"""Combat runtime: equipment holds, health and match state for a PhysicsWorld.

Attached to a world only when cfg["combat"] exists (a plain island never pays
for it). This module holds NO simulation of its own: it is a state + rules layer
over the real physics. Equipment bodies are real movable MuJoCo objects built
from the equipment registry (see world/spec.py), holds reuse the world's carry
hold, and damage (phase 2) is computed from measured MuJoCo collision events —
never from an invented counter.

The per-step callback (`on_step`) is registered through the world's existing
`register_step_callback` hook, so combat adds no new stepping pathway.
"""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from .equipment import EQUIPMENT, EquipmentItem


class AgentCombatState:
    """Per-agent combat bookkeeping. Health, incapacity and holds start at full."""

    __slots__ = ("agent", "team", "health", "incapacitated",
                 "item", "equipped", "attack_ready_at", "bracing")

    def __init__(self, agent: str, team: str, health_max: float):
        self.agent = agent
        self.team = team
        self.health = float(health_max)
        self.incapacitated = False
        self.item: EquipmentItem | None = None   # the equipped EquipmentItem
        self.equipped: str | None = None         # the world object id carrying it
        self.attack_ready_at = 0.0
        self.bracing = False

    def equipped_dict(self) -> dict | None:
        it = self.item
        if it is None:
            return None
        return {
            "object": self.equipped,
            "item": it.equip_id,
            "ammo": it.ammo,
            "ammo_capacity": int(it.spec.get("ammo_capacity", 0)),
            "grip": it.spec["grip"],
            "class": it.spec["combat_class"],
        }


class CombatRuntime:
    def __init__(self, world: Any, cfg: dict, action_error=None):
        self.world = world
        self.cfg = dict(cfg)
        self.action_error = action_error  # the physics module's ActionError class
        self.health_max = float(cfg.get("health", 100.0))
        self.teams: dict[str, list[str]] = {str(k): [str(a) for a in (v or [])]
                                            for k, v in (cfg.get("teams") or {}).items()}
        self.items: list[dict] = list(cfg.get("items") or [])
        self._equipment_ids: set[str] = set()
        self._team_of: dict[str, str] = {}
        self._agents: dict[str, AgentCombatState] = {}
        self.match_over = False
        self.winner: str | None = None
        self._event_cursor = 0   # event-stream cursor for the damage pass
        # Damage constants (documented in docs/BUGLOG_2026-11-17.md §10): damage is
        # derived ONLY from measured collision impulses. The floor keeps contacts
        # that cannot injure (walk-ins, nudges, brushes) out of the damage path;
        # the cap bounds a single hit.
        self.damage_floor = float(cfg.get("damage_min_impulse", 0.8))
        self.damage_max = float(cfg.get("damage_max", 95.0))
        self._pellet_pool: list[str] = []
        self._flight: dict[str, dict] = {}   # pellet id -> {"scale", "item", "agent"}

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        """Bind teams + per-agent state once the world is fully built."""
        w = self.world
        for team, aids in self.teams.items():
            for aid in aids:
                if aid not in w._agents:
                    self._fail(f"combat team '{team}' names unknown agent '{aid}'")
                self._team_of[aid] = team
        unassigned = [aid for aid in w._agents if aid not in self._team_of]
        if unassigned:
            self._fail(f"agents not assigned to a combat team: {unassigned}")
        for entry in self.items:
            oid = str(entry.get("id"))
            if oid not in w.specs:
                self._fail(f"combat item '{oid}' is not in the world spec")
            item = w.specs[oid].properties.get("equipment")
            if not item or item not in EQUIPMENT:
                self._fail(f"combat item '{oid}' does not carry a known equipment spec")
            self._equipment_ids.add(oid)
        for aid in sorted(w._agents):
            self._agents[aid] = AgentCombatState(aid, self._team_of.get(aid, ""), self.health_max)
        self._pellet_pool = [oid for oid, s in w.specs.items() if s.kind == "pellet"]
        self._pellet_pool.sort()

    def _fail(self, message: str):
        if self.action_error is not None:
            raise self.action_error("combat_config", message)
        raise ValueError(message)

    def _agent_data(self, agent: str, data: dict) -> dict:
        """Event attribution mirroring the world's own agent-scoped vocabulary."""
        if self.world._multi:
            return {"agent": agent, **data}
        return dict(data)

    def _unknown_agent(self, agent: str):
        raise self.action_error("unknown_agent",
                                f"No agent id '{agent}' in this combat roster.",
                                known_agents=sorted(self._agents)) from None

    # ------------------------------------------------------------------ equipment verbs
    def equip(self, agent: str, oid: str) -> dict:
        w = self.world
        acs = self._agents.get(agent)
        if acs is None:
            self._unknown_agent(agent)
        if acs.incapacitated:
            raise self.action_error("incapacitated",
                                    f"'{agent}' cannot equip: incapacitated.") from None
        spec = w.specs.get(oid)
        if spec is None or oid not in w.active:
            raise self.action_error("unknown_object",
                                    f"There is no object '{oid}' to equip.",
                                    known_ids=sorted(self._equipment_ids)) from None
        item_id = spec.properties.get("equipment") if spec.kind == "equipment" else None
        if not item_id or item_id not in EQUIPMENT:
            raise self.action_error(
                "not_equipment",
                f"'{oid}' is not equipment; it carries no usable item spec.") from None
        # a hold is a hold: one object at a time, through the real carry machinery
        if acs.equipped is not None or w._agents[agent].held:
            raise self.action_error("hands_full",
                                    f"'{agent}' already holds an object; put it down first.",
                                    holding=w._agents[agent].held) from None
        # reaching distance is enforced by the real pickup reach check
        w.pick_up(oid, agent=agent)
        acs.item = EquipmentItem(w, item_id, launcher=None, object_id=oid)
        if acs.item.spec["combat_class"] == "melee":
            # a melee weapon rides the hand: an arm swing carries it into the
            # target (see attack())
            w._agents[agent].grip_anchor = "hand"
        if acs.item.spec["combat_class"] == "shield":
            # The ordinary hold pins the body and REMOVES its collisions
            # (blocks would scrape the carrier). A shield plate's whole job is
            # to take the incoming contact, so its geoms stay collidable while
            # held; the carry pin keeps it exactly 0.30 m ahead of the chest,
            # clear of the holder's own torso (torso half-depth 0.09 m).
            w._set_collidable(oid, True)
        if acs.item.spec["combat_class"] == "ranged":
            acs.item.launcher = self._make_launcher(acs)
        # firing requires the item's hold on the item state machine too
        needs_both = acs.item.spec["grip"] in ("two_hand", "dual")
        acs.item.grip(left=True, right=needs_both)
        acs.equipped = oid
        w._agents[agent].combat_item = item_id
        ev = w.emit("equipment_equipped", **self._agent_data(
            agent, {"object": oid, "item": item_id, "ammo": acs.item.ammo}))
        return {"equipped": oid, "item": item_id, "ammo": acs.item.ammo,
                "held_object": oid, "event_n": ev.get("n")}

    def unequip(self, agent: str) -> dict:
        w = self.world
        acs = self._agents.get(agent)
        if acs is None:
            self._unknown_agent(agent)
        if acs.incapacitated:
            raise self.action_error("incapacitated",
                                    f"'{agent}' cannot unequip: incapacitated.") from None
        if acs.equipped is None:
            raise self.action_error("not_equipped",
                                    f"'{agent}' is not holding any equipment.") from None
        oid = acs.equipped
        if w._agents[agent].held == oid:
            w.put_down(oid, agent=agent)
        acs.item, acs.equipped = None, None
        w._agents[agent].combat_item = None
        w._agents[agent].grip_anchor = "chest"
        w.emit("equipment_unequipped", **self._agent_data(agent, {"object": oid}))
        return {"dropped": oid, "held_object": None}

    # ------------------------------------------------------------------ state surface
    def state(self, agent: str) -> dict:
        acs = self._agents.get(agent)
        if acs is None:
            self._unknown_agent(agent)
        self._refresh(acs)
        return {
            "agent": agent,
            "team": acs.team,
            "health": round(acs.health, 3),
            "health_max": self.health_max,
            "incapacitated": acs.incapacitated,
            "equipped": acs.equipped_dict(),
            "bracing": acs.bracing,
            "match_over": self.match_over,
            "winner": self.winner,
        }

    def report_bits(self, agent: str) -> dict:
        """The measured combat facts that belong in agents_report rows."""
        acs = self._agents.get(agent)
        if acs is None:
            return {"team": "", "health": 0.0, "incapacitated": True}
        self._refresh(acs)
        return {"team": acs.team, "health": round(acs.health, 3),
                "incapacitated": acs.incapacitated}

    # ------------------------------------------------------------------ firing (M4 phase 2)
    def _refresh(self, acs: AgentCombatState) -> None:
        """Finalise any staged reload once its window has elapsed."""
        it = acs.item
        if it is None:
            return
        if it.reloading_until is not None and it._now() >= it.reloading_until:
            it.ammo = it.spec["ammo_capacity"]
            it.reloading_until = None

    def _make_launcher(self, acs: AgentCombatState):
        """The real-world launcher for one held item: it spawns a pool pellet
        at the measured muzzle (the held object's own position) and gives it
        its launch velocity — the same release-velocity boundary condition the
        throw verb uses. No stepping here: the caller's world steps itself."""

        def launch(origin, direction, spread_deg) -> list[str]:
            return self._launch(acs, np.asarray(origin, dtype=float),
                                np.asarray(direction, dtype=float), float(spread_deg))

        return launch

    def _pop_pellet(self) -> str | None:
        return self._pellet_pool.pop(0) if self._pellet_pool else None

    def _spread_direction(self, d: np.ndarray, spread_deg: float, k: int, n: int) -> np.ndarray:
        """One deterministic cone offset (golden-ratio ring) applied in the
        horizontal plane + a small vertical tilt; identical shots stay identical."""
        u = d / max(float(np.linalg.norm(d)), 1e-9)
        if spread_deg <= 0.0 or n <= 1:
            return u
        golden = 0.6180339887
        yaw = math.radians(spread_deg) * (2.0 * ((k * golden) % 1.0) - 1.0)
        pitch = math.radians(spread_deg) * (2.0 * ((n - 1.0 - k) * golden) % 1.0 - 1.0)
        cy, sy = math.cos(yaw), math.sin(yaw)
        u = np.array([cy * u[0] - sy * u[1], sy * u[0] + cy * u[1], u[2]])
        tilt = np.array([u[0], u[1]]) * math.sin(pitch)
        u = np.array([u[0], u[1], u[2] + math.cos(pitch) - 1.0 + 0.0])
        u[0:2] += tilt
        u[2] = max(u[2], -0.60)  # no ground-dunking offsets
        return u / max(float(np.linalg.norm(u)), 1e-9)

    def _launch(self, acs: AgentCombatState, origin: np.ndarray, direction: np.ndarray,
                spread_deg: float) -> list[str]:
        w = self.world
        it = acs.item
        p = it.spec["projectile_cfg"]
        speed = float(p["speed"])
        n = max(1, int(it.spec["pellets_per_shot"]))
        scale = float(it.spec["damage_scale"])
        u = direction / max(float(np.linalg.norm(direction)), 1e-9)
        muzzle = origin + u * 0.05
        pids: list[str] = []
        for k in range(n):
            pid = self._pop_pellet()
            if pid is None:
                break
            d = self._spread_direction(u, spread_deg, k, n)
            w._unpark(pid)
            w.spawn_object(pid, [round(float(v), 4) for v in muzzle])
            w.set_object_velocity(pid, [round(float(v), 4) for v in (d * speed)])
            w.active.add(pid)
            w.generated.append(pid)
            self._flight[pid] = {"scale": scale, "item": it.equip_id, "agent": acs.agent}
            w.emit("pellet_launched", **self._agent_data(acs.agent, {
                "object": pid, "item": it.equip_id, "speed": speed, "pellet": k,
                "direction": [round(float(v), 4) for v in d],
            }))
            pids.append(pid)
        mujoco.mj_forward(w.m, w.d)
        return pids

    def fire(self, agent: str, direction) -> dict:
        w = self.world
        acs = self._agents.get(agent)
        if acs is None:
            self._unknown_agent(agent)
        if acs.incapacitated:
            raise self.action_error("incapacitated",
                                    f"'{agent}' cannot fire: incapacitated.") from None
        it = acs.item
        if it is None:
            raise self.action_error("not_equipped",
                                    f"'{agent}' is not holding any weapon to fire.") from None
        if it.spec["combat_class"] in ("melee", "shield"):
            raise self.action_error(
                "not_a_launcher",
                f"'{it.equip_id}' is a {it.spec['combat_class']} item, not a launcher.") from None
        self._refresh(acs)
        origin = np.asarray(w.obj_pos(acs.equipped), dtype=float)  # the muzzle: the held item's body
        d = np.asarray(direction, dtype=float)
        if float(np.linalg.norm(d)) < 1e-9:
            raise self.action_error("invalid_argument", "The firing direction cannot be zero.") from None
        res = it.fire(d.tolist(), origin=origin)
        if it.spec["combat_class"] == "throwable" and res.get("fired"):
            # a throw releases the hold (world.throw did it) — the combat state follows
            acs.item, acs.equipped = None, None
            w._agents[agent].combat_item = None
        elif res.get("fired") and not res.get("proj_ids"):
            # the pool is empty: an honest refusal, no ammo burned
            it.ammo += 1
            res["fired"] = False
            res["reason"] = "pool_empty"
            res.pop("shortfall", None)
        res["muzzle"] = [round(float(v), 4) for v in origin]
        return res

    def reload(self, agent: str) -> dict:
        w = self.world
        acs = self._agents.get(agent)
        if acs is None:
            self._unknown_agent(agent)
        if acs.incapacitated:
            raise self.action_error("incapacitated",
                                    f"'{agent}' cannot reload: incapacitated.") from None
        it = acs.item
        if it is None:
            raise self.action_error("not_equipped",
                                    f"'{agent}' is not holding anything to reload.") from None
        if it.spec["combat_class"] not in ("ranged", "throwable"):
            raise self.action_error("not_a_launcher",
                                    f"'{it.equip_id}' has nothing to reload.") from None
        self._refresh(acs)
        res = it.reload()
        w.emit("weapon_reload", **self._agent_data(agent, {
            "object": acs.equipped, "item": it.equip_id, "until": round(float(res.get("until") or 0.0), 4),
        }))
        return res

    # ------------------------------------------------------------------ combat actions (M4 phase 3)
    def attack(self, agent: str, direction) -> dict:
        """Drive the held melee weapon into the target with a body lunge.

        This rig has no free arms — its only engine-honest actuation is the
        contact-driven root walk — so a strike is a physical charge: the gait
        surges along the aim for the weapon's swing window, the weapon (pinned
        ahead of the chest) contacts first, and the measured impact impulse
        flows through the ordinary damage pass. No synthetic counters."""
        w = self.world
        acs = self._agents.get(agent)
        if acs is None:
            self._unknown_agent(agent)
        if acs.incapacitated:
            raise self.action_error("incapacitated",
                                    f"'{agent}' cannot attack: incapacitated.") from None
        it = acs.item
        if it is None:
            raise self.action_error("not_equipped",
                                    f"'{agent}' is not holding anything to attack with.") from None
        if it.spec["combat_class"] != "melee":
            raise self.action_error(
                "not_a_melee_weapon",
                f"'{it.equip_id}' is a {it.spec['combat_class']} item, not a melee weapon.") from None
        now = w.time
        if now < acs.attack_ready_at:
            return {"attacked": False, "reason": "rate_limit",
                    "ready_at": round(acs.attack_ready_at, 4)}
        # the held body cannot strike while its geoms are non-collidable (the
        # hold contract); restored the moment the strike ends. Only the strike
        # is collidable: while raising, the folded weapon still overlaps its
        # own forearm, and a collidable pin there would shove the holder.
        w.emit("attack_started", **self._agent_data(agent, {"object": acs.equipped, "item": it.equip_id}))
        swing = float(it.spec["swing_time_s"])
        gain = float(it.spec["swing_shoulder_gain"])
        kd = float(it.spec["swing_shoulder_kd"])
        rt = w._agents[agent]
        shoulder = rt.name_map.get("left_shoulder", "left_shoulder")
        elbow = rt.name_map.get("left_elbow", "left_elbow")
        t_start = w.time
        try:
            # 1. raise the weapon up-forward (still non-collidable)
            n_r = max(1, int(0.45 / w.dt))
            for i in range(n_r):
                f = (i + 1) / n_r
                w.set_joint_target(shoulder, 1.45 * f, kp=gain, kd=kd)
                w.set_joint_target(elbow, 2.35 * f, kp=0.8 * gain, kd=kd)
                w._step()
            # 2. strike: a fast straight-arm thrust — the extended arm holds
            #    the weapon clear of its own body, so its geoms can take and
            #    give real contacts through the entire thrust
            w._set_collidable(acs.equipped, True)
            n_s = max(1, int(swing / w.dt))
            for i in range(n_s):
                f = (i + 1) / n_s
                w.set_joint_target(shoulder, 1.45 - 0.40 * f, kp=gain, kd=kd)
                w.set_joint_target(elbow, 2.35 - 2.25 * f, kp=0.8 * gain, kd=kd)
                w._step()
        finally:
            w._set_collidable(acs.equipped, False)
        t_end = w.time
        # 3. return the arm to the carry fold
        for _ in range(max(1, int(0.6 / w.dt))):
            w.set_joint_target(shoulder, 0.35, kp=40.0, kd=5.0)
            w.set_joint_target(elbow, 0.10, kp=32.0, kd=4.0)
            w._step()
        contacts = []
        for ev in w.events:
            objs = ev["data"].get("objects") or []
            if ev.get("type") == "collision" and acs.equipped in objs \
                    and t_start <= float(ev.get("t", 0.0)) <= t_end + 0.4:
                others = [o for o in objs if o != acs.equipped]
                if not others:
                    continue
                contacts.append({"with": others, "impulse": round(float(ev["data"].get("impulse") or 0.0), 4),
                                 "t": round(float(ev["t"]), 3)})
        acs.attack_ready_at = t_end + float(it.spec["rate_limit_s"])
        return {"attacked": True, "weapon": it.equip_id,
                "swing_time_s": swing,
                "contacts": contacts[-3:],
                "rate_ready_at": round(acs.attack_ready_at, 4)}

    def defend(self, agent: str, stance: str = "guard") -> dict:
        """Brace with the held shield: the carry pin lifts the plate into a
        guard pose (higher, further out). Fully physical — the plate is the
        same pinned body, only the pin pose changes, and it keeps taking the
        real impacts with its measured absorb."""
        w = self.world
        acs = self._agents.get(agent)
        if acs is None:
            self._unknown_agent(agent)
        if acs.incapacitated:
            raise self.action_error("incapacitated",
                                    f"'{agent}' cannot guard: incapacitated.") from None
        it = acs.item
        if it is None:
            raise self.action_error("not_equipped",
                                    f"'{agent}' is not holding anything to guard with.") from None
        if it.spec["combat_class"] != "shield":
            raise self.action_error("no_shield",
                                    f"'{it.equip_id}' is a {it.spec['combat_class']} item, not a shield.") from None
        if stance not in ("guard", "carry"):
            raise self.action_error("invalid_argument",
                                    "stance must be 'guard' or 'carry'.") from None
        rt = w._agents[agent]
        if stance == "guard":
            acs.bracing = True
            rt.spec["hold_height_humanoid"] = float(it.spec["guard_height"])
            rt.spec["hold_forward"] = float(it.spec["guard_forward"])
        else:
            acs.bracing = False
            rt.spec["hold_height_humanoid"] = 0.84   # the rig spec's carry pose
            rt.spec["hold_forward"] = 0.30
        for _ in range(max(1, int(0.25 / w.dt))):
            w._step()
        w.emit("guard_pose", **self._agent_data(agent, {
            "object": acs.equipped, "stance": stance,
            "position": [round(float(v), 3) for v in w.obj_pos(acs.equipped or "")],
        }))
        return {"guarded": stance == "guard", "stance": stance,
                "plate_position": [round(float(v), 3) for v in w.obj_pos(acs.equipped)]}

    def _incapacitate(self, agent: str) -> None:
        w = self.world
        acs = self._agents[agent]
        if acs.incapacitated:
            return
        acs.incapacitated = True
        rt = w._agents[agent]
        # a real release: the weapon drops where the agent stood
        if rt.held:
            w.put_down(rt.held, agent=agent)
        acs.item, acs.equipped = None, None
        rt.combat_item = None
        rt.grip_anchor = "chest"
        w.incapacitate_rig(agent)
        w.emit("agent_incapacitated", **self._agent_data(agent, {
            "health": 0.0, "health_max": self.health_max,
            "position": [round(float(v), 3) for v in w.agent_pos(agent)],
        }))

    def _check_match(self, w: Any) -> None:
        if self.match_over or len(self.teams) < 2:
            return
        alive = {team: [a for a in aids if not self._agents[a].incapacitated]
                 for team, aids in self.teams.items()}
        living = [t for t, xs in alive.items() if xs]
        if len(living) <= 1:
            self.match_over = True
            self.winner = living[0] if living else None
            w.emit("match_ended", winner=self.winner,
                   remaining={t: len(xs) for t, xs in alive.items()})

    # ------------------------------------------------------------------ damage pass (M4 phase 2)
    def on_step(self, world: Any) -> None:
        """Per-step pass over the NEW collision events; damage derives only
        from their measured impulse/impulse-scale values."""
        events = world.events
        i = self._event_cursor
        if i > len(events):
            i = 0
        self._event_cursor = len(events)
        if i < len(events):
            for ev in events[i:]:
                if ev.get("type") != "collision":
                    continue
                data = ev.get("data", {})
                objs = data.get("objects") or []
                if len(objs) < 2:
                    continue
                self._process_collision(data, objs)
        self._check_match(world)

    def _process_collision(self, data: dict, objs: list) -> None:
        impulse = float(data.get("impulse") or 0.0)
        parts = data.get("agent_parts") or {}
        seen: set[str] = set()
        for aid in objs:
            if aid in self._agents and aid not in seen:
                seen.add(aid)
                others = [o for o in objs if o != aid]
                if others:
                    self._apply_damage(aid, others[0], data, parts, impulse)
        # A hit on held defensive gear: the collision names the plate, not the
        # agent, so attribute it to the shield's holder (absorbed fraction).
        for oid in objs:
            if oid in self._agents:
                continue
            holder = self._shield_holder(oid)
            if holder is None or holder in seen:
                continue
            others = [o for o in objs if o != oid]
            if not others or others[0] in self._agents:
                continue  # body-vs-plate contact is melee territory (phase 3)
            seen.add(holder)
            self._apply_damage(holder, others[0], data, parts, impulse, via_plate=True)

    def _shield_holder(self, oid: str) -> str | None:
        """The agent this equipped shield plate belongs to (None if not one)."""
        for aid, acs in self._agents.items():
            if acs.equipped == oid and acs.item is not None \
                    and acs.item.spec["combat_class"] == "shield":
                return aid
        return None

    def _apply_damage(self, target: str, other: str, data: dict, parts: dict,
                      impulse: float, via_plate: bool = False) -> None:
        w = self.world
        acs = self._agents[target]
        scale, kind = self._damage_source(other)
        if scale is None or kind is None:
            return
        if impulse < self.damage_floor:
            return
        if w._agents[target].held == other:
            # an agent and THEIR OWN held weapon: the kinematic hold welds the
            # two bodies together, so their solver contacts are constraint
            # artifacts, not strikes — measured, but not damage
            return
        part = str(parts.get(target, ""))
        mult = self._part_multiplier(part)
        if via_plate or (other == acs.equipped and acs.item is not None
                         and acs.item.spec["combat_class"] == "shield"):
            # the measured impact was taken by the holder's own plate
            absorb = 1.0 - float(acs.item.spec.get("absorb", 0.0))
        else:
            absorb = 1.0
        damage = min(impulse * scale * mult * absorb, self.damage_max)
        acs.health = max(0.0, acs.health - damage)
        w.emit("agent_damaged", **self._agent_data(target, {
            "source": other, "source_kind": kind, "impulse": round(impulse, 4),
            "damage_scale": scale, "part": part, "part_multiplier": mult,
            "shielded": absorb < 0.999, "absorb": round(absorb, 3),
            "damage": round(damage, 3), "health": round(acs.health, 3),
            "health_max": self.health_max,
            "position": list(data.get("position") or []),
        }))
        if acs.health <= 0.0:
            self._incapacitate(target)

    def _damage_source(self, other: str) -> tuple[float | None, str | None]:
        """(damage_scale, kind) of the object that struck an agent, or (None, None)
        when the contact carries no damage (gentle bodies, inert objects)."""
        fl = self._flight.pop(other, None)
        if fl is not None:
            return float(fl["scale"]), "pellet"
        spec = self.world.specs.get(other)
        if spec is not None:
            item_id = spec.properties.get("equipment")
            if item_id and item_id in EQUIPMENT and float(EQUIPMENT[item_id].get("damage_scale", 0)) > 0:
                return float(EQUIPMENT[item_id]["damage_scale"]), "equipment"
        return None, None

    def _absorb(self, target_agent: str, other: str) -> float:
        """What fraction of the measured impulse reaches the target: a held
        shield plate that takes the contact absorbs its spec fraction."""
        acs = self._agents.get(target_agent)
        if acs is not None and acs.item is not None and other == acs.equipped \
                and acs.item.spec["combat_class"] == "shield":
            return 1.0 - float(acs.item.spec.get("absorb", 0.0))
        return 1.0

    @staticmethod
    def _part_multiplier(part: str) -> float:
        """Body-part sensitivity of an impact (documented combat constants)."""
        p = part.lower()
        if any(tag in p for tag in ("head", "neck")):
            return 1.5
        if "torso" in p:
            return 1.0
        if any(tag in p for tag in ("hip", "pelvis", "waist")):
            return 0.9
        if p:
            return 0.6  # limbs: legs, shoulders, arms
        return 1.0
