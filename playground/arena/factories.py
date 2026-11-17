"""Production arena factories (M5).

These are the real factories :meth:`ArenaMatch.play` exists for. Every piece wraps
the live engine — no stubs:

* :func:`build_match_setup` translates a match configuration into the real
  application config a multi-agent combat :class:`~playground.world.physics.PhysicsWorld`
  is built from (per-agent bodies, ``cfg["combat"]`` items laid out within equip
  reach, the latent pellet pool).
* :func:`production_world_factory` builds that world, spawns and ORIENTS every
  combatant through :meth:`PhysicsWorld.setup_agent_pose`, and wraps it in
  :class:`ArenaWorldAdapter`, which steps the engine and translates the real combat
  events (``pellet_launched``, ``agent_damaged``, ``agent_incapacitated``,
  ``match_ended``) into the scorer's impact contract with real proj_id -> shooter
  attribution.
* :class:`ArenaSessionHandle` + the session launcher inside :func:`run_arena_match`
  launch each combatant's REAL recorded session through
  :func:`~playground.session.runner.run_session` (mode ``arena_solo`` or
  ``arena_coop``, prompt id + system prompt hash recorded, the shared world's
  events/frames fanned out to every participating session) and drive the adapter one
  turn between physics ticks. All world access stays on the match thread.
* :func:`run_arena_match` wires plan + factories + scorer end to end, augments the
  result with the runtime's measured KO arbitration and finalizes every session.
* :func:`cmd_arena` is the ``playground arena`` CLI subcommand.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from ..agents.adapters import Adapter
from ..agents.prompts import continue_nudge
from ..config import load_config, resolve_path
from ..session.runner import finish_session, is_session_stop_requested, run_session
from ..world.physics import PhysicsWorld
from .match import ArenaMatch, MatchResult, export_match_report

DEFAULT_TICK = 0.05           # match loop cadence (20 Hz)
DEFAULT_TURN_INTERVAL = 2.0   # one adapter turn per combatant every 2 s of match time
DEFAULT_PELLETS = 24
DEFAULT_SPAWN_DISTANCE = 2.8  # opposing-team line spacing (the M4-proven duel geometry)
TEAM_SPACING = 1.4            # gap between teammates on the same line
ITEM_FORWARD = 0.6            # equipment rest position in front of the team line
ITEM_LATERAL = 0.45           # lateral spacing between a team's equipment items


class ArenaFactoryError(ValueError):
    """An arena match configuration the production factories refuse honestly."""


def _oid(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "team"


class MatchSetup(NamedTuple):
    """Everything the production factories need to build and run one match."""
    world_cfg: Dict[str, Any]           # the real PhysicsWorld application config
    agent_ids: List[str]                # combatant ids (== world._agents keys)
    poses: Dict[str, float]             # combatant id -> facing yaw (world heading, deg)
    combat_teams: Dict[str, List[str]]  # team name -> combatant ids


def build_match_setup(match_cfg: dict, root_cfg: dict | None = None) -> MatchSetup:
    """Translate a match configuration into the real application config for its world.

    Team 0 spawns on the -x side facing +x, team 1 on the +x side facing -x,
    teammates spaced along y. Each team's equipment items rest in front of its line,
    within equip reach, laterally spaced. Override points: ``arena.spawn_distance``,
    ``arena.team_spacing``, ``arena.item_offset``, ``arena.agent_yaws`` and per-agent
    ``arena.agent_starts``.
    """
    from ..combat.equipment import EQUIPMENT
    world_cfg = copy.deepcopy(root_cfg) if root_cfg is not None else load_config()
    world_cfg["agent_body"] = {**world_cfg.get("agent_body", {}), "type": "humanoid"}

    teams = match_cfg.get("teams") or []
    if not teams:
        raise ArenaFactoryError("match config has no teams")
    if len(teams) < 2:
        raise ArenaFactoryError("an arena match needs at least two teams")
    arena = match_cfg.get("arena") or {}
    rules = match_cfg.get("rules") or {}
    spawn_distance = float(arena.get("spawn_distance", DEFAULT_SPAWN_DISTANCE))
    team_spacing = float(arena.get("team_spacing", TEAM_SPACING))
    item_forward = float(arena.get("item_offset", ITEM_FORWARD))
    override_starts = {str(k): [float(v[0]), float(v[1])]
                       for k, v in (arena.get("agent_starts") or {}).items()}
    override_yaws = arena.get("agent_yaws") or {}

    agent_bodies: Dict[str, Dict[str, Any]] = {}
    combat_teams: Dict[str, List[str]] = {}
    items: List[Dict[str, Any]] = []
    poses: Dict[str, float] = {}
    agent_ids: List[str] = []
    for ti, team in enumerate(teams):
        name = team.get("name")
        members = team.get("agents") or []
        if not name or not members:
            raise ArenaFactoryError("each team requires a name and at least one agent")
        facing = 0.0 if ti % 2 == 0 else 180.0
        aids: List[str] = []
        first_spawn: Optional[List[float]] = None
        for ai, acfg in enumerate(members):
            aid = str(acfg.get("id") or f"{_oid(str(name))}_agent_{ai + 1}")
            if aid in agent_bodies:
                raise ArenaFactoryError(f"combatant ids must be unique: '{aid}' appears twice")
            side = -1.0 if ti % 2 == 0 else 1.0  # team 0 on -x, team 1 on +x (facing across)
            line_x = side * spawn_distance / 2.0
            start = override_starts.get(aid) or [
                round(line_x, 4),
                round((ai - (len(members) - 1) / 2.0) * team_spacing, 4)]
            agent_bodies[aid] = {"start": start}
            poses[aid] = float(override_yaws.get(aid, facing))
            if first_spawn is None:
                first_spawn = list(start)
            aids.append(aid)
            agent_ids.append(aid)
        combat_teams[str(name)] = aids
        for k, item_name in enumerate(team.get("equipment") or []):
            if item_name not in EQUIPMENT:
                raise ArenaFactoryError(
                    f"unknown equipment item '{item_name}'; known: {sorted(EQUIPMENT)}")
            yaw0 = math.radians(poses[aids[0]])
            fx, fy = math.cos(yaw0), math.sin(yaw0)          # facing direction
            px, py = -fy, fx                                 # lateral direction
            base = override_starts.get(aids[0]) or first_spawn
            lat = k * ITEM_LATERAL
            items.append({
                "id": f"{_oid(str(name))}_item_{k}",
                "item": item_name,
                "at": [round(float(base[0]) + fx * item_forward + px * lat, 4),
                       round(float(base[1]) + fy * item_forward + py * lat, 4)],
            })
    world_cfg["agent_bodies"] = agent_bodies
    world_cfg["combat"] = {
        "teams": combat_teams,
        "items": items,
        "pellets": int(rules.get("pellets", DEFAULT_PELLETS)),
    }
    return MatchSetup(world_cfg=world_cfg, agent_ids=agent_ids, poses=poses,
                      combat_teams=combat_teams)


class ArenaWorldAdapter:
    """Bridges the real shared PhysicsWorld to the match/scorer contract.

    :meth:`step` advances the engine, then translates the newly arrived real events:
    a ``pellet_launched`` registers its true pellet id with the scorer; an
    ``agent_damaged`` from a pellet becomes the scorer's ``projectile_impact`` event
    with real attribution (pellets through the runtime's launched flights, melee
    strikes via the wielder of the striking item); every translated event is returned
    so the match timeline records it verbatim.
    """

    def __init__(self, world: PhysicsWorld, scorer):
        self.world = world
        self.scorer = scorer
        self.runtime = world.combat
        self._cursor = 0
        self._melee_seq = 0
        self._handles: List["ArenaSessionHandle"] = []
        self._fanout = False
        self.match_ended: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------ registry helpers
    def team_of(self, agent_id: str) -> str:
        if self.runtime is not None:
            for team, aids in self.runtime.teams.items():
                if agent_id in aids:
                    return team
        return "unknown"

    def _wielder_of(self, object_id: str) -> Optional[str]:
        """The combatant currently holding the striking item (melee attribution)."""
        if self.runtime is not None:
            for aid, acs in getattr(self.runtime, "_agents", {}).items():
                if getattr(acs, "equipped", None) == object_id:
                    return aid
        return None

    # ------------------------------------------------------------ world stepping
    def step(self, dt: float) -> List[Dict[str, Any]]:
        n = max(1, int(round(dt / float(self.world.dt))))
        self.world.step(n)
        return self._translate()

    def events(self) -> List[Dict[str, Any]]:
        """Translate events that arrived since the last call, without stepping."""
        return self._translate()

    def _translate(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        events = self.world.events
        while self._cursor < len(events):
            ev = events[self._cursor]
            self._cursor += 1
            etype = ev.get("type")
            data = ev.get("data", {}) or {}
            if etype == "collision":
                continue  # engine detail; damage/launch attribution below is the truth
            if etype == "pellet_launched":
                pid = data.get("object")
                shooter = data.get("agent")
                if pid and shooter:
                    self.scorer.register_projectile(pid, shooter, self.team_of(shooter))
                out.append({"type": "pellet_shot", "proj_id": pid, "shooter_agent": shooter,
                            "shooter_team": self.team_of(shooter or ""),
                            "item": data.get("item"), "speed": data.get("speed")})
            elif etype == "agent_damaged":
                target = data.get("agent")
                src = data.get("source")
                kind = data.get("source_kind")
                entry = {"type": "projectile_impact", "proj_id": src, "target_agent": target,
                         "team": self.team_of(target or ""), "damage": data.get("damage"),
                         "health": data.get("health"), "impulse": data.get("impulse"),
                         "part": data.get("part")}
                if kind == "pellet":
                    out.append(entry)
                elif kind == "equipment":
                    wielder = self._wielder_of(src)
                    if wielder is not None:
                        pid = f"melee:{src}:{self._melee_seq}"
                        self._melee_seq += 1
                        self.scorer.register_projectile(pid, wielder, self.team_of(wielder))
                        out.append({**entry, "proj_id": pid, "melee": True})
                    else:
                        out.append({**entry, "unattributed": True})
                else:
                    out.append({**entry, "unattributed": True})
            elif etype in ("agent_incapacitated", "match_ended", "weapon_reload"):
                agent = data.get("agent")
                out.append({"type": etype, "agent": agent,
                            **{k: v for k, v in data.items() if k != "agent"}})
                if etype == "match_ended" and self.match_ended is None:
                    self.match_ended = {"winner": data.get("winner")}
            # everything else (engine chatter) stays out of the match
        return out

    # ------------------------------------------------------------ session stream fan-out
    def attach_handle(self, handle: "ArenaSessionHandle", expected: int) -> None:
        """Accept a freshly launched combatant session; once every combatant has one,
        fan the shared world's events/frames out to all participating sessions."""
        self._handles.append(handle)
        if len(self._handles) >= expected:
            self.install_fanout(expected)

    def install_fanout(self, expected: int) -> None:
        """Wire the world's event/frame sinks so every session streams the shared world."""
        if self._fanout or len(self._handles) < expected:
            return
        handles = list(self._handles)

        def on_event(ev: dict) -> None:
            for h in handles:
                try:
                    h.env._on_event(ev)
                except Exception:  # noqa: BLE001 - one session's stream must not stop the rest
                    pass

        def on_frame(frame: dict) -> None:
            for h in handles:
                try:
                    h.env._on_frame(frame)
                except Exception:  # noqa: BLE001
                    pass
        self.world.event_sink = on_event
        self.world.frame_sink = on_frame
        self._fanout = True


class DrivenAdapterBridge(Adapter):
    """Presents a real per-agent adapter to run_session's recording path.

    run_session records the session, writes the system prompt and opening, and
    returns the live Environment without driving the adapter (``driven_externally``);
    the arena match then drives the inner adapter turn by turn through its handle.
    """

    driven_externally = True

    def __init__(self, inner: Adapter):
        self.inner = inner
        self.name = getattr(inner, "name", "arena")
        self.model = getattr(inner, "model", "")

    def begin(self, system: str, tool_names: list) -> None:
        return None  # the handle begins the inner adapter at match start

    def step(self, user_text, tool_results):
        raise NotImplementedError("arena sessions are driven by the match loop")

    def describe(self) -> dict:
        if hasattr(self.inner, "describe"):
            try:
                return self.inner.describe()
            except Exception:  # noqa: BLE001
                pass
        return super().describe()


class ArenaSessionHandle:
    """One combatant's recorded session, driven turn by turn by the match loop.

    Mirrors the runner loop mechanics (env.call tool execution, assistant message
    rows, no-tool-call nudges, stop signal, usage accounting) in arena tick shape:
    one adapter turn every ``turn_interval_s`` of match time, always on the match
    thread. Returns no match events — impacts surface through the world adapter.
    """

    def __init__(self, summary: dict, adapter: Adapter, bridge: DrivenAdapterBridge,
                 team: str, mode: str, turn_interval_s: float, cfg: dict, db,
                 teammates: List[str], opponents: List[str]):
        self.summary = summary
        self.sid = summary["session_id"]
        self.env = summary["environment"]
        self.adapter = adapter
        self.bridge = bridge
        self.team = team
        self.mode = mode
        self.turn_interval_s = float(turn_interval_s)
        self.user_text: Optional[str] = summary["opening"]
        self.results: List[Tuple[Any, dict]] = []
        self.turns = 0
        self.usage: Dict[str, float] = {}
        self.done = False
        self.error: Optional[str] = None
        self._began = False
        self._since = self.turn_interval_s  # the first turn happens on the first tick
        self.cfg = cfg
        self.db = db

    def step(self, dt: float) -> List[Dict[str, Any]]:
        if self.done:
            return []
        if not self._began:
            self.adapter.begin(self.summary["system"], self.summary["tools"])
            self._began = True
        self._since += dt
        if is_session_stop_requested(self.sid):
            self.done = True
            return []
        if self._since < self.turn_interval_s:
            return []
        self._since = 0.0
        self._drive_turn()
        return []

    def _drive_turn(self) -> None:
        sid = self.sid
        try:
            turn = self.adapter.step(self.user_text, self.results)
        except Exception as exc:  # noqa: BLE001 - a dead adapter must not stop the match
            self.error = f"{type(exc).__name__}: {exc}"
            self.done = True
            return
        self.user_text, self.results = None, []
        self.turns += 1
        for k, v in (turn.usage or {}).items():
            if isinstance(v, (int, float)):
                self.usage[k] = self.usage.get(k, 0.0) + v
        if turn.text and turn.text.strip():
            self.db.add_message(sid, self.turns, "assistant", turn.text.strip())
        if not turn.tool_calls:
            if getattr(self.adapter, "_done", False):
                self.done = True
                return
            self.user_text = continue_nudge(
                self.env.remaining() if self.env.phase == "active" else None)
            self.db.add_message(sid, self.turns, "user", self.user_text)
            return
        for call in turn.tool_calls:
            res = self.env.call(call.name, call.args, source="agent", turn=self.turns)
            self.results.append((call, res))
            if is_session_stop_requested(sid):
                self.done = True
                return
        if self.env.phase == "closed":
            self.done = True

    def finalize_row(self) -> dict:
        return {"session_id": self.sid, "agent_id": self.env.agent_id, "team": self.team,
                "mode": self.mode, "turns": self.turns}

    def finalize(self, match_over: bool) -> dict:
        """Close the recorded session with runner semantics (status, diary, memory)."""
        if self.env.end_reason:
            end_reason = self.env.end_reason
        elif self.done and self.error:
            end_reason = "runner_error"
        elif self.done:
            end_reason = "agent_stopped"
        elif match_over:
            end_reason = "match_over"
        else:
            end_reason = "match_time_limit"
        try:
            finish_session(self.cfg, self.db, self.env, self.bridge, self.turns,
                           self.usage, end_reason, self.error)
        except Exception as exc:  # noqa: BLE001 - finalize best-effort, never mask the result
            self.error = self.error or f"finalize: {type(exc).__name__}: {exc}"
        row = self.finalize_row()
        row["end_reason"] = end_reason
        row["error"] = self.error
        return row


def mode_for(teams: List[Dict[str, Any]]) -> str:
    """'arena_solo' for the classic 1v1 duel, 'arena_coop' for everything else."""
    if len(teams) == 2 and all(len(t.get("agents") or []) == 1 for t in teams):
        return "arena_solo"
    return "arena_coop"


def production_world_factory(setup: MatchSetup, scorer) -> ArenaWorldAdapter:
    """Build the REAL multi-agent combat world: bodies, equipment, pellet pool."""
    world = PhysicsWorld(setup.world_cfg)
    for aid, yaw in setup.poses.items():
        world.setup_agent_pose(aid, yaw_degrees=yaw)
        # Combat stance: without a gait command the humanoid's balance loop idles and
        # the rig topples ~2.6 s after spawn (measured 2026-11-17, P1/P2/P4 bisect).
        # 'stand' keeps the balance regulation engaged while stationary.
        world.set_gait("stand", agent=aid)
    return ArenaWorldAdapter(world, scorer=scorer)


def run_arena_match(match_cfg: dict, adapter_factory: Callable[[str, dict], Adapter], *,
                    db=None, root_cfg: dict | None = None,
                    turn_interval_s: float = DEFAULT_TURN_INTERVAL,
                    tick: float = DEFAULT_TICK,
                    match_name: Optional[str] = None) -> MatchResult:
    """Plan, build, launch and play one arena match with the production factories.

    adapter_factory(agent_id, plan_item) returns a fresh adapter per combatant
    (PolicyAdapter for scripted tests, make_adapter(...) for live models). Sessions
    are recorded in ``db`` and finalized when the match ends. The result carries the
    real timeline plus ``end_reason`` ('incapacitated' | 'score_target' | 'timeout')
    and per-session rows.
    """
    match = ArenaMatch(match_cfg)
    match.plan()
    if root_cfg is None:
        root_cfg = load_config()
    if db is None:
        from ..storage import Database
        db = Database(resolve_path(root_cfg, root_cfg["database"]))
    runtime_duration_s = float(match_cfg.get("rules", {}).get("duration_s", 60))
    setup = build_match_setup(match_cfg, root_cfg)
    world_adapter = production_world_factory(setup, match._scorer)
    mode = mode_for(match_cfg.get("teams") or [])

    def _launch(plan_item: dict) -> ArenaSessionHandle:
        aid = str(plan_item["agent_cfg"].get("id"))
        team = plan_item["team"]
        teammates = [a for a in setup.combat_teams.get(team, []) if a != aid]
        opponents = [a for t, aids in setup.combat_teams.items() if t != team for a in aids]
        # Rendering/inspection manifest (recorded with the session): which combatants
        # belong to which team and which world objects are combat equipment. The
        # dashboard reads this from the session's config to draw markers/pellets with
        # real team colours instead of grey primitives (M5.5).
        arena_context={"team": team, "teammates": teammates, "opponents": opponents,
                       "match": {"name": match_name or "match", "mode": mode,
                                 "teams": {t: list(a) for t, a in setup.combat_teams.items()},
                                 "items": [dict(it) for it in setup.world_cfg["combat"]["items"]],
                                 "pellets": int(setup.world_cfg["combat"]["pellets"])}}
        session_cfg = copy.deepcopy(root_cfg)
        # the session lives while the match lives: budget = match time (plus wrapup),
        # so the runner's wall-clock phase machine never closes a live combatant early.
        minutes = max(1, math.ceil(runtime_duration_s / 60.0) + 2)
        session_cfg["session"] = {**session_cfg.get("session", {}), "duration_minutes": minutes}
        adapter = adapter_factory(aid, plan_item)
        bridge = DrivenAdapterBridge(adapter)
        summary = run_session(session_cfg, db, aid, bridge, mode=mode,
                              label=f"arena:{match_name or 'match'}",
                              world=world_adapter.world,
                              arena_context=arena_context)
        handle = ArenaSessionHandle(summary, adapter, bridge, team=team, mode=mode,
                                    turn_interval_s=turn_interval_s, cfg=session_cfg,
                                    db=db, teammates=teammates, opponents=opponents)
        world_adapter.attach_handle(handle, expected=len(setup.agent_ids))
        return handle

    match.play(lambda _wcfg: world_adapter, _launch)

    result = match.result()
    result["end_reason"] = "timeout" if result["timeout"] else "score_target"
    if world_adapter.match_ended is not None:
        # the runtime's measured KO arbitration outranks the hit-count scorer
        result["end_reason"] = "incapacitated"
        if world_adapter.match_ended.get("winner") is not None:
            result["winner"] = world_adapter.match_ended["winner"]
    session_rows = [h.finalize(result["end_reason"] == "incapacitated")
                    for h in world_adapter._handles]
    result["sessions"] = session_rows
    result["adapter"] = world_adapter  # live handles/world for callers (never exported)
    return result


# --------------------------------------------------------------------- CLI
def cmd_arena(args) -> int:
    """CLI: run an arena match from a JSON config file through the production factories."""
    cfg_path = Path(args.config)
    match_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if getattr(args, "root", None):
        root_cfg = load_config(None, {"_root": args.root})
    else:
        root_cfg = load_config()
    from ..storage import Database
    db = Database(resolve_path(root_cfg, root_cfg["database"]) if not args.db else args.db)

    def adapter_factory(agent_id: str, plan_item: dict) -> Adapter:
        from ..agents import make_adapter
        return make_adapter(root_cfg, args.adapter, model=args.model)

    out_path = Path(args.out) if args.out else cfg_path.with_suffix(".report.json")
    result = run_arena_match(match_cfg, adapter_factory, db=db, root_cfg=root_cfg,
                             match_name=cfg_path.stem)
    export_match_report(result, str(out_path))
    print(json.dumps({
        "winner": result.get("winner"),
        "end_reason": result.get("end_reason"),
        "duration": result.get("duration"),
        "team_stats": result.get("team_stats"),
        "agent_stats": result.get("agent_stats"),
        "timeline_events": len(result.get("timeline") or []),
        "sessions": result.get("sessions"),
        "report": str(out_path),
    }, indent=2))
    return 0
