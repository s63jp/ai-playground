"""Physics engine wrapper around MuJoCo.

This module owns the simulated island. It knows nothing about agents, memory or
language models: it exposes a body that can walk, jump, climb, carry and throw,
and it records what physically happened (events, flights, collisions, frames).

Design notes
------------
* Time only advances while an action is being carried out, so a run is fully
  reproducible from its sequence of actions.
* The body is a non-rotating box on three slide joints. Its geom is frictionless;
  all horizontal traction comes from a force-limited walking controller that only
  acts while the body is supported. In the air the body is purely ballistic.
* A carried object is attached rigidly: its collisions are disabled and its mass
  is added to the body, so everything the legs do has to move the load as well.
* Throwing gives the object a velocity relative to the body and applies the
  opposite momentum to the body.
"""
from __future__ import annotations

import math
from typing import Any, Callable
import os


import importlib.util, os
# Ensure offscreen rendering for headless environments

# Load real MuJoCo bindings from the virtual environment site-packages, bypassing the local stub.
_mujoco_pkg = os.path.join(os.path.dirname(__file__), '..', '..', '.venv', 'Lib', 'site-packages', 'mujoco', '__init__.py')
_spec = importlib.util.spec_from_file_location('mujoco_real', _mujoco_pkg)
_mujoco = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mujoco)
# Expose as "mujoco" for backward compatibility
mujoco = _mujoco

import numpy as np

from .mathutil import euler_deg_to_quat, heading_deg, quat_to_euler_deg, quat_to_mat, rnd
from .spec import PARK_ORIGIN, ObjectSpec, build_mjcf, build_specs

RAY_GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # world + active objects only
MIN_CATCH_DELAY = 0.05   # an object must be free for at least this long before catch() can grab it


def _plain(obj):
    """Convert numpy containers to plain Python at full precision (for exact checkpoints)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


class ActionError(Exception):
    """An action that could not be carried out. Never silently swallowed."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class PhysicsWorld:
    def __init__(self, cfg: dict,
                 event_sink: Callable[[dict], None] | None = None,
                 frame_sink: Callable[[dict], None] | None = None):
        self.cfg = cfg
        self.wc = cfg["world"]
        self.ac = cfg["agent_body"]
        self.rc = cfg["recording"]
        self.rule = cfg["rules"]["ring_generation"]
        self.event_sink = event_sink
        self.frame_sink = frame_sink
        self._pd = {}  # joint PD controllers: name -> dict with target, kp, kd
        self._step_callbacks: list[Callable[['PhysicsWorld'], None]] = []  # step callbacks
        self._gait_controller = None  # optional GaitController instance
        self.fallen_settle_s = 0.3  # sustained-condition window before agent_fallen() is True
        self._fallen_since = None  # first sim time the fallen condition was observed

        spec_list = build_specs(cfg)
        self.specs: dict[str, ObjectSpec] = {s.id: s for s in spec_list}
        self.xml = build_mjcf(cfg, spec_list)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.dt = float(self.m.opt.timestep)

        m = self.m
        self.agent_bid = m.body("agent").id
        try:
            self.agent_gid = m.geom("g:agent").id
        except Exception:
            self.agent_gid = None
        self.aq = int(m.joint("agent_x").qposadr[0])
        self.av = int(m.joint("agent_x").dofadr[0])
        self.hw = float(self.ac["width"]) / 2
        self.hh = float(self.ac["height"]) / 2
        self.base_mass = float(self.ac["mass"])
        self.half = float(self.wc["island"]["half_extent"])
        self.sea = float(self.wc["island"]["sea_level"])

        # Object-level index: every geom maps to one named thing.
        self.onames: list[str] = ["agent", "boundary"]
        self.oidx: dict[str, int] = {n: i for i, n in enumerate(self.onames)}
        for s in spec_list:
            self.oidx[s.id] = len(self.onames)
            self.onames.append(s.id)
        self.geom_oidx = np.zeros(m.ngeom, dtype=np.int64)
        self.gids: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            name = m.geom(g).name
            if name == "g:agent":
                key = "agent"
            elif name.startswith("g:boundary"):
                key = "boundary"
            else:
                key = name[2:].split(":")[0]
            if key not in self.oidx:
                key = 'agent'
            self.geom_oidx[g] = self.oidx[key]
            self.gids.setdefault(key, []).append(g)
        self.agent_oi = self.oidx["agent"]

        self.movable: list[str] = [s.id for s in spec_list if s.movable]
        self.bid: dict[str, int] = {}
        self.qadr: dict[str, int] = {}
        self.dadr: dict[str, int] = {}
        for i, oid in enumerate(self.movable):
            self.bid[oid] = m.body(oid).id
            j = m.joint(f"j:{oid}")
            self.qadr[oid] = int(j.qposadr[0])
            self.dadr[oid] = int(j.dofadr[0])
        self.slot = {oid: i for i, oid in enumerate(self.movable)}
        # Per-object linear dof start for velocity lookups (-1 for static things).
        self.lin_dof = np.full(len(self.onames), -1, dtype=np.int64)
        self.lin_dof[self.agent_oi] = self.av
        for oid in self.movable:
            self.lin_dof[self.oidx[oid]] = self.dadr[oid]

        self.active: set[str] = {s.id for s in spec_list if s.movable and not s.latent}
        self.latent: list[str] = [s.id for s in spec_list if s.movable and s.latent]
        self.lost: set[str] = set()
        self.generated: list[str] = []

        # Controller / body state
        self.held: str | None = None
        self.held_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.facing = 0.0
        self.target_vel = np.zeros(2)
        self.grounded = False
        self.support: list[str] = []
        self.side_contacts: list[str] = []
        self._jump_left = 0
        self._jumped = False
        self._jump_support: list[str] = []
        self._zero3 = np.zeros(3)
        self._climb: dict | None = None
        self._air: dict | None = None
        self.last_air: dict | None = None

        # Recording state
        self.steps = 0
        self.seq = 0
        self.events: list[dict] = []
        self.flights: dict[str, dict] = {}
        self.completed_flights: list[dict] = []
        self._flight_seq = 0
        self._pairs: set[tuple[int, int]] = set()
        self._pair_last: dict[tuple[int, int], float] = {}
        self._pending: list[dict] = []
        self._prev_qvel = np.zeros(m.nv)
        self._last_frame: dict[str, list[float]] = {}
        self._frame_count = 0
        self._recording = False

        # World-rule state
        self.rule_charges = int(self.rule["charges"]) if self.rule.get("enabled") else 0
        self._rule_armed = True
        self._rule_since: float | None = None

        for oid in self.latent:
            self._park(oid)
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(0.6 / self.dt)):
            self._step()
        self.d.time = 0.0
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.steps = 0
        self._air = None
        self.last_air = None
        self._recording = True
        self.initial_positions = {oid: self.obj_pos(oid).tolist() for oid in sorted(self.active)}
        self.emit("world_ready", objects=len(self.active), agent=rnd(self.agent_pos()))
        # Humanoid-only spawn reset (see _humanoid_spawn_reset): limb joints
        # only, feet-on-ground; box worlds' block freejoint slots stay untouched.
        if self.ac.get("type") == "humanoid":
            self._humanoid_spawn_reset(0.01)


    # ------------------------------------------------------------------ basics
    @property
    def time(self) -> float:
        return float(self.d.time)

    def agent_pos(self) -> np.ndarray:
        return self.d.qpos[self.aq:self.aq + 3] + self._agent_origin

    @property
    def _agent_origin(self) -> np.ndarray:
        if not hasattr(self, "_origin_cache"):
            self._origin_cache = np.array(self.m.body_pos[self.agent_bid], dtype=float)
        return self._origin_cache

    def agent_vel(self) -> np.ndarray:
        """Current linear velocity of the agent (slide joints)."""
        return self.d.qvel[self.av:self.av + 3].copy()


    def set_joint_target(self, joint: str, position: float, kp: float = 15.0, kd: float = 1.5) -> None:
        """Register a PD controller for a named joint.
        The target position (radians) will be tracked each simulation step.
        kp – proportional gain, kd – derivative gain.
        """
        self._pd[joint] = {"target": position, "kp": kp, "kd": kd}
        # Removed hip forward displacement hack – movement now relies on PD torques and contact forces.

        self.emit("joint_control", hidden=None, joint=joint, target=position, kp=kp, kd=kd)


    def total_mass(self) -> float:
        """Total mass of the agent (agent container + humanoid limbs + held load).

        The humanoid's figure hangs off the container (waist joint), so its
        real mass is the subtree mass; for the box agent the subtree equals
        body_mass, identical to the HEAD semantics.
        """
        m = float(self.m.body_subtreemass[self.agent_bid])
        if m <= 1e-6:
            m = float(self.m.body_mass[self.agent_bid])
        return float(m)



    def feet(self) -> float:
        """Height of the agent's feet above the ground plane (z coordinate minus half height)."""
        return float(self.agent_pos()[2] - self.hh)

    def obj_pos(self, oid: str) -> np.ndarray:
        """Position (x, y, z) of object `oid`."""
        a = self.qadr[oid]
        return self.d.qpos[a:a + 3].copy()

    def obj_quat(self, oid: str) -> np.ndarray:
        a = self.qadr[oid]
        return self.d.qpos[a + 3:a + 7].copy()

    def obj_vel(self, oid: str) -> np.ndarray:
        a = self.dadr[oid]
        return self.d.qvel[a:a + 3].copy()

    def obj_angvel(self, oid: str) -> np.ndarray:
        a = self.dadr[oid]
        return self.d.qvel[a + 3:a + 6].copy()

    def obj_speed(self, oid: str) -> float:
        return float(np.linalg.norm(self.obj_vel(oid)))

    def half_extent_z(self, oid: str, quat=None) -> float:
        s = self.specs[oid]
        q = self.obj_quat(oid) if quat is None else quat
        if s.shape == "sphere":
            return float(s.dimensions["radius"])
        R = quat_to_mat(q)
        if s.shape == "box":
            d = s.dimensions
            return float(abs(R[2, 0]) * d["x"] / 2 + abs(R[2, 1]) * d["y"] / 2 + abs(R[2, 2]) * d["z"] / 2)
        nz = abs(float(R[2, 2]))
        return float(s.dimensions["major_radius"] * math.sqrt(max(0.0, 1 - nz * nz)) + s.dimensions["tube_radius"])

    def bounding_radius(self, oid: str) -> float:
        s = self.specs[oid]
        if s.shape == "sphere":
            return float(s.dimensions["radius"])
        if s.shape == "box":
            d = s.dimensions
            return 0.5 * math.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
        return float(s.dimensions["major_radius"] + s.dimensions["tube_radius"])

    def emit(self, etype: str, hidden: dict | None = None, **data: Any) -> dict | None:
        if not self._recording:
            return None
        self.seq += 1
        ev = {"seq": self.seq, "t": round(self.time, 4), "step": self.steps, "type": etype, "data": rnd(data)}
        if hidden:
            ev["hidden"] = rnd(hidden)
        self.events.append(ev)
        if self.event_sink:
            self.event_sink(ev)
        return ev

    # ------------------------------------------------------------------ stepping
    def _step(self) -> None:
        d = self.d
        v0 = self.av
        d.qfrc_applied[:] = 0.0

        if self._climb is not None:
            self._advance_climb()
        else:
            fx = fy = fz = 0.0
            reaction_on: list[str] = []
            if self.grounded and self.ac.get("type") != "humanoid":
                # Root traction is the BOX agent's locomotion model: a
                # horizontal force at the root, paired with a frictionless
                # base geom. The humanoid walks CONTACT-DRIVEN instead:
                # stance-foot friction + alternating hip torque (see
                # playground/locomotion.py) — applying the root traction to
                # an articulated leg rig folds the legs in (~240 Nm couple,
                # hips slam their ±90° limits, z 0.81 -> 0.21 kneel-drag).
                M = self.total_mass()
                tau = float(self.ac["traction_tau"])
                fx = M * (self.target_vel[0] - d.qvel[v0]) / tau
                fy = M * (self.target_vel[1] - d.qvel[v0 + 1]) / tau
                mag = math.hypot(fx, fy)
                lim = float(self.ac["walk_force"])
                if mag > lim:
                    fx *= lim / mag
                    fy *= lim / mag
                reaction_on = [o for o in self.support if o in self.active]
            if self._jump_left > 0:
                fz = float(self.ac["jump_force"])
                self._jump_left -= 1
                reaction_on = [o for o in self._jump_support if o in self.active and o != self.held]
            d.qfrc_applied[v0] = fx
            d.qfrc_applied[v0 + 1] = fy
            d.qfrc_applied[v0 + 2] = fz
            # Newton's third law: whatever the legs push against is pushed back.
            if reaction_on and (fx or fy or fz):
                p = self.agent_pos()
                point = np.array([p[0], p[1], p[2] - self.hh])
                share = np.array([-fx, -fy, -fz]) / len(reaction_on)
                for oid in reaction_on:
                    mujoco.mj_applyFT(self.m, d, share, self._zero3, point, self.bid[oid], d.qfrc_applied)
        if self._gait_controller is not None:
            self._gait_controller.update(self)
        # PD joint controllers (set_joint_target): applied at method level so
        # direct joint targeting works with or without a gait controller.
        for jname in list(self._pd.keys()):
            cfgj = self._pd[jname]
            try:
                jid = self.m.joint(jname).id
            except (ValueError, KeyError):
                # Joint does not exist in this model (e.g. box agent): drop the
                # target and record it instead of corrupting the step.
                del self._pd[jname]
                self.emit("joint_target_dropped", joint=jname)
                continue
            qaddr = int(self.m.jnt_qposadr[jid])
            dofadr = int(self.m.jnt_dofadr[jid])
            target = cfgj.get('target', 0.0)
            kp = cfgj.get('kp', 0.0)
            kd = cfgj.get('kd', 0.0)
            qpos = d.qpos[qaddr]
            qvel = d.qvel[dofadr]
            torque = kp * (target - qpos) - kd * qvel
            d.qfrc_applied[dofadr] += torque
        if self.held:
            self._pin_held()
        # NOTE: the former per-step 'torso quat reset' write at qpos[aq+3:aq+7] is REMOVED;
        # it corrupted every box-agent world (the write landed on block_01's qpos slot)
        # and teleported the humanoid torso root each step. Posture comes from spine PD
        # targets (see playground/locomotion.py), never from qpos writes.
        # --- integrate the world (restore of the HEAD integration block; the
        # humanoid edits accidentally DROPPED mj_step, freezing all physics:
        # every world ran in time 0.0, no falls/flights/collisions ever happened) ---
        self._prev_qvel[:] = d.qvel
        mujoco.mj_step(self.m, d)
        if self.held:
            self._pin_held()
        self.steps += 1
        self._post_step()
        # invoke registered step callbacks
        for cb in self._step_callbacks:
            try:
                cb(self)
            except Exception as e:
                self.emit('step_callback_error', error=str(e))

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._step()

    # ------------------------------------------------------------------ API
    def register_step_callback(self, fn: Callable[["PhysicsWorld"], None]) -> None:
        """Register a callable to be invoked after each simulation step.
        The function receives the PhysicsWorld instance.
        """
        self._step_callbacks.append(fn)

    def set_gait(self, mode: str, speed: float = 1.0, heading: float | None = None) -> None:
        """Configure the gait controller.

        mode: one of 'walk', 'run', 'stand', 'stop', 'crouch'.
        speed: target forward speed (m/s).
        heading: optional yaw angle (degrees) to rotate the movement direction.

        Physics honesty: this ONLY configures joint-space PD targets. Forward motion
        must emerge from foot–ground contact; qpos/qvel are never written here.
        """
        from playground.locomotion import GaitController, DEFAULTS
        if self._gait_controller is None:
            self._gait_controller = GaitController(self, DEFAULTS.copy())
        self._gait_controller.configure(mode, speed=speed, heading=heading)
        self.emit("gait_change", mode=mode, speed=float(speed),
                  heading=(float(heading) if heading is not None else None))


    def stand_up(self) -> None:
        """Attempt a physics-based recovery from a fallen pose.

        No teleports: qpos/qvel are never written. The gait controller drives a
        scheduled sequence of joint-space PD targets (knee/hip flexion, push-off,
        spine bracing, posture restore). Emits 'recovery_started' now; the
        controller emits 'recovery_complete' or 'recovery_failed' from its per-step
        update once the agent is upright (agent_fallen() False) or the schedule
        times out (~6 s of sim time).
        """
        from playground.locomotion import GaitController, DEFAULTS
        if self._gait_controller is None:
            self._gait_controller = GaitController(self, DEFAULTS.copy())
        self._gait_controller.begin_recovery()
        self.emit("recovery_started")

    def _humanoid_spawn_reset(self, z: float = 0.01) -> None:
        """Reset the humanoid's OWN joints at spawn time (agent-type gated).

        Limb hinge qpos go to 0 (neutral posture) and all agent dofs to zero
        velocity. The reset NEVER touches qpos past the agent's own limb
        joints: after the 3 slide joints the limb hinge slots follow, and the
        object freejoint slots after those belong to block_01.. — the old
        unconditional ``qpos[aq+3:aq+7]`` write teleported block_01 to (1,0,0)
        in box worlds (Nov 17 bug log).

        ``z`` is the raw slide-z so the feet sit just above the ground: with
        the spec.py torso offset fix the feet land at agent_pos z, i.e. at
        qpos_z, so 0.01 leaves a 1 cm settle drop instead of the old 0.8 m
        free-fall.
        """
        for j in range(self.m.njnt):
            if int(self.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                qadr = int(self.m.jnt_qposadr[j])
                if qadr >= self.aq + 3:
                    self.d.qpos[qadr] = 0.0
                    self.d.qvel[int(self.m.jnt_dofadr[j])] = 0.0
        self.d.qpos[self.aq:self.aq + 3] = np.array([0.0, 0.0, float(z)])
        self.d.qvel[self.av:self.av + 3] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def agent_fallen(self) -> bool:
        """True when the humanoid has definitively fallen.

        The up-vector is measured from the torso body's world rotation matrix
        (d.xmat) — never from synthetic/or-quaternion slots. A fall requires the
        condition to persist for ``fallen_settle_s`` sim seconds so a stumble
        during gait is not misread as a fall (settle window: training termination,
        recovery behaviour and reward evaluation all depend on this).
        """
        try:
            bid = self.m.body("torso").id
        except Exception:
            bid = self.agent_bid
        up_z = float(np.asarray(self.d.xmat[bid], dtype=float).reshape(3, 3)[2, 2])
        # Height threshold. The humanoid cannot tip (rigid slide-root column,
        # no pitch dof) so "fallen" means leg collapse: the folded-leg floor of
        # the rig is ~0.42 m = 0.52x the 0.81 m stand height, while standing is
        # 0.80-0.81 m and walking holds 0.78-0.80 m.
        # Other agent types keep the legacy 40% half-height rule.
        if self.ac.get("type") == "humanoid":
            # stand height is ~1.0*hh (the container rides at hh); the
            # folded-leg floor is ~0.52*hh, so 0.55*hh separates a sit/crumple
            # from standing (0.80) and walking (0.78-0.80).
            height_ok = float(self.agent_pos()[2]) > self.hh * 0.55
        else:
            height_ok = float(self.agent_pos()[2]) > self.hh * 0.4
        fallen_now = (up_z < 0.5) or (not height_ok)
        if not fallen_now:
            self._fallen_since = None
            return False
        if self._fallen_since is None:
            self._fallen_since = self.time
        return (self.time - self._fallen_since) >= float(self.fallen_settle_s)

    def agent_skeleton(self) -> list[dict]:
        """Return a list of body parts info for the current agent.
        For a humanoid this includes torso, hips, knees, etc.; for a box returns a single entry.
        """
        # Use spec list to order bodies; torso first
        bodies = []
        # The main agent body is always present
        def body_info(name, qaddr):
            pos = self.d.qpos[qaddr:qaddr+3].tolist()
            quat = self.d.qpos[qaddr+3:qaddr+7].tolist()
            return {"name": name, "pos": pos, "quat": quat}
        # torso (agent body)
        bodies.append(body_info('torso', self.aq))
        # movable limbs (if humanoid)
        for oid in self.movable:
            qaddr = self.qadr[oid]
            bodies.append(body_info(oid, qaddr))
        return bodies

    def spawn_object(self, oid: str, pos: list[float], quat: list[float] | None = None) -> None:
        """Place an existing object at a new position (and optional orientation)."""
        if oid not in self.oidx:
            raise ValueError(f"Object {oid} not known")
        qaddr = self.qadr[oid]
        self.d.qpos[qaddr:qaddr+3] = np.array(pos, dtype=float)
        if quat is not None:
            self.d.qpos[qaddr+3:qaddr+7] = np.array(quat, dtype=float)
        self.emit('object_spawned', object=oid, pos=pos, quat=quat)

    def set_object_velocity(self, oid: str, vel: list[float], angular: list[float] | None = None) -> None:
        """Set linear (and optional angular) velocity of an object."""
        if oid not in self.oidx:
            raise ValueError(f"Object {oid} not known")
        vaddr = self.dadr[oid]
        self.d.qvel[vaddr:vaddr+3] = np.array(vel, dtype=float)
        if angular is not None:
            self.d.qvel[vaddr+3:vaddr+6] = np.array(angular, dtype=float)
        self.emit('object_velocity_set', object=oid, vel=vel, angular=angular)

    def _pin_held(self) -> None:
        oid = self.held
        a, v = self.qadr[oid], self.dadr[oid]
        p = self.agent_pos()
        if self.ac.get("type") == "humanoid":
            # The humanoid carries the block in front of the figure — the
            # figure faces local +Y rotated by the waist yaw, so the carry
            # offset rotates with it. The box pin position (feet + hold_height
            # = 1.2 m under the pelvis centre) would put the block INSIDE the
            # torso geom (0.90-1.40 m band at the centre line).
            yaw_q = float(self.d.qpos[int(self.m.jnt_qposadr[self.m.joint("waist_yaw").id])])
            fy = float(self.ac.get("hold_forward", 0.30))
            self.d.qpos[a:a + 3] = (p[0] - fy * math.sin(yaw_q),
                                    p[1] + fy * math.cos(yaw_q),
                                    p[2] - self.hh + float(self.ac.get("hold_height_humanoid", 0.84)))
            self.d.qpos[a + 3:a + 7] = self.held_quat
            self.d.qvel[v:v + 3] = self.d.qvel[self.av:self.av + 3]
            self.d.qvel[v + 3:v + 6] = 0.0
            return
        self.d.qpos[a:a + 3] = (p[0], p[1], p[2] - self.hh + float(self.ac["hold_height"]))
        self.d.qpos[a + 3:a + 7] = self.held_quat
        self.d.qvel[v:v + 3] = self.d.qvel[self.av:self.av + 3]
        self.d.qvel[v + 3:v + 6] = 0.0

    def _post_step(self) -> None:
        d = self.d
        ncon = d.ncon
        grounded = False
        support: set[str] = set()
        side: set[str] = set()
        pairs: set[tuple[int, int]] = set()
        if ncon:
            # Vectorised: contacts are reduced to unique object pairs in numpy, and only
            # the handful that involve the body are examined individually.
            oi = self.geom_oidx[d.contact.geom[:ncon]]
            ca, cb = oi[:, 0], oi[:, 1]
            keep = ca != cb
            if keep.any():
                n = len(self.onames)
                codes = np.unique(np.minimum(ca, cb)[keep] * n + np.maximum(ca, cb)[keep])
                pairs = {(c // n, c % n) for c in codes.tolist()}
                ag = self.agent_oi
                for i in np.nonzero(keep & ((ca == ag) | (cb == ag)))[0].tolist():
                    a, b = int(ca[i]), int(cb[i])
                    nz = float(d.contact.frame[i, 2])
                    other = self.onames[b if a == ag else a]
                    up = -nz if a == ag else nz
                    if up > 0.7:
                        grounded = True
                        support.add(other)
                    elif abs(nz) < 0.3:
                        side.add(other)
        if self._climb is None:
            if self.grounded and not grounded:
                p = self.agent_pos()
                self._air = {"t0": self.time, "feet0": float(p[2] - self.hh), "pos0": p.copy(),
                             "max_feet": float(p[2] - self.hh), "cause": "jump" if self._jumped else "fall",
                             "v0": self.agent_vel()}
            elif grounded and not self.grounded and self._air is not None:
                self._finish_air(sorted(support))
            if self._air is not None:
                f = self.feet()
                if f > self._air["max_feet"]:
                    self._air["max_feet"] = f
        if grounded and self._jump_left == 0:
            self._jumped = False
        self.grounded = grounded
        self.support = sorted(support)
        self.side_contacts = sorted(side)

        if self._recording:
            new = pairs - self._pairs
            if new:
                self._start_collisions(sorted(new))
            if self._pending:
                self._advance_collisions()
            if self.flights:
                self._advance_flights(pairs)
        self._pairs = pairs

        if self.steps % 25 == 0:
            self._check_lost()
        if self.steps % 50 == 0 and self.rule_charges > 0:
            self._check_rule()
        if self._recording and self.steps % int(self.rc["frame_interval"]) == 0:
            self._record_frame()

    def _finish_air(self, landed_on: list[str]) -> None:
        a = self._air
        p = self.agent_pos()
        feet_now = float(p[2] - self.hh)
        min_h = float(self.rc.get("min_air_height", 0.03))
        if (a["cause"] == "fall" and self.time - a["t0"] < float(self.rc.get("min_air_time", 0.15))
                and abs(feet_now - a["feet0"]) < min_h and a["max_feet"] - a["feet0"] < min_h):
            # Sub-threshold contact chatter (e.g. the rebound after a landing): not a fall.
            self._air = None
            return
        rec = {
            "cause": a["cause"],
            "air_time": self.time - a["t0"],
            "takeoff_position": a["pos0"], "landing_position": p.copy(),
            "takeoff_velocity": a["v0"],
            "takeoff_feet_height": a["feet0"], "max_feet_height": a["max_feet"],
            "apex_above_takeoff": a["max_feet"] - a["feet0"],
            "horizontal_distance": float(np.hypot(p[0] - a["pos0"][0], p[1] - a["pos0"][1])),
            "impact_speed": float(np.linalg.norm(self._prev_qvel[self.av:self.av + 3])),
            "landed_on": landed_on, "carried": self.held, "total_mass": self.total_mass(),
            "t_start": a["t0"], "t_end": self.time,
        }
        self.last_air = rnd(rec)
        self._air = None
        self.emit("agent_landing", **rec)

    # ------------------------------------------------------------------ collisions
    def _lin_vel_prev(self, oi: int) -> np.ndarray:
        a = self.lin_dof[oi]
        if a < 0:
            return np.zeros(3)
        return self._prev_qvel[a:a + 3]

    def _start_collisions(self, new_pairs: list[tuple[int, int]]) -> None:
        thr = float(self.rc["collision_speed_threshold"])
        cooldown = float(self.rc.get("collision_cooldown", 0.25))
        for a, b in new_pairs:
            rel = self._lin_vel_prev(a) - self._lin_vel_prev(b)
            speed = float(np.linalg.norm(rel))
            if speed < thr:
                continue
            # Contact chatter (the same two things bouncing or scraping) is one collision, not dozens.
            last = self._pair_last.get((a, b))
            self._pair_last[(a, b)] = self.time
            if last is not None and self.time - last < cooldown:
                continue
            pos = None
            oi = self.geom_oidx[self.d.contact.geom[:self.d.ncon]]
            for i in range(self.d.ncon):
                x, y = int(oi[i, 0]), int(oi[i, 1])
                if (x == a and y == b) or (x == b and y == a):
                    pos = np.array(self.d.contact.pos[i])
                    break
            self._pending.append({"a": a, "b": b, "t": self.time, "speed": speed, "pos": pos,
                                  "impulse": 0.0, "peak": 0.0, "left": 10})

    def _advance_collisions(self) -> None:
        d = self.d
        f6 = np.zeros(6)
        oi = self.geom_oidx[d.contact.geom[:d.ncon]] if d.ncon else np.zeros((0, 2), dtype=np.int64)
        done = []
        for pc in self._pending:
            total = 0.0
            for i in range(d.ncon):
                x, y = int(oi[i, 0]), int(oi[i, 1])
                if (x == pc["a"] and y == pc["b"]) or (x == pc["b"] and y == pc["a"]):
                    mujoco.mj_contactForce(self.m, d, i, f6)
                    total += abs(float(f6[0]))
            pc["impulse"] += total * self.dt
            pc["peak"] = max(pc["peak"], total)
            pc["left"] -= 1
            if pc["left"] <= 0:
                done.append(pc)
        for pc in done:
            self._pending.remove(pc)
            self.emit("collision", objects=[self.onames[pc["a"]], self.onames[pc["b"]]],
                      relative_speed=pc["speed"], impulse=pc["impulse"], peak_force=pc["peak"],
                      position=pc["pos"], t_contact=pc["t"])

    # ------------------------------------------------------------------ flights
    def _begin_flight(self, oid: str, kind: str, vel: np.ndarray, extra: dict | None = None) -> dict:
        self._flight_seq += 1
        p = self.obj_pos(oid)
        fl = {"flight_id": self._flight_seq, "object": oid, "kind": kind, "t0": self.time,
              "launch_position": p.copy(), "launch_velocity": np.array(vel, dtype=float),
              "launch_speed": float(np.linalg.norm(vel)), "max_height": float(p[2]), "max_height_t": self.time,
              "first_impact": None, "impacts": 0, "_contact": False, "_quiet": 0.0, "done": False}
        if extra:
            fl.update(extra)
        self.flights[oid] = fl
        return fl

    def _advance_flights(self, pairs: set[tuple[int, int]]) -> None:
        rest_v = float(self.rc["rest_speed"])
        rest_t = float(self.rc["rest_time"])
        for oid in sorted(self.flights):
            fl = self.flights[oid]
            oi = self.oidx[oid]
            p = self.obj_pos(oid)
            if p[2] > fl["max_height"]:
                fl["max_height"] = float(p[2])
                fl["max_height_t"] = self.time
            touching = sorted(self.onames[b if a == oi else a] for a, b in pairs if a == oi or b == oi)
            if touching and not fl["_contact"]:
                fl["impacts"] += 1
                if fl["first_impact"] is None:
                    v = self._prev_qvel[self.dadr[oid]:self.dadr[oid] + 3]
                    fl["first_impact"] = {"t": self.time, "position": p.copy(), "with": touching,
                                          "speed": float(np.linalg.norm(v)), "velocity": np.array(v)}
            fl["_contact"] = bool(touching)
            if fl["first_impact"] is not None:
                quiet = self.obj_speed(oid) < rest_v and float(np.linalg.norm(self.obj_angvel(oid))) < 0.5
                fl["_quiet"] = fl["_quiet"] + self.dt if quiet else 0.0
                if fl["_quiet"] >= rest_t:
                    self._end_flight(oid, "rest")

    def _end_flight(self, oid: str, outcome: str) -> dict | None:
        fl = self.flights.pop(oid, None)
        if fl is None:
            return None
        p = self.obj_pos(oid)
        fi = fl["first_impact"]
        lp = fl["launch_position"]
        rec = {
            "flight_id": fl["flight_id"], "object": oid, "kind": fl["kind"], "outcome": outcome,
            "launch_position": lp, "launch_velocity": fl["launch_velocity"], "launch_speed": fl["launch_speed"],
            "agent_velocity_at_release": fl.get("agent_velocity"),
            "agent_grounded_at_release": fl.get("agent_grounded"),
            "force": fl.get("force"),
            "max_height": fl["max_height"], "max_height_above_launch": fl["max_height"] - float(lp[2]),
            "time_to_max_height": fl["max_height_t"] - fl["t0"],
            "flight_time": (fi["t"] - fl["t0"]) if fi else None,
            "first_impact_position": fi["position"] if fi else None,
            "first_impact_with": fi["with"] if fi else None,
            "impact_speed": fi["speed"] if fi else None,
            "horizontal_range": float(np.hypot(fi["position"][0] - lp[0], fi["position"][1] - lp[1])) if fi else None,
            "impacts": fl["impacts"],
            "final_position": p, "total_time": self.time - fl["t0"],
            "resting_on": self.support_graph().get(oid, []) if outcome == "rest" else [],
            "t_start": fl["t0"], "t_end": self.time,
        }
        rec = rnd(rec)
        self.completed_flights.append(rec)
        self.emit("flight_end", **rec)
        return rec

    # ------------------------------------------------------------------ world bookkeeping
    def _check_lost(self) -> None:
        for oid in sorted(self.active):
            if oid == self.held:
                continue
            p = self.obj_pos(oid)
            if p[2] < self.sea:
                self.emit("object_lost", object=oid, last_position=p, reason="fell into the sea")
                if oid in self.flights:
                    self._end_flight(oid, "lost")
                self.active.discard(oid)
                self.lost.add(oid)
                self._park(oid)

    def _park(self, oid: str) -> None:
        """Take a body out of the world: no collisions, gravity cancelled, parked far away."""
        a, v = self.qadr[oid], self.dadr[oid]
        self.d.qpos[a:a + 3] = (PARK_ORIGIN[0] + 2.0 * self.slot[oid], PARK_ORIGIN[1], PARK_ORIGIN[2])
        self.d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        self.d.qvel[v:v + 6] = 0.0
        self._set_collidable(oid, False)
        b = self.bid[oid]
        self.d.xfrc_applied[b, :] = 0.0
        self.d.xfrc_applied[b, 2] = float(self.m.body_mass[b]) * float(self.wc["gravity"])

    def _unpark(self, oid: str) -> None:
        self.d.xfrc_applied[self.bid[oid], :] = 0.0
        self._set_collidable(oid, True)

    def _set_collidable(self, oid: str, on: bool) -> None:
        for g in self.gids[oid]:
            self.m.geom_contype[g] = 1 if on else 0
            self.m.geom_conaffinity[g] = 1 if on else 0
            self.m.geom_group[g] = 0 if on else 2
        # MuJoCo's broadphase filters on body-level masks that are only rebuilt from the
        # geoms by mj_setConst. Keep them in sync here, otherwise a just-released object
        # is invisible to collision checks until the next mass update.
        b = self.bid[oid]
        self.m.body_contype[b] = 1 if on else 0
        self.m.body_conaffinity[b] = 1 if on else 0

    def _set_agent_mass(self, mass: float) -> None:
        # The humanoid's figure hangs off the pelvis container; rewriting the
        # container mass for a carried load would inflate the walked figure to
        # ~94 kg and fold the legs through the floor mid-stride. The humanoid
        # carries via the hold pin only (the load rides the container's
        # slide-mass math as a visual rider, which keeps every leg contact
        # reaction in its calibrated regime); the box agent keeps the legacy
        # behaviour below.
        if self.ac.get("type") == "humanoid":
            return
        d = self.d
        qpos, qvel, t = d.qpos.copy(), d.qvel.copy(), d.time
        self.m.body_mass[self.agent_bid] = mass
        mujoco.mj_setConst(self.m, d)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.time = t
        mujoco.mj_forward(self.m, d)

    def _check_rule(self) -> None:
        """Hidden world rule: see docs/RESEARCHER_NOTES.md. Never described to agents."""
        ring = self.wc["ring"]["id"]
        target = None
        if ring in self.active and ring != self.held and self.latent:
            rp = self.obj_pos(ring)
            flat = abs(float(quat_to_mat(self.obj_quat(ring))[2, 2])) > 0.95
            if flat and self.obj_speed(ring) < 0.05:
                rs = self.specs[ring].dimensions
                inner = rs["major_radius"] - rs["tube_radius"]
                for oid in sorted(self.active):
                    s = self.specs[oid]
                    if s.kind != "block" or oid == self.held:
                        continue
                    bp = self.obj_pos(oid)
                    if (math.hypot(bp[0] - rp[0], bp[1] - rp[1]) < inner - 0.3
                            and abs(bp[2] - rp[2]) < 0.3 and self.obj_speed(oid) < 0.05):
                        target = oid
                        break
        if target is None:
            self._rule_since = None
            self._rule_armed = True
            return
        if not self._rule_armed:
            return
        if self._rule_since is None:
            self._rule_since = self.time
            return
        if self.time - self._rule_since < float(self.rule["dwell"]):
            return
        new = self.latent.pop(0)
        bp = self.obj_pos(target)
        top, _ = self.ray_down(bp[0], bp[1], bp[2] + 30.0)
        z = (top if top is not None else bp[2] + 0.25) + self.specs[new].dimensions["z"] / 2 + 0.03
        a, v = self.qadr[new], self.dadr[new]
        self.d.qpos[a:a + 3] = (bp[0], bp[1], z)
        self.d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        self.d.qvel[v:v + 6] = 0.0
        self._unpark(new)
        self.active.add(new)
        self.generated.append(new)
        self.rule_charges -= 1
        self._rule_armed = False
        self._rule_since = None
        mujoco.mj_forward(self.m, self.d)
        self.emit("object_appeared", object=new, position=[bp[0], bp[1], z],
                  hidden={"cause": "ring_generation", "ring_around": target, "charges_left": self.rule_charges})

    def _record_frame(self, key: bool = False) -> None:
        if not self.frame_sink and not key:
            # Still keep counters consistent even without a sink.
            self._frame_count += 1
            return
        is_key = key or self._frame_count % int(self.rc["keyframe_every"]) == 0
        objs: dict[str, list[float]] = {}
        for oid in sorted(self.active):
            a = self.qadr[oid]
            pose = [round(float(x), 3) for x in self.d.qpos[a:a + 7]]
            if is_key or self._last_frame.get(oid) != pose:
                objs[oid] = pose
                self._last_frame[oid] = pose
        if is_key:
            for oid in list(self._last_frame):
                if oid not in self.active:
                    del self._last_frame[oid]
        frame = {"i": self._frame_count, "t": round(self.time, 3), "key": is_key,
                 "agent": [round(float(x), 3) for x in self.agent_pos()],
                 "held": self.held, "objects": objs}
        if is_key:
            frame["present"] = sorted(self.active)
        self._frame_count += 1
        if self.frame_sink:
            self.frame_sink(frame)

    # ------------------------------------------------------------------ queries
    def ray_down(self, x: float, y: float, z_from: float, include_agent: bool = False):
        gid = np.zeros(1, dtype=np.int32)
        mask = RAY_GROUPS.copy()
        if include_agent:
            mask[1] = 1
        dist = mujoco.mj_ray(self.m, self.d, np.array([x, y, z_from], dtype=float),
                             np.array([0.0, 0.0, -1.0]), mask, 1, -1, gid)
        if dist < 0 or gid[0] < 0:
            return None, None
        return float(z_from - dist), self.onames[int(self.geom_oidx[int(gid[0])])]

    def support_graph(self) -> dict[str, list[str]]:
        """upper object -> things it rests on, from the current contact set."""
        d = self.d
        res: dict[str, set[str]] = {}
        if not d.ncon:
            return {}
        oi = self.geom_oidx[d.contact.geom[:d.ncon]]
        for i in range(d.ncon):
            a, b = int(oi[i, 0]), int(oi[i, 1])
            if a == b:
                continue
            nz = float(d.contact.frame[i, 2])
            if nz > 0.6:
                res.setdefault(self.onames[b], set()).add(self.onames[a])
            elif nz < -0.6:
                res.setdefault(self.onames[a], set()).add(self.onames[b])
        return {k: sorted(v) for k, v in res.items()}

    def is_quiet(self) -> bool:
        rest_v = float(self.rc["rest_speed"])
        if self._climb is not None or self._jump_left > 0:
            return False
        if not self.grounded or float(np.linalg.norm(self.agent_vel())) > rest_v:
            return False
        for oid in self.active:
            if oid == self.held:
                continue
            if self.obj_speed(oid) > rest_v or float(np.linalg.norm(self.obj_angvel(oid))) > 0.5:
                return False
        return True

    def settle(self, max_time: float = 10.0, min_time: float = 0.0) -> dict:
        need = float(self.rc["rest_time"])
        quiet = 0.0
        t0 = self.time
        self.target_vel[:] = 0.0
        while self.time - t0 < max_time:
            self._step()
            quiet = quiet + self.dt if self.is_quiet() else 0.0
            if quiet >= need and self.time - t0 >= min_time:
                return {"settled": True, "elapsed": round(self.time - t0, 4)}
        return {"settled": False, "elapsed": round(self.time - t0, 4)}

    def agent_state(self) -> dict:
        p = self.agent_pos()
        below, what = self.ray_down(p[0], p[1], p[2] - self.hh + 0.05)
        return rnd({
            "position": p, "feet_height": p[2] - self.hh, "head_height": p[2] + self.hh,
            "velocity": self.agent_vel(), "speed": float(np.linalg.norm(self.agent_vel())),
            "on_ground": self.grounded, "standing_on": self.support,
            "surface_below": {"height": below, "object": what},
            "facing_degrees": self.facing, "holding": self.held,
            "body": {"width": 2 * self.hw, "height": 2 * self.hh, "mass": self.base_mass},
            "carried_mass": max(0.0, self.total_mass() - self.base_mass),
        })

    def object_state(self, oid: str, graph: dict | None = None) -> dict:
        if oid not in self.specs:
            raise ActionError("unknown_object", f"There is no object with id '{oid}'.",
                              known_ids=sorted(self.visible_ids()))
        s = self.specs[oid]
        if s.movable and oid not in self.active:
            if oid in self.lost:
                return {"id": oid, "name": s.name, "kind": s.kind, "in_world": False,
                        "interaction_state": {"lost": True, "note": "left the island during this session"}}
            raise ActionError("unknown_object", f"There is no object with id '{oid}'.",
                              known_ids=sorted(self.visible_ids()))
        base = {"id": oid, "name": s.name, "kind": s.kind, "shape": s.shape, "unique": s.unique,
                "movable": s.movable, "colour": s.colour, "dimensions": dict(s.dimensions),
                "mass": s.mass, "physical_properties": {"friction": s.friction, "rigid": True},
                "in_world": True}
        if not s.movable:
            base.update({"position": list(s.position), "rotation_quaternion": list(s.rotation),
                         "rotation_euler_degrees": [0.0, 0.0, 0.0],
                         "top_height": s.properties.get("top_height"),
                         "extent": {"x": [s.position[0] - s.dimensions["x"] / 2, s.position[0] + s.dimensions["x"] / 2],
                                    "y": [s.position[1] - s.dimensions["y"] / 2, s.position[1] + s.dimensions["y"] / 2],
                                    "z": [s.position[2] - s.dimensions["z"] / 2, s.position[2] + s.dimensions["z"] / 2]},
                         "interaction_state": {"held": False, "at_rest": True, "fixed": True}})
            return rnd(base)
        graph = self.support_graph() if graph is None else graph
        p, q = self.obj_pos(oid), self.obj_quat(oid)
        v, w = self.obj_vel(oid), self.obj_angvel(oid)
        hz = self.half_extent_z(oid)
        held = oid == self.held
        supporting = sorted(u for u, lows in graph.items() if oid in lows)
        base.update({
            "position": p, "rotation_quaternion": q, "rotation_euler_degrees": quat_to_euler_deg(q),
            "bottom_height": p[2] - hz, "top_height": p[2] + hz,
            "velocity": v, "angular_velocity": w,
            "interaction_state": {
                "held": held, "at_rest": (not held) and float(np.linalg.norm(v)) < float(self.rc["rest_speed"]),
                "in_flight": oid in self.flights and self.flights[oid]["first_impact"] is None,
                "resting_on": [] if held else graph.get(oid, []),
                "supporting": [] if held else supporting,
                "generated_this_session": oid in self.generated,
            },
        })
        return rnd(base)

    def visible_ids(self) -> list[str]:
        return [s.id for s in self.specs.values() if (not s.movable) or s.id in self.active]

    def boundaries(self) -> dict:
        return {
            "coordinate_system": {"units": "metres, kilograms, seconds",
                                  "axes": {"x": "east (+) / west (-)", "y": "north (+) / south (-)", "z": "up (+)"},
                                  "origin": "centre of the island at ground level"},
            "walkable_area": {"x": [-self.half, self.half], "y": [-self.half, self.half]},
            "ground_height": 0.0, "sea_level": self.sea,
            "start_position": [float(self.wc["start"][0]), float(self.wc["start"][1]), 0.0],
            "note": "The body cannot walk past the island edge. Objects are not stopped by the edge.",
        }

    def point_info(self, x: float, y: float, z: float) -> dict:
        inside = []
        for oid in self.visible_ids():
            if self._contains(oid, np.array([x, y, z], dtype=float)):
                inside.append(oid)
        top, what = self.ray_down(x, y, z + 1e-6)
        col_top, col_what = self.ray_down(x, y, 60.0)
        ap = self.agent_pos()
        return rnd({
            "position": [x, y, z],
            "inside_island_bounds": abs(x) <= self.half and abs(y) <= self.half,
            "occupied_by": inside,
            "surface_below": {"height": top, "object": what,
                              "gap": None if top is None else z - top},
            "highest_surface_in_column": {"height": col_top, "object": col_what},
            "horizontal_distance_from_agent": float(np.hypot(x - ap[0], y - ap[1])),
            "height_above_agent_feet": z - self.feet(),
            "within_reach": self._in_reach(np.array([x, y, z], dtype=float)),
        })

    def _contains(self, oid: str, pt: np.ndarray) -> bool:
        s = self.specs[oid]
        if s.movable:
            c, q = self.obj_pos(oid), self.obj_quat(oid)
        else:
            c, q = np.array(s.position), np.array(s.rotation)
        local = quat_to_mat(q).T @ (pt - c)
        if s.shape == "box":
            d = s.dimensions
            return bool(abs(local[0]) <= d["x"] / 2 and abs(local[1]) <= d["y"] / 2 and abs(local[2]) <= d["z"] / 2)
        if s.shape == "sphere":
            return bool(np.linalg.norm(local) <= s.dimensions["radius"])
        ring_r = math.hypot(local[0], local[1]) - s.dimensions["major_radius"]
        return bool(math.hypot(ring_r, local[2]) <= s.dimensions["tube_radius"])

    def _in_reach(self, pt: np.ndarray) -> bool:
        ap = self.agent_pos()
        f = self.feet()
        return bool(np.hypot(pt[0] - ap[0], pt[1] - ap[1]) <= float(self.ac["reach"])
                    and f - float(self.ac["reach_down"]) <= pt[2] <= f + float(self.ac["reach_up"]))

    # ------------------------------------------------------------------ movement
    def _require_free(self) -> None:
        if self._climb is not None:
            raise ActionError("busy", "The body is in the middle of a climb.")

    def wait(self, seconds: float) -> dict:
        if not (0 < seconds <= 60):
            raise ActionError("invalid_argument", "seconds must be between 0 and 60.")
        t0 = self.time
        self.target_vel[:] = 0.0
        self.step(int(round(seconds / self.dt)))
        return {"elapsed": round(self.time - t0, 4), "agent_position": rnd(self.agent_pos()),
                "agent_on_ground": self.grounded}

    def catch(self, oid: str, max_wait: float = 5.0) -> dict:
        """Wait for a moving object to come within reach and grab it."""
        self._require_free()
        s = self._movable(oid)
        if self.held == oid:
            raise ActionError("already_holding", f"'{oid}' is already being held.")
        if self.held:
            raise ActionError("hands_full", f"Already holding '{self.held}'. Only one object can be carried.",
                              holding=self.held)
        if not (0 < max_wait <= 30):
            raise ActionError("invalid_argument", "max_wait must be between 0 and 30 seconds.")
        fl = self.flights.get(oid)
        if fl is None or fl["first_impact"] is not None:
            raise ActionError("not_in_flight", f"'{oid}' is not in the air. Use pick_up for objects at rest.",
                              object_position=rnd(self.obj_pos(oid)))
        t0 = self.time
        closest = float("inf")
        vmax = float(self.ac["catch_max_speed"])
        while self.time - t0 < max_wait:
            fl = self.flights.get(oid)
            if oid not in self.active:
                raise ActionError("object_lost", f"'{oid}' left the island before it could be caught.")
            if fl is None or fl["first_impact"] is not None:
                raise ActionError("missed", f"'{oid}' touched "
                                  f"{', '.join(fl['first_impact']['with']) if fl else 'something'} before it "
                                  "came within reach.", waited=round(self.time - t0, 4),
                                  object_position=rnd(self.obj_pos(oid)),
                                  closest_approach=None if closest == float("inf") else round(closest, 4))
            pos = self.obj_pos(oid)
            ap = self.agent_pos()
            closest = min(closest, float(np.hypot(pos[0] - ap[0], pos[1] - ap[1])))
            rel = float(np.linalg.norm(self.obj_vel(oid) - self.agent_vel()))
            if self.time - t0 >= MIN_CATCH_DELAY and self._in_reach(pos) and rel <= vmax:
                res = self.pick_up(oid)
                res["waited"] = round(self.time - t0, 4)
                res["airborne_time"] = round(self.time - fl["t0"], 4)
                return res
            self._step()
        raise ActionError("timeout", f"'{oid}' did not come within reach in {max_wait} s.",
                          object_position=rnd(self.obj_pos(oid)), waited=round(self.time - t0, 4))

    def move(self, direction: np.ndarray, distance: float, speed: float | None = None,
             stop: bool = True) -> dict:
        self._require_free()
        d2 = np.array([direction[0], direction[1]], dtype=float)
        n = float(np.linalg.norm(d2))
        if n < 1e-6:
            raise ActionError("invalid_argument", "A walking direction needs a horizontal component.")
        d2 /= n
        if not (0 < distance <= 4 * self.half):
            raise ActionError("invalid_argument", f"distance must be between 0 and {4 * self.half} metres.")
        vmax = float(self.ac["max_speed"])
        speed = float(self.ac["walk_speed"]) if speed is None else float(speed)
        if not (0.1 <= speed <= vmax):
            raise ActionError("invalid_argument", f"speed must be between 0.1 and {vmax} m/s.")
        start = self.agent_pos().copy()
        feet0 = self.feet()
        t0 = self.time
        before = {oid: self.obj_pos(oid) for oid in self.active if oid != self.held}
        self.facing = heading_deg(d2)
        # Humanoid: root traction is disabled (physics._step) — the legs are the
        # locomotion system. Route this call through the gait controller so the
        # requested distance is covered by real foot-ground actuation in the
        # shared progress/blocked/timeout loop below.
        is_humanoid = self.ac.get("type") == "humanoid"
        heading_req = math.degrees(math.atan2(d2[1], d2[0]))
        if is_humanoid:
            gc_cfg = dict(self._gait_controller.cfg) if self._gait_controller is not None else {}
            if (gc_cfg.get("mode") != "walk"
                    or abs(float(gc_cfg.get("speed", 0.0)) - speed) > 1e-9
                    or abs(float(gc_cfg.get("heading_deg", 0.0)) - heading_req) > 1e-9):
                self.set_gait("walk", speed=speed, heading=heading_req)
        blocked_by: list[str] = []
        left_ground = False
        timeout = distance / speed * 2.5 + 4.0
        last_check_t, last_check_s = self.time, 0.0
        travelled = 0.0
        while True:
            p = self.agent_pos()
            travelled = float((p[0] - start[0]) * d2[0] + (p[1] - start[1]) * d2[1])
            remaining = distance - travelled
            if remaining <= 0.004:
                break
            if not is_humanoid:
                if stop:
                    a_max = 0.7 * float(self.ac["walk_force"]) / self.total_mass()
                    v_t = min(speed, max(0.04, math.sqrt(2 * a_max * max(remaining, 0.0))))
                    self.target_vel[:] = d2 * v_t
                else:
                    # stop=False: cruise at the requested speed and leave the
                    # cruise request active on return (legacy semantics — the
                    # running-throw HVE case depends on it).
                    self.target_vel[:] = d2 * speed

            self._step()
            if not self.grounded:
                left_ground = True
            if self.time - last_check_t >= (2.0 if is_humanoid else 0.5):
                # The humanoid starts its stride under a 1.5 s amplitude ramp
                # (a full-amplitude first swing kicks the stance leg), so the
                # box-tuned 0.5 s stall window would verdict "blocked" on a
                # healthy ramp start. The box keeps the head behaviour.
                if self.grounded and travelled - last_check_s < 0.01:
                    blocked_by = list(self.side_contacts) or ["unknown obstacle"]
                    break
                last_check_t, last_check_s = self.time, travelled
            if self.time - t0 > timeout:
                # Not an obstacle: the body was deflected (slid along something) and never closed the remaining
                # distance along the requested heading. Say so precisely rather than blaming an obstacle.
                blocked_by = list(self.side_contacts) or ["deflected_off_course"]
                break
        if blocked_by or stop:
            # Brake: box agents drop the traction request; the humanoid switches
            # the gait to stand so the legs hold posture instead of striding on.
            if is_humanoid and self._gait_controller is not None:
                self.set_gait("stand")
            self.target_vel[:] = 0.0
            t1 = self.time
            while self.time - t1 < 3.0:
                self._step()
                v = self.agent_vel()
                if self.grounded and math.hypot(v[0], v[1]) < 0.02 and abs(v[2]) < 0.02:
                    break
        end = self.agent_pos().copy()
        disturbed = []
        for oid, p0 in sorted(before.items()):
            if oid in self.active:
                dp = self.obj_pos(oid) - p0
                if float(np.linalg.norm(dp)) > 0.05:
                    disturbed.append({"object": oid, "displacement": rnd(dp, 3),
                                      "distance_moved": round(float(np.linalg.norm(dp)), 3),
                                      "now_at": rnd(self.obj_pos(oid), 3)})
        res = {
            "start_position": start, "end_position": end,
            "requested_distance": distance,
            "distance_travelled": float(np.hypot(end[0] - start[0], end[1] - start[1])),
            "height_change": self.feet() - feet0, "left_ground_during_move": left_ground,
            "elapsed": self.time - t0, "standing_on": self.support, "still_moving": not (stop or blocked_by),
            "velocity": self.agent_vel(), "objects_disturbed": disturbed,
        }
        if blocked_by:
            if blocked_by == ["deflected_off_course"] and is_humanoid:
                # The leg-driven walk reorients by twisting at the waist while
                # the planted feet resist with friction, so a long straight move
                # drifts off the requested line even while striding healthily;
                # for the humanoid this is reported, not raised (the box's
                # traction course-keeping doesn't apply to a yaw-limited walk).
                res["off_course"] = True
                res["facing_error_deg"] = round(math.degrees(math.atan2(
                    (end[1] - start[1]) * 1.0, (end[0] - start[0]) * 1.0))
                    - self.facing, 1)
                return rnd(res)
            if blocked_by == ["deflected_off_course"]:
                off = float(abs((end[0] - start[0]) * -d2[1] + (end[1] - start[1]) * d2[0]))
                raise ActionError("blocked", f"Movement did not finish: after {res['distance_travelled']:.2f} m the "
                                  f"body had been pushed {off:.2f} m sideways off the requested line and could not "
                                  f"close the remaining distance.", blocked_by=blocked_by,
                                  sideways_offset=round(off, 3), **rnd(res))
            raise ActionError("blocked", f"Movement stopped after {res['distance_travelled']:.2f} m of "
                              f"{distance:.2f} m: blocked by {', '.join(blocked_by)}.",
                              blocked_by=blocked_by, **rnd(res))
        return rnd(res)

    def jump(self, wait: str = "landing") -> dict:
        self._require_free()
        if wait not in ("landing", "apex", "takeoff"):
            raise ActionError("invalid_argument", "wait must be one of: landing, apex, takeoff.")
        if not self.grounded:
            # A body that is a hair above a surface (settling after a landing or a recoil)
            # touches down within moments; anything else is genuinely airborne.
            p = self.agent_pos()
            below, _ = self.ray_down(p[0], p[1], p[2] - self.hh + 0.05)
            if below is not None and self.feet() - below < 0.05:
                t_w = self.time
                while not self.grounded and self.time - t_w < 0.3:
                    self._step()
        if not self.grounded:
            raise ActionError("not_on_ground", "Cannot jump: the body is not standing on anything.",
                              feet_height=round(self.feet(), 4), velocity=rnd(self.agent_vel()))
        t0 = self.time
        start = self.agent_pos().copy()
        feet0 = self.feet()
        self._jumped = True
        self._jump_support = list(self.support)
        self._jump_left = int(round(float(self.ac["jump_push_time"]) / self.dt))
        while self._jump_left > 0:
            self._step()
        takeoff_v = self.agent_vel()
        self.target_vel[:] = 0.0
        if self._air is not None:
            self._air["v0"] = takeoff_v.copy()
        self.emit("jump_takeoff", position=self.agent_pos().copy(), velocity=takeoff_v, carried=self.held,
                  total_mass=self.total_mass(), pushed_off=list(self._jump_support),
                  takeoff_feet_height=feet0)
        res: dict[str, Any] = {"start_position": start, "takeoff_velocity": takeoff_v,
                               "takeoff_feet_height": feet0, "carried": self.held,
                               "total_mass": self.total_mass(), "returned_at": wait}
        if wait == "apex":
            while self.agent_vel()[2] > 0 and self.time - t0 < 10.0 and not (self.grounded and self.time - t0 > 0.3):
                self._step()
        elif wait == "landing":
            landing = None
            while self.time - t0 < 15.0:
                self._step()
                if self.last_air and self.last_air.get("t_start", -1) >= t0 - 1e-9 \
                        and self.last_air.get("cause") == "jump":
                    landing = self.last_air
                    break
                if self.grounded and self._air is None and self.time - t0 > float(self.ac["jump_push_time"]) + 0.05:
                    break   # never left the support (e.g. something overhead)
            # Stand still until the body is stably down: grounded and quiet for 40 ms.
            t1 = self.time
            quiet = 0
            while self.time - t1 < 2.0 and quiet < 20:
                self._step()
                quiet = quiet + 1 if (self.grounded and abs(self.agent_vel()[2]) < 0.02) else 0
            if landing:
                res.update({k: landing[k] for k in (
                    "air_time", "max_feet_height", "apex_above_takeoff", "horizontal_distance",
                    "impact_speed", "landed_on", "landing_position")})
            else:
                res["note"] = "The body did not leave its support."
        res.update({"position": self.agent_pos().copy(), "feet_height": self.feet(),
                    "velocity": self.agent_vel(), "on_ground": self.grounded, "elapsed": self.time - t0})
        return rnd(res)

    def _footprint_ok(self, cx: float, cy: float, z_from: float, h: float) -> tuple[bool, float]:
        centre, _ = self.ray_down(cx, cy, z_from)
        if centre is None or abs(centre - h) > 0.03:
            return False, -1.0
        top = centre
        e = self.hw - 0.02
        for dx, dy in ((e, e), (e, -e), (-e, e), (-e, -e), (e, 0), (-e, 0), (0, e), (0, -e)):
            hit, _ = self.ray_down(cx + dx, cy + dy, z_from)
            if hit is not None and hit > top:
                top = hit
        return top <= h + 0.03, centre

    def climb(self, direction: np.ndarray | None = None) -> dict:
        self._require_free()
        carried = max(0.0, self.total_mass() - self.base_mass)
        if carried > float(self.ac["climb_max_carry"]) + 1e-9:
            raise ActionError("too_heavy_to_climb",
                              f"Cannot climb while carrying {carried:.1f} kg.", carried_mass=round(carried, 3))
        if direction is not None:
            dirs = [np.array([direction[0], direction[1]], dtype=float)]
            if np.linalg.norm(dirs[0]) < 1e-6:
                raise ActionError("invalid_argument", "A climbing direction needs a horizontal component.")
            dirs[0] /= np.linalg.norm(dirs[0])
        else:
            f = math.radians(self.facing)
            dirs = [np.array([math.cos(f + k * math.pi / 4), math.sin(f + k * math.pi / 4)]) for k in
                    (0, 1, -1, 2, -2, 3, -3, 4)]
        p = self.agent_pos().copy()
        feet = self.feet()
        max_up = float(self.ac["max_climb"])
        z_from = feet + max_up + 3.0
        best = None
        nearest_too_high = None
        for d2 in dirs:
            r = 0.04
            while r <= float(self.ac["climb_reach"]) + 1e-9:
                ex, ey = p[0] + d2[0] * (self.hw + r), p[1] + d2[1] * (self.hw + r)
                h, what = self.ray_down(ex, ey, z_from)
                if h is not None and h > feet + 0.05:
                    if h <= feet + max_up + 1e-6:
                        cx, cy = p[0] + d2[0] * (2 * self.hw + r - 0.02), p[1] + d2[1] * (2 * self.hw + r - 0.02)
                        ok, _ = self._footprint_ok(cx, cy, z_from, h)
                        if ok and abs(cx) <= self.half - self.hw and abs(cy) <= self.half - self.hw:
                            best = (cx, cy, h, what, d2)
                            break
                    else:
                        gap = h - feet
                        if nearest_too_high is None or gap < nearest_too_high[0]:
                            nearest_too_high = (gap, what)
                    break
                r += 0.04
            if best:
                break
        if best is None:
            if nearest_too_high:
                raise ActionError("ledge_too_high",
                                  f"The nearest ledge ({nearest_too_high[1]}) is {nearest_too_high[0]:.2f} m above "
                                  "the feet, which is out of climbing reach from here.",
                                  ledge=nearest_too_high[1], ledge_height_above_feet=round(nearest_too_high[0], 4),
                                  feet_height=round(feet, 4))
            raise ActionError("nothing_to_climb", "There is no ledge within climbing reach in that direction.",
                              feet_height=round(feet, 4))
        cx, cy, h, what, d2 = best
        self.facing = heading_deg(d2)
        was_airborne = not self.grounded
        t0 = self.time
        n = max(2, int(round(float(self.ac["climb_time"]) / self.dt)))
        self._climb = {"start": p, "top": np.array([p[0], p[1], h + 0.02 + self.hh]),
                       "end": np.array([cx, cy, h + 0.02 + self.hh]), "n": n, "i": 0}
        self.target_vel[:] = 0.0
        self._jump_left = 0
        self._air = None
        while self._climb is not None:
            self._step()
        self.settle(max_time=2.0)
        res = {"start_position": p, "end_position": self.agent_pos().copy(), "climbed_onto": what,
               "height_gained": self.feet() - feet, "ledge_height": h, "started_airborne": was_airborne,
               "elapsed": self.time - t0, "standing_on": self.support}
        self.emit("climb", **res)
        return rnd(res)

    def _advance_climb(self) -> None:
        c = self._climb
        c["i"] += 1
        f = c["i"] / c["n"]
        if f < 0.6:
            pos = c["start"] + (c["top"] - c["start"]) * (f / 0.6)
        else:
            pos = c["top"] + (c["end"] - c["top"]) * ((f - 0.6) / 0.4)
        self.d.qpos[self.aq:self.aq + 3] = pos - self._agent_origin
        self.d.qvel[self.av:self.av + 3] = 0.0
        if c["i"] >= c["n"]:
            self._climb = None
            self.grounded = False

    # ------------------------------------------------------------------ objects
    def _movable(self, oid: str) -> ObjectSpec:
        if oid not in self.specs or (self.specs[oid].movable and oid not in self.active and oid not in self.lost):
            raise ActionError("unknown_object", f"There is no object with id '{oid}'.",
                              known_ids=sorted(self.visible_ids()))
        s = self.specs[oid]
        if not s.movable:
            raise ActionError("not_movable", f"'{oid}' ({s.name}) is part of the island and cannot be moved.")
        if oid in self.lost:
            raise ActionError("object_lost", f"'{oid}' left the island earlier in this session.")
        return s

    def pick_up(self, oid: str) -> dict:
        self._require_free()
        s = self._movable(oid)
        if self.held == oid:
            raise ActionError("already_holding", f"'{oid}' is already being held.")
        if self.held:
            raise ActionError("hands_full", f"Already holding '{self.held}'. Only one object can be carried.",
                              holding=self.held)
        p = self.obj_pos(oid)
        ap = self.agent_pos()
        dist = float(np.hypot(p[0] - ap[0], p[1] - ap[1]))
        if not self._in_reach(p):
            raise ActionError("out_of_reach", f"'{oid}' is out of reach: it is {dist:.2f} m away horizontally "
                              f"(reach is {float(self.ac['reach']):.2f} m).",
                              horizontal_distance=round(dist, 4),
                              height_above_feet=round(float(p[2] - self.feet()), 4),
                              object_position=rnd(p), agent_position=rnd(ap))
        if s.mass > float(self.ac["max_carry"]):
            raise ActionError("too_heavy", f"'{oid}' is too heavy to lift.", mass=s.mass)
        if oid in self.support:
            raise ActionError("standing_on_object", f"Cannot pick up '{oid}' while standing on it.")
        v_obj = self.obj_vel(oid)
        v_ag = self.agent_vel()
        rel = float(np.linalg.norm(v_obj - v_ag))
        moving = self.obj_speed(oid) > float(self.rc["rest_speed"])
        in_flight = oid in self.flights and self.flights[oid]["first_impact"] is None
        if rel > float(self.ac["catch_max_speed"]):
            raise ActionError("moving_too_fast", f"'{oid}' is moving too fast to grab.", relative_speed=round(rel, 4))
        was_on = self.support_graph().get(oid, [])
        if oid in self.flights:
            self._end_flight(oid, "caught")
        # Momentum is conserved when a moving object is grabbed.
        M = self.total_mass()
        new_v = (M * v_ag + s.mass * v_obj) / (M + s.mass)
        yaw = quat_to_euler_deg(self.obj_quat(oid))[2] if s.shape == "box" else 0.0
        self.held_quat = euler_deg_to_quat(0.0, 0.0, yaw)
        self._set_collidable(oid, False)
        self.held = oid
        self._set_agent_mass(self.base_mass + s.mass)
        if moving:
            self.d.qvel[self.av:self.av + 3] = new_v
        self._pin_held()
        mujoco.mj_forward(self.m, self.d)
        res = {"object": oid, "picked_up_from": p, "was_resting_on": was_on, "was_moving": moving,
               "caught_in_flight": in_flight, "object_velocity_before": v_obj, "relative_speed": rel,
               "object_mass": s.mass, "total_mass": self.total_mass(), "agent_position": ap.copy(),
               "agent_on_ground": self.grounded}
        self.emit("catch" if in_flight else "pickup", **res)
        return rnd(res)

    def _release(self, oid: str, pos: np.ndarray, quat: np.ndarray, vel: np.ndarray) -> None:
        """Let go of the held object at a pose. Reverts and raises if the space is occupied."""
        a, v = self.qadr[oid], self.dadr[oid]
        self.held = None
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = quat
        self.d.qvel[v:v + 3] = vel
        self.d.qvel[v + 3:v + 6] = 0.0
        self._set_collidable(oid, True)
        mujoco.mj_forward(self.m, self.d)
        oi = self.oidx[oid]
        worst, hit = 0.0, None
        if self.d.ncon:
            idx = self.geom_oidx[self.d.contact.geom[:self.d.ncon]]
            for i in range(self.d.ncon):
                x, y = int(idx[i, 0]), int(idx[i, 1])
                if x == y or (x != oi and y != oi):
                    continue
                depth = -float(self.d.contact.dist[i])
                if depth > worst:
                    worst, hit = depth, self.onames[y if x == oi else x]
        if worst > 0.02:
            self._set_collidable(oid, False)
            self.held = oid
            self._pin_held()
            mujoco.mj_forward(self.m, self.d)
            raise ActionError("space_occupied",
                              f"Cannot release '{oid}' there: the space is occupied by {hit}.",
                              occupied_by=hit, overlap=round(worst, 4), requested_position=rnd(pos))
        self._set_agent_mass(self.base_mass)

    def _held(self, oid: str | None) -> str:
        if self.held is None:
            raise ActionError("not_holding", "Nothing is being held.")
        if oid is not None and oid != self.held:
            self._movable(oid)
            raise ActionError("not_holding", f"'{oid}' is not being held (holding '{self.held}').",
                              holding=self.held)
        return self.held

    def _wait_flight(self, oid: str, fid: int, mode: str, max_wait: float) -> tuple[dict | None, bool]:
        t0 = self.time
        self.target_vel[:] = 0.0
        while self.time - t0 < max_wait:
            fl = self.flights.get(oid)
            if fl is None or fl["flight_id"] != fid:
                break
            if mode == "impact" and fl["first_impact"] is not None:
                return None, True
            self._step()
        for rec in reversed(self.completed_flights):
            if rec["flight_id"] == fid:
                return rec, True
        return None, False

    def throw(self, oid: str | None, direction: np.ndarray, force: float, wait: str = "rest",
              max_wait: float = 30.0) -> dict:
        self._require_free()
        oid = self._held(oid)
        if wait not in ("rest", "impact", "none"):
            raise ActionError("invalid_argument", "wait must be one of: rest, impact, none.")
        fmax = float(self.ac["throw_force_max"])
        if not (isinstance(force, (int, float)) and math.isfinite(force) and 0 < force <= fmax):
            raise ActionError("invalid_argument", f"force must be greater than 0 and at most {fmax} N.",
                              max_force=fmax)
        s = self.specs[oid]
        dvec = np.array(direction, dtype=float)
        v_rel = dvec * math.sqrt(2.0 * force * float(self.ac["throw_stroke"]) / s.mass)
        v_ag = self.agent_vel()
        ap = self.agent_pos()
        feet = self.feet()
        r = self.bounding_radius(oid)
        horiz = math.hypot(dvec[0], dvec[1])
        if horiz > 0.3:
            h = np.array([dvec[0], dvec[1]]) / horiz
            self.facing = heading_deg(h)
            pos = np.array([ap[0] + h[0] * (self.hw + r + 0.08), ap[1] + h[1] * (self.hw + r + 0.08),
                            feet + float(self.ac["hold_height"])])
        elif dvec[2] > 0:
            pos = np.array([ap[0], ap[1], feet + 2 * self.hh + r + 0.05])
        else:
            f = math.radians(self.facing)
            pos = np.array([ap[0] + math.cos(f) * (self.hw + r + 0.08), ap[1] + math.sin(f) * (self.hw + r + 0.08),
                            feet + float(self.ac["hold_height"])])
        grounded = self.grounded
        v_world = v_ag + v_rel
        self._release(oid, pos, self.held_quat.copy(), v_world)
        # Newton's third law: the body receives the opposite momentum.
        self.d.qvel[self.av:self.av + 3] = v_ag - (s.mass / self.base_mass) * v_rel
        fl = self._begin_flight(oid, "throw", v_world, {"agent_velocity": v_ag.copy(), "agent_grounded": grounded,
                                                        "force": float(force)})
        launch = {"object": oid, "force": float(force), "direction": dvec, "release_position": pos,
                  "release_velocity": v_world, "release_speed": float(np.linalg.norm(v_world)),
                  "velocity_relative_to_body": v_rel, "agent_velocity_at_release": v_ag,
                  "agent_on_ground_at_release": grounded, "agent_feet_height_at_release": feet,
                  "object_mass": s.mass}
        self.emit("throw_release", flight_id=fl["flight_id"], **launch)
        res: dict[str, Any] = dict(launch)
        res["flight_id"] = fl["flight_id"]
        res["waited_for"] = wait
        if wait == "none":
            res["note"] = "Returned immediately; the object is still moving."
            return rnd(res)
        rec, ok = self._wait_flight(oid, fl["flight_id"], wait, max_wait)
        if rec:
            res.update({k: rec[k] for k in ("outcome", "max_height", "max_height_above_launch",
                                            "time_to_max_height", "flight_time", "first_impact_position",
                                            "first_impact_with", "impact_speed", "horizontal_range", "impacts",
                                            "final_position", "total_time", "resting_on")})
        elif ok:
            fl = self.flights[oid]
            fi = fl["first_impact"]
            res.update({"outcome": "first_impact", "max_height": fl["max_height"],
                        "max_height_above_launch": fl["max_height"] - float(pos[2]),
                        "flight_time": fi["t"] - fl["t0"], "first_impact_position": fi["position"],
                        "first_impact_with": fi["with"], "impact_speed": fi["speed"],
                        "note": "Returned at first impact; the object may still be moving."})
        else:
            res.update({"outcome": "still_moving", "note": f"The object had not come to rest after {max_wait} s."})
        return rnd(res)

    # ------------------------------------------------------------------ contact manipulation
    # One primitive for every "apply a force to an object you are NOT holding" verb (push, pull,
    # slide, roll, kick, nudge). The verbs differ only in their force range, default direction and
    # a couple of preconditions — the PHYSICS is identical and contains no task knowledge, so an
    # agent discovers technique rather than having it hardcoded (action-space expansion 2026-11-13).
    # The object must be in reach and resting (not held, not in flight). An impulse is applied along
    # a horizontal direction; the engine then simulates freely and records the outcome the same way
    # a throw is recorded (a "slide" flight), so verifiers and NN datasets pick these up automatically.
    CONTACT_VERBS = {
        #            min_N  max_N  up_bias  needs_contact_dir
        "push":     (5.0,  120.0, 0.0),
        "pull":     (5.0,   60.0, 0.0),
        "slide":    (2.0,   80.0, 0.0),
        "nudge":    (1.0,   15.0, 0.0),
        "kick":     (20.0, 160.0, 0.12),
        "roll":     (5.0,  120.0, 0.0),
    }

    def impulse_object(self, oid: str, direction: np.ndarray, force: float, verb: str = "push",
                       wait: str = "rest", max_wait: float = 20.0) -> dict:
        """Apply a contact impulse to an unheld, resting object in reach. Shared by push/pull/slide/roll/kick/nudge."""
        self._require_free()
        if self.held == oid:
            raise ActionError("holding_object", f"'{oid}' is being held; put it down before you {verb} it.")
        s = self._movable(oid)
        if verb not in self.CONTACT_VERBS:
            raise ActionError("invalid_argument", f"unknown contact verb '{verb}'.")
        fmin, fmax, up_bias = self.CONTACT_VERBS[verb]
        if not (isinstance(force, (int, float)) and math.isfinite(force) and fmin <= force <= fmax):
            raise ActionError("invalid_argument", f"force for {verb} must be between {fmin} and {fmax} N.",
                              min_force=fmin, max_force=fmax)
        if wait not in ("rest", "impact", "none"):
            raise ActionError("invalid_argument", "wait must be one of: rest, impact, none.")
        p = self.obj_pos(oid)
        ap = self.agent_pos()
        if not self._in_reach(p):
            dist = float(np.hypot(p[0] - ap[0], p[1] - ap[1]))
            raise ActionError("out_of_reach", f"'{oid}' is {dist:.2f} m away; reach is {float(self.ac['reach']):.2f} m.",
                              horizontal_distance=round(dist, 4), object_position=rnd(p))
        if self.obj_speed(oid) > float(self.ac["catch_max_speed"]):
            raise ActionError("moving_too_fast", f"'{oid}' is already moving too fast to {verb}.")
        if oid in self.support:
            raise ActionError("standing_on_object", f"Cannot {verb} '{oid}' while standing on it.")
        dvec = np.array(direction, dtype=float)
        horiz = math.hypot(dvec[0], dvec[1])
        if horiz < 1e-6:
            raise ActionError("invalid_argument", f"{verb} needs a horizontal direction.")
        # Normalise to a horizontal unit vector, then add any small upward bias (kick lofts slightly).
        h = np.array([dvec[0] / horiz, dvec[1] / horiz, 0.0])
        d3 = h.copy()
        d3[2] = up_bias
        d3 = d3 / (np.linalg.norm(d3) or 1.0)
        self.facing = heading_deg(h)
        # Same impulse model as throw: v = dir * sqrt(2 * F * stroke / m). A shorter contact stroke
        # than a throw (the hands are not winding up), so the same force imparts less speed.
        stroke = float(self.ac.get("push_stroke", 0.25))
        v_mag = math.sqrt(2.0 * float(force) * stroke / s.mass)
        v_world = d3 * v_mag
        # Apply to the object's free joint; the body keeps its own velocity (feet braced).
        vadr = self.dadr[oid]
        self.d.qvel[vadr:vadr + 3] = self.obj_vel(oid) + v_world
        mujoco.mj_forward(self.m, self.d)
        fl = self._begin_flight(oid, "slide", v_world, {"verb": verb, "force": float(force),
                                                        "agent_position": ap.copy()})
        launch = {"object": oid, "verb": verb, "force": float(force), "direction": rnd(d3),
                  "contact_position": rnd(p), "launch_velocity": rnd(v_world),
                  "launch_speed": round(float(v_mag), 4), "object_mass": s.mass,
                  "agent_position": rnd(ap)}
        self.emit("object_impulse", flight_id=fl["flight_id"], **launch)
        res: dict[str, Any] = dict(launch)
        res["flight_id"] = fl["flight_id"]
        res["waited_for"] = wait
        if wait == "none":
            res["note"] = f"Returned immediately; '{oid}' is still moving."
            return rnd(res)
        rec, ok = self._wait_flight(oid, fl["flight_id"], wait, max_wait)
        start = np.array(p, dtype=float)
        if rec:
            fp = np.array(rec.get("final_position", p), dtype=float)
            res.update({k: rec[k] for k in ("outcome", "flight_time", "first_impact_position",
                                            "first_impact_with", "impact_speed", "horizontal_range",
                                            "impacts", "final_position", "total_time", "resting_on")
                        if k in rec})
            res["distance_moved"] = round(float(np.hypot(fp[0] - start[0], fp[1] - start[1])), 4)
            res["toppled"] = bool(rec.get("impacts", 0) >= 1 and s.shape == "box"
                                  and self._tipped(oid))
        else:
            res["outcome"] = "still_moving"
            res["note"] = f"'{oid}' had not come to rest after {max_wait} s."
        return rnd(res)

    def _tipped(self, oid: str) -> bool:
        """True if a box object is now resting on a face other than the one it started upright on."""
        if self.specs[oid].shape != "box":
            return False
        up = quat_to_mat(self.obj_quat(oid))[:, 2]
        return bool(abs(up[2]) < 0.8)   # local +z no longer points up => it tipped over

    def place(self, oid: str | None, position, wait: bool = True) -> dict:
        self._require_free()
        oid = self._held(oid)
        try:
            vals = [float(v) for v in position]
        except (TypeError, ValueError):
            raise ActionError("invalid_argument", "position must be [x, y] or [x, y, z].") from None
        if len(vals) not in (2, 3) or not all(math.isfinite(v) for v in vals):
            raise ActionError("invalid_argument", "position must be [x, y] or [x, y, z] with finite numbers.")
        hz = self.half_extent_z(oid, self.held_quat)
        feet = self.feet()
        x, y = vals[0], vals[1]
        auto = len(vals) == 2
        if auto:
            top, what = self.ray_down(x, y, 60.0)
            if top is None:
                raise ActionError("no_surface", "There is no surface below that point to place the object on.",
                                  requested_position=[x, y])
            z = top + hz + 0.005
            if z > feet + float(self.ac["reach_up"]) + 1e-9:
                raise ActionError("out_of_reach",
                                  f"The top surface there ({what}) is {top - feet:.2f} m above the feet; the object "
                                  f"would have to be released {z - feet:.2f} m above the feet, which is too high "
                                  "to reach from here.", surface=what, surface_height=round(top, 4),
                                  release_height_above_feet=round(z - feet, 4), feet_height=round(feet, 4))
        else:
            z = vals[2]
        pos = np.array([x, y, z])
        ap = self.agent_pos()
        if not self._in_reach(pos):
            raise ActionError("out_of_reach", "That position is out of reach.",
                              horizontal_distance=round(float(np.hypot(x - ap[0], y - ap[1])), 4),
                              height_above_feet=round(z - feet, 4), requested_position=rnd(pos))
        v_ag = self.agent_vel() if not self.grounded else np.zeros(3)
        self._release(oid, pos, self.held_quat.copy(), v_ag)
        fl = self._begin_flight(oid, "place", v_ag, {"agent_velocity": v_ag.copy(), "agent_grounded": self.grounded})
        self.emit("release", object=oid, kind="place", position=pos, flight_id=fl["flight_id"])
        res: dict[str, Any] = {"object": oid, "requested_position": pos, "release_position": pos,
                               "flight_id": fl["flight_id"]}
        if not wait:
            return rnd(res)
        rec, _ = self._wait_flight(oid, fl["flight_id"], "rest", 20.0)
        if rec:
            fp = np.array(rec["final_position"])
            res.update({"final_position": rec["final_position"], "resting_on": rec["resting_on"],
                        "outcome": rec["outcome"], "settle_time": rec["total_time"],
                        "displacement_from_requested": float(np.linalg.norm(fp - pos)),
                        "fall_distance": float(pos[2] - fp[2]),
                        "stayed_where_placed": bool(np.hypot(fp[0] - x, fp[1] - y) < 0.1 and abs(pos[2] - fp[2]) < 0.1)})
        else:
            res.update({"outcome": "still_moving", "final_position": self.obj_pos(oid)})
        return rnd(res)

    def put_down(self, oid: str | None) -> dict:
        self._require_free()
        oid = self._held(oid)
        r = self.bounding_radius(oid)
        ap = self.agent_pos()
        last_err: ActionError | None = None
        for k in (0, 1, -1, 2, -2, 3, -3, 4):
            f = math.radians(self.facing) + k * math.pi / 4
            x = ap[0] + math.cos(f) * (self.hw + r + 0.1)
            y = ap[1] + math.sin(f) * (self.hw + r + 0.1)
            try:
                return self.place(oid, [x, y])
            except ActionError as e:
                last_err = e
        raise ActionError("no_room", f"There is no free space next to the body to put '{oid}' down.",
                          last_error=last_err.message if last_err else None)

    def rotate_object(self, oid: str, rotation) -> dict:
        self._require_free()
        s = self._movable(oid)
        try:
            r, p, y = (float(v) for v in rotation)
        except (TypeError, ValueError):
            raise ActionError("invalid_argument",
                              "rotation must be [roll, pitch, yaw] in degrees.") from None
        if not all(math.isfinite(v) for v in (r, p, y)):
            raise ActionError("invalid_argument", "rotation values must be finite.")
        q = euler_deg_to_quat(r, p, y)
        if oid == self.held:
            self.held_quat = q
            self._pin_held()
            return rnd({"object": oid, "held": True, "rotation_euler_degrees": [r, p, y],
                        "rotation_quaternion": q})
        pos = self.obj_pos(oid)
        if not self._in_reach(pos):
            raise ActionError("out_of_reach", f"'{oid}' is out of reach.", object_position=rnd(pos))
        if oid in self.support:
            raise ActionError("standing_on_object", f"Cannot rotate '{oid}' while standing on it.")
        if self.held:
            raise ActionError("hands_full", f"Hands are full (holding '{self.held}').")
        a, v = self.qadr[oid], self.dadr[oid]
        old_q = self.obj_quat(oid)
        old_p = pos.copy()
        bottom = pos[2] - self.half_extent_z(oid)
        new_p = np.array([pos[0], pos[1], bottom + self.half_extent_z(oid, q) + 0.005])
        self.d.qpos[a:a + 3] = new_p
        self.d.qpos[a + 3:a + 7] = q
        self.d.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        oi = self.oidx[oid]
        worst, hit = 0.0, None
        if self.d.ncon:
            idx = self.geom_oidx[self.d.contact.geom[:self.d.ncon]]
            for i in range(self.d.ncon):
                x, yy = int(idx[i, 0]), int(idx[i, 1])
                if x == yy or (x != oi and yy != oi):
                    continue
                depth = -float(self.d.contact.dist[i])
                if depth > worst:
                    worst, hit = depth, self.onames[yy if x == oi else x]
        if worst > 0.02:
            self.d.qpos[a:a + 3] = old_p
            self.d.qpos[a + 3:a + 7] = old_q
            mujoco.mj_forward(self.m, self.d)
            raise ActionError("space_occupied", f"Cannot rotate '{oid}' there: it would overlap {hit}.",
                              occupied_by=hit)
        self.emit("rotate", object=oid, rotation_euler_degrees=[r, p, y])
        st = self.settle(max_time=6.0)
        return rnd({"object": oid, "held": False, "requested_rotation_euler_degrees": [r, p, y],
                    "final_rotation_euler_degrees": quat_to_euler_deg(self.obj_quat(oid)),
                    "final_position": self.obj_pos(oid), "settled": st["settled"]})

    # ------------------------------------------------------------------ state
    def describe(self) -> dict:
        graph = self.support_graph()
        return {"time": round(self.time, 4), "agent": self.agent_state(),
                "objects": [self.object_state(oid, graph) for oid in self.visible_ids()],
                "lost_objects": sorted(self.lost), "generated_objects": list(self.generated)}

    def _scan_contacts(self) -> None:
        """Recompute support information from the current contacts without recording anything."""
        was, n_events = self._recording, self.seq
        self._recording = False
        air = self._air
        self.grounded = True          # suppress a spurious take-off transition
        self._post_step_contacts_only()
        self._air = air
        self._recording = was
        assert self.seq == n_events

    def _post_step_contacts_only(self) -> None:
        d = self.d
        grounded, support, side, pairs = False, set(), set(), set()
        if d.ncon:
            oi = self.geom_oidx[d.contact.geom[:d.ncon]]
            ca, cb = oi[:, 0], oi[:, 1]
            keep = ca != cb
            if keep.any():
                n = len(self.onames)
                codes = np.unique(np.minimum(ca, cb)[keep] * n + np.maximum(ca, cb)[keep])
                pairs = {(c // n, c % n) for c in codes.tolist()}
                ag = self.agent_oi
                for k in np.nonzero(keep & ((ca == ag) | (cb == ag)))[0].tolist():
                    a, b = int(ca[k]), int(cb[k])
                    nz = float(d.contact.frame[k, 2])
                    up = -nz if a == ag else nz
                    other = self.onames[b if a == ag else a]
                    if up > 0.7:
                        grounded = True
                        support.add(other)
                    elif abs(nz) < 0.3:
                        side.add(other)
        self.grounded, self.support, self.side_contacts = grounded, sorted(support), sorted(side)
        self._pairs = pairs

    def get_state(self) -> dict:
        """Complete checkpoint: restoring it reproduces the simulation bit for bit."""
        size = mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_INTEGRATION)
        buf = np.zeros(size)
        mujoco.mj_getState(self.m, self.d, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
        return {"time": self.time, "steps": self.steps, "mj": buf.tolist(),
                "active": sorted(self.active), "latent": list(self.latent), "lost": sorted(self.lost),
                "generated": list(self.generated), "held": self.held, "held_quat": self.held_quat.tolist(),
                "facing": self.facing, "rule_charges": self.rule_charges, "rule_armed": self._rule_armed,
                "rule_since": self._rule_since, "seq": self.seq, "frame_count": self._frame_count,
                "flight_seq": self._flight_seq, "air": _plain(self._air), "jumped": self._jumped,
                "flights": _plain(self.flights), "pending": _plain(self._pending),
                "last_air": _plain(self.last_air), "pairs": sorted([int(a), int(b)] for a, b in self._pairs),
                "grounded": self.grounded, "support": list(self.support)}

    def set_state(self, st: dict, keep_counters: bool = False) -> None:
        """Restore a checkpoint taken with get_state() (session recovery, sandbox clones)."""
        self.active = set(st["active"])
        self.latent = list(st["latent"])
        self.lost = set(st["lost"])
        self.generated = list(st["generated"])
        self.facing = float(st["facing"])
        self.rule_charges = int(st["rule_charges"])
        self._rule_armed = bool(st["rule_armed"])
        self._rule_since = st.get("rule_since")
        self.steps = int(st["steps"])
        if not keep_counters:
            self.seq = int(st["seq"])
            self._frame_count = int(st["frame_count"])
            self._flight_seq = int(st["flight_seq"])
        self.flights = {k: dict(v) for k, v in (st.get("flights") or {}).items()}
        self._pending = [dict(v) for v in (st.get("pending") or [])]
        self.last_air = st.get("last_air")
        self.held = st.get("held")
        self.held_quat = np.array(st["held_quat"], dtype=float)
        self.d.xfrc_applied[:] = 0.0
        g = float(self.wc["gravity"])
        for oid in self.movable:
            on = oid in self.active and oid != self.held
            self._set_collidable(oid, on)
            if oid not in self.active:
                b = self.bid[oid]
                self.d.xfrc_applied[b, 2] = float(self.m.body_mass[b]) * g
        mass = self.base_mass + (float(self.specs[self.held].mass) if self.held else 0.0)
        self.m.body_mass[self.agent_bid] = mass
        mujoco.mj_setConst(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, np.array(st["mj"], dtype=float), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.m, self.d)
        self.target_vel[:] = 0.0
        self._jump_left = 0
        self._climb = None
        air = st.get("air")
        if air:
            air = dict(air)
            air["pos0"] = np.array(air["pos0"], dtype=float)
            air["v0"] = np.array(air["v0"], dtype=float)
        self._jumped = bool(st.get("jumped", False))
        self._scan_contacts()
        self._air = air
        if "pairs" in st:
            # The contact set of the checkpointed step, so the next step sees the same "new" contacts.
            self._pairs = {(int(a), int(b)) for a, b in st["pairs"]}
            self.grounded = bool(st["grounded"])
            self.support = list(st["support"])
        self._last_frame = {}
        # NOTE: the humanoid-era spawn reset that used to sit here was removed:
        # set_state() must faithfully restore a checkpoint. Besides, with the 3
        # agent slide joints first, qpos[aq+3:aq+7] belongs to block_01's
        # freejoint in box worlds — the old unconditional write teleported
        # block_01 to (1,0,0) on every restore (Nov 17 bug log).


    def clone(self) -> "PhysicsWorld":
        """Independent copy of the world in its current state. Nothing it does is recorded here."""
        other = PhysicsWorld(self.cfg)
        other.set_state(self.get_state())
        other.events.clear()
        other.completed_flights.clear()
        return other

    def state_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.d.qpos.tobytes() + self.d.qvel.tobytes()).hexdigest()

    # ------------------------------------------------------------------ researcher-side setup
    def setup_place(self, oid: str, position, rotation_euler_degrees=(0.0, 0.0, 0.0)) -> None:
        """Position an object directly. For benchmark / test setup only: never exposed to agents."""
        self._movable(oid)
        a, v = self.qadr[oid], self.dadr[oid]
        self.d.qpos[a:a + 3] = [float(x) for x in position]
        self.d.qpos[a + 3:a + 7] = euler_deg_to_quat(*rotation_euler_degrees)
        self.d.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def setup_remove(self, oid: str) -> None:
        """Take an object out of the scene. For benchmark / test setup only: never exposed to agents."""
        self._movable(oid)
        if oid == self.held:
            raise ActionError("invalid_argument", "Cannot remove the held object.")
        self.active.discard(oid)
        self._park(oid)
        mujoco.mj_forward(self.m, self.d)

    def setup_agent(self, x: float, y: float, feet: float | None = None) -> None:
        """Position the body directly. For benchmark / test setup only: never exposed to agents."""
        if feet is None:
            top, _ = self.ray_down(x, y, 50.0)
            feet = (top or 0.0) + 0.002
        self.d.qpos[self.aq:self.aq + 3] = np.array([x, y, feet + self.hh]) - self._agent_origin
        self.d.qvel[self.av:self.av + 3] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self._scan_contacts()

    def setup_finish(self) -> None:
        """Let a prepared scene come to rest and make it the session's starting point."""
        was = self._recording
        self._recording = False
        self.settle(max_time=5.0)
        t_keep = self.d.time
        self._recording = was
        self.initial_positions = {oid: self.obj_pos(oid).tolist() for oid in sorted(self.active)}
        self.emit("scene_prepared", sim_time=round(t_keep, 4))
        # Humanoid-only spawn reset (see _humanoid_spawn_reset): limb joints
        # only, feet-on-ground; box worlds' block freejoint slots stay untouched.
        if self.ac.get("type") == "humanoid":
            self._humanoid_spawn_reset(0.01)
