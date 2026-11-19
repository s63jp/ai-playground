"""LocoEnv — the single authoritative locomotion RL environment (M6-B B2 → M7-0a).

Contract version 4 by default ("rl_env/4"; M7-0a per the approved freeze
reports/M7_0_CONTRACT_FREEZE_20261118_DRAFT.md v2 + reports/M7_0A_ENV_V4_CHANGE.md),
documented in full in docs/RL_ENVIRONMENT.md and machine-readable via
observation_layout() / action_layout(); the B5 .aipg manifest reproduces
exactly this schema. One env, one reward pipeline (rl/rewards.py), one
curriculum mechanism (rl/curriculum.py applying phases through set_phase).

Observation v4 (float32[68]; APPEND-ONLY over v3, which is APPEND-ONLY over
v2 — the 43-dim v3 prefix is unchanged and bit-identical, verified by the
N1 test):
    [0:14]   joint positions, normalized (2q - lo - hi)/(hi - lo) in [-1, 1]
    [14:28]  joint velocities, normalized  qvel / 6.0 rad/s
    [28:31]  torso up-vector, world frame (unit)
    [31:34]  torso linear velocity, world frame, / 5.0 m/s
    [34:35]  upright margin (= torso up_z, raw)
    [35:38]  torso forward (chest) axis, world frame (unit; xmat column 1)
    [38:40]  per-foot ground-support normal force / 1000.0 N (left, right)
    [40:43]  COMMANDED target velocity v*, WORLD frame, / 5.0 m/s
    [43:58]  limb_pose_target (15, action-mapping encoding; 0 = neutral)
    [58:60]  com_shift (2, normalized; measured anchor ± scales)
    [60:63]  waypoint (ux, uy, d̂; yaw-only heading frame, d̂ = dist/4.0)
    [63:64]  stop_request flag
    [64:68]  cmd_mask: {0,1} x (limb, balance, waypoint, stop) — the stage
             selector in-obs. Mask 0 ⇒ zeros in obs AND no reward consumes
             the group. The ACTION space is never masked.

Action (15 normalized joints, [-1, 1], NAME-ordered: the 14 legacy joints in
unchanged order + waist_yaw at index 14) maps to PD targets
    target = lo + (a + 1) * 0.5 * (hi - lo)
through PhysicsWorld.set_joint_target — the world's single validated
joint-force path (B1: non-finite rejection, gain ceilings, per-joint torque
caps, saturation attribution). The PD tracks at 500 Hz (dt 0.002 s); a
decision is held for `action_repeat` substeps (default 10 -> 20 ms policy
period) — decimation lives here, the world is untouched. The waist carries
the measured posture gains (50, 5) and the B1 cap 120 N·m (P1/P2).

contract_version 3 (legacy, read-only evaluation surfaces) reproduces the
pre-v4 tree bit-identically: obs 43, action 14 (the waist again a
conventional posture hinge), no command block, no command setters — the
historical lineages replay through the exact old contract.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from ..world import PhysicsWorld
from ..world.physics import ActionError

from .rewards import (COMPONENT_KEYS, DEFAULTS, MeasuredStep, compute_reward,
                      heading_progress_raw, COM_ANCHOR_XY, COM_SHIFT_SCALE_XY,
                      WAYPOINT_D_REF, LIMB_GROUPS)

CONTRACT_VERSION = 4
# M7-0a (reports/M7_0A_ENV_V4_CHANGE.md): the env speaks rl_env/4 by default.
# contract_version 3 remains available as the read-only legacy evaluation mode
# (bit-identical to the pre-v4 tree — playback of the historical lineages).
_LEGACY_CONTRACT_VERSION = 3

# The frozen NAME-ORDERED joint list, the v2-freeze §2 law. v3's 14 are the
# exact model-enumeration order on this rig (verified twice, probe +
# construction test); the waist is APPENDED at 14 and is never taken from
# model order (waist_yaw is MODEL joint index 3 — the root's rotational dof —
# so naive enumeration would put it first and break the append-only prefix).
_V3_JOINT_ORDER = (
    "spine_L2", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
    "head_pitch", "head_yaw", "head_roll", "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
)
_V4_JOINT_ORDER = _V3_JOINT_ORDER + ("waist_yaw",)
_COMMAND_GROUPS = ("limb", "balance", "waypoint", "stop")

# M7-1a: the measured P2 per-joint holdable candidate set for command
# schedules (ai-playground-audit/m7_p2_pd_reference.json — every non-waist
# joint completed holds at ±0.35/±0.70 rad; the waist is neutral-hold only).
_M7_P2_HOLDABLE_RADS = (0.0, -0.7, -0.35, 0.35, 0.7)

# Fixed normalisation constants (rl.observation in config; documented). Vel
# scales come from measured scripting envelopes: knee peak 4.83 rad/s during
# the scripted walk (ai-playground-audit/probe_b2_audit), root speed capped
# at 4.0 m/s by config.
_JOINT_VEL_SCALE = 6.0     # rad/s
_ROOT_VEL_SCALE = 5.0      # m/s
_CONTACT_FORCE_SCALE = 1000.0  # N per obs unit


class LocoEnv:
    # v4 default (rl_env/4): 43 (v3, unchanged prefix) + 15 limb targets + 2
    # com_shift + 3 waypoint + 1 stop + 4 cmd_mask. v3-mode instances carry
    # 43 / 14 / version 3 (see __init__).
    obs_dim = 43 + 15 + 2 + 3 + 1 + 4   # 68, layout v4
    action_dim = 15
    contract_version = CONTRACT_VERSION

    # v4 passive posture hinges excluded from the action space (held by the
    # conventional PD registered at reset): left/right_foot_yaw only — the
    # waist JOINS the policy under v4 (frozen name order above). The v3
    # legacy mode keeps its third passive hinge (waist_yaw) via _EXCLUDED_V3.
    _EXCLUDED = ("left_foot_yaw", "right_foot_yaw")
    _EXCLUDED_V3 = ("left_foot_yaw", "right_foot_yaw", "waist_yaw")

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        # Load full configuration and enforce humanoid body
        from ..config import load_config
        overrides = cfg.copy() if isinstance(cfg, dict) else {}
        agent_body = overrides.get('agent_body', {})
        agent_body['type'] = 'humanoid'
        overrides['agent_body'] = agent_body
        full_cfg = load_config(None, overrides=overrides)
        self.cfg = full_cfg
        self._rl_cfg = dict(full_cfg.get('rl') or {})
        # ---- contract switch (M7-0a): 4 = rl_env/4 default; 3 = legacy ----
        self.contract_version = int(self._rl_cfg.get('contract_version',
                                                     CONTRACT_VERSION))
        if self.contract_version not in (CONTRACT_VERSION,
                                         _LEGACY_CONTRACT_VERSION):
            raise ValueError(
                f"unsupported contract_version {self.contract_version}; "
                f"allowed: {(_LEGACY_CONTRACT_VERSION, CONTRACT_VERSION)}")
        self.world = PhysicsWorld(cfg=full_cfg)
        # Curriculum state: assistance lives in THIS env (joint-space PD target
        # blending; never teleports, never direct qpos writes). Base gravity
        # captured once so phase application never spirals the model's gravity.
        self._base_gravity = np.array(self.world.m.opt.gravity, dtype=float).copy()
        self._cur_gravity_scale = 1.0
        self.upright_assist = float(cfg.get('upright_assist', 0.0) or 0.0)
        self._last_targets = {}
        self.max_steps = int(self._rl_cfg.get('max_episode_steps', 400))
        self.action_repeat = max(1, int(self._rl_cfg.get('action_repeat', 10)))
        self._target_vel = np.asarray(
            self._rl_cfg.get('target_velocity', [0.5, 0.0, 0.0]), dtype=np.float64)
        self._reward_cfg = rl_cfg_reward_defaults(self._rl_cfg)
        obs_cfg = dict(self._rl_cfg.get('observation') or {})
        self._joint_vel_scale = float(obs_cfg.get('joint_vel_scale', _JOINT_VEL_SCALE))
        self._root_vel_scale = float(obs_cfg.get('root_vel_scale', _ROOT_VEL_SCALE))
        self._contact_force_scale = float(obs_cfg.get('contact_force_scale',
                                                      _CONTACT_FORCE_SCALE))
        self._start_jitter = float(self._rl_cfg.get('start_pose_jitter', 0.15))
        self._is_v3 = self.contract_version == _LEGACY_CONTRACT_VERSION
        if self._is_v3:
            # ---- legacy mode: the exact pre-v4 construction, untouched. ----
            self.joint_names = list(_V3_JOINT_ORDER)
            self.action_dim = 14
            self.obs_dim = 43
            self._obs_joint_names = self.joint_names   # the same list (obs = action joints)
            self._EXCL = self._EXCLUDED_V3
            self._gains = {
                "left_hip": (60.0, 5.0), "right_hip": (60.0, 5.0),
                "left_knee": (60.0, 5.0), "right_knee": (60.0, 5.0),
                "left_ankle": (25.0, 1.0), "right_ankle": (25.0, 1.0),
                "spine_L2": (35.0, 4.0),
                "left_shoulder": (25.0, 3.0), "right_shoulder": (25.0, 3.0),
                "left_elbow": (10.0, 2.0), "right_elbow": (10.0, 2.0),
                "head_pitch": (8.0, 1.0), "head_yaw": (8.0, 1.0),
                "head_roll": (8.0, 1.0),
            }
            self._posture_joints = ("waist_yaw", "left_foot_yaw", "right_foot_yaw")
            self._posture_gains = {"waist_yaw": (50.0, 5.0),
                                   "left_foot_yaw": (8.0, 1.0),
                                   "right_foot_yaw": (8.0, 1.0)}
        else:
            # ---- v4: name-ordered 15-joint action; waist is POLICY. ----
            self.joint_names = list(_V4_JOINT_ORDER)
            # The OBS slices stay the fixed 14-name layout in BOTH modes: the
            # v3 prefix [0:43] is bit-identical forever and the WAIST was
            # never observed in v3 (its measured position reaches the reward
            # pipeline, never the observation — measured state ≠ commanded
            # obs). The action list is the one that grows.
            self._obs_joint_names = list(_V3_JOINT_ORDER)
            self.action_dim = 15
            self.obs_dim = 68
            self._EXCL = self._EXCLUDED
            self._gains = {
                "left_hip": (60.0, 5.0), "right_hip": (60.0, 5.0),
                "left_knee": (60.0, 5.0), "right_knee": (60.0, 5.0),
                "left_ankle": (25.0, 1.0), "right_ankle": (25.0, 1.0),
                "spine_L2": (35.0, 4.0),
                "left_shoulder": (25.0, 3.0), "right_shoulder": (25.0, 3.0),
                "left_elbow": (10.0, 2.0), "right_elbow": (10.0, 2.0),
                "head_pitch": (8.0, 1.0), "head_yaw": (8.0, 1.0),
                "head_roll": (8.0, 1.0),
                "waist_yaw": (50.0, 5.0),   # measured prior P1A/P2; N2 re-measures
            }
            self._posture_joints = ("left_foot_yaw", "right_foot_yaw")
            self._posture_gains = {"left_foot_yaw": (8.0, 1.0),
                                   "right_foot_yaw": (8.0, 1.0)}
        # Enumerate-or-verify: in legacy mode the frozen 14-name list MUST
        # equal the model enumeration (drift detector for the rig itself).
        if self._is_v3:
            self._verify_joint_enumeration()
        # Joint ranges: read the REAL limited-joint ranges from the model;
        # fallback only for unlimited joints.
        self.joint_range = {}
        for name in self.joint_names:
            j = self.world.m.joint(name)
            if int(self.world.m.jnt_limited[j.id]):
                self.joint_range[name] = (float(self.world.m.jnt_range[j.id][0]),
                                          float(self.world.m.jnt_range[j.id][1]))
            else:
                self.joint_range[name] = (-np.pi / 2, np.pi / 2)
        # dof addresses for the policy joints (effort measurement; 14 legacy
        # / 15 with the waist under v4 — the N2 gate measures the growth).
        self._dofadr = [self._dof_of(n) for n in self.joint_names]
        self._collect_bodies()
        # ---- M7-0a v4 command state (commands are STAGE state; reset() does
        # not clear them — set_phase rewrites them non-cumulatively, absent ⇒
        # zeros. v3-mode envs reject the setters: that contract has no
        # command block.) ----
        self._commands = {"mask": (0, 0, 0, 0), "limb_targets": None,
                          "com_shift": (0.0, 0.0), "waypoint": (0.0, 0.0, 0.0),
                          "stop": 0, "limb_group": "all"}
        # ---- M7-1a: per-episode command schedule (S1 limb stages; spec set
        # by set_phase, plan built per reset(seed), ticked per step) ----
        self._cmd_schedule = None    # validated spec dict, or None
        self._cmd_plan = None        # this episode's segments [(prev, tgt, len)]
        self._cmd_window = 0         # decision windows since the last reset

        self._target_point = None   # env-side target point, metres world xyz
        self._upright_window = deque(maxlen=25)  # stability window (decisions)
        for name in self.joint_names:
            kp, kd = self._gains.get(name, (15.0, 1.5))
            self.world.set_joint_target(name, 0.0, kp=kp, kd=kd, source="neural_policy")
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        # B1 boundary accounting (per episode; reset() zeroes it):
        self._reset_boundary_stats()
        self._metrics_reset(seed=None)

    # ------------------------------------------------------------ helpers
    def _dof_of(self, name: str) -> int:
        j = self.world.m.joint(name)
        return int(j.dofadr[0])

    def _verify_joint_enumeration(self) -> None:
        """M7-0a drift detector: the frozen 14-name v3 order MUST equal the
        model-enumeration order on this rig (hinges minus the movable objects
        and the excluded passive hinges). A rig change that reorders the
        policy hinges fails here instead of silently changing the contract."""
        import mujoco as _mj
        hinge_code = int(_mj.mjtJoint.mjJNT_HINGE)
        m = self.world.m
        names = []
        for i in range(m.njnt):
            if int(m.jnt_type[i]) == hinge_code:
                name = _mj.mj_id2name(m, int(_mj.mjtObj.mjOBJ_JOINT), i)
                if isinstance(name, (bytes, bytearray)):
                    name = name.decode()
                if name and not str(name).startswith("j:") and \
                        name not in self._EXCL:
                    names.append(str(name))
        if tuple(names[:14]) != tuple(_V3_JOINT_ORDER):
            raise RuntimeError(
                "rig joint order drifted from the frozen v3 contract: "
                f"model {names[:14]} != frozen {_V3_JOINT_ORDER}")

    def _com_xy(self) -> tuple:
        """Full-body COM xy (metres, world): every rig body's mass ⊗ xipos,
        INCLUDING the agent root column (the mass the feet carry — the P4
        method). v4 measurement only."""
        m, d = self.world.m, self.world.d
        mass, acc = 0.0, np.zeros(3)
        for bid in self._rig_body_ids:   # includes the agent root column
            bmass = float(m.body_mass[bid])
            mass += bmass
            acc += bmass * np.asarray(d.xipos[bid], dtype=float)
        if mass <= 0.0:
            return (0.0, 0.0)
        com = acc / mass
        return (float(com[0]), float(com[1]))

    def _target_dist(self) -> float | None:
        """Horizontal (xy) distance from the agent root to the env target
        point, or None when no target point exists (M7-0a default; M7-5
        places targets)."""
        if self._target_point is None:
            return None
        p = np.asarray(self.world.agent_pos(), dtype=float)[:2]
        t = np.asarray(self._target_point, dtype=float)[:2]
        return float(np.linalg.norm(t - p))

    def set_target_point(self, xyz) -> None:
        """Set (or clear, None) the env-side target point, metres world xyz.

        The placement itself rides setup mechanics (M7-5: setup_place inside
        reset); this state EXISTS at M7-0a so the approach/stop reward terms
        have their engine-measured quantities and tests can exercise them.
        v3-mode envs refuse (that contract has no waypoint command).
        """
        if self._is_v3:
            raise ValueError("set_target_point requires contract version 4")
        if xyz is None:
            self._target_point = None
            return
        t = np.asarray(xyz, dtype=float).reshape(3)
        if not np.all(np.isfinite(t)):
            raise ValueError("target_point must be finite")
        self._target_point = (float(t[0]), float(t[1]), float(t[2]))

    def set_commands(self, mask=None, limb_targets=None, com_shift=None,
                     waypoint=None, stop=None, limb_group=None) -> None:
        """Set the v4 command state (the obs command block + reward gates).

        All values are VALIDATED loudly: mask = 4 bits (limb, balance,
        waypoint, stop) in {0,1}; limb_targets = 15 normalized values in
        [−1, 1] (action-mapping encoding; the waist at index 14); com_shift =
        2 normalized values in [−1, 1] (metres via COM_SHIFT_SCALE_XY from
        the measured anchor); waypoint = (ux, uy, d̂); stop ∈ {0, 1};
        limb_group ∈ LIMB_GROUPS. Partial updates update only what is given.
        The action space is NEVER masked: commands gate observation zeros and
        reward consumption only (the v2 freeze §1 semantics). v3-mode envs
        refuse: that contract has no command block.
        """
        if self._is_v3:
            raise ValueError("set_commands requires contract version 4")
        if mask is not None:
            m = tuple(int(x) for x in mask)
            if len(m) != 4 or any(x not in (0, 1) for x in m):
                raise ValueError("mask must be 4 bits in {0,1} "
                                 "(limb, balance, waypoint, stop)")
            self._commands["mask"] = m
        if limb_targets is not None:
            t = np.asarray(limb_targets, dtype=float).reshape(-1)
            if t.shape != (15,) or not np.all(np.isfinite(t)) or \
                    np.any(np.abs(t) > 1.0):
                raise ValueError(
                    "limb_targets must be 15 finite normalized values in [−1,1]")
            self._commands["limb_targets"] = tuple(float(x) for x in t)
        if com_shift is not None:
            c = np.asarray(com_shift, dtype=float).reshape(-1)
            if c.shape != (2,) or not np.all(np.isfinite(c)) or \
                    np.any(np.abs(c) > 1.0):
                raise ValueError(
                    "com_shift must be 2 finite normalized values in [−1,1] "
                    "(±0.070 m x / ±0.040 m y via COM_SHIFT_SCALE_XY)")
            self._commands["com_shift"] = (float(c[0]), float(c[1]))
        if waypoint is not None:
            w = np.asarray(waypoint, dtype=float).reshape(-1)
            if w.shape != (3,) or not np.all(np.isfinite(w)) or \
                    float(w[2]) < 0.0:
                raise ValueError(
                    "waypoint must be (ux, uy, d̂), finite, d̂ ≥ 0")
            self._commands["waypoint"] = (float(w[0]), float(w[1]), float(w[2]))
        if stop is not None:
            s = int(stop)
            if s not in (0, 1):
                raise ValueError("stop must be 0 or 1")
            self._commands["stop"] = s
        if limb_group is not None:
            if limb_group not in LIMB_GROUPS:
                raise ValueError(
                    f"limb_group must be one of {sorted(LIMB_GROUPS)}")
            self._commands["limb_group"] = limb_group

    # ---------------------------------------------------------- M7-1a schedule
    def _set_command_schedule(self, spec) -> None:
        """Validate + store a per-episode limb command schedule (M7-1a).

        Spec (all fields validated loudly; None clears — non-cumulative like
        every phase knob):
          group               name in LIMB_GROUPS (the stage's active group)
          targets_rad         {joint: [candidate hold targets in rad]} — every
                              candidate must be inside the joint's range_rad
                              AND inside the MEASURED P2 holdable set
                              {0, ±0.35, ±0.70}; waist candidates ⊆ {0} (the
                              P1B neutral-hold gate); ankle candidates ⊆ {0}
                              (P2: ground-pinned, an M7-2 COM-shift finding)
          hold_windows        decision windows held per segment (>0)
          transition_windows  windows ramping between holds (≥0)
          shuffle             false (default): candidates cycle in listed
                              order per joint; true: seeded per-episode draw
        """
        if self._is_v3:
            raise ValueError(
                "phase 'command_schedule' requires contract version 4")
        if spec is None:
            self._cmd_schedule = None
            self._cmd_plan = None
            self._cmd_window = 0
            return
        if not isinstance(spec, dict):
            raise TypeError("phase 'command_schedule' must be a dict or None")
        g = spec.get("group")
        if g not in LIMB_GROUPS:
            raise ValueError(
                f"command_schedule group must be one of {sorted(LIMB_GROUPS)}")
        targets = spec.get("targets_rad")
        if not isinstance(targets, dict) or not targets:
            raise ValueError("command_schedule needs targets_rad: "
                             "{joint: [candidate rads]}")
        hold = int(spec.get("hold_windows", 120))
        trans = int(spec.get("transition_windows", 30))
        if hold <= 0:
            raise ValueError("command_schedule hold_windows must be > 0")
        if trans < 0:
            raise ValueError("command_schedule transition_windows must be >= 0")
        if hold + trans > self.max_steps:
            raise ValueError(
                f"command_schedule segment ({hold}+{trans}) exceeds the "
                f"episode max_steps ({self.max_steps})")
        idxs = LIMB_GROUPS[g]
        sets = {}
        for i in idxs:
            j = self.joint_names[i]
            if j not in targets and \
                    not (j in ("waist_yaw",)
                         or j.endswith("_ankle")
                         or j.endswith("_foot_yaw")):
                raise ValueError(
                    f"command_schedule is missing joint '{j}' (group '{g}'); "
                    "every group joint needs an explicit candidate list — "
                    "pinned-neutral joints declare [0.0]")
            cands = targets.get(j, [0.0])
            if not isinstance(cands, (list, tuple)) or not cands:
                raise ValueError(f"command_schedule targets for '{j}' must "
                                 "be a non-empty list of rads")
            norm = []
            lo, hi = self.joint_range[j]
            for rad in cands:
                r = float(rad)
                if not np.isfinite(r):
                    raise ValueError(f"command_schedule target for '{j}' "
                                     "must be finite")
                if not (lo <= r <= hi):
                    raise ValueError(
                        f"command_schedule target {r} rad outside joint "
                        f"'{j}' range [{lo}, {hi}]")
                if j == "waist_yaw":
                    if r != 0.0:
                        raise ValueError(
                            "the waist is NEVER commanded in S1 (P1B/P2 "
                            "dead-band): candidates must be [0.0]")
                elif j.endswith("_ankle") or j.endswith("_foot_yaw"):
                    if r != 0.0:
                        raise ValueError(
                            f"'{j}' is ground-pinned under load (P2: 0.344 "
                            "rad tracking err — the M7-2 COM-shift finding): "
                            "S1 schedules must pin it at 0.0")
                elif min(abs(r - allowed) for allowed in
                         _M7_P2_HOLDABLE_RADS) > 1e-9:
                    raise ValueError(
                        f"command_schedule target {r} rad is outside the "
                        "measured P2 holdable set {0, ±0.35, ±0.70} for "
                        f"'{j}'")
                norm.append((r - lo) / (hi - lo) * 2.0 - 1.0)   # action encoding
            sets[j] = tuple(sorted(set(norm)))
        # joint-name keyed candidate sets, in NORMALIZED action units
        self._cmd_schedule = {"group": g,
                              "joints": tuple(self.joint_names[i] for i in idxs),
                              "idxs": tuple(int(x) for x in idxs),
                              "sets": sets,
                              "hold_windows": hold,
                              "transition_windows": trans,
                              "shuffle": bool(spec.get("shuffle", False))}
        self._cmd_plan = None
        self._cmd_window = 0
        # The scheduled group IS the active limb group: arm the limb mask and
        # the group here (one source of truth) — a schedule without a live
        # limb mask would be a silent no-op reward, which is exactly the
        # silent-misconfig pattern the loud-validated command seam exists to
        # prevent. Conflicts with a phase-declared 'commands' block are
        # refused in set_phase.
        self.set_commands(mask=(1, 0, 0, 0), limb_group=g)

    def _plan_command_episode(self, seed) -> None:
        """Build this episode's hold plan from the schedule spec. Deterministic
        per (spec, seed); the default (shuffle off) cycles each commanded
        joint's candidates in listed order (seeded rotation for variety) —
        identical seeds give identical plans, shuffled mode seeds its draws."""
        self._cmd_plan = None
        self._cmd_window = 0
        if self._cmd_schedule is None:
            return
        import zlib
        sched = self._cmd_schedule
        # int-entropy SeedSequence (string seeds are refused); crc32 keeps
        # the group derivation stable across processes/python runs
        entropy = [int(seed or 0), zlib.crc32(
            sched["group"].encode("utf-8"))]
        rng = np.random.default_rng(entropy)
        hold, trans = sched["hold_windows"], sched["transition_windows"]
        seg = hold + trans
        n_seg = int(np.ceil(max(self.max_steps, 1) / seg)) + 1
        rot = int(rng.integers(0, 10_000))    # fixed per-episode phase offset
        neutral = np.zeros(self.action_dim, dtype=float)
        plan, prev = [], neutral
        for k in range(n_seg):
            tgt = neutral.copy()
            for pos, idx in enumerate(sched["idxs"]):
                j = sched["joints"][pos]
                cands = sched["sets"][j]
                if sched["shuffle"]:
                    tgt[idx] = float(cands[int(rng.integers(len(cands)))])
                else:
                    tgt[idx] = float(cands[(rot + k + pos) % len(cands)])
            plan.append((prev, tgt, seg))
            prev = tgt
        self._cmd_plan = plan
        self._cmd_window = 0
        v = self._schedule_value_at(0)
        self.set_commands(limb_targets=tuple(float(x) for x in v))

    def _schedule_value_at(self, window: int) -> np.ndarray:
        """The commanded 15-vector at decision window `window` under the
        episode plan: transitions interpolate linearly, holds are flat."""
        seg = self._cmd_schedule["hold_windows"] + \
            self._cmd_schedule["transition_windows"]
        s = (window // seg) % len(self._cmd_plan)
        t = window - (window // seg) * seg
        prev_t, tgt, _ = self._cmd_plan[s]
        t_trans = self._cmd_schedule["transition_windows"]
        if t < t_trans and t_trans > 0:
            a = t / t_trans
            return (1.0 - a) * prev_t + a * tgt
        return tgt.copy()

    def _tick_command_schedule(self) -> None:
        """Advance the schedule one decision window and publish the window's
        command (engine-owned: the command block feeds obs + reward this
        window; the action space is never masked)."""
        if self._cmd_schedule is None or self._cmd_plan is None:
            return
        self._cmd_window += 1
        v = self._schedule_value_at(self._cmd_window)
        self.set_commands(limb_targets=tuple(float(x) for x in v))

    def _command_block_obs(self) -> np.ndarray:
        """The 25-dim v4 command block: masked groups emit their command
        values, inactive groups emit zeros (the policy reads zeros; the value
        semantics live in rewards/commands, never in the obs encoding)."""
        mask = self._commands["mask"]
        limb = np.zeros(15, dtype=np.float32)
        if mask[0] and self._commands["limb_targets"] is not None:
            limb = np.asarray(self._commands["limb_targets"],
                              dtype=np.float32)
        balance = np.zeros(2, dtype=np.float32)
        if mask[1]:
            balance = np.asarray(self._commands["com_shift"],
                                 dtype=np.float32)
        wp = np.zeros(3, dtype=np.float32)
        if mask[2]:
            wp = np.asarray(self._commands["waypoint"], dtype=np.float32)
        stop = np.zeros(1, dtype=np.float32)
        if mask[3]:
            stop = np.asarray([float(self._commands["stop"])],
                              dtype=np.float32)
        return np.concatenate([limb, balance, wp, stop,
                               np.asarray(mask, dtype=np.float32)])

    def _collect_bodies(self) -> None:
        """Cache the default agent's rig subtree (self-pair contact filter)
        and the foot body ids. Rig membership: every body whose parent chain
        reaches the default agent's container body."""
        m = self.world.m
        try:
            container = int(self.world._ag(None).bid)
        except Exception:
            container = int(m.body("agent").id)
        rig = set()
        for i in range(m.nbody):
            bid = i
            while bid != 0:
                if bid == container:
                    rig.add(i)
                    break
                bid = int(m.body_parentid[bid])
        self._rig_body_ids = rig
        self._foot_bids = {s: int(m.body(f"{s}_foot").id) for s in ("left", "right")}

    def _reset_boundary_stats(self) -> None:
        self.clip_stats = {"calls": 0, "calls_with_clipping": 0, "elements_clipped": 0,
                           "max_elements_in_call": 0, "rejected_calls": 0,
                           "faulted_steps": 0}
        self._warn_baseline = dict(self.world.physics_warnings())

    def _metrics_reset(self, seed=None) -> None:
        self._metrics = {
            "steps": 0, "sim_time": 0.0, "start_position": None, "seeds_used": [],
            "path_len": 0.0, "heading_progress_rewarded": 0.0,
            "heading_progress_raw": 0.0, "up_z_sum": 0.0, "up_z_min": None,
            "effort_total_j": 0.0, "components": {}, "reward_total": 0.0,
            "falls": 0, "faulted_windows": 0, "truncated_at": None, "final_position": None,
        }
        self._episode_seed = seed
        self._upright_window.clear()
        self._last_position = None

    def action_clipping(self) -> dict:
        """B1: record and expose action clipping so excessive clipping is
        detectable. elements_clipped / calls is the clipping rate."""
        cs = dict(self.clip_stats)
        cs["clipping_rate"] = (cs["calls_with_clipping"] / cs["calls"]) if cs["calls"] else 0.0
        return cs

    def faulted(self) -> bool:
        """True once physics went numerically invalid this episode (warnings seen)."""
        return self.clip_stats["faulted_steps"] > 0

    # ------------------------------------------------------------ curriculum
    def set_phase(self, phase: dict) -> None:
        """Apply one curriculum phase. THE phase-appliance every stage uses.

        Non-cumulative semantics: every call REWRITES the task parameters from
        their sources of truth, never from the previous phase's values:
          gravity_scale   key present -> rewrite from the CAPTURED BASE gravity;
                          absent -> leave gravity as-is (physical sim state)
          upright_assist  ALWAYS rewritten (absent -> 0.0)
          target_velocity ALWAYS rewritten (absent -> the config default)
          reward_coeffs   ALWAYS rewritten: config defaults, plus the phase's
                          partial overrides when the key is present
          max_steps       key present -> set; absent -> leave (duration knob)
        Unknown reward keys raise, so a configuration mistake cannot pass
        silently (see also rl/rewards.compute_reward).
        """
        if not isinstance(phase, dict):
            raise TypeError("phase must be a dict")
        if 'gravity_scale' in phase:
            g = float(phase['gravity_scale'])
            self._cur_gravity_scale = g
            self.world.m.opt.gravity[:] = self._base_gravity * g
        self.upright_assist = (min(1.0, max(0.0, float(phase['upright_assist'])))
                               if 'upright_assist' in phase else 0.0)
        if 'target_velocity' in phase:
            tv = np.asarray(phase['target_velocity'], dtype=float).reshape(3)
            if not np.all(np.isfinite(tv)):
                raise ValueError("target_velocity must be finite")
            self._target_vel = tv
        else:
            self._target_vel = np.asarray(
                self._rl_cfg.get('target_velocity', [0.5, 0.0, 0.0]),
                dtype=np.float64)
        rc = dict(phase['reward_coeffs']) if 'reward_coeffs' in phase else {}
        unknown = set(rc) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown reward coefficients: {sorted(unknown)}")
        self._reward_cfg = {**rl_cfg_reward_defaults(self._rl_cfg), **rc}
        if 'commands' in phase:
            # M7-0a: optional per-stage command group (mask bits + values).
            # Non-cumulative like every phase knob; partial dicts fill the
            # rest with the inactive defaults.
            if self._is_v3:
                raise ValueError(
                    "phase 'commands' requires contract version 4")
            c = phase['commands']
            if not isinstance(c, dict):
                raise TypeError("phase 'commands' must be a dict")
            self.set_commands(mask=c.get('mask', (0, 0, 0, 0)),
                              limb_targets=c.get('limb_targets'),
                              com_shift=c.get('com_shift'),
                              waypoint=c.get('waypoint'),
                              stop=c.get('stop'),
                              limb_group=c.get('limb_group', 'all'))
        elif not self._is_v3:
            # non-cumulative: a phase without commands ⇒ zero command state
            # (mask all off) — the standing/controlled/sustained phases are
            # exactly the v3 behaviour still.
            self._commands = {"mask": (0, 0, 0, 0), "limb_targets": None,
                              "com_shift": (0.0, 0.0),
                              "waypoint": (0.0, 0.0, 0.0), "stop": 0,
                              "limb_group": "all"}
        if 'command_schedule' in phase:
            # M7-1a: per-episode limb hold schedule (the plan itself is built
            # per reset(seed) — stage-agnostic episode data). Non-cumulative
            # like every phase knob: a phase without the key clears it. A
            # schedule declares its own limb mask + group; a phase-declared
            # 'commands' block must not contradict it (the schedule sets a
            # consistent one itself).
            c = phase.get('commands')
            if isinstance(c, dict):
                sched_g = phase['command_schedule'].get('group')
                if 'limb_group' in c and c['limb_group'] != sched_g:
                    raise ValueError(
                        "phase 'commands.limb_group' contradicts "
                        "command_schedule.group "
                        f"({c.get('limb_group')!r} != {sched_g!r})")
                if c.get('mask') is not None and tuple(c['mask'])[0] != 1:
                    raise ValueError(
                        "phase 'commands.mask' with a command_schedule must "
                        "arm the limb bit (bit 0 = 1)")
            self._set_command_schedule(phase['command_schedule'])
        elif not self._is_v3:
            self._set_command_schedule(None)
        if 'max_steps' in phase:
            self.max_steps = int(phase['max_steps'])

    def set_target_velocity(self, vec) -> None:
        """Set the world-frame velocity the reward tracks ([m/s], 3-vector)."""
        tv = np.asarray(vec, dtype=float).reshape(3)
        if not np.all(np.isfinite(tv)):
            raise ValueError("target_velocity must be finite")
        self._target_vel = tv

    @property
    def target_velocity(self) -> np.ndarray:
        return self._target_vel.copy()

    # ------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None):
        """Fresh episode: rebuild the world (no state leakage), re-apply the
        curriculum phase state, re-register PD, zero all episode accounting.

        Deterministic: identical calls give identical episodes. A fixed
        `seed` draws the start position from a small documented jitter box
        through the world's own setup mechanism (initial conditions only);
        `seed=None` keeps the exact default spawn position. Nothing else is
        randomized at reset.
        """
        from ..world import PhysicsWorld
        self.world = PhysicsWorld(cfg=self.cfg)
        self.world.m.opt.gravity[:] = self._base_gravity * self._cur_gravity_scale
        if seed is not None:
            rng = np.random.default_rng(int(seed))
            jit = self._start_jitter
            dx, dy = rng.uniform(-jit, jit, size=2)
            base = np.asarray(self.world.agent_pos(), dtype=float)
            self.world.setup_agent_pose(self.world.default_agent,
                                        position=[float(base[0] + dx),
                                                  float(base[1] + dy)])
        # Passive posture hinges: conventional PD (env infrastructure, tagged
        # separately from the policy's demands for saturation attribution).
        for name, (kp, kd) in self._posture_gains.items():
            self.world.set_joint_target(name, 0.0, kp=kp, kd=kd, source="env_posture")
        for name in self.joint_names:
            kp, kd = self._gains.get(name, (15.0, 1.5))
            self.world.set_joint_target(name, 0.0, kp=kp, kd=kd, source="neural_policy")
        # M7-1a: build this episode's command plan AFTER the reset pose is
        # placed and the neutral PD is registered; apply window 0's command.
        self._plan_command_episode(seed)
        self._collect_bodies()
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        self._reset_boundary_stats()
        self._metrics_reset(seed=seed)
        self._last_position = np.asarray(self.world.agent_pos(), dtype=float).copy()
        self._metrics["start_position"] = self._last_position.tolist()
        return self._get_obs()

    # ------------------------------------------------------------ stepping
    def step(self, action: np.ndarray):
        """One decision window: map the action to PD targets, run
        `action_repeat` physics substeps (500 Hz PD), then compute the
        consolidated reward from engine measurements.

        Returns (obs, reward, terminated, truncated, info).
        """
        # ---- B1 action-boundary guards (before ANY physical effect) ----
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            self.clip_stats["rejected_calls"] += 1
            raise ActionError("invalid_action_shape",
                              f"action must have shape ({self.action_dim},); got {action.shape}",
                              shape=list(action.shape))
        if not np.all(np.isfinite(action)):
            bad = np.nonzero(~np.isfinite(action))[0].tolist()
            self.clip_stats["rejected_calls"] += 1
            self.world._safety_log_event("env_action_rejected",
                                         joints=[self.joint_names[i] for i in bad],
                                         source="neural_policy")
            raise ActionError("non_finite_action",
                              "action contains NaN or infinity; rejected before joint mapping",
                              bad_indices=bad)
        over = np.abs(action) > 1.0
        clipped_elems = 0
        if over.any():
            clipped_elems = int(over.sum())
            self.clip_stats["calls_with_clipping"] += 1
            self.clip_stats["elements_clipped"] += clipped_elems
            self.clip_stats["max_elements_in_call"] = \
                max(self.clip_stats["max_elements_in_call"], clipped_elems)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.clip_stats["calls"] += 1
        # M7-1a: publish this window's scheduled command (before the PD
        # mapping, so the window's obs, MeasuredStep.commands and reward all
        # read the same command state; actions are never masked by it).
        self._tick_command_schedule()
        # map normalized action to target positions (documented encoding)
        for i, name in enumerate(self.joint_names):
            low, high = self.joint_range[name]
            target = low + (action[i] + 1) * 0.5 * (high - low)
            self._last_targets[name] = float(target)
            kp, kd = self._gains.get(name, (15.0, 1.5))
            self.world.set_joint_target(name, target, kp=kp, kd=kd, source="neural_policy")
        # Optional curriculum assistance (strength 0..1): blend the policy's
        # targets toward a neutral standing posture (0 rad) on the balance-
        # critical joints — spine + hips + knees — through the SAME PD torque
        # path as every other command. Physics-only: the MuJoCo solver
        # produces all motion; no teleports, no direct qpos writes.
        if self.upright_assist > 0.0:
            w = min(1.0, float(self.upright_assist))
            for name in ("spine_L2", "left_hip", "right_hip", "left_knee", "right_knee"):
                if name in self._last_targets:
                    blended = (1.0 - w) * self._last_targets[name]
                    kp, kd = self._gains.get(name, (15.0, 1.5))
                    self.world.set_joint_target(name, blended, kp=kp, kd=kd,
                                                source="neural_policy")
                    self._last_targets[name] = blended
        # ---- advance simulation: `action_repeat` substeps, PD tracks each ----
        pos_before = np.asarray(self.world.agent_pos(), dtype=float).copy()
        dist_before = self._target_dist()          # v4 target-point seam
        effort_j = 0.0
        dt = float(self.world.m.opt.timestep)
        fault = None
        for _ in range(self.action_repeat):
            self.world.step()
            # mechanical work of the POLICY's own PD torques over this substep
            # (qfrc_applied holds the torque the just-finished step applied)
            qf = np.asarray(self.world.d.qfrc_applied, dtype=np.float64)[self._dofadr]
            qv = np.asarray(self.world.d.qvel, dtype=np.float64)[self._dofadr]
            effort_j += float(np.abs(qf * qv).sum()) * dt
            # B1 physics-health watch: a new MuJoCo warning means the engine
            # suppressed physics inside this window.
            warn = self.world.physics_warnings()
            new_warns = {k: v for k, v in warn.items()
                         if v > self._warn_baseline.get(k, 0)}
            if new_warns:
                fault = {"physics_warnings": sorted(new_warns)}
                self._warn_baseline = dict(warn)
                break
        obs = self._get_obs()
        pos_after = np.asarray(self.world.agent_pos(), dtype=float).copy()
        vel = self.world.agent_vel().astype(np.float64)
        dpos = pos_after - pos_before
        if self._last_position is None:
            self._last_position = pos_before.copy()
        # physical termination truth (debounced agent_fallen)
        fell = bool(self.world.agent_fallen())
        self.step_count += 1
        self._metrics["steps"] = self.step_count
        self._metrics["sim_time"] = float(self.world.time)
        self._metrics["path_len"] += float(np.linalg.norm(dpos))
        up_z = float(obs[34])
        self._upright_window.append(up_z)
        self._metrics["up_z_sum"] += up_z
        self._metrics["up_z_min"] = (up_z if self._metrics["up_z_min"] is None
                                     else min(self._metrics["up_z_min"], up_z))
        self._metrics["effort_total_j"] += effort_j
        is_fault = bool(fault) or bool(np.any(~np.isfinite(obs)))
        if fell:
            self._metrics["falls"] += 1
        # ---- consolidated reward from engine measurements (one pipeline) ----
        if is_fault:
            # B1: the window's physics was (partially) suppressed — reward is
            # zeroed and the window is excluded from valid learning data by
            # contract (info['faulted']).
            reward = 0.0
            components = {k: 0.0 for k in COMPONENT_KEYS}
            obs = np.where(np.isfinite(obs), obs, 0.0).astype(np.float32)
        else:
            prog_raw = heading_progress_raw(dpos, self._target_vel)
            prog_rewarded = min(prog_raw,
                                float(np.linalg.norm(self._target_vel)) * self.action_repeat * dt)
            self._metrics["heading_progress_raw"] += prog_raw
            self._metrics["heading_progress_rewarded"] += prog_rewarded
            # B4 posture ask measurement: the chest's world up-vector z. The
            # spine_L2 BODY (same name as its hinge) is the chest — it carries
            # the g:chest geom and the arms/head subtree, and its hinge pitch
            # is the chest tilt the posture probes measure. Same channel as
            # the torso up-vector (xmat col 2, z component) so it composes
            # with the existing reward weights without renormalisation.
            chest_mat = np.asarray(self.world.d.xmat[
                int(self.world.m.body('spine_L2').id)], dtype=np.float64)
            chest_up_z = float(chest_mat.reshape(3, 3)[2, 2])
            # M7-0a v4 engine measurements (None in v3 mode → components 0.0):
            # full-body COM xy (P4 method), normalized joint positions over
            # the contract's joint list, and target-point distances.
            if self._is_v3:
                com_xy_v4, jpos_norm = None, None
            else:
                com_xy_v4 = self._com_xy()
                jpos_norm = np.zeros(len(self.joint_names), dtype=np.float64)
                for i, name in enumerate(self.joint_names):
                    jj = self.world.m.joint(name)
                    lo, hi = self.joint_range[name]
                    span = (hi - lo) if (hi - lo) != 0.0 else 1.0
                    qposadr = int(jj.qposadr[0])
                    jpos_norm[i] = (float(self.world.d.qpos[qposadr])
                                    - lo) / span * 2.0 - 1.0
            dist_after = self._target_dist()
            ms = MeasuredStep(
                position_before=pos_before, position_after=pos_after,
                velocity=vel, up_z=up_z, effort_j=effort_j,
                window_s=self.action_repeat * dt, fell=fell,
                up_z_window=list(self._upright_window),
                chest_up_z=chest_up_z,
                target_velocity=tuple(float(x) for x in self._target_vel),
                commands={**self._commands},
                joint_pos_norm=(None if jpos_norm is None else jpos_norm),
                com_xy=com_xy_v4,
                dist_to_target=(None if dist_before is None or dist_after is None
                                else (float(dist_before), float(dist_after))))
            reward, components = compute_reward(ms, action, self.prev_action,
                                                self._reward_cfg)
        self.prev_action = action.copy()
        # ---- termination / truncation ----
        terminated = bool(is_fault or fell)
        truncated = bool(not terminated and self.step_count >= self.max_steps)
        if truncated:
            self._metrics["truncated_at"] = self.step_count
        if is_fault:
            self.clip_stats["faulted_steps"] += 1
            self._metrics["faulted_windows"] += 1
        # ---- episode accounting ----
        self._metrics["reward_total"] += reward
        for k, v in components.items():
            self._metrics["components"][k] = \
                self._metrics["components"].get(k, 0.0) + float(v)
        self._metrics["final_position"] = pos_after.tolist()
        info = {
            "components": components,
            "clipped_elements": clipped_elems,
            "contacts": self._foot_contacts_n(),
            "effort_j": effort_j,
            "target_velocity": self._target_vel.tolist(),
        }
        if is_fault:
            info['faulted'] = True
            info['physics_warnings'] = fault.get("physics_warnings", []) if fault else []
        if fell:
            info['fallen'] = True
        return obs, float(reward), bool(terminated), truncated, info

    # ------------------------------------------------------------ observation
    def _foot_contacts_n(self) -> tuple:
        """Per-foot ground-support normal force (N): contacts against bodies
        OUTSIDE this rig (never self-pairs — the shipped asset embeds each
        foot partly inside its shin capsule, a persistent internal contact
        pair that must not read as support), |normal z| > 0.7 — the same
        support-contact rule PhysicsWorld._post_step uses for `grounded`.
        Snapshot at the decision boundary."""
        import mujoco as _mj
        d = self.world.d
        out = {"left": 0.0, "right": 0.0}
        body_of = self.world.m.geom_bodyid
        for ci in range(int(d.ncon)):
            c = d.contact[ci]
            b1 = int(body_of[int(c.geom1)])
            b2 = int(body_of[int(c.geom2)])
            for side, bid in self._foot_bids.items():
                if bid in (b1, b2):
                    other = b2 if b1 == bid else b1
                    if other in self._rig_body_ids:
                        break  # self-pair (asset quirk) — never ground support
                    nz = float(d.contact.frame[ci, 2])
                    if abs(nz) <= 0.7:
                        break
                    f6 = np.zeros(6)
                    _mj.mj_contactForce(self.world.m, d, ci, f6)
                    out[side] += abs(float(f6[0]))
                    break
        return (out["left"], out["right"])

    def _get_obs(self) -> np.ndarray:
        # joint positions and velocities from world data — the fixed 14-name
        # layout in BOTH contract modes (the v3 prefix [0:43] is forever;
        # the waist's measured position is reward-side only, never observed).
        qpos = self.world.d.qpos.astype(np.float32)
        qvel = self.world.d.qvel.astype(np.float32)
        joint_pos = []
        joint_vel = []
        for name in self._obs_joint_names:
            j = self.world.m.joint(name)
            pos_idx = int(j.qposadr[0])
            vel_idx = int(j.dofadr[0])
            lo, hi = self.joint_range[name]
            span = (hi - lo) if (hi - lo) != 0.0 else 1.0
            joint_pos.append(float((qpos[pos_idx] - lo) / span * 2.0 - 1.0))
            joint_vel.append(float(qvel[vel_idx]))
        joint_pos = np.array(joint_pos, dtype=np.float32)
        joint_vel = np.array(joint_vel, dtype=np.float32) / np.float32(self._joint_vel_scale)
        # torso orientation from rotation matrix; LOCAL +Y (column 1) is the
        # chest/heading axis of this rig (measured: the gait walks heading
        # h with col1 -> (cos h, sin h) after the waist turn).
        torso_bid = self.world.m.body('torso').id
        row = np.asarray(self.world.d.xmat[torso_bid], dtype=np.float32)
        mat3 = row.reshape((3, 3))
        torso_up = mat3[:, 2]
        torso_fwd = mat3[:, 1]
        # torso linear velocity (slide joints, world frame)
        torso_vel = (self.world.agent_vel().astype(np.float32)
                     / np.float32(self._root_vel_scale))
        # upright margin (z component of up vector)
        upright_margin = np.array([torso_up[2]], dtype=np.float32)
        cl, cr = self._foot_contacts_n()
        contacts = (np.array([cl, cr], dtype=np.float32)
                    / np.float32(self._contact_force_scale))
        # task conditioning (v3, APPEND-only): the commanded target velocity the
        # reward tracks, world frame, same scale as the observed torso velocity
        # (root_vel_scale) so commanded and measured velocities are comparable
        command = (self._target_vel.astype(np.float32)
                   / np.float32(self._root_vel_scale))
        obs = np.concatenate([joint_pos, joint_vel, torso_up, torso_vel,
                              upright_margin, torso_fwd, contacts, command])
        if not self._is_v3:
            # rl_env/4 APPEND-ONLY: the command block after the fixed v3 43.
            obs = np.concatenate([obs, self._command_block_obs()])
        return obs.astype(np.float32)

    # ------------------------------------------------------------ diagnostics
    def episode_summary(self) -> dict:
        """Everything measured this episode, from engine state. Does not reset."""
        up_steps = max(1, self.step_count)
        sim = self._metrics["sim_time"]
        return {
            "steps": self.step_count,
            "sim_time_s": sim,
            "start_position": self._metrics["start_position"],
            "final_position": self._metrics["final_position"],
            "path_len_m": self._metrics["path_len"],
            "net_displacement_m": (None if self._metrics["start_position"] is None
                                   or self._metrics["final_position"] is None else
                                   float(np.linalg.norm(
                                       np.asarray(self._metrics["final_position"])
                                       - np.asarray(self._metrics["start_position"])))),
            "mean_speed_m_s": (self._metrics["path_len"] / sim if sim > 0 else 0.0),
            "heading_progress_rewarded_m": self._metrics["heading_progress_rewarded"],
            "heading_progress_raw_m": self._metrics["heading_progress_raw"],
            "upright_fraction": (self._metrics["up_z_sum"] / up_steps),
            "min_up_z": self._metrics["up_z_min"],
            "effort_total_j": self._metrics["effort_total_j"],
            "effort_mean_w": (self._metrics["effort_total_j"] / sim if sim > 0 else 0.0),
            "reward_total": self._metrics["reward_total"],
            "components": dict(self._metrics["components"]),
            "falls": self._metrics["falls"],
            "faulted_windows": self._metrics["faulted_windows"],
            "truncated_at": self._metrics["truncated_at"],
            "action_clipping": self.action_clipping(),
            "seed": self._episode_seed,
            "target_velocity": self._target_vel.tolist(),
        }

    # ------------------------------------------------------------ schema
    def observation_layout(self) -> dict:
        """Machine-readable observation contract (the .aipg manifest carries it)."""
        sections = [
                {"name": "joint_pos", "slice": [0, 14], "unit": "normalized",
                 "note": "(q - lo)/(hi - lo)*2 - 1; order = joint_order"},
                {"name": "joint_vel", "slice": [14, 28], "unit": "rad/s",
                 "note": f"raw qvel / {self._joint_vel_scale} "
                         "(config rl.observation.joint_vel_scale; walk peak 4.83 + margin)"},
                {"name": "torso_up", "slice": [28, 31], "unit": "unit vector",
                 "note": "torso xmat column 2, world frame"},
                {"name": "torso_lin_vel", "slice": [31, 34], "unit": "m/s",
                 "note": f"root slide velocity, world, / {self._root_vel_scale} "
                         f"(max_speed 4.0 + margin)"},
                {"name": "upright_margin", "slice": [34, 35], "unit": "raw",
                 "note": "torso up z"},
                {"name": "torso_forward", "slice": [35, 38], "unit": "unit vector",
                 "note": "torso xmat column 1 (chest/heading axis), world frame"},
                {"name": "foot_ground_force", "slice": [38, 40], "unit": "N",
                 "note": f"ground-support normal force (|nz|>0.7, non-rig contacts), "
                         f"/ {self._contact_force_scale} (left, right)"},
                {"name": "command", "slice": [40, 43], "unit": "m/s",
                 "note": f"COMMANDED target velocity v* the reward tracks, WORLD "
                         f"frame, / {self._root_vel_scale} (same scale as "
                         f"torso_lin_vel); task conditioning (B3); APPEND-only "
                         f"slice over v2"},
        ]
        command_block = None
        if not self._is_v3:
            sections = sections + [
                {"name": "limb_pose_target", "slice": [43, 58], "unit": "normalized",
                 "note": "commanded pose targets, the ACTION mapping constants: "
                         "target = lo + (n+1)/2*(hi-lo), n in [-1,1]; 0 = neutral "
                         "(0 rad, symmetric ranges); order = joint_order; "
                         "inactive (mask bit 0) = zeros"},
                {"name": "com_shift", "slice": [58, 60], "unit": "normalized",
                 "note": f"commanded COM shift (x, y) normalized n in [-1,1] = "
                         f"n * {COM_SHIFT_SCALE_XY[0]} / {COM_SHIFT_SCALE_XY[1]} m "
                         f"from the measured standing anchor {COM_ANCHOR_XY} m "
                         "(P4); inactive = zeros"},
                {"name": "waypoint", "slice": [60, 63], "unit": "normalized",
                 "note": f"(ux, uy, dh): direction in the yaw-only heading frame "
                         f"(torso col-1 horizontal projection + left-hand "
                         f"perpendicular), dh = dist/{WAYPOINT_D_REF} m; "
                         "inactive = zeros"},
                {"name": "stop_request", "slice": [63, 64], "unit": "flag",
                 "note": "{0,1}: park near the waypoint; inactive = 0"},
                {"name": "cmd_mask", "slice": [64, 68], "unit": "flags",
                 "note": "the stage selector in-obs: {0,1} x (limb, balance, "
                         "waypoint, stop), fixed order; no task one-hot; "
                         "gate = obs zeros + reward consumption ONLY — the "
                         "action space is never masked (M7-0 freeze §1)"},
            ]
            command_block = {
                "mask_order": list(_COMMAND_GROUPS),
                "limb_encoding": "same as action mapping (normalized targets)",
                "limb_targets_order": list(self.joint_names) if not self._is_v3
                                      else None,
                "limb_group_index": {k: list(v) for k, v in LIMB_GROUPS.items()},
                "com_shift_anchor_xy_m": list(COM_ANCHOR_XY),
                "com_shift_scale_xy_m": list(COM_SHIFT_SCALE_XY),
                "waypoint_d_ref_m": WAYPOINT_D_REF,
                "waypoint_frame": "yaw-only heading frame (heading-invariant)",
                "schedules": "stage-defined, seeded, deterministic episode data",
            }
        out = {
            "layout_version": self.contract_version,
            "dim": self.obs_dim,
            "dtype": "float32",
            "sections": sections,
            "joint_order": list(self._obs_joint_names),
            "joint_ranges_rad": {n: [float(self.joint_range[n][0]),
                                     float(self.joint_range[n][1])]
                                 for n in self._obs_joint_names},
            "normalisation": {
                "joint_pos": "min-max to [-1,1] per joint range",
                "joint_vel": f"/ {_JOINT_VEL_SCALE}",
                "root_vel": f"/ {_ROOT_VEL_SCALE}",
                "contact_force": f"/ {_CONTACT_FORCE_SCALE}",
                "command": f"/ {_ROOT_VEL_SCALE} (root_vel_scale; matches torso_lin_vel)",
                "unit_vectors_up_fwd_and_margin": "none",
                "command_block": " limb targets normalized [-1,1]; com_shift [-1,1]; "
                                 "waypoint unitless; stop flag; mask bits",
            },
            "command_frame": "world",
            "fault_semantics": {
                "non_finite_observation": "transition FAULTED, zeroed, terminated",
                "mjwarn_during_window": "transition FAULTED, zeroed, terminated",
            },
        }
        if command_block is not None:
            out["command_block"] = command_block
        return out

    def action_layout(self) -> dict:
        """Machine-readable action contract (the .aipg manifest carries it)."""
        return {
            "layout_version": self.contract_version,
            "action_dim": self.action_dim,
            "encoding": "joint_targets_normalized",
            "mapping": "target = lo + (a + 1)*0.5*(hi - lo)",
            "joints": [{"name": n,
                        "range_rad": [float(self.joint_range[n][0]),
                                      float(self.joint_range[n][1])],
                        "kp": float(self._gains.get(n, (15.0, 1.5))[0]),
                        "kd": float(self._gains.get(n, (15.0, 1.5))[1])}
                       for n in self.joint_names],
            "passive_conventional_pd": [
                {"name": n, "target_rad": 0.0, "kp": float(kp), "kd": float(kd)}
                for n, (kp, kd) in self._posture_gains.items()],
            "decimation_steps": self.action_repeat,
            "control_period_ms": self.action_repeat * float(
                self.world.m.opt.timestep) * 1000.0,
            "pd_control_hz": round(1.0 / float(self.world.m.opt.timestep), 1),
            "actuation_mode": "pd_via_set_joint_target",
            "source_tag": "neural_policy",
        }


def rl_cfg_reward_defaults(rl_cfg: dict | None) -> dict:
    """Reward coefficient defaults from the resolved rl reward config block."""
    d = dict(DEFAULTS)
    rlc = dict((rl_cfg or {}).get('reward') or {})
    for k, v in rlc.items():
        if k in d:
            d[k] = v
    return d
