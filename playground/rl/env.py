"""LocoEnv — the single authoritative locomotion RL environment (M6-B B2).

Contract version 3 ("rl_env/3"): documented in full in docs/RL_ENVIRONMENT.md
and machine-readable via observation_layout() / action_layout(); the B5 .aipg
manifest reproduces exactly this schema. One env, one reward pipeline (rl/
rewards.py), one curriculum mechanism (rl/curriculum.py applying phases
through set_phase).

Observation v3 (float32[43]; v3 is an APPEND-ONLY extension of v2, which was
an APPEND-ONLY extension of v1: all v2 slices keep their meaning):
    [0:14]   joint positions, normalized (2q - lo - hi)/(hi - lo) in [-1, 1]
    [14:28]  joint velocities, normalized  qvel / 6.0 rad/s
    [28:31]  torso up-vector, world frame (unit)
    [31:34]  torso linear velocity, world frame, / 5.0 m/s
    [34:35]  upright margin (= torso up_z, raw)
    [35:38]  torso forward (chest) axis, world frame (unit; xmat column 1)
    [38:40]  per-foot ground-support normal force / 1000.0 N (left, right)
    [40:43]  COMMANDED target velocity v*, WORLD frame, / 5.0 m/s (B3: the
             movement objective the reward tracks is observed; see
             "task conditioning" in docs/RL_ENVIRONMENT.md §1.5)

Action (14 normalized joints, [-1, 1]) maps to PD targets
    target = lo + (a + 1) * 0.5 * (hi - lo)
through PhysicsWorld.set_joint_target — the world's single validated
joint-force path (B1: non-finite rejection, gain ceilings, per-joint torque
caps, saturation attribution). The PD tracks at 500 Hz (dt 0.002 s); a
decision is held for `action_repeat` substeps (default 10 -> 20 ms policy
period) — decimation lives here, the world is untouched.

Return protocol changed at B2 (documented migration, gym-style):
    step(action) -> (obs, reward, terminated, truncated, info)
- terminated: the episode ended for a PHYSICAL reason — the figure fell
  (agent_fallen truth) or the window is FAULTED (a MuJoCo mjWARN fired or the
  observation went non-finite; B1 semantics, reward zeroed, excluded from
  training data by contract).
- truncated: the time budget (max_steps decisions) ran out.

B1 boundary guards preserved unchanged:
- non-finite / wrong-shape actions are REJECTED (ActionError) before any
  mapping; finite out-of-range actions are CLIPPED into [-1, 1] with per-
  episode clip statistics (action_clipping()); invalid actions are never
  silently converted into valid ones;
- safety events are observable via the world safety log/stats, and MuJoCo
  numerical warnings mark transitions FAULTED.

Reset: PhysicsWorld is rebuilt per episode (project convention — no state
leakage). reset(seed=None) is fully deterministic; with a seed it draws the
start position from a small jittered box through the world's own setup
mechanism (initial-configuration class, documented; no mid-episode qpos/qvel
writes anywhere in this env — the MuJoCo solver produces all motion).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from ..world import PhysicsWorld
from ..world.physics import ActionError

from .rewards import (COMPONENT_KEYS, DEFAULTS, MeasuredStep, compute_reward,
                      heading_progress_raw)

CONTRACT_VERSION = 3

# Fixed normalisation constants (rl.observation in config; documented). Vel
# scales come from measured scripting envelopes: knee peak 4.83 rad/s during
# the scripted walk (ai-playground-audit/probe_b2_audit), root speed capped
# at 4.0 m/s by config.
_JOINT_VEL_SCALE = 6.0     # rad/s
_ROOT_VEL_SCALE = 5.0      # m/s
_CONTACT_FORCE_SCALE = 1000.0  # N per obs unit


class LocoEnv:
    obs_dim = 14 + 14 + 3 + 3 + 1 + 3 + 2 + 3   # 43, layout v3 (v2 40 + command)
    action_dim = 14
    contract_version = CONTRACT_VERSION

    # Passive posture hinges excluded from the action space (held by the
    # conventional PD registered at reset): waist_yaw, left/right_foot_yaw.
    _EXCLUDED = ("left_foot_yaw", "right_foot_yaw", "waist_yaw")

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
        # Enumerate hinge joints for the humanoid; exclude movable-object
        # freejoint naming ('j:<oid>') and the agent root slide joints
        # (agent_x/y/z). Deterministic model order, probe-verified:
        # [spine_L2, l_shoulder, l_elbow, r_shoulder, r_elbow, head_pitch,
        #  head_yaw, head_roll, l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
        try:
            import mujoco as _mj
            hinge_code = int(_mj.mjtJoint.mjJNT_HINGE)
        except Exception:
            hinge_code = 0
        m = self.world.m
        joint_names = []
        for i in range(m.njnt):
            if int(m.jnt_type[i]) == hinge_code:
                try:
                    name = _mj.mj_id2name(m, int(_mj.mjtObj.mjOBJ_JOINT), i)
                except Exception:
                    name = None
                if isinstance(name, (bytes, bytearray)):
                    name = name.decode()
                if name and not str(name).startswith("j:"):
                    if name in self._EXCLUDED:
                        continue
                    joint_names.append(str(name))
        self.joint_names = joint_names[:14]
        m = self.world.m
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
        # Register PD controllers with locomotion-grade gains. The world's
        # set_joint_target default (kp 15) is far too soft for load-bearing
        # joints. These are controller parameters through the same honest
        # joint-target pathway (no qpos/qvel writes); gains are part of the
        # action contract (action_layout()).
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
        # Passive posture hinges: fixed conventional PD targets (rig posture
        # the policy does not command; same magnitudes the gait controller
        # uses for these families). Registered at every reset, tagged
        # source="env_posture" so saturation attribution can separate them
        # from the policy's demands.
        self._posture_joints = ("waist_yaw", "left_foot_yaw", "right_foot_yaw")
        self._posture_gains = {"waist_yaw": (50.0, 5.0),
                               "left_foot_yaw": (8.0, 1.0),
                               "right_foot_yaw": (8.0, 1.0)}
        # dof addresses for the 14 policy joints (effort measurement)
        self._dofadr = [self._dof_of(n) for n in self.joint_names]
        self._collect_bodies()
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
            ms = MeasuredStep(
                position_before=pos_before, position_after=pos_after,
                velocity=vel, up_z=up_z, effort_j=effort_j,
                window_s=self.action_repeat * dt, fell=fell,
                up_z_window=list(self._upright_window),
                chest_up_z=chest_up_z,
                target_velocity=tuple(float(x) for x in self._target_vel))
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
        # joint positions and velocities from world data using joint names
        qpos = self.world.d.qpos.astype(np.float32)
        qvel = self.world.d.qvel.astype(np.float32)
        joint_pos = []
        joint_vel = []
        for name in self.joint_names:
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
        return {
            "layout_version": CONTRACT_VERSION,
            "dim": self.obs_dim,
            "dtype": "float32",
            "sections": [
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
            ],
            "joint_order": list(self.joint_names),
            "joint_ranges_rad": {n: [float(lo), float(hi)]
                                 for n, (lo, hi) in self.joint_range.items()},
            "normalisation": {
                "joint_pos": "min-max to [-1,1] per joint range",
                "joint_vel": f"/ {_JOINT_VEL_SCALE}",
                "root_vel": f"/ {_ROOT_VEL_SCALE}",
                "contact_force": f"/ {_CONTACT_FORCE_SCALE}",
                "command": f"/ {_ROOT_VEL_SCALE} (root_vel_scale; matches torso_lin_vel)",
                "unit_vectors_up_fwd_and_margin": "none",
            },
            "command_frame": "world",
            "fault_semantics": {
                "non_finite_observation": "transition FAULTED, zeroed, terminated",
                "mjwarn_during_window": "transition FAULTED, zeroed, terminated",
            },
        }

    def action_layout(self) -> dict:
        """Machine-readable action contract (the .aipg manifest carries it)."""
        return {
            "layout_version": CONTRACT_VERSION,
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
