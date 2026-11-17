
import numpy as np
from ..world import PhysicsWorld


class LocoEnv:
    """Simple locomotion environment for the humanoid.
    Observation layout (float32):
        14 joint positions, 14 joint velocities,
        torso up-vector (3), torso linear velocity (3), upright margin (1).
    Action: 14 normalized [-1,1] joint targets -> radians via simple linear scaling.
    """
    obs_dim = 14 + 14 + 3 + 3 + 1
    action_dim = 14

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        # Ensure humanoid agent
        world_cfg = cfg.get('world', {})
        # Load full configuration and enforce humanoid body
        from ..config import load_config
        overrides = cfg.copy() if isinstance(cfg, dict) else {}
        # Ensure top-level agent_body with humanoid type
        agent_body = overrides.get('agent_body', {})
        agent_body['type'] = 'humanoid'
        overrides['agent_body'] = agent_body
        full_cfg = load_config(None, overrides=overrides)
        self.cfg = full_cfg
        self.world = PhysicsWorld(cfg=full_cfg)
        # Curriculum state: assistance lives in THIS env (joint-space PD target blending;
        # never teleports, never direct qpos writes). Base gravity captured once so phase
        # application never spirals the model's gravity.
        self._base_gravity = np.array(self.world.m.opt.gravity, dtype=float).copy()
        self._cur_gravity_scale = 1.0
        self.upright_assist = float(cfg.get('upright_assist', 0.0) or 0.0)
        self._last_targets = {}
        # Enumerate hinge joints for the humanoid; exclude movable-object freejoint naming
        # ('j:<oid>') and the agent root slide joints (agent_x/y/z).
        try:
            import mujoco as _mj
            hinge_code = int(_mj.mjtJoint.mjJNT_HINGE)
        except Exception:
            hinge_code = 0
        m = self.world.m
        joint_names = []
        for i in range(m.njnt):
            if int(m.jnt_type[i]) == hinge_code:
                # MjModel exposes names via mj_id2name (there is no m.jnt_name array).
                try:
                    name = _mj.mj_id2name(m, int(_mj.mjtObj.mjOBJ_JOINT), i)
                except Exception:
                    name = None
                if isinstance(name, (bytes, bytearray)):
                    name = name.decode()
                if name and not str(name).startswith("j:"):
                    # The waist turn and the foot-yaw hinges are posture/servo
                    # joints (the gait controller holds them); excluding them
                    # keeps the locomotion action space at the 14 core DOFs the
                    # reward/obs layout was built for (waist/foot additions to
                    # the rig are Nov 17; the obs slots must stay stable).
                    if name in ("left_foot_yaw", "right_foot_yaw",
                                "waist_yaw"):
                        continue
                    joint_names.append(str(name))
        # Keep first 14 humanoid joints
        self.joint_names = joint_names[:14]
        m = self.world.m
        # Joint ranges: read the REAL limited-joint ranges from the model; fallback only
        # for unlimited joints.
        self.joint_range = {}
        for name in self.joint_names:
            j = self.world.m.joint(name)
            if int(self.world.m.jnt_limited[j.id]):
                self.joint_range[name] = (float(self.world.m.jnt_range[j.id][0]),
                                          float(self.world.m.jnt_range[j.id][1]))
            else:
                self.joint_range[name] = (-np.pi / 2, np.pi / 2)
        # Register PD controllers with locomotion-grade gains. The world's
        # set_joint_target default (kp 15) is far too soft: 15 Nm does not
        # swing a leg against the body weight and a scripted/RL walk never
        # moves. These are controller parameters through the same honest
        # joint-target pathway (no qpos/qvel writes).
        self._gains = {
            "left_hip": (60.0, 5.0), "right_hip": (60.0, 5.0),
            "left_knee": (60.0, 5.0), "right_knee": (60.0, 5.0),
            "left_ankle": (25.0, 1.0), "right_ankle": (25.0, 1.0),
            "spine_L2": (35.0, 4.0),
            "left_shoulder": (25.0, 3.0), "right_shoulder": (25.0, 3.0),
            "left_elbow": (10.0, 2.0), "right_elbow": (10.0, 2.0),
        }
        for name in self.joint_names:
            kp, kd = self._gains.get(name, (15.0, 1.5))
            self.world.set_joint_target(name, 0.0, kp=kp, kd=kd)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        self.max_steps = cfg.get('max_steps', 1000)

    def set_phase(self, phase: dict) -> None:
        """Apply one curriculum phase: {'gravity_scale', 'upright_assist', 'max_steps'}.
        Gravity is rewritten from the CAPTURED BASE gravity (never cumulative), and
        assistance is joint-space PD target blending (see step()) — physics-only.
        The MuJoCo solver produces all motion; this env never teleports bodies."""
        if not isinstance(phase, dict):
            raise TypeError("phase must be a dict")
        g = float(phase.get('gravity_scale', 1.0))
        self._cur_gravity_scale = g
        self.world.m.opt.gravity[:] = self._base_gravity * g
        if 'upright_assist' in phase:
            self.upright_assist = min(1.0, max(0.0, float(phase['upright_assist'])))
        if 'max_steps' in phase:
            self.max_steps = int(phase['max_steps'])

    def reset(self):
        # PhysicsWorld exposes no reset(); the project's convention is a fresh world per
        # episode (see tests/conftest.py fixtures). Rebuild and re-apply curriculum state.
        from ..world import PhysicsWorld
        self.world = PhysicsWorld(cfg=self.cfg)
        self.world.m.opt.gravity[:] = self._base_gravity * self._cur_gravity_scale
        for name in self.joint_names:
            kp, kd = getattr(self, "_gains", {}).get(name, (15.0, 1.5))
            self.world.set_joint_target(name, 0.0, kp=kp, kd=kd)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (self.action_dim,)
        # map normalized action to target positions
        for i, name in enumerate(self.joint_names):
            low, high = self.joint_range[name]
            target = low + (action[i] + 1) * 0.5 * (high - low)
            self._last_targets[name] = float(target)
            kp, kd = self._gains.get(name, (15.0, 1.5))
            self.world.set_joint_target(name, target, kp=kp, kd=kd)
        # Optional curriculum assistance (strength 0..1): blend the policy's targets toward
        # a neutral standing posture (0 rad) on the balance-critical joints — spine + hips
        # + knees — through the SAME PD torque path as every other command. Physics-only:
        # the MuJoCo solver produces all motion; no teleports, no direct qpos writes.
        if self.upright_assist > 0.0:
            w = min(1.0, float(self.upright_assist))
            for name in ("spine_L2", "left_hip", "right_hip", "left_knee", "right_knee"):
                if name in self._last_targets:
                    blended = (1.0 - w) * self._last_targets[name]
                    kp, kd = self._gains.get(name, (15.0, 1.5))
                    self.world.set_joint_target(name, blended, kp=kp, kd=kd)
                    self._last_targets[name] = blended
        # advance simulation
        self.world.step()
        obs = self._get_obs()
        # compute reward using configured components (no action_magnitude)
        from .rewards import DEFAULTS, velocity_tracking, upright, alive, energy_penalty
        coeffs = self.cfg.get('reward_coeffs', DEFAULTS)
        r = 0.0
        r += coeffs.get('velocity_tracking',0)*velocity_tracking(obs)
        r += coeffs.get('upright',0)*upright(obs)
        r += coeffs.get('alive',0)*alive(obs)
        r += coeffs.get('energy_penalty',0)*energy_penalty(action)
        reward = float(r)
        self.prev_action = action.copy()
        self.step_count += 1
        done = self.step_count >= self.max_steps
        # check fall termination if available
        if hasattr(self.world, 'agent_fallen') and self.world.agent_fallen():
            done = True
            info = {'fallen': True}
        else:
            info = {}
        return obs, reward, done, info

    def _get_obs(self):
        # joint positions and velocities from world data using joint names
        qpos = self.world.d.qpos.astype(np.float32)
        qvel = self.world.d.qvel.astype(np.float32)
        joint_pos = []
        joint_vel = []
        for name in self.joint_names:
            j = self.world.m.joint(name)
            pos_idx = int(j.qposadr[0])
            vel_idx = int(j.dofadr[0])
            joint_pos.append(qpos[pos_idx])
            joint_vel.append(qvel[vel_idx])
        joint_pos = np.array(joint_pos, dtype=np.float32)
        joint_vel = np.array(joint_vel, dtype=np.float32)
        # torso up vector from rotation matrix (third column); d.xmat is (nbody, 9) 2-D
        torso_bid = self.world.m.body('torso').id
        row = np.asarray(self.world.d.xmat[torso_bid], dtype=np.float32)
        mat3 = row.reshape((3, 3))
        torso_up = mat3[:,2]
        # torso linear velocity
        torso_vel = self.world.agent_vel().astype(np.float32)
        # upright margin (z component of up vector)
        upright_margin = np.array([torso_up[2]], dtype=np.float32)
        obs = np.concatenate([joint_pos, joint_vel, torso_up, torso_vel, upright_margin])
        return obs
