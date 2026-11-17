import math
import os

import numpy as np
import yaml


def test_humanoid_pd_walk():
    """Direct PD-driven walk: the coordinated joint-target pattern (hips
    antiphase cosine sweep, heel-back swing-knee bumps, level-foot ankles) is
    written straight to set_joint_target -- no gait controller, no move() call.
    Forward displacement must emerge from leg actuation and foot-ground contact.

    (The old version drove two equal-and-opposite hip sinusoids, which cancel
    exactly on a symmetric rig -- net motion ~1e-14 m -- and relied on the
    removed root-traction pathway via move(); a leg-driven walk needs the
    stance/swing asymmetry, which is what this pattern encodes.)"""
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['agent_body']['type'] = 'humanoid'
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(cfg)
    # no gait controller: raw PD targets only
    world._gait_controller = None

    freq = 1.1
    amp = 0.22                      # stride sweep (rad)
    ka = 0.55                       # swing-knee heel-back flexion bump
    start = world.agent_pos().copy()
    start_time = world.time
    while world.time - start_time < 4.0:
        phi = (world.time - start_time) * freq
        th = 2.0 * math.pi * phi
        c, s = math.cos(th), math.sin(th)
        ramp = min(1.0, phi / 1.5)
        a = amp * ramp
        # left foot lands ahead at phi=0; roles swap at phi=0.5
        hip_l, hip_r = a * c, -a * c
        knee_l = -ka * max(0.0, -s)
        knee_r = -ka * max(0.0, s)
        world.set_joint_target('left_hip', hip_l, kp=120.0, kd=5.0)
        world.set_joint_target('right_hip', hip_r, kp=120.0, kd=5.0)
        world.set_joint_target('left_knee', knee_l, kp=50.0, kd=5.0)
        world.set_joint_target('right_knee', knee_r, kp=50.0, kd=5.0)
        world.set_joint_target('left_ankle', -(hip_l + knee_l), kp=25.0, kd=1.0)
        world.set_joint_target('right_ankle', -(hip_r + knee_r), kp=25.0, kd=1.0)
        world._step()
    end = world.agent_pos().copy()
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    # expect modest PD-driven displacement (legs only, no traction)
    assert dist >= 0.05, f"walked {dist:.3f}m, expected >= 0.05m"
    assert not np.isnan(world.d.qacc).any(), "solver blow-up during PD walk"
    # the walk must not have collapsed: the root column stays above the
    # fallen threshold (z > 0.44 m, see agent_fallen calibration)
    assert end[2] > 0.44, f"collapsed during PD walk: z={end[2]:.3f}"
    # Verify PD targets were registered for hips
    assert 'left_hip' in world._pd and 'right_hip' in world._pd, "Hip PD controllers not set"
