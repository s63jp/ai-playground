import os
import yaml
import math

def test_humanoid_pd_walk():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['agent_body']['type'] = 'humanoid'
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(cfg)
    # Simple sinusoidal hip target for 3 seconds
    def set_targets(t):
        amp = 0.25
        freq = 1.0
        target = amp * math.sin(2 * math.pi * freq * t)
        world.set_joint_target('left_hip', target, kp=15.0, kd=1.5)
        world.set_joint_target('right_hip', -target, kp=15.0, kd=1.5)
    duration = 3.0
    speed = 0.4
    start = world.agent_pos().copy()
    start_time = world.time
    while world.time - start_time < duration:
        set_targets(world.time)
        world.move([1.0, 0.0, 0.0], speed * world.dt, stop=False)
        world._step()
    end = world.agent_pos().copy()
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    # Expect at least 80% of nominal distance (speed*duration)
    assert dist >= 0.8 * speed * duration, f"walked {dist:.3f}m, expected >= {0.8 * speed * duration:.3f}m"
