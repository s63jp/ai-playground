import os
import yaml

def test_humanoid_loaded():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['agent_body']['type'] = 'humanoid'
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(cfg)
    # Check that key joints exist
    assert world.m.joint('left_knee') is not None, "left_knee joint missing"
    assert world.m.joint('head_pitch') is not None, "head_pitch joint missing"
    # Run a few steps to ensure stability
    for _ in range(100):
        world._step()
    # Ensure positions are finite
    assert not any(abs(v) > 1e6 or v != v for v in world.d.qpos.ravel()), "simulation diverged"
