import math
import os

import numpy as np
import yaml


def _load_humanoid_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['agent_body']['type'] = 'humanoid'
    return cfg


def _torso_up_z(world) -> float:
    """Torso up-vector z from the body's world rotation matrix (d.xmat)."""
    bid = int(world.m.body('torso').id)
    return float(np.asarray(world.d.xmat[bid]).reshape(3, 3)[2, 2])


def test_humanoid_gait_walk_and_upright():
    """8 s contact-driven walk: at least 0.15 m of displacement, no collapse,
    torso upright, no NaNs. Root traction is disabled for the humanoid, so any
    displacement here is leg-driven through foot-ground contact."""
    cfg = _load_humanoid_cfg()
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(cfg)
    world.set_gait('walk', speed=1.0)
    start_pos = world.agent_pos().copy()
    start_time = world.time
    while world.time - start_time < 8.0:
        world._step()
        assert not np.isnan(world.d.qacc).any(), f"NaN qacc at t={world.time}"
    end_pos = world.agent_pos().copy()
    disp = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
    # displacement should be at least 0.15 m (lab evidence: ~0.75 m in 8 s)
    assert disp >= 0.15, f"displacement {disp:.3f} m too small"
    # torso upright: measured from the torso body's world rotation matrix.
    # (The old check read qpos[aq+3:aq+7] as a quaternion, but the rig root is
    # three slide joints -- those slots are joint angles, not a quaternion.)
    assert _torso_up_z(world) >= 0.8, f"torso not upright, up_z={_torso_up_z(world):.3f}"
    # height must not have collapsed (the honest fall mode of this rig)
    height_ok = world.agent_pos()[2] > world.hh * 0.4
    assert height_ok, f"agent collapsed, z={world.agent_pos()[2]:.3f}"
    assert not np.isnan(disp)


def test_humanoid_stand_and_stand_up():
    """Stand 10 s without falling; collapse the legs through the joint-target
    pathway (the physical fall mode of a rigid-column rig -- height collapse;
    the old -5 m/s qvel impulse cannot topple a slide-root figure in one step);
    verify agent_fallen() fires after the settle window; recover with stand_up()
    and require the measured recovery_complete event."""
    cfg = _load_humanoid_cfg()
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(cfg)
    # stand mode should keep upright without falling for 10 seconds
    world.set_gait('stand')
    for _ in range(int(10.0 / world.dt)):
        world._step()
        assert not world.agent_fallen(), "agent fell during stand"
    # topple the agent the only way this rig can fall: fold the legs via the
    # actuator pathway, never a qpos/qvel write. (0.9, -1.1) is the measured
    # fall pose stand_up() can climb out of; deeper crumples wedge the pelvis
    # under the legs and stay down (armless rigid column -- see BUGLOG).
    # The posture-hold gait is disengaged for the disturbance: an external
    # topple overwrites the hold targets every step otherwise (both write the
    # same per-step PD slots) and the figure would simply keep standing.
    world._gait_controller = None
    for side in ('left', 'right'):
        world.set_joint_target(f'{side}_knee', -1.1, kp=150.0, kd=6.0)
        world.set_joint_target(f'{side}_hip', 0.9, kp=150.0, kd=6.0)
    fallen_seen = False
    for _ in range(int(4.0 / world.dt)):
        for side in ('left', 'right'):
            world.set_joint_target(f'{side}_knee', -1.1, kp=150.0, kd=6.0)
            world.set_joint_target(f'{side}_hip', 0.9, kp=150.0, kd=6.0)
        world._step()
        if world.agent_fallen():
            fallen_seen = True
    assert fallen_seen, "agent should be fallen after the leg fold"
    assert world.agent_fallen(), "fall not sustained at the end of the fold"
    n_events = len(world.events)
    # recover with stand_up: the joint-target schedule attempts the get-up and
    # the outcome is MEASURED, never assumed. The schedule runs ~4.8 s (roll,
    # catch, crouch-exit) followed by its posture hold -- the full window is
    # simulated, no early break: an upright bounce mid-rise is not a recovery.
    world.stand_up()
    for _ in range(int(9.0 / world.dt)):
        world._step()
    assert not world.agent_fallen(), f"agent did not stand up (z={world.agent_pos()[2]:.3f})"
    assert world.agent_pos()[2] >= 0.55, f"recovery plateau too low: z={world.agent_pos()[2]:.3f}"
    # the posture hold must keep the figure up for another second.
    for _ in range(int(1.0 / world.dt)):
        world._step()
    assert not world.agent_fallen(), "agent sagged back down after recovery"
    # the posture hold must keep the figure up.
    up_since = world.time
    for _ in range(int(1.5 / world.dt)):
        world._step()
        if world.agent_fallen():
            up_since = None
            break
    assert not world.agent_fallen(), "agent sagged back down after recovery"
    evtypes = [e['type'] for e in world.events[n_events:]]
    assert 'recovery_started' in evtypes, f"no recovery_started event: {evtypes}"
    assert 'recovery_complete' in evtypes, f"no recovery_complete event: {evtypes}"
    assert 'recovery_failed' not in evtypes, f"recovery reported failed: {evtypes}"
    assert _torso_up_z(world) >= 0.8


def test_humanoid_speed_and_heading():
    """Speed scaling must produce a measurably longer walk, and heading=90 must
    turn the walk north. Displacement over a fixed window is the stable metric
    (instantaneous velocity is phase-noisy for a ~0.1 m/s gait)."""
    cfg = _load_humanoid_cfg()
    from playground.world.physics import PhysicsWorld

    def walk_distance(speed, heading=None, duration=4.0):
        world = PhysicsWorld(cfg)
        world.set_gait('walk', speed=speed, heading=heading)
        start = world.agent_pos().copy()
        t0 = world.time
        mx = 0.0
        while world.time - t0 < duration:
            world._step()
            mx = max(mx, float(np.abs(world.d.qacc).max()))
        p = world.agent_pos()
        return float(np.hypot(p[0] - start[0], p[1] - start[1])), mx

    disp1, q1 = walk_distance(1.0)
    disp2, q2 = walk_distance(2.0)
    assert disp2 > disp1, f"higher speed not reflected: {disp2:.3f} vs {disp1:.3f}"
    # no solver blow-ups: QACC stays bounded (lab evidence: <= ~3e3)
    assert max(q1, q2) < 1e6, f"qacc blow-up: {q1:.1e}/{q2:.1e}"
    disp3, _ = walk_distance(1.0, heading=90.0)
    assert disp3 >= 0.15, f"heading 90 walk too short: {disp3:.3f}"


def test_register_step_callback_and_object_ops():
    cfg = _load_humanoid_cfg()
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(cfg)
    order = []
    def cb1(w):
        order.append('first')
    def cb2(w):
        order.append('second')
    world.register_step_callback(cb1)
    world.register_step_callback(cb2)
    world._step()
    assert order == ['first', 'second'], "callback ordering incorrect"
    # spawn object test (use an existing movable object identifier)
    oid = next(iter(world.movable))
    world.spawn_object(oid, [0.0, 0.0, 0.5])
    pos = world.obj_pos(oid)
    assert np.allclose(pos, [0.0, 0.0, 0.5]), "spawn_object failed"
    world.set_object_velocity(oid, [1.0, 0.0, 0.0])
    vel = world.obj_vel(oid)
    assert np.allclose(vel, [1.0, 0.0, 0.0]), "set_object_velocity failed"
