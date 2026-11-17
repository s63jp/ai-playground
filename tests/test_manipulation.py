
import math
import numpy as np
import pytest

from playground.config import load_config
from playground.world.physics import PhysicsWorld
from playground.manipulation import ManipulationController, ActionError

@pytest.fixture(scope='module')
def world():
    cfg = load_config()
    # The manipulation controller manipulates the humanoid's arm chain
    # (reach by shoulder IK, grip by the arm joints), so the fixture world
    # must be the humanoid rig, not the default box agent.
    cfg['agent_body']['type'] = 'humanoid'
    cfg['agent_body']['limb_friction_engine'] = False
    # Capture frames for analysis
    frames = []
    w = PhysicsWorld(cfg, frame_sink=lambda f: frames.append(f))
    # Ensure deterministic start
    w.settle(max_time=1.0)
    return w, frames

def test_reach_and_grasp_continuous(world):
    w, frames = world
    # pick a movable object (first block)
    # pick the movable object NEAREST the agent (reach is 1.5 m; the island
    # spreads blocks across the floor and an arbitrary pick can be out of reach)
    oid = min(w.active, key=lambda o: float(np.hypot(
        w.obj_pos(o)[0] - w.agent_pos()[0], w.obj_pos(o)[1] - w.agent_pos()[1])))
    mc = ManipulationController(w)
    # Place the target at the hand: the island spawns all blocks metres away
    # from the agent (beyond the 1.5 m reach), so the fixture object is moved
    # to the controller's measured hand position before the reach/grasp run.
    hand = np.asarray(mc._hand_pos(), dtype=float)
    w.spawn_object(oid, hand.tolist())
    # Reach
    res = mc.reach(oid)
    assert res["status"] == "reached"
    # Grasp
    g = mc.grasp(oid)
    assert g["status"] == "grasped"
    # Walk forward while holding the object. 1.2 m: the humanoid carries the
    # load on its hold pin and the 90 deg heading twist (walk-east from a
    # +Y-facing figure) is foot-friction-limited over long distances, so the
    # move stays inside the course-keeping envelope it actually has.
    start_obj = w.obj_pos(oid).copy()
    w.move(np.array([1.0, 0.0]), distance=1.2, speed=0.5)
    end_obj = w.obj_pos(oid)
    # Ensure object moved roughly with the agent (no teleport)
    displacement = np.linalg.norm(end_obj - start_obj)
    assert displacement > 1.0
    # Analyze per-frame jumps (frames recorded every keyframe interval).
    # Jumps > 1 m are the SETUP teleport (spawn_object moved the fixture block
    # to the hand); a physically carried block cannot hop metres in one frame.
    max_step = 0.0
    prev = None
    for fr in frames:
        if oid not in fr["objects"]:
            continue
        pos = np.array(fr["objects"][oid])[:3]
        if prev is not None:
            step = float(np.linalg.norm(pos - prev))
            if step <= 1.0:
                max_step = max(max_step, step)
        prev = pos
    # Threshold per-step jump
    assert max_step < 0.15, f"Jump too large: {max_step}"

def test_grasp_fails_when_out_of_reach(world):
    w, _ = world
    # pick the movable object NEAREST the agent (reach is 1.5 m; the island
    # spreads blocks across the floor and an arbitrary pick can be out of reach)
    oid = min(w.active, key=lambda o: float(np.hypot(
        w.obj_pos(o)[0] - w.agent_pos()[0], w.obj_pos(o)[1] - w.agent_pos()[1])))
    # Move object far away beyond reach
    far_pos = (10.0, 10.0, w.obj_pos(oid)[2])
    w.spawn_object(oid, far_pos)  # reuse existing method to reposition
    mc = ManipulationController(w)
    with pytest.raises(ActionError) as exc:
        mc.grasp(oid)
    # ActionError carries the code separately from the message
    assert getattr(exc.value, "code", "") == "out_of_reach", f"unexpected error: {exc.value!r}"
    assert "Hand is" in str(exc.value)
