"""HVE: Autonomous Physical Agent Integration (docs/AGENT_PHYSICAL_INTEGRATION.md).

Happy / Validate / Error over the agent-facing physical surface added in Phase 12:
lifecycle statuses (Environment), the new movement/object tools, the skills interface
(run_skill / run_chain), the composition chain, and the ManipulationController reach
fix. Machine-independent: fresh worlds per test, generous measured margins.
"""
import math

import numpy as np
import pytest

from playground.config import load_config
from playground.interface import Environment
from playground.interface.schema import SPECS
from playground.manipulation import ManipulationController
from playground.memory import MemoryStore
from playground.storage import Database
from playground.world.physics import PhysicsWorld

NEW_MOVEMENT_TOOLS = ["run_to", "turn_to", "stop_moving", "stand", "stand_up", "approach_object"]
NEW_OBJECT_TOOLS = ["reach_for", "carry_to"]
NEW_SKILL_TOOLS = ["run_skill", "run_chain"]


# ------------------------------------------------------------------ fixtures

def _humanoid_cfg(tmp_path):
    cfg = load_config(overrides={"_root": str(tmp_path), "session": {"max_steps": 800}})
    cfg["agent_body"]["type"] = "humanoid"
    return cfg


@pytest.fixture
def hworld(tmp_path):
    cfg = _humanoid_cfg(tmp_path)
    w = PhysicsWorld(cfg)
    w.settle(max_time=1.0)
    return w


@pytest.fixture
def henv(tmp_path):
    cfg = _humanoid_cfg(tmp_path)
    db = Database(tmp_path / "test.db")
    counter = {"n": 0}
    mem = MemoryStore(tmp_path / "agents", "agent_t", cfg, db=db)
    counter["n"] += 1
    sid = db.create_session(agent_id="agent_t", mode="exploration", session_number=1, adapter="test")
    env = Environment(cfg, db, mem, "agent_t", sid, 1)
    yield env
    env.world.settle(max_time=0.0)
    db.close()


def collapse_to_fall(w: PhysicsWorld) -> bool:
    """Fold the legs through the actuator pathway (the rig's physical fall mode).

    A brief stand phase first registers the posture-hold PD targets (ankles,
    feet, waist) in the world's persistent joint-target table; without them the
    fold crumples past the sit pose into a wedged floor collapse that recovery
    cannot climb out of (see test_locomotion's proven sequence). Returns whether
    a fall was MEASURED during the fold: the thrash makes the sustained flag
    flicker, so it is checked per step, never at the end alone.
    """
    w.set_gait("stand")
    for _ in range(int(1.5 / w.dt)):
        w._step()
    w._gait_controller = None
    fallen_seen = False
    for _ in range(int(4.0 / w.dt)):
        for side in ("left", "right"):
            w.set_joint_target(f"{side}_knee", -1.1, kp=150.0, kd=6.0)
            w.set_joint_target(f"{side}_hip", 0.9, kp=150.0, kd=6.0)
        w._step()
        if w.agent_fallen():
            fallen_seen = True
    return fallen_seen


def place_ahead(w: PhysicsWorld, oid: str, ahead: float, side: float = 0.0):
    ap = w.agent_pos()
    # The rig faces +Y at rest; spawn with soles' clearance under it.
    z = max(0.06, w.specs[oid].dimensions.get("z", 0.2) * 0.5)
    w.spawn_object(oid, [ap[0] + side, ap[1] + ahead, z * 0.5 + 0.01])
    for _ in range(int(0.6 / w.dt)):
        w._step()
    return w.obj_pos(oid)


def shoulder_qpos(w: PhysicsWorld) -> float:
    j = w.m.joint("left_shoulder")
    return float(w.d.qpos[int(j.qposadr[0])])


# ------------------------------------------------------------------ Happy

def test_agent_state_reports_fallen(hworld):
    body = hworld.agent_state()
    assert body["fallen"] is False
    assert collapse_to_fall(hworld) is True
    assert hworld.agent_state()["fallen"] is True


def test_reach_tracks_requested_target(hworld):
    mc = ManipulationController(hworld)
    oid = "block_01"
    ap = hworld.agent_pos()
    # In-plane target 0.95 m ahead, ground-floored, clear of the straight-arm
    # sweep (the aim's tip lands ~0.06 m short of the block's near face).
    dims = hworld.specs[oid].dimensions.get("z", 0.5)
    hworld.spawn_object(oid, [ap[0], ap[1] + 0.95, dims * 0.5 + 0.01])
    for _ in range(int(0.5 / hworld.dt)):
        hworld._step()
    res = mc.reach(oid)
    assert res["status"] == "reached"
    for _ in range(int(1.0 / hworld.dt)):
        hworld._step()
    req = float(res["requested_shoulder_rad"])
    got = shoulder_qpos(hworld)
    # The arm must actually TRACK the requested angle (radians), never slam a limit.
    assert abs(got - req) < 0.15, f"shoulder {got:.3f} rad did not track request {req:.3f}"
    assert abs(got) < 1.2, "shoulder driven to an extreme"
    # The measured hand lies along the shoulder->target ray (sagittal projection).
    hand = np.asarray(mc._hand_pos(), dtype=float)
    shoulder = np.asarray(mc._shoulder_pos(), dtype=float)
    tgt = hworld.obj_pos(oid)
    hand_dir = hand - shoulder
    tgt_dir = tgt - shoulder
    cos = float(np.dot(hand_dir, tgt_dir) / (np.linalg.norm(hand_dir) * np.linalg.norm(tgt_dir)))
    assert cos > 0.95, f"hand direction off target (cos={cos:.3f})"


def test_reach_for_tool_reports_measurement(henv):
    env = henv
    ap = env.world.agent_pos()
    dims = env.world.specs["block_01"].dimensions.get("z", 0.5)
    env.world.spawn_object("block_01", [ap[0], ap[1] + 0.8, dims * 0.5 + 0.01])
    for _ in range(int(0.5 / env.world.dt)):
        env.world._step()
    res = env.call("reach_for", {"object_id": "block_01"})
    assert res["success"] is True
    assert res["status"] == "completed"
    # The reported distance is the measured shortfall of the aim: a straight arm
    # cannot lay a fingertip on a ground-level object ~1 m away — the payload
    # tells the agent exactly how far off the hand still is.
    assert 0.0 < res["hand_to_object"] < 1.0, f"hand_to_object={res['hand_to_object']}"


def test_turn_to_faces_and_reports(henv):
    env = henv
    start = env.world.agent_pos().copy()
    res = env.call("turn_to", {"heading": 180.0})
    assert res["success"] is True
    assert res["status"] == "completed"
    assert res["facing_degrees"] == 180.0
    drift = float(np.hypot(*(env.world.agent_pos()[:2] - start[:2])))
    assert drift < 0.6, f"turn walked {drift:.3f} m (should stay near-footprint)"
    assert "waist_yaw_deg" in res


def test_run_to_covers_distance(henv):
    env = henv
    ap = env.world.agent_pos()
    res = env.call("run_to", {"x": float(ap[0] + 1.5), "y": float(ap[1]), "speed": 1.5})
    assert res["success"] is True
    assert res["status"] == "completed"
    assert res["distance_travelled"] >= 0.3, f"only {res['distance_travelled']:.3f} m"


def test_stand_and_hold_via_env(henv):
    env = henv
    res = env.call("stand", {})
    assert res["success"] is True and res["status"] == "completed"
    env.call("wait", {"seconds": 3.0})
    z = env.world.agent_pos()[2]
    assert z > 0.55 * env.world.hh, f"collapsed to z={z:.3f}"


def test_stop_moving_brakes(henv):
    env = henv
    env.world.move(np.array([1.0, 0.0]), 0.4, speed=1.0, stop=False)
    res = env.call("stop_moving", {})
    assert res["success"] is True
    assert res["status"] == "completed"
    assert res["settled"] is True
    v = env.world.agent_vel()
    assert float(np.hypot(v[0], v[1])) < 0.15, f"still moving at {v[:2]}"


def test_stand_up_recovers_after_fall(henv):
    env = henv
    assert collapse_to_fall(env.world) is True
    res = env.call("stand_up", {})
    assert res["success"] is True
    assert res["status"] == "completed"
    assert res["recovered"] is True
    assert res["recovery_event"] == "recovery_complete"
    assert env.world.agent_fallen() is False


def test_approach_pickup_carry_chain(henv):
    env = henv
    ap = env.world.agent_pos()
    place_ahead(env.world, "block_01", 1.6)
    res = env.call("approach_object", {"object_id": "block_01"})
    assert res["success"] is True and res["status"] == "completed"
    # The humanoid walk drifts off the straight line by waist-twist vs foot
    # friction (documented rig behaviour); the approach still ends near the object.
    assert res["distance_remaining"] <= 1.2
    got = env.call("pick_up", {"object_id": "block_01"})
    assert got["success"] is True and env.world.held == "block_01"
    carry = env.call("carry_to", {"object_id": "block_01", "x": float(ap[0] + 0.4), "y": float(ap[1] - 0.6)})
    assert carry["success"] is True and carry["status"] == "completed"
    assert len(carry["ledger"]) >= 2
    final = env.world.obj_pos("block_01")
    miss = float(np.hypot(final[0] - (ap[0] + 0.4), final[1] - (ap[1] - 0.6)))
    # The humanoid walk drifts off the straight line (waist-twist vs planted-foot
    # friction, BUGLOG 2026-11-17), and the drop lands at the carry pin beside
    # the agent. The measured distance_to_target reports the honest shortfall.
    assert miss < 2.2, f"object left {miss:.3f} m from the drop point"
    assert carry["distance_to_target"] == pytest.approx(miss, abs=0.02)


def test_run_skill_walking(henv):
    env = henv
    res = env.call("run_skill", {"skill": "walking",
                                 "params": {"direction": "north", "distance": 0.5}})
    assert res["success"] is True
    assert res["status"] == "completed"
    assert res["skill_status"] == "succeeded"
    assert res["distance_travelled"] >= 0.2


def test_run_chain_pickup_then_carry(henv):
    """The brief's composition example: approach/reach -> grip -> carry."""
    env = henv
    ap = env.world.agent_pos()
    dims = env.world.specs["block_01"].dimensions.get("z", 0.5)
    env.world.spawn_object("block_01", [ap[0], ap[1] + 0.55, dims * 0.5 + 0.01])
    for _ in range(int(0.6 / env.world.dt)):
        env.world._step()
    start = env.world.obj_pos("block_01").copy()
    res = env.call("run_chain", {"steps": [
        {"skill": "grasping", "args": {"object_id": "block_01"}},
        {"skill": "carrying", "args": {"direction": "west", "distance": 0.4}},
    ]})
    assert res["success"] is True
    assert res["status"] == "completed"
    assert [e["status"] for e in res["ledger"]] == ["succeeded", "succeeded"]
    assert res["completed_steps"] == 2
    assert env.world.held == "block_01"
    moved = float(np.linalg.norm(env.world.obj_pos("block_01") - start))
    assert moved > 0.2, f"carried block moved only {moved:.3f} m"


# ------------------------------------------------------------------ Validate

def test_run_skill_unknown_skill(henv):
    res = henv.call("run_skill", {"skill": "teleport", "params": {}})
    assert res["success"] is False
    assert res["error"]["code"] == "unknown_skill"
    assert res["status"] == "failed"


def test_reach_for_out_of_reach_fails(henv):
    env = henv
    ap = env.world.agent_pos()
    env.world.spawn_object("block_01", [ap[0] + 5.0, ap[1] + 5.0, 0.1])
    for _ in range(int(0.4 / env.world.dt)):
        env.world._step()
    res = env.call("reach_for", {"object_id": "block_01"})
    assert res["success"] is False
    assert res["error"]["code"] == "out_of_reach"
    assert res["status"] == "failed"


def test_turn_to_nonnumeric_heading_fails(henv):
    res = henv.call("turn_to", {"heading": "north"})
    assert res["success"] is False
    assert res["error"]["code"] == "invalid_argument"
    assert res["status"] == "failed"


def test_run_chain_empty_rejected(henv):
    res = henv.call("run_chain", {"steps": []})
    assert res["success"] is False
    assert res["error"]["code"] == "invalid_argument"


def test_approach_unknown_object(henv):
    res = henv.call("approach_object", {"object_id": "no_such_thing"})
    assert res["success"] is False
    assert res["error"]["code"] == "unknown_object"
    assert res["status"] == "failed"


# ------------------------------------------------------------------ Error

def test_lifecycle_classifier():
    """Layer 5: the vocabulary is mapped from measured evidence, not from return codes."""
    from playground.interface.environment import classify_lifecycle
    assert classify_lifecycle("blocked", {"distance_travelled": 1.18}) == "partial"
    assert classify_lifecycle("unknown_object", {}) == "failed"
    assert classify_lifecycle("invalid_argument", {"required": ["x"]}) == "failed"
    assert classify_lifecycle("chain_failed", {"completed_steps": 2}) == "partial"
    assert classify_lifecycle("recovery_failed", {}) == "failed"


def test_run_chain_aborts_and_marks_rest_interrupted(henv):
    env = henv
    ap = env.world.agent_pos()
    dims = env.world.specs["block_01"].dimensions.get("z", 0.5)
    # Grip target within pick-up reach, clear of the body.
    env.world.spawn_object("block_01", [ap[0], ap[1] + 0.55, dims * 0.5 + 0.01])
    # A resting object for the failing step: catch rejects resting objects
    # outright (not_in_flight) — a deterministic, physical failure. (A walk
    # into loose blocks is NOT deterministic: the 300 N walk force drags a
    # row of 5 kg blocks aside, so a "wall" shove can complete honestly.)
    env.world.spawn_object("block_02", [ap[0] + 4.0, ap[1] + 4.0, 0.26])
    for _ in range(int(0.5 / env.world.dt)):
        env.world._step()
    steps = [
        {"skill": "grasping", "args": {"object_id": "block_01"}},
        {"skill": "catching", "args": {"object_id": "block_02", "max_wait": 1.0}},
        {"skill": "jumping", "args": {}},
    ]
    res = env.call("run_chain", {"steps": steps})
    assert res["success"] is False
    assert res["error"]["code"] == "chain_failed"
    assert res["status"] == "partial"
    ledger = res["details"]["ledger"]
    assert [e["status"] for e in ledger] == ["succeeded", "failed", "interrupted"]
    assert ledger[2]["skill"] == "jumping"
    assert env.world.held == "block_01"


# ------------------------------------------------------------------ schema / free-will

@pytest.mark.parametrize("tool", NEW_MOVEMENT_TOOLS + NEW_OBJECT_TOOLS + NEW_SKILL_TOOLS)
def test_new_tools_registered_with_honest_descriptions(tool):
    spec = SPECS.get(tool)
    assert spec is not None, f"{tool} missing from the tool catalogue"
    d = spec["description"]
    lowered = d.lower()
    for banned in ("you should", "you must", "you have to", "never do", "always do",
                   "you are supposed", "goal is", "objective is"):
        assert banned not in lowered, f"{tool}: directive phrasing {banned!r} in description"
    assert d.strip() and "?" not in d, f"{tool}: description must be factual prose"


def test_new_tools_are_physical(hworld):
    from playground.interface.physical import PHYSICAL_TOOL_NAMES
    for tool in NEW_MOVEMENT_TOOLS + NEW_OBJECT_TOOLS:
        assert tool in PHYSICAL_TOOL_NAMES, f"{tool} not dispatched as a physical tool"
