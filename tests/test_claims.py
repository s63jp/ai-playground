"""Tests for claim-parameter strictness and the tool-facing error path.

Every malformed-parameter case goes through env.call(...) (the same path a live
agent takes), never through the verifier functions in isolation: a malformed
claim must surface as `invalid_argument` with the expected parameter names,
and must NOT be recorded as an ordinary verified-false claim.

The threaded cases below use the benchmark/test scene API (setup_place /
setup_finish) to build isolated world states, plus stub worlds for states a
legal settlement cannot hold (e.g. the tilted wedged ring of session 87).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from playground.integrity import CLAIM_HELP, VERIFIERS
from playground.integrity.claims import verify_claim
from playground.interface.schema import SPECS
from playground.world.mathutil import euler_deg_to_quat
from playground.world.physics import PhysicsWorld
from playground.world.spec import ObjectSpec


# =========================================================== helpers
def _err(env, tool, args):
    """Call a tool and return its failure result (asserts it failed)."""
    r = env.call(tool, args)
    assert r["success"] is False, f"expected {tool} to fail, got {r}"
    assert "error" in r
    return r


# =========================================================== Phase A: strict parameters
class TestMalformedParameters:
    def test_unknown_type_is_invalid_argument(self, make_env):
        env = make_env()
        r = _err(env, "claim_achievement", {"type": "teleport", "params": {}})
        assert r["error"]["code"] == "invalid_argument"
        assert r["details"]["claim_types"] == CLAIM_HELP

    def test_incorrect_parameter_names_rejected(self, make_env):
        # the exact shape of the session-87 claim: object_id/target_id instead of object/surface
        env = make_env()
        r = _err(env, "claim_achievement", {"type": "object_on",
                                            "params": {"object_id": "ring", "target_id": "spire"}})
        assert r["error"]["code"] == "invalid_argument"
        assert "object_id" in r["error"]["message"]
        assert r["details"]["expected"] == CLAIM_HELP["object_on"]

    def test_unknown_extra_parameter_rejected(self, make_env):
        env = make_env()
        r = _err(env, "claim_achievement", {"type": "block_count",
                                            "params": {"at_least": 13, "also": "x"}})
        assert r["error"]["code"] == "invalid_argument"
        assert "also" in r["error"]["message"]

    def test_missing_required_identifiers_rejected(self, make_env):
        env = make_env()
        for ctype, params, missing in (("object_on", {}, ["object", "surface"]),
                                       ("agent_on", {}, ["surface"]),
                                       ("juggle", {"cycles": 1}, ["objects"]),
                                       ("object_on", {"object": "ring"}, ["surface"])):
            r = _err(env, "claim_achievement", {"type": ctype, "params": params})
            assert r["error"]["code"] == "invalid_argument", (ctype, params)
            for m in missing:
                assert m in r["error"]["message"], (ctype, params, m)

    def test_height_reached_needs_a_threshold(self, make_env):
        env = make_env()
        r = _err(env, "claim_achievement", {"type": "height_reached", "params": {"object": "ring"}})
        assert r["error"]["code"] == "invalid_argument"
        assert "min_height" in r["error"]["message"] and "target" in r["error"]["message"]

    def test_malformed_claim_is_not_recorded_as_a_verification(self, make_env):
        env = make_env()
        before = env.db.records(kind="claim", session_id=env.sid)
        _err(env, "claim_achievement", {"type": "object_on",
                                        "params": {"object_id": "ring", "target_id": "spire"}})
        after = env.db.records(kind="claim", session_id=env.sid)
        assert len(after) == len(before)
        # nothing was swallowed into a normal rejection either: no integrity_flag
        flags = env.db.records(kind="integrity_flag", session_id=env.sid)
        assert all(f["data"].get("kind") != "claim_not_verified" for f in flags)

    def test_optional_params_and_defaults_still_accepted(self, make_env):
        env = make_env()
        r = env.call("claim_achievement", {"type": "block_count", "params": {"at_least": 5}})
        assert r["success"] is True and r["verified"] is True          # 12 blocks exist
        r = env.call("claim_achievement", {"type": "block_count", "params": {}})   # default 13
        assert r["success"] is True and r["verified"] is False and r["measured"]["blocks_in_world"] == 12

    def test_correct_names_reach_the_verifier(self, make_env):
        env = make_env()
        # correct names but false claim: a real rejection naming the agent's words
        r = env.call("claim_achievement", {"type": "object_on",
                                           "params": {"object": "block_01", "surface": "spire"}})
        assert r["success"] is True and r["verified"] is False
        assert "block_01" in r["criteria"] and "spire" in r["criteria"]
        assert "resting_on" in r["measured"] or "in_world" in r["measured"]

    def test_unknown_object_ids_rejected_not_passed(self, make_env):
        env = make_env()
        r = env.call("claim_achievement",
                     {"type": "object_on", "params": {"object": "block_99", "surface": "spire"}})
        assert r["success"] is True and r["verified"] is False
        assert r["measured"].get("in_world") is False
        r = env.call("claim_achievement",
                     {"type": "object_on", "params": {"object": "block_01", "surface": "not_a_thing"}})
        assert r["success"] is True and r["verified"] is False     # block_01 exists but not on that


# =========================================================== Phase A: both entry points
class TestClaimEntryPoints:
    def test_update_goal_malformed_claim_is_rejected(self, make_env):
        env = make_env()
        g = env.call("set_goal", {"title": "ring on spire"})
        r = _err(env, "update_goal", {"goal_id": g["goal_id"], "status": "completed",
                                      "claim": {"type": "object_on",
                                                "params": {"object_id": "ring", "target_id": "spire"}}})
        assert r["error"]["code"] == "invalid_argument"
        # the malformed claim must not complete the goal either
        assert env.memory.get_goal(g["goal_id"])["status"] != "completed"

    def test_update_goal_and_claim_achievement_agree(self, make_env):
        env = make_env()
        g = env.call("set_goal", {"title": "blocks in world"})
        a = env.call("claim_achievement", {"type": "block_count", "params": {"at_least": 5}})
        r = env.call("update_goal", {"goal_id": g["goal_id"], "status": "completed",
                                     "claim": {"type": "block_count", "params": {"at_least": 5}}})
        assert a["verified"] is True
        assert r["success"] is True and r["claim_check"]["verified"] is True
        # and the same wrong-name claim fails in both
        wrong = {"type": "object_on", "params": {"object_id": "ring", "target_id": "spire"}}
        assert _err(env, "claim_achievement", wrong)["error"]["code"] == "invalid_argument"
        assert _err(env, "update_goal", {"goal_id": g["goal_id"], "status": "completed",
                                         "claim": wrong})["error"]["code"] == "invalid_argument"

    def test_all_claim_types_exposed(self):
        assert set(VERIFIERS) == set(CLAIM_HELP)
        assert all(t in SPECS["claim_achievement"]["description"] for t in VERIFIERS)


# =========================================================== claim help stays truthful
class TestClaimHelpMatchesSignatures:
    def test_required_and_optional_params_documented(self):
        """Every parameter name the verifiers accept appears in CLAIM_HELP for that type."""
        for ctype, fn in VERIFIERS.items():
            sig = inspect.signature(fn)
            names = {n for n, p in sig.parameters.items()
                     if n not in ("world", "events") and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            text = " ".join(CLAIM_HELP[ctype].split())
            missing = [n for n in names if n.split("_")[0] not in text and n not in text]
            assert not missing, f"{ctype}: CLAIM_HELP does not mention {missing}"

    def test_required_params_have_no_defaults(self):
        """Identifiers the help calls 'required' are actually required (no default)."""
        for ctype in ("object_on", "agent_on", "juggle"):
            fn = VERIFIERS[ctype]
            for n, p in inspect.signature(fn).parameters.items():
                if n in ("world", "events") or n in ("min_blocks", "count", "tolerance",
                                                     "at_least", "cycles", "min_height",
                                                     "target", "objects", "min_objects", "min_levels"):
                    continue
                assert p.default is inspect.Parameter.empty, f"{ctype}.{n} lost its required status"


# =========================================================== verify_claim itself
class TestVerifyClaimDirect:
    def test_wrong_names_raise_typeerror(self):
        with pytest.raises(TypeError):
            verify_claim("object_on", {"object_id": "x", "target_id": "y"}, object(), [])
        with pytest.raises(TypeError):
            verify_claim("object_on", {}, object(), [])


# =========================================================== Phase C/D: threaded
def _tilted_session87_stub(tilt_deg: float):
    """The engine state recorded at session 87's claim time (invest-s87/snapshot.db,
    action idx 50): ring wedged around the spire at ~26-33 degrees, at rest."""
    ring = ObjectSpec(id="ring", name="The Ring", kind="ring", shape="torus", movable=True,
                      position=(8.0094, 0.0105, 5.6274),
                      rotation=tuple(float(v) for v in euler_deg_to_quat(tilt_deg, 0.0, 0.0)),
                      dimensions={"major_radius": 0.7, "tube_radius": 0.06,
                                  "outer_diameter": 1.52, "inner_diameter": 1.28},
                      mass=2.0, unique=True, properties={"segments": 16})
    spire = ObjectSpec(id="spire", name="THE SPIRE", kind="spire", shape="box", movable=False,
                       position=(8.0, 0.0, 3.0), dimensions={"x": 0.8, "y": 0.8, "z": 6.0}, unique=True)
    terrace = ObjectSpec(id="platform_terrace", name="Spire Terrace", kind="platform", shape="box",
                         movable=False, position=(6.1, 0.0, 1.0),
                         dimensions={"x": 3.0, "y": 3.0, "z": 2.0}, properties={"top_height": 2.0})

    class _D:
        ncon = 0

    class _W:
        def __init__(self_):
            self_.active = {"ring"}
            self_.held = None
            self_.d = _D()
            self_.oidx = {}
            self_.specs = {"ring": ring, "spire": spire, "platform_terrace": terrace}

        def obj_pos(self_, oid):
            return np.array(ring.position)

        def obj_quat(self_, oid):
            return np.array(ring.rotation)

        def obj_speed(self_, oid):
            return 0.002          # as recorded at claim time

        def obj_angvel(self_, oid):
            return np.array([0.0116, 0.0042, 0.0001])

        def support_graph(self_):
            return {}

    return _W()


@pytest.fixture
def pgworld(cfg):
    """A real PhysicsWorld with the hidden ring rule disabled (scene-style tests)."""
    cfg["rules"]["ring_generation"]["enabled"] = False
    return PhysicsWorld(cfg)


def _setup(w: PhysicsWorld, ring_xy: tuple[float, float, float], agent_xy=(0.0, 0.0)):
    w.setup_place("ring", ring_xy)
    w.setup_agent(*agent_xy)
    w.setup_finish()


class TestThreadedVerdicts:
    def test_block_column_through_flat_ring(self, pgworld):
        _setup(pgworld, (-2.5, -3.0, 0.06), (0.7, -2.1))    # the ring-generation pose, rule off
        r = verify_claim("threaded", {"object": "ring", "target": "block_01"}, pgworld, [])
        assert r["verified"] is True
        assert r["measured"]["aperture_occupied"] is True
        assert r["measured"]["target_through_the_plane"] is True

    def test_tilted_wedged_ring_around_spire(self, pgworld):
        _setup(pgworld, (8.0, 0.0, 3.0), (9.2, 0.0))
        r = verify_claim("threaded", {"object": "ring", "target": "spire"}, pgworld, [])
        assert r["verified"] is True
        assert r["measured"]["cross_section_gap"] is not None
        assert r["measured"]["cross_section_gap"] <= r["measured"]["aperture_inner_radius"]
        assert r["measured"]["contacts_with_target"] >= 1     # wedged, not floating

    def test_ring_bracelet_on_rock(self, pgworld):
        _setup(pgworld, (-3.0, 2.0, 0.06), (-1.2, 3.2))
        r = verify_claim("threaded", {"object": "ring", "target": "rock"}, pgworld, [])
        assert r["verified"] is True

    @pytest.mark.parametrize("tilt", [0.0, 26.0, 30.0, 33.0])
    def test_tilted_session87_state_is_threaded_on_spire_only(self, tilt):
        # the recorded claim-time state: threaded on the spire at every measured tilt
        w = _tilted_session87_stub(tilt)
        r = verify_claim("threaded", {"object": "ring", "target": "spire"}, w, [])
        assert r["verified"] is True
        assert r["measured"]["cross_section_gap"] == 0.0
        # ...and object_on remains false for the same state: threading is not support
        on = verify_claim("object_on", {"object": "ring", "surface": "spire"}, w, [])
        assert on["verified"] is False

    def test_resting_on_the_terrace_is_not_threading(self, pgworld):
        _setup(pgworld, (8.0, 0.0, 3.0), (9.2, 0.0))
        r = verify_claim("threaded", {"object": "ring", "target": "platform_terrace"}, pgworld, [])
        assert r["verified"] is False

    def test_ring_beside_the_wall_touching_it(self, pgworld):
        _setup(pgworld, (8.0, 1.16, 0.06), (7.0, 2.4))
        r = verify_claim("threaded", {"object": "ring", "target": "spire"}, pgworld, [])
        assert r["verified"] is False
        assert r["measured"]["contacts_with_target"] >= 1
        assert r["measured"]["aperture_occupied"] is False

    def test_hole_over_empty_terrace(self, pgworld):
        _setup(pgworld, (5.5, 0.0, 2.06), (5.5, 3.5))
        r = verify_claim("threaded", {"object": "ring", "target": "spire"}, pgworld, [])
        assert r["verified"] is False

    def test_ring_on_ground_is_not_threaded_on_the_ground(self, pgworld):
        _setup(pgworld, (-2.5, -3.0, 0.06), (0.7, -2.1))
        r = verify_claim("threaded", {"object": "ring", "target": "ground"}, pgworld, [])
        assert r["verified"] is False

    def test_object_held(self, pgworld):
        _setup(pgworld, (8.0, 0.0, 3.0), (9.2, 0.0))
        pgworld.pick_up("ring")
        r = verify_claim("threaded", {"object": "ring", "target": "spire"}, pgworld, [])
        assert r["verified"] is False and r["measured"]["held"] is True

    def test_target_held(self, pgworld):
        _setup(pgworld, (8.0, 0.0, 3.0), (9.2, 0.0))
        pgworld.setup_agent(-2.0, -2.6)
        pgworld.pick_up("block_01")
        r = verify_claim("threaded", {"object": "ring", "target": "block_01"}, pgworld, [])
        assert r["verified"] is False and r["measured"]["held"] is True

    def test_object_moving(self, pgworld):
        _setup(pgworld, (8.0, 0.0, 3.0), (9.2, 0.0))
        a = pgworld.dadr["ring"]
        pgworld.d.qvel[a:a + 3] = [0.5, 0.3, 0.0]     # engine state: the ring is moving
        r = verify_claim("threaded", {"object": "ring", "target": "spire"}, pgworld, [])
        assert r["verified"] is False
        assert r["measured"]["at_rest"]["object"] is False

    def test_non_torus_object(self, pgworld):
        r = verify_claim("threaded", {"object": "block_01", "target": "spire"}, pgworld, [])
        assert r["verified"] is False and r["measured"]["shape"] == "box"

    def test_unknown_and_latent_targets(self, pgworld):
        r = verify_claim("threaded", {"object": "ring", "target": "nope"}, pgworld, [])
        assert r["verified"] is False and "unknown" in r["criteria"]
        r = verify_claim("threaded", {"object": "ring", "target": "block_13"}, pgworld, [])
        assert r["verified"] is False          # reserve body exists but is not in the world


class TestThreadedToolPath:
    def test_threaded_true_claim_through_the_tool(self, cfg, make_env):
        cfg["rules"]["ring_generation"]["enabled"] = False
        env = make_env()
        env.world.setup_place("ring", [8.0, 0.0, 3.0])
        env.world.setup_agent(9.2, 0.0)
        env.world.setup_finish()
        r = env.call("claim_achievement", {"type": "threaded",
                                           "params": {"object": "ring", "target": "spire"}})
        assert r["success"] is True and r["verified"] is True
        claims = env.db.records(kind="claim", session_id=env.sid)
        assert len(claims) == 1 and claims[0]["data"]["type"] == "threaded"
        flags = env.db.records(kind="integrity_flag", session_id=env.sid)
        assert not flags

    def test_threaded_wrong_names_are_invalid_argument_through_the_tool(self, cfg, make_env):
        cfg["rules"]["ring_generation"]["enabled"] = False
        env = make_env()
        r = env.call("claim_achievement", {"type": "threaded",
                                           "params": {"object_id": "ring", "target_id": "spire"}})
        assert r["success"] is False
        assert r["error"]["code"] == "invalid_argument"
        assert r["details"]["expected"] == CLAIM_HELP["threaded"]
        assert "threaded" in SPECS["claim_achievement"]["description"]

    def test_threaded_claim_via_update_goal(self, cfg, make_env):
        cfg["rules"]["ring_generation"]["enabled"] = False
        env = make_env()
        g = env.call("set_goal", {"title": "thread the ring"})
        wrong = {"type": "threaded", "params": {"id": "ring", "thing": "spire"}}
        r = env.call("update_goal", {"goal_id": g["goal_id"], "status": "completed", "claim": wrong})
        assert r["success"] is False and r["error"]["code"] == "invalid_argument"
        assert env.memory.get_goal(g["goal_id"])["status"] == "active"
