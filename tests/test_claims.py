"""Tests for claim-parameter strictness and the tool-facing error path.

Every malformed-parameter case goes through env.call(...) (the same path a live
agent takes), never through the verifier functions in isolation: a malformed
claim must surface as `invalid_argument` with the expected parameter names,
and must NOT be recorded as an ordinary verified-false claim.
"""
from __future__ import annotations

import inspect

import pytest

from playground.interface.schema import SPECS
from playground.integrity import CLAIM_HELP, VERIFIERS
from playground.integrity.claims import verify_claim


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
        class W:
            active: set = set()
        with pytest.raises(TypeError):
            verify_claim("object_on", {"object_id": "x", "target_id": "y"}, W(), [])
        with pytest.raises(TypeError):
            verify_claim("object_on", {}, W(), [])
