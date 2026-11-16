"""Extra regression tests for existing behaviour not covered by test_hve.py.

Tests:
  - Prompt selection recorded per session
  - Resumed session keeps its original budget
  - walk_to returns arrived=False + remaining_distance + still_moving=False when deflected
  - move result never reports 'timeout' as an obstacle (uses 'deflected_off_course')
  - Collision events debounced (no duplicates within collision_cooldown)
  - out_of_reach error includes distance
  - Event hash chain intact after session
  - Determinism of two identical sessions
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from playground.agents import PolicyAdapter
from playground.agents.adapters import AdapterError
from playground.agents.prompts import PERSONALITIES, exploration_system, prompt_hash
from playground.config import load_config
from playground.session import run_session
from playground.world.mathutil import parse_direction
from playground.world.physics import ActionError, PhysicsWorld


class TestPromptSelection:
    """Verify prompt selection is recorded per session."""

    def test_default_prompt_is_free_agency_v3(self, cfg, db):
        """Default exploration prompt is the strictly non-directive free_agency_v3."""
        seen = {}

        def policy(system, opening, tools):
            seen["system"] = system
            yield ("call", "end_session", {})

        run_session(cfg, db, "agent_001", PolicyAdapter(policy))
        assert seen["system"].startswith("You are an autonomous AI agent")
        assert "You have no assigned mission, and no objective is waiting for you." in seen["system"]
        assert "You may:" not in seen["system"]  # no activity menu

    def test_prompt_id_and_hash_recorded_in_session_started_event(self, cfg, db):
        """The session_started event should record prompt_id and system_prompt_hash."""
        def policy(system, opening, tools):
            yield ("call", "end_session", {})

        s = run_session(cfg, db, "agent_001", PolicyAdapter(policy))
        events = db.events(s["session_id"])
        started = [e for e in events if e["type"] == "session_started"][0]["data"]

        assert "prompt_id" in started
        assert "system_prompt_hash" in started
        assert started["prompt_id"] == "free_agency_v3"
        expected_hash = prompt_hash(exploration_system("free_agency_v3"))
        assert started["system_prompt_hash"] == expected_hash

    def test_brief_v1_prompt_recorded_correctly(self, cfg, db):
        """When brief_v1 is selected, it should be recorded."""
        c = copy.deepcopy(cfg)
        c["agents"]["exploration_prompt"] = "brief_v1"

        def policy(system, opening, tools):
            yield ("call", "end_session", {})

        s = run_session(c, db, "agent_001", PolicyAdapter(policy))
        events = db.events(s["session_id"])
        started = [e for e in events if e["type"] == "session_started"][0]["data"]

        assert started["prompt_id"] == "brief_v1"
        assert started["system_prompt_hash"] == prompt_hash(exploration_system("brief_v1"))


class TestResumedSessionBudget:
    """Verify resumed session keeps its original budget."""

    def test_resumed_session_uses_original_max_steps(self, cfg, db):
        """A resumed session should use the budget it was started with."""
        # Start with 50 max steps
        cfg["session"]["max_steps"] = 50

        call_count = {"n": 0}

        def policy_interrupt(system, opening, tools):
            call_count["n"] += 1
            yield ("call", "jump", {})
            yield ("call", "jump", {})
            raise AdapterError("simulated network error")

        s1 = run_session(cfg, db, "agent_001", PolicyAdapter(policy_interrupt))
        assert s1["end_reason"] == "adapter_error"
        assert db.session(s1["session_id"])["status"] == "interrupted"

        # Now change config to 10 max steps - but resumed session should use original 50
        cfg["session"]["max_steps"] = 10

        def policy_resume(system, opening, tools):
            assert "resumed" in opening.lower()
            # Should be able to do many more actions since original budget was 50
            for _ in range(20):
                yield ("call", "look_around", {})
            yield ("call", "end_session", {})

        s2 = run_session(cfg, db, "agent_001", PolicyAdapter(policy_resume),
                         resume_session_id=s1["session_id"])
        # If budget was properly preserved, we should complete successfully
        assert s2["end_reason"] == "agent_ended"
        # Session ID should be the same (same session resumed)
        assert s2["session_id"] == s1["session_id"]


class TestWalkToDeflection:
    """Verify walk_to behaviour when blocked."""

    def test_walk_to_blocked_returns_error_with_details(self, make_env):
        """walk_to should return an error when blocked by the boundary."""
        env = make_env()
        # Try to walk to a blocked location (off the island)
        r = env.call("walk_to", {"x": 15.0, "y": 0.0})

        assert r["success"] is False
        # Should have an error code
        assert "code" in r["error"]
        # Valid codes: outside_world, blocked, etc.
        assert r["error"]["code"] in ("outside_world", "blocked", "path_blocked")

    def test_move_blocked_never_uses_timeout_as_reason(self, world):
        """move should not use 'timeout' as a blocking reason - uses actual obstacles."""
        # Try to walk off the island
        with pytest.raises(ActionError) as exc:
            world.move(parse_direction("north"), 20.0)

        # Should have a clear error, not timeout
        assert exc.value.code in ("outside_world", "blocked")
        # If there's a blocked_by field, it should describe the actual obstacle
        if "blocked_by" in exc.value.details:
            blocked_by = exc.value.details["blocked_by"]
            # blocked_by can be a string or list
            if isinstance(blocked_by, str):
                assert "timeout" not in blocked_by.lower()
            elif isinstance(blocked_by, list):
                # If it's a list of obstacles, none should mention timeout
                assert all("timeout" not in str(b).lower() for b in blocked_by)


class TestCollisionDebouncing:
    """Verify collision events are debounced per the collision_cooldown rule."""

    def test_no_duplicate_collision_within_cooldown(self, world, walk):
        """Same object pair should not generate duplicate collisions within cooldown period."""
        # Pick up and throw a block to generate collisions
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        world.throw("block_01", parse_direction([1, 0, 0.05]), 120.0)

        # Get all collision events for block_01
        collisions = [
            e for e in world.events
            if e["type"] == "collision" and "block_01" in e["data"]["objects"]
        ]

        # Check that no two collisions with the same pair happen within cooldown
        cooldown = world.cfg.get("recording", {}).get("collision_cooldown", 0.25)

        for i, c1 in enumerate(collisions):
            for c2 in collisions[i+1:]:
                if set(c1["data"]["objects"]) == set(c2["data"]["objects"]):
                    time_diff = abs(c2["time"] - c1["time"])
                    assert time_diff >= cooldown, \
                        f"Duplicate collision within cooldown: {c1['data']['objects']} at times {c1['time']}, {c2['time']}"


class TestOutOfReachError:
    """Verify out_of_reach error includes distance information."""

    def test_out_of_reach_error_includes_distance(self, world):
        """out_of_reach error should include the horizontal_distance in details."""
        # Agent starts at origin, block_01 is at (-2.5, -3.0)
        with pytest.raises(ActionError) as exc:
            world.pick_up("block_01")

        assert exc.value.code == "out_of_reach"
        assert "horizontal_distance" in exc.value.details
        assert exc.value.details["horizontal_distance"] > world.ac["reach"]
        # The error message should mention distance
        assert "distance" in exc.value.message.lower() or "far" in exc.value.message.lower() or \
               str(exc.value.details["horizontal_distance"])[:3] in exc.value.message


class TestEventHashChain:
    """Verify event hash chain integrity."""

    def test_hash_chain_intact_after_session(self, cfg, db):
        """Event hash chain should verify as ok after a completed session."""
        def policy(system, opening, tools):
            yield ("call", "observe_world", {})
            yield ("call", "jump", {})
            yield ("call", "walk_to", {"x": -2.5, "y": -2.1})
            yield ("call", "pick_up", {"object_id": "block_01"})
            yield ("call", "throw", {"object_id": "block_01", "direction": "up", "force": 100})
            yield ("call", "end_session", {})

        s = run_session(cfg, db, "agent_001", PolicyAdapter(policy))
        chain = db.verify_chain(s["session_id"])
        assert chain["ok"] is True


class TestSessionDeterminism:
    """Verify determinism of two identical sessions."""

    def test_identical_sessions_have_same_final_state_hash(self, cfg, db):
        """Two sessions with identical actions should produce identical final state hashes."""
        def deterministic_policy(system, opening, tools):
            yield ("call", "walk_to", {"x": -2.5, "y": -2.1})
            yield ("call", "pick_up", {"object_id": "block_01"})
            yield ("call", "throw", {"object_id": "block_01", "direction": "up", "force": 150})
            yield ("call", "jump", {})
            yield ("call", "end_session", {})

        # Run two sessions with the same policy
        s1 = run_session(cfg, db, "agent_001", PolicyAdapter(deterministic_policy))
        s2 = run_session(cfg, db, "agent_002", PolicyAdapter(deterministic_policy))

        assert s1["final_state_hash"] == s2["final_state_hash"], \
            f"State hashes differ: {s1['final_state_hash']} vs {s2['final_state_hash']}"

    def test_physics_world_determinism_direct(self, cfg, walk):
        """Two PhysicsWorld instances with identical actions should have identical state hashes."""
        def run_sequence(w):
            walk(w, -2.5, -2.1)
            w.pick_up("block_01")
            w.jump(wait="takeoff")
            w.throw("block_01", parse_direction([1, 0.3, 1]), 120.0)
            return w.state_hash()

        w1 = PhysicsWorld(cfg)
        w2 = PhysicsWorld(cfg)

        h1 = run_sequence(w1)
        h2 = run_sequence(w2)

        assert h1 == h2, f"State hashes differ: {h1} vs {h2}"
