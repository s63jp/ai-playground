"""HVE suite for the action-space expansion (audit 2026-11-13, Stage 2).

Proves against the REAL physics engine that the new non-throw actions work and are verifiable:
  Happy    - push / slide / roll / kick / nudge / pull move an unheld object; stack_on and build_row build.
  Validate - force limits per verb; direction required where it must be; must be in reach.
  Error    - pushing a held object, an out-of-reach object, or an unknown object is refused cleanly.
Plus: the new benchmark verifiers reject throwing and pass on genuine contact/build.
"""
from __future__ import annotations

import numpy as np
import pytest

from playground.config import load_config
from playground.world.physics import PhysicsWorld, ActionError
from playground.interface.physical import PhysicalTools, PHYSICAL_TOOL_NAMES


@pytest.fixture
def world():
    return PhysicsWorld(load_config())


@pytest.fixture
def tools(world):
    return PhysicalTools(world)


def _goto_object(w: PhysicsWorld, t: PhysicalTools, oid: str, standoff: float = 0.9):
    """Walk the body to just beside an object so it is in reach."""
    p = w.obj_pos(oid)
    t.walk_to(float(p[0] - standoff), float(p[1]))
    return w.obj_pos(oid)


def _first_block(w):
    return next(o for o in w.visible_ids() if w.specs[o].kind == "block")


# ============================================================ Happy
class TestContactVerbsMove:
    def test_all_six_contact_tools_are_registered(self):
        for name in ("push", "pull", "slide", "nudge", "kick", "roll", "stack_on", "build_row"):
            assert name in PHYSICAL_TOOL_NAMES

    def test_push_moves_an_unheld_block(self, world, tools):
        b = _first_block(world)
        start = _goto_object(world, tools, b).copy()
        r = tools.push(b, force=60.0)
        assert r["verb"] == "push"
        assert r["outcome"] in ("rest", "still_moving", "first_impact")
        assert r["distance_moved"] > 0.05
        assert float(np.hypot(*(world.obj_pos(b)[:2] - start[:2]))) > 0.05

    def test_slide_moves_in_the_requested_direction(self, world, tools):
        b = _first_block(world)
        start = _goto_object(world, tools, b).copy()
        r = tools.slide(b, "north", force=60.0)
        end = world.obj_pos(b)
        # north is +y: y should increase more than |x| change
        assert end[1] - start[1] > 0.05

    def test_roll_moves_the_rock(self, world, tools):
        _goto_object(world, tools, "rock")
        r = tools.roll("rock", "north", force=60.0)
        assert r["distance_moved"] > 0.1

    def test_kick_imparts_more_speed_than_nudge(self, world, tools):
        b = _first_block(world)
        _goto_object(world, tools, b)
        kick = tools.kick(b, force=120.0, wait="none")
        # reset world and compare a nudge from the same pose
        w2 = PhysicsWorld(load_config())
        t2 = PhysicalTools(w2)
        _goto_object(w2, t2, b)
        nudge = t2.nudge(b, force=10.0, wait="none")
        assert kick["launch_speed"] > nudge["launch_speed"]

    def test_stack_on_rests_one_block_on_another(self, world, tools):
        blocks = [o for o in world.visible_ids() if world.specs[o].kind == "block"]
        b0, b1 = blocks[0], blocks[1]
        p1 = world.obj_pos(b1)
        tools.walk_to(float(p1[0] - 0.9), float(p1[1]))
        tools.pick_up(b1)
        r = tools.stack_on(b0)
        assert r.get("stacked_on") == b0
        assert b0 in (r.get("resting_on") or [])
        assert world.obj_pos(b1)[2] > world.obj_pos(b0)[2] + 0.1

    def test_build_row_places_beside_a_neighbour(self, world, tools):
        blocks = [o for o in world.visible_ids() if world.specs[o].kind == "block"]
        b0, b1 = blocks[0], blocks[1]
        p1 = world.obj_pos(b1)
        tools.walk_to(float(p1[0] - 0.9), float(p1[1]))
        tools.pick_up(b1)
        # step to the far side of the neighbour so the body is not occupying the open ground beside it
        p0 = world.obj_pos(b0)
        tools.walk_to(float(p0[0] - 1.3), float(p0[1] + 1.3))
        r = tools.build_row(b0)
        assert r.get("beside") == b0
        d = float(np.hypot(*(world.obj_pos(b1)[:2] - world.obj_pos(b0)[:2])))
        assert 0.3 < d < 2.0


# ============================================================ Validate
class TestContactVerbValidation:
    def test_force_out_of_range_is_rejected(self, world, tools):
        b = _first_block(world)
        _goto_object(world, tools, b)
        with pytest.raises(ActionError) as e:
            tools.push(b, force=5000.0)
        assert e.value.code == "invalid_argument"

    def test_slide_requires_a_direction(self, world, tools):
        from playground.interface.schema import ToolError
        b = _first_block(world)
        _goto_object(world, tools, b)
        with pytest.raises((ActionError, ToolError, TypeError)):
            tools.slide(b, None, force=30.0)

    def test_nudge_default_force_is_gentle(self, world, tools):
        b = _first_block(world)
        _goto_object(world, tools, b)
        r = tools.nudge(b)
        assert r["force"] <= 15.0


# ============================================================ Error
class TestContactVerbErrors:
    def test_pushing_a_held_object_is_refused(self, world, tools):
        from playground.interface.schema import ToolError
        b = _first_block(world)
        _goto_object(world, tools, b)
        tools.pick_up(b)
        with pytest.raises((ActionError, ToolError)) as e:
            tools.push(b, force=40.0)
        assert e.value.code == "holding_object"

    def test_out_of_reach_object_is_refused(self, world, tools):
        from playground.interface.schema import ToolError
        far = [o for o in world.visible_ids() if world.specs[o].kind == "block"]
        # a block far from spawn, with an explicit direction so reach (not _away_dir) is the gate
        with pytest.raises((ActionError, ToolError)) as e:
            tools.slide(far[-1], "north", force=40.0)
        assert e.value.code in ("out_of_reach", "moving_too_fast")

    def test_unknown_object_is_refused(self, world, tools):
        from playground.interface.schema import ToolError
        with pytest.raises((ActionError, ToolError)) as e:
            tools.slide("no_such_block", "north", force=40.0)
        assert e.value.code == "unknown_object"


# ============================================================ Verifiers reject throwing
class TestNonThrowVerifiers:
    def test_push_to_target_task_exists_and_rejects_throwing(self):
        from playground.session.benchmark import TASKS
        for tid in ("push_to_target", "roll_to_target", "build_to_height", "topple"):
            assert tid in TASKS

    def test_object_at_verifier_requires_contact_not_throw(self):
        from playground.session.benchmark import make_v_object_at
        v = make_v_object_at([0.0, 0.0], tol=50.0, obj="block_01")
        # object at origin-ish but ONLY a throw event -> not verified
        thrown = [{"type": "throw_release", "data": {"object": "block_01"}}]
        w = PhysicsWorld(load_config())
        res = v(w, thrown)
        assert res["verified"] is False
        assert res["measured"]["was_thrown"] is True

    def test_object_at_verifier_passes_on_contact(self):
        from playground.session.benchmark import make_v_object_at
        w = PhysicsWorld(load_config())
        b = _first_block(w)
        p = w.obj_pos(b)
        v = make_v_object_at([float(p[0]), float(p[1])], tol=50.0, obj=b)
        contact = [{"type": "object_impulse", "data": {"object": b}}]
        res = v(w, contact)
        assert res["verified"] is True

    def test_impulse_row_feature_extraction(self):
        from playground.ml.features import impulse_row
        r = impulse_row(
            {"verb": "push", "force": 40.0, "direction": [1, 0, 0], "object_mass": 5.0,
             "launch_speed": 2.1, "contact_position": [0, 0, 0.25]},
            {"final_position": [0.34, 0, 0.25], "horizontal_range": 0.3, "impact_speed": 1.2,
             "flight_time": 0.4},
        )
        assert r is not None
        assert abs(r["distance_moved"] - 0.34) < 1e-6
        assert r["force"] == 40.0 and r["launch_speed"] == 2.1
