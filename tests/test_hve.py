"""HVE suite: Happy path, Validation, Error handling for every claimed capability.

Each section maps to a phase of the brief. Physics assertions compare against analytic
expectations or against the engine's own record, never against hard-coded "nice" numbers.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from playground.agents import PolicyAdapter, make_adapter
from playground.analysis import classify_session
from playground.config import agent_ids, load_config
from playground.integrity import check_flight, verify_claim, verify_experiment
from playground.interface.schema import SPECS, ToolError, normalise_args, openai_tools, anthropic_tools
from playground.memory import MemoryError_, MemoryStore, evaluate_status
from playground.session import ensure_agents, run_agents, run_session
from playground.session.benchmark import TASKS, run_ablation, run_benchmark
from playground.world.mathutil import parse_direction
from playground.world.physics import ActionError, PhysicsWorld

G = 9.81


# =============================================================== Phase 2: world
class TestWorldHappy:
    def test_initial_objects_have_every_required_property(self, world):
        objs = world.describe()["objects"]
        ids = [o["id"] for o in objs]
        assert len(ids) == len(set(ids))
        assert sum(1 for o in objs if o["kind"] == "block") == 12
        assert sum(1 for o in objs if o["kind"] == "rock") == 1
        assert sum(1 for o in objs if o["kind"] == "ring") == 1 and next(o for o in objs if o["kind"] == "ring")["unique"]
        assert sum(1 for o in objs if o["kind"] == "spire") == 1
        assert sum(1 for o in objs if o["kind"] == "platform") >= 3
        assert len({o["top_height"] for o in objs if o["kind"] == "platform"}) >= 3   # different heights
        for o in objs:
            for key in ("id", "name", "position", "rotation_quaternion", "dimensions", "mass", "colour",
                        "physical_properties", "interaction_state"):
                assert key in o, (o["id"], key)

    def test_spire_is_tall_narrow_and_beyond_a_jump_from_its_terrace(self, world):
        sp = world.object_state("spire")
        terr = world.object_state("platform_terrace")
        assert sp["dimensions"]["z"] >= 5 * max(sp["dimensions"]["x"], sp["dimensions"]["y"])
        # terrace is adjacent: its east face touches the spire's west face
        assert terr["extent"]["x"][1] == pytest.approx(sp["extent"]["x"][0], abs=1e-6)
        j = world.jump()
        assert sp["top_height"] - terr["top_height"] > j["apex_above_takeoff"] + world.ac["max_climb"]

    def test_world_starts_at_rest_on_the_ground(self, world):
        assert world.grounded and world.support == ["ground"]
        for oid in world.active:
            assert world.obj_speed(oid) < 1e-3

    def test_boundaries_and_coordinate_system(self, world):
        b = world.boundaries()
        assert b["walkable_area"]["x"] == [-12.0, 12.0]
        assert "x" in b["coordinate_system"]["axes"]


class TestPhysicsHappy:
    def test_jump_matches_impulse_ballistics(self, world):
        j = world.jump()
        m = world.total_mass()
        F, T = world.ac["jump_force"], world.ac["jump_push_time"]
        a = F / m - G
        expected = 0.5 * a * T * T + (a * T) ** 2 / (2 * G)
        assert j["apex_above_takeoff"] == pytest.approx(expected, rel=0.02)
        assert j["landed_on"] == ["ground"]

    def test_carried_mass_reduces_jump_height(self, world, walk):
        free = world.jump()["apex_above_takeoff"]
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        loaded = world.jump()["apex_above_takeoff"]
        assert loaded < free * 0.9

    def test_stationary_throw_obeys_ballistics_and_recorded_flight(self, world, walk):
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        r = world.throw("block_01", parse_direction("up"), 150.0)
        v = math.sqrt(2 * 150.0 * world.ac["throw_stroke"] / 5.0)
        assert r["release_speed"] == pytest.approx(v, rel=1e-3)
        assert r["max_height_above_launch"] == pytest.approx(v * v / (2 * G), rel=0.02)
        assert check_flight({**r, "launch_velocity": r["release_velocity"], "launch_speed": r["release_speed"],
                             "launch_position": r["release_position"]}) == []

    def test_jump_throw_goes_higher_than_standing_throw(self, cfg, walk):
        def peak(jumping):
            w = PhysicsWorld(cfg)
            walk(w, -2.5, -2.1)
            w.pick_up("block_01")
            if jumping:
                w.jump(wait="takeoff")
            return w.throw("block_01", parse_direction("up"), 200.0)
        s, j = peak(False), peak(True)
        assert j["max_height"] > s["max_height"] + 1.5
        # the extra speed is the body's upward velocity at release
        assert j["release_velocity"][2] == pytest.approx(s["release_velocity"][2] + j["agent_velocity_at_release"][2],
                                                        abs=0.05)

    def test_running_throw_carries_further(self, cfg, walk):
        def rng(run):
            w = PhysicsWorld(cfg)
            walk(w, -2.5, -2.1)
            w.pick_up("block_01")
            walk(w, -6, 0)
            if run:
                w.move(parse_direction("east"), 3.0, speed=4.0, stop=False)
            return w.throw("block_01", parse_direction([1, 0, 1]), 150.0)["horizontal_range"]
        assert rng(True) > rng(False) + 1.0

    def test_throw_recoil_conserves_momentum(self, world, walk):
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        v0 = world.agent_vel().copy()
        r = world.throw("block_01", parse_direction("east"), 200.0, wait="none")
        dv_body = world.agent_vel() - v0
        p_obj = 5.0 * np.array(r["velocity_relative_to_body"])
        assert np.allclose(dv_body * 50.0, -p_obj, atol=1e-6)

    def test_stacking_five_blocks_is_stable(self, world, walk):
        base = (-2.5, -3.0)
        for b in ["block_02", "block_03", "block_04"]:
            p = world.obj_pos(b)
            walk(world, p[0], p[1] + 0.9)
            world.pick_up(b)
            walk(world, base[0], base[1] + 0.9)
            r = world.place(b, list(base))
            assert r["stayed_where_placed"], r
        world.settle(3.0)
        heights = [world.obj_pos(b)[2] for b in ["block_01", "block_02", "block_03", "block_04"]]
        assert heights == pytest.approx([0.25, 0.75, 1.25, 1.75], abs=0.01)

    def test_collisions_are_recorded_with_impulse(self, world, walk):
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        world.throw("block_01", parse_direction("east"), 150.0)
        cols = [e for e in world.events if e["type"] == "collision" and "block_01" in e["data"]["objects"]]
        assert cols and cols[0]["data"]["impulse"] > 0 and cols[0]["data"]["relative_speed"] > 0.3

    def test_friction_stops_a_sliding_block(self, world, walk):
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        r = world.throw("block_01", parse_direction([1, 0, 0.05]), 120.0)
        assert r["outcome"] == "rest"
        assert world.obj_speed("block_01") < 0.05

    def test_hidden_ring_rule_generates_a_block(self, world, walk):
        walk(world, 3.0, 2.1)
        world.pick_up("ring")
        walk(world, -2.5, -1.75)
        world.place("ring", [-2.5, -3.0])
        world.wait(2.5)
        assert "block_13" in world.active
        ev = [e for e in world.events if e["type"] == "object_appeared"]
        assert ev[0]["hidden"]["cause"] == "ring_generation"

    def test_determinism_and_exact_checkpoint_restore(self, cfg, walk):
        def run(w):
            walk(w, -2.5, -2.1)
            w.pick_up("block_01")
            w.jump(wait="takeoff")
            w.throw("block_01", parse_direction([1, 0.3, 1]), 120.0, wait="none")
        a, b = PhysicsWorld(cfg), PhysicsWorld(cfg)
        run(a)
        run(b)
        assert a.state_hash() == b.state_hash()
        st = json.loads(json.dumps(a.get_state()))
        c = PhysicsWorld(cfg)
        c.set_state(st)
        a.wait(2.0)
        c.wait(2.0)
        assert a.state_hash() == c.state_hash()


class TestPhysicsValidate:
    def test_cannot_walk_off_the_island(self, world):
        with pytest.raises(ActionError) as e:
            world.move(parse_direction("north"), 20.0)       # nothing between the start and the north edge
        assert e.value.code == "blocked" and "boundary" in e.value.details["blocked_by"]
        assert world.agent_pos()[1] < 12.0

    def test_cannot_climb_the_terrace_or_jump_and_climb_it(self, world, walk):
        walk(world, 4.3, 0.0)
        with pytest.raises(ActionError) as e:
            world.climb(parse_direction("east"))
        assert e.value.code == "ledge_too_high"

    def test_pick_up_out_of_reach(self, world):
        with pytest.raises(ActionError) as e:
            world.pick_up("block_01")
        assert e.value.code == "out_of_reach" and e.value.details["horizontal_distance"] > 1.5

    def test_one_object_at_a_time(self, world, walk):
        walk(world, -2.0, -2.1)
        world.pick_up("block_01")
        with pytest.raises(ActionError) as e:
            world.pick_up("block_02")
        assert e.value.code == "hands_full"

    def test_force_limits(self, world, walk):
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        for bad in (0.0, -5.0, 201.0, float("nan")):
            with pytest.raises(ActionError):
                world.throw("block_01", parse_direction("up"), bad)
        assert world.held == "block_01"

    def test_place_into_occupied_space_is_refused_and_object_kept(self, world, walk):
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        with pytest.raises(ActionError) as e:
            world.place("block_01", [-1.5, -3.0, 0.25])
        assert e.value.code == "space_occupied" and world.held == "block_01"

    def test_cannot_jump_in_the_air(self, world):
        world.jump(wait="apex")
        with pytest.raises(ActionError) as e:
            world.jump()
        assert e.value.code == "not_on_ground"

    def test_fixed_structures_cannot_be_moved(self, world):
        with pytest.raises(ActionError) as e:
            world.pick_up("spire")
        assert e.value.code == "not_movable"


# =============================================================== Phase 3: tool interface
class TestInterfaceHappy:
    def test_every_tool_in_the_brief_exists(self):
        brief = ["observe_world", "inspect_object", "inspect_position", "get_inventory", "get_world_boundaries",
                 "get_object_positions", "move", "jump", "climb", "walk_to", "look_around", "pick_up",
                 "put_down", "throw", "place", "rotate_object", "measure_height", "measure_distance",
                 "measure_velocity", "measure_flight_time", "inspect_collision", "run_physics_test",
                 "inspect_world_history", "inspect_recent_events", "save_program", "execute_program"]
        assert set(brief) <= set(SPECS)
        assert len(openai_tools()) == len(anthropic_tools()) == len(SPECS)

    def test_throw_returns_structured_measurements(self, make_env):
        env = make_env()
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        env.call("pick_up", {"object_id": "block_01"})
        r = env.call("throw", {"object_id": "block_01", "direction": "up", "force": 150})
        assert r["success"] and r["action"] == "throw" and isinstance(r["action_index"], int)
        for k in ("release_position", "final_position", "flight_time", "max_height", "release_speed"):
            assert k in r

    def test_sandbox_test_does_not_change_the_world(self, make_env):
        env = make_env()
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        before = env.world.state_hash()
        r = env.call("run_physics_test", {"parameters": {"steps": [
            {"tool": "pick_up", "args": {"object_id": "block_01"}},
            {"tool": "throw", "args": {"object_id": "block_01", "direction": "up", "force": 150}}]}})
        assert r["success"] and r["results"][1]["max_height"] > 3
        assert env.world.state_hash() == before

    def test_programs_save_run_and_persist(self, make_env):
        env = make_env()
        r = env.call("save_program", {"name": "toss", "steps": [
            {"tool": "pick_up", "args": {"object_id": "${b}"}},
            {"tool": "throw", "args": {"object_id": "${b}", "direction": "up", "force": "${f}"}}]})
        assert r["success"] and r["parameters"] == ["b", "f"]
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        p = env.call("execute_program", {"name": "toss", "params": {"b": "block_01", "f": 100}})
        assert p["success"] and p["steps_failed"] == 0
        assert [a["source"] for a in env.db.actions(env.sid)][-2:] == ["program", "program"]
        assert (env.memory.dir / "programs" / "toss.json").exists()


class TestInterfaceValidate:
    @pytest.mark.parametrize("tool,args,code", [
        ("fly", {}, "unknown_tool"),
        ("move", {"direction": "north"}, "invalid_argument"),
        ("move", {"direction": "sideways", "distance": 1}, "invalid_argument"),
        ("pick_up", {"object_id": "block_99"}, "unknown_object"),
        ("inspect_object", {"object_id": "block_13"}, "unknown_object"),   # latent block is invisible
        ("move", {"direction": "north", "distance": 1, "bogus": 1}, "invalid_argument"),
        ("jump", {"wait": "never"}, "invalid_argument"),
    ])
    def test_bad_calls_fail_loudly(self, make_env, tool, args, code):
        r = make_env().call(tool, args)
        assert r["success"] is False and r["error"]["code"] == code and r["error"]["message"]

    def test_failures_are_logged_as_failures(self, make_env):
        env = make_env()
        env.call("pick_up", {"object_id": "block_01"})
        a = env.db.actions(env.sid)[-1]
        assert a["success"] is False and a["error_code"] == "out_of_reach"

    def test_program_missing_parameter_runs_nothing(self, make_env):
        env = make_env()
        env.call("save_program", {"name": "p", "steps": [{"tool": "jump", "args": {}},
                                                          {"tool": "wait", "args": {"seconds": "${s}"}}]})
        n = len(env.db.actions(env.sid))
        r = env.call("execute_program", {"name": "p"})
        assert r["error"]["code"] == "missing_parameter"
        assert len(env.db.actions(env.sid)) == n + 1        # only the failed call itself

    def test_memory_tools_cannot_go_in_programs(self, make_env):
        r = make_env().call("save_program", {"name": "x", "steps": [{"tool": "record_discovery", "args": {}}]})
        assert r["error"]["code"] == "invalid_argument"

    def test_argument_coercion(self):
        assert normalise_args("place", {"object_id": "a", "position": "[1, 2]"})["position"] == [1, 2]
        with pytest.raises(ToolError):
            normalise_args("wait", {"seconds": "soon"})


class TestInterfaceError:
    def test_internal_crash_is_reported_as_failure(self, make_env, monkeypatch):
        env = make_env()
        monkeypatch.setattr(env.tools, "jump", lambda **k: 1 / 0)
        r = env.call("jump", {})
        assert r["success"] is False and r["error"]["code"] == "internal_error"
        assert env.db.records(kind="integrity_flag")[0]["data"]["kind"] == "internal_error"

    def test_session_closed_rejects_actions(self, make_env):
        env = make_env()
        env.call("end_session", {})
        assert env.call("observe_world")["error"]["code"] == "session_closed"

    def test_step_limit_triggers_wrapup_then_close(self, make_env, cfg):
        cfg["session"]["max_steps"] = 3
        env = make_env()
        for _ in range(3):
            env.call("look_around", {})
        r = env.call("jump", {})
        assert r["error"]["code"] == "session_time_over"
        assert env.call("write_diary", {"text": "still allowed"})["success"]


# =============================================================== Phase 5/7/9: memory + evidence
class TestMemoryHappy:
    def _throw(self, env):
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        env.call("pick_up", {"object_id": "block_01"})
        return env.call("throw", {"object_id": "block_01", "direction": "up", "force": 150})

    def test_directory_structure(self, cfg, db):
        ensure_agents(cfg, db)
        from playground.session import memory_root
        for a in agent_ids(cfg):
            d = memory_root(cfg) / a
            for f in ("diary.md", "handbook.md", "discoveries.json", "experiments.json", "mistakes.md",
                      "unfinished_goals.md"):
                assert (d / f).exists(), (a, f)
            assert (d / "programs").is_dir() and (d / "session_history").is_dir()
        assert len(agent_ids(cfg)) == 13

    def test_evidence_gate_promotes_only_with_two_measured_experiments(self, make_env):
        env = make_env()
        t1 = self._throw(env)
        e1 = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "r",
                                            "measurements": {"max_height": t1["max_height"]},
                                            "action_refs": [t1["action_index"]], "supported": True})
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"]]})
        assert d["granted_status"] == "probable"
        t2 = self._throw(env)
        e2 = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "r",
                                            "measurements": {"max_height": t2["max_height"]},
                                            "action_refs": [t2["action_index"]], "supported": True})
        u = env.call("update_discovery", {"discovery_id": d["discovery_id"], "status": "confirmed",
                                          "add_experiment_ids": [e2["experiment_id"]]})
        assert u["granted_status"] == "confirmed"

    def test_briefing_is_read_from_files(self, cfg, db, tmp_path):
        m = MemoryStore(tmp_path / "agents", "a", cfg, db=db)
        m.add_diary(1, 3.0, "UNIQUE-DIARY-MARKER")
        (m.dir / "handbook.md").write_text("HAND-EDITED HANDBOOK", encoding="utf-8")
        b = m.briefing()
        assert "HAND-EDITED HANDBOOK" in b and "UNIQUE-DIARY-MARKER" in b


class TestMemoryValidate:
    def test_untested_hypothesis_never_becomes_fact(self, make_env):
        r = make_env().call("record_discovery", {"statement": "The ring is magic.", "status": "confirmed"})
        assert r["granted_status"] == "hypothesis" and r["status_granted_as_requested"] is False

    def test_fabricated_measurement_is_contradicted(self, make_env):
        env = make_env()
        t = TestMemoryHappy()._throw(env)
        e = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "32 m",
                                           "measurements": {"max_height": 32.0},
                                           "action_refs": [t["action_index"]], "supported": True})
        assert e["evidence_check"]["verdict"] == "contradicted"
        d = env.call("record_discovery", {"statement": "32 m", "status": "confirmed",
                                          "experiment_ids": [e["experiment_id"]]})
        assert d["granted_status"] == "hypothesis"

    def test_same_evidence_twice_counts_once(self):
        e = {"supported": True, "action_refs": [5], "session": 1,
             "verification": {"verified": True, "level": "measured"}}
        assert evaluate_status("confirmed", [e, dict(e)], {})[0] == "probable"

    def test_disproven_requires_checked_contradiction(self):
        assert evaluate_status("disproven", [], {})[0] == "hypothesis"
        e = {"supported": False, "action_refs": [3], "verification": {"verified": True, "level": "measured"}}
        assert evaluate_status("disproven", [e], {})[0] == "disproven"

    def test_invalid_references(self):
        v = verify_experiment({"measurements": {}}, {999: None})
        assert v["verified"] is False


class TestMemoryAdministration:
    def _confirmed(self, make_env):
        env = make_env()
        ids = []
        for _ in range(2):
            t = TestMemoryHappy()._throw(env)
            e = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "r",
                                               "measurements": {"max_height": t["max_height"]},
                                               "action_refs": [t["action_index"]], "supported": True})
            ids.append(e["experiment_id"])
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed", "experiment_ids": ids})
        assert d["granted_status"] == "confirmed"
        return env, d["discovery_id"], ids

    def test_remove_challenge_unverify_with_audit(self, make_env):
        env, did, eids = self._confirmed(make_env)
        m = env.memory
        m.challenge(did, "suspicious")
        assert m.get_discovery(did)["flags"]["challenged"]
        m.mark_unverified(did, "re-test")
        assert m.get_discovery(did)["status"] == "hypothesis"
        assert "Marked unverified" in (m.dir / "handbook.md").read_text(encoding="utf-8")
        m.remove_item(did, "ablation")
        assert m.get_discovery(did) is None
        ops = [r["op"] for r in env.db.q("SELECT op FROM memory_audit ORDER BY id")]
        assert ops == ["challenge", "mark_unverified", "remove"]
        with pytest.raises(Exception):
            env.db.x("DELETE FROM memory_audit")

    def test_removing_evidence_demotes_discovery(self, make_env):
        env, did, eids = self._confirmed(make_env)
        env.memory.remove_item(eids[0], "test")
        assert env.memory.get_discovery(did)["status"] == "probable"

    def test_ablation_copy_leaves_original_untouched(self, make_env, tmp_path):
        env, did, _ = self._confirmed(make_env)
        copy = env.memory.copy_to(tmp_path / "abl", remove_ids=[did])
        assert copy.get_discovery(did) is None and env.memory.get_discovery(did) is not None

    def test_unknown_item(self, cfg, tmp_path):
        with pytest.raises(MemoryError_):
            MemoryStore(tmp_path / "x", "a", cfg).remove_item("D-404")


# =============================================================== Phase 12: integrity
class TestIntegrity:
    def test_event_log_is_append_only_and_hash_chained(self, make_env):
        env = make_env()
        env.call("jump", {})
        assert env.db.verify_chain(env.sid)["ok"]
        with pytest.raises(Exception, match="immutable"):
            env.db.x("UPDATE events SET data_json='{}' WHERE session_id=?", (env.sid,))
        with pytest.raises(Exception, match="immutable"):
            env.db.x("DELETE FROM actions")

    def test_tampering_is_detected(self, make_env):
        env = make_env()
        env.call("jump", {})
        env.db.x("DROP TRIGGER events_no_update")
        env.db.x("UPDATE events SET data_json='{\"forged\":1}' WHERE session_id=? AND n=2", (env.sid,))
        r = env.db.verify_chain(env.sid)
        assert r["ok"] is False and "altered" in r["error"]

    def test_false_claim_rejected_true_claim_accepted(self, make_env):
        env = make_env()
        assert env.call("claim_achievement", {"type": "tower", "params": {"min_blocks": 2}})["verified"] is False
        env.call("walk_to", {"x": -1.5, "y": -2.1})
        env.call("pick_up", {"object_id": "block_02"})
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        env.call("place", {"object_id": "block_02", "position": [-2.5, -3.0]})
        assert env.call("claim_achievement", {"type": "tower", "params": {"min_blocks": 2}})["verified"] is True

    def test_juggling_claim_requires_real_catches(self, world, walk):
        walk(world, -2.5, -2.1)
        world.pick_up("block_01")
        for _ in range(5):                     # throw and grab instantly: not a toss
            world.throw("block_01", parse_direction("up"), 150.0, wait="none")
            world.pick_up("block_01")
        assert verify_claim("juggle", {"objects": ["block_01"], "cycles": 3}, world, world.events)["verified"] is False
        for _ in range(3):                     # genuine tosses caught in the air
            world.throw("block_01", parse_direction("up"), 150.0, wait="none")
            world.catch("block_01")
        v = verify_claim("juggle", {"objects": ["block_01"], "cycles": 3}, world, world.events)
        assert v["verified"] is True and v["measured"]["verified_cycles"] == 3

    def test_goal_completion_claim_is_checked(self, make_env):
        env = make_env()
        g = env.call("set_goal", {"title": "spire"})
        r = env.call("update_goal", {"goal_id": g["goal_id"], "status": "completed",
                                     "claim": {"type": "object_on", "params": {"object": "block_01", "surface": "spire"}}})
        assert r["status"] == "completed" and r["claim_check"]["verified"] is False and "warning" in r

    def test_implausible_flight_detected(self):
        assert check_flight({"launch_velocity": [0, 0, 5], "launch_speed": 5, "max_height_above_launch": 9.0})


# =============================================================== Phase 4/5/6: sessions
def _policy(steps):
    def policy(system, opening, tools):
        for tool, args in steps:
            r = yield ("call", tool, args)
            assert r is not None
    return policy


class TestSessions:
    def test_exploration_prompt_has_personality_and_no_physics_hints(self, cfg, db):
        seen = {}

        def policy(system, opening, tools):
            seen["system"], seen["opening"] = system, opening
            yield ("call", "end_session", {})
        run_session(cfg, db, "a1", PolicyAdapter(policy))
        s = seen["system"]
        # Default is the researcher's free-agency prompt, verbatim (PERSONALITY_FREE_AGENCY_V2).
        assert s.startswith("You are an autonomous AI agent inhabiting a simulated physical world.")
        assert "You have no assigned mission." in s and "You decide what happens next." in s
        assert "Never claim to have accomplished something without evidence from the environment." in s
        import re as _re
        for leak in ("ring", "the ring", "jump-throw", "take-off", "momentum", "generate", "velocity"):
            pat = _re.compile(rf"\b{_re.escape(leak)}\b", _re.I)        # whole words: "exploring" is fine
            assert not pat.search(s) and not pat.search(seen["opening"]), leak

    def test_exploration_prompt_is_selectable_and_recorded(self, cfg, db):
        import copy
        from playground.agents.prompts import PERSONALITY_BRIEF_V1, exploration_system, prompt_hash
        seen = {}

        def policy(system, opening, tools):
            seen["system"] = system
            yield ("call", "end_session", {})
        c = copy.deepcopy(cfg)
        c["agents"]["exploration_prompt"] = "brief_v1"
        s = run_session(c, db, "a1", PolicyAdapter(policy))
        assert seen["system"].startswith(PERSONALITY_BRIEF_V1)
        started = [e for e in db.events(s["session_id"]) if e["type"] == "session_started"][0]["data"]
        assert started["prompt_id"] == "brief_v1"
        assert started["system_prompt_hash"] == prompt_hash(exploration_system("brief_v1"))
        # the stored system message is exactly what the agent received
        assert db.q("SELECT content FROM messages WHERE session_id=? AND role='system'",
                    (s["session_id"],))[0]["content"] == seen["system"]

    def test_unknown_exploration_prompt_is_rejected(self, cfg, db):
        import copy
        c = copy.deepcopy(cfg)
        c["agents"]["exploration_prompt"] = "does_not_exist"
        with pytest.raises(ValueError, match="unknown exploration prompt"):
            run_session(c, db, "a1", PolicyAdapter(_policy([])))

    def test_reset_and_persistence_across_sessions(self, cfg, db):
        run_session(cfg, db, "a1", PolicyAdapter(_policy([
            ("walk_to", {"x": -2.5, "y": -2.1}), ("pick_up", {"object_id": "block_01"}),
            ("throw", {"object_id": "block_01", "direction": "east", "force": 200}),
            ("write_diary", {"text": "I threw block_01 east."}), ("end_session", {})])))
        seen = {}

        def p2(system, opening, tools):
            seen["opening"] = opening
            r = yield ("call", "inspect_object", {"object_id": "block_01"})
            seen["pos"] = r["object"]["position"]
            yield ("call", "end_session", {})
        s2 = run_session(cfg, db, "a1", PolicyAdapter(p2))
        assert "I threw block_01 east." in seen["opening"]          # memory carried over
        assert seen["pos"] == pytest.approx([-2.5, -3.0, 0.25], abs=1e-3)   # world reset
        assert s2["session_number"] == 2
        assert s2["island_reset"]["this_session_started_from"] == s2["island_reset"]["next_session_starts_from"]
        from playground.session import memory_root
        hist = memory_root(cfg) / "a1" / "session_history"
        assert (hist / "session_001" / "final_world_state.json").exists()

    def test_agents_are_isolated(self, cfg, db):
        run_session(cfg, db, "a1", PolicyAdapter(_policy([("write_diary", {"text": "SECRET-A1"}),
                                                          ("end_session", {})])))
        seen = {}

        def p(system, opening, tools):
            seen["o"] = opening
            r = yield ("call", "read_memory", {"file": "diary"})
            seen["d"] = r["content"]
            yield ("call", "end_session", {})
        run_session(cfg, db, "a2", PolicyAdapter(p))
        assert "SECRET-A1" not in seen["o"] and "SECRET-A1" not in seen["d"]

    def test_thirteen_agents_concurrently(self, cfg, db):
        res = run_agents(cfg, db, lambda a: PolicyAdapter(_policy([("jump", {}), ("write_diary", {"text": a}),
                                                                   ("end_session", {})])),
                         sessions=1, concurrency=4)
        assert res["sessions_run"] == 13 and res["completed"] == 13 and not res["errors"]
        ids = {r["agent_id"] for r in db.q("SELECT agent_id FROM sessions")}
        assert ids == set(agent_ids(cfg))
        for sid in [r["id"] for r in db.q("SELECT id FROM sessions")]:
            assert db.verify_chain(sid)["ok"]

    def test_time_limit_ends_session(self, cfg, db):
        cfg["session"]["duration_minutes"] = 1.0
        t = {"now": 0.0}

        def clock():
            t["now"] += 7.0                     # every call advances a fake wall clock by 7 s
            return t["now"]

        def p(system, opening, tools):
            while True:
                r = yield ("call", "look_around", {})
                if not r["success"]:
                    break
            yield ("call", "write_diary", {"text": "time up"})
        s = run_session(cfg, db, "a1", PolicyAdapter(p), clock=clock)
        assert s["end_reason"] == "time_limit"

    def test_adapter_error_marks_session_resumable_and_resume_is_exact(self, cfg, db):
        class Boom(Exception):
            pass
        from playground.agents.adapters import AdapterError

        def p1(system, opening, tools):
            yield ("call", "walk_to", {"x": -2.5, "y": -2.1})
            yield ("call", "pick_up", {"object_id": "block_01"})
            yield ("call", "throw", {"object_id": "block_01", "direction": "east", "force": 120})
            raise AdapterError("network went away")
        s = run_session(cfg, db, "a1", PolicyAdapter(p1))
        assert s["end_reason"] == "adapter_error" and db.session(s["session_id"])["status"] == "interrupted"
        seen = {}

        def p2(system, opening, tools):
            seen["opening"] = opening
            r = yield ("call", "inspect_object", {"object_id": "block_01"})
            seen["pos"] = r["object"]["position"]
            yield ("call", "end_session", {})
        s2 = run_session(cfg, db, "a1", PolicyAdapter(p2), resume_session_id=s["session_id"])
        assert s2["session_id"] == s["session_id"] and "resumed" in seen["opening"]
        assert seen["pos"][0] > 0                       # block is where the throw left it, not reset
        rep = [e for e in db.events(s["session_id"], types=["session_resumed"])][0]["data"]
        assert rep["state_matches_log"] is True
        assert db.verify_chain(s["session_id"])["ok"]


# =============================================================== Phase 8: behaviour
class TestBehaviour:
    def test_categories_from_recorded_behaviour(self, cfg, db):
        steps = [("set_goal", {"title": "Build a castle with only three blocks"}),
                 ("assign_meaning", {"target": ["block_01", "block_02"], "meaning": "the castle gate"}),
                 ("walk_to", {"x": -2.5, "y": -2.1})]
        for f in (60, 90, 120):
            steps += [("pick_up", {"object_id": "block_01"}),
                      ("throw", {"object_id": "block_01", "direction": "north", "force": f}),
                      ("walk_to", {"x": -2.5, "y": -2.1})]
        steps += [("end_session", {})]
        s = run_session(cfg, db, "a1", PolicyAdapter(_policy(steps)))
        c = classify_session(db, s["session_id"])
        assert c["self_directed_goals"]["count"] == 1
        assert c["self_directed_goals"]["self_imposed_constraints"]
        assert c["pretend_play"]["count"] == 1 and c["pretend_play"]["group_meanings"] == 1
        assert any(v["activity"] == "throw" and v["distinct_parameterisations"] == 3
                   for v in c["repetition_with_variation"]["varied_repetitions"])
        assert "No statement is made about experience" in c["disclaimer"]

    def test_no_goals_means_activity_counts_as_own_sake(self, cfg, db):
        s = run_session(cfg, db, "a1", PolicyAdapter(_policy([("jump", {}), ("jump", {}), ("end_session", {})])))
        c = classify_session(db, s["session_id"])
        assert c["self_directed_goals"]["count"] == 0 and c["activity_for_its_own_sake"]["count"] == 2


# =============================================================== Phase 10: benchmarking
class TestBenchmark:
    def test_task_verifiers_reject_untouched_world(self, world):
        for t in TASKS.values():
            if t.scene:
                continue
            assert t.verify(world, [])["verified"] is False, t.id

    def test_unfinished_construction_scene(self, cfg):
        w = PhysicsWorld(cfg)
        TASKS["complete_construction"].scene(w)
        w.setup_finish()
        assert TASKS["complete_construction"].verify(w, w.events)["score"] == 3

    def test_groups_and_ablation_end_to_end(self, cfg, db):
        cfg["session"]["max_steps"] = 70
        for _ in range(2):
            run_session(cfg, db, "agent_001", make_adapter(cfg, "scripted"))
        b = run_benchmark(cfg, db, lambda a: make_adapter(cfg, "scripted"), ["agent_001"], ["fresh_001"],
                          tasks=["spire_landing", "generate_blocks"])
        res = {(r["group"], r["task_id"]): r for r in b["results"]}
        assert res[("A", "spire_landing")]["memory_use"]["relevant_handbook_entries_at_start"]
        assert res[("B", "spire_landing")]["memory_use"]["memory_empty_at_start"]
        for r in b["results"]:
            assert r["completed"] in (True, False) and r["end_reason"] == "agent_ended"
        from playground.session import memory_root
        real = MemoryStore(memory_root(cfg), "agent_001", cfg)
        did = next(d["id"] for d in real.discoveries["items"] if d.get("category") == "ring_generation")
        before = json.dumps(real.discoveries)
        a = run_ablation(cfg, db, lambda x: make_adapter(cfg, "scripted"), "agent_001", [did],
                         tasks=["generate_blocks"])
        conds = {r["condition"]: r for r in a["results"]}
        assert did in conds["full_memory"]["memory_use"]["relevant_handbook_entries_at_start"]
        assert did not in conds["ablated"]["memory_use"]["relevant_handbook_entries_at_start"]
        assert json.dumps(MemoryStore(memory_root(cfg), "agent_001", cfg).discoveries) == before

    def test_unknown_task_and_item(self, cfg, db):
        with pytest.raises(ValueError):
            run_benchmark(cfg, db, lambda a: make_adapter(cfg, "scripted"), ["x"], tasks=["nope"])
        ensure_agents(cfg, db, ["agent_009"])
        with pytest.raises(ValueError):
            run_ablation(cfg, db, lambda a: make_adapter(cfg, "scripted"), "agent_009", ["D-999"])


# =============================================================== Phase 13: adapters + config
class TestAdaptersAndConfig:
    def test_openai_compat_adapter_against_a_fake_endpoint(self, cfg, db, monkeypatch):
        import httpx
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            assert body["tools"] and body["messages"][0]["role"] == "system"
            if len(calls) == 1:
                return httpx.Response(200, json={"choices": [{"message": {"content": "Let me look.", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "observe_world", "arguments": "{}"}}]}}]})
            assert body["messages"][-1]["role"] == "tool" and '"success":true' in body["messages"][-1]["content"]
            return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "end_session", "arguments": "{}"}}]}}]})
        from playground.agents import OpenAICompatAdapter
        ad = OpenAICompatAdapter("http://fake/v1", "m")
        ad.client = httpx.Client(transport=httpx.MockTransport(handler))
        s = run_session(cfg, db, "a1", ad)
        assert s["end_reason"] == "agent_ended" and len(calls) == 2
        assert db.messages(s["session_id"])[-1]["content"] == "Let me look."

    def test_anthropic_adapter_against_a_fake_endpoint(self, cfg, db):
        import httpx
        n = {"i": 0}

        def handler(request):
            n["i"] += 1
            body = json.loads(request.content)
            assert request.headers["x-api-key"] == "k" and body["tools"][0]["input_schema"]
            name = "jump" if n["i"] == 1 else "end_session"
            return httpx.Response(200, json={"content": [{"type": "tool_use", "id": f"t{n['i']}", "name": name,
                                                          "input": {}}]})
        from playground.agents import AnthropicAdapter
        ad = AnthropicAdapter("claude-x", "k")
        ad.client = httpx.Client(transport=httpx.MockTransport(handler))
        s = run_session(cfg, db, "a1", ad)
        assert s["end_reason"] == "agent_ended" and n["i"] == 2

    def test_adapter_http_error_is_not_success(self, cfg, db):
        import httpx
        from playground.agents import OpenAICompatAdapter
        ad = OpenAICompatAdapter("http://fake/v1", "m")
        ad.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
        s = run_session(cfg, db, "a1", ad)
        assert s["end_reason"] == "adapter_error" and "401" in s["error"]

    def test_config_validation(self):
        with pytest.raises(ValueError):
            load_config(overrides={"rules": {"ring_generation": {"charges": 99}}})
        with pytest.raises(Exception):
            make_adapter(load_config(), "nonexistent")
