"""HVE tests for the 2026-11-13 audit fixes: resume-adapter guard and neural-network on/off switch."""
import pytest

from playground.agents.adapters import PolicyAdapter
from playground.integrity.evidence import verify_experiment
from playground.session.runner import ResumeAdapterMismatch, check_resume_adapter, ml_enabled_for, run_session


class _A:
    def __init__(self, name, model):
        self.name, self.model = name, model


def _end_policy(system, briefing, tools):
    yield ("call", "end_session", {})


class TestResumeGuard:
    def test_happy_same_adapter_and_model(self):
        check_resume_adapter({"id": 1, "adapter": "lmstudio_unsloth4b", "model": "unslothqen 4b"},
                             _A("lmstudio_unsloth4b", "unslothqen 4b"))

    def test_validate_session19_scenario_refused(self):
        # The real contamination: LLM session resumed with the CLI default 'scripted' adapter.
        with pytest.raises(ResumeAdapterMismatch):
            check_resume_adapter({"id": 19, "adapter": "lmstudio_unsloth4b", "model": "unslothqen 4b"},
                                 _A("scripted", "scripted-reference"))

    def test_validate_different_model_refused(self):
        with pytest.raises(ResumeAdapterMismatch):
            check_resume_adapter({"id": 2, "adapter": "lmstudio", "model": "qwen/qwen3.5-9b"},
                                 _A("lmstudio", "lowkeyqwen3.52b"))

    def test_error_legacy_row_without_adapter_is_allowed(self):
        check_resume_adapter({"id": 3, "adapter": None, "model": None}, _A("scripted", "x"))


class TestMlSwitch:
    def test_happy_default_on(self):
        assert ml_enabled_for({}, "agent_001")

    def test_validate_global_off(self):
        assert not ml_enabled_for({"ml": {"enabled": False}}, "agent_001")

    def test_validate_per_agent_override(self):
        cfg = {"ml": {"enabled": False, "agents": {"agent_003": {"enabled": True}}}}
        assert ml_enabled_for(cfg, "agent_003") and not ml_enabled_for(cfg, "agent_001")

    def test_session_offers_learning_tools_by_default(self, cfg, db):
        ad = PolicyAdapter(_end_policy, label="policy")
        run_session(cfg, db, "a1", ad)
        assert any(t.startswith("nn_") for t in ad._tools)

    def test_error_session_offers_no_learning_tools_when_off(self, cfg, db):
        cfg["ml"] = {"enabled": False}
        ad = PolicyAdapter(_end_policy, label="policy")
        run_session(cfg, db, "a1", ad)
        assert not any(t.startswith("nn_") for t in ad._tools)


def _survey_then_record(system, briefing, tools):
    """A tiny policy that records a mistake (claim-free, always accepted) then ends."""
    yield ("call", "record_mistake", {"what": "walked off the mid deck", "lesson": "check the drop first"})
    yield ("call", "end_session", {})


class TestMemoryProvenance:
    def test_happy_item_is_stamped_with_the_authoring_brain(self, cfg, db):
        ad = PolicyAdapter(_survey_then_record, label="lmstudio_unsloth4b", model="unslothqen 4b")
        s = run_session(cfg, db, "agent_prov", ad)
        from playground.memory import MemoryStore
        from playground.session.runner import memory_root
        mem = MemoryStore(memory_root(cfg), "agent_prov", cfg, db=db)
        assert mem.state["mistakes"], "a mistake should have been recorded"
        author = mem.state["mistakes"][-1]["author"]
        assert author["adapter"] == "lmstudio_unsloth4b" and author["model"] == "unslothqen 4b"
        assert s["end_reason"] == "agent_ended"

    def test_validate_two_brains_leave_distinguishable_stamps(self, cfg, db):
        # Same agent, two sessions, two different brains: each item carries its own author.
        from playground.memory import MemoryStore
        from playground.session.runner import memory_root
        run_session(cfg, db, "agent_two", PolicyAdapter(_survey_then_record, label="brainA", model="mA"))
        run_session(cfg, db, "agent_two", PolicyAdapter(_survey_then_record, label="brainB", model="mB"))
        mem = MemoryStore(memory_root(cfg), "agent_two", cfg, db=db)
        authors = {m["author"]["adapter"] for m in mem.state["mistakes"]}
        assert authors == {"brainA", "brainB"}

    def test_error_legacy_store_without_author_set_does_not_crash(self, cfg, db, tmp_path):
        from playground.memory import MemoryStore
        mem = MemoryStore(tmp_path / "mem", "agent_legacy", cfg, db=db)
        # author never set (direct use, as in older code): stamp is still well-formed, brain unknown.
        m = mem.add_mistake(1, 0.0, "did a thing", None, None)
        assert m["author"]["adapter"] is None and m["author"]["session"] == 1


class TestResumeAdapterResolution:
    def test_happy_bare_resume_keeps_the_sessions_own_adapter(self, cfg, db):
        from playground.session.runner import resolve_resume_adapter
        sid = db.create_session(agent_id="a1", mode="exploration", session_number=1,
                                adapter="lmstudio_unsloth4b", model="unslothqen 4b", config_hash="h",
                                config_json={}, memory_dir="x", budget_json={})
        assert resolve_resume_adapter(db, sid, None, None) == ("lmstudio_unsloth4b", "unslothqen 4b")

    def test_validate_explicit_adapter_overrides_on_resume(self, cfg, db):
        from playground.session.runner import resolve_resume_adapter
        sid = db.create_session(agent_id="a1", mode="exploration", session_number=1,
                                adapter="lmstudio_unsloth4b", model="unslothqen 4b", config_hash="h",
                                config_json={}, memory_dir="x", budget_json={})
        assert resolve_resume_adapter(db, sid, "claude", None) == ("claude", None)


def _measure_action(idx, tool, result, category="experiment", success=True):
    return {idx: {"idx": idx, "tool": tool, "category": category, "success": success, "result": result}}


class TestEvidenceCoincidence:
    """The value-match coincidence bug: a claimed number must not count as a measurement merely because
    it equals an unrelated positional coordinate the engine logged (audit 2026-11-13, E-009/E-010)."""

    def test_error_count_matching_a_coordinate_is_not_measured(self):
        # platform_count=3 vs the ring's x-coordinate objects.ring[0]=3.0 (the real contamination).
        cited = _measure_action(1, "observe_world", {"objects": {"ring": [3.0, 5.69, 0.75]}})
        v = verify_experiment({"measurements": {"platform_count": 3.0}}, cited)
        assert v["level"] != "measured"
        assert v["verdict"] == "grounded"
        assert v["weak_matches"] and v["weak_matches"][0]["match"] == "value_coordinate"

    def test_happy_named_scalar_match_still_counts_as_measured(self):
        # spire_height=6.0 vs the engine's named top_height=6.0: a real measurement, must stay "measured".
        cited = _measure_action(1, "inspect_object", {"top_height": 6.0, "base_height": 0.0})
        v = verify_experiment({"measurements": {"spire_height": 6.0}}, cited)
        assert v["level"] == "measured" and v["verdict"] == "consistent"

    def test_happy_same_field_name_match_still_counts(self):
        cited = _measure_action(1, "measure_height", {"max_height": 7.24})
        v = verify_experiment({"measurements": {"max_height": 7.2354}}, cited)
        assert v["level"] == "measured" and v["matches"][0]["match"] == "same_field"

    def test_validate_wrong_same_field_value_is_contradicted(self):
        cited = _measure_action(1, "measure_height", {"max_height": 4.5})
        v = verify_experiment({"measurements": {"max_height": 32.0}}, cited)
        assert v["verdict"] == "contradicted" and v["verified"] is False


class TestContextBudget:
    """The history-trim + context-budget fix (audit 2026-11-13). The old _trim never shrank history and
    ignored the model's real token window, so LM Studio silently truncated the middle of the prompt."""

    def _adapter(self, **kw):
        from playground.agents.adapters import OpenAICompatAdapter
        ad = OpenAICompatAdapter("http://fake/v1", "m", **kw)
        ad.begin("SYSTEM PROMPT " * 20, ["observe_world", "jump", "throw", "move", "look_around"])
        ad.messages.append({"role": "user", "content": "OPENING " * 50})
        return ad

    def _grow(self, ad, n):
        for i in range(n):
            ad.messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "look_around", "arguments": "{}"}}]})
            ad.messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "RESULT " * 80})

    def test_happy_history_actually_shrinks_under_token_budget(self):
        # 2B-sized window: the whole request must fit, so a long history gets trimmed down and stays down.
        ad = self._adapter(context_tokens=12544)
        self._grow(ad, 200)
        ad._trim()
        budget = ad._history_budget_chars()
        body = sum(len(__import__("json").dumps(m)) for m in ad.messages[2:]) + ad._fixed_chars()
        assert body <= budget, f"request {body} still exceeds budget {budget}"
        assert ad.last_trim and ad.last_trim["removed_exchanges"] > 0

    def test_validate_budget_accounts_for_tool_schema_and_system(self):
        # The fixed part (system + tools + opening) is subtracted from the window, not ignored.
        ad = self._adapter(context_tokens=12544)
        assert ad._history_budget_chars() < int(12544 * ad.chars_per_token) - ad._fixed_chars() + 1
        assert ad._fixed_chars() > 0

    def test_validate_trim_is_idempotent_and_keeps_head(self):
        ad = self._adapter(context_tokens=12544)
        self._grow(ad, 200)
        ad._trim()
        size1 = len(ad.messages)
        ad._trim()  # a second trim with no new messages must not keep deleting.
        assert len(ad.messages) == size1
        assert ad.messages[0]["role"] == "system" and "OPENING" in ad.messages[1]["content"]

    def test_error_no_context_tokens_falls_back_to_char_limit(self):
        # Unknown window (server offline): behave as before, bounded by max_history_chars only.
        ad = self._adapter(context_tokens=None, max_history_chars=5000)
        self._grow(ad, 200)
        ad._trim()
        body = sum(len(__import__("json").dumps(m)) for m in ad.messages[2:]) + ad._fixed_chars()
        # fixed part may exceed 5000 on its own, but the trimmable middle is bounded.
        middle = sum(len(__import__("json").dumps(m)) for m in ad.messages[3:])
        assert middle <= 5000


class TestModelDiscovery:
    """Listing/probe of an openai_compat server's models (audit 2026-11-13 steer: pick any LM Studio
    model, and derive the real context window instead of guessing)."""

    def _server(self, payload, status=200):
        import httpx
        return httpx.MockTransport(lambda req: httpx.Response(status, json=payload))

    def test_happy_lists_models_from_v0_endpoint(self, monkeypatch):
        import httpx
        from playground.agents import adapters
        payload = {"data": [{"id": "qwenA", "state": "loaded", "loaded_context_length": 14538,
                             "max_context_length": 262144},
                            {"id": "qwenB", "state": "not-loaded", "max_context_length": 131072}]}
        monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, json=payload))
        models = adapters.list_openai_models("http://127.0.0.1:1234/v1")
        assert [m["id"] for m in models] == ["qwenA", "qwenB"]

    def test_happy_probe_prefers_loaded_context_length(self, monkeypatch):
        import httpx
        from playground.agents import adapters
        payload = {"data": [{"id": "qwenA", "loaded_context_length": 14538, "max_context_length": 262144}]}
        monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, json=payload))
        assert adapters.probe_context_tokens("http://127.0.0.1:1234/v1", "qwenA") == 14538

    def test_error_offline_server_yields_none_not_crash(self, monkeypatch):
        import httpx
        from playground.agents import adapters

        def boom(url, **kw):
            raise httpx.ConnectError("refused")
        monkeypatch.setattr(httpx, "get", boom)
        assert adapters.list_openai_models("http://127.0.0.1:1234/v1") == []
        assert adapters.probe_context_tokens("http://127.0.0.1:1234/v1", "anything") is None


class TestRepeatObservation:
    """Repeat-observation feedback (audit 2026-11-13): a read-only observation of an unchanged world is
    flagged, not blocked, so a weak model stops looping without losing autonomy. A 2B model once made
    126 identical observations in a row."""

    def test_happy_first_observation_is_not_flagged(self, make_env):
        env = make_env()
        r = env.call("look_around", {})
        assert r["success"] and "repeat_of_action" not in r

    def test_error_repeated_identical_observation_is_flagged(self, make_env):
        env = make_env()
        first = env.call("look_around", {})
        again = env.call("look_around", {})
        assert again["repeat_of_action"] == first["action_index"]
        assert "world has not changed" in again["repeat_notice"]

    def test_validate_physical_action_resets_repeat_tracking(self, make_env):
        env = make_env()
        env.call("look_around", {})
        env.call("look_around", {})            # flagged
        env.call("move", {"direction": "north", "distance": 2})
        r = env.call("look_around", {})        # world moved: fresh observation, not flagged
        assert "repeat_of_action" not in r

    def test_validate_different_args_is_a_different_observation(self, make_env):
        env = make_env()
        env.call("look_around", {})
        r = env.call("look_around", {"radius": 10})
        assert "repeat_of_action" not in r

    def test_error_measurement_is_not_treated_as_a_repeat_observation(self, make_env):
        env = make_env()
        env.call("measure_height", {"target": "spire"})
        r = env.call("measure_height", {"target": "spire"})
        assert "repeat_of_action" not in r


def _continue_policy(system, briefing, tools):
    # Record a goal (unfinished work), then end asking to continue.
    yield ("call", "set_goal", {"title": "map the whole island", "description": "only did the west half"})
    yield ("call", "end_session", {"reason": "out of time", "continue_next_session": True})


def _finish_policy(system, briefing, tools):
    yield ("call", "end_session", {"reason": "done", "continue_next_session": False})


class TestSessionLifecycle:
    """Session-lifecycle fixes (audit 2026-11-13): agents stopped at 1-51% of budget, read their briefing
    as a summary request, and had no way to carry research into another session."""

    def test_happy_opening_frames_memory_as_reference_not_a_task(self):
        from playground.agents.prompts import exploration_opening
        briefing = "----- handbook.md -----\nD-001 the spire is 6m tall"
        opening = exploration_opening("agent_001", 3, briefing, 60.0, None)
        assert "reference only" in opening
        # Free-will contract (Nov 2026): the opening gives no assigned objective
        # and no productivity mandate — what to do is entirely the agent's choice.
        assert "no assigned objective" in opening
        assert "your job" not in opening.lower()
        # The situation note is the LAST thing the model reads, after the memory block.
        assert opening.index("end of saved memory") < opening.index("end_session")

    def test_happy_end_session_records_continuation_request(self, make_env):
        env = make_env()
        r = env.call("end_session", {"reason": "unfinished", "continue_next_session": True})
        assert r["continue_next_session"] is True and env.continue_requested is True

    def test_validate_end_session_without_continue_defaults_false(self, make_env):
        env = make_env()
        r = env.call("end_session", {"reason": "done"})
        assert r["continue_next_session"] is False and env.continue_requested is False

    def test_happy_continuation_grants_one_extra_session(self, cfg, db):
        from playground.session.manager import run_agents
        calls = {"n": 0}

        def factory(agent_id):
            calls["n"] += 1
            # First session asks to continue; the granted extra session finishes.
            return PolicyAdapter(_continue_policy if calls["n"] == 1 else _finish_policy, label="p")
        res = run_agents(cfg, db, factory, ["agent_c"], sessions=1, max_continuations=1)
        assert res["sessions_run"] == 2  # base + 1 continuation
        assert res["continuations_granted"] >= 1

    def test_validate_continuation_is_bounded(self, cfg, db):
        from playground.session.manager import run_agents
        # Every session asks to continue, but the cap is 2 extra: 1 base + 2 = 3 total, no runaway.
        res = run_agents(cfg, db, lambda a: PolicyAdapter(_continue_policy, label="p"),
                         ["agent_c"], sessions=1, max_continuations=2)
        assert res["sessions_run"] == 3

    def test_happy_budget_nudge_reports_remaining(self):
        from playground.agents.prompts import continue_nudge
        msg = continue_nudge({"steps": 42, "seconds": 1800, "phase": "active"})
        assert "42 actions" in msg and "more minutes" in msg


class TestGoalsAndBriefing:
    """Structured goals + bounded briefing (audit 2026-11-13). Agents never used set_goal (dashboard
    'none set') and one briefing grew to 22k chars, crowding a 12-16k-token window."""

    def test_happy_goal_records_kind_and_next_step(self, make_env):
        env = make_env()
        r = env.call("set_goal", {"title": "why do blocks spawn?", "kind": "question",
                                  "next_step": "lay the ring flat around block_01"})
        assert r["kind"] == "question"
        g = env.memory.state["goals"][-1]
        assert g["kind"] == "question" and g["next_step"] == "lay the ring flat around block_01"

    def test_validate_goal_kind_defaults_to_objective(self, make_env):
        env = make_env()
        env.call("set_goal", {"title": "build a tower"})
        assert env.memory.state["goals"][-1]["kind"] == "objective"

    def test_happy_briefing_is_bounded(self, cfg, db, tmp_path):
        from playground.memory import MemoryStore
        cfg = {**cfg, "memory": {**cfg["memory"], "briefing_char_budget": 2000}}
        mem = MemoryStore(tmp_path / "m", "agent_big", cfg, db=db)
        # Pile on far more diary than the budget allows.
        for i in range(200):
            mem.add_diary(i, float(i), "A very long diary entry number %d " % i * 20)
        brief = mem.briefing()
        assert len(brief) <= 2000 + 500  # core parts + at most one trimmed diary block

    def test_validate_briefing_keeps_handbook_and_goals(self, cfg, db, tmp_path):
        from playground.memory import MemoryStore
        cfg = {**cfg, "memory": {**cfg["memory"], "briefing_char_budget": 1500}}
        mem = MemoryStore(tmp_path / "m", "agent_keep", cfg, db=db)
        mem.add_goal(1, 0.0, "map the island", "half done", kind="explore")
        for i in range(100):
            mem.add_diary(i, float(i), "filler diary %d " % i * 20)
        brief = mem.briefing()
        # The durable open goal survives even though most diary is dropped.
        assert "map the island" in brief


class TestGeneralTransitionNetwork:
    """The NN was throw-only by construction (features.py INPUT/TARGET_COLUMNS). The transition module
    adds a general (state, action) -> next-state representation across every physical action, so the
    architecture is no longer a throwing specialist. Audit 2026-11-13."""

    def _acts(self):
        # move, pick_up, place: three different action types with engine-style results.
        return [
            {"tool": "move", "args_json": '{"direction":[1,0,0],"distance":5}',
             "result_json": '{"success":true,"end_position":[5,0,0.8],"objects_disturbed":[{"distance_moved":1.9}]}'},
            {"tool": "pick_up", "args_json": '{"object_id":"block_01"}',
             "result_json": '{"success":true,"agent_position":[5,0,0.8],"holding":"block_01"}'},
            {"tool": "jump", "args_json": '{}',
             "result_json": '{"success":true,"position":[5,0,1.1],"left_ground_during_move":true}'},
        ]

    def test_happy_builds_rows_across_multiple_action_types(self):
        from playground.ml.transition_features import dataset_from_actions
        ds = dataset_from_actions(self._acts())
        assert ds["manifest"]["rows"] == 3
        tools = {r["action_id"] for r in ds["rows"]}
        assert len(tools) == 3  # three distinct action ids, not just throws

    def test_happy_row_captures_movement_and_object_change(self):
        from playground.ml.transition_features import transition_row
        before = {"position": [0, 0, 0.8], "speed": 0.0, "holding": False}
        r = transition_row("move", {"direction": [1, 0, 0], "distance": 5},
                           {"success": True, "end_position": [5, 0, 0.8],
                            "objects_disturbed": [{"distance_moved": 1.9}]}, before)
        assert round(r["agent_moved"], 1) == 5.0 and round(r["object_moved"], 1) == 1.9

    def test_validate_non_transition_tool_yields_no_row(self):
        from playground.ml.transition_features import transition_row
        assert transition_row("look_around", {}, {"success": True}, {"position": [0, 0, 0]}) is None

    def test_happy_spec_validates_against_transition_vocabulary(self):
        from playground.ml.nets import validate_spec
        from playground.ml.transition_features import TRANSITION_INPUTS, TRANSITION_TARGETS
        spec = validate_spec({"name": "motion", "inputs": ["action_id", "param_distance"],
                              "target": "agent_moved"}, TRANSITION_INPUTS, TRANSITION_TARGETS)
        assert spec["target"] == "agent_moved"

    def test_error_transition_column_rejected_by_throw_vocabulary(self):
        import pytest as _pytest
        from playground.ml import MLError
        from playground.ml.nets import validate_spec
        # Default (throw) vocabulary must NOT silently accept transition columns.
        with _pytest.raises(MLError):
            validate_spec({"name": "m", "inputs": ["action_id"], "target": "agent_moved"})


def _survey_policy(system, briefing, tools):
    yield ("call", "look_around", {})
    yield ("call", "move", {"direction": "north", "distance": 2})
    yield ("call", "set_goal", {"title": "find the edge", "kind": "explore"})
    yield ("call", "end_session", {"reason": "done"})


class TestModelComparison:
    """Equal-conditions model comparison (audit 2026-11-13, section 8). The 2B/4B/9B runs were not
    comparable; this harness holds agents, action cap and context window equal across models."""

    def test_happy_session_profile_reports_behaviour(self, cfg, db):
        from playground.session.model_compare import session_profile
        from playground.session.runner import run_session
        s = run_session(cfg, db, "agent_p", PolicyAdapter(_survey_policy, label="p", model="m1"))
        prof = session_profile(db, s["session_id"])
        assert prof["distinct_tools"] >= 2 and prof["goals_set"] == 1
        assert 0.0 <= prof["observation_share"] <= 1.0 and prof["model"] == "m1"

    def test_happy_compare_runs_every_model_under_equal_conditions(self, cfg, db):
        from playground.session.model_compare import compare_models
        res = compare_models(cfg, db, lambda model, aid: PolicyAdapter(_survey_policy, label="p", model=model),
                             ["modelA", "modelB"], agents_per_model=2, sessions=1, max_steps=40)
        assert set(res["by_model"]) == {"modelA", "modelB"}
        # Equal conditions: same agent count and same action cap recorded for both.
        assert res["by_model"]["modelA"]["sessions"] == 2
        assert res["by_model"]["modelB"]["sessions"] == 2
        assert res["conditions"]["max_steps"] == 40

    def test_validate_fresh_agents_are_model_scoped(self, cfg, db):
        from playground.session.model_compare import compare_models
        res = compare_models(cfg, db, lambda model, aid: PolicyAdapter(_survey_policy, label="p", model=model),
                             ["modelA", "modelB"], agents_per_model=1, sessions=1)
        ids_a = {p["agent_id"] for p in res["profiles"]["modelA"]}
        ids_b = {p["agent_id"] for p in res["profiles"]["modelB"]}
        # No agent id is shared between models, so no memory leaks across the comparison.
        assert ids_a.isdisjoint(ids_b)

    def test_validate_context_tokens_pinned_equally(self, cfg, db):
        from playground.session.model_compare import compare_models
        captured = []

        def factory(model, aid):
            ad = PolicyAdapter(_survey_policy, label="p", model=model)
            captured.append(ad)
            return ad
        compare_models(cfg, db, factory, ["modelA"], agents_per_model=1, sessions=1, context_tokens=8000)
        # PolicyAdapter has no context_tokens attr; the harness must not crash trying to pin it.
        assert captured and getattr(captured[0], "context_tokens", None) in (None, 8000)


def _mixed_actions():
    return [
        {"tool": "climb", "args_json": "{}",
         "result_json": '{"success":true,"ledge_height":0.5,"height_gained":0.5,"started_airborne":false,"start_position":[0,0,0]}'},
        {"tool": "jump", "args_json": "{}",
         "result_json": '{"success":true,"total_mass":52,"takeoff_feet_height":0,"apex_above_takeoff":1.1,"air_time":0.9,"max_feet_height":1.1,"impact_speed":4.2}'},
        {"tool": "pick_up", "args_json": '{"object_id":"block_01"}',
         "result_json": '{"success":true,"object_mass":5,"relative_speed":0.1,"was_moving":false,"agent_on_ground":true}'},
        {"tool": "walk_to", "args_json": '{"x":2,"y":2}',
         "result_json": '{"success":true,"distance_travelled":2.8,"requested_distance":2.8,"objects_disturbed":[],"height_change":0}'},
    ]


class TestBehaviourDatasets:
    """More datasets than throws (audit 2026-11-13): climb, jump, locomotion, manipulation, placement and
    the cross-action 'affordance' self-model — each built from real engine records."""

    def test_happy_catalogue_lists_non_throw_datasets(self):
        from playground.ml.datasets import available_datasets
        names = {d["name"] for d in available_datasets()}
        assert {"climb", "jump", "locomotion", "manipulation", "placement", "affordance"} <= names

    def test_happy_builds_climb_and_jump_rows(self):
        from playground.ml.datasets import build_dataset
        acts = _mixed_actions()
        climb = build_dataset(acts, "climb")
        jump = build_dataset(acts, "jump")
        assert climb["manifest"]["rows"] == 1 and "ledge_height" in climb["rows"][0]
        assert jump["manifest"]["rows"] == 1 and jump["rows"][0]["apex_above_takeoff"] == 1.1

    def test_happy_affordance_is_cross_action(self):
        from playground.ml.datasets import build_dataset
        ds = build_dataset(_mixed_actions(), "affordance")
        # Every physical action contributes one affordance row (climb, jump, pick_up, walk_to).
        assert ds["manifest"]["rows"] == 4
        assert {"action_id", "success"} <= set(ds["rows"][0])

    def test_validate_vocab_from_manifest_picks_dataset_columns(self):
        from playground.ml.datasets import build_dataset, vocab_from_manifest
        ds = build_dataset(_mixed_actions(), "climb")
        ins, tgts = vocab_from_manifest(ds["manifest"])
        assert "ledge_height" in ins and "height_gained" in tgts

    def test_validate_throw_manifest_falls_back_to_default_vocab(self):
        from playground.ml.datasets import vocab_from_manifest
        # A throw dataset manifest has no inputs/targets -> (None, None) -> default throw vocab.
        assert vocab_from_manifest({"source": "engine_throws", "rows": 100}) == (None, None)

    def test_error_unknown_kind_raises(self):
        import pytest as _pytest
        from playground.ml.datasets import build_dataset
        with _pytest.raises(KeyError):
            build_dataset(_mixed_actions(), "telepathy")

    def test_happy_climb_dataset_trains_against_its_vocabulary(self):
        import pytest as _pytest
        from playground.ml import torch_status
        if not torch_status()["available"]:
            _pytest.skip("torch not available")
        from playground.ml.datasets import columns_for
        from playground.ml.nets import validate_spec
        ic, tc = columns_for("climb")
        spec = validate_spec({"name": "climb_ok", "inputs": ["ledge_height", "started_airborne"],
                              "target": "success"}, ic, tc)
        assert spec["target"] == "success"


class TestDashboardAccess:
    """Everything the audit added, reachable from the dashboard API (audit 2026-11-13 steer:
    'make everything accessible from the dashboard')."""

    def _client(self, cfg, db):
        from fastapi.testclient import TestClient
        from playground.dashboard.server import create_app
        return TestClient(create_app(cfg, db))

    def test_happy_adapters_and_dataset_kinds_listed(self, cfg, db):
        c = self._client(cfg, db)
        ad = c.get("/api/adapters").json()
        assert any(a["type"] == "openai_compat" for a in ad["adapters"])
        kinds = c.get("/api/ml/dataset-kinds").json()["kinds"]
        names = {k["name"] for k in kinds}
        assert {"throw", "climb", "jump", "affordance"} <= names

    def test_happy_ml_config_toggle(self, cfg, db):
        c = self._client(cfg, db)
        assert c.get("/api/ml/config").json()["enabled"] in (True, False)
        r = c.post("/api/ml/config", json={"enabled": False}).json()
        assert r["enabled"] is False
        r = c.post("/api/ml/config", json={"agent": "agent_001", "agent_enabled": True}).json()
        assert r["resolved"]["agent_001"] is True

    def test_happy_unsloth_is_a_configured_provider(self, cfg, db):
        c = self._client(cfg, db)
        names = {a["name"] for a in c.get("/api/adapters").json()["adapters"]}
        assert "unsloth" in names

    def test_happy_session_profile_endpoint(self, cfg, db):
        from playground.session.runner import run_session
        s = run_session(cfg, db, "agent_dash", PolicyAdapter(_survey_policy, label="p", model="m1"))
        c = self._client(cfg, db)
        prof = c.get(f"/api/sessions/{s['session_id']}/profile").json()
        assert prof["goals_set"] == 1 and "observation_share" in prof

    def test_happy_agent_goals_endpoint_reads_memory(self, cfg, db):
        from playground.session.runner import run_session
        run_session(cfg, db, "agent_g2", PolicyAdapter(_survey_policy, label="p", model="m1"))
        c = self._client(cfg, db)
        g = c.get("/api/agents/agent_g2/goals").json()
        assert any(x["title"] == "find the edge" for x in g["open_goals"])
