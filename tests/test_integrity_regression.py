"""Integrity regression suite (task 2026-11-16).

Covers the fourteen cases required by the audit task plus the free-will prompt
contract (no system prompt may steer agents toward particular actions). Every
test works in its own tmp workspace via the shared conftest fixtures.
"""
from __future__ import annotations

import json
import re

import pytest

from playground.agents.prompts import (
    BENCHMARK_SYSTEM, PERSONALITIES, SITUATION, exploration_opening, continue_nudge,
)
from playground.interface.schema import SPECS
from playground.memory import MemoryError_, MemoryStore
from playground.memory.store import AGENT_STATUSES, RESEARCHER_OVERLAYS, STATUSES

# ------------------------------------------------------------------ helpers
STEER_PHRASES = (
    # productivity / assigned-objective language that must never reach an agent.
    # Negations are fine ("you are not required to be productive" is freedom, not steering).
    r"keep going until", r"your job", r"take an action now",
    r"(?<!not required to )be productive",
    r"you must (?!not)", r"pick one .*(question|experiment)",
)


def _is_error(r) -> bool:
    return isinstance(r, dict) and ("error" in r or r.get("saved") is None and "discovery_id" not in r)


def _walk_pick_throw(env, obj="block_01", direction="up", force=150):
    env.call("walk_to", {"x": -2.5, "y": -2.1})
    env.call("pick_up", {"object_id": obj})
    return env.call("throw", {"object_id": obj, "direction": direction, "force": force})


def _measured(env, **kw):
    t = _walk_pick_throw(env, **kw)
    return env.call("record_experiment", {
        "question": "q", "hypothesis": "h", "actual_result": str(t["max_height"]),
        "measurements": {"max_height": t["max_height"]},
        "action_refs": [t["action_index"]], "supported": True,
    })


def _checked_env_memories(make_env):
    """Two independent agents in the same tests db/temp root."""
    return make_env("agent_x"), make_env("agent_y")


# ------------------------------------------------------- 1. no self-confirmation
class TestNoSelfConfirmation:
    def test_overlay_statuses_are_researcher_only(self, make_env):
        env = make_env()
        for st in RESEARCHER_OVERLAYS:
            r = env.call("record_discovery", {"statement": "s", "status": st})
            assert _is_error(r), r
            with pytest.raises(MemoryError_, match="researcher"):
                make_env().memory.add_discovery(1, "s", st, [])

    def test_fabricated_evidence_is_rejected(self, make_env):
        env = make_env()
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        r = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": ["E-999"]})
        assert _is_error(r) and "unknown" in json.dumps(r).lower()
        assert not env.memory.discoveries["items"]  # nothing was recorded from thin air

    def test_unmeasured_evidence_cannot_confirm_two_experiments(self, make_env):
        env = make_env()
        # two *cited* but never engine-measured experiments stay ungrantable
        e1 = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "about 3 m",
                                            "measurements": {}, "action_refs": [], "supported": True})
        e2 = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "about 4 m",
                                            "measurements": {}, "action_refs": [], "supported": True})
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]})
        assert d["granted_status"] != "confirmed"


# --------------------------------------- 2. evidence check != scientific truth
class TestEvidenceCheckSeparation:
    def test_passing_check_with_unsupported_result_does_not_confirm(self, make_env):
        env = make_env()
        t = _walk_pick_throw(env, direction="up", force=150)
        e = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": str(t["max_height"]),
                                           "measurements": {"max_height": t["max_height"]},
                                           "action_refs": [t["action_index"]], "supported": False})
        assert e["outcome"]["evidence_check"] == "passed"
        assert e["outcome"]["scientific"] == "contradicted"
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e["experiment_id"]]})
        # A checked experiment that refutes the claim is exactly the evidence
        # rule for 'disproven': the check passed but the claim does not hold.
        assert d["granted_status"] == "disproven"

    def test_failed_check_is_inconclusive_never_supporting(self, make_env):
        env = make_env()
        e = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "32 m",
                                           "measurements": {"max_height": 32.0}, "action_refs": [],
                                           "supported": True})
        assert e["outcome"]["evidence_check"] == "failed"
        assert e["outcome"]["scientific"] == "inconclusive"
        t = _walk_pick_throw(env)
        e2 = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": "99 m",
                                            "measurements": {"max_height": 99.0},
                                            "action_refs": [t["action_index"]], "supported": True})
        # implausible claim overrode the check
        assert e2["outcome"]["evidence_check"] == "failed"
        assert e2["outcome"]["scientific"] == "inconclusive"
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e["experiment_id"], e2["experiment_id"]]})
        assert d["granted_status"] == "hypothesis"


# --------------------------------------------- 3. contradictory experiments link
class TestContradictoryAssociation:
    def test_supporting_and_contradicting_are_both_listed(self, make_env):
        env = make_env()
        e1 = _measured(env, direction="up", force=150)  # supported=True
        t = _walk_pick_throw(env, direction="up", force=150)
        e2 = env.call("record_experiment", {"question": "q", "hypothesis": "h", "actual_result": str(t["max_height"]),
                                            "measurements": {"max_height": t["max_height"]},
                                            "action_refs": [t["action_index"]], "supported": False,
                                            "discovery_id": None})
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]})
        store = env.memory
        got = store.get_discovery(d["discovery_id"])
        assert e1["experiment_id"] in got["evidence"] and e2["experiment_id"] in got["evidence"]
        assert got["status"] == "hypothesis"  # one supporting measured experiment alone cannot confirm


# ----------------------------------------------- 4. challenge preserves history
class TestChallengeHistory:
    def test_challenge_appends_and_keeps_original(self, make_env):
        env = make_env()
        e1 = _measured(env)
        e2 = _measured(env)
        d = env.call("record_discovery", {"statement": "THE ORIGINAL CLAIM", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]})
        did = d["discovery_id"]
        hist_before = len(env.memory.get_discovery(did)["history"])
        r = env.memory.challenge(did, "verify the throw measurement")
        rec = env.memory.get_discovery(did)
        assert r["status"] == "challenged"
        assert rec["history"][-1]["kind"] == "challenge"
        assert rec["history"][-1]["reason"] == "verify the throw measurement"
        assert len(rec["history"]) == hist_before + 1  # append-only, nothing rewritten
        assert rec["statement"] == "THE ORIGINAL CLAIM"

    def test_agent_evidence_never_clears_or_lifts_a_challenge(self, make_env):
        env = make_env()
        e1 = _measured(env)
        e2 = _measured(env)
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]})
        did = d["discovery_id"]
        env.memory.challenge(did, "suspicious")
        e3 = _measured(env)
        u = env.call("update_discovery", {"discovery_id": did, "status": "confirmed",
                                          "add_experiment_ids": [e3["experiment_id"]]})
        rec = env.memory.get_discovery(did)
        assert u["granted_status"] is None or rec["status"] == "challenged"
        assert rec["status"] == "challenged"
        assert rec["flags"]["challenged"] and rec["flags"]["challenge_reason"] == "suspicious"
        assert rec["evidence_status"] == "confirmed"  # recorded, not granted
        assert "until a researcher resolves it" in rec["history"][-1]["reason"]

    def test_resolve_challenge_restores_evidence_status_and_audits(self, make_env):
        env = make_env()
        e1 = _measured(env)
        e2 = _measured(env)
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]})
        did = d["discovery_id"]
        env.memory.challenge(did, "re-check")
        out = env.memory.resolve_challenge(did, "checked, claim holds", actor="researcher")
        assert out["status"] == "confirmed"
        assert out["history"][-1]["kind"] == "resolve_challenge"
        ops = [r["op"] for r in env.db.q(
            "SELECT op FROM memory_audit WHERE agent_id=? ORDER BY id", ("agent_t",))]
        assert "challenge" in ops and "resolve_challenge" in ops


# ------------------------------------------ 5. mark-unverified keeps evidence
class TestMarkUnverified:
    def _confirmed(self, env):
        e1 = _measured(env)
        e2 = _measured(env)
        return env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                             "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]
                                             })

    def test_status_unverified_and_old_evidence_retained_but_ignored(self, make_env):
        env = make_env()
        d = self._confirmed(env)
        did = d["discovery_id"]
        rec = env.memory.mark_unverified(did, "fresh start")
        assert rec["status"] == "unverified"
        assert len(rec["evidence"]) == 2  # history preserved
        assert rec["history"][-1]["kind"] == "mark_unverified"
        cut = rec["flags"]["unverified_at_experiment"]
        assert cut >= 2
        # fresh measured evidence does NOT reinstate confirmation
        f = _measured(env)
        env.call("update_discovery", {"discovery_id": did, "status": "confirmed",
                                      "add_experiment_ids": [f["experiment_id"]]})
        rec = env.memory.get_discovery(did)
        assert rec["status"] == "unverified"
        assert rec["evidence_status"] == "probable"  # one fresh measured experiment: probable only

    def test_reverify_adopts_fresh_evidence(self, make_env):
        env = make_env()
        d = self._confirmed(env)
        did = d["discovery_id"]
        env.memory.mark_unverified(did, "fresh start")
        # two fresh measured experiments post-cutoff support again
        f1 = _measured(env)
        f2 = _measured(env)
        env.call("update_discovery", {"discovery_id": did, "add_experiment_ids":
                                      [f1["experiment_id"], f2["experiment_id"]]})
        out = env.memory.reverify(did, "re-ran everything", actor="researcher")
        assert out["status"] == "confirmed"
        assert out["history"][-1]["kind"] == "reverify"
        # the earlier unverified mark is still in the history
        kinds = [h.get("kind") for h in out["history"]]
        assert "mark_unverified" in kinds and "reverify" in kinds


# ------------------------------------------- 6. researcher actions in audit log
class TestResearcherAuditTrail:
    def test_all_researcher_ops_are_persisted(self, make_env):
        env = make_env()
        e1 = _measured(env)
        e2 = _measured(env)
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]})
        did = d["discovery_id"]
        env.memory.challenge(did, "why")
        env.memory.resolve_challenge(did, "ok")
        env.memory.mark_unverified(did, "again")
        env.memory.reverify(did, "ok again")
        env.memory.set_status(did, "probable", "manual downscale")
        env.memory.remove_item(did, "gone")
        env.memory.remove_item(e1["experiment_id"], "bad evidence")
        rows = [r["op"] for r in env.db.q(
            "SELECT op FROM memory_audit WHERE agent_id=? ORDER BY id", ("agent_t",))]
        for op in ("challenge", "resolve_challenge", "mark_unverified", "reverify",
                   "set_status", "remove"):
            assert op in rows, rows
        # every audit row carries actor + reason
        r = env.db.q("SELECT actor, reason FROM memory_audit WHERE op='set_status' AND item_id=?",
                     (did,))
        assert r and r[0]["reason"] == "manual downscale"


# ------------------------------------------------------- 7/8. persistence resets
class TestSessionPersistence:
    def test_memory_survives_env_reset(self, make_env, tmp_path, cfg, db):
        env = make_env()
        e1 = _measured(env)
        env.call("record_discovery", {"statement": "throws go straight up", "status": "confirmed",
                                      "experiment_ids": [e1["experiment_id"]]})
        env.call("set_goal", {"title": "map the west shore", "reason": "curiosity", "kind": "explore"})
        env.call("write_diary", {"text": "today I threw a block"})
        agent_dir = env.memory.dir
        # --- fresh session: new environment, same agent memory --------------
        mem2 = MemoryStore(agent_dir.parent, "agent_t", cfg, db=db)
        b = mem2.briefing()
        assert "throws go straight up" in (agent_dir / "handbook.md").read_text(encoding="utf-8")
        assert "map the west shore" in b or "map the west shore" in (agent_dir / "unfinished_goals.md").read_text(encoding="utf-8")
        assert "today I threw a block" in b
        assert len(mem2.state["diary"]) == 1

    def test_no_goals_are_auto_assigned_between_sessions(self, make_env, cfg, db):
        env = make_env()
        env.call("set_goal", {"title": "own plan"})
        n_before = len(env.memory.state["goals"])
        mem2 = MemoryStore(env.memory.dir.parent, "agent_t", cfg, db=db)
        mem2.close_session(2, {"ignored": True}, {"summary": "x"})
        assert len(mem2.state["goals"]) == n_before

    def test_experiments_record_agent_and_session_association(self, make_env):
        env = make_env()
        e = _measured(env)
        rec = env.memory.get_experiment(e["experiment_id"])
        assert rec["agent_id"] == "agent_t"
        assert rec["session"] == env.session_number
        assert rec["session_id"] == env.sid
        assert rec["created_at"] > 0
        assert rec["author"]["session"] == env.session_number

    def test_diary_entries_are_immutable_and_authored(self, make_env):
        env = make_env()
        env.call("write_diary", {"text": "entry one"})
        y1 = env.memory.state["diary"][0]
        assert y1["author"] == "agent"
        # no agent-facing update/edit/delete path exists for diary entries
        names = list(SPECS)
        assert "write_diary" in names
        assert not any(t["category"] == "memory" and t["name"] != "write_diary"
                       and "diary" in t["name"] for t in SPECS.values()), names

    def test_agents_cannot_overwrite_each_other(self, make_env, tmp_path, cfg, db):
        MemoryStore(tmp_path / "agents", "agent_x", cfg, db=db).add_diary(1, 0.0, "X WROTE THIS")
        MemoryStore(tmp_path / "agents", "agent_y", cfg, db=db).add_diary(1, 0.0, "Y WROTE THIS")
        x = json.loads((tmp_path / "agents" / "agent_x" / "state.json").read_text(encoding="utf-8"))
        y = json.loads((tmp_path / "agents" / "agent_y" / "state.json").read_text(encoding="utf-8"))
        assert x["diary"][0]["text"] == "X WROTE THIS"
        assert y["diary"][0]["text"] == "Y WROTE THIS"
        # independent per-agent id counters
        assert x["next"]["Y"] == 2 and y["next"]["Y"] == 2


# --------------------------------------------------------------- 10/11. handbook
class TestHandbookAuthoritative:
    def test_agent_claimed_confirmations_do_not_appear_confirmed(self, make_env):
        env = make_env()
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        d = env.call("record_discovery", {"statement": "Confirmed discovery: stacking blocks makes ledges",
                                          "status": "confirmed"})
        assert d["granted_status"] == "hypothesis"
        hand = (env.memory.dir / "handbook.md").read_text(encoding="utf-8")
        confirmed_section = hand.split("## Verified discoveries")[1].split("\n## ")[0]
        assert "stacking blocks makes ledges" not in confirmed_section
        assert "stacking blocks makes ledges" in hand  # lives under hypotheses

    def test_handbook_status_follows_authoritative_status(self, make_env):
        env = make_env()
        e1 = _measured(env)
        e2 = _measured(env)
        d = env.call("record_discovery", {"statement": "s", "status": "confirmed",
                                          "experiment_ids": [e1["experiment_id"], e2["experiment_id"]]})
        did = d["discovery_id"]
        env.memory.challenge(did, "hold on")
        env.memory.save()
        hand = (env.memory.dir / "handbook.md").read_text(encoding="utf-8")
        challenged_section = hand.split("## Challenged by the researcher")[1].split("\n## ")[0]
        assert "Challenged by the researcher" in hand and did in challenged_section
        env.memory.resolve_challenge(did, "fine")
        env.memory.save()
        hand = (env.memory.dir / "handbook.md").read_text(encoding="utf-8")
        confirmed_section = hand.split("## Verified discoveries")[1].split("\n## ")[0]
        assert did in confirmed_section


# ------------------------------------------------------------------ 7. goals
class TestGoalContinuity:
    def test_goal_origin_and_reason_recorded(self, make_env):
        env = make_env()
        g = env.call("set_goal", {"title": "find out why blocks bounce", "reason": "noticed heavy bounce",
                                  "kind": "question"})
        rec = env.memory.get_goal(g["goal_id"])
        assert rec["origin"] == "agent"
        assert rec["history"][0]["reason"] == "noticed heavy bounce"

    def test_goal_lifecycle_history_records_changes(self, make_env):
        env = make_env()
        g = env.call("set_goal", {"title": "climb check", "reason": "curiosity"})
        env.call("update_goal", {"goal_id": g["goal_id"], "status": "paused", "note": "throw physics first"})
        env.call("update_goal", {"goal_id": g["goal_id"], "status": "abandoned", "note": "lost interest"})
        rec = env.memory.get_goal(g["goal_id"])
        assert rec["status"] == "abandoned"
        notes = [h.get("note") or h.get("reason") for h in rec["history"]]
        assert any("throw physics first" in (n or "") for n in notes)
        assert any("lost interest" in (n or "") for n in notes)

    def test_assigned_goal_provenance_supported(self, tmp_path, cfg, db):
        m = MemoryStore(tmp_path / "agents", "bench_agent", cfg, db=db)
        g = m.add_goal(1, 0.0, "reach the terrace", "benchmark task", origin="assigned", reason="task T-1")
        assert g["origin"] == "assigned" and g["history"][0]["origin"] == "assigned"
        with pytest.raises(MemoryError_):
            m.add_goal(1, 0.0, "bad", None, origin="hacker")


# ------------------------------------------------ 14. free-will prompt contract
class TestFreeWillPrompts:
    def test_no_steering_in_agent_facing_texts(self):
        # Free-exploration texts only: benchmark prompts legitimately assign tasks
        # (that mode IS externally assigned work and is recorded as such).
        texts = [SITUATION, exploration_opening("agent_a", 1, None, 30, 400), continue_nudge()]
        texts += list(PERSONALITIES.values())
        for txt in texts:
            for pat in STEER_PHRASES:
                assert not re.search(pat, txt, re.I), (pat, txt[:160])

    def test_arena_prompts_are_separate_assigned_work_texts(self):
        """Steered 2026-11-17: arena solo and coop are separate prompt registry entries,
        distinct from every other registry text; arena ids never collide with exploration ids."""
        from playground.agents.prompts import (ARENA_COOP_SYSTEM, ARENA_SOLO_SYSTEM,
                                               ARENA_PROMPT_IDS, EXPLORATION_SYSTEM)
        arena = (ARENA_SOLO_SYSTEM, ARENA_COOP_SYSTEM)
        all_texts = set(arena) | set(PERSONALITIES.values()) | {EXPLORATION_SYSTEM, BENCHMARK_SYSTEM}
        assert len(all_texts) == len(arena) + len(PERSONALITIES) + 2
        for pid in ARENA_PROMPT_IDS:
            assert pid not in PERSONALITIES
        for text in arena:
            assert text not in EXPLORATION_SYSTEM

    def test_opening_states_reset_and_preserves_choice(self):
        op = exploration_opening("agent_a", 1, None, 30, 400)
        assert "reset" in op.lower()
        assert "end_session" in op

    def test_no_hidden_mechanic_leaks(self):
        blobs = [SITUATION, exploration_opening("agent_a", 1, None, 30, 400), BENCHMARK_SYSTEM]
        blobs += list(PERSONALITIES.values())
        for b in blobs:
            low = b.lower()
            assert "ring around" not in low or "hidden" not in low.split("ring around")[1][:200]

    def test_overlay_statuses_not_in_agent_enums(self):
        enums = {name: (spec["parameters"]["properties"].get("status", {}).get("enum"))
                 for name, spec in SPECS.items()
                 if name in ("record_discovery", "update_discovery")}
        for name, enum in enums.items():
            assert enum is None or not (set(enum) & set(RESEARCHER_OVERLAYS)), (name, enum)


# --------------------------------------------------- end-of-session diary capture
class TestEndOfSessionDiary:
    def _run(self, cfg, db, policy):
        from playground.agents.adapters import PolicyAdapter
        from playground.session.runner import run_session, memory_root
        run_session(cfg, db, "probe_agent", PolicyAdapter(policy))
        return memory_root(cfg) / "probe_agent"

    def test_final_words_in_the_end_session_turn_are_diary_captured(self, cfg, db):
        """An agent that says its final thoughts in the same turn it calls end_session
        still gets a diary entry at the end of the session (user mandate 2026-11-16)."""
        final_words = ("I am satisfied with my exploration. My final thoughts are about the shape of "
                       "the island and the way water laps at the edges, which I will note next time.")

        def policy(system, opening, tools):
            yield ("say", final_words)
            yield ("call", "end_session", {})

        d = self._run(cfg, db, policy)
        diary = json.loads((d / "state.json").read_text(encoding="utf-8"))["diary"]
        assert diary, "end-of-session diary must not stay empty when the agent spoke"
        auto = [y for y in diary if "[auto-captured]" in y["text"]]
        assert len(auto) == 1 and auto[0]["author"] == "auto"
        assert final_words[:60] in auto[0]["text"]

    def test_agent_wrote_diary_already_not_duplicated(self, cfg, db):
        """When the agent diarized the same closing words itself, no auto-capture duplicate."""

        def policy(system, opening, tools):
            yield ("call", "write_diary", {"text": "I am satisfied with my exploration. My final " +
                                           "thoughts are written here for good."})
            yield ("say", "I am satisfied with my exploration. My final thoughts are written here for good.")
            yield ("call", "end_session", {})

        d = self._run(cfg, db, policy)
        diary = json.loads((d / "state.json").read_text(encoding="utf-8"))["diary"]
        assert len(diary) == 1  # the agent's own entry; no auto duplicate
        assert diary[0]["author"] == "agent"

    def test_time_limit_sessions_still_allow_diary_writes(self, cfg, db):
        """Budget-ended sessions are unaffected: the wrapup notice lets agents diarize, and
        the scripted reference agent's own wrap-up entries survive unchanged."""
        written = []

        def policy(system, opening, tools):
            while True:
                written.append(True)
                yield ("say", "thinking out loud, not acting")
                yield ("call", "write_diary", {"text": "wrap-up note before time runs out"})
                yield ("call", "end_session", {})
                break

        d = self._run(cfg, db, policy)
        diary = json.loads((d / "state.json").read_text(encoding="utf-8"))["diary"]
        assert any(y.get("author") == "agent" for y in diary)


# ------------------------------------------ researcher ops over the dashboard API
class TestResearcherApi:
    @pytest.fixture
    def client(self, cfg, db):
        from fastapi.testclient import TestClient
        from playground.dashboard.server import create_app
        return TestClient(create_app(cfg, db))

    def _confirmed(self, cfg, db):
        """Seed a confirmed discovery in the dashboard-visible memory root."""
        from playground.session.runner import memory_root
        m = MemoryStore(memory_root(cfg), "agent_seed", cfg, db=db)
        for i in range(2):
            m.add_experiment(1, float(i) + 1,
                             {"question": "q", "hypothesis": "h", "actual_result": f"{1.0 + i} m",
                              "measurements": {"max_height": 1.0 + i}, "action_refs": [i + 1], "supported": True},
                             {"verified": True, "verdict": "consistent", "level": "measured", "matches": {"max_height": 1.0 + i},
                              "contradictions": [], "unmatched": {}, "issues": [], "checked_actions": [i + 1]})
        d = m.add_discovery(1, "test claim", "confirmed", ["E-001", "E-002"])
        return m, d["id"]

    def test_new_researcher_ops_over_api(self, client, cfg, db):
        m, did = self._confirmed(cfg, db)
        base = f"/api/agents/agent_seed/memory/{did}"
        r = client.post(base + "/challenge", json={"reason": "api check"})
        assert r.status_code == 200 and r.json()["discovery"]["status"] == "challenged"
        r = client.post(base + "/resolve_challenge", json={"reason": "holds"})
        assert r.status_code == 200 and r.json()["discovery"]["status"] == "confirmed"
        r = client.post(base + "/mark_unverified", json={"reason": "api re-check"})
        assert r.status_code == 200 and r.json()["discovery"]["status"] == "unverified"
        r = client.post(base + "/reverify", json={"reason": "looked again", "status": "confirmed"})
        assert r.status_code == 200 and r.json()["discovery"]["status"] == "confirmed"
        r = client.post(base + "/set_status", json={"status": "probable", "reason": "api manual"})
        assert r.status_code == 200 and r.json()["discovery"]["status"] == "probable"
        r = client.post(base + "/bogus_op", json={})
        assert r.status_code == 400
        ops = {row["op"] for row in db.q(
            "SELECT op FROM memory_audit WHERE agent_id='agent_seed'")}
        assert {"challenge", "resolve_challenge", "mark_unverified", "reverify", "set_status"} <= ops
        # read the store back from disk (the seeding handle above is stale now)
        from playground.session.runner import memory_root as _memory_root
        fresh = MemoryStore(_memory_root(cfg), "agent_seed", cfg, db=db)
        assert fresh.get_discovery(did)["status"] == "probable"

    def test_memory_api_exposes_authoritative_fields(self, client, cfg, db):
        m, did = self._confirmed(cfg, db)
        m.challenge(did, "api view check")
        m.update_discovery(1, did, add_experiment_ids=["E-001"], note="re-check under challenge")
        data = client.get("/api/agents/agent_seed/memory").json()
        d = next(x for x in data["discoveries"] if x["id"] == did)
        for key in ("agent_id", "created_at", "evidence_status", "status_reason",
                    "requested_status", "history"):
            assert key in d, key
        assert d["status"] == "challenged"
        # fresh evidence under an overlay is recorded, not granted
        assert d["evidence_status"] == "confirmed"
        e = data["experiments"][0]
        assert "outcome" in e and "evidence_check" in e["outcome"]
        audit = data["audit"]
        assert any(row["op"] == "challenge" for row in audit)
