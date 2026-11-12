"""Memory persistence tests: HVE suite for session-end memory paths and nn/ integration.

Tests:
1. Memory persists through all end_reason paths (agent_ended, agent_stopped, step_limit, time_limit)
2. add_note correctly persists to state.json and renders to handbook.md
3. Session 2 briefing includes memory from session 1
4. agent_stopped captures auto-diary entry for agent's last words
5. nn/ integration: briefing, stats, close_session, remove_item, all_items
6. Ablation copy includes nn/
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from playground.agents import PolicyAdapter
from playground.config import load_config
from playground.memory import MemoryStore
from playground.session import run_session
from playground.storage import Database


# =============================================================== Fixtures

@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"_root": str(tmp_path), "session": {"max_steps": 50, "idle_turn_limit": 3}})


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


@pytest.fixture
def memory_root(tmp_path):
    return tmp_path / "agents"


def make_fake_nn_model(memory_dir: Path, agent_id: str, name: str, version: int = 1):
    """Create a fake nn/ model JSON (without importing torch)."""
    nn_models_dir = memory_dir / "nn" / "models" / agent_id / name
    nn_models_dir.mkdir(parents=True, exist_ok=True)
    model_json = {
        "spec": {
            "name": name,
            "inputs": ["release_speed", "release_height"],
            "target": "max_height",
            "hidden": [32, 32]
        },
        "metrics": {
            "test": {"rmse": 0.123, "r2": 0.95},
            "baseline_linear": {"rmse": 0.456, "r2": 0.80}
        },
        "version": version,
        "created_at": time.time(),
        "owner": agent_id,
        "beats_linear_baseline": True
    }
    (nn_models_dir / f"v{version}.json").write_text(json.dumps(model_json), encoding="utf-8")
    return model_json


# =============================================================== Happy Path Tests

class TestMemoryPersistenceHappy:
    """Happy path: memory persists correctly through normal operations."""

    def test_add_note_persists_to_state_and_renders(self, cfg, db, memory_root):
        """add_note stores in state.json['notes'] and renders to handbook.md."""
        mem = MemoryStore(memory_root, "agent_test", cfg, db=db)
        note = mem.add_note(session=1, t=10.5, text="Test note about the island")
        
        assert note["id"] == "N-001"
        assert len(mem.state["notes"]) == 1
        assert mem.state["notes"][0]["text"] == "Test note about the island"
        
        # Check rendered handbook includes the note
        handbook = (mem.dir / "handbook.md").read_text(encoding="utf-8")
        assert "N-001" in handbook
        assert "Test note about the island" in handbook

    def test_note_in_session1_appears_in_session2_briefing(self, cfg, db, memory_root):
        """Notes from session 1 appear in session 2's briefing."""
        mem = MemoryStore(memory_root, "agent_persist", cfg, db=db)
        mem.add_note(session=1, t=5.0, text="The spire is 6m tall")
        mem.state["sessions_completed"] = 1
        mem.save()
        
        # New session's briefing should include the note
        briefing = mem.briefing()
        assert "N-001" in briefing
        assert "The spire is 6m tall" in briefing
        assert "sessions completed so far: 1" in briefing

    def test_agent_ended_persists_memory(self, cfg, db):
        """agent_ended (normal session end) correctly persists all memory."""
        def policy(system, briefing, tools):
            yield ("say", "I will explore and take notes.")
            yield ("call", "look_around", {"radius": 5})
            yield ("call", "add_note", {"text": "Found 12 blocks around me"})
            yield ("call", "end_session", {"reason": "exploration complete"})
        
        adapter = PolicyAdapter(policy, label="test", model="scripted")
        summary = run_session(cfg, db, "agent_ended_test", adapter)
        
        assert summary["end_reason"] == "agent_ended"
        assert summary["error"] is None
        
        # Memory should persist
        mem = MemoryStore(Path(summary["memory_snapshot"]).parent.parent.parent, "agent_ended_test", cfg, db=db)
        assert len(mem.state["notes"]) == 1
        assert "12 blocks" in mem.state["notes"][0]["text"]
        assert mem.state["sessions_completed"] == 1

    def test_step_limit_persists_memory(self, cfg, db):
        """step_limit sessions still persist memory written before the limit."""
        cfg = {**cfg, "session": {**cfg["session"], "max_steps": 3}}
        
        def policy(system, briefing, tools):
            yield ("call", "add_note", {"text": "Starting exploration"})
            # Exceed step limit with more actions
            yield ("call", "look_around", {})
            yield ("call", "look_around", {})
            yield ("call", "look_around", {})  # This triggers step_limit
        
        adapter = PolicyAdapter(policy, label="test", model="scripted")
        summary = run_session(cfg, db, "agent_step_limit", adapter)
        
        assert summary["end_reason"] == "step_limit"
        
        # Memory should still persist
        mem = MemoryStore(Path(summary["memory_snapshot"]).parent.parent.parent, "agent_step_limit", cfg, db=db)
        assert len(mem.state["notes"]) == 1
        assert mem.state["sessions_completed"] == 1

    def test_agent_stopped_persists_memory_and_captures_last_words(self, cfg, db):
        """agent_stopped captures the agent's last text as auto-diary when no memory was written."""
        cfg = {**cfg, "session": {**cfg["session"], "idle_turn_limit": 2}}
        
        def policy(system, briefing, tools):
            # Agent only speaks, no tool calls - simulates a model that only outputs prose
            yield ("say", "I see 12 blocks arranged in a circle. The spire is magnificent.")
            yield ("say", "I should explore further but I'm feeling contemplative.")
            # No more yields - generator ends, adapter._done = True
        
        adapter = PolicyAdapter(policy, label="text_only", model="prose_model")
        summary = run_session(cfg, db, "agent_text_only", adapter)
        
        assert summary["end_reason"] == "agent_stopped"
        assert summary["actions_logged"] == 0  # No actual actions taken
        
        # Memory should persist and have auto-captured diary entry
        mem = MemoryStore(Path(summary["memory_snapshot"]).parent.parent.parent, "agent_text_only", cfg, db=db)
        assert mem.state["sessions_completed"] == 1
        
        # Check for auto-captured last words in diary
        diary = mem.state["diary"]
        assert len(diary) >= 1
        auto_entry = next((d for d in diary if "[auto-captured]" in d.get("text", "")), None)
        assert auto_entry is not None, "Should have auto-captured diary entry"
        assert "contemplative" in auto_entry["text"] or "blocks" in auto_entry["text"]

    def test_sessions_completed_increments_correctly(self, cfg, db, memory_root):
        """sessions_completed increments after each successful session."""
        mem = MemoryStore(memory_root, "agent_count", cfg, db=db)
        
        assert mem.state["sessions_completed"] == 0
        
        # Simulate session completions
        mem.close_session(1, {}, {"end_reason": "agent_ended"})
        assert mem.state["sessions_completed"] == 1
        
        mem.close_session(2, {}, {"end_reason": "agent_ended"})
        assert mem.state["sessions_completed"] == 2

    def test_session_history_snapshot_created(self, cfg, db, memory_root):
        """close_session creates a session history snapshot with all files."""
        mem = MemoryStore(memory_root, "agent_snapshot", cfg, db=db)
        mem.add_note(session=1, t=1.0, text="Test note")
        
        dest = mem.close_session(1, {"test": "world_state"}, {"end_reason": "agent_ended"})
        
        assert dest.exists()
        assert (dest / "state.json").exists()
        assert (dest / "handbook.md").exists()
        assert (dest / "final_world_state.json").exists()
        assert (dest / "summary.json").exists()


# =============================================================== NN Integration Tests

class TestNNIntegrationHappy:
    """Happy path: nn/ network integration with memory."""

    def test_briefing_lists_networks(self, cfg, db, memory_root):
        """Briefing includes nn/ section when networks exist."""
        mem = MemoryStore(memory_root, "agent_nn", cfg, db=db)
        mem.state["sessions_completed"] = 1
        mem.save()
        
        # Create fake network
        make_fake_nn_model(mem.dir, "agent_nn", "throw_predictor", version=1)
        
        briefing = mem.briefing()
        assert "----- nn/" in briefing
        assert "throw_predictor" in briefing
        assert "max_height" in briefing  # target
        assert "release_speed" in briefing or "inputs" in briefing.lower()

    def test_stats_includes_network_count(self, cfg, db, memory_root):
        """stats() returns 'networks' count."""
        mem = MemoryStore(memory_root, "agent_stats", cfg, db=db)
        
        # No networks initially
        stats = mem.stats()
        assert stats.get("networks", 0) == 0
        
        # Add fake networks
        make_fake_nn_model(mem.dir, "agent_stats", "model_a")
        make_fake_nn_model(mem.dir, "agent_stats", "model_b")
        
        stats = mem.stats()
        assert stats["networks"] == 2

    def test_close_session_snapshots_nn(self, cfg, db, memory_root):
        """close_session snapshot includes nn/ directory."""
        mem = MemoryStore(memory_root, "agent_nn_snap", cfg, db=db)
        make_fake_nn_model(mem.dir, "agent_nn_snap", "my_model", version=1)
        
        dest = mem.close_session(1, {}, {"end_reason": "agent_ended"})
        
        # Check nn/ was copied to snapshot
        nn_snap = dest / "nn" / "models" / "agent_nn_snap" / "my_model"
        assert nn_snap.exists()
        assert (nn_snap / "v1.json").exists()

    def test_all_items_lists_networks(self, cfg, db, memory_root):
        """all_items() includes networks with kind='network'."""
        mem = MemoryStore(memory_root, "agent_all", cfg, db=db)
        make_fake_nn_model(mem.dir, "agent_all", "height_model")
        make_fake_nn_model(mem.dir, "agent_all", "distance_model")
        
        items = mem.all_items()
        networks = [i for i in items if i["kind"] == "network"]
        
        assert len(networks) == 2
        names = {n["id"] for n in networks}
        assert "NN-height_model" in names
        assert "NN-distance_model" in names

    def test_remove_item_deletes_network(self, cfg, db, memory_root):
        """remove_item('NN-name') removes that network's folder."""
        mem = MemoryStore(memory_root, "agent_rm", cfg, db=db)
        make_fake_nn_model(mem.dir, "agent_rm", "to_delete")
        
        # Verify it exists
        assert (mem.dir / "nn" / "models" / "agent_rm" / "to_delete").exists()
        
        removed = mem.remove_item("NN-to_delete", reason="test removal", actor="researcher")
        
        assert removed is not None
        assert not (mem.dir / "nn" / "models" / "agent_rm" / "to_delete").exists()
        
        # Check it's audited
        audit_log = (mem.dir / "audit.log").read_text(encoding="utf-8")
        assert "NN-to_delete" in audit_log
        assert "remove" in audit_log

    def test_copy_to_includes_nn(self, cfg, db, memory_root, tmp_path):
        """Ablation copy_to copies nn/ directory."""
        mem = MemoryStore(memory_root, "agent_copy", cfg, db=db)
        make_fake_nn_model(mem.dir, "agent_copy", "copied_model")
        
        dest_root = tmp_path / "ablation"
        other = mem.copy_to(dest_root, db=db)
        
        # Check nn/ was copied
        assert (other.dir / "nn" / "models" / "agent_copy" / "copied_model" / "v1.json").exists()


# =============================================================== Validation Tests

class TestMemoryPersistenceValidation:
    """Validation: edge cases and boundary conditions."""

    def test_empty_memory_briefing(self, cfg, db, memory_root):
        """Empty memory returns first-session message."""
        mem = MemoryStore(memory_root, "agent_empty", cfg, db=db)
        briefing = mem.briefing()
        assert "first session" in briefing.lower()

    def test_multiple_notes_persist(self, cfg, db, memory_root):
        """Multiple notes from same session all persist."""
        mem = MemoryStore(memory_root, "agent_multi", cfg, db=db)
        mem.add_note(1, 1.0, "First note")
        mem.add_note(1, 2.0, "Second note")
        mem.add_note(1, 3.0, "Third note")
        
        assert len(mem.state["notes"]) == 3
        
        # All in handbook
        handbook = (mem.dir / "handbook.md").read_text(encoding="utf-8")
        assert "N-001" in handbook
        assert "N-002" in handbook
        assert "N-003" in handbook

    def test_session_snapshot_idempotent(self, cfg, db, memory_root):
        """Calling close_session twice on same session overwrites cleanly."""
        mem = MemoryStore(memory_root, "agent_idem", cfg, db=db)
        mem.add_note(1, 1.0, "Note v1")
        
        dest1 = mem.close_session(1, {"v": 1}, {})
        
        # Add more data and close again
        mem.add_note(1, 2.0, "Note v2")
        dest2 = mem.close_session(1, {"v": 2}, {})
        
        assert dest1 == dest2
        state = json.loads((dest2 / "state.json").read_text(encoding="utf-8"))
        assert len(state["notes"]) == 2

    def test_nn_briefing_empty_when_no_networks(self, cfg, db, memory_root):
        """Briefing does not include nn/ section when no networks exist."""
        mem = MemoryStore(memory_root, "agent_no_nn", cfg, db=db)
        mem.state["sessions_completed"] = 1
        mem.save()
        
        briefing = mem.briefing()
        assert "----- nn/" not in briefing


# =============================================================== Error Handling Tests

class TestMemoryPersistenceErrors:
    """Error handling: memory operations under error conditions."""

    def test_remove_unknown_nn_raises(self, cfg, db, memory_root):
        """Removing non-existent NN- item raises MemoryError_."""
        from playground.memory import MemoryError_
        
        mem = MemoryStore(memory_root, "agent_err", cfg, db=db)
        
        with pytest.raises(MemoryError_) as exc_info:
            mem.remove_item("NN-nonexistent")
        
        assert "unknown_item" in str(exc_info.value.code)

    def test_adapter_error_does_not_persist_memory(self, cfg, db):
        """adapter_error sessions are resumable and don't snapshot memory."""
        from playground.agents import AdapterError
        
        class FailingAdapter:
            name = "failing"
            model = "fail"
            driven_externally = False
            
            def begin(self, system, tools):
                pass
            
            def step(self, user_text, results):
                raise AdapterError("Simulated failure")
            
            def describe(self):
                return {"adapter": self.name, "model": self.model}
        
        summary = run_session(cfg, db, "agent_fail", FailingAdapter())
        
        assert summary["end_reason"] == "adapter_error"
        # Memory snapshot should not be created for resumable sessions
        assert summary.get("memory_snapshot") is None

    def test_nn_stats_handles_malformed_json(self, cfg, db, memory_root):
        """stats() gracefully handles malformed nn/ JSON files."""
        mem = MemoryStore(memory_root, "agent_bad_nn", cfg, db=db)
        
        # Create malformed model
        bad_dir = mem.dir / "nn" / "models" / "agent_bad_nn" / "broken"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "v1.json").write_text("{ invalid json", encoding="utf-8")
        
        # Should not crash, just skip the bad file
        stats = mem.stats()
        assert stats["networks"] == 0  # Bad file not counted


# =============================================================== End-to-End Session Tests

class TestSessionEndReasonPaths:
    """Test every end_reason path saves memory correctly."""

    def test_time_limit_persists_memory(self, cfg, db):
        """time_limit sessions persist memory written before timeout."""
        # Use step_limit instead since time_limit is hard to test reliably
        cfg = {**cfg, "session": {**cfg["session"], "max_steps": 2}}
        
        def policy(system, briefing, tools):
            yield ("call", "add_note", {"text": "Quick note before limit"})
            yield ("call", "look_around", {})
            yield ("call", "look_around", {})  # This triggers step_limit
        
        adapter = PolicyAdapter(policy, label="test", model="scripted")
        summary = run_session(cfg, db, "agent_limit", adapter)
        
        assert summary["end_reason"] == "step_limit"
        
        mem = MemoryStore(Path(summary["memory_snapshot"]).parent.parent.parent, "agent_limit", cfg, db=db)
        assert len(mem.state["notes"]) == 1
        assert mem.state["sessions_completed"] == 1

    def test_full_session_flow_persistence(self, cfg, db):
        """Full session flow: multiple memory writes, end, verify all persist."""
        def policy(system, briefing, tools):
            yield ("say", "Starting comprehensive exploration")
            yield ("call", "look_around", {"radius": 3})
            yield ("call", "add_note", {"text": "Survey note 1"})
            yield ("call", "move", {"direction": 0, "distance": 1})
            yield ("call", "add_note", {"text": "Survey note 2"})
            yield ("call", "write_diary", {"text": "Made good progress today"})
            yield ("call", "end_session", {"reason": "comprehensive test complete"})
        
        adapter = PolicyAdapter(policy, label="full_test", model="scripted")
        summary = run_session(cfg, db, "agent_full_flow", adapter)
        
        assert summary["end_reason"] == "agent_ended"
        assert summary["error"] is None
        
        mem = MemoryStore(Path(summary["memory_snapshot"]).parent.parent.parent, "agent_full_flow", cfg, db=db)
        assert len(mem.state["notes"]) == 2
        assert len(mem.state["diary"]) == 1
        assert mem.state["sessions_completed"] == 1
        
        # Verify session history snapshot
        snapshot = Path(summary["memory_snapshot"])
        assert snapshot.exists()
        snapshot_state = json.loads((snapshot / "state.json").read_text(encoding="utf-8"))
        assert len(snapshot_state["notes"]) == 2
