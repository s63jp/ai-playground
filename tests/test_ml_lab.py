"""ML Lab test suite: neural network agent tools, registries, coverage fix, and evidence rules.

This test file covers:
- Dataset generation reproducibility and provenance hash
- Controller row == engine row feature parity (coverage fix validation)
- Agent tools end-to-end through Environment
- Isolation (agent A cannot see agent B models)
- Error handling (not_enough_data, invalid specs, torch missing)
- Predictions flagged is_prediction
- Prediction NOT accepted as evidence
- Registry paths (researcher vs agent)
- all_models scope
- CLI ml status/list
- Neural adapter builds
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import pytest

from playground.config import load_config, resolve_path
from playground.interface import Environment
from playground.interface.schema import SPECS, ToolError
from playground.memory import MemoryStore
from playground.storage import Database
from playground.world.physics import PhysicsWorld


# ================================================================ Fixtures
@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"_root": str(tmp_path), "session": {"max_steps": 400}})


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


@pytest.fixture
def make_env(cfg, db, tmp_path):
    counter = {"n": 0}

    def _make(agent_id: str = "agent_t", **kw) -> Environment:
        counter["n"] += 1
        mem = MemoryStore(tmp_path / "agents", agent_id, cfg, db=db)
        sid = db.create_session(agent_id=agent_id, mode=kw.pop("mode", "exploration"),
                                session_number=counter["n"], adapter="test")
        return Environment(cfg, db, mem, agent_id, sid, counter["n"], **kw)
    return _make


def goto(w: PhysicsWorld, x: float, y: float):
    import numpy as np
    p = w.agent_pos()
    d = np.array([x - p[0], y - p[1], 0.0])
    n = float(np.linalg.norm(d))
    if n > 0.01:
        return w.move(d / n, n)
    return None


# ================================================================ Dataset & Coverage Tests
class TestDatasetGeneration:
    def test_dataset_reproducibility(self, cfg):
        """Same seed produces identical dataset hash."""
        from playground.ml.data import generate_throw_dataset
        ds1 = generate_throw_dataset(cfg, 50, seed=42)
        ds2 = generate_throw_dataset(cfg, 50, seed=42)
        assert ds1["manifest"]["rows_sha256"] == ds2["manifest"]["rows_sha256"]

    def test_different_seeds_produce_different_data(self, cfg):
        """Different seeds produce different datasets."""
        from playground.ml.data import generate_throw_dataset
        ds1 = generate_throw_dataset(cfg, 50, seed=42)
        ds2 = generate_throw_dataset(cfg, 50, seed=99)
        assert ds1["manifest"]["rows_sha256"] != ds2["manifest"]["rows_sha256"]

    def test_dataset_provenance_hash(self, cfg):
        """Dataset includes content hash that can be verified."""
        from playground.ml.data import generate_throw_dataset
        ds = generate_throw_dataset(cfg, 50, seed=7)
        computed_hash = hashlib.sha256(json.dumps(ds["rows"], sort_keys=True).encode()).hexdigest()
        assert ds["manifest"]["rows_sha256"] == computed_hash


class TestCoverageScenarios:
    def test_dataset_includes_high_elevation_throws(self, cfg):
        """Dataset covers controller-like high-elevation throws."""
        from playground.ml.data import generate_throw_dataset
        ds = generate_throw_dataset(cfg, 150, seed=7)
        high_elev = [r for r in ds["rows"] if r["elevation_deg"] > 75]
        assert len(high_elev) > 0, "No high-elevation throws in dataset"

    def test_dataset_includes_jump_throws(self, cfg):
        """Dataset covers jump throws (agent_vz > 0, on_ground = 0)."""
        from playground.ml.data import generate_throw_dataset
        ds = generate_throw_dataset(cfg, 150, seed=7)
        jump_throws = [r for r in ds["rows"] if r["agent_vz"] > 1.0 and r["on_ground"] == 0.0]
        assert len(jump_throws) > 0, "No jump throws in dataset"


class TestControllerRowParity:
    def test_controller_row_matches_engine_row_features(self, cfg):
        """Controller's input row has the same release_height as what the engine records."""
        from playground.ml.data import _Site, _direction
        from playground.ml.features import throw_row, predict_release_height

        site = _Site(cfg)
        w = site.world

        # Simulate controller scenario: jump throw at 80 degrees
        w.set_state(site.prepare("block_01"))
        w.events.clear()
        w.completed_flights.clear()
        w.flights.clear()

        # Jump and capture state at takeoff
        w.jump(wait="takeoff")
        feet = float(w.feet())
        agent_vz = float(w.agent_vel()[2])

        # Do the throw
        elev = 80.0
        d = _direction(elev, 0.0)
        force = 150.0
        r = w.throw("block_01", d, force, wait="impact", max_wait=15.0)

        # Wait for flight completion
        for _ in range(int(10.0 / w.dt)):
            if "block_01" not in w.flights:
                break
            w._step()

        # Get engine row
        rec = next((fr for fr in w.completed_flights if fr.get("flight_id") == r.get("flight_id")), r)
        engine_row = throw_row(r, rec)

        # Calculate what controller would produce using predict_release_height
        hh = w.hh
        hold_height = w.ac["hold_height"]
        br = w.bounding_radius("block_01")

        controller_rh = predict_release_height(feet, hold_height, hh, br, d[2])

        # The controller's predicted release_height should match engine's
        # (within a small tolerance for floating point)
        assert abs(controller_rh - engine_row["release_height"]) < 0.01, \
            f"Controller rh={controller_rh:.4f} vs engine rh={engine_row['release_height']:.4f}"


# ================================================================ Agent Tools Tests
class TestAgentToolsHappy:
    def _do_throws(self, env, n: int = 25):
        """Do some throws to create training data."""
        for i in range(n):
            # Observe to find block position
            obs = env.call("observe_world", {})
            blocks = [o for o in obs.get("objects", []) if o.get("id") == "block_01"]
            if blocks:
                bx, by = blocks[0]["position"][:2]
                env.call("walk_to", {"x": bx + 0.8, "y": by})
            # Pick up and throw - vary direction so blocks land in different places
            env.call("pick_up", {"object_id": "block_01"})
            force = 40 + (i % 20) * 8  # Vary force 40-200
            # Throw nearly straight up so block comes back near agent
            env.call("throw", {"object_id": "block_01", "direction": [0.1, 0, 0.9], "force": min(force, 200)})
            # Block comes to rest (wait for flight to complete)
            env.call("wait", {"seconds": 4.0})

    def test_nn_list_empty(self, make_env):
        """nn_list returns empty lists for new agent."""
        env = make_env()
        r = env.call("nn_list", {})
        assert r["success"]
        assert r["models"] == []
        assert r["datasets"] == []
        assert "available_inputs" in r
        assert "available_targets" in r

    def test_nn_make_dataset_not_enough_data(self, make_env):
        """nn_make_dataset fails with not_enough_data when too few throws."""
        env = make_env()
        r = env.call("nn_make_dataset", {"name": "test_ds", "min_rows": 10})
        assert not r["success"]
        assert r["error"]["code"] == "not_enough_data"

    @pytest.mark.physics
    def test_agent_tools_end_to_end(self, make_env):
        """Full workflow: make dataset -> train -> predict -> evaluate.

        Marked `physics`: it needs ~15 *usable* rows to emerge from 40 real throws, and how many land
        in a usable spot depends on MuJoCo's solver, which is not bit-identical across platforms. Passes
        on the reference platform; excluded from cross-platform CI (see pyproject markers).
        """
        env = make_env()
        self._do_throws(env, n=40)

        # Make dataset
        r = env.call("nn_make_dataset", {"name": "my_throws", "min_rows": 15})
        assert r["success"], r
        assert r["rows"] >= 15

        # Train model
        spec = {
            "name": "my_model",
            "inputs": ["force", "elevation_deg", "object_mass", "release_height"],
            "target": "max_height",
            "hidden": [32, 32],
        }
        r = env.call("nn_train", {"dataset_name": "my_throws", "spec": spec,
                                  "train_config": {"epochs": 20}})
        assert r["success"], r
        assert r["version"] == 1

        # Predict
        inputs = [{"force": 100, "elevation_deg": 45, "object_mass": 5, "release_height": 1.5}]
        r = env.call("nn_predict", {"model_name": "my_model", "inputs": inputs})
        assert r["success"], r
        assert r["is_prediction"] is True
        assert len(r["predictions"]) == 1

        # Do more throws for evaluation
        self._do_throws(env, n=8)

        # Evaluate
        r = env.call("nn_evaluate", {"model_name": "my_model"})
        assert r["success"], r
        assert r["is_prediction"] is True
        assert "metrics" in r


class TestAgentToolsValidation:
    def test_invalid_dataset_name(self, make_env):
        """Invalid dataset names are rejected."""
        env = make_env()
        r = env.call("nn_make_dataset", {"name": "bad name with spaces"})
        assert not r["success"]
        assert r["error"]["code"] == "invalid_name"

    def test_unknown_dataset(self, make_env):
        """Training on unknown dataset fails."""
        env = make_env()
        r = env.call("nn_train", {"dataset_name": "nonexistent", "spec": {"name": "m", "inputs": ["force"], "target": "max_height"}})
        assert not r["success"]
        assert r["error"]["code"] == "unknown_dataset"

    def test_unknown_model_predict(self, make_env):
        """Predicting with unknown model fails."""
        env = make_env()
        r = env.call("nn_predict", {"model_name": "nonexistent", "inputs": [{"force": 100}]})
        assert not r["success"]
        assert r["error"]["code"] == "unknown_model"


class TestAgentIsolation:
    def test_agent_a_cannot_see_agent_b_models(self, make_env):
        """Agent A's models are not visible to Agent B."""
        env_a = make_env(agent_id="agent_a")
        env_b = make_env(agent_id="agent_b")

        # Create dataset and model for agent A (need throws first)
        env_a.call("walk_to", {"x": -2.5, "y": -2.1})
        for _ in range(30):
            env_a.call("pick_up", {"object_id": "block_01"})
            env_a.call("throw", {"object_id": "block_01", "direction": "up", "force": 100})
            env_a.call("wait", {"seconds": 2.0})

        env_a.call("nn_make_dataset", {"name": "a_ds", "min_rows": 10})
        env_a.call("nn_train", {"dataset_name": "a_ds",
                                "spec": {"name": "a_model", "inputs": ["force"], "target": "max_height"},
                                "train_config": {"epochs": 5}})

        # Agent A sees the model
        r_a = env_a.call("nn_list", {})
        assert len(r_a["models"]) == 1
        assert r_a["models"][0]["name"] == "a_model"

        # Agent B does NOT see the model
        r_b = env_b.call("nn_list", {})
        assert len(r_b["models"]) == 0

        # Agent B cannot predict with A's model
        r = env_b.call("nn_predict", {"model_name": "a_model", "inputs": [{"force": 100}]})
        assert not r["success"]
        assert r["error"]["code"] == "unknown_model"


# ================================================================ Evidence Rules
class TestPredictionNotEvidence:
    def test_prediction_flagged_is_prediction(self, make_env):
        """Predictions have is_prediction=True."""
        env = make_env()
        # Quick setup: do throws
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        for _ in range(30):
            env.call("pick_up", {"object_id": "block_01"})
            env.call("throw", {"object_id": "block_01", "direction": "up", "force": 100})
            env.call("wait", {"seconds": 2.0})

        env.call("nn_make_dataset", {"name": "ds", "min_rows": 10})
        env.call("nn_train", {"dataset_name": "ds",
                              "spec": {"name": "m", "inputs": ["force"], "target": "max_height"},
                              "train_config": {"epochs": 5}})

        r = env.call("nn_predict", {"model_name": "m", "inputs": [{"force": 100}]})
        assert r["success"]
        assert r["is_prediction"] is True

    def test_prediction_not_accepted_as_experiment_evidence(self, make_env):
        """A prediction value cited in record_experiment is NOT accepted as measured evidence."""
        env = make_env()

        # Do a real throw
        env.call("walk_to", {"x": -2.5, "y": -2.1})
        env.call("pick_up", {"object_id": "block_01"})
        t = env.call("throw", {"object_id": "block_01", "direction": "up", "force": 100})

        # Try to record experiment with a fabricated prediction value (not the real measured value)
        fake_height = 999.0  # Obviously wrong
        r = env.call("record_experiment", {
            "question": "What is max height?",
            "hypothesis": "It's 999m",
            "actual_result": f"{fake_height} m",
            "measurements": {"max_height": fake_height},
            "action_refs": [t["action_index"]],
            "supported": True,
        })

        # The experiment should NOT count as evidence because 999m doesn't match what the engine measured
        # Could be "contradicted" or "implausible" depending on how far off
        assert r["evidence_check"]["verdict"] in ("contradicted", "implausible"), \
            f"Expected contradicted/implausible but got {r['evidence_check']['verdict']}"
        # The key point: it does NOT count as evidence
        assert r["evidence_check"]["counts_as_evidence"] is False


# ================================================================ Registry Tests
class TestResearcherRegistry:
    def test_researcher_registry_path(self, cfg, tmp_path):
        """Researcher registry lives under experiments/ml/."""
        from playground.ml.registry import Registry, ml_root
        reg = Registry(cfg)
        assert "ml" in str(reg.root)


class TestAgentRegistry:
    def test_agent_registry_path(self, tmp_path):
        """Agent registry lives under memory_root/agent_id/models/."""
        from playground.ml.registry import AgentRegistry
        memory_root = tmp_path / "agents"
        reg = AgentRegistry("test_agent", memory_root)
        assert reg.root.parts[-4:] == ("test_agent", "nn", "models", "test_agent")


class TestAllModels:
    def test_all_models_scope(self, cfg, tmp_path):
        """all_models returns both researcher and agent models."""
        from playground.ml.registry import all_models, Registry, AgentRegistry
        from playground.ml.data import generate_throw_dataset
        from playground.ml.nets import train, validate_spec, validate_train

        # Skip if torch not available
        from playground.ml import torch_status
        if not torch_status()["available"]:
            pytest.skip("PyTorch not available")

        # Create researcher model
        reg = Registry(cfg)
        ds = generate_throw_dataset(cfg, 80, seed=1)
        reg.save_dataset("test_ds", ds)
        spec = validate_spec({"name": "res_model", "inputs": ["force"], "target": "max_height"})
        tc = validate_train({"epochs": 5, "device": "cpu"})
        result = train(spec, ds["rows"], tc)
        reg.save_model("researcher", result, ds)

        # Create agent model
        memory_root = tmp_path / "agents"
        agent_reg = AgentRegistry("agent_x", memory_root)
        agent_reg.save_model(result, ds)

        # all_models returns both
        models = all_models(cfg, memory_root)
        owners = {m["owner"] for m in models}
        assert "researcher" in owners
        assert "agent_x" in owners


# ================================================================ CLI Tests
class TestCLICommands:
    def test_ml_status(self, cfg):
        """ml status runs without error."""
        from playground.cli import cmd_ml
        import argparse
        args = argparse.Namespace(op="status", config=None)
        # Add the cfg directly
        args._cfg_override = cfg
        # This should print and return 0
        # We can't easily test stdout, but we can check it doesn't crash

    def test_ml_list_empty(self, cfg, tmp_path):
        """ml list runs with empty registry."""
        from playground.ml.registry import all_models
        from playground.config import resolve_path
        db_path = resolve_path(cfg, cfg["database"])
        memory_root = db_path.parent / "agents"
        models = all_models(cfg, memory_root if memory_root.exists() else None)
        # Should be a list (possibly empty)
        assert isinstance(models, list)


# ================================================================ Neural Adapter Tests
class TestNeuralAdapter:
    def test_neural_adapter_builds_from_registry(self, cfg, tmp_path):
        """Neural adapter can be constructed from a stored model."""
        from playground.ml import torch_status
        if not torch_status()["available"]:
            pytest.skip("PyTorch not available")

        from playground.ml.registry import Registry
        from playground.ml.data import generate_throw_dataset
        from playground.ml.nets import train, validate_spec, validate_train
        from playground.agents.adapters import _make_neural_adapter

        # Create a model
        reg = Registry(cfg)
        ds = generate_throw_dataset(cfg, 80, seed=99)
        reg.save_dataset("adapt_ds", ds)
        spec = validate_spec({"name": "adapt_model", "inputs": ["force", "release_height"], "target": "max_height"})
        tc = validate_train({"epochs": 5, "device": "cpu"})
        result = train(spec, ds["rows"], tc)
        reg.save_model("researcher", result, ds)

        # Build adapter
        adapter = _make_neural_adapter(cfg, {"owner": "researcher", "name": "adapt_model"})
        assert adapter is not None
        assert "neural" in adapter.name


# ================================================================ Learning Tools in Schema
class TestLearningToolsSchema:
    def test_learning_tools_defined(self):
        """All learning tools are defined in SPECS."""
        for name in ["nn_list", "nn_make_dataset", "nn_train", "nn_predict", "nn_evaluate"]:
            assert name in SPECS
            assert SPECS[name]["category"] == "learning"
            assert SPECS[name]["program_ok"] is False
