"""HVE coverage for the persistent model leaderboard (Nov 2026 feature batch)."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from playground.dashboard import create_app
from playground.ml import leaderboard as lb
from playground.ml.registry import Registry


@pytest.fixture
def client(cfg, db):
    return TestClient(create_app(cfg, db))


class TestLeaderboardStore:
    def test_record_appends_and_persists(self, cfg):
        lb.record(cfg, {"source": "train", "owner": "researcher", "name": "m1",
                        "test_r2": 0.8, "test_rmse": 0.4, "beats_linear": True, "ts": 100.0})
        lb.record(cfg, {"source": "train", "owner": "researcher", "name": "m1",
                        "test_r2": 0.9, "test_rmse": 0.3, "beats_linear": True, "ts": 200.0})
        saved = json.loads((lb._path(cfg)).read_text(encoding="utf-8"))
        assert [e["test_r2"] for e in saved] == [0.8, 0.9]

    def test_by_model_aggregates_trend_and_compare(self, cfg):
        lb.record(cfg, {"source": "train", "owner": "researcher", "name": "m1",
                        "test_r2": 0.8, "test_rmse": 0.4, "beats_linear": True})
        lb.record(cfg, {"source": "train", "owner": "researcher", "name": "m1",
                        "test_r2": 0.9, "test_rmse": 0.3, "beats_linear": True})
        lb.record(cfg, {"source": "compare", "owner": "researcher", "name": "m1",
                        "actions_mean": 51.0, "distinct_tools_mean": 2.0})
        b = lb.by_model(cfg)
        assert len(b) == 1
        m = b[0]
        assert m["best_test_r2"] == 0.9 and m["best_test_rmse"] == 0.3
        assert m["train_count"] == 2 and m["compare_count"] == 1
        assert m["trend_r2"] == pytest.approx(0.1)     # improving trend
        assert m["latest_compare"]["actions_mean"] == 51.0
        assert m["beats_linear"] == 2

    def test_entries_filter_by_source(self, cfg):
        lb.record(cfg, {"source": "train", "owner": "researcher", "name": "m1", "test_r2": 0.5})
        lb.record(cfg, {"source": "compare", "owner": "researcher", "name": "m_other", "actions_mean": 1})
        assert [e["name"] for e in lb.entries(cfg, source="train")] == ["m1"]
        assert [e["name"] for e in lb.entries(cfg, source="compare")] == ["m_other"]


class TestLeaderboardApi:
    def test_leaderboard_endpoint_shape(self, client):
        r = client.get("/api/ml/leaderboard")
        assert r.status_code == 200
        body = r.json()
        assert "entries" in body and "by_model" in body

    def test_training_job_records_a_leaderboard_entry(self, cfg, db, client):
        """A finished train job must leave one persistent train entry for its model."""
        from playground.ml.data import generate_throw_dataset
        reg = Registry(cfg)
        reg.save_dataset("lb_throws", generate_throw_dataset(cfg, 60, seed=2))
        r = client.post("/api/ml/train", json={
            "dataset": "lb_throws",
            "spec": {"name": "lb_model", "inputs": ["force", "feet_height"], "target": "max_height",
                     "hidden": [16]},
            "train": {"epochs": 10, "device": "cpu"},
            "scope": "researcher", "owner": "researcher"})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        deadline = time.time() + 120
        while time.time() < deadline:
            j = client.get(f"/api/ml/jobs/{job_id}").json()
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert j["status"] == "done", j.get("error")
        deadline = time.time() + 10
        entries = []
        while time.time() < deadline:
            entries = client.get("/api/ml/leaderboard").json()["entries"]
            if entries:
                break
            time.sleep(0.1)
        assert entries, "training did not record a leaderboard entry"
        mine = [e for e in entries if e["name"] == "lb_model" and e["source"] == "train"]
        assert mine and mine[0]["dataset"] == "lb_throws"
        assert mine[0]["test_r2"] is not None
