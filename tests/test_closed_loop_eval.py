"""HVE coverage for the closed-loop eval widget (network vs scripted baseline, one click).

The network runs AS the agent's brain through the real benchmark pipeline and the same task runs with
the scripted reference policy under the same conditions; both are judged by the task's own independent
verifier. Torch required (skipped otherwise — same gate as the NN-lab suite).
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from playground.dashboard import create_app
from playground.ml import torch_status

torch_ok = torch_status()["available"]
needs_torch = pytest.mark.skipif(not torch_ok, reason="PyTorch not available")


@pytest.fixture
def client(cfg, db):
    return TestClient(create_app(cfg, db))


@needs_torch
class TestClosedLoopCompareEndpoint:
    def test_both_sides_run_and_verdict_compares_scores(self, cfg, db, client):
        from test_nn_lab_stage1 import _small_model
        from playground.ml.registry import registry_for
        reg = registry_for(cfg, "researcher", "researcher")
        info, _ds = _small_model(cfg, reg, "clv_net", "researcher", n=140, epochs=60)

        r = client.post("/api/ml/closed-loop-compare", json={
            "owner": "researcher", "name": "clv_net", "task_id": "throw_height"})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        deadline = time.time() + 240
        while time.time() < deadline:
            j = client.get(f"/api/ml/jobs/{job_id}").json()
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert j["status"] == "done", j.get("error")
        summary = j["result_summary"]
        assert summary["task_id"] == "throw_height"
        # Both sides were actually driven through the world and independently verified.
        for side in ("model", "baseline"):
            res = summary[side]
            assert res["session_id"] > 0
            assert res["adapter"]["adapter"], res
            assert isinstance(res["failed_actions"], int)
            assert "score" in res and "completed" in res
        v = summary["verdict"]
        assert set(v) == {"beats_baseline", "model_completed", "baseline_completed", "score_delta"}
        if v["score_delta"] is not None:
            assert v["beats_baseline"] == (v["score_delta"] > 0)

    def test_unknown_task_rejected_400(self, client):
        r = client.post("/api/ml/closed-loop-compare", json={"name": "x", "task_id": "nope"})
        assert r.status_code == 400

    def test_missing_name_rejected_400(self, client):
        r = client.post("/api/ml/closed-loop-compare", json={})
        assert r.status_code == 400
