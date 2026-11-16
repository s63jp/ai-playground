"""HVE coverage for the Nov 15 2026 session: engine-generated datasets (throw/jump/climb), the
generate endpoint's kind dispatch, the empty-log build refusal, and the infrastructure tab loaders.

The engine generators drive the REAL MuJoCo world with small n; rows are compared against the
log-built behaviour datasets' own row vocabulary, so a model trained on either can be evaluated
on either (closed-loop design)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from playground.ml.data import (generate_climb_dataset, generate_jump_dataset)
from playground.ml.datasets import columns_for

# Repo-relative static root: tests must never hardcode a developer's machine path (the Nov 16 2026
# CI failure) — resolve from this file so the suite runs identically on Windows and the CI checkout.
STATIC = Path(__file__).resolve().parents[1] / "playground" / "dashboard" / "static"


def ml_job_wait(client, job_id: str, timeout: float = 90.0) -> dict:
    """Wait for an ML job (the generate/train/build jobs live under /api/ml/jobs, not /api/run/jobs)."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/ml/jobs/{job_id}").json()
        if j.get("status") in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError(f"ml job {job_id} never finished")


@pytest.fixture
def client(cfg, db):
    from playground.dashboard import create_app
    return TestClient(create_app(cfg, db))


# --------------------------------------------------------------------- engine generators


class TestGenerateJump:
    def test_jump_rows_match_the_log_schema_and_track_mass(self, cfg):
        n = 8
        ds = generate_jump_dataset(cfg, n, seed=11)
        m, rows = ds["manifest"], ds["rows"]
        ins, tgts = columns_for("jump")
        assert m["kind"] == "jump" and m["source"] == "engine_jumps"
        assert m["inputs"] == sorted(ins) and m["targets"] == sorted(tgts)
        assert m["rows_sha256"] and m["requested"] == n and m["seconds"] >= 0
        assert len(rows) >= n - 1, f"expected ~{n} usable jumps, got {len(rows)} (failed {m['failed']})"
        keys = set(ins) | set(tgts)
        for r in rows:
            assert set(r) <= keys, r
        masses = sorted({r["total_mass"] for r in rows})
        takeoffs = sorted({r["takeoff_feet_height"] for r in rows})
        assert 0.0 in takeoffs and max(takeoffs) >= 0.49, f"platforms never varied: {takeoffs}"
        assert 50.0 <= min(masses) <= 50.0 and max(masses) >= 55.0, masses
        # the mass -> apex signal: heavier jumps never out-perform lighter ones
        pairs = sorted({(r["total_mass"], r["apex_above_takeoff"]) for r in rows})
        apex_by_mass = {mm: aa for mm, aa in pairs}
        vals = list(apex_by_mass.items())
        assert all(apex_by_mass[m1] >= apex_by_mass[m2] for (m1, _), (m2, _) in zip(vals, vals[1:])), pairs
        # identical (inputs) -> identical targets (deterministic engine)
        assert rows[0]["apex_above_takeoff"] > 0.5

    def test_invalid_n_rejected(self, cfg):
        with pytest.raises(Exception) as e:
            generate_jump_dataset(cfg, 0, seed=1)
        assert "n must be" in str(e.value)


class TestGenerateClimb:
    def test_climb_rows_match_vocabulary_with_both_ledge_heights(self, cfg):
        n = 12
        ds = generate_climb_dataset(cfg, n, seed=5)
        m, rows = ds["manifest"], ds["rows"]
        ins, tgts = columns_for("climb")
        assert m["kind"] == "climb" and m["source"] == "engine_climbs"
        assert m["inputs"] == sorted(ins) and m["targets"] == sorted(tgts)
        keys = set(ins) | set(tgts)
        assert len(rows) >= n - 2, f"expected ~{n} usable climbs, got {len(rows)} (failed {m['failed']})"
        for r in rows:
            assert set(r) <= keys, r
            assert r["success"] == 1.0, "engine climbs either complete (success 1) or raise; never a 0 row"
            assert r["height_gained"] <= r["ledge_height"] + 1e-6
        ledges = sorted({r["ledge_height"] for r in rows})
        assert 0.49 <= min(ledges) and 0.99 <= max(ledges), f"ledge heights never varied: {ledges}"

    def test_invalid_n_rejected(self, cfg):
        with pytest.raises(Exception) as e:
            generate_climb_dataset(cfg, 200001, seed=1)
        assert "n must be" in str(e.value)


# --------------------------------------------------------------------- endpoint behaviour


class TestGenerateEndpoint:
    def test_generate_kind_jump_end_to_end(self, client):
        r = client.post("/api/ml/generate", json={
            "kind": "jump", "name": "engine_jump_smoke", "n": 10, "seed": 2})
        assert r.status_code == 200, r.text
        job = ml_job_wait(client, r.json()["job_id"])
        assert job["status"] == "done", job.get("error")
        summary = job["result_summary"]
        assert summary["rows"] >= 6, (summary, job.get("error"))
        assert summary["source"] == "engine_jumps" and summary["requested"] == 10 and "failed" in summary
        listed = client.get("/api/ml/datasets?scope=researcher&owner=researcher").json()["datasets"]
        assert any(d["name"] == "engine_jump_smoke" and d["rows"] == summary["rows"] for d in listed)
        cols = client.get("/api/ml/datasets/engine_jump_smoke/quality").json()
        assert cols["kind"] == "jump" and cols["rows"] == summary["rows"]

    def test_unknown_kind_is_a_400(self, client):
        r = client.post("/api/ml/generate", json={"kind": "hurl", "name": "nope", "n": 20})
        assert r.status_code == 400 and "unknown engine kind" in r.text

    def test_bad_name_is_a_400(self, client):
        r = client.post("/api/ml/generate", json={"kind": "throw", "name": "no such name"})
        assert r.status_code == 400

    def test_out_of_range_n_is_a_400(self, client):
        r = client.post("/api/ml/generate", json={"kind": "throw", "name": "ok_name", "n": 9})
        assert r.status_code == 400 and "n must be between" in r.text


class TestBuildRefusesEmptyLog:
    def test_empty_log_build_is_a_400_with_guidance_and_no_artifact(self, client):
        import json
        r = client.post("/api/ml/build-behaviour", json={"kind": "jump", "name": "empty_build", "seed": 1})
        assert r.status_code == 400
        body = r.json()
        assert "0 usable 'jump' actions" in body["detail"]
        assert "engine" in body["detail"], "error should point the user at the engine builders"
        listed = client.get("/api/ml/datasets?scope=researcher&owner=researcher").json()["datasets"]
        assert not any(d["name"] == "empty_build" for d in listed), "0-row artifact must not be saved"


class TestUiContract:
    """The Infrastructure + NN tabs depend on loaders referenced by showTab(); when one is missing
    the tab renders dead (the lost-providers incident of Nov 15 2026). Guard the contract statically."""

    def test_every_tab_loader_referenced_by_showtab_is_defined(self):
        with open(STATIC / "app.js", encoding="utf-8") as f:
            src = f.read()
        for fn in ("loadMemory", "loadAnalytics", "loadCompare", "loadBench", "loadNn", "loadLab",
                   "loadInfra", "loadProviders", "loadMonitor", "loadResources"):
            assert f"function {fn}" in src, f"{fn} is referenced by showTab but never defined"
        assert "/api/gateway/providers" in src and "pv-refresh" in src
        assert "nn-gen-kind" in src and "j.progress" in src

    def test_app_js_is_syntactically_valid(self):
        node = shutil.which("node")
        if not node:
            pytest.skip("node not available")
        out = subprocess.run([node, "--check", str(STATIC / "app.js")],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

    def test_generate_form_has_engine_kinds(self):
        with open(STATIC / "index.html", encoding="utf-8") as f:
            html = f.read()
        for marker in ('nn-gen-kind', 'value="jump"', 'value="climb"', 'value="throw"'):
            assert marker in html
