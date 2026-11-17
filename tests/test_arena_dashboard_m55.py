"""M5.5 dashboard run-mode integration (real HTTP surface over a real tmp backend).

HVE shape:
- Happy: GET /api/run/modes lists exactly the four backend-supported modes plus the real
  equipment catalogue; POST /api/run/arena runs a REAL 1v1 duel as a run job (recorded
  arena_solo sessions, manifest recorded in the session config and surfaced by
  /api/sessions/{sid}); POST /api/run/benchmark runs a REAL scripted benchmark task as a
  run job (mode='benchmark' recorded, benchmark run row + result).
- Validate: wrong mode, third team, unknown equipment, 2v1 layout submitted as arena_solo,
  unknown task and unknown adapter are all refused with 400 BEFORE any job starts.
- Error: GET /api/run/jobs/{bogus} -> 404.
"""
from __future__ import annotations

import time

import pytest


def _client(tmp_path):
    from fastapi.testclient import TestClient
    from playground.config import load_config
    from playground.dashboard.server import create_app
    from playground.storage import Database
    cfg = load_config(overrides={"_root": str(tmp_path), "session": {"max_steps": 400}})
    db = Database(tmp_path / "dash.db")
    return TestClient(create_app(cfg, db))


def _wait_job(client, job_id: str, timeout_s: float = 240.0) -> dict:
    t0 = time.time()
    last: dict = {}
    while time.time() - t0 < timeout_s:
        r = client.get(f"/api/run/jobs/{job_id}")
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") != "running":
            return last
        time.sleep(0.4)
    return {"status": "wait-timeout", "detail": last}


class TestHappy:
    def test_run_modes_list_backend_modes_and_real_equipment(self, tmp_path):
        from playground.combat.equipment import EQUIPMENT
        c = _client(tmp_path)
        r = c.get("/api/run/modes")
        assert r.status_code == 200
        data = r.json()
        ids = [m["id"] for m in data["modes"]]
        # Exactly what the backend can run: runner dispatch + benchmark + arena factories.
        assert ids == ["exploration", "benchmark", "arena_solo", "arena_coop"]
        assert all(m.get("label") for m in data["modes"])
        opts = data["equipment"]
        assert [o["id"] for o in opts] == sorted(EQUIPMENT)
        pistol = next(o for o in opts if o["id"] == "paintball_pistol")
        assert pistol["combat_class"] == "ranged" and pistol["dims_m"] == [0.05, 0.19, 0.12]
        assert data["limits"]["duration_s"] == [2, 600] and data["limits"]["teams"] == 2

    def test_arena_solo_run_job_records_duel_with_manifest(self, tmp_path):
        c = _client(tmp_path)
        body = {
            "mode": "arena_solo",
            "name": "dashm55",
            "rules": {"duration_s": 6.0, "pellets": 24, "score_target": 100, "friendly_fire": False},
            "teams": [
                {"name": "Red", "agents": [{"id": "agent_dial_r1", "adapter": "scripted"}],
                 "equipment": ["paintball_rifle"]},
                {"name": "Blue", "agents": [{"id": "agent_dial_b1", "adapter": "scripted"}],
                 "equipment": ["paintball_pistol"]},
            ],
        }
        r = c.post("/api/run/arena", json=body)
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        job = _wait_job(c, job_id)
        assert job["status"] == "done", job
        result = job["result"]
        assert result["mode"] == "arena_solo"
        assert result["end_reason"] in ("timeout", "incapacitated", "score_target")
        assert len(result["sessions"]) == 2
        sids = sorted(int(s["session_id"]) for s in result["sessions"])
        # Every combatant session is a real arena_solo session carrying the real manifest.
        for sid in sids:
            d = c.get(f"/api/sessions/{sid}").json()
            assert d["mode"] == "arena_solo"
            assert d["label"] == "arena:dashm55"
            ac = d["arena_context"]
            assert ac["match"]["mode"] == "arena_solo"
            assert set(ac["match"]["teams"]) == {"Red", "Blue"}
            assert ac["match"]["pellets"] == 24
            items = ac["match"]["items"]
            assert len(items) == 2
            by_team = {it["id"].split("_item_")[0]: it for it in items}
            assert by_team["red"]["item"] == "paintball_rifle"
            assert by_team["blue"]["item"] == "paintball_pistol"
            # The combatant's own team context is per-session and consistent.
            own_team = ac["team"]
            assert own_team in ("Red", "Blue") and sid in [int(s["session_id"]) for s in result["sessions"]]

    def test_benchmark_run_job_records_task_session(self, tmp_path):
        c = _client(tmp_path)
        r = c.post("/api/run/benchmark", json={"task_id": "topple", "agent_id": "agent_bench_dial",
                                               "adapter": "scripted"})
        assert r.status_code == 200, r.text
        job = _wait_job(c, r.json()["job_id"])
        assert job["status"] == "done", job
        result = job["result"]
        assert result["task_id"] == "topple"
        assert int(result["session_id"]) > 0
        d = c.get(f"/api/sessions/{int(result['session_id'])}").json()
        assert d["mode"] == "benchmark" and d["task_id"] == "topple"


class TestValidate:
    def test_unknown_mode_refused(self, tmp_path):
        c = _client(tmp_path)
        r = c.post("/api/run/arena", json={"mode": "arena_extra"})
        assert r.status_code == 400 and "arena_solo" in r.json()["detail"]

    def test_three_teams_refused(self, tmp_path):
        c = _client(tmp_path)
        body = {"mode": "arena_coop", "teams": [
            {"name": "A", "agents": [{"id": "a"}]}, {"name": "B", "agents": [{"id": "b"}]},
            {"name": "C", "agents": [{"id": "c"}]}]}
        r = c.post("/api/run/arena", json=body)
        assert r.status_code == 400 and "two teams" in r.json()["detail"]

    def test_unknown_equipment_refused(self, tmp_path):
        c = _client(tmp_path)
        body = {"mode": "arena_coop", "teams": [
            {"name": "A", "agents": [{"id": "a"}], "equipment": ["paintball_laser"]},
            {"name": "B", "agents": [{"id": "b"}]}]}
        r = c.post("/api/run/arena", json=body)
        assert r.status_code == 400 and "paintball_laser" in r.json()["detail"]

    def test_solo_layout_mismatch_refused(self, tmp_path):
        c = _client(tmp_path)
        body = {"mode": "arena_solo", "teams": [
            {"name": "A", "agents": [{"id": "a"}, {"id": "a2"}]},
            {"name": "B", "agents": [{"id": "b"}]}]}
        r = c.post("/api/run/arena", json=body)
        assert r.status_code == 400 and "arena_coop" in r.json()["detail"]

    def test_unknown_combatant_adapter_refused(self, tmp_path):
        c = _client(tmp_path)
        body = {"mode": "arena_solo", "teams": [
            {"name": "A", "agents": [{"id": "a", "adapter": "nope"}]},
            {"name": "B", "agents": [{"id": "b"}]}]}
        r = c.post("/api/run/arena", json=body)
        assert r.status_code == 400 and "nope" in r.json()["detail"]

    def test_unknown_task_refused(self, tmp_path):
        c = _client(tmp_path)
        r = c.post("/api/run/benchmark", json={"task_id": "not_a_task", "agent_id": "agent_x"})
        assert r.status_code == 400 and "available" in r.json()["detail"]

    def test_benchmark_unknown_agent_refused(self, tmp_path):
        c = _client(tmp_path)
        r = c.post("/api/run/benchmark", json={"task_id": "topple", "agent_id": "BAD ID!"})
        assert r.status_code == 400


class TestError:
    def test_unknown_job_404(self, tmp_path):
        c = _client(tmp_path)
        assert c.get("/api/run/jobs/deadbeef").status_code == 404
