"""Smoke tests: the whole system starts, serves and runs end to end."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from playground.agents import make_adapter
from playground.cli import main
from playground.dashboard import create_app
from playground.session import run_session


def test_scripted_exploration_produces_a_discovery(cfg, db):
    cfg["session"]["max_steps"] = 90
    s = run_session(cfg, db, "agent_001", make_adapter(cfg, "scripted"))
    assert s["end_reason"] == "agent_ended" and s["error"] is None
    assert s["memory"]["discoveries"]["confirmed"] >= 1
    assert s["integrity"]["event_chain"]["ok"]


def test_dashboard_and_api(cfg, db):
    cfg["session"]["max_steps"] = 40
    s = run_session(cfg, db, "agent_001", make_adapter(cfg, "scripted"))
    c = TestClient(create_app(cfg, db))
    assert c.get("/api/health").json()["ok"]
    # the page serves the real app (app.js) AND ships the live neural
    # training tab; the old '"<canvas" not in page' premise is obsolete —
    # the rl_live tab's sparklines/viewport legitimately use canvases.
    home = c.get("/")
    assert "app.js" in home.text and "rl_live_tab.js" in home.text
    assert c.get("/static/vendor/three.module.min.js").status_code == 200
    assert len(c.get("/api/agents").json()["agents"]) == 13
    sid = s["session_id"]
    assert c.get(f"/api/sessions/{sid}").json()["status"] == "completed"
    assert c.get(f"/api/sessions/{sid}/frames").json()["total"] > 10
    assert c.get(f"/api/sessions/{sid}/integrity").json()["event_chain"]["ok"]
    assert "self_directed_goals" in c.get(f"/api/sessions/{sid}/behaviour").json()
    assert len(c.get("/api/compare").json()["agents"]) == 13
    assert c.get("/api/sessions/99999").status_code == 404


def test_external_agent_drives_a_session_over_http(cfg, db):
    c = TestClient(create_app(cfg, db))
    r = c.post("/api/agent/sessions", json={"agent_id": "hermes_001", "max_steps": 20}).json()
    assert "You have no assigned mission, and no objective is waiting for you." in r["system_prompt"] and len(r["tools"]) > 30
    sid = r["session_id"]
    assert c.post(f"/api/agent/sessions/{sid}/call", json={"tool": "observe_world"}).json()["success"]
    bad = c.post(f"/api/agent/sessions/{sid}/call", json={"tool": "fly", "args": {}}).json()
    assert bad["success"] is False
    j = c.post(f"/api/agent/sessions/{sid}/call", json={"tool": "jump", "args": {}, "say": "hop"}).json()
    assert j["success"] and j["apex_above_takeoff"] > 0.5
    summ = c.post(f"/api/agent/sessions/{sid}/finish", json={"reason": "done"}).json()
    assert summ["actions_logged"] == 3 and summ["integrity"]["event_chain"]["ok"]
    assert c.post(f"/api/agent/sessions/{sid}/call", json={"tool": "jump"}).status_code == 404
    assert c.post("/api/agent/sessions", json={"agent_id": "../etc"}).status_code == 400


def test_external_benchmark_task_is_verified_by_the_server(cfg, db):
    c = TestClient(create_app(cfg, db))
    r = c.post("/api/agent/sessions", json={"agent_id": "ext", "mode": "benchmark",
                                            "task_id": "generate_blocks"}).json()
    summ = c.post(f"/api/agent/sessions/{r['session_id']}/finish", json={}).json()
    assert summ["task_result"]["verified"] is False


def test_cli(tmp_path, capsys):
    cfgf = tmp_path / "c.yaml"
    cfgf.write_text(json.dumps({"_root": str(tmp_path), "session": {"max_steps": 30}}))
    assert main(["--config", str(cfgf), "init"]) == 0
    assert main(["--config", str(cfgf), "explore", "--agent", "agent_002", "--adapter", "scripted"]) == 0
    assert main(["--config", str(cfgf), "memory", "show", "agent_002"]) == 0
    assert main(["--config", str(cfgf), "verify-log", "1"]) == 0
    out = capsys.readouterr().out
    assert "agent_002" in out and '"ok": true' in out
