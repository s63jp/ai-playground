"""The FastAPI mount (rest.py) — the dashboard integration surface.

TestClient-based; same injected-fake policy as the sidecar tests. Torch-free
except through lazily-import availability checks.
"""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi import FastAPI

from playground.rl_live import interop
from playground.rl_live.rest import register_rl_live_routes


@pytest.fixture
def env(tmp_path):
    b3p = tmp_path / "b3" / "progress.jsonl"
    b3p.parent.mkdir(parents=True, exist_ok=True)
    with b3p.open("w", encoding="utf-8") as f:
        for i in (1, 2):
            f.write(json.dumps({"iteration": i, "total_valid_steps": i * 2048,
                                "stage": "controlled"}) + "\n")
    app = FastAPI()
    handle = register_rl_live_routes(
        app, {"rl_live": {"b3_progress": str(b3p)}},
        out_root=tmp_path / "runs", frame_hz=10_000)
    from fastapi.testclient import TestClient
    return TestClient(app), handle, tmp_path


def test_status_objectives_b3(env):
    client, handle, _tmp = env
    st = client.get("/api/rl_live/status").json()
    assert st["session"]["state"] in ("idle", "error")
    assert st["b3_available"] is True
    assert st["session"]["trainer_available"] == interop.trainer_available()
    objs = client.get("/api/rl_live/objectives").json()
    assert [o["id"] for o in objs["objectives"]] == ["walking"]
    assert objs["stage_ladder"][0]["id"] == "standing"
    b3 = client.get("/api/rl_b3/telemetry?limit=1").json()
    assert b3["available"] is True and b3["latest"]["iteration"] == 2
    assert b3["records"] == [b3["latest"]]


def test_spec_404_until_started(env):
    client, handle, _tmp = env
    r = client.get("/api/rl_live/spec")
    assert r.status_code == 404
    assert r.json()["available"] is False


def test_start_refuses_without_trainer(tmp_path, monkeypatch):
    """The dashboard mount refuses start while the trainer layers are
    unavailable. Post-B3-merge the modules exist, so unavailability is
    FORCED through the probe (still a real production state when torch is
    missing); the honest-refusal contract stays covered instead of skipped."""
    monkeypatch.setattr("playground.rl_live.interop.trainer_available",
                        lambda: False)
    b3p = tmp_path / "b3" / "progress.jsonl"
    b3p.parent.mkdir(parents=True, exist_ok=True)
    b3p.write_text(json.dumps({"iteration": 1, "total_valid_steps": 2048,
                               "stage": "standing"}) + "\n", encoding="utf-8")
    from fastapi.testclient import TestClient
    app = FastAPI()
    register_rl_live_routes(app, {"rl_live": {"b3_progress": str(b3p)}},
                            out_root=tmp_path / "runs_unavailable",
                            frame_hz=10_000)
    client = TestClient(app)
    r = client.post("/api/rl_live/start", json={})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "trainer_unavailable"


def test_sse_stream(env):
    client, handle, _tmp = env

    def pump():
        time.sleep(0.25)
        for i in range(6):
            handle["bus"].publish_route({"t": "frame", "episode": 1,
                                         "decision": i, "bodies": {}})
            handle["bus"].publish_event("note", {"i": i})
            time.sleep(0.02)

    threading.Thread(target=pump, daemon=True).start()
    event_lines = []
    with client.stream("GET", "/api/rl_live/stream", timeout=20) as r:
        assert r.status_code == 200
        seen_frames = 0
        for line in r.iter_lines():
            if line.startswith("event:"):
                event_lines.append(line)
                if line == "event: frame":
                    seen_frames += 1
            if seen_frames >= 5 and any(l == "event: note" for l in event_lines) \
                    and any(l == "event: status" for l in event_lines):
                break
    assert seen_frames >= 5                     # frames streamed live
    assert event_lines[0] in ("event: status", "event: spec", "event: iteration")


def test_error_shape_and_codes(env):
    client, handle, _tmp = env
    r = client.post("/api/rl_live/save", json={"tag": "x"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "nothing_to_save"
