"""The playback FastAPI mount (rest_playback.py) — the dashboard playback
surface that replaces the standalone :8766 sidecar.

TestClient-based, mirroring tests/test_rl_live_rest.py. The idle/status shape,
the guard codes and the SSE transport run without a checkpoint (the returned
`bus` handle drives the stream directly). The end-to-end start test spins a
REAL PlaybackSession through the /start route and is skipped when the recorded
b3_run1 checkpoint is absent; the heavy per-session proofs (frame contract,
recorded-band replay, read-only guarantees) live in test_rl_playback.py against
PlaybackSession directly.

These migrate the two server-dependent cases that used the raw http.server
sidecar (test_sse_delivers_spec_and_frames_to_an_early_subscriber and
test_playback_status_shape_through_the_app) onto the one dashboard mount.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from playground.rl_live.rest_playback import register_rl_playback_routes

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "experiments" / "rl" / "b3_run1" / "ckpt_final.json"


@pytest.fixture
def pb(tmp_path):
    app = FastAPI()
    handle = register_rl_playback_routes(
        app, {"rl_live": {}}, out_root=tmp_path / "runs", frame_hz=10_000)
    return TestClient(app), handle, tmp_path


def test_status_idle_shape(pb):
    client, _handle, _tmp = pb
    j = client.get("/api/rl_playback/status").json()
    assert j["server"]["surface"] == "dashboard"
    assert j["server"]["bus"]["frame_hz"] == 10_000
    assert j["session"]["state"] == "idle"
    assert j["session"]["is_playback"] is True
    assert j["session"]["config"] is None


def test_default_frame_hz_is_30(tmp_path):
    # The sidecar's status test asserted the bus runs at 30 Hz by default; the
    # mount keeps that default when no frame_hz override is passed.
    app = FastAPI()
    register_rl_playback_routes(app, {"rl_live": {}}, out_root=tmp_path / "runs")
    client = TestClient(app)
    assert client.get("/api/rl_playback/status").json()[
        "server"]["bus"]["frame_hz"] == 30.0


def test_spec_404_until_started(pb):
    client, _handle, _tmp = pb
    r = client.get("/api/rl_playback/spec")
    assert r.status_code == 404
    assert r.json()["available"] is False


def test_control_without_session_409(pb):
    client, _handle, _tmp = pb
    r = client.post("/api/rl_playback/pause")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "not_active"


def test_unknown_op_404(pb):
    client, _handle, _tmp = pb
    r = client.post("/api/rl_playback/frobnicate")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_op"


def test_start_rejects_path_outside_roots(pb):
    client, _handle, _tmp = pb
    r = client.post("/api/rl_playback/start", json={"path": "/etc/passwd"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] in ("invalid_config", "start_failed")


def test_start_rejects_traversal_in_experiment(pb):
    client, _handle, _tmp = pb
    r = client.post("/api/rl_playback/start",
                    json={"experiment": "../etc", "checkpoint": "ckpt.json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] in ("invalid_config", "start_failed")


def test_sse_transport(pb):
    # The mount's SSE delivers published events and frames to a subscriber,
    # exactly like the live mount — proven here through the returned bus handle,
    # so it runs without a checkpoint or any real MuJoCo.
    client, handle, _tmp = pb

    def pump():
        time.sleep(0.25)
        handle["bus"].publish_event("spec", {"spec_version": 1, "bodies": []})
        for i in range(6):
            handle["bus"].publish_route({"t": "frame", "episode": 1,
                                         "decision": i, "bodies": {}})
            handle["bus"].publish_event("note", {"i": i})
            time.sleep(0.02)

    threading.Thread(target=pump, daemon=True).start()
    seen_frames = 0
    events = []
    with client.stream("GET", "/api/rl_playback/stream", timeout=20) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line)
                if line == "event: frame":
                    seen_frames += 1
            if seen_frames >= 5 and any(l == "event: note" for l in events):
                break
    assert seen_frames >= 5                        # frames streamed live
    assert events[0] in ("event: status", "event: spec")


# --------------------------------------------- end-to-end real-session proof
@pytest.fixture(scope="module")
def need_ckpt():
    if not CKPT.exists():
        pytest.skip("b3_run1/ckpt_final.json not present on this tree")
    return CKPT


def test_real_playback_streams_spec_and_frames_through_the_mount(need_ckpt,
                                                                 tmp_path):
    """A real PlaybackSession started through /api/rl_playback/start delivers
    the rig spec and live frames over /stream, and /status reports the playback
    config. Migrates the sidecar's early-subscriber SSE + status-shape tests
    onto the dashboard mount (writes land under the gitignored
    experiments/rl_playback)."""
    app = FastAPI()
    register_rl_playback_routes(app, {"rl_live": {}},
                                out_root=tmp_path / "runs", frame_hz=10_000)
    client = TestClient(app)
    res = client.post("/api/rl_playback/start", json={
        "path": str(CKPT), "stage": "sustained",
        "inference": "deterministic", "speed": 10.0}).json()
    assert res["ok"] is True
    st = client.get("/api/rl_playback/status").json()["session"]
    assert st["is_playback"] is True
    assert st["config"]["inference"] == "deterministic"

    seen_spec = seen_frame = False
    deadline = time.time() + 60
    try:
        with client.stream("GET", "/api/rl_playback/stream", timeout=30) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line == "event: spec":
                    seen_spec = True
                elif line == "event: frame":
                    seen_frame = True
                if (seen_spec and seen_frame) or time.time() > deadline:
                    break
    finally:
        client.post("/api/rl_playback/stop")
    assert seen_spec, "no spec event over the playback mount"
    assert seen_frame, "a subscriber of a running playback session got no frames"
