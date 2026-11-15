"""HVE coverage for the SSE live console (Nov 2026 feature batch).

The stream reads the shared SQLite log, so a session running in a thread appears in the stream and the
client receives session-lifecycle + action events as server-sent events. These tests run a REAL uvicorn
server on an ephemeral port: the in-process TestClient buffers streaming responses until completion,
which can never happen for an endless SSE stream (and would also be untrue of production).
"""
from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from playground.dashboard import create_app
from playground.session.runner import run_session
import playground.session.runner as runner_mod

from test_ml_dataset_and_stop import _SlowAdapter


@pytest.fixture
def server(cfg, db):
    """A real uvicorn server on a free port, wired to the tmp cfg/db."""
    app = create_app(cfg, db)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    deadline = time.time() + 15
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=15)


def _wait_registered(db, agent_id: str, timeout: float = 25.0) -> int:
    """Wait until the runner has created a running session row for THIS agent and registered its
    stop signal. (The stop-test file's helper hardcodes agent_id='agent_t'; these tests run
    'agent_sse', so they need their own wait — polling the wrong agent id was the whole failure.)"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = db.q("SELECT id FROM sessions WHERE agent_id=? AND status='running'", [agent_id])
        if rows and rows[0]["id"] in runner_mod._session_stop_events:
            return rows[0]["id"]
        time.sleep(0.01)
    raise AssertionError(f"session for {agent_id} never started "
                         f"(rows: {db.q('SELECT id, agent_id, status FROM sessions')})")


def _start_session_in_thread(cfg, db, results):
    th = threading.Thread(
        target=lambda: results.append(run_session(cfg, db, "agent_sse", _SlowAdapter(), mode="exploration")),
        daemon=True)
    th.start()
    return th, _wait_registered(db, "agent_sse")


def _iter_sse(resp, limit=6):
    """Yield parsed data frames from a streaming SSE response, byte-level."""
    buf = b""
    for chunk in resp.iter_bytes(chunk_size=8):
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.decode("utf-8", "replace").splitlines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])
                    if limit:
                        limit -= 1
                        if not limit:
                            return


def _stop_and_join(th, sid, results, timeout=60.0):
    runner_mod.request_stop_session(sid)
    th.join(timeout=timeout)


class TestSseLiveConsole:
    def test_stream_delivers_session_and_action_events(self, cfg, db, server):
        results: list = []
        th, sid = _start_session_in_thread(cfg, db, results)
        try:
            with httpx.stream("GET", f"{server}/api/stream/events", params={"max_events": 6},
                              timeout=20.0) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                events = list(_iter_sse(resp))
        finally:
            _stop_and_join(th, sid, results)
        assert events, "stream produced no events"
        session_events = [e for e in events if e["type"] == "session"]
        action_events = [e for e in events if e["type"] == "action"]
        assert session_events and any(e.get("agent_id") == "agent_sse" for e in session_events)
        assert action_events and action_events[0]["tool"]
        assert results[0]["end_reason"] == "stopped_by_user"

    def test_agent_filter_only_streams_that_agent(self, cfg, db, server):
        results: list = []
        th, sid = _start_session_in_thread(cfg, db, results)
        try:
            with httpx.stream("GET", f"{server}/api/stream/events",
                              params={"agent": "agent_sse", "max_events": 4}, timeout=20.0) as resp:
                events = list(_iter_sse(resp))
            assert events
            for e in events:
                assert e.get("agent_id") == "agent_sse" or e.get("session_id") == sid, e
        finally:
            _stop_and_join(th, sid, results)

    def test_stream_headers_and_frames_are_valid_json(self, cfg, db, server):
        results: list = []
        th, sid = _start_session_in_thread(cfg, db, results)
        try:
            with httpx.stream("GET", f"{server}/api/stream/events", params={"max_events": 2},
                              timeout=20.0) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                assert resp.headers.get("cache-control") == "no-cache"
                got = 0
                for ev in _iter_sse(resp):
                    assert isinstance(ev, dict) and ev.get("type") in ("session", "action")
                    got += 1
                    if got >= 2:
                        break
                assert got >= 1
        finally:
            _stop_and_join(th, sid, results)

    def test_untracked_action_rows_reach_the_stream(self, cfg, db, server):
        """Actions logged by ANY code — even outside a live runner — must appear: the stream follows the
        log, not an in-memory registry (CLI-driven sessions in other processes stream the same way)."""
        from test_ml_dataset_and_stop import _climb_action
        sid = db.create_session(agent_id="agent_sse2", mode="exploration", session_number=1, adapter="test")
        with httpx.stream("GET", f"{server}/api/stream/events", params={"agent": "agent_sse2"},
                          timeout=20.0) as resp:
            frames = []
            db.append_action(sid, **_climb_action(1))    # logged while the stream is open
            for ev in _iter_sse(resp):
                frames.append(ev)
                if any(e["type"] == "action" for e in frames) and len(frames) >= 2:
                    break
        actions = [e for e in frames if e["type"] == "action"]
        assert actions and actions[0]["tool"] == "climb"
