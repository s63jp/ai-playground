"""The rl_live sidecar server: endpoints, SSE fanout, static guards, and the
read-only B3 telemetry view (torch-free; trainer-unavailability tests FORCE
the unavailable state through the probe, which is still a real production
state when torch is missing)."""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError

import pytest

from playground.rl_live import interop
from playground.rl_live.bus import Bus
from playground.rl_live.server import B3Telemetry, App, LiveHTTPServer
from playground.rl_live.session import LiveBrainSession


# local fakes (no cross-test imports: tests/ is not a package)
class FakeTrainer:
    def __init__(self, target, on_iteration, should_stop):
        self.total_valid_steps = 0
        self.iteration = 0
        self._target = target
        self.on_iteration = on_iteration
        self.should_stop = should_stop
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        while self.total_valid_steps < self._target:
            if self.should_stop():
                return {"stopped": True, "reason": "should_stop"}
            time.sleep(0.01)
            self.iteration += 1
            self.total_valid_steps += 10
            self.on_iteration({"iteration": self.iteration,
                               "total_valid_steps": self.total_valid_steps,
                               "stage": "standing", "stage_transition_to": None})


def fake_saver(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fake": True}), encoding="utf-8")
    return {"path": str(path), "content_sha256": "x" * 64,
            "bytes": path.stat().st_size}


def wait_state(sess, want, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sess.state == want:
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def app(tmp_path):
    def factory(env, curriculum, *, name, seed, total_steps, stage_budgets,
                ppo_overrides, on_iteration, should_stop):
        return {"trainer": FakeTrainer(total_steps, on_iteration, should_stop),
                "actor": None, "critic": None, "ppo_cfg_dict": {}}

    b3p = tmp_path / "b3" / "progress.jsonl"
    b3p.parent.mkdir(parents=True, exist_ok=True)
    with b3p.open("w", encoding="utf-8") as f:
        for i in (1, 2):
            f.write(json.dumps({"iteration": i, "total_valid_steps": i * 2048,
                                "stage": "controlled"}) + "\n")
    session = LiveBrainSession(trainer_factory=factory, saver=fake_saver,
                               root=tmp_path / "runs")
    bus = Bus(frame_hz=10_000)
    theApp = App(session=session, bus=bus,
                 b3=B3Telemetry(b3p, bus=bus, interval=0.05),
                 out_root=tmp_path / "runs", refresher=True)
    return theApp


@pytest.fixture
def http(app):
    httpd = LiveHTTPServer(("127.0.0.1", 0), app)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _get(base, path, want_code=200):
    r = urlreq.urlopen(base + path, timeout=10)
    assert r.status == want_code
    return r.read()


def test_health_and_pages(http):
    body = _get(http, "/api/health")
    assert json.loads(body)["ok"] is True
    html = _get(http, "/rl_live.html")
    assert b"Live Neural Training" in html
    index = _get(http, "/")                      # 302 → followed to the page
    assert b"Live Neural Training" in index
    vendor = _get(http, "/static/vendor/three.module.min.js")
    assert vendor[:5] == b'"use' or len(vendor) > 100_000   # served + real


def test_status_objectives_b3(http):
    st = json.loads(_get(http, "/api/rl_live/status"))
    assert st["server"]["version"] == "rl_live/1"
    assert st["session"]["state"] in ("idle", "error")
    assert st["b3_available"] is True
    objs = json.loads(_get(http, "/api/rl_live/objectives"))
    assert [o["id"] for o in objs["objectives"]] == ["walking"]
    assert objs["stage_ladder"][0]["id"] == "standing"
    assert objs["trainer_available"] == interop.trainer_available()
    assert isinstance(objs["trainer_available"], bool)
    b3 = json.loads(_get(http, "/api/rl_b3/telemetry?limit=1"))
    assert b3["available"] is True
    assert b3["latest"]["iteration"] == 2
    assert len(b3["records"]) == 1
    assert "read-only" in b3["note"]
    # the session's own field reflects ITS factory path (injected fake → True)
    assert st["session"]["trainer_available"] is True


def test_static_traversal_guard(http):
    """A raw request escaping the static roots must 404."""
    s = socket.create_connection(("127.0.0.1", int(http.rsplit(":", 1)[1])),
                                 timeout=10)
    try:
        s.sendall(b"GET /static/../../pyproject.toml HTTP/1.1\r\n"
                  b"Host: x\r\nConnection: close\r\n\r\n")
        data = b""
        s.settimeout(10)
        while b"\r\n\r\n" not in data:
            data += s.recv(4096)
        assert b" 404 " in data.split(b"\r\n", 1)[0] + b" "
    finally:
        s.close()


def test_sse_stream_receives_frames_and_b3(http, app):
    s = socket.create_connection(("127.0.0.1", int(http.rsplit(":", 1)[1])),
                                 timeout=10)
    try:
        s.sendall(b"GET /api/rl_live/stream HTTP/1.1\r\nHost: x\r\n\r\n")
        s.settimeout(15)
        buf = b""

        def pump():
            for i in range(4):
                time.sleep(0.05)
                app.bus.publish_route({"t": "frame", "episode": 1,
                                       "decision": i, "bodies": {}})
                app.bus.publish_event("note", {"hello": i})

        threading.Thread(target=pump, daemon=True).start()
        deadline = time.time() + 15
        while time.time() < deadline:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            frames = buf.count(b"event: frame")
            if frames >= 4 and b"event: b3_iteration" in buf \
                    and b"event: note" in buf:
                break
        assert buf.count(b"event: frame") >= 4
        assert b'event: b3_iteration\ndata: {"iteration": 1' in buf  # tmp B3 tail
        assert b"event: note" in buf
    finally:
        s.close()


def test_start_refuses_without_trainer(http, tmp_path, monkeypatch):
    """The default interop path refuses to start while the trainer layers
    are UNAVAILABLE. Post-B3-merge the modules exist, so unavailability is
    FORCED through the probe (still a real production state when torch is
    missing) — the honest-refusal contract stays covered instead of skipped."""
    monkeypatch.setattr("playground.rl_live.interop.trainer_available",
                        lambda: False)
    session = LiveBrainSession(root=tmp_path / "runs_default")   # default factory
    other = App(session=session, bus=Bus(frame_hz=10_000),
                b3=B3Telemetry(tmp_path / "nonexistent.jsonl"),
                out_root=tmp_path / "runs_default", refresher=False)
    httpd = LiveHTTPServer(("127.0.0.1", 0), other)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        body = json.loads(_post(f"http://127.0.0.1:{port}",
                                "/api/rl_live/start", {}))
        assert body["ok"] is False
        assert body["error"]["code"] == "trainer_unavailable"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_save_without_session(http):
    body = json.loads(_post(http, "/api/rl_live/save", {"tag": "x"}))
    assert body["error"]["code"] == "nothing_to_save"


def test_bad_json_is_400(http, app):
    s = socket.create_connection(("127.0.0.1", int(http.rsplit(":", 1)[1])),
                                 timeout=10)
    try:
        s.sendall(b"POST /api/rl_live/start HTTP/1.1\r\nHost: x\r\n"
                  b"Content-Type: application/json\r\nContent-Length: 7\r\n"
                  b"Connection: close\r\n\r\nnotjson")
        s.settimeout(10)
        data = s.recv(4096)
        assert b" 400 " in data.split(b"\r\n", 1)[0] + b" "
    finally:
        s.close()


def _get404(http, path):
    try:
        urlreq.urlopen(http + path, timeout=10)
        return None
    except HTTPError as e:
        return e.code


def test_unknown_route_404(http):
    assert _get404(http, "/api/nope") == 404


def test_real_session_controls_over_http(app, http):
    """Full loop over HTTP with the fake trainer: start → running → pause."""
    body = json.loads(_post(http, "/api/rl_live/start",
                            {"name": "http_walk", "seed": 3,
                             "total_steps": 5_000}))
    assert body["ok"] is True
    assert wait_state(app.session, "running")
    spec_path = _get(http, "/api/rl_live/spec")
    # a real env was built by the session thread → a real spec exists
    spec = json.loads(spec_path)
    assert spec.get("spec_version") == 1
    st = json.loads(_get(http, "/api/rl_live/status"))
    assert st["session"]["config"]["name"] == "http_walk"
    # the first iteration record arrives a beat after "running": wait for it
    # (a start/pause race, not an availability fact)
    first = None
    t0 = time.time()
    while time.time() - t0 < 10.0:
        tele = json.loads(_get(http, "/api/rl_live/telemetry.json?limit=5"))["records"]
        if tele and tele[0]["iteration"] >= 1:
            first = tele[0]
            break
        time.sleep(0.05)     # the fake trainer advances 10 steps/10 ms
    assert first is not None, "fake trainer produced no iteration record in 10 s"
    app.session.pause()
    assert wait_state(app.session, "paused")
    save = json.loads(_post(http, "/api/rl_live/save", {"tag": "peek"}))
    assert save["ok"] is True
    app.session.stop()
    assert wait_state(app.session, "stopped")


def _post(base, path, body):
    r = urlreq.Request(base + path, data=json.dumps(body).encode(),
                       headers={"Content-Type": "application/json"},
                       method="POST")
    return urlreq.urlopen(r, timeout=30).read()
