"""The live-training sidecar: localhost-only HTTP + SSE server.

Run from the repository root:
    .venv/Scripts/python.exe -m playground.rl_live.server
    → http://127.0.0.1:8766/rl_live.html

A separate process from `playground serve` (the island dashboard stays
exactly as it is). Endpoints:

  GET  /rl_live.html                     the viewer page
  GET  /static/rl_live.js                this module's own static file
  GET  /static/<file>                    READ-ONLY reuse of the dashboard
                                         static dir (three.js vendor,
                                         OrbitControls, HumanoidView, css)
  GET  /api/rl_live/status               session + server state
  GET  /api/rl_live/objectives           the objective registry + stage ladder
  GET  /api/rl_live/spec                 the cached scene spec (after a start)
  GET  /api/rl_live/stream               SSE: frame|reset|episode_end|iteration|
                                         stage_transition|status|saved|spec|
                                         b3_iteration
  GET  /api/rl_live/telemetry.json       last N iteration records (own run)
  GET  /api/rl_b3/telemetry.json         last N records of the RUNNING B3
                                         run's progress.jsonl — READ-ONLY
  POST /api/rl_live/start|pause|resume|stop|save

Safety posture (mirrors the project's): binds 127.0.0.1 only, no auth, no
DNS exposure; static file serving is path-traversal-guarded; the only write
targets are the session's own run dir under <root> (default
experiments/rl_live/). The B3 experiment directory is only ever read.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .bus import Bus
from .session import OBJECTIVES, LiveBrainSession
from .telemetry import B3Telemetry, read_records

ROOT = Path(__file__).resolve().parents[2]
MODULE_STATIC = Path(__file__).resolve().parent / "static"
DASHBOARD_STATIC = ROOT / "playground" / "dashboard" / "static"
SERVER_VERSION = "rl_live/1"


# ------------------------------------------------------------------- the app
class App:
    def __init__(self, *, session=None, bus=None, b3=None, out_root=None,
                 frame_hz=30.0, refresher=True):
        self.session = session or LiveBrainSession(root=out_root)
        self.bus = bus or Bus(frame_hz=frame_hz)
        self.b3 = b3
        self._broadcaster = self.bus.publish_event
        # wire the session's two sinks into the bus; on_broadcast records the
        # spec for /api/rl_live/spec + SSE replay, then forwards to clients
        self.session._frame_sink = self.bus.publish_route
        self.session._broadcast = self.on_broadcast
        self._last_spec: dict | None = None
        self.started = time.time()
        if refresher:
            threading.Thread(target=self._refresh_loop, daemon=True,
                             name="rl_live_b3_refresher").start()

    def on_broadcast(self, kind, payload):
        if kind == "spec":
            self._last_spec = payload
        self._broadcaster(kind, payload)

    def _refresh_loop(self):
        while True:
            try:
                if self.bus.subscriber_count > 0 and self.b3 is not None:
                    self.b3.refresh()
            except Exception:
                pass
            time.sleep(1.0)

    # ------------------------------------------------------- helper payloads
    def objectives_payload(self) -> dict:
        from .interop import trainer_available
        stages = []
        for i, s in enumerate(_stage_ladder()):
            stages.append({"index": i, "id": s.id, "name": s.name,
                           "description": s.description,
                           "env_ready": s.env_ready})
        return {"objectives": [{"id": k, "label": v["label"],
                                "stage_budgets": v["stage_budgets"],
                                "default": k == "walking"}
                               for k, v in OBJECTIVES.items()],
                "stage_ladder": stages,
                "trainer_available": trainer_available(),
                "note": "the first supported objective is walking (the B3 "
                        "ladder); climbing / navigation / obstacle avoidance "
                        "deliberately not offered yet"}

    def status_payload(self) -> dict:
        full = self.session.status()
        st = full["session"]
        st["trainer_available"] = full["trainer_available"]
        return {"server": {"version": SERVER_VERSION,
                           "started_at": self.started,
                           "bus": self.bus.stats(),
                           "root": str(self.session.root)},
                "session": st,
                "b3_available": bool(self.b3 and
                                     self.b3.path and self.b3.path.exists())}


def _stage_ladder():
    """The B2 curriculum stage ladder (metadata for display only)."""
    from playground.rl.curriculum import DEFAULT_STAGES
    return DEFAULT_STAGES


# -------------------------------------------------------------------- server
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = SERVER_VERSION

    @property
    def app(self) -> "App":
        return self.server.app      # per-server instance, never class-level

    def log_message(self, fmt, *args):   # quiet: localhost sidecar
        pass

    # ------------------------------------------------------------ dispatch
    def do_GET(self):
        route = self.path.split("?", 1)
        path = route[0]
        try:
            if path in ("/", "/index.html"):
                return self._redirect("/rl_live.html")
            if path == "/rl_live.html":
                return self._file(MODULE_STATIC / "rl_live.html", "text/html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/rl_live/status":
                return self._json(self.app.status_payload())
            if path == "/api/rl_live/objectives":
                return self._json(self.app.objectives_payload())
            if path == "/api/rl_live/spec":
                spec = self.app._last_spec
                if spec is None:
                    return self._json({"available": False,
                                       "note": "start a session to build the "
                                               "rig scene from the live model"})
                return self._json(spec)
            if path == "/api/rl_live/stream":
                return self._stream()
            if path == "/api/rl_live/telemetry.json":
                return self._json({"records": self.app.session.recent_iterations(
                    limit=self._limit(200)), "session_state":
                    self.app.session.state})
            if path == "/api/rl_b3/telemetry":
                return self._json(self.app.b3.snapshot(limit=self._limit(30)))
            if path == "/api/health":
                return self._json({"ok": True, "version": SERVER_VERSION})
            return self._json({"error": {"code": "not_found",
                               "message": path}}, 404)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            return self._json({"error": {"code": "invalid_json",
                                 "message": "body is not valid JSON"}}, 400)
        s = self.app.session
        try:
            if path == "/api/rl_live/start":
                return self._json(s.start(
                    objective=body.get("objective", "walking"),
                    name=body.get("name"), seed=body.get("seed"),
                    total_steps=body.get("total_steps"),
                    ppo_overrides=body.get("ppo_overrides")))
            if path == "/api/rl_live/pause":
                return self._json(s.pause())
            if path == "/api/rl_live/resume":
                return self._json(s.resume())
            if path == "/api/rl_live/stop":
                return self._json(s.stop())
            if path == "/api/rl_live/save":
                return self._json(s.save(tag=body.get("tag")))
            return self._json({"error": {"code": "not_found",
                               "message": path}}, 404)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # ------------------------------------------------------------ plumbing
    def _limit(self, default):
        try:
            q = self.path.split("?", 1)[1]
            from urllib.parse import parse_qs
            return max(1, min(500, int(parse_qs(q).get("limit", [default])[0])))
        except Exception:
            return default

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(min(length, 1_048_576))
        return json.loads(raw or b"{}")

    def _json(self, obj, code=200):
        data = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, loc):
        self.send_response(302)
        self.send_header("Location", loc)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_bytes(self, data: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, p: Path, ctype: str):
        if not p.exists():
            return self._json({"error": {"code": "missing_file",
                                 "message": str(p)}}, 404)
        self._send_bytes(p.read_bytes(), ctype)

    def _static(self, rel: str):
        rel = rel.lstrip("/")
        candidates = []
        if rel in ("rl_live.js", "rl_live.css"):
            candidates.append(MODULE_STATIC / rel)
        candidates.append(DASHBOARD_STATIC / rel)
        for p in candidates:
            resolved = p.resolve()
            if not (resolved.is_file() and
                    resolved.as_posix().startswith(
                        p.parent.resolve().as_posix())):
                continue
            ctype = self._ctype(resolved.suffix)
            if ctype:
                return self._send_bytes(resolved.read_bytes(), ctype)
        return self._json({"error": {"code": "not_found", "message": rel}}, 404)

    @staticmethod
    def _ctype(suffix: str):
        return {".html": "text/html", ".js": "text/javascript",
                ".css": "text/css", ".map": "application/json",
                ".svg": "image/svg+xml", ".ico": "image/x-icon",
                ".png": "image/png"}.get(suffix)

    # ------------------------------------------------------------ SSE
    def _stream(self):
        sub = self.app.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            # replay context: spec (if a scene exists) + status + last iteration
            if self.app._last_spec is not None:
                self._sse("spec", self.app._last_spec)
            last_iter = self.app.session.recent_iterations(limit=1)
            state = self.app.status_payload()
            self._sse("status", state["session"])
            if last_iter:
                self._sse("iteration", last_iter[0])
            last_sent = time.monotonic()
            while True:
                try:
                    sub.cond.acquire()
                    sub.cond.wait(timeout=5.0)
                finally:
                    sub.cond.release()
                events, frame, _ = sub.drain()
                for kind, payload in events:
                    if kind == "spec":
                        continue      # spec replayed only on connect (once)
                    self._sse(kind, payload)
                if frame is not None:
                    self._sse("frame", frame)
                now = time.monotonic()
                if not events and frame is None and now - last_sent > 15.0:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_sent = now
                if events or frame is not None:
                    last_sent = now
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.bus.unsubscribe(sub)
            self.close_connection = True

    def _sse(self, kind: str, payload):
        try:
            data = json.dumps(payload, default=str)
            self.wfile.write(f"event: {kind}\ndata: {data}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise


class LiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, app: App):
        super().__init__(addr, _Handler)
        # handlers reach their app via self.server — per-instance, so several
        # servers can coexist in one process (tests, multi-port runs)
        self.app = app


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="loopback only by policy; any other value is refused")
    ap.add_argument("--frame-hz", type=float, default=30.0,
                    help="viewer frame rate (MuJoCo decisions run at 50 Hz)")
    ap.add_argument("--out-root", default=str(ROOT / "experiments" / "rl_live"),
                    help="live run dirs live here (default experiments/rl_live)")
    ap.add_argument("--b3-progress",
                    default=str(ROOT / "experiments" / "rl" / "b3_run1" /
                                "progress.jsonl"),
                    help="the RUNNING B3 run's progress.jsonl, read-only")
    ap.add_argument("--no-b3", action="store_true",
                    help="disable the B3 read-only telemetry panel entirely")
    args = ap.parse_args(argv)
    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        ap.error("refusing to bind anything but loopback (project policy: the "
                 "sidecar has no authentication)")
    b3_path = None if args.no_b3 else args.b3_progress
    bus = Bus(frame_hz=args.frame_hz)
    b3 = None if b3_path is None else B3Telemetry(b3_path, bus=bus)
    session = LiveBrainSession(root=Path(args.out_root))
    app = App(session=session, bus=bus, b3=b3, out_root=Path(args.out_root),
              frame_hz=args.frame_hz, refresher=not args.no_b3)
    httpd = LiveHTTPServer(("127.0.0.1", args.port), app)
    shown = lambda: f"http://127.0.0.1:{httpd.server_address[1]}/rl_live.html"
    print(f"[rl_live] sidecar on {shown()}", flush=True)
    print(f"[rl_live] B3 protection: read-only tail of {args.b3_progress}"
          f" ({'configured' if b3 else 'disabled'}); live runs write only "
          f"under {args.out_root}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
