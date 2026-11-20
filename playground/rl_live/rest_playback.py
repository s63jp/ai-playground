"""FastAPI mount for checkpoint PLAYBACK on the main dashboard server.

This migrates the standalone sidecar's unique playback functionality
(`playground.rl_live.playback.PlaybackSession`) into the one dashboard
backend, so there is no second server/process. It mirrors the live mount
(rest.py) but on the `/api/rl_playback/*` route family with its own Bus/SSE,
so live training and playback can coexist on the same server without colliding
on the shared `/api/rl_live/*` surface.

    from ..rl_live.rest_playback import register_rl_playback_routes
    register_rl_playback_routes(app, cfg)     # dashboard/server.py
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi.responses import JSONResponse, StreamingResponse

from .bus import Bus
from .playback import PlaybackError, PlaybackSession

ROOT = Path(__file__).resolve().parents[2]


def register_rl_playback_routes(app, cfg: dict | None = None, *,
                                out_root: Path | None = None,
                                frame_hz: float = 30.0):
    rl_live_cfg = dict((cfg or {}).get("rl_live") or {})
    live_root = Path(out_root or rl_live_cfg.get(
        "out_root") or ROOT / "experiments" / "rl_live")
    pb_root = ROOT / "experiments" / "rl_playback"
    # checkpoints may be played from any of the training output roots
    allowed_roots = [live_root, ROOT / "experiments" / "rl", pb_root]

    bus = Bus(frame_hz=frame_hz)
    spec_holder: dict = {"spec": None}
    state: dict = {"session": None}       # current PlaybackSession or None
    lock = threading.RLock()

    def broadcast(kind, payload):
        if kind == "spec":
            spec_holder["spec"] = payload
        bus.publish_event(kind, payload)

    def _sse(kind: str, payload) -> str:
        return f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"

    def current() -> PlaybackSession | None:
        return state["session"]

    def status_payload() -> dict:
        s = current()
        if s is None:
            return {"server": {"surface": "dashboard", "bus": bus.stats()},
                    "session": {"state": "idle", "is_playback": True,
                                "config": None}}
        full = s.status()
        st = full["session"]
        st["trainer_available"] = full["trainer_available"]
        return {"server": {"surface": "dashboard", "bus": bus.stats()},
                "session": st}

    def _resolve_checkpoint(body: dict) -> Path:
        """Resolve {experiment, checkpoint} under the live root, or a validated
        `path` under an allowed training-output root. Raises PlaybackError."""
        name = body.get("experiment")
        ckpt = str(body.get("checkpoint", "ckpt_final.json"))
        if name:
            if any(b in str(name) for b in ("/", "\\", "..")) or \
                    any(b in ckpt for b in ("/", "\\", "..")):
                raise PlaybackError("experiment and checkpoint must be plain names")
            return live_root / str(name) / ckpt
        raw = body.get("path")
        if not raw:
            raise PlaybackError("provide {experiment, checkpoint} or a path")
        p = Path(str(raw)).resolve()
        if not any(str(p).startswith(str(r.resolve())) for r in allowed_roots):
            raise PlaybackError("checkpoint path is outside the training roots")
        return p

    # --------------------------------------------------------------- routes
    @app.get("/api/rl_playback/checkpoints")
    def pb_checkpoints():
        from .interop import list_experiments
        return list_experiments(live_root)

    @app.post("/api/rl_playback/start")
    def pb_start(body: dict | None = None):
        body = body or {}
        with lock:
            prev = current()
            if prev is not None and prev.busy():
                try:
                    prev.stop()
                except Exception:
                    pass
            try:
                ckpt = _resolve_checkpoint(body)
                sess = PlaybackSession(
                    checkpoint=str(ckpt),
                    stage=body.get("stage", "sustained"),
                    inference=body.get("inference", "deterministic"),
                    speed=float(body.get("speed", 1.0)),
                    name=body.get("name"), root=pb_root,
                    broadcaster=broadcast, frame_sink=bus.publish_route)
            except PlaybackError as exc:
                return JSONResponse(status_code=400, content={
                    "ok": False, "error": {"code": "invalid_config",
                                           "message": str(exc)}})
            except Exception as exc:
                return JSONResponse(status_code=400, content={
                    "ok": False, "error": {"code": "start_failed",
                                           "message": str(exc)}})
            spec_holder["spec"] = None
            state["session"] = sess
            res = sess.start()
        if not res.get("ok"):
            return JSONResponse(status_code=409, content=res)
        return res

    _OPS = {"pause", "resume", "reset", "mode", "speed", "stop"}

    @app.post("/api/rl_playback/{op}")
    def pb_control(op: str, body: dict | None = None):
        if op not in _OPS:
            return JSONResponse(status_code=404, content={
                "ok": False, "error": {"code": "unknown_op",
                                       "message": f"ops: {sorted(_OPS)}"}})
        s = current()
        if s is None:
            return JSONResponse(status_code=409, content={
                "ok": False, "error": {"code": "not_active",
                                       "message": "no playback session started"}})
        res = s.playback_control(op, body or {})
        if not res.get("ok"):
            code = res["error"]["code"]
            status = {"not_running": 409, "not_paused": 409,
                      "invalid_config": 400}.get(code, 400)
            return JSONResponse(status_code=status, content=res)
        return res

    @app.get("/api/rl_playback/status")
    def pb_status():
        return status_payload()

    @app.get("/api/rl_playback/spec")
    def pb_spec():
        spec = spec_holder["spec"]
        if spec is None:
            return JSONResponse(status_code=404, content={
                "available": False,
                "note": "start a playback session to build the rig scene"})
        return spec

    @app.get("/api/rl_playback/stream")
    def pb_stream():
        sub = bus.subscribe()

        def gen():
            try:
                if spec_holder["spec"] is not None:
                    yield _sse("spec", spec_holder["spec"])
                yield _sse("status", status_payload()["session"])
                last_sent = time.monotonic()
                while True:
                    try:
                        sub.cond.acquire()
                        sub.cond.wait(timeout=0.5)
                    finally:
                        sub.cond.release()
                    events, frame, _drops = sub.drain()
                    wrote = False
                    for kind, payload in events:
                        yield _sse(kind, payload)
                        wrote = True
                    if frame is not None:
                        yield _sse("frame", frame)
                        wrote = True
                    now = time.monotonic()
                    if not wrote and now - last_sent > 15.0:
                        yield ": ping\n\n"
                        last_sent = now
                    elif wrote:
                        last_sent = now
            finally:
                bus.unsubscribe(sub)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return {"playback_state": state, "bus": bus}
