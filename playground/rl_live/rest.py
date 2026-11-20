"""FastAPI mount for Live Neural Training on the main dashboard server.

This is the primary integration surface (Jamie: "the project has a full
dashboard it can be added to"). The standalone sidecar
(`-m playground.rl_live.server`) remains as an optional isolated entry
point; both drive the same session/bus/telemetry modules.

    from ..rl_live.rest import register_rl_live_routes    # dashboard/server.py
    register_rl_live_routes(app, cfg)

Endpoints (all under the dashboard origin, same SSE conventions as
/api/stream/events):

  GET  /api/rl_live/status            session + bus state
  GET  /api/rl_live/objectives        objective registry + stage ladder
  GET  /api/rl_live/spec              cached scene spec
  GET  /api/rl_live/stream            SSE (frame|reset|episode_end|iteration|
                                       stage_transition|status|saved|spec|
                                       b3_iteration)
  GET  /api/rl_live/telemetry.json    recent iteration records (own run)
  GET  /api/rl_b3/telemetry           READ-ONLY tail of the running B3 file
  POST /api/rl_live/start|pause|resume|stop|save
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi.responses import JSONResponse, StreamingResponse

from .bus import Bus
from .session import OBJECTIVES, LiveBrainSession
from .telemetry import B3Telemetry, read_records

ROOT = Path(__file__).resolve().parents[2]


def _stage_ladder():
    from playground.rl.curriculum import DEFAULT_STAGES
    return DEFAULT_STAGES


def _b3_default_path(cfg: dict | None) -> Path:
    rl_live_cfg = dict((cfg or {}).get("rl_live") or {})
    p = rl_live_cfg.get("b3_progress")
    if p:
        return Path(p)
    # conventional location of the running B3 experiment (read-only)
    return ROOT / "experiments" / "rl" / "b3_run1" / "progress.jsonl"


def register_rl_live_routes(app, cfg: dict | None = None, *,
                            out_root: Path | None = None,
                            frame_hz: float = 30.0):
    rl_live_cfg = dict((cfg or {}).get("rl_live") or {})
    root = Path(out_root or rl_live_cfg.get(
        "out_root") or ROOT / "experiments" / "rl_live")
    b3_path = None if rl_live_cfg.get("no_b3") else _b3_default_path(cfg)

    bus = Bus(frame_hz=frame_hz)
    spec_holder: dict = {"spec": None}

    def broadcast(kind, payload):
        if kind == "spec":
            spec_holder["spec"] = payload
        bus.publish_event(kind, payload)

    session = LiveBrainSession(root=root, broadcaster=broadcast,
                               frame_sink=bus.publish_route)
    b3 = B3Telemetry(b3_path, bus)

    def _refresh_loop():
        while True:
            try:
                if bus.subscriber_count > 0:
                    b3.refresh()
            except Exception:
                pass
            time.sleep(1.0)

    threading.Thread(target=_refresh_loop, daemon=True,
                     name="rl_live_b3_refresher").start()

    # ------------------------------------------------------------- helpers
    def ok_or(response: dict, code_map=None):
        """Wrap a session result into a JSONResponse with a real status code."""
        if response.get("ok"):
            return response
        code = (code_map or {}).get(response["error"]["code"], 400)
        return JSONResponse(status_code=code, content=response)

    ERROR_CODES = {"busy": 409, "not_running": 409, "not_paused": 409,
                   "not_active": 409, "trainer_unavailable": 409,
                   "invalid_config": 400, "unknown_objective": 400,
                   "save_failed": 409, "nothing_to_save": 409,
                   "save_requires_paused": 409}

    def _sse(kind: str, payload) -> str:
        return f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"

    def status_payload() -> dict:
        full = session.status()
        st = full["session"]
        st["trainer_available"] = full["trainer_available"]
        return {"server": {"surface": "dashboard",
                           "bus": bus.stats(),
                           "root": str(root)},
                "session": st,
                "b3_available": bool(b3.path and b3.path.exists())}

    def objectives_payload() -> dict:
        from .interop import trainer_available
        return {"objectives": [{"id": k, "label": v["label"],
                                "stage_budgets": v["stage_budgets"],
                                "default": k == "walking"}
                               for k, v in OBJECTIVES.items()],
                "stage_ladder": [{"index": i, "id": s.id, "name": s.name,
                                  "description": s.description,
                                  "env_ready": s.env_ready}
                                 for i, s in enumerate(_stage_ladder())],
                "trainer_available": trainer_available(),
                "note": "the first supported objective is walking (the B3 "
                        "ladder); climbing / navigation / obstacle avoidance "
                        "deliberately not offered yet"}

    # ------------------------------------------------------------- routes
    @app.get("/api/rl_live/status")
    def rl_live_status():
        return status_payload()

    @app.get("/api/rl_live/objectives")
    def rl_live_objectives():
        return objectives_payload()

    @app.get("/api/rl_live/spec")
    def rl_live_spec():
        spec = spec_holder["spec"]
        if spec is None:
            return JSONResponse(status_code=404, content={
                "available": False,
                "note": "start a session to build the rig scene from the "
                        "live model"})
        return spec

    @app.get("/api/rl_live/mesh/{name}")
    def rl_live_mesh(name: str):
        """Compiled vertex/face data for one mesh asset (the humanoid_v2 shell).

        Populated as the scene spec is built, so it is available for exactly
        the meshes the current model carries. The name is looked up in that
        table, never used as a path.
        """
        from .frames import cached_mesh, cached_mesh_names
        md = cached_mesh(name)
        if md is None:
            return JSONResponse(status_code=404, content={
                "available": False, "mesh": name,
                "known": cached_mesh_names(),
                "note": "unknown mesh; start a session so the scene spec is "
                        "built from the live model"})
        return md

    @app.get("/api/rl_live/telemetry.json")
    def rl_live_telemetry(limit: int = 200):
        recs = session.recent_iterations(limit=min(500, max(1, limit)))
        return {"records": recs, "session_state": session.state}

    @app.get("/api/rl_b3/telemetry")
    def rl_b3_telemetry(limit: int = 30):
        return b3.snapshot(limit=min(500, max(1, limit)))

    @app.post("/api/rl_live/start")
    def rl_live_start(body: dict | None = None):
        body = body or {}
        return ok_or(session.start(
            objective=body.get("objective", "walking"),
            name=body.get("name"), seed=body.get("seed"),
            total_steps=body.get("total_steps"),
            ppo_overrides=body.get("ppo_overrides"),
            # visual-only; e.g. {"agent_body.visual": "humanoid_v2"}
            env_overrides=body.get("env_overrides")), ERROR_CODES)

    @app.post("/api/rl_live/pause")
    def rl_live_pause():
        return ok_or(session.pause(), ERROR_CODES)

    @app.post("/api/rl_live/resume")
    def rl_live_resume():
        return ok_or(session.resume(), ERROR_CODES)

    @app.post("/api/rl_live/stop")
    def rl_live_stop():
        return ok_or(session.stop(), ERROR_CODES)

    @app.post("/api/rl_live/save")
    def rl_live_save(body: dict | None = None):
        return ok_or(session.save(tag=(body or {}).get("tag")), ERROR_CODES)

    @app.get("/api/rl_live/stream")
    def rl_live_stream():
        sub = bus.subscribe()
        holder = spec_holder

        def gen():
            try:
                if holder["spec"] is not None:
                    yield _sse("spec", holder["spec"])
                yield _sse("status", status_payload()["session"])
                last = session.recent_iterations(limit=1)
                if last:
                    yield _sse("iteration", last[0])
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
                        # every live event is forwarded, spec included: the
                        # session broadcasts it exactly once per start and a
                        # stream connected before the start got no replay
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

    return {"session": session, "bus": bus, "b3": b3}
