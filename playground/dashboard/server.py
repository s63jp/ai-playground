"""HTTP API and dashboard server.

Three surfaces:
  /api/agent/...   lets any external AI (Hermes, a custom model, a human with curl) drive a session;
  /api/...         read-only researcher views of everything recorded, plus memory administration;
  /                the monitoring dashboard (static files, polling the read API).

Watching never interferes: dashboard endpoints only read the database.
Sessions run by the CLI in another process are visible here because both use the same SQLite file.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..agents.adapters import ExternalAdapter
from ..analysis.behaviour import classify_session
from ..config import agent_ids, config_hash, load_config, resolve_path
from ..interface.schema import catalogue
from ..memory import MemoryError_, MemoryStore
from ..session.benchmark import TASKS
from ..session.manager import ensure_agents
from ..session.runner import finish_session, integrity_report, memory_root, run_session, session_summary
from ..storage import Database
from ..world.spec import build_specs

STATIC = Path(__file__).resolve().parent / "static"
AGENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,39}$")


def create_app(cfg: dict | None = None, db: Database | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db = db or Database(resolve_path(cfg, cfg["database"]))
    ensure_agents(cfg, db)
    app = FastAPI(title="AI Experimental Playground", version=__version__)
    app.state.cfg, app.state.db = cfg, db
    live: dict[int, dict] = {}          # external sessions driven through this server
    lock = threading.RLock()

    def mem(agent_id: str) -> MemoryStore:
        known = set(agent_ids(cfg)) | {p.name for p in memory_root(cfg).iterdir() if p.is_dir()}
        if agent_id not in known:
            raise HTTPException(404, f"unknown agent '{agent_id}'")
        return MemoryStore(memory_root(cfg), agent_id, cfg, db=db)

    def sess(sid: int) -> dict:
        row = db.session(sid)
        if not row:
            raise HTTPException(404, f"no session {sid}")
        return row

    # ---------------------------------------------------------------- meta
    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "config_hash": config_hash(cfg), "db": db.path,
                "agents": len(agent_ids(cfg)), "live_external_sessions": len(live)}

    @app.get("/api/tools")
    def tools():
        return {"tools": catalogue()}

    @app.get("/api/world/spec")
    def world_spec():
        specs = [s.describe() for s in build_specs(cfg) if not s.latent]
        return {"objects": specs, "half_extent": cfg["world"]["island"]["half_extent"],
                "sea_level": cfg["world"]["island"]["sea_level"],
                "agent_body": {"width": cfg["agent_body"]["width"], "height": cfg["agent_body"]["height"]},
                "block_size": cfg["world"]["blocks"]["size"],
                "colours": {s.id: s.colour for s in build_specs(cfg)}}

    # ---------------------------------------------------------------- external agent control
    @app.post("/api/agent/sessions")
    def agent_start(body: dict = Body(...)):
        agent_id = body.get("agent_id")
        if not agent_id or not AGENT_ID_RE.match(str(agent_id)):
            raise HTTPException(400, "agent_id is required: 1-40 characters of a-z, 0-9, '_' or '-'")
        mode = body.get("mode", "exploration")
        task = TASKS.get(body.get("task_id")) if body.get("task_id") else None
        if body.get("task_id") and not task:
            raise HTTPException(400, f"unknown task; available: {sorted(TASKS)}")
        if mode == "benchmark" and not task:
            raise HTTPException(400, "benchmark mode needs a task_id")
        m = MemoryStore(memory_root(cfg), agent_id, cfg, db=db)
        c = json.loads(json.dumps(cfg))
        if body.get("max_steps"):
            c["session"]["max_steps"] = int(body["max_steps"])
        if body.get("minutes"):
            c["session"]["duration_minutes"] = float(body["minutes"])
        ad = ExternalAdapter()
        ad.model = str(body.get("model") or "external")
        info = run_session(c, db, agent_id, ad, mode=mode, memory=m, label=body.get("label"),
                           scene=task.scene if task else None, task_prompt=task.prompt if task else None)
        env = info["environment"]
        with lock:
            live[env.sid] = {"env": env, "adapter": ad, "task": task, "cfg": c}
        return {"session_id": env.sid, "session_number": env.session_number, "system_prompt": info["system"],
                "opening_message": info["opening"], "tools": [t for t in catalogue() if t["name"] in info["tools"]],
                "call_endpoint": f"/api/agent/sessions/{env.sid}/call",
                "finish_endpoint": f"/api/agent/sessions/{env.sid}/finish"}

    def _live(sid: int) -> dict:
        with lock:
            item = live.get(sid)
        if item is None:
            raise HTTPException(404, f"session {sid} is not an open external session on this server")
        return item

    @app.post("/api/agent/sessions/{sid}/call")
    def agent_call(sid: int, body: dict = Body(...)):
        item = _live(sid)
        if "tool" not in body:
            raise HTTPException(400, "body must be {\"tool\": name, \"args\": {...}}")
        if body.get("say"):
            db.add_message(sid, item["env"].agent_calls + 1, "assistant", str(body["say"])[:4000])
        return item["env"].call(body["tool"], body.get("args") or {}, source="agent")

    @app.post("/api/agent/sessions/{sid}/say")
    def agent_say(sid: int, body: dict = Body(...)):
        item = _live(sid)
        db.add_message(sid, item["env"].agent_calls + 1, "assistant", str(body.get("text", ""))[:4000])
        return {"ok": True}

    @app.post("/api/agent/sessions/{sid}/finish")
    def agent_finish(sid: int, body: dict = Body(default={})):
        item = _live(sid)
        task = item["task"]

        def verify(env):
            r = task.verify(env.world, db.events(env.sid, limit=10 ** 9))
            r["task_id"] = task.id
            return r
        summary = finish_session(item["cfg"], db, item["env"], item["adapter"], item["env"].agent_calls, {},
                                 body.get("reason") or item["env"].end_reason or "external_finish", None,
                                 verify if task else None)
        with lock:
            live.pop(sid, None)
        return summary

    # ---------------------------------------------------------------- researcher views
    @app.get("/api/agents")
    def agents():
        out = []
        for a in agent_ids(cfg):
            m = MemoryStore(memory_root(cfg), a, cfg)
            last = db.one("SELECT id, status, session_number, mode, started_at FROM sessions WHERE agent_id=? "
                          "ORDER BY id DESC LIMIT 1", (a,))
            out.append({**m.stats(), "last_session": last,
                        "exploration_sessions": db.one("SELECT COUNT(*) c FROM sessions WHERE agent_id=? AND "
                                                       "mode='exploration'", (a,))["c"]})
        return {"agents": out}

    @app.get("/api/agents/{agent_id}/memory")
    def agent_memory(agent_id: str):
        m = mem(agent_id)
        files = {n: (m.dir / f"{n}.md").read_text(encoding="utf-8")
                 for n in ("diary", "handbook", "mistakes", "unfinished_goals")}
        return {"agent_id": agent_id, "stats": m.stats(), "files": files,
                "discoveries": m.discoveries["items"], "experiments": m.experiments["items"],
                "goals": m.state["goals"], "programs": m.list_programs(), "items": m.all_items(),
                "history": sorted(p.name for p in (m.dir / "session_history").iterdir()),
                "audit": db.q("SELECT * FROM memory_audit WHERE agent_id=? ORDER BY id DESC LIMIT 50", (agent_id,))}

    @app.post("/api/agents/{agent_id}/memory/{item_id}/{op}")
    def memory_op(agent_id: str, item_id: str, op: str, body: dict = Body(default={})):
        m = mem(agent_id)
        reason = body.get("reason")
        try:
            if op == "remove":
                return {"removed": m.remove_item(item_id, reason)}
            if op == "challenge":
                return {"discovery": m.challenge(item_id, reason or "challenged from the dashboard")}
            if op == "mark_unverified":
                return {"discovery": m.mark_unverified(item_id, reason)}
        except MemoryError_ as e:
            raise HTTPException(400, e.message) from None
        raise HTTPException(400, "op must be remove, challenge or mark_unverified")

    @app.get("/api/sessions")
    def sessions(agent_id: str | None = None, mode: str | None = None, limit: int = 200):
        sql, p = ("SELECT id, agent_id, mode, session_number, label, adapter, model, status, end_reason, "
                  "started_at, ended_at, sim_time, action_count, wall_seconds, task_id, condition, "
                  "benchmark_run_id FROM sessions WHERE 1=1"), []
        if agent_id:
            sql += " AND agent_id=?"
            p.append(agent_id)
        if mode:
            sql += " AND mode=?"
            p.append(mode)
        return {"sessions": db.q(sql + " ORDER BY id DESC LIMIT ?", [*p, limit])}

    @app.get("/api/sessions/{sid}")
    def session_detail(sid: int):
        sess(sid)
        s = session_summary(db, sid)
        s.pop("config", None)
        s["frames"] = db.frame_count(sid)
        return s

    @app.get("/api/sessions/{sid}/actions")
    def session_actions(sid: int, after: int = 0, limit: int = 500):
        sess(sid)
        return {"actions": db.actions(sid, after=after, limit=limit)}

    @app.get("/api/sessions/{sid}/events")
    def session_events(sid: int, after: int = 0, limit: int = 1000, types: str | None = None):
        sess(sid)
        return {"events": db.events(sid, after=after, limit=limit, types=types.split(",") if types else None)}

    @app.get("/api/sessions/{sid}/frames")
    def session_frames(sid: int, start: int = 0, limit: int = 2000):
        sess(sid)
        return {"frames": db.frames(sid, start, limit), "total": db.frame_count(sid)}

    @app.get("/api/sessions/{sid}/messages")
    def session_messages(sid: int, after: int = 0):
        sess(sid)
        return {"messages": db.messages(sid, after_id=after)}

    @app.get("/api/sessions/{sid}/integrity")
    def session_integrity(sid: int):
        sess(sid)
        return integrity_report(db, sid)

    @app.get("/api/sessions/{sid}/behaviour")
    def session_behaviour(sid: int):
        sess(sid)
        return classify_session(db, sid)

    @app.get("/api/sessions/{sid}/records")
    def session_records(sid: int, kind: str | None = None):
        sess(sid)
        return {"records": db.records(kind=kind, session_id=sid)}

    @app.get("/api/live")
    def live_view():
        rows = db.q("SELECT id, agent_id, mode, session_number, task_id, started_at, action_count, sim_time "
                    "FROM sessions WHERE status='running' ORDER BY id DESC LIMIT 20")
        stale = time.time() - 600
        out = []
        for r in rows:
            last = db.one("SELECT MAX(wall_ts) w FROM actions WHERE session_id=?", (r["id"],))
            r["last_activity"] = last["w"] if last else None
            r["stale"] = (r["last_activity"] or r["started_at"] or 0) < stale
            out.append(r)
        return {"running": out}

    @app.get("/api/analytics")
    def analytics(agent_id: str | None = None):
        where, p = ("WHERE agent_id=?", [agent_id]) if agent_id else ("", [])
        timeline = db.q(f"SELECT id, agent_id, mode, session_number, sim_time, action_count, end_reason, "
                        f"started_at, summary_json FROM sessions {where} ORDER BY id", p)
        for t in timeline:
            s = json.loads(t.pop("summary_json") or "null") or {}
            iv = (s.get("integrity") or {}).get("what_was_independently_verified", {})
            t.update({"behaviour": s.get("behaviour"), "failed_actions": (s.get("integrity") or {})
                      .get("what_the_agent_did", {}).get("failed_actions"),
                      "experiments_ok": iv.get("experiments_consistent_with_engine"),
                      "experiments_bad": (iv.get("experiments_contradicted_by_engine") or 0) +
                      (iv.get("experiments_without_valid_evidence") or 0),
                      "claims_verified": iv.get("claims_verified"), "claims_rejected": iv.get("claims_rejected"),
                      "discoveries": (s.get("memory") or {}).get("discoveries")})
        return {"timeline": timeline}

    @app.get("/api/compare")
    def compare_agents():
        rows = []
        for a in agent_ids(cfg):
            m = MemoryStore(memory_root(cfg), a, cfg)
            st = m.stats()
            beh = {"self_directed_goals": 0, "pretend_play": 0, "repetition_with_variation": 0,
                   "activity_for_its_own_sake": 0}
            for r in db.records(kind="behaviour", agent_id=a):
                for k in beh:
                    beh[k] += r["data"][k]["count"]
            bench = db.q("SELECT metrics_json FROM benchmark_results WHERE agent_id=? AND grp='A'", (a,))
            done = [json.loads(b["metrics_json"])["completed"] for b in bench]
            claims = db.records(kind="claim", agent_id=a)
            st.update({"behaviour": beh, "benchmark_runs": len(done),
                       "benchmark_completion_rate": round(sum(done) / len(done), 3) if done else None,
                       "claims": len(claims), "claims_verified": sum(1 for c in claims if c["data"]["verified"]),
                       "memory_quality": round(st["experiments_verified"] / st["experiments"], 3)
                       if st["experiments"] else None,
                       "discovery_statements": [{"id": d["id"], "status": d["status"], "category": d.get("category"),
                                                 "statement": d["statement"]} for d in m.discoveries["items"]]})
            rows.append(st)
        return {"agents": rows}

    @app.get("/api/benchmarks")
    def benchmarks():
        runs = db.q("SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT 50")
        for r in runs:
            r["params"] = json.loads(r.pop("params_json") or "null")
            r["summary"] = json.loads(r.pop("summary_json") or "null")
            res = db.q("SELECT * FROM benchmark_results WHERE run_id=? ORDER BY id", (r["id"],))
            r["results"] = [{**{k: x[k] for k in ("session_id", "agent_id", "grp", "condition", "task_id")},
                             **json.loads(x["metrics_json"])} for x in res]
        return {"runs": runs}

    @app.get("/api/benchmarks/tasks")
    def bench_tasks():
        return {"tasks": [{"id": t.id, "title": t.title, "prompt": t.prompt, "max_steps": t.max_steps}
                          for t in TASKS.values()]}

    # ---------------------------------------------------------------- ML lab
    ml_jobs: dict[str, dict] = {}
    ml_lock = threading.Lock()
    ml_running: dict[str, bool] = {"busy": False}  # single job at a time

    def _all_models_fallback() -> list[dict]:
        """Fallback when WP3's registry.all_models is not yet available."""
        from ..ml.registry import Registry, NAME_RE
        out = []
        # Researcher models
        reg = Registry(cfg)
        for m in reg.list_models():
            out.append({**m, "scope": "researcher", "agent_id": None})
        # Agent models - scan memory_root/<agent_id>/nn/models/*/*/v*.json
        mroot = memory_root(cfg)
        if mroot.exists():
            for agent_dir in mroot.iterdir():
                if not agent_dir.is_dir():
                    continue
                nn_models = agent_dir / "nn" / "models"
                if not nn_models.exists():
                    continue
                for p in sorted(nn_models.glob("*/*/v*.json")):
                    try:
                        b = json.loads(p.read_text(encoding="utf-8"))
                        owner = p.parent.parent.name
                        nm = p.parent.name
                        m = b["metrics"]
                        out.append({
                            "owner": owner, "name": nm, "version": b["version"],
                            "target": b["spec"]["target"], "inputs": b["spec"]["inputs"],
                            "hidden": b["spec"]["hidden"], "parameters": b["parameters"],
                            "test_rmse": m["test"]["rmse"], "test_r2": m["test"]["r2"],
                            "linear_rmse": m["baseline_linear"]["rmse"],
                            "beats_linear": m["beats_linear_baseline"],
                            "dataset_rows": b["dataset"].get("rows"),
                            "created_at": b.get("created_at"),
                            "closed_loop": b.get("closed_loop"),
                            "scope": "agent", "agent_id": agent_dir.name
                        })
                    except Exception:
                        pass
        return out

    def _get_all_models() -> list[dict]:
        """Try WP3's registry.all_models, fall back to scanning."""
        try:
            from ..ml.registry import all_models
            return all_models(cfg)
        except (ImportError, AttributeError):
            return _all_models_fallback()

    @app.get("/api/ml/status")
    def ml_status():
        from ..ml import torch_status
        return torch_status()

    @app.get("/api/ml/columns")
    def ml_columns():
        from ..ml.features import INPUT_COLUMNS, TARGET_COLUMNS
        return {"inputs": INPUT_COLUMNS, "targets": TARGET_COLUMNS}

    @app.get("/api/ml/datasets")
    def ml_datasets():
        from ..ml.registry import Registry
        reg = Registry(cfg)
        return {"datasets": reg.dataset_info()}

    @app.get("/api/ml/models")
    def ml_models():
        models = _get_all_models()
        # Never return state_dict
        for m in models:
            m.pop("state_dict", None)
        return {"models": models}

    @app.get("/api/ml/models/{scope}/{owner}/{name}")
    def ml_model_detail(scope: str, owner: str, name: str, version: int | None = None):
        from ..ml.registry import Registry
        from ..ml import MLError
        # Find the right registry based on scope
        if scope == "researcher":
            reg = Registry(cfg)
        elif scope == "agent":
            agent_nn_root = memory_root(cfg) / owner / "nn"
            if not agent_nn_root.exists():
                raise HTTPException(404, f"No agent registry for '{owner}'.")
            reg = Registry(cfg, root=agent_nn_root)
        else:
            raise HTTPException(400, f"scope must be 'researcher' or 'agent', got '{scope}'.")
        try:
            model = reg.load_model(owner, name, version)
        except MLError as e:
            if e.code == "unknown_model":
                raise HTTPException(404, e.message) from None
            raise HTTPException(400, e.message) from None
        # Never return state_dict
        result = {k: v for k, v in model.items() if k != "state_dict"}
        return result

    @app.post("/api/ml/train")
    def ml_train(body: dict = Body(...)):
        from ..ml.registry import Registry, NAME_RE
        from ..ml.nets import validate_spec, validate_train
        from ..ml import MLError
        # Check not busy
        with ml_lock:
            if ml_running["busy"]:
                raise HTTPException(409, "A training or evaluation job is already running.")
            ml_running["busy"] = True
        try:
            dataset_name = body.get("dataset")
            spec = body.get("spec", {})
            train_cfg = body.get("train", {})
            # Force device to cpu if not specified or auto
            if train_cfg.get("device") in (None, "auto"):
                train_cfg["device"] = "cpu"
            # Validate
            try:
                spec = validate_spec(spec)
                train_cfg = validate_train(train_cfg)
            except MLError as e:
                with ml_lock:
                    ml_running["busy"] = False
                raise HTTPException(400, e.message) from None
            # Load dataset
            reg = Registry(cfg)
            try:
                ds = reg.load_dataset(dataset_name)
            except MLError as e:
                with ml_lock:
                    ml_running["busy"] = False
                raise HTTPException(400, e.message) from None
            # Create job
            job_id = str(uuid.uuid4())[:8]
            ml_jobs[job_id] = {"status": "queued", "spec": spec, "dataset": dataset_name}

            def run_training():
                try:
                    ml_jobs[job_id]["status"] = "running"
                    from ..ml.nets import train, summarise
                    result = train(spec, ds["rows"], train_cfg)
                    saved = reg.save_model("researcher", result, ds)
                    summary = summarise(result)
                    summary["saved"] = saved
                    ml_jobs[job_id]["status"] = "done"
                    ml_jobs[job_id]["result_summary"] = summary
                except Exception as e:
                    ml_jobs[job_id]["status"] = "error"
                    ml_jobs[job_id]["error"] = str(e)
                finally:
                    with ml_lock:
                        ml_running["busy"] = False

            t = threading.Thread(target=run_training, daemon=True)
            t.start()
            return {"job_id": job_id}
        except HTTPException:
            raise
        except Exception as e:
            with ml_lock:
                ml_running["busy"] = False
            raise HTTPException(500, str(e)) from None

    @app.get("/api/ml/jobs/{job_id}")
    def ml_job_status(job_id: str):
        if job_id not in ml_jobs:
            raise HTTPException(404, f"No job '{job_id}'.")
        return ml_jobs[job_id]

    @app.post("/api/ml/evaluate")
    def ml_evaluate(body: dict = Body(...)):
        from ..ml.registry import Registry
        from ..ml import MLError
        # Check not busy
        with ml_lock:
            if ml_running["busy"]:
                raise HTTPException(409, "A training or evaluation job is already running.")
            ml_running["busy"] = True
        try:
            owner = body.get("owner")
            name = body.get("name")
            version = body.get("version")
            kind = body.get("kind", "fresh")
            n = body.get("n", 100)
            if kind not in ("fresh", "closed_loop"):
                with ml_lock:
                    ml_running["busy"] = False
                raise HTTPException(400, "kind must be 'fresh' or 'closed_loop'.")
            # Load model (researcher registry only for now)
            reg = Registry(cfg)
            try:
                model = reg.load_model(owner, name, version)
            except MLError as e:
                with ml_lock:
                    ml_running["busy"] = False
                raise HTTPException(400, e.message) from None

            job_id = str(uuid.uuid4())[:8]
            ml_jobs[job_id] = {"status": "queued", "kind": kind, "model": f"{owner}/{name}"}

            def run_eval():
                try:
                    ml_jobs[job_id]["status"] = "running"
                    from ..ml.evaluate import fresh_engine_test, closed_loop_height
                    if kind == "fresh":
                        result = fresh_engine_test(cfg, model, n=n)
                    else:
                        # Get dataset rows for linear baseline
                        try:
                            ds = reg.load_dataset(body.get("dataset") or model["dataset"].get("source", ""))
                            linear_rows = ds["rows"]
                        except Exception:
                            linear_rows = None
                        result = closed_loop_height(cfg, model, linear_rows=linear_rows)
                    # Attach result to model
                    actual_version = version or model["version"]
                    try:
                        reg.attach(owner, name, actual_version, kind if kind == "closed_loop" else "fresh_check", result)
                    except Exception:
                        pass  # attachment is optional
                    ml_jobs[job_id]["status"] = "done"
                    ml_jobs[job_id]["result_summary"] = result
                except Exception as e:
                    ml_jobs[job_id]["status"] = "error"
                    ml_jobs[job_id]["error"] = str(e)
                finally:
                    with ml_lock:
                        ml_running["busy"] = False

            t = threading.Thread(target=run_eval, daemon=True)
            t.start()
            return {"job_id": job_id}
        except HTTPException:
            raise
        except Exception as e:
            with ml_lock:
                ml_running["busy"] = False
            raise HTTPException(500, str(e)) from None

    # ---------------------------------------------------------------- dashboard
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.exception_handler(Exception)
    async def crash(request, exc):  # noqa: ANN001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app
