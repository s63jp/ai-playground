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
import copy
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
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

    # Model-gateway surface (provider discovery, resource monitoring, and — later stages —
    # the OpenAI-compatible /v1/chat/completions passthrough + request monitor + queue).
    # Mounted on THIS app so there is no second process or extra proxy.
    from ..gateway.api import register_gateway_routes
    register_gateway_routes(app, cfg, db)

    # Register player API routes
    from .player_api import router as player_router
    app.include_router(player_router)

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
        _refresh_datasets_async()   # the log just grew; keep behaviour datasets current
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
        status = body.get("status")
        actor = body.get("actor") or "researcher"
        try:
            if op == "remove":
                return {"removed": m.remove_item(item_id, reason, actor=actor)}
            if op == "challenge":
                return {"discovery": m.challenge(item_id, reason or "challenged from the dashboard", actor=actor)}
            if op == "mark_unverified":
                return {"discovery": m.mark_unverified(item_id, reason, actor=actor)}
            if op == "resolve_challenge":
                return {"discovery": m.resolve_challenge(item_id, reason, actor=actor,
                                                        status=status)}
            if op == "reverify":
                return {"discovery": m.reverify(item_id, reason, actor=actor, status=status)}
            if op == "set_status":
                if not status:
                    raise HTTPException(400, "set_status needs a 'status' in the body")
                return {"discovery": m.set_status(item_id, status, reason, actor=actor)}
        except MemoryError_ as e:
            raise HTTPException(400, e.message) from None
        raise HTTPException(400, "op must be remove, challenge, mark_unverified, resolve_challenge, "
                                 "reverify or set_status")

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

    @app.get("/api/arena")
    def arena_matches(limit: int = 50):
        """Arena matches (M5): every recorded session whose mode is an arena mode, grouped
        into match RUNS (sessions of the same label that started within 15 s of the run's
        anchor) so each duel/coop event can be opened as one unit."""
        rows = db.q("SELECT id, agent_id, mode, session_number, label, adapter, model, status, "
                    "end_reason, started_at, ended_at, sim_time, action_count, wall_seconds "
                    "FROM sessions WHERE mode LIKE 'arena%' ORDER BY id DESC LIMIT ?", [limit * 4])
        runs: list = []
        for r in rows:
            anchor = (r["label"] or "arena", r["started_at"] or 0.0)
            if runs and runs[-1]["_label"] == r["label"] \
                    and (runs[-1]["_anchor"] - anchor[1]) <= 15.0:
                runs[-1]["sessions"].append(r)
            else:
                runs.append({"_label": r["label"] or "arena", "_anchor": anchor[1],
                             "label": r["label"] or "arena", "sessions": [r]})
        matches = []
        for run in runs[:limit]:
            ss = sorted(run["sessions"], key=lambda s: s["id"])
            matches.append({
                "label": run["label"],
                "started_at": min(s["started_at"] or 0.0 for s in ss),
                "status": ("completed" if all(s["status"] == "completed" for s in ss)
                           else "running" if any(s["status"] != "completed" for s in ss) else "unknown"),
                "modes": sorted({s["mode"] for s in ss}),
                "sessions": ss,
            })
        return {"matches": matches}

    @app.get("/api/sessions/{sid}")
    def session_detail(sid: int):
        sess(sid)
        s = session_summary(db, sid)
        s["arena_context"] = (s.pop("config", None) or {}).get("arena")
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

    @app.post("/api/sessions/{sid}/stop")
    def stop_session(sid: int):
        from ..session.runner import request_stop_session, finish_session
        row = db.session(sid)
        if not row:
            raise HTTPException(404, f"no session {sid}")
        if row["status"] != "running":
            raise HTTPException(409, f"session {sid} is not running")
        # External sessions driven through this server are finished directly.
        with lock:
            item = live.pop(sid, None)
        if item is not None:
            task = item.get("task")
            def verify(env):
                return task.verify(env.world, db.events(env.sid, limit=10 ** 9)) if task else None
            finish_session(item["cfg"], db, item["env"], item["adapter"], item["env"].agent_calls, {},
                           "stopped_by_user", None, verify if task else None)
            _refresh_datasets_async()   # log grew; refresh datasets
            return {"ok": True, "stopped": True, "note": "external session finished"}
        # In-process sessions get a stop signal that the runner checks between turns.
        stopped = request_stop_session(sid)
        return {"ok": True, "stopped": stopped, "note": "stop signal sent" if stopped else "session not currently running in this process"}

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

    def _ml_registry(scope: str, owner: str):
        """Resolve the right registry for a scope/owner, raising clean HTTP errors (audit 2026-11-13)."""
        from ..ml.registry import registry_for
        from ..ml import MLError
        scope = scope or "researcher"
        owner = owner or "researcher"
        if scope not in ("researcher", "agent"):
            raise HTTPException(400, f"scope must be 'researcher' or 'agent', got '{scope}'.")
        if scope == "agent":
            known = set(agent_ids(cfg)) | {p.name for p in memory_root(cfg).iterdir() if p.is_dir()} \
                if memory_root(cfg).exists() else set(agent_ids(cfg))
            if owner not in known:
                raise HTTPException(404, f"unknown agent '{owner}'")
        try:
            return registry_for(cfg, scope, owner)
        except MLError as e:
            raise HTTPException(400, e.message) from None

    @app.get("/api/ml/status")
    def ml_status():
        from ..ml import torch_status
        return torch_status()

    @app.get("/api/ml/columns")
    def ml_columns(dataset: str | None = None, scope: str = "researcher", owner: str = "researcher"):
        from ..ml.datasets import vocab_from_manifest
        from ..ml.features import INPUT_COLUMNS, TARGET_COLUMNS
        if dataset:
            reg = _ml_registry(scope, owner)
            try:
                ds = reg.load_dataset(dataset)
            except Exception as e:
                raise HTTPException(400, f"Could not load dataset: {e}") from None
            in_cols, tgt_cols = vocab_from_manifest(ds.get("manifest", {}))
            if in_cols is None or tgt_cols is None:
                in_cols, tgt_cols = INPUT_COLUMNS, TARGET_COLUMNS
            return {"inputs": in_cols, "targets": tgt_cols, "dataset": dataset}
        return {"inputs": INPUT_COLUMNS, "targets": TARGET_COLUMNS}

    @app.get("/api/ml/datasets")
    def ml_datasets(scope: str = "researcher", owner: str = "researcher"):
        reg = _ml_registry(scope, owner)
        return {"scope": scope, "owner": owner, "datasets": reg.dataset_info()}

    @app.get("/api/ml/datasets/{name}/quality")
    def ml_dataset_quality(name: str, scope: str = "researcher", owner: str = "researcher"):
        """Quality report for one dataset: rows/age, per-tool coverage from the live log, class balance
        on any binary target, warnings (0 rows, below training minimum, class skew) and hash lineage."""
        reg = _ml_registry(scope, owner)
        try:
            ds = reg.load_dataset(name)
        except Exception as e:
            raise HTTPException(400, f"Could not load dataset: {e}") from None
        rows = ds.get("rows") or []
        m = ds.get("manifest") or {}
        from ..ml.datasets import DATASETS
        kind = m.get("kind")
        s = DATASETS.get(kind)
        tools = list(s.tools) if s else []
        coverage = {}
        for t in tools:
            c = db.one("SELECT COUNT(*) c FROM actions WHERE tool=?", (t,))
            coverage[t] = c["c"] if c else 0
        # Class balance on a binary target column ({0.0, 1.0}). Prefer a column that actually has both
        # classes, and prefer 'success'-style columns, so a constant measure isn't mistaken for the
        # classifier target.
        balance = None
        tgt_cols = list(m.get("targets") or []) if isinstance(m.get("targets"), list) else list(m.get("targets") or {})
        best = None
        for col in tgt_cols:
            values = [r.get(col) for r in rows if r.get(col) is not None]
            if not rows or not values or not set(values) <= {0.0, 1.0}:
                continue
            n1 = sum(1 for v in values if v == 1.0)
            both = 0 < n1 < len(values)
            prefer = (4 if both else 0) + (2 if "success" in col.lower() else 0) + 1
            if best is None or prefer > best[0]:
                best = (prefer, col, n1, len(values) - n1)
        if best:
            balance = {"column": best[1], "class_1": best[2], "class_0": best[3]}
        warnings = []
        if not rows:
            warnings.append("0 rows — nothing usable for training; run sessions that log this behaviour")
        elif len(rows) < 20:
            warnings.append(f"only {len(rows)} rows — below the 20-row training minimum, keep logging")
        if balance:
            minority = min(balance["class_1"], balance["class_0"])
            if len(rows) and minority < max(2, 0.1 * len(rows)):
                skew = minority / len(rows)
                warnings.append(f"class skew: {balance['column']} is {balance['class_1']}/{balance['class_0']} "
                                f"(minority {skew:.0%}) — the model may learn to always answer the majority")
        age_min = None
        if m.get("updated_at"):
            age_min = max(0.0, (time.time() - m["updated_at"]) / 60)
        stale = age_min is not None and age_min > 60
        return {"name": name, "kind": kind, "rows": len(rows), "updated_at": m.get("updated_at"),
                "age_minutes": None if age_min is None else round(age_min, 1), "stale": stale,
                "coverage": coverage, "balance": balance, "warnings": warnings,
                "rows_sha256": m.get("rows_sha256"), "source": m.get("source"),
                "targets": tgt_cols, "inputs": list(m.get("inputs") or []) if isinstance(m.get("inputs"), list)
                else list(m.get("inputs") or {})}

    @app.get("/api/ml/leaderboard")
    def ml_leaderboard():
        from ..ml import leaderboard as lb
        return {"entries": lb.entries(cfg, limit=100), "by_model": lb.by_model(cfg)}

    @app.post("/api/ml/generate")
    def ml_generate(body: dict = Body(...)):
        """Generate a dataset by running the real MuJoCo world, saved to the chosen registry.

        kind: throw (default; force/angle -> flight outcomes), jump (carried load x take-off
        platform -> apex), climb (ledge x gap x load -> height gained). Each generator emits rows
        in the same schema as that behaviour's log-built datasets. Background job (N can be large).
        """
        from ..ml.data import generate_climb_dataset, generate_jump_dataset, generate_throw_dataset
        from ..ml.registry import NAME_RE
        from ..ml import MLError
        name = body.get("name")
        if not name or not NAME_RE.match(name):
            raise HTTPException(400, "name must be 1-40 chars of letters, digits, '_' or '-'.")
        kind = str(body.get("kind") or "throw")
        gens = {"throw": generate_throw_dataset, "jump": generate_jump_dataset,
                "climb": generate_climb_dataset}
        if kind not in gens:
            raise HTTPException(400, f"unknown engine kind '{kind}'; available: {sorted(gens)}")
        try:
            n = int(body.get("n", 1500) if body.get("n") is not None else 1500)
            seed = int(body.get("seed", 7))
        except (TypeError, ValueError):
            raise HTTPException(400, "n and seed must be integers.") from None
        if not (10 <= n <= 20000):
            raise HTTPException(400, "n must be between 10 and 20000.")
        scope = body.get("scope", "researcher")
        owner = body.get("owner", "researcher")
        reg = _ml_registry(scope, owner)
        with ml_lock:
            if ml_running["busy"]:
                raise HTTPException(409, "A training, evaluation or generation job is already running.")
            ml_running["busy"] = True
        job_id = str(uuid.uuid4())[:8]
        ml_jobs[job_id] = {"status": "queued", "kind": "generate", "dataset": name, "n": n,
                           "dataset_kind": kind}

        def run_generate():
            def prog(done, of):
                ml_jobs[job_id]["progress"] = {"done": done, "of": of}
            try:
                ml_jobs[job_id]["status"] = "running"
                ds = gens[kind](cfg, n, seed=seed, progress=prog)
                info = reg.save_dataset(name, ds)
                ml_jobs[job_id]["status"] = "done"
                ml_jobs[job_id]["result_summary"] = {**info, "scope": scope, "owner": owner,
                                                     "requested": n,
                                                     "failed": ds["manifest"].get("failed", 0)}
            except MLError as e:
                ml_jobs[job_id]["status"] = "error"
                ml_jobs[job_id]["error"] = str(getattr(e, "message", e))
            except Exception as e:  # noqa: BLE001
                ml_jobs[job_id]["status"] = "error"
                ml_jobs[job_id]["error"] = str(e)
            finally:
                with ml_lock:
                    ml_running["busy"] = False

        threading.Thread(target=run_generate, daemon=True).start()
        return {"job_id": job_id}

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
            scope = body.get("scope", "researcher")
            owner = body.get("owner", "researcher")
            # Force device to cpu if not specified or auto
            if train_cfg.get("device") in (None, "auto"):
                train_cfg["device"] = "cpu"
            # Resolve registry for the chosen owner (researcher or a specific agent) and load the
            # dataset from THAT registry so an agent trains on its own data (audit 2026-11-13).
            try:
                reg = _ml_registry(scope, owner)
            except HTTPException:
                with ml_lock:
                    ml_running["busy"] = False
                raise
            try:
                ds = reg.load_dataset(dataset_name)
            except MLError as e:
                with ml_lock:
                    ml_running["busy"] = False
                raise HTTPException(400, e.message) from None
            # The dataset carries its own column vocabulary (climb/jump/affordance/...); throw datasets
            # use the default. Validate the spec against THAT vocabulary (audit 2026-11-13).
            from ..ml.datasets import vocab_from_manifest
            in_cols, tgt_cols = vocab_from_manifest(ds.get("manifest", {}))
            try:
                spec = validate_spec(spec, in_cols, tgt_cols)
                train_cfg = validate_train(train_cfg)
            except MLError as e:
                with ml_lock:
                    ml_running["busy"] = False
                raise HTTPException(400, e.message) from None
            # Create job
            job_id = str(uuid.uuid4())[:8]
            ml_jobs[job_id] = {"status": "queued", "spec": spec, "dataset": dataset_name,
                               "scope": scope, "owner": owner}

            def run_training():
                try:
                    ml_jobs[job_id]["status"] = "running"
                    from ..ml.nets import train, summarise
                    result = train(spec, ds["rows"], train_cfg, in_cols, tgt_cols)
                    saved = reg.save_model(owner, result, ds)
                    summary = summarise(result)
                    summary["saved"] = saved
                    summary["scope"], summary["owner"] = scope, owner
                    ml_jobs[job_id]["status"] = "done"
                    ml_jobs[job_id]["result_summary"] = summary
                    # Leaderboard: one persistent profile entry per finished training run.
                    try:
                        from ..ml import leaderboard as lb
                        mets = result["metrics"]
                        lb.record(cfg, {
                            "source": "train", "scope": scope, "owner": owner,
                            "name": spec.get("name", "?"), "version": (saved or {}).get("version"),
                            "dataset": dataset_name, "dataset_rows": len(ds["rows"]),
                            "target": spec.get("target"), "parameters": result.get("parameters"),
                            "test_rmse": mets["test"]["rmse"], "test_r2": mets["test"]["r2"],
                            "test_mae": mets["test"]["mae"],
                            "beats_linear": bool(mets["beats_linear_baseline"]),
                            "beats_mean": bool(mets["beats_mean_baseline"]),
                            "linear_rmse": mets["baseline_linear"]["rmse"],
                            "epochs_run": result.get("epochs_run"), "device": result.get("device"),
                        })
                    except Exception:  # leaderboard must never break training
                        pass
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
            # Load the model from the right scope so agent-owned networks evaluate too (audit 2026-11-13).
            scope = body.get("scope", "researcher" if owner == "researcher" else "agent")
            try:
                reg = _ml_registry(scope, owner)
            except HTTPException:
                with ml_lock:
                    ml_running["busy"] = False
                raise
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

    @app.post("/api/ml/controller")
    def ml_controller(body: dict = Body(...)):
        """Run a trained network AS an agent's brain on a benchmark task, judged by the task's own
        verifier (never the agent). This is the 'test my custom network in the world' button
        (audit 2026-11-13). owner='researcher' or an agent id; the network is loaded from that scope.
        """
        from ..ml.registry import NAME_RE
        from ..ml import MLError
        from ..agents.adapters import make_neural_adapter, AdapterError
        from ..session.benchmark import run_controller_benchmark, TASKS
        owner = body.get("owner", "researcher")
        name = body.get("name")
        version = body.get("version")
        task_id = body.get("task_id", "throw_height")
        if not name:
            raise HTTPException(400, "name (model name) is required.")
        if task_id not in TASKS:
            raise HTTPException(400, f"unknown task_id; available: {sorted(TASKS)}")
        with ml_lock:
            if ml_running["busy"]:
                raise HTTPException(409, "A training, evaluation or generation job is already running.")
            ml_running["busy"] = True
        job_id = str(uuid.uuid4())[:8]
        ml_jobs[job_id] = {"status": "queued", "kind": "controller",
                           "model": f"{owner}/{name}", "task_id": task_id}

        def run_controller():
            try:
                ml_jobs[job_id]["status"] = "running"
                adapter = make_neural_adapter(cfg, owner=owner, name=name, version=version)
                result = run_controller_benchmark(cfg, db, adapter, task_id=task_id,
                                                  label=f"dashboard:{owner}/{name}")
                ml_jobs[job_id]["status"] = "done"
                ml_jobs[job_id]["result_summary"] = result
            except (MLError, AdapterError) as e:
                ml_jobs[job_id]["status"] = "error"
                ml_jobs[job_id]["error"] = getattr(e, "message", str(e))
            except Exception as e:  # noqa: BLE001
                ml_jobs[job_id]["status"] = "error"
                ml_jobs[job_id]["error"] = str(e)
            finally:
                with ml_lock:
                    ml_running["busy"] = False

        threading.Thread(target=run_controller, daemon=True).start()
        return {"job_id": job_id}

    @app.post("/api/ml/closed-loop-compare")
    def ml_closed_loop_compare(body: dict = Body(...)):
        """One click (Nov 2026): run a trained network AS the agent's brain on a benchmark task AND the
        same task with the scripted reference policy under identical conditions. Both are judged by the
        task's own verifier; the result is a side-by-side with a verdict. Uses the ML lock like the
        other eval jobs so nothing overloads the world."""
        from ..ml.registry import NAME_RE
        from ..ml import MLError
        from ..agents.adapters import make_adapter, make_neural_adapter, AdapterError
        from ..session.benchmark import run_controller_benchmark, TASKS
        owner = body.get("owner", "researcher")
        name = body.get("name")
        version = body.get("version")
        task_id = body.get("task_id", "throw_height")
        if not name:
            raise HTTPException(400, "name (model name) is required.")
        if task_id not in TASKS:
            raise HTTPException(400, f"unknown task_id; available: {sorted(TASKS)}")
        with ml_lock:
            if ml_running["busy"]:
                raise HTTPException(409, "A training, evaluation or generation job is already running.")
            ml_running["busy"] = True
        job_id = str(uuid.uuid4())[:8]
        ml_jobs[job_id] = {"status": "queued", "kind": "closed_loop_compare",
                           "model": f"{owner}/{name}", "task_id": task_id}

        def run_compare_job():
            try:
                ml_jobs[job_id]["status"] = "running"
                neural = make_neural_adapter(cfg, owner=owner, name=name, version=version)
                model_result = run_controller_benchmark(cfg, db, neural, task_id=task_id,
                                                        label=f"clv:{owner}/{name}")
                scripted = make_adapter(cfg, "scripted")
                baseline_result = run_controller_benchmark(cfg, db, scripted, task_id=task_id,
                                                           label=f"clv:{owner}/{name}:baseline")
                def better(a, b):
                    if a is None or b is None:
                        return None
                    return a > b
                verdict = {
                    "beats_baseline": better(model_result.get("score"), baseline_result.get("score")),
                    "model_completed": model_result.get("completed"),
                    "baseline_completed": baseline_result.get("completed"),
                    "score_delta": (None if model_result.get("score") is None or
                                    baseline_result.get("score") is None
                                    else round(model_result["score"] - baseline_result["score"], 3)),
                }
                ml_jobs[job_id]["status"] = "done"
                ml_jobs[job_id]["result_summary"] = {
                    "task_id": task_id, "model": model_result, "baseline": baseline_result,
                    "verdict": verdict}
            except (MLError, AdapterError) as e:
                ml_jobs[job_id]["status"] = "error"
                ml_jobs[job_id]["error"] = getattr(e, "message", str(e))
            except Exception as e:  # noqa: BLE001
                ml_jobs[job_id]["status"] = "error"
                ml_jobs[job_id]["error"] = str(e)
            finally:
                with ml_lock:
                    ml_running["busy"] = False

        threading.Thread(target=run_compare_job, daemon=True).start()
        return {"job_id": job_id}

    # ---------------------------------------------------------------- audit 2026-11-13 dashboard access
    # Everything the audit added to the CLI, exposed to the dashboard: model discovery, the ML on/off
    # switch, per-session behavioural profiles, the provenance check, and launching exploration /
    # model-comparison runs from the browser.
    run_jobs: dict[str, dict] = {}
    run_lock = threading.Lock()
    run_state: dict[str, bool] = {"busy": False}

    @app.get("/api/adapters")
    def list_adapters():
        table = cfg.get("adapters", {})
        return {"adapters": [{"name": n, "type": s.get("type"), "model": s.get("model"),
                              "base_url": s.get("base_url")} for n, s in sorted(table.items())]}

    @app.get("/api/models")
    def list_models(adapter: str = "lmstudio"):
        """Every model the given openai_compat server (e.g. LM Studio) is offering, so a model can be
        chosen in the UI instead of being limited to the config (audit 2026-11-13 steer)."""
        from ..agents.adapters import list_openai_models
        spec = cfg.get("adapters", {}).get(adapter)
        if not spec or spec.get("type") != "openai_compat":
            raise HTTPException(400, f"'{adapter}' is not an openai_compat adapter.")
        models = list_openai_models(spec["base_url"], spec.get("api_key_env"))
        return {"adapter": adapter, "base_url": spec["base_url"],
                "models": [{"id": m.get("id"), "state": m.get("state"),
                            "context": m.get("loaded_context_length") or m.get("max_context_length")}
                           for m in models]}

    @app.get("/api/ml/config")
    def ml_config_get():
        from ..session.runner import ml_enabled_for
        ml = cfg.get("ml", {}) or {}
        return {"enabled": bool(ml.get("enabled", True)), "agents": ml.get("agents", {}),
                "resolved": {a: ml_enabled_for(cfg, a) for a in agent_ids(cfg)}}

    @app.post("/api/ml/config")
    def ml_config_set(body: dict = Body(...)):
        """Turn the neural-network tools on/off globally or per agent, live (audit 2026-11-13 steer)."""
        ml = cfg.setdefault("ml", {})
        if "enabled" in body:
            ml["enabled"] = bool(body["enabled"])
        if "agent" in body and "agent_enabled" in body:
            ml.setdefault("agents", {}).setdefault(str(body["agent"]), {})["enabled"] = bool(body["agent_enabled"])
        return ml_config_get()

    @app.get("/api/sessions/{sid}/profile")
    def session_profile_api(sid: int):
        from ..session.model_compare import session_profile
        sess(sid)
        return session_profile(db, sid)

    @app.get("/api/ml/dataset-kinds")
    def ml_dataset_kinds():
        """The catalogue of behaviour datasets the lab can build — not just throws (audit 2026-11-13)."""
        from ..ml.datasets import available_datasets
        return {"kinds": [{"name": "throw", "description": "Throw outcomes (force/angle -> height/range).",
                           "tools": ["throw"]}] + available_datasets()}

    @app.post("/api/ml/build-behaviour")
    def ml_build_behaviour(body: dict = Body(...)):
        """Build a named behaviour dataset (climb/jump/locomotion/manipulation/placement/affordance) from
        the recorded log and save it to the chosen registry (audit 2026-11-13). Background job.

        The log is read and the dataset built at submit time: an empty result is refused here with a
        400 (nothing actionable was recorded) instead of writing a 0-row dataset artifact.
        """
        from ..ml.datasets import DATASETS, build_dataset
        from ..ml.registry import NAME_RE
        name = body.get("name")
        kind = body.get("kind")
        if not name or not NAME_RE.match(name):
            raise HTTPException(400, "name must be 1-40 chars of letters, digits, '_' or '-'.")
        if kind not in DATASETS:
            raise HTTPException(400, f"unknown kind '{kind}'; available: {sorted(DATASETS)}")
        scope = body.get("scope", "researcher")
        owner = body.get("owner", "researcher")
        reg = _ml_registry(scope, owner)
        # Which sessions' actions to draw from: an agent's own, or every session for the researcher scope.
        if scope == "agent":
            rows = db.q("SELECT id FROM sessions WHERE agent_id=? ORDER BY id", (owner,))
        else:
            rows = db.q("SELECT id FROM sessions ORDER BY id")
        actions: list = []
        for r in rows:
            actions.extend(db.actions(r["id"], limit=10**6))
        ds = build_dataset(actions, kind)
        if not ds["rows"]:
            raise HTTPException(
                400, f"the action log has 0 usable '{kind}' actions for {scope}/{owner} — nothing to build. "
                     f"Run sessions that use the '{kind}' behaviour first, or build an engine dataset "
                     f"instead (throws, jumps and climbs can be generated without any log history).")
        with ml_lock:
            if ml_running["busy"]:
                raise HTTPException(409, "A training, evaluation or generation job is already running.")
            ml_running["busy"] = True
        job_id = str(uuid.uuid4())[:8]
        ml_jobs[job_id] = {"status": "queued", "kind": "build_behaviour", "dataset": name, "behaviour": kind}

        def run_build():
            try:
                ml_jobs[job_id]["status"] = "running"
                import hashlib as _hl
                ds["manifest"]["rows_sha256"] = _hl.sha256(
                    json.dumps(ds["rows"], sort_keys=True).encode()).hexdigest()
                ds["manifest"]["updated_at"] = time.time()
                info = reg.save_dataset(name, ds)
                ml_jobs[job_id]["status"] = "done"
                ml_jobs[job_id]["result_summary"] = {**info, "kind": kind, "rows": ds["manifest"]["rows"],
                                                     "scope": scope, "owner": owner}
            except Exception as e:  # noqa: BLE001
                ml_jobs[job_id]["status"] = "error"
                ml_jobs[job_id]["error"] = str(e)
            finally:
                with ml_lock:
                    ml_running["busy"] = False

        threading.Thread(target=run_build, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/agents/{agent_id}/goals")
    def agent_goals(agent_id: str):
        """An agent's OPEN goals from its memory (not just this session) so the dashboard's
        'Self-selected objectives' reflects reality instead of showing 'none set' (audit 2026-11-13)."""
        m = mem(agent_id)
        open_goals = [g for g in m.state["goals"] if g["status"] in ("active", "paused")]
        return {"agent_id": agent_id, "open_goals": open_goals,
                "all_goals": m.state["goals"]}

    @app.get("/api/audit/provenance")
    def audit_provenance():
        """Sessions whose recorded brain differs from the adapter that actually ran them, and the memory
        items that are therefore mis-attributed (the session-19 scripted contamination)."""
        import subprocess, sys as _sys
        from pathlib import Path as _Path
        script = _Path(__file__).resolve().parent.parent.parent / "scripts" / "audit_provenance.py"
        if not script.exists():
            return {"available": False, "note": "scripts/audit_provenance.py not found"}
        try:
            out = subprocess.run([_sys.executable, str(script), "--db", db.path, "--json",
                                  "--memory-root", str(memory_root(cfg))],
                                 capture_output=True, text=True, timeout=60)
            return json.loads(out.stdout) if out.stdout.strip().startswith("{") else {"raw": out.stdout}
        except Exception as e:  # noqa: BLE001
            return {"available": False, "error": str(e)}

    def _start_run_job(kind: str, target) -> dict:
        with run_lock:
            if run_state["busy"]:
                raise HTTPException(409, "A run is already in progress.")
            run_state["busy"] = True
        job_id = str(uuid.uuid4())[:8]
        run_jobs[job_id] = {"status": "running", "kind": kind, "progress": [], "result": None}

        def wrap():
            try:
                target(run_jobs[job_id])
                run_jobs[job_id]["status"] = "done"
                # Incremental datasets: the run just added actions to the log; rebuild the log-derived
                # behaviour datasets so the NN lab trains on current data.
                _refresh_datasets_async(on_done=lambda out: run_jobs[job_id].setdefault(
                    "dataset_refresh", out))
            except Exception as e:  # noqa: BLE001
                run_jobs[job_id]["status"] = "error"
                run_jobs[job_id]["error"] = str(e)
            finally:
                with run_lock:
                    run_state["busy"] = False
        threading.Thread(target=wrap, daemon=True).start()
        return {"job_id": job_id}

    dataset_refresh_lock = threading.Lock()

    def _all_registries():
        """The researcher registry plus every agent nn/ registry that holds datasets."""
        from ..ml.registry import Registry, registry_for
        regs = [registry_for(cfg, "researcher", "researcher")]
        mem_root = resolve_path(cfg, cfg["memory"]["root"])
        mem_root = Path(mem_root)
        if mem_root.exists():
            for d in sorted(mem_root.iterdir()):
                if (d / "nn" / "datasets").is_dir():
                    regs.append(Registry(cfg, root=d / "nn"))
        return regs

    def _refresh_datasets_async(on_done=None):
        """Rebuild log-derived behaviour datasets in a background thread (no-op if disabled)."""
        if not (cfg.get("ml", {}).get("auto_refresh_datasets", True)):
            return None

        def run():
            out: list = []
            try:
                from ..ml.datasets import refresh_log_datasets
                with dataset_refresh_lock:
                    for reg in _all_registries():
                        out.extend(refresh_log_datasets(db, reg))
            except Exception as e:  # noqa: BLE001
                out = [{"error": str(e)}]
            if on_done:
                try:
                    on_done(out)
                except Exception:
                    pass
            return out
        threading.Thread(target=run, daemon=True).start()
        return True


    @app.get("/api/run/jobs/{job_id}")
    def run_job_status(job_id: str):
        if job_id not in run_jobs:
            raise HTTPException(404, f"No run job '{job_id}'.")
        return run_jobs[job_id]

    @app.post("/api/run/jobs/{job_id}/stop")
    def run_job_stop(job_id: str):
        if job_id not in run_jobs:
            raise HTTPException(404, f"No run job '{job_id}'.")
        with run_lock:
            run_jobs[job_id]["stop"] = True
        return {"ok": True, "job_id": job_id, "note": "stop requested"}

    @app.post("/api/panic")
    def panic_stop():
        """Kill switch: stop every running session and run job this server knows about, in one call."""
        from ..session.runner import _session_stop_events, request_stop_session, finish_session
        sessions_requested = 0
        # In-process loop sessions (runner registry): ask each to stop at its next safe point.
        for sid in list(_session_stop_events.keys()):
            if request_stop_session(sid):
                sessions_requested += 1
        # Externally driven sessions: finish them directly.
        with lock:
            external_items = list(live.items())
            live.clear()
        external_finished = 0
        for sid, item in external_items:
            task = item.get("task")
            def verify(env):
                return task.verify(env.world, db.events(env.sid, limit=10 ** 9)) if task else None
            finish_session(item["cfg"], db, item["env"], item["adapter"], item["env"].agent_calls, {},
                           "stopped_by_user", None, verify if task else None)
            external_finished += 1
            sessions_requested += 1
        _refresh_datasets_async()   # the killed sessions' actions are now in the log
        # Run jobs: mark every running job as stopping.
        jobs_stopped = 0
        with run_lock:
            for jid, job in run_jobs.items():
                if job.get("status") == "running":
                    job["stop"] = True
                    jobs_stopped += 1
        return {"ok": True, "sessions_signalled": sessions_requested,
                "external_finished": external_finished, "run_jobs_stopping": jobs_stopped}

    @app.get("/api/run/active")
    def run_active():
        return {"busy": run_state["busy"], "jobs": {k: v["status"] for k, v in run_jobs.items()}}

    @app.get("/api/stream/events")
    async def stream_events(request: Request, agent: str | None = None, max_events: int = 0):
        """Server-Sent Events: push every recorded action (plus session lifecycle) as it appears in the
        log, across ALL sessions at once. The stream reads the shared SQLite log, so sessions driven by
        the CLI in another process stream here too — watching never interferes. `max_events` (test hook)
        ends the stream after N events; `agent` filters to one agent."""
        import asyncio as _asyncio
        from fastapi.responses import StreamingResponse

        state = {"last_action": 0, "sessions": {}}

        def poll() -> list[dict]:
            out: list[dict] = []
            where, params = "", []
            if agent:
                where = "AND s.agent_id=?"
                params = [agent]
            rows = db.q(f"SELECT id, agent_id, mode, session_number, status, end_reason FROM sessions s "
                        f"ORDER BY id DESC LIMIT 40")
            live_now = {}
            for r in rows:
                live_now[r["id"]] = r
                prev = state["sessions"].get(r["id"])
                if prev is None or prev.get("status") != r["status"] or prev.get("end_reason") != r["end_reason"]:
                    state["sessions"][r["id"]] = r
                    out.append({"type": "session", "session_id": r["id"], "agent_id": r["agent_id"],
                                "mode": r["mode"], "session_number": r["session_number"],
                                "status": r["status"], "end_reason": r["end_reason"]})
            for sid in list(state["sessions"]):
                if sid not in live_now:
                    del state["sessions"][sid]
            acts = db.q(f"SELECT a.id, a.session_id, a.tool, a.category, a.success, a.error_code, "
                        f"a.turn, a.t_start, a.args_json FROM actions a JOIN sessions s ON s.id=a.session_id "
                        f"WHERE a.id > ? {where} ORDER BY a.id LIMIT 200",
                        [state["last_action"], *params])
            for a in acts:
                state["last_action"] = max(state["last_action"], a["id"])
                try:
                    args = json.loads(a.get("args_json") or "{}")
                except (TypeError, ValueError):
                    args = {}
                out.append({"type": "action", "id": a["id"], "session_id": a["session_id"],
                            "tool": a["tool"], "category": a["category"], "success": bool(a["success"]),
                            "error_code": a["error_code"], "turn": a["turn"],
                            "args": {k: args[k] for k in list(args)[:6] if not isinstance(args[k], (dict, list))}})
            return out

        async def gen():
            sent = 0
            # Initial state: the last 5 actions per running session so a fresh console is not empty.
            for r in db.q("SELECT id, agent_id, mode, session_number, status FROM sessions "
                          "WHERE status='running' ORDER BY id DESC LIMIT 8"):
                state["sessions"][r["id"]] = dict(r)
                yield f"data: {json.dumps({'type': 'session', **r})}\n\n"
            idle_ticks = 0
            while True:
                events = poll()
                for e in events:
                    sent += 1
                    yield f"data: {json.dumps(e, default=str)}\n\n"
                if max_events and sent >= max_events:
                    return
                idle_ticks += 1
                if idle_ticks % 30 == 0:      # ~15 s heartbeat at 0.5 s cadence
                    yield ": heartbeat\n\n"
                await _asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/run/explore")
    def run_explore(body: dict = Body(...)):
        """Launch exploration sessions for one agent from the browser (audit 2026-11-13).

        Accepted knobs (the dashboard's run form): agent_id, adapter, model, sessions (per run),
        max_steps (steps per session), concurrency (sessions running at a time), max_continuations.
        """
        from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
        from ..agents.adapters import make_adapter
        from ..session.runner import run_session, request_stop_session
        agent_id = str(body.get("agent_id") or "agent_001")
        adapter_name = str(body.get("adapter") or "lmstudio")
        model = body.get("model")
        sessions = max(1, min(50, int(body.get("sessions", 1))))
        concurrency = max(1, min(8, int(body.get("concurrency", 1))))
        max_cont = max(0, min(20, int(body.get("max_continuations", 0))))
        max_steps = body.get("max_steps")
        if adapter_name not in cfg.get("adapters", {}):
            raise HTTPException(400, f"unknown adapter '{adapter_name}'")

        def run_one(state):
            ad = make_adapter(cfg, adapter_name, model=model)
            run_cfg = cfg
            if max_steps is not None:
                run_cfg = copy.deepcopy(cfg)
                run_cfg.setdefault("session", {})["max_steps"] = max(1, int(max_steps))
            return run_session(run_cfg, db, agent_id, ad, mode="exploration",
                               on_event=lambda etype, data: state.setdefault("current_sessions", set()).add(data["session_id"]))

        def job(state):
            remaining, granted = sessions, 0
            stop_sent: set[int] = set()
            progress = state["progress"]

            def collect(f) -> None:
                """Record one finished session and steer the remaining budget."""
                nonlocal remaining, granted
                try:
                    s = f.result()
                except Exception as e:
                    progress.append({"session_id": None, "end_reason": "runner_error", "actions": 0,
                                     "continue_requested": False, "error": str(e)})
                    remaining = 0
                    return
                cur: set = state.get("current_sessions") or set()
                cur.discard(s["session_id"])
                progress.append({"session_id": s["session_id"], "end_reason": s["end_reason"],
                                 "actions": s["actions_logged"],
                                 "continue_requested": s.get("continue_requested"),
                                 "error": s.get("error")})
                if s["end_reason"] in ("adapter_error", "runner_error", "interrupted", "stopped_by_user"):
                    remaining = 0
                elif remaining == 0 and s.get("continue_requested") and granted < max_cont:
                    granted += 1
                    remaining = 1

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures: set = set()
                while remaining > 0 or futures:
                    # A pending stop: signal every in-flight session to stop at its next safe point,
                    # spawn nothing new, give the runner up to 60 s to write them out.
                    if state.get("stop"):
                        cur: set = state.get("current_sessions") or set()
                        for sid in list(cur):
                            if sid not in stop_sent:
                                request_stop_session(sid)
                                stop_sent.add(sid)
                        if not futures:
                            break
                        done, futures = wait(futures, timeout=60, return_when=FIRST_COMPLETED)
                        for f in done:
                            collect(f)
                        if not futures:
                            break
                        continue
                    while len(futures) < concurrency and remaining > 0:
                        remaining -= 1
                        futures.add(pool.submit(run_one, state))
                    if not futures:
                        break
                    # 0.5 s slice so a stop request is honoured mid-run, not only between sessions.
                    done, futures = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                    for f in done:
                        collect(f)
            state["result"] = {"sessions_run": len(progress), "continuations_granted": granted,
                               "stopped": bool(state.get("stop"))}
        return _start_run_job("explore", job)

    @app.post("/api/run/compare")
    def run_compare(body: dict = Body(...)):
        """Launch an equal-conditions model comparison from the browser (audit 2026-11-13)."""
        from ..agents.adapters import make_adapter
        from ..session.model_compare import compare_models
        models = [m.strip() for m in (body.get("models") or []) if str(m).strip()]
        if not models:
            raise HTTPException(400, "models must be a non-empty list of model ids")
        adapter_name = str(body.get("adapter") or "lmstudio")
        agents = max(1, min(10, int(body.get("agents", 2))))
        max_steps = int(body.get("max_steps", 60))
        ctx = body.get("context_tokens")

        def adapter_for(model, aid):
            return make_adapter(cfg, adapter_name, model=model)

        def job(state):
            res = compare_models(cfg, db, adapter_for, models, agents_per_model=agents, sessions=1,
                                 context_tokens=int(ctx) if ctx else None, max_steps=max_steps,
                                 progress=lambda p: state["progress"].append(
                                     {"model": p["model"], "session_id": p["session_id"],
                                      "actions": p["actions"], "distinct_tools": p["distinct_tools"],
                                      "observation_share": p["observation_share"], "goals_set": p["goals_set"]}))
            state["result"] = res["by_model"]
            # Leaderboard: one persistent behavioural entry per compared model.
            try:
                from ..ml import leaderboard as lb
                for model_id, agg in res["by_model"].items():
                    lb.record(cfg, {
                        "source": "compare", "owner": "researcher", "name": model_id,
                        "adapter": adapter_name, "agents": agents, "sessions_per_agent": 1,
                        "max_steps": max_steps, "context_tokens": ctx,
                        "actions_mean": agg.get("actions_mean"),
                        "distinct_tools_mean": agg.get("distinct_tools_mean"),
                        "goals_set_mean": agg.get("goals_set_mean"),
                        "confirmed_discoveries_mean": agg.get("confirmed_discoveries_mean"),
                        "failed_share_mean": agg.get("failed_share_mean"),
                        "experiments_mean": agg.get("experiments_mean"),
                        "errored": agg.get("errored"),
                    })
            except Exception:
                pass
        return _start_run_job("compare", job)

    # ---------------------------------------------------------------- run modes: arena + task (M5.5)
    # Modes the BACKEND genuinely supports, for the Run & models selector. Grounded in
    # session/runner.py's mode dispatch (exploration/arena_*), session/benchmark.run_task
    # (mode='benchmark') and arena/factories.run_arena_match + mode_for().
    RUN_MODES = [
        {"id": "exploration", "label": "Free play (exploration)",
         "note": "One agent roaming the island with the full toolset, memory on."},
        {"id": "benchmark", "label": "Task run (single agent)",
         "note": "One agent on a fixed benchmark task whose result the engine verifies."},
        {"id": "arena_solo", "label": "Arena duel (1v1)",
         "note": "Two combatants, one per team, real paintball physics and scoring."},
        {"id": "arena_coop", "label": "Arena team match",
         "note": "Two teams; teammates share a line, friendly fire stays off by default."},
    ]

    def _equipment_options() -> list[dict]:
        from ..combat.equipment import EQUIPMENT
        out: list[dict] = []
        for eid, es in sorted(EQUIPMENT.items()):
            entry = {"id": eid, "name": es.get("name", eid), "combat_class": es.get("combat_class"),
                     "grip": es.get("grip"), "dims_m": list(es.get("dims_m") or []),
                     "ammo_capacity": es.get("ammo_capacity")}
            pc = es.get("projectile_cfg")
            if pc:
                # the launcher's real projectile (radius drives the pellet mesh size)
                entry["projectile"] = {"speed": pc.get("speed"), "radius": pc.get("radius"),
                                       "colour": pc.get("colour"), "restitution": pc.get("restitution")}
            out.append(entry)
        return out

    @app.get("/api/run/modes")
    def run_modes():
        """Run-mode registry for the Run & models selector: only modes the backend runs,
        plus the real equipment catalogue (M5 factories validate against the same one)."""
        return {"modes": RUN_MODES, "equipment": _equipment_options(),
                "limits": {"duration_s": [2, 600], "pellets": [1, 96], "score_target": [1, 999],
                           "combatants_per_team": [1, 4], "teams": 2}}

    @app.post("/api/run/arena")
    def run_arena_api(body: dict = Body(...)):
        """Launch a REAL arena match (M5 production factories) as a run job.

        Body: {"mode": "arena_solo"|"arena_coop", "name": str?, "rules": {"duration_s", "pellets",
        "score_target", "friendly_fire"}, "teams": [{"name", "agents": [{"id", "adapter", "model"}],
        "equipment": [<EQUIPMENT id>]}]}. The job runs run_arena_match end-to-end; every
        combatant session is recorded exactly like a CLI match."""
        from ..arena.factories import mode_for, run_arena_match
        from ..combat.equipment import EQUIPMENT
        from ..agents.adapters import make_adapter

        mode = str(body.get("mode") or "")
        if mode not in ("arena_solo", "arena_coop"):
            raise HTTPException(400, f"mode must be 'arena_solo' or 'arena_coop' (got '{mode}').")
        teams_in = body.get("teams")
        if not isinstance(teams_in, list) or len(teams_in) != 2:
            raise HTTPException(400, "arena runs need exactly two teams")
        limits = {"duration_s": (2, 600), "pellets": (1, 96), "score_target": (1, 999)}
        rules_in = body.get("rules") or {}
        rules: dict = {}
        for key, (lo, hi) in limits.items():
            v = float(rules_in.get(key, {"duration_s": 60, "pellets": 24, "score_target": 10}[key]))
            if not (lo <= v <= hi):
                raise HTTPException(400, f"{key} out of range: {v} (allowed {lo}..{hi})")
            rules[key] = int(v) if key != "duration_s" else round(v, 2)
        rules["friendly_fire"] = bool(rules_in.get("friendly_fire", False))

        match_cfg: dict = {"rules": rules, "teams": []}
        adapter_for_agent: dict[str, tuple[str, str | None]] = {}
        for ti, team in enumerate(teams_in):
            if not isinstance(team, dict):
                raise HTTPException(400, f"team {ti} must be an object")
            name = str(team.get("name") or "").strip() or (f"Team{ti + 1}")
            members = team.get("agents") or [{}]
            if not (1 <= len(members) <= 4):
                raise HTTPException(400, f"team '{name}' needs 1-4 agents (mode '{mode}': "
                                         "arena_solo requires exactly one per team)")
            equipment = [str(e) for e in (team.get("equipment") or [])]
            unknown = [e for e in equipment if e not in EQUIPMENT]
            if unknown:
                raise HTTPException(400, f"unknown equipment for team '{name}': {unknown}; "
                                         f"known: {sorted(EQUIPMENT)}")
            agents_out: list[dict] = []
            for ai, acfg in enumerate(members):
                aid = str(acfg.get("id") or "").strip()
                if not aid:
                    raise HTTPException(400, f"team '{name}' agent {ai + 1}: 'id' is required")
                if not AGENT_ID_RE.match(aid):
                    raise HTTPException(400, f"invalid combatant id '{aid}': 1-40 chars "
                                             "a-z 0-9 '_' '-' (must start alphanumeric)")
                adapter_name = str(acfg.get("adapter") or "").strip() or "scripted"
                if adapter_name not in set((cfg.get("adapters") or {})):
                    raise HTTPException(400, f"combatant '{aid}': unknown adapter '{adapter_name}'. "
                                             f"Configured: {sorted(cfg.get('adapters') or {})}")
                model = str(acfg.get("model") or "").strip() or None
                adapter_for_agent[aid] = (adapter_name, model)
                agents_out.append({"id": aid})
            match_cfg["teams"].append({"name": name, "agents": agents_out, "equipment": equipment})

        resolved_mode = mode_for(match_cfg["teams"])
        if resolved_mode != mode:
            raise HTTPException(400, f"team layout is a {resolved_mode} match, not '{mode}': "
                                     "arena_solo needs exactly 2 teams of 1 combatant")
        match_name = str(body.get("name") or "").strip()
        if match_name and not re.match(r"^[a-zA-Z0-9_\-]{1,40}$", match_name):
            raise HTTPException(400, "name must be 1-40 chars of letters, digits, '_' or '-'")

        def adapter_factory(agent_id: str, plan_item: dict):
            aname, amodel = adapter_for_agent.get(agent_id, ("scripted", None))
            return make_adapter(cfg, aname, model=amodel)

        def job(state):
            result = run_arena_match(match_cfg, adapter_factory, db=db, root_cfg=cfg,
                                     match_name=match_name or "dashboard")
            payload = {
                "mode": resolved_mode, "end_reason": result.get("end_reason"),
                "winner": result.get("winner"), "duration": result.get("duration"),
                "timeout": bool(result.get("timeout")),
                "team_stats": result.get("team_stats"), "agent_stats": result.get("agent_stats"),
                "sessions": [dict(s) for s in (result.get("sessions") or [])],
                "timeline_events": len(result.get("timeline") or []),
                "timeline": [t for t in (result.get("timeline") or [])
                             if isinstance(t, dict)][:200],
            }
            state["result"] = payload
            state["progress"].append({"sessions": payload["sessions"]})

        return _start_run_job("arena", job)

    @app.post("/api/run/benchmark")
    def run_benchmark_api(body: dict = Body(...)):
        """Launch ONE benchmark task session as a run job (the backend's real single-agent
        task mode: session/benchmark.run_task, with the engine verifying the result)."""
        from ..agents.adapters import make_adapter
        from ..session.benchmark import TASKS, run_task
        task_id = str(body.get("task_id") or "")
        if task_id not in TASKS:
            raise HTTPException(400, f"unknown task_id; available: {sorted(TASKS)}")
        agent_id = str(body.get("agent_id") or "").strip()
        if not AGENT_ID_RE.match(agent_id):
            raise HTTPException(400, "agent_id is required: 1-40 chars of a-z, 0-9, '_' or '-'")
        adapter_name = str(body.get("adapter") or "").strip() or "scripted"
        if adapter_name not in set((cfg.get("adapters") or {})):
            raise HTTPException(400, f"unknown adapter '{adapter_name}'. "
                                     f"Configured: {sorted(cfg.get('adapters') or {})}")
        model = str(body.get("model") or "").strip() or None

        def job(state):
            from ..memory import MemoryStore
            run_id = db.x("INSERT INTO benchmark_runs(label,kind,created_at,params_json) "
                          "VALUES(?,?,?,?)",
                          (f"bench:{agent_id}:{task_id}", "single", time.time(),
                           json.dumps({"agent_id": agent_id, "task_id": task_id,
                                       "adapter": adapter_name, "model": model})))
            task = TASKS[task_id]
            mem = MemoryStore(memory_root(cfg), agent_id, cfg, db=db)
            row = run_task(cfg, db, task, agent_id,
                           make_adapter(cfg, adapter_name, model=model), mem, run_id,
                           "A", "experienced")
            state["result"] = {"run_id": run_id, **row}
            state["progress"].append({"session_id": row["session_id"], "task_id": task_id,
                                      "action_count": row.get("action_count")})

        return _start_run_job("benchmark", job)

    # ---------------------------------------------------------------- dashboard
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.exception_handler(Exception)
    async def crash(request, exc):  # noqa: ANN001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app
