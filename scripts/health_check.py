"""Health check: is everything needed to run an experiment present and working?

Exits 0 when healthy. Each check prints PASS/FAIL with the evidence it used.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            ok, info = fn()
        except Exception as exc:  # noqa: BLE001
            ok, info = False, f"{type(exc).__name__}: {exc}"
        results.append((name, ok, info))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        return fn
    return deco


@check("imports")
def _():
    import fastapi, httpx, mujoco, numpy, uvicorn, yaml  # noqa: E401, F401
    return True, f"mujoco {mujoco.__version__}, numpy {numpy.__version__}, fastapi {fastapi.__version__}"


@check("config")
def _():
    from playground.config import config_hash, load_config
    cfg = load_config()
    return cfg["agents"]["count"] == 13, f"13 agents, physics config {config_hash(cfg)[:12]}"


@check("physics determinism")
def _():
    from playground.config import load_config
    from playground.world.mathutil import parse_direction
    from playground.world.physics import PhysicsWorld
    cfg = load_config()
    hs = []
    for _ in range(2):
        w = PhysicsWorld(cfg)
        w.move(parse_direction([-2.5, -2.1, 0]), float((2.5 ** 2 + 2.1 ** 2) ** 0.5))
        w.pick_up("block_01")
        w.jump(wait="takeoff")
        w.throw("block_01", parse_direction("up"), 200.0)
        hs.append(w.state_hash())
    return hs[0] == hs[1], f"two runs -> {hs[0][:12]} / {hs[1][:12]}"


@check("physics speed")
def _():
    from playground.config import load_config
    from playground.world.physics import PhysicsWorld
    w = PhysicsWorld(load_config())
    t = time.perf_counter()
    w.step(5000)
    rt = 5000 * w.dt / (time.perf_counter() - t)
    return rt > 2, f"{rt:.1f}x real time"


@check("database + session + hash chain")
def _():
    from playground.agents import PolicyAdapter
    from playground.config import load_config
    from playground.session import run_session
    from playground.storage import Database
    tmp = Path(tempfile.mkdtemp())
    cfg = load_config(overrides={"_root": str(tmp)})
    db = Database(tmp / "h.db")

    def p(s, o, t):
        yield ("call", "observe_world", {})
        yield ("call", "jump", {})
        yield ("call", "end_session", {})
    s = run_session(cfg, db, "health", PolicyAdapter(p))
    ok = s["end_reason"] == "agent_ended" and db.verify_chain(s["session_id"])["ok"]
    return ok, f"session {s['session_id']} ended '{s['end_reason']}', chain ok={db.verify_chain(s['session_id'])['ok']}"


@check("dashboard assets")
def _():
    st = ROOT / "playground" / "dashboard" / "static"
    need = ["index.html", "app.js", "style.css", "vendor/three.module.min.js", "vendor/OrbitControls.js"]
    missing = [n for n in need if not (st / n).exists()]
    return not missing, "all present" if not missing else f"missing {missing}"


@check("experiment workspace")
def _():
    from playground.config import load_config, resolve_path
    cfg = load_config()
    root = resolve_path(cfg, cfg["memory"]["root"])
    n = len([p for p in root.iterdir() if p.is_dir()]) if root.exists() else 0
    return True, f"{root} ({n} agent folders; run 'playground init' to create all 13)"


@check("model endpoints (informational)")
def _():
    import httpx
    out = []
    try:
        r = httpx.get("http://127.0.0.1:1234/v1/models", timeout=3)
        out.append(f"LM Studio up ({len(r.json().get('data', []))} models)")
    except Exception:  # noqa: BLE001
        out.append("LM Studio not reachable")
    return True, "; ".join(out)


failed = [n for n, ok, _ in results if not ok]
print(json.dumps({"healthy": not failed, "failed": failed}))
sys.exit(1 if failed else 0)
