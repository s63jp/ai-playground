"""Health check: is everything needed to run an experiment present and working?

Exits 0 when healthy. Each check prints PASS/FAIL with the evidence it used.

Usage:
    python scripts/health_check.py [--workspace PATH] [--no-xpu]
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Health check for the AI Playground")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Experiment workspace path (uses config default if not given)")
    parser.add_argument("--no-xpu", action="store_true",
                        help="Skip XPU/GPU checks (for CI or CPU-only environments)")
    args = parser.parse_args()

    overrides = {}
    if args.workspace:
        overrides["_root"] = args.workspace

    # === Core imports ===
    @check("imports")
    def _():
        import fastapi, httpx, mujoco, numpy, uvicorn, yaml  # noqa: E401, F401
        return True, f"mujoco {mujoco.__version__}, numpy {numpy.__version__}, fastapi {fastapi.__version__}"

    # === Config ===
    @check("config")
    def _():
        from playground.config import config_hash, load_config
        cfg = load_config(overrides=overrides)
        return cfg["agents"]["count"] == 13, f"13 agents, physics config {config_hash(cfg)[:12]}"

    # === Physics determinism ===
    @check("physics determinism")
    def _():
        from playground.config import load_config
        from playground.world.mathutil import parse_direction
        from playground.world.physics import PhysicsWorld
        cfg = load_config(overrides=overrides)
        hs = []
        for _ in range(2):
            w = PhysicsWorld(cfg)
            w.move(parse_direction([-2.5, -2.1, 0]), float((2.5 ** 2 + 2.1 ** 2) ** 0.5))
            w.pick_up("block_01")
            w.jump(wait="takeoff")
            w.throw("block_01", parse_direction("up"), 200.0)
            hs.append(w.state_hash())
        return hs[0] == hs[1], f"two runs -> {hs[0][:12]} / {hs[1][:12]}"

    # === Physics speed ===
    @check("physics speed")
    def _():
        from playground.config import load_config
        from playground.world.physics import PhysicsWorld
        cfg = load_config(overrides=overrides)
        w = PhysicsWorld(cfg)
        t = time.perf_counter()
        w.step(5000)
        rt = 5000 * w.dt / (time.perf_counter() - t)
        return rt > 2, f"{rt:.1f}x real time"

    # === Database + session + hash chain ===
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

    # === Dashboard assets ===
    @check("dashboard assets")
    def _():
        st = ROOT / "playground" / "dashboard" / "static"
        need = ["index.html", "app.js", "style.css", "vendor/three.module.min.js", "vendor/OrbitControls.js"]
        missing = [n for n in need if not (st / n).exists()]
        return not missing, "all present" if not missing else f"missing {missing}"

    # === Experiment workspace ===
    @check("experiment workspace")
    def _():
        from playground.config import load_config, resolve_path
        cfg = load_config(overrides=overrides)
        root = resolve_path(cfg, cfg["memory"]["root"])
        n = len([p for p in root.iterdir() if p.is_dir()]) if root.exists() else 0
        return True, f"{root} ({n} agent folders; run 'playground init' to create all 13)"

    # === ML stack (informational - never fails) ===
    @check("ML stack (optional)")
    def _():
        try:
            import torch
            version = torch.__version__
            xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
            cuda_available = torch.cuda.is_available()
            device_info = []
            if xpu_available:
                device_info.append(f"XPU available ({torch.xpu.device_count()} devices)")
            if cuda_available:
                device_info.append(f"CUDA available ({torch.cuda.device_count()} devices)")
            if not device_info:
                device_info.append("CPU only")
            return True, f"torch {version}; {'; '.join(device_info)}"
        except ImportError:
            return True, "torch not installed (ML features disabled)"

    # === ML registry dir writable ===
    @check("ML registry writable")
    def _():
        from playground.config import load_config, resolve_path
        cfg = load_config(overrides=overrides)
        ml_dir = resolve_path(cfg, cfg.get("_root", "experiments")) / "ml"
        try:
            ml_dir.mkdir(parents=True, exist_ok=True)
            test_file = ml_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True, f"{ml_dir} is writable"
        except Exception as e:
            return False, f"{ml_dir} not writable: {e}"

    # === Prompts present ===
    @check("exploration prompts")
    def _():
        from playground.agents.prompts import PERSONALITIES, DEFAULT_PERSONALITY
        from playground.config import load_config
        cfg = load_config(overrides=overrides)
        required = ["free_agency_v2", "brief_v1"]
        missing = [p for p in required if p not in PERSONALITIES]
        if missing:
            return False, f"missing prompts: {missing}"
        config_prompt = cfg.get("agents", {}).get("exploration_prompt", DEFAULT_PERSONALITY)
        valid = config_prompt in PERSONALITIES
        return valid, f"free_agency_v2, brief_v1 present; config uses '{config_prompt}' (valid={valid})"

    # === Adapters configured ===
    @check("adapters configured")
    def _():
        from playground.config import load_config
        cfg = load_config(overrides=overrides)
        adapters = list(cfg.get("adapters", {}).keys())
        return len(adapters) > 0, f"{len(adapters)} adapters: {', '.join(adapters)}"

    # === Model endpoints (informational) ===
    @check("model endpoints (informational)")
    def _():
        import httpx
        out = []
        try:
            r = httpx.get("http://127.0.0.1:1234/v1/models", timeout=3)
            models = r.json().get("data", [])
            out.append(f"LM Studio up ({len(models)} model{'s' if len(models) != 1 else ''})")
        except Exception:  # noqa: BLE001
            out.append("LM Studio not reachable")
        return True, "; ".join(out)

    failed = [n for n, ok, _ in results if not ok]
    print(json.dumps({"healthy": not failed, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
