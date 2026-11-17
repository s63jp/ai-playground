#!/usr/bin/env python3
"""Profile: where does step time go, and what cheap wins exist?

Usage:
    python scripts/profile_step.py [--workspace PATH] [--steps N] [--benchmark]

Default behaviour is the legacy raw-vs-wrapper profile. --benchmark adds the
physics-extension suite: box vs humanoid agent_body step time (median/p95 over
>= 200 steps, warm-up excluded), scaling with movable-object count (5/25/50)
and RSS memory delta. Results are also written to
experiments/perf/<timestamp>_step.json as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playground.config import load_config  # noqa: E402
from playground.world.physics import PhysicsWorld  # noqa: E402


def _rss_mb() -> float:
    """Resident set size in MB (psutil if present, else resource/proc fallbacks)."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        pass
    try:
        # Windows fallback via psapi
        import ctypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return float(pmc.WorkingSetSize) / (1024 * 1024)
    except Exception:
        return float("nan")


def _steps_median_p95(world, steps: int, warmup: int = 50) -> tuple[float, float, float]:
    """Run `steps` world._step() calls; return (median_ms, p95_ms, steps_per_s)."""
    for _ in range(warmup):
        world._step()
    samples: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        world._step()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples)
    return float(np.median(arr)), float(np.percentile(arr, 95)), steps / float(arr.sum() / 1000.0)


def _bench(body_type: str, steps: int, extras: dict | None = None) -> dict:
    overrides = {"agent_body": {"type": body_type}}
    if extras:
        overrides.update(extras)
    world = PhysicsWorld(load_config(overrides=overrides))
    med, p95, speed = _steps_median_p95(world, steps)
    return {"agent_body": body_type, "extras": extras or {},
            "median_ms": round(med, 3), "p95_ms": round(p95, 3),
            "steps_per_s": round(speed, 1), "timestep": world.dt}


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile physics step performance")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Experiment workspace path")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of steps to profile (default: 5000)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the physics-extension benchmark suite")
    args = parser.parse_args()

    overrides = {}
    if args.workspace:
        overrides["_root"] = args.workspace

    cfg = load_config(overrides=overrides)
    w = PhysicsWorld(cfg)
    m, d = w.m, w.d
    N = args.steps
    report: dict = {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "steps": N, "mujoco": mujoco.__version__}

    print(f"nv {m.nv}  ngeom {m.ngeom}  ncon at rest {d.ncon}  nefc {d.nefc}")

    # Raw mj_step
    t = time.perf_counter()
    for _ in range(N):
        mujoco.mj_step(m, d)
    raw = time.perf_counter() - t
    report["raw_mj_step_per_s"] = round(N / raw)
    print(f"raw mj_step      : {N / raw:8,.0f} steps/s")

    # Wrapped _step
    w = PhysicsWorld(cfg)
    t = time.perf_counter()
    for _ in range(N):
        w._step()
    wrapped = time.perf_counter() - t
    overhead_us = (wrapped - raw) / N * 1e6
    report["wrapper_step_per_s"] = round(N / wrapped)
    report["python_overhead_us_per_step"] = round(overhead_us, 1)
    print(f"wrapper _step    : {N / wrapped:8,.0f} steps/s  (python overhead {overhead_us:.1f} us/step)")

    # Post-step alone
    t = time.perf_counter()
    for _ in range(N):
        w._post_step()
    post = time.perf_counter() - t
    report["post_step_us_per_call"] = round(post / N * 1e6, 1)
    print(f"_post_step alone : {post / N * 1e6:.1f} us/call")

    # Feature flags
    print(f"sleep flag available: {hasattr(mujoco.mjtEnableBit, 'mjENBL_SLEEP')}  "
          f"| island flag: {hasattr(mujoco.mjtEnableBit, 'mjENBL_ISLAND')}")
    enable_bits = [n for n in dir(mujoco.mjtEnableBit) if n.startswith("mjENBL")]
    print(f"enable bits: {enable_bits}")
    print(f"solver: {m.opt.solver}  iterations {m.opt.iterations}  cone {m.opt.cone}  jacobian {m.opt.jacobian}")

    if args.benchmark:
        bench_steps = max(200, min(N, 400))
        rss0 = _rss_mb()
        bench = []
        for body in ("box", "humanoid"):
            try:
                bench.append(_bench(body, bench_steps))
            except Exception as exc:  # noqa: BLE001
                bench.append({"agent_body": body, "error": f"{type(exc).__name__}: {exc}"})
        scaling = []
        for count in (5, 25, 50):
            try:
                scaling.append(_bench("box", bench_steps, {"world": {"blocks": {"count": count}}}))
            except Exception as exc:  # noqa: BLE001
                scaling.append({"blocks": count, "error": f"{type(exc).__name__}: {exc}"})
        rss1 = _rss_mb()
        report["benchmark"] = {"agent_bodies": bench, "object_scaling": scaling,
                               "rss_start_mb": round(rss0, 1), "rss_end_mb": round(rss1, 1),
                               "rss_delta_mb": round(rss1 - rss0, 1)}
        print(json.dumps(report["benchmark"], indent=2))

    out_dir = ROOT / "experiments" / "perf"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_step.json"
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"written: {out_file}")
    except Exception as exc:  # noqa: BLE001
        print(f"(experiments/perf not writable: {exc})")
    print(json.dumps({k: v for k, v in report.items() if k != "benchmark"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
