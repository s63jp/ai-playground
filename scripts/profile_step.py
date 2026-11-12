#!/usr/bin/env python3
"""Profile: where does step time go, and what cheap wins exist?

Usage:
    python scripts/profile_step.py [--workspace PATH] [--steps N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playground.config import load_config  # noqa: E402
from playground.world.physics import PhysicsWorld  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile physics step performance")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Experiment workspace path")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of steps to profile (default: 5000)")
    args = parser.parse_args()

    overrides = {}
    if args.workspace:
        overrides["_root"] = args.workspace

    cfg = load_config(overrides=overrides)
    w = PhysicsWorld(cfg)
    m, d = w.m, w.d
    N = args.steps

    print(f"nv {m.nv}  ngeom {m.ngeom}  ncon at rest {d.ncon}  nefc {d.nefc}")

    # Raw mj_step
    t = time.perf_counter()
    for _ in range(N):
        mujoco.mj_step(m, d)
    raw = time.perf_counter() - t
    print(f"raw mj_step      : {N / raw:8,.0f} steps/s")

    # Reset for wrapper test
    w = PhysicsWorld(cfg)

    # Wrapped _step
    t = time.perf_counter()
    for _ in range(N):
        w._step()
    wrapped = time.perf_counter() - t
    overhead_us = (wrapped - raw) / N * 1e6
    print(f"wrapper _step    : {N / wrapped:8,.0f} steps/s  (python overhead {overhead_us:.1f} us/step)")

    # Post-step alone
    t = time.perf_counter()
    for _ in range(N):
        w._post_step()
    post = time.perf_counter() - t
    print(f"_post_step alone : {post / N * 1e6:.1f} us/call")

    # Feature flags
    print(f"sleep flag available: {hasattr(mujoco.mjtEnableBit, 'mjENBL_SLEEP')}  "
          f"| island flag: {hasattr(mujoco.mjtEnableBit, 'mjENBL_ISLAND')}")
    enable_bits = [n for n in dir(mujoco.mjtEnableBit) if n.startswith("mjENBL")]
    print(f"enable bits: {enable_bits}")
    print(f"solver: {m.opt.solver}  iterations {m.opt.iterations}  cone {m.opt.cone}  jacobian {m.opt.jacobian}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
