#!/usr/bin/env python3
"""M3 §9 — measure real physics speed with N humanoids in one world.

Reports wall-clock steps/second, contact counts and total mass per world size
(1, 2 and 4 independently controlled humanoids). No estimation: every number
is measured from this run. Usage:

    ./.venv/Scripts/python.exe scripts/profile_multiagent.py [--seconds 4] [--agents 1 2 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.config import load_config
from playground.world.physics import PhysicsWorld

import numpy as np


def build(n: int) -> PhysicsWorld:
    cfg = load_config()
    cfg["agent_body"]["type"] = "humanoid"
    cfg["agent_bodies"] = {}
    for i in range(n):
        aid = f"agent_{i + 1:03d}"
        cfg["agent_bodies"][aid] = {"start": [-3.0 + 2.4 * i, 2.4 if i else 0.0]}
    return PhysicsWorld(cfg)


def probe(n: int, seconds: float) -> dict:
    world = build(n)
    # every agent starts walking east at a modest speed: steady-state locomotion cost
    for aid in world._agents:
        world.set_gait("walk", speed=0.6, agent=aid)
    n_steps = int(seconds / world.dt)
    t0 = time.perf_counter()
    for i in range(n_steps):
        world._step()
        if i == n_steps // 2:
            mid_contacts = int(world.d.ncon)
    wall = time.perf_counter() - t0
    # stability: every agent upright, no NaNs
    states = [world.agent_state(aid) for aid in world._agents]
    z = [float(s["position"][2]) for s in states]
    up = [float(s["position"][2]) > 0.55 * s["body"]["height"] / 2 + 0.4 or not world.agent_fallen(aid)
          for aid, s in zip(world._agents, states)]
    nq = world.m.nq
    qacc_max = float(np.max(np.abs(world.d.qacc))) if world.d.qacc.size else 0.0
    mem_mb = 0.0
    return {
        "agents": n,
        "nq": nq,
        "steps": n_steps,
        "wall_s": round(wall, 3),
        "steps_per_s": round(n_steps / wall, 1),
        "realtime_factor": round(seconds / wall, 1),
        "contacts_midrun": mid_contacts,
        "z_after": [round(v, 3) for v in z],
        "all_upright_alive": all(up),
        "max_abs_qacc": round(qacc_max, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile N-humanoid physics speed (M3 §9).")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--agents", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()
    rows = [probe(n, args.seconds) for n in args.agents]
    print(json.dumps(rows, indent=2))
    out = Path(__file__).resolve().parent.parent / "experiments" / "multiagent_perf.json"
    out.write_text(json.dumps({"measured": rows}, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
