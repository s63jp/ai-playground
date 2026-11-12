#!/usr/bin/env python3
"""Engine probe: real physics measurements for researcher validation.

Prints real measurements from the MuJoCo engine:
  - Jump apex (standing vs with carried mass)
  - Throw peak (standing vs jump-throw)
  - Tower stacking stability
  - Ring generation rule
  - Determinism check (two identical runs hash-equal)
  - Throughput (steps/s)

Usage:
    python scripts/engine_probe.py [--workspace PATH] [--json] [--quiet]

The script works against any workspace (or a temp dir if none given).
"""
from __future__ import annotations

import argparse
import hashlib
import json as jsonlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playground.config import load_config  # noqa: E402
from playground.world.mathutil import parse_direction  # noqa: E402
from playground.world.physics import ActionError, PhysicsWorld  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Engine physics probe - real measurements")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Experiment workspace path (temp dir used if not given)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", action="store_true", help="Only output final summary")
    args = parser.parse_args()

    workspace = Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="pg_probe_"))
    cfg = load_config(overrides={"_root": str(workspace)})
    results: dict = {"workspace": str(workspace), "measurements": {}}

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg)

    def fresh() -> PhysicsWorld:
        return PhysicsWorld(cfg)

    def goto(w: PhysicsWorld, x: float, y: float) -> None:
        p = w.agent_pos()
        d = np.array([x - p[0], y - p[1], 0.0])
        dist = float(np.linalg.norm(d))
        if dist > 0.01:
            w.move(d / dist, dist)

    # --- 1. Jump apex (standing) ---
    t0 = time.perf_counter()
    w = fresh()
    build_time = time.perf_counter() - t0
    log(f"build+settle: {build_time:.2f}s  objects active={len(w.active)} latent={len(w.latent)}")

    j = w.jump()
    jump_apex = j["apex_above_takeoff"]
    log(f"JUMP (standing):  apex={jump_apex:.4f} m  air_time={j['air_time']:.4f} s")
    results["measurements"]["jump_apex_standing"] = round(jump_apex, 4)

    # --- 2. Jump apex with carried mass ---
    w2 = fresh()
    goto(w2, -2.5, -2.1)
    w2.pick_up("block_01")
    j2 = w2.jump()
    jump_apex_loaded = j2["apex_above_takeoff"]
    log(f"JUMP (carrying 5kg):  apex={jump_apex_loaded:.4f} m")
    results["measurements"]["jump_apex_loaded_5kg"] = round(jump_apex_loaded, 4)
    results["measurements"]["carried_mass_reduces_jump"] = jump_apex_loaded < jump_apex * 0.95

    # --- 3. Standing throw peak ---
    w3 = fresh()
    goto(w3, -2.5, -2.1)
    w3.pick_up("block_01")
    th1 = w3.throw("block_01", parse_direction("up"), 150.0)
    standing_throw_peak = th1["max_height"]
    log(f"THROW (standing, 150N up):  max_height={standing_throw_peak:.4f} m  "
        f"max_above_launch={th1['max_height_above_launch']:.4f} m  flight_time={th1['flight_time']:.4f} s")
    results["measurements"]["standing_throw_150N_max_height"] = round(standing_throw_peak, 4)

    # --- 4. Jump-throw peak (release at takeoff) ---
    w4 = fresh()
    goto(w4, -2.5, -2.1)
    w4.pick_up("block_01")
    w4.jump(wait="takeoff")
    th2 = w4.throw("block_01", parse_direction("up"), 150.0)
    jump_throw_peak = th2["max_height"]
    log(f"THROW (jump at takeoff, 150N up):  max_height={jump_throw_peak:.4f} m  "
        f"agent_vz_at_release={th2['agent_velocity_at_release'][2]:.4f}")
    results["measurements"]["jump_throw_150N_max_height"] = round(jump_throw_peak, 4)
    results["measurements"]["jump_throw_goes_higher"] = jump_throw_peak > standing_throw_peak + 0.5

    # --- 5. Three-block tower stability (max reachable from ground) ---
    w5 = fresh()
    base = (-2.5, -3.0)
    goto(w5, base[0], base[1] + 0.9)
    placed = 1  # block_01 is already at base
    for b in ["block_02", "block_03"]:  # Only stack 3 - higher is unreachable from ground
        try:
            bp = w5.obj_pos(b)
            goto(w5, bp[0], bp[1] + 0.9)
            w5.pick_up(b)
            goto(w5, base[0], base[1] + 0.9)
            r = w5.place(b, [base[0], base[1]])
            if r.get("stayed_where_placed"):
                placed += 1
        except ActionError as e:
            log(f"  tower place failed: {b} [{e.code}] {e.message}")
            break
    w5.settle(3.0)
    heights = [round(float(w5.obj_pos(f"block_0{i}")[2]), 4) for i in range(1, 4)]
    nominal = [0.25, 0.75, 1.25]
    max_drift = max(abs(h - n) for h, n in zip(heights, nominal))
    log(f"TOWER (3 blocks):  heights={heights}  max_drift={max_drift:.4f} m  placed={placed}")
    results["measurements"]["tower_3block_heights"] = heights
    results["measurements"]["tower_max_drift_m"] = round(max_drift, 4)
    results["measurements"]["tower_stable"] = max_drift < 0.02 and placed >= 3

    # --- 6. Ring generation rule ---
    w6 = fresh()
    goto(w6, 3.0, 2.1)
    w6.pick_up("ring")
    goto(w6, -2.5, -1.75)
    try:
        w6.place("ring", [-2.5, -3.0])
        before = sorted(o for o in w6.active if o.startswith("block"))
        w6.wait(2.5)
        after = sorted(o for o in w6.active if o.startswith("block"))
        new_blocks = sorted(set(after) - set(before))
        appeared_events = [e for e in w6.events if e["type"] == "object_appeared"]
        ring_rule_worked = len(new_blocks) > 0 and any(
            e.get("hidden", {}).get("cause") == "ring_generation" for e in appeared_events
        )
        log(f"RING RULE:  blocks before={len(before)} after={len(after)}  new={new_blocks}  worked={ring_rule_worked}")
        results["measurements"]["ring_rule_generated_blocks"] = new_blocks
        results["measurements"]["ring_rule_worked"] = ring_rule_worked
    except ActionError as e:
        log(f"RING RULE:  placement failed [{e.code}]")
        results["measurements"]["ring_rule_worked"] = False

    # --- 7. Determinism check ---
    def determinism_run():
        x = fresh()
        goto(x, -2.5, -2.1)
        x.pick_up("block_01")
        x.jump(wait="takeoff")
        x.throw("block_01", parse_direction([1, 0.3, 1]), 120.0)
        return hashlib.sha256(x.d.qpos.tobytes() + x.d.qvel.tobytes()).hexdigest()

    h1, h2 = determinism_run(), determinism_run()
    deterministic = h1 == h2
    log(f"DETERMINISM:  equal={deterministic}  hash={h1[:16]}")
    results["measurements"]["determinism_equal"] = deterministic

    # --- 8. Throughput ---
    w7 = fresh()
    t = time.perf_counter()
    w7.step(5000)
    elapsed = time.perf_counter() - t
    steps_per_sec = int(5000 / elapsed)
    realtime_factor = 5000 * w7.dt / elapsed
    log(f"THROUGHPUT:  {steps_per_sec:,} steps/s = {realtime_factor:.1f}x real time")
    results["measurements"]["throughput_steps_per_sec"] = steps_per_sec
    results["measurements"]["throughput_realtime_factor"] = round(realtime_factor, 1)

    # --- Summary ---
    all_ok = (
        results["measurements"]["carried_mass_reduces_jump"] and
        results["measurements"]["jump_throw_goes_higher"] and
        results["measurements"]["tower_stable"] and
        results["measurements"]["determinism_equal"]
    )
    results["all_checks_passed"] = all_ok

    if args.json:
        print(jsonlib.dumps(results, indent=2))
    else:
        log(f"\n{'='*60}")
        log(f"All physics checks passed: {all_ok}")
        if not args.quiet:
            log(f"Workspace: {workspace}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
