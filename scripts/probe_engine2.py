"""Engine probe 2: the scenarios the first probe got wrong or did not reach.

Prints real measurements for: overlap refusal, the hidden ring rule, a five-block
tower, whether the spire can be reached at all (and how), reaction forces on a
stack that is jumped from, and wrapper throughput.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.config import load_config  # noqa: E402
from playground.world.mathutil import parse_direction  # noqa: E402
from playground.world.physics import ActionError, PhysicsWorld  # noqa: E402

cfg = load_config()


def fresh():
    return PhysicsWorld(cfg)


def goto(w, x, y):
    p = w.agent_pos()
    d = np.array([x - p[0], y - p[1], 0.0])
    dist = float(np.linalg.norm(d))
    if dist > 0.01:
        return w.move(d / dist, dist)


def attempt(label, fn):
    try:
        r = fn()
        print(f"{label}: OK", r if not isinstance(r, dict) else {k: r[k] for k in list(r)[:6]})
        return r
    except ActionError as e:
        print(f"{label}: REFUSED [{e.code}] {e.message}")
        return None


# --- 1. overlap refusal (was silently launching the ring)
w = fresh()
goto(w, 3.0, 2.1)
w.pick_up("ring")
goto(w, -2.5, -2.1)
attempt("1a place ring overlapping own body", lambda: w.place("ring", [-2.5, -3.0, 0.9]))
print("   still holding:", w.held, "| ring active:", "ring" in w.active)

# --- 2. hidden ring rule: lay the ring flat around a block
goto(w, -2.5, -1.75)
r = attempt("2a place ring around block_01 (auto height)", lambda: w.place("ring", [-2.5, -3.0]))
if r:
    print("   ring final:", r["final_position"], "resting_on:", r["resting_on"])
before = sorted(o for o in w.active if o.startswith("block"))
w.wait(2.5)
after = sorted(o for o in w.active if o.startswith("block"))
print("2b blocks before/after 2.5 s:", len(before), "->", len(after), "| new:", sorted(set(after) - set(before)))
print("   object_appeared events:", [(e["data"], e.get("hidden")) for e in w.events if e["type"] == "object_appeared"])
w.wait(3.0)
print("2c after 3 more s without re-arming:", len([o for o in w.active if o.startswith("block")]), "blocks")

# --- 3. five-block tower, moving between pick-ups
w = fresh()
base = (-2.5, -3.0)
placed = 1
for b in ["block_02", "block_03", "block_04", "block_05"]:
    bp = w.obj_pos(b)
    goto(w, bp[0], bp[1] + 0.9)
    w.pick_up(b)
    goto(w, base[0], base[1] + 0.9)
    try:
        r = w.place(b, [base[0], base[1]])
        placed += 1 if r.get("stayed_where_placed") else 0
    except ActionError as e:
        print("   tower place failed:", b, e.code, e.message)
        break
w.settle(3.0)
tops = [round(float(w.obj_pos(b)[2]), 4) for b in ["block_01", "block_02", "block_03", "block_04", "block_05"]]
print("3  tower centre heights:", tops, "| placed:", placed, "| nominal [0.25,0.75,1.25,1.75,2.25]")

# --- 4. can the terrace and the spire be reached?
w = fresh()
try:
    goto(w, 4.0, 0.0)
except ActionError as e:
    print("4a walk to terrace foot:", e.code, e.details.get("end_position"))
attempt("4b climb terrace from ground", lambda: w.climb(parse_direction("east")))
w.jump(wait="apex")
attempt("4c jump then climb terrace", lambda: w.climb(parse_direction("east")))
w.settle(2.0)

# build a 2-block step against the terrace wall, then climb up it
w = fresh()
for n, b in enumerate(["block_06", "block_05"]):
    bp = w.obj_pos(b)
    goto(w, bp[0], bp[1] + 0.9)
    w.pick_up(b)
    goto(w, 3.3, 0.0)
    r = attempt(f"4d place {b} at terrace foot", lambda: w.place(b, [4.3, 0.0]))
print("   step stack:", [rnd for rnd in (w.object_state("block_06")["top_height"], w.object_state("block_05")["top_height"])])
r = attempt("4e climb onto the 2-block step", lambda: w.climb(parse_direction("east")))
if r:
    print("   feet now:", round(w.feet(), 4), "standing on", w.support)
    r = attempt("4f climb from step onto terrace", lambda: w.climb(parse_direction("east")))
    print("   feet now:", round(w.feet(), 4), "standing on", w.support)
if w.feet() > 1.9:
    goto(w, 7.0, 0.0)
    print("4g on terrace beside spire:", np.round(w.agent_pos(), 3), "| spire top is", round(6.0 - w.feet(), 3), "m above feet")
    attempt("4h climb spire from terrace", lambda: w.climb(parse_direction("east")))
    j = w.jump()
    print("4i plain jump on terrace: apex above takeoff", j.get("apex_above_takeoff"), "| needs", round(6.0 - 2.0, 3))
    w.jump(wait="apex")
    attempt("4j jump then climb spire", lambda: w.climb(parse_direction("east")))
    w.settle(2.0)

# --- 5. reaction force: jumping off a single loose block
w = fresh()
goto(w, -2.5, -2.3)
attempt("5a climb onto block_01", lambda: w.climb(parse_direction("south")))
print("   standing on", w.support, "feet", round(w.feet(), 4))
b0 = w.obj_pos("block_01").copy()
j = w.jump()
print("5b jump off block: apex", j.get("apex_above_takeoff"), "| block moved", np.round(w.obj_pos("block_01") - b0, 4))

# --- 6. throughput
w = fresh()
t = time.perf_counter()
w.step(10000)
el = time.perf_counter() - t
print(f"6  wrapper throughput at rest: {10000 / el:,.0f} steps/s = {10000 * w.dt / el:.1f}x real time")
