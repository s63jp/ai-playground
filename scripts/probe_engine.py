"""Engine probe: exercise the physics wrapper end to end and print real measurements."""
import hashlib
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


t = time.perf_counter()
w = fresh()
print(f"build+settle: {time.perf_counter() - t:.2f}s  objects active={len(w.active)} latent={len(w.latent)}")
print("agent:", w.agent_state()["position"], "grounded", w.grounded, "on", w.support)
print("block_01:", w.object_state("block_01")["position"], "ring:", w.object_state("ring")["position"])

# 1. jump
j = w.jump()
print("JUMP  apex=%.4f air=%.4f takeoff_vz=%.4f" % (j["apex_above_takeoff"], j["air_time"], j["takeoff_velocity"][2]))

# 2. walk to a block, pick up, stationary throw up
m = w.move(parse_direction([-2.5, -3.0 + 1.0, 0]), float(np.hypot(2.5, 2.0)))
print("MOVE  end=", m["end_position"], "elapsed", m["elapsed"])
pk = w.pick_up("block_01")
print("PICK  total_mass", pk["total_mass"])
th = w.throw("block_01", parse_direction("up"), 150.0)
print("THROW stationary up: max_above_launch=%.4f flight=%.4f release_speed=%.4f" % (
    th["max_height_above_launch"], th["flight_time"], th["release_speed"]))

# 3. jump-throw
w2 = fresh()
w2.move(parse_direction([-2.5, -2.0, 0]), float(np.hypot(2.5, 2.0)))
w2.pick_up("block_01")
jl = w2.jump()
print("JUMP carrying 5kg apex=%.4f" % jl["apex_above_takeoff"])
w2.jump(wait="takeoff")
th2 = w2.throw("block_01", parse_direction("up"), 150.0)
print("THROW at takeoff up: max_above_launch=%.4f abs_max=%.4f release_speed=%.4f agent_v=%s" % (
    th2["max_height_above_launch"], th2["max_height"], th2["release_speed"], th2["agent_velocity_at_release"]))

# 4. tower of five
w3 = fresh()
base = np.array([-2.5, -3.0])
w3.move(parse_direction([base[0], base[1] + 0.9, 0]), float(np.hypot(base[0], base[1] + 0.9)))
print("agent at", w3.agent_state()["position"])
ok = True
for i, b in enumerate(["block_02", "block_03", "block_07", "block_08"]):
    try:
        w3.pick_up(b)
        r = w3.place(b, [base[0], base[1]])
        print(f"  place {b}: final={r['final_position']} on={r['resting_on']} stayed={r['stayed_where_placed']}")
    except ActionError as e:
        ok = False
        print("  FAIL", b, e.code, e.message, e.details)
w3.settle(3.0)
print("TOWER heights:", [w3.object_state(b)["position"][2] for b in ["block_01", "block_02", "block_03", "block_07", "block_08"]])

# 5. spire reach from the terrace: climb onto terrace needs help (2.0 high)
w4 = fresh()
try:
    w4.move(parse_direction("east"), 4.3)
except ActionError as e:
    print("MOVE east blocked:", e.code, e.details.get("blocked_by"), e.details.get("end_position"))
try:
    print(w4.climb(parse_direction("east")))
except ActionError as e:
    print("CLIMB terrace from ground:", e.code, e.message)
w4.jump(wait="apex")
try:
    print("JUMP+CLIMB:", w4.climb(parse_direction("east")))
except ActionError as e:
    print("JUMP+CLIMB terrace:", e.code, e.message)

# 6. ring rule
w5 = fresh()
w5.move(parse_direction([3.0, 3.0 - 0.9, 0]), float(np.hypot(3.0, 2.1)))
w5.pick_up("ring")
w5.move(parse_direction([-2.5 - 3.0, -3.0 + 0.9 - 2.1, 0]), float(np.hypot(-5.5, -4.2)))
print("agent", w5.agent_state()["position"])
r = w5.place("ring", [-2.5, -3.0, 0.9])
print("RING place:", r["final_position"], r["resting_on"])
w5.wait(2.0)
print("events:", [(e["type"], e["data"].get("object")) for e in w5.events if e["type"] in ("object_appeared",)])
print("active blocks:", sorted(o for o in w5.active if o.startswith("block")))
print("block_13:", w5.object_state("block_13")["position"] if "block_13" in w5.active else None)

# 7. determinism
def run():
    x = fresh()
    x.move(parse_direction([-2.5, -2.0, 0]), float(np.hypot(2.5, 2.0)))
    x.pick_up("block_01")
    x.jump(wait="takeoff")
    x.throw("block_01", parse_direction([1, 0.3, 1]), 120.0)
    return hashlib.sha256(x.d.qpos.tobytes() + x.d.qvel.tobytes()).hexdigest()


a, b = run(), run()
print("DETERMINISM:", a == b, a[:16])
t = time.perf_counter()
w.step(5000)
el = time.perf_counter() - t
print(f"throughput with wrapper: {5000 / el:,.0f} steps/s = {5000 * w.dt / el:.1f}x real time")
