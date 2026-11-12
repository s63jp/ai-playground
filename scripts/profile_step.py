"""Profile: where does step time go, and what cheap wins exist?"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.config import load_config  # noqa: E402
from playground.world.physics import PhysicsWorld  # noqa: E402

cfg = load_config()
w = PhysicsWorld(cfg)
m, d = w.m, w.d
print("nv", m.nv, "ngeom", m.ngeom, "ncon at rest", d.ncon, "nefc", d.nefc)
on_shelf = int(np.sum((w.geom_oidx[d.contact.geom[:d.ncon]] == w.shelf_oi).any(axis=1)))
print("contacts with the hidden shelf:", on_shelf)

N = 5000
t = time.perf_counter()
for _ in range(N):
    mujoco.mj_step(m, d)
raw = time.perf_counter() - t
print(f"raw mj_step      : {N / raw:8,.0f} steps/s")

t = time.perf_counter()
for _ in range(N):
    w._step()
wrapped = time.perf_counter() - t
print(f"wrapper _step    : {N / wrapped:8,.0f} steps/s  (python overhead {(wrapped - raw) / N * 1e6:.1f} us/step)")

t = time.perf_counter()
for _ in range(N):
    w._post_step()
print(f"_post_step alone : {(time.perf_counter() - t) / N * 1e6:.1f} us/call")

print("sleep flag available:", hasattr(mujoco.mjtEnableBit, "mjENBL_SLEEP"),
      "| island flag:", hasattr(mujoco.mjtEnableBit, "mjENBL_ISLAND"))
print("enable bits:", [n for n in dir(mujoco.mjtEnableBit) if n.startswith("mjENBL")])
print("solver:", m.opt.solver, "iterations", m.opt.iterations, "cone", m.opt.cone, "jacobian", m.opt.jacobian)
