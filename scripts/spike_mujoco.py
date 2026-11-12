"""Spike: validate MuJoCo for the playground before committing to the architecture.

Checks: API surface on the installed version, bitwise determinism, stability of a
5-block tower with a heavy slide-jointed body on top, ray casting, contact force
readout, runtime mass change, and raw step throughput.
"""
import hashlib
import time

import mujoco
import numpy as np

XML = """
<mujoco model="spike">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <default>
    <geom friction="0.9 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001" condim="4"/>
  </default>
  <worldbody>
    <geom name="ground" type="box" pos="0 0 -0.5" size="12 12 0.5"/>
    {blocks}
    <body name="agent" pos="0 0 {agent_z}">
      <joint name="ax" type="slide" axis="1 0 0"/>
      <joint name="ay" type="slide" axis="0 1 0"/>
      <joint name="az" type="slide" axis="0 0 1"/>
      <geom name="agent_geom" type="box" size="0.22 0.22 0.8" mass="50" friction="0 0 0"/>
    </body>
    <body name="rock" pos="3 0 0.3">
      <freejoint/>
      <geom name="rock_geom" type="sphere" size="0.3" mass="12" condim="6" friction="0.9 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(n_blocks=5):
    blocks = "\n".join(
        f'<body name="b{i}" pos="0 0 {0.25 + 0.5 * i}"><freejoint/>'
        f'<geom name="g{i}" type="box" size="0.25 0.25 0.25" mass="5"/></body>'
        for i in range(n_blocks)
    )
    agent_z = 0.5 * n_blocks + 0.8 + 0.001
    m = mujoco.MjModel.from_xml_string(XML.format(blocks=blocks, agent_z=agent_z))
    return m, mujoco.MjData(m)


def run(steps=5000):
    m, d = build()
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return m, d


print("mujoco", mujoco.__version__)
m, d = run(5000)  # 10 s
h1 = hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes()).hexdigest()
m2, d2 = run(5000)
h2 = hashlib.sha256(d2.qpos.tobytes() + d2.qvel.tobytes()).hexdigest()
print("determinism bitwise equal:", h1 == h2, h1[:16])

# Tower stability: each block should still be at x,y ~ 0 and its nominal height.
worst_xy, worst_z = 0.0, 0.0
for i in range(5):
    p = d.body(f"b{i}").xpos
    worst_xy = max(worst_xy, float(np.hypot(p[0], p[1])))
    worst_z = max(worst_z, abs(float(p[2]) - (0.25 + 0.5 * i)))
ap = d.body("agent").xpos
print(f"tower after 10s with 50kg body on top: worst xy drift={worst_xy*1000:.3f} mm, "
      f"worst z error={worst_z*1000:.3f} mm, agent feet z={ap[2]-0.8:.4f} (nominal 2.5)")
print("max |qvel| at rest:", float(np.abs(d.qvel).max()))

# Contact force readout
f6 = np.zeros(6)
tot = 0.0
for i in range(d.ncon):
    c = d.contact[i]
    n1 = m.geom(c.geom1).name
    n2 = m.geom(c.geom2).name
    if "agent_geom" in (n1, n2):
        mujoco.mj_contactForce(m, d, i, f6)
        tot += f6[0]
print(f"normal force on agent from contacts: {tot:.2f} N (weight {50*9.81:.2f} N), ncon={d.ncon}")

# Ray cast straight down onto the tower top (exclude nothing)
geomid = np.zeros(1, dtype=np.int32)
dist = mujoco.mj_ray(m, d, np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
print("ray hit:", m.geom(int(geomid[0])).name, "at z =", round(10.0 - dist, 4))

# Runtime mass change affects dynamics (needed for carried-load physics)
m3, d3 = build(0)
for _ in range(500):
    mujoco.mj_step(m3, d3)
aid = m3.body("agent").id
zadr = m3.joint("az").dofadr[0]
d3.qvel[zadr] = 200.0 / m3.body_mass[aid]
z0 = d3.qpos[m3.joint("az").qposadr[0]]
peak = z0
for _ in range(1000):
    mujoco.mj_step(m3, d3)
    peak = max(peak, d3.qpos[m3.joint("az").qposadr[0]])
print(f"jump apex with impulse 200 N*s on 50 kg: {peak - z0:.4f} m (analytic {4.0**2/(2*9.81):.4f})")

# Throughput
m4, d4 = build(5)
t = time.perf_counter()
for _ in range(20000):
    mujoco.mj_step(m4, d4)
el = time.perf_counter() - t
print(f"throughput: {20000/el:,.0f} steps/s -> {20000*0.002/el:.1f}x real time")
