"""Probe: runtime body-mass change (carried load) and frictionless-priority agent geom."""
import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <default><geom condim="4" friction="0.9 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <geom name="ground" type="box" pos="0 0 -0.5" size="12 12 0.5"/>
    <body name="agent" pos="0 0 0.801">
      <joint name="ax" type="slide" axis="1 0 0"/><joint name="ay" type="slide" axis="0 1 0"/><joint name="az" type="slide" axis="0 0 1"/>
      <geom name="agent_geom" type="box" size="0.22 0.22 0.8" mass="50" friction="0 0 0" priority="1" condim="1"/>
    </body>
    <body name="blk" pos="2 0 0.25"><freejoint/><geom type="box" size="0.25 0.25 0.25" mass="5"/></body>
  </worldbody>
</mujoco>
"""


def jump(extra_mass, set_const):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    for _ in range(300):
        mujoco.mj_step(m, d)
    aid = m.body("agent").id
    zq = m.joint("az").qposadr[0]
    zd = m.joint("az").dofadr[0]
    xd = m.joint("ax").dofadr[0]
    q_before = d.qpos.copy()
    if extra_mass:
        m.body_mass[aid] += extra_mass
        if set_const:
            mujoco.mj_setConst(m, d)
    print(f"  qpos preserved after mass change: {np.allclose(q_before, d.qpos)}  mass={m.body_mass[aid]}")
    z0 = d.qpos[zq]
    peak = z0
    for i in range(1500):
        d.qfrc_applied[:] = 0
        if i < 50:
            d.qfrc_applied[zd] = 2200.0
        if i < 50:
            d.qfrc_applied[xd] = 100.0   # sideways shove to check frictionless slide
        mujoco.mj_step(m, d)
        peak = max(peak, d.qpos[zq])
    return peak - z0, d.qpos[m.joint("ax").qposadr[0]]


for extra, sc in ((0.0, False), (5.0, False), (5.0, True), (12.0, True)):
    apex, x = jump(extra, sc)
    mt = 50 + extra
    a = 2200 / mt - 9.81
    v = a * 0.1
    analytic = 0.5 * a * 0.01 + v * v / (2 * 9.81)
    print(f"extra={extra:5.1f} setConst={sc!s:5}  apex={apex:.4f}  analytic={analytic:.4f}  x_drift={x:.3f}")
