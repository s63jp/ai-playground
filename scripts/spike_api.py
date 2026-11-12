"""Probe the MuJoCo 4.x Python API surface the engine wrapper relies on."""
import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="box" pos="0 0 -0.5" size="5 5 0.5"/>
    <body name="a" pos="0 0 0.25"><freejoint name="a_j"/><geom name="a_g" type="box" size="0.25 0.25 0.25" mass="5"/></body>
    <body name="b" pos="0 0 0.75"><freejoint name="b_j"/><geom name="b_g" type="box" size="0.25 0.25 0.25" mass="5" group="3"/></body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
for _ in range(200):
    mujoco.mj_step(m, d)

print("ncon", d.ncon)
c = d.contact
for attr in ("geom1", "geom2", "geom", "dist", "frame", "pos", "efc_address"):
    try:
        v = getattr(c, attr)
        print(f"contact.{attr}: type={type(v).__name__} shape={getattr(v, 'shape', None)}")
    except Exception as e:  # noqa: BLE001
        print(f"contact.{attr}: ERROR {type(e).__name__}: {e}")

print("geom_bodyid", m.geom_bodyid, "geom_group", m.geom_group, "contype", m.geom_contype)
print("body a id", m.body("a").id, "jnt qposadr", m.joint("a_j").qposadr, "dofadr", m.joint("a_j").dofadr)
print("body_mass writable:", m.body_mass.flags.writeable, "gravcomp:", m.body_gravcomp)

# ray with geomgroup mask (exclude group 3)
gid = np.zeros(1, dtype=np.int32)
mask = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)
dist = mujoco.mj_ray(m, d, np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), mask, 1, -1, gid)
print("ray masked: hit", m.geom(int(gid[0])).name, "z", round(5 - dist, 4))
dist = mujoco.mj_ray(m, d, np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
print("ray all   : hit", m.geom(int(gid[0])).name, "z", round(5 - dist, 4))
dist = mujoco.mj_ray(m, d, np.array([9.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
print("ray miss  : dist", dist, "geom", int(gid[0]))

# contact force + object velocity
f6 = np.zeros(6)
mujoco.mj_contactForce(m, d, 0, f6)
print("contact force[0]", np.round(f6, 3))
v6 = np.zeros(6)
mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("a").id, v6, 0)
print("objectVelocity (rot, lin)", np.round(v6, 5))
print("freejoint qvel layout: lin", d.qvel[0:3], "ang", d.qvel[3:6])

# toggling collisions at runtime
m.geom_contype[m.geom("b_g").id] = 0
m.geom_conaffinity[m.geom("b_g").id] = 0
mujoco.mj_forward(m, d)
print("ncon after disabling b:", d.ncon)
print("mj_state size:", mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
print("contact frame row0 (normal):", np.round(np.array(d.contact.frame[0][:3]), 3) if d.ncon else None)
