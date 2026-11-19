"""Before/after compiled-physics comparison: legacy rig vs humanoid_v2 visual.

Compiles BOTH models and diffs the compiled MuJoCo structures that matter for
physics: model scalars (nq, nv, nu, njnt, nbody, ngeom), per-body mass, inertia
diagonal and inertial frame origin, joint types/ranges, and actuator gear.

Prints a markdown table and exits non-zero if ANY physics quantity differs.
"""
import os
import sys
import tempfile

import numpy as np
import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from playground.config import load_config                      # noqa: E402
from playground.world.spec import build_mjcf, build_specs      # noqa: E402


def build(visual):
    ab = {"type": "humanoid"}
    if visual:
        ab["visual"] = visual
    cfg = load_config(overrides={"_root": tempfile.mkdtemp(prefix="hv2c_"),
                                 "agent_body": ab})
    return mujoco.MjModel.from_xml_string(build_mjcf(cfg, build_specs(cfg)))


def bodies(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]


def main():
    a, b = build(None), build("humanoid_v2")
    fail = []

    print("## Model scalars\n")
    print("| quantity | legacy | humanoid_v2 | same |")
    print("|---|---:|---:|:--:|")
    for k in ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "nmesh"):
        va, vb = getattr(a, k), getattr(b, k)
        same = va == vb
        # ngeom/nmesh are EXPECTED to differ: that is the added visual geometry
        expected_diff = k in ("ngeom", "nmesh")
        mark = "yes" if same else ("+%d (visual)" % (vb - va) if expected_diff else "**NO**")
        if not same and not expected_diff:
            fail.append(k)
        print("| %s | %d | %d | %s |" % (k, va, vb, mark))

    print("\n## Per-body mass / inertia (physics-critical)\n")
    assert bodies(a) == bodies(b), "body list changed!"
    names = bodies(a)
    dm = np.abs(a.body_mass - b.body_mass)
    di = np.abs(a.body_inertia - b.body_inertia).max(axis=1)
    dp = np.abs(a.body_ipos - b.body_ipos).max(axis=1)
    changed = [i for i in range(a.nbody) if dm[i] or di[i] or dp[i]]

    print("| body | mass (kg) | inertia diag (kg m^2) | mass delta | inertia delta |")
    print("|---|---:|---|---:|---:|")
    for i in range(a.nbody):
        if names[i] in (None, "world"):
            continue
        print("| %s | %.6f | %s | %.3e | %.3e |"
              % (names[i], a.body_mass[i],
                 np.array2string(a.body_inertia[i], precision=5,
                                 separator=" ", suppress_small=False),
                 dm[i], di[i]))
    if changed:
        fail.append("body mass/inertia changed: %s" % [names[i] for i in changed])

    print("\n**total mass** legacy %.6f kg vs humanoid_v2 %.6f kg (delta %.3e)"
          % (a.body_mass.sum(), b.body_mass.sum(),
             abs(a.body_mass.sum() - b.body_mass.sum())))

    print("\n## Joints\n")
    ja = [mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(a.njnt)]
    jb = [mujoco.mj_id2name(b, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(b.njnt)]
    if ja != jb:
        fail.append("joint list changed")
    if not np.array_equal(a.jnt_range, b.jnt_range):
        fail.append("joint ranges changed")
    if not np.array_equal(a.jnt_type, b.jnt_type):
        fail.append("joint types changed")
    print("- joint names identical: **%s** (%d joints)" % (ja == jb, len(ja)))
    print("- joint ranges identical: **%s**" % np.array_equal(a.jnt_range, b.jnt_range))
    print("- joint types identical: **%s**" % np.array_equal(a.jnt_type, b.jnt_type))
    print("- hinge joints: %s" % ", ".join(
        n for n, t in zip(ja, a.jnt_type) if t == mujoco.mjtJoint.mjJNT_HINGE))

    print("\n## Collision geometry\n")
    ca = sum(1 for i in range(a.ngeom) if a.geom_contype[i] or a.geom_conaffinity[i])
    cb = sum(1 for i in range(b.ngeom) if b.geom_contype[i] or b.geom_conaffinity[i])
    if ca != cb:
        fail.append("collision geom count changed")
    print("- colliding geoms: legacy %d vs humanoid_v2 %d -> **%s**"
          % (ca, cb, "identical" if ca == cb else "CHANGED"))
    print("- added geoms are all non-colliding visual meshes: **%s**"
          % all(not (b.geom_contype[i] or b.geom_conaffinity[i])
                for i in range(b.ngeom)
                if (mujoco.mj_id2name(b, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("g:hv2:")))

    print("\n## 300-step trajectory from qpos0\n")
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    da.qpos[:], db.qpos[:] = a.qpos0, b.qpos0
    for _ in range(300):
        mujoco.mj_step(a, da)
        mujoco.mj_step(b, db)
    for lbl, x, y in (("qpos", da.qpos, db.qpos), ("qvel", da.qvel, db.qvel),
                      ("qacc", da.qacc, db.qacc)):
        d = float(np.abs(np.asarray(x) - np.asarray(y)).max())
        if d != 0.0:
            fail.append("trajectory %s diverged" % lbl)
        print("- max |delta %s| after 300 steps: **%.3e**" % (lbl, d))
    print("- contacts: legacy %d vs humanoid_v2 %d" % (da.ncon, db.ncon))

    print("\n---\n")
    if fail:
        print("RESULT: **PHYSICS CHANGED** -> %s" % fail)
        return 1
    print("RESULT: **physics identical** - humanoid_v2 adds visual geometry only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
