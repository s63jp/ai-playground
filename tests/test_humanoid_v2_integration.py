"""Integration tests for the humanoid_v2 visual geometry in the EXISTING MuJoCo
rig (spec section 5).

HAPPY    - the integrated model loads and carries every expected visual mesh
VALIDATE - physics/contract preserved: bodies, joints, masses, RL dims,
           ground contact, symmetry, connectivity, dynamics, articulation
ERROR    - negative controls: flag off adds nothing, unknown flag fails closed,
           missing mesh file must not pass silently

Physics preservation is asserted at three levels: model scalars (nq/nv/nu/njnt),
per-body mass/inertia (exact equality), and an actual 200-step trajectory from
qpos0 compared against the legacy model.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mujoco
from playground.config import load_config
from playground.world.spec import build_mjcf, build_specs

MESH_DIR = os.path.join(REPO, "assets", "meshes", "humanoid_v2")
MANIFEST = json.load(open(os.path.join(MESH_DIR, "manifest.json"), encoding="utf-8"))
PARTS = sorted(MANIFEST["parts"])                      # 35 expected visuals
ALL_JOINTS = ["head_pitch", "head_yaw", "head_roll", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle",
              "spine_L2", "waist_yaw"]


def _model(visual=None):
    ab = {"type": "humanoid"}
    if visual:
        ab["visual"] = visual
    cfg = load_config(overrides={"_root": tempfile.mkdtemp(prefix="hv2t_"),
                                 "agent_body": ab})
    xml = build_mjcf(cfg, build_specs(cfg))
    return cfg, xml, mujoco.MjModel.from_xml_string(xml)


@pytest.fixture(scope="module")
def legacy():
    return _model(None)


@pytest.fixture(scope="module")
def integrated():
    return _model("humanoid_v2")


def mesh_geoms(m):
    """part name -> geom id for the hv2 visual geoms."""
    out = {}
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith("g:hv2:"):
            out[name.split(":", 2)[2]] = i
    return out


def _qrot(q, v):
    """Rotate Nx3 points by quaternion q = (w, x, y, z)."""
    w, x, y, z = [float(t) for t in q]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return v @ R.T


def mesh_world(m, d, gid):
    """World-space vertices of a mesh geom.

    MuJoCo keeps mesh_vert in the GEOM's own frame (already bbox-centred), and
    the compiler sets geom_pos = mesh_pos plus geom_quat = mesh_quat, so
    d.geom_xpos/xmat carry every offset and axis rotation:
        world = geom_xmat (x) mesh_vert + geom_xpos
    Verified against the source OBJs: foot/head/chest/hand z-extents come out
    0..0.096, 1.525..1.755, 1.19..1.44, 0.645..0.835 - the designed values.
    Re-applying mesh_quat (or using body xpos) silently scrambles axes and was
    the source of earlier 9-component / 2.05 m height failures.
    """
    vid = m.geom_dataid[gid]
    v = np.asarray(m.mesh_vert[m.mesh_vertadr[vid]: m.mesh_vertadr[vid]
                               + m.mesh_vertnum[vid]], dtype=np.float64)
    R = np.asarray(d.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    return v @ R.T + np.asarray(d.geom_xpos[gid], dtype=np.float64)


def settle(m, steps=200):
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for _ in range(steps):
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)
    return d


# ------------------------------------------------------------------- HAPPY ---

def test_happy_integrated_model_loads(integrated):
    cfg, xml, m = integrated
    assert m.nmesh == 35, m.nmesh
    assert 'meshdir=' in xml and 'hv2_' in xml


def test_happy_all_expected_visual_meshes_present(integrated):
    _, _, m = integrated
    found = mesh_geoms(m)
    assert sorted(found) == PARTS, {"missing": sorted(set(PARTS) - set(found)),
                                    "extra": sorted(set(found) - set(PARTS))}
    assert len(found) == 35


def test_happy_mesh_files_exist_on_disk():
    missing = [p for p in PARTS if not os.path.exists(
        os.path.join(MESH_DIR, p + ".obj"))]
    assert missing == [], missing
    assert MANIFEST["source"]["sha256"]


# ---------------------------------------------------------------- VALIDATE ---

def test_validate_physical_bodies_and_joints_intact(legacy, integrated):
    _, _, a = legacy
    _, _, b = integrated
    assert (b.nq, b.nv, b.nu, b.njnt, b.nbody) == (a.nq, a.nv, a.nu, a.njnt, a.nbody)
    ja = [mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(a.njnt)]
    jb = [mujoco.mj_id2name(b, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(b.njnt)]
    assert ja == jb
    for jn in ALL_JOINTS:                                # un-prefixed legacy rig
        assert jn in jb, jn
    # collision geoms unchanged (only non-colliding visuals added)
    coll_a = sum(int(a.geom_contype[i] or a.geom_conaffinity[i]) for i in range(a.ngeom))
    coll_b = sum(int(b.geom_contype[i] or b.geom_conaffinity[i]) for i in range(b.ngeom))
    assert coll_b == coll_a, (coll_a, coll_b)
    assert b.ngeom - a.ngeom == 35


def test_validate_mass_and_inertia_preserved_exactly(legacy, integrated):
    """Adding mesh geoms must not shift mass: bodies without an explicit
    <inertial> derive it from geoms, which is why inertials.json exists."""
    _, _, a = legacy
    _, _, b = integrated
    assert np.array_equal(a.body_mass, b.body_mass), "body_mass drifted"
    assert np.array_equal(a.body_inertia, b.body_inertia), "body_inertia drifted"
    assert np.array_equal(a.body_ipos, b.body_ipos), "body_ipos drifted"
    assert abs(a.body_mass.sum() - b.body_mass.sum()) == 0.0


def test_validate_rl_contract_unchanged(integrated):
    """v4 contract: 68 observations / 15 actions, waist_yaw IS in the action
    space (action 15, index 14). Attaching visual meshes must not touch it."""
    from playground.rl.env import LocoEnv
    assert LocoEnv.obs_dim == 68, LocoEnv.obs_dim
    assert LocoEnv.action_dim == 15, LocoEnv.action_dim
    assert LocoEnv.obs_dim == 43 + 15 + 2 + 3 + 1 + 4, "v4 layout changed"
    # waist joins the policy under v4; only the foot yaws stay passive
    assert "waist_yaw" not in LocoEnv._EXCLUDED
    assert LocoEnv._EXCLUDED == ("left_foot_yaw", "right_foot_yaw")
    assert "waist_yaw" in LocoEnv._EXCLUDED_V3, "v3 legacy mode must still exclude it"
    _, _xml, m = integrated
    assert m.nu == 0, "visual integration must not add actuators"


def test_validate_waist_is_action_15(integrated):
    """The brief's 'waist as action 15' - index 14 in the frozen v4 order."""
    from playground.rl.env import LocoEnv
    env = LocoEnv({"agent_body": {"type": "humanoid", "visual": "humanoid_v2"}})
    env.reset()
    names = list(env.joint_names)
    assert len(names) == 15, names
    assert names[14] == "waist_yaw", names
    obs = env.observe() if hasattr(env, "observe") else None
    if obs is not None:
        assert len(obs) == 68, len(obs)


def test_validate_visual_geometry_stands_on_ground(integrated):
    _, _, m = integrated
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    g = mesh_geoms(m)
    allv = np.vstack([mesh_world(m, d, g[p]) for p in PARTS])
    height = allv[:, 2].max() - allv[:, 2].min()
    assert 1.70 <= height <= 1.80, height                      # ~1.755 m figure
    sole = min(mesh_world(m, d, g[p])[:, 2].min() for p in ("foot_L", "foot_R"))
    assert abs(sole) < 1e-4, sole                             # feet at ground level
    # left/right symmetry of the visual set
    for p in [x for x in PARTS if x.endswith("_L")]:
        a = mesh_world(m, d, g[p]).copy()
        b = mesh_world(m, d, g[p[:-2] + "_R"])
        a[:, 0] *= -1.0
        assert a.shape == b.shape
        a = a[np.lexsort((a[:, 2], a[:, 1], np.round(a[:, 0], 6)))]
        b = b[np.lexsort((b[:, 2], b[:, 1], np.round(b[:, 0], 6)))]
        assert np.abs(a - b).max() < 1e-5, p


def test_validate_no_floating_visual_parts(integrated):
    _, _, m = integrated
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    g = mesh_geoms(m)
    boxes = {p: (mesh_world(m, d, g[p]).min(axis=0),
                 mesh_world(m, d, g[p]).max(axis=0)) for p in PARTS}
    parent = {p: p for p in PARTS}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    names = sorted(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.all(boxes[a][0] <= boxes[b][1] + 1e-6) and \
                    np.all(boxes[b][0] <= boxes[a][1] + 1e-6):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    comps = len({find(p) for p in PARTS})
    assert comps == 1, f"{comps} disconnected visual components"


def test_validate_dynamics_identical(legacy, integrated):
    """200 steps from qpos0: the integrated model must track the legacy one
    exactly (same masses, inertias, contacts => same trajectory)."""
    _, _, a = legacy
    _, _, b = integrated
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    da.qpos[:] = a.qpos0
    db.qpos[:] = b.qpos0
    for _ in range(200):
        for m, d in ((a, da), (b, db)):
            mujoco.mj_forward(m, d)
            mujoco.mj_step(m, d)
    assert np.abs(da.qpos - db.qpos).max() == 0.0
    assert np.abs(da.qvel - db.qvel).max() == 0.0
    assert np.abs(da.qacc - db.qacc).max() == 0.0
    assert da.ncon == db.ncon


def test_validate_mesh_geoms_never_collide(integrated):
    _, _, m = integrated
    g = mesh_geoms(m)
    bad = [p for p, i in g.items()
           if m.geom_contype[i] or m.geom_conaffinity[i]]
    assert bad == [], bad
    d = settle(m, 200)
    names = {i for i in g.values()}
    touching = [c for c in d.contact[:d.ncon]
                if int(c.geom[0]) in names or int(c.geom[1]) in names]
    assert touching == [], f"visual mesh took {len(touching)} contacts"


def test_validate_articulation_moves_visuals_and_restores(integrated):
    """Temporary joint deflections: the visual meshes must follow their bodies,
    then qpos0 must restore the exact rest pose."""
    _, _, m = integrated
    g = mesh_geoms(m)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    rest = {p: mesh_world(m, d, g[p]) for p in PARTS}
    assert np.isfinite(d.qpos).all()

    jid = {jn: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
           for jn in ALL_JOINTS if "waist" not in jn}
    assert all(i >= 0 for i in jid.values())
    targets = {"head_pitch": 0.30, "head_yaw": 0.40, "head_roll": 0.20,
               "left_shoulder": -0.50, "left_elbow": 0.80, "left_hip": 0.40,
               "left_knee": -0.60, "left_ankle": 0.20}
    for jn, val in targets.items():
        d.qpos[m.jnt_qposadr[jid[jn]]] = val
    mujoco.mj_forward(m, d)
    assert np.isfinite(d.qpos).all(), "NaN in qpos after articulation"
    assert np.isfinite(d.xpos).all(), "NaN in body positions after articulation"

    moved, unmoved = [], []
    for p in PARTS:
        delta = float(np.abs(mesh_world(m, d, g[p]) - rest[p]).max())
        left_chain = p.endswith("_L") or p in ("head", "faceplate", "neck",
                                               "neck_ring_1", "neck_ring_2")
        (moved if left_chain else unmoved).append((p, delta))
    assert min(d for _, d in moved) > 1e-4, \
        f"articulated visuals did not move: {sorted(moved, key=lambda t: t[1])[:4]}"
    assert max(d for _, d in unmoved) < 1e-9, \
        f"right-side visuals moved unexpectedly: {sorted(unmoved, key=lambda t: -t[1])[:4]}"

    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    for p in PARTS:
        assert np.array_equal(mesh_world(m, d, g[p]), rest[p]), \
            f"{p} did not return to its rest pose"


# ------------------------------------------------------------------ ERROR ---

def test_error_flag_off_adds_nothing(legacy):
    """Without agent_body.visual the generated XML must be untouched."""
    _, xml, m = legacy
    assert "hv2" not in xml and "meshdir" not in xml
    assert m.nmesh == 0
    assert m.ngeom == 60, m.ngeom            # legacy geom count, no visuals


def test_error_unknown_visual_value_fails_closed():
    """A typo must fall back to the legacy model, never a half-applied state."""
    _, xml, m = _model("humanoid_v3")
    assert "hv2" not in xml
    assert m.nmesh == 0


def test_error_missing_mesh_file_is_reported(integrated):
    """Point meshdir at an empty dir: MuJoCo must refuse to load (loud failure,
    not a silently mesh-less model)."""
    _, xml, _ = integrated
    empty = tempfile.mkdtemp(prefix="hv2empty_")
    broken = xml
    import re
    broken = re.sub(r'meshdir="[^"]*"', 'meshdir="%s"' % empty.replace("\\", "/"),
                    broken, count=1)
    assert broken != xml, "meshdir not found in XML - injection missing?"
    with pytest.raises((mujoco.FatalError, ValueError)):
        mujoco.MjModel.from_xml_string(broken)     # must fail LOUDLY, never load


# -------------------------------------------------------------- CHECKPOINT ---

def test_checkpoint_roundtrip_under_the_visual_model():
    """A checkpoint saved against the humanoid_v2 world must load back, and a
    LEGACY-world checkpoint must load against the visual world.

    This is the real guarantee for existing brains: load_checkpoint() verifies
    obs/action dims, section slices, joint order, mapping and PD gains, so if
    attaching visual meshes perturbed the RL contract in any way the
    fingerprint check would reject the file.
    """
    pytest.importorskip("torch")
    from playground.rl.env import LocoEnv
    from playground.rl.ppo import TanhGaussianActor, Critic
    from playground.rl.checkpoint import save_checkpoint, load_checkpoint

    env = LocoEnv({"agent_body": {"type": "humanoid", "visual": "humanoid_v2"}})
    env.reset()
    assert env.obs_dim == 68 and env.action_dim == 15
    path = os.path.join(tempfile.mkdtemp(prefix="hv2ck_"), "ckpt.json")
    save_checkpoint(path, actor=TanhGaussianActor(env.obs_dim, env.action_dim),
                    critic=Critic(env.obs_dim), env=env,
                    trainer_state={"step": 1})
    assert load_checkpoint(path, env=env) is not None   # raises on drift

    legacy = LocoEnv({"agent_body": {"type": "humanoid"}})
    legacy.reset()
    lpath = os.path.join(tempfile.mkdtemp(prefix="hv2ck2_"), "legacy.json")
    save_checkpoint(lpath,
                    actor=TanhGaussianActor(legacy.obs_dim, legacy.action_dim),
                    critic=Critic(legacy.obs_dim), env=legacy,
                    trainer_state={"step": 1})
    # the two worlds share ONE contract - that is what visual-only means
    assert load_checkpoint(lpath, env=env) is not None
