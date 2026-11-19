"""Humanoid V2 visual retargeting, Phase 1 HVE suite (Happy / Validate / Error).

Live tests build the real LocoEnv the way rl_live does (agent_body humanoid) and
drive MuJoCo with mj_forward only, so no simulation state is ever stepped by the
joint tests. The retargeting module itself is checked against independent formulas.
"""
from __future__ import annotations

import ast
import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np
import pytest

from playground.rl.env import LocoEnv
from playground.rl_live import humanoid_v2_retarget as R
from playground.rl_live.frames import FrameIndex, rig_body_names

AMT = 0.3  # rad perturbation applied to one hinge at a time
HINGES = [
    "waist_yaw", "spine_L2",
    "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
    "head_pitch", "head_yaw", "head_roll",
    "left_hip", "left_knee", "left_ankle", "left_foot_yaw",
    "right_hip", "right_knee", "right_ankle", "right_foot_yaw",
]


# ---------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def skel():
    return R.load_skeleton()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = LocoEnv(cfg={"_root": str(tmp_path_factory.mktemp("v2_env")),
                     "agent_body": {"type": "humanoid"}})
    e.reset(seed=0)
    return e


@pytest.fixture(scope="module")
def env_step(tmp_path_factory):
    e = LocoEnv(cfg={"_root": str(tmp_path_factory.mktemp("v2_step")),
                     "agent_body": {"type": "humanoid"}})
    e.reset(seed=0)
    return e


def _frame(env):
    idx = FrameIndex(env)
    idx.ensure()
    return idx.body_dict()


def _reset_frame(env):
    env.reset(seed=0)
    return _frame(env)


def _capture(env, joint: str, amount: float):
    """Reference frame, then the frame after perturbing only `joint` with mj_forward."""
    ref = _reset_frame(env)
    m, d = env.world.m, env.world.d
    adr = int(m.joint(joint).qposadr[0])
    d.qpos[adr] += amount
    mujoco.mj_forward(m, d)
    pert = _frame(env)
    return ref, pert


def _subtree(m, body_name: str) -> set[str]:
    """Rig bodies whose MuJoCo ancestor chain includes body_name (inclusive)."""
    target = int(m.body(body_name).id)
    out = set()
    for name in R.RIG_BODIES:
        cur = int(m.body(name).id)
        while cur != 0:
            if cur == target:
                out.add(name)
                break
            cur = int(m.body_parentid[cur])
    return out


def _skeleton_data():
    return json.loads(R.SKELETON_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Happy: static mapping
def test_skeleton_counts(skel):
    assert len(skel.bones) == 19
    assert len(skel.meshes) == 35
    assert skel.excluded_meshes == ("setup_floor",)


def test_bone_hierarchy_is_a_tree_rooted_at_pelvis(skel):
    roots = [b for b, v in skel.bones.items() if v["parent"] is None]
    assert roots == ["pelvis"]
    assert skel.bones["clavicle.L"]["parent"] == "chest"
    assert skel.bones["hand.R"]["parent"] == "forearm.R"
    assert skel.bones["foot.L"]["parent"] == "shin.L"


def test_bone_and_mesh_classification_counts(skel):
    bone_modes = {}
    for b in skel.bones:
        mode, _ = R.bone_driver(b, skel)
        bone_modes[mode] = bone_modes.get(mode, 0) + 1
    assert bone_modes == {"direct": 11, "approximate": 2, "rigid": 6}
    mesh_modes = {}
    for m in skel.meshes:
        mode, _ = R.bone_driver(skel.meshes[m]["bone"], skel)
        mesh_modes[mode] = mesh_modes.get(mode, 0) + 1
    assert mesh_modes == R.EXPECTED_MODE_MESH_COUNTS


def test_every_bone_resolves_to_a_rig_body(skel):
    for b in skel.bones:
        _, body = R.bone_driver(b, skel)
        assert body in R.RIG_BODIES


def test_direct_and_approximate_drivers_are_one_to_one(skel):
    bodies = list(R.DIRECT_BONE_BODY.values()) + list(R.APPROX_BONE_BODY.values())
    assert len(bodies) == len(set(bodies)) == 13


def test_left_right_bones_and_meshes_stay_distinct(skel):
    assert len(set(skel.bones)) == 19 and len(set(skel.meshes)) == 35
    for bone, body in R.DIRECT_BONE_BODY.items():
        if bone.endswith(".L"):
            assert body.startswith("left_"), bone
            assert skel.bones[bone]["head_world"][0] > 0, bone
        if bone.endswith(".R"):
            assert body.startswith("right_"), bone
            assert skel.bones[bone]["head_world"][0] < 0, bone
    for mesh, info in skel.meshes.items():
        if mesh.endswith("_L"):
            assert info["bone"].endswith(".L"), mesh
        if mesh.endswith("_R"):
            assert info["bone"].endswith(".R"), mesh


def test_module_is_pure_python_and_imports_no_simulator_or_browser():
    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert not roots & {"mujoco", "three", "playground", "requests", "js"}
    assert roots <= {"__future__", "json", "collections", "dataclasses", "pathlib", "numpy"}


# ---------------------------------------------------------------- Happy: transform maths
def test_quaternion_convention_matches_mujoco(env):
    env.reset(seed=0)
    m, d = env.world.m, env.world.d
    mujoco.mj_forward(m, d)
    checked = 0
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if name not in R.RIG_BODIES:
            continue
        np.testing.assert_allclose(R.quat_to_mat3(d.xquat[b]),
                                   d.xmat[b].reshape(3, 3), atol=1e-9)
        checked += 1
    assert checked == 16


def test_identity_at_reference_rest_pose(env, skel):
    frame = _reset_frame(env)
    rest = R.RestPose.from_frame(frame, skel)
    out = R.retarget(frame, rest)
    for body, D in out["deltas"].items():
        np.testing.assert_allclose(D, np.eye(4), atol=1e-9, err_msg=body)
    for bone in skel.bones:
        np.testing.assert_allclose(out["bones"][bone],
                                   rest.A @ skel.bones[bone]["rest_world"], atol=1e-9)


def test_rest_alignment_residuals_are_finite_and_bounded(env, skel):
    rest = R.RestPose.from_frame(_reset_frame(env), skel)
    assert set(rest.residuals) == {b for b in skel.bones
                                   if rest.drivers[b][0] in ("direct", "approximate")}
    for bone, r in rest.residuals.items():
        assert np.all(np.isfinite(r))
        assert np.linalg.norm(r) < 0.3, f"{bone} pivot offset {np.linalg.norm(r):.3f} m"


def test_waist_yaw_is_represented_in_torso_quaternion_and_delta(env, skel):
    ref, pert = _capture(env, "waist_yaw", AMT)
    half = AMT / 2.0
    np.testing.assert_allclose(pert["torso"]["q"],
                               [math.cos(half), 0.0, 0.0, math.sin(half)], atol=1e-4)
    rest = R.RestPose.from_frame(ref, skel)
    out = R.retarget(pert, rest)
    c, s = math.cos(AMT), math.sin(AMT)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    np.testing.assert_allclose(out["deltas"]["torso"][:3, :3], Rz, atol=1e-4)


def test_pivot_compensation_matches_independent_formula(env, skel):
    """Bone head after a body rotation: p(t) + Rd (H_A - p0), with Rd = R(t) R0^T."""
    ref, pert = _capture(env, "left_elbow", AMT)
    rest = R.RestPose.from_frame(ref, skel)
    out = R.retarget(pert, rest)
    checked = 0
    for bone, (mode, body) in rest.drivers.items():
        if mode not in ("direct", "approximate"):
            continue
        R0 = R.quat_to_mat3(ref[body]["q"])
        Rt = R.quat_to_mat3(pert[body]["q"])
        p0 = np.asarray(ref[body]["p"], float)
        pt = np.asarray(pert[body]["p"], float)
        Rd = Rt @ R0.T
        H_A = (rest.A @ np.append(skel.bones[bone]["head_world"], 1.0))[:3]
        expected = pt + Rd @ (H_A - p0)
        # bone_world[:3, 3] is the bone origin (= head) after retargeting
        np.testing.assert_allclose(out["bones"][bone][:3, 3], expected, atol=2e-4, err_msg=bone)
        checked += 1
    assert checked == 13


def test_no_accumulated_drift_over_repeated_frames(env, skel):
    ref = _reset_frame(env)
    rest = R.RestPose.from_frame(ref, skel)
    first = R.retarget(ref, rest)
    _, pert = _capture(env, "left_hip", AMT)
    for _ in range(300):
        R.retarget(pert, rest)
        R.retarget(ref, rest)
    again = R.retarget(ref, rest)
    for bone in first["bones"]:
        np.testing.assert_array_equal(first["bones"][bone], again["bones"][bone])
    for mesh in first["meshes"]:
        np.testing.assert_array_equal(first["meshes"][mesh], again["meshes"][mesh])


def test_retarget_does_not_mutate_input_frame(env, skel):
    frame = _reset_frame(env)
    before = copy.deepcopy(frame)
    rest = R.RestPose.from_frame(frame, skel)
    R.retarget(frame, rest)
    assert frame == before


# ---------------------------------------------------------------- Happy: live names & joints
def test_live_rig_body_names_match_the_map_exactly(env):
    assert tuple(rig_body_names(env)) == R.RIG_BODIES
    assert set(_frame(env)) == set(R.RIG_BODIES)


def test_live_model_has_exactly_the_17_expected_hinges(env):
    m = env.world.m
    hinges = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in range(m.njnt) if int(m.jnt_type[j]) == 3]
    assert sorted(hinges) == sorted(HINGES)
    assert len(hinges) == 17


@pytest.mark.parametrize("joint", HINGES)
def test_each_live_joint_moves_only_its_subtree(env, skel, joint):
    ref, pert = _capture(env, joint, AMT)
    m = env.world.m
    jbody = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                              int(m.joint(joint).bodyid[0]))
    subtree = _subtree(m, jbody)
    rest = R.RestPose.from_frame(ref, skel)
    out_ref = R.retarget(ref, rest)
    out_pert = R.retarget(pert, rest)

    # the joint's own body rotates by exactly AMT about a world axis
    assert abs(R.rotation_angle(out_pert["deltas"][jbody]) - AMT) < 2e-3, jbody
    for body in R.RIG_BODIES:
        if body in subtree:
            assert R.rotation_angle(out_pert["deltas"][body]) > 1e-3 or \
                not np.allclose(out_pert["deltas"][body], np.eye(4), atol=1e-6), body
        else:
            # unrelated bodies: bit-identical delta, so no unintended movement
            np.testing.assert_array_equal(out_pert["deltas"][body],
                                          out_ref["deltas"][body], err_msg=body)

    responders, silent = [], []
    for bone in skel.bones:
        _, drv = rest.drivers[bone]
        if drv in subtree:
            assert not np.array_equal(out_pert["bones"][bone], out_ref["bones"][bone]), bone
            responders.append(bone)
        else:
            np.testing.assert_array_equal(out_pert["bones"][bone],
                                          out_ref["bones"][bone], err_msg=bone)
            silent.append(bone)
    for mesh, info in skel.meshes.items():
        _, drv = rest.drivers[info["bone"]]
        if drv in subtree:
            assert not np.array_equal(out_pert["meshes"][mesh], out_ref["meshes"][mesh]), mesh
        else:
            np.testing.assert_array_equal(out_pert["meshes"][mesh],
                                          out_ref["meshes"][mesh], err_msg=mesh)
    print(f"\n[{joint}] body={jbody} subtree={sorted(subtree)} "
          f"responding_bones={len(responders)} silent_bones={len(silent)}")


def test_all_35_meshes_assigned_once_and_match_independent_formula(env, skel):
    ref, pert = _capture(env, "left_knee", AMT)
    rest = R.RestPose.from_frame(ref, skel)
    out = R.retarget(pert, rest)
    assert len(out["meshes"]) == 35
    assert sorted(out["meshes"]) == sorted(skel.meshes)
    for mesh, info in skel.meshes.items():
        _, body = rest.drivers[info["bone"]]
        R0 = R.quat_to_mat3(ref[body]["q"])
        Rt = R.quat_to_mat3(pert[body]["q"])
        p0 = np.asarray(ref[body]["p"], float)
        pt = np.asarray(pert[body]["p"], float)
        Rd = Rt @ R0.T
        X = rest.A @ info["matrix_world"]          # rest placement, applied once
        expected = np.eye(4)
        expected[:3, :3] = Rd @ X[:3, :3]
        expected[:3, 3] = pt + Rd @ (X[:3, 3] - p0)
        np.testing.assert_allclose(out["meshes"][mesh], expected, atol=2e-4, err_msg=mesh)


def test_rigid_follow_meshes_take_their_verified_driver_delta(env, skel):
    ref, pert = _capture(env, "left_elbow", AMT)
    rest = R.RestPose.from_frame(ref, skel)
    out = R.retarget(pert, rest)
    rigid = [m for m, i in skel.meshes.items() if rest.drivers[i["bone"]][0] == "rigid"]
    assert len(rigid) == 8
    for mesh in rigid:
        bone = skel.meshes[mesh]["bone"]
        _, body = rest.drivers[bone]
        assert body in ("left_elbow", "spine_L2", "torso") or body.startswith("right_")
    # hands hang off the elbow: each hand mesh equals the elbow delta applied once to its rest placement
    for mesh in ("hand_L", "wrist_L", "hand_R", "wrist_R"):
        _, body = rest.drivers[skel.meshes[mesh]["bone"]]
        assert body in ("left_elbow", "right_elbow")
        expected = out["deltas"][body] @ rest.A @ skel.meshes[mesh]["matrix_world"]
        np.testing.assert_array_equal(out["meshes"][mesh], expected, err_msg=mesh)
        expected_angle = AMT if body == "left_elbow" else 0.0
        assert abs(R.rotation_angle(out["deltas"][body]) - expected_angle) < 2e-3, body


# ---------------------------------------------------------------- Happy: reset & stationarity
def test_reset_restores_identical_reference_and_no_accumulation(env, skel):
    frames = []
    for _ in range(4):
        frames.append(_reset_frame(env))
        m, d = env.world.m, env.world.d
        d.qpos[int(m.joint("left_knee").qposadr[0])] += 0.4
        mujoco.mj_forward(m, d)
    for f in frames[1:]:
        assert f == frames[0]
    rests = [R.RestPose.from_frame(f, skel) for f in frames]
    for r in rests[1:]:
        np.testing.assert_array_equal(r.A, rests[0].A)
        for b in skel.bones:
            np.testing.assert_array_equal(R.retarget(frames[0], r)["bones"][b],
                                          R.retarget(frames[0], rests[0])["bones"][b])


def test_settled_sim_is_stationary_and_stationary_frames_give_identical_output(env_step, skel):
    env_step.reset(seed=0)
    zero = np.zeros(env_step.action_dim, dtype=np.float32)
    history = []
    settled_steps = None
    for k in range(400):
        _obs, _reward, term, trunc, _info = env_step.step(zero)
        f = _frame(env_step)
        history.append(f)
        if term or trunc:
            break
        if len(history) >= 11:
            worst = max(np.max(np.abs(np.asarray(history[-1][n]["p"]) - np.asarray(history[-1 - i][n]["p"])))
                        for n in R.RIG_BODIES for i in range(1, 11))
            if worst < 1e-4:
                settled_steps = k + 1
                break
    print(f"\nsettling: steps={len(history)} settled_at={settled_steps}")
    frame = history[-1]
    rest = R.RestPose.from_frame(frame, skel)
    a = R.retarget(frame, rest)
    b = R.retarget(_frame(env_step), rest)     # no state change in between
    for bone in skel.bones:
        np.testing.assert_array_equal(a["bones"][bone], b["bones"][bone])
    assert settled_steps is not None, "rig did not settle within 400 zero-action steps"


# ---------------------------------------------------------------- Validate: bad inputs
def test_load_rejects_wrong_bone_count():
    data = _skeleton_data()
    data["bones"].pop()
    with pytest.raises(R.RetargetError):
        R.skeleton_from_dict(data)


def test_mesh_bound_to_two_bones_is_rejected():
    data = _skeleton_data()
    mesh = next(m for m in data["meshes"] if m["name"] == "head")
    mesh["dominant_bone_counts"] = {"head": 400, "neck": mesh["n_verts"] - 400}
    with pytest.raises(R.RetargetError):
        R.skeleton_from_dict(data)


def test_unbound_mesh_is_rejected():
    data = _skeleton_data()
    mesh = next(m for m in data["meshes"] if m["name"] == "head")
    mesh["unweighted_verts"] = 1
    with pytest.raises(R.RetargetError):
        R.skeleton_from_dict(data)


def test_secondary_weights_are_rejected():
    data = _skeleton_data()
    mesh = next(m for m in data["meshes"] if m["name"] == "head")
    mesh["verts_with_secondary_weight_gt_0.01"] = 3
    with pytest.raises(R.RetargetError):
        R.skeleton_from_dict(data)


def test_frame_with_missing_rig_body_fails_closed(env, skel):
    frame = _reset_frame(env)
    del frame["torso"]
    with pytest.raises(R.RetargetError, match="missing"):
        R.RestPose.from_frame(frame, skel)


def test_prefixed_multiagent_body_name_fails_closed(env, skel):
    frame = _reset_frame(env)
    frame["agent_001:torso"] = frame.pop("torso")
    with pytest.raises(R.RetargetError, match="mismatch"):
        R.RestPose.from_frame(frame, skel)


def test_unexpected_extra_body_fails_closed(env, skel):
    frame = _reset_frame(env)
    frame["block_01"] = {"p": [0, 0, 0], "q": [1, 0, 0, 0]}
    with pytest.raises(R.RetargetError, match="unexpected"):
        R.retarget(frame, R.RestPose.from_frame(_reset_frame(env), skel))


@pytest.mark.parametrize("bad", [
    {"p": [0.0, 0.0, float("nan")], "q": [1, 0, 0, 0]},
    {"p": [0.0, 0.0, 0.0], "q": [0, 0, 0, 0]},
    {"p": [0.0, 0.0], "q": [1, 0, 0, 0]},
])
def test_non_finite_zero_or_malformed_pose_rejected(env, skel, bad):
    frame = _reset_frame(env)
    frame["head"] = bad
    with pytest.raises(R.RetargetError):
        R.RestPose.from_frame(frame, skel)


def test_frame_must_be_a_mapping(skel):
    with pytest.raises(R.RetargetError):
        R.validate_frame([1, 2, 3])


# ---------------------------------------------------------------- Error: unresolved mapping
def test_unmapped_root_bone_raises(skel):
    data = _skeleton_data()
    for b in data["bones"]:
        if b["name"] == "neck":
            b["parent"] = None
    s = R.skeleton_from_dict(data)
    with pytest.raises(R.RetargetError, match="no verified driver"):
        R.bone_driver("neck", s)


def test_mapping_rows_cover_every_bone_and_mesh_once(env, skel):
    rest = R.RestPose.from_frame(_reset_frame(env), skel)
    rows = R.mapping_rows(skel, rest)
    assert [r["bone"] for r in rows] == list(skel.bones)
    seen = [m for r in rows for m in r["meshes"]]
    assert sorted(seen) == sorted(skel.meshes)
    assert len(seen) == len(set(seen)) == 35
