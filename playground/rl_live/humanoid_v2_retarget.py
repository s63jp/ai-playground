"""Humanoid V2 visual retargeting, Phase 1 (pure Python + numpy).

Consumes the existing rl_live frame body map {name: {"p": [x,y,z], "q": [w,x,y,z]}}
for exactly the 16 rig bodies and produces Blender-space 4x4 transforms for the
19 armature bones and the 35 skinned V2 meshes.

Guarantees
* Pure function of (frame, rest). No per-frame state, so repeated frames cannot accumulate error.
* Writes nothing to simulation state. Imports no mujoco, no browser or Three.js code.
* Fails closed: missing, extra or prefixed body names raise RetargetError.

Equations (4x4 homogeneous; both frames Z-up, +X left, +Y forward; identity conversion)

  Reset capture, once per episode:
    T0[b]           = pose(p0_b, q0_b)                       body world pose at reset
    H[bone]         = (armature_world @ [head_local, 1])[:3] bone head, Blender world
    t*              = mean over direct+approximate bones of (p0_drv - H)
    A               = translate(t*)                          one rigid rest alignment
    residual[bone]  = p0_drv - (H + t*)                      pivot offset, reported

  Per frame:
    D[b](t)         = pose(p_b(t), q_b(t)) @ inv(T0[b])      body delta, applied once
    bone_world      = D[drv(bone)](t) @ A @ rest_world(bone)
    mesh_world      = D[drv(mesh)](t) @ A @ mesh_matrix_world

  At reset D = I, so bone_world = A @ rest_world: the Blender rest pose placed on the
  sim figure. Rigid-follow bones and meshes take the delta of their verified driver.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SKELETON_JSON = Path(__file__).resolve().parent / "humanoid_v2" / "blender_skeleton.json"

RIG_BODIES = (
    "agent", "torso", "spine_L2",
    "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
    "head",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
)

# Verified classification (see humanoid_v2/blender_skeleton.json and the Phase 1 report).
# direct: Blender bone head lies at the MuJoCo joint pivot of the named body (offset reported).
DIRECT_BONE_BODY = {
    "head": "head",
    "upperarm.L": "left_shoulder", "forearm.L": "left_elbow",
    "thigh.L": "left_hip", "shin.L": "left_knee", "foot.L": "left_foot",
    "upperarm.R": "right_shoulder", "forearm.R": "right_elbow",
    "thigh.R": "right_hip", "shin.R": "right_knee", "foot.R": "right_foot",
}
# approximate: the MuJoCo body is the nearest available frame, not the same anatomical joint.
APPROX_BONE_BODY = {"spine": "torso", "chest": "spine_L2"}
# rigid-follow roots: bones with no Blender parent that has a driver.
RIGID_ROOT_BODY = {"pelvis": "torso"}

EXPECTED_BONES = 19
EXPECTED_MESHES = 35
EXPECTED_MODE_MESH_COUNTS = {"direct": 22, "approximate": 5, "rigid": 8}
NON_RIG_MESHES = ("setup_floor",)
DRIVEN_MODES = ("direct", "approximate")


class RetargetError(ValueError):
    """Raised when inputs do not match the verified contract. Never silently repaired."""


def quat_to_mat3(q) -> np.ndarray:
    """MuJoCo convention: q = [w, x, y, z]; normalised before conversion."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise RetargetError(f"quaternion must be 4 finite values, got {q!r}")
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise RetargetError("quaternion has zero norm")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(p, q) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise RetargetError(f"position must be 3 finite values, got {p!r}")
    T = np.eye(4)
    T[:3, :3] = quat_to_mat3(q)
    T[:3, 3] = p
    return T


def inv_pose(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rotation_angle(M: np.ndarray) -> float:
    """Magnitude of the rotation part of a 4x4 transform, in radians."""
    c = (float(np.trace(np.asarray(M, dtype=float)[:3, :3])) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def validate_frame(frame) -> None:
    """Fail closed unless the frame holds exactly the 16 rig bodies with valid poses."""
    if not isinstance(frame, Mapping):
        raise RetargetError("frame must be a mapping of body name to pose")
    got, want = set(frame), set(RIG_BODIES)
    if got != want:
        raise RetargetError(
            f"frame body set mismatch: missing={sorted(want - got)} "
            f"unexpected={sorted(got - want)}")
    for name in RIG_BODIES:
        entry = frame[name]
        if not isinstance(entry, Mapping) or "p" not in entry or "q" not in entry:
            raise RetargetError(f"{name}: pose needs 'p' and 'q'")
        try:
            pose_matrix(entry["p"], entry["q"])
        except RetargetError as exc:
            raise RetargetError(f"{name}: {exc}") from None


@dataclass(frozen=True)
class Skeleton:
    armature: str
    armature_world: np.ndarray
    bones: dict          # name -> {"parent", "head_local", "rest_world" (4x4), "head_world" (3,)}
    meshes: dict         # name -> {"bone", "matrix_world" (4x4)}
    excluded_meshes: tuple

    def bone_parent(self, bone: str):
        return self.bones[bone]["parent"]


def skeleton_from_dict(data: dict) -> Skeleton:
    arm_world = np.asarray(data["armature_matrix_world"], dtype=float)
    if arm_world.shape != (4, 4):
        raise RetargetError("armature_matrix_world must be 4x4")
    bones = {}
    for b in data["bones"]:
        local = np.asarray(b["matrix_local"], dtype=float)
        head = np.asarray(b["head_local"], dtype=float)
        if not np.allclose(local[:3, 3], head, atol=1e-5):
            raise RetargetError(f"bone {b['name']}: matrix translation != head_local")
        bones[b["name"]] = {
            "parent": b["parent"],
            "head_local": head,
            "rest_world": arm_world @ local,
            "head_world": (arm_world @ np.append(head, 1.0))[:3],
        }
    if len(bones) != EXPECTED_BONES:
        raise RetargetError(f"expected {EXPECTED_BONES} bones, found {len(bones)}")
    for name, b in bones.items():
        if b["parent"] is not None and b["parent"] not in bones:
            raise RetargetError(f"bone {name}: parent {b['parent']} not in skeleton")

    meshes, excluded = {}, []
    for m in data["meshes"]:
        if not m["armature_modifiers"]:
            excluded.append(m["name"])
            continue
        if m["name"] in meshes:
            raise RetargetError(f"duplicate mesh {m['name']}")
        counts = m["dominant_bone_counts"]
        if m["unweighted_verts"] or len(counts) != 1 or sum(counts.values()) != m["n_verts"]:
            raise RetargetError(f"mesh {m['name']} is not rigidly bound to exactly one bone: {counts}")
        (bone,) = counts
        if bone not in bones:
            raise RetargetError(f"mesh {m['name']} bound to unknown bone {bone}")
        if m["verts_with_secondary_weight_gt_0.01"]:
            raise RetargetError(f"mesh {m['name']} has secondary weights; rigid binding does not hold")
        meshes[m["name"]] = {"bone": bone,
                             "matrix_world": np.asarray(m["matrix_world"], dtype=float)}
    if len(meshes) != EXPECTED_MESHES:
        raise RetargetError(f"expected {EXPECTED_MESHES} skinned meshes, found {len(meshes)}")
    if tuple(sorted(excluded)) != tuple(sorted(NON_RIG_MESHES)):
        raise RetargetError(f"unexpected non-rig meshes: {sorted(excluded)}")
    return Skeleton(armature=data["armature"], armature_world=arm_world, bones=bones,
                    meshes=meshes, excluded_meshes=tuple(excluded))


def load_skeleton(path=SKELETON_JSON) -> Skeleton:
    return skeleton_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def bone_driver(bone: str, skel: Skeleton) -> tuple[str, str]:
    """Return (mode, driving MuJoCo body) for a bone. Rigid bones inherit their parent's driver."""
    if bone in DIRECT_BONE_BODY:
        return "direct", DIRECT_BONE_BODY[bone]
    if bone in APPROX_BONE_BODY:
        return "approximate", APPROX_BONE_BODY[bone]
    if bone in RIGID_ROOT_BODY:
        return "rigid", RIGID_ROOT_BODY[bone]
    parent = skel.bone_parent(bone)
    if parent is None:
        raise RetargetError(f"bone {bone} is a root with no verified driver")
    _, body = bone_driver(parent, skel)
    return "rigid", body


@dataclass(frozen=True)
class RestPose:
    skeleton: Skeleton
    body_T0: dict        # body -> 4x4 world pose at reset
    A: np.ndarray        # 4x4 rest alignment, Blender rest -> sim world
    drivers: dict        # bone -> (mode, body)
    residuals: dict = field(default_factory=dict)   # bone -> pivot offset vector (m)

    @classmethod
    def from_frame(cls, frame, skeleton: Skeleton) -> "RestPose":
        validate_frame(frame)
        T0 = {n: pose_matrix(frame[n]["p"], frame[n]["q"]) for n in RIG_BODIES}
        drivers = {b: bone_driver(b, skeleton) for b in skeleton.bones}
        pairs = []
        for bone, (mode, body) in drivers.items():
            if mode in DRIVEN_MODES:
                pairs.append((bone, body, skeleton.bones[bone]["head_world"]))
        if not pairs:
            raise RetargetError("no driven bones available for rest alignment")
        t = np.mean([T0[body][:3, 3] - H for _, body, H in pairs], axis=0)
        A = np.eye(4)
        A[:3, 3] = t
        residuals = {bone: T0[body][:3, 3] - (H + t) for bone, body, H in pairs}
        return cls(skeleton=skeleton, body_T0=T0, A=A, drivers=drivers, residuals=residuals)


def retarget(frame, rest: RestPose) -> dict:
    """Blender-space transforms for one frame. Pure: depends only on (frame, rest)."""
    validate_frame(frame)
    skel = rest.skeleton
    deltas = {n: pose_matrix(frame[n]["p"], frame[n]["q"]) @ inv_pose(rest.body_T0[n])
              for n in RIG_BODIES}
    bones = {}
    for bone in skel.bones:
        _, body = rest.drivers[bone]
        bones[bone] = deltas[body] @ rest.A @ skel.bones[bone]["rest_world"]
    meshes = {}
    for mesh, info in skel.meshes.items():
        _, body = rest.drivers[info["bone"]]
        meshes[mesh] = deltas[body] @ rest.A @ info["matrix_world"]
    return {"deltas": deltas, "bones": bones, "meshes": meshes}


def mapping_rows(skel: Skeleton, rest: RestPose) -> list[dict]:
    """Report rows: one per bone with its driver, residual and the meshes it drives."""
    rows = []
    for bone in skel.bones:
        mode, body = rest.drivers[bone]
        meshes = sorted(m for m, info in skel.meshes.items() if info["bone"] == bone)
        res = rest.residuals.get(bone)
        rows.append({"bone": bone, "mode": mode, "body": body,
                     "residual_m": None if res is None else round(float(np.linalg.norm(res)), 4),
                     "meshes": meshes})
    return rows
