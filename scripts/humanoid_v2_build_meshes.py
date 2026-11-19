"""Retarget the Blender humanoid onto AI Playground's existing MuJoCo skeleton
as VISUAL geometry (integration stage 1: visuals only, physics untouched).

Contract:
  * Existing MuJoCo bodies/joints stay authoritative. Each Blender part maps to
    ONE existing body and is baked into that body's LOCAL frame at qpos0.
  * Both figures are Z-up and face +Y => no global rotation.
  * Segment lengths differ (Blender thigh 0.43 m vs skeleton 0.38 m), so limb
    parts are fitted BETWEEN the skeleton's real joint frames: rotated onto the
    skeleton's bone axis and scaled ALONG that axis only (cross-section kept),
    so each part's distal end lands exactly on the next joint - no gaps when
    the joints articulate.
  * Axial parts (head/neck/chest/abdomen/pelvis) keep Blender proportions
    (identity), preserving the ~1.755 m figure height the spec requires.
  * Right-side parts use M_R = Mirror . M_L . Mirror (det = +1), so L/R
    symmetry is exact AND face winding stays correct.

Outputs: assets/meshes/humanoid_v2/<part>.obj (35 files) + manifest.json
(part -> body, matrices, landing assertions, source SHA256).

Run:  .venv/Scripts/python.exe scripts/humanoid_v2_build_meshes.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)

SOURCE_OBJ = os.path.join(REPO, "assets", "humanoid_v2", "source",
                          "humanoid_robot_rigged.obj")
OUT_DIR = os.path.join(REPO, "assets", "meshes", "humanoid_v2")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")

MATERIAL_RGBA = {  # sRGB values documented in the Blender asset docs
    "RBT_Shell_BlueGrey": [0.42, 0.50, 0.58, 1.0],
    "RBT_Joint_Gunmetal": [0.30, 0.32, 0.35, 1.0],
    "RBT_Accent_Light": [0.60, 0.66, 0.71, 1.0],
    "RBT_Visor_Dark": [0.11, 0.12, 0.14, 1.0],
}

# axial parts keep Blender's own placement (identity in world frame)
AXIAL_MAP = {
    "pelvis": "torso", "abdomen": "torso", "spine_joint_low": "torso",
    "spine_joint_up": "spine_L2", "chest": "spine_L2", "chest_plate": "spine_L2",
    "neck": "head", "neck_ring_1": "head", "neck_ring_2": "head",
    "head": "head", "faceplate": "head",
}
# limb parts: part -> (target body, role). Left side drives; right mirrors it.
LIMB_MAP = {
    "shoulder_L": ("left_shoulder", "shoulder"),
    "upperarm_L": ("left_shoulder", "upperarm"),
    "elbow_L": ("left_elbow", "elbow"),
    "forearm_L": ("left_elbow", "forearm"),
    "wrist_L": ("left_elbow", "wrist"),      # skeleton has no wrist DOF
    "hand_L": ("left_elbow", "hand"),        # hand rides the forearm body
    "hip_L": ("left_hip", "hip"),
    "thigh_L": ("left_hip", "thigh"),
    "knee_L": ("left_knee", "knee"),
    "shin_L": ("left_knee", "shin"),
    "ankle_L": ("left_ankle", "ankle"),
    "foot_L": ("left_foot", "foot"),
}
M4 = np.diag([-1.0, 1.0, 1.0, 1.0])          # mirror across the YZ plane


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_obj_parts(path: str):
    """OBJ with `o <name>` markers -> {part: verts}, {part: faces}, {part: mtl}."""
    verts: list[list[float]] = []
    faces: dict[str, list[list[int]]] = {}
    mat_of: dict[str, str] = {}
    cur = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("o "):
                cur = line[2:].strip()
                faces[cur] = []
            elif line.startswith("usemtl ") and cur:
                mat_of[cur] = line.split(None, 1)[1].strip()
            elif line.startswith("f ") and cur:
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                faces[cur].append([(i - 1) if i > 0 else len(verts) + i for i in idx])
    V = np.asarray(verts, dtype=np.float64)
    out_v, out_f = {}, {}
    for name, fl in faces.items():
        used = sorted({i for f in fl for i in f})
        remap = {g: k for k, g in enumerate(used)}
        out_v[name] = V[used]
        out_f[name] = [[remap[i] for i in f] for f in fl]
    return out_v, out_f, mat_of


def rot_align(d_from: np.ndarray, d_to: np.ndarray) -> np.ndarray:
    """3x3 rotation taking d_from onto d_to (Rodrigues)."""
    a = d_from / np.linalg.norm(d_from)
    b = d_to / np.linalg.norm(d_to)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def scale_along(d: np.ndarray, s: float) -> np.ndarray:
    """Scale s along unit direction d, 1.0 elsewhere (cross-section preserved)."""
    u = d / np.linalg.norm(d)
    return np.eye(3) + (s - 1.0) * np.outer(u, u)


def compose(anchor_bl, anchor_ex, R, S=None) -> np.ndarray:
    """4x4 for p' = anchor_ex + R . S . (p - anchor_bl)."""
    M = np.eye(4)
    M[:3, :3] = R @ (S if S is not None else np.eye(3))
    M[:3, 3] = anchor_ex - M[:3, :3] @ anchor_bl
    return M


def skeleton():
    """Compile the real playground world (humanoid) and read frames at qpos0."""
    import mujoco
    from playground.config import load_config
    from playground.world.spec import build_mjcf, build_specs
    from playground.world.physics import PhysicsWorld

    tmp = tempfile.mkdtemp(prefix="hv2_")
    cfg = load_config(overrides={"_root": tmp, "agent_body": {"type": "humanoid"}})
    w = PhysicsWorld(cfg)
    w.d.qpos[:] = w.m.qpos0
    mujoco.mj_forward(w.m, w.d)

    def body(name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"body {name} missing from generated world"
        mat = np.array(w.d.xmat[bid], dtype=np.float64).reshape(3, 3)
        assert np.allclose(mat, np.eye(3), atol=1e-9), \
            f"body {name} not axis-aligned at qpos0; baking would be wrong"
        return np.array(w.d.xpos[bid], dtype=np.float64)

    def geom(name: str):
        gid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"geom {name} missing from generated world"
        return (np.array(w.d.geom_xpos[gid], dtype=np.float64),
                np.array(w.m.geom_size[gid], dtype=np.float64),
                np.array(w.d.geom_xmat[gid], dtype=np.float64).reshape(3, 3))

    return {"cfg": cfg, "body": body, "geom": geom}


def bbox_center(V: np.ndarray) -> np.ndarray:
    return (V.min(axis=0) + V.max(axis=0)) / 2.0


def write_obj(path: str, name: str, V: np.ndarray, faces, header) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# humanoid_v2 retargeted visual mesh\n")
        for h in header:
            fh.write(f"# {h}\n")
        fh.write(f"o {name}\n")
        for p in V:
            fh.write("v %.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
        for f in faces:                       # fan-triangulate (MuJoCo-safe)
            for k in range(1, len(f) - 1):
                fh.write("f %d %d %d\n" % (f[0] + 1, f[k] + 1, f[k + 1] + 1))


def main() -> int:
    if not os.path.exists(SOURCE_OBJ):
        raise SystemExit(f"source OBJ missing: {SOURCE_OBJ}\n"
                         "copy the Blender export into assets/humanoid_v2/source/")
    os.makedirs(OUT_DIR, exist_ok=True)
    V, F, MAT = read_obj_parts(SOURCE_OBJ)
    expect = set(AXIAL_MAP) | set(LIMB_MAP) | {p.replace("_L", "_R") for p in LIMB_MAP}
    if expect - set(V):
        raise SystemExit(f"missing parts in source OBJ: {sorted(expect - set(V))}")
    print(f"source: {len(V)} parts, {sum(len(v) for v in V.values())} verts, "
          f"sha256={sha256(SOURCE_OBJ)[:16]}...")

    # ---------------- skeleton landmarks at qpos0 ----------------
    sk = skeleton()
    B, G = sk["body"], sk["geom"]
    SHX, ELX = B("left_shoulder"), B("left_elbow")
    HIPX, KNX, ANX = B("left_hip"), B("left_knee"), B("left_ankle")
    fpos, fsize, fmat = G("g:left_forearm")           # wrist anchor = distal axis end
    axis = fmat[:, 2]
    e1, e2 = fpos + axis * fsize[1], fpos - axis * fsize[1]
    WRX = e1 if e1[2] < e2[2] else e2                # whichever end hangs lower
    assert WRX[2] < fpos[2], f"forearm capsule axis {axis} gives no distal end"
    bpos, bsize, _ = G("g:left_foot")                 # sole = collision box bottom
    SOLE = float(bpos[2] - bsize[2])
    BODY_X = {n: B(n) for n in
              ("torso", "spine_L2", "head",
               "left_shoulder", "left_elbow", "left_hip", "left_knee",
               "left_ankle", "left_foot",
               "right_shoulder", "right_elbow", "right_hip", "right_knee",
               "right_ankle", "right_foot")}
    print(f"skeleton: shoulder={SHX.round(4)} elbow={ELX.round(4)} "
          f"wrist={WRX.round(4)} hip={HIPX.round(4)} knee={KNX.round(4)} "
          f"ankle={ANX.round(4)} sole={SOLE:.4f}")

    # ---------------- Blender anchors (read from the mesh itself) ----------------
    cSH = bbox_center(V["shoulder_L"])
    cEL = bbox_center(V["elbow_L"])
    cWR = bbox_center(V["wrist_L"])
    cHIP = bbox_center(V["hip_L"])
    cKN = bbox_center(V["knee_L"])
    cAN = bbox_center(V["ankle_L"])
    thigh_top = np.array([V["thigh_L"][:, 0].mean(), V["thigh_L"][:, 1].mean(),
                          V["thigh_L"][:, 2].max()])
    shin_top = np.array([V["shin_L"][:, 0].mean(), V["shin_L"][:, 1].mean(),
                         V["shin_L"][:, 2].max()])
    seg = {}
    for nm, bl, ex in (("upperarm", cEL - cSH, ELX - SHX),
                       ("forearm", cWR - cEL, WRX - ELX),
                       ("thigh", cKN - thigh_top, KNX - HIPX),
                       ("shin", cAN - shin_top, ANX - KNX)):
        s = float(np.linalg.norm(ex) / np.linalg.norm(bl))
        seg[nm] = {"scale": round(s, 6),
                   "bl_len": round(float(np.linalg.norm(bl)), 6),
                   "ex_len": round(float(np.linalg.norm(ex)), 6),
                   "R": rot_align(bl, ex)}
        print(f"segment {nm:9s}: blender {seg[nm]['bl_len']:.4f} m -> "
              f"skeleton {seg[nm]['ex_len']:.4f} m  (axial scale {s:.4f})")

    # ---------------- per-part transforms ----------------
    arm_R = seg["upperarm"]["R"]
    limb_M = {
        "shoulder": (cSH, SHX, arm_R, None),
        "upperarm": (cSH, SHX, arm_R,
                     scale_along(cEL - cSH, seg["upperarm"]["scale"])),
        "elbow": (cEL, ELX, arm_R, None),
        "forearm": (cEL, ELX, arm_R,
                    scale_along(cWR - cEL, seg["forearm"]["scale"])),
        "wrist": (cWR, WRX, arm_R, None),
        "hand": (cWR, WRX, arm_R, None),
        "hip": (cHIP, HIPX, np.eye(3), None),
        "thigh": (thigh_top, HIPX, seg["thigh"]["R"],
                  scale_along(cKN - thigh_top, seg["thigh"]["scale"])),
        "knee": (cKN, KNX, np.eye(3), None),
        "shin": (shin_top, KNX, seg["shin"]["R"],
                 scale_along(cAN - shin_top, seg["shin"]["scale"])),
        "ankle": (cAN, ANX, np.eye(3), None),
    }
    cF = bbox_center(V["foot_L"])
    foot_t = np.array([BODY_X["left_foot"][0] - cF[0], bpos[1] - cF[1],
                       SOLE - float(V["foot_L"][:, 2].min())])
    print(f"foot translate: {foot_t.round(4)} (sole target {SOLE:.4f})")

    mats: dict[str, np.ndarray] = {}
    for part in AXIAL_MAP:
        mats[part] = np.eye(4)
    for part, (body, role) in LIMB_MAP.items():
        if role == "foot":
            m = np.eye(4)
            m[:3, 3] = foot_t
            mats[part] = m
        else:
            abl, aex, R, S = limb_M[role]
            mats[part] = compose(abl, aex, R, S)
    for part in LIMB_MAP:                                  # exact mirror, det=+1
        mats[part.replace("_L", "_R")] = M4 @ mats[part] @ M4

    part_body = dict(AXIAL_MAP)
    for part, (body, role) in LIMB_MAP.items():
        part_body[part] = body
        part_body[part.replace("_L", "_R")] = body.replace("left_", "right_")

    # ---------------- bake into body-local frames ----------------
    out_v = {}
    for part in sorted(V):
        M = mats[part]
        world = (M[:3, :3] @ V[part].T).T + M[:3, 3]
        out_v[part] = world - BODY_X[part_body[part]]     # body orientation = I

    def world_of(part: str) -> np.ndarray:
        return out_v[part] + BODY_X[part_body[part]]

    def distal(part: str) -> np.ndarray:
        """Centre of the segment's DISTAL cap face (its most downward-facing
        normal). Vertex argmin breaks on tilted cap rings: the ring spans
        z +/- r*sin(tilt), so 'lowest vertex' sits a full radius off-axis."""
        P = world_of(part)
        best, best_s = None, 1e9
        for f in F[part]:
            pts = P[f]
            n = np.zeros(3)                                # Newell normal
            for i in range(len(pts)):
                n += np.cross(pts[i], pts[(i + 1) % len(pts)])
            ln = float(np.linalg.norm(n))
            if ln < 1e-12:
                continue
            s = n[2] / ln
            if s < best_s:
                best_s, best = s, f
        assert best is not None and best_s <= -0.5, \
            f"{part}: no downward-facing distal cap (got {best_s})"
        return P[best].mean(axis=0)

    checks = {
        "upperarm_distal_to_elbow": float(np.linalg.norm(distal("upperarm_L") - ELX)),
        "forearm_distal_to_wrist": float(np.linalg.norm(distal("forearm_L") - WRX)),
        "thigh_distal_to_knee": float(np.linalg.norm(distal("thigh_L") - KNX)),
        "shin_distal_to_ankle": float(np.linalg.norm(distal("shin_L") - ANX)),
        "foot_sole_z": float(world_of("foot_L")[:, 2].min()),
        "head_top_z": float(world_of("head")[:, 2].max()),
    }
    failures = []
    for k, v in checks.items():
        print(f"check {k:26s} = {v:.6f}")
    for k in ("upperarm_distal_to_elbow", "forearm_distal_to_wrist",
              "thigh_distal_to_knee", "shin_distal_to_ankle"):
        if checks[k] > 1e-6:
            failures.append(f"{k} off by {checks[k]:.6f}")
    if abs(checks["foot_sole_z"]) > 1e-6:
        failures.append(f"foot sole at {checks['foot_sole_z']}")

    # L/R exact mirror
    worst = 0.0
    for part in sorted(p for p in V if p.endswith("_L")):
        a = out_v[part].copy()
        a[:, 0] *= -1.0
        b = out_v[part.replace("_L", "_R")]
        if a.shape != b.shape:
            failures.append(f"{part}: vertex count differs from right side")
            continue
        a = a[np.lexsort((a[:, 2], a[:, 1], np.round(a[:, 0], 9)))]
        b = b[np.lexsort((b[:, 2], b[:, 1], np.round(b[:, 0], 9)))]
        worst = max(worst, float(np.abs(a - b).max()))
    print(f"L/R mirror max deviation: {worst:.3e}")
    if worst > 1e-9:
        failures.append(f"L/R asymmetry {worst}")

    # rest-pose connectivity: no floating parts
    names = sorted(V)
    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    boxes = {n: (out_v[n].min(axis=0), out_v[n].max(axis=0)) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.all(boxes[a][0] <= boxes[b][1] + 1e-6) and \
                    np.all(boxes[b][0] <= boxes[a][1] + 1e-6):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    comps: dict[str, list[str]] = {}
    for n in names:
        comps.setdefault(find(n), []).append(n)
    print(f"rest-pose connected components: {len(comps)}")
    if len(comps) != 1:
        failures.append(f"{len(comps)} components: "
                        f"{[sorted(v) for v in comps.values()]}")

    allz = np.vstack([world_of(n) for n in names])
    height = float(allz[:, 2].max() - allz[:, 2].min())
    print(f"visual figure: z {allz[:, 2].min():.4f} .. {allz[:, 2].max():.4f} "
          f"-> height {height:.4f} m")
    if not (1.70 <= height <= 1.80):
        failures.append(f"height {height} out of range")

    # ---------------- write ----------------
    src_rel = os.path.relpath(SOURCE_OBJ, REPO).replace(os.sep, "/")
    for part in sorted(V):
        hdr = [f"body: {part_body[part]}", f"source: {src_rel}",
               "retarget: " + ("identity" if part in AXIAL_MAP
                               else "matrix in manifest")]
        write_obj(os.path.join(OUT_DIR, f"{part}.obj"), part, out_v[part],
                  F[part], hdr)

    manifest = {
        "generator": "scripts/humanoid_v2_build_meshes.py",
        "source": {"path": src_rel, "sha256": sha256(SOURCE_OBJ)},
        "frame": "qpos0, body-local, Z-up, +Y forward (matches Blender)",
        "parts": {p: {"body": part_body[p],
                      "matrix": [round(float(x), 9) for x in mats[p].ravel()],
                      "material": MAT.get(p, "RBT_Shell_BlueGrey"),
                      "rgba": MATERIAL_RGBA.get(MAT.get(p, "RBT_Shell_BlueGrey")),
                      "verts": int(len(out_v[p])),
                      "mode": "identity" if p in AXIAL_MAP else "retarget"}
                  for p in sorted(V)},
        "bodies": {b: sorted(p for p in V if part_body[p] == b)
                   for b in sorted(set(part_body.values()))},
        "segments": {k: {kk: vv for kk, vv in v.items() if kk != "R"}
                     for k, v in seg.items()},
        "checks": {**{k: round(v, 9) for k, v in checks.items()},
                   "lr_mirror_max_dev": worst,
                   "components": len(comps), "height_m": round(height, 6)},
    }
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {len(V)} meshes + {os.path.relpath(MANIFEST, REPO)}")

    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    print("MESH_BUILD_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
