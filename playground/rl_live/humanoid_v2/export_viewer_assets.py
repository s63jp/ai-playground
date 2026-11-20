"""Build the browser assets for the Humanoid V2 viewer (re-runnable, deterministic).

Writes, under playground/dashboard/static/assets/humanoid_v2/:
  humanoid_robot_rigged.fbx   copy of the verified Blender export (sha256 checked)
  blender_skeleton.json       copy of the verified skeleton dump (sha256 checked)
  retarget_plan.json          bone/mesh driver table exported from humanoid_v2_retarget.py

And writes tests/fixtures/humanoid_v2_viewer_parity.json: real LocoEnv frames with the
expected Python retarget outputs, so the JS port can be checked for exact parity.
Physics, MJCF, checkpoints are only read.
"""
import hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # ai-playground
sys.path.insert(0, str(ROOT))
from playground.rl.env import LocoEnv                       # noqa: E402
from playground.rl_live import humanoid_v2_retarget as R    # noqa: E402
from playground.rl_live.frames import FrameIndex            # noqa: E402

SRC_FBX = ROOT.parent / "humanoid-robot" / "assets" / "humanoid_robot_rigged.fbx"
SRC_SKEL = R.SKELETON_JSON
ASSET_DIR = ROOT / "playground" / "dashboard" / "static" / "assets" / "humanoid_v2"
FIXTURE = ROOT / "tests" / "fixtures" / "humanoid_v2_viewer_parity.json"
PARITY_PERTURB = [
    ("reset", {}),
    ("waist_yaw +0.3", {"waist_yaw": 0.3}),
    ("spine_L2 -0.3", {"spine_L2": -0.3}),
    ("left_elbow +0.5", {"left_elbow": 0.5}),
    ("right_hip -0.3", {"right_hip": -0.3}),
    ("head_yaw +0.4", {"head_yaw": 0.4}),
    ("left_foot_yaw +0.2", {"left_foot_yaw": 0.2}),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_verified(src: Path, dst: Path) -> str:
    shutil.copyfile(src, dst)
    a, b = sha256(src), sha256(dst)
    if a != b:
        raise RuntimeError(f"copy mismatch {dst}: {a} != {b}")
    return a


def rows(M) -> list:
    return np.asarray(M, dtype=float).reshape(4, 4).tolist()


def build_plan(skel: R.Skeleton) -> dict:
    bones = {}
    for name, b in skel.bones.items():
        mode, body = R.bone_driver(name, skel)
        bones[name] = {"parent": b["parent"], "mode": mode, "body": body,
                       "head_world": b["head_world"].tolist(),
                       "rest_world": rows(b["rest_world"])}
    meshes = {}
    for name, info in skel.meshes.items():
        mode, body = R.bone_driver(info["bone"], skel)
        meshes[name] = {"bone": info["bone"], "mode": mode, "body": body}
    return {"armature": skel.armature, "rig_bodies": list(R.RIG_BODIES),
            "excluded_meshes": list(skel.excluded_meshes),
            "counts": {m: sum(1 for v in meshes.values() if v["mode"] == m)
                       for m in ("direct", "approximate", "rigid")},
            "bones": bones, "meshes": meshes}


def build_fixture(skel: R.Skeleton) -> dict:
    scratch = Path(os.environ.get("SCRATCH", r"C:/Users/jamie/AppData/Local/hermes/cache/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="v2fix_", dir=str(scratch))
    env = LocoEnv(cfg={"_root": tmp, "agent_body": {"type": "humanoid"}})
    cases, rest = [], None
    for label, pert in PARITY_PERTURB:
        env.reset(seed=0)
        m, d = env.world.m, env.world.d
        for j, a in pert.items():
            d.qpos[int(m.joint(j).qposadr[0])] += a
        mujoco.mj_forward(m, d)
        idx = FrameIndex(env)
        idx.ensure()
        frame = idx.body_dict()
        if rest is None:
            rest = R.RestPose.from_frame(frame, skel)
        out = R.retarget(frame, rest)
        cases.append({
            "label": label, "perturb": pert, "bodies": frame,
            "expected": {
                "deltas": {n: rows(out["deltas"][n]) for n in R.RIG_BODIES},
                "object_matrix": {mesh: rows(out["deltas"][rest.drivers[skel.meshes[mesh]["bone"]][1]] @ rest.A)
                                  for mesh in skel.meshes},
                "bone_world": {b: rows(out["bones"][b]) for b in skel.bones},
            },
        })
    return {"A": rows(rest.A), "cases": cases}


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    fbx_sha = copy_verified(SRC_FBX, ASSET_DIR / "humanoid_robot_rigged.fbx")
    skel_sha = copy_verified(SRC_SKEL, ASSET_DIR / "blender_skeleton.json")
    skel = R.load_skeleton(SRC_SKEL)
    plan = build_plan(skel)
    (ASSET_DIR / "retarget_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    fixture = build_fixture(skel)
    fixture["provenance"] = {"fbx_sha256": fbx_sha, "skeleton_sha256": skel_sha}
    FIXTURE.write_text(json.dumps(fixture), encoding="utf-8")
    # browser copy: the same real-LocoEnv perturbation frames, served to the viewer check page
    (ASSET_DIR / "viewer_parity_fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
    print("FBX_SHA256", fbx_sha)
    print("SKELETON_SHA256", skel_sha)
    print("PLAN bones", len(plan["bones"]), "meshes", len(plan["meshes"]), "counts", plan["counts"])
    print("FIXTURE cases", len(fixture["cases"]), "->", FIXTURE)


if __name__ == "__main__":
    main()
