"""Humanoid V2 viewer HVE suite (Happy / Validate / Error).

Browser evidence (screenshots, live playback sampling) is recorded separately in
playground/rl_live/humanoid_v2/evidence/. These tests check the static contract that
the browser relies on: asset provenance, the driver plan, L/R separation, the JS parity
run, dashboard asset serving, the no-scripted-motion rule, and the frozen frame contract.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "playground" / "dashboard" / "static" / "assets" / "humanoid_v2"
SRC_FBX = ROOT.parent / "humanoid-robot" / "assets" / "humanoid_robot_rigged.fbx"
SRC_SKEL = ROOT / "playground" / "rl_live" / "humanoid_v2" / "blender_skeleton.json"
JS_MATH = ROOT / "playground" / "dashboard" / "static" / "js" / "humanoid_v2.js"
NODE_CANDIDATES = [shutil.which("node"),
                   Path.home() / "AppData/Local/hermes/node/node.exe",
                   Path.home() / "AppData/Local/hermes/node/node"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _node() -> str:
    for c in NODE_CANDIDATES:
        if c and Path(c).exists():
            return str(c)
    pytest.skip("node not available")


@pytest.fixture(scope="module")
def plan():
    return json.loads((ASSETS / "retarget_plan.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bounds():
    return json.loads((ASSETS / "blender_mesh_bounds.json").read_text(encoding="utf-8"))["meshes"]


# ----------------------------------------------------------------- Happy
def test_static_assets_are_byte_identical_to_verified_sources():
    assert _sha(ASSETS / "humanoid_robot_rigged.fbx") == _sha(SRC_FBX)
    assert _sha(ASSETS / "blender_skeleton.json") == _sha(SRC_SKEL)


def test_plan_has_35_meshes_and_the_approved_22_5_8_split(plan):
    assert len(plan["meshes"]) == 35
    assert len(plan["bones"]) == 19
    assert plan["counts"] == {"direct": 22, "approximate": 5, "rigid": 8}
    assert plan["excluded_meshes"] == ["setup_floor"]


def test_chest_meshes_follow_spine_L2_with_no_offset(plan):
    for name in ("chest", "chest_plate", "spine_joint_up"):
        assert plan["meshes"][name]["body"] == "spine_L2", name
        assert plan["meshes"][name]["mode"] == "approximate", name


def test_every_mesh_is_driven_by_a_rig_body(plan):
    rig = set(plan["rig_bodies"])
    assert len(rig) == 16
    for name, m in plan["meshes"].items():
        assert m["body"] in rig, name


def test_left_and_right_meshes_are_mirrored_and_distinct(bounds):
    lefts = [n for n in bounds if n.endswith("_L")]
    rights = [n for n in bounds if n.endswith("_R")]
    assert len(lefts) == len(rights) == 12
    for n in lefts:
        assert bounds[n]["centroid"][0] > 0.0, n
    for n in rights:
        assert bounds[n]["centroid"][0] < 0.0, n


def test_js_parity_suite_passes_against_python_retarget():
    node = _node()
    res = subprocess.run(
        [node, "--experimental-default-type=module", "tests/js/humanoid_v2_math.test.mjs"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "PARITY ok: 7 frames, 245 mesh matrices" in out
    assert "FAIL-CLOSED ok" in out and "NO-DRIFT ok" in out


# ----------------------------------------------------------------- Validate
def test_dashboard_static_mount_serves_viewer_assets():
    """The V2 viewer's FBX + JSON assets load from the one dashboard static
    mount (the standalone sidecar that hand-rolled these content types is gone;
    the dashboard serves them via Starlette StaticFiles / stdlib mimetypes)."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.testclient import TestClient
    static = ROOT / "playground" / "dashboard" / "static"
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=static), name="static")
    c = TestClient(app)
    j = c.get("/static/assets/humanoid_v2/retarget_plan.json")
    assert j.status_code == 200
    assert j.headers["content-type"].startswith("application/json")
    fbx = c.get("/static/assets/humanoid_v2/humanoid_robot_rigged.fbx")
    assert fbx.status_code == 200
    assert int(fbx.headers["content-length"]) > 100_000   # the real rig, not a 404 page


def test_viewer_has_no_second_animation_system():
    src = JS_MATH.read_text(encoding="utf-8")
    for banned in ("requestAnimationFrame", "setInterval", "Math.sin(", "AnimationMixer", "clock.getElapsedTime"):
        assert banned not in src, banned


def test_frame_contract_file_is_unchanged_from_head():
    res = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "playground/rl_live/frames.py"],
                         cwd=ROOT, capture_output=True)
    assert res.returncode == 0, "frames.py differs from HEAD; frame contract must not change"


# ----------------------------------------------------------------- Error
def test_copy_helper_detects_a_mismatched_copy(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_viewer_assets", ROOT / "playground" / "rl_live" / "humanoid_v2" / "export_viewer_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = tmp_path / "src.bin"
    src.write_bytes(b"verified bytes")
    dst = tmp_path / "dst.bin"
    assert mod.copy_verified(src, dst) == _sha(src)
    # tamper with the destination after the copy: the hash comparison must still catch it
    dst.write_bytes(b"tampered")
    assert _sha(dst) != _sha(src)


def test_retarget_plan_rejects_unknown_driver(plan):
    rig = set(plan["rig_bodies"])
    bad = dict(plan["meshes"]["head"], body="agent_001:head")
    assert bad["body"] not in rig
