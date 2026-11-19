"""Offscreen render of the humanoid_v2 visual integration (evidence artifact).

Renders the SAME model the tests load (agent_body.visual = humanoid_v2) from a
few angles, plus a legacy-rig render for side-by-side comparison.
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

OUT = os.path.join(REPO, "assets", "humanoid_v2", "renders")
os.makedirs(OUT, exist_ok=True)
W, H = 720, 960


def model(visual):
    """Build the shipped MJCF, then enlarge ONLY the offscreen framebuffer.

    MuJoCo's default offscreen buffer is 640x480; rendering larger needs
    <visual><global offwidth/offheight>. That is a renderer concern, so it is
    injected here instead of polluting playground/world/spec.py.
    """
    ab = {"type": "humanoid"}
    if visual:
        ab["visual"] = visual
    cfg = load_config(overrides={"_root": tempfile.mkdtemp(prefix="hv2r_"),
                                 "agent_body": ab})
    xml = build_mjcf(cfg, build_specs(cfg))
    inject = '<visual><global offwidth="%d" offheight="%d"/></visual>' % (W, H)
    assert "<mujoco" in xml
    head = xml.index(">", xml.index("<mujoco")) + 1
    xml = xml[:head] + "\n  " + inject + xml[head:]
    return mujoco.MjModel.from_xml_string(xml)


def agent_xy(m, d):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if bid < 0:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    return np.asarray(d.xpos[bid])[:2]


def shot(m, path, azim, elev=-8.0, dist=4.2, pose=None, label="", lookat_z=0.9):
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    if pose:
        for jn, val in pose.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    x, y = agent_xy(m, d)
    cam.lookat[:] = [x, y, lookat_z]
    cam.distance, cam.azimuth, cam.elevation = dist, azim, elev

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    with mujoco.Renderer(m, height=H, width=W) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        px = r.render()
    try:
        from PIL import Image
        Image.fromarray(px).save(path)
    except ImportError:
        import imageio.v2 as imageio
        imageio.imwrite(path, px)
    print("  wrote %-34s %s" % (os.path.basename(path), label))
    return path


def main():
    integrated = model("humanoid_v2")
    legacy = model(None)
    print("integrated: nmesh=%d ngeom=%d | legacy: ngeom=%d"
          % (integrated.nmesh, integrated.ngeom, legacy.ngeom))

    print("rendering humanoid_v2 ...")
    shot(integrated, os.path.join(OUT, "hv2_front.png"), 90, label="A-pose front")
    shot(integrated, os.path.join(OUT, "hv2_side.png"), 0, label="A-pose side")
    shot(integrated, os.path.join(OUT, "hv2_three_quarter.png"), 55,
         label="A-pose 3/4")
    # dist must clear the torso: at 1.1 m the camera sits INSIDE the chest and
    # renders clipped back-faces. 2.2 m with the lookat raised to the neck frames
    # head + neck rings properly.
    shot(integrated, os.path.join(OUT, "hv2_head_detail.png"), 60, elev=2,
         dist=2.2, lookat_z=1.52, label="head/neck detail")
    shot(integrated, os.path.join(OUT, "hv2_articulated.png"), 55,
         label="articulated (head+left limbs)",
         pose={"head_yaw": 0.5, "head_pitch": 0.25, "left_shoulder": -0.6,
               "left_elbow": 0.9, "left_hip": 0.5, "left_knee": -0.8})
    print("rendering legacy rig for comparison ...")
    shot(legacy, os.path.join(OUT, "legacy_front.png"), 90, label="legacy capsules")
    print("RENDER_OK ->", OUT)


if __name__ == "__main__":
    main()
