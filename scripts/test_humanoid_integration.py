"""Integration test for the humanoid player-model asset (standalone script).

Run: .venv/Scripts/python.exe scripts/test_humanoid_integration.py
The same checks live in the pytest suite as tests/test_humanoid_asset.py.
"""
import sys
from pathlib import Path

import mujoco

XML_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "humanoid_mujoco.xml")


def run() -> bool:
    print("=== Testing Humanoid Asset Integration ===\n")
    ok = True

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    print(f"✓ Model loaded: mujoco {mujoco.__version__}, bodies={model.nbody} "
          f"joints={model.njnt} actuators={model.nu}")

    for _ in range(250):
        mujoco.mj_step(model, data)
    assert not any(
        (abs(x) > 1e6 or x != x) for x in data.qpos.ravel()
    ), "simulation diverged"
    print("✓ 250 sim steps stable (damping holds the rest pose)")

    # Every actuator must resolve to a named joint
    resolvable = 0
    for a in range(model.nu):
        jid = model.actuator_trnid[a][0]
        if jid >= 0 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid):
            resolvable += 1
    print(f"✓ Actuators mapped to named joints: {resolvable}/{model.nu}")
    ok &= resolvable == model.nu

    # A motor must actually move its joint
    before = float(data.qpos[model.jnt_qposadr[model.joint("left_knee").id]])
    aid = model.actuator("left_knee").id
    data.ctrl[aid] = 12.0
    for _ in range(60):
        mujoco.mj_step(model, data)
    after = float(data.qpos[model.jnt_qposadr[model.joint("left_knee").id]])
    print(f"✓ Motor drives left_knee: qpos {before:.3f} -> {after:.3f} rad")
    ok &= abs(after - before) > 0.01

    # Neutral ctrl returns the joint back toward rest
    data.ctrl[aid] = 0.0
    for _ in range(120):
        mujoco.mj_step(model, data)
    settled = float(data.qpos[model.jnt_qposadr[model.joint("left_knee").id]])
    print(f"✓ Joint settles back toward neutral: {settled:.3f} rad")
    ok &= abs(settled) < abs(after)

    print(f"\n=== Integration Test Complete: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
