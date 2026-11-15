"""HVE coverage for the humanoid player-model asset (assets/humanoid_mujoco.xml).

The asset arrived broken (unnamed joints, actuators referencing non-existent 'torso/...' targets, a
ball joint driven by three motors, limbs laid out along the wrong axis, an absolute meshdir). It was
corrected in Nov 2026 and must stay verifiable: it loads, simulates stably, every actuator maps to a
named joint, and motors actually drive their joints.
"""
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / "assets" / "humanoid_mujoco.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(XML))


class TestHumanoidAsset:
    def test_xml_file_exists(self):
        assert XML.exists()

    def test_xml_is_portable(self):
        """No absolute paths baked into the compiler line."""
        text = XML.read_text(encoding="utf-8")
        assert "meshdir" not in text, "remove the absolute meshdir; the asset uses no mesh files"
        assert "C:/" not in text

    def test_loads_with_expected_structure(self, model):
        assert model.nbody == 14          # world + torso + spine + 2x(shoulder,elbow) + head + 2x(hip,knee,ankle)
        assert model.njnt == 14
        assert model.nu == 12             # spine, 2x arm, 2x leg, 3 head
        for name in ("spine_L2", "left_shoulder", "right_elbow", "left_hip", "right_knee", "head_yaw"):
            assert model.joint(name).id >= 0, f"joint '{name}' missing"

    def test_every_actuator_targets_a_named_joint(self, model):
        for a in range(model.nu):
            jid = int(model.actuator_trnid[a][0])
            assert jid >= 0
            assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid), f"actuator {a} has no joint"

    def test_simulation_is_stable(self, model):
        data = mujoco.MjData(model)
        for _ in range(250):
            mujoco.mj_step(model, data)
        qpos = data.qpos.ravel()
        assert all(abs(x) < 1e6 and x == x for x in qpos), "simulation diverged"

    def test_motor_drives_its_joint(self, model):
        data = mujoco.MjData(model)
        knee_joint = model.joint("left_knee").id
        before = float(data.qpos[model.jnt_qposadr[knee_joint]])
        data.ctrl[model.actuator("left_knee").id] = 12.0     # Nm, inside ctrlrange
        for _ in range(60):
            mujoco.mj_step(model, data)
        after = float(data.qpos[model.jnt_qposadr[knee_joint]])
        assert abs(after - before) > 0.01, "motor produced no joint motion"

    def test_head_has_three_drivable_hinges(self, model):
        data = mujoco.MjData(model)
        for name, ctrl in (("head_pitch", 1.0), ("head_yaw", 1.0), ("head_roll", 1.0)):
            adr = model.jnt_qposadr[model.joint(name).id]
            before = float(data.qpos[adr])
            data.ctrl[model.actuator(name).id] = ctrl
            for _ in range(40):
                mujoco.mj_step(model, data)
            assert abs(float(data.qpos[adr]) - before) > 1e-3, f"{name} did not move"
            data.ctrl[model.actuator(name).id] = 0.0
