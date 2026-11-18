"""Dashboard RL surface (M6-B B2): read-only contract/curriculum + bounded
probe runs. No trainer exists yet — the surface must not imply one."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from playground.dashboard.server import create_app


@pytest.fixture
def app(cfg, db):
    return create_app(cfg, db)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestRlContract:
    def test_contract_matches_env_layout(self, client):
        r = client.get("/api/rl/contract")
        assert r.status_code == 200
        data = r.json()
        # M7-0a: the dashboard serves the LIVE contract (v4; freeze section 2)
        assert data["contract_version"] == 4
        assert data["obs_dim"] == 68 and data["action_dim"] == 15
        obs = data["observation"]
        assert obs["layout_version"] == 4 and obs["dim"] == 68
        slices = {sec["name"]: sec["slice"] for sec in obs["sections"]}
        assert sum(sec["slice"][1] - sec["slice"][0] for sec in obs["sections"]) == 68
        # the v3 prefix sections keep their slices unchanged
        assert slices["joint_pos"] == [0, 14] and slices["command"] == [40, 43]
        # the command block appends 43..68 in the fixed order
        for name, sl in (("limb_pose_target", [43, 58]), ("com_shift", [58, 60]),
                         ("waypoint", [60, 63]), ("stop_request", [63, 64]),
                         ("cmd_mask", [64, 68])):
            assert slices[name] == sl
        act = data["action"]
        assert act["action_dim"] == 15
        assert len(act["joints"]) == 15
        assert act["joints"][-1]["name"] == "waist_yaw"
        assert act["decimation_steps"] == act["control_period_ms"] / 2.0
        assert {p["name"] for p in act["passive_conventional_pd"]} == {
            "left_foot_yaw", "right_foot_yaw"}


class TestRlCurriculum:
    def test_stage_table(self, client):
        r = client.get("/api/rl/curriculum")
        assert r.status_code == 200
        stages = r.json()["stages"]
        assert [s["id"] for s in stages] == ["standing", "controlled", "sustained",
                                             "disturbance", "terrain", "navigation"]
        assert [s["env_ready"] for s in stages][:3] == [True, True, True]
        assert any(not s["env_ready"] for s in stages)


class TestRlProbe:
    def test_zero_leg_measured(self, client):
        r = client.post("/api/rl/probe", json={"leg": "zero", "seconds": 1.0})
        assert r.status_code == 200
        s = r.json()["summary"]
        assert s["leg"] == "zero"
        assert s["steps"] == 50  # 1.0 s at the 20-ms decision period
        assert s["faulted_windows"] == 0 and s["falls"] == 0
        assert "components" in s and "progress" in s["components"]

    def test_gait_leg_bounded(self, client):
        r = client.post("/api/rl/probe",
                        json={"leg": "gait", "seconds": 2.0, "speed": 1.0})
        assert r.status_code == 200
        s = r.json()["summary"]
        assert s["leg"] == "gait"
        assert s["steps"] == 100
        # the scripted benchmark walks from real contact (measured displacement)
        assert s["net_displacement_m"] > 0.05

    def test_invalid_inputs_rejected(self, client):
        assert client.post("/api/rl/probe", json={"leg": "ppo"}).status_code == 400
        assert client.post("/api/rl/probe", json={"seconds": 99}).status_code == 400
        assert client.post("/api/rl/probe", json={"leg": "gait",
                                                  "speed": 50}).status_code == 400
