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
        assert data["contract_version"] == 3
        assert data["obs_dim"] == 43 and data["action_dim"] == 14
        obs = data["observation"]
        assert obs["layout_version"] == 3 and obs["dim"] == 43
        assert sum(s["slice"][1] - s["slice"][0] for s in obs["sections"]) == 43
        assert obs["sections"][-1]["name"] == "command" \
            and obs["sections"][-1]["slice"] == [40, 43]
        act = data["action"]
        assert act["action_dim"] == 14
        assert len(act["joints"]) == 14
        assert act["decimation_steps"] == act["control_period_ms"] / 2.0
        assert {p["name"] for p in act["passive_conventional_pd"]} == {
            "waist_yaw", "left_foot_yaw", "right_foot_yaw"}


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
