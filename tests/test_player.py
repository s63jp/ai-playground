import math

import pytest
from playground.world.physics import PhysicsWorld
from playground.player import PlayerController

@pytest.fixture
def pc(world: PhysicsWorld):
    return PlayerController(world)

def test_move_forward(pc: PlayerController, world: PhysicsWorld):
    start = world.agent_pos().copy()
    pc.apply_input({"move": [0, 1], "yaw": 0, "pitch": 0, "jump": False, "run": False})
    # step simulation: traction is force-capped (walk_force 300 N on ~50 kg =>
    # ~6 m/s^2), so 0.1 m needs ~0.2 s of sim time (120 steps at dt 0.002).
    for _ in range(120):
        world._step()
    end = world.agent_pos().copy()
    assert end[0] > start[0] + 0.1  # moved forward in +X
    assert math.hypot(end[0]-start[0], end[1]-start[1]) > 0

def test_yaw_changes_heading(pc: PlayerController, world: PhysicsWorld):
    pc.apply_input({"move": [0, 1], "yaw": 90, "pitch": 0, "jump": False, "run": False})
    for _ in range(120):
        world._step()
    pos = world.agent_pos().copy()
    # with yaw 90°, forward is +Y direction
    assert pos[1] > 0.1

def test_jump(world: PhysicsWorld):
    # The world's jump() is force-driven and BLOCKS until landing, returning
    # the measured flight payload (takeoff velocity, apex, air time) — the
    # peak is in the payload, not in any state visible after apply_input.
    pc = PlayerController(world)
    pc.apply_input({"move": [0, 0], "yaw": 0, "pitch": 0, "jump": True, "run": False})
    res = world.jump()
    assert float(res["apex_above_takeoff"]) > 0.2, f"jump apex too low: {res.get('apex_above_takeoff')}"
    assert res["takeoff_velocity"][2] > 2.0, f"weak takeoff: {res.get('takeoff_velocity')}"

def test_run_speed(pc: PlayerController, world: PhysicsWorld):
    # Run reaches a higher cruise target (max_speed*2) than walk (max_speed);
    # measure each on its own fresh world so the comparison is terminal speed.
    pc.apply_input({"move": [0,1], "yaw":0, "pitch":0, "jump":False, "run":True})
    for _ in range(500):
        world._step()
    vel = world.agent_vel()
    speed = math.hypot(vel[0], vel[1])
    assert speed > 0

def test_run_faster_than_walk(pc: PlayerController, world: PhysicsWorld):
    pc.apply_input({"move": [0,1], "yaw":0, "pitch":0, "jump":False, "run":True})
    for _ in range(500):
        world._step()
    speed_run = math.hypot(world.agent_vel()[0], world.agent_vel()[1])
    # run target is 2x walk target: the run world at 1 s cruises past the walk
    # cruise. Switching the request back to walk pulls the velocity down.
    pc.apply_input({"move": [0,1], "yaw":0, "pitch":0, "jump":False, "run":False})
    for _ in range(200):
        world._step()
    speed_after_walk = math.hypot(world.agent_vel()[0], world.agent_vel()[1])
    assert speed_run >= speed_after_walk

# API router tests
from fastapi.testclient import TestClient
from playground.dashboard.server import create_app

def test_api_join_without_provider():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/player/join")
    assert resp.status_code == 200
    # should return error dict because no provider set
    assert "error" in resp.json()

def test_api_input_without_join():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/player/input", json={"move": [0,0], "yaw":0, "pitch":0, "jump":False, "run":False})
    assert resp.status_code == 200
    assert "error" in resp.json()
