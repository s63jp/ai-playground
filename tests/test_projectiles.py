"""Real-simulator tests for the P5 projectile/impact/explosion subsystem.

Every test drives an actual PhysicsWorld (no mocks). The incidence-angle
convention is documented in playground/combat/projectiles.py::update:
angle between the projectile's incoming velocity and the surface normal
(0 deg for a straight-down drop onto flat ground).
"""
from __future__ import annotations

import numpy as np
import pytest

from playground.combat.projectiles import ProjectileSystem, explosion
from playground.config import load_config
from playground.world.physics import PhysicsWorld


@pytest.fixture()
def world() -> PhysicsWorld:
    return PhysicsWorld(load_config())


def test_launch_and_impact_on_flat_ground(world):
    """Drop a pooled block straight down onto the island ground and verify the impact
    contract: sane position, unit surface normal pointing up, ~0 deg incidence."""
    sys = ProjectileSystem(world, ["block_01"])
    # launch beside the agent spawn (0,0): dropping a block onto the agent's
    # own position first hits the agent body, not the bare ground.
    result = sys.launch([2.5, 0.0, 0.6], [0.0, 0.0, -1.0], 4.0)
    assert "proj_id" in result and result["proj_id"] == "block_01"

    impacts = []
    for _ in range(400):
        impacts = sys.update()
        if impacts:
            break
    assert impacts, "no impact detected within the 400-step budget"

    imp = impacts[0]
    assert imp["proj_id"] == "block_01"
    assert imp["surface"] == "ground"
    assert isinstance(imp["speed"], float)
    assert imp["t"] is not None
    pos = np.array(imp["pos"], dtype=float)
    assert pos[2] < 0.35, f"impact position z={pos[2]} is not near the ground plane"
    normal = np.array(imp["normal"], dtype=float)
    assert abs(float(np.linalg.norm(normal)) - 1.0) < 0.05, "impact normal is not unit length"
    assert normal[2] > 0.9, f"ground normal should point up, got {normal}"
    assert imp["angle_deg"] < 20.0, (
        f"straight-down drop must be near 0 deg incidence, got {imp['angle_deg']} deg")


def test_pool_exhaustion_returns_error_dict(world):
    sys = ProjectileSystem(world, ["block_01", "block_02", "block_03"])
    for i, oid in enumerate(["block_01", "block_02", "block_03"]):
        out = sys.launch([2.5, 0.0, 0.6 + i * 0.3], [0.0, 0.0, -1.0], 3.0)
        assert out.get("proj_id") == oid
    out4 = sys.launch([2.5, 0.0, 1.0], [1.0, 0.0, 0.0], 3.0)
    assert out4 == {"error": "pool_exhausted"}


def test_explosion_applies_impulse_and_emits_payload(world):
    from playground.world.physics import PhysicsWorld as _W  # noqa: F401
    rock_pos = np.array(world.obj_pos("rock"), dtype=float)
    before = float(np.linalg.norm(world.obj_vel("rock")))
    payload = explosion(world, (rock_pos + np.array([0.5, 0.0, 0.0])).tolist(),
                        radius=2.0, force=80.0, particle_count=64)
    after = float(np.linalg.norm(world.obj_vel("rock")))
    assert after > before, f"explosion did not move the rock (before {before}, after {after})"

    dirs = np.array(payload["particle_dirs"], dtype=float)
    assert dirs.shape == (64, 3)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-9), "Fibonacci particle dirs must be unit length"
    payload2 = explosion(world, (rock_pos + np.array([0.5, 0.0, 0.0])).tolist(),
                         radius=2.0, force=80.0, particle_count=64)
    assert payload2["particle_dirs"] == payload["particle_dirs"], (
        "Fibonacci sampling must be deterministic")

    kinds = [ev.get("type") for ev in world.events]
    assert "explosion" in kinds, "explosion event missing from world events"
