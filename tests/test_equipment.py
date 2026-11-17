import numpy as np

import types
import pytest
from playground.combat.equipment import EQUIPMENT, EquipmentItem, validate_equipment

# Helper stub world
class StubWorld:
    def __init__(self):
        self.time = 0.0
        self.thrown = []
    def throw(self, oid, direction, force=1.0, wait="rest"):
        pid = f"throw-{oid}-{self.time}"
        self.thrown.append((oid, direction, force, wait))
        return {"proj_ids": [pid]}

# Stub launcher records calls with origin, direction, spread_deg
class StubLauncher:
    def __init__(self):
        self.calls = []
    def __call__(self, origin, direction, spread_deg):
        pid = f"proj-{origin if isinstance(origin, (list, tuple, np.ndarray)) else 'unknown'}-{direction}"
        self.calls.append((origin, direction, spread_deg))
        return [pid]


def test_registry_validation():
    errs = validate_equipment()
    assert not errs, f"Registry errors: {errs}"
    # Spot‑check required keys for one entry
    spec = EQUIPMENT["paintball_pistol"]
    for key in ["id", "grip", "hands_required", "projectile_cfg", "ammo_capacity"]:
        assert key in spec

def test_grip_state_machine():
    w = StubWorld()
    item = EquipmentItem(w, "paintball_pistol")
    # Initially not gripped
    assert not (item.left_gripped or item.right_gripped)
    # Can grip with either hand free
    assert item.can_grip(True, True)
    item.grip(left=True, right=False)
    assert item.left_gripped and not item.right_gripped
    item.ungrip()
    assert not (item.left_gripped or item.right_gripped)
    # Two‑hand equipment
    rifle = EquipmentItem(w, "paintball_rifle")
    assert rifle.can_grip(True, True)
    rifle.grip()
    assert rifle.left_gripped and rifle.right_gripped
    rifle.ungrip()
    assert not (rifle.left_gripped or rifle.right_gripped)

def test_fire_rate_and_ammo_and_reload():
    w = StubWorld()
    launcher = StubLauncher()
    pistol = EquipmentItem(w, "paintball_pistol", launcher=launcher)
    pistol.grip(left=True)
    # First fire succeeds
    res1 = pistol.fire([1, 0, 0])
    assert res1["fired"]
    assert res1["ammo"] == pistol.spec["ammo_capacity"] - 1
    # Immediate second fire fails due to rate limit
    res2 = pistol.fire([1, 0, 0])
    assert not res2["fired"] and res2["reason"] == "rate_limit"
    # Fast‑forward beyond rate limit
    pistol._fast_forward(pistol.spec["rate_limit_s"] + 0.01)
    res3 = pistol.fire([1, 0, 0])
    assert res3["fired"]
    # Deplete ammo
    pistol.ammo = 0
    res4 = pistol.fire([1, 0, 0])
    assert not res4["fired"] and res4["reason"] == "no_ammo"
    # Reload and fast‑forward past reload time
    pistol.reload()
    pistol._fast_forward(pistol.spec["reload_steps_s"] + 0.01)
    assert pistol.ammo == pistol.spec["ammo_capacity"]
    # Fire after reload
    res5 = pistol.fire([1, 0, 0])
    assert res5["fired"]

def test_spread_determinism_and_shotgun_multi_pellet():
    w = StubWorld()
    launcher = StubLauncher()
    shotgun = EquipmentItem(w, "paintball_shotgun", launcher=launcher)
    shotgun.grip()
    # Fire twice with fast‑forward between to reset rate limit
    shotgun._fast_forward(shotgun.spec["rate_limit_s"] + 0.01)
    first = shotgun.fire([0, 1, 0])
    shotgun._fast_forward(shotgun.spec["rate_limit_s"] + 0.01)
    second = shotgun.fire([0, 1, 0])
    # The spread_deg values recorded by launcher should be identical and match spec
    assert launcher.calls[0][2] == launcher.calls[1][2] == shotgun.spec["accuracy_spread_deg"]
    # Verify shortfall reported correctly
    expected_shortfall = shotgun.spec["pellets_per_shot"] - len(first["proj_ids"])
    assert first.get("shortfall", 0) == expected_shortfall
    # proj_ids length should be whatever launcher returned (1 in our stub)
    assert len(first["proj_ids"]) == 1


def test_throwable_delegates_to_world_throw():
    w = StubWorld()
    grenade = EquipmentItem(w, "paintball_grenade")
    grenade.grip(left=True)
    res = grenade.fire([0, 0, 1])
    assert res["fired"]
    # Verify that world.throw was called with the correct force from spec
    expected_force = EQUIPMENT["paintball_grenade"]["throw_force"]
    assert w.thrown[0][2] == expected_force

    assert "proj_ids" in res
    assert len(res["proj_ids"]) == 1
