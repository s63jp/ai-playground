"""M4 phase 1 — equipment is a real physical object and can be physically held.

Spec: docs/specifications/claims-remediation-mission.md M4 — "equipment that
agents can physically pick up, hold, carry and use". Equipment must interact
with the physical world, so the items are real movable MuJoCo bodies built from
the equipment registry, held through the existing carry hold (one hold at a
time) and dropped by the existing release.

HVE shape:
- Happy: a combat-configured multi-agent world builds the equipment bodies at
  their configured places, the equip tool pins the item to the body with the
  real carry hold (it rides the agent), combat_state/me/agents_report carry
  measured combat state, and unequip drops the item physically.
- Validate: equip rejects unknown ids, non-equipment objects, out-of-reach
  items and a second held object.
- Error: worlds without a combat config keep the pre-M4 surface exactly
  (world.combat is None, combat tools refuse, reports unchanged).
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _base_cfg():
    from playground.config import load_config
    cfg = load_config()
    cfg["agent_body"]["type"] = "humanoid"
    return cfg


def _combat_cfg():
    cfg = _base_cfg()
    cfg["agent_bodies"] = {
        "agent_001": {"start": [0.4, 0.0]},
        "agent_002": {"start": [2.6, 0.0]},
    }
    cfg["combat"] = {
        "teams": {"blue": ["agent_001"], "red": ["agent_002"]},
        "items": [
            {"id": "pistol_01", "item": "paintball_pistol", "at": [1.0, 0.0]},
            {"id": "shield_01", "item": "paint_shield", "at": [0.9, 1.2]},
            {"id": "baton_01", "item": "training_baton", "at": [3.0, 0.0]},
        ],
        "pellets": 8,
    }
    return cfg


@pytest.fixture()
def combat_world():
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(_combat_cfg())
    world._agents["agent_001"]._climb = None
    return world


@pytest.fixture()
def plain_world():
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(_base_cfg())
    world._agents["agent_001"]._climb = None
    return world


# ------------------------------------------------------------------ Happy
def test_a_combat_world_builds_equipment_bodies(combat_world):
    from playground.combat.equipment import EQUIPMENT
    w = combat_world
    assert w.combat is not None
    for oid, item, at in (("pistol_01", "paintball_pistol", (1.0, 0.0)),
                          ("shield_01", "paint_shield", (0.9, 1.2)),
                          ("baton_01", "training_baton", (3.0, 0.0))):
        s = w.specs[oid]
        assert s.kind == "equipment" and s.movable
        assert s.properties.get("equipment") == item
        p = w.obj_pos(oid)
        assert math.hypot(float(p[0]) - at[0], float(p[1]) - at[1]) < 0.05
        assert float(p[2]) > 0.0, f"{oid} must rest on the ground"
        assert w.obj_speed(oid) < 0.05, f"{oid} must start at rest"
        assert float(s.mass) == pytest.approx(EQUIPMENT[item]["mass_kg"])
    # pellets exist as latent parked pool bodies, not in the world yet
    for i in range(8):
        pid = f"pellet_{i + 1:02d}"
        assert pid in w.specs and w.specs[pid].latent
        assert pid not in w.active
    # both agents are upright and far enough apart
    a1, a2 = w.agent_pos("agent_001"), w.agent_pos("agent_002")
    assert float(np.linalg.norm(a1 - a2)) > 1.0
    assert not w.agent_fallen("agent_001") and not w.agent_fallen("agent_002")


def test_b_equip_pins_item_with_real_carry_hold(combat_world):
    from playground.interface.physical import PhysicalTools
    w = combat_world
    tools = PhysicalTools(w, agent_id="agent_001")
    out = tools.equip("pistol_01")
    assert out["equipped"] == "pistol_01"
    assert out["item"] == "paintball_pistol"
    assert out["ammo"] == 12
    assert w._agents["agent_001"].held == "pistol_01"
    assert out["held_object"] == "pistol_01"
    # the item is mounted ahead of the chest: it rides the walking body
    d0 = float(np.linalg.norm(np.asarray(w.obj_pos("pistol_01"))
                              - np.asarray(w.agent_pos("agent_001"))))
    w.set_gait("walk", speed=0.8, heading=90.0, agent="agent_001")
    for _ in range(int(0.7 / w.dt)):
        w._step()
        assert not np.isnan(w.d.qacc).any()
    w.set_gait("stand", agent="agent_001")
    for _ in range(int(0.8 / w.dt)):
        w._step()
    d1 = float(np.linalg.norm(np.asarray(w.obj_pos("pistol_01"))
                              - np.asarray(w.agent_pos("agent_001"))))
    assert d1 <= d0 + 0.25, f"carry hold drifted: {d0:.2f} m -> {d1:.2f} m"
    assert not w.agent_fallen("agent_001")
    # combat_state reports the measured hold
    st = tools.combat_state()
    assert st["equipped"]["object"] == "pistol_01"
    assert st["equipped"]["item"] == "paintball_pistol"
    assert st["health"] == pytest.approx(100.0)
    assert st["incapacitated"] is False
    assert st["team"] == "blue"


def test_c_unequip_drops_item_physically(combat_world):
    from playground.interface.physical import PhysicalTools
    w = combat_world
    tools = PhysicalTools(w, agent_id="agent_001")
    tools.equip("pistol_01")
    out = tools.unequip()
    assert out["dropped"] == "pistol_01"
    assert w._agents["agent_001"].held is None
    for _ in range(int(1.0 / w.dt)):
        w._step()
    a = np.asarray(w.agent_pos("agent_001"))
    p = np.asarray(w.obj_pos("pistol_01"))
    assert math.hypot(float(p[0] - a[0]), float(p[1] - a[1])) < 0.9
    assert w.obj_speed("pistol_01") < 0.05, "dropped item must come to rest"
    assert tools.combat_state()["equipped"] is None


def test_f_agents_report_carries_combat_state(combat_world):
    w = combat_world
    rows = w.agents_report("agent_001")
    assert [r["id"] for r in rows] == ["agent_002"]
    assert rows[0]["team"] == "red"
    assert rows[0]["health"] == pytest.approx(100.0)
    assert rows[0]["incapacitated"] is False


def test_g_combat_tools_in_catalogue():
    from playground.interface.schema import CATEGORIES, SPECS
    for name in ("equip", "unequip", "combat_state"):
        assert name in SPECS, name
        assert SPECS[name]["category"] == "combat"
    assert "combat" in CATEGORIES


# ------------------------------------------------------------------ Validate
def test_d_equip_rejects_invalid_holds(combat_world):
    from playground.interface.physical import PhysicalTools
    from playground.world.physics import ActionError
    w = combat_world
    tools = PhysicalTools(w, agent_id="agent_001")
    with pytest.raises(ActionError) as e:
        tools.equip("no_such_object")
    assert e.value.code == "unknown_object"
    with pytest.raises(ActionError) as e:
        tools.equip("block_01")
    assert e.value.code == "not_equipment"
    # baton_01 rests 2.6 m away — beyond arm reach, equipping must refuse
    with pytest.raises(ActionError) as e:
        tools.equip("baton_01")
    assert e.value.code == "out_of_reach"
    # one hold at a time (the existing carry contract)
    tools.equip("pistol_01")
    with pytest.raises(ActionError) as e:
        tools.equip("shield_01")
    assert e.value.code == "hands_full"


# ------------------------------------------------------------------ Error/legacy guard
def test_e_no_combat_config_keeps_pre_m4_surface(plain_world):
    from playground.interface.physical import PhysicalTools
    from playground.world.physics import ActionError
    w = plain_world
    assert w.combat is None
    tools = PhysicalTools(w, agent_id="agent_001")
    with pytest.raises(ActionError) as e:
        tools.equip("block_01")
    assert e.value.code == "no_combat"
    with pytest.raises(ActionError) as e:
        tools.combat_state()
    assert e.value.code == "no_combat"
    # no equipment anywhere in a plain world, and no combat keys in reports
    assert not [s for s in w.specs.values() if s.kind == "equipment"]
    assert w.agents_report("agent_001") == []
