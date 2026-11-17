"""M4 phase 3 — combat actions, physical incapacitation and match termination.

Spec: docs/specifications/claims-remediation-mission.md M4 + CLAIMS_MATRIX rows
20/23/24 — physical attacks and defensive actions; damage, health, incapacity
and match termination stay physics-derived.

HVE shape:
- Happy: a melee swing drives the held baton through the opposing rig through
  the waist articulation; its real MuJoCo contact produces damage. A depleted
  health pool physically collapses the rig (posture actuation released — the
  figure folds under gravity) and the elimination of one side ends the match.
- Validate: a melee attack requires a melee weapon and respects its rate
  limit; the guard pose needs a shield and physically raises the plate; a
  downed agent's movement verbs refuse.
- Error: worlds without combat refuse attack/defend with the no_combat code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _combat_cfg(pellets: int = 8):
    from playground.config import load_config
    cfg = load_config()
    cfg["agent_body"]["type"] = "humanoid"
    cfg["agent_bodies"] = {
        "agent_001": {"start": [0.0, 0.0]},
        # the left arm swings at x ≈ +0.18; the target stands clear of the
        # resting weapon corridor but inside the raise arc's reach
        "agent_002": {"start": [0.19, 0.8]},
    }
    cfg["combat"] = {
        "teams": {"blue": ["agent_001"], "red": ["agent_002"]},
        "health": 30.0,          # a small pool: 2-3 measured hits incapacitate
        "items": [
            {"id": "baton_01", "item": "training_baton", "at": [0.55, 0.35]},
            {"id": "pistol_01", "item": "paintball_pistol", "at": [0.9, 0.6]},
            {"id": "shield_01", "item": "paint_shield", "at": [0.75, 0.85]},
        ],
        "pellets": pellets,
    }
    return cfg


@pytest.fixture()
def combat_world():
    from playground.world.physics import PhysicsWorld
    world = PhysicsWorld(_combat_cfg())
    world._agents["agent_001"]._climb = None
    return world


def _tools(world, aid):
    from playground.interface.physical import PhysicalTools
    return PhysicalTools(world, agent_id=aid)


def _events(world, etype):
    return [e for e in world.events if e["type"] == etype]


def _torso_z(world, aid: str) -> float:
    rt = world._agents[aid]
    return float(np.asarray(world.d.xpos)[rt.torso_bid, 2])


def _pistol_aim(w):
    """Real aim vector: from the held pistol's body to the target's torso."""
    m = np.asarray(w.obj_pos("pistol_01"))
    t = np.asarray(w.agent_pos("agent_002")) + np.array([0.0, 0.0, 0.35])
    d = t - m
    return (d / float(np.linalg.norm(d))).tolist()


def _down_the_target(w):
    """Fire the pistol until agent_002's pool empties (real measured damage)."""
    tools = _tools(w, "agent_001")
    state = tools.combat_state()
    if state["equipped"] is None or state["equipped"]["item"] != "paintball_pistol":
        tools.equip("pistol_01")
    st = w.combat_state("agent_002")
    for _ in range(24):
        out = tools.fire(_pistol_aim(w))
        for _ in range(int(0.35 / w.dt)):
            w._step()
        st = w.combat_state("agent_002")
        if st["incapacitated"]:
            return True
        if out.get("reason") == "pool_empty":
            break
    return st["incapacitated"]


# ------------------------------------------------------------------ melee
def test_a_melee_attack_drives_the_weapon_through_the_target(combat_world):
    w = combat_world
    tools = _tools(w, "agent_001")
    tools.equip("baton_01")
    out = tools.attack([0.0, 1.0, 0.0])  # north, straight at agent_002
    assert out["attacked"] is True
    assert out["weapon"] == "training_baton"
    assert out["swing_time_s"] == pytest.approx(0.35)
    contacts = [e for e in _events(w, "collision")
                if "baton_01" in e["data"].get("objects", [])
                and "agent_002" in e["data"].get("objects", [])]
    assert contacts, "the weapon must physically contact the opposing rig"
    assert float(contacts[0]["data"]["impulse"]) > 0.3
    assert not w.agent_fallen("agent_001")
    assert not w.agent_fallen("agent_002")


def test_b_melee_attack_needs_a_melee_weapon_and_respects_rate_limit(combat_world):
    from playground.world.physics import ActionError
    w = combat_world
    tools = _tools(w, "agent_001")
    with pytest.raises(ActionError) as e:
        tools.attack([0.0, 1.0, 0.0])
    assert e.value.code == "not_equipped"
    tools.equip("pistol_01")
    with pytest.raises(ActionError) as e:
        tools.attack([0.0, 1.0, 0.0])
    assert e.value.code == "not_a_melee_weapon"
    tools.unequip()
    tools.equip("baton_01")
    out = tools.attack([0.0, 1.0, 0.0])
    assert out["attacked"] is True
    out = tools.attack([0.0, 1.0, 0.0])
    assert out["attacked"] is False and out.get("reason") == "rate_limit"
    for _ in range(int(1.5 / w.dt)):
        w._step()
    out = tools.attack([0.0, 1.0, 0.0])
    assert out["attacked"] is True, f"attack after the rate window: {out}"


def test_c_defend_raises_the_plate_into_a_guard_pose(combat_world):
    w = combat_world
    tools = _tools(w, "agent_001")
    tools.equip("baton_01")
    from playground.world.physics import ActionError
    with pytest.raises(ActionError) as e:
        tools.defend()
    assert e.value.code == "no_shield"
    tools.unequip()
    tools.equip("shield_01")
    before = float(np.asarray(w.obj_pos("shield_01"))[2])
    out = tools.defend()
    assert out["guarded"] is True
    after = float(np.asarray(w.obj_pos("shield_01"))[2])
    assert after > before + 0.1, f"guard must raise the plate: {before:.2f} -> {after:.2f}"
    assert tools.combat_state()["bracing"] is True
    out = tools.defend(stance="carry")
    assert tools.combat_state()["bracing"] is False


# ------------------------------------------------------------------ incapacity
def test_d_depleted_health_physically_collapses_the_rig(combat_world):
    w = combat_world
    z0 = _torso_z(w, "agent_002")
    assert _down_the_target(w), "the pelting must incapacitate agent_002"
    inc = _events(w, "agent_incapacitated")
    assert inc and inc[0]["data"].get("agent") == "agent_002"
    assert w.combat_state("agent_002")["health"] == pytest.approx(0.0)
    # the collapse is physical: the torso sinks once posture actuation is gone
    for _ in range(int(2.0 / w.dt)):
        w._step()
    z1 = _torso_z(w, "agent_002")
    assert z0 - z1 > 0.35, f"torso height {z0:.2f} -> {z1:.2f}: no physical collapse"
    assert w._agents["agent_002"].held is None


def test_e_movement_refused_while_incapacitated(combat_world):
    from playground.world.physics import ActionError
    w = combat_world
    assert _down_the_target(w), "the pelting must incapacitate agent_002"
    with pytest.raises(ActionError) as e:
        w.set_gait("walk", speed=0.5, heading=90.0, agent="agent_002")
    assert e.value.code == "incapacitated"
    with pytest.raises(ActionError) as e:
        w.set_gait("stand", agent="agent_002")
    assert e.value.code == "incapacitated"


def test_f_match_ends_when_a_side_is_eliminated(combat_world):
    w = combat_world
    assert _down_the_target(w), "the pelting must incapacitate agent_002"
    for _ in range(int(0.8 / w.dt)):
        w._step()
    assert w.combat_state("agent_001")["match_over"] is True
    end_events = _events(w, "match_ended")
    assert end_events and end_events[0]["data"].get("winner") == "blue"
    assert w.combat_state("agent_001")["winner"] == "blue"
    assert w.agents_report("agent_001")[0]["incapacitated"] is True


# ------------------------------------------------------------------ errors
def test_g_no_combat_worlds_refuse_the_combat_verbs():
    from playground.interface.physical import PhysicalTools
    from playground.world.physics import ActionError, PhysicsWorld
    from playground.config import load_config
    cfg = load_config()
    cfg["agent_body"]["type"] = "humanoid"
    w = PhysicsWorld(cfg)
    w._agents["agent_001"]._climb = None
    tools = PhysicalTools(w, agent_id="agent_001")
    with pytest.raises(ActionError) as e:
        tools.attack([0.0, 1.0, 0.0])
    assert e.value.code == "no_combat"
    with pytest.raises(ActionError) as e:
        tools.defend()
    assert e.value.code == "no_combat"
