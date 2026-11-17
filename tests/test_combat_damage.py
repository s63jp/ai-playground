"""M4 phase 2 — physical projectiles and measured contact-event damage.

Spec: docs/specifications/claims-remediation-mission.md M4 — impacts detected
through real MuJoCo contacts, damage derived from physical interactions +
equipment properties + impact conditions (never an invented counter).

HVE shape:
- Happy: firing launches a pool pellet at the measured muzzle; the pellet's
  real MuJoCo contact with the opposing agent produces a collision event with
  a measured impulse, and the runtime derives damage from it (health drops by
  exactly the amount the event carries).
- Validate: melee/shield items refuse fire (not_a_launcher); firing with empty
  hands refuses (not_equipped); an exhausted pool refuses honestly; a reload
  is staged for the item's reload time and unblocks firing afterwards.
- Error/guard: a gentle walk-in collision (measured impulse below the damage
  floor) damages nobody; thrown equipment leaves the hold.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _combat_cfg(pellets: int = 12):
    from playground.config import load_config
    cfg = load_config()
    cfg["agent_body"]["type"] = "humanoid"
    cfg["agent_bodies"] = {
        "agent_001": {"start": [0.4, 0.0]},
        "agent_002": {"start": [2.6, 0.0]},
    }
    cfg["combat"] = {
        "teams": {"blue": ["agent_001"], "red": ["agent_002"]},
        "items": [
            {"id": "pistol_01", "item": "paintball_pistol", "at": [0.6, 0.0]},
            {"id": "shield_01", "item": "paint_shield", "at": [2.2, 0.6]},
            {"id": "grenade_01", "item": "paintball_grenade", "at": [0.3, 0.8]},
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


def _armed(world):
    tools = _tools(world, "agent_001")
    tools.equip("pistol_01")
    return tools


def _aim(w, at: str = "agent_002", item: str = "pistol_01") -> list[float]:
    """Straight-line direction from the held item's body to the target's torso.

    The rig's root sits ~0.35 m below the torso centre; aiming at the root
    can thread the leg gap at mid-line, so combat aiming points at the body.
    """
    t = np.asarray(w.agent_pos(at)) + np.array([0.0, 0.0, 0.35])
    m = np.asarray(w.obj_pos(item))
    d = t - m
    return (d / float(np.linalg.norm(d))).tolist()


def _events(world, etype):
    return [e for e in world.events if e["type"] == etype]


def _damage_events(world, agent: str):
    return [e for e in _events(world, "agent_damaged") if e["data"].get("agent") == agent]


def _hit(world, pid: str, agent: str) -> bool:
    for e in _events(world, "collision"):
        objs = e["data"].get("objects", [])
        if pid in objs and agent in objs:
            return True
    return False


def _run_until(world, predicate, seconds: float = 3.0) -> bool:
    """Step until predicate(world) or the budget is spent."""
    for _ in range(int(seconds / world.dt)):
        world._step()
        if predicate(world):
            return True
    return False


# ------------------------------------------------------------------ Happy
def test_a_fire_launches_pellet_and_damage_comes_from_the_measured_impact(combat_world):
    w = combat_world
    tools = _armed(w)
    out = tools.fire(_aim(w))
    assert out["fired"] is True
    assert out["ammo"] == 11
    assert len(out["proj_ids"]) == 1
    pid = out["proj_ids"][0]
    assert pid in w.active
    p = np.asarray(w.obj_pos(pid))
    assert float(np.linalg.norm(p)) > 0.1, "the pellet must leave the muzzle"
    v = np.asarray(w.obj_vel(pid))
    assert 14.0 <= float(np.linalg.norm(v)) <= 24.0, f"launch speed {float(np.linalg.norm(v)):.2f}"
    hit = _run_until(w, lambda _w: _hit(_w, pid, "agent_002"))
    assert hit, "pellet never contacted agent_002 through real physics"
    # the collision event carries a measured impulse and the struck body part
    ev = [e for e in _events(w, "collision") if pid in e["data"].get("objects", [])][-1]
    assert float(ev["data"]["impulse"]) > 0.3
    assert "agent_parts" in ev["data"], "agent collisions must name the struck body part"
    dmg = _damage_events(w, "agent_002")
    assert len(dmg) >= 1
    d = dmg[0]["data"]
    assert d["source"] == pid
    assert d["health_max"] == 100.0
    st = w.combat_state("agent_002")
    assert st["health"] == pytest.approx(100.0 - d["damage"])
    assert 5.0 <= d["damage"] <= 60.0, f"damage {d['damage']:.2f} out of the measured band"


def test_b_gentle_walk_in_produces_no_damage(combat_world):
    w = combat_world
    _armed(w)  # a fully armed shooter walking into a standing agent
    w.set_gait("walk", speed=0.5, heading=90.0, agent="agent_001")
    for _ in range(int(6.0 / w.dt)):
        w._step()
    w.set_gait("stand", agent="agent_001")
    contacts = sum(1 for e in _events(w, "collision")
                   if "agent_002" in e["data"].get("objects", []))
    assert contacts >= 1, "walk-in must still produce measured contact events"
    assert _damage_events(w, "agent_002") == [], "walk-in contacts must stay below the damage floor"
    assert w.combat_state("agent_002")["health"] == pytest.approx(100.0)
    assert not w.agent_fallen("agent_002")


def test_c_shield_absorbs_a_direct_hit():
    from playground.world.physics import PhysicsWorld

    def _shield_cfg():
        """The target faces WEST (researcher-side setup, the same initial
        pose the arena factories will give combatants), so its held plate
        stands between it and the westbound-incoming shooter."""
        from playground.config import load_config
        cfg = load_config()
        cfg["agent_body"]["type"] = "humanoid"
        cfg["agent_bodies"] = {
            "agent_001": {"start": [0.4, 0.9]},
            "agent_002": {"start": [2.4, 1.0]},
        }
        cfg["combat"] = {
            "teams": {"blue": ["agent_001"], "red": ["agent_002"]},
            "items": [
                {"id": "pistol_01", "item": "paintball_pistol", "at": [0.6, 0.7]},
                {"id": "shield_01", "item": "paint_shield", "at": [2.1, 1.2]},
            ],
            "pellets": 4,
        }
        return cfg

    def run_shot(shielded: bool):
        w = PhysicsWorld(_shield_cfg())
        w._agents["agent_001"]._climb = None
        w.setup_agent_pose("agent_002", yaw_degrees=180.0)  # chest to the west
        if shielded:
            # equip immediately: the plate is a held body from step 0
            _tools(w, "agent_002").equip("shield_01")
        tools = _armed(w)
        for _ in range(int(0.4 / w.dt)):
            w._step()
        out = tools.fire(_aim(w))
        pid = out["proj_ids"][0]
        assert _run_until(w, lambda _w: _hit(_w, pid, "agent_002") or _hit(_w, pid, "shield_01"))
        for _ in range(int(0.5 / w.dt)):
            w._step()
        dmg = _damage_events(w, "agent_002")
        assert len(dmg) >= 1, "damage missing in %s shot" % ("shielded" if shielded else "baseline")
        return float(dmg[0]["data"]["damage"]), float(dmg[0]["data"]["impulse"]), dmg[0]["data"]

    dmg_a, imp_a, _ = run_shot(shielded=False)
    dmg_b, imp_b, d2 = run_shot(shielded=True)
    assert d2["shielded"] is True
    assert imp_b > 0.3, "the shielded hit is still measured"
    assert dmg_b < 0.5 * dmg_a, f"shield absorbed too little: {dmg_b:.2f} vs {dmg_a:.2f}"


def test_d_fire_reload_and_pool_validation():
    from playground.world.physics import ActionError, PhysicsWorld
    w = PhysicsWorld(_combat_cfg(pellets=6))
    w._agents["agent_001"]._climb = None
    tools = _tools(w, "agent_001")
    with pytest.raises(ActionError) as e:
        _tools(w, "agent_002").fire([1.0, 0.0, 0.0])
    assert e.value.code == "not_equipped"
    _tools(w, "agent_002").equip("shield_01")
    with pytest.raises(ActionError) as e:
        _tools(w, "agent_002").fire([1.0, 0.0, 0.0])
    assert e.value.code == "not_a_launcher"
    tools.equip("pistol_01")
    # 6 shots consume the pool (0.2 s rate limit, 0.3 s of stepping between)
    for i in range(6):
        out = tools.fire([1.0, 0.0, 0.0])
        assert out["fired"] is True, f"shot {i} fired=False: {out}"
        for _ in range(int(0.3 / w.dt)):
            w._step()
    # the 7th shot has no pellet left: an honest refusal, ammo untouched
    stale = w.combat_state("agent_001")["equipped"]["ammo"]
    out = tools.fire([1.0, 0.0, 0.0])
    assert out["fired"] is False
    assert out.get("reason") == "pool_empty"
    assert w.combat_state("agent_001")["equipped"]["ammo"] == stale, \
        "a refused shot must not consume ammo"
    # reload stages for the item's reload time; during the window fire refuses
    re = tools.reload()
    assert re.get("reloading") is True
    for _ in range(int(0.3 / w.dt)):
        w._step()  # clear the 0.2 s rate limit left over from the previous shot
    out = tools.fire([1.0, 0.0, 0.0])
    assert out["fired"] is False and out.get("reason") == "reloading"
    for _ in range(int(2.0 / w.dt)):
        w._step()
    out = tools.fire([1.0, 0.0, 0.0])
    assert out["fired"] is False and out.get("reason") == "pool_empty", \
        "a reload without pellets cannot make a shot"
    assert w.combat_state("agent_001")["equipped"]["ammo"] == 12


def test_e_grenade_is_a_real_throwable(combat_world):
    w = combat_world
    tools = _tools(w, "agent_001")
    tools.equip("grenade_01")
    out = tools.fire([1.0, 0.0, 0.0])
    assert out["fired"] is True
    assert out["ammo"] == 0
    assert out.get("flight_id") is not None, "the real throw must return its flight record"
    for _ in range(int(1.5 / w.dt)):
        w._step()
    g = np.asarray(w.obj_pos("grenade_01"))
    a = np.asarray(w.agent_pos("agent_001"))
    assert math.hypot(float(g[0] - a[0]), float(g[1] - a[1])) > 0.5, "grenade must leave the agent"
    assert any(e["type"] == "throw_release" for e in w.events)
    assert tools.combat_state()["equipped"] is None, "a thrown item is no longer held"
    assert w._agents["agent_001"].held is None


def test_f_agents_report_health_reflects_damage(combat_world):
    w = combat_world
    tools = _armed(w)
    out = tools.fire(_aim(w))
    pid = out["proj_ids"][0]
    assert _run_until(w, lambda _w: _hit(_w, pid, "agent_002"))
    for _ in range(int(0.5 / w.dt)):
        w._step()
    rows = w.agents_report("agent_001")
    assert rows and rows[0]["id"] == "agent_002"
    assert rows[0]["health"] < 100.0
