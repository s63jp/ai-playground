"""Production arena factories (M5): REAL MuJoCo worlds + REAL recorded sessions.

HVE shape:
- Happy: real solo-duel world build (agents / equipment / latent pellet pool / facing /
  spacing), real event translation with true proj_id -> shooter attribution, a scripted
  duel end-to-end through run_arena_match with recorded arena_solo sessions (prompt id +
  system prompt hash + frames + opening), a 2v2 world dispatching arena_coop with
  ARENA_COOP_SYSTEM recorded, and the `playground arena` CLI (direct call + subprocess,
  tmp db / tmp root, real report written).
- Validate: unknown equipment item refused; duplicate combatant id refused; a
  single-team config refused.
- Error: the refusals are the honest ArenaFactoryError surface.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from playground.agents.adapters import PolicyAdapter
from playground.agents.prompts import ARENA_COOP_SYSTEM, ARENA_SOLO_SYSTEM, prompt_hash
from playground.arena.factories import (ArenaFactoryError, build_match_setup, mode_for,
                                        production_world_factory, run_arena_match,
                                        DEFAULT_SPAWN_DISTANCE)
from playground.arena.match import export_match_report
from playground.arena.scoring import ArenaScorer
from playground.config import load_config
from playground.storage import Database
from playground.world.physics import PhysicsWorld

ROOT = Path(__file__).resolve().parent.parent


def _match_cfg(**over):
    cfg = {
        "teams": [
            {"name": "Red", "agents": [{"id": "agent_001"}], "equipment": ["paintball_pistol"]},
            {"name": "Blue", "agents": [{"id": "agent_002"}], "equipment": ["paintball_pistol"]},
        ],
        "rules": {"score_target": 100, "duration_s": 14.0, "friendly_fire": False, "pellets": 24},
        "arena": {},
    }
    cfg.update(over)
    return cfg


def _root_cfg(tmp_path):
    return load_config(overrides={"_root": str(tmp_path), "session": {"max_steps": 400}})


def _aim(w, agent_id: str, at_id: str, item_id: str) -> List[float]:
    """Straight-line direction from the held item's body to the target's torso (M4 math)."""
    t = np.asarray(w.agent_pos(at_id)) + np.array([0.0, 0.0, 0.35])
    m = np.asarray(w.obj_pos(item_id))
    d = t - m
    return (d / float(np.linalg.norm(d))).tolist()


def _scripted_policy(item_id: str, aim_dir: List[float]):
    def policy(system, opening, tools):
        yield ("say", f"Taking position; equipping {item_id}.")
        yield ("call", "equip", {"object_id": item_id})
        while True:  # the match ends the session; ammo/pool refusals are honest
            yield ("call", "fire", {"direction": aim_dir})
    return policy


def _duel_factory(item_by_agent, aim_by_agent):
    def factory(agent_id, plan_item):
        return PolicyAdapter(_scripted_policy(item_by_agent[agent_id], aim_by_agent[agent_id]))
    return factory


def _idle_policy():
    def policy(system, opening, tools):
        yield ("say", "Holding my ground.")
        yield ("call", "combat_state", {})
        yield ("call", "end_session", {})
    return policy


# ------------------------------------------------------------------ Happy
def test_build_setup_translates_real_combat_cfg(tmp_path):
    setup = build_match_setup(_match_cfg(), _root_cfg(tmp_path))
    wc = setup.world_cfg
    assert wc["agent_body"]["type"] == "humanoid"
    assert wc["agent_bodies"] == {
        "agent_001": {"start": [-1.4, 0.0]},
        "agent_002": {"start": [1.4, 0.0]},
    }
    assert setup.combat_teams == {"Red": ["agent_001"], "Blue": ["agent_002"]}
    assert wc["combat"]["pellets"] == 24
    items = {i["id"]: i for i in wc["combat"]["items"]}
    assert items["red_item_0"]["item"] == "paintball_pistol"
    assert items["red_item_0"]["at"] == pytest.approx([-0.8, 0.0], abs=1e-3)
    assert items["blue_item_0"]["at"] == pytest.approx([0.8, 0.0], abs=1e-3)
    assert setup.poses == {"agent_001": 0.0, "agent_002": 180.0}


def test_production_world_builds_real_combat_world(tmp_path):
    setup = build_match_setup(_match_cfg(), _root_cfg(tmp_path))
    adapter = production_world_factory(setup, ArenaScorer({"score_target": 100}))
    w = adapter.world
    assert isinstance(w, PhysicsWorld) and w._multi is True
    assert sorted(w._agents) == ["agent_001", "agent_002"]
    assert w.combat is not None and sorted(w.combat.teams) == ["Blue", "Red"]
    for oid in ("red_item_0", "blue_item_0"):
        assert oid in w.specs and w.specs[oid].kind == "equipment"
    pellets = [k for k in w.specs if k.startswith("pellet_")]
    assert len(pellets) == 24  # the pool is real
    gap = float(np.linalg.norm(w.agent_pos("agent_001") - w.agent_pos("agent_002")))
    assert gap == pytest.approx(DEFAULT_SPAWN_DISTANCE, abs=0.3)
    for aid in ("agent_001", "agent_002"):
        rt = w._agents[aid]
        up = float(np.asarray(w.d.xmat[rt.torso_bid]).reshape(3, 3)[2, 2])
        assert up > 0.9, f"{aid} must spawn upright"


def test_translate_real_events_attribute_shooter(tmp_path):
    setup = build_match_setup(_match_cfg(), _root_cfg(tmp_path))
    scorer = ArenaScorer({"score_target": 100})
    adapter = production_world_factory(setup, scorer)
    w = adapter.world
    equip_out = w.combat.equip("agent_001", "red_item_0")
    assert "error" not in equip_out, f"equip failed: {equip_out}"
    fired = w.combat.fire("agent_001", _aim(w, "agent_001", "agent_002", "red_item_0"))
    pid = (fired.get("proj_ids") or [None])[0]
    assert pid and pid.startswith("pellet_")
    shot = impact = None
    for _ in range(120):  # 6 s of sim time: flight + impact + damage pass
        for e in adapter.step(0.05):
            if e["type"] == "pellet_shot" and shot is None:
                shot = e
            if e["type"] == "projectile_impact" and impact is None:
                impact = e
            if e["type"] == "projectile_impact":
                scorer.process_event(e)  # the match loop's scoring job, replayed directly
        if impact is not None:
            break
    assert shot is not None, "the pellet launch must reach the match timeline"
    assert shot["proj_id"] == pid and shot["shooter_agent"] == "agent_001"
    assert shot["shooter_team"] == "Red"
    assert impact is not None, "an aimed shot must register a real impact"
    assert impact["target_agent"] == "agent_002" and impact["team"] == "Blue"
    assert (impact.get("damage") or 0) >= 0
    # the scorer holds the real registry and scored the real hit
    assert scorer._projectile_owner[pid] == ("agent_001", "Red")
    assert scorer.tally()[0].get("Red", 0) >= 1


def test_solo_duel_end_to_end_with_recorded_sessions(tmp_path):
    match_cfg = _match_cfg()
    root = _root_cfg(tmp_path)
    db = Database(tmp_path / "arena.db")
    factory = _duel_factory(
        {"agent_001": "red_item_0", "agent_002": "blue_item_0"},
        {"agent_001": [1.0, 0.0, 0.05], "agent_002": [-1.0, 0.0, 0.05]},
    )
    result = run_arena_match(match_cfg, factory, db=db, root_cfg=root, match_name="solo")
    # the result contract
    assert result["end_reason"] in ("timeout", "score_target", "incapacitated")
    sessions = result["sessions"]
    assert [s["team"] for s in sessions] == ["Red", "Blue"]
    assert all(s["mode"] == "arena_solo" for s in sessions)
    shots = [e for e in result["timeline"] if e["type"] == "pellet_shot"]
    assert len(shots) >= 2 and all("elapsed" in e for e in result["timeline"])
    assert len(result["timeline"]) >= len(shots)
    # combat stance: the rigs must survive the whole match upright (the idle-topple gap)
    world_adapter = result["adapter"]
    for aid in ("agent_001", "agent_002"):
        z = float(world_adapter.world.agent_pos(aid)[2])
        assert z >= 0.65, f"{aid} must stand through the match (z={z})"
    # the real recorded sessions: ids, arena mode, prompt id + hash, opening, frames
    from playground.agents.prompts import prompt_hash as ph
    for s in sessions:
        sid = s["session_id"]
        row = db.session(sid)
        assert row["mode"] == "arena_solo"
        assert json.loads(row["budget_json"])["minutes"] == 3  # match-time budget, not wall
        note = db.events(sid, types=["session_started"])[0]["data"]
        assert note["prompt_id"] == "arena_solo"
        assert note["system_prompt_hash"] == ph(ARENA_SOLO_SYSTEM)
        sys_msgs = [m for m in db.messages(sid) if m["role"] == "system"]
        assert sys_msgs[0]["content"] == ARENA_SOLO_SYSTEM
        openings = [m for m in db.messages(sid) if m["role"] == "user"]
        assert any("Arena match session" in m["content"] for m in openings)
        assert db.frame_count(sid) > 0, "the shared world must stream frames to each session"
    db.close()


def test_coop_config_dispatches_arena_coop(tmp_path):
    match_cfg = _match_cfg(
        teams=[
            {"name": "Red", "agents": [{"id": "agent_001"}, {"id": "agent_002"}],
             "equipment": ["paintball_pistol", "paintball_pistol"]},
            {"name": "Blue", "agents": [{"id": "agent_003"}, {"id": "agent_004"}],
             "equipment": ["paintball_pistol", "paintball_pistol"]},
        ],
        rules={"score_target": 100, "duration_s": 6.0, "friendly_fire": False, "pellets": 24},
    )
    root = _root_cfg(tmp_path)
    db = Database(tmp_path / "coop.db")
    result = run_arena_match(match_cfg, lambda aid, plan: PolicyAdapter(_idle_policy()),
                             db=db, root_cfg=root, turn_interval_s=5.0, match_name="coop")
    assert len(result["sessions"]) == 4
    assert all(s["mode"] == "arena_coop" for s in result["sessions"])
    for s in result["sessions"]:
        sid = s["session_id"]
        note = db.events(sid, types=["session_started"])[0]["data"]
        assert note["prompt_id"] == "arena_coop"
        assert note["system_prompt_hash"] == prompt_hash(ARENA_COOP_SYSTEM)
        sys_msgs = [m for m in db.messages(sid) if m["role"] == "system"]
        assert sys_msgs[0]["content"] == ARENA_COOP_SYSTEM
    db.close()


def test_mode_for_two_teams_one_agent_is_arena_solo():
    assert mode_for(_match_cfg()["teams"]) == "arena_solo"
    assert mode_for(_match_cfg(
        teams=[{"name": "Red", "agents": [{"id": "a1"}, {"id": "a2"}], "equipment": []},
               {"name": "Blue", "agents": [{"id": "a3"}], "equipment": []}])["teams"]) == "arena_coop"


def test_report_json_carries_timeline_and_end_reason(tmp_path):
    root = _root_cfg(tmp_path)
    db = Database(tmp_path / "rep.db")
    result = run_arena_match(_match_cfg(),
                             _duel_factory({"agent_001": "red_item_0", "agent_002": "blue_item_0"},
                                           {"agent_001": [1.0, 0.0, 0.05],
                                            "agent_002": [-1.0, 0.0, 0.05]}),
                             db=db, root_cfg=root, match_name="rep")
    out_path = tmp_path / "report.json"
    export_match_report(result, str(out_path))
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["timeline"] == result["timeline"]
    assert data["end_reason"] == result["end_reason"]
    assert len(data["sessions"]) == 2
    db.close()


# ------------------------------------------------------------------ CLI
def _write_match_json(tmp_path) -> Path:
    cfg_path = tmp_path / "duel.json"
    cfg_path.write_text(json.dumps(_match_cfg(rules={"score_target": 100,
                                               "duration_s": 6.0,
                                               "friendly_fire": False,
                                               "pellets": 24})), encoding="utf-8")
    return cfg_path


def test_cli_cmd_arena_direct_writes_report(tmp_path):
    from playground.arena.factories import cmd_arena
    cfg_path = _write_match_json(tmp_path)
    args = SimpleNamespace(config=str(cfg_path), out=str(tmp_path / "direct.report.json"),
                           adapter="scripted", model=None, db=str(tmp_path / "cli.db"),
                           root=str(tmp_path))
    assert cmd_arena(args) == 0
    report = json.loads((tmp_path / "direct.report.json").read_text(encoding="utf-8"))
    assert "winner" in report and report["end_reason"] in ("timeout", "score_target", "incapacitated")
    assert isinstance(report["timeline"], list)


def test_cli_subcommand_runs_a_real_match(tmp_path):
    cfg_path = _write_match_json(tmp_path)
    cmd = [sys.executable, "-m", "playground", "arena", str(cfg_path),
           "--out", str(tmp_path / "sub.report.json"), "--db", str(tmp_path / "sub.db"),
           "--root", str(tmp_path)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"arena CLI failed:\n{proc.stdout}\n{proc.stderr}"
    report = json.loads((tmp_path / "sub.report.json").read_text(encoding="utf-8"))
    assert report["end_reason"] in ("timeout", "score_target", "incapacitated")
    assert isinstance(report["timeline"], list)
    summary = json.loads(proc.stdout)  # the CLI prints its result summary
    assert summary["report"].endswith("sub.report.json")


def test_dashboard_exposes_arena_matches(tmp_path):
    from fastapi.testclient import TestClient
    from playground.dashboard.server import create_app
    root = _root_cfg(tmp_path)
    database = Database(tmp_path / "dash.db")
    app = create_app(root, database)
    client = TestClient(app)
    result = run_arena_match(
        _match_cfg(rules={"score_target": 100, "duration_s": 6.0,
                          "friendly_fire": False, "pellets": 24}),
        lambda aid, plan: PolicyAdapter(_idle_policy()),
        db=database, root_cfg=root, match_name="dash")
    r = client.get("/api/arena")
    assert r.status_code == 200
    matches = r.json()["matches"]
    assert matches, "the recorded arena match must be visible to the dashboard"
    top = matches[0]
    assert top["label"] == "arena:dash"
    assert len(top["sessions"]) == 2
    assert {s["id"] for s in top["sessions"]} == {s["session_id"] for s in result["sessions"]}
    assert top["status"] == "completed"


# ------------------------------------------------------------------ Validate / Error
def test_unknown_equipment_item_is_refused(tmp_path):
    cfg = _match_cfg()
    cfg["teams"][0]["equipment"] = ["blaster_3000"]
    with pytest.raises(ArenaFactoryError, match="unknown equipment item"):
        build_match_setup(cfg, _root_cfg(tmp_path))


def test_duplicate_combatant_id_is_refused(tmp_path):
    cfg = _match_cfg(teams=[
        {"name": "Red", "agents": [{"id": "agent_001"}], "equipment": ["paintball_pistol"]},
        {"name": "Blue", "agents": [{"id": "agent_001"}], "equipment": ["paintball_pistol"]},
    ])
    with pytest.raises(ArenaFactoryError, match="unique"):
        build_match_setup(cfg, _root_cfg(tmp_path))


def test_single_team_config_is_refused(tmp_path):
    with pytest.raises(ArenaFactoryError, match="two teams"):
        build_match_setup(_match_cfg(teams=[
            {"name": "Red", "agents": [{"id": "agent_001"}],
             "equipment": ["paintball_pistol"]}]), _root_cfg(tmp_path))
