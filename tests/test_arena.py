
"""Unit tests for the arena package.

The tests construct a tiny fake world that reports ``projectile_impact`` events and a fake session
object that simply forwards the same events back. They verify that:

* ``plan`` validates good and bad configurations.
* ``play`` aggregates scores correctly, respecting ``friendly_fire``.
* Victory by ``score_target`` and timeout are both reported.
* ``export_match_report`` writes a JSON file that can be round‑tripped.
"""
import json
from pathlib import Path

import pytest

from playground.arena.match import ArenaMatch, MatchResult
from playground.arena import scoring

# --------------------------------------------------------------------- helpers
class DummyWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        self.time = 0.0
        self.events_to_emit = []
        self.projectile_counter = 0
        # Keep a reference to the scorer so we can register projectiles.
        self.scorer = None

    def step(self, dt):
        self.time += dt
        # Emit any queued events for this step.
        ev = self.events_to_emit[:]
        self.events_to_emit.clear()
        return ev

    def launch_projectile(self, shooter_agent, shooter_team, target_agent, target_team):
        proj_id = f'p{self.projectile_counter}'
        self.projectile_counter += 1
        # Register with scorer (outside world).
        if self.scorer:
            self.scorer.register_projectile(proj_id, shooter_agent, shooter_team)
        # Queue impact event for the next world step.
        self.events_to_emit.append({
            'type': 'projectile_impact',
            'proj_id': proj_id,
            'target_agent': target_agent,
            'team': target_team,
        })

class DummySession:
    def __init__(self, plan_item, world: DummyWorld):
        self.plan = plan_item
        self.world = world
        self.time = 0.0
        self.actions = []

    def step(self, dt):
        # For the test we cause each agent to fire once at the start.
        if self.time == 0.0:
            # Fire a projectile at a dummy opponent.
            opponent = 'agentB' if self.plan['agent_cfg'].get('id') != 'agentB' else 'agentA'
            opponent_team = 'Blue' if self.plan['team'] == 'Red' else 'Red'
            self.world.launch_projectile(
                shooter_agent=self.plan['agent_cfg'].get('id', 'unknown'),
                shooter_team=self.plan['team'],
                target_agent=opponent,
                target_team=opponent_team,
            )
        self.time += dt
        return []

# --------------------------------------------------------------------- tests

def test_plan_validation_success():
    cfg = {
        'teams': [
            {'name': 'Red', 'agents': [{'id': 'agentA'}]},
            {'name': 'Blue', 'agents': [{'id': 'agentB'}]},
        ],
        'rules': {'score_target': 1, 'duration_s': 5, 'friendly_fire': False},
        'arena': {'obstacles': [], 'cover': []},
    }
    m = ArenaMatch(cfg)
    # Should not raise.
    m.plan()


def test_plan_validation_failure():
    cfg = {'teams': []}
    m = ArenaMatch(cfg)
    with pytest.raises(ValueError):
        m.plan()


def _run_match(friendly_fire: bool, return_match: bool = False):
    cfg = {
        'teams': [
            {'name': 'Red', 'agents': [{'id': 'agentA'}]},
            {'name': 'Blue', 'agents': [{'id': 'agentB'}]},
        ],
        'rules': {'score_target': 2, 'duration_s': 1.0, 'friendly_fire': friendly_fire},
        'arena': {},
    }
    match = ArenaMatch(cfg)
    match.plan()
    # Prepare world and hook scorer.
    world = DummyWorld({})
    world.scorer = match._scorer
    def world_factory(_cfg):
        return world
    def launcher(plan_item):
        return DummySession(plan_item, world)
    match.play(world_factory, launcher)
    if return_match:
        return match
    return match.result()

def test_victory_by_score():
    res = _run_match(friendly_fire=False)
    # Each side fires once; score_target=2, so no victory – timeout should be true.
    assert res.timeout is True
    assert res.winner is None

def test_friendly_fire_enabled_allows_self_score():
    cfg = {
        'teams': [
            {'name': 'Red', 'agents': [{'id': 'agentA'}]},
        ],
        'rules': {'score_target': 1, 'duration_s': 5, 'friendly_fire': True},
        'arena': {},
    }
    m = ArenaMatch(cfg)
    m.plan()
    world = DummyWorld({})
    world.scorer = m._scorer
    def wf(_c):
        return world
    def launch(plan):
        # Agent shoots at itself.
        sess = DummySession(plan, world)
        # Override step to fire at self.
        def step(dt):
            if sess.time == 0.0:
                world.launch_projectile('agentA', 'Red', 'agentA', 'Red')
            sess.time += dt
            return []
        sess.step = step
        return sess
    m.play(wf, launch)
    res = m.result()
    assert res.winner == 'Red'
    assert res.team_stats['Red'] == 1

def test_agent_stats_recorded():
    cfg = {
        'teams': [
            {'name': 'Red', 'agents': [{'id': 'agentA'}]},
            {'name': 'Blue', 'agents': [{'id': 'agentB'}]},
        ],
        'rules': {'score_target': 5, 'duration_s': 0.5, 'friendly_fire': False},
        'arena': {},
    }
    m = ArenaMatch(cfg)
    m.plan()
    world = DummyWorld({})
    world.scorer = m._scorer
    def wf(_c):
        return world
    def launch(p):
        return DummySession(p, world)
    m.play(wf, launch)
    res = m.result()
    # Each agent should have one hit dealt and one hit taken (the opponent).
    assert res.agent_stats['agentA']['hits_dealt'] == 1
    assert res.agent_stats['agentA']['hits_taken'] == 1
    assert res.agent_stats['agentB']['hits_dealt'] == 1
    assert res.agent_stats['agentB']['hits_taken'] == 1

def test_export_match_report(tmp_path):
    from playground.arena import match as arena_mod
    cfg = {
        'teams': [
            {'name': 'Red', 'agents': [{'id': 'a1'}]},
            {'name': 'Blue', 'agents': [{'id': 'b1'}]},
        ],
        'rules': {'score_target': 1, 'duration_s': 0.1, 'friendly_fire': False},
        'arena': {},
    }
    m = arena_mod.ArenaMatch(cfg)
    m.plan()
    world = DummyWorld({})
    world.scorer = m._scorer
    def wf(_c):
        return world
    def launch(p):
        return DummySession(p, world)
    m.play(wf, launch)
    res = m.result()
    out_path = tmp_path / 'report.json'
    arena_mod.export_match_report(res, str(out_path))
    data = json.loads(out_path.read_text())
    assert data['duration'] == res.duration
    assert data['team_stats'] == res.team_stats
    assert data['winner'] == res.winner


class _QuietSession(DummySession):
    """A session that never fires: proves an eventless match really has an empty timeline."""

    def step(self, dt):
        self.time += dt
        return []


def test_timeline_records_events_in_encounter_order():
    m = _run_match(friendly_fire=False, return_match=True)
    tl = m.result().timeline
    # Both sessions fire once at t=0; the impacts are emitted on the next world step.
    assert len(tl) == 2
    assert all(e['type'] == 'projectile_impact' for e in tl)
    # Encounter order: agentA (first plan item) registered p0 before agentB's p1.
    assert tl[0]['proj_id'] == 'p0' and tl[1]['proj_id'] == 'p1'
    assert tl[0]['shooter_agent'] == 'agentA' and tl[0]['shooter_team'] == 'Red'
    assert tl[1]['shooter_agent'] == 'agentB' and tl[1]['shooter_team'] == 'Blue'
    assert tl[0]['target_agent'] == 'agentB' and tl[1]['target_agent'] == 'agentA'
    assert all(e['elapsed'] == pytest.approx(0.05) for e in tl)


def test_export_match_report_round_trips_timeline(tmp_path):
    m = _run_match(friendly_fire=False, return_match=True)
    res = m.result()
    out_path = tmp_path / 'report_timeline.json'
    from playground.arena.match import export_match_report
    export_match_report(res, str(out_path))
    data = json.loads(out_path.read_text())
    assert data['timeline'] == res.timeline
    assert data['timeline'][0]['proj_id'] == 'p0'


def test_timeline_empty_when_no_events():
    cfg = {
        'teams': [
            {'name': 'Red', 'agents': [{'id': 'agentA'}]},
            {'name': 'Blue', 'agents': [{'id': 'agentB'}]},
        ],
        'rules': {'score_target': 2, 'duration_s': 0.2, 'friendly_fire': False},
        'arena': {},
    }
    m = ArenaMatch(cfg)
    m.plan()
    world = DummyWorld({})
    world.scorer = m._scorer
    m.play(lambda _c: world, lambda plan: _QuietSession(plan, world))
    res = m.result()
    assert res.timeline == []  # genuinely no events -> empty timeline (not a placeholder)
    assert res.team_stats == {}


