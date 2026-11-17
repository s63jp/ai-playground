
"""Arena match orchestration.

Provides :class:`ArenaMatch` which validates a match configuration, builds a world via a
factory, launches agent sessions via a launcher, steps the simulation and computes scores.

It does not depend on the MuJoCo engine directly: it works with *callables* supplied by the
caller for the world and the sessions (the same surface the production factories implement;
the unit tests inject fakes).
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List

from . import scoring, modes


class MatchResult(dict):
    """Result dictionary produced by :meth:`ArenaMatch.result`.

    It is a plain ``dict`` subclass so callers can access keys as attributes if they wish.
    Expected keys:
        - ``team_stats``: mapping team name → stats dict
        - ``agent_stats``: mapping agent id → stats dict
        - ``winner``: team name or ``None`` if timeout/no victor
        - ``duration``: elapsed seconds
        - ``timeout``: bool indicating match ended by time limit
        - ``timeline``: ordered list of recorded match events (impacts with elapsed time)
    """

    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
        self.__dict__.update(data)


class ArenaMatch:
    """Controller for a single arena match.

    Parameters
    ----------
    cfg: dict
        Full match configuration (see specification in the task description).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._world = None
        self._session_plan = None
        self._scorer = scoring.ArenaScorer(cfg.get('rules', {}))
        self._elapsed = 0.0
        self._winner: str | None = None
        self._timeout = False
        # Timeline of events with timestamps and shooter info.
        self._timeline: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------- validation & planning
    def plan(self) -> None:
        """Validate the configuration and prepare world/agent plan.

        Populates ``self._world_cfg`` and ``self._session_plan`` which are later passed to the
        injected factories.
        """
        # Basic structural validation – raise ``ValueError`` with a clear message on failure.
        if not isinstance(self.cfg, dict):
            raise ValueError('config must be a dict')
        # Teams
        teams = self.cfg.get('teams')
        if not isinstance(teams, list) or not teams:
            raise ValueError('config must contain a non‑empty list of teams')
        for t in teams:
            if not isinstance(t, dict) or 'name' not in t or 'agents' not in t:
                raise ValueError('each team requires a name and agents list')
            if not isinstance(t['agents'], list) or not t['agents']:
                raise ValueError('team agents must be a non‑empty list')
        # Rules
        rules = self.cfg.get('rules', {})
        if not isinstance(rules, dict):
            raise ValueError('rules must be a dict')
        # Arena geometry – we simply store it; deeper validation is out of scope.
        self._world_cfg = modes.build_world_cfg(self.cfg.get('arena', {}))
        # Build a flat session plan – each entry is ``{'team': team_name, 'agent_cfg': cfg}``
        self._session_plan = []
        for team in teams:
            for ag in team['agents']:
                plan = {
                    'team': team['name'],
                    'agent_cfg': ag,
                    'equipment': team.get('equipment', []),
                }
                self._session_plan.append(plan)
        # Nothing else to do – the heavy lifting (world creation, session launch) is delegated.

    # --------------------------------------------------------------------- execution
    def _consume_events(self, events: List[Dict[str, Any]], at: float) -> None:
        """Feed world/session events into the scorer and the match timeline."""
        for e in events:
            entry = dict(e)
            owner = self._scorer.owner_of(e.get('proj_id'))
            if owner:
                entry['shooter_agent'], entry['shooter_team'] = owner
            entry['elapsed'] = at
            self._timeline.append(entry)
            self._scorer.process_event(e)

    def play(self, world_factory: Callable[[Dict[str, Any]], Any],
             session_launcher: Callable[[Dict[str, Any]], Any]) -> None:
        """Run the match.

        ``world_factory`` receives the world configuration dictionary (from ``plan``) and must
        return an object exposing ``step(dt)`` and ``event_stream()``.
        ``session_launcher`` receives a plan item (as produced by ``plan``) and must return a *session*
        object exposing ``advance(dt)`` (or ``step``) and ``final_stats()``.
        The orchestration loops until a victory condition or duration limit is reached.
        """
        if self._world_cfg is None or self._session_plan is None:
            raise RuntimeError('plan() must be called before play()')
        # Create the world.
        self._world = world_factory(self._world_cfg)
        # Launch sessions.
        self._sessions = [session_launcher(p) for p in self._session_plan]
        duration = float(self.cfg.get('rules', {}).get('duration_s', 60))
        time_step = 0.05  # 20 Hz simulation – fine enough for scoring.
        elapsed = 0.0
        # Main loop.
        while elapsed < duration:
            # Step world – collect any events for scoring.
            events = self._world.step(time_step) if hasattr(self._world, 'step') else []
            self._consume_events(events, elapsed)
            # Advance each session.
            for sess in self._sessions:
                sess_events = sess.step(time_step) if hasattr(sess, 'step') else []
                self._consume_events(sess_events, elapsed)

            elapsed += time_step
            # Victory check.
            if self._scorer.check_victory():
                break
        self._elapsed = elapsed
        self._timeout = elapsed >= duration
        self._winner = self._scorer.winner()

    # --------------------------------------------------------------------- incremental API
    def step(self, dt: float) -> None:
        """Advance the match by ``dt`` seconds – useful for incremental integration.
        Returns ``None``; after each call one may inspect ``self.result()``.
        """
        if self._world is None:
            raise RuntimeError('play() or world_factory not yet invoked')
        events = self._world.step(dt) if hasattr(self._world, 'step') else []
        self._consume_events(events, self._elapsed + dt)
        for sess in self._sessions:
            sess_events = sess.step(dt) if hasattr(sess, 'step') else []
            self._consume_events(sess_events, self._elapsed + dt)
        self._elapsed += dt
        if self._scorer.check_victory():
            self._winner = self._scorer.winner()

    def score(self) -> Dict[str, Any]:
        """Return the current scoring snapshot – useful for UI updates.
        """
        return self._scorer.snapshot()

    def result(self) -> MatchResult:
        """Return the final match result dictionary.
        Must be called after the match has finished (either by ``play`` or by external stepping).
        """
        if self._world is None:
            raise RuntimeError('match not yet executed')
        # Gather per‑team and per‑agent stats from scorer.
        team_stats, agent_stats = self._scorer.tally()
        data = {
            'team_stats': team_stats,
            'agent_stats': agent_stats,
            'winner': self._winner,
            'duration': self._elapsed,
            'timeout': self._timeout,
            'timeline': self._timeline,
        }
        return MatchResult(data)

def export_match_report(result: MatchResult, path: str) -> None:
    """Write a JSON file with the full match result.

    The format mirrors the keys of :class:`MatchResult`, including the recorded match-event
    ``timeline`` (an empty list only when no events actually occurred).
    """
    import json, os

    def _res_get(key, default=None):
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)
    data = {
        "end_reason": _res_get("end_reason", None),
        "sessions": _res_get("sessions", []),
        'team_stats': result.team_stats,
        'agent_stats': result.agent_stats,
        'winner': result.winner,
        'duration': result.duration,
        'timeout': result.timeout,
        'timeline': getattr(result, 'timeline', []),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2)

