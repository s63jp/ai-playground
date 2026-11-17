
"""Scoring utilities for arena matches.

The public contract expects handling of ``projectile_impact`` events with the payload
``{"proj_id": str, "pos": [...], "surface": str, "shooter": str, ...}``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


class ArenaScorer:
    """Simple scorer that tracks hits per team and per agent.

    Parameters
    ----------
    rules: dict
        Scoring configuration – currently supports ``score_target`` (int) and ``friendly_fire`` (bool).
    """

    def __init__(self, rules: Dict[str, Any] | None):
        self.rules = rules or {}
        self.score_target = int(self.rules.get('score_target', 10))
        self.friendly_fire = bool(self.rules.get('friendly_fire', False))
        # Mapping proj_id -> (shooter_agent, shooter_team).
        self._projectile_owner: Dict[str, Tuple[str, str]] = {}
        self._team_scores: Dict[str, int] = {}
        self._agent_stats: Dict[str, Dict[str, int]] = {}
        self._winner: str | None = None

    # --------------------------------------------------------------------- external hooks
    def register_projectile(self, proj_id: str, shooter_agent: str, shooter_team: str) -> None:
        """Record which agent and team launched a projectile.

        The scorer now maintains a full mapping from ``proj_id`` to the shooter identity, enabling
        per-agent credit for hits.
        """
        self._projectile_owner[proj_id] = (shooter_agent, shooter_team)
        self._team_scores.setdefault(shooter_team, 0)
        self._agent_stats.setdefault(shooter_agent, {'hits_dealt': 0, 'hits_taken': 0, 'shots_fired': 0})
        self._agent_stats[shooter_agent]['shots_fired'] += 1

    # --------------------------------------------------------------------- event processing
    def owner_of(self, proj_id: str) -> Tuple[str, str] | None:
        """The registered ``(shooter_agent, shooter_team)`` for a projectile, or ``None``."""
        return self._projectile_owner.get(proj_id)

    def process_event(self, ev: Dict[str, Any]) -> None:
        """Consume a raw event dictionary.

        Expected format for a hit event::

            {"type": "projectile_impact", "proj_id": "p1", "target_agent": "a2", "team": "Red"}
        """
        if not isinstance(ev, dict):
            return
        etype = ev.get('type') or ev.get('event_type') or ev.get('type_name')
        if etype != 'projectile_impact':
            return
        proj_id = ev.get('proj_id')
        target = ev.get('target_agent')
        target_team = ev.get('team')
        shooter_info = self._projectile_owner.get(proj_id)
        if not shooter_info:
            return
        shooter_agent, shooter_team = shooter_info
        if not self.friendly_fire and shooter_team == target_team:
            return
        self._team_scores[shooter_team] = self._team_scores.get(shooter_team, 0) + 1
        shooter_stats = self._agent_stats.setdefault(shooter_agent, {'hits_dealt': 0, 'hits_taken': 0, 'shots_fired': 0})
        shooter_stats['hits_dealt'] += 1
        if target:
            target_stats = self._agent_stats.setdefault(target, {'hits_dealt': 0, 'hits_taken': 0, 'shots_fired': 0})
            target_stats['hits_taken'] += 1

    # --------------------------------------------------------------------- victory logic
    def check_victory(self) -> bool:
        """Return True if any team reached score_target. Sets _winner.
        """
        for team, sc in self._team_scores.items():
            if sc >= self.score_target:
                self._winner = team
                return True
        return False

    def winner(self) -> str | None:
        return self._winner

    # --------------------------------------------------------------------- snapshots
    def snapshot(self) -> Dict[str, Any]:
        return {'team_scores': dict(self._team_scores), 'friendly_fire': self.friendly_fire}

    def tally(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (team_stats, agent_stats) for final result.
        """
        return dict(self._team_scores), dict(self._agent_stats)




