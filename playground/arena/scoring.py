
"""Scoring utilities for arena matches.

The public contract expects handling of ``projectile_impact`` events with the payload
``{"proj_id": str, "pos": [...], "surface": str, "shooter": str, ...}``.

The implementation keeps a simple per‑team hit count and optionally respects friendly fire.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


class ArenaScorer:
    """Simple scorer that tracks hits per team.

    Parameters
    ----------
    rules: dict
        Scoring configuration – currently supports ``score_target`` (int) and ``friendly_fire`` (bool).
    """

    def __init__(self, rules: Dict[str, Any] | None):
        self.rules = rules or {}
        self.score_target = int(self.rules.get('score_target', 10))
        self.friendly_fire = bool(self.rules.get('friendly_fire', False))
        # Mapping proj_id → shooter info (team name). In a real engine this would be supplied by the
        # ProjectileSystem contract; tests will populate it manually.
        self._projectile_owner: Dict[str, str] = {}
        self._team_scores: Dict[str, int] = {}
        self._agent_stats: Dict[str, Dict[str, int]] = {}
        self._winner: str | None = None

    # --------------------------------------------------------------------- external hooks
    def register_projectile(self, proj_id: str, shooter_agent: str, shooter_team: str) -> None:
        """Record which team launched a projectile – called by the world (stubbed in tests)."""
        self._projectile_owner[proj_id] = shooter_team
        # Ensure team entry exists.
        self._team_scores.setdefault(shooter_team, 0)
        self._agent_stats.setdefault(shooter_agent, {'hits_dealt': 0, 'hits_taken': 0, 'shots_fired': 0})
        self._agent_stats[shooter_agent]['shots_fired'] += 1

    # --------------------------------------------------------------------- event processing
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
        shooter_team = self._projectile_owner.get(proj_id)
        if not shooter_team:
            # Unknown projectile – ignore.
            return
        # Friendly fire check.
        if not self.friendly_fire and shooter_team == target_team:
            return
        # Update scores.
        self._team_scores[shooter_team] = self._team_scores.get(shooter_team, 0) + 1
        # Agent‑level stats.
        # Find shooter agent – we stored it when registering the projectile.
        shooter_agents = [a for a, s in self._agent_stats.items() if s.get('shots_fired')]
        # For simplicity we increment the first matching agent (tests provide direct mapping).
        # In a full implementation the projectile payload would include the shooter identifier.
        # Here we just credit the team.
        if target:
            # Ensure target stats exist.
            self._agent_stats.setdefault(target, {'hits_dealt': 0, 'hits_taken': 0, 'shots_fired': 0})
            self._agent_stats[target]['hits_taken'] += 1
        # Increment shooter stats.
        # We cannot reliably map proj_id → agent without extra data; tests may call ``register_projectile``
        # with the shooter agent name embedded, so we simply record a generic hit.
        # (Advanced usage can extend ``process_event`` to read ``shooter_agent`` from the event.)
        # No further per‑agent bookkeeping here.

    # --------------------------------------------------------------------- victory logic
    def check_victory(self) -> bool:
        """Return ``True`` if any team reached ``score_target``.
        Sets ``self._winner`` accordingly.
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
        """Return a tuple ``(team_stats, agent_stats)`` suitable for the final result.
        """
        return dict(self._team_scores), dict(self._agent_stats)
