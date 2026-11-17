
"""Adapter protocol and world capability wrapper for the playground.
Only minimal logic is included; missing capabilities are reported via
`CapabilityUnavailable` errors that callers translate to an "unavailable"
status.
"""
from __future__ import annotations
import typing as t

class CapabilityUnavailable(RuntimeError):
    """Raised when a requested capability is not present in the world."""

class CapabilityAdapter(t.Protocol):
    """Protocol defining all skill‑level capabilities.
    Implementations must return a dict with at least a ``status`` key.
    """

    def set_gait(self, mode: str, speed: float = 1.0, heading: t.Optional[float] = None) -> dict: ...
    def stand_up(self) -> dict: ...
    def agent_fallen(self) -> dict: ...
    def reach(self, object_id: str) -> dict: ...
    def grasp(self, object_id: str) -> dict: ...
    def release(self) -> dict: ...
    def pick_up(self, object_id: str) -> dict: ...
    def throw(self, object_id: str, direction: t.Sequence[float], force: float) -> dict: ...


class WorldCapabilities:
    """Wraps a physics world and optional manipulation controller.
    The wrapper exposes only the capabilities that are actually present.
    """

    def __init__(self, world: t.Any, manipulation: t.Optional[t.Any] = None):
        self.world = world
        self.manipulation = manipulation
        # Detect which methods are available on the underlying objects
        self._has_gait = hasattr(world, "set_gait")
        self._has_stand = hasattr(world, "stand_up")
        self._has_fallen = hasattr(world, "agent_fallen")
        self._has_reach = manipulation is not None and hasattr(manipulation, "reach")
        self._has_grasp = manipulation is not None and hasattr(manipulation, "grasp")
        self._has_release = manipulation is not None and hasattr(manipulation, "release")
        self._has_pick_up = manipulation is not None and hasattr(manipulation, "pick_up")
        self._has_throw = manipulation is not None and hasattr(manipulation, "throw")

    # Helper to return unavailable dict uniformly
    def _unavailable(self, name: str) -> dict:
        return {"status": "unavailable", "detail": f"Capability '{name}' not present in world"}

    def set_gait(self, mode: str, speed: float = 1.0, heading: t.Optional[float] = None) -> dict:
        if not self._has_gait:
            return self._unavailable("set_gait")
        return self.world.set_gait(mode, speed, heading)

    def stand_up(self) -> dict:
        if not self._has_stand:
            return self._unavailable("stand_up")
        return self.world.stand_up()

    def agent_fallen(self) -> dict:
        if not self._has_fallen:
            return self._unavailable("agent_fallen")
        return self.world.agent_fallen()

    def reach(self, object_id: str) -> dict:
        if not self._has_reach:
            return self._unavailable("reach")
        return self.manipulation.reach(object_id)

    def grasp(self, object_id: str) -> dict:
        if not self._has_grasp:
            return self._unavailable("grasp")
        return self.manipulation.grasp(object_id)

    def release(self) -> dict:
        if not self._has_release:
            return self._unavailable("release")
        return self.manipulation.release()

    def pick_up(self, object_id: str) -> dict:
        if not self._has_pick_up:
            return self._unavailable("pick_up")
        return self.manipulation.pick_up(object_id)

    def throw(self, object_id: str, direction: t.Sequence[float], force: float) -> dict:
        if not self._has_throw:
            return self._unavailable("throw")
        return self.manipulation.throw(object_id, direction, force)
