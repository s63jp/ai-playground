"""Backend-enforced concurrency limits and a fair, prioritised request queue.

Stops one application or agent from monopolising the model servers. Every gateway request
must acquire admission across all applicable scopes — global, per-provider, per-model and
per-agent — before it is forwarded. When a scope is at its limit the request waits in a
queue ordered by priority then arrival (FIFO), and per-agent caps mean one agent's flood
cannot hold every slot while others starve. Enforcement is here in the request path, not in
the dashboard — the dashboard only displays what this enforces.

Disabled by default: with no limits configured, `enabled` is False and acquire() is a no-op
fast path, so the gateway behaves exactly as before Stage C.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


class QueueFull(Exception):
    """The configured queue is full; the request is rejected rather than queued (429)."""


class Cancelled(Exception):
    """The request was cancelled while waiting in the queue."""


@dataclass
class _Waiter:
    seq: int
    priority: int
    agent: str
    provider: str
    model: str
    scopes: list
    cancelled: bool = False


@dataclass
class Admission:
    scopes: list = field(default_factory=list)


class ResourceManager:
    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.g = int(c.get("max_concurrent", 0) or 0)
        self.provider_default = int(c.get("provider_default", 0) or 0)
        self.model_default = int(c.get("model_default", 0) or 0)
        self.agent_default = int(c.get("agent_default", 0) or 0)
        self.per_provider = dict(c.get("per_provider") or {})
        self.per_model = dict(c.get("per_model") or {})
        self.per_agent = dict(c.get("per_agent") or {})
        self.queue_max = int(c.get("queue_max", 0) or 0)
        self.default_priority = int(c.get("default_priority", 5))
        self._cv = threading.Condition()
        self._active: dict[str, int] = defaultdict(int)
        self._waiters: list[_Waiter] = []
        self._seq = 0

    # ------------------------------------------------------------ limits
    def _limit(self, scope: str) -> int:
        kind, _, name = scope.partition(":")
        if kind == "global":
            return self.g
        if kind == "provider":
            return int(self.per_provider.get(name, self.provider_default) or 0)
        if kind == "model":
            return int(self.per_model.get(name, self.model_default) or 0)
        if kind == "agent":
            return int(self.per_agent.get(name, self.agent_default) or 0)
        return 0

    def _scopes_for(self, provider: str, model: str, agent: str) -> list[str]:
        # Only scopes with a positive limit constrain the request; the rest are unlimited.
        candidate = ["global", f"provider:{provider}", f"model:{model}", f"agent:{agent}"]
        return [s for s in candidate if self._limit(s) > 0]

    @property
    def enabled(self) -> bool:
        return bool(self.g or self.provider_default or self.model_default or self.agent_default
                    or self.per_provider or self.per_model or self.per_agent)

    def _has_capacity(self, scopes) -> bool:
        return all(self._active[s] < self._limit(s) for s in scopes)

    def _fair_order(self) -> list[_Waiter]:
        # Higher priority first, then FIFO by arrival. Monopoly is prevented by per-agent caps
        # (an agent at its limit simply cannot be admitted, so capacity flows to others).
        return sorted(self._waiters, key=lambda w: (-w.priority, w.seq))

    def _next_admittable(self) -> _Waiter | None:
        for w in self._fair_order():
            if not w.cancelled and self._has_capacity(w.scopes):
                return w
        return None

    # ------------------------------------------------------------ admission
    def acquire(self, provider: str, model: str, agent: str, priority: int | None = None,
                is_cancelled=None, poll: float = 0.1) -> Admission:
        """Block until admitted across every applicable scope, then return an Admission to be
        released later. Raises QueueFull if the queue is at capacity, Cancelled if is_cancelled()
        becomes true while waiting."""
        scopes = self._scopes_for(provider, model, agent)
        if not scopes:
            return Admission(scopes=[])        # unconstrained: no queue, no accounting
        pr = self.default_priority if priority is None else int(priority)
        with self._cv:
            if self.queue_max and len(self._waiters) >= self.queue_max:
                raise QueueFull()
            self._seq += 1
            w = _Waiter(self._seq, pr, agent, provider, model, scopes)
            self._waiters.append(w)
            self._cv.notify_all()
            try:
                while True:
                    if is_cancelled and is_cancelled():
                        raise Cancelled()
                    if self._next_admittable() is w:
                        for s in scopes:
                            self._active[s] += 1
                        self._waiters.remove(w)
                        self._cv.notify_all()
                        return Admission(scopes=scopes)
                    self._cv.wait(timeout=poll)    # wake on release/arrival, and poll for cancellation
            except BaseException:
                if w in self._waiters:
                    self._waiters.remove(w)
                self._cv.notify_all()
                raise

    def release(self, adm: Admission) -> None:
        if not adm or not adm.scopes:
            return
        with self._cv:
            for s in adm.scopes:
                self._active[s] = max(0, self._active[s] - 1)
            self._cv.notify_all()

    # ------------------------------------------------------------ introspection / config
    def snapshot(self) -> dict:
        with self._cv:
            return {
                "enabled": self.enabled,
                "limits": self._limits_dict(),
                "active": {k: v for k, v in self._active.items() if v},
                "active_total": sum(v for k, v in self._active.items() if k == "global") or None,
                "queue_depth": len(self._waiters),
                "queued": [{"seq": w.seq, "priority": w.priority, "agent": w.agent,
                            "provider": w.provider, "model": w.model} for w in self._fair_order()],
            }

    def _limits_dict(self) -> dict:
        return {"max_concurrent": self.g, "provider_default": self.provider_default,
                "model_default": self.model_default, "agent_default": self.agent_default,
                "per_provider": dict(self.per_provider), "per_model": dict(self.per_model),
                "per_agent": dict(self.per_agent), "queue_max": self.queue_max,
                "default_priority": self.default_priority}

    def update_limits(self, patch: dict) -> dict:
        """Change limits live (dashboard control). Returns the new limits. Waiters are re-evaluated."""
        with self._cv:
            for k in ("max_concurrent", "provider_default", "model_default", "agent_default",
                      "queue_max", "default_priority"):
                if k in patch and patch[k] is not None:
                    setattr(self, {"max_concurrent": "g"}.get(k, k), int(patch[k]))
            for k, attr in (("per_provider", "per_provider"), ("per_model", "per_model"),
                            ("per_agent", "per_agent")):
                if isinstance(patch.get(k), dict):
                    setattr(self, attr, dict(patch[k]))
            self._cv.notify_all()
            return self._limits_dict()
