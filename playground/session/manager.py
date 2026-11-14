"""Running many independent agents, sequentially or concurrently.

Agents never share anything: each has its own memory directory, its own sessions
and its own freshly built island. Concurrency is only a scheduling choice.
"""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..agents.adapters import Adapter
from ..config import agent_ids
from ..memory import MemoryStore
from .runner import memory_root, run_session


def ensure_agents(cfg: dict, db=None, ids: list[str] | None = None) -> list[str]:
    """Create the memory directory structure for every agent (idempotent)."""
    ids = ids or agent_ids(cfg)
    for a in ids:
        MemoryStore(memory_root(cfg), a, cfg, db=db)
    return ids


def run_agents(cfg: dict, db, adapter_factory: Callable[[str], Adapter], ids: list[str] | None = None,
               sessions: int = 1, concurrency: int = 1, label: str | None = None,
               progress: Callable[[dict], None] | None = None, max_continuations: int = 0) -> dict:
    """Run `sessions` exploration sessions for each agent. Returns per-agent summaries.

    If an agent ends a session with continue_next_session=true (it has unfinished experiments or open
    questions), up to `max_continuations` EXTRA sessions are granted beyond the base `sessions` count,
    so a curious agent can carry its research across sessions instead of each one being isolated. The
    cap keeps this bounded - no uncontrolled loop. Audit 2026-11-13.
    """
    ids = ensure_agents(cfg, db, ids)
    started = time.time()

    def one(agent_id: str) -> list[dict]:
        out = []
        granted_extra = 0
        remaining = sessions
        while remaining > 0:
            summary = run_session(copy.deepcopy(cfg), db, agent_id, adapter_factory(agent_id),
                                  mode="exploration", label=label)
            out.append(summary)
            if progress:
                progress(summary)
            if summary["end_reason"] in ("adapter_error", "runner_error", "interrupted"):
                break
            remaining -= 1
            # The agent asked to keep going and we have continuations left: grant one more session.
            if remaining == 0 and summary.get("continue_requested") and granted_extra < max_continuations:
                granted_extra += 1
                remaining = 1
        return out

    results: dict[str, list[dict]] = {}
    if concurrency <= 1:
        for a in ids:
            results[a] = one(a)
    else:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="agent") as pool:
            futures = {pool.submit(one, a): a for a in ids}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
    flat = [s for a in ids for s in results[a]]
    return {"agents": ids, "sessions_run": len(flat), "concurrency": concurrency,
            "wall_seconds": round(time.time() - started, 2),
            "continuations_granted": sum(1 for s in flat if s.get("continue_requested")),
            "completed": sum(1 for s in flat if s["end_reason"] not in ("adapter_error", "runner_error", "interrupted")),
            "errors": [{"agent_id": s["agent_id"], "session_id": s["session_id"], "end_reason": s["end_reason"],
                        "error": s["error"]} for s in flat if s["error"]],
            "by_agent": {a: [{"session_id": s["session_id"], "session_number": s["session_number"],
                              "end_reason": s["end_reason"], "actions": s["actions_logged"],
                              "continue_requested": s.get("continue_requested", False),
                              "memory": s["memory"]} for s in results[a]] for a in ids}}
