"""Model comparison under equivalent conditions.

Audit 2026-11-13 found the 2B/4B/9B runs were not comparable: different context windows, different
agents, different accumulated memory, and wildly different action counts (the 9B produced 32 actions
total, every session ending with "model unloaded"). This module makes a fair comparison possible.

Two parts:
  * session_profile(db, sid): a behavioural profile of ONE session, computed only from the recorded log
    (never from anything an agent claimed). These are the quantities section 8 of the brief asks about:
    behavioural diversity, exploration quality, objective generation, experimentation, memory use, tool
    selection and session continuation.
  * compare_models(...): runs the SAME number of FRESH agents for each model, with the SAME pinned
    context window, action cap and starting memory (empty), then aggregates the profiles per model.

"Fair" is the whole point: a faster model is not assumed to be a better explorer. The harness holds
every condition equal except the model, so any difference in the profiles is attributable to the model.
"""
from __future__ import annotations

import copy
import statistics
from typing import Callable

from ..agents.adapters import Adapter
from .manager import ensure_agents
from .runner import run_session

# Tool categories that count as "observation" (read-only) vs "physical" (world-changing), mirroring the
# interface schema so the profile does not drift from it.
OBSERVATION = ("observation",)
PHYSICAL = ("movement", "object")


def session_profile(db, sid: int) -> dict:
    """A behavioural profile of one session, from the log alone."""
    sess = db.session(sid)
    actions = db.actions(sid)
    recs = db.records(session_id=sid)
    n = len(actions)

    by_tool: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    failed = 0
    for a in actions:
        by_tool[a["tool"]] = by_tool.get(a["tool"], 0) + 1
        by_cat[a["category"]] = by_cat.get(a["category"], 0) + 1
        if not a["success"]:
            failed += 1

    kinds: dict[str, int] = {}
    confirmed = 0
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        if r["kind"] == "discovery" and (r["data"] or {}).get("status") == "confirmed":
            confirmed += 1
    goals_set = sum(1 for r in recs if r["kind"] == "goal" and (r["data"] or {}).get("event") == "set")

    # Budget used: actions against the session's step cap, or wall against the minutes budget.
    budget = sess.get("budget_json")
    import json
    cap = None
    if budget:
        try:
            cap = json.loads(budget).get("max_steps")
        except (TypeError, ValueError):
            cap = None
    budget_used = round(n / cap, 3) if cap else None

    obs = sum(by_cat.get(c, 0) for c in OBSERVATION)
    phys = sum(by_cat.get(c, 0) for c in PHYSICAL)
    summary = {}
    if sess.get("summary_json"):
        try:
            summary = json.loads(sess["summary_json"])
        except (TypeError, ValueError):
            summary = {}

    return {
        "session_id": sid,
        "agent_id": sess["agent_id"],
        "model": sess.get("model"),
        "adapter": sess.get("adapter"),
        "end_reason": sess.get("end_reason"),
        "actions": n,
        "distinct_tools": len(by_tool),                      # behavioural diversity
        "throw_share": round(by_tool.get("throw", 0) / n, 3) if n else 0.0,
        "observation_share": round(obs / n, 3) if n else 0.0,  # exploration vs. looping
        "physical_share": round(phys / n, 3) if n else 0.0,
        "failed_share": round(failed / n, 3) if n else 0.0,
        "goals_set": goals_set,                              # objective generation
        "experiments": kinds.get("experiment", 0),           # experimentation
        "discoveries": kinds.get("discovery", 0),
        "confirmed_discoveries": confirmed,
        "mistakes": kinds.get("mistake", 0),
        "memory_writes": sum(kinds.get(k, 0) for k in ("diary", "experiment", "discovery", "mistake", "goal")),
        "budget_used": budget_used,                          # did it use its time?
        "continue_requested": bool(summary.get("continue_requested")),  # session continuation
        "tool_histogram": dict(sorted(by_tool.items(), key=lambda kv: -kv[1])),
    }


def _agg(profiles: list[dict]) -> dict:
    """Mean/median of the numeric profile fields across sessions of one model."""
    if not profiles:
        return {}
    numeric = ["actions", "distinct_tools", "throw_share", "observation_share", "physical_share",
               "failed_share", "goals_set", "experiments", "discoveries", "confirmed_discoveries",
               "mistakes", "memory_writes"]
    out = {"sessions": len(profiles)}
    for k in numeric:
        vals = [p[k] for p in profiles if isinstance(p.get(k), (int, float))]
        if vals:
            out[f"{k}_mean"] = round(statistics.mean(vals), 3)
            out[f"{k}_median"] = round(statistics.median(vals), 3)
    bu = [p["budget_used"] for p in profiles if isinstance(p.get("budget_used"), (int, float))]
    if bu:
        out["budget_used_mean"] = round(statistics.mean(bu), 3)
    out["continued"] = sum(1 for p in profiles if p.get("continue_requested"))
    out["errored"] = sum(1 for p in profiles if p.get("end_reason") in ("adapter_error", "runner_error"))
    return out


def compare_models(cfg: dict, db, adapter_for: Callable[[str, str], Adapter], models: list[str],
                   agents_per_model: int = 3, sessions: int = 1, context_tokens: int | None = None,
                   max_steps: int | None = 60, agent_prefix: str = "cmp",
                   progress: Callable[[dict], None] | None = None) -> dict:
    """Run every model under identical conditions and return per-model aggregated profiles.

    Equivalent conditions enforced here:
      * same number of FRESH agents per model (ids are model-scoped and start with empty memory);
      * same pinned max_steps (action cap) so no model simply gets more turns;
      * same context_tokens budget (if given) so no model gets a larger window;
      * same world (every session builds the untouched island).

    adapter_for(model, agent_id) builds the adapter for one model+agent. The caller injects how a model
    name maps to an adapter so this stays provider-neutral.
    """
    base = copy.deepcopy(cfg)
    base.setdefault("session", {})
    if max_steps is not None:
        base["session"]["max_steps"] = max_steps
    results: dict[str, dict] = {}
    all_profiles: dict[str, list[dict]] = {}
    for model in models:
        # Fresh, model-scoped agent ids so no accumulated or cross-model memory leaks in.
        ids = [f"{agent_prefix}_{_slug(model)}_{i:02d}" for i in range(agents_per_model)]
        ensure_agents(base, db, ids)
        profiles: list[dict] = []
        for aid in ids:
            for _ in range(sessions):
                ad = adapter_for(model, aid)
                if context_tokens is not None and hasattr(ad, "context_tokens"):
                    ad.context_tokens = context_tokens
                summary = run_session(copy.deepcopy(base), db, aid, ad, mode="exploration")
                prof = session_profile(db, summary["session_id"])
                prof["model"] = model
                profiles.append(prof)
                if progress:
                    progress(prof)
        all_profiles[model] = profiles
        results[model] = _agg(profiles)
    return {"conditions": {"agents_per_model": agents_per_model, "sessions": sessions,
                           "max_steps": max_steps, "context_tokens": context_tokens},
            "by_model": results, "profiles": all_profiles}


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())[:16] or "model"
