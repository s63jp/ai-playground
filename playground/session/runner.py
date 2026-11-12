"""Session lifecycle: brief the agent from its files, let it act, save, reset.

A session always starts from a freshly built island. The only thing carried from
one session to the next is the agent's memory directory.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from ..agents.adapters import Adapter, AdapterError, ToolCall
from ..agents.prompts import (BENCHMARK_SYSTEM, CONTINUE_NUDGE, DEFAULT_PERSONALITY, benchmark_opening,
                              exploration_opening, exploration_system, prompt_hash)
from ..analysis.behaviour import store_classification
from ..config import config_hash, resolve_path
from ..interface.environment import Environment
from ..interface.schema import SPECS
from ..memory import MemoryStore
from ..world.physics import PhysicsWorld

_canonical_cache: dict[str, str] = {}


def canonical_initial_hash(cfg: dict) -> str:
    """State hash of the untouched island for this configuration."""
    key = config_hash(cfg)
    if key not in _canonical_cache:
        _canonical_cache[key] = PhysicsWorld(cfg).state_hash()
    return _canonical_cache[key]


def memory_root(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["memory"]["root"])


def integrity_report(db, session_id: int) -> dict:
    """What the agent claimed, set against what was independently measured."""
    sess = db.session(session_id)
    recs = db.records(session_id=session_id)
    claims = [r["data"] for r in recs if r["kind"] == "claim"]
    exps = [r["data"] for r in recs if r["kind"] == "experiment"]
    flags = [{"action": r["action_idx"], **r["data"]} for r in recs if r["kind"] == "integrity_flag"]
    discs = [r["data"] for r in recs if r["kind"] == "discovery"]
    acts = db.actions(session_id)
    return {
        "session_id": session_id, "agent_id": sess["agent_id"],
        "event_chain": db.verify_chain(session_id),
        "what_the_agent_did": {"actions": len(acts), "failed_actions": sum(1 for a in acts if not a["success"]),
                               "physical_actions": sum(1 for a in acts if a["category"] in ("movement", "object"))},
        "what_the_agent_claims": {"achievement_claims": len(claims),
                                  "experiments_recorded": len(exps),
                                  "discoveries_requested_confirmed": sum(
                                      1 for d in discs if d.get("requested_status") == "confirmed")},
        "what_was_independently_verified": {
            "claims_verified": sum(1 for c in claims if c["verified"]),
            "claims_rejected": sum(1 for c in claims if not c["verified"]),
            "experiments_consistent_with_engine": sum(1 for e in exps if e["verification"]["verified"]),
            "experiments_contradicted_by_engine": sum(1 for e in exps if e["verification"]["verdict"] == "contradicted"),
            "experiments_without_valid_evidence": sum(
                1 for e in exps if e["verification"]["verdict"] in ("invalid_references", "no_evidence", "implausible")),
            "discoveries_granted_confirmed": sum(1 for d in discs if d.get("status") == "confirmed")},
        "flags": flags,
        "claims": claims,
    }


def _summarise(env: Environment, db, sid: int, end_reason: str, adapter: Adapter, turns: int,
               usage: dict, error: str | None) -> dict:
    behaviour = store_classification(db, sid)
    integ = integrity_report(db, sid)
    return {"session_id": sid, "agent_id": env.agent_id, "mode": env.mode, "session_number": env.session_number,
            "end_reason": end_reason, "error": error, "adapter": adapter.describe(), "turns": turns,
            "agent_calls": env.agent_calls, "actions_logged": env._idx, "sim_time": round(env.world.time, 3),
            "wall_seconds": round(env.elapsed(), 2), "counters": dict(env.counters), "token_usage": usage,
            "initial_state_hash": env.initial_state_hash, "final_state_hash": env.world.state_hash(),
            "integrity": {k: integ[k] for k in ("event_chain", "what_the_agent_did", "what_the_agent_claims",
                                                "what_was_independently_verified")},
            "integrity_flags": len(integ["flags"]),
            "behaviour": {k: behaviour[k]["count"] for k in ("self_directed_goals", "pretend_play",
                                                             "repetition_with_variation",
                                                             "activity_for_its_own_sake")},
            "memory": env.memory.stats()}


def run_session(cfg: dict, db, agent_id: str, adapter: Adapter, mode: str = "exploration",
                memory: MemoryStore | None = None, label: str | None = None,
                scene: Callable[[PhysicsWorld], None] | None = None, task_prompt: str | None = None,
                exclude_tools: list[str] | None = None, include_briefing: bool = True,
                extra: dict | None = None, clock: Callable[[], float] | None = None,
                resume_session_id: int | None = None, on_event: Callable[[str, dict], None] | None = None,
                on_finish: Callable[[Environment], dict] | None = None) -> dict:
    """Run one complete session and return its summary."""
    if memory is None:
        memory = MemoryStore(memory_root(cfg), agent_id, cfg, db=db)
    if resume_session_id is not None:
        # A resumed session keeps the budget it was started with, whatever the current CLI flags say.
        row0 = db.session(resume_session_id)
        if row0 is not None and row0.get("budget_json"):
            b = json.loads(row0["budget_json"])
            cfg = {**cfg, "session": {**cfg["session"], "max_steps": b.get("max_steps"),
                                      "duration_minutes": b.get("minutes", cfg["session"]["duration_minutes"])}}
    s_cfg = cfg["session"]
    tool_names = [n for n in SPECS if n not in (exclude_tools or [])]
    resumed = None
    if resume_session_id is not None:
        row = db.session(resume_session_id)
        if row is None or row["status"] == "completed":
            raise ValueError(f"session {resume_session_id} cannot be resumed (status: {row and row['status']})")
        sid, number = resume_session_id, row["session_number"]
        env = Environment(cfg, db, memory, agent_id, sid, number, mode=row["mode"], allowed_tools=tool_names,
                          scene=scene, clock=clock, wall_used=float(row["wall_seconds"] or 0.0))
        resumed = env.restore()
        db.update_session(sid, status="running")
    else:
        number = db.next_session_number(agent_id, mode)
        sid = db.create_session(agent_id=agent_id, mode=mode, session_number=number, label=label,
                                adapter=adapter.name, model=adapter.model, config_hash=config_hash(cfg),
                                config_json={k: cfg[k] for k in ("world", "agent_body", "rules", "session", "memory")},
                                memory_dir=str(memory.dir),
                                budget_json={"minutes": s_cfg["duration_minutes"], "max_steps": s_cfg.get("max_steps")},
                                **(extra or {}))
        env = Environment(cfg, db, memory, agent_id, sid, number, mode=mode, allowed_tools=tool_names,
                          scene=scene, clock=clock)
    briefing = memory.briefing() if include_briefing else None
    prompt_id = None
    if mode == "exploration":
        prompt_id = cfg.get("agents", {}).get("exploration_prompt") or DEFAULT_PERSONALITY
        system = exploration_system(prompt_id)
        opening = exploration_opening(agent_id, number, briefing, float(s_cfg["duration_minutes"]),
                                      s_cfg.get("max_steps"))
    else:
        system = BENCHMARK_SYSTEM
        opening = benchmark_opening(task_prompt or "", briefing, float(s_cfg["duration_minutes"]),
                                    s_cfg.get("max_steps"))
    if resumed:
        opening += ("\n\n(This session was interrupted and has been resumed. The island is exactly as you left "
                    f"it after action {resumed['resumed_at_action']}; observe it to re-orient yourself.)")
    env.note("session_started", mode=mode, session_number=number, adapter=adapter.describe(),
             memory_empty=memory.is_empty(), briefing_chars=len(briefing or ""),
             island_matches_original=env.initial_state_hash == canonical_initial_hash(cfg) if scene is None else None,
             resumed=bool(resumed), prompt_id=prompt_id, system_prompt_hash=prompt_hash(system))
    env._flush_only()
    db.add_message(sid, 0, "system", system)
    db.add_message(sid, 0, "user", opening)
    if on_event:
        on_event("session_started", {"session_id": sid, "agent_id": agent_id})

    turns, idle, usage, error = 0, 0, {}, None
    end_reason = None
    if adapter.driven_externally:
        return {"session_id": sid, "environment": env, "system": system, "opening": opening,
                "tools": tool_names, "memory": memory}
    try:
        adapter.begin(system, tool_names)
        user_text: str | None = opening
        results: list[tuple[ToolCall, dict]] = []
        while True:
            turn = adapter.step(user_text, results)
            turns += 1
            for k, v in (turn.usage or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            if turn.text and turn.text.strip():
                db.add_message(sid, turns, "assistant", turn.text.strip())
            user_text, results = None, []
            if not turn.tool_calls:
                if getattr(adapter, "_done", False):
                    end_reason = "agent_stopped"
                    break
                idle += 1
                if idle >= int(s_cfg["idle_turn_limit"]):
                    end_reason = "agent_stopped"
                    break
                if env.phase == "closed":
                    break
                user_text = CONTINUE_NUDGE
                db.add_message(sid, turns, "user", user_text)
                continue
            idle = 0
            for call in turn.tool_calls:
                res = env.call(call.name, call.args, source="agent", turn=turns)
                results.append((call, res))
            if env.phase == "closed":
                break
    except AdapterError as exc:
        error, end_reason = str(exc), "adapter_error"
    except KeyboardInterrupt:
        error, end_reason = "interrupted by operator", "interrupted"
    except Exception as exc:  # noqa: BLE001
        error, end_reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}", "runner_error"
    return finish_session(cfg, db, env, adapter, turns, usage,
                          end_reason if error else (env.end_reason or end_reason or "closed"), error, on_finish)


def finish_session(cfg: dict, db, env: Environment, adapter: Adapter, turns: int, usage: dict,
                   end_reason: str, error: str | None,
                   on_finish: Callable[[Environment], dict] | None = None) -> dict:
    """Save everything, snapshot the memory files and discard the world (the reset)."""
    sid = env.sid
    resumable = end_reason in ("adapter_error", "interrupted", "runner_error")
    
    # Auto-capture agent's last words when session ends without memory being written
    # This ensures agents that only produce text (no tool calls) still have their thoughts preserved
    if end_reason == "agent_stopped" and env.mode == "exploration" and not resumable:
        # Get the last assistant messages (their final thoughts before stopping)
        all_msgs = db.messages(sid)
        assistant_msgs = [m["content"] for m in all_msgs if m["role"] == "assistant"]
        last_messages = assistant_msgs[-3:] if len(assistant_msgs) >= 3 else assistant_msgs
        if last_messages:
            combined = " ".join(m.strip() for m in last_messages if m and m.strip())
            if combined and len(combined) > 20:  # Only capture if there's meaningful content
                # Truncate if too long
                if len(combined) > 500:
                    combined = combined[:497] + "..."
                auto_text = f"[auto-captured] Session ended ({end_reason}). Final thoughts: {combined}"
                env.memory.add_diary(env.session_number, env.world.time, auto_text)
    
    env.note("session_finished", end_reason=end_reason, error=error)
    env._flush_only()
    env.checkpoint()
    final_state = env.snapshot()
    task_result = None
    if on_finish is not None:
        # Independent measurement of the final world, before it is discarded.
        task_result = on_finish(env)
        env.note("task_verified", **{k: task_result[k] for k in ("task_id", "verified") if k in task_result})
        env._flush_only()
    if not resumable and env.mode == "exploration":
        env.memory.state["sessions_completed"] = max(env.memory.state["sessions_completed"], env.session_number)
    summary = _summarise(env, db, sid, end_reason, adapter, turns, usage, error)
    if task_result is not None:
        summary["task_result"] = task_result
    if not resumable:
        dest = env.memory.close_session(env.session_number, final_state, summary) if env.mode == "exploration" \
            else None
        summary["memory_snapshot"] = str(dest) if dest else None
        # The reset: this world is discarded, and the next session builds the island from its specification.
        summary["island_reset"] = {"next_session_starts_from": canonical_initial_hash(cfg),
                                   "this_session_started_from": env.initial_state_hash}
    db.update_session(sid, status="interrupted" if resumable else "completed", end_reason=end_reason,
                      ended_at=time.time(), sim_time=round(env.world.time, 4), physics_steps=env.world.steps,
                      action_count=env._idx, wall_seconds=round(env.elapsed(), 2),
                      final_state_json=final_state, summary_json=summary)
    return summary


def session_summary(db, sid: int) -> dict | None:
    row = db.session(sid)
    if not row:
        return None
    for k in ("summary_json", "final_state_json", "config_json", "budget_json"):
        row[k[:-5]] = json.loads(row.pop(k)) if row.get(k) else None
    return row
