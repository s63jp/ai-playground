"""Session lifecycle: brief the agent from its files, let it act, save, reset.

A session always starts from a freshly built island. The only thing carried from
one session to the next is the agent's memory directory.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from ..agents.adapters import Adapter, AdapterError, ToolCall
from ..agents.prompts import (BENCHMARK_SYSTEM, DEFAULT_PERSONALITY, REVIEW_BEFORE_STOP,
                              benchmark_opening, continue_nudge, exploration_opening, exploration_system,
                              prompt_hash)
from ..analysis.behaviour import store_classification
from ..config import config_hash, resolve_path
from ..interface.environment import Environment
from ..interface.schema import SPECS
from ..memory import MemoryStore
from ..world.physics import PhysicsWorld

_canonical_cache: dict[str, str] = {}

# Per-session stop signals. Any thread running a session can be asked to stop cleanly; the main loop
# checks this flag between turns. External sessions are stopped by finishing them directly.
_session_stop_events: dict[int, threading.Event] = {}


def request_stop_session(sid: int) -> bool:
    """Ask the running session `sid` to stop at the next safe point. Returns True if the session was known."""
    ev = _session_stop_events.get(sid)
    if ev is not None:
        ev.set()
        return True
    return False


def is_session_stop_requested(sid: int) -> bool:
    return _session_stop_events.get(sid, threading.Event()).is_set()


def _register_session_stop(sid: int) -> threading.Event:
    ev = threading.Event()
    _session_stop_events[sid] = ev
    return ev


def _unregister_session_stop(sid: int) -> None:
    _session_stop_events.pop(sid, None)


def canonical_initial_hash(cfg: dict) -> str:
    """State hash of the untouched island for this configuration."""
    key = config_hash(cfg)
    if key not in _canonical_cache:
        _canonical_cache[key] = PhysicsWorld(cfg).state_hash()
    return _canonical_cache[key]


def memory_root(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["memory"]["root"])


class ResumeAdapterMismatch(ValueError):
    """A session may only be resumed by the same brain that started it."""


def check_resume_adapter(row: dict, adapter: Adapter) -> None:
    """Refuse to resume a session with a different adapter or model than the one that started it.

    Audit 2026-11-13: session 19 (agent_003) was started by `lmstudio_unsloth4b`, hit an adapter error and
    was resumed with the CLI's default adapter (`scripted`). 70 of its 71 actions - and every throwing
    'discovery' in agent_003's memory - were therefore written by the scripted reference policy while the
    session row still named the language model. Resuming must keep the brain, or the record lies.
    """
    want_a, want_m = row.get("adapter"), row.get("model")
    got_a, got_m = getattr(adapter, "name", None), getattr(adapter, "model", None)
    if want_a and got_a and want_a != got_a:
        raise ResumeAdapterMismatch(
            f"session {row['id']} was started with adapter '{want_a}' (model '{want_m}'); it cannot be resumed "
            f"with adapter '{got_a}'. Pass --adapter {want_a}, or start a new session.")
    if want_m and got_m and want_m != got_m:
        raise ResumeAdapterMismatch(
            f"session {row['id']} was started with model '{want_m}'; it cannot be resumed with model '{got_m}'.")


def resolve_resume_adapter(db, resume_session_id: int, adapter_name: str | None,
                           model: str | None) -> tuple[str | None, str | None]:
    """Decide which adapter/model to resume a session with.

    If the caller passed --adapter explicitly, use that (check_resume_adapter will then refuse a true
    mismatch). Otherwise default to the adapter/model the session was originally started with, so a bare
    `--resume` never silently swaps an AI for the scripted reference policy (audit 2026-11-13).
    """
    row = db.session(resume_session_id)
    if adapter_name is not None:
        return adapter_name, model
    if row is None:
        return adapter_name, model
    return row.get("adapter"), (model if model is not None else row.get("model"))


def ml_enabled_for(cfg: dict, agent_id: str) -> bool:
    """Whether neural-network (learning) tools are offered to this agent.

    `ml.enabled` switches them off for everyone; `ml.agents.<id>.enabled` overrides per agent.
    Default: on (backwards compatible).
    """
    ml = cfg.get("ml") or {}
    per = ((ml.get("agents") or {}).get(agent_id) or {})
    if "enabled" in per:
        return bool(per["enabled"])
    return bool(ml.get("enabled", True))


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
            "continue_requested": bool(getattr(env, "continue_requested", False)),
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
    if not ml_enabled_for(cfg, agent_id):
        # The researcher has switched neural-network tools off (globally or for this agent).
        tool_names = [n for n in tool_names if SPECS[n]["category"] != "learning"]
    resumed = None
    if resume_session_id is not None:
        row = db.session(resume_session_id)
        if row is None or row["status"] == "completed":
            raise ValueError(f"session {resume_session_id} cannot be resumed (status: {row and row['status']})")
        check_resume_adapter(row, adapter)
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
        # Register the stop signal as soon as the session exists, so a stop requested during startup is honored.
        _unregister_session_stop(sid)
        _register_session_stop(sid)
        env = Environment(cfg, db, memory, agent_id, sid, number, mode=mode, allowed_tools=tool_names,
                          scene=scene, clock=clock)
    briefing = memory.briefing() if include_briefing else None
    # Stamp every memory item written this session with the brain that actually produced it.
    memory.author = {"adapter": adapter.name, "model": adapter.model, "source": "agent"}
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
    reviewed = False
    end_reason = None
    if adapter.driven_externally:
        return {"session_id": sid, "environment": env, "system": system, "opening": opening,
                "tools": tool_names, "memory": memory}
    try:
        adapter.begin(system, tool_names)
        user_text: str | None = opening
        results: list[tuple[ToolCall, dict]] = []
        while True:
            if is_session_stop_requested(sid):
                end_reason = "stopped_by_user"
                break
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
                # One turn before the idle limit would end the session, give exploration agents a single
                # review-and-continue prompt: a last chance to resume unfinished work instead of drifting
                # to a stop. After that, the normal budget-aware nudge. Audit 2026-11-13.
                if (env.mode == "exploration" and idle == int(s_cfg["idle_turn_limit"]) - 1
                        and not reviewed and env.phase == "active"):
                    user_text, reviewed = REVIEW_BEFORE_STOP, True
                else:
                    user_text = continue_nudge(env.remaining() if env.phase == "active" else None)
                db.add_message(sid, turns, "user", user_text)
                continue
            idle = 0
            for call in turn.tool_calls:
                res = env.call(call.name, call.args, source="agent", turn=turns)
                results.append((call, res))
                if is_session_stop_requested(sid):
                    end_reason = "stopped_by_user"
                    break
            if end_reason == "stopped_by_user":
                break
            if env.phase == "closed":
                break
    except AdapterError as exc:
        error, end_reason = str(exc), "adapter_error"
    except KeyboardInterrupt:
        error, end_reason = "interrupted by operator", "interrupted"
    except Exception as exc:  # noqa: BLE001
        error, end_reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}", "runner_error"
    finally:
        _unregister_session_stop(sid)
    return finish_session(cfg, db, env, adapter, turns, usage,
                          end_reason if error else (env.end_reason or end_reason or "closed"), error, on_finish)


def finish_session(cfg: dict, db, env: Environment, adapter: Adapter, turns: int, usage: dict,
                   end_reason: str, error: str | None,
                   on_finish: Callable[[Environment], dict] | None = None) -> dict:
    """Save everything, snapshot the memory files and discard the world (the reset)."""
    sid = env.sid
    _unregister_session_stop(sid)   # every finished session (loop, external, stopped) drops its stop signal
    resumable = end_reason in ("adapter_error", "interrupted", "runner_error")
    
    # Auto-capture the agent's last words when the session ends and the final
    # words are not already in the diary.
    # - agent_stopped (text-only stop / idle limit / external done): the last
    #   turns' words ARE the final thoughts; capture up to three.
    # - agent_ended (the agent called end_session): only words said in the same
    #   turn as the end_session call count as final thoughts; older narration
    #   is stale and is not captured.
    if (end_reason in ("agent_stopped", "agent_ended") and env.mode == "exploration"
            and not resumable):
        all_msgs = db.messages(sid)
        assistant_msgs = [m for m in all_msgs if m["role"] == "assistant"]
        if end_reason == "agent_ended":
            last_end = db.q("SELECT turn FROM actions WHERE session_id=? AND tool='end_session' "
                            "ORDER BY idx DESC LIMIT ?", (sid, 1))
            end_turn = last_end[0].get("turn") if last_end else None
            final_source = [m["content"].strip() for m in assistant_msgs
                            if end_turn is not None and m.get("turn") == end_turn
                            and (m.get("content") or "").strip()]
        else:
            texts = [m["content"] for m in assistant_msgs]
            final_source = [t.strip() for t in (texts[-3:] if len(texts) >= 3 else texts) if t and t.strip()]
        if final_source:
            combined = " ".join(final_source)
            if combined and len(combined) > 20:  # Only capture if there's meaningful content
                # Skip when the agent already wrote these words down themselves
                # (e.g. a scripted wrap-up diary entry) — no duplicate capture.
                if not any(combined[:60] in (e.get("text") or "") for e in env.memory.state["diary"]):
                    # Truncate if too long
                    if len(combined) > 500:
                        combined = combined[:497] + "..."
                    auto_text = f"[auto-captured] Session ended ({end_reason}). Final thoughts: {combined}"
                    env.memory.add_diary(env.session_number, env.world.time, auto_text, author="auto")
    
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
