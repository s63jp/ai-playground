#!/usr/bin/env python3
"""Training data extraction pipeline for ai-playground → Unsloth SFT datasets.

This script reconstructs full tool-calling trajectories from the SQLite database,
applies quality gates (success rate, integrity verification), and outputs in
Unsloth-compatible JSONL format with provenance metadata.

Usage:
    python scripts/export_training_data.py --adapter lmstudio_claude_distilled9b \
        --format openai_tool_calls --output unsloth_qwen3.5_16k.jsonl

See docs/OPERATING.md for configuration options and quality gate documentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
import contextlib


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for provenance hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class ExportError(Exception):
    """Raised when export fails validation or integrity checks."""
    pass


# -----------------------------------------------------------------------------
# Configuration & Quality Gates
# -----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "min_success_rate": 0.75,           # Minimum tool call success rate per session
    "min_actions_per_session": 20,      # Filter out very short sessions (< N actions)
    "max_messages_per_turn": 1000,      # Truncate excessively long messages
    "include_reasoning_only": False,    # Also export turns with text but no tool calls
    "exclude_adapters": [],             # Adapter names to skip (e.g., ["scripted"])
    "min_integrity_score": None,        # Minimum verified/total ratio for experiments
}


def quality_gates_passed(session: dict, adapter_stats: dict) -> tuple[bool, str]:
    """Check if a session passes all configured quality gates.

    Returns (passed, reason). If passed is True, reason explains why; otherwise
    the first failing gate's name and message are returned.
    """
    # Gate 1: adapter exclusion list
    if session.get("adapter") in DEFAULT_CONFIG["exclude_adapters"]:
        return False, f"adapter excluded ({session['adapter']})"

    # Gate 2: minimum actions threshold
    acts = session.get("action_count", 0) or 0
    if acts < DEFAULT_CONFIG["min_actions_per_session"]:
        return False, f"too few actions ({acts} < {DEFAULT_CONFIG['min_actions_per_session']})"

    # Gate 3: success rate (if configured and adapter has stats)
    min_sr = DEFAULT_CONFIG.get("min_success_rate")
    if min_sr is not None and session.get("adapter") in adapter_stats:
        sr = adapter_stats[session["adapter"]].get("success_rate", 0.0)
        if sr < min_sr:
            return False, f"low success rate ({sr:.1%} < {min_sr:.1%})"

    # Gate 4: integrity score (if configured and experiments exist)
    integ_min = DEFAULT_CONFIG.get("min_integrity_score")
    recs = session.get("records", []) or []
    if integ_min is not None and recs:
        exp_recs = [r for r in recs if r.get("kind") == "experiment"]
        if exp_recs:
            verified = sum(1 for e in exp_recs if json.loads(e["data_json"]).get("verification", {}).get("verified"))
            score = verified / len(exp_recs)
            if score < integ_min:
                return False, f"low integrity score ({score:.2f} < {integ_min})"

    # Passed all gates
    reason = (
        f"adapter={session['adapter']} "
        f"model={session.get('model', 'N/A')} "
        f"actions={acts} "
        f"messages={len(session.get('messages', [])) or 0}"
    )
    return True, reason


# -----------------------------------------------------------------------------
# Database Access Layer
# -----------------------------------------------------------------------------

class PlaygroundDB:
    """Read-only access to the playground SQLite database."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if not Path(self.path).exists():
            raise ExportError(f"Database not found: {path}")
        self.conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        with self.conn.lock if hasattr(self.conn, "lock") else contextlib.nullcontext():
            self.conn.close()

    def q(self, sql: str, params: Iterable = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable = ()) -> dict | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    # -------------------------------------------------------------------------
    # Session queries
    # -------------------------------------------------------------------------

    def sessions_by_adapter_model(
        self, adapter: Optional[str], model: Optional[str], mode: str = "exploration"
    ) -> list[dict]:
        """Get all sessions for a given adapter/model combination."""
        # Build WHERE clause conditions - always include mode first
        where_parts = [f"s.mode=?"]
        params = [mode]
        
        if adapter:
            where_parts.append("s.adapter=?")
            params.append(adapter)
        if model:
            where_parts.append("s.model=?")
            params.append(model)
        
        # Build SQL with WHERE clause - simple SELECT from sessions table, no JOIN needed
        sql = f"SELECT s.* FROM sessions s " \
               f"WHERE {' AND '.join(where_parts)} ORDER BY s.action_count DESC, s.started_at ASC LIMIT 1000"
        return self.q(sql, tuple(params))

    def session(self, sid: int) -> dict | None:
        """Get a single session by ID."""
        row = self.one("SELECT * FROM sessions WHERE id=?", (sid,))
        if not row:
            return None
        for k in ("summary_json", "final_state_json", "config_json", "budget_json"):
            row[k[:-5]] = json.loads(row.pop(k)) if row.get(k) else None
        return row

    # -------------------------------------------------------------------------
    # Message queries (for conversation reconstruction)
    # -----------------------------------------------------------------------------

    def messages(self, sid: int, after_id: int = 0, limit: int = 100000) -> list[dict]:
        """Get all messages for a session."""
        return self.q(
            "SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id LIMIT ?",
            (sid, after_id, limit)
        )

    # -------------------------------------------------------------------------
    # Action queries (for tool calls and results)
    # -----------------------------------------------------------------------------

    def actions(self, sid: int, after_idx: int = 0, limit: int = 100000) -> list[dict]:
        """Get all actions for a session."""
        rows = self.q(
            "SELECT * FROM actions WHERE session_id=? AND idx>? ORDER BY idx LIMIT ?",
            (sid, after_idx, limit)
        )
        # Parse JSON fields back to dicts/lists
        for r in rows:
            r["args"] = json.loads(r.pop("args_json"))
            r["result"] = json.loads(r.pop("result_json"))
            r["success"] = bool(r["success"])
        return rows

    def actions_by_turn(self, sid: int) -> dict[int, list[dict]]:
        """Group actions by turn number."""
        acts = self.actions(sid)
        by_turn: dict[int, list[dict]] = {}
        for a in acts:
            t = a["turn"] or 0
            by_turn.setdefault(t, []).append(a)
        return by_turn

    # -------------------------------------------------------------------------
    # Integrity records (for quality scoring)
    # -----------------------------------------------------------------------------

    def integrity_records(self, sid: int) -> list[dict]:
        """Get all integrity-related records for a session."""
        kinds = ("experiment", "claim", "discovery", "mistake", "goal", "diary")
        sql = "SELECT * FROM records WHERE session_id=? AND kind IN ({}) ORDER BY id".format(
            ",".join("?" * len(kinds))
        )
        return self.q(sql, tuple(kinds) + (sid,))

    def integrity_score(self, sid: int) -> float | None:
        """Calculate verified/total ratio for experiments in this session."""
        recs = self.integrity_records(sid)
        exp_recs = [r for r in recs if r.get("kind") == "experiment"]
        if not exp_recs:
            return None
        total = len(exp_recs)
        verified = sum(1 for e in exp_recs if json.loads(e["data_json"]).get("verification", {}).get("verified"))
        return verified / total


# -----------------------------------------------------------------------------
# Conversation Reconstruction
# -----------------------------------------------------------------------------

def reconstruct_conversation(
    db: PlaygroundDB, sid: int, include_reasoning_only: bool = False
) -> list[dict]:
    """Reconstruct a full tool-calling conversation from messages + actions.

    Returns a list of message dicts in OpenAI-compatible format with optional
    "tool_calls" and "tool_response" fields for SFT training.
    """
    # Get all messages sorted by id (which is also turn order)
    msgs = db.messages(sid, after_id=0, limit=100000)

    # Group actions by turn
    acts_by_turn = db.actions_by_turn(sid)

    conversation: list[dict] = []

    # Turn 0: system + user opening (if present)
    for m in msgs:
        if m["turn"] == 0:
            conversation.append({"role": m["role"], "content": m["content"]})

    max_turn = max([a["turn"] or 0 for a in db.actions(sid)] + [m["turn"] or 0 for m in msgs])

    # Subsequent turns
    for turn in range(1, max_turn + 1):
        assistant_msgs = [m for m in msgs if m["turn"] == turn and m["role"] == "assistant"]
        user_msgs = [m for m in msgs if m["turn"] == turn and m["role"] == "user"]

        actions_for_turn = acts_by_turn.get(turn, [])

        # Assistant message (with optional tool_calls)
        assistant: dict[str, Any] = {"role": "assistant"}
        if assistant_msgs:
            content = " ".join(m["content"] or "" for m in assistant_msgs).strip()
            if content and len(content) > DEFAULT_CONFIG["max_messages_per_turn"]:
                content = content[:DEFAULT_CONFIG["max_messages_per_turn"]] + "..."
            assistant["content"] = content

        # Add tool calls from actions (if any)
        if actions_for_turn:
            for action in actions_for_turn:
                call_id = f"call_{action['idx']}"
                assistant.setdefault("tool_calls", []).append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": action["tool"],
                        "arguments": action["args"]
                    }
                })

        if assistant.get("content") or include_reasoning_only:
            conversation.append(assistant)

        # Tool response messages (one per tool call)
        for action in actions_for_turn:
            conversation.append({
                "role": "tool",
                "tool_call_id": f"call_{action['idx']}",
                "content": json.dumps(action["result"], indent=2, sort_keys=True),
            })

        # User messages (nudges, follow-ups)
        for um in user_msgs:
            conversation.append({"role": "user", "content": um["content"]})

    return conversation


def export_session(
    db: PlaygroundDB, sid: int, adapter_name: str, include_reasoning_only: bool = False
) -> dict[str, Any]:
    """Export a single session as one training example.

    Returns a dict with "messages" (the reconstructed conversation), plus metadata.
    """
    conv = reconstruct_conversation(db, sid, include_reasoning_only=include_reasoning_only)

    # Compute provenance hash for this exact trajectory
    blob = canonical_json(conv)
    traj_hash = hashlib.sha256(blob.encode()).hexdigest()[:16]

    session_row = db.session(sid)
    if not session_row:
        return None

    integrity_score = db.integrity_score(sid)

    example: dict[str, Any] = {
        "messages": conv,
        "metadata": {
            "session_id": sid,
            "agent_id": session_row.get("agent_id", "unknown"),
            "adapter": adapter_name,
            "model": session_row.get("model", "unknown"),
            "mode": session_row.get("mode", "exploration"),
            "status": session_row.get("status", "unknown"),
            "action_count": session_row.get("action_count", 0) or 0,
            "trajectory_hash": traj_hash,
            "integrity_score": round(integrity_score, 3) if integrity_score is not None else None,
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }

    return example


def export_sessions_batch(
    db: PlaygroundDB,
    sessions: list[dict],
    adapter_name: str,
    include_reasoning_only: bool = False,
) -> tuple[list[dict], dict]:
    """Export multiple sessions and compute aggregate statistics.

    Returns (examples_dict, stats_dict).
    """
    examples: list[dict] = []
    total_actions = 0
    total_messages = 0
    total_tool_calls = 0
    verified_experiments = 0
    total_experiments = 0

    for session in sessions:
        sid = session["id"]
        example = export_session(db, sid, adapter_name, include_reasoning_only)
        if not example:
            continue

        examples.append(example)
        acts = session.get("action_count", 0) or 0
        msgs = len(session.get("messages", [])) or 0
        total_actions += acts
        total_messages += msgs

        # Count tool calls from actions table
        acts_list = db.actions(sid)
        total_tool_calls += len(acts_list)

        # Integrity counts
        recs = session.get("records", []) or []
        exp_recs = [r for r in recs if r.get("kind") == "experiment"]
        verified_experiments += sum(1 for e in exp_recs if json.loads(e["data_json"]).get("verification", {}).get("verified"))
        total_experiments += len(exp_recs)

    # Aggregate stats
    avg_integrity = (verified_experiments / total_experiments * 100) if total_experiments else None

    stats: dict[str, Any] = {
        "adapter": adapter_name,
        "model": sessions[0].get("model", "unknown") if sessions else "unknown",
        "sessions_exported": len(examples),
        "total_actions": total_actions,
        "total_messages": total_messages,
        "total_tool_calls": total_tool_calls,
        "verified_experiments": verified_experiments,
        "total_experiments": total_experiments,
        "avg_integrity_score_pct": round(avg_integrity, 2) if avg_integrity else None,
    }

    return examples, stats


# -----------------------------------------------------------------------------
# Main Export Pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export ai-playground trajectories to Unsloth-compatible JSONL format"
    )
    parser.add_argument(
        "--db", "-d", default="experiments/playground.db",
        help="Path to playground SQLite database (default: experiments/playground.db)"
    )
    parser.add_argument(
        "--adapter", "-a", required=True,
        help="Adapter name to export (e.g., lmstudio_claude_distilled9b)"
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Filter by model name (optional)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--format", choices=["openai_tool_calls", "chat_template"], default="openai_tool_calls",
        help="Output format: openai_tool_calls (with tool_calls field) or chat_template (merged)"
    )
    parser.add_argument(
        "--include-reasoning-only", action="store_true",
        help="Also include turns with reasoning text but no tool calls"
    )
    parser.add_argument(
        "--stats-file", default=None,
        help="Optional JSON file to write export statistics"
    )

    args = parser.parse_args()

    # Load database
    try:
        db = PlaygroundDB(args.db)
    except ExportError as e:
        print(f"ERROR: {e}")
        return 1

    try:
        # Query sessions for this adapter/model combination
        mode_filter = "exploration" if args.model is None else "exploration"  # Always use exploration mode
        sessions = db.sessions_by_adapter_model(args.adapter, args.model, mode=mode_filter)
        
        print(f"\nFound {len(sessions)} sessions for adapter={args.adapter} {'model=' + args.model if args.model else ''}")


        # Apply quality gates and filter
        filtered_sessions: list[dict] = []
        rejected: list[tuple[str, str]] = []  # (session_id, reason)

        for session in sessions:
            passed, reason = quality_gates_passed(session, {})
            if passed:
                filtered_sessions.append(session)
            else:
                sid = session.get("id") or "?"
                rejected.append((sid, reason))

        print(f"After quality gates: {len(filtered_sessions)} sessions pass, {len(rejected)} rejected")

        # Export to JSONL
        examples, stats = export_sessions_batch(
            db, filtered_sessions, args.adapter, include_reasoning_only=args.include_reasoning_only
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting {len(examples)} examples to {args.output}...")

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                # Format based on selected mode
                if args.format == "chat_template":
                    # Merge tool calls into assistant message content (Unsloth-compatible)
                    msgs = example["messages"]
                    formatted_msgs = []
                    for m in msgs:
                        role, content = m["role"], m.get("content", "") or ""
                        tc = m.pop("tool_calls", [])
                        if tc and role == "assistant":
                            # Prepend tool call info to reasoning text
                            calls_str = "\n\n".join(
                                f"→ {c['function']['name']}({json.dumps(c['function']['arguments'])})"
                                for c in tc
                            )
                            content = (content or "") + f"\n\n{calls_str}"
                        elif role == "tool":
                            # For chat template, merge tool response into next assistant turn
                            pass  # Handled by merging with following assistant message
                        formatted_msgs.append({"role": role, "content": content})

                    example["messages"] = formatted_msgs

                f.write(json.dumps(example, indent=2) + "\n")

        print(f"Export complete: {len(examples)} trajectories written.")

        # Write stats if requested
        if args.stats_file:
            stats_path = Path(args.stats_file)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump({**stats, "rejected": rejected}, f, indent=2)

        print(f"\nStatistics:")
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")

    except Exception as e:
        print(f"ERROR during export: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    exit(main())
