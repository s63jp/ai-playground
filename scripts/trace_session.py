#!/usr/bin/env python3
"""Print a readable trace of one session from the database (researcher tool).

Usage:
    python scripts/trace_session.py DATABASE [SESSION_ID]
    python scripts/trace_session.py --db experiments/playground.db --session 1
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a readable trace of one session from the database"
    )
    parser.add_argument("database", nargs="?", default=None,
                        help="Path to the SQLite database")
    parser.add_argument("session_id", nargs="?", type=int, default=None,
                        help="Session ID to trace (default: 1)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to the SQLite database (alternative to positional)")
    parser.add_argument("--session", type=int, default=None,
                        help="Session ID to trace (alternative to positional)")
    args = parser.parse_args()

    # Resolve database path
    db_path = args.database or args.db
    if not db_path:
        parser.error("Database path is required (either positional or --db)")

    # Resolve session ID
    sid = args.session_id or args.session or 1

    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        return 1

    row = db.execute(
        "SELECT status, end_reason, action_count, sim_time, model FROM sessions WHERE id=?",
        (sid,)
    ).fetchone()

    if not row:
        print(f"Session {sid} not found", file=sys.stderr)
        return 1

    print(f"session {sid}: status={row['status']} end_reason={row['end_reason']} "
          f"actions={row['action_count']} sim_time={row['sim_time']} model={row['model']}")

    # Check for resume events
    for (t, d) in db.execute(
        "SELECT type, data_json FROM events WHERE session_id=? AND type='session_resumed'",
        (sid,)
    ):
        print(f"RESUMED: {d}")

    # Get assistant messages by turn
    msgs: dict[int, list[str]] = {}
    for turn, c in db.execute(
        "SELECT turn, content FROM messages WHERE session_id=? AND role='assistant'",
        (sid,)
    ):
        msgs.setdefault(turn, []).append(c)

    # Print actions with context
    shown: set[int] = set()
    for row in db.execute(
        "SELECT idx, tool, args_json, success, result_json, turn, source FROM actions "
        "WHERE session_id=? ORDER BY idx",
        (sid,)
    ):
        idx, tool, args, ok, res, turn, src = row

        # Show assistant messages for this turn
        if turn in msgs and turn not in shown:
            shown.add(turn)
            for c in msgs[turn]:
                print(f"    SAYS: {c[:400].replace(chr(10), ' ')}")

        r = json.loads(res)
        extra = ""

        if tool == "throw" and ok:
            extra = f" peak={r.get('max_height')} rest_on={r.get('resting_on')}"
        elif tool == "jump" and ok:
            extra = f" apex={r.get('apex_above_takeoff')}"
        elif tool == "record_experiment" and ok:
            extra = f" verdict={r['evidence_check']['verdict']}"
        elif tool in ("record_discovery", "update_discovery") and ok:
            extra = f" asked={r['requested_status']} granted={r['granted_status']}"
        elif tool == "claim_achievement" and ok:
            extra = f" verified={r['verified']}"
        elif tool in ("move", "walk_to") and ok and r.get("objects_disturbed"):
            extra = f" disturbed={[o['object'] for o in r['objects_disturbed']]}"

        tag = "" if src == "agent" else f"[{src}] "
        if ok:
            print(f"  #{idx} {tag}{tool} {args[:110]} -> OK{extra}")
        else:
            err = r.get("error", {})
            print(f"  #{idx} {tag}{tool} {args[:110]} -> FAIL {err.get('code')}: {err.get('message', '')[:100]}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
