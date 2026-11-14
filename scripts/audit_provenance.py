#!/usr/bin/env python
"""Audit memory-item provenance and quarantine items written by a brain other than the session's own.

Background (audit 2026-11-13): session 19 (agent_003 session #5) was started as the `lmstudio_unsloth4b`
language model, hit an adapter error, and was resumed with the CLI's old default adapter `scripted` — the
reference policy, which is NOT an AI. 70 of its 71 actions, and every throwing "discovery" that then filled
agent_003's briefings, were produced by the scripted policy while the session row still named the LLM.

New sessions now stamp each memory item with its author (adapter/model) and resuming with a different brain
is refused, so this can't recur. This script repairs the EXISTING contaminated records:

  * --report (default): list, per agent, every memory item whose authoring session ran (even partly) under a
    different adapter than the session row claims, or under the scripted reference policy.
  * --apply: mark the affected discoveries as challenged (never silently deleted), with a reason that names
    the real author, so the evidence system stops treating them as settled. Experiments/goals/mistakes are
    reported but left in place (they are already only as strong as their cited actions).

Usage:
    python scripts/audit_provenance.py --db experiments/playground.db
    python scripts/audit_provenance.py --db experiments/playground.db --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def _session_adapters(con: sqlite3.Connection, sid: int) -> list[str]:
    """Adapter names recorded across this session's session_started events (one per start/resume)."""
    rows = con.execute(
        "SELECT data_json FROM events WHERE session_id=? AND type=? ORDER BY n",
        (sid, "session_started"),
    ).fetchall()
    out = []
    for (dj,) in rows:
        d = json.loads(dj)
        desc = d.get("adapter") or {}
        # adapter.describe() returns {"adapter": name, "model": ...}; older rows may use "name".
        name = desc.get("adapter") or desc.get("name")
        if name:
            out.append(name)
    return out


def _action_sources(con: sqlite3.Connection, sid: int) -> dict:
    """Count action program/source; the scripted policy marks its actions with source='agent' too, so we
    rely on the session_started adapter trail rather than the source column alone."""
    rows = con.execute("SELECT source, COUNT(*) FROM actions WHERE session_id=? GROUP BY source", (sid,)).fetchall()
    return {s: n for s, n in rows}


def find_contaminated(con: sqlite3.Connection) -> list[dict]:
    """Sessions whose actual brain differs from the session row's adapter (resume swapped the adapter)."""
    out = []
    for sid, aid, adapter, model in con.execute("SELECT id, agent_id, adapter, model FROM sessions").fetchall():
        seen = _session_adapters(con, sid)
        distinct = [a for a in dict.fromkeys(seen)]
        # Contaminated if more than one adapter ran in the session, or the only adapter that ran differs
        # from what the session row claims.
        actually = set(distinct)
        mixed = len(actually) > 1
        swapped = bool(distinct) and adapter is not None and adapter not in actually
        if mixed or swapped:
            out.append({"session_id": sid, "agent_id": aid, "row_adapter": adapter, "row_model": model,
                        "adapters_seen": distinct, "scripted_ran": "scripted" in actually})
    return out


def items_from_session(mem_dir: Path, sid_number: int) -> dict:
    """Memory items (by file) whose created/origin session number matches a contaminated session number."""
    found = defaultdict(list)
    for fname, key, idfield in (("discoveries.json", "items", "created_session"),
                                ("experiments.json", "items", "session")):
        p = mem_dir / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for it in data.get(key, []):
            if it.get(idfield) == sid_number:
                found[fname].append(it["id"])
    state_p = mem_dir / "state.json"
    if state_p.exists():
        state = json.loads(state_p.read_text(encoding="utf-8"))
        for coll, field in (("goals", "origin_session"), ("mistakes", "session")):
            for it in state.get(coll, []):
                if it.get(field) == sid_number:
                    found[coll].append(it["id"])
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="experiments/playground.db")
    ap.add_argument("--memory-root", default="experiments/agents")
    ap.add_argument("--apply", action="store_true", help="challenge affected discoveries (default: report only)")
    ap.add_argument("--json", action="store_true", help="emit a JSON report (for the dashboard)")
    args = ap.parse_args(argv)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    contaminated = find_contaminated(con)

    if args.json:
        mismatches = []
        for c in contaminated:
            row = con.execute("SELECT session_number FROM sessions WHERE id=?", (c["session_id"],)).fetchone()
            snum = row[0] if row else None
            items = items_from_session(Path(args.memory_root) / c["agent_id"], snum)
            mismatches.append({"session_id": c["session_id"], "agent_id": c["agent_id"],
                               "row_adapter": c["row_adapter"], "ran": c["adapters_seen"],
                               "scripted_ran": c["scripted_ran"], "items": dict(items)})
        con.close()
        print(json.dumps({"available": True, "mismatches": mismatches}, indent=2))
        return 0

    if not contaminated:
        print("No provenance mismatches found: every session ran under a single, matching adapter.")
        return 0

    # Map contaminated DB session ids to their session_number per agent (memory items key off the number).
    print(f"Found {len(contaminated)} session(s) whose brain differs from the session row:\n")
    to_challenge = defaultdict(list)  # agent_id -> [(discovery_id, reason)]
    for c in contaminated:
        row = con.execute("SELECT session_number FROM sessions WHERE id=?", (c["session_id"],)).fetchone()
        snum = row[0] if row else None
        mem_dir = Path(args.memory_root) / c["agent_id"]
        items = items_from_session(mem_dir, snum)
        real = "scripted reference policy (not an AI)" if c["scripted_ran"] else "/".join(c["adapters_seen"])
        print(f"  session id {c['session_id']} (agent {c['agent_id']} #{snum}): row says "
              f"'{c['row_adapter']}', actually ran {c['adapters_seen']}")
        for fname, ids in items.items():
            print(f"      {fname}: {', '.join(ids) or '(none)'}")
        reason = (f"authored in session {snum} by the {real}, not by {c['row_adapter']} as the session row "
                  f"claimed (provenance audit 2026-11-13)")
        for did in items.get("discoveries.json", []):
            to_challenge[c["agent_id"]].append((did, reason))
    con.close()

    if not args.apply:
        print("\nReport only. Re-run with --apply to mark the affected discoveries as challenged.")
        return 0

    # Apply: challenge each affected discovery through the real MemoryStore so audit logs are written.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from playground.config import load_config
    from playground.memory import MemoryStore
    from playground.storage import Database

    cfg = load_config()
    db = Database(Path(args.db))
    changed = 0
    for agent_id, entries in to_challenge.items():
        mem = MemoryStore(args.memory_root, agent_id, cfg, db=db)
        for did, reason in entries:
            try:
                mem.challenge(did, reason, actor="provenance_audit")
                changed += 1
                print(f"  challenged {agent_id} {did}")
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP {agent_id} {did}: {exc}")
    db.close()
    print(f"\nChallenged {changed} discovery item(s). They now carry a researcher challenge and must be "
          f"re-tested before counting as evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
