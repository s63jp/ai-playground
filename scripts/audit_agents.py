#!/usr/bin/env python3
"""Researcher audit of the multi-agent run: isolation, persistence, reset, integrity.

Reads the DB and memory files to verify:
  - Each agent's sessions use only their own experiments (isolation)
  - Hash chains are intact
  - All sessions started from identical reset island state

Usage:
    python scripts/audit_agents.py [WORKSPACE]
    python scripts/audit_agents.py --workspace experiments
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playground.storage import Database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit multi-agent runs for isolation, persistence, and integrity"
    )
    parser.add_argument("workspace", nargs="?", default=None,
                        help="Experiment workspace path (default: experiments)")
    parser.add_argument("--workspace", "-w", dest="workspace_opt", type=str, default=None,
                        help="Experiment workspace path (alternative to positional)")
    args = parser.parse_args()

    # Resolve workspace path
    workspace = args.workspace or args.workspace_opt or "experiments"
    root = Path(workspace)

    if not root.exists():
        print(f"Error: workspace '{root}' does not exist", file=sys.stderr)
        return 1

    db_path = root / "playground.db"
    if not db_path.exists():
        print(f"Error: database '{db_path}' does not exist", file=sys.stderr)
        return 1

    agents_dir = root / "agents"
    if not agents_dir.exists():
        print(f"Error: agents directory '{agents_dir}' does not exist", file=sys.stderr)
        return 1

    db = Database(db_path)

    print("agent      sess  confirmed  probable  experiments  ok/contradicted  chain  s2 read memory  s2 start hash")
    start_hashes: set[str | None] = set()
    problems: list[str] = []

    for ad in sorted(agents_dir.iterdir()):
        if not ad.is_dir():
            continue

        a = ad.name
        try:
            disc = json.loads((ad / "discoveries.json").read_text())["items"]
            exps = json.loads((ad / "experiments.json").read_text())["items"]
            st = json.loads((ad / "state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"{a}: Error reading memory files: {e}", file=sys.stderr)
            continue

        rows = db.q(
            "SELECT id, session_number, summary_json FROM sessions WHERE agent_id=? AND mode='exploration' "
            "ORDER BY session_number",
            (a,)
        )

        for r in rows:
            r["initial_state_hash"] = json.loads(r["summary_json"] or "{}").get("initial_state_hash")

        chains = [db.verify_chain(r["id"])["ok"] for r in rows]

        for r in rows:
            start_hashes.add(r["initial_state_hash"])

        s2 = [r for r in rows if r["session_number"] == 2]
        reads = 0
        if s2:
            reads = sum(1 for x in db.actions(s2[0]["id"]) if x["tool"] == "read_memory")

        # Check isolation: every experiment this agent cites must come from its own sessions
        own = {r["id"] for r in rows}
        foreign = [
            e["id"] for e in exps
            if e.get("session_id") is not None and e["session_id"] not in own
        ]
        if foreign:
            problems.append(f"{a}: experiments from another agent's sessions {foreign}")

        ok = sum(1 for e in exps if e.get("verification", {}).get("verified"))
        bad = sum(1 for e in exps if e.get("verification", {}).get("verdict") == "contradicted")

        def n(s: str) -> int:
            return sum(1 for d in disc if d["status"] == s)

        chain_status = "ok" if all(chains) else "BROKEN"
        s2_hash = (s2[0]["initial_state_hash"] or "")[:12] if s2 else "-"

        print(f"{a}  {st['sessions_completed']:>4}  {n('confirmed'):>9}  {n('probable'):>8}  {len(exps):>11}  "
              f"{ok:>6}/{bad:<8}  {chain_status:>5}  {reads:>14}  {s2_hash}")

    total_sessions = db.q("SELECT COUNT(*) n FROM sessions")[0]["n"]
    # Filter out None values
    valid_hashes = [h for h in start_hashes if h is not None]
    print(f"\ndistinct initial world states across all {total_sessions} sessions: "
          f"{len(valid_hashes)} (1 = every session started from the identical reset island)")
    print("isolation problems:", problems or "none")

    db.close()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
