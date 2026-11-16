#!/usr/bin/env python3
"""Migrate agent memory records to the integrity schema (task 2026-11-16).

Adds, to every existing record that lacks it:

- ``agent_id``            — stamped from the directory name.
- ``created_at``          — wall clock at migration time (original wall time was
                            not recorded; the value is explicitly labelled with
                            ``created_at_source: "migrated"``).
- ``session_id``          — looked up from playground.db sessions by
                            (agent_id, session_number) when a unique match exists.
- ``outcome``             — evidence_outcome(verification, supported) so the
                            evidence-check result and the scientific conclusion
                            are stored separately for legacy records too.
- diary entries ``author``— "auto" when the text carries the ``[auto-captured]``
                            marker, else "agent".
- goal ``origin``         — "agent" (nothing in the code ever assigned goals).

Before touching anything the script copies ``experiments/`` to
``backups/migrate-memory-<ts>/`` and writes a per-agent JSON report. Records are
never deleted; the script is idempotent (already-stamped fields are left alone).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playground.integrity.evidence import evidence_outcome  # noqa: E402
from playground.memory.store import _write_atomic, GOAL_ORIGINS  # noqa: E402


def main() -> int:
    data = ROOT / "experiments"
    agents_dir = data / "agents"
    db_path = data / "playground.db"
    if not agents_dir.is_dir():
        print(f"no agents dir at {agents_dir}")
        return 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "backups" / f"migrate-memory-{ts}"
    backup.mkdir(parents=True)
    shutil.copytree(data, backup / "experiments")
    print(f"backup of experiments/ -> {backup}/experiments")

    db = sqlite3.connect(db_path) if db_path.exists() else None
    sessions_map: dict[tuple[str, int], int] = {}
    if db is not None:
        for agent_id, num, sid in db.execute("SELECT agent_id, session_number, id FROM sessions"):
            sessions_map.setdefault((agent_id, num), sid)

    report: dict[str, dict[str, int]] = {}
    now = round(time.time(), 3)
    for agent_dir in sorted(p for p in agents_dir.iterdir() if (p / "state.json").exists()
                            or (p / "discoveries.json").exists()):
        agent_id = agent_dir.name
        counts = {"discoveries": 0, "experiments": 0, "diary": 0, "goals": 0, "skipped": 0}

        def load(name: str, default) -> dict:
            p = agent_dir / name
            if not p.exists():
                return default
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"!! {agent_id}/{name} is not valid JSON; skipped")
                counts["skipped"] += 1
                return default

        sessions_map_full = sessions_map
        # --- discoveries -------------------------------------------------------
        if (agent_dir / "discoveries.json").exists():
            disc = load("discoveries.json", {"items": [], "next_id": 1})
            changed = False
            for d in disc.get("items", []):
                if "agent_id" not in d:
                    d["agent_id"] = agent_id
                    changed = True
                if "created_at" not in d:
                    d["created_at"] = now
                    d["created_at_source"] = "migrated"
                    changed = True
                if "session_id" not in d and d.get("created_session") is not None:
                    s = sessions_map_full.get((agent_id, d["created_session"]))
                    if s is None:
                        s = sessions_map_full.get((agent_id, d.get("updated_session")))
                    if s is not None:
                        d["session_id"] = s
                        changed = True
                if changed or "outcome" not in d:
                    pass  # discoveries have no verification to classify
            if changed:
                _write_atomic(agent_dir / "discoveries.json", json.dumps(disc, indent=1, ensure_ascii=False))
                counts["discoveries"] = sum(1 for d in disc["items"] if "agent_id" in d)

        # --- experiments -------------------------------------------------------
        if (agent_dir / "experiments.json").exists():
            exps = load("experiments.json", {"items": [], "next_id": 1})
            changed = False
            for e in exps.get("items", []):
                touched = False
                if "agent_id" not in e:
                    e["agent_id"] = agent_id
                    touched = True
                if "created_at" not in e:
                    e["created_at"] = now
                    e["created_at_source"] = "migrated"
                    touched = True
                if "session_id" not in e:
                    s = sessions_map_full.get((agent_id, e.get("session")))
                    if s is not None:
                        e["session_id"] = s
                        touched = True
                if "outcome" not in e:
                    e["outcome"] = evidence_outcome(e.get("verification"), e.get("supported"))
                    touched = True
                if touched:
                    changed = True
            if changed:
                _write_atomic(agent_dir / "experiments.json", json.dumps(exps, indent=1, ensure_ascii=False))
                counts["experiments"] = sum(1 for e in exps["items"] if "agent_id" in e)

        # --- state.json: diary authors + goal origins --------------------------
        if (agent_dir / "state.json").exists():
            st = load("state.json", {})
            changed = False
            for y in st.get("diary", []):
                if "author" not in y:
                    y["author"] = "auto" if "[auto-captured]" in y.get("text", "") else "agent"
                    changed = True
            for g in st.get("goals", []):
                if "origin" not in g:
                    g["origin"] = "agent"
                    changed = True
                for h in g.get("history", []):
                    if "origin" not in h:
                        h["origin"] = g["origin"]
                        changed = True
            if changed:
                _write_atomic(agent_dir / "state.json", json.dumps(st, indent=1, ensure_ascii=False))
                counts["diary"] = sum(1 for y in st.get("diary", []) if "author" in y)
                counts["goals"] = sum(1 for g in st.get("goals", []) if "origin" in g)

        report[agent_id] = counts
        print(f"{agent_id}: {counts}")

    rep_path = backup / "migration_report.json"
    _write_atomic(rep_path, json.dumps({"created": ts, "agents": report}, indent=1))
    with open(ROOT / "experiments" / "migrate-memory-last.json", "w", encoding="utf-8") as fh:
        json.dump({"created": ts, "backup": str(backup), "agents": report}, fh, indent=1)
    print(f"report -> {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
