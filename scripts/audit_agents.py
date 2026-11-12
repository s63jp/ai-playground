"""Researcher audit of the multi-agent run: isolation, persistence, reset, integrity. Reads the DB and files only."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playground.storage import Database  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments")
db = Database(root / "playground.db")
agents_dir = root / "agents"

print("agent      sess  confirmed  probable  experiments  ok/contradicted  chain  s2 read memory  s2 start hash")
start_hashes, problems = set(), []
for ad in sorted(agents_dir.iterdir()):
    a = ad.name
    disc = json.loads((ad / "discoveries.json").read_text())["items"]
    exps = json.loads((ad / "experiments.json").read_text())["items"]
    st = json.loads((ad / "state.json").read_text())
    rows = db.q("SELECT id, session_number, summary_json FROM sessions WHERE agent_id=? AND mode='exploration' "
                "ORDER BY session_number", (a,))
    for r in rows:
        r["initial_state_hash"] = json.loads(r["summary_json"] or "{}").get("initial_state_hash")
    chains = [db.verify_chain(r["id"])["ok"] for r in rows]
    for r in rows:
        start_hashes.add(r["initial_state_hash"])
    s2 = [r for r in rows if r["session_number"] == 2]
    reads = 0
    if s2:
        reads = sum(1 for x in db.actions(s2[0]["id"]) if x["tool"] == "read_memory")
    # isolation: every experiment this agent cites must come from its own sessions
    own = {r["id"] for r in rows}
    foreign = [e["id"] for e in exps if e.get("session_id") is not None and e["session_id"] not in own]
    if foreign:
        problems.append(f"{a}: experiments from another agent's sessions {foreign}")
    ok = sum(1 for e in exps if e.get("verification", {}).get("verified"))
    bad = sum(1 for e in exps if e.get("verification", {}).get("verdict") == "contradicted")
    n = lambda s: sum(1 for d in disc if d["status"] == s)  # noqa: E731
    print(f"{a}  {st['sessions_completed']:>4}  {n('confirmed'):>9}  {n('probable'):>8}  {len(exps):>11}  "
          f"{ok:>6}/{bad:<8}  {'ok' if all(chains) else 'BROKEN':>5}  {reads:>14}  "
          f"{(s2[0]['initial_state_hash'] or '')[:12] if s2 else '-'}")
print(f"\ndistinct initial world states across all {db.q('SELECT COUNT(*) n FROM sessions')[0]['n']} sessions: "
      f"{len(start_hashes)} (1 = every session started from the identical reset island)")
print("isolation problems:", problems or "none")
