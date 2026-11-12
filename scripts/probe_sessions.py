"""Probe: run scripted exploration sessions and print what actually happened."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.agents import make_adapter  # noqa: E402
from playground.config import load_config  # noqa: E402
from playground.session import run_session  # noqa: E402
from playground.storage import Database  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 45
n_sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 2
agent = sys.argv[3] if len(sys.argv) > 3 else "agent_001"
tmp = Path(tempfile.mkdtemp(prefix="pg_sess_"))
cfg = load_config(overrides={"_root": str(tmp), "session": {"max_steps": steps}})
db = Database(tmp / "pg.db")
for s in range(n_sessions):
    t = time.perf_counter()
    summ = run_session(cfg, db, agent, make_adapter(cfg, "scripted"))
    sid = summ["session_id"]
    print(f"\n===== session {summ['session_number']} (id {sid}) end={summ['end_reason']} error={summ['error']} "
          f"calls={summ['agent_calls']} actions={summ['actions_logged']} sim={summ['sim_time']}s "
          f"wall={time.perf_counter() - t:.1f}s")
    for a in db.actions(sid):
        r = a["result"]
        extra = ""
        if a["tool"] == "throw" and a["success"]:
            extra = f" peak={r.get('max_height')} rest_on={r.get('resting_on')} final={r.get('final_position')}"
        elif a["tool"] in ("record_discovery", "update_discovery") and a["success"]:
            extra = f" requested={r['requested_status']} granted={r['granted_status']}"
        elif a["tool"] == "record_experiment" and a["success"]:
            extra = f" {r['experiment_id']} verdict={r['evidence_check']['verdict']}"
        elif a["tool"] == "update_goal" and a["success"]:
            extra = f" status={r['status']} check={r.get('claim_check', {}).get('verified')}"
        elif a["tool"] == "walk_to":
            extra = f" -> {r.get('end_position') or r.get('details', {}).get('end_position')}"
        elif a["tool"] == "place" and a["success"]:
            extra = f" final={r.get('final_position')} on={r.get('resting_on')}"
        elif a["tool"] == "get_object_positions":
            extra = f" count={r.get('count')}"
        err = "" if a["success"] else f"  !! {r['error']['code']}: {r['error']['message'][:90]}"
        print(f"  {a['idx']:3d} {a['source'][:5]:5s} {a['tool']:20s}{extra}{err}")
    print("  messages:")
    for m in db.messages(sid):
        if m["role"] == "assistant":
            print("    >", m["content"][:230].replace("\n", " "))
    print("  behaviour:", summ["behaviour"], "| integrity flags:", summ["integrity_flags"])
    print("  verified:", json.dumps(summ["integrity"]["what_was_independently_verified"]))
    print("  memory:", json.dumps(summ["memory"]))
print("\ntmp:", tmp)
