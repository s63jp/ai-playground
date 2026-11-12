"""Print a readable trace of one session from the database (researcher tool)."""
import json
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
sid = int(sys.argv[2]) if len(sys.argv) > 2 else 1
row = db.execute("select status,end_reason,action_count,sim_time,model from sessions where id=?", (sid,)).fetchone()
print("session", sid, row)
for (t, d) in db.execute("select type,data_json from events where session_id=? and type='session_resumed'", (sid,)):
    print("RESUMED:", d)
msgs = {}
for turn, c in db.execute("select turn,content from messages where session_id=? and role='assistant'", (sid,)):
    msgs.setdefault(turn, []).append(c)
shown = set()
for idx, tool, args, ok, res, turn, src in db.execute(
        "select idx,tool,args_json,success,result_json,turn,source from actions where session_id=? order by idx",
        (sid,)):
    if turn in msgs and turn not in shown:
        shown.add(turn)
        for c in msgs[turn]:
            print("    SAYS:", c[:400].replace("\n", " "))
    r = json.loads(res)
    extra = ""
    if tool == "throw" and ok:
        extra = f" peak={r.get('max_height')} rest_on={r.get('resting_on')}"
    elif tool == "jump" and ok:
        extra = f" apex={r.get('apex_above_takeoff')}"
    elif tool == "record_experiment" and ok:
        extra = " verdict=" + r["evidence_check"]["verdict"]
    elif tool in ("record_discovery", "update_discovery") and ok:
        extra = f" asked={r['requested_status']} granted={r['granted_status']}"
    elif tool == "claim_achievement" and ok:
        extra = f" verified={r['verified']}"
    elif tool in ("move", "walk_to") and ok and r.get("objects_disturbed"):
        extra = f" disturbed={[o['object'] for o in r['objects_disturbed']]}"
    tag = "" if src == "agent" else f"[{src}] "
    print(f"  #{idx} {tag}{tool} {args[:110]} ->", ("OK" + extra) if ok else
          "FAIL " + r["error"]["code"] + ": " + r["error"]["message"][:100])
