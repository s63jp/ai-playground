"""Environment probe: drive the tool interface directly and print what comes back."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.config import load_config  # noqa: E402
from playground.interface import Environment  # noqa: E402
from playground.memory import MemoryStore  # noqa: E402
from playground.storage import Database  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="pg_probe_"))
cfg = load_config(overrides={"_root": str(tmp)})
db = Database(tmp / "pg.db")
mem = MemoryStore(tmp / "agents", "agent_001", cfg, db=db)
sid = db.create_session(agent_id="agent_001", mode="exploration", session_number=1, adapter="probe")
env = Environment(cfg, db, mem, "agent_001", sid, 1)


def show(label, r, keys=None):
    if not r["success"]:
        print(f"{label}: FAIL [{r['error']['code']}] {r['error']['message']}")
    else:
        body = {k: r[k] for k in (keys or list(r)[:7]) if k in r}
        print(f"{label}: ok #{r['action_index']} {json.dumps(body)[:330]}")
    return r


c = env.call
o = show("observe", c("observe_world"), ["towers", "structures"])
print("   objects:", len(o["objects"]), "| agent:", o["agent"]["position"], "| ids:", [x["id"] for x in o["objects"]][:8])
show("look_around", c("look_around", {"radius": 4}), ["objects"])
show("bad tool", c("fly", {}))
show("bad arg", c("move", {"direction": "north"}))
show("bad arg2", c("throw", {"object_id": "block_01", "direction": "up", "force": 500}))
show("unknown obj", c("pick_up", {"object_id": "block_99"}))
show("hidden block", c("inspect_object", {"object_id": "block_13"}))
show("walk", c("walk_to", {"x": -2.5, "y": -2.1}), ["arrived", "distance_travelled"])
show("pick", c("pick_up", {"object_id": "block_01"}), ["object", "total_mass"])
t1 = show("throw standing", c("throw", {"object_id": "block_01", "direction": "up", "force": 150}),
          ["max_height", "max_height_above_launch", "flight_time", "release_speed"])
show("flight time", c("measure_flight_time", {"target": "block_01"}), ["flight_time", "max_height", "impact_speed"])
show("collisions", c("inspect_collision", {"object_id": "block_01", "limit": 2}), ["count", "collisions"])
sb = show("sandbox", c("run_physics_test", {"parameters": {"label": "jump-throw", "steps": [
    {"tool": "pick_up", "args": {"object_id": "block_02"}},
    {"tool": "jump", "args": {"wait": "takeoff"}},
    {"tool": "throw", "args": {"object_id": "block_02", "direction": "up", "force": 150}}]}}),
          ["steps_run", "steps_failed", "note"])
print("   sandbox throw max_height:", sb["results"][2].get("max_height"), "| real world block_02 still at:",
      c("inspect_object", {"object_id": "block_02"})["object"]["position"])

# experiment records: honest, fabricated, uncited
e1 = show("exp honest", c("record_experiment", {
    "question": "How high does a block go when thrown straight up at 150 N?", "hypothesis": "About 2 m above my hands",
    "actual_result": f"max height {t1['max_height']}", "measurements": {"max_height": t1["max_height"]},
    "action_refs": [t1["action_index"]], "supported": True}), ["experiment_id", "evidence_check"])
e2 = show("exp fabricated", c("record_experiment", {
    "question": "same", "hypothesis": "It goes very high", "actual_result": "max height 32",
    "measurements": {"max_height": 32.0}, "action_refs": [t1["action_index"]], "supported": True}),
          ["experiment_id", "evidence_check"])
e3 = show("exp bad refs", c("record_experiment", {
    "question": "x", "hypothesis": "y", "actual_result": "z", "action_refs": [999], "supported": True}),
          ["experiment_id", "evidence_check"])
d1 = show("discovery confirmed? (1 exp)", c("record_discovery", {
    "statement": "A 150 N upward throw of a block peaks about 1.8 m above the release point.",
    "status": "confirmed", "experiment_ids": [e1["experiment_id"]]}), ["granted_status", "reason"])
d2 = show("discovery from fabricated", c("record_discovery", {
    "statement": "Blocks fly to 32 m.", "status": "confirmed", "experiment_ids": [e2["experiment_id"]]}),
          ["granted_status", "reason"])
show("discovery no evidence", c("record_discovery", {"statement": "The ring is magic.", "status": "confirmed"}),
     ["granted_status", "reason"])
# second independent experiment -> promotion
c("walk_to", {"x": -2.5, "y": -2.1})
c("pick_up", {"object_id": "block_01"})
t2 = c("throw", {"object_id": "block_01", "direction": "up", "force": 150})
e4 = c("record_experiment", {"question": "repeat", "hypothesis": "Same height again",
                              "actual_result": "same", "measurements": {"max_height": t2["max_height"]},
                              "action_refs": [t2["action_index"]], "supported": True})
show("promote with 2nd exp", c("update_discovery", {"discovery_id": d1["discovery_id"], "status": "confirmed",
                                                   "add_experiment_ids": [e4["experiment_id"]]}),
     ["granted_status", "reason"])
# programs
show("save program", c("save_program", {"name": "toss", "description": "pick up and throw up", "steps": [
    {"tool": "pick_up", "args": {"object_id": "${block}"}},
    {"tool": "throw", "args": {"object_id": "${block}", "direction": "up", "force": "${force}"}}]}),
     ["program", "parameters"])
p = show("run program", c("execute_program", {"name": "toss", "params": {"block": "block_01", "force": 100}}),
         ["steps_run", "steps_failed"])
print("   program throw max_height:", p["results"][-1].get("max_height"))
show("program missing param", c("execute_program", {"name": "toss", "params": {"block": "block_01"}}))
show("program bad step", c("save_program", {"name": "bad", "steps": [{"tool": "record_discovery", "args": {}}]}))
# claims
show("claim false tower", c("claim_achievement", {"type": "tower", "params": {"min_blocks": 5}}),
     ["verified", "measured"])
g = c("set_goal", {"title": "Build a two-block tower"})
c("walk_to", {"x": -1.5, "y": -2.1})
c("pick_up", {"object_id": "block_02"})
show("place on block_03", c("place", {"object_id": "block_02", "position": [-0.5, -3.0]}),
     ["final_position", "resting_on", "stayed_where_placed"])
show("goal completed w/ claim", c("update_goal", {"goal_id": g["goal_id"], "status": "completed",
                                                  "claim": {"type": "tower", "params": {"min_blocks": 2}}}),
     ["status", "claim_check"])
show("history", c("inspect_world_history", {"types": ["flight_end"], "limit": 2}), ["count"])
show("diary", c("write_diary", {"text": "Threw blocks. Height is repeatable."}), ["entry_id"])
show("end", c("end_session", {"reason": "done for now"}), ["ended"])
show("after end", c("observe_world"))
print("chain:", db.verify_chain(sid))
print("actions logged:", len(db.actions(sid)), "| events:", db.one("SELECT COUNT(*) c FROM events")["c"],
      "| frames:", db.frame_count(sid))
print("integrity flags:", [(r["data"]["kind"]) for r in db.records("integrity_flag")])
try:
    db.x("UPDATE events SET data_json='{}' WHERE n=5")
except Exception as e:  # noqa: BLE001
    print("tamper attempt blocked:", e)
print("--- handbook.md ---")
print((mem.dir / "handbook.md").read_text(encoding="utf-8"))
print("tmp:", tmp)
