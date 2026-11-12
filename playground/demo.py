"""End-to-end demonstration.

Runs in its own workspace (never touches the real experiment data) and shows an agent:
  observing the island, choosing its own objective, attempting it, failing, changing
  approach, discovering a mechanic, recording it, having the island reset, returning in a
  later session and using its saved knowledge to do something it previously could not.

The headline comparison is a controlled one: after session 1, the agent's memory is
copied twice. One copy is left intact, the other has the relevant discovery removed.
Both copies return to the same reset island with the same remaining goal, so any
difference is attributable to that one memory entry. Every number printed comes from
the physics engine's record; the script only reads it back.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .agents import make_adapter
from .config import load_config, resolve_path
from .memory import MemoryStore
from .session import run_session
from .storage import Database


BUDGET = 46   # long enough to discover and record a mechanic, too short to also exploit it


def _line(s: str = "") -> None:
    print(s, flush=True)


def _trace(db, sid: int, limit: int = 80) -> list[str]:
    out = []
    msgs = {m["turn"]: m["content"] for m in db.messages(sid) if m["role"] == "assistant"}
    shown = set()
    for a in db.actions(sid)[:limit]:
        if a["turn"] in msgs and a["turn"] not in shown:
            shown.add(a["turn"])
            out.append(f'      says: "{msgs[a["turn"]][:200]}"')
        r = a["result"]
        note = ""
        if a["tool"] == "throw" and a["success"]:
            note = f"peak {r['max_height']} m, came to rest on {r['resting_on']}"
        elif a["tool"] == "jump" and a["success"] and r.get("apex_above_takeoff") is not None:
            note = f"feet rose {r['apex_above_takeoff']} m"
        elif a["tool"] == "set_goal" and a["success"]:
            note = f"goal {r['goal_id']}: {a['args']['title']}"
        elif a["tool"] == "record_experiment" and a["success"]:
            note = f"{r['experiment_id']}, evidence check: {r['evidence_check']['verdict']}"
        elif a["tool"] in ("record_discovery", "update_discovery") and a["success"]:
            note = f"asked for '{r['requested_status']}', granted '{r['granted_status']}'"
        elif a["tool"] == "update_goal" and a["success"]:
            note = f"status {r['status']}" + (f", independent check verified={r['claim_check']['verified']}"
                                              if r.get("claim_check") else "")
        elif a["tool"] == "record_mistake" and a["success"]:
            note = a["args"]["what"]
        elif a["tool"] == "read_memory":
            note = a["args"]["file"]
        elif a["tool"] == "execute_program" and a["success"]:
            note = f"program '{a['args']['name']}' saved in an earlier session"
        if not a["success"]:
            note = f"FAILED [{r['error']['code']}] {r['error']['message'][:110]}"
        if a["tool"] in ("walk_to", "observe_world", "get_world_boundaries", "list_programs", "measure_height",
                         "look_around", "pick_up") and a["success"]:
            continue
        out.append(f"   #{a['idx']:<3d} {a['tool']:<18s} {note}")
    return out


def run_demo(cfg: dict | None = None, out: str = "experiments/demo", adapter: str = "scripted",
             model: str | None = None) -> int:
    base = load_config()
    root = Path(out) if Path(out).is_absolute() else Path(base.get("_root") or Path(__file__).resolve().parent.parent) / out
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    cfg = load_config(overrides={"_root": str(root), "session": {"max_steps": BUDGET, "wrap_up_notice_steps": 3}})
    db = Database(resolve_path(cfg, cfg["database"]))
    (root / "demo_config.yaml").write_text(f"_root: {root.as_posix()}\n", encoding="utf-8")
    agent = "agent_001"
    mk = lambda: make_adapter(cfg, adapter, model=model)  # noqa: E731

    _line("=" * 78)
    _line("AI EXPERIMENTAL PLAYGROUND: end-to-end demonstration")
    _line(f"workspace: {root}")
    _line(f"agent brain: {adapter}" + (" (a hand-written reference policy, not an AI; see "
                                       "playground/agents/scripted.py)" if adapter == "scripted" else ""))
    _line("=" * 78)

    # ---- session 1: short on purpose, so the goal is left unfinished
    _line(f"\nSESSION 1: no memory, no objective given. {BUDGET}-action budget.")
    s1 = run_session(cfg, db, agent, mk())
    for line in _trace(db, s1["session_id"]):
        _line(line)
    m = MemoryStore(root / "experiments" / "agents", agent, cfg)
    _line(f"\n  end of session 1: {s1['end_reason']}. Memory written to {m.dir}")
    _line("  handbook now:")
    for d in m.discoveries["items"]:
        _line(f"    {d['id']} [{d['status']}] {d['statement'][:120]}")
    open_goals = [g for g in m.state["goals"] if g["status"] in ("active", "paused")]
    for g in open_goals:
        _line(f"    unfinished goal {g['id']}: {g['title']}")
    _line(f"  island reset: next session starts from state "
          f"{s1['island_reset']['next_session_starts_from'][:16]}, the same as session 1 started from "
          f"{s1['island_reset']['this_session_started_from'][:16]}: "
          f"{s1['island_reset']['next_session_starts_from'] == s1['island_reset']['this_session_started_from']}")

    # ---- controlled return: same memory, minus one entry
    jt = next((d["id"] for d in m.discoveries["items"] if d.get("category") == "jump_throw"), None)
    results = {}
    for cond in ("with_memory", "without_that_discovery"):
        croot = root / "conditions" / cond
        rm = [jt] if (cond == "without_that_discovery" and jt) else []
        m.copy_to(croot, remove_ids=rm, reason="demonstration control")
        ccfg = load_config(overrides={"_root": str(root), "memory": {"root": str(croot)},
                                      "session": {"max_steps": BUDGET, "wrap_up_notice_steps": 3}})
        s = run_session(ccfg, db, agent, mk(), memory=MemoryStore(croot, agent, ccfg, db=db),
                        label=f"demo:{cond}")
        claims = db.records(kind="claim", session_id=s["session_id"])
        spire = [c["data"] for c in claims if c["data"]["type"] == "object_on" and c["data"]["params"].get(
            "surface") == "spire"]
        results[cond] = {"session_id": s["session_id"], "actions": s["agent_calls"],
                         "spire_landing_verified": any(c["verified"] for c in spire),
                         "throws": sum(1 for a in db.actions(s["session_id"]) if a["tool"] == "throw"),
                         "summary": s}
        title = ("SESSION 2 (memory intact)" if cond == "with_memory" else
                 f"SESSION 2, CONTROL (same memory with {jt} removed)")
        _line(f"\n{title}: island reset; the agent starts by reading its files.")
        for line in _trace(db, s["session_id"]):
            _line(line)

    # ---- verdict, from the engine's own record
    w, wo = results["with_memory"], results["without_that_discovery"]
    s1_spire = any(c["data"]["verified"] for c in db.records(kind="claim", session_id=s1["session_id"])
                   if c["data"]["type"] == "object_on")
    _line("\n" + "=" * 78)
    _line("RESULT (independently verified from physics events, not from the agent's account)")
    _line(f"  session 1, no memory:               block on spire verified = {s1_spire}")
    _line(f"  session 2, memory intact:           block on spire verified = {w['spire_landing_verified']}"
          f"  ({w['throws']} throws, {w['actions']} actions)")
    _line(f"  session 2, minus {jt}:            block on spire verified = {wo['spire_landing_verified']}"
          f"  ({wo['throws']} throws, {wo['actions']} actions)")
    helped = w["spire_landing_verified"] and (not s1_spire)
    attributable = helped and (not wo["spire_landing_verified"] or wo["actions"] > w["actions"])
    _line(f"  achieved in a later session what it could not in the first: {helped}")
    _line(f"  difference attributable to the removed memory entry ({jt}): {attributable}")
    _line(f"  event-log hash chains intact: "
          f"{all(db.verify_chain(x)['ok'] for x in [s1['session_id'], w['session_id'], wo['session_id']])}")
    _line("=" * 78)
    report = {"session_1": s1["session_id"], "with_memory": {k: v for k, v in w.items() if k != "summary"},
              "without": {k: v for k, v in wo.items() if k != "summary"}, "removed": jt,
              "s1_spire": s1_spire, "later_success": helped, "attributable": attributable}
    (root / "demo_report.json").write_text(json.dumps(report, indent=2))
    _line(f"report: {root / 'demo_report.json'}")
    _line(f"watch it: python -m playground --config {(root / 'demo_config.yaml').as_posix()} serve")
    return 0 if helped else 1
