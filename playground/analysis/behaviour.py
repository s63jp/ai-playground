"""Behaviour classification from the recorded log.

Four research categories are tracked. Every entry is a count plus the evidence
(action numbers, record ids, quoted text) that produced it. Nothing here states
or implies what an agent felt: only what it was recorded doing.
"""
from __future__ import annotations

import re
from typing import Any

PHYSICAL = ("movement", "object")
CONSTRAINT_WORDS = re.compile(r"\b(without|only|using just|as many|as high|as far|in a row|blindfold|one[- ]handed|"
                              r"no more than|fewer than|at most|exactly|challenge|record)\b", re.I)
SYMBOLIC_WORDS = re.compile(r"\b(represents?|pretend|imagine|looks like|is my|is a|as a|call(ed)? (it|this|them)|"
                            r"named?|symboli[sz]es?|stands for|monument|statue|altar|house|castle|bridge|garden|"
                            r"lighthouse|throne|gate)\b", re.I)


def _numeric_signature(args: dict) -> tuple:
    out = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append((k, round(float(v), 3)))
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            out.append((k, tuple(round(float(x), 3) for x in v)))
        elif isinstance(v, str):
            out.append((k, v))
    return tuple(out)


def classify_session(db, session_id: int) -> dict:
    sess = db.session(session_id)
    agent = sess["agent_id"]
    actions = db.actions(session_id)
    recs = db.records(session_id=session_id)
    by_kind: dict[str, list[dict]] = {}
    for r in recs:
        by_kind.setdefault(r["kind"], []).append(r)
    goals = by_kind.get("goal", [])
    labels = by_kind.get("label", [])
    experiments = by_kind.get("experiment", [])
    claims = by_kind.get("claim", [])
    diary = by_kind.get("diary", [])

    # ---- 1. self-directed goals
    set_events = [g for g in goals if g["data"].get("event") == "set"]
    updates = [g for g in goals if g["data"].get("event") == "update"]
    challenges = [g for g in set_events if CONSTRAINT_WORDS.search(
        (g["data"].get("title") or "") + " " + (g["data"].get("description") or ""))]
    set_ids = {g["key"] for g in set_events}
    revisited = sorted({g["key"] for g in updates if g["key"] not in set_ids})
    self_directed = {
        "count": len(set_events),
        "goals_set": [{"id": g["key"], "title": g["data"]["title"], "action": g["action_idx"]} for g in set_events],
        "self_imposed_constraints": [{"id": g["key"], "title": g["data"]["title"]} for g in challenges],
        "status_changes": [{"id": g["key"], "status": g["data"]["status"], "action": g["action_idx"],
                            "independently_verified": g["data"].get("independently_verified")} for g in updates],
        "abandoned": sum(1 for g in updates if g["data"]["status"] == "abandoned"),
        "completed_claimed": sum(1 for g in updates if g["data"]["status"] == "completed"),
        "completed_verified": sum(1 for g in updates if g["data"]["status"] == "completed"
                                  and g["data"].get("independently_verified") is True),
        "revisited_from_earlier_sessions": revisited,
        "criteria": "set_goal calls (no goal is ever assigned in exploration mode); constraints detected by wording",
    }

    # ---- 2. pretend play
    textual = []
    for r in diary + set_events:
        text = r["data"].get("text") or r["data"].get("title") or ""
        m = SYMBOLIC_WORDS.search(text)
        if m:
            textual.append({"record": r["key"], "kind": r["kind"], "matched": m.group(0), "text": text[:200]})
    pretend = {
        "count": len(labels),
        "meanings_assigned": [{"targets": l["data"]["targets"], "meaning": l["data"]["meaning"],
                               "action": l["action_idx"]} for l in labels],
        "group_meanings": sum(1 for l in labels if len(l["data"]["targets"]) > 1),
        "textual_indicators": textual,
        "criteria": "assign_meaning calls (strong evidence); representational wording in diary/goals (weak evidence, "
                    "listed separately and not counted)",
    }

    # ---- 3. repetition with variation
    runs, cur = [], []
    for a in actions:
        if a["category"] not in PHYSICAL or a["tool"] in ("move", "walk_to", "wait"):
            if a["category"] in PHYSICAL:
                continue
            continue
        if cur and cur[-1]["tool"] != a["tool"]:
            runs.append(cur)
            cur = []
        cur.append(a)
    if cur:
        runs.append(cur)
    by_tool: dict[str, list[dict]] = {}
    for a in actions:
        if a["category"] in PHYSICAL and a["tool"] not in ("move", "walk_to", "wait"):
            by_tool.setdefault(a["tool"], []).append(a)
    variations = []
    for tool, acts in by_tool.items():
        sigs = [_numeric_signature({k: v for k, v in a["args"].items() if k != "object_id"}) for a in acts]
        distinct = len(set(sigs))
        if len(acts) >= 3 and distinct >= 2:
            variations.append({"activity": tool, "repetitions": len(acts), "distinct_parameterisations": distinct,
                               "actions": [a["idx"] for a in acts][:40]})
    prog_runs: dict[str, list] = {}
    for a in actions:
        if a["tool"] == "execute_program":
            prog_runs.setdefault(a["args"].get("name", "?"), []).append(a)
    for name, acts in prog_runs.items():
        sigs = {repr(sorted((a["args"].get("params") or {}).items())) for a in acts}
        if len(acts) >= 2:
            variations.append({"activity": f"program:{name}", "repetitions": len(acts),
                               "distinct_parameterisations": len(sigs), "actions": [a["idx"] for a in acts]})
    placed_on: dict[str, list[int]] = {}
    for a in actions:
        if a["tool"] in ("place", "put_down") and a["success"]:
            for base in a["result"].get("resting_on") or []:
                if base.startswith("block") or base in ("ring", "rock"):
                    placed_on.setdefault(base, []).append(a["idx"])
    repetition = {
        "count": len(variations),
        "varied_repetitions": sorted(variations, key=lambda v: -v["repetitions"]),
        "experiments_repeating_a_hypothesis": _repeated_hypotheses(experiments),
        "criteria": "the same physical activity performed 3+ times with at least 2 different parameter sets; "
                    "saved programs run 2+ times",
    }

    # ---- 4. activity for its own sake
    cited = set()
    for e in experiments:
        cited.update(e["data"].get("action_refs") or [])
    intervals = _goal_intervals(goals, len(actions))
    outside = []
    for a in actions:
        if a["category"] not in PHYSICAL or a["source"] != "agent":
            continue
        in_goal = any(lo <= a["idx"] <= hi for lo, hi in intervals)
        if not in_goal and a["idx"] not in cited:
            outside.append(a["idx"])
    physical_total = sum(1 for a in actions if a["category"] in PHYSICAL and a["source"] == "agent")
    own_sake = {
        "count": len(outside),
        "physical_actions_outside_any_goal_or_experiment": outside[:200],
        "share_of_physical_actions": round(len(outside) / physical_total, 3) if physical_total else 0.0,
        "achievement_claims_without_a_goal": [{"type": c["data"]["type"], "verified": c["data"]["verified"],
                                               "action": c["action_idx"]} for c in claims
                                              if not c["data"].get("goal_id")],
        "criteria": "physical actions taken while no self-set goal was active and never cited as experimental "
                    "evidence; this measures the absence of a recorded purpose, not a motive",
    }
    return {"session_id": session_id, "agent_id": agent,
            "self_directed_goals": self_directed, "pretend_play": pretend,
            "repetition_with_variation": repetition, "activity_for_its_own_sake": own_sake,
            "totals": {"actions": len(actions), "physical_actions": physical_total,
                       "experiments": len(experiments), "claims": len(claims),
                       "claims_verified": sum(1 for c in claims if c["data"]["verified"])},
            "disclaimer": "Classification of recorded behaviour only. No statement is made about experience, "
                          "enjoyment, boredom or any other internal state."}


def _repeated_hypotheses(experiments: list[dict]) -> list[dict]:
    seen: dict[str, list[str]] = {}
    for e in experiments:
        key = re.sub(r"[^a-z0-9 ]", "", (e["data"].get("hypothesis") or "").lower())[:60]
        if key:
            seen.setdefault(key, []).append(e["key"])
    return [{"hypothesis": k, "experiments": v} for k, v in seen.items() if len(v) >= 2]


def _goal_intervals(goals: list[dict], n_actions: int) -> list[tuple[int, int]]:
    opened: dict[str, int] = {}
    out: list[tuple[int, int]] = []
    for g in goals:
        gid, idx, status = g["key"], g["action_idx"] or 0, g["data"]["status"]
        if status == "active":
            opened.setdefault(gid, idx)
        elif gid in opened:
            out.append((opened.pop(gid), idx))
    out.extend((start, n_actions + 1) for start in opened.values())
    return out


def store_classification(db, session_id: int) -> dict[str, Any]:
    result = classify_session(db, session_id)
    db.delete_records("behaviour", session_id)
    db.add_record(result["agent_id"], "behaviour", result, session_id=session_id)
    return result
