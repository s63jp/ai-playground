"""Evidence checks: does what an agent wrote down agree with what the engine measured?

Nothing in here trusts the agent's account. An experiment record cites action
numbers; those actions are looked up in the immutable action log and the claimed
measurements are compared with the values the physics engine returned.
"""
from __future__ import annotations

import math
import re
from typing import Any

PHYSICAL_CATEGORIES = ("movement", "object", "experiment")
MEASURING_TOOLS = ("measure_height", "measure_distance", "measure_velocity", "measure_flight_time",
                   "inspect_collision", "inspect_object", "inspect_position", "observe_world",
                   "get_object_positions", "get_inventory")
IGNORED_KEYS = {"flight_id", "seq", "idx", "action_index", "step", "t", "t_start", "t_end", "sim_time",
                "force", "elapsed", "object_mass", "mass", "total_mass"}


def numeric_leaves(obj: Any, path: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append((path, float(obj)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(numeric_leaves(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(numeric_leaves(v, f"{path}[{i}]"))
    return out


def _key(path: str) -> str:
    last = path.split(".")[-1]
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"\[\d+\]", "", last).lower()).strip("_")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.02, 0.02 * abs(b))


def verify_experiment(fields: dict, cited: dict[int, dict | None]) -> dict:
    """Check an experiment record against the actions it cites.

    `cited` maps each referenced action number to its logged action (or None if absent).
    Verdicts: consistent | grounded | contradicted | invalid_references | no_evidence.
    """
    missing = sorted(i for i, a in cited.items() if a is None)
    present = {i: a for i, a in cited.items() if a is not None}
    physical = {i: a for i, a in present.items()
                if a["category"] in PHYSICAL_CATEGORIES or a["tool"] in MEASURING_TOOLS}
    succeeded = {i: a for i, a in physical.items() if a["success"]}
    issues: list[str] = []
    if missing:
        issues.append(f"cited action(s) {missing} do not exist in this session's log")
    if not cited:
        return {"verified": False, "verdict": "no_evidence", "level": None, "checked_actions": [],
                "matches": [], "contradictions": [], "unmatched": [],
                "issues": ["no actions were cited as evidence"]}
    if not physical:
        return {"verified": False, "verdict": "invalid_references", "level": None,
                "checked_actions": sorted(present), "matches": [], "contradictions": [], "unmatched": [],
                "issues": issues + ["none of the cited actions is a physical action or measurement"]}

    engine: list[tuple[int, str, float]] = []
    for i, a in sorted(physical.items()):
        res = a["result"]
        for path, val in numeric_leaves(res):
            if _key(path) in IGNORED_KEYS:
                continue
            engine.append((i, path, val))

    matches, contradictions, unmatched = [], [], []
    for name, claimed in numeric_leaves(fields.get("measurements") or {}):
        ckey = _key(name)
        same_key = [(i, p, v) for i, p, v in engine if _key(p) == ckey]
        if same_key:
            hit = next(((i, p, v) for i, p, v in same_key if _close(claimed, v)), None)
            if hit:
                matches.append({"claim": name, "claimed": claimed, "engine": hit[2], "action": hit[0],
                                "field": hit[1], "match": "same_field"})
            else:
                nearest = min(same_key, key=lambda x: abs(x[2] - claimed))
                contradictions.append({"claim": name, "claimed": claimed, "engine": nearest[2],
                                       "action": nearest[0], "field": nearest[1]})
            continue
        hit = next(((i, p, v) for i, p, v in engine if _close(claimed, v) and abs(v) > 1e-9), None)
        if hit:
            matches.append({"claim": name, "claimed": claimed, "engine": hit[2], "action": hit[0],
                            "field": hit[1], "match": "value"})
        else:
            unmatched.append({"claim": name, "claimed": claimed,
                              "note": "not found in the cited actions' measurements (derived or unsupported)"})

    if contradictions:
        verdict, verified, level = "contradicted", False, None
        issues.append("a claimed measurement disagrees with the value the engine recorded")
    elif missing and not succeeded and not matches:
        verdict, verified, level = "invalid_references", False, None
    elif matches:
        verdict, verified, level = "consistent", True, "measured"
    else:
        verdict, verified, level = "grounded", True, "cited"
    return {"verified": verified, "verdict": verdict, "level": level, "checked_actions": sorted(physical),
            "missing_actions": missing, "matches": matches, "contradictions": contradictions,
            "unmatched": unmatched, "issues": issues}


def check_flight(flight: dict, g: float = 9.81) -> list[str]:
    """Physical plausibility of one recorded flight. Returns a list of problems (empty = plausible)."""
    problems: list[str] = []
    v = flight.get("launch_velocity") or [0.0, 0.0, 0.0]
    speed = flight.get("launch_speed")
    if speed is None:
        speed = math.sqrt(sum(float(x) ** 2 for x in v))
    vz = float(v[2])
    rise = flight.get("max_height_above_launch")
    if rise is not None:
        bound = (vz * vz) / (2 * g) if vz > 0 else 0.0
        if rise > bound + 0.05 + 0.02 * bound:
            problems.append(f"rose {rise:.3f} m above launch, more than the ballistic limit {bound:.3f} m "
                            f"for a vertical launch speed of {vz:.3f} m/s")
        if rise < -1e-6:
            problems.append("maximum height is below the launch height")
    ft = flight.get("flight_time")
    if ft is not None and ft < 0:
        problems.append("negative flight time")
    tmax = flight.get("time_to_max_height")
    if ft is not None and tmax is not None and tmax > ft + 0.01 and flight.get("impacts", 0) <= 1:
        if rise is not None and rise > 0.05 and flight.get("outcome") == "rest":
            pass  # the apex can legitimately follow a bounce
    if speed and speed > 60:
        problems.append(f"launch speed {speed:.1f} m/s exceeds anything the body can produce")
    imp, lp, ip = flight.get("impact_speed"), flight.get("launch_position"), flight.get("first_impact_position")
    if imp is not None and lp and ip:
        e_max = speed * speed + 2 * g * (float(lp[2]) - float(ip[2]))
        if imp * imp > e_max + 1.0 + 0.05 * abs(e_max):
            problems.append(f"impact speed {imp:.2f} m/s implies the object gained energy in flight")
    return problems


def check_claim_limits(measurements: dict, world_limits: dict) -> list[str]:
    """Flag claimed numbers that cannot be true of this world regardless of evidence."""
    problems = []
    for name, val in numeric_leaves(measurements or {}):
        k = _key(name)
        if any(w in k for w in ("height", "apex", "altitude")) and val > world_limits.get("max_height", 60.0):
            problems.append(f"claimed {name}={val} exceeds the highest value reachable in this world")
        if any(w in k for w in ("time", "duration")) and val < 0:
            problems.append(f"claimed {name}={val} is a negative time")
        if any(w in k for w in ("speed", "velocity")) and abs(val) > world_limits.get("max_speed", 60.0):
            problems.append(f"claimed {name}={val} exceeds any speed attainable in this world")
    return problems
