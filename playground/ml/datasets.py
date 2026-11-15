"""A registry of named behaviour datasets — not just throws.

Audit 2026-11-13 follow-up: the ML lab could only learn throw outcomes. This module adds purpose-built
datasets for the other things the agent does, each built from the engine's OWN action records (the
`actions` table's result_json — the tool result the world returned), never from anything an agent typed:

  * climb        — will a climb onto a ledge succeed, and how much height is gained?
  * jump         — how a carried load and take-off height shape a jump (apex, air time, landing)
  * locomotion   — how far the body actually travels and what it disturbs, from a move/walk request
  * manipulation — can a given object be picked up (mass / whether it was moving)?  [affordance]
  * placement    — how accurately a placed object ends up where it was asked to go
  * affordance   — the CROSS-ACTION self-model: given the action and a little state, will it succeed?
                   This is the "does this work for me?" dataset — a model of the agent's own capabilities.

Every dataset carries its own input/target column vocabulary, so it trains through the same
nets.validate_spec / nets.train as throws (both take an explicit vocabulary). New behaviours are added by
appending one DatasetSpec — extend, don't duplicate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable


def _f(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _z(v: Any) -> float:
    """Coerce to a finite float, defaulting to 0.0 (for optional state fields)."""
    x = _f(v, 0.0)
    return x if x is not None else 0.0


@dataclass
class DatasetSpec:
    name: str
    description: str
    tools: tuple               # action tools this dataset draws from
    inputs: dict               # column -> human description (known before/at the action)
    targets: dict              # column -> human description (measured by the engine after)
    build_row: Callable[[dict, dict], dict | None]  # (args, result) -> row or None


# ----------------------------------------------------------------------------- climb
def _climb_row(args: dict, r: dict) -> dict | None:
    sp = r.get("start_position") or [0, 0, 0]
    if _f(r.get("ledge_height")) is None:
        return None
    return {
        "ledge_height": _z(r.get("ledge_height")),
        "start_feet_height": _z(sp[2] if len(sp) > 2 else 0.0),
        "started_airborne": 1.0 if r.get("started_airborne") else 0.0,
        "success": 1.0 if r.get("success") else 0.0,
        "height_gained": _z(r.get("height_gained")),
    }


# ----------------------------------------------------------------------------- jump
def _jump_row(args: dict, r: dict) -> dict | None:
    if _f(r.get("apex_above_takeoff")) is None:      # only completed jumps (not wait=takeoff stubs)
        return None
    return {
        "total_mass": _z(r.get("total_mass")),
        "takeoff_feet_height": _z(r.get("takeoff_feet_height")),
        "apex_above_takeoff": _z(r.get("apex_above_takeoff")),
        "air_time": _z(r.get("air_time")),
        "max_feet_height": _z(r.get("max_feet_height")),
        "impact_speed": _z(r.get("impact_speed")),
    }


# ----------------------------------------------------------------------------- locomotion
def _loco_row(args: dict, r: dict) -> dict | None:
    req = _f(args.get("distance")) or _f(r.get("requested_distance"))
    if _f(r.get("distance_travelled")) is None:
        return None
    disturbed = r.get("objects_disturbed") or []
    return {
        "requested_distance": _z(req),
        "is_walk_to": 1.0 if r.get("action") == "walk_to" else 0.0,
        "distance_travelled": _z(r.get("distance_travelled")),
        "objects_disturbed": float(len(disturbed)),
        "height_change": _z(r.get("height_change")),
        "left_ground": 1.0 if r.get("left_ground_during_move") else 0.0,
    }


# ----------------------------------------------------------------------------- manipulation (pick_up affordance)
def _pick_row(args: dict, r: dict) -> dict | None:
    if _f(r.get("object_mass")) is None:
        return None
    return {
        "object_mass": _z(r.get("object_mass")),
        "relative_speed": _z(r.get("relative_speed")),
        "was_moving": 1.0 if r.get("was_moving") else 0.0,
        "agent_on_ground": 1.0 if r.get("agent_on_ground") else 0.0,
        "success": 1.0 if r.get("success") else 0.0,
    }


# ----------------------------------------------------------------------------- placement accuracy
def _place_row(args: dict, r: dict) -> dict | None:
    if _f(r.get("displacement_from_requested")) is None:
        return None
    return {
        "fall_distance": _z(r.get("fall_distance")),
        "displacement_from_requested": _z(r.get("displacement_from_requested")),
        "stayed_where_placed": 1.0 if r.get("stayed_where_placed") else 0.0,
        "settle_time": _z(r.get("settle_time")),
    }


# ----------------------------------------------------------------------------- affordance / cognition (cross-action)
_AFFORD_TOOLS = ("move", "walk_to", "jump", "climb", "pick_up", "put_down", "place", "throw", "catch",
                 "rotate_object")


def _afford_row(args: dict, r: dict) -> dict | None:
    tool = r.get("action")
    if tool not in _AFFORD_TOOLS:
        return None
    sp = r.get("agent_position") or r.get("start_position") or r.get("position") or [0, 0, 0]
    return {
        "action_id": float(_AFFORD_TOOLS.index(tool)),
        "feet_height": _z(sp[2] if len(sp) > 2 else 0.0),
        "has_target_object": 1.0 if (args.get("object_id") or args.get("object")) else 0.0,
        "has_distance": 1.0 if args.get("distance") is not None else 0.0,
        "has_force": 1.0 if args.get("force") is not None else 0.0,
        "success": 1.0 if r.get("success", True) else 0.0,
    }


DATASETS: dict[str, DatasetSpec] = {
    "climb": DatasetSpec(
        "climb", "Will a climb onto a ledge succeed, and how much height is gained?", ("climb",),
        {"ledge_height": "height of the ledge above the feet, m", "start_feet_height": "feet height before, m",
         "started_airborne": "1 if the agent was airborne when it tried, else 0"},
        {"success": "1 if it climbed up, else 0", "height_gained": "metres of height gained"},
        _climb_row),
    "jump": DatasetSpec(
        "jump", "How a carried load and take-off height shape a jump.", ("jump",),
        {"total_mass": "agent + carried mass at take-off, kg", "takeoff_feet_height": "feet height at take-off, m"},
        {"apex_above_takeoff": "peak height above take-off, m", "air_time": "seconds airborne",
         "max_feet_height": "highest the feet reached, m", "impact_speed": "speed at landing, m/s"},
        _jump_row),
    "locomotion": DatasetSpec(
        "locomotion", "How far the body actually travels and what it disturbs, from a move/walk request.",
        ("move", "walk_to"),
        {"requested_distance": "distance asked for, m", "is_walk_to": "1 if walk_to, 0 if move"},
        {"distance_travelled": "distance actually moved, m", "objects_disturbed": "count of objects nudged",
         "height_change": "change in feet height, m", "left_ground": "1 if the agent left the ground, else 0"},
        _loco_row),
    "manipulation": DatasetSpec(
        "manipulation", "Can a given object be picked up (affordance by mass / motion)?", ("pick_up",),
        {"object_mass": "mass of the target object, kg", "relative_speed": "relative speed at grab, m/s",
         "was_moving": "1 if the object was moving, else 0", "agent_on_ground": "1 if the agent stood on something"},
        {"success": "1 if the pick-up succeeded, else 0"},
        _pick_row),
    "placement": DatasetSpec(
        "placement", "How accurately a placed object ends up where it was asked to go.", ("place", "put_down"),
        {"fall_distance": "how far the object fell after release, m"},
        {"displacement_from_requested": "distance from the requested spot, m",
         "stayed_where_placed": "1 if it stayed put, else 0", "settle_time": "seconds to come to rest"},
        _place_row),
    "affordance": DatasetSpec(
        "affordance", "The cross-action self-model: given the action and a little state, will it succeed? "
                      "A model of the agent's own capabilities (the 'cognitive' dataset).", _AFFORD_TOOLS,
        {"action_id": "which action, as an index into the action list", "feet_height": "feet height before, m",
         "has_target_object": "1 if the action named an object", "has_distance": "1 if a distance was given",
         "has_force": "1 if a force was given"},
        {"success": "1 if the action succeeded, else 0"},
        _afford_row),
}


def available_datasets() -> list[dict]:
    """Catalogue of the behaviour datasets, for the CLI, the dashboard and the agent tools."""
    return [{"name": s.name, "description": s.description, "tools": list(s.tools),
             "inputs": sorted(s.inputs), "targets": sorted(s.targets)} for s in DATASETS.values()]


def columns_for(name: str) -> tuple[dict, dict]:
    """The (input_columns, target_columns) vocabulary for a named dataset — pass to nets.validate_spec/train."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; available: {sorted(DATASETS)}")
    s = DATASETS[name]
    return s.inputs, s.targets


def vocab_from_manifest(manifest: dict) -> tuple[dict | None, dict | None]:
    """Derive the training vocabulary from a saved dataset's manifest.

    Behaviour datasets record their own inputs/targets; throw datasets do not (they use the default throw
    vocabulary). Returns (None, None) for a throw dataset so callers fall back to features.py. This lets
    one train path serve every dataset kind without hardcoding which columns are legal.
    """
    kind = manifest.get("kind")
    if kind in DATASETS:
        return columns_for(kind)
    ins = manifest.get("inputs")
    tgts = manifest.get("targets")
    if isinstance(ins, list) and isinstance(tgts, list) and ins and tgts:
        return {c: "" for c in ins}, {c: "" for c in tgts}
    return None, None


def build_dataset(actions: list[dict], name: str) -> dict:
    """Build a named behaviour dataset from `actions` rows (each with tool/args_json/result_json).

    Returns {"rows": [...], "manifest": {...}} just like the throw dataset, so it saves and trains the
    same way. Only successful, well-formed engine records contribute rows.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; available: {sorted(DATASETS)}")
    s = DATASETS[name]
    rows: list[dict] = []
    for a in actions:
        if a.get("tool") not in s.tools:
            continue
        # Database.actions() returns parsed dicts under `args`/`result`; raw JSON exports use `args_json`/`result_json`.
        raw_args = a.get("args")
        raw_result = a.get("result")
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(a.get("args_json") or "{}")
            except (TypeError, ValueError):
                args = {}
        if isinstance(raw_result, dict):
            result = raw_result
        else:
            try:
                result = json.loads(a.get("result_json") or "{}")
            except (TypeError, ValueError):
                result = {}
        # The engine's result normally carries "action"; guarantee it so builders that key off the tool
        # name (e.g. the cross-action affordance model) work even on records that omit it.
        result.setdefault("action", a.get("tool"))
        row = s.build_row(args, result)
        if row is not None:
            rows.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    return {"rows": rows,
            "manifest": {"source": f"behaviour:{name}", "kind": name, "rows": len(rows),
                         "inputs": sorted(s.inputs), "targets": sorted(s.targets),
                         "description": s.description}}


def refresh_log_datasets(db, reg) -> list[dict]:
    """Incremental refresh (Nov 2026): rebuild every log-derived behaviour dataset from the current
    action log and save it. Throw datasets are seed-generated (independent of the log) and are left
    alone. Returns one summary per refreshed dataset: {name, kind, rows, changed} where `changed`
    compares the rows hash against what was saved before, so the UI can flash real updates."""
    import hashlib as _hl
    import time as _time
    actions: list[dict] = []
    for r in db.q("SELECT id FROM sessions ORDER BY id"):
        actions.extend(db.actions(r["id"], limit=10 ** 9))
    refreshed: list[dict] = []
    for name in reg.list_datasets():
        try:
            existing = reg.load_dataset(name)
        except Exception:
            continue
        kind = (existing.get("manifest") or {}).get("kind")
        if kind not in DATASETS or not (existing.get("manifest") or {}).get("source", "").startswith("behaviour:"):
            continue
        ds = build_dataset(actions, kind)
        ds["manifest"]["rows_sha256"] = _hl.sha256(
            json.dumps(ds["rows"], sort_keys=True).encode()).hexdigest()
        ds["manifest"]["updated_at"] = _time.time()
        old_sha = (existing.get("manifest") or {}).get("rows_sha256")
        info = reg.save_dataset(name, ds)
        refreshed.append({"name": name, "kind": kind, "rows": len(ds["rows"]), "path": info["path"],
                          "changed": bool(ds["rows"]) and old_sha != ds["manifest"]["rows_sha256"]})
    return refreshed
