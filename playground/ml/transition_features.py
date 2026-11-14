"""General transition features: how ANY physical action changes the world.

The original ML lab learns one thing only — where a thrown object lands (features.py, throw rows). Audit
2026-11-13 found that this makes the network a throwing specialist by construction: its inputs and targets
name forces, elevations and flight ranges and nothing else, so it cannot represent — let alone learn —
movement, placement, climbing, or the relationships between objects, distances and positions.

This module adds the missing general case: a transition row pairs the world state BEFORE an action, the
action and its parameters, and the world state AFTER, measured by the engine. A network trained on these
rows learns a (state, action) -> next-state model of the island across every physical action, not just
throws. It is purely additive: the throw pipeline is untouched and remains available as a focused
special case for anyone who wants it.

Rows are built only from the engine's own action records (the `actions` table's result_json, which is the
tool result the world itself returned), never from anything an agent typed as prose.
"""
from __future__ import annotations

import json
import math
from typing import Any

# The physical actions whose before/after transition we can featurise. Observation, measurement and
# memory tools do not change the world, so they are not transitions.
TRANSITION_TOOLS = ("move", "walk_to", "jump", "climb", "pick_up", "put_down", "place", "throw", "catch",
                    "rotate_object")

# Inputs available before/at the action: the agent's own state and a compact description of the action.
TRANSITION_INPUTS = {
    "agent_x": "agent x position before the action, m",
    "agent_y": "agent y position before the action, m",
    "agent_z": "agent (feet) height before the action, m",
    "agent_speed": "agent speed before the action, m/s",
    "holding": "1 if the agent was holding an object before the action, else 0",
    "action_id": "which action this is, as an index into TRANSITION_TOOLS",
    "param_distance": "requested distance/magnitude of the action if any, m (0 when not applicable)",
    "param_dir_x": "x component of the action's direction if any (0 when not applicable)",
    "param_dir_y": "y component of the action's direction if any",
    "param_dir_z": "z component of the action's direction if any",
    "param_force": "force applied by the action if any, N (0 when not applicable)",
}
# Targets: how the world changed, measured by the engine.
TRANSITION_TARGETS = {
    "d_agent_x": "change in agent x, m",
    "d_agent_y": "change in agent y, m",
    "d_agent_z": "change in agent height, m",
    "agent_moved": "distance the agent moved, m",
    "object_moved": "distance the most-disturbed object moved, m (0 if none)",
    "left_ground": "1 if the agent left the ground during the action, else 0",
    "success": "1 if the action succeeded, else 0",
}
TRANSITION_ALL = {**TRANSITION_INPUTS, **TRANSITION_TARGETS}


def _f(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _vec3(v: Any) -> tuple[float, float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (_f(v[0], 0.0), _f(v[1], 0.0), _f(v[2], 0.0) if len(v) > 2 else 0.0)
    return (0.0, 0.0, 0.0)


def transition_row(tool: str, args: dict, result: dict, before: dict) -> dict | None:
    """Build one transition row from an action and the agent state sampled just before it.

    Args:
        tool: the tool name (must be in TRANSITION_TOOLS).
        args: the normalised tool arguments.
        result: the engine's tool result (the action's result_json).
        before: {"position": [x,y,z], "speed": s, "holding": bool} sampled before the action.

    Returns None for tools that are not transitions or results without usable position data.
    """
    if tool not in TRANSITION_TOOLS or not isinstance(result, dict):
        return None
    bx, by, bz = _vec3(before.get("position"))
    end = result.get("end_position") or result.get("agent_position") or result.get("position")
    ex, ey, ez = _vec3(end) if end is not None else (bx, by, bz)

    # The most-disturbed object, if the engine reported any.
    disturbed = result.get("objects_disturbed") or []
    obj_moved = 0.0
    for d in disturbed:
        obj_moved = max(obj_moved, _f(d.get("distance_moved"), 0.0) or 0.0)

    dx, dy, dz = args.get("direction"), None, None
    pdx, pdy, pdz = 0.0, 0.0, 0.0
    if isinstance(dx, (list, tuple)):
        pdx, pdy, pdz = _vec3(dx)

    row = {
        "agent_x": round(bx, 6), "agent_y": round(by, 6), "agent_z": round(bz, 6),
        "agent_speed": round(_f(before.get("speed"), 0.0) or 0.0, 6),
        "holding": 1.0 if before.get("holding") else 0.0,
        "action_id": float(TRANSITION_TOOLS.index(tool)),
        "param_distance": round(_f(args.get("distance"), 0.0) or 0.0, 6),
        "param_dir_x": round(pdx, 6), "param_dir_y": round(pdy, 6), "param_dir_z": round(pdz, 6),
        "param_force": round(_f(args.get("force"), 0.0) or 0.0, 6),
        "d_agent_x": round(ex - bx, 6), "d_agent_y": round(ey - by, 6), "d_agent_z": round(ez - bz, 6),
        "agent_moved": round(math.dist((bx, by, bz), (ex, ey, ez)), 6),
        "object_moved": round(obj_moved, 6),
        "left_ground": 1.0 if result.get("left_ground_during_move") or result.get("on_ground") is False else 0.0,
        "success": 1.0 if result.get("success", True) else 0.0,
    }
    return row


def dataset_from_actions(actions: list[dict]) -> dict:
    """Build a general transition dataset from a session's `actions` rows (each with args_json/result_json).

    The agent state 'before' each action is reconstructed from the previous action's end position, so the
    rows form a coherent trajectory. Returns {"rows": [...], "manifest": {...}} like the throw dataset.
    """
    rows: list[dict] = []
    before = {"position": [0.0, 0.0, 0.0], "speed": 0.0, "holding": False}
    for a in actions:
        tool = a.get("tool")
        try:
            args = json.loads(a.get("args_json") or "{}")
            result = json.loads(a.get("result_json") or "{}")
        except (TypeError, ValueError):
            continue
        row = transition_row(tool, args, result, before)
        if row is not None:
            rows.append(row)
        # Advance the 'before' state for the next action from whatever the engine last reported.
        end = result.get("end_position") or result.get("agent_position") or result.get("position")
        if end is not None:
            before = {"position": end,
                      "speed": result.get("speed", 0.0) if isinstance(result.get("speed"), (int, float)) else 0.0,
                      "holding": bool(result.get("holding") or result.get("carried") or before.get("holding"))}
    return {"rows": rows,
            "manifest": {"source": "transitions", "rows": len(rows), "kind": "transition",
                         "inputs": sorted(TRANSITION_INPUTS), "targets": sorted(TRANSITION_TARGETS)}}
