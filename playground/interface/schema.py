"""Tool catalogue: the complete, provider-neutral description of what an agent can do.

Descriptions say what a tool does and what its arguments mean. They deliberately
say nothing about how the world's physics behaves: that is for agents to find out.
"""
from __future__ import annotations

import json
import math
from typing import Any

from ..integrity import CLAIM_HELP

NUM = {"type": "number"}
STR = {"type": "string"}
BOOL = {"type": "boolean"}
INT = {"type": "integer"}
VEC = {"type": "array", "items": {"type": "number"}}
DIRECTION = {"anyOf": [STR, NUM, VEC],
             "description": "A compass name (north, south, east, west, northeast, ..., up, down), a heading in "
                            "degrees (0 = east, 90 = north), or a vector [dx, dy, dz]."}
TARGET = {"anyOf": [STR, VEC],
          "description": "'self', an object id, or a point [x, y, z]."}
STEPS = {"type": "array", "items": {"type": "object", "properties": {"tool": STR, "args": {"type": "object"}},
                                    "required": ["tool"]},
         "description": "Ordered tool calls: [{\"tool\": name, \"args\": {...}}, ...]."}


def _tool(name: str, category: str, description: str, props: dict | None = None,
          required: list[str] | None = None, program_ok: bool = True) -> dict:
    return {"name": name, "category": category, "description": description, "program_ok": program_ok,
            "parameters": {"type": "object", "properties": props or {}, "required": required or []}}


TOOLS: list[dict] = [
    # ---------------------------------------------------------------- observation
    _tool("observe_world", "observation",
          "Full structured snapshot: your body, every object and structure, and what rests on what."),
    _tool("inspect_object", "observation", "Everything known about one object.",
          {"object_id": STR}, ["object_id"]),
    _tool("inspect_position", "observation",
          "What is at a point: what occupies it, the surface below it, whether you can reach it.",
          {"x": NUM, "y": NUM, "z": NUM}, ["x", "y", "z"]),
    _tool("get_inventory", "observation", "What you are holding and the state of your body."),
    _tool("get_world_boundaries", "observation", "The coordinate system and the limits of the island."),
    _tool("get_object_positions", "observation", "Compact list of every object's id and position."),
    _tool("look_around", "observation",
          "What is near you: objects and ledges sorted by distance, with bearings and relative heights.",
          {"radius": {**NUM, "description": "Metres. Default 6."}}),
    # ---------------------------------------------------------------- movement
    _tool("move", "movement", "Walk in a straight line.",
          {"direction": DIRECTION, "distance": {**NUM, "description": "Metres."},
           "speed": {**NUM, "description": "Metres per second, 0.1 to 4. Default 2."},
           "keep_moving": {**BOOL, "description": "If true, do not brake at the end of the walk. Default false."}},
          ["direction", "distance"]),
    _tool("walk_to", "movement", "Walk in a straight line to a point. There is no path-finding.",
          {"x": NUM, "y": NUM, "z": {**NUM, "description": "Optional. The feet height you expect to end at."},
           "speed": NUM}, ["x", "y"]),
    _tool("climb", "movement", "Climb a ledge in a given direction.",
          {"direction": DIRECTION}, ["direction"]),
    # ---------------------------------------------------------------- joint control
    _tool("jump", "movement", "Jump.",
          {"wait": {"type": "string", "enum": ["landing", "apex", "takeoff"],
                    "description": "When this call returns, so that another action can follow at that moment. "
                                   "Default landing."}}),
    _tool("set_joint_target", "control", "Set a PD target for a named joint.",
          {"joint": STR, "position": NUM, "kp": NUM, "kd": NUM},
          ["joint", "position"]),
    _tool("wait", "movement", "Let time pass.", {"seconds": {**NUM, "description": "0 to 60."}}, ["seconds"]),

    # ------------------------------------------------- movement, physical extension (Phase 12)
    _tool("run_to", "movement", "Move to a point at a raised speed. There is no path-finding.",
          {"x": NUM, "y": NUM, "speed": {**NUM, "description": "Target speed in m/s. Default: a fast pace."}},
          ["x", "y"]),
    _tool("turn_to", "movement", "Face a compass heading given in degrees (0 = east, 90 = north).",
          {"heading": {**NUM, "description": "Degrees: 0 = east, 90 = north, 180 = west."}}, ["heading"]),
    _tool("stop_moving", "movement", "End the current walk and come to a standstill."),
    _tool("stand", "movement", "Hold a neutral standing posture for a while.",
          {"seconds": {**NUM, "description": "0 to 60. Default 2."}}),
    _tool("stand_up", "movement",
          "Attempt to rise from a collapsed or fallen pose. The outcome is reported as measured."),
    _tool("approach_object", "movement", "Walk until the named object is close. There is no path-finding.",
          {"object_id": STR,
           "stop_short": {**NUM, "description": "Metres to stop short of the object. Default 0.6."},
           "speed": {**NUM, "description": "Target speed in m/s."}}, ["object_id"]),

    # ------------------------------------------------- object, physical extension (Phase 12)
    _tool("reach_for", "object",
          "Point one arm toward a nearby object. The response carries the measured joint angles and distances.",
          {"object_id": STR}, ["object_id"]),
    _tool("carry_to", "object",
          "Take an object to a point: close in, take hold, walk there, set it down. Every part is reported.",
          {"object_id": STR, "x": NUM, "y": NUM,
           "speed": {**NUM, "description": "Target speed in m/s."}},
          ["object_id", "x", "y"]),

    # ------------------------------------------------- skills, physical extension (Phase 12)
    _tool("run_skill", "skill",
          "Execute one named skill from the physical skill catalogue. The response carries what physically happened.",
          {"skill": STR, "params": {"type": "object",
                                    "description": "Named arguments the skill accepts."}}, ["skill"]),
    _tool("run_chain", "skill",
          "Run skills in order, stopping at the first one that does not complete. Every step is reported.",
          {"steps": {"type": "array",
                     "items": {"type": "object",
                               "properties": {"skill": STR, "args": {"type": "object"}},
                               "required": ["skill"]}}}, ["steps"]),
    # ---------------------------------------------------------------- objects
    _tool("pick_up", "object", "Pick up an object. You can carry one object at a time.",
          {"object_id": STR}, ["object_id"]),
    _tool("put_down", "object", "Put the object you are holding down next to you.",
          {"object_id": {**STR, "description": "Optional; defaults to the held object."}}),
    _tool("throw", "object", "Throw the object you are holding.",
          {"object_id": STR, "direction": DIRECTION,
           "force": {**NUM, "description": "Newtons, greater than 0 and at most 200."},
           "wait": {"type": "string", "enum": ["rest", "impact", "none"],
                    "description": "Return when the object has come to rest (default), at its first impact, "
                                   "or immediately."}},
          ["object_id", "direction", "force"]),
    _tool("place", "object", "Set the object you are holding down at a position.",
          {"object_id": STR,
           "position": {**VEC, "description": "[x, y] to set it on top of whatever is there, or [x, y, z] to "
                                              "release it with its centre at exactly that point."}},
          ["object_id", "position"]),
    _tool("rotate_object", "object", "Turn an object you are holding or one within reach.",
          {"object_id": STR, "rotation": {**VEC, "description": "[roll, pitch, yaw] in degrees."}},
          ["object_id", "rotation"]),
    _tool("catch", "object", "Wait for a moving object to come within reach and grab it.",
          {"object_id": STR, "max_wait": {**NUM, "description": "Seconds to wait at most. Default 5."}},
          ["object_id"]),
    # ---------------------------------------------------------------- contact manipulation
    # Ways to move an object you are NOT holding. The object must be within reach and resting.
    _tool("push", "object", "Push an object that is within reach, away from you (or along a direction).",
          {"object_id": STR, "direction": {**DIRECTION, "description": DIRECTION["description"] +
                                            " Optional: without it, the object is pushed directly away from you."},
           "force": {**NUM, "description": "Newtons, 5 to 120. Default 40."},
           "wait": {"type": "string", "enum": ["rest", "impact", "none"]}},
          ["object_id"]),
    _tool("pull", "object", "Pull an object that is within reach towards you.",
          {"object_id": STR, "force": {**NUM, "description": "Newtons, 5 to 60. Default 25."},
           "wait": {"type": "string", "enum": ["rest", "impact", "none"]}},
          ["object_id"]),
    _tool("slide", "object", "Slide an object that is within reach along the ground in a direction.",
          {"object_id": STR, "direction": DIRECTION,
           "force": {**NUM, "description": "Newtons, 2 to 80. Default 30."},
           "wait": {"type": "string", "enum": ["rest", "impact", "none"]}},
          ["object_id", "direction"]),
    _tool("nudge", "object", "Give an object within reach a small gentle push.",
          {"object_id": STR, "direction": {**DIRECTION, "description": DIRECTION["description"] +
                                            " Optional: defaults to directly away from you."},
           "force": {**NUM, "description": "Newtons, 1 to 15. Default 6."},
           "wait": {"type": "string", "enum": ["rest", "impact", "none"]}},
          ["object_id"]),
    _tool("kick", "object", "Kick an object within reach, sending it off with some lift.",
          {"object_id": STR, "direction": {**DIRECTION, "description": DIRECTION["description"] +
                                            " Optional: defaults to directly away from you."},
           "force": {**NUM, "description": "Newtons, 20 to 160. Default 80."},
           "wait": {"type": "string", "enum": ["rest", "impact", "none"]}},
          ["object_id"]),
    _tool("roll", "object", "Roll an object within reach along the ground (works best on round objects).",
          {"object_id": STR, "direction": DIRECTION,
           "force": {**NUM, "description": "Newtons, 5 to 120. Default 40."},
           "wait": {"type": "string", "enum": ["rest", "impact", "none"]}},
          ["object_id", "direction"]),
    # ---------------------------------------------------------------- combat (arena worlds only)
    # These tools are physical: they hold real objects through the ordinary carry
    # hold and every consequence follows the measured engine events. They are only
    # meaningful where a combat configuration exists; elsewhere they raise no_combat.
    _tool("equip", "combat", "Take hold of an equipment object lying within reach (it uses your one carry hold).",
          {"object_id": STR}, ["object_id"]),
    _tool("unequip", "combat", "Release the equipment object you are holding and put it down.",
          {"object_id": {**STR, "description": "Optional; defaults to the equipped object."}}),
    _tool("combat_state", "combat", "Your combat state: health, incapacity, team and what you are holding."),
    _tool("fire", "combat", "Fire the launcher you are holding toward a direction.",
          {"direction": DIRECTION}, ["direction"]),
    _tool("reload", "combat",
          "Reload the launcher you are holding; it becomes ready again after the item's reload time."),
    _tool("attack", "combat", "Swing the melee weapon you are holding through the target direction.",
          {"direction": DIRECTION}, ["direction"]),
    _tool("defend", "combat",
          "Raise the shield you are holding into a guard pose; stance 'carry' returns it to the carry pose.",
          {"stance": {"type": "string", "enum": ["guard", "carry"]}}),
    # ---------------------------------------------------------------- construction (convenience)
    # Thin, discoverable wrappers over pick_up + carry + place. They encode no strategy: they just
    # spare an agent the exact release-coordinate arithmetic, so stacking/lining up is as cheap to
    # TRY as throwing. Whether they help is for the agent to judge from the result.
    _tool("stack_on", "object", "Put the object you are holding on top of another object.",
          {"object_id": {**STR, "description": "Optional; defaults to the held object."},
           "target_id": {**STR, "description": "The object to stack on top of."}},
          ["target_id"]),
    _tool("build_row", "object",
          "Place the object you are holding next to another one, forming or extending a row on the ground.",
          {"object_id": {**STR, "description": "Optional; defaults to the held object."},
           "beside_id": {**STR, "description": "The object to place next to."},
           "direction": {**DIRECTION, "description": DIRECTION["description"] +
                         " Which way from the neighbour to place it. Optional; defaults to the open side."}},
          ["beside_id"]),
    # ---------------------------------------------------------------- experimentation
    _tool("measure_height", "experiment", "Measure heights of yourself, an object or the surface at a point.",
          {"target": TARGET}),
    _tool("measure_distance", "experiment", "Measure the distance and bearing between two things.",
          {"from": TARGET, "to": TARGET}, ["to"]),
    _tool("measure_velocity", "experiment", "Measure the current velocity of yourself or an object.",
          {"target": TARGET}),
    _tool("measure_flight_time", "experiment",
          "The engine's measurements of the most recent flight of an object, or of your own last jump or fall.",
          {"target": TARGET}),
    _tool("inspect_collision", "experiment", "The engine's records of recent collisions.",
          {"object_id": {**STR, "description": "Optional: only collisions involving this object."},
           "limit": {**INT, "description": "Default 5."}}),
    _tool("run_physics_test", "experiment",
          "Try a sequence of actions in a private copy of the world as it is right now. The real world is not "
          "changed. Returns what each step measured.",
          {"parameters": {"type": "object", "properties": {"steps": STEPS, "label": STR}, "required": ["steps"]}},
          ["parameters"], program_ok=False),
    # ---------------------------------------------------------------- world history and programs
    _tool("inspect_world_history", "world",
          "Search this session's record of physical events (throws, landings, collisions, appearances...).",
          {"types": {"type": "array", "items": STR}, "object_id": STR,
           "since_action": INT, "limit": {**INT, "description": "Default 30."}}),
    _tool("inspect_recent_events", "world", "The most recent physical events.",
          {"limit": {**INT, "description": "Default 12."}}),
    _tool("save_program", "world",
          "Save a reusable sequence of tool calls. Use ${name} inside args for values supplied when it is run. "
          "Programs persist between sessions.",
          {"name": STR, "description": STR, "steps": STEPS}, ["name", "steps"], program_ok=False),
    _tool("execute_program", "world", "Run a saved program.",
          {"name": STR, "params": {"type": "object", "description": "Values for ${name} placeholders."},
           "repeat": {**INT, "description": "1 to 20. Default 1."},
           "on_error": {"type": "string", "enum": ["stop", "continue"]}}, ["name"], program_ok=False),
    _tool("list_programs", "world", "List your saved programs.", program_ok=False),
    # ---------------------------------------------------------------- memory and reflection
    _tool("read_memory", "memory", "Read one of your memory files.",
          {"file": {"type": "string", "enum": ["handbook", "diary", "mistakes", "goals", "discoveries",
                                               "experiments", "programs"]}}, ["file"], program_ok=False),
    _tool("write_diary", "memory", "Add a diary entry.", {"text": STR}, ["text"], program_ok=False),
    _tool("add_note", "memory", "Add a free-form working note to your handbook.", {"text": STR}, ["text"],
          program_ok=False),
    _tool("record_experiment", "memory",
          "Record an experiment. Cite the action_index numbers of the actions that produced the evidence; the "
          "record is checked against what those actions actually measured.",
          {"question": {**STR, "description": "What you wanted to find out."},
           "hypothesis": STR, "action_description": {**STR, "description": "What you did."},
           "conditions": {**STR, "description": "Experimental conditions: starting positions, held objects, "
                            "settings that differed between trials."},
           "expected_result": STR, "actual_result": STR,
           "measurements": {"type": "object", "description": "Named numeric results, e.g. {\"max_height\": 3.1}."},
           "action_refs": {"type": "array", "items": INT},
           "discovery_id": {**STR, "description": "Optional handbook entry (D-…) this experiment tests."},
           "supported": {**BOOL, "description": "Whether the result supports the hypothesis."},
           "needs_followup": BOOL, "conclusion": STR},
          ["question", "hypothesis", "actual_result", "action_refs", "supported"], program_ok=False),
    _tool("record_discovery", "memory",
          "Add an entry to your handbook. The status you ask for is granted only if the cited experiments "
          "support it; otherwise the entry is filed under the status the evidence justifies.",
          {"statement": STR,
           "status": {"type": "string", "enum": ["confirmed", "probable", "hypothesis", "disproven"]},
           "experiment_ids": {"type": "array", "items": STR}, "category": STR},
          ["statement", "status"], program_ok=False),
    _tool("update_discovery", "memory", "Add evidence or a note to a handbook entry, or ask for a new status.",
          {"discovery_id": STR,
           "status": {"type": "string", "enum": ["confirmed", "probable", "hypothesis", "disproven"]},
           "add_experiment_ids": {"type": "array", "items": STR}, "note": STR}, ["discovery_id"],
          program_ok=False),
    _tool("record_mistake", "memory", "Record a mistake and what you learned from it.",
          {"what": STR, "why": STR, "lesson": STR, "action_refs": {"type": "array", "items": INT}}, ["what"],
          program_ok=False),
    _tool("set_goal", "memory", "Note an objective you have chosen for yourself. Use 'kind' to say what "
          "it is: a question you want answered, a hypothesis you want to test, or a thing you want to "
          "build or do. 'next_step' is the very next action you intend to take towards it.",
          {"title": STR, "description": STR,
           "reason": {**STR, "description": "Why you are setting this goal now (optional)."},
           "kind": {"type": "string", "enum": ["question", "hypothesis", "build", "explore", "objective"],
                    "description": "What sort of goal this is. Default: objective."},
           "next_step": {**STR, "description": "The next concrete action you plan to take towards it."}},
          ["title"], program_ok=False),
    _tool("update_goal", "memory",
          "Change a goal's status. A goal marked completed can carry a claim, which is checked independently.",
          {"goal_id": STR, "status": {"type": "string", "enum": ["active", "paused", "completed", "abandoned"]},
           "note": STR,
           "claim": {"type": "object", "description": "Optional: {\"type\": ..., \"params\": {...}}. Give "
                                                      "params the exact parameter names documented for "
                                                      "claim_achievement; unknown names are rejected, "
                                                      "never ignored."}},
          ["goal_id", "status"], program_ok=False),
    _tool("assign_meaning", "memory",
          "Give an object, or a group of objects, a name or meaning of your own.",
          {"target": {"anyOf": [STR, {"type": "array", "items": STR}]}, "meaning": STR}, ["target", "meaning"],
          program_ok=False),
    _tool("claim_achievement", "memory",
          "State that you have achieved something and have it checked against the physical record. Give "
          "params a dict with exactly the parameter names the type expects (checked strictly: wrong or "
          "unknown names are rejected as invalid_argument, never ignored). Types and their exact params: "
          + "; ".join(f"{k}: {v}" for k, v in CLAIM_HELP.items()),
          {"type": STR, "params": {"type": "object", "description": "Keys must match the parameter names "
                                    "listed for the chosen type in this description; used verbatim by "
                                    "the independent verifier."}, "description": STR},
          ["type"], program_ok=False),
    _tool("end_session", "memory", "Finish for now. Your memory files are kept; the island is reset. "
          "If you want a follow-up session with this same memory, set continue_next_session=true; "
          "otherwise the session simply ends.",
          {"reason": STR, "continue_next_session": {"type": "boolean",
           "description": "True if you want to resume exploring in a follow-up session."}},
          program_ok=False),
    # ---------------------------------------------------------------- learning (neural networks)
    _tool("nn_list", "learning",
          "List your trained models and datasets. Shows available input columns and prediction targets.",
          program_ok=False),
    _tool("nn_make_dataset", "learning",
          "Build a dataset from your own engine-recorded actions. kind='throw' (default) learns throw "
          "outcomes; 'climb','jump','locomotion','manipulation','placement','affordance' learn other "
          "behaviours (affordance = a self-model of which actions succeed). nn_list shows the catalogue.",
          {"name": {**STR, "description": "1-40 chars: letters, digits, underscores, hyphens."},
           "kind": {**STR, "description": "throw (default), climb, jump, locomotion, manipulation, "
                                          "placement, or affordance."},
           "min_rows": {**INT, "description": "Minimum usable rows required. Default 20."}},
          ["name"], program_ok=False),
    _tool("nn_train", "learning",
          "Train a neural network on one of your datasets. Training happens on CPU.",
          {"dataset_name": STR,
           "spec": {"type": "object", "description": "Model spec: {name, inputs: [...], target, hidden: [...]}"},
           "train_config": {"type": "object", "description": "Optional: {epochs, lr, batch_size, patience}."}},
          ["dataset_name", "spec"], program_ok=False),
    _tool("nn_predict", "learning",
          "Make predictions using one of your trained models. Results are flagged is_prediction=true.",
          {"model_name": STR, "inputs": {"type": "array", "items": {"type": "object"}},
           "version": {**INT, "description": "Model version. Default: latest."}},
          ["model_name", "inputs"], program_ok=False),
    _tool("nn_evaluate", "learning",
          "Evaluate a model against fresh throws from this session.",
          {"model_name": STR, "version": {**INT, "description": "Model version. Default: latest."}},
          ["model_name"], program_ok=False),
]

SPECS: dict[str, dict] = {t["name"]: t for t in TOOLS}
# "control" (set_joint_target) and "skill" (run_skill) were used by tools but
# missing from this enumeration (M6-A audit finding); downstream category
# listings now match the real tool set.
CATEGORIES = ("observation", "movement", "object", "combat", "experiment", "world", "memory",
              "learning", "control", "skill")


class ToolError(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


def openai_tools(names: list[str] | None = None) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["parameters"]}}
            for t in TOOLS if names is None or t["name"] in names]


def anthropic_tools(names: list[str] | None = None) -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOLS if names is None or t["name"] in names]


def catalogue() -> list[dict]:
    return [{"name": t["name"], "category": t["category"], "description": t["description"],
             "parameters": t["parameters"]} for t in TOOLS]


def _coerce(value: Any, schema: dict) -> Any:
    """Be forgiving about encodings models commonly produce ('[1,2]' strings, {'x':..} dicts)."""
    if isinstance(value, str):
        v = value.strip()
        kinds = {schema.get("type")} | {s.get("type") for s in schema.get("anyOf", [])}
        if v[:1] in "[{" and kinds & {"array", "object"}:
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return value
        if schema.get("type") in ("number", "integer"):
            try:
                return int(v) if schema["type"] == "integer" else float(v)
            except ValueError:
                return value
        if schema.get("type") == "boolean" and v.lower() in ("true", "false"):
            return v.lower() == "true"
    if isinstance(value, dict) and schema.get("type") == "array" and {"x", "y"} <= set(value):
        return [value["x"], value["y"]] + ([value["z"]] if "z" in value else [])
    return value


def normalise_args(tool: str, args: Any) -> dict:
    spec = SPECS[tool]
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolError("invalid_argument", f"Arguments for {tool} are not valid JSON: {exc}.") from None
    if not isinstance(args, dict):
        raise ToolError("invalid_argument", f"Arguments for {tool} must be an object of named values.")
    props = spec["parameters"]["properties"]
    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError("invalid_argument", f"{tool} has no argument(s) {unknown}. It accepts: {sorted(props)}.",
                        accepted=sorted(props))
    missing = [r for r in spec["parameters"]["required"] if args.get(r) is None]
    if missing:
        raise ToolError("invalid_argument", f"{tool} needs argument(s) {missing}.", required=missing)
    out = {}
    for k, v in args.items():
        if v is None:
            continue
        v = _coerce(v, props[k])
        typ = props[k].get("type")
        ok = {"number": isinstance(v, (int, float)) and not isinstance(v, bool),
              "integer": isinstance(v, int) and not isinstance(v, bool),
              "string": isinstance(v, str), "boolean": isinstance(v, bool),
              "array": isinstance(v, list), "object": isinstance(v, dict)}.get(typ, True)
        if typ == "integer" and isinstance(v, float) and v.is_integer():
            v, ok = int(v), True
        if not ok:
            raise ToolError("invalid_argument", f"Argument '{k}' of {tool} must be of type {typ}.",
                            argument=k, received=repr(v)[:80])
        # B1: JSON admits NaN/Infinity literals; numbers must be finite at the tool
        # boundary. (Downstream world validation rejects them too — this catches
        # them before any tool body runs.)
        if typ in ("number", "integer") and isinstance(v, (int, float)) and not math.isfinite(float(v)):
            raise ToolError("invalid_argument", f"Argument '{k}' of {tool} must be a finite number.",
                            argument=k, received=repr(v)[:80])
        if "enum" in props[k] and v not in props[k]["enum"]:
            raise ToolError("invalid_argument", f"Argument '{k}' of {tool} must be one of {props[k]['enum']}.",
                            argument=k, received=repr(v)[:80])
        out[k] = v
    return out
