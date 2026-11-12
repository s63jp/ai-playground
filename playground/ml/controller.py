"""A neural-network agent brain.

The controller drives a body through exactly the same tool interface as the language-model agents
(so the same verifiers, logs and benchmarks apply). It is a hybrid, and says so:

  * a small fixed routine handles logistics (walk to a block, pick it up, walk to a clear spot);
  * every decision about *how to throw* is made by a trained network used as an inverse model:
    candidate (force, jump) actions are scored by the network's predicted peak, and the best is executed.

The routine contains no knowledge of the physics. Whatever the controller "knows" about throwing
is in the network's weights, which were learned from engine data.

Currently implemented task: the `throw_height` benchmark (land a throw's peak within tolerance of a target).
For other tasks the controller observes, reports that it has no policy for the task, and ends.
"""
from __future__ import annotations

import math
import re

from . import MLError
from .nets import Predictor

CLEAR_SPOT = (-9.5, 0.0)          # open ground on the west side, away from every structure
ELEVATION = 80.0                  # degrees above horizontal, thrown east


def _direction(elev_deg: float) -> list[float]:
    e = math.radians(elev_deg)
    return [round(math.cos(e), 6), 0.0, round(math.sin(e), 6)]


def make_controller_policy(model: dict, max_attempts: int = 4):
    """Return a policy generator function for PolicyAdapter."""
    pred = Predictor(model)
    if pred.spec["target"] != "max_height" or "force" not in pred.spec["inputs"]:
        raise MLError("wrong_model", "The neural controller needs a model that predicts max_height from force.")

    def policy(system: str, opening: str, tools: list[str]):
        m = re.search(r"Target height: (\d+(?:\.\d+)?)", opening)
        if not m:
            yield ("say", "I am a neural-network controller; I only have a policy for throw-height tasks.")
            yield ("call", "observe_world", {})
            yield ("call", "end_session", {"reason": "no learned policy for this task"})
            return
        target = float(m.group(1))
        obs = yield ("call", "observe_world", {})
        blocks = [o for o in obs.get("objects", []) if o.get("kind") == "block" and o.get("movable")]
        if not blocks:
            yield ("call", "end_session", {"reason": "no block to throw"})
            return
        ax, ay = obs["agent"]["position"][:2]
        blocks.sort(key=lambda o: math.hypot(o["position"][0] - ax, o["position"][1] - ay))
        attempts = 0
        for b in blocks:
            if attempts >= max_attempts:
                break
            bx, by = b["position"][:2]
            # stand on the near side of the block, then pick it up
            r = yield ("call", "walk_to", {"x": round(bx + 0.9, 3), "y": round(by, 3)})
            r = yield ("call", "pick_up", {"object_id": b["id"]})
            if not r.get("success"):
                r = yield ("call", "walk_to", {"x": round(bx, 3), "y": round(by + 0.9, 3)})
                r = yield ("call", "pick_up", {"object_id": b["id"]})
                if not r.get("success"):
                    continue
            yield ("call", "walk_to", {"x": CLEAR_SPOT[0], "y": CLEAR_SPOT[1]})
            inv = yield ("call", "get_inventory", {})
            body = inv.get("agent") or {}
            feet = float(body.get("feet_height", 0.0) or 0.0)
            mass = float(inv.get("carried_mass") or 5.0)
            # ---- the network decides: score every candidate (force, jump) and take the closest prediction
            cands = []
            d = _direction(ELEVATION)
            for jump, vz in ((False, 0.0), (True, 3.02)):
                for f in [20.0 + 2.0 * i for i in range(91)]:
                    cands.append({"force": f, "jump": jump, "row": {
                        "force": f, "dir_x": d[0], "dir_y": d[1], "dir_z": d[2], "elevation_deg": ELEVATION,
                        "agent_vx": 0.0, "agent_vy": 0.0, "agent_vz": vz, "object_mass": mass,
                        "feet_height": feet, "on_ground": 0.0 if jump else 1.0,
                        "release_height": feet + 1.2}})
            preds = pred.predict_rows([c["row"] for c in cands])
            best = min(range(len(cands)), key=lambda i: abs(preds[i] - target))
            c = cands[best]
            yield ("say", f"My network predicts a peak of {preds[best]:.3f} m for force {c['force']:.0f} N"
                          f"{' released at take-off' if c['jump'] else ' standing'}; target {target} m.")
            if c["jump"]:
                yield ("call", "jump", {"wait": "takeoff"})
            t = yield ("call", "throw", {"object_id": b["id"], "direction": d, "force": c["force"]})
            attempts += 1
            peak = t.get("max_height")
            if peak is not None and abs(peak - target) <= 0.25:
                yield ("say", f"Measured peak {peak} m, within tolerance.")
                break
        yield ("call", "end_session", {"reason": "neural controller finished"})

    return policy


def make_controller_adapter(model: dict, label: str | None = None):
    from ..agents.adapters import PolicyAdapter
    name = model["spec"]["name"]
    return PolicyAdapter(make_controller_policy(model), label=label or "neural",
                         model=f"nn:{model.get('owner', 'researcher')}/{name}/v{model.get('version', '?')}")
