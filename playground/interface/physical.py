"""Physical tools: observation, movement, object handling and measurement.

Bound to a PhysicsWorld and nothing else, so exactly the same code serves the real
world and the private sandbox copy used by run_physics_test.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..integrity.claims import find_structures, find_towers
from ..world.mathutil import compass, heading_deg, parse_direction, rnd
from ..world.physics import ActionError, PhysicsWorld
from .schema import ToolError

AGENT_EVENT_TYPES = ("throw_release", "flight_end", "collision", "agent_landing", "jump_takeoff", "pickup",
                     "catch", "release", "climb", "rotate", "object_appeared", "object_lost")


def visible_event(ev: dict) -> dict:
    """An event as an agent may see it: measurements only, never the researcher-side annotations."""
    return {"seq": ev.get("n", ev.get("seq")), "t": ev["t"], "type": ev["type"], "data": ev["data"],
            **({"action_index": ev["action_idx"]} if ev.get("action_idx") is not None else {})}


class PhysicalTools:
    def __init__(self, world: PhysicsWorld):
        self.w = world

    # ------------------------------------------------------------ helpers
    def _direction(self, direction) -> np.ndarray:
        try:
            return parse_direction(direction)
        except ValueError as exc:
            raise ToolError("invalid_argument", str(exc)) from None

    def _point(self, target, default_self: bool = True) -> tuple[np.ndarray, str]:
        w = self.w
        if target is None and default_self:
            target = "self"
        if isinstance(target, str):
            key = target.strip()
            if key.lower() in ("self", "me", "agent", "body"):
                return w.agent_pos().copy(), "self"
            if key in w.specs and (not w.specs[key].movable or key in w.active):
                if w.specs[key].movable:
                    return w.obj_pos(key), key
                return np.array(w.specs[key].position, dtype=float), key
            raise ToolError("unknown_object", f"There is no object with id '{key}'.",
                            known_ids=sorted(w.visible_ids()))
        try:
            vals = [float(v) for v in target]
        except (TypeError, ValueError):
            raise ToolError("invalid_argument", "A target is 'self', an object id, or a point [x, y, z].") from None
        if len(vals) == 2:
            top, _ = w.ray_down(vals[0], vals[1], 60.0)
            vals.append(top if top is not None else 0.0)
        if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
            raise ToolError("invalid_argument", "A point needs 2 or 3 finite numbers.")
        return np.array(vals), "point"

    # ------------------------------------------------------------ observation
    def observe_world(self) -> dict:
        w = self.w
        graph = w.support_graph()
        objects = []
        for oid in w.visible_ids():
            s = w.specs[oid]
            if s.movable:
                p = w.obj_pos(oid)
                hz = w.half_extent_z(oid)
                objects.append({"id": oid, "name": s.name, "kind": s.kind, "movable": True,
                                "position": rnd(p, 3), "top_height": round(float(p[2] + hz), 3),
                                "resting_on": [] if oid == w.held else graph.get(oid, []),
                                "held": oid == w.held, "moving": w.obj_speed(oid) > 0.05})
            else:
                objects.append({"id": oid, "name": s.name, "kind": s.kind, "movable": False,
                                "position": rnd(list(s.position), 3), "size": dict(s.dimensions),
                                "top_height": s.properties.get("top_height")})
        return {"agent": w.agent_state(), "objects": objects,
                "towers": [t for t in find_towers(w) if t["height_blocks"] >= 2],
                "structures": find_structures(w),
                "objects_no_longer_in_world": sorted(w.lost),
                "coordinate_system": "x east, y north, z up; metres; origin at the island centre, ground level"}

    def inspect_object(self, object_id: str) -> dict:
        try:
            return {"object": self.w.object_state(object_id)}
        except ActionError as e:
            raise ToolError(e.code, e.message, **e.details) from None

    def inspect_position(self, x: float, y: float, z: float) -> dict:
        if not all(math.isfinite(float(v)) for v in (x, y, z)):
            raise ToolError("invalid_argument", "x, y and z must be finite numbers.")
        return self.w.point_info(float(x), float(y), float(z))

    def get_inventory(self) -> dict:
        w = self.w
        held = w.object_state(w.held) if w.held else None
        return {"holding": w.held, "held_object": held, "hands_free": w.held is None,
                "carried_mass": round(w.total_mass() - w.base_mass, 3), "agent": w.agent_state()}

    def get_world_boundaries(self) -> dict:
        return self.w.boundaries()

    def get_object_positions(self) -> dict:
        w = self.w
        out = {}
        for oid in w.visible_ids():
            s = w.specs[oid]
            out[oid] = rnd(w.obj_pos(oid), 3) if s.movable else rnd(list(s.position), 3)
        return {"agent": rnd(w.agent_pos(), 3), "objects": out, "count": len(out)}

    def look_around(self, radius: float = 6.0) -> dict:
        w = self.w
        if not (0 < radius <= 40):
            raise ToolError("invalid_argument", "radius must be between 0 and 40 metres.")
        ap, feet = w.agent_pos(), w.feet()
        near, ledges = [], []
        for oid in w.visible_ids():
            s = w.specs[oid]
            if s.kind == "ground":
                continue
            if s.movable:
                p = w.obj_pos(oid)
                top = float(p[2] + w.half_extent_z(oid))
                d = float(np.hypot(p[0] - ap[0], p[1] - ap[1]))
                bearing = heading_deg(p - ap) % 360
            else:
                c, dm = np.array(s.position), s.dimensions
                nx = min(max(ap[0], c[0] - dm["x"] / 2), c[0] + dm["x"] / 2)
                ny = min(max(ap[1], c[1] - dm["y"] / 2), c[1] + dm["y"] / 2)
                d = float(np.hypot(nx - ap[0], ny - ap[1]))
                top = float(s.properties.get("top_height", c[2]))
                p = c
                bearing = heading_deg(np.array([nx - ap[0], ny - ap[1]])) % 360 if d > 1e-6 else 0.0
            if d > radius:
                continue
            entry = {"id": oid, "name": s.name, "kind": s.kind, "distance": round(d, 3),
                     "bearing_degrees": round(bearing, 1), "direction": compass(bearing),
                     "top_height_above_feet": round(top - feet, 3)}
            if s.movable:
                entry["within_reach"] = w._in_reach(p)
                entry["held"] = oid == w.held
                near.append(entry)
            else:
                ledges.append(entry)
        near.sort(key=lambda e: e["distance"])
        ledges.sort(key=lambda e: e["distance"])
        return {"position": rnd(ap, 3), "feet_height": round(feet, 3), "facing_degrees": round(w.facing, 1),
                "standing_on": list(w.support), "radius": radius, "objects": near, "fixed_structures": ledges,
                "distance_to_island_edge": {k: round(v, 3) for k, v in (
                    ("east", w.half - ap[0]), ("west", w.half + ap[0]),
                    ("north", w.half - ap[1]), ("south", w.half + ap[1]))}}

    # ------------------------------------------------------------ movement
    def move(self, direction, distance: float, speed: float | None = None, keep_moving: bool = False) -> dict:
        return self.w.move(self._direction(direction), float(distance), speed, stop=not keep_moving)

    def walk_to(self, x: float, y: float, z: float | None = None, speed: float | None = None) -> dict:
        w = self.w
        ap = w.agent_pos()
        if abs(x) > w.half or abs(y) > w.half:
            raise ToolError("outside_world", f"({x}, {y}) is beyond the island edge (limits are "
                            f"+/-{w.half} in x and y).", limits=[-w.half, w.half])
        d = np.array([x - ap[0], y - ap[1], 0.0])
        dist = float(np.linalg.norm(d))
        if dist < 0.02:
            res = {"start_position": rnd(ap), "end_position": rnd(ap), "distance_travelled": 0.0,
                   "note": "Already there."}
        else:
            res = w.move(d / dist, dist, speed, stop=True)
        res["target"] = [x, y] + ([z] if z is not None else [])
        end = w.agent_pos()
        short = float(np.hypot(end[0] - x, end[1] - y))
        res["arrived"] = bool(short < 0.1)
        if not res["arrived"]:
            # Never leave "not arrived" open to interpretation: the walk has finished, here is how far off it is.
            res["remaining_distance"] = round(short, 4)
            res["note"] = (f"The walk has finished but ended {short:.2f} m from the target (the body was pushed "
                           f"off the straight line). Nothing is still moving you: every action is complete when "
                           f"its result is returned. Call walk_to again to cover the rest.")
        res["still_moving"] = False
        if z is not None:
            res["feet_height"] = round(w.feet(), 4)
            res["height_matches_target"] = bool(abs(w.feet() - float(z)) < 0.1)
        return res

    def jump(self, wait: str = "landing") -> dict:
        return self.w.jump(wait)

    def climb(self, direction=None) -> dict:
        return self.w.climb(None if direction is None else self._direction(direction))

    def wait(self, seconds: float) -> dict:
        return self.w.wait(float(seconds))

    # ------------------------------------------------------------ objects
    def pick_up(self, object_id: str) -> dict:
        return self.w.pick_up(object_id)

    def put_down(self, object_id: str | None = None) -> dict:
        return self.w.put_down(object_id)

    def throw(self, object_id: str, direction, force: float, wait: str = "rest") -> dict:
        return self.w.throw(object_id, self._direction(direction), float(force), wait)

    def place(self, object_id: str, position) -> dict:
        return self.w.place(object_id, position)

    def rotate_object(self, object_id: str, rotation) -> dict:
        return self.w.rotate_object(object_id, rotation)

    def catch(self, object_id: str, max_wait: float = 5.0) -> dict:
        return self.w.catch(object_id, float(max_wait))

    # ------------------------------------------------------------ measurement
    def measure_height(self, target=None) -> dict:
        w = self.w
        p, what = self._point(target)
        if what == "self":
            below, surf = w.ray_down(p[0], p[1], p[2] - w.hh + 0.05)
            return rnd({"target": "self", "feet_height": w.feet(), "head_height": p[2] + w.hh,
                        "height_above_surface_below": None if below is None else w.feet() - below,
                        "surface_below": surf, "on_ground": w.grounded})
        if what == "point":
            top, surf = w.ray_down(p[0], p[1], 60.0)
            return rnd({"target": [p[0], p[1]], "surface_height": top, "surface": surf,
                        "height_above_agent_feet": None if top is None else top - w.feet()})
        s = w.specs[what]
        if not s.movable:
            top = s.properties.get("top_height")
            return rnd({"target": what, "top_height": top, "base_height": s.position[2] - s.dimensions["z"] / 2,
                        "top_height_above_agent_feet": top - w.feet()})
        hz = w.half_extent_z(what)
        below, surf = w.ray_down(p[0], p[1], p[2] - hz - 0.01)
        return rnd({"target": what, "centre_height": p[2], "bottom_height": p[2] - hz, "top_height": p[2] + hz,
                    "height_above_surface_below": None if below is None else p[2] - hz - below,
                    "surface_below": surf, "top_height_above_agent_feet": p[2] + hz - w.feet()})

    def measure_distance(self, to, **kw) -> dict:
        a, an = self._point(kw.get("from"))
        b, bn = self._point(to, default_self=False)
        d = b - a
        horiz = float(np.hypot(d[0], d[1]))
        return rnd({"from": an if an != "point" else a, "to": bn if bn != "point" else b,
                    "from_position": a, "to_position": b, "distance": float(np.linalg.norm(d)),
                    "horizontal_distance": horiz, "vertical_difference": float(d[2]),
                    "bearing_degrees": (heading_deg(d) % 360) if horiz > 1e-6 else None,
                    "direction": compass(heading_deg(d)) if horiz > 1e-6 else None})

    def measure_velocity(self, target=None) -> dict:
        w = self.w
        _, what = self._point(target)
        if what == "self":
            v = w.agent_vel()
            return rnd({"target": "self", "velocity": v, "speed": float(np.linalg.norm(v)),
                        "horizontal_speed": float(np.hypot(v[0], v[1])), "vertical_speed": float(v[2]),
                        "on_ground": w.grounded})
        if what == "point" or not w.specs[what].movable:
            return {"target": what if what != "point" else "point", "velocity": [0.0, 0.0, 0.0], "speed": 0.0,
                    "note": "Fixed points and structures do not move."}
        v, om = w.obj_vel(what), w.obj_angvel(what)
        return rnd({"target": what, "velocity": v, "speed": float(np.linalg.norm(v)),
                    "horizontal_speed": float(np.hypot(v[0], v[1])), "vertical_speed": float(v[2]),
                    "angular_velocity": om, "held": what == w.held})

    def measure_flight_time(self, target=None) -> dict:
        w = self.w
        _, what = self._point(target)
        if what == "self":
            if w._air is not None:
                return rnd({"target": "self", "in_flight": True, "airborne_for": w.time - w._air["t0"],
                            "max_feet_height_so_far": w._air["max_feet"]})
            if w.last_air is None:
                raise ToolError("no_flight_recorded", "Your body has not been airborne yet in this session.")
            return {"target": "self", "in_flight": False, **w.last_air}
        if what == "point":
            raise ToolError("invalid_argument", "measure_flight_time needs 'self' or an object id.")
        fl = w.flights.get(what)
        if fl is not None and fl["first_impact"] is None:
            return rnd({"target": what, "in_flight": True, "airborne_for": w.time - fl["t0"],
                        "max_height_so_far": fl["max_height"], "flight_id": fl["flight_id"]})
        for rec in reversed(w.completed_flights):
            if rec["object"] == what:
                return {"target": what, "in_flight": False, **rec}
        if fl is not None:
            fi = fl["first_impact"]
            return rnd({"target": what, "in_flight": False, "still_settling": True, "flight_id": fl["flight_id"],
                        "flight_time": fi["t"] - fl["t0"], "max_height": fl["max_height"],
                        "first_impact_position": fi["position"], "first_impact_with": fi["with"],
                        "impact_speed": fi["speed"]})
        raise ToolError("no_flight_recorded", f"'{what}' has not been thrown or released in this session.")

    def inspect_collision(self, object_id: str | None = None, limit: int = 5) -> dict:
        w = self.w
        if object_id is not None and object_id not in w.specs and object_id != "agent":
            raise ToolError("unknown_object", f"There is no object with id '{object_id}'.",
                            known_ids=sorted(w.visible_ids()))
        limit = max(1, min(int(limit), 50))
        found = []
        for ev in reversed(w.events):
            if ev["type"] != "collision":
                continue
            if object_id is None or object_id in ev["data"]["objects"]:
                found.append({"t": ev["t"], **ev["data"]})
                if len(found) >= limit:
                    break
        return {"object_id": object_id, "collisions": found, "count": len(found),
                "note": None if found else "No collisions above the recording threshold match."}


PHYSICAL_TOOL_NAMES = [n for n in dir(PhysicalTools) if not n.startswith("_")]


def call_physical(tools: PhysicalTools, name: str, args: dict[str, Any]) -> dict:
    fn = getattr(tools, name)
    if name == "measure_distance":
        return fn(**args)
    return fn(**args)
