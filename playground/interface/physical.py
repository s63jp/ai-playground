"""Physical tools: observation, movement, object handling and measurement.

Bound to a PhysicsWorld and nothing else, so exactly the same code serves the real
world and the private sandbox copy used by run_physics_test.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from ..integrity.claims import find_structures, find_towers
from ..world.mathutil import compass, heading_deg, parse_direction, rnd
try:
    from ..world.physics import ActionError, PhysicsWorld
except Exception:
    class ActionError(Exception):
        pass
    class PhysicsWorld:
        def __init__(self, cfg=None, *args, **kwargs):
            self.cfg = cfg
            self.d = type("Data", (), {"qpos": np.zeros(100, dtype=np.float32), "qvel": np.zeros(100, dtype=np.float32)})
        def reset(self):
            pass
        def step(self, n=1):
            pass
        def set_joint_target(self, joint, pos, kp=15., kd=1.5):
            pass
        def agent_vel(self):
            return np.zeros(3, dtype=np.float32)
        def opt(self):
            return type("Opt", (), {"gravity": [0., -9.81, 0.]})
from .schema import ToolError

AGENT_EVENT_TYPES = ("throw_release", "object_impulse", "flight_end", "collision", "agent_landing",
                     "jump_takeoff", "pickup", "catch", "release", "climb", "rotate", "object_appeared",
                     "object_lost")


def visible_event(ev: dict) -> dict:
    """An event as an agent may see it: measurements only, never the researcher-side annotations."""
    return {"seq": ev.get("n", ev.get("seq")), "t": ev["t"], "type": ev["type"], "data": ev["data"],
            **({"action_index": ev["action_idx"]} if ev.get("action_idx") is not None else {})}


class PhysicalTools:
    def __init__(self, world: PhysicsWorld):
        self.w = world
        # Phase 12 lazily-built skill plumbing (skills library, adapter, arm controller).
        self._skill_lib = None
        self._skill_adapter = None
        self._mc = None

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
        br = w.bounding_radius(w.held) if w.held else None
        return {"holding": w.held, "held_object": held, "hands_free": w.held is None,
                "carried_mass": round(w.total_mass() - w.base_mass, 3),
                "carried_bounding_radius": round(br, 4) if br else None,
                "agent": w.agent_state()}

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

    def set_joint_target(self, joint: str, position: float, kp: float = 0.0, kd: float = 0.0) -> dict:
        """Register a PD controller for a joint. Returns acknowledgment.
        The controller will be applied each physics step.
        """
        self.w.set_joint_target(joint, float(position), float(kp), float(kd))
        return {"joint": joint, "target": position, "kp": kp, "kd": kd}


    def climb(self, direction=None) -> dict:
        return self.w.climb(None if direction is None else self._direction(direction))

    def wait(self, seconds: float) -> dict:
        return self.w.wait(float(seconds))
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

    # ------------------------------------------------------------ contact manipulation
    def _away_dir(self, object_id: str):
        """A horizontal unit vector pointing from the body towards the object (default push/kick direction)."""
        if object_id not in self.w.specs:
            raise ToolError("unknown_object", f"There is no object with id '{object_id}'.")
        if self.w.held == object_id:
            raise ToolError("holding_object", f"'{object_id}' is being held; put it down first.")
        p = self.w.obj_pos(object_id)
        ap = self.w.agent_pos()
        d = np.array([p[0] - ap[0], p[1] - ap[1], 0.0])
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-6:
            raise ToolError("invalid_argument", "The object is directly beneath you; give a direction.")
        return d / n

    def push(self, object_id: str, direction=None, force: float = 40.0, wait: str = "rest") -> dict:
        d = self._away_dir(object_id) if direction is None else self._direction(direction)
        return self.w.impulse_object(object_id, d, float(force), verb="push", wait=wait)

    def pull(self, object_id: str, force: float = 25.0, wait: str = "rest") -> dict:
        d = -self._away_dir(object_id)   # towards the body
        return self.w.impulse_object(object_id, d, float(force), verb="pull", wait=wait)

    def slide(self, object_id: str, direction, force: float = 30.0, wait: str = "rest") -> dict:
        return self.w.impulse_object(object_id, self._direction(direction), float(force), verb="slide", wait=wait)

    def nudge(self, object_id: str, direction=None, force: float = 6.0, wait: str = "rest") -> dict:
        d = self._away_dir(object_id) if direction is None else self._direction(direction)
        return self.w.impulse_object(object_id, d, float(force), verb="nudge", wait=wait)

    def kick(self, object_id: str, direction=None, force: float = 80.0, wait: str = "rest") -> dict:
        d = self._away_dir(object_id) if direction is None else self._direction(direction)
        return self.w.impulse_object(object_id, d, float(force), verb="kick", wait=wait)

    def roll(self, object_id: str, direction, force: float = 40.0, wait: str = "rest") -> dict:
        return self.w.impulse_object(object_id, self._direction(direction), float(force), verb="roll", wait=wait)

    # ------------------------------------------------------------ construction convenience
    def stack_on(self, target_id: str, object_id: str | None = None) -> dict:
        """Place the held object centred on top of target_id. No strategy — just the release-point arithmetic."""
        w = self.w
        held = object_id or w.held
        if not held:
            raise ToolError("not_holding", "You are not holding anything to stack.")
        if target_id not in w.specs:
            raise ToolError("unknown_object", f"There is no object with id '{target_id}'.")
        tp = w.obj_pos(target_id)
        top = tp[2] + w.half_extent_z(target_id)
        hz = w.half_extent_z(held, w.held_quat)
        res = w.place(held, [float(tp[0]), float(tp[1]), float(top + hz + 0.005)])
        res["stacked_on"] = target_id
        return res

    def build_row(self, beside_id: str, object_id: str | None = None, direction=None) -> dict:
        """Place the held object on the ground next to beside_id, forming/extending a row."""
        w = self.w
        held = object_id or w.held
        if not held:
            raise ToolError("not_holding", "You are not holding anything to place.")
        if beside_id not in w.specs:
            raise ToolError("unknown_object", f"There is no object with id '{beside_id}'.")
        bp = w.obj_pos(beside_id)
        try:
            bw = float(w.specs[beside_id].dimensions.get("x", 0.5)) / 2
        except Exception:
            bw = 0.3
        try:
            hw = float(w.specs[held].dimensions.get("x", 0.5)) / 2
        except Exception:
            hw = 0.3
        step = bw + hw + 0.12
        # Try the requested side first (or away-from-body by default), then fall back to other sides
        # if that space is occupied — same spirit as put_down, so a row still forms near a wall/platform.
        if direction is None:
            base = self._away_dir(beside_id)
        else:
            base = self._direction(direction)
        import math as _m
        base_ang = _m.atan2(base[1], base[0])
        last = None
        for k in (0, 1, -1, 2, -2, 3, -3, 4):
            ang = base_ang + k * _m.pi / 4
            x = float(bp[0] + _m.cos(ang) * step)
            y = float(bp[1] + _m.sin(ang) * step)
            try:
                res = w.place(held, [x, y])
                res["beside"] = beside_id
                return res
            except ActionError as exc:
                last = exc
        raise ToolError("no_room", f"No free space next to '{beside_id}' to place the object "
                        f"({last.message if last else 'unknown'}).")

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


    # ------------------------------------------------------------------ physical extension (Phase 12)
    # These wrappers ride the existing gait/manipulation/world implementations
    # only: no second locomotion system, no qpos/qvel writes
    # (docs/AGENT_PHYSICAL_INTEGRATION.md).

    def _skill_setup(self) -> None:
        """Build the skill library + adapter lazily (cheap worlds stay cheap)."""
        if self._skill_lib is None:
            import tempfile
            from ..skills.agent_adapter import AgentSkillAdapter
            from ..skills.catalog import ALL_SKILLS
            from ..skills.library import SkillsLibrary
            from ..manipulation import ManipulationController
            lib = SkillsLibrary(root=os.path.join(tempfile.gettempdir(), "playground_skills"))
            known = {s.id for s in lib.list()}
            for spec in ALL_SKILLS:
                if spec.id not in known:
                    lib.register(spec)
            self._skill_lib = lib
            self._mc = ManipulationController(self.w)
            self._skill_adapter = AgentSkillAdapter(self.w, self, self._mc)

    def _resolve_skill(self, name: str):
        self._skill_setup()
        key = str(name).strip().lower()
        for spec in self._skill_lib.list():
            if spec.id == key or spec.id == f"skill-{key}" or spec.name.lower() == key:
                return spec
        raise ToolError("unknown_skill", f"There is no skill named '{str(name)[:60]}'.",
                        available_skills=sorted(s.id for s in self._skill_lib.list()))

    def run_to(self, x: float, y: float, speed: float | None = None) -> dict:
        """A run is a move at a raised requested speed; the payload reports what the legs managed."""
        res = self.walk_to(float(x), float(y), speed=float(speed) if speed is not None else 2.8)
        res["requested_speed"] = round(float(speed) if speed is not None else 2.8, 3)
        return res

    def turn_to(self, heading: float) -> dict:
        """Face a compass heading. Heading changes ride the contact-driven walk —
        this rig turns by striding (measured drift and waist angle are reported)."""
        w = self.w
        h = float(heading)
        current = float(getattr(w, "facing", 90.0))
        diff = (h - current + 180.0) % 360.0 - 180.0
        ap0 = w.agent_pos().copy()
        walked = 0.0
        if abs(diff) > 5.0:
            d2 = np.array([math.cos(math.radians(h)), math.sin(math.radians(h))], dtype=float)
            res = w.move(d2, 0.12, speed=0.5, stop=True)
            walked = float(res.get("distance_travelled", 0.0))
            end_face = round(float(w.facing), 2)
        else:
            end_face = round(current, 2)
        j = w.m.joint("waist_yaw")
        waist = float(w.d.qpos[int(j.qposadr[0])])
        end = w.agent_pos()
        return {"requested_heading": h, "previous_facing_degrees": round(current, 2),
                "facing_degrees": end_face,
                "walked_for_turn": round(walked, 4),
                "displacement_during_turn": round(float(math.hypot(end[0] - ap0[0], end[1] - ap0[1])), 4),
                "waist_yaw_deg": round(math.degrees(waist), 2)}

    def stop_moving(self) -> dict:
        """End the current walk: the gait controller holds posture until the body is still."""
        w = self.w
        if getattr(w, "_gait_controller", None) is not None:
            w.set_gait("stop")
        w.target_vel[:] = 0.0
        t1 = w.time
        while w.time - t1 < 3.0:
            w._step()
            v = w.agent_vel()
            if w.grounded and math.hypot(v[0], v[1]) < 0.02 and abs(v[2]) < 0.02:
                break
        v = w.agent_vel()
        speed = float(math.hypot(v[0], v[1]))
        return {"settled": bool(w.grounded and speed < 0.05), "speed": round(speed, 4),
                "velocity": [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)],
                "elapsed_s": round(w.time - t1, 3)}

    def stand(self, seconds: float = 2.0) -> dict:
        """Hold the neutral standing posture for a while; reports the measured state after."""
        w = self.w
        w.set_gait("stand")
        payload = w.wait(float(seconds)) if seconds and float(seconds) > 0 else {}
        return {"upright_after": bool(not w.agent_fallen()), "agent_state": w.agent_state(),
                **{k: v for k, v in payload.items() if k != "status"}}

    def stand_up(self) -> dict:
        """Attempt to rise from a fallen pose. The outcome comes from the measured
        recovery events; a recovery that never sustained is reported as failed."""
        w = self.w
        if not w.agent_fallen():
            return {"recovered": True, "recovery_event": None, "agent_was_upright": True,
                    "agent_state": w.agent_state()}
        w.stand_up()
        deadline = w.time + 9.0
        idx, events, event = 0, w.events, None
        while w.time < deadline and event is None:
            w._step()
            while idx < len(events):
                etype = events[idx].get("type")
                if etype in ("recovery_complete", "recovery_failed"):
                    event = etype
                    break
                idx += 1
        if event != "recovery_complete":
            raise ActionError("recovery_failed", "The stand-up attempt ended without a sustained upright pose.",
                              event=event, fallen_after=bool(w.agent_fallen()))
        return {"recovered": True, "recovery_event": event,
                "recovery_after_s": round(w.time - deadline + 9.0, 3),
                "agent_state": w.agent_state()}

    def approach_object(self, object_id: str, stop_short: float = 0.6, speed: float | None = None) -> dict:
        """Walk until the object is close, stopping short of it."""
        w = self.w
        w._movable(object_id)  # validates: ActionError unknown_object for unknown ids
        pos = np.asarray(w.obj_pos(object_id), dtype=float)
        ap = w.agent_pos()
        off = np.array([ap[0] - pos[0], ap[1] - pos[1]], dtype=float)
        n = float(math.hypot(off[0], off[1]))
        if n < 1e-6:
            off = np.array([0.0, -1.0])
            n = 1.0
        short = max(0.0, min(float(stop_short), n - 0.25))
        if n > short + 0.05:
            tgt = pos[:2] + off / n * short
            res = dict(self.walk_to(float(tgt[0]), float(tgt[1]), speed=speed))
        else:
            res = {"distance_travelled": 0.0, "start_position": rnd(ap).tolist(),
                   "end_position": rnd(ap).tolist(), "elapsed": 0.0}
        end = w.agent_pos()
        remaining = float(math.hypot(end[0] - pos[0], end[1] - pos[1]))
        res["object"] = object_id
        res["distance_remaining"] = round(remaining, 4)
        res["within_reach"] = bool(remaining <= float(w.ac.get("reach", 1.5)))
        return res

    def reach_for(self, object_id: str) -> dict:
        """Point one arm toward the object; the aim is verified against the measured joints."""
        w = self.w
        w._movable(object_id)  # validates: ActionError unknown_object for unknown ids
        self._skill_setup()
        out = dict(self._mc.reach(object_id))
        obj = np.asarray(w.obj_pos(object_id), dtype=float)
        hand = np.asarray(self._mc._hand_pos(), dtype=float)
        out["hand_to_object"] = round(float(np.linalg.norm(hand - obj)), 4)
        out["object_position"] = [round(float(obj[0]), 4), round(float(obj[1]), 4), round(float(obj[2]), 4)]
        return out

    def carry_to(self, object_id: str, x: float, y: float, speed: float | None = None) -> dict:
        """Take an object to a point: close in, take hold, walk there, set it down.
        Every part is reported; nothing teleports — the object rides the carry pin."""
        w = self.w
        if abs(float(x)) > w.half or abs(float(y)) > w.half:
            raise ToolError("outside_world", f"({x}, {y}) is beyond the island edge (limits are "
                            f"+/-{w.half} in x and y).", limits=[-w.half, w.half])
        w._movable(object_id)  # validates: ActionError unknown_object for unknown ids
        ledger: list[dict] = []
        pos = np.asarray(w.obj_pos(object_id), dtype=float)
        if w.held and w.held != object_id:
            raise ActionError("hands_full", f"Already holding '{w.held}'; put it down before taking '{object_id}'.",
                              holding=w.held)
        if not w.held:
            ap = w.agent_pos()
            off = np.array([ap[0] - pos[0], ap[1] - pos[1]], dtype=float)
            n = float(math.hypot(off[0], off[1]))
            if n > 0.5:
                step_to = pos[:2] + off / n * 0.5
                app = dict(self.walk_to(float(step_to[0]), float(step_to[1]), speed=speed))
                ledger.append({"step": "approach", "arrived": app.get("arrived"),
                               "distance_travelled": app.get("distance_travelled")})
            got = dict(self.pick_up(object_id))
            ledger.append({"step": "pick_up", "held": w.held,
                           "was_moving": got.get("was_moving")})
        carry = dict(self.walk_to(float(x), float(y), speed=speed))
        ledger.append({"step": "carry", "arrived": carry.get("arrived"),
                       "distance_travelled": carry.get("distance_travelled"),
                       "still_holding": w.held})
        if w.held == object_id:
            down = dict(self.put_down(object_id))
            ledger.append({"step": "put_down", "released": True})
        final = w.obj_pos(object_id)
        miss = float(math.hypot(final[0] - float(x), final[1] - float(y)))
        return {"ledger": ledger, "object": object_id, "target": [float(x), float(y)],
                "final_position": [round(float(final[0]), 4), round(float(final[1]), 4)],
                "distance_to_target": round(miss, 4), "held_after": w.held}

    def run_skill(self, skill: str, params: dict | None = None) -> dict:
        """Execute one catalogue skill and return what physically happened."""
        spec = self._resolve_skill(skill)
        out = self._skill_lib.invoke(spec.id, self._skill_adapter, dict(params or {}))
        body = {"skill": spec.id, "skill_name": spec.name, "skill_status": out.get("status")}
        body.update(out.get("result") or {})
        if out.get("detail"):
            body["detail"] = out["detail"]
        if out.get("status") == "failed":
            raise ActionError("skill_failed", str(out.get("detail") or "the skill failed"), **body)
        if out.get("status") == "unavailable":
            raise ActionError("skill_unavailable", str(out.get("detail") or "this world lacks the capability"), **body)
        return body

    def run_chain(self, steps: list) -> dict:
        """Run skills in order, stopping at the first that does not complete.
        Each step is reported; the steps after a failure are marked interrupted."""
        if not isinstance(steps, list) or not steps:
            raise ToolError("invalid_argument", "run_chain needs a non-empty ordered list of steps.")
        ledger: list[dict] = []
        completed = 0
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or not step.get("skill"):
                raise ToolError("invalid_argument", f"Step {i} needs a 'skill' name.")
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            try:
                body = self.run_skill(step["skill"], args)
                ledger.append({"index": i, "skill": step["skill"], "status": "succeeded", "result": body})
                completed += 1
            except (ActionError, ToolError) as exc:
                code = getattr(exc, "code", "skill_failed")
                msg = getattr(exc, "message", str(exc))
                ledger.append({"index": i, "skill": step["skill"], "status": "failed",
                               "error": {"code": code, "message": msg}})
                for j in range(i + 1, len(steps)):
                    nxt = steps[j].get("skill") if isinstance(steps[j], dict) else None
                    ledger.append({"index": j, "skill": nxt, "status": "interrupted"})
                raise ActionError(
                    "chain_failed",
                    f"Step {i} ({step['skill']}) did not complete ({code}); "
                    f"the remaining {len(steps) - i - 1} step(s) were not run.",
                    ledger=ledger, completed_steps=completed, failed_step=i)
        return {"ledger": ledger, "completed_steps": completed, "total_steps": len(steps)}


PHYSICAL_TOOL_NAMES = [n for n in dir(PhysicalTools) if not n.startswith("_")]


def call_physical(tools: PhysicalTools, name: str, args: dict[str, Any]) -> dict:
    fn = getattr(tools, name)
    if name == "measure_distance":
        return fn(**args)
    return fn(**args)
