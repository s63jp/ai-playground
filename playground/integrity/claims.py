"""Independent verification of achievements.

Every verifier reads the physics engine's state or its event log. None of them
takes the agent's word, a program's return value or a memory file as input.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..world.mathutil import quat_to_mat

MIN_TOSS_TIME = 0.2      # a throw followed by a grab sooner than this is not a toss
_GEOM_EPS = 1e-6         # metres: slop for near-flush geometry decisions


def _at_rest(world, oid: str) -> bool:
    return (oid != world.held and world.obj_speed(oid) < 0.1
            and float(np.linalg.norm(world.obj_angvel(oid))) < 1.0)


def find_towers(world, xy_tol: float = 0.2) -> list[dict]:
    """Columns of blocks: each block resting on exactly the block below, the lowest on a non-block."""
    graph = world.support_graph()
    blocks = [o for o in sorted(world.active) if world.specs[o].kind == "block" and _at_rest(world, o)]
    above: dict[str, list[str]] = {}
    for b in blocks:
        lows = [x for x in graph.get(b, []) if x != "agent"]
        if len(lows) == 1 and lows[0] in blocks:
            above.setdefault(lows[0], []).append(b)
    towers = []
    for b in blocks:
        lows = [x for x in graph.get(b, []) if x != "agent"]
        if not lows or any(x in blocks for x in lows):
            continue
        chain = [b]
        while True:
            nxt = [c for c in above.get(chain[-1], [])
                   if math.hypot(*(world.obj_pos(c)[:2] - world.obj_pos(chain[-1])[:2])) <= xy_tol]
            if len(nxt) != 1:
                break
            chain.append(nxt[0])
        base = world.obj_pos(chain[0])
        top = world.obj_pos(chain[-1])
        towers.append({"blocks": chain, "height_blocks": len(chain), "base": lows,
                       "base_position": [round(float(v), 3) for v in base],
                       "top_height": round(float(top[2] + world.half_extent_z(chain[-1])), 3),
                       "lean": round(float(math.hypot(top[0] - base[0], top[1] - base[1])), 3)})
    return sorted(towers, key=lambda t: (-t["height_blocks"], t["blocks"][0]))


def find_structures(world) -> list[dict]:
    """Connected groups of two or more resting movable objects that touch one another."""
    d = world.d
    adj: dict[str, set[str]] = {}
    if d.ncon:
        oi = world.geom_oidx[d.contact.geom[:d.ncon]]
        for i in range(d.ncon):
            a, b = world.onames[int(oi[i, 0])], world.onames[int(oi[i, 1])]
            if a != b and a in world.active and b in world.active and a != world.held and b != world.held:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
    seen, out = set(), []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        if len(comp) < 2 or not all(_at_rest(world, o) for o in comp):
            continue
        pts = np.array([world.obj_pos(o) for o in comp])
        tops = [float(world.obj_pos(o)[2] + world.half_extent_z(o)) for o in comp]
        out.append({"objects": sorted(comp), "count": len(comp),
                    "centre": [round(float(v), 3) for v in pts.mean(axis=0)],
                    "footprint": [round(float(np.ptp(pts[:, 0])), 3), round(float(np.ptp(pts[:, 1])), 3)],
                    "top_height": round(max(tops), 3),
                    "levels": len({round(float(z), 1) for z in pts[:, 2]})})
    return out


def verify_tower(world, events, min_blocks: int = 5, count: int = 1) -> dict:
    towers = [t for t in find_towers(world) if t["height_blocks"] >= int(min_blocks)]
    return {"verified": len(towers) >= int(count), "measured": {"towers": towers, "qualifying": len(towers)},
            "criteria": f"at least {count} column(s) of {min_blocks}+ blocks at rest"}


def verify_object_on(world, events, *, object: str, surface: str) -> dict:
    if object not in world.active:
        return {"verified": False, "measured": {"in_world": False}, "criteria": f"{object} at rest on {surface}"}
    on = world.support_graph().get(object, [])
    rest = _at_rest(world, object)
    return {"verified": bool(rest and surface in on),
            "measured": {"resting_on": on, "at_rest": rest,
                         "position": [round(float(v), 3) for v in world.obj_pos(object)]},
            "criteria": f"{object} at rest and in supporting contact with {surface}"}


def verify_agent_on(world, events, *, surface: str) -> dict:
    return {"verified": bool(world.grounded and surface in world.support),
            "measured": {"standing_on": list(world.support), "feet_height": round(world.feet(), 3)},
            "criteria": f"body standing on {surface}"}


def verify_height_reached(world, events, *, object: str, min_height: float | None = None,
                          target: float | None = None, tolerance: float = 0.25) -> dict:
    if min_height is None and target is None:
        raise TypeError("a height_reached claim needs 'min_height' or 'target'")
    flights = [e["data"] for e in events if e["type"] == "flight_end" and e["data"].get("object") == object
               and e["data"].get("kind") == "throw"]
    peaks = [f["max_height"] for f in flights]
    best = max(peaks) if peaks else None
    if target is not None:
        errs = [abs(p - float(target)) for p in peaks]
        err = min(errs) if errs else None
        ok = err is not None and err <= float(tolerance)
        return {"verified": ok, "measured": {"peaks": peaks, "closest_error": err},
                "criteria": f"a throw of {object} peaking within {tolerance} m of {target} m"}
    ok = best is not None and min_height is not None and best >= float(min_height)
    return {"verified": bool(ok), "measured": {"peaks": peaks, "highest": best},
            "criteria": f"a throw of {object} peaking at or above {min_height} m"}


def verify_block_count(world, events, at_least: int = 13) -> dict:
    n = sum(1 for o in world.active if world.specs[o].kind == "block")
    appeared = [e["data"].get("object") for e in events if e["type"] == "object_appeared"]
    return {"verified": n >= int(at_least), "measured": {"blocks_in_world": n, "appeared_this_session": appeared},
            "criteria": f"at least {at_least} blocks present in the world"}


def verify_juggle(world, events, *, objects: list[str], cycles: int = 1) -> dict:
    """Count genuine throw-and-catch tosses from physics events and simulation time.

    A toss counts only if the object was released, stayed in the air for at least
    MIN_TOSS_TIME and was grabbed before touching anything. A cycle is one counted
    toss of every listed object. A drop resets the running streak.
    """
    objs = list(objects or [])
    if not objs:
        return {"verified": False, "measured": {}, "criteria": "objects must be listed"}
    airborne: dict[str, float] = {}
    counts = {o: 0 for o in objs}
    streak = {o: 0 for o in objs}
    best = drops = rejected = overlaps = 0
    t_first = t_last = None
    for e in events:
        o = e["data"].get("object")
        if o not in counts:
            continue
        if e["type"] == "throw_release":
            if any(x != o for x in airborne):
                overlaps += 1
            airborne[o] = e["t"]
            t_first = e["t"] if t_first is None else t_first
        elif e["type"] == "catch" and o in airborne:
            dt = e["t"] - airborne.pop(o)
            if e["data"].get("caught_in_flight") and dt >= MIN_TOSS_TIME:
                counts[o] += 1
                streak[o] += 1
                t_last = e["t"]
                best = max(best, min(streak.values()))
            else:
                rejected += 1
        elif e["type"] == "flight_end" and e["data"].get("outcome") in ("rest", "lost") and o in airborne:
            airborne.pop(o, None)
            drops += 1
            streak = {k: 0 for k in streak}
        elif e["type"] == "pickup" and o in airborne:
            airborne.pop(o, None)       # picked up after it had already hit something
            drops += 1
            streak = {k: 0 for k in streak}
    need_overlap = len(objs) > 1
    ok = best >= int(cycles) and (overlaps > 0 or not need_overlap)
    return {"verified": bool(ok),
            "measured": {"verified_cycles": best, "tosses": counts, "drops": drops,
                         "rejected_instant_grabs": rejected, "throws_with_another_object_airborne": overlaps,
                         "sim_time_span": None if t_first is None or t_last is None else round(t_last - t_first, 3)},
            "criteria": f"{cycles} unbroken cycle(s) of {', '.join(objs)}; each toss airborne >= {MIN_TOSS_TIME} s "
                        "and caught before touching anything"
                        + ("; at least one throw made while another object was airborne" if need_overlap else "")}


def verify_structure(world, events, objects: list[str] | None = None, min_objects: int = 2,
                     min_levels: int = 1) -> dict:
    structs = find_structures(world)
    want = set(objects or [])
    hits = [s for s in structs if s["count"] >= int(min_objects) and s["levels"] >= int(min_levels)
            and want.issubset(set(s["objects"]))]
    return {"verified": bool(hits), "measured": {"structures": structs},
            "criteria": f"a connected resting structure of {min_objects}+ objects on {min_levels}+ level(s)"
                        + (f" containing {sorted(want)}" if want else "")}


# ------------------------------------------------------------------ threaded
def _obj_pose(world, oid: str) -> tuple[np.ndarray, np.ndarray]:
    """Centre and quaternion of any known object, movable or structural."""
    s = world.specs[oid]
    if s.movable:
        return world.obj_pos(oid), world.obj_quat(oid)
    return np.array(s.position, dtype=float), np.array(s.rotation, dtype=float)


def _torus_dims(spec) -> tuple[float, float, int]:
    return (float(spec.dimensions["major_radius"]), float(spec.dimensions["tube_radius"]),
            int(spec.properties.get("segments", 16)))


def _support_along(shape: str, dims: dict, quat, n: np.ndarray) -> float:
    """Max of (p·n) over the shape centred at its origin, in world orientation quat."""
    R = quat_to_mat(quat)
    if shape == "box":
        nl = R.T @ n
        return float((abs(nl[0]) * dims["x"] + abs(nl[1]) * dims["y"] + abs(nl[2]) * dims["z"]) / 2)
    if shape == "sphere":
        return float(dims["radius"])
    a = R[:, 2]                                   # torus axis: the tube sweeps a circle around it
    return float(dims["major_radius"] * math.sqrt(max(0.0, 1.0 - (n @ a) ** 2)) + dims["tube_radius"])


def _project_box(c: np.ndarray, quat, dims: dict):
    R = quat_to_mat(quat)
    h = np.array([dims["x"], dims["y"], dims["z"]], dtype=float) / 2

    def proj(x: np.ndarray) -> np.ndarray:
        return c + R @ np.clip(R.T @ (x - c), -h, h)
    return proj


def _project_ball(c: np.ndarray, r: float):
    def proj(x: np.ndarray) -> np.ndarray:
        v = x - c
        d = float(np.linalg.norm(v))
        return c + v if d <= r else c + v * (r / d)
    return proj


def _project_capsule(a: np.ndarray, b: np.ndarray, r: float):
    ab = b - a
    dd = float(ab @ ab)

    def proj(x: np.ndarray) -> np.ndarray:
        t = 0.0 if dd < 1e-18 else float(np.clip((x - a) @ ab / dd, 0.0, 1.0))
        e = a + t * ab
        v = x - e
        d = float(np.linalg.norm(v))
        return e + v if d <= r else e + v * (r / d)
    return proj


def _plane_gap(proj_a, p0: np.ndarray, n: np.ndarray, max_iter: int = 800) -> float:
    """Distance from p0 to solid(A) ∩ {x : (x − p0)·n = 0}, where p0 lies on that plane.

    Dykstra's alternating projection between A and the plane converges to the exact
    nearest point when the two convex sets intersect; callers pre-check that.
    """
    x, p, q = p0.copy(), np.zeros(3), np.zeros(3)
    z = x
    for _ in range(max_iter):
        y = proj_a(x + p)
        p = x + p - y
        z = y + q
        z = z - ((z - p0) @ n) * n
        q = y + q - z
        if float(np.linalg.norm(z - y)) < 1e-12:
            break
        x = z
    return float(np.linalg.norm(z - p0))


def _torus_capsules(c: np.ndarray, quat, major: float, tube: float, segments: int
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """The engine's torus collision geometry: capsule centre segments in world coordinates."""
    R = quat_to_mat(quat)
    pts = [c + R @ np.array([major * math.cos(2 * math.pi * k / segments),
                             major * math.sin(2 * math.pi * k / segments), 0.0])
           for k in range(segments)]
    return [(pts[k], pts[(k + 1) % segments]) for k in range(segments)]


def _aperture_gap(world, oid: str, p0: np.ndarray, n: np.ndarray) -> float | None:
    """Nearest distance from p0 to the given object's solid cross-section at the plane
    through p0 with normal n; None when the solid does not meet the plane at all."""
    c, quat = _obj_pose(world, oid)
    s = world.specs[oid]
    depth = float((c - p0) @ n)
    if s.shape == "box":
        if abs(depth) > _support_along("box", s.dimensions, quat, n):
            return None
        return _plane_gap(_project_box(c, quat, s.dimensions), p0, n)
    if s.shape == "sphere":
        r = float(s.dimensions["radius"])
        if abs(depth) >= r:
            return None
        disc = c - depth * n
        return max(0.0, float(np.linalg.norm(disc - p0)) - math.sqrt(r * r - depth * depth))
    major, tube, segments = _torus_dims(s)
    best = None
    for a, b in _torus_capsules(c, quat, major, tube, segments):
        da, db = float((a - p0) @ n), float((b - p0) @ n)
        min_depth = 0.0 if da * db < 0 else min(abs(da), abs(db))
        if min_depth > tube:
            continue
        g = _plane_gap(_project_capsule(a, b, tube), p0, n)
        best = g if best is None else min(best, g)
    return best


def _contacts_between(world, a: str, b: str) -> int:
    """Contact points between two known objects in the current engine state."""
    d = world.d
    if not d.ncon:
        return 0
    oi = world.geom_oidx[d.contact.geom[:d.ncon]]
    ia, ib = world.oidx[a], world.oidx[b]
    return sum(1 for i in range(d.ncon)
               if (int(oi[i, 0]), int(oi[i, 1])) in ((ia, ib), (ib, ia)))


def verify_threaded(world, events, *, object: str, target: str) -> dict:
    """Independent check of {target} passing through the aperture of {object}.

    An aperture is a torus's hole. The relationship holds when the target's solid
    reaches the aperture plane (the plane through the ring's centre, normal to the
    ring's axis) within the ring's clear hole, and its material extends beyond that
    plane on both sides; the ring must be present, free and at rest. Geometry is
    taken from the engine state and the collision-geometry shapes (a settled torus is
    its capsule segments; the clear hole's radius is major·cos(pi/segments) − tube),
    so it works for any orientation — a tilted ring is evaluated in its own plane.
    The distance query is exact convex geometry (Dykstra projection); it does not
    assume the ring is horizontal and uses no object-specific constants.
    """
    def fail(measured: dict, why: str) -> dict:
        return {"verified": False, "measured": measured,
                "criteria": f"{object} threaded on {target}: {why}"}

    if object not in world.specs or target not in world.specs:
        unknown = [x for x in (object, target) if x not in world.specs]
        return fail({}, f"unknown object id(s) {unknown}")
    so, st = world.specs[object], world.specs[target]
    if so.shape != "torus":
        return fail({"shape": so.shape}, f"{object} is a {so.shape}; a threaded claim needs "
                                         "an object with an aperture (a torus)")
    o_in = (not so.movable) or object in world.active
    t_in = (not st.movable) or target in world.active
    if not o_in or not t_in:
        return fail({"in_world": {"object": o_in, "target": t_in}}, "not present in the world")
    if world.held in (object, target):
        return fail({"held": True}, "held by the body")
    rest_o = True if not so.movable else _at_rest(world, object)
    rest_t = True if not st.movable else _at_rest(world, target)
    if not (rest_o and rest_t):
        return fail({"at_rest": {"object": rest_o, "target": rest_t}}, "not sufficiently stationary")

    po, qo = _obj_pose(world, object)
    n = quat_to_mat(qo)[:, 2]                       # the aperture faces along the torus axis
    major, tube, segments = _torus_dims(so)
    r_aperture = major * math.cos(math.pi / segments) - tube   # clear hole of the capsule ring
    gap = _aperture_gap(world, target, po, n)
    c, quat = _obj_pose(world, target)
    depth = float((c - po) @ n)
    sup = _support_along(st.shape, st.dimensions, quat, n)
    lo, hi = depth - sup, depth + sup
    occupied = gap is not None and gap <= r_aperture + _GEOM_EPS
    through = hi > _GEOM_EPS and lo < -_GEOM_EPS
    measured = {
        "aperture_inner_radius": round(r_aperture, 4),
        "cross_section_gap": None if gap is None else round(gap, 4),
        "target_extent_along_axis": [round(lo, 4), round(hi, 4)],
        "axis": [round(float(v), 3) for v in n],
        "aperture_occupied": bool(occupied),
        "target_through_the_plane": bool(through),
        "contacts_with_target": _contacts_between(world, object, target),
        "object_speed": round(world.obj_speed(object), 4),
        "target_speed": None if not st.movable else round(world.obj_speed(target), 4),
    }
    return {"verified": bool(occupied and through), "measured": measured,
            "criteria": f"{object} threaded on {target}: {target}'s material reaches the aperture "
                        f"plane within {round(r_aperture, 3)} m of the ring centre and extends "
                        f"beyond it on both sides, with the ring at rest"}


VERIFIERS: dict[str, Callable[..., dict]] = {
    "tower": verify_tower, "object_on": verify_object_on, "agent_on": verify_agent_on,
    "height_reached": verify_height_reached, "block_count": verify_block_count,
    "juggle": verify_juggle, "structure": verify_structure, "threaded": verify_threaded,
}

CLAIM_HELP = {
    "tower": ("params: min_blocks (optional, default 5), count (optional, default 1). "
              "Asserts count+ column(s) of at least min_blocks blocks resting on one another at rest."),
    "object_on": ("params: object, surface (both required object ids). "
                  "Asserts <object> is at rest and in supporting contact with <surface>."),
    "agent_on": ("params: surface (required object id). Asserts the body is standing on <surface>."),
    "height_reached": ("params: object (required id), and exactly one of min_height or target "
                       "(tolerance optional, default 0.25). Asserts a throw of <object> peaked "
                       "at or above min_height, or within tolerance of target."),
    "block_count": ("params: at_least (optional, default 13). "
                    "Asserts at_least or more blocks exist in the world."),
    "juggle": ("params: objects (required list of object ids), cycles (optional, default 1). "
               "Asserts cycles unbroken toss(es) of every listed object, each airborne "
               ">= 0.2 s and caught before touching anything."),
    "structure": ("params: min_objects (optional, default 2), min_levels (optional, default 1), "
                  "objects (optional list of ids that must all be part of the structure). "
                  "Asserts a connected resting structure meeting those."),
    "threaded": ("params: object, target (both required object ids; object must be a torus, i.e. "
                 "have an aperture). Asserts target passes through object's aperture: its material "
                 "reaches the aperture plane inside the clear hole and extends beyond the plane on "
                 "both sides, with the ring at rest (works while the ring is tilted)."),
}


def verify_claim(claim_type: str, params: dict[str, Any], world, events: list[dict]) -> dict:
    if claim_type not in VERIFIERS:
        raise KeyError(claim_type)
    res = VERIFIERS[claim_type](world, events, **(params or {}))
    res["type"] = claim_type
    res["params"] = params or {}
    return res
