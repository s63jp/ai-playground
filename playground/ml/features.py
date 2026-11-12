"""Feature extraction: turning engine-measured throws into rows a network can learn from.

A throw row pairs what was under the thrower's control (force, direction, its own motion at release,
the object's mass) with what the physics engine then measured (peak height, flight time, range...).
Rows are only ever built from the engine's own records (throw results / throw_release + flight_end
events), never from anything an agent typed.
"""
from __future__ import annotations

import math
from typing import Any

# Columns a network may take as input (all known at the moment of release).
INPUT_COLUMNS = {
    "force": "throwing force in newtons",
    "dir_x": "x (east) component of the unit throw direction",
    "dir_y": "y (north) component of the unit throw direction",
    "dir_z": "z (up) component of the unit throw direction",
    "elevation_deg": "angle of the throw direction above horizontal, degrees",
    "agent_vx": "thrower's own x velocity at release, m/s",
    "agent_vy": "thrower's own y velocity at release, m/s",
    "agent_vz": "thrower's own vertical velocity at release, m/s",
    "object_mass": "mass of the thrown object, kg",
    "feet_height": "height of the thrower's feet at release, m",
    "on_ground": "1 if the thrower stood on something at release, else 0",
    "release_height": "height of the object at release, m",
}
# Columns a network may predict (measured by the engine after release).
TARGET_COLUMNS = {
    "max_height": "highest point of the flight above the ground, m",
    "max_height_above_launch": "peak height above the release point, m",
    "time_to_max_height": "seconds from release to the peak",
    "flight_time": "seconds from release to first impact",
    "horizontal_range": "horizontal distance from release point to first impact, m",
    "impact_speed": "speed at first impact, m/s",
}
ALL_COLUMNS = {**INPUT_COLUMNS, **TARGET_COLUMNS}


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def throw_row(release: dict, flight: dict | None = None) -> dict | None:
    """Build one row from a throw's release record and its flight record.

    `release` may be a throw tool result (which contains both) or a throw_release event's data;
    `flight` is the matching flight_end event's data when the two arrive separately.
    Returns None when the record does not describe a throw with a measured peak.
    """
    fl = flight or release
    d = release.get("direction")
    pos = release.get("release_position")
    av = release.get("agent_velocity_at_release")
    if d is None or pos is None or av is None or _f(fl.get("max_height")) is None:
        return None
    dx, dy, dz = (float(v) for v in d)
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dx, dy, dz = dx / n, dy / n, dz / n
    rng = _f(fl.get("horizontal_range"))
    fi = fl.get("first_impact_position")
    if rng is None and fi is not None:
        rng = math.hypot(float(fi[0]) - float(pos[0]), float(fi[1]) - float(pos[1]))
    row = {
        "force": _f(release.get("force")),
        "dir_x": dx, "dir_y": dy, "dir_z": dz,
        "elevation_deg": math.degrees(math.asin(max(-1.0, min(1.0, dz)))),
        "agent_vx": _f(av[0]), "agent_vy": _f(av[1]), "agent_vz": _f(av[2]),
        "object_mass": _f(release.get("object_mass")),
        "feet_height": _f(release.get("agent_feet_height_at_release")),
        "on_ground": 1.0 if release.get("agent_on_ground_at_release") else 0.0,
        "release_height": _f(pos[2]),
        "max_height": _f(fl.get("max_height")),
        "max_height_above_launch": _f(fl.get("max_height_above_launch")),
        "time_to_max_height": _f(fl.get("time_to_max_height")),
        "flight_time": _f(fl.get("flight_time")),
        "horizontal_range": rng,
        "impact_speed": _f(fl.get("impact_speed")),
    }
    if row["force"] is None or row["object_mass"] is None:
        return None
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
