"""Small maths helpers shared by the world modules."""
from __future__ import annotations

import math

import numpy as np

CARDINALS = {
    "east": (1.0, 0.0, 0.0), "west": (-1.0, 0.0, 0.0),
    "north": (0.0, 1.0, 0.0), "south": (0.0, -1.0, 0.0),
    "northeast": (0.70710678, 0.70710678, 0.0), "northwest": (-0.70710678, 0.70710678, 0.0),
    "southeast": (0.70710678, -0.70710678, 0.0), "southwest": (-0.70710678, -0.70710678, 0.0),
    "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0),
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def rnd(value, nd: int = 4):
    """Round floats / arrays / nested containers for structured output."""
    if isinstance(value, (float, np.floating)):
        v = round(float(value), nd)
        return 0.0 if v == 0 else v
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return [rnd(v, nd) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [rnd(v, nd) for v in value]
    if isinstance(value, dict):
        return {k: rnd(v, nd) for k, v in value.items()}
    return value


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_euler_deg(q) -> list[float]:
    """Quaternion (w,x,y,z) -> [roll, pitch, yaw] in degrees (intrinsic Z-Y-X)."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def euler_deg_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = (math.radians(float(v)) / 2 for v in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def parse_direction(direction) -> np.ndarray:
    """Accept a compass name, a heading in degrees, or a 2/3-vector. Returns a unit 3-vector."""
    if isinstance(direction, str):
        key = direction.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        if direction.strip() in CARDINALS:
            key = direction.strip()
        if key in CARDINALS:
            return np.array(CARDINALS[key], dtype=float)
        try:
            direction = float(direction)
        except ValueError as exc:
            raise ValueError(
                f"unknown direction '{direction}'. Use north/south/east/west/up/down (and diagonals), "
                "a heading in degrees (0 = east, 90 = north), or a vector [dx, dy, dz]."
            ) from exc
    if isinstance(direction, (int, float)) and not isinstance(direction, bool):
        a = math.radians(float(direction))
        return np.array([math.cos(a), math.sin(a), 0.0])
    if isinstance(direction, dict):
        direction = [direction.get("x", 0.0), direction.get("y", 0.0), direction.get("z", 0.0)]
    try:
        vec = np.array([float(v) for v in direction], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("direction must be a name, a heading in degrees, or a vector [dx, dy, dz]") from exc
    if vec.shape == (2,):
        vec = np.array([vec[0], vec[1], 0.0])
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError("direction vector must have 2 or 3 finite components")
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("direction vector has zero length")
    return vec / norm


def heading_deg(vec) -> float:
    return math.degrees(math.atan2(float(vec[1]), float(vec[0])))


def compass(angle_deg: float) -> str:
    names = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]
    return names[int(((angle_deg % 360) + 22.5) // 45) % 8]
