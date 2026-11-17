"""Projectile system for P5.
Implements launch, update (detect impacts) and simple explosion utility.
"""

import math
import numpy as np
from typing import List, Dict, Any


class ProjectileSystem:
    """Manage a pool of pre‑created movable objects as projectiles.

    world: PhysicsWorld instance
    pool_ids: list of object IDs that exist in the world spec and are marked as movable.
    """

    def __init__(self, world, pool_ids: List[str]):
        self.world = world
        # copy so we can pop without mutating caller's list
        self._available = list(pool_ids)
        self._active_flights: Dict[str, Dict[str, Any]] = {}
        self._last_event_index = 0

    def launch(self, origin: List[float], direction: List[float], speed: float, cfg: dict | None = None) -> Dict[str, Any]:
        """Spawn a projectile from the pool.
        Returns a dict with ``proj_id`` on success.
        If the pool is empty returns ``{'error': 'pool_exhausted'}``.
        """
        if not self._available:
            return {"error": "pool_exhausted"}
        proj_id = self._available.pop(0)
        # place the object
        self.world.spawn_object(proj_id, origin)
        # normalise direction
        d = np.array(direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm == 0:
            raise ValueError("direction vector must be non‑zero")
        vel = (d / norm * speed).tolist()
        self.world.set_object_velocity(proj_id, vel)
        # record that a flight is in progress – we will later pull impact info from world events
        self._active_flights[proj_id] = {"origin": origin, "velocity": vel, "cfg": cfg}
        return {"proj_id": proj_id}

    def update(self) -> List[Dict[str, Any]]:
        """Advance the world one step and return any new impact dicts.
        The dict format matches the contract:
        {'proj_id', 'pos', 'normal', 'angle_deg', 'surface', 'speed', 't'}

        angle_deg convention: incidence angle between the projectile's incoming
        velocity and the surface normal — 0 deg for a straight-down drop on flat
        ground (velocity direction opposite the normal).
        """
        # step a small amount – enough for collisions to be registered
        self.world.step(1)
        # Refresh in-flight velocities BEFORE scanning contacts, so the impact angle uses
        # the pre-contact velocity (gravity bends the trajectory during flight).
        for pid, flight in self._active_flights.items():
            try:
                flight["last_velocity"] = np.array(self.world.obj_vel(pid), dtype=float).tolist()
            except Exception:
                pass
        new_impacts = []
        events = self.world.events
        for ev in events[self._last_event_index :]:
            if ev.get("type") != "collision":
                continue
            data = ev.get("data", {})
            objs = data.get("objects", [])
            # find any projectile in this collision
            proj = None
            other = None
            for o in objs:
                if o in self._active_flights:
                    proj = o
                else:
                    other = o
            if not proj:
                continue
            # Real contact-frame normal when a live contact exists for the projectile.
            normal = self._contact_normal(proj)
            if normal is None:
                normal = [0.0, 0.0, 1.0]
                angle = 0.0
            else:
                n = np.array(normal, dtype=float)
                n /= float(np.linalg.norm(n) + 1e-12)
                # Incidence angle uses the velocity recorded JUST BEFORE contact
                # (refreshed on every update() call while in flight; gravity bends the
                # trajectory, so launch velocity alone would be an approximation).
                v_dir = np.array(self._active_flights[proj].get("last_velocity",
                                  self._active_flights[proj]["velocity"]), dtype=float)
                v_dir /= float(np.linalg.norm(v_dir) + 1e-12)
                cos_inc = float(np.clip(np.dot(-v_dir, n), -1.0, 1.0))
                angle = math.degrees(math.acos(cos_inc))
            impact = {
                "proj_id": proj,
                "pos": data.get("position", [0.0, 0.0, 0.0]),
                "normal": [float(c) for c in normal],
                "angle_deg": round(float(angle), 2),
                "surface": other,
                "speed": float(self.world.obj_speed(proj)),
                "t": ev.get("t"),
            }
            new_impacts.append(impact)
            # remove flight tracking – assume one impact ends the flight
            self._active_flights.pop(proj, None)
        self._last_event_index = len(events)
        return new_impacts

    def _contact_normal(self, proj: str):
        """Return the MuJoCo contact-frame normal (unit vector, oriented out of the
        surface toward the projectile) for the first live contact involving the
        projectile's geoms, or None when no contact is live.

        Mapping: world.geom_oidx maps geom ids to object indexes; world.onames is the
        object-index -> name list (same table the collision processor uses).
        """
        w = self.world
        d = w.d
        ncon = int(d.ncon)
        for i in range(ncon):
            c = d.contact[i]
            try:
                name_a = w.onames[int(w.geom_oidx[int(c.geom1)])]
                name_b = w.onames[int(w.geom_oidx[int(c.geom2)])]
            except Exception:
                continue
            if name_a == proj or name_b == proj:
                # mujoco 4.x MjContact has no .normal; the contact frame's z-axis
                # (frame[6:9]) is the normal direction.
                n = np.array(c.frame[6:9], dtype=float)
                norm = float(np.linalg.norm(n))
                if norm < 1e-9:
                    return None
                # Orient the normal out of the surface toward the projectile
                # (verified empirically by the vertical-drop incidence test).
                if name_a == proj:
                    n = -n
                return (n / norm).tolist()
        return None


def explosion(world, pos: List[float], radius: float, force: float, particle_count: int = 64) -> Dict[str, Any]:
    """Apply a radial impulse to all movable objects within ``radius``.
    Emits an ``explosion`` event with particle directions generated via a Fibonacci sphere.
    Returns the payload dict that was emitted.
    """
    # generate particle direction vectors (unit) using deterministic seed
    rs = np.random.RandomState(12345)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    particle_dirs = []
    for i in range(particle_count):
        y = 1 - (i / float(particle_count - 1)) * 2  # y from 1 to -1
        radius_xy = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius_xy
        z = math.sin(theta) * radius_xy
        particle_dirs.append([x, y, z])

    # apply impulse to each movable object inside radius
    for oid in world.visible_ids():
        spec = world.specs.get(oid)
        if not spec or not spec.movable:
            continue
        obj_pos = world.obj_pos(oid)
        vec = np.array(obj_pos) - np.array(pos)
        dist = float(np.linalg.norm(vec))
        if dist == 0 or dist > radius:
            continue
        # linear fall‑off
        magnitude = force * (1.0 - dist / radius)
        direction = (vec / dist).tolist()
        impulse = (np.array(direction) * magnitude).tolist()
        # get current velocity, add impulse / mass
        cur_vel = world.obj_vel(oid)
        mass = float(spec.mass) if spec.mass else 1.0
        new_vel = (cur_vel + np.array(impulse) / mass).tolist()
        world.set_object_velocity(oid, new_vel)

    payload = {
        "pos": pos,
        "radius": radius,
        "force": force,
        "particle_count": particle_count,
        "particle_dirs": particle_dirs,
    }
    world.emit("explosion", hidden=None, **payload)
    return payload
