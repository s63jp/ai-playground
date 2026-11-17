"""Agent-facing skill adapter for the playground.

Dispatches the skill catalogue (playground/skills/catalog.py) onto the EXISTING
implementations — the PhysicsWorld gait API, GaitController recovery, the
ManipulationController and the world verbs. It introduces no new actuation: every
skill is a parameter bundle over what the world already does, and every result
carries the measurements the simulation produced
(docs/AGENT_PHYSICAL_INTEGRATION.md, layers 2-5).
"""
from __future__ import annotations

import math

import numpy as np

from ..world.physics import ActionError


class AgentSkillAdapter:
    """Dispatch table for the 16 catalogue skills.

    Method names match the catalogue's ``controller.method`` values. Anything the
    skill cannot honestly do raises ``ActionError`` — the library turns that into
    a ``failed`` skill status and the caller reports the measurement, never an
    assumed success.
    """

    RUN_SPEED = 2.8  # requested run target speed; the payload reports what actually happened

    def __init__(self, world, tools, manipulation=None):
        self.w = world
        self.tools = tools
        self.mc = manipulation

    # ---------------------------------------------------------------- helpers
    def _dir(self, direction):
        return np.asarray(self.tools._direction(direction if direction is not None else "north"), dtype=float)

    def _move(self, direction, distance, speed):
        return dict(self.w.move(self._dir(direction), float(distance), speed, stop=True))

    # ---------------------------------------------------------------- locomotion
    def stand(self, seconds: float = 2.0, **extra) -> dict:
        self.w.set_gait("stand")
        payload = dict(self.w.wait(max(0.05, min(60.0, float(seconds))))) if seconds else {}
        return {"status": "succeeded", "upright_after": bool(not self.w.agent_fallen()), **payload}

    def _walk_dir(self, direction, heading):
        """Direction vector from a compass name/vector or a heading in degrees."""
        if direction is not None:
            return self._dir(direction)
        if heading is not None:
            rad = math.radians(float(heading))
            return np.array([math.cos(rad), math.sin(rad), 0.0])
        return self._dir(None)

    def set_gait(self, mode: str = "walk", speed: float | None = None, heading=None,
                 direction=None, distance: float = 0.0, **extra) -> dict:
        """Gait skill. With a distance (and a direction or heading) this DRIVES a
        real contact-governed walk via the existing world.move; without one it
        only configures the gait controller and reports the change."""
        if mode in ("walk", "run") and distance and float(distance) > 0.0:
            res = dict(self.w.move(self._walk_dir(direction, heading), float(distance),
                                   speed, stop=True))
            res["status"] = "succeeded"
            return res
        self.w.set_gait(mode, speed=float(speed) if speed is not None else 1.0,
                        heading=None if heading is None else float(heading))
        return {"status": "succeeded", "gait": mode,
                "heading_degrees": None if heading is None else round(float(heading), 2)}

    def move(self, direction=None, distance: float = 0.5, speed: float | None = None, **extra) -> dict:
        res = self._move(direction, distance, speed)
        res["status"] = "succeeded"
        return res

    def turn(self, heading, **extra) -> dict:
        if heading is None:
            raise ActionError("invalid_argument", "Turning needs a heading in degrees (0 = east, 90 = north).")
        res = dict(self.tools.turn_to(float(heading)))
        res["status"] = "succeeded"
        return res

    def jump(self, wait: str = "landing", **extra) -> dict:
        res = dict(self.w.jump(wait))
        res["status"] = "succeeded"
        return res

    def jump_landing(self, **extra) -> dict:
        """Jump with the measured touchdown report."""
        res = dict(self.w.jump("landing"))
        res["status"] = "succeeded"
        return res

    def balance(self, seconds: float = 2.0, **extra) -> dict:
        self.w.set_gait("stand")
        t0 = self.w.time
        peak = 0.0
        while self.w.time - t0 < max(0.05, min(60.0, float(seconds))):
            self.w._step()
            v = self.w.agent_vel()
            peak = max(peak, float(np.linalg.norm(v[:2])))
        upright = not self.w.agent_fallen()
        return {"status": "succeeded" if upright else "failed", "upright_after": upright,
                "peak_sway_speed": round(peak, 4)}

    def agent_fallen(self, **extra) -> dict:
        fallen = bool(self.w.agent_fallen())
        return {"status": "succeeded", "fallen": fallen, "agent_state": self.w.agent_state()}

    def stand_up(self, **extra) -> dict:
        """Recovery: the outcome is measured from the recovery events — never assumed."""
        if not self.w.agent_fallen():
            return {"status": "succeeded", "recovered": True, "agent_was_upright": True}
        self.w.stand_up()
        deadline = self.w.time + 9.0
        idx, events, event = 0, self.w.events, None
        while self.w.time < deadline and event is None:
            self.w._step()
            while idx < len(events):
                etype = events[idx].get("type")
                if etype in ("recovery_complete", "recovery_failed"):
                    event = etype
                    break
                idx += 1
        if event == "recovery_complete":
            return {"status": "succeeded", "recovered": True, "recovery_event": event,
                    "recovery_after_s": round(self.w.time - deadline + 9.0, 3),
                    "agent_state": self.w.agent_state()}
        raise ActionError("recovery_failed", "The stand-up attempt ended without a sustained upright pose.",
                          event=event, fallen_after=bool(self.w.agent_fallen()))

    # ---------------------------------------------------------------- manipulation
    def reach(self, object_id=None, **extra) -> dict:
        if self.mc is None:
            raise ActionError("skill_unavailable", "No manipulation controller in this world.")
        if not object_id:
            raise ActionError("invalid_argument", "Reaching needs an object_id.")
        return dict(self.mc.reach(str(object_id)))

    def pick_up(self, object_id=None, **extra) -> dict:
        if not object_id:
            raise ActionError("invalid_argument", "Grasping needs an object_id.")
        res = dict(self.w.pick_up(str(object_id)))
        res["status"] = "succeeded"
        return res

    def lift(self, object_id=None, **extra) -> dict:
        """Take hold of an object and report the measured carried state."""
        already = False
        if object_id and self.w.held == str(object_id):
            already = True
        elif object_id:
            try:
                self.w.pick_up(str(object_id))
            except ActionError as exc:
                if exc.code != "already_holding":
                    raise
                already = True
        if not self.w.held:
            raise ActionError("nothing_held", "Nothing is held, so nothing is lifted.")
        st = self.w.agent_state()
        return {"status": "succeeded", "held": self.w.held, "already_held": already,
                "carried_mass": st.get("carried_mass"), "feet_height": st.get("feet_height")}

    def carry(self, direction=None, distance: float = 0.5, speed: float | None = None, **extra) -> dict:
        if not self.w.held:
            raise ActionError("nothing_held", "Pick something up before carrying it.")
        oid = self.w.held
        start = np.asarray(self.w.obj_pos(oid), dtype=float)[:2].copy()
        res = self._move(direction, distance, speed)
        moved = float(np.linalg.norm(np.asarray(self.w.obj_pos(oid), dtype=float)[:2] - start))
        res.update({"status": "succeeded", "carried": oid, "object_displacement": round(moved, 4)})
        return res

    def throw(self, direction=None, force: float = 60.0, object_id=None, wait: str = "rest", **extra) -> dict:
        oid = object_id or self.w.held
        if not oid:
            raise ActionError("nothing_held", "Nothing is held to throw.")
        if direction is None:
            raise ActionError("invalid_argument", "Throwing needs a direction.")
        res = dict(self.w.throw(str(oid), self._dir(direction), float(force), wait))
        res["status"] = "succeeded"
        return res

    def catch(self, object_id=None, max_wait: float = 5.0, **extra) -> dict:
        if not object_id:
            raise ActionError("invalid_argument", "Catching needs an object_id to watch for.")
        res = dict(self.w.catch(str(object_id), float(max_wait)))
        res["status"] = "succeeded"
        return res

    def climb(self, direction=None, **extra) -> dict:
        res = dict(self.w.climb(None if direction is None else self._dir(direction)))
        res["status"] = "succeeded"
        return res
