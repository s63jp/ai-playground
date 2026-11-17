
"""Manipulation controller for P4.
Implements reach, grasp, release, push, pull.
Uses world.set_joint_target for PD control and world.pick_up for grip.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any

from .world.physics import PhysicsWorld, ActionError

DEFAULT_CFG = {
    "reach_distance": 1.5,   # max distance for arm reach (m)
    "grasp_tolerance": 0.07,  # max distance between hand and object to consider contact (m)
    "push_force_max": 120.0,   # same as CONTACT_VERBS limits for push
}

class ManipulationController:
    def __init__(self, world: PhysicsWorld, cfg: dict | None = None):
        self.world = world
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        # Simple mapping of arm joints
        self.left_shoulder = "left_shoulder"
        self.left_elbow = "left_elbow"
        # No explicit step callback registration: rely on world._pin_held handling.

    UPPER_ARM = 0.28   # MJCF capsule lengths of the hanging arm chain
    FOREARM = 0.26

    def _shoulder_pos(self) -> np.ndarray:
        """Current world-space position of the left shoulder (body)."""
        bid = self.world.m.body(self.left_shoulder).id
        return np.array(self.world.d.xpos[bid])

    def _hand_pos(self) -> np.ndarray:
        """MEASURED hand tip: forearm tip in world frame from the elbow body pose.

        The MJCF arm chain ends at the elbow body (the forearm capsule is the
        elbow body's geometry, extending FOREARM along the body's local -Z), so
        the honest hand position is the elbow body's pose rotated, never an
        assumed offset from a neutral pose.
        """
        bid = int(self.world.m.body(self.left_elbow).id)
        pos = np.asarray(self.world.d.xpos[bid], dtype=float)
        rot = np.asarray(self.world.d.xmat[bid], dtype=float).reshape(3, 3)
        return pos + rot @ np.array([0.0, 0.0, -self.FOREARM])

    def reach(self, oid: str) -> dict:
        """Point the left arm at the target object with 2-link sagittal-plane IK.

        The shoulder and elbow are X-hinges in the figure's sagittal plane; the
        target is projected onto that plane (side components are unreachable and
        the horizontal distance check still applies). Joint targets are set in
        RADIANS (they were degrees once — the old code drove the shoulder toward
        15.0 radians, i.e. its physical limit, see BUGLOG 2026-11-17). After
        settling, the MEASURED joint angles are compared against the request and
        'reached' is only reported when the arm actually tracked the targets:
        never report success because the target was accepted.
        """
        obj_pos = np.asarray(self.world.obj_pos(oid), dtype=float)
        shoulder = self._shoulder_pos()
        vec = obj_pos - shoulder
        horiz_dist = math.hypot(vec[0], vec[1])
        max_reach = float(self.cfg["reach_distance"])
        if horiz_dist > max_reach:
            raise ActionError("out_of_reach", f"Object {oid} is {horiz_dist:.2f} m away, beyond reach {max_reach:.2f}.")
        # Brace the figure with the existing stand posture while the arm aims:
        # without it the arm's swing topples/slides the free-standing column
        # (Nov 17 probe: the agent slid ~1 m and the aim read backwards). The
        # posture actuates the legs/spine through the standard gait pathway;
        # the shoulder/elbow targets below are separate joints and coexist.
        had_gait = getattr(self.world, "_gait_controller", None) is not None
        if not had_gait:
            self.world.set_gait("stand")
        # Sagittal plane: local forward (+Y when standing) and down (-Z).
        fwd = float(vec[1])
        down = -float(vec[2])
        d = math.hypot(fwd, down)
        total = self.UPPER_ARM + self.FOREARM
        # Keep off the straight-arm singular pole, but when the target sits at
        # or beyond full extension DO NOT bend the elbow to chase it: aim the
        # straight arm at the target and let the (measured, reported) shortfall
        # speak. An elbow sweep at full extension ploughs the forearm through
        # whatever sits at the hand (ejecta, BUGLOG 2026-11-17).
        if d >= total * 0.995:
            d_eff = d
        else:
            d_eff = min(max(d, 1e-3), total * 0.96)
        c = (self.UPPER_ARM ** 2 + d_eff ** 2 - self.FOREARM ** 2) / (2.0 * self.UPPER_ARM * d_eff)
        alpha = math.acos(max(-1.0, min(1.0, c)))          # interior angle at the shoulder
        c2 = (self.UPPER_ARM ** 2 + self.FOREARM ** 2 - d_eff ** 2) / (2.0 * self.UPPER_ARM * self.FOREARM)
        beta = math.acos(max(-1.0, min(1.0, c2)))          # interior angle at the elbow
        phi = math.atan2(fwd, max(down, 1e-6))             # 0 = hanging; + toward forward
        th_s = max(-0.05, min(1.6, phi - alpha))           # shoulder hinge (rad)
        th_e = max(0.0, min(2.4, math.pi - beta))          # elbow hinge folds forward (rad)
        elbow = f"{self.left_elbow}"
        self.world.set_joint_target(self.left_shoulder, th_s, kp=50.0, kd=7.0)
        self.world.set_joint_target(elbow, th_e, kp=35.0, kd=5.0)
        for _ in range(int(2.5 / self.world.dt)):
            self.world._step()
        sh = self.world.m.joint(self.left_shoulder)
        el = self.world.m.joint(elbow)
        got_s = float(self.world.d.qpos[int(sh.qposadr[0])])
        got_e = float(self.world.d.qpos[int(el.qposadr[0])])
        if abs(got_s - th_s) > 0.25:
            raise ActionError("reach_untracked", f"The shoulder settled {got_s:.3f} rad, not the requested {th_s:.3f} rad.")
        # Leave the world as it was found: a stand brace created here would keep
        # steering the next walk off course (Nov 17 probe: a 1.2 m walk lost its
        # heading with a leftover 'stand' controller; with the controller
        # restored to the pre-reach state the same walk covers 1.46 m).
        if not had_gait:
            self.world._gait_controller = None
        return {"status": "reached", "object": oid,
                "requested_shoulder_rad": round(th_s, 4), "requested_elbow_rad": round(th_e, 4),
                "shoulder_angle": round(math.degrees(got_s), 2), "shoulder_angle_rad": round(got_s, 4),
                "elbow_angle_rad": round(got_e, 4),
                "target_distance": round(horiz_dist, 3)}

    def grasp(self, oid: str) -> dict:
        """Attempt to grasp an object. Requires hand within grasp tolerance.
        Uses world.pick_up to attach the object.
        """
        hand = self._hand_pos()
        obj = self.world.obj_pos(oid)
        dist = float(np.linalg.norm(hand - obj))
        if dist > float(self.cfg["grasp_tolerance"]):
            raise ActionError("out_of_reach", f"Hand is {dist:.3f} m from object; need <= {self.cfg['grasp_tolerance']:.3f} m.")
        # Use existing world.pick_up which will pin the object.
        result = self.world.pick_up(oid)
        return {"status": "grasped", "object": oid, **result}

    def release(self) -> dict:
        """Release held object at current hand position."""
        if not self.world.held:
            raise ActionError("nothing_held", "No object is currently held.")
        oid = self.world.held
        hand = self._hand_pos()
        # Release with zero velocity relative to world.
        self.world._release(oid, hand, self.world.held_quat.copy(), np.zeros(3))
        return {"status": "released", "object": oid}

    def push(self, oid: str, direction: np.ndarray, force: float) -> dict:
        """Apply a contact impulse to an object (push)."""
        # Reuse world.impulse_object with verb='push'
        return self.world.impulse_object(oid, direction, force, verb='push')

    def pull(self, oid: str, direction: np.ndarray, force: float) -> dict:
        """Pull is push with opposite direction."""
        dir_vec = -np.array(direction, dtype=float)
        return self.world.impulse_object(oid, dir_vec, force, verb='push')
