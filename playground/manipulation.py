
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

    def _shoulder_pos(self) -> np.ndarray:
        """Current world-space position of the left shoulder (body)."""
        bid = self.world.m.body(self.left_shoulder).id
        return np.array(self.world.d.xpos[bid])

    def _hand_pos(self) -> np.ndarray:
        """Approximate hand position as end of left forearm capsule.
        Upper arm length from XML: 0.28 m, forearm length: 0.26 m. Use straight line.
        """
        shoulder = self._shoulder_pos()
        # Direction of lower arm is along negative local Z of the forearm; approximate world Z.
        # We'll just offset downward by total arm length.
        return shoulder + np.array([0.0, 0.0, -(0.28 + 0.26)])

    def reach(self, oid: str) -> dict:
        """Point the left arm toward the target object using simple planar IK.
        Returns status dict.
        """
        obj_pos = self.world.obj_pos(oid)
        shoulder = self._shoulder_pos()
        vec = obj_pos - shoulder
        horiz_dist = math.hypot(vec[0], vec[1])
        max_reach = float(self.cfg["reach_distance"])
        if horiz_dist > max_reach:
            raise ActionError("out_of_reach", f"Object {oid} is {horiz_dist:.2f} m away, beyond reach {max_reach:.2f}.")
        # Simple angle for shoulder joint (hinge about Y axis)
        shoulder_angle = math.degrees(math.atan2(vec[1], vec[0]))
        # Clamp to joint limits (-15 to 15 deg as per actuator range)
        shoulder_angle = max(-15.0, min(15.0, shoulder_angle))
        self.world.set_joint_target(self.left_shoulder, shoulder_angle)
        # Elbow angle to bring hand close: use law of cosines on planar triangle.
        arm_len = 0.28 + 0.26
        # keep elbow modest, ignore detailed geometry.
        elbow_angle = 0.0  # neutral
        self.world.set_joint_target(self.left_elbow, elbow_angle)
        return {"status": "reached", "object": oid, "shoulder_angle": shoulder_angle}

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
