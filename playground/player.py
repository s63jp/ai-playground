"""Player controller for human input.
Applies movement, yaw/pitch, jump and run to the PhysicsWorld using the existing API.
"""

import math
from typing import Any


class PlayerController:
    def __init__(self, world: Any, cfg: dict | None = None):
        self.world = world
        self.cfg = cfg or {}
        # Default limits from world acceleration config if present
        ac = getattr(world, "ac", {})
        self.max_force = self.cfg.get("max_force", float(ac.get("walk_force", 10.0)))
        self.max_speed = self.cfg.get("max_speed", float(ac.get("max_speed", 2.0)))
        self.jump_force = self.cfg.get("jump_force", float(ac.get("jump_force", 7.0)))
        self.jump_cooldown = self.cfg.get("jump_cooldown", 0.2)  # seconds
        self._last_jump_time = -self.jump_cooldown
        self._state = {
            "move": [0.0, 0.0],
            "yaw": 0.0,
            "pitch": 0.0,
            "jump": False,
            "run": False,
        }
        # register step callback
        if hasattr(world, "register_step_callback"):
            world.register_step_callback(self._step)

    def apply_input(self, state: dict) -> dict:
        """Update internal desired state and apply forces for this step.
        Returns the current controller state after processing.
        """
        # merge provided keys
        for k in self._state:
            if k in state:
                self._state[k] = state[k]
        # calculate target velocity vector based on yaw and move inputs
        move = self._state["move"]
        yaw = math.radians(self._state["yaw"])
        # forward vector
        fwd = (math.cos(yaw), math.sin(yaw))
        # right (strafe) vector
        right = (-math.sin(yaw), math.cos(yaw))
        vx = fwd[0] * move[1] + right[0] * move[0]
        vy = fwd[1] * move[1] + right[1] * move[0]
        # speed scaling: the move vector is a unit input — full deflection
        # commands the world's max_speed (doubled while running); the world's
        # force-capped traction still bounds the achievable acceleration.
        # (The old min(…, 1.0) clamp meant [0,1] mapped to 1 m/s and the run
        # multiplier could never amplify anything.)
        speed = math.hypot(vx, vy)
        if speed > 0:
            scale = self.max_speed * (2.0 if self._state["run"] else 1.0) / speed
            vx *= scale
            vy *= scale
        # apply to world target velocity (clamped by max_force inside world step)
        if hasattr(self.world, "target_vel"):
            self.world.target_vel[0] = vx
            self.world.target_vel[1] = vy
        # handle jump
        now = getattr(self.world, "time", 0.0)
        if self._state["jump"] and now - self._last_jump_time >= self.jump_cooldown:
            if getattr(self.world, "grounded", False):
                try:
                    self.world.jump()
                    self._last_jump_time = now
                except Exception:
                    pass
        return self._state.copy()

    def state(self) -> dict:
        """Return current positional and orientation state of the player.
        """
        pos = self.world.agent_pos()
        vel = self.world.agent_vel()
        return {
            "pos": pos.tolist() if hasattr(pos, "tolist") else list(pos),
            "vel": vel.tolist() if hasattr(vel, "tolist") else list(vel),
            "yaw": self._state["yaw"],
            "pitch": self._state["pitch"],
            "grounded": getattr(self.world, "grounded", False),
        }

    def camera_pose(self, mode: str = "first_person") -> dict:
        """Return eye and target vectors for the requested camera mode.
        """
        pos = self.world.agent_pos()
        # simple implementation: eye at agent height + offset, target forward
        yaw_rad = math.radians(self._state["yaw"])
        forward = (math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
        height = getattr(self.world, "hh", 0.5)
        if mode == "first_person":
            eye = [pos[0], pos[1], pos[2] + height]
            target = [eye[0] + forward[0], eye[1] + forward[1], eye[2]]
        else:  # third_person
            distance = 5.0
            eye = [pos[0] - forward[0] * distance, pos[1] - forward[1] * distance, pos[2] + height + 2.0]
            target = [pos[0], pos[1], pos[2] + height]
        return {"eye": eye, "target": target}

    def _step(self):
        """Placeholder step callback – currently no per‑step logic needed.
        """
        pass
