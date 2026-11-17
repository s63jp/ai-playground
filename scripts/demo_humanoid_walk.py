#!/usr/bin/env python
"""Demo script: simple PD‑controlled walk for the articulated humanoid.

The script loads the default config with ``agent_body.type = "humanoid"`` and
creates an ``Environment`` (the high‑level API used by the agents). It then
applies a sinusoidal target to the hip joints while issuing ``move`` commands
to walk forward. The PD controller registered via ``set_joint_target`` will
apply torques each physics step, producing smooth limb motion.
"""

import math
import numpy as np
from pathlib import Path

import yaml

# Load config and switch to humanoid body
cfg_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
cfg["agent_body"]["type"] = "humanoid"

# Initialise the simulation world directly (no server needed)
from playground.world.physics import PhysicsWorld
world = PhysicsWorld(cfg)

# Helper to set hip joint targets (left and right) with a phase offset
def set_hip_targets(t, amplitude=0.3, frequency=1.0):
    # target angle in radians
    target = amplitude * math.sin(2 * math.pi * frequency * t)
    world.set_joint_target("left_hip", target, kp=20.0, kd=2.0)
    world.set_joint_target("right_hip", -target, kp=20.0, kd=2.0)

# Walk forward while applying the PD targets for a few seconds
walk_duration = 5.0  # seconds
speed = 0.5          # m/s forward
start_pos = world.agent_pos().copy()
start_time = world.time
print('time,pos_x,pos_y,pos_z,up_z')
while world.time - start_time < walk_duration:
    # Apply sinusoidal hip movement
    set_hip_targets(world.time)
    # Issue a forward move (the existing move tool handles slide joints)
    world.move([1.0, 0.0, 0.0], speed * world.dt, stop=False)
    # Step the simulation
    world._step()
    pos = world.agent_pos()
    quat = world.d.qpos[world.aq+3:world.aq+7]
    from playground.world.mathutil import quat_to_mat
    up = quat_to_mat(quat) @ np.array([0.0, 0.0, 1.0])
    print(f"{world.time:.2f},{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},{up[2]:.3f}")

end_pos = world.agent_pos().copy()
distance = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
print(f"Walked {distance:.3f} m in {walk_duration}s (expected ≈{speed * walk_duration:.3f} m)")
