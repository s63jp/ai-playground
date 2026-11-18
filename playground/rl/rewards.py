"""Authoritative reward pipeline for the locomotion RL environment (M6-B B2).

This module is the ONE place where reward is defined, summed and made
observable. Playground.rl.env builds a :class:`MeasuredStep` from real engine
measurements each decision window and calls :func:`compute_reward`; nothing
else in the package sums reward (the old second copy that lived in
rl/train.py is deleted — see docs/RL_ENVIRONMENT.md).

Every component consumes an ENGINE MEASUREMENT of the physical world, never a
normalized observation slice:

- ``progress``              metres advanced along the commanded heading, from
                            measured world positions, clamped per decision at
                            |v*|*dt so no single window can out-pay real travel.
- ``velocity_tracking``     exp(-||v - v*||^2 / sigma) on the FULL 3-D world
                            velocity: sideways and backwards motion cannot
                            out-score honest progress (anti-exploit, v1).
- ``upright``               min(torso, chest) up z in the world frame —
                            CHEST-AWARE since the B4 posture ask (b4_posture1,
                            Jamie 2026-11-18): the slide-root pelvis cannot
                            pitch, so the torso up-vector alone was blind to a
                            folding chest (measured in b4_head1: torso up ≈ 1
                            while chest_up_z mean 0.115 in the sustained fold).
- ``alive``                 ±1 on the same chest-aware upright margin.
- ``effort``                mechanical work the policy's own PD torques did
                            this window, sum(|qfrc_applied*qvel|)*dt over the
                            14 policy dofs, in joules (measured, not the
                            normalized command magnitude).
- ``smoothness``            action-rate penalty ||a - a_prev||^2 (control
                            chatter, not physics — documented as shaping).
- ``stability``             variance of up_z over the recent decision window
                            (body jitter penalty).
- ``fall``                  one-shot penalty when the episode terminated
                            because the figure fell (agent_fallen truth).

M7-0a v4 command-block components (config-default 0.0; every one gates on its
cmd_mask bit — an inactive group can never contribute, whatever the coeff):
- ``limb_tracking``      −w · mean over the active limb-group joints of
                         (measured − commanded)² in normalized joint units.
- ``com_tracking``       −w · squared metres error of the measured full-body
                         COM against the commanded shift from the measured
                         standing anchor.
- ``approach_progress``  +w · metres the agent moved toward the env's target
                         point this window (never negative).
- ``stop_tolerance``     −w · metres beyond the stop band at window end, only
                         when the stop request is armed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Reward defaults (config key rl.reward). Weights are per decision window
# (20 ms at the default decimation). The standing-vs-walking arithmetic these
# imply, measured on the real model, is recorded in reports/m6b/B2_ENVIRONMENT.md.
#
# M7-0a (env contract v4, reports/M7_0A_ENV_V4_CHANGE.md): FOUR NEW KEYED
# COMPONENTS with config-default 0.0 — every existing fixture and run computes
# the identical reward until a v4 stage turns a mask+coefficient on (M7-1+).
# Exact equations live in the change doc §4; summary:
#   limb_tracking     −c · mean_{j∈limb_group} (p_j − t_j)²        (normalized joints)
#   com_tracking      −c · ‖(com_xy − anchor) − n⊙scale‖²          (metres²)
#   approach_progress +c · max(0, d_before − d_after)              (metres/window)
#   stop_tolerance    −c · max(0, d_after − stop_band_m) · stop    (metres)
# All four gate on the cmd_mask bits (mask off ⇒ the term cannot contribute).
DEFAULTS = {
    "progress": 8.0,           # per metre of measured heading progress
    "velocity_tracking": 1.0,  # kernel weight
    "tracking_sigma": 0.1,     # rad of the exp kernel (kept v1 for comparability)
    "upright": 0.5,
    "alive": 1.0,
    "alive_margin": 0.4,
    "effort": 0.005,           # per joule of measured work (walk ~3.3 J/window)
    "smoothness": 0.1,
    "stability": 0.2,
    "stability_window": 25,
    "fall": 5.0,
    # ---- M7-0a command-block components (v4 contract; defaults 0.0) ----
    "limb_tracking": 0.0,
    "com_tracking": 0.0,
    "approach_progress": 0.0,
    "stop_tolerance": 0.0,
    "stop_band_m": 0.5,
}

# Component KEYS (the observable breakdown terms; DEFAULTS additionally holds
# config-only keys like tracking_sigma/stability_window/alive_margin/stop_band_m).
COMPONENT_KEYS = ("progress", "velocity_tracking", "upright", "alive",
                  "effort", "smoothness", "stability", "fall",
                  "limb_tracking", "com_tracking", "approach_progress",
                  "stop_tolerance")

# ---- Frozen v4 command constants (MEASURED; M7-0 probes P3/P4) ----
# com anchor: the settled full-body (47.05 kg) standing COM xy
# (ai-playground-audit/m7_p4_support_hull.json, seed 20261117).
COM_ANCHOR_XY = (-0.07926, 0.08871)
# com_shift command conversion: normalized n ∈ [−1,1] → metres (P4 margins:
# x ±6.50 cm, y 3.08/6.92 cm at the command bound corners).
COM_SHIFT_SCALE_XY = (0.070, 0.040)
# waypoint normalisation: d̂ = dist / D_REF (P3: horizon-reach 400 × 0.02 s ×
# 0.5 m/s; island half 12.0 m bounds placement).
WAYPOINT_D_REF = 4.0

# M7 limb groups over the 15-v4 joint order (indices into _V4_JOINT_ORDER
# in rl/env.py). The active group is env state (`limb_group`); "all" covers
# every commanded joint (the M7-0a default).
LIMB_GROUPS = {
    "all": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
    "torso": (0,),
    "left_arm": (1, 2),
    "right_arm": (3, 4),
    "head": (5, 6, 7),
    "left_leg": (8, 9, 10),
    "right_leg": (11, 12, 13),
    "waist": (14,),
}

# Default zero command state (all groups inactive): used by MeasuredStep so
# hand-built fixtures stay bit-identical without importing env machinery.
ZERO_COMMANDS = {
    "mask": (0, 0, 0, 0),
    "limb_targets": None,
    "com_shift": (0.0, 0.0),
    "waypoint": (0.0, 0.0, 0.0),
    "stop": 0,
    "limb_group": "all",
}


def heading_progress_raw(delta_position, target_velocity) -> float:
    """Metres moved along the commanded heading by the LAST window (max 0).

    Net-projection rule: oscillating back-and-forth movement nets out to ~0
    and backward walking pays nothing, so progress cannot be farmed by any
    repeated motion that does not close distance along the heading.
    """
    t = np.asarray(target_velocity, dtype=np.float64)[:3]
    speed = float(np.linalg.norm(t))
    if speed <= 0.0:
        return 0.0
    heading = t / speed
    d = np.asarray(delta_position, dtype=np.float64)[:3]
    return max(0.0, float(d @ heading))


def progress_reward(delta_position, target_velocity, window_s, weight) -> float:
    """Clamped heading-progress reward.

    The raw measured displacement along the heading is capped at
    |v*| * window_s so a policy cannot collect more reward than sustained
    motion at the commanded speed would pay (duty-cycled bursts can never
    out-earn the sustained walk that travels the same distance in the same
    time; real displacement, not velocity sampling, is what pays).
    """
    t = np.asarray(target_velocity, dtype=np.float64)[:3]
    speed = float(np.linalg.norm(t))
    raw = heading_progress_raw(delta_position, t)
    cap = speed * max(0.0, float(window_s))
    return weight * min(raw, cap)


def velocity_tracking_reward(velocity, target_velocity, sigma, weight) -> float:
    """exp(-||v - v*||^2 / sigma) on the full 3-D WORLD velocity error."""
    vel = np.asarray(velocity, dtype=np.float64)[:3]
    t = np.asarray(target_velocity, dtype=np.float64)[:3]
    err = vel - t
    return weight * float(np.exp(-(err @ err) / float(sigma)))


def upright_reward(up_z, weight) -> float:
    """Per-decision upright posture value: torso up z in the world frame."""
    return weight * float(up_z)


def alive_reward(up_z, margin_threshold, reward) -> float:
    """+reward while the upright margin stays above the threshold, else -reward."""
    return float(reward) if float(up_z) > float(margin_threshold) else -float(reward)


def effort_penalty(effort_j, weight) -> float:
    """-weight * (mechanical work done by the policy's PD torques this window)."""
    return -float(weight) * max(0.0, float(effort_j))


def smoothness_penalty(action, prev_action, weight) -> float:
    """-weight * ||a_t - a_{t-1}||^2 over the CLIPPED, mapped-encoded actions."""
    d = np.asarray(action, dtype=np.float64) - np.asarray(prev_action, dtype=np.float64)
    return -float(weight) * float(d @ d)


def stability_penalty(up_z_window, weight) -> float:
    """-weight * variance of up_z over the recent decision window."""
    w = [float(x) for x in up_z_window]
    if len(w) < 2:
        return 0.0
    arr = np.asarray(w)
    return -float(weight) * float(arr.var())


# ---------------------------------------------------------------- M7-0a (v4)
def limb_tracking_penalty(m: MeasuredStep, weight) -> float:
    """−w · mean_{j∈active group} (measured_norm_j − commanded_norm_j)².

    MEASURED joint positions (m.joint_pos_norm) against the commanded
    normalized targets; consumes only joints in the ACTIVE limb group. Gates:
    mask bit 0 (limb) ⇒ 0.0; no measured positions ⇒ 0.0.
    """
    cmd = m.commands
    if not cmd.get("mask", (0, 0, 0, 0))[0]:
        return 0.0
    targets = cmd.get("limb_targets")
    if m.joint_pos_norm is None or targets is None:
        return 0.0
    idx = LIMB_GROUPS.get(cmd.get("limb_group", "all"))
    if not idx:
        return 0.0
    p = np.asarray(m.joint_pos_norm, dtype=float)
    t = np.asarray(targets, dtype=float)
    err = p[list(idx)] - t[list(idx)]
    return -float(weight) * float(np.mean(err * err))


def com_tracking_penalty(m: MeasuredStep, weight) -> float:
    """−w · ‖(com_xy − anchor) − n ⊙ scale‖² (metres²).

    The MEASURED full-body COM against the commanded normalized shift from
    the measured standing anchor. Gate: mask bit 1 (balance) ⇒ live; no
    measured COM ⇒ 0.0.
    """
    cmd = m.commands
    if not cmd.get("mask", (0, 0, 0, 0))[1]:
        return 0.0
    if m.com_xy is None:
        return 0.0
    n = np.asarray(cmd.get("com_shift", (0.0, 0.0)), dtype=float)
    off = np.asarray(m.com_xy, dtype=float) - np.asarray(COM_ANCHOR_XY, dtype=float)
    want = n * np.asarray(COM_SHIFT_SCALE_XY, dtype=float)
    err = off - want
    return -float(weight) * float(err @ err)


def approach_progress_reward(m: MeasuredStep, weight) -> float:
    """+w · max(0, d_before − d_after) metres toward the target point.

    Gate: mask bit 2 (waypoint) AND an env target point (no placement
    machinery exists at M7-0a ⇒ the env state is None ⇒ 0.0).
    """
    cmd = m.commands
    if not cmd.get("mask", (0, 0, 0, 0))[2]:
        return 0.0
    if not m.dist_to_target:
        return 0.0
    before, after = m.dist_to_target
    return float(weight) * float(max(0.0, float(before) - float(after)))


def stop_tolerance_penalty(m: MeasuredStep, weight, stop_band_m) -> float:
    """−w · max(0, d_after − stop_band_m) while the stop request is armed.

    Gate: mask bit 3 (stop) AND an env target point. stop_band_m is a config
    key (default 0.5 m; re-measured at M7-5 from real stop rows).
    """
    cmd = m.commands
    if not cmd.get("mask", (0, 0, 0, 0))[3]:
        return 0.0
    if not cmd.get("stop"):
        return 0.0
    if not m.dist_to_target:
        return 0.0
    after = float(m.dist_to_target[1])
    excess = max(0.0, after - float(stop_band_m))
    return -float(weight) * excess


@dataclass
class MeasuredStep:
    """Engine measurements for ONE decision window (built by LocoEnv.step).

    All fields are raw engine quantities in SI units (m, m/s, rad/s, J).
    """

    position_before: np.ndarray      # agent root world position at window start
    position_after: np.ndarray       # agent root world position at window end
    velocity: np.ndarray             # torso linear velocity (world, m/s)
    up_z: float                      # torso up-vector z (world)
    effort_j: float                  # measured policy-torque work, joules
    window_s: float                  # simulated seconds in this window
    fell: bool                       # terminated because agent_fallen() tripped
    up_z_window: list = field(default_factory=list)  # recent up_z samples (stability)
    chest_up_z: float = 1.0          # chest (spine_L2 body) up-vector z, world
                                     # frame, [−1, 1], +1 upright. B4 posture
                                     # ask: built by LocoEnv.step; default 1.0
                                     # keeps hand-built fixtures at the old
                                     # (pelvis-only) behaviour bit-for-bit.
    target_velocity: tuple = (0.5, 0.0, 0.0)
    # ---- M7-0a v4 command-block fields (defaults keep fixtures identical) ----
    commands: dict = field(default_factory=lambda: dict(ZERO_COMMANDS))
                                     # the LIVE command state (mask bits 0/1,
                                     # limb targets normalized [15], com_shift
                                     # normalized [2], waypoint [3], stop, and
                                     # the active limb_group name). v3-mode
                                     # envs never set it (stays ZERO_COMMANDS).
    joint_pos_norm: object = None    # engine-measured normalized joint positions
                                     # (15-vector, v4 envs; None otherwise) —
                                     # limb_tracking consumes MEASURED state,
                                     # never the observation slice.
    com_xy: object = None            # engine-measured full-body COM xy, metres
                                     # (incl. the root column; None in v3 mode
                                     # or before the first measurement).
    dist_to_target: object = None    # (d_before, d_after) metres to the env's
                                     # target point, or None when no target
                                     # point exists (M7-0a default).


def compute_reward(m: MeasuredStep, action, prev_action, coeffs: dict | None = None):
    """Sum the authoritative reward; returns (total, components).

    `components` carries every term separately so an episode's score can
    always be explained term by term (mission requirement; consumed by
    LocoEnv.info['components'] and episode_summary()).
    """
    c = dict(DEFAULTS)
    if coeffs:
        unknown = set(coeffs) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown reward coefficients: {sorted(unknown)}")
        c.update(coeffs)
    action = np.asarray(action, dtype=np.float64)
    prev = np.asarray(prev_action, dtype=np.float64)
    # B4 posture ask (b4_posture1, ONE reward change per Jamie's decision):
    # the upright/alive margin is CHEST-AWARE — min of the pelvis and chest
    # (spine_L2 body) world up-vector z components. The pelvis cannot pitch
    # (slide root, no pitch dof), so the torso term alone was blind to the
    # chest fold; the chest term is what the folding failure actually lives on.
    # Standing equilibrium (spine 0 => chest_up_z 1.0) is bit-identical:
    # min(1, 1) = 1. Nothing else changes — stability stays on torso up_z
    # (repoint deferred), the fall truth stays agent_fallen(), and chest_up_z
    # is an engine MEASUREMENT, never an observation slice.
    posture = min(float(m.up_z), float(m.chest_up_z))
    comp = {
        "progress": progress_reward(m.position_after - m.position_before,
                                    m.target_velocity, m.window_s, c["progress"]),
        "velocity_tracking": velocity_tracking_reward(
            m.velocity, m.target_velocity, c["tracking_sigma"], c["velocity_tracking"]),
        "upright": upright_reward(posture, c["upright"]),
        "alive": alive_reward(posture, c["alive_margin"], c["alive"]),
        "effort": effort_penalty(m.effort_j, c["effort"]),
        "smoothness": smoothness_penalty(action, prev, c["smoothness"]),
        "stability": stability_penalty(m.up_z_window, c["stability"]),
        "fall": -float(c["fall"]) if m.fell else 0.0,
        # ---- M7-0a v4 command-block components (all config-default 0.0; all
        # gate on their cmd_mask bit — inactive groups CANNOT contribute) ----
        "limb_tracking": limb_tracking_penalty(m, c["limb_tracking"]),
        "com_tracking": com_tracking_penalty(m, c["com_tracking"]),
        "approach_progress": approach_progress_reward(m, c["approach_progress"]),
        "stop_tolerance": stop_tolerance_penalty(m, c["stop_tolerance"],
                                                 c["stop_band_m"]),
    }
    total = float(sum(comp.values()))
    return total, {k: float(v) for k, v in comp.items()}
