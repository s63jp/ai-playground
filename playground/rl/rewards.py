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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Reward defaults (config key rl.reward). Weights are per decision window
# (20 ms at the default decimation). The standing-vs-walking arithmetic these
# imply, measured on the real model, is recorded in reports/m6b/B2_ENVIRONMENT.md.
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
}

# Component KEYS (the observable breakdown terms; DEFAULTS additionally holds
# config-only keys like tracking_sigma/stability_window/alive_margin).
COMPONENT_KEYS = ("progress", "velocity_tracking", "upright", "alive",
                  "effort", "smoothness", "stability", "fall")


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
    }
    total = float(sum(comp.values()))
    return total, {k: float(v) for k, v in comp.items()}
