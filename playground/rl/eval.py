"""Held-out evaluation for the locomotion RL environment (M6-B stage B3).

One harness for every comparison the milestone requires, over MATCHED
conditions: the same env configuration, episode horizon, target velocity
(stage phase), start-pose seed list (start poses) and physics settings for
every policy evaluated:

- fresh untrained neural policy (the B3 baseline, defined HERE — the old
  `Policy` class was deleted with the broken trainer at B2),
- trained neural policy (deterministic actions; never the rollout samples),
- scripted walking gait benchmark (a BENCHMARK, never labelled learned),
- zero-action control.

Training and evaluation data are separate by construction: evaluation
resets use a disjoint seed base (default 97010000+), episodes run
deterministically (actor mean action), and no evaluation call mutates any
parameter.
"""

from __future__ import annotations

import numpy as np

EVAL_SEED_BASE = 97_010_000     # disjoint from the training seed space
DEFAULT_EPISODES = 10
DEFAULT_MAX_STEPS = 400         # the 8-s benchmark horizon (400 x 20 ms)


def _stage_phase_dict(curriculum, stage_id):
    for s in curriculum.stages:
        if s.id == stage_id:
            return dict(s.phase)
    raise ValueError(f"unknown stage id {stage_id!r}")


def _summary_with_axis(env) -> dict:
    s = dict(env.episode_summary())
    tv = np.asarray(s.get("target_velocity") or (0.0, 0.0, 0.0), dtype=float)[:3]
    sp = float(np.linalg.norm(tv))
    if sp > 1e-9:
        h = tv / sp
        d = (np.asarray(s["final_position"]) - np.asarray(s["start_position"]))
        s["axis_displacement_m"] = float(d[:3] @ h)
    else:
        s["axis_displacement_m"] = 0.0
    cs = s.get("action_clipping") or {}
    s["action_clip_rate"] = float(cs.get("clipping_rate", 0.0))
    return s


def run_policy_episode(actor, env, *, seed, phase, max_steps=DEFAULT_MAX_STEPS,
                       deterministic=True) -> dict:
    """One held-out evaluation episode of a neural policy (deterministic
    mean action by default). Applies the phase, resets with the seed, runs
    to truncation/termination, returns the engine summary + axis displacement.
    Never mutates policy parameters."""
    env.set_phase(dict(phase))
    obs = env.reset(seed=int(seed))
    for _ in range(int(max_steps)):
        a = actor.deterministic_action(obs) if deterministic else None
        if a is None:
            raise ValueError("stochastic evaluation is not part of the B3 "
                             "protocol (deterministic=True only)")
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return _summary_with_axis(env)


def run_zero_episode(env, *, seed, phase, max_steps=DEFAULT_MAX_STEPS) -> dict:
    env.set_phase(dict(phase))
    env.reset(seed=int(seed))
    zero = np.zeros(env.action_dim, dtype=np.float32)
    for _ in range(int(max_steps)):
        obs, r, term, trunc, info = env.step(zero)
        if term or trunc:
            break
    return _summary_with_axis(env)


def run_gait_episode(env, *, seed, phase, heading=90.0, speed=1.0,
                     max_steps=DEFAULT_MAX_STEPS) -> dict:
    """The SCRIPTED walking benchmark under matched conditions (the gait
    is a benchmark, never a learned policy). `heading` selects the gait's
    walk direction (degrees; 90 = world +Y = the walkable chest axis)."""
    env.set_phase(dict(phase))
    env.reset(seed=int(seed))
    env.world.set_gait("walk", speed=float(speed), heading=float(heading))
    zero = np.zeros(env.action_dim, dtype=np.float32)
    for _ in range(int(max_steps)):
        obs, r, term, trunc, info = env.step(zero)
        if term or trunc:
            break
    return _summary_with_axis(env)


METRIC_KEYS = ("net_displacement_m", "axis_displacement_m", "mean_speed_m_s",
               "upright_fraction", "min_up_z", "effort_mean_w", "reward_total",
               "action_clip_rate", "path_len_m", "heading_progress_rewarded_m")


def summarize_episodes(episodes: list) -> dict:
    """Mean / std / median / min / max for the required eval metrics, the
    aggregate failure counters, and the UPRIGHT-TRAVEL breakdown: displacement
    statistics computed over episodes that stayed upright throughout (falls 0)
    are reported separately from fall-terminated episodes, so "travelled
    without falling" can never be conflated with "moved then collapsed"."""
    out = {}
    for key in METRIC_KEYS:
        vals = [float(e[key]) for e in episodes
                if e.get(key) is not None and np.isfinite(float(e[key]))]
        if vals:
            a = np.asarray(vals, dtype=float)
            out[key] = {"mean": float(a.mean()), "std": float(a.std()),
                        "median": float(np.median(a)), "min": float(a.min()),
                        "max": float(a.max())}
        else:
            out[key] = None
    out["episodes"] = len(episodes)
    out["falls_total"] = int(sum(int(e.get("falls", 0)) for e in episodes))
    out["faulted_windows_total"] = int(sum(int(e.get("faulted_windows", 0))
                                           for e in episodes))
    out["completed_truncated"] = int(sum(1 for e in episodes
                                         if e.get("truncated_at") is not None))
    for key in ("axis_displacement_m", "net_displacement_m"):
        upright_vals = [float(e[key]) for e in episodes
                        if int(e.get("falls", 0)) == 0
                        and e.get(key) is not None]
        fell_vals = [float(e[key]) for e in episodes
                     if int(e.get("falls", 0)) > 0 and e.get(key) is not None]
        def stat(vals):
            if not vals:
                return None
            a = np.asarray(vals, dtype=float)
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "min": float(a.min()), "max": float(a.max()), "n": len(vals)}
        out[f"{key}_no_fall"] = stat(upright_vals)
        out[f"{key}_fell"] = stat(fell_vals)
    return out


def eval_policy(actor, env, *, episodes=DEFAULT_EPISODES, seed_base=EVAL_SEED_BASE,
                phase, max_steps=DEFAULT_MAX_STEPS) -> dict:
    eps = [run_policy_episode(actor, env, seed=seed_base + i, phase=phase,
                              max_steps=max_steps)
           for i in range(int(episodes))]
    return {"episodes": eps, "summary": summarize_episodes(eps)}


def eval_zero(env, *, episodes=DEFAULT_EPISODES, seed_base=EVAL_SEED_BASE,
              phase, max_steps=DEFAULT_MAX_STEPS) -> dict:
    eps = [run_zero_episode(env, seed=seed_base + i, phase=phase,
                            max_steps=max_steps)
           for i in range(int(episodes))]
    return {"episodes": eps, "summary": summarize_episodes(eps)}


def eval_gait(env, *, episodes=1, seed_base=EVAL_SEED_BASE, phase,
              heading=90.0, speed=1.0, max_steps=DEFAULT_MAX_STEPS) -> dict:
    eps = [run_gait_episode(env, seed=seed_base + i, phase=phase,
                            heading=heading, speed=speed, max_steps=max_steps)
           for i in range(int(episodes))]
    return {"episodes": eps, "summary": summarize_episodes(eps)}


def eval_fresh_actor(env, *, episodes=DEFAULT_EPISODES, seed_base=EVAL_SEED_BASE,
                     phase, hidden=(64, 64), seed=0,
                     max_steps=DEFAULT_MAX_STEPS) -> dict:
    """The B3 untrained-policy baseline under matched conditions: a freshly
    initialised TanhGaussianActor (deterministic actions). NOTE: this is the
    fresh-baseline DEFINITION for the B2 contract era (historical 0.0245 m
    came from the deleted pre-M6 Policy class and stays an immutable
    historical context number, not a direct comparison)."""
    from .ppo import TanhGaussianActor
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, hidden=hidden, seed=seed)
    return eval_policy(actor, env, episodes=episodes, seed_base=seed_base,
                       phase=phase, max_steps=max_steps)
