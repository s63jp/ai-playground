"""S1 capability evaluation — the HONEST mastery gate for an individual limb.

Given a trained checkpoint and an S1 limb stage, this runs held-out
DETERMINISTIC episodes on that limb's command schedule and measures ACTUAL
physical behaviour: per-joint steady tracking abs-err (rad), pinned-joint rest,
chest posture, and falls. It compares the measurements against the recorded
gate bands (reports/M7_1A_S1_PROBE.json — P2 measured err x margin) and returns
a verdict. There is NO training here: a limb is reported "mastered" ONLY on
measured evidence that meets every band, never on step count or reward.

Used by the CLI (`scripts/rl_ppo.py eval-s1`), the live-training session
(post-run capability report), and the tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .curriculum import M7_S1_STAGES
from .env import LocoEnv
from .rewards import LIMB_GROUPS

_ROOT = Path(__file__).resolve().parents[2]
_PROBE_JSON = _ROOT / "reports" / "M7_1A_S1_PROBE.json"

EVAL_SEED_BASE = 97_010_000      # the S1 held-out eval seed base (report §5)
EVAL_EPISODES = 12
PINNED_NEUTRAL_BAND_RAD = 0.30   # pinned joints (waist/ankle) must hold neutral
CHEST_UP_Z_MIN = 0.90            # B4 chest posture pass band (report §5.3)


def _stage_by_id(stage_id: str):
    for s in M7_S1_STAGES:
        if s.id == stage_id:
            return s
    raise ValueError(f"unknown S1 stage {stage_id!r}; choose one of "
                     f"{[s.id for s in M7_S1_STAGES[1:]]}")


def load_gate_bands(stage_id: str, probe_json: Path | None = None) -> dict:
    """Per-joint steady abs-err bands (rad) for a limb stage, from the recorded
    calibration probe. Raises if the evidence is missing — the mastery gate is
    never run without recorded bands."""
    path = Path(probe_json) if probe_json else _PROBE_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"calibration evidence {path} is missing — run "
            f"reports/m7_s1_probe.py before evaluating S1 capability")
    data = json.loads(path.read_text(encoding="utf-8"))
    grp = data.get("groups", {}).get(stage_id)
    if not grp:
        raise KeyError(f"probe evidence has no bands for stage {stage_id!r}")
    return dict(grp.get("gate_bands_abs_err_rad", {}))


def _chest_up_z(env) -> float:
    m = np.asarray(env.world.d.xmat[int(env.world.m.body("spine_L2").id)],
                   dtype=float).reshape(3, 3)
    return float(m[2, 2])


def _denorm(env, joint, norm: float) -> float:
    lo, hi = env.joint_range[joint]
    return (float(norm) + 1.0) / 2.0 * (hi - lo) + lo


def evaluate_s1_limb(checkpoint, stage_id: str, *, episodes: int = EVAL_EPISODES,
                     seed_base: int = EVAL_SEED_BASE, bands: dict | None = None,
                     write_to=None, policy=None) -> dict:
    """Run the held-out capability eval for ONE S1 limb.

    `checkpoint` is a path to a saved checkpoint, or None when `policy` (an
    object exposing deterministic_action) is supplied directly (tests). Returns
    a verdict dict: mastered (bool), per-joint measured steady abs-err vs band,
    pinned-joint rest, chest posture, falls, and the episode seeds used.
    """
    stage = _stage_by_id(stage_id)
    if stage_id == "standing":
        raise ValueError("standing is the shared base, not an S1 limb stage")
    bands = dict(bands) if bands is not None else load_gate_bands(stage_id)

    env = LocoEnv()
    env.set_phase(stage.phase)
    group = stage.phase["commands"]["limb_group"]
    idx = LIMB_GROUPS[group]
    names = [env.joint_names[i] for i in idx]

    if policy is None:
        if checkpoint is None:
            raise ValueError("provide a checkpoint path or a policy object")
        from .checkpoint import load_checkpoint
        policy = load_checkpoint(str(checkpoint), env=env)["actor"]

    # pinned joints: schedule candidate set is exactly {neutral}
    sched_sets = env._cmd_schedule["sets"]
    pinned = set()
    commanded = []
    for n in names:
        cands = sched_sets[n]
        # a joint whose only candidate maps to 0 rad is pinned-neutral
        rads = sorted({round(_denorm(env, n, c), 6) for c in cands})
        if rads == [0.0]:
            pinned.add(n)
        else:
            commanded.append(n)

    abs_err_sum = {n: 0.0 for n in names}
    hold_count = 0
    min_chest = 1.0
    falls = 0
    seeds = [int(seed_base + i) for i in range(episodes)]
    sched = env._cmd_schedule
    seg = sched["hold_windows"] + sched["transition_windows"]
    trans = sched["transition_windows"]

    for sd in seeds:
        obs = env.reset(seed=sd)
        fell_ep = False
        for k in range(env.max_steps):
            action = policy.deterministic_action(obs).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            min_chest = min(min_chest, _chest_up_z(env))
            if info.get("fallen") or term:
                fell_ep = True
                break
            # hold window = the flat tail of a segment (published window k+1)
            if ((k + 1) % seg) >= trans:
                hold_count += 1
                cmd = env._commands["limb_targets"]
                for i, n in zip(idx, names):
                    q = float(env.world.d.qpos[int(env.world.m.joint(n).qposadr[0])])
                    cmd_rad = _denorm(env, n, cmd[i])
                    abs_err_sum[n] += abs(q - cmd_rad)
        if fell_ep:
            falls += 1

    denom = max(hold_count, 1)
    steady = {n: abs_err_sum[n] / denom for n in names}

    joints = {}
    all_pass = True
    for n in names:
        if n in pinned:
            band = PINNED_NEUTRAL_BAND_RAD
            kind = "pinned_neutral"
        else:
            band = float(bands.get(n, 0.0))
            kind = "commanded"
        passed = (band > 0.0) and (steady[n] <= band)
        if not passed:
            all_pass = False
        joints[n] = {"kind": kind, "steady_abs_err_rad": round(steady[n], 5),
                     "band_rad": round(band, 5), "pass": bool(passed)}

    posture_pass = bool(min_chest >= CHEST_UP_Z_MIN)
    falls_pass = (falls == 0)
    mastered = bool(all_pass and posture_pass and falls_pass)

    verdict = {
        "stage_id": stage_id,
        "group": group,
        "episodes": episodes,
        "seeds": seeds,
        "mastered": mastered,
        "joints": joints,
        "commanded_joints": commanded,
        "pinned_joints": sorted(pinned),
        "posture": {"chest_up_z_min": round(min_chest, 4),
                    "band": CHEST_UP_Z_MIN, "pass": posture_pass},
        "falls": {"count": falls, "pass": falls_pass},
        "checkpoint": (str(checkpoint) if checkpoint is not None else None),
        "bands_source": (str(_PROBE_JSON) if bands is None else "provided"),
    }
    if write_to is not None:
        dest = Path(write_to)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        verdict["written_to"] = str(dest)
    return verdict
