"""M7-1a untrained-baseline + limb_tracking coefficient probe (read-only).

reports/M7_1A_LIMB_STAGES_CHANGE.md §4/§5. NO training, NO model writes; the
only side effect is this script's own evidence JSON under reports/.

What it does, per S1 limb group (left_arm, right_arm, head, torso, left_leg,
right_leg, all):
  1. Replays the stage's command_schedule with the limb_tracking coefficient
     set to 1.0 so each window's reward component equals exactly -mean_err^2
     over the active group (normalized joint units — the same units the
     trained reward uses). Two control conditions:
        - zero-action   (PD holds neutral while commanded to the hold targets)
        - untrained policy (a fresh 256x256 TanhGaussianActor, deterministic
          mean action) — corroborates the zero-action "bad tracking" row.
  2. Reads the P2 PD-reference "good tracking" row for the same joints
     (ai-playground-audit/m7_p2_pd_reference.json) and converts its measured
     steady abs-err (rad) into the same normalized err^2 units.
  3. Fixes the limb_tracking coefficient w so:
        (a) signal floor:  w * (err2_bad - err2_p2) >= 0.05 reward/window for
            EVERY commanded group (the trained reward must separate good from
            bad tracking), and
        (b) equilibrium ceiling: when tracking well (err2 ~ err2_p2) the
            residual per-window penalty w * err2_p2 stays <= 5% of the base
            standing reward per window, so the coefficient cannot swamp the
            standing equilibrium.
  4. Records the §5 capability gate bands (per-joint P2 abs-err * margin 2,
     plus the verbatim B4 posture bands) as the launch spec.

The script DOES NOT modify curriculum.py or launch any run; it prints the
measured w and whether the provisional 2.0 is inside the measured window.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playground.rl.curriculum import M7_S1_STAGES          # noqa: E402
from playground.rl.env import LocoEnv                       # noqa: E402
from playground.rl.ppo import TanhGaussianActor             # noqa: E402
from playground.rl.rewards import LIMB_GROUPS               # noqa: E402

P2_PATH = ROOT.parent / "ai-playground-audit" / "m7_p2_pd_reference.json"

SEED = 97_010_000           # the S1 held-out eval seed base (report §5)
MARGIN = 2.0                # gate band = P2 abs-err * margin (report §5)
GRADIENT_FLOOR = 0.05       # required reward separation per window (report §4)
EQUIL_FRACTION = 0.05       # good-tracking residual penalty <= 5% base/window
POLICY_HIDDEN = (256, 256)  # the M7 architecture the ladder trains at
POLICY_SEED = 20_261_118

# Verbatim B4 posture bands (report §5.3) — recorded as the launch spec.
POSTURE_BANDS = {
    "head_rel_pitch_abs_max": [0.35, 0.60],
    "head_world_up_z_min": [0.9, 0.8],
    "chest_tilt_abs_max": [0.35, 0.60],
}


def _span(env, joint):
    lo, hi = env.joint_range[joint]
    return (hi - lo) if (hi - lo) != 0.0 else 1.0


def _p2_err2_norm(env, joint, p2_summary):
    """P2 good-tracking err^2 in normalized units for a joint, or None."""
    s = p2_summary.get(joint)
    if s is None:
        return None
    e_rad = float(s["abs_err_mean_max_rad"])
    # jpos_norm = (q - lo)/span*2 - 1  =>  d(norm) = 2/span * d(q)
    return (e_rad * 2.0 / _span(env, joint)) ** 2


def _replay(stage, policy=None):
    """Replay one stage's schedule. Returns per-window err^2 over the active
    group (via the limb_tracking=1.0 component), base reward/window, and a
    hold-window mask. Deterministic for a fixed seed."""
    env = LocoEnv()
    phase = copy.deepcopy(stage.phase)
    phase["reward_coeffs"] = dict(phase.get("reward_coeffs", {}))
    phase["reward_coeffs"]["limb_tracking"] = 1.0   # component == -mean_err^2
    env.set_phase(phase)
    obs = env.reset(seed=SEED)
    group = phase["commands"]["limb_group"]
    sched = env._cmd_schedule
    seg = sched["hold_windows"] + sched["transition_windows"]
    trans = sched["transition_windows"]

    err2, base, hold_mask = [], [], []
    for k in range(env.max_steps):
        if policy is None:
            action = np.zeros(env.action_dim, dtype=np.float32)
        else:
            action = policy.deterministic_action(obs).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)
        lt = float(info["components"].get("limb_tracking", 0.0))
        err2.append(-lt)                 # mean_err^2 over the active group
        base.append(float(reward) - lt)  # base reward without the limb penalty
        # the window published during step k is window (k+1); hold portion is
        # the flat tail of each segment (after the transition ramp)
        hold_mask.append(((k + 1) % seg) >= trans)
        if term:
            break
    return {"group": group,
            "err2": np.asarray(err2, dtype=float),
            "base": np.asarray(base, dtype=float),
            "hold": np.asarray(hold_mask, dtype=bool)}


def main():
    p2 = json.loads(P2_PATH.read_text(encoding="utf-8"))
    p2_summary = p2["summary"]
    env0 = LocoEnv()   # for joint ranges / names

    groups = [(s.id, s.phase["commands"]["limb_group"]) for s in M7_S1_STAGES[1:]]
    rows = {}
    for stage in M7_S1_STAGES[1:]:
        gid, group = stage.id, stage.phase["commands"]["limb_group"]
        policy = TanhGaussianActor(env0.obs_dim, env0.action_dim,
                                   hidden=POLICY_HIDDEN, seed=POLICY_SEED)
        zero = _replay(stage)
        pol = _replay(stage, policy=policy)

        def steady(r):
            m = r["hold"]
            return float(np.mean(r["err2"][m])) if m.any() else float(np.mean(r["err2"]))

        idx = LIMB_GROUPS[group]
        # P2 good-tracking floor over the group's joints (pinned joints with no
        # P2 case contribute 0 — commanded to neutral, they sit at neutral).
        per_joint_p2 = {}
        p2_terms = []
        for j in idx:
            name = env0.joint_names[j]
            e2 = _p2_err2_norm(env0, name, p2_summary)
            per_joint_p2[name] = e2
            p2_terms.append(0.0 if e2 is None else e2)
        err2_p2 = float(np.mean(p2_terms)) if p2_terms else 0.0

        # gate bands: per-joint steady abs-err band = P2 abs-err * margin (rad)
        gate_bands_rad = {}
        for j in idx:
            name = env0.joint_names[j]
            s = p2_summary.get(name)
            if s is not None:
                gate_bands_rad[name] = round(
                    float(s["abs_err_mean_max_rad"]) * MARGIN, 4)

        err2_bad_zero = steady(zero)
        err2_bad_pol = steady(pol)
        base_pw = float(np.mean(zero["base"]))
        rows[gid] = {
            "group": group,
            "joint_indices": list(idx),
            "err2_bad_zero_action": round(err2_bad_zero, 6),
            "err2_bad_untrained_policy": round(err2_bad_pol, 6),
            "err2_p2_reference": round(err2_p2, 6),
            "separation_zero": round(err2_bad_zero - err2_p2, 6),
            "base_reward_per_window": round(base_pw, 4),
            "n_hold_windows": int(zero["hold"].sum()),
            "per_joint_p2_err2_norm": {k: (None if v is None else round(v, 6))
                                       for k, v in per_joint_p2.items()},
            "gate_bands_abs_err_rad": gate_bands_rad,
        }

    # ---- fix w from the measured rows (report §4) -------------------------
    commanded = {gid: r for gid, r in rows.items()
                 if r["separation_zero"] > 0 and r["err2_p2_reference"] > 0}
    seps = {gid: r["separation_zero"] for gid, r in commanded.items()}
    equil = {gid: EQUIL_FRACTION * r["base_reward_per_window"]
             / r["err2_p2_reference"] for gid, r in commanded.items()
             if r["base_reward_per_window"] > 0}

    w_min = GRADIENT_FLOOR / min(seps.values()) if seps else None
    w_max = min(equil.values()) if equil else None
    w_floor_group = min(seps, key=seps.get) if seps else None
    w_ceil_group = min(equil, key=equil.get) if equil else None

    chosen = None
    feasible = (w_min is not None and w_max is not None and w_min <= w_max)
    if feasible:
        geo = float(np.sqrt(w_min * w_max))
        chosen = round(geo * 2) / 2.0           # nearest 0.5 in-range
        chosen = min(max(chosen, w_min), w_max)
    provisional_in_range = (feasible and w_min <= 2.0 <= w_max)

    out = {
        "probe": "m7_s1_probe",
        "date": "2026-11-19",
        "seed_base": SEED,
        "policy_hidden": list(POLICY_HIDDEN),
        "method": {
            "gradient_floor_per_window": GRADIENT_FLOOR,
            "equilibrium_fraction": EQUIL_FRACTION,
            "gate_margin": MARGIN,
            "units": "normalized joint positions (obs encoding, [-1,1])",
        },
        "groups": rows,
        "coefficient": {
            "w_min_signal_floor": None if w_min is None else round(w_min, 4),
            "w_min_binding_group": w_floor_group,
            "w_max_equilibrium_ceiling": None if w_max is None else round(w_max, 4),
            "w_max_binding_group": w_ceil_group,
            "feasible_window": feasible,
            "w_chosen": chosen,
            "provisional_2p0_in_range": provisional_in_range,
        },
        "posture_bands_B4": POSTURE_BANDS,
    }
    dest = ROOT / "reports" / "M7_1A_S1_PROBE.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"wrote {dest}")
    for gid, r in rows.items():
        print(f"  {gid:16s} bad(zero)={r['err2_bad_zero_action']:.4f} "
              f"bad(pol)={r['err2_bad_untrained_policy']:.4f} "
              f"p2={r['err2_p2_reference']:.4f} "
              f"sep={r['separation_zero']:.4f} base/w={r['base_reward_per_window']:.2f}")
    print(f"w_min(signal) = {w_min}  binding={w_floor_group}")
    print(f"w_max(equil)  = {w_max}  binding={w_ceil_group}")
    print(f"feasible={feasible}  w_chosen={chosen}  provisional_2.0_in_range={provisional_in_range}")


if __name__ == "__main__":
    main()
