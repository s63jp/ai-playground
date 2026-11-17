"""Humanoid locomotion experiment lab (Nov 17).

Runs controlled A/B experiments on the humanoid rig and records three
metric families separately:

  translation      root displacement, average velocity
  postural         torso height drift, orientation, joint-limit proximity,
                   progressive joint drift, foot ground penetration
  gait quality     hip L/R antiphase correlation, foot swing clearance,
                   extreme-acceleration events, NaN flags

Standing matrix (per the review protocol):
  {no controller | gait-stand} x {asset-original friction | engine friction}
The friction factor is cfg-switched via agent_body.limb_friction_engine
(true = frictionless/normal-only limb contacts; false = asset original).

Walking isolation: traction ramp tested INDEPENDENTLY of the joint
parameter changes (2x2: ramp x params), both friction configs reported.

Usage:
  python scripts/humanoid_walk_lab.py [--suite stand|walk|all] [--json PATH]
Writes reports/humanoid_walk_lab.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

JOINTS = ("left_ankle", "right_ankle", "left_knee", "right_knee",
          "left_hip", "right_hip", "spine_L2", "left_shoulder", "right_shoulder")
ACC_ELEVATED = 1e4   # |qacc| threshold counted as an instability event

# Walking parameter presets matched to the gait-controller knobs.
PARAMS_GENTLE = {"step_frequency": 0.9, "hip_amplitude": 0.22, "knee_amplitude": 0.05,
                 "hip_damp": 8.0, "ankle_damp": 5.0}
PARAMS_ORIGINAL = {"step_frequency": 1.4, "hip_amplitude": 0.30, "knee_amplitude": 0.15,
                   "hip_damp": 5.0, "ankle_damp": 3.0}


def _load_cfg(friction_engine: bool) -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["agent_body"]["type"] = "humanoid"
    cfg["agent_body"]["limb_friction_engine"] = bool(friction_engine)
    return cfg


def _foot_geoms(w):
    ids = []
    for bname in ("left_foot", "right_foot"):
        try:
            bid = int(w.m.body(bname).id)
        except (ValueError, KeyError):
            continue
        for g in range(w.m.ngeom):
            if int(w.m.geom_bodyid[g]) == bid:
                ids.append(g)
    return ids


def _repro_meta(controller, friction_engine, duration, gait_overrides):
    """Reproducibility record per the review protocol: the results must stay
    traceable even if the working tree changes while an experiment runs."""
    import hashlib
    import subprocess
    src = os.path.abspath(__file__)
    try:
        with open(src, "rb") as f:
            src_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        src_hash = "unavailable"
    commit = "unavailable"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10,
                                cwd=os.path.dirname(src)).stdout.strip() or "dirty"
    except Exception:
        pass
    import mujoco
    return {
        "harness_source_sha256_16": src_hash,
        "git_commit": commit,
        "mujoco_version": mujoco.__version__,
        "working_tree_dirty_hint": "see git status; physics.py/spec.py edits are uncommitted",
        "controller": controller,
        "friction_engine": friction_engine,
        "duration_s": duration,
        "gait_overrides": gait_overrides or {},
        "seed": "none-deterministic",
    }


def run_case(name, controller, friction_engine,
             duration=10.0, gait_overrides=None, speed=1.0):
    """Run one case; returns metrics dict. controller: 'none'|'stand'|'walk'."""
    from playground.world.physics import PhysicsWorld

    cfg = _load_cfg(friction_engine)
    w = PhysicsWorld(cfg)
    start_pos = np.array(w.agent_pos().copy())
    torso_bid = int(w.m.body("torso").id)
    fgs = _foot_geoms(w)

    if controller != "none":
        if controller == "walk":
            w.set_gait("walk", speed=speed)
        else:
            w.set_gait(controller)
        if gait_overrides and w._gait_controller is not None:
            w._gait_controller.cfg.update(gait_overrides)

    qadr = {j: int(w.m.joint(j).qposadr[0]) for j in JOINTS}
    # Limit proximity is computed ONLY for joints the compiled model confirms
    # as limited (m.jnt_limited compiles Boolean-like {0,1}, not a tristate)
    # with a valid, non-zero range span — never silently classify unlimited
    # joints as limited.
    lim_norm = {}
    for j in JOINTS:
        jid = int(w.m.joint(j).id)
        lim = int(w.m.jnt_limited[jid])
        rng = (float(w.m.jnt_range[jid][0]), float(w.m.jnt_range[jid][1]))
        if lim == 1 and (rng[1] - rng[0]) > 1e-9:
            lim_norm[j] = max(abs(rng[0]), abs(rng[1])) or 1.0

    dt = float(w.dt)
    series = []
    max_qacc = 0.0
    unstable_steps = 0
    nan_seen = False
    foot_bottom_min = float("inf")
    sample_every = max(1, int(0.25 / dt))

    for i in range(int(duration / dt)):
        w._step()
        qa = float(np.max(np.abs(w.d.qacc)))
        if not math.isfinite(qa):
            nan_seen = True
        else:
            if qa > max_qacc:
                max_qacc = qa
            if qa > ACC_ELEVATED:
                unstable_steps += 1
        if i % sample_every == 0:
            angles = {}
            for j in JOINTS:
                v = float(w.d.qpos[qadr[j]])
                if math.isfinite(v):
                    angles[j] = round(v, 5)
            foot_bots = [float(w.d.geom_xpos[g][2]) - float(w.m.geom_rbound[g]) for g in fgs]
            fz = min(foot_bots) if foot_bots else float("nan")
            if math.isfinite(fz):
                foot_bottom_min = min(foot_bottom_min, fz)
            torso_xmat = np.array(w.d.xmat[torso_bid]).reshape(3, 3)
            series.append({
                "t": round(float(w.time), 4),
                "root_z": round(float(w.agent_pos()[2]), 4),
                "torso_z": round(float(w.d.xpos[torso_bid][2]), 4),
                "up_z": round(float(torso_xmat[2, 2]), 4),
                "angles": angles,
                "foot_min_z": round(float(fz), 5),
                "qacc": round(qa, 1),
            })

    pos = np.array(w.agent_pos().copy())
    disp = float(np.hypot(pos[0] - start_pos[0], pos[1] - start_pos[1]))
    ramp_t = float((gait_overrides or {}).get("traction_ramp_s", 1.5))
    # conservative lower-bound mean velocity: displacement achieved after the
    # ramp window (assumes all translation happened in the last duration-ramp_t)
    mean_v = disp / max(duration - ramp_t, 1e-6)

    zs = [s["root_z"] for s in series]
    ups = [s["up_z"] for s in series]
    z_drift = max(abs(s["root_z"] - zs[1]) for s in series[1:]) if len(zs) > 1 else 0.0
    joint_drift = {}
    limit_frac = {}
    for j in JOINTS:
        vals = [s["angles"].get(j) for s in series if j in s["angles"]]
        vals = [v for v in vals if v is not None]
        if vals:
            a1 = vals[min(1, len(vals) - 1)]
            joint_drift[j] = round(max(abs(v - a1) for v in vals), 4)
            if j in lim_norm:
                limit_frac[j] = round(max(abs(v) for v in vals) / lim_norm[j], 3)
    hl = np.array([s["angles"].get("left_hip", float("nan")) for s in series])
    hr = np.array([s["angles"].get("right_hip", float("nan")) for s in series])
    mask = np.isfinite(hl) & np.isfinite(hr)
    antiphase = None
    if mask.sum() > 4:
        c = np.corrcoef(hl[mask], hr[mask])[0, 1]
        antiphase = float(c) if math.isfinite(c) else None

    eff_cfg = dict(w._gait_controller.cfg) if w._gait_controller is not None else None
    return {
        "name": name,
        "meta": _repro_meta(controller, friction_engine, duration, gait_overrides),
        "timestep": dt,
        "effective_gait_cfg": eff_cfg,
        "controller": controller,
        "friction_engine": friction_engine,
        "gait_overrides": gait_overrides or {},
        # translation
        "disp_m": round(disp, 4),
        "mean_v_lbound_mps": round(mean_v, 4),
        # postural
        "root_z_end": round(float(w.agent_pos()[2]), 4),
        "up_z_min": round(min(ups), 4) if ups else float("nan"),
        "root_z_max_abs_drift_after_settle": round(float(z_drift), 4),
        "joint_drift_rad": joint_drift,
        "joint_limit_proximity_max": (max(limit_frac.values()) if limit_frac else 0.0),
        "foot_bottom_min_z": round(float(foot_bottom_min), 5),
        # gait quality
        "hip_antiphase_corr": round(antiphase, 3) if antiphase is not None else None,
        "foot_swing_clearance_max": (round(max(s["foot_min_z"] for s in series), 5)
                                     if series else float("nan")),
        "max_qacc": f"{max_qacc:.3g}",
        "unstable_steps_gt1e4": unstable_steps,
        "nan_seen": nan_seen,
        "fallen": bool(w.agent_fallen()),
        "series": series,
    }


def suite_stand():
    out = []
    for fric in (False, True):
        label = "engine" if fric else "asset"
        out.append(run_case(f"stand/noctl/{label}", "none", fric, 10.0))
        out.append(run_case(f"stand/gait/{label}", "stand", fric, 10.0))
    return out


def suite_walk():
    out = []
    for fric in (False, True):
        label = "engine" if fric else "asset"
        for pname, pv in (("orig", PARAMS_ORIGINAL), ("gentle", PARAMS_GENTLE)):
            for ramp in (0.0, 1.5):
                ov = dict(pv)
                ov["traction_ramp_s"] = ramp
                out.append(run_case(
                    f"walk/{label}/{pname}/ramp{ramp:g}", "walk", fric,
                    8.0, gait_overrides=ov))
    return out


def summarize(res):
    drift = res.get("joint_drift_rad", {})
    worst_drift = max(drift.values()) if drift else 0.0
    worst_joint = max(drift, key=drift.get) if drift else "-"
    lim = res.get("joint_limit_proximity_max", 0.0)
    return (f"{res['name']:30s} disp={res['disp_m']:7.3f} z_end={res['root_z_end']:7.3f} "
            f"upz_min={res['up_z_min']:6.3f} drift={worst_drift:6.3f}({worst_joint}) "
            f"limfrac={lim:4.2f} qacc={res['max_qacc']:>8s} uns={res['unstable_steps_gt1e4']:5d} "
            f"corr={res['hip_antiphase_corr']} fallen={res['fallen']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["stand", "walk", "all"])
    ap.add_argument("--json", default=os.path.join(
        os.path.dirname(__file__), "..", "reports", "humanoid_walk_lab.json"))
    args = ap.parse_args()

    results = []
    if args.suite in ("stand", "all"):
        print("=== STANDING BASELINE MATRIX (10 s) ===")
        for r in suite_stand():
            results.append(r)
            print(summarize(r))
    if args.suite in ("walk", "all"):
        print("=== WALKING ISOLATION MATRIX (8 s) ===")
        for r in suite_walk():
            results.append(r)
            print(summarize(r))

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"full metrics + series -> {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()
