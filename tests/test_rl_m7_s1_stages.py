"""M7-1a tests — the individual limb-control stages (S1), reports/
M7_1A_LIMB_STAGES_CHANGE.md: the separate M7 ladder, the command-schedule
seam, measured-set validation, per-seed determinism, the reward gate, and
the DEFAULT_STAGES fixture identity."""
import numpy as np
import pytest

from playground.rl.curriculum import DEFAULT_STAGES, M7_S1_STAGES, Curriculum
from playground.rl.env import LocoEnv
from playground.rl.rewards import (LIMB_GROUPS, MeasuredStep,
                                   limb_tracking_penalty)

_EXPECTED_DEFAULT_IDS = ("standing", "controlled", "sustained", "disturbance",
                         "terrain", "navigation")

_LIMB = slice(43, 58)   # the v4 limb_pose_target command block in the obs


def _zero_action(env):
    return np.zeros(env.action_dim, dtype=np.float32)


# --------------------------------------------------------------- the ladder
def test_m7_ladder_shape_and_ids():
    ids = tuple(s.id for s in M7_S1_STAGES)
    assert ids == ("standing", "m7_s1_left_arm", "m7_s1_right_arm",
                   "m7_s1_head", "m7_s1_torso", "m7_s1_left_leg",
                   "m7_s1_right_leg", "m7_s1_all")
    assert M7_S1_STAGES[0] is DEFAULT_STAGES[0]   # bit-identical base stage
    for s in M7_S1_STAGES[1:]:
        assert s.env_ready is True
        ph = s.phase
        # no walking: every M7 phase names its ZERO target velocity EXPLICITLY
        assert list(ph["target_velocity"]) == [0.0, 0.0, 0.0]
        assert ph["commands"]["mask"] == (1, 0, 0, 0)
        assert ph["commands"]["limb_group"] in LIMB_GROUPS
        assert ph["reward_coeffs"].get("limb_tracking", 0.0) > 0.0
        assert "command_schedule" in ph


def test_default_ladder_untouched():
    assert tuple(s.id for s in DEFAULT_STAGES) == _EXPECTED_DEFAULT_IDS
    for sid, want in (("standing", 0.6), ("controlled", 0.3),
                      ("sustained", 0.0)):
        idx = [i for i, s in enumerate(DEFAULT_STAGES) if s.id == sid][0]
        assert DEFAULT_STAGES[idx].phase["upright_assist"] == want


def test_ladder_selects_and_applies():
    env = LocoEnv()
    cur = Curriculum(env, stages=M7_S1_STAGES)
    cur.select("m7_s1_left_arm")
    assert cur.index == 1 and cur.stage.id == "m7_s1_left_arm"
    assert env._commands["mask"] == (1, 0, 0, 0)
    assert env._commands["limb_group"] == "left_arm"
    # standing re-select: bit-identical zero command state, schedule cleared
    cur.select(0)
    assert env._commands["mask"] == (0, 0, 0, 0)
    assert env._cmd_schedule is None


# ---------------------------------------------------- schedule validation
def _arm_schedule():
    return {"group": "left_arm",
            "targets_rad": {"left_shoulder": (-0.7, -0.35, 0.35, 0.7),
                            "left_elbow": (-0.7, -0.35, 0.35, 0.7)},
            "hold_windows": 60, "transition_windows": 20}


def _all_targets(waist=(0.0,), left_ankle=(0.0,)):
    return {"left_shoulder": (0.0,), "left_elbow": (0.0,),
            "right_shoulder": (0.0,), "right_elbow": (0.0,),
            "head_pitch": (0.0,), "head_yaw": (0.0,), "head_roll": (0.0,),
            "spine_L2": (0.0,), "left_hip": (0.0,), "left_knee": (0.0,),
            "left_ankle": left_ankle, "right_hip": (0.0,),
            "right_knee": (0.0,), "right_ankle": (0.0,),
            "waist_yaw": waist}


@pytest.mark.parametrize("spec, match", [
    ({"group": "nose", "targets_rad": {"left_shoulder": (0.0,)},
      "hold_windows": 60, "transition_windows": 20}, "group"),
    ({"group": "left_arm", "targets_rad": {"left_shoulder": (1.05,),
                                           "left_elbow": (0.0,)}},
     "measured P2 holdable set"),
    (_arm_schedule() | {"targets_rad": {"left_shoulder": (0.5,),
                                        "left_elbow": (0.0,)}},
     "measured P2 holdable set"),
    (_arm_schedule() | {"targets_rad": {"left_shoulder": (5.0,),
                                        "left_elbow": (0.0,)}},
     "outside joint"),
    ({"group": "all", "targets_rad": _all_targets(left_ankle=(0.35,)),
      "hold_windows": 60, "transition_windows": 20}, "ground-pinned"),
    ({"group": "all", "targets_rad": _all_targets(waist=(0.35,)),
      "hold_windows": 60, "transition_windows": 20}, "waist is NEVER commanded"),
    ({"group": "all",
      "targets_rad": {k: v for k, v in _all_targets().items()
                      if k != "head_roll"},
      "hold_windows": 60, "transition_windows": 20}, "missing joint"),
])


def test_schedule_validation_refusals(spec, match):
    env = LocoEnv()
    with pytest.raises(ValueError, match=match):
        env._set_command_schedule(spec)
    assert env._cmd_schedule is None


def test_schedule_waist_never_commanded():
    env = LocoEnv()
    bad = {"group": "all",
           "targets_rad": _all_targets(waist=(0.35,)),
           "hold_windows": 60, "transition_windows": 20}
    with pytest.raises(ValueError, match="waist is NEVER commanded"):
        env._set_command_schedule(bad)
    assert env._cmd_schedule is None


def test_schedule_segment_bounds():
    env = LocoEnv()
    spec = _arm_schedule()
    spec["hold_windows"] = 200
    spec["transition_windows"] = 300     # 500 > max_steps 400
    with pytest.raises(ValueError, match="exceeds"):
        env._set_command_schedule(spec)
    assert env._cmd_schedule is None


# ------------------------------------------------ determinism + boundaries
def test_plan_deterministic_per_seed():
    env1 = LocoEnv()
    env2 = LocoEnv()
    for env in (env1, env2):
        env._set_command_schedule(_arm_schedule())
    env1.reset(seed=20261117)
    env2.reset(seed=20261117)
    for _ in range(90):
        o1 = env1.step(_zero_action(env1))[0][_LIMB]
        o2 = env2.step(_zero_action(env2))[0][_LIMB]
        assert np.array_equal(o1, o2)
    # a different seed rotates the schedule: some plan targets differ
    env3 = LocoEnv()
    env3._set_command_schedule(_arm_schedule())
    env3.reset(seed=99010000)
    env1.reset(seed=20261117)
    env1.step(_zero_action(env1))
    env3.step(_zero_action(env3))
    p1 = [np.asarray(seg[1]) for seg in env1._cmd_plan]
    p3 = [np.asarray(seg[1]) for seg in env3._cmd_plan]
    assert any(not np.array_equal(a, b) for a, b in zip(p1, p3))


def test_hold_boundaries_and_interpolation():
    env = LocoEnv()
    env._set_command_schedule(_arm_schedule())
    env.reset(seed=7)
    sh, el, waist = (env.joint_names.index(n) for n in
                     ("left_shoulder", "left_elbow", "waist_yaw"))
    # The episode's first hold target comes from the seeded candidate
    # rotation — the plan is seed-deterministic, not fixed to +0.7 rad
    # (candidates cycle in listed order with a per-episode phase offset).
    # Assert against the plan's OWN target, not a hard-coded one.
    tgt0 = np.asarray(env._cmd_plan[0][1], dtype=float)

    def limb(o):
        return np.asarray(o[_LIMB], dtype=float)

    # window 0 = the ramp start (neutral)
    assert np.allclose(limb(env._get_obs()), 0.0, atol=1e-12)
    # during the transition the command follows the planned interpolation
    # (float32 obs rounding — the block is stored float32)
    for k in range(19):
        o = env.step(_zero_action(env))[0]
        target = env._schedule_value_at(k + 1)
        assert abs(limb(o)[sh] - target[sh]) < 1e-6
    # window 30 = the flat hold at the plan's first target
    o = env.step(_zero_action(env))[0]
    assert abs(limb(o)[sh] - tgt0[sh]) < 1e-6
    flat = limb(o).copy()
    for _ in range(50):     # holds are FLAT
        o = env.step(_zero_action(env))[0]
        assert np.allclose(limb(o), flat, atol=1e-12)
    # the waist slot is NEVER commanded (stays neutral through the stage)
    assert env._commands["limb_targets"][waist] == 0.0


def test_leg_stage_pins_ankles():
    env = LocoEnv()
    cur = Curriculum(env, stages=M7_S1_STAGES)
    cur.select("m7_s1_left_leg")
    env.reset(seed=3)
    la = env.joint_names.index("left_ankle")
    for _ in range(40):
        v = np.asarray(env.step(_zero_action(env))[0][_LIMB], dtype=float)
        assert v[la] == 0.0            # ground-pinned: never commanded
    knee = env._cmd_schedule["sets"]["left_knee"]
    lo, hi = env.joint_range["left_knee"]
    want = (0.7 - lo) / (hi - lo) * 2.0 - 1.0
    assert len(knee) == 4 and abs(max(knee) - want) < 1e-9


# --------------------------------------------------------------- the reward
def test_limb_tracking_active_group_only():
    m = MeasuredStep(
        position_before=np.zeros(3), position_after=np.zeros(3),
        velocity=np.zeros(3), up_z=1.0, effort_j=0.0, window_s=0.02,
        fell=False,
        commands={"mask": (1, 0, 0, 0),
                  "limb_targets": tuple(0.3 * np.ones(15)),
                  "limb_group": "left_arm"},
        joint_pos_norm=np.zeros(15))
    m.joint_pos_norm[1] = m.joint_pos_norm[2] = 0.1   # err = 0.1 − 0.3 = −0.2
    w = 2.0
    got = limb_tracking_penalty(m, w)
    assert abs(got - (-w * (0.2 * 0.2))) < 1e-12
    # mask off ⇒ structurally zero, whatever the coeff
    m.commands = dict(m.commands, mask=(0, 0, 0, 0))
    assert limb_tracking_penalty(m, w) == 0.0


# -------------------------------------------------------------- launcher
def test_launcher_refuses_provisional_m7_ladder(tmp_path):
    """The launch-time gate: no M7 lineage trains while the ladder's
    limb_tracking coefficient is provisional (the untrained-baseline probe
    must first record the measured w + gate bands and clear the flag)."""
    import argparse
    import scripts.rl_ppo as rl_ppo
    from playground.rl.ppo import PPOError
    args = argparse.Namespace(
        name="m7", steps=512, seed=20261117, rollout_steps=512, ent_coef=None,
        lr=None, stage_budget=None, resume=None, fork_from=None,
        out=str(tmp_path / "m7_run"), operation="fresh", stages="m7_s1")
    with pytest.raises(PPOError) as e:
        rl_ppo.cmd_train(args)
    assert "STILL PROVISIONAL" in str(e.value)


def test_launcher_records_stage_ladder_tag(tmp_path):
    import argparse
    import json
    import scripts.rl_ppo as rl_ppo
    out = tmp_path / "m7_run"
    args = argparse.Namespace(
        name="rec", steps=512, seed=20261117, rollout_steps=512,
        ent_coef=None, lr=None, stage_budget=None, resume=None,
        fork_from=None, out=str(out), operation="fresh", stages=None)
    rl_ppo.cmd_train(args)
    cfgj = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfgj["stage_ladder"] == "default"
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert res["stage_ladder"] == "default"


def test_launcher_refuses_unknown_ladder(tmp_path):
    import argparse
    import scripts.rl_ppo as rl_ppo
    from playground.rl.ppo import PPOError
    args = argparse.Namespace(
        name="x", steps=512, seed=1, rollout_steps=512, ent_coef=None,
        lr=None, stage_budget=None, resume=None, fork_from=None,
        out=str(tmp_path / "x"), operation="fresh", stages="nonsense")
    with pytest.raises(PPOError) as e:
        rl_ppo.cmd_train(args)
    assert "unknown --stages" in str(e.value)


# --------------------------------------------------- reset metrics invariant
def test_reset_calls_metrics_reset_exactly_once(monkeypatch):
    """Regression (reports/M7_1A_LIMB_STAGES_CHANGE.md review): the M7-1a
    limb-schedule insertion in reset() briefly double-called
    `_metrics_reset(seed=seed)` — once beside `_plan_command_episode`, once in
    the pre-existing tail. A second reset re-zeros every accumulator written
    between the two calls, so reset() must call it exactly once per episode."""
    env = LocoEnv()
    calls = {"n": 0}
    orig = env._metrics_reset

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(env, "_metrics_reset", counting)
    env.reset(seed=11)
    assert calls["n"] == 1
    # and the episode starts from a fresh, fully-populated metrics block
    assert env._metrics["steps"] == 0
    assert env._metrics["start_position"] is not None
    assert env._metrics["reward_total"] == 0.0
    assert env._metrics["falls"] == 0


def test_reset_metrics_survive_command_schedule(monkeypatch):
    """The scheduled limb command (set in `_plan_command_episode`) and the
    metrics block must both be intact after reset — a stray second
    `_metrics_reset` would not clear commands, but this pins the ordering
    contract so the schedule and a populated start_position coexist."""
    env = LocoEnv()
    env._set_command_schedule(_arm_schedule())
    env.reset(seed=11)
    assert env._cmd_plan is not None
    assert env._commands["mask"] == (1, 0, 0, 0)
    assert env._metrics["start_position"] is not None
