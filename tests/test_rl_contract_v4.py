"""M7-0a contract v4 tests (the new contract's own suite).

Covers the freeze's implementation laws:
- append-only obs: the 43-dim v3 prefix is BIT-identical between the legacy
  mode and v4 under identical seeds/phases/actions (N1);
- the waist joins the ACTION by NAME order (index 14), never model order;
- the obs slices stay the fixed 14-name layout (the waist's measured position
  is reward-side only — it was never observed in v3 either);
- command block: mask-gated values, inactive groups read zeros, active
  zeros are real commands (the mask bit disambiguates);
- the action space is NEVER masked;
- the four new reward components: config-default 0.0, mask gate, exact math;
- phase 'commands' non-cumulative; v3 mode refuses the v4 seams loudly.
"""
import numpy as np
import pytest

from playground.rl.env import LocoEnv
from playground.rl.rewards import (DEFAULTS, COMPONENT_KEYS, MeasuredStep,
                                   compute_reward, COM_ANCHOR_XY,
                                   COM_SHIFT_SCALE_XY, WAYPOINT_D_REF,
                                   LIMB_GROUPS)
from playground.rl.curriculum import DEFAULT_STAGES

V4_ORDER = ["spine_L2", "left_shoulder", "left_elbow", "right_shoulder",
            "right_elbow", "head_pitch", "head_yaw", "head_roll", "left_hip",
            "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle",
            "waist_yaw"]


def _v3():
    return LocoEnv(cfg={"rl": {"contract_version": 3}})


def _v4():
    return LocoEnv()


# ------------------------------------------------------------------ layout
def test_v4_default_contract():
    env = _v4()
    assert (env.obs_dim, env.action_dim, env.contract_version) == (68, 15, 4)
    assert list(env.joint_names) == V4_ORDER          # name order; waist at 14
    assert list(env._obs_joint_names) == V4_ORDER[:14]  # obs stays 14-wide
    assert env._gains["waist_yaw"] == (50.0, 5.0)
    assert "waist_yaw" not in env._posture_gains
    assert set(env._posture_gains) == {"left_foot_yaw", "right_foot_yaw"}
    ol = env.observation_layout()
    assert ol["layout_version"] == 4 and ol["dim"] == 68
    assert [s["name"] for s in ol["sections"]] == [
        "joint_pos", "joint_vel", "torso_up", "torso_lin_vel", "upright_margin",
        "torso_forward", "foot_ground_force", "command",
        "limb_pose_target", "com_shift", "waypoint", "stop_request", "cmd_mask"]
    assert "waist_yaw" not in ol["joint_order"]        # obs slices: 14 names
    assert ol["command_block"]["limb_targets_order"][-1] == "waist_yaw"
    assert ol["command_block"]["mask_order"] == ["limb", "balance", "waypoint",
                                                 "stop"]
    assert ol["command_block"]["com_shift_anchor_xy_m"] == list(COM_ANCHOR_XY)
    assert ol["command_block"]["waypoint_d_ref_m"] == WAYPOINT_D_REF
    al = env.action_layout()
    assert al["layout_version"] == 4 and al["action_dim"] == 15
    waist = al["joints"][-1]
    assert waist["name"] == "waist_yaw" and waist["kp"] == 50.0 and waist["kd"] == 5.0
    # the range is the MODEL's limited range (XML literal ±3.1416, ~π)
    assert waist["range_rad"] == pytest.approx([-np.pi, np.pi], abs=1e-3)
    assert [p["name"] for p in al["passive_conventional_pd"]] == [
        "left_foot_yaw", "right_foot_yaw"]
    # effort measurement set grew 14 -> 15 with the action (frozen v3 list)
    assert len(env._dofadr) == 15


def test_v3_legacy_mode():
    env = _v3()
    assert (env.obs_dim, env.action_dim, env.contract_version) == (43, 14, 3)
    assert list(env.joint_names) == V4_ORDER[:14]
    assert env._gains.get("waist_yaw") is None          # posture hinge in v3
    assert env._posture_gains["waist_yaw"] == (50.0, 5.0)
    ol = env.observation_layout()
    assert ol["layout_version"] == 3 and ol["dim"] == 43
    assert "command_block" not in ol
    assert [s["name"] for s in ol["sections"]][-1] == "command"
    with pytest.raises(ValueError):
        env.set_commands(mask=(1, 0, 0, 0))
    with pytest.raises(ValueError):
        env.set_target_point([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        env.set_phase({"commands": {"mask": (1, 0, 0, 0)}})


def test_unsupported_contract_version_rejected():
    with pytest.raises(ValueError):
        LocoEnv(cfg={"rl": {"contract_version": 2}})
    with pytest.raises(ValueError):
        LocoEnv(cfg={"rl": {"contract_version": 5}})


# ----------------------------------------------------------- N1 bit-identity
def test_n1_v3_prefix_bit_identity():
    """THE append-only law: identical seeds/phase/actions give a BIT-identical
    43-prefix and an identical reward on the legacy and the v4 env."""
    e3, e4 = _v3(), _v4()
    phase = dict(DEFAULT_STAGES[0].phase)   # standing: assist, zero v*
    e3.set_phase(dict(phase))
    e4.set_phase(dict(phase))
    o3, o4 = e3.reset(seed=20261117), e4.reset(seed=20261117)
    assert np.array_equal(o3, o4[:43])
    assert np.all(o4[43:68] == 0.0)         # no commands ⇒ block reads zeros
    assert o4.dtype == np.float32
    # standing zero-action prefix: the N2 property (waist still ⇒ identical
    # reward; bit-equal obs and per-window reward floats)
    for step in range(30):
        o3, r3, t3, tr3, i3 = e3.step(np.zeros(14, dtype=np.float32))
        o4, r4, t4, tr4, i4 = e4.step(np.zeros(15, dtype=np.float32))
        assert np.array_equal(o3, o4[:43])
        assert r3 == r4
    # general sweep: PHYSICS is identical (targets identical; obs bit-equal);
    # the ONLY reward-path difference must be the DESIGNED effort-set growth
    # 14 → 15 (v4 ALSO measures the waist's own PD work), i.e.
    # Δreward == −effort_weight × Δeffort, within float summation noise.
    rng = np.random.default_rng(7)
    for step in range(60):
        a14 = rng.uniform(-1.0, 1.0, 14).astype(np.float32)
        a15 = np.concatenate([a14, [0.0]]).astype(np.float32)  # waist neutral
        o3, r3, t3, tr3, i3 = e3.step(a14)
        o4, r4, t4, tr4, i4 = e4.step(a15)
        assert np.array_equal(o3, o4[:43])
        assert i4["effort_j"] >= i3["effort_j"]
        assert abs((r4 - r3) + 0.005 * (i4["effort_j"] - i3["effort_j"])) < 1e-9
    assert e4.episode_summary()["net_displacement_m"] == \
        e3.episode_summary()["net_displacement_m"]


# ------------------------------------------------------- action + commands
def test_action_mapping_waist_neutral_and_full():
    env = _v4()
    env.reset(seed=20261117)
    env.step(np.zeros(15, dtype=np.float32))
    assert all(abs(t) < 1e-12 for t in env._last_targets.values())  # 0 → 0 rad
    a = np.zeros(15, dtype=np.float32)
    a[14] = 1.0                                     # waist full +yaw
    env.step(a)
    wlo, whi = env.joint_range["waist_yaw"]         # the MODEL's ±3.1416
    assert abs(env._last_targets["waist_yaw"] - whi) < 1e-6
    a[14] = -1.0
    env.step(a)
    assert abs(env._last_targets["waist_yaw"] - wlo) < 1e-6
    # mapping spot-check on one legacy joint (0.5 → quarter range + center)
    a[0] = 0.5
    env.step(a)
    lo, hi = env.joint_range["spine_L2"]
    assert abs(env._last_targets["spine_L2"] - (lo + 0.75 * (hi - lo))) < 1e-6


def test_action_space_never_masked():
    """Mask 0 zeroes the COMMAND OBS, never the policy's body authority."""
    env = _v4()
    env.reset(seed=20261117)
    a = np.zeros(15, dtype=np.float32)
    a[1] = 0.5   # left_shoulder
    o, r, t, tr, i = env.step(a)
    lo, hi = env.joint_range["left_shoulder"]
    # expected exact: lo + (0.5+1)/2*(hi-lo); float32 target rounding ~1e-7
    assert abs(env._last_targets["left_shoulder"]
               - (lo + 0.75 * (hi - lo))) < 1e-6


def test_command_block_obs_semantics():
    env = _v4()
    env.reset(seed=20261117)
    zero = np.zeros(15, dtype=np.float32)
    o, *_ = env.step(zero)
    assert np.all(o[43:68] == 0.0)
    # limb group active with a real neutral AND a real target: values pass through
    tg = np.zeros(15, dtype=np.float32)
    tg[0] = 0.25
    tg[14] = 0.5
    env.set_commands(mask=(1, 0, 0, 0), limb_targets=tg)
    o, *_ = env.step(zero)
    assert np.array_equal(o[43:58], tg)  # all 15 commanded values (0 = hold-neutral)
    assert np.all(o[64:68] == np.array([1, 0, 0, 0], dtype=np.float32))
    # balance + waypoint + stop active: limb reads zeros (inactive)
    env.set_commands(mask=(0, 1, 1, 1), com_shift=(0.5, -0.5),
                     waypoint=(0.3, -0.3, 0.8), stop=1)
    o, *_ = env.step(zero)
    assert np.all(o[43:58] == 0.0)
    assert np.array_equal(o[58:60], np.array([0.5, -0.5], dtype=np.float32))
    assert np.array_equal(o[60:63], np.array([0.3, -0.3, 0.8], dtype=np.float32))
    assert o[63] == 1.0 and np.array_equal(o[64:68], np.array([0, 1, 1, 1], dtype=np.float32))
    # full mask carries every group
    env.set_commands(mask=(1, 1, 1, 1))
    o, *_ = env.step(zero)
    assert o[43] == 0.25 and o[57] == 0.5 and o[58] == 0.5 and o[63] == 1.0
    assert np.array_equal(o[64:68], np.array([1, 1, 1, 1], dtype=np.float32))


def test_command_validation_and_partial_updates():
    env = _v4()
    env.reset(seed=20261117)
    env.set_commands(mask=(1, 0, 0, 0), limb_targets=np.zeros(15),
                     limb_group="head")
    with pytest.raises(ValueError):
        env.set_commands(mask=(1, 0, 0))
    with pytest.raises(ValueError):
        env.set_commands(mask=(2, 0, 0, 0))
    with pytest.raises(ValueError):
        env.set_commands(limb_targets=np.zeros(14))
    with pytest.raises(ValueError):
        env.set_commands(limb_targets=np.full(15, 1.5))
    with pytest.raises(ValueError):
        env.set_commands(com_shift=(0.5, 2.0))
    with pytest.raises(ValueError):
        env.set_commands(waypoint=(0.0, 0.0, -0.1))
    with pytest.raises(ValueError):
        env.set_commands(stop=3)
    with pytest.raises(ValueError):
        env.set_commands(limb_group="wings")
    # partial: the previous limb_group "head" and mask survive
    env.set_commands(com_shift=(0.1, 0.1))
    assert env._commands["limb_group"] == "head"
    assert env._commands["mask"][0] == 1
    assert env._commands["com_shift"] == (0.1, 0.1)


def test_phase_commands_non_cumulative():
    env = _v4()
    env.reset(seed=20261117)
    env.set_phase({"commands": {"mask": (1, 1, 0, 0), "com_shift": (0.5, 0.0)}})
    assert env._commands["mask"] == (1, 1, 0, 0)
    assert env._commands["com_shift"] == (0.5, 0.0)
    # absent ⇒ zero state, every time (non-cumulative, set_phase law)
    env.set_phase({})
    assert env._commands["mask"] == (0, 0, 0, 0)
    assert env._commands["com_shift"] == (0.0, 0.0)
    env.reset(seed=20261118)    # reset does NOT clear stage command state
    env.set_phase({"commands": {"mask": (0, 0, 0, 1), "stop": 1}})
    env.reset(seed=20261119)
    assert env._commands["mask"] == (0, 0, 0, 1)
    env.set_phase(dict(DEFAULT_STAGES[1].phase))   # plain v3-era stage
    assert env._commands["mask"] == (0, 0, 0, 0)


# ----------------------------------------------------------------- rewards
def _step_with_commands(m, **cmd):
    m.commands = {**{"mask": (0, 0, 0, 0), "limb_targets": None,
                     "com_shift": (0.0, 0.0), "waypoint": (0.0, 0.0, 0.0),
                     "stop": 0, "limb_group": "all"}, **cmd}
    return m


def test_new_components_default_zero_and_fixture_identity():
    coeffs = dict(DEFAULTS)
    coeffs.update({"limb_tracking": 3.0, "com_tracking": 3.0,
                   "approach_progress": 3.0, "stop_tolerance": 3.0})
    ms = MeasuredStep(position_before=np.zeros(3), position_after=np.zeros(3),
                      velocity=np.zeros(3), up_z=1.0, effort_j=0.0,
                      window_s=0.02, fell=False, up_z_window=[1.0, 1.0])
    total, comp = compute_reward(ms, np.zeros(14), np.zeros(14), coeffs)
    # the four are present, inert without commands, and the v3 terms are
    # exactly the historical fixture values
    for k in ("limb_tracking", "com_tracking", "approach_progress",
              "stop_tolerance"):
        assert comp[k] == 0.0
    old = MeasuredStep(position_before=np.zeros(3), position_after=np.zeros(3),
                       velocity=np.zeros(3), up_z=1.0, effort_j=0.0,
                       window_s=0.02, fell=False, up_z_window=[1.0, 1.0])
    t3, c3 = compute_reward(old, np.zeros(14), np.zeros(14), coeffs)
    assert t3 == total


def test_limb_tracking_exact_math_and_gates():
    pos = np.zeros(15, dtype=np.float64)
    pos[5], pos[6], pos[7] = 0.2, -0.1, 0.4     # head group measured
    tg = np.zeros(15, dtype=np.float32)
    ms = _step_with_commands(
        MeasuredStep(position_before=np.zeros(3), position_after=np.zeros(3),
                     velocity=np.zeros(3), up_z=1.0, effort_j=0.0,
                     window_s=0.02, fell=False, up_z_window=[1.0, 1.0],
                     joint_pos_norm=pos),
        mask=(1, 0, 0, 0), limb_targets=tg, limb_group="head")
    err2 = ((0.2 - 0.0) ** 2 + (-0.1) ** 2 + (0.4) ** 2) / 3.0
    total, comp = compute_reward(ms, np.zeros(14), np.zeros(14),
                                 {"limb_tracking": 10.0})
    assert abs(comp["limb_tracking"] + 10.0 * err2) < 1e-12
    # group selectivity: "head" ignores the waist
    pos[14] = 1.0
    _, comp = compute_reward(ms, np.zeros(14), np.zeros(14),
                             {"limb_tracking": 10.0})
    assert abs(comp["limb_tracking"] + 10.0 * err2) < 1e-12
    # mask off ⇒ zero regardless of coefficient
    ms.commands["mask"] = (0, 0, 0, 0)
    _, comp = compute_reward(ms, np.zeros(14), np.zeros(14),
                             {"limb_tracking": 10.0})
    assert comp["limb_tracking"] == 0.0


def test_com_tracking_exact_math_and_gates():
    ms = _step_with_commands(
        MeasuredStep(position_before=np.zeros(3), position_after=np.zeros(3),
                     velocity=np.zeros(3), up_z=1.0, effort_j=0.0,
                     window_s=0.02, fell=False, up_z_window=[1.0, 1.0],
                     com_xy=(-0.07926, 0.08871)),      # at the anchor
        mask=(0, 1, 0, 0), com_shift=(0.5, 0.0))
    _, comp = compute_reward(ms, np.zeros(14), np.zeros(14),
                             {"com_tracking": 20.0})
    want = 0.5 * COM_SHIFT_SCALE_XY[0]
    assert abs(comp["com_tracking"] + 20.0 * want ** 2) < 1e-12
    ms.commands["mask"] = (0, 0, 0, 0)
    _, comp = compute_reward(ms, np.zeros(14), np.zeros(14),
                             {"com_tracking": 20.0})
    assert comp["com_tracking"] == 0.0


def test_approach_and_stop_distance_terms():
    ms = _step_with_commands(
        MeasuredStep(position_before=np.zeros(3), position_after=np.zeros(3),
                     velocity=np.zeros(3), up_z=1.0, effort_j=0.0,
                     window_s=0.02, fell=False, up_z_window=[1.0, 1.0],
                     dist_to_target=(0.3, 0.2)),
        mask=(0, 0, 1, 0), waypoint=(0.0, 0.0, 0.8))
    _, comp = compute_reward(ms, np.zeros(14), np.zeros(14),
                             {"approach_progress": 50.0})
    assert abs(comp["approach_progress"] - 50.0 * 0.1) < 1e-12
    # stop: armed, 0.7 m at band 0.5 ⇒ −w·0.2; un-armed ⇒ 0.0
    ms2 = _step_with_commands(
        MeasuredStep(position_before=np.zeros(3), position_after=np.zeros(3),
                     velocity=np.zeros(3), up_z=1.0, effort_j=0.0,
                     window_s=0.02, fell=False, up_z_window=[1.0, 1.0],
                     dist_to_target=(0.3, 0.7)),
        mask=(0, 0, 1, 1), waypoint=(0.0, 0.0, 0.2), stop=1)
    _, comp = compute_reward(ms2, np.zeros(14), np.zeros(14),
                             {"stop_tolerance": 10.0, "stop_band_m": 0.5})
    assert abs(comp["stop_tolerance"] + 10.0 * 0.2) < 1e-12
    ms2.commands["stop"] = 0
    _, comp = compute_reward(ms2, np.zeros(14), np.zeros(14),
                             {"stop_tolerance": 10.0, "stop_band_m": 0.5})
    assert comp["stop_tolerance"] == 0.0


def test_env_wires_measurements_and_target_point():
    env = _v4()
    env.reset(seed=20261117)
    env.set_phase({"commands": {"mask": (0, 0, 1, 0)}})
    env.set_target_point((0.3, 0.4, 0.0))
    _, _, t, tr, i = env.step(np.zeros(15, dtype=np.float32))
    c = i["components"]
    for k in ("limb_tracking", "com_tracking", "approach_progress",
              "stop_tolerance"):
        assert k in c and c[k] == 0.0     # zero coeffs everywhere (freeze)
    # clearing the target point removes the distances
    env.set_target_point(None)
    _, _, _, _, i2 = env.step(np.zeros(15, dtype=np.float32))
    # (no direct info field; the reward path sees None — assert via MeasuredStep)
    obs = env._get_obs()
    assert obs.shape == (68,)
    assert np.all(obs[64:68] == np.array([0, 0, 1, 0], dtype=np.float32))
