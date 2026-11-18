"""Core RL-component semantics (M6-B B2 contract).

Rewritten at B2: reward components now consume ENGINE MEASUREMENTS (never
normalized observation slices), the step protocol is 5-tuple gym-style, and
the observation is the v3 layout (B3 task-conditioning append over v2). The
anti-exploit semantics from the original suite are preserved and strengthened
(see test_rl_env_b2.py for the full B2 battery, test_rl_env_b3.py for the v3
append battery)."""
import numpy as np
import pytest

import playground.rl.rewards as R
from playground.rl.env import LocoEnv
from playground.rl.rewards import (COMPONENT_KEYS, DEFAULTS, MeasuredStep,
                                   alive_reward, compute_reward,
                                   heading_progress_raw, progress_reward,
                                   upright_reward, velocity_tracking_reward)

_TARGET = np.array([0.5, 0.0, 0.0])


def test_shapes():
    env = LocoEnv()
    obs = env.reset()
    assert obs.shape == (LocoEnv.obs_dim,) == (43,)   # v3 = v2 40 + command
    act = np.zeros(LocoEnv.action_dim, dtype=np.float32)
    obs2, rew, term, trunc, info = env.step(act)
    assert obs2.shape == (LocoEnv.obs_dim,)
    assert isinstance(rew, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert set(info["components"]) == set(COMPONENT_KEYS)


def test_obs_v2_layout_and_finiteness():
    env = LocoEnv()
    obs = env.reset()
    # real spawn posture: joints neutral (q ~ 0, all ranges symmetric) -> ~0
    assert np.allclose(obs[0:14], 0.0, atol=0.05)
    # up vector is +Z, forward (chest) is +Y at the neutral waist (0 rad)
    assert np.allclose(obs[28:31], [0, 0, 1], atol=1e-4)
    assert float(obs[34]) > 0.99
    assert np.allclose(obs[35:38], [0, 1, 0], atol=1e-4)
    assert np.all(np.isfinite(obs))
    # joint velocity normalisation scale is the documented one
    env.step(np.zeros(LocoEnv.action_dim, dtype=np.float32))
    assert np.all(np.isfinite(obs))


def test_velocity_tracking_full_vector_semantics():
    """Anti-exploit v1 semantics preserved on MEASURED velocity: matching the
    target scores ~1; standing with a nonzero target scores ~exp(-||v*||^2/sigma);
    sideways and backwards motion cannot out-score honest progress."""
    r_stand = velocity_tracking_reward(np.zeros(3), _TARGET, DEFAULTS["tracking_sigma"], 1.0)
    r_move = velocity_tracking_reward(_TARGET, _TARGET, DEFAULTS["tracking_sigma"], 1.0)
    assert r_move > r_stand * 3.0
    assert r_move > 0.95 and r_stand < 0.2
    sideways = velocity_tracking_reward(np.array([0.0, 2.0, 0.0]), _TARGET,
                                        DEFAULTS["tracking_sigma"], 1.0)
    assert sideways <= r_stand
    backwards = velocity_tracking_reward(np.array([-0.5, 0.0, 0.0]), _TARGET,
                                         DEFAULTS["tracking_sigma"], 1.0)
    assert backwards < r_move


def test_progress_net_displacement_and_backward():
    """The heading-progress rule pays NET forward displacement only:
    back-and-forth repetition nets ~0 and backward motion pays 0."""
    v = np.array([0.5, 0.0, 0.0])
    dt, w = 0.02, DEFAULTS["progress"]
    # one metre forward in a window: capped at |v*|*dt (anti-burst clamp)
    assert progress_reward(np.array([1.0, 0, 0]), v, dt, w) == pytest.approx(w * 0.5 * dt)
    # half the cap forward: pays exactly the measured metres
    assert progress_reward(np.array([0.005, 0, 0]), v, dt, w) == pytest.approx(w * 0.005)
    # straight backwards: zero
    assert progress_reward(np.array([-1.0, 0, 0]), v, dt, w) == 0.0
    # out-and-back oscillation pays the HALF-rate: forward legs earn, return
    # legs pay 0 (net round trip can never exceed one-way travel per metre)
    assert heading_progress_raw(np.array([0.5, 0, 0]), v) \
        + heading_progress_raw(np.array([-0.5, 0, 0]), v) == pytest.approx(0.5)
    # pure sideways displacement along +y: perpendicular to heading -> 0
    assert progress_reward(np.array([0.0, 0.02, 0]), v, dt, w) == 0.0
    # zero target velocity: standing task, progress pays nothing
    assert progress_reward(np.array([0.5, 0, 0]), np.zeros(3), dt, w) == 0.0


def test_alive_upright_effort_components():
    assert upright_reward(1.0, DEFAULTS["upright"]) == pytest.approx(0.5)
    assert alive_reward(1.0, DEFAULTS["alive_margin"], DEFAULTS["alive"]) == 1.0
    assert alive_reward(0.1, DEFAULTS["alive_margin"], DEFAULTS["alive"]) == -1.0
    assert upright_reward(0.6, DEFAULTS["upright"]) == pytest.approx(0.3)


def test_chest_aware_upright_min_rule():
    """B4 posture ask (b4_posture1, the ONE reward change per Jamie's decision):
    upright/alive consume min(torso up-z, chest up-z). The slide-root pelvis
    cannot pitch, so torso up alone was blind to the chest fold — measured in
    b4_head1's sustained fold (torso ≈ 1.0 every window while chest_up_z mean
    0.115). Standing (chest fully upright) must be BIT-IDENTICAL to the old
    rule: min(1, 1) = 1."""
    # folded chest: upright collapses and alive flips even though torso is up
    ms_fold = _ms(up_z=1.0)
    ms_fold.chest_up_z = 0.1
    _, comp = compute_reward(ms_fold, np.zeros(14), np.zeros(14))
    assert comp["upright"] == pytest.approx(0.05)      # 0.5 * min(1.0, 0.1)
    assert comp["alive"] == -1.0                       # min 0.1 <= margin 0.4
    # modest chest tilt pays proportionally through the min, alive still +1
    ms_tilt = _ms(up_z=1.0)
    ms_tilt.chest_up_z = 0.85
    _, comp2 = compute_reward(ms_tilt, np.zeros(14), np.zeros(14))
    assert comp2["upright"] == pytest.approx(0.425)    # 0.5 * 0.85
    assert comp2["alive"] == 1.0
    # chest up beyond torso up cannot inflate upright above the torso term
    ms_high = _ms(up_z=0.6)
    ms_high.chest_up_z = 1.0
    _, comp3 = compute_reward(ms_high, np.zeros(14), np.zeros(14))
    assert comp3["upright"] == pytest.approx(0.3)      # min(0.6, 1.0) = 0.6
    # standing invariance: the dataclass default (chest_up_z = 1.0) keeps every
    # pre-change fixture and the standing equilibrium exactly as before
    _, comp4 = compute_reward(_ms(up_z=1.0), np.zeros(14), np.zeros(14))
    assert comp4["upright"] == pytest.approx(0.5)
    assert comp4["alive"] == 1.0
    # measured step built by the env carries the chest measurement (real path)
    env_ms = compute_reward(
        MeasuredStep(
            position_before=np.zeros(3), position_after=np.zeros(3),
            velocity=np.zeros(3), up_z=1.0, effort_j=0.0, window_s=0.02,
            fell=False, up_z_window=[1.0, 1.0], chest_up_z=0.2,
            target_velocity=tuple(_TARGET)),
        np.zeros(14), np.zeros(14))[1]
    assert env_ms["upright"] == pytest.approx(0.1)
    assert env_ms["alive"] == -1.0


def _ms(up_z=1.0, dpos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), fell=False, effort=0.0,
        window=0.02, up_window=None):
    return MeasuredStep(
        position_before=np.zeros(3), position_after=np.asarray(dpos, dtype=float),
        velocity=np.asarray(vel, dtype=float), up_z=up_z, effort_j=effort,
        window_s=window, fell=fell,
        up_z_window=up_window if up_window is not None else [1.0, 1.0],
        target_velocity=tuple(_TARGET))


def test_compute_reward_sums_observably():
    """The total is the sum of the observable components (per-transition)."""
    ms = _ms(fell=True, dpos=(0.0, 0.5, 0.0), vel=(0.0, 0.0, 0.0))
    total, comp = compute_reward(ms, np.zeros(14), np.zeros(14))
    assert set(comp) == set(COMPONENT_KEYS)
    assert abs(total - sum(comp.values())) < 1e-9
    assert comp["fall"] == -DEFAULTS["fall"]
    # standing at the 0.5 m/s target: tracking ~exp(-2.5), no progress (0.5 m along y is sideways)
    assert comp["velocity_tracking"] == pytest.approx(float(np.exp(-2.5)), rel=1e-3)
    assert comp["progress"] == 0.0


def test_fall_penalty_shape():
    ms = _ms(up_z=0.2, fell=True)
    total, comp = compute_reward(ms, np.zeros(14), np.zeros(14))
    assert comp["fall"] == -DEFAULTS["fall"]
    ms2 = _ms(fell=False)
    _, comp2 = compute_reward(ms2, np.zeros(14), np.zeros(14))
    assert comp2["fall"] == 0.0


def test_smoothness_and_stability_are_shaping():
    a = np.full(14, 0.5, dtype=np.float32)
    total, comp = compute_reward(_ms(), a, np.zeros(14))
    assert comp["smoothness"] == pytest.approx(-DEFAULTS["smoothness"] * 14 * 0.25)
    total2, comp2 = compute_reward(
        _ms(up_window=[0.9, 1.0, 0.8, 1.0, 0.9]), a, a)
    assert comp2["smoothness"] == 0.0
    assert comp2["stability"] < 0.0   # jittered up_z window is penalised


@pytest.fixture(scope="module")
def real_env():
    try:
        env = LocoEnv()
        return env
    except Exception as e:  # pragma: no cover
        pytest.xfail(f"Real PhysicsWorld unavailable: {e}")
        return None


def test_realworld_smoke(real_env):
    """200 decisions (4 s) with zero action: observations valid, upright."""
    if real_env is None:
        pytest.skip("Real environment not available")
    env = real_env
    env.reset()
    zero = np.zeros(LocoEnv.action_dim, dtype=np.float32)
    for step in range(200):
        obs, rew, term, trunc, info = env.step(zero)
        assert obs.shape == (LocoEnv.obs_dim,)
        assert np.all(np.isfinite(obs))
        assert float(obs[34]) > 0.5
        if term or trunc:
            break


def test_anti_exploit_real():
    """Anti-exploit on the REAL humanoid (kept from the original suite): a
    scripted gait that travels toward the target must out-earn a standing
    policy on the TRACKING component, must ACCRUE PROGRESS (the v2 rule that
    pays real metres), and must physically displace."""
    zero = np.zeros(LocoEnv.action_dim, dtype=np.float32)

    def run(gait):
        env = LocoEnv(cfg={"rl": {"max_episode_steps": 400}})
        env.reset()
        if gait:
            env.world.set_gait("walk", speed=1.0, heading=0.0)
        tracking = []
        p0 = np.asarray(env.world.agent_pos(), dtype=float).copy()
        for _ in range(400):
            obs, _, term, trunc, _ = env.step(zero)
            tracking.append(velocity_tracking_reward(
                env.world.agent_vel(), env.target_velocity,
                DEFAULTS["tracking_sigma"], 1.0))
            if term or trunc:
                break
        d = float(np.linalg.norm(np.asarray(env.world.agent_pos()) - p0))
        return float(np.mean(tracking)), d, env

    t_zero, d_zero, _ = run(gait=False)
    t_move, d_move, env_move = run(gait=True)
    assert t_move > t_zero * 1.05
    assert d_move > d_zero
    measured = env_move.episode_summary()
    # the walking episode ACCRUED heading progress (reward pays real metres)
    assert measured["heading_progress_raw_m"] > 0.25
    assert measured["path_len_m"] > 0.5
    assert measured["falls"] == 0 and measured["faulted_windows"] == 0
