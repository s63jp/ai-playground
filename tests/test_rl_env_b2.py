"""M6-B stage B2 — the consolidated RL environment gate battery.

Covers the B2 acceptance areas:
- observation contract (v2 layout, finiteness, determinism, normalisation)
- action contract (validation, joint mapping/order, clipping accounting)
- the ONE reward pipeline (components observable; anti-exploit on real physics)
- termination vs truncation semantics; safety-fault propagation (B1 intact)
- reset/seed reproducibility and episode isolation (no state leakage)
- curriculum transitions (deterministic, explicit, non-cumulative)
- episode metrics/diagnostics
- scripted-locomotion compatibility inside the env (gait benchmark horizon)

Physical probes that only make sense as REPORTS (not in-suite assertions)
live in ai-playground-audit/probe_b2_*.py; their numbers feed
reports/m6b/B2_ENVIRONMENT.md.
"""
import json
import math

import numpy as np
import pytest

from playground.config import load_config
from playground.rl.curriculum import DEFAULT_STAGES, Curriculum
from playground.rl.env import LocoEnv
from playground.rl.rewards import COMPONENT_KEYS, DEFAULTS as R_DEFAULTS
from playground.world.physics import ActionError
from playground.world.physics import PhysicsWorld


def env_cfg(**over):
    return load_config(overrides={"agent_body": {"type": "humanoid"}, **over})


def _zero(env):
    return np.zeros(env.action_dim, dtype=np.float32)


# ---------------------------------------------------------------- observation
def test_obs_contract_dimensions_and_finiteness():
    """v4 (M7-0a) = the 43-dim v3 slice APPEND-only + the 25-dim command
    block; every v2/v3 section keeps its meaning (N1 bit-identity test in
    test_rl_contract_v4.py)."""
    env = LocoEnv()
    obs = env.reset()
    assert obs.shape == (68,)
    zero = _zero(env)
    for _ in range(25):
        obs, r, term, trunc, info = env.step(zero)
        assert obs.shape == (68,)
        assert np.all(np.isfinite(obs)), "observation went non-finite without fault"
        assert not term
    layout = env.observation_layout()
    assert layout["layout_version"] == 4 and layout["dim"] == 68
    covered = sum(hi - lo for _, lo, hi in
                  [(s["name"], s["slice"][0], s["slice"][1]) for s in layout["sections"]])
    assert covered == 68
    # APPEND-only guarantee: the first 7 v2 sections keep their exact slices
    assert [(s["name"], s["slice"]) for s in layout["sections"][:7]] == [
        ("joint_pos", [0, 14]), ("joint_vel", [14, 28]), ("torso_up", [28, 31]),
        ("torso_lin_vel", [31, 34]), ("upright_margin", [34, 35]),
        ("torso_forward", [35, 38]), ("foot_ground_force", [38, 40])]
    # the v3 append is the command section
    assert layout["sections"][7]["name"] == "command" \
        and layout["sections"][7]["slice"] == [40, 43]
    assert layout["command_frame"] == "world"
    assert layout["joint_order"] == ["spine_L2", "left_shoulder", "left_elbow",
                                     "right_shoulder", "right_elbow", "head_pitch",
                                     "head_yaw", "head_roll", "left_hip", "left_knee",
                                     "left_ankle", "right_hip", "right_knee",
                                     "right_ankle"]


def test_obs_joint_order_matches_action_order():
    env = LocoEnv()
    layout_a = env.action_layout()
    layout_o = env.observation_layout()
    names_a = [j["name"] for j in layout_a["joints"]]
    # M7-0a: the ACTION list grew to 15 (waist appended by name order); the
    # OBS joint order stays the fixed 14 (= the action list's prefix, which
    # is exactly the v3 order). The v3 legacy mode keeps full equality.
    assert layout_o["joint_order"] == names_a[:14]
    assert len(names_a) == 15 and names_a[-1] == "waist_yaw"
    assert "waist_yaw" not in layout_o["joint_order"]
    legacy = LocoEnv(cfg={"rl": {"contract_version": 3}})
    assert legacy.observation_layout()["joint_order"] ==         list(legacy.action_layout()["joints"][i]["name"] for i in range(14))


def test_obs_normalisation_values():
    """Documented normalisation: neutral joints -> ~0 (symmetric ranges),
    up = +Z unit, contacts in [0, ~1.2] (force/1000)."""
    env = LocoEnv()
    obs = env.reset()
    assert np.allclose(obs[0:14], 0.0, atol=0.05)
    assert np.allclose(obs[28:31], [0, 0, 1], atol=1e-4)
    assert np.allclose(obs[35:38], [0, 1, 0], atol=1e-4)  # chest +Y at waist 0
    assert obs[38] >= 0.0 and obs[39] >= 0.0
    # the commanded target is visible (v3): config default v* = [0.5, 0, 0],
    # world frame, / 5.0 root_vel_scale
    assert np.allclose(obs[40:43], [0.1, 0.0, 0.0], atol=1e-6)
    # stand briefly: feet support ~ body weight -> contact forces become large
    zero = _zero(env)
    for _ in range(50):
        obs, *_ = env.step(zero)
    assert obs[38] > 0.05 and obs[39] > 0.05, \
        f"expected real support forces in obs contacts, got {obs[38]}, {obs[39]}"


def test_obs_deterministic_same_construction():
    a = LocoEnv()
    b = LocoEnv()
    oa, ob = a.reset(), b.reset()
    assert np.array_equal(oa, ob)
    z = _zero(a)
    for _ in range(30):
        oa2, *_ = a.step(z)
        ob2, *_ = b.step(z)
        assert np.array_equal(oa2, ob2)


# ---------------------------------------------------------------- action
def test_action_validation_rejects():
    env = LocoEnv()
    env.reset()
    for bad in (float("nan"), float("inf"), float("-inf")):
        a = _zero(env)
        a[7] = bad
        with pytest.raises(ActionError) as e:
            env.step(a)
        assert e.value.code == "non_finite_action"
    for shape in ((10,), (16,), (15, 1)):   # (15,) is the VALID v4 shape now
        with pytest.raises(ActionError) as e:
            env.step(np.zeros(shape, dtype=np.float32))
        assert e.value.code == "invalid_action_shape"
    cs = env.action_clipping()
    assert cs["rejected_calls"] == 6 and cs["calls"] == 0
    # physics was never touched by the rejected calls
    assert env.world.physics_warnings() == {}
    assert not np.isnan(env.world.d.qpos).any()


def test_action_mapping_and_joint_ranges():
    """+1 maps the joint to its range HIGH, -1 to its LOW (documented mapping)."""
    env = LocoEnv()
    env.reset()
    a = np.ones(env.action_dim, dtype=np.float32)
    env.step(a)
    for name in env.joint_names:
        low, high = env.joint_range[name]
        assert env.world._pd[name]["target"] == pytest.approx(high, abs=1e-6)
    env.step(-np.ones(env.action_dim, dtype=np.float32))
    for name in env.joint_names:
        low, high = env.joint_range[name]
        assert env.world._pd[name]["target"] == pytest.approx(low, abs=1e-6)


def test_action_passive_posture_hinges_conventional_pd():
    """The passive hinges are commanded by the env (conventional PD), NOT the
    policy's mapping, and their source tag separates them from neural_policy.
    M7-0a: v4 has TWO passive hinges (left/right foot_yaw); the waist JOINED
    the policy at the same posture gains (50, 5) — the v3 legacy mode keeps
    all three passive hinges (see test_rl_contract_v4)."""
    env = LocoEnv()
    env.reset()
    env.step(_zero(env))
    for name, (kp, kd) in env._posture_gains.items():
        assert name not in [j["name"] for j in env.action_layout()["joints"]]
        assert env.world._pd[name]["source"] == "env_posture"
        assert env.world._pd[name]["target"] == 0.0
    for name in env.joint_names:
        assert env.world._pd[name]["source"] == "neural_policy"


def test_policy_never_touched_passive_joints():
    """A policy at its extremes leaves the posture hinges' PD registered by the
    env (they are conventional assistance, outside the action contract)."""
    env = LocoEnv()
    env.reset()
    env.step(np.full(env.action_dim, 0.9, dtype=np.float32))
    for name in env._posture_joints:
        assert env.world._pd[name]["source"] == "env_posture"
        assert env.world._pd[name]["kp"] == env._posture_gains[name][0]


# ---------------------------------------------------------------- reward
def test_reward_components_observable_per_step():
    env = LocoEnv()
    env.reset()
    for _ in range(20):   # let the spawn settle (initial z transient)
        obs, r, term, trunc, info = env.step(_zero(env))
    comp = info["components"]
    assert set(comp) == set(COMPONENT_KEYS)
    assert abs(float(r) - sum(comp.values())) < 1e-9
    # standing at the default target: tracking ~ exp(-2.5), no progress, no fall
    assert comp["velocity_tracking"] == pytest.approx(float(np.exp(-2.5)), rel=2e-2)
    # settled standing has only settling micro-drift along the heading (~1e-20
    # m per window measured) — the progress term pays nothing there
    assert comp["progress"] == pytest.approx(0.0, abs=1e-9)
    assert comp["alive"] == 1.0 and comp["upright"] == pytest.approx(0.5, abs=1e-6)
    assert comp["fall"] == 0.0


def test_reward_unknown_coeff_rejected():
    env = LocoEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.set_phase({"reward_coeffs": {"teleport": 1.0}})


def test_reward_coeffs_phase_override_non_cumulative():
    """set_phase rewrites the effective coeff set from defaults; applying a
    second phase WITHOUT the key restores the configuration defaults."""
    env = LocoEnv()
    env.reset()
    env.set_phase({"reward_coeffs": {"upright": 2.0}})
    obs, r, term, trunc, info = env.step(_zero(env))
    assert info["components"]["upright"] == pytest.approx(2.0, abs=1e-6)
    env.set_phase({})  # explicit no-op phase: full default set restored
    obs, r, term, trunc, info = env.step(_zero(env))
    assert info["components"]["upright"] == pytest.approx(0.5, abs=1e-6)


def test_reward_farming_stop_and_go_real():
    """The progress term pays DISPLACEMENT, not velocity sampling: a stop-and-go
    profile cannot beat a sustained walk that travels further, and a velocity-
    jiggle policy that travels nowhere accrues zero progress."""
    def run(profile):
        assert profile in ("sustained", "bursty", "jiggle")
        env = LocoEnv(cfg={"rl": {"max_episode_steps": 400}})
        env.reset()
        zero = _zero(env)
        mode_now = None
        for k in range(400):
            if profile == "jiggle":
                a = np.zeros(env.action_dim, dtype=np.float32)
                # chatter at the encoding extremes: velocity noise, no net travel
                a[:2] = 1.0 if k % 2 == 0 else -1.0
                env.step(a)
                continue
            mode = ("walk" if profile == "sustained"
                    else ("walk" if (k // 100) % 2 == 0 else "stop"))
            # reconfigure only on mode changes: every set_gait call resets the
            # controller's traction ramp clock, which would handicap BOTH arms
            if mode != mode_now:
                if mode == "walk":
                    env.world.set_gait("walk", speed=1.0, heading=0.0)
                else:
                    env.world.set_gait("stop")
                mode_now = mode
            env.step(zero)
        return env.episode_summary()

    s_sustain = run("sustained")
    s_burst = run("bursty")
    s_jiggle = run("jiggle")
    # jiggle: velocity noise with ~no travel -> progress pays ~nothing
    assert s_jiggle["heading_progress_rewarded_m"] < 0.05
    assert s_jiggle["path_len_m"] < 0.3
    # sustained walk: real metres along the heading
    assert s_sustain["heading_progress_raw_m"] > 0.4
    # stop-and-go travels less and earns less progress (no duty-cycle bonus)
    assert s_burst["heading_progress_rewarded_m"] <= s_sustain["heading_progress_rewarded_m"]
    assert s_sustain["net_displacement_m"] > s_burst["net_displacement_m"]


def test_reward_progress_backward_real():
    """A scripted round trip (walk +heading for 2 s, walk -heading for 6 s)
    travels both directions, but the heading-progress rule pays the HALF
    rate: forward legs earn, return legs pay 0 — repeated/return movement
    can never out-earn straight travel, and the velocity kernel penalises
    the reversed legs near-fully."""
    env = LocoEnv(cfg={"rl": {"max_episode_steps": 400}})
    env.reset()
    zero = _zero(env)
    env.world.set_gait("walk", speed=1.0, heading=0.0)
    for _ in range(100):
        env.step(zero)
    mid = np.asarray(env.world.agent_pos(), dtype=float).copy()
    env.world.set_gait("walk", speed=1.0, heading=180.0)
    for _ in range(300):
        env.step(zero)
    end = np.asarray(env.world.agent_pos(), dtype=float)
    s = env.episode_summary()
    # it physically travelled both directions...
    assert s["path_len_m"] > 0.5, s
    # ...yet the REWARDED heading progress stayed at the half rate or below
    assert s["heading_progress_rewarded_m"] < 0.6 * s["path_len_m"], s
    assert s["heading_progress_rewarded_m"] > 0.0, "forward legs must earn"


def test_episode_metrics_and_effort():
    env = LocoEnv(cfg={"rl": {"max_episode_steps": 200}})
    env.reset()
    zero = _zero(env)
    effs = 0.0
    for _ in range(200):
        obs, r, term, trunc, info = env.step(zero)
        effs += info["effort_j"]
        if term or trunc:
            break
    s = env.episode_summary()
    assert s["steps"] == 200 and s["truncated_at"] == 200
    assert s["sim_time_s"] == pytest.approx(200 * 0.02, rel=1e-3)
    assert s["effort_total_j"] == pytest.approx(effs, rel=1e-3)
    assert s["falls"] == 0 and s["faulted_windows"] == 0
    assert s["reward_total"] == pytest.approx(sum(s["components"].values()))
    assert s["upright_fraction"] == pytest.approx(1.0, abs=1e-6)
    # standing effort is tiny next to the measured walking envelope (165 W)
    assert s["effort_mean_w"] < 10.0


# ---------------------------------------------------------------- termination
def test_truncation_semantics():
    env = LocoEnv(cfg={"rl": {"max_episode_steps": 5}})
    env.reset()
    zero = _zero(env)
    for k in range(4):
        obs, r, term, trunc, info = env.step(zero)
        assert not term and not trunc
    obs, r, term, trunc, info = env.step(zero)
    assert trunc and not term
    assert info["components"]  # a truncated window still reports components


def test_faulted_transition_is_terminated_not_success():
    """B1 semantics on the v2 protocol: a physically suppressed window is
    terminated + zero-reward and is identifiable for exclusion (info)."""
    cfg = env_cfg(**{"world": {"safety": {"enforce_torque_limits": False,
                                          "clamp_target": False,
                                          "kp_ceiling": 1e9, "kd_ceiling": 1e9}}})
    env = LocoEnv(cfg)
    env.reset()
    env.world._pd["left_foot_yaw"] = {"target": 1e30, "kp": 1e6, "kd": 0.0,
                                      "source": "rogue"}
    obs, r, term, trunc, info = env.step(_zero(env))
    assert term is True and trunc is False
    assert r == 0.0
    assert info.get("faulted") is True
    assert "mjWARN_BADQACC" in info["physics_warnings"]
    assert env.faulted()
    s = env.episode_summary()
    assert s["faulted_windows"] == 1 and s["steps"] == 1
    assert all(v == 0.0 for v in s["components"].values())


def test_fall_termination_reports_components():
    """Drive the collapse through the POLICY's own honest mapping (knees to
    full heel-back flexion at -1, hips forward) so the real PD pathway folds
    the figure; the debounced agent_fallen truth then terminates the episode
    and the fall penalty lands in the components, separated from truncation."""
    env = LocoEnv(cfg={"rl": {"max_episode_steps": 4000}})
    env.reset()
    fold = np.zeros(env.action_dim, dtype=np.float32)
    fold[8] = 0.8    # left_hip  -> forward
    fold[11] = 0.8   # right_hip -> forward
    fold[9] = -1.0   # left_knee -> full heel-back flexion
    fold[12] = -1.0  # right_knee -> full heel-back flexion
    saw_fall = None
    for _ in range(int(6.0 / 0.02)):
        obs, r, term, trunc, info = env.step(fold)
        if term:
            saw_fall = info
            break
    assert saw_fall is not None, "the collapse must terminate the episode"
    assert saw_fall.get("fallen") is True
    assert saw_fall["components"]["fall"] == -R_DEFAULTS["fall"]
    s = env.episode_summary()
    assert s["falls"] == 1 and s["components"]["fall"] == -R_DEFAULTS["fall"]


# ---------------------------------------------------------------- lifecycle
def test_reset_reproducible_and_seeded():
    env = LocoEnv()
    o1 = env.reset()
    o2 = env.reset()
    assert np.array_equal(o1, o2), "unseeded resets must be deterministic"
    e2 = LocoEnv()
    e2.reset(seed=123)
    p2 = np.asarray(e2.world.agent_pos(), dtype=float)
    e3 = LocoEnv()
    e3.reset(seed=123)
    p3 = np.asarray(e3.world.agent_pos(), dtype=float)
    assert np.allclose(p2, p3, atol=1e-12), "same seed -> same start pose"
    e4 = LocoEnv()
    e4.reset(seed=456)
    p4 = np.asarray(e4.world.agent_pos(), dtype=float)
    e5 = LocoEnv()
    e5.reset(seed=789)
    p5 = np.asarray(e5.world.agent_pos(), dtype=float)
    assert not np.allclose(p4, p5, atol=1e-6), "different seeds -> different poses"
    base = np.asarray(env.world.agent_pos(), dtype=float)
    for p in (p2, p4, p5):
        assert abs(p[0] - base[0]) <= 0.16 and abs(p[1] - base[1]) <= 0.16, p
        assert p[2] == pytest.approx(base[2], abs=1e-6)  # height untouched
    # NOTE: the OBSERVATION vectors for different start positions are
    # identical here by design — absolute root position is NOT an observation
    # field in layout v2 (documented); the jitter moves the WORLD start pose.


def test_seeded_reproducible_episodes():
    """Same seed, same action script -> identical episode metric-for-metric."""
    def run(seed):
        env = LocoEnv(cfg={"rl": {"max_episode_steps": 60}})
        env.reset(seed=seed)
        rng = np.random.default_rng(7)
        for _ in range(60):
            a = (rng.uniform(-1, 1, env.action_dim) * 0.3).astype(np.float32)
            env.step(a)
        return env.episode_summary()

    s1 = run(99)
    s2 = run(99)
    for k in ("reward_total", "path_len_m", "effort_total_j", "net_displacement_m"):
        assert s1[k] == pytest.approx(s2[k], rel=1e-9), k
    assert s1["steps"] == s2["steps"]


def test_episode_isolation_no_state_leakage():
    """After termination, reset zeroes episode accounting and the world is
    fresh: identical action scripts on consecutive episodes measure the same."""
    env = LocoEnv(cfg={"rl": {"max_episode_steps": 40}})
    env.reset(seed=5)
    rng = np.random.default_rng(3)
    acts = [(rng.uniform(-1, 1, env.action_dim) * 0.4).astype(np.float32)
            for _ in range(40)]
    for a in acts:
        env.step(a)
    s1 = env.episode_summary()
    env.reset(seed=5)
    for a in acts:
        env.step(a)
    s2 = env.episode_summary()
    for k in ("reward_total", "path_len_m", "effort_total_j"):
        assert s1[k] == pytest.approx(s2[k], rel=1e-6), k
    # counters truly zeroed by reset (no leakage from the first episode)
    obs = env.reset(seed=6)
    assert env.step_count == 0
    assert env.episode_summary()["steps"] == 0
    assert env.action_clipping()["calls"] == 0
    assert env.episode_summary()["faulted_windows"] == 0


def test_obs_action_dims_stable_across_episodes():
    # M7-0a: the DEFAULT contract is v4 (43-dim v3 prefix + command block);
    # the read-only legacy mode still constructs the exact v3 contract.
    env = LocoEnv()
    o = env.reset()
    assert o.shape == (68,)
    env.reset(seed=1)
    o2 = env.reset()
    assert o2.shape == (68,)
    assert LocoEnv.action_dim == 15 and LocoEnv.contract_version == 4
    legacy = LocoEnv(cfg={"rl": {"contract_version": 3}})
    assert (legacy.obs_dim, legacy.action_dim, legacy.contract_version) ==         (43, 14, 3)


# ---------------------------------------------------------------- curriculum
def test_curriculum_transitions_deterministic_and_explicit():
    env = LocoEnv()
    env.reset()
    c = Curriculum(env)
    assert c.stage.id == "standing"
    assert env.target_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert env.upright_assist == pytest.approx(0.6)
    transitions = []
    for _ in range(c.stages[0].duration_decisions + 20):
        env.step(_zero(env))
        if c.step():
            transitions.append((c.stage.id, c.steps_in_stage))
    assert transitions == [("controlled", 0)], transitions
    # B3 measured correction: stages command the walkable chest axis (world +Y
    # at spawn; probe_b3_yaw: |dy| 70x |dx| for waist-locked policies)
    assert env.target_velocity == pytest.approx([0.0, 0.25, 0.0])
    assert env.upright_assist == pytest.approx(0.3)
    # reproducible: a second run of the same sequence gives the same crossings
    env.reset()
    c2 = Curriculum(env)
    n = 0
    for _ in range(c2.stages[0].duration_decisions + 20):
        env.step(_zero(env))
        if c2.step():
            n += 1
    assert n == 1 and c2.stage.id == "controlled"


def test_curriculum_phase_application_non_cumulative():
    """The classic dead-code defect (gravity multiplied every apply) must not
    be reachable: applying the same stage repeatedly keeps gravity identical."""
    env = LocoEnv()
    env.reset()
    c = Curriculum(env)
    g0 = np.array(env.world.m.opt.gravity, dtype=float).copy()
    c.select("controlled")
    c.select("standing")
    c.select("controlled")
    g = np.array(env.world.m.opt.gravity, dtype=float)
    assert np.allclose(g, g0), "gravity must rewrite from base, never accumulate"


def test_curriculum_reserved_stages_fail_loudly():
    c = Curriculum()
    with pytest.raises(ValueError):
        c.select("disturbance")
    with pytest.raises(ValueError):
        c.advance(3)
    with pytest.raises(ValueError):
        c.select("no_such_stage")


def test_curriculum_reward_phase_explicit():
    """Stage phases may (deliberately) configure reward coefficients; the
    application is idempotent (non-cumulative) by set_phase semantics."""
    from dataclasses import asdict, replace
    from playground.rl.curriculum import Stage
    stages = [DEFAULT_STAGES[0],
              replace(DEFAULT_STAGES[1],
                      phase={"target_velocity": [0.25, 0.0, 0.0],
                             "upright_assist": 0.3, "max_steps": 400,
                             "reward_coeffs": {"progress": 4.0,
                                               "velocity_tracking": 2.0}})]
    env = LocoEnv()
    env.reset()
    c = Curriculum(env, stages=stages)
    for _ in range(stages[0].duration_decisions + 5):
        env.step(_zero(env))
        if c.step():
            break
    assert c.stage.id == "controlled"
    obs, r, term, trunc, info = env.step(_zero(env))
    # velocity at ~0 against the 0.25 target: exp(-0.0625/0.1), doubled weight
    assert info["components"]["velocity_tracking"] == pytest.approx(
        float(np.exp(-0.0625 / 0.1) * 2.0), rel=5e-3)


def test_curriculum_snapshot_records_contract():
    c = Curriculum()
    snap = c.snapshot()
    ids = [s["id"] for s in snap["stages"]]
    assert ids == ["standing", "controlled", "sustained", "disturbance",
                   "terrain", "navigation"]
    assert [s["env_ready"] for s in snap["stages"]][:3] == [True, True, True]
    assert [s["env_ready"] for s in snap["stages"]][3:] == [False, False, False]


# ---------------------------------------------------------------- compat
def test_scripted_gait_benchmark_horizon_in_env():
    """The scripted gait (benchmark, never a learning claim) still works inside
    the env. Anchor re-measured for the B4 rig (2026-11-18, Jamie decision):
    the head inertial fix removes ~10 kg from the upper body, changing the
    scripted walk's stride energetics — in-harness measurement on the fixed
    rig is 0.7301 m over the 8 s horizon (0 falls, upright 1.000). The old-rig
    anchor 0.761 ± 0.05 is preserved as historical B3 evidence in
    reports/m6b/ and was bit-stable across the whole B2/B3 eras."""
    env = LocoEnv(cfg={"rl": {"max_episode_steps": 400}})
    env.reset()
    env.world.set_gait("walk", speed=1.0, heading=0.0)
    zero = _zero(env)
    dist = None
    for _ in range(400):
        env.step(zero)
    s = env.episode_summary()
    dist = s["net_displacement_m"]
    assert dist == pytest.approx(0.730, abs=0.05), f"scripted baseline moved: {dist}"
    assert s["falls"] == 0 and s["faulted_windows"] == 0
    assert s["upright_fraction"] == pytest.approx(1.0, abs=1e-4)


def test_decimation_hold_semantics():
    """One decision holds its mapped targets across action_repeat substeps
    (a slower policy at the documented 20 ms; PD still every 2 ms)."""
    env = LocoEnv()
    env.reset()
    a = np.full(env.action_dim, 0.5, dtype=np.float32)
    t_before = float(env.world.time)
    env.step(a)
    t_after = float(env.world.time)
    assert (t_after - t_before) == pytest.approx(env.action_repeat * 0.002, rel=1e-6)
    name = env.joint_names[0]
    low, high = env.joint_range[name]
    assert env.world._pd[name]["target"] == pytest.approx(
        low + (0.5 + 1) * 0.5 * (high - low), abs=1e-6)
