"""M6-B stage B3 — PPO integration battery on the REAL B2 environment.

Runs genuine (small) training through `LocoEnv` + the real reward pipeline
and real MuJoCo physics: parameter updates on actual rollouts, faulted-window
exclusion, checkpoint round-trip on the live contract, evaluation separation
(held-out seeds, deterministic actions, no parameter mutation).

The expensive "trains a walking policy" acceptance demonstration is a
separately documented script+command (reports/m6b/B3_PPO_LEARNING.md), not an
in-suite test.
"""
import json

import numpy as np
import pytest

from playground.config import load_config
from playground.rl import eval as rl_eval
from playground.rl.checkpoint import (CheckpointError, load_checkpoint,
                                      save_checkpoint)
from playground.rl.curriculum import Curriculum
from playground.rl.env import LocoEnv
from playground.rl.ppo import (PPOConfig, PPOError, PPOTrainer,
                               TanhGaussianActor, Critic, collect_rollout)

SMALL_CFG = {"rollout_steps": 384, "epochs": 2, "minibatch": 128, "gamma": 0.9,
             "gae_lambda": 0.9, "lr": 3e-4}


def _small_ppo_cfg():
    c = PPOConfig(rollout_steps=SMALL_CFG["rollout_steps"],
                  epochs=SMALL_CFG["epochs"], minibatch=SMALL_CFG["minibatch"],
                  gamma=SMALL_CFG["gamma"], gae_lambda=SMALL_CFG["gae_lambda"],
                  lr=SMALL_CFG["lr"])
    return c


def _env():
    return LocoEnv(cfg={"agent_body": {"type": "humanoid"},
                        "rl": {"max_episode_steps": 400}})


def _train_a_bit(env, actor, critic, iterations=3, curriculum=None,
                 stage_budgets=None, on_iteration=None):
    cfg = _small_ppo_cfg()
    trainer = PPOTrainer(env, actor, critic, cfg, seed=20261117,
                         total_steps=iterations * cfg.rollout_steps,
                         stage_budgets=stage_budgets or {},
                         curriculum=curriculum, on_iteration=on_iteration)
    summary = trainer.train()
    return trainer, summary


def test_real_training_updates_parameters_and_records_metrics():
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    torch.manual_seed(20261117)
    env = _env()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=20261117)
    critic = Critic(env.obs_dim, seed=20261117)
    before = {k: v.numpy().copy() for k, v in actor.state_dict().items()}
    recs = []
    trainer, summary = _train_a_bit(env, actor, critic, iterations=3,
                                    on_iteration=recs.append)
    assert summary["stopped"] is False
    assert trainer.total_valid_steps == 3 * SMALL_CFG["rollout_steps"]
    after = {k: v.numpy() for k, v in actor.state_dict().items()}
    delta = sum(float(((after[k] - before[k]) ** 2).sum()) for k in before)
    assert delta > 1e-10, "genuine updates must move the network weights"
    assert len(recs) == 3
    for rec in recs:
        assert rec["episodes"] >= 1
        assert not rec["update_rejected"]
        assert math_isfinite(rec["policy_loss"]) and math_isfinite(rec["value_loss"])
        assert rec["roll_episodes"] == rec["episodes"]
        assert "roll_axis_displacement_mean" in rec
        assert rec["faulted_windows"] == 0
        assert rec["roll_upright_fraction_mean"] > 0.0
    # B1 intact during training: no faults were produced and no rejections
    assert trainer.rollout_rejections == 0
    assert trainer.faulted_windows_total == 0
    assert env.clip_stats["rejected_calls"] == 0


def math_isfinite(x):
    return x is not None and np.isfinite(x)


def test_rollout_buffer_contents_from_real_env():
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    env = _env()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=77)
    critic = Critic(env.obs_dim, seed=78)
    buf = collect_rollout(env, actor, critic, 220, seed=20261117, episode_counter=[0])
    assert len(buf) == 220
    assert buf.obs.shape == (220, env.obs_dim) == (220, 43)
    assert buf.z.shape == (220, 14) and buf.actions.shape == (220, 14)
    assert np.all(np.isfinite(buf.obs)) and np.all(np.isfinite(buf.z))
    # obs shape equals the v3 contract; actions inside bounds
    assert np.all(np.abs(buf.actions) <= 1.0)
    # deterministic re-evaluation: old logp equals current-policy logp of z
    with actor.torch.no_grad():
        logp_now, _ = actor.evaluate_z(buf.obs, buf.z)
    assert np.allclose(logp_now.numpy(), buf.logp.astype(np.float32), atol=1e-4), \
        "stored logp must be the logp of the SAME z under the policy that sampled it"


def test_faulted_window_excluded_and_prev_row_bootstrapped_off():
    """A faulted (B1) window must never enter the data; the trajectory ends
    there (the previous valid row loses its bootstrap)."""
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    env = _env()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=90)
    critic = Critic(env.obs_dim, seed=91)

    class FaultyOnce(LocoEnv):
        """Real env; window #5 is flagged FAULTED (the B1 semantics exercised
        here by contract, the fault PATH itself is proven in the B1/B2 suites
        with real MuJoCo warning injection)."""
        def step(self, action):
            out = super().step(action)
            if self.step_count == 5:
                obs, r, term, trunc, info = out
                info = dict(info)
                info["faulted"] = True
                info["physics_warnings"] = ["simulated_b3_exclusion_test"]
                zeroed = np.where(np.isfinite(obs), obs, 0.0).astype(np.float32)
                return zeroed, 0.0, True, False, info
            return out

    fenv = FaultyOnce(cfg={"agent_body": {"type": "humanoid"},
                           "rl": {"max_episode_steps": 400}})
    buf = collect_rollout(fenv, actor, critic, 300, seed=20261117,
                          episode_counter=[0])
    # windows 1-4 -> rows 0-3; window 5 is FAULTED and must be absent; the
    # previous valid row (index 3) is marked terminated (bootstrap void) and
    # row 4 is the first row of the fresh episode.
    assert len(buf) == 300
    assert bool(buf.terminated[3]) and bool(buf.done[3])
    # the row before the fault carried a valid (non-fault) window reward
    assert buf.rewards[3] > 0.0
    # after the boundary the observation must come from a fresh reset:
    # standing pose -> upright margin near 1, zero velocity
    assert abs(float(buf.obs[4][34]) - 1.0) < 1e-3
    assert np.allclose(buf.obs[4][31:34], 0.0, atol=1e-6)


def test_trainer_advances_curriculum_explicitly():
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    env = _env()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=101)
    critic = Critic(env.obs_dim, seed=102)
    cur = Curriculum(env)     # binds + applies standing
    recs = []
    cfg = _small_ppo_cfg()
    trainer = PPOTrainer(env, actor, critic, cfg, seed=1,
                         total_steps=2 * cfg.rollout_steps,
                         stage_budgets={"standing": cfg.rollout_steps,
                                        "controlled": cfg.rollout_steps},
                         curriculum=cur, on_iteration=recs.append)
    trainer.train()
    # explicit advances at the configured budgets: standing -> controlled
    # (iteration 1), controlled -> sustained (iteration 2); each advance is a
    # Curriculum.advance() whose set_phase rewrite is observable in the env
    assert recs[0]["stage_transition_to"] == "controlled"
    assert recs[0]["stage"] == "controlled"
    assert recs[1]["stage_transition_to"] == "sustained"
    assert recs[1]["stage"] == "sustained"
    assert np.allclose(env.target_velocity, [0.0, 0.5, 0.0])
    assert env.upright_assist == pytest.approx(0.0)


def test_checkpoint_roundtrip_on_real_contract(tmp_path):
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    env = _env()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=110)
    critic = Critic(env.obs_dim, seed=111)
    trainer, _ = _train_a_bit(env, actor, critic, iterations=1)
    cp = save_checkpoint(tmp_path / "real.json", actor=actor, critic=critic,
                         optimizer_actor=trainer.opt_a,
                         optimizer_critic=trainer.opt_c, env=env,
                         cfg_dict=trainer.cfg.as_dict(), seed=20261117,
                         trainer_state=trainer.rng_state())
    ck = load_checkpoint(tmp_path / "real.json", env=env)
    out = ck["actor"]
    # identical deterministic evaluation on the live env's actual observations
    obs = env.reset(seed=555)
    for _ in range(30):
        a1 = actor.deterministic_action(obs)
        a2 = out.deterministic_action(obs)
        assert np.array_equal(a1, a2), "checkpoint must reproduce the policy exactly"
        obs, r, term, trunc, info = env.step(a1)
        if term or trunc:
            env.reset(seed=556)
    # optimiser state restored for exact continuation
    assert ck["optimizer_actor"] is not None
    assert abs(ck["optimizer_actor"].param_groups[0]["lr"]
               - trainer.opt_a.param_groups[0]["lr"]) < 1e-15
    # loading does not modify the checkpoint file
    data1 = (tmp_path / "real.json").read_text(encoding="utf-8")
    load_checkpoint(tmp_path / "real.json", env=env)
    assert (tmp_path / "real.json").read_text(encoding="utf-8") == data1


def test_checkpoint_rejects_foreign_contract(tmp_path):
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    env = _env()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=120)
    critic = Critic(env.obs_dim, seed=121)
    save_checkpoint(tmp_path / "v2.json", actor=actor, critic=critic,
                    env=env)
    body = json.loads((tmp_path / "v2.json").read_text(encoding="utf-8"))
    body["meta"]["contract"]["contract_version"] = 2
    body["meta"]["contract"]["obs_dim"] = 40
    import hashlib
    body["content_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in body.items() if k != "content_sha256"},
        sort_keys=True).encode()).hexdigest()
    (tmp_path / "v2.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(CheckpointError) as e:
        load_checkpoint(tmp_path / "v2.json", env=env)
    assert "obs_dim 40 != 43" in str(e.value)


def test_eval_separation_no_param_mutation_and_reproducible():
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    env = _env()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=130)
    before = {k: v.numpy().copy() for k, v in actor.state_dict().items()}
    phase = {"target_velocity": [0.0, 0.5, 0.0], "upright_assist": 0.0,
             "gravity_scale": 1.0, "max_steps": 400}
    r1 = rl_eval.eval_policy(actor, env, episodes=2, seed_base=97010_000,
                             phase=phase, max_steps=400)
    after = {k: v.numpy() for k, v in actor.state_dict().items()}
    for k in before:
        assert np.array_equal(before[k], after[k]), \
            "evaluation must never mutate policy parameters"
    env2 = _env()
    r2 = rl_eval.eval_policy(actor, env2, episodes=2, seed_base=97010_000,
                             phase=phase, max_steps=400)
    for k in ("axis_displacement_m", "net_displacement_m", "upright_fraction"):
        assert r1["summary"][k]["mean"] == pytest.approx(r2["summary"][k]["mean"],
                                                         rel=0, abs=0.0), \
            "matched held-out evaluation must be bit-reproducible"
    # held-out: eval seeds are disjoint from the training seed space
    assert 97010_000 > 20261117 + 10_000


def test_zero_and_gait_eval_harness_contract():
    env = _env()
    phase = {"target_velocity": [0.0, 0.5, 0.0], "upright_assist": 0.0,
             "gravity_scale": 1.0, "max_steps": 400}
    z = rl_eval.eval_zero(env, episodes=1, seed_base=97010_000, phase=phase,
                          max_steps=400)
    assert z["summary"]["upright_fraction"]["mean"] > 0.9, \
        "the zero-action control stands the whole horizon (B2-measured)"
    assert abs(z["summary"]["axis_displacement_m"]["mean"]) < 0.05
    g = rl_eval.eval_gait(env, episodes=1, seed_base=97010_000, phase=phase,
                          heading=90.0, max_steps=400)
    # B4 re-measured anchor (Jamie decision 2026-11-18): the harness gait at
    # h=90 measures 0.7057 m on the fixed-head rig (head inertial fix; old-rig
    # anchor 0.761 ± 0.05 → harness 0.7491 m preserved in B3 evidence).
    assert g["summary"]["net_displacement_m"]["mean"] == pytest.approx(0.706, abs=0.05), \
        "the scripted gait benchmark must reproduce through the harness"
    assert g["summary"]["falls_total"] == 0 and g["summary"]["faulted_windows_total"] == 0


def test_upright_travel_breakdown_separates_travel_from_collapses():
    """Jamie's B3 requirement: commanded-axis displacement must never be
    conflated with walking. The summary carries a no-fall breakdown so
    'travelled while upright' and 'moved then collapsed' are separate
    statistics."""
    env = _env()
    phase = {"target_velocity": [0.0, 0.5, 0.0], "upright_assist": 0.0,
             "gravity_scale": 1.0, "max_steps": 400}
    g = rl_eval.eval_gait(env, episodes=2, seed_base=97010_000, phase=phase,
                          heading=90.0, max_steps=400)
    nf = g["summary"].get("axis_displacement_m_no_fall")
    assert nf is not None and nf["n"] == 2, \
        "no-fall episodes must drive the upright-travel statistic"
    assert abs(nf["mean"] - g["summary"]["axis_displacement_m"]["mean"]) < 1e-9
    assert g["summary"].get("axis_displacement_m_fell") is None
    # and a per-episode record carries its own fall flag for report tables
    for e in g["episodes"]:
        assert e["falls"] == 0
        assert "axis_displacement_m" in e
