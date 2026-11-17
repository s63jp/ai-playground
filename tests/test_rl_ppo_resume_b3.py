"""M6-B stage B3 — checkpoint RESUME continuity proof (real environment).

Jamie's requirement: a resume path is only proven when it can be shown that
a run interrupted after k iterations and continued from its checkpoint ends
in EXACTLY the same state as an uninterrupted run of the same seed and
schedule — no silent restart, no duplicated progress — with every part of
the state restored:

  1. actor + critic weights (bit-exact),
  2. optimiser state (Adam moment tensors, bit-exact through identical
     updates),
  3. torch RNG state (identical sampling sequences),
  4. trainer counters (valid steps, iteration, episode counter — the
     episode seed sequence continues without repeats),
  5. curriculum stage (the checkpointed stage is re-selected; its set_phase
     rewrite is the documented non-cumulative rewrite, a semantically exact
     no-op).

The interrupted twin continues at the recorded stage (standing → controlled
transition happens identically in both runs) and the final network states
are compared bit-for-bit.
"""
import numpy as np
import pytest

from playground.rl.checkpoint import apply_rng, load_checkpoint, save_checkpoint
from playground.rl.curriculum import Curriculum
from playground.rl.env import LocoEnv
from playground.rl.ppo import PPOConfig, PPOTrainer, TanhGaussianActor, Critic

ROLL = 384
SEED = 4242


def _cfg():
    return PPOConfig(rollout_steps=ROLL, epochs=2, minibatch=128, gamma=0.9,
                     gae_lambda=0.9, lr=3e-4)


def _env():
    return LocoEnv(cfg={"agent_body": {"type": "humanoid"},
                        "rl": {"max_episode_steps": 400}})


def test_resume_continuity_bit_exact(tmp_path):
    torch = TanhGaussianActor(3, 2).torch
    torch.set_num_threads(1)
    cfg = _cfg()
    torch.manual_seed(SEED)

    # ---- Run A: uninterrupted, 3 iterations (transition happens after it 1)
    env_a = _env()
    torch.manual_seed(SEED)
    actor_a = TanhGaussianActor(env_a.obs_dim, env_a.action_dim, seed=SEED)
    critic_a = Critic(env_a.obs_dim, seed=SEED)
    cur_a = Curriculum(env_a)          # binds + applies standing
    trainer_a = PPOTrainer(env_a, actor_a, critic_a, cfg, seed=SEED,
                           total_steps=3 * ROLL,
                           stage_budgets={"standing": ROLL}, curriculum=cur_a)
    trainer_a.train()

    # ---- Run B: interrupted after iteration 1, checkpointed exactly as the
    # CLI does (weights + optimiser + rng + counters + curriculum pointer)
    env_b = _env()
    torch.manual_seed(SEED)
    actor_b = TanhGaussianActor(env_b.obs_dim, env_b.action_dim, seed=SEED)
    critic_b = Critic(env_b.obs_dim, seed=SEED)
    cur_b = Curriculum(env_b)
    trainer_b = PPOTrainer(env_b, actor_b, critic_b, cfg, seed=SEED,
                           total_steps=1 * ROLL,
                           stage_budgets={"standing": ROLL}, curriculum=cur_b)
    trainer_b.train()                  # stops after 1 iteration
    assert trainer_b.total_valid_steps == ROLL
    ck_path = tmp_path / "resume.json"
    save_checkpoint(ck_path, actor=actor_b, critic=critic_b,
                    optimizer_actor=trainer_b.opt_a,
                    optimizer_critic=trainer_b.opt_c, env=env_b,
                    cfg_dict=cfg.as_dict(), seed=SEED,
                    trainer_state=trainer_b.rng_state(),
                    extra_meta={"curriculum": {"index": cur_b.index,
                                               "stage_id": cur_b.stage.id},
                                "training_steps": trainer_b.total_valid_steps})
    # after 1 standing-budget iteration the advance must already have happened
    assert cur_b.stage.id == "controlled"

    # ---- Run B2: fresh instance, restore EVERYTHING, continue to 3 iterations
    env_b2 = _env()
    ck = load_checkpoint(ck_path, env=env_b2)   # contract-fingerprint checked
    actor_b2, critic_b2 = ck["actor"], ck["critic"]
    cur_b2 = Curriculum(env_b2)                  # binds standing first
    # counters, torch RNG, optimiser, curriculum stage — in the CLI's order
    trainer_b2 = PPOTrainer(env_b2, actor_b2, critic_b2, cfg, seed=SEED,
                            total_steps=3 * ROLL,
                            stage_budgets={"standing": ROLL}, curriculum=cur_b2)
    trainer_b2.opt_a = ck["optimizer_actor"]
    trainer_b2.opt_c = ck["optimizer_critic"]
    apply_rng(ck["rng"])
    trainer_b2.load_rng_state(ck["rng"])
    cur_b2.select(int(ck["meta"]["curriculum"]["index"]))
    assert cur_b2.stage.id == "controlled", "curriculum stage must restore"
    trainer_b2.train()

    # ---- continuity assertions ------------------------------------------------
    # 1+2. weights AND optimiser-driven behaviour: the final networks are
    #      bit-identical to the uninterrupted run's (any restoration gap —
    #      weights, Adam moments, RNG, seeds, stage — desynchronises them)
    sd_a = {k: v.numpy() for k, v in actor_a.state_dict().items()}
    sd_b2 = {k: v.numpy() for k, v in actor_b2.state_dict().items()}
    assert set(sd_a) == set(sd_b2)
    for k in sd_a:
        assert np.array_equal(sd_a[k], sd_b2[k]), \
            f"actor {k} diverged — restoration is not exact"
    cd_a = {k: v.numpy() for k, v in critic_a.state_dict().items()}
    cd_b2 = {k: v.numpy() for k, v in critic_b2.state_dict().items()}
    for k in cd_a:
        assert np.array_equal(cd_a[k], cd_b2[k]), \
            f"critic {k} diverged — restoration is not exact"
    # checkpointed intermediate state proved the weight restore itself:
    w_saved = np.asarray(load_checkpoint(ck_path, env=env_b2)["actor"]
                         .state_dict()["log_std"].numpy())
    w_b = np.asarray(actor_b.state_dict()["log_std"].numpy())
    assert np.array_equal(w_saved, w_b), "saved weights differ from the saver"
    # 3. torch RNG: both runs sampled the same z stream — verified indirectly
    #    by the bit-equality above (z drives the actions and the update).
    # 4. counters: no restart, no duplication
    assert trainer_b2.total_valid_steps == trainer_a.total_valid_steps == 3 * ROLL
    assert trainer_b2.iteration == trainer_a.iteration
    assert trainer_b2.episode_counter[0] == trainer_a.episode_counter[0], \
        "episode seed sequence must continue, never restart"
    assert trainer_b2.faulted_windows_total == trainer_a.faulted_windows_total
    # 5. curriculum: same final stage, same phase in the env
    assert cur_b2.stage.id == cur_a.stage.id
    assert np.allclose(env_b2.target_velocity, env_a.target_velocity)
    assert env_b2.upright_assist == pytest.approx(env_a.upright_assist)
    # optimiser state carries through (lr and moments count)
    assert len(trainer_b2.opt_a.state) == len(trainer_a.opt_a.state)
    assert abs(trainer_b2.opt_a.param_groups[0]["lr"]
               - trainer_a.opt_a.param_groups[0]["lr"]) < 1e-18
