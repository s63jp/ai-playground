"""PPO network scaling — 64×64 / 128×128 / 256×256 selectable (M7-era).

The architecture is ONE config knob, `rl.ppo.hidden` (PPOConfig.hidden),
already consumed by every fresh construction site (scripts/rl_ppo.py fresh,
rl_live interop); a run can also override it per-run with `train --hidden`.
This battery proves the full lifecycle for non-default sizes:

- construction + forward-pass shapes for 64/128/256 and arbitrary widths
  (the mechanism was never preset-restricted);
- per-run configuration: YAML default preserved at [64, 64], the CLI flag
  validated by the SAME from_rl_cfg rules, every selection recorded;
- checkpoints RECORD their actual architecture (policy.hidden) and the
  strict loader RESTORES it — with the recorded-vs-tensor mismatch rejected
  loudly in both directions (no silent substitution, no lenient load path);
- PRE-M7-0b checkpoints (no recorded widths) still load: the historical
  default (64, 64) matches their tensors, and a mismatched shape is refused;
- optimiser state restores shape-consistently (Adam moments fit the
  rebuilt architecture and a fresh update runs on them);
- transfer-fork surgery PRESERVES the parent's recorded widths for
  non-default parents (and derives them from the parent trunk when absent)
  with misrecorded architectures refused BEFORE any port.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from playground.rl.checkpoint import (CheckpointError, _hash,
                                      load_checkpoint, save_checkpoint)
from playground.rl.env import LocoEnv
from playground.rl.ppo import (PPOConfig, PPOError, RolloutBuffer,
                               TanhGaussianActor, Critic, ppo_update)
from playground.rl.transfer import (TransferError, parent_sha256,
                                    surgery_transfer)

_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts.parent) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_scripts.parent))
import scripts.rl_ppo as rl_ppo  # noqa: E402


@pytest.fixture(scope="module")
def _v4_env():
    """ONE shared read-only v4 env (dims + fingerprints only)."""
    return LocoEnv()


ENV_SIZES = ((64, 64), (128, 128), (256, 256))


def _trunk_widths(actor):
    sd = {k: np.asarray(v) for k, v in actor.state_dict().items()
          if k.startswith("trunk.") and k.endswith(".weight")}
    return tuple(int(sd[k].shape[0]) for k in sorted(sd, key=lambda s: int(s.split(".")[1])))


def _critic_widths(critic):
    sd = {k: np.asarray(v) for k, v in critic.state_dict().items()
          if k.endswith(".weight")}
    ordered = [sd[k] for k in sorted(sd, key=lambda s: int(s.split(".")[0]))]
    return tuple(int(w.shape[0]) for w in ordered[:-1])


# ----------------------------------------------------------- construction
@pytest.mark.parametrize("hidden", (*ENV_SIZES, (48, 96)))
def test_actor_critic_construction_per_size(_v4_env, hidden):
    """Every selectable width builds BOTH networks consistently, including
    arbitrary positive widths (never preset-restricted)."""
    obs_dim, act_dim = _v4_env.obs_dim, _v4_env.action_dim
    actor = TanhGaussianActor(obs_dim, act_dim, hidden=hidden, seed=1)
    critic = Critic(obs_dim, hidden=hidden, seed=2)
    n = len(hidden)
    assert _trunk_widths(actor) == hidden
    sd = actor.state_dict()
    assert int(np.asarray(sd["trunk.0.weight"]).shape[1]) == obs_dim
    last = f"trunk.{2 * (n - 1)}.weight"
    assert np.asarray(sd["mu_head.weight"]).shape == (act_dim, hidden[-1])
    assert int(np.asarray(sd[last]).shape[0]) == hidden[-1]
    assert _critic_widths(critic) == hidden
    csd = critic.state_dict()
    assert np.asarray(csd["0.weight"]).shape == (hidden[0], obs_dim)
    assert np.asarray(csd[f"{2 * n}.weight"]).shape == (1, hidden[-1])
    # the mu_head/init contract holds at every width (no output saturation)
    act0 = actor.deterministic_action(np.zeros(obs_dim, dtype=np.float32))
    assert act0.shape == (act_dim,) and np.all(np.isfinite(act0))
    assert np.max(np.abs(act0)) < 0.05


@pytest.mark.parametrize("hidden", ENV_SIZES)
def test_forward_pass_output_dimensions(_v4_env, hidden):
    actor = TanhGaussianActor(_v4_env.obs_dim, _v4_env.action_dim,
                              hidden=hidden, seed=3)
    critic = Critic(_v4_env.obs_dim, hidden=hidden, seed=4)
    obs = np.random.default_rng(5).normal(0, 0.5, _v4_env.obs_dim).astype(np.float32)
    act, z, logp = actor.sample(obs)
    assert act.shape == (15,) and z.shape == (15,)
    assert np.all(np.isfinite(logp)) and np.all(np.abs(act) < 1.0)
    batch = np.repeat(obs[None, :], 7, axis=0)
    assert critic.values_batch(batch).shape == (7,)
    probe = np.array([actor.deterministic_action(p) for p in batch])
    assert probe.shape == (7, 15)


def test_deep_trunk_width_derivation_six_layers():
    """The recorded policy.hidden must be the TRUE widths for deep trunks:
    layer state-dict indexes reach 12+ and a bare key sort would put
    'trunk.10.weight' before 'trunk.2.weight' (the M7-0b first-cut bug,
    generalised to 6+ layers)."""
    from playground.rl.checkpoint import _actor_hidden
    from playground.rl.transfer import _actor_hidden_widths
    deep = (8, 12, 16, 24, 24, 16)
    actor = TanhGaussianActor(43, 14, hidden=deep, seed=6)
    assert _actor_hidden(actor) == list(deep)
    assert _actor_hidden_widths(actor.state_dict()) == deep


# ------------------------------------------------------------ config path
@pytest.mark.parametrize("listed,expected", (
    ([64, 64], (64, 64)), ([128, 128], (128, 128)), ([256, 256], (256, 256)),
    ([32], (32,)),
))
def test_config_loads_each_supported_size(listed, expected):
    cfg = PPOConfig.from_rl_cfg({"ppo": {"hidden": listed}})
    assert cfg.hidden == expected
    assert cfg.as_dict()["hidden"] == list(expected)   # config.json echo


def test_config_rejects_invalid_width_lists():
    for bad in ([0, 0], [-32], [64] * 9, "big", ["x"]):
        with pytest.raises(PPOError):
            PPOConfig.from_rl_cfg({"ppo": {"hidden": bad}})


def test_yaml_default_stays_64x64_and_knob_is_live():
    from playground.config import load_config
    cfg = load_config(None)
    assert cfg["rl"]["ppo"]["hidden"] == [64, 64]     # B3/B4 default preserved
    assert PPOConfig.from_rl_cfg(cfg["rl"]).hidden == (64, 64)
    merged = load_config(None, overrides={"rl": {"ppo": {"hidden": [256, 256]}}})
    assert PPOConfig.from_rl_cfg(merged["rl"]).hidden == (256, 256)


def test_launcher_hidden_override_helper():
    """--hidden parses HERE and is validated by the SAME from_rl_cfg rules
    (spliced into the rl dict before PPOConfig construction)."""
    rl = rl_ppo._rl_cfg_with_hidden_override({"ppo": {"lr": 3e-4}}, "256,256")
    assert rl["ppo"]["hidden"] == [256, 256]
    assert rl["ppo"]["lr"] == 3e-4
    thin = rl_ppo._rl_cfg_with_hidden_override({}, "32")
    assert thin["ppo"]["hidden"] == [32]
    assert rl_ppo._rl_cfg_with_hidden_override({"ppo": {"hidden": [8]}},
                                               None) == {"ppo": {"hidden": [8]}}
    with pytest.raises(PPOError, match="--hidden"):
        rl_ppo._rl_cfg_with_hidden_override({}, "a,b")
    with pytest.raises(PPOError):
        PPOConfig.from_rl_cfg(
            rl_ppo._rl_cfg_with_hidden_override({}, "0,0"))   # validated path


# ------------------------------------------------------ checkpoint lifecycle
def _envless_fps(env):
    return {"obs_fingerprint": env.observation_layout(),
            "action_fingerprint": env.action_layout()}


def _live_fp(env):
    return {"observation": env.observation_layout(),
            "action": env.action_layout(),
            "contract_version": env.contract_version,
            "obs_dim": int(env.obs_dim), "action_dim": int(env.action_dim)}


def _restore_widths_from_loaded(loaded):
    return (_trunk_widths(loaded["actor"]), _critic_widths(loaded["critic"]))


@pytest.mark.parametrize("hidden", ((128, 128), (256, 256)))
def test_checkpoint_records_and_restores_architecture(_v4_env, tmp_path, hidden):
    obs_dim = _v4_env.obs_dim
    actor = TanhGaussianActor(obs_dim, 15, hidden=hidden, seed=7)
    critic = Critic(obs_dim, hidden=hidden, seed=8)
    saved = save_checkpoint(tmp_path / "c.json", actor=actor, critic=critic,
                            **_envless_fps(_v4_env), seed=9)
    body = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
    assert body["meta"]["policy"]["hidden"] == list(hidden)   # recorded truth
    out = load_checkpoint(saved["path"], **_envless_fps(_v4_env))
    assert _restore_widths_from_loaded(out) == (hidden, hidden)
    # weights restored BIT-EXACTLY at the recorded architecture
    for k, v in actor.state_dict().items():
        assert np.allclose(np.asarray(out["actor"].state_dict()[k]),
                           np.asarray(v), rtol=0, atol=0), k
    for k, v in critic.state_dict().items():
        assert np.allclose(np.asarray(out["critic"].state_dict()[k]),
                           np.asarray(v), rtol=0, atol=0), k
    probe = np.random.default_rng(10).normal(0, 0.5, (4, obs_dim)).astype(np.float32)
    assert np.array_equal(
        np.asarray(out["critic"].values_batch(probe)),
        np.asarray(critic.values_batch(probe)))


def _rehash(path, mutate):
    """Rewrite a checkpoint body through `mutate` and re-seal the content
    hash the same way the loader recomputes it (the documented hand-built
    body path — makes a hash-consistent but lying architecture reachable)."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    mutate(body)
    body.pop("content_sha256", None)
    body["content_sha256"] = _hash({k: v for k, v in body.items()
                                    if k != "content_sha256"})
    Path(path).write_text(json.dumps(body), encoding="utf-8")
    return body


@pytest.mark.parametrize("recorded", ([64, 64], [256, 256]))
def test_no_silent_architecture_substitution(_v4_env, tmp_path, recorded):
    """A checkpoint whose TENSOR shapes disagree with its recorded widths is
    refused with a typed CheckpointError — there is no strict=False path and
    no partial load: the failure names the recorded architecture."""
    hidden = (128, 128)
    actor = TanhGaussianActor(_v4_env.obs_dim, 15, hidden=hidden, seed=11)
    critic = Critic(_v4_env.obs_dim, hidden=hidden, seed=12)
    saved = save_checkpoint(tmp_path / "c.json", actor=actor, critic=critic,
                            **_envless_fps(_v4_env), seed=13)
    _rehash(saved["path"], lambda b: b["meta"]["policy"].__setitem__(
        "hidden", recorded))
    with pytest.raises(CheckpointError, match="do not fit") as e:
        load_checkpoint(saved["path"], **_envless_fps(_v4_env))
    assert str(list(recorded)) in str(e.value)   # the recorded widths named


def test_config_hidden_must_be_a_list():
    with pytest.raises(PPOError):
        PPOConfig.from_rl_cfg({"ppo": {"hidden": 64}})   # not a list at all


def test_sizeless_historical_checkpoint_loads_at_64x64(tmp_path):
    """A PRE-M7-0b body carries NO policy.hidden: the loader falls back to
    the (64, 64) default and the strict load CONFIRMS it against the
    checkpoint's actual tensors (the historical-compatibility law; the real
    lineage files are re-verified bit-level outside the suite)."""
    obs_fp = {"dim": 43, "layout_version": 3,
              "sections": [{"name": "s", "slice": [0, 43]}],
              "joint_order": [f"j{i}" for i in range(14)]}
    act_fp = {"action_dim": 14, "layout_version": 3,
              "mapping": "target = lo + (a + 1)*0.5*(hi - lo)",
              "joints": [{"name": f"j{i}"} for i in range(14)]}
    actor = TanhGaussianActor(43, 14, seed=14)
    critic = Critic(43, hidden=(64, 64), seed=15)
    saved = save_checkpoint(tmp_path / "old.json", actor=actor, critic=critic,
                            obs_fingerprint=obs_fp, action_fingerprint=act_fp,
                            seed=16)
    body = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
    body["meta"]["policy"].pop("hidden")
    body.pop("content_sha256", None)
    body["content_sha256"] = _hash({k: v for k, v in body.items()
                                    if k != "content_sha256"})
    Path(saved["path"]).write_text(json.dumps(body), encoding="utf-8")
    out = load_checkpoint(saved["path"], obs_fingerprint=obs_fp,
                          action_fingerprint=act_fp)
    assert _trunk_widths(out["actor"]) == (64, 64)    # exact historical arch
    assert _critic_widths(out["critic"]) == (64, 64)
    assert np.array_equal(
        np.asarray(out["actor"].deterministic_action(
            np.zeros(43, dtype=np.float32))),
        np.asarray(actor.deterministic_action(np.zeros(43, dtype=np.float32))))


def test_sizeless_historical_architecture_mismatch_refused(tmp_path):
    """AND a sizeless body whose tensors are NOT (64, 64) is refused loudly —
    the default fallback never substitutes a silently different shape."""
    obs_fp = {"dim": 43, "layout_version": 3,
              "sections": [{"name": "s", "slice": [0, 43]}],
              "joint_order": [f"j{i}" for i in range(14)]}
    act_fp = {"action_dim": 14, "layout_version": 3,
              "mapping": "target = lo + (a + 1)*0.5*(hi - lo)",
              "joints": [{"name": f"j{i}"} for i in range(14)]}
    actor = TanhGaussianActor(43, 14, hidden=(128, 128), seed=17)
    critic = Critic(43, hidden=(128, 128), seed=18)
    saved = save_checkpoint(tmp_path / "weird.json", actor=actor, critic=critic,
                            obs_fingerprint=obs_fp, action_fingerprint=act_fp,
                            seed=19)
    _rehash(saved["path"], lambda b: b["meta"]["policy"].pop("hidden"))
    with pytest.raises(CheckpointError, match="64, 64"):
        load_checkpoint(saved["path"], obs_fingerprint=obs_fp,
                        action_fingerprint=act_fp)


# --------------------------------------------------- optimiser restoration
def _tiny_buffer(actor, critic, n=96, seed=21):
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 0.5, (n, actor.obs_dim)).astype(np.float32)
    z = rng.normal(0, 1.0, (n, actor.action_dim)).astype(np.float32)
    next_obs = rng.normal(0, 0.5, (n, actor.obs_dim)).astype(np.float32)
    with actor.torch.no_grad():
        logp_old, _ = actor.evaluate_z(obs, z)
    values = np.asarray(critic.values_batch(obs), dtype=np.float64)
    buf = RolloutBuffer(obs_dim=actor.obs_dim, action_dim=actor.action_dim)
    for i in range(n):
        buf.add(obs[i], next_obs[i], np.tanh(z[i]), z[i],
                float(logp_old[i]), float(values[i]), 1.0, False, False)
    return buf.finalize()


def test_optimiser_state_restores_fit_to_architecture(_v4_env, tmp_path):
    """The Adam moments of a 128×128 run restore shape-consistently into the
    rebuilt 128×128 networks and a further update runs on them (the restore
    is genuinely the same optimisation, not a fresh silent reset)."""
    hidden = (128, 128)
    actor = TanhGaussianActor(_v4_env.obs_dim, 15, hidden=hidden, seed=22)
    critic = Critic(_v4_env.obs_dim, hidden=hidden, seed=23)
    cfg = PPOConfig.from_rl_cfg({"ppo": {"rollout_steps": 96, "epochs": 1,
                                         "minibatch": 32, "hidden": list(hidden)}})
    # the TRAINER pattern: the persistent optimisers are passed INTO the
    # update (ppo_update builds throwaway ones when omitted) — exactly what
    # PPOTrainer saves into every checkpoint
    torch = actor.torch
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    stats = ppo_update(actor, critic, _tiny_buffer(actor, critic), cfg,
                       iteration_seed=1, opt_actor=opt_a, opt_critic=opt_c)
    assert stats["batches"] > 0
    saved = save_checkpoint(tmp_path / "opt.json", actor=actor, critic=critic,
                            optimizer_actor=opt_a, optimizer_critic=opt_c,
                            **_envless_fps(_v4_env), seed=24)
    # rebuild a checkpoint WITH optimiser state the way the trainer saves it
    from playground.rl.checkpoint import _opt_from_json, _opt_to_json
    body = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
    body["optimizer"] = {"actor": _opt_to_json(opt_a), "critic": _opt_to_json(opt_c)}
    body.pop("content_sha256", None)
    body["content_sha256"] = _hash({k: v for k, v in body.items()
                                    if k != "content_sha256"})
    Path(saved["path"]).write_text(json.dumps(body), encoding="utf-8")
    out = load_checkpoint(saved["path"], **_envless_fps(_v4_env))
    assert _restore_widths_from_loaded(out) == (hidden, hidden)
    opt_a2, opt_c2 = out["optimizer_actor"], out["optimizer_critic"]
    sd = opt_a2.state_dict()
    assert len(sd["state"]) == len(list(actor.parameters()))   # every tensor
    for idx, param in enumerate(actor.parameters()):
        assert idx in sd["state"], f"missing Adam state for param {idx}"
        assert tuple(sd["state"][idx]["exp_avg"].shape) == tuple(param.shape)
    # a fresh PPO step runs on the restored optimisers without error
    stats2 = ppo_update(out["actor"], out["critic"],
                        _tiny_buffer(out["actor"], out["critic"], seed=25),
                        cfg, iteration_seed=2, opt_actor=opt_a2,
                        opt_critic=opt_c2)
    assert stats2["batches"] > 0


# ------------------------------------------------------------ transfer forks
def _v3_parent(tmp_path, hidden, name="p"):
    env = LocoEnv(cfg={"rl": {"contract_version": 3}})
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, hidden=hidden, seed=26)
    critic = Critic(env.obs_dim, hidden=hidden, seed=27)
    with actor.torch.no_grad():
        actor.log_std.copy_(
            actor.torch.linspace(-1.2, 0.4, env.action_dim))
    ck = save_checkpoint(tmp_path / f"{name}.json", actor=actor, critic=critic,
                         env=env, seed=28)
    return ck["path"]


@pytest.mark.parametrize("hidden", ((64, 64), (128, 128), (256, 256)))
def test_surgery_preserves_parent_architecture(tmp_path, hidden):
    """Transfer-fork surgery keeps the PARENT'S widths in the child for every
    selectable architecture — the port never truncates/reshapes, the parent
    file is byte-identical, and the recorded sizes ride the fork record."""
    path = _v3_parent(tmp_path, hidden)
    sha0 = parent_sha256(path)
    fp = _live_fp(LocoEnv())
    child = surgery_transfer(path, fp, log_std_init=0.0)
    assert parent_sha256(path) == sha0              # parent untouched
    assert child["parent_hidden"] == list(hidden)   # recorded preservation
    assert _trunk_widths(child["actor"]) == hidden
    assert _critic_widths(child["critic"]) == hidden
    assert child["round_trip"]["passed"] is True
    assert child["round_trip"]["deterministic_appended_action_exact_zero"]
    assert child["actor"].action_dim == 15   # v4 action dim (the child speaks
                                             # the live contract)


def test_surgery_refuses_misrecorded_parent_architecture(tmp_path):
    """A parent LYING about its widths (recorded (64,64), tensors (128,128))
    is refused BEFORE the port runs — TransferError, nothing written."""
    path = _v3_parent(tmp_path, (128, 128), name="lying")
    _rehash(path, lambda b: b["meta"]["policy"].__setitem__("hidden", [64, 64]))
    sha0 = parent_sha256(path)          # post-rewrite baseline: the surgery
    fp = _live_fp(LocoEnv())            # itself must not touch the byte
    with pytest.raises(TransferError, match="does not match") as e:
        surgery_transfer(path, fp, log_std_init=0.0)
    assert "[64, 64]" in str(e.value)
    assert "[128, 128]" in str(e.value)
    assert parent_sha256(path) == sha0              # nothing touched


def test_surgery_sizeless_parent_derives_true_widths(tmp_path):
    """A pre-M7-0b parent (no recorded widths) hands the child its TRUE trunk
    widths, derived from the tensors — not an assumed (64, 64)."""
    path = _v3_parent(tmp_path, (128, 128), name="old128")
    _rehash(path, lambda b: b["meta"]["policy"].pop("hidden"))
    fp = _live_fp(LocoEnv())
    child = surgery_transfer(path, fp, log_std_init=0.0)
    assert child["parent_hidden"] == [128, 128]
    assert _trunk_widths(child["actor"]) == (128, 128)
    assert child["round_trip"]["passed"] is True


def test_surgery_sizeless_64x64_parent_unchanged(tmp_path):
    """The historical (64, 64) sizeless parent behaves exactly as the M7-0b
    gate proved: derived widths == default, round trip passes."""
    path = _v3_parent(tmp_path, (64, 64), name="old64")
    _rehash(path, lambda b: b["meta"]["policy"].pop("hidden"))
    fp = _live_fp(LocoEnv())
    child = surgery_transfer(path, fp, log_std_init=0.0)
    assert child["parent_hidden"] == [64, 64]
    assert child["round_trip"]["passed"] is True


# ------------------------------------------------------------- launcher law
TINY = dict(steps=512, rollout_steps=512)   # one PPO iteration; no stage flips


def _namespace(name, out, **kw):
    return argparse.Namespace(
        name=name, steps=kw.get("steps", TINY["steps"]),
        seed=kw.get("seed", 20261117), rollout_steps=kw.get("rollout_steps",
                                                            TINY["rollout_steps"]),
        ent_coef=None, lr=None, stage_budget=None, resume=kw.get("resume"),
        fork_from=kw.get("fork_from"), out=str(out),
        operation=kw.get("operation"), hidden=kw.get("hidden"),
        stages=None)


def test_launcher_fresh_selects_and_records_architecture(tmp_path):
    """--hidden selects the run's architecture END TO END: construction
    (config.json), the checkpoint's recorded truth, and the restored
    networks — with the untouched default still (64, 64)."""
    run128 = tmp_path / "r128"
    rl_ppo.cmd_train(_namespace("r128", run128, hidden="128,128"))
    cfgj = json.loads((run128 / "config.json").read_text(encoding="utf-8"))
    assert cfgj["ppo"]["hidden"] == [128, 128]
    meta = json.loads((run128 / "ckpt_final.json").read_text(
        encoding="utf-8"))["meta"]
    assert meta["policy"]["hidden"] == [128, 128]
    out = load_checkpoint(run128 / "ckpt_final.json", env=LocoEnv())
    assert _trunk_widths(out["actor"]) == (128, 128)
    assert _critic_widths(out["critic"]) == (128, 128)

    run_def = tmp_path / "rdef"
    rl_ppo.cmd_train(_namespace("rdef", run_def))
    cfgj = json.loads((run_def / "config.json").read_text(encoding="utf-8"))
    assert cfgj["ppo"]["hidden"] == [64, 64]
    meta = json.loads((run_def / "ckpt_final.json").read_text(
        encoding="utf-8"))["meta"]
    assert meta["policy"]["hidden"] == [64, 64]


def test_launcher_continue_fork_preserves_architecture(tmp_path):
    """continue_fork restores the parent's architecture FROM THE CHECKPOINT
    (never from the child run's config): a default-config child of a 128×128
    parent stays 128×128 in every checkpoint it writes."""
    parent = tmp_path / "p128"
    rl_ppo.cmd_train(_namespace("p128", parent, hidden="128,128"))
    pck = parent / "ckpt_final.json"
    child = tmp_path / "c128"
    rl_ppo.cmd_train(_namespace("c128", child, fork_from=str(pck),
                                operation="continue_fork"))
    meta = json.loads((child / "ckpt_final.json").read_text(
        encoding="utf-8"))["meta"]
    assert meta["policy"]["hidden"] == [128, 128]
    out = load_checkpoint(child / "ckpt_final.json", env=LocoEnv())
    assert _trunk_widths(out["actor"]) == (128, 128)


def test_launcher_fork_architecture_and_recorded_round_trip(tmp_path):
    """transfer_fork through the launcher preserves the parent widths under a
    non-default architecture (v3 128×128 parent → v4 child, end-to-end with
    op tags + the recorded round-trip proof; same-version transfers are
    refused by design — 'for equal contracts use continue-fork')."""
    parent = _v3_parent(tmp_path, (128, 128), name="p128")
    child = tmp_path / "tf128"
    rl_ppo.cmd_train(_namespace("tf128", child, fork_from=str(parent),
                                operation="transfer_fork"))
    cfgj = json.loads((child / "config.json").read_text(encoding="utf-8"))
    assert cfgj["run_parent"]["parent_hidden"] == [128, 128]
    meta = json.loads((child / "ckpt_final.json").read_text(
        encoding="utf-8"))["meta"]
    assert meta["operation"] == "transfer_fork"
    assert meta["run_parent"]["parent_hidden"] == [128, 128]
    assert meta["run_parent"]["round_trip"]["passed"] is True
