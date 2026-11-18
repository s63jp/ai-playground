"""M7-0b tests — the three-operation loading law.

- the MANDATORY fingerprint extension: kd + joint ranges + contract version
  ride the strict diff (two brains with matching dims but different control
  mechanics can no longer load);
- the surgery loader: the v3→v4 declared prefix pair ports with the exact
  fork-moment identity (round-trip), preserves the parent's per-dim σ and
  appends the new dim at log_std_init;
- surgery refusals: non-prefix sections, non-upgrade versions, differing
  control mechanics on the SHARED slice, normalisation drift;
- the launcher law: op-tag refusals, the strict loader refusing a v3 parent
  under continue_fork (the load direction), the transfer path through the
  real launcher with tags + fresh counters, and the parent-immutability
  watchdog FIRING (the poisoned-parent case);
- the launcher-law companion rides this commit: the B1 inverse-semantics
  regression (rogue waist write overwritten by the action path) lives in
  tests/test_safety_b1.py beside the moved foot_yaw rogue test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from playground.rl.checkpoint import (CheckpointError, load_checkpoint,
                                      save_checkpoint)
from playground.rl.env import LocoEnv
from playground.rl.ppo import (PPOConfig, PPOError, TanhGaussianActor, Critic)
from playground.rl.transfer import (TransferError, parent_sha256,
                                    subset_check, surgery_transfer)

LIVE = {"contract_version": 4, "obs_dim": 68, "action_dim": 15}


def _live_fingerprint(env):
    return {
        "observation": env.observation_layout(),
        "action": env.action_layout(),
        "contract_version": env.contract_version,
        "obs_dim": int(env.obs_dim),
        "action_dim": int(env.action_dim),
    }


# ------------------------------------------------- mandatory diff extension
def _synthetic_fp(env, mutation):
    fp = _live_fingerprint(env)
    if mutation == "kd":
        fp["action"]["joints"][0]["kd"] = 2.5          # differs only in kd
    elif mutation == "range":
        fp["action"]["joints"][0]["range_rad"] = [-1.0, 1.0]
    elif mutation == "version":
        # the envless loader reads the contract version from the LAYOUTS'
        # layout_version stamps — a top-level mutation is invisible there
        fp["observation"]["layout_version"] = 99
    return fp


def test_fingerprint_diff_extends_to_kd_ranges_version(tmp_path):
    env = LocoEnv()
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=1)
    critic = Critic(env.obs_dim, seed=2)
    save_checkpoint(tmp_path / "c.json", actor=actor, critic=critic,
                    env=env, seed=3)
    ok = _live_fingerprint(env)
    load_checkpoint(tmp_path / "c.json", obs_fingerprint=ok["observation"],
                    action_fingerprint=ok["action"])        # identical loads
    for mutation in ("kd", "range", "version"):
        mutated = _synthetic_fp(env, mutation)
        with pytest.raises(CheckpointError) as e:
            load_checkpoint(tmp_path / "c.json",
                            obs_fingerprint=mutated["observation"],
                            action_fingerprint=mutated["action"])
        if mutation in ("kd", "range"):
            assert "differs" in str(e.value)


def test_range_and_kd_only_differences_carry_the_field_name():
    """The refusal must name WHICH control mechanic differs (the reviewable
    failure mode Jamie's requirement targets: two 'compatible-looking'
    brains with different mechanics)."""
    from playground.rl.checkpoint import _fingerprint_diff
    a = {"observation": {}, "action": {"mapping": "m", "joints": [
        {"name": "j", "range_rad": [-1.0, 1.0], "kp": 50.0, "kd": 5.0},
        {"name": "k", "range_rad": [-2.0, 2.0], "kp": 25.0, "kd": 1.0}]}}
    b_kd = {"observation": {}, "action": {"mapping": "m", "joints": [
        {"name": "j", "range_rad": [-1.0, 1.0], "kp": 50.0, "kd": 9.0},
        {"name": "k", "range_rad": [-2.0, 2.0], "kp": 25.0, "kd": 1.0}]}}
    b_range = {"observation": {}, "action": {"mapping": "m", "joints": [
        {"name": "j", "range_rad": [-1.0, 1.0], "kp": 50.0, "kd": 5.0},
        {"name": "k", "range_rad": [-2.1, 2.0], "kp": 25.0, "kd": 1.0}]}}
    assert any("kd" in d for d in _fingerprint_diff(a, b_kd))
    assert any("range" in d for d in _fingerprint_diff(a, b_range))
    assert _fingerprint_diff(a, {"observation": {}, "action": a["action"]}) == []


# ---------------------------------------------------------------- surgery
def _v3_parent(tmp_path):
    """A strict-saved v3-mode parent. The per-dim sigma is deliberately
    non-trivial: with log_std_init 0.0 every entry would be 0 and the
    D5-amended preserve-σ assertion would be vacuous."""
    env = LocoEnv(cfg={"rl": {"contract_version": 3}})
    actor = TanhGaussianActor(env.obs_dim, env.action_dim, seed=11)
    critic = Critic(env.obs_dim, seed=12)
    with actor.torch.no_grad():
        actor.log_std.copy_(
            actor.torch.linspace(-1.2, 0.4, env.action_dim))
    ck = save_checkpoint(tmp_path / "v3.json", actor=actor, critic=critic,
                         env=env, seed=13)
    return ck["path"], env, actor, critic


def test_surgery_round_trip_v3_to_v4(tmp_path):
    path, penv, pactor, pcritic = _v3_parent(tmp_path)
    torch = pactor.torch
    ls_parent = np.asarray(pactor.log_std.detach().numpy(), dtype=np.float64)
    assert not np.allclose(ls_parent, 0.0)      # the preserve-σ row is real
    child = surgery_transfer(path, _live_fingerprint(LocoEnv()),
                             log_std_init=0.0)
    assert child["round_trip"]["passed"] is True
    assert child["round_trip"]["deterministic_appended_action_exact_zero"]
    assert child["round_trip"]["log_std_shared_slice_exact_equal"]
    # D5-amended: the parent's per-dim sigma is PRESERVED on the shared slice
    assert np.allclose(np.asarray(child["actor"].log_std.detach().numpy())[:14],
                       ls_parent, atol=0)
    # the 15th (waist) sigma is the appended log_std_init
    assert float(child["actor"].log_std.detach().numpy()[14]) == 0.0
    # the child speaks the live contract
    assert child["actor"].obs_dim == 68 and child["actor"].action_dim == 15
    assert child["parent_sha256"] == parent_sha256(path)
    # the zero-init record names the honest port sites (first trunk layer's
    # input cols = obs growth; output-head rows = action growth)
    zv = child["zero_init"]
    assert any("trunk.0.weight" in s and "[43:68]" in s for s in zv)
    assert any("mu_head.weight" in s and "rows [14:15]" in s for s in zv)
    # the child architecture is the parent's (hidden widths validated)
    assert child["parent_hidden"] == [64, 64]


def test_surgery_refuses_non_prefix_and_downgrades(tmp_path):
    path, _, _, _ = _v3_parent(tmp_path)
    live = _live_fingerprint(LocoEnv())
    # non-prefix observation sections (a stored layout with foreign slices)
    bad = json.loads(json.dumps(live))
    bad["contract_version"] = 5
    bad["observation"] = json.loads(json.dumps(bad["observation"]))
    bad["observation"]["sections"] = [
        {"name": "weird", "slice": [0, 7]}] + bad["observation"]["sections"][1:]
    bad["obs_dim"] = 68
    with pytest.raises(TransferError):
        surgery_transfer(path, bad, log_std_init=0.0)
    # no downgrade: stored contract_version must be BELOW the live one
    with pytest.raises(TransferError):
        surgery_transfer(path, {"observation": live["observation"],
                                "action": live["action"],
                                "contract_version": 3,
                                "obs_dim": 43, "action_dim": 14},
                         log_std_init=0.0)
    # differing control mechanics ON THE SHARED SLICE = refused
    mech = json.loads(json.dumps(live))
    mech["action"] = json.loads(json.dumps(mech["action"]))
    mech["action"]["joints"][0]["kd"] = 42.0
    with pytest.raises(TransferError):
        surgery_transfer(path, mech, log_std_init=0.0)
    mech2 = json.loads(json.dumps(live))
    mech2["action"] = json.loads(json.dumps(mech2["action"]))
    mech2["action"]["joints"][0]["range_rad"] = [-1.0, 1.0]
    with pytest.raises(TransferError):
        surgery_transfer(path, mech2, log_std_init=0.0)
    # normalisation drift on the shared keys = refused
    norm = json.loads(json.dumps(live))
    norm["observation"] = json.loads(json.dumps(norm["observation"]))
    norm["observation"]["normalisation"]["joint_vel"] = "different scaling"
    with pytest.raises(TransferError):
        surgery_transfer(path, norm, log_std_init=0.0)


# ------------------------------------------------------------ launcher law
_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts.parent) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_scripts.parent))
import scripts.rl_ppo as rl_ppo  # noqa: E402  (ROOT already on sys.path)

TINY = dict(steps=512, rollout_steps=512)   # one PPO iteration; no stage flips


def _namespace(name, out, **kw):
    return argparse.Namespace(
        name=name, steps=kw.get("steps", TINY["steps"]),
        seed=kw.get("seed", 20261117), rollout_steps=kw.get("rollout_steps",
                                                            TINY["rollout_steps"]),
        ent_coef=None, lr=None, stage_budget=None, resume=kw.get("resume"),
        fork_from=kw.get("fork_from"), out=str(out),
        operation=kw.get("operation"))


def _tiny_run(ns):
    """cmd_train with a tiny budget; returns the raised exception or None."""
    try:
        rl_ppo.cmd_train(ns)
        return None
    except Exception as exc:      # the test asserts on the type + message
        return exc


def test_launcher_refusals(tmp_path):
    path, _, _, _ = _v3_parent(tmp_path)
    # a fork op requires --fork-from
    exc = _tiny_run(_namespace("r1", tmp_path / "r1",
                               operation="continue_fork"))
    assert isinstance(exc, PPOError) and "requires --fork-from" in str(exc)
    # an unknown operation name is refused
    exc = _tiny_run(_namespace("u", tmp_path / "u", operation="merge"))
    assert isinstance(exc, PPOError) and "unknown --operation" in str(exc)
    # fresh takes no parent pointers
    exc = _tiny_run(_namespace("a_run", tmp_path / "a_run", resume=str(path),
                               operation="fresh"))
    assert isinstance(exc, PPOError) and "fresh takes no" in str(exc)
    # a fork writing into the parent's run dir is refused (new lineage rule)
    parent_dir = Path(path).resolve().parent
    exc = _tiny_run(_namespace("p", parent_dir, fork_from=str(path),
                               operation="continue_fork"))
    assert isinstance(exc, PPOError) and "parent run dir" in str(exc)
    # a fork target that already holds a run is refused (lineage mixing)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "progress.jsonl").write_text("", encoding="utf-8")
    exc = _tiny_run(_namespace("q", busy, fork_from=str(path),
                               operation="continue_fork"))
    assert isinstance(exc, PPOError) and "already holds a run" in str(exc)
    # resume into a different dir IS a fork — refused with guidance
    exc = _tiny_run(_namespace("c_run", tmp_path / "c_run", resume=str(path),
                               operation="resume"))
    assert isinstance(exc, PPOError) and "continue_fork" in str(exc)


def test_fresh_tags_then_continue_fork_restores_and_records(tmp_path):
    # fresh parent: the real launcher flow, tiny single-iteration budget
    parent_run = tmp_path / "parent"
    rl_ppo.cmd_train(_namespace("parent", parent_run, operation=None))
    pck = parent_run / "ckpt_final.json"
    pmeta = json.loads(pck.read_text(encoding="utf-8"))["meta"]
    assert pmeta["operation"] == "fresh"          # op tags ride fresh too
    pres = json.loads((parent_run / "result.json").read_text(encoding="utf-8"))
    assert pres["operation"] == "fresh"
    parent_steps = int(pmeta["training_steps"])

    # continue_fork: exact restore into a NEW dir, parent byte-unchanged
    sha0 = parent_sha256(pck)
    child_run = tmp_path / "child"
    rl_ppo.cmd_train(_namespace("child", child_run, fork_from=str(pck),
                                operation="continue_fork"))
    assert parent_sha256(pck) == sha0, \
        "the parent brain must be byte-unchanged after a fork"
    cmeta = json.loads((child_run / "ckpt_final.json").read_text(
        encoding="utf-8"))["meta"]
    assert cmeta["operation"] == "continue_fork"
    rp = cmeta["run_parent"]
    assert rp["parent_sha256"] == sha0
    assert int(rp["parent_training_steps"]) == parent_steps
    # counters are PRESERVED (continued, not restarted)
    assert int(cmeta["training_steps"]) > parent_steps
    cfgj = json.loads((child_run / "config.json").read_text(encoding="utf-8"))
    assert cfgj["operation"] == "continue_fork"
    assert cfgj["run_parent"]["parent_sha256"] == sha0
    res = json.loads((child_run / "result.json").read_text(encoding="utf-8"))
    assert res["operation"] == "continue_fork"


def test_continue_fork_refuses_v3_parent_strict_loader(tmp_path):
    """Both directions of the loading law, proven at the launcher: the strict
    path (continue_fork) may NOT quietly succeed where surgery would — a v3
    parent under a live v4 env is a fingerprint mismatch, refused loudly
    (dims + contract_version), and nothing is written."""
    path, _, _, _ = _v3_parent(tmp_path)
    child_run = tmp_path / "cf"
    exc = _tiny_run(_namespace("cf", child_run, fork_from=str(path),
                               operation="continue_fork"))
    assert isinstance(exc, CheckpointError) and "incompatible" in str(exc)
    assert not (child_run / "ckpt_final.json").exists()


def test_transfer_fork_ports_v3_parent_to_v4_child(tmp_path):
    """The surgery path through the REAL launcher: transfer_fork is the ONLY
    route to surgery; the child starts fresh (counters + optimisers + stage
    0) with the ported weights and the recorded round-trip proof."""
    path, _, _, _ = _v3_parent(tmp_path)
    sha0 = parent_sha256(path)
    child_run = tmp_path / "tf"
    rl_ppo.cmd_train(_namespace("tf", child_run, fork_from=str(path),
                                operation="transfer_fork"))
    assert parent_sha256(path) == sha0
    meta = json.loads((child_run / "ckpt_final.json").read_text(
        encoding="utf-8"))["meta"]
    assert meta["operation"] == "transfer_fork"
    rp = meta["run_parent"]
    assert (rp["obs_parent"], rp["act_parent"]) == (43, 14)
    assert list(rp["parent_contract_pair"]) == [3, 4]
    assert rp["round_trip"]["passed"] is True
    assert rp["log_std_appended"] == 0.0
    # the child lineage starts at standing with FRESH counters (one rollout)
    assert meta["curriculum"]["index"] == 0
    assert 500 <= int(meta["training_steps"]) < 560
    res = json.loads((child_run / "result.json").read_text(encoding="utf-8"))
    assert res["operation"] == "transfer_fork"
    cfgj = json.loads((child_run / "config.json").read_text(encoding="utf-8"))
    assert cfgj["operation"] == "transfer_fork"
    assert cfgj["run_parent"]["parent_contract_pair"] == [3, 4]


def test_transfer_fork_parent_watchdog_fires(tmp_path, monkeypatch):
    """The parent-lineage watchdog: if the parent file changes between the
    fork start and a checkpoint write, the write is REFUSED before any
    poisoned checkpoint lands."""
    path, _, _, _ = _v3_parent(tmp_path)
    real = rl_ppo.parent_sha256
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        return real(p) if calls["n"] == 1 else ("f" * 64)

    monkeypatch.setattr(rl_ppo, "parent_sha256", flaky)
    child_run = tmp_path / "wd"
    exc = _tiny_run(_namespace("wd", child_run, fork_from=str(path),
                               operation="transfer_fork"))
    assert isinstance(exc, PPOError) and "PARENT LINEAGE CHANGED" in str(exc)
    # the poisoned fork aborted at the checkpoint write: nothing durable
    assert not (child_run / "ckpt_final.json").exists()
    assert not (child_run / "result.json").exists()
    assert (child_run / "progress.jsonl").exists()   # training had begun
    assert calls["n"] >= 2                           # watchdog read the file
