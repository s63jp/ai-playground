"""PPO trainer / evaluator CLI for the locomotion environment (M6-B stage B3).

Drives ONLY the authoritative stack: playground/rl/env.py (LocoEnv),
playground/rl/rewards.py, playground/rl/curriculum.py, playground/rl/ppo.py
(trainer) and playground/rl/checkpoint.py. There is no second training engine.

Commands
--------
  train   Run a seeded PPO training run (fresh or resumed from a checkpoint).
          Progress -> <out>/<name>/progress.jsonl; periodic + final checkpoints,
          run config echo and a final result summary.
  eval    Evaluate a checkpoint policy against the zero-action control and the
          scripted gait benchmark under MATCHED held-out conditions.
  baselines  The matched B3 baselines: fresh untrained policy, zero action,
          gait benchmark (same seeds, horizon, stage phase).

Run from the repository root, e.g.:
  .venv/Scripts/python.exe scripts/rl_ppo.py train --name b3_sustained_run1
  .venv/Scripts/python.exe scripts/rl_ppo.py eval --checkpoint <ckpt.json>
  .venv/Scripts/python.exe scripts/rl_ppo.py baselines --stage sustained
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from playground.config import load_config  # noqa: E402
from playground.rl import eval as rl_eval  # noqa: E402
from playground.rl.checkpoint import (CheckpointError, load_checkpoint,  # noqa: E402
                                      save_checkpoint)
from playground.rl.curriculum import DEFAULT_STAGES, Curriculum  # noqa: E402
from playground.rl.env import LocoEnv  # noqa: E402
from playground.rl.ppo import (PPOConfig, PPOError, PPOTrainer,  # noqa: E402
                               TanhGaussianActor, Critic)
from playground.rl.transfer import (TransferError, parent_sha256,  # noqa: E402
                                    surgery_transfer)

OPERATIONS = ("fresh", "resume", "continue_fork", "transfer_fork")
FORK_OPS = ("continue_fork", "transfer_fork")


def _rl_cfg(cfg):
    return dict(cfg.get("rl") or {})


def _contract_fingerprint_of(env):
    return {
        "observation": env.observation_layout(),
        "action": env.action_layout(),
        "contract_version": env.contract_version,
        "obs_dim": int(env.obs_dim),
        "action_dim": int(env.action_dim),
    }


def _run_dir(cfg, name):
    out = Path(cfg.get("rl", {}).get("ppo_run", {}).get("out_root")
               or (ROOT / "experiments" / "rl"))
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_operation(a) -> tuple:
    """M7-0b loading law: resolve the operation tag + parent, enforce the
    refusals BEFORE any training. Returns (operation, fork_from | None)."""
    op = (a.operation or "").strip() or ("resume" if a.resume else "fresh")
    if op not in OPERATIONS:
        raise PPOError(f"unknown --operation {op!r}; allowed {OPERATIONS}")
    if op == "fresh":
        if a.resume or a.fork_from:
            raise PPOError("--operation fresh takes no --resume/--fork-from")
        return op, None
    if a.resume and a.fork_from:
        raise PPOError("give only one of --resume (same-run continuation) or "
                       "--fork-from (fork)")
    if op == "resume":
        if a.fork_from:
            raise PPOError("--operation resume continues the SAME run with "
                           "--resume; branching to a new dir is continue_fork")
        if not a.resume:
            raise PPOError("--operation resume requires --resume")
        return op, None
    # fork operations
    if not a.fork_from:
        raise PPOError(f"--operation {op} requires --fork-from <checkpoint>")
    return op, Path(a.fork_from)


def _parse_stage_budgets(s: str | None):
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def _rl_cfg_with_hidden_override(rl: dict, hidden_arg) -> dict:
    """Per-run architecture override (scripts/rl_ppo.py train --hidden).

    `--hidden 256,256` selects the actor/critic trunk widths for THIS run
    without editing `config/default.yaml` (whose 64x64 default stays the B3/B4
    baseline). The override is spliced into the rl dict BEFORE PPOConfig
    validation, so the resulting widths pass through exactly the same
    from_rl_cfg rules as a YAML-set value (1..8 layers of positive ints,
    PPOError otherwise), and the selected widths are recorded wherever the
    config is echoed (config.json carries ppo_cfg.as_dict()).
    """
    if hidden_arg is None or str(hidden_arg).strip() == "":
        return rl
    try:
        widths = [int(h) for h in str(hidden_arg).split(",")]
    except ValueError as e:
        raise PPOError(f"--hidden must be comma-separated layer widths "
                       f"(e.g. 256,256): {e}") from e
    return {**rl, "ppo": {**(rl.get("ppo") or {}), "hidden": widths}}


def _make_env_and_curriculum(cfg):
    env = LocoEnv(cfg={"agent_body": {"type": "humanoid"},
                       **(_rl_cfg(cfg) or {})})
    cur = Curriculum(env)            # binds + applies the standing stage
    return env, cur


# --------------------------------------------------------------------- train
def cmd_train(a):
    cfg = load_config(None, overrides={})
    rl = _rl_cfg(cfg)
    run = dict(rl.get("ppo_run") or {})
    total_steps = a.steps if a.steps is not None else int(run.get("total_steps", 2_000_000))
    seed = a.seed if a.seed is not None else int(run.get("seed", 20261117))
    cp_freq = int(run.get("checkpoint_freq", 200_000))
    stage_budgets = _parse_stage_budgets(a.stage_budget) if a.stage_budget \
        else run.get("stage_budgets")
    ppo_cfg = PPOConfig.from_rl_cfg(
        _rl_cfg_with_hidden_override(rl, getattr(a, "hidden", None)))
    if a.ent_coef is not None:
        ppo_cfg.ent_coef = float(a.ent_coef)
    if a.lr is not None:
        ppo_cfg.lr = float(a.lr)
    if a.rollout_steps is not None:
        ppo_cfg.rollout_steps = int(a.rollout_steps)

    torch = __import__("torch")
    torch.set_num_threads(1)     # tiny MLPs: thread pool overhead dominates
    torch.manual_seed(seed)

    env, cur = _make_env_and_curriculum(cfg)
    op, fork_from = _resolve_operation(a)
    if getattr(a, "stages", None):
        from playground.rl.curriculum import (STAGE_LADDERS,
                                              m7_s1_launch_block_reason)
        if a.stages not in STAGE_LADDERS:
            raise PPOError(f"unknown --stages {a.stages!r}: allowed "
                           f"{sorted(STAGE_LADDERS)}")
        if a.stages == "m7_s1":
            reason = m7_s1_launch_block_reason()
            if reason:
                raise PPOError(f"M7-1a limb ladder blocked: {reason}")
        run_ladder = a.stages
        cur = Curriculum(env, stages=STAGE_LADDERS[run_ladder])
        cur.select(0)
    else:
        run_ladder = "default"
    out = Path(a.out) if a.out else _run_dir(cfg, a.name)
    out.mkdir(parents=True, exist_ok=True)
    fork_record = None
    if op == "resume":
        # continues the SAME run: the out dir must be the parent's
        parent_dir = Path(a.resume).resolve().parent
        if out.resolve() != parent_dir:
            raise PPOError(
                "--operation resume continues the SAME run: the out directory "
                f"({out}) is not the parent run dir ({parent_dir}). "
                "Branching to a new dir is --operation continue_fork.")
    if op in FORK_OPS:
        parent_dir = fork_from.resolve().parent
        if out.resolve() == parent_dir:
            raise PPOError(
                f"--operation {op} must not write into the parent run dir "
                f"({parent_dir}): a fork is a NEW lineage. (same-run "
                "continuation is --operation resume)")
        if (out / "progress.jsonl").exists():
            raise PPOError(
                f"the fork target {out} already holds a run (progress.jsonl "
                "exists): forking would mix lineages — refuse")
        parent_sha = parent_sha256(fork_from)
        if op == "transfer_fork":
            tk = surgery_transfer(fork_from, _contract_fingerprint_of(env),
                                  log_std_init=float(ppo_cfg.log_std_init))
            if not tk["round_trip"]["passed"]:
                raise PPOError(
                    "transfer surgery round-trip proof FAILED "
                    f"({tk['round_trip']}) — refusing to fork")
            actor, critic = tk["actor"], tk["critic"]
            fork_record = {
                "parent_checkpoint": str(fork_from),
                "parent_sha256": parent_sha,
                "parent_contract_pair": list(tk["subset"]["contract_pair"]),
                "obs_parent": tk["subset"]["obs_parent"],
                "act_parent": tk["subset"]["act_parent"],
                "parent_hidden": tk["parent_hidden"],
                "zero_init": tk["zero_init"],
                "log_std_appended": tk["log_std_appended"],
                "round_trip": tk["round_trip"],
            }
            print(f"[transfer_fork] parent {fork_from} sha {parent_sha[:16]}… "
                  f"pair {fork_record['parent_contract_pair']} "
                  f"(obs {fork_record['obs_parent']}->{env.obs_dim}, "
                  f"act {fork_record['act_parent']}->{env.action_dim}); "
                  "round-trip passed", flush=True)
            trainer_state = None
            meta_steps = 0
            if total_steps <= meta_steps:
                total_steps = meta_steps + ppo_cfg.rollout_steps
            cur = Curriculum(env)   # the child lineage starts at stage 0
        else:
            ck = load_checkpoint(fork_from, env=env)
            actor, critic = ck["actor"], ck["critic"]
            trainer_state = dict(ck["rng"])
            from playground.rl.checkpoint import apply_rng
            apply_rng(ck["rng"])
            meta_steps = int(trainer_state.get("total_valid_steps", 0))
            if total_steps <= meta_steps:
                total_steps = meta_steps + ppo_cfg.rollout_steps
            fork_record = {"parent_checkpoint": str(fork_from),
                           "parent_sha256": parent_sha,
                           "parent_training_steps":
                               ck["meta"].get("training_steps")}
            print(f"[continue_fork] parent {fork_from} sha {parent_sha[:16]}… "
                  f"restored at step {meta_steps}", flush=True)
    elif a.resume:
        ck = load_checkpoint(a.resume, env=env)
        actor, critic = ck["actor"], ck["critic"]
        trainer_state = dict(ck["rng"])
        from playground.rl.checkpoint import apply_rng
        apply_rng(ck["rng"])
        meta_steps = int(trainer_state.get("total_valid_steps", 0))
        print(f"resumed from {a.resume}: step {meta_steps}", flush=True)
        if total_steps <= meta_steps:
            total_steps = meta_steps + ppo_cfg.rollout_steps
    else:
        actor = TanhGaussianActor(env.obs_dim, env.action_dim,
                                  hidden=ppo_cfg.hidden,
                                  log_std_init=ppo_cfg.log_std_init, seed=seed)
        critic = Critic(env.obs_dim, hidden=ppo_cfg.hidden, seed=seed)
        trainer_state = None
    run_meta = {"operation": op}
    if fork_record is not None:
        run_meta["run_parent"] = fork_record

    trainer = PPOTrainer(env, actor, critic, ppo_cfg, seed=seed,
                         total_steps=total_steps, stage_budgets=stage_budgets,
                         curriculum=cur)
    if op in ("resume", "continue_fork") and (a.resume or op == "continue_fork"):
        # restore optimiser + rng + counters + CURRICULUM STAGE for exact
        # continuation (the loader verifies the contract fingerprint;
        # curriculum select() re-applies the saved stage phase, which is the
        # documented non-cumulative rewrite — a semantically exact no-op)
        trainer.opt_a = ck["optimizer_actor"]
        trainer.opt_c = ck["optimizer_critic"]
        trainer.load_rng_state(trainer_state)
        saved_cur = ck["meta"].get("curriculum") or {}
        if "index" in saved_cur:
            cur.select(int(saved_cur["index"]))
            print(f"  curriculum restored to stage '{cur.stage.id}' "
                  f"(index {cur.index})", flush=True)
        else:
            print("  WARNING: checkpoint carries no curriculum record; "
                  "training resumes from stage 0 (standing)", flush=True)

    progress_path = out / "progress.jsonl"
    pf = progress_path.open("a", encoding="utf-8")
    last_ckpt = {"step": trainer.total_valid_steps}
    _parent_sha = fork_record.get("parent_sha256") if fork_record else None

    def _checkpoint(tag):
        extra = {"stage": cur.stage.id,
                 "operation": op,
                 "run_total_steps": total_steps,
                 "run": a.name,
                 "curriculum": {"index": cur.index,
                                "stage_id": cur.stage.id},
                 "training_steps": trainer.total_valid_steps}
        if fork_record is not None:
            extra["run_parent"] = fork_record
        if _parent_sha is not None:
            # the parent-lineage watchdog: the recorded parent brain must be
            # byte-unchanged whenever a fork writes a checkpoint
            now_sha = parent_sha256(fork_from)
            if now_sha != _parent_sha:
                raise PPOError(
                    "PARENT LINEAGE CHANGED DURING FORK: "
                    f"{fork_from} sha {now_sha[:16]}… != recorded "
                    f"{_parent_sha[:16]}… — aborting")
        cp = save_checkpoint(out / tag, actor=actor, critic=critic,
                             optimizer_actor=trainer.opt_a,
                             optimizer_critic=trainer.opt_c, env=env,
                             cfg_dict=ppo_cfg.as_dict(), seed=seed,
                             trainer_state=trainer.rng_state(),
                             extra_meta=extra)
        print(f"  checkpoint: {cp['path']} ({cp['bytes']} bytes)", flush=True)
        return cp

    def on_iteration(rec):
        pf.write(json.dumps(rec) + "\n")
        pf.flush()
        if rec["total_valid_steps"] - last_ckpt["step"] >= cp_freq:
            last_ckpt["step"] = rec["total_valid_steps"]
            _checkpoint(f"ckpt_{rec['total_valid_steps']:09d}.json")

    trainer.on_iteration = on_iteration
    t0 = time.perf_counter()
    print(f"[train] {a.name}: operation={op} total_steps={total_steps} seed={seed} "
          f"rollout={ppo_cfg.rollout_steps} ent_coef={ppo_cfg.ent_coef} "
          f"stages={stage_budgets}", flush=True)
    summary = trainer.train()
    wall = time.perf_counter() - t0
    final = _checkpoint("ckpt_final.json")
    (out / "config.json").write_text(json.dumps({
        "ppo": ppo_cfg.as_dict(), "total_steps": total_steps, "seed": seed,
        "stage_budgets": stage_budgets,
        "stage_ladder": run_ladder,
        "operation": op,
        "fork_from": (str(fork_from) if fork_from else None),
        "run_parent": fork_record,
        "env_contract_version": env.contract_version,
        "obs_dim": env.obs_dim, "action_dim": env.action_dim,
        "started_from_checkpoint": a.resume or None,
    }, indent=2), encoding="utf-8")
    (out / "result.json").write_text(json.dumps({
        **summary, "wall_s": round(wall, 1),
        "operation": op,
        "stage_ladder": run_ladder,
        "fork_from": (str(fork_from) if fork_from else None),
        "final_checkpoint": final["path"],
        "faulted_windows_total": trainer.faulted_windows_total,
        "rollout_rejections": trainer.rollout_rejections,
    }, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "wall_s": round(wall, 1)}, indent=2), flush=True)


# ----------------------------------------------------------------------- eval
def cmd_eval(a):
    policy_eval_checkpoint(a)


def policy_eval_checkpoint(a):
    cfg = load_config(None, overrides={})
    rl = _rl_cfg(cfg)
    torch = __import__("torch")
    torch.set_num_threads(1)
    env = LocoEnv(cfg={"agent_body": {"type": "humanoid"}, **rl})
    ck = load_checkpoint(a.checkpoint, env=env)
    actor, meta = ck["actor"], ck["meta"]
    episodes = a.episodes or rl_eval.DEFAULT_EPISODES
    seed_base = a.seed_base or rl_eval.EVAL_SEED_BASE
    max_steps = int(rl.get("max_episode_steps", 400))
    phase = rl_eval._stage_phase_dict(Curriculum(), a.stage)
    out_dir = Path(a.out) if a.out else Path(a.checkpoint).parent
    results = {"checkpoint": a.checkpoint, "stage": a.stage,
               "episodes": episodes, "seed_base": seed_base,
               "max_steps": max_steps, "meta_stage": meta.get("stage"),
               "training_steps": meta.get("training_steps")}
    tables = []
    t_policy = rl_eval.eval_policy(actor, env, episodes=episodes, seed_base=seed_base,
                                   phase=phase, max_steps=max_steps)
    results["policy"] = t_policy["summary"]
    t_zero = rl_eval.eval_zero(env, episodes=episodes, seed_base=seed_base,
                               phase=phase, max_steps=max_steps)
    results["zero_action"] = t_zero["summary"]
    gait_heading = a.gait_heading if a.gait_heading is not None else 0.0
    t_gait = rl_eval.eval_gait(env, episodes=1, seed_base=seed_base, phase=phase,
                               heading=gait_heading, max_steps=max_steps)
    results["gait_benchmark"] = t_gait["summary"]
    results["gait_heading_deg"] = gait_heading

    def row(label, s):
        d = s.get("axis_displacement_m") or {}
        ns = s.get("net_displacement_m") or {}
        up = s.get("upright_fraction") or {}
        return {"policy": label,
                "axis_disp_mean_m": round(d.get("mean", 0.0), 4),
                "axis_disp_std_m": round(d.get("std", 0.0), 4),
                "net_disp_mean_m": round(ns.get("mean", 0.0), 4),
                "upright_mean": round(up.get("mean", 0.0), 4),
                "falls_total": s.get("falls_total"),
                "faulted_total": s.get("faulted_windows_total"),
                "completed": s.get("completed_truncated")}
    tables = [row("trained_policy", results["policy"]),
              row("zero_action", results["zero_action"]),
              row(f"gait(h={gait_heading:.0f})", results["gait_benchmark"])]
    print(json.dumps({"evaluation": tables, "summary_full": {
        k: results[k] for k in ("policy", "zero_action", "gait_benchmark")}},
        indent=2), flush=True)
    outp = out_dir / (f"eval_{a.stage}_s{seed_base}_{episodes}ep.json")
    outp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"written: {outp}", flush=True)


# ------------------------------------------------------------------ baselines
def cmd_eval_s1(a):
    """Held-out S1 capability (mastery) eval for one individual limb — the
    honest gate: a limb is reported mastered ONLY on measured tracking against
    the recorded bands, never on step count."""
    from playground.rl.s1_eval import (EVAL_EPISODES, EVAL_SEED_BASE,
                                        evaluate_s1_limb)
    verdict = evaluate_s1_limb(
        a.checkpoint, a.stage,
        episodes=(a.episodes or EVAL_EPISODES),
        seed_base=(a.seed_base if a.seed_base is not None else EVAL_SEED_BASE),
        write_to=a.out)
    print(json.dumps(verdict, indent=2))
    if a.require_mastered and not verdict["mastered"]:
        raise PPOError(f"{a.stage} is NOT mastered "
                       f"({sum(1 for r in verdict['joints'].values() if not r['pass'])} "
                       f"joint band(s) failing)")


def cmd_baselines(a):
    cfg = load_config(None, overrides={})
    rl = _rl_cfg(cfg)
    torch = __import__("torch")
    torch.set_num_threads(1)
    env = LocoEnv(cfg={"agent_body": {"type": "humanoid"}, **rl})
    episodes = a.episodes or rl_eval.DEFAULT_EPISODES
    seed_base = a.seed_base or rl_eval.EVAL_SEED_BASE
    max_steps = int(rl.get("max_episode_steps", 400))
    phase = rl_eval._stage_phase_dict(Curriculum(), a.stage)
    results = {"stage": a.stage, "episodes": episodes, "seed_base": seed_base,
               "max_steps": max_steps}
    print(f"[baselines] stage={a.stage} episodes={episodes} "
          f"seeds={seed_base}+0..{episodes - 1}", flush=True)
    t_fresh = rl_eval.eval_fresh_actor(env, episodes=episodes, seed_base=seed_base,
                                       phase=phase, max_steps=max_steps)
    results["fresh_untrained"] = t_fresh["summary"]
    t_zero = rl_eval.eval_zero(env, episodes=episodes, seed_base=seed_base,
                               phase=phase, max_steps=max_steps)
    results["zero_action"] = t_zero["summary"]
    for heading in (0.0, 90.0):
        t_g = rl_eval.eval_gait(env, episodes=1, seed_base=seed_base, phase=phase,
                                heading=heading, max_steps=max_steps)
        results[f"gait_heading_{int(heading)}"] = t_g["summary"]
    for k, v in results.items():
        if k in ("stage", "episodes", "seed_base", "max_steps"):
            continue
        d = (v.get("axis_displacement_m") or {})
        print(f"  {k:22s} axis_disp mean {d.get('mean', 0.0):+.4f} "
              f"std {d.get('std', 0.0):.4f} | falls {v.get('falls_total')} "
              f"faults {v.get('faulted_windows_total')} completed {v.get('completed_truncated')}",
              flush=True)
    outp = Path(a.out) if a.out else _run_dir(cfg, "baselines")
    outp.mkdir(parents=True, exist_ok=True)
    dest = outp / f"baselines_{a.stage}_s{seed_base}_{episodes}ep.json"
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"written: {dest}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--rollout-steps", type=int, default=None)
    t.add_argument("--ent-coef", type=float, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--stage-budget", default=None,
                   help="e.g. standing=400000,controlled=500000")
    t.add_argument("--resume", default=None)
    t.add_argument("--fork-from", dest="fork_from", default=None,
                   help="the parent checkpoint for continue_fork/transfer_fork")
    t.add_argument("--operation", default=None,
                   choices=list(OPERATIONS),
                   help="the loading operation; default: resume when --resume "
                        "is given, fresh otherwise (the tag is recorded in "
                        "every checkpoint + the run metadata)")
    t.add_argument("--out", default=None)
    t.add_argument("--hidden", default=None,
                   help="comma-separated actor/critic trunk widths for this "
                        "run (e.g. 256,256); default: rl.ppo.hidden from the "
                        "YAML config (64,64)")
    t.add_argument("--stages", default=None,
                   help="the stage ladder for this run: a ladder name "
                        "exported by playground.rl.curriculum (e.g. "
                        "m7_s1); default: the B2 standing ladder")
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--stage", default="sustained")
    e.add_argument("--episodes", type=int, default=None)
    e.add_argument("--seed-base", type=int, default=None)
    e.add_argument("--gait-heading", type=float, default=0.0)
    e.add_argument("--out", default=None)
    es = sub.add_parser("eval-s1")
    es.add_argument("--checkpoint", required=True)
    es.add_argument("--stage", required=True,
                    help="S1 limb stage id (e.g. m7_s1_head, m7_s1_left_leg)")
    es.add_argument("--episodes", type=int, default=None)
    es.add_argument("--seed-base", type=int, default=None)
    es.add_argument("--out", default=None, help="write the verdict JSON here")
    es.add_argument("--require-mastered", action="store_true",
                    help="exit non-zero unless the limb meets every gate band")
    b = sub.add_parser("baselines")
    b.add_argument("--stage", default="sustained")
    b.add_argument("--episodes", type=int, default=None)
    b.add_argument("--seed-base", type=int, default=None)
    b.add_argument("--out", default=None)
    args = p.parse_args(argv)
    try:
        if args.cmd == "train":
            cmd_train(args)
        elif args.cmd == "eval":
            cmd_eval(args)
        elif args.cmd == "eval-s1":
            cmd_eval_s1(args)
        elif args.cmd == "baselines":
            cmd_baselines(args)
    except CheckpointError as exc:
        print(f"CHECKPOINT ERROR: {exc}", file=sys.stderr)
        return 2
    except TransferError as exc:
        print(f"TRANSFER ERROR: {exc}", file=sys.stderr)
        return 4
    except PPOError as exc:
        print(f"PPO ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
