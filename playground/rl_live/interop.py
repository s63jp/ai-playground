"""The single seam between the live-training layer and the B3 trainer modules.

`playground.rl.ppo` and `playground.rl.checkpoint` are the ONE training
algorithm and THE checkpoint format. This module never reimplements either:
it only wires them up for a live session. When those modules are not on the
tree (the live-training branch is based on B2 until B3 lands), every entry
point raises `InteropUnavailable` with a clear message and the session
surfaces it rather than pretending.

Import policy: the imports happen inside the functions, so importing THIS
module stays cheap and the absence of the B3 modules is a runtime condition,
not a module-level crash.
"""

from __future__ import annotations


class InteropUnavailable(RuntimeError):
    """B3 trainer modules are not on this tree (they land with the B3 commit)."""


def trainer_available() -> bool:
    try:
        import playground.rl.ppo  # noqa: F401
        import playground.rl.checkpoint  # noqa: F401
        return True
    except Exception:
        return False


def build_session_trainer(env, curriculum, *, name: str, seed: int,
                          total_steps: int, stage_budgets: dict | None,
                          ppo_overrides: dict | None, on_iteration, should_stop):
    """Construct the real PPO trainer exactly as scripts/rl_ppo.py does.

    Returns {"trainer", "actor", "critic", "ppo_cfg_dict"}.
    """
    try:
        from playground.rl.ppo import PPOConfig, PPOTrainer, TanhGaussianActor, Critic
    except ImportError as e:
        raise InteropUnavailable(
            "the B3 trainer modules (playground/rl/ppo.py, checkpoint.py) are "
            "not present on this tree — live training starts once the B3 "
            "commit lands; nothing else about the live layer depends on them"
        ) from e
    from playground.config import load_config
    torch = __import__("torch")
    torch.set_num_threads(1)   # same convention as scripts/rl_ppo.py

    cfg = load_config(None, overrides={})
    rl = dict(cfg.get("rl") or {})
    ppo_cfg = PPOConfig.from_rl_cfg(rl)
    overrides = dict(ppo_overrides or {})
    for key in ("lr", "ent_coef", "clip_eps", "gamma", "gae_lambda",
                "rollout_steps", "epochs", "minibatch", "max_grad_norm"):
        if key in overrides and overrides[key] is not None:
            setattr(ppo_cfg, key, type(getattr(ppo_cfg, key))(overrides[key]))

    if "hidden" in overrides and overrides["hidden"] is not None:
        try:
            hidden = tuple(int(width) for width in overrides["hidden"])
        except (TypeError, ValueError) as exc:
            raise ValueError("hidden must be a sequence of positive layer widths") from exc

        if not hidden or len(hidden) > 8 or any(width < 1 for width in hidden):
            raise ValueError("hidden must contain 1-8 positive layer widths")

        ppo_cfg.hidden = hidden
    torch.manual_seed(int(seed))

    env_cfg = {"agent_body": {"type": "humanoid"}, **rl}
    # env was already constructed with the same config by the caller; the
    # dims below are read through it (tap passthrough keeps this honest).
    actor = TanhGaussianActor(env.obs_dim, env.action_dim,
                              hidden=ppo_cfg.hidden,
                              log_std_init=ppo_cfg.log_std_init, seed=int(seed))
    critic = Critic(env.obs_dim, hidden=ppo_cfg.hidden, seed=int(seed))
    trainer = PPOTrainer(env, actor, critic, ppo_cfg, seed=int(seed),
                         total_steps=int(total_steps),
                         stage_budgets=stage_budgets, curriculum=curriculum,
                         on_iteration=on_iteration, should_stop=should_stop)
    return {"trainer": trainer, "actor": actor, "critic": critic,
            "ppo_cfg_dict": ppo_cfg.as_dict()}


def build_session_env(env_cfg_overrides: dict | None = None, *,
                      limb_stage_id: str | None = None):
    """Construct LocoEnv + bound Curriculum.

    Default: the B3/B4 walking ladder (standing stage applied), unchanged.
    When `limb_stage_id` is given (an M7-1a S1 limb), the curriculum is the
    2-stage live limb ladder — the standing base then that one limb — so the
    live session trains an individual limb exactly as the CLI `--stages m7_s1`
    lineage would, but focused on the selected limb. The launch gate is
    enforced by the caller (the session) before this runs.
    """
    from playground.config import load_config
    from playground.rl.curriculum import Curriculum
    from playground.rl.env import LocoEnv
    cfg = load_config(None, overrides={})
    rl = dict(cfg.get("rl") or {})
    rl.update(dict(env_cfg_overrides or {}))
    env = LocoEnv(cfg={"agent_body": {"type": "humanoid"}, **rl})
    if limb_stage_id:
        from playground.rl.curriculum import m7_s1_limb_ladder
        cur = Curriculum(env, stages=m7_s1_limb_ladder(limb_stage_id))
        cur.select(0)         # standing base first
    else:
        cur = Curriculum(env)  # binds + applies the standing stage
    return env, cur


def list_experiments(root) -> dict:
    """List experiment run dirs under `root` with their checkpoints, S1 eval
    verdicts, and objective/limb metadata. Read-only; shared by both the
    dashboard mount and the standalone sidecar so the two never drift."""
    import json
    from pathlib import Path
    root = Path(root)
    items = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            cfg = None
            cfgp = d / "run_config.json"
            if cfgp.exists():
                try:
                    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
                except Exception:
                    cfg = None
            items.append({"name": d.name,
                          "checkpoints": sorted(p.name for p in d.glob("ckpt_*.json")),
                          "evals": sorted(p.name for p in d.glob("s1_eval_*.json")),
                          "objective": (cfg or {}).get("objective"),
                          "limb_stage_id": (cfg or {}).get("limb_stage_id"),
                          "s1": bool((cfg or {}).get("s1"))})
    return {"experiments": items, "root": str(root)}


def run_s1_evaluation(root, *, experiment, checkpoint="ckpt_final.json",
                      stage_id=None, episodes=12) -> dict:
    """Run the S1 capability (mastery) eval on a saved checkpoint under
    `root/experiment`. Returns {"ok": True, "verdict": ...} or
    {"ok": False, "error": {"code", "message"}}. The caller guards the
    training-running state and maps errors to HTTP status."""
    import json
    from pathlib import Path
    root = Path(root)
    try:
        episodes = max(1, min(24, int(episodes)))
    except (TypeError, ValueError):
        episodes = 12
    name, ckpt = str(experiment or ""), str(checkpoint or "")
    if not name or any(b in name for b in ("/", "\\", "..")) or \
            any(b in ckpt for b in ("/", "\\", "..")):
        return {"ok": False, "error": {"code": "invalid_config",
                "message": "experiment and checkpoint must be plain names"}}
    run_dir = root / name
    ckpt_path = run_dir / ckpt
    if not ckpt_path.exists():
        return {"ok": False, "error": {"code": "not_found",
                "message": f"checkpoint {ckpt_path} does not exist"}}
    if not stage_id:
        cfgp = run_dir / "run_config.json"
        if cfgp.exists():
            try:
                stage_id = json.loads(cfgp.read_text(encoding="utf-8")).get("limb_stage_id")
            except Exception:
                stage_id = None
    if not stage_id:
        return {"ok": False, "error": {"code": "invalid_config",
                "message": "stage_id required (no limb_stage_id in run_config)"}}
    try:
        from playground.rl.s1_eval import evaluate_s1_limb
        verdict = evaluate_s1_limb(str(ckpt_path), stage_id, episodes=episodes,
                                   write_to=run_dir / f"s1_eval_{stage_id}.json")
    except Exception as exc:
        return {"ok": False, "error": {"code": "eval_failed", "message": str(exc)}}
    return {"ok": True, "verdict": verdict}


def save_session_checkpoint(path, *, trainer, curriculum, seed: int,
                            name: str, total_steps: int,
                            stage_budgets: dict | None,
                            ppo_cfg_dict: dict | None) -> dict:
    """Write a training checkpoint through THE checkpoint module (B3)."""
    try:
        from playground.rl.checkpoint import save_checkpoint
    except ImportError as e:
        raise InteropUnavailable(
            "playground/rl/checkpoint.py is not present on this tree — "
            "checkpoint saving becomes available when the B3 commit lands"
        ) from e
    return save_checkpoint(
        path,
        actor=trainer.actor, critic=trainer.critic,
        optimizer_actor=trainer.opt_a, optimizer_critic=trainer.opt_c,
        env=trainer.env, cfg_dict=ppo_cfg_dict, seed=int(seed),
        trainer_state=trainer.rng_state(),
        extra_meta={"stage": (curriculum.stage.id if curriculum is not None else None),
                    "run_total_steps": int(total_steps),
                    "run": str(name),
                    "curriculum": ({"index": curriculum.index,
                                    "stage_id": curriculum.stage.id}
                                   if curriculum is not None else {}),
                    "training_steps": trainer.total_valid_steps,
                    "origin": "rl_live sidecar"})
