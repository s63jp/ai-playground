"""RL training checkpoints (M6-B stage B3).

NOT the `.aipg` portable-brain format (B5): these are training checkpoints in
the repository's own convention — JSON documents with a SHA-256 content hash
(same pattern as `playground/ml/registry.py`), stored beside the run's
records. A checkpoint preserves the actual trained parameters: actor weights,
critic weights, optimiser states, RNG states, the training step, the seed and
the FULL environment/policy contract fingerprints so an incompatible load is
rejected loudly instead of silently mis-firing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

FORMAT = "rl_checkpoint/1"


class CheckpointError(RuntimeError):
    """A checkpoint is unreadable or incompatible with the live contract."""


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _tensor_to_list(t) -> list:
    return np.asarray(t, dtype=np.float64).tolist()


def _opt_to_json(opt) -> dict:
    """Flatten a torch optimizer into JSON (Adam state + param groups)."""
    sd = opt.state_dict()
    out = {"param_groups": sd["param_groups"], "state": {}}
    for idx, st in sd["state"].items():
        d = {}
        for k, v in st.items():
            if hasattr(v, "tolist"):
                d[k] = _tensor_to_list(v)
            elif hasattr(v, "item"):
                d[k] = float(v.item())
            else:
                d[k] = v
        out["state"][int(idx)] = d
    return out


def _opt_from_json(owner_params, data):
    """Rebuild a torch optimizer (Adam) with its state. `owner_params` is the
    ordered param list the state indices refer to."""
    torch = _torch()
    st = {"state": {int(k): {kk: (torch.tensor(vv) if isinstance(vv, list) else vv)
                             for kk, vv in v.items()}
                    for k, v in data["state"].items()},
          "param_groups": data["param_groups"]}
    opt = torch.optim.Adam(owner_params, lr=data["param_groups"][0]["lr"])
    opt.load_state_dict(st)
    return opt


def _torch():
    import torch
    return torch


def _contract_fingerprint(env) -> dict:
    return {
        "observation": env.observation_layout(),
        "action": env.action_layout(),
        "contract_version": env.contract_version,
        "obs_dim": int(env.obs_dim),
        "action_dim": int(env.action_dim),
    }


def _torch_rng_to_json() -> str:
    torch = _torch()
    return base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode()


def _torch_rng_from_json(s: str) -> None:
    torch = _torch()
    raw = base64.b64decode(s)
    t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
    torch.set_rng_state(t)


def save_checkpoint(path, *, actor, critic, optimizer_actor=None,
                    optimizer_critic=None, env=None, obs_fingerprint=None,
                    action_fingerprint=None, cfg_dict=None, seed=0,
                    trainer_state=None, extra_meta=None) -> dict:
    """Write a checkpoint. Returns {"path", "content_sha256", "bytes"}.

    `env` (preferred) supplies the live contract fingerprints; for envless
    saves (unit tests) pass `obs_fingerprint`/`action_fingerprint` dicts.
    `trainer_state` carries step/episode/rng counters (PPOTrainer.rng_state).
    """
    torch = _torch()
    if env is not None:
        contract = _contract_fingerprint(env)
    else:
        if obs_fingerprint is None or action_fingerprint is None:
            raise CheckpointError("save_checkpoint needs env or both fingerprints")
        contract = {"observation": obs_fingerprint, "action": action_fingerprint,
                    "obs_dim": int(obs_fingerprint.get("dim", -1)),
                    "action_dim": int(action_fingerprint.get("action_dim", -1))}
    body = {
        "format": FORMAT,
        "meta": {
            "created_at": time.time(),
            "seed": int(seed),
            "contract": contract,
            "policy": {
                "class": "TanhGaussianActor",
                "obs_dim": int(actor.obs_dim),
                "action_dim": int(actor.action_dim),
                "log_std_min": float(actor.log_std_min),
                "log_std_max": float(actor.log_std_max),
                "log_std": _tensor_to_list(actor.log_std.detach().numpy()),
                "deterministic_eval": "a = tanh(mu(s))",
            },
            "ppo_config": dict(cfg_dict or {}),
            "trained_through": ("playground/rl/env.py LocoEnv (rl_env contract) "
                                "+ playground/rl/rewards.py + playground/rl/curriculum.py"),
            "aipg_note": ("training checkpoint only; the .aipg portable brain "
                          "format is milestone B5"),
            **(extra_meta or {}),
        },
        "actor": {k: _tensor_to_list(v) for k, v in actor.state_dict().items()},
        "critic": {k: _tensor_to_list(v) for k, v in critic.state_dict().items()},
        "optimizer": {"actor": (_opt_to_json(optimizer_actor) if optimizer_actor else None),
                      "critic": (_opt_to_json(optimizer_critic) if optimizer_critic else None)},
        "rng": {"torch_rng": _torch_rng_to_json(),
                **{k: v for k, v in (trainer_state or {}).items()
                   if not k.startswith("torch_rng")}},
    }
    body["content_sha256"] = _hash({k: v for k, v in body.items()
                                    if k != "content_sha256"})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body), encoding="utf-8")
    return {"path": str(p), "content_sha256": body["content_sha256"],
            "bytes": p.stat().st_size}


def _fingerprint_diff(stored: dict, live: dict) -> list:
    diffs = []
    if int(stored.get("obs_dim", -1)) != int(live.get("obs_dim", -2)):
        diffs.append(f"obs_dim {stored.get('obs_dim')} != {live.get('obs_dim')}")
    if int(stored.get("action_dim", -1)) != int(live.get("action_dim", -2)):
        diffs.append(f"action_dim {stored.get('action_dim')} != {live.get('action_dim')}")
    o_st = stored.get("observation") or {}
    o_lv = live.get("observation") or {}
    if [s.get("slice") for s in o_st.get("sections", [])] != \
            [s.get("slice") for s in o_lv.get("sections", [])]:
        diffs.append("observation sections differ")
    if list(o_st.get("joint_order", [])) != list(o_lv.get("joint_order", [])):
        diffs.append("joint order differs")
    a_st = stored.get("action") or {}
    a_lv = live.get("action") or {}
    if list(a_st.get("mapping", "")) != list(a_lv.get("mapping", "")) or \
            [j.get("name") for j in a_st.get("joints", [])] != \
            [j.get("name") for j in a_lv.get("joints", [])] or \
            [j.get("kp") for j in a_st.get("joints", [])] != \
            [j.get("kp") for j in a_lv.get("joints", [])]:
        diffs.append("action mapping/joints/gains differ")
    return diffs


def load_checkpoint(path, obs_fingerprint=None, action_fingerprint=None,
                    env=None):
    """Load and verify a checkpoint. Returns a dict with rebuilt actor/critic,
    optimisers, rng restore data and meta. Raises CheckpointError on:

    - missing/corrupt file, wrong format tag, hash mismatch (tamper check),
    - contract incompatibility: the checkpoint's observation/action contract
      fingerprint must match the live env (or the fingerprints passed in) —
      obs/action dims, section slices, joint order, mapping and PD gains.
    """
    torch = _torch()
    p = Path(path)
    if not p.exists():
        raise CheckpointError(f"no checkpoint at {p}")
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CheckpointError(f"checkpoint is not valid JSON: {e}") from e
    if body.get("format") != FORMAT:
        raise CheckpointError(f"unknown checkpoint format {body.get('format')!r}")
    if body.get("content_sha256") != _hash({k: v for k, v in body.items()
                                            if k != "content_sha256"}):
        raise CheckpointError("checkpoint content hash mismatch (tampered or "
                              "hand-edited); refusing to load")
    meta = body["meta"]
    if env is not None:
        live = _contract_fingerprint(env)
    else:
        if obs_fingerprint is None or action_fingerprint is None:
            raise CheckpointError("load_checkpoint needs env or both fingerprints")
        live = {"observation": obs_fingerprint, "action": action_fingerprint,
                "obs_dim": int(obs_fingerprint.get("dim", -1)),
                "action_dim": int(action_fingerprint.get("action_dim", -1))}
    diffs = _fingerprint_diff(meta["contract"], live)
    if diffs:
        raise CheckpointError(
            "checkpoint is incompatible with the live contract: "
            + "; ".join(diffs))
    # rebuild the policy exactly as saved
    pol = meta["policy"]
    if pol.get("class") != "TanhGaussianActor":
        raise CheckpointError(f"unknown actor class {pol.get('class')!r}")
    from .ppo import TanhGaussianActor, Critic  # local import: no cycle
    actor = TanhGaussianActor(int(pol["obs_dim"]), int(pol["action_dim"]),
                              log_std_min=float(pol["log_std_min"]),
                              log_std_max=float(pol["log_std_max"]))
    actor.load_state_dict({k: torch.tensor(v, dtype=torch.float32)
                           if not isinstance(v, torch.Tensor) else v
                           for k, v in body["actor"].items()})
    critic = Critic(int(pol["obs_dim"]))
    critic.load_state_dict({k: torch.tensor(v, dtype=torch.float32)
                            for k, v in body["critic"].items()})
    out = {"path": str(p), "meta": meta, "actor": actor, "critic": critic,
           "content_sha256": body["content_sha256"]}
    if meta.get("policy", {}).get("log_std") is not None:
        # log_std is part of the actor state dict already; verify equality
        saved = np.asarray(meta["policy"]["log_std"], dtype=np.float64)
        live_ls = actor.log_std.detach().numpy().astype(np.float64)
        if not np.allclose(saved, live_ls, atol=1e-12):
            raise CheckpointError("actor log_std inconsistent with saved state")
    if body.get("optimizer") and body["optimizer"].get("actor"):
        opt_a = _opt_from_json(actor.parameters(), body["optimizer"]["actor"])
        opt_c = _opt_from_json(critic.parameters(), body["optimizer"]["critic"])
        out["optimizer_actor"], out["optimizer_critic"] = opt_a, opt_c
    out["rng"] = body.get("rng") or {}
    return out


def apply_rng(rng_state: dict) -> None:
    """Restore the torch RNG (and return any trainer counters) from a
    checkpoint's rng section."""
    if rng_state.get("torch_rng"):
        _torch_rng_from_json(rng_state["torch_rng"])
