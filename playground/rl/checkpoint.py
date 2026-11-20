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


def _actor_hidden(actor) -> list:
    """Hidden widths derived from the trunk weight shapes (works for every
    checkpoint regardless of meta age; additive field since M7-0b).

    Derivation: trunk layer i has weight (out_i, in_i) with in_0 = obs_dim
    and in_i = out_{i-1}; the hidden sequence is therefore in_1..in_{n-1}
    plus out_{n-1}. e.g. hidden=(64, 64) has weights (64, obs_dim) and
    (64, 64) -> widths (64, 64). (The first cut, widths[1:], returned only
    the interior in-dims — (64,) for (64, 64) — which would have made the
    surgery rebuild a one-hidden-layer child for two-hidden-layer parents;
    caught by the M7-0b tests before any fork ran.)"""
    # sort by LAYER INDEX (the trunk holds up to eight Linear layers under
    # PPOConfig's 1..8 rule, so state-dict indexes reach 14 and a bare key
    # sort would put "trunk.10.weight" before "trunk.2.weight" — recording
    # a wrong architecture for 6+ layer trunks)
    items = sorted(((k, v) for k, v in actor.state_dict().items()
                    if k.startswith("trunk.") and k.endswith(".weight")),
                   key=lambda kv: int(kv[0].split(".")[1]))
    if not items:
        return []
    widths = [int(np.asarray(v).shape[1]) for _, v in items][1:]
    widths.append(int(np.asarray(items[-1][1]).shape[0]))
    return widths


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
        for src in (obs_fingerprint, action_fingerprint):
            if src.get("layout_version") is not None:
                # the layouts carry the contract version (observation_layout
                # and action_layout both stamp it) — keep it in the saved
                # contract so later loads can verify it
                contract["contract_version"] = int(src["layout_version"])
                break
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
                # M7-0b (additive): the hidden widths, so surgery validates
                # architecture, not just dims. Older checkpoints lack the key
                # (the surgery derives widths from their weight shapes).
                "hidden": _actor_hidden(actor),
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
    """M7-0b (MANDATORY per Jamie's directive 2026-11-18): the compatibility
    check now covers the CONTROL MECHANICS, not just the shapes — kd and the
    joint ranges ride the diff alongside the dims/order/slices/mapping/kp
    checks. Two brains whose observation and action dimensions match but
    whose gains or ranges differ are NOT compatible and cannot load."""
    diffs = []
    # dims and version: compared ONLY when both sides carry them (a hand-
    # built diff-test dict without dims is not a dims disagreement). The
    # STRICT loader always fills them (env or the fingerprints' dims stamps),
    # so this leniency never reaches a real load.
    if stored.get("obs_dim") is not None and live.get("obs_dim") is not None \
            and int(stored.get("obs_dim")) != int(live.get("obs_dim")):
        diffs.append(f"obs_dim {stored.get('obs_dim')} != {live.get('obs_dim')}")
    if stored.get("action_dim") is not None and \
            live.get("action_dim") is not None and \
            int(stored.get("action_dim")) != int(live.get("action_dim")):
        diffs.append(f"action_dim {stored.get('action_dim')} != "
                     f"{live.get('action_dim')}")
    if stored.get("contract_version") is not None and \
            live.get("contract_version") is not None and \
            int(stored.get("contract_version")) != \
            int(live.get("contract_version")):
        diffs.append(f"contract_version {stored.get('contract_version')} != "
                     f"{live.get('contract_version')}")
    o_st = stored.get("observation") or {}
    o_lv = live.get("observation") or {}
    if [s.get("slice") for s in o_st.get("sections", [])] != \
            [s.get("slice") for s in o_lv.get("sections", [])]:
        diffs.append("observation sections differ")
    if list(o_st.get("joint_order", [])) != list(o_lv.get("joint_order", [])):
        diffs.append("joint order differs")
    st_ranges = o_st.get("joint_ranges_rad") or {}
    lv_ranges = o_lv.get("joint_ranges_rad") or {}
    if {k: [float(x) for x in v] for k, v in st_ranges.items() if k in lv_ranges} != \
            {k: [float(x) for x in v] for k, v in lv_ranges.items() if k in st_ranges}:
        diffs.append("observation joint ranges differ")
    a_st = stored.get("action") or {}
    a_lv = live.get("action") or {}
    if list(a_st.get("mapping", "")) != list(a_lv.get("mapping", "")):
        diffs.append("action mapping differs")
    st_j = list(a_st.get("joints") or [])
    lv_j = list(a_lv.get("joints") or [])
    if [j.get("name") for j in st_j] != [j.get("name") for j in lv_j]:
        diffs.append("action joint order differs")
    for i, (sj, lj) in enumerate(zip(st_j, lv_j)):
        if sj.get("kp") != lj.get("kp"):
            diffs.append(f"action joint {i} kp differs "
                         f"({sj.get('kp')} != {lj.get('kp')})")
        if sj.get("kd") != lj.get("kd"):
            diffs.append(f"action joint {i} kd differs "
                         f"({sj.get('kd')} != {lj.get('kd')})")
        if [float(x) for x in (sj.get("range_rad") or [])] != \
                [float(x) for x in (lj.get("range_rad") or [])]:
            diffs.append(f"action joint {i} range differs")
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
        for src in (obs_fingerprint, action_fingerprint):
            # the layouts stamp layout_version == contract_version; carrying
            # it here is what makes the M7-0b version comparison meaningful
            # on the envless loader path (hand-built dicts without it fall
            # back to the both-sides-present skip in _fingerprint_diff)
            if src.get("layout_version") is not None:
                live["contract_version"] = int(src["layout_version"])
                break
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
    # Rebuild actor and critic at the CHECKPOINT'S OWN architecture: the
    # saved `policy.hidden` widths are authoritative when present (M7-0b
    # additive field; every save records the trunk-derived widths), or fall
    # back to the historical default when absent. Every pre-M7-0b checkpoint
    # in this repository is (64, 64) — and should that ever be wrong, the
    # strict tensor load below refuses the shape mismatch loudly, so the
    # fallback can never silently substitute an architecture.
    hidden = pol.get("hidden", None)
    if hidden is not None:
        try:
            hidden_tuple = tuple(int(h) for h in hidden)
        except (TypeError, ValueError) as e:
            raise CheckpointError(
                f"checkpoint policy.hidden is not a list of integer widths: "
                f"{e}") from e
        if not hidden_tuple or any(h < 1 for h in hidden_tuple):
            raise CheckpointError(
                f"checkpoint policy.hidden must be positive layer widths "
                f"(got {pol['hidden']!r})")
    else:
        hidden_tuple = None
    actor_kwargs = {
        "obs_dim": int(pol["obs_dim"]),
        "action_dim": int(pol["action_dim"]),
        "log_std_min": float(pol["log_std_min"]),
        "log_std_max": float(pol["log_std_max"])
    }
    if hidden_tuple is not None:
        actor_kwargs["hidden"] = hidden_tuple
    critic_kwargs = {"obs_dim": int(pol["obs_dim"])}
    if hidden_tuple is not None:
        critic_kwargs["hidden"] = hidden_tuple
    hidden_label = (str(list(hidden_tuple)) if hidden_tuple is not None
                    else "(64, 64) default")
    try:
        actor = TanhGaussianActor(**actor_kwargs)
        actor.load_state_dict(
            {k: torch.tensor(v, dtype=torch.float32)
             if not isinstance(v, torch.Tensor) else v
             for k, v in body["actor"].items()})
    except (RuntimeError, KeyError) as e:
        raise CheckpointError(
            f"checkpoint actor weights do not fit the recorded architecture "
            f"(hidden={hidden_label}): {e}") from e
    try:
        critic = Critic(**critic_kwargs)
        critic.load_state_dict(
            {k: torch.tensor(v, dtype=torch.float32)
             for k, v in body["critic"].items()})
    except (RuntimeError, KeyError) as e:
        raise CheckpointError(
            f"checkpoint critic weights do not fit the recorded architecture "
            f"(hidden={hidden_label}): {e}") from e
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
