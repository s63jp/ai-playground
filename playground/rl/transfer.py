"""M7-0b — the transfer-fork surgery loader (the DOCUMENTED SECOND PATH).

The strict loader (`checkpoint.load_checkpoint`) refuses ANY contract
difference. Loading a brain whose contract is a DECLARED PREFIX SUBSET of
the live contract is a different, deliberate operation — transfer-fork — and
it lives here and ONLY here: `surgery_transfer`. The run launcher cannot
reach it without the `transfer_fork` operation tag.

Contract subset rule (verified BEFORE any port): the stored contract must be
a strict prefix of the live contract —
  - stored contract_version < live contract_version (no downgrade);
  - stored observation sections == the live sections' first k (names + slices);
  - stored obs/action dims within the live dims;
  - stored obs joint_order == the live obs joint_order prefix;
  - stored obs joint_ranges_rad == the prefix keys and values;
  - stored action `joints` == the live action list's exact prefix INCLUDING
    kp, kd AND range_rad (the M7-0b control-mechanics requirement — two
    brains with matching dims but different kd/ranges are not the same
    policy);
  - mapping string equal.

Port:
  - actor trunk input rows [0:obs_parent] copied, [obs_parent:] ZERO;
  - deeper trunk layers + biases copied;
  - mu_head weight/bias: first [act_parent] columns copied, appended ZERO;
  - log_std: per-dim [0:act_parent] copied (the parent's LEARNED sigma is
    preserved), the appended entry = log_std_init;
  - critic input rows same rule; everything else copied.

Round-trip proof runs inside the surgery (results returned in the record):
deterministic outputs on the shared slice equal to 1e-6, log_std exact-equal
on the shared slice, critic value equal to 1e-5.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "rl_checkpoint/1"


class TransferError(RuntimeError):
    """A transfer-fork surgery is not possible for this parent."""


def parent_sha256(path) -> str:
    """SHA-256 over the checkpoint file BYTES (the lineage pointer)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_verified_body(path) -> dict:
    """Integrity-verified body: format tag + content_sha256 (the same tamper
    check the strict loader uses), WITHOUT the fingerprint equality — the
    surgery's own prefix-subset check replaces it."""
    p = Path(path)
    if not p.exists():
        raise TransferError(f"no checkpoint at {p}")
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TransferError(f"checkpoint is not valid JSON: {e}") from e
    if body.get("format") != FORMAT:
        raise TransferError(f"unknown checkpoint format {body.get('format')!r}")
    calc = hashlib.sha256(json.dumps(
        {k: v for k, v in body.items() if k != "content_sha256"},
        sort_keys=True).encode()).hexdigest()
    if body.get("content_sha256") != calc:
        raise TransferError("checkpoint content hash mismatch (tampered or "
                            "hand-edited); refusing to transfer")
    return body


def _prefix_ranges(ranges: dict, order: list) -> dict:
    return {k: [float(x) for x in ranges[k]] for k in order if k in ranges}


def subset_check(stored_contract: dict, live_contract: dict) -> dict:
    """Declared prefix-subset verification. Returns the subset shapes;
    raises TransferError on any violation."""
    st_v = int(stored_contract.get("contract_version", 0) or 0)
    lv_v = int(live_contract.get("contract_version", 0) or 0)
    if not (st_v < lv_v):
        raise TransferError(
            f"stored contract_version {st_v} is not below the live {lv_v}: "
            "a transfer needs a declared prefix-subset parent (for equal "
            "contracts use continue-fork with the strict loader)")
    o_st = stored_contract.get("observation") or {}
    o_lv = live_contract.get("observation") or {}
    st_sec, lv_sec = o_st.get("sections") or [], o_lv.get("sections") or []
    if [(s.get("name"), s.get("slice")) for s in st_sec] != \
            [(s.get("name"), s.get("slice")) for s in lv_sec[:len(st_sec)]]:
        raise TransferError(
            "stored observation sections are not an exact prefix of the live "
            "sections (append-only discipline violated)")
    obs_parent, obs_child = int(stored_contract.get("obs_dim", 0)), \
        int(live_contract.get("obs_dim", 0))
    act_parent, act_child = int(stored_contract.get("action_dim", 0)), \
        int(live_contract.get("action_dim", 0))
    if not (0 < obs_parent <= obs_child and 0 < act_parent <= act_child):
        raise TransferError("stored dims must be within the live dims")
    if list(o_st.get("joint_order") or []) != list(o_lv.get("joint_order") or []):
        raise TransferError("stored obs joint order is not the live prefix")
    if (o_st.get("joint_ranges_rad") or {}) != \
            _prefix_ranges(o_lv.get("joint_ranges_rad") or {},
                           o_st.get("joint_order") or []):
        raise TransferError("stored obs joint ranges are not the live prefix "
                            "(the control-mechanics equality requirement)")
    a_st = stored_contract.get("action") or {}
    a_lv = live_contract.get("action") or {}
    if a_st.get("mapping") != a_lv.get("mapping"):
        raise TransferError("action mapping differs")
    n_st = o_st.get("normalisation") or {}
    n_lv = o_lv.get("normalisation") or {}
    shared_norm = {k: n_st[k] for k in n_st if k in n_lv}
    if shared_norm != {k: n_lv[k] for k in shared_norm}:
        raise TransferError("stored observation normalisation is not the "
                            "live prefix (scaling semantics must match)")
    st_j, lv_j = a_st.get("joints") or [], a_lv.get("joints") or []
    if len(st_j) != act_parent or len(lv_j) < act_parent:
        raise TransferError("action joint lists inconsistent with dims")
    for i in range(act_parent):
        sj, lj = st_j[i], lv_j[i]
        for field in ("name", "kp", "kd"):
            if sj.get(field) != lj.get(field):
                raise TransferError(
                    f"action joint {i} {field} differs ({sj.get(field)} != "
                    f"{lj.get(field)}): control mechanics must be identical")
        if [float(x) for x in (sj.get("range_rad") or [])] != \
                [float(x) for x in (lj.get("range_rad") or [])]:
            raise TransferError(
                f"action joint {i} range differs: control mechanics must be "
                "identical (M7-0b mandatory requirement)")
    return {"obs_parent": obs_parent, "obs_child": obs_child,
            "act_parent": act_parent, "act_child": act_child,
            "contract_pair": (st_v, lv_v)}


def _actor_hidden_widths(actor_sd: dict) -> tuple:
    """The TRUE hidden widths from a trunk state dict: in-dims[1:] of the
    trunk weights plus the final layer's out-dim (kept identical to
    checkpoint._actor_hidden — the built-child check compares these)."""
    items = sorted(((k, v) for k, v in actor_sd.items()
                    if k.startswith("trunk.") and k.endswith(".weight")),
                   key=lambda kv: kv[0])
    if not items:
        return ()
    widths = [int(np.asarray(v).shape[1]) for _, v in items][1:]
    widths.append(int(np.asarray(items[-1][1]).shape[0]))
    return tuple(widths)


def _torch():
    import torch
    return torch


def _tensor(torch, arr, ref):
    return torch.as_tensor(np.asarray(arr, dtype=np.float64), dtype=ref.dtype)


def surgery_transfer(path, env_fingerprints: dict, *, log_std_init: float = 0.0):
    """Transfer-fork surgery. Returns:

        {"actor", "critic", "subset", "parent_sha256", "parent_content_sha",
         "round_trip", "log_std_appended", "zero_init", "parent_hidden"}

    `env_fingerprints` = the live env's _contract_fingerprint() dict
    (observation/action layouts + dims + contract_version). Refuses anything
    that is not a declared prefix subset.
    """
    torch = _torch()
    body = _load_verified_body(path)
    meta = body["meta"]
    subset = subset_check(meta["contract"], env_fingerprints)
    hidden = meta.get("policy", {}).get("hidden") or None
    from .ppo import TanhGaussianActor, Critic
    kw = {"hidden": tuple(hidden)} if hidden else {}
    child_actor = TanhGaussianActor(subset["obs_child"], subset["act_child"],
                                    log_std_init=float(log_std_init), **kw)
    child_critic = Critic(subset["obs_child"], **kw)
    built_hidden = _actor_hidden_widths(child_actor.state_dict())
    if hidden and tuple(hidden) != built_hidden:
        raise TransferError(f"hidden widths {hidden} do not match the child "
                            f"architecture actually built {built_hidden}")
    zeros = []
    # mu_head.* and log_std are ported by the explicit blocks below ONLY:
    # the generic row/column rules describe the trunk's IO growth, not the
    # output head (column rule) or the per-dim sigma (copy/append law), and
    # the first cut also crashed indexing a 1-D log_std with c_shape[1].
    _explicit = ("mu_head.weight", "mu_head.bias", "log_std")
    with torch.no_grad():
        sd = child_actor.state_dict()      # detached CLONES (actor.state_dict copies)
        parent_sd = body["actor"]
        for key, c_param in sd.items():
            if key in _explicit:
                continue
            p_val = parent_sd.get(key)
            c_shape = tuple(c_param.shape)
            if p_val is None:
                c_param.zero_()
                zeros.append(f"actor.{key} (absent in parent)")
                continue
            p_arr = np.asarray(p_val, dtype=np.float64)
            if c_shape == p_arr.shape:
                c_param.copy_(_tensor(torch, p_arr, c_param))
            elif c_shape[1] >= p_arr.shape[1] and c_shape[0] == p_arr.shape[0]:
                c_param[:, :p_arr.shape[1]].copy_(_tensor(torch, p_arr[:, :], c_param))
                c_param[:, p_arr.shape[1]:].zero_()
                zeros.append(f"actor.{key} input cols [{p_arr.shape[1]}:{c_shape[1]}]")
            elif c_shape[0] >= p_arr.shape[0] and c_shape[1] == p_arr.shape[1]:
                c_param[:p_arr.shape[0]].copy_(_tensor(torch, p_arr, c_param))
                c_param[p_arr.shape[0]:].zero_()
                zeros.append(f"actor.{key} hidden rows [{p_arr.shape[0]}:{c_shape[0]}]")
            else:
                raise TransferError(
                    f"unexpected shape mismatch for actor.{key}: parent "
                    f"{p_arr.shape} vs child {c_shape}")
        cs_ap, cs_ac = subset["act_parent"], subset["act_child"]
        pw = np.asarray(parent_sd["mu_head.weight"], dtype=np.float64)
        pb = np.asarray(parent_sd["mu_head.bias"], dtype=np.float64)
        # mu_head is nn.Linear(hidden -> action_dim): the APPENDED action is
        # the weight's ROW dim ((15, 64) child vs (14, 64) parent) — port the
        # first act_parent rows, zero the appended rows (the first cut ported
        # columns, the input dim, and crashed the round-trip)
        sd["mu_head.weight"][:cs_ap].copy_(_tensor(torch, pw[:cs_ap],
                                                   sd["mu_head.weight"]))
        sd["mu_head.weight"][cs_ap:].zero_()
        zeros.append(f"actor.mu_head.weight rows [{cs_ap}:{cs_ac}]")
        sd["mu_head.bias"][:cs_ap].copy_(_tensor(torch, pb[:cs_ap],
                                                 sd["mu_head.bias"]))
        sd["mu_head.bias"][cs_ap:].zero_()
        zeros.append(f"actor.mu_head.bias entries [{cs_ap}:{cs_ac}]")
        p_ls = np.asarray(parent_sd["log_std"], dtype=np.float64)
        sd["log_std"][:cs_ap].copy_(_tensor(torch, p_ls, sd["log_std"]))
        sd["log_std"][cs_ap] = float(log_std_init)
        child_actor.load_state_dict(sd)

        pc_sd = body["critic"]   # critic.state_dict() returns LIVE references:
        # the in-place copies below ARE the port; no further load needed.
        for key, c_param in child_critic.state_dict().items():
            p_val = pc_sd.get(key)
            c_shape = tuple(c_param.shape)
            if p_val is None:
                c_param.zero_()
                zeros.append(f"critic.{key} (absent in parent)")
                continue
            p_arr = np.asarray(p_val, dtype=np.float64)
            if c_shape == p_arr.shape:
                c_param.copy_(_tensor(torch, p_arr, c_param))
            elif c_shape[1] >= p_arr.shape[1] and c_shape[0] == p_arr.shape[0]:
                c_param[:, :p_arr.shape[1]].copy_(
                    _tensor(torch, p_arr[:, :], c_param))
                c_param[:, p_arr.shape[1]:].zero_()
                zeros.append(f"critic.{key} input cols")
            else:
                raise TransferError(f"unexpected critic shape for {key}")
    pol = meta["policy"]
    parent_actor = TanhGaussianActor(subset["obs_parent"], subset["act_parent"],
                                     log_std_init=0.0,
                                     log_std_min=float(pol.get("log_std_min", -5.0)),
                                     log_std_max=float(pol.get("log_std_max", 2.0)))
    parent_actor.load_state_dict({k: torch.as_tensor(
        np.asarray(v, dtype=np.float32)) for k, v in body["actor"].items()})
    parent_critic = Critic(subset["obs_parent"])
    parent_critic.load_state_dict({k: torch.as_tensor(
        np.asarray(v, dtype=np.float32)) for k, v in body["critic"].items()})
    round_trip = _round_trip_proof(parent_actor, parent_critic, child_actor,
                                   child_critic, subset, torch)
    return {"actor": child_actor, "critic": child_critic, "subset": subset,
            "parent_sha256": parent_sha256(path),
            "parent_content_sha": body["content_sha256"],
            "parent_hidden": list(built_hidden),
            "round_trip": round_trip,
            "log_std_appended": float(log_std_init),
            "zero_init": zeros}


def _noguard():
    """A no-op context manager (torch.no_grad can't be nested oddly inside a
    `with torch.no_grad()` block that also closes over plain assignments)."""
    import contextlib
    return contextlib.nullcontext()


def _round_trip_proof(parent_actor, parent_critic, child_actor, child_critic,
                      subset, torch, n=16, seed=0) -> dict:
    """The fork-moment identity: deterministic outputs on the shared slice
    equal, log_std exact-equal, critic value equal."""
    state = torch.get_rng_state()      # the probe must not reseed the caller's
    torch.manual_seed(seed)            # training stream: restore the pre-probe
    rng = np.random.default_rng(seed)  # state on the way out
    obs = rng.uniform(-0.5, 0.5, (n, subset["obs_child"])).astype(np.float32)
    parent_obs = np.ascontiguousarray(obs[:, :subset["obs_parent"]])
    try:
        with torch.no_grad():
            # deterministic_action is a single-row API (reshapes to [1, -1]):
            # probe row by row, not a flattened batch
            pa = np.stack([parent_actor.deterministic_action(parent_obs[i])
                           for i in range(n)])
            ca = np.stack([child_actor.deterministic_action(obs[i])
                           for i in range(n)])
            pv = float(parent_critic.value(parent_obs[0]))
            cv = float(child_critic.value(obs[0]))
    finally:
        torch.set_rng_state(state)
    ok_act = float(np.max(np.abs(pa - ca[:, :subset["act_parent"]])))
    ok_val = abs(pv - cv)
    ls_p = np.asarray(parent_actor.log_std.detach().numpy(), dtype=np.float64)
    ls_c = np.asarray(child_actor.log_std.detach().numpy(), dtype=np.float64)
    ls_eq = bool(np.array_equal(ls_p, ls_c[:subset["act_parent"]]))
    return {
        "deterministic_shared_slice_max_abs_diff": ok_act,
        "deterministic_appended_action_exact_zero":
            bool(np.all(ca[:, subset["act_parent"]:] == 0.0)),
        "critic_value_abs_diff": ok_val,
        "log_std_shared_slice_exact_equal": ls_eq,
        "passed": bool(ok_act < 1e-6 and ok_val < 1e-5 and ls_eq),
    }
