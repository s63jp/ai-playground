"""Neural networks: a small, declarative model spec and a PyTorch trainer.

A model is described by JSON, so agents, the CLI and the dashboard can all create one:

    {"name": "throw_peak", "inputs": ["force", "elevation_deg", "agent_vz", "object_mass"],
     "target": "max_height", "hidden": [64, 64], "activation": "relu", "dropout": 0.0}

Training standardises inputs and target (statistics are stored with the model), splits the data
into train / validation / test by a fixed seed, early-stops on validation loss and reports every
metric on the untouched test split. Two baselines are always computed on the same test split so a
network's numbers can be judged: predicting the training mean, and ordinary least squares.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any

from . import MLError, torch_status
from .features import INPUT_COLUMNS, TARGET_COLUMNS

ACTIVATIONS = ("relu", "tanh", "gelu", "silu", "sigmoid")
LIMITS = {"max_layers": 8, "max_width": 1024, "max_epochs": 5000, "min_rows": 20}


def validate_spec(spec: dict) -> dict:
    """Check a model spec and fill in defaults. Raises MLError with a precise reason."""
    if not isinstance(spec, dict):
        raise MLError("invalid_spec", "A model spec must be an object.")
    s = {"name": spec.get("name"), "inputs": spec.get("inputs"), "target": spec.get("target"),
         "hidden": spec.get("hidden", [64, 64]), "activation": spec.get("activation", "relu"),
         "dropout": spec.get("dropout", 0.0), "description": spec.get("description", "")}
    if not isinstance(s["name"], str) or not s["name"].replace("_", "").replace("-", "").isalnum() \
            or not 1 <= len(s["name"]) <= 40:
        raise MLError("invalid_spec", "name must be 1-40 characters of letters, digits, '_' or '-'.")
    ins = s["inputs"]
    if not isinstance(ins, list) or not ins or len(set(ins)) != len(ins):
        raise MLError("invalid_spec", "inputs must be a non-empty list of distinct column names.",
                      available_inputs=sorted(INPUT_COLUMNS))
    bad = [c for c in ins if c not in INPUT_COLUMNS]
    if bad:
        raise MLError("invalid_spec", f"Unknown input column(s) {bad}. Inputs must be known at the moment of release.",
                      available_inputs=sorted(INPUT_COLUMNS))
    if s["target"] not in TARGET_COLUMNS:
        raise MLError("invalid_spec", f"target must be one of {sorted(TARGET_COLUMNS)}.",
                      available_targets=sorted(TARGET_COLUMNS))
    h = s["hidden"]
    if not isinstance(h, list) or len(h) > LIMITS["max_layers"] or \
            not all(isinstance(x, int) and 1 <= x <= LIMITS["max_width"] for x in h):
        raise MLError("invalid_spec", f"hidden must be a list of at most {LIMITS['max_layers']} layer widths, "
                                      f"each 1-{LIMITS['max_width']} ([] means a linear model).")
    if s["activation"] not in ACTIVATIONS:
        raise MLError("invalid_spec", f"activation must be one of {list(ACTIVATIONS)}.")
    if not isinstance(s["dropout"], (int, float)) or not 0 <= s["dropout"] < 0.9:
        raise MLError("invalid_spec", "dropout must be a number in [0, 0.9).")
    return s


def validate_train(cfg: dict) -> dict:
    t = {"epochs": cfg.get("epochs", 300), "lr": cfg.get("lr", 3e-3), "batch_size": cfg.get("batch_size", 64),
         "weight_decay": cfg.get("weight_decay", 0.0), "patience": cfg.get("patience", 40),
         "seed": cfg.get("seed", 0), "device": cfg.get("device", "auto"),
         "val_fraction": cfg.get("val_fraction", 0.15), "test_fraction": cfg.get("test_fraction", 0.15)}
    if not isinstance(t["epochs"], int) or not 1 <= t["epochs"] <= LIMITS["max_epochs"]:
        raise MLError("invalid_spec", f"epochs must be an integer 1-{LIMITS['max_epochs']}.")
    if not isinstance(t["lr"], (int, float)) or not 1e-6 <= t["lr"] <= 1.0:
        raise MLError("invalid_spec", "lr must be between 1e-6 and 1.")
    if not isinstance(t["batch_size"], int) or not 1 <= t["batch_size"] <= 65536:
        raise MLError("invalid_spec", "batch_size must be an integer 1-65536.")
    if t["device"] not in ("auto", "cpu", "xpu", "cuda"):
        raise MLError("invalid_spec", "device must be auto, cpu, xpu or cuda.")
    if not (0 < t["val_fraction"] < 0.5 and 0 < t["test_fraction"] < 0.5):
        raise MLError("invalid_spec", "val_fraction and test_fraction must each be in (0, 0.5).")
    return t


def _require_torch():
    st = torch_status()
    if not st["available"]:
        raise MLError("torch_missing", st["reason"])
    import torch
    return torch


def pick_device(pref: str = "auto") -> str:
    torch = _require_torch()
    if pref == "auto":
        # Small MLPs are faster on CPU than through a GPU launch queue; use the GPU only when asked.
        return "cpu"
    if pref == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise MLError("device_unavailable", "device 'xpu' was requested but no Intel XPU is available.")
    if pref == "cuda" and not torch.cuda.is_available():
        raise MLError("device_unavailable", "device 'cuda' was requested but CUDA is not available.")
    return pref


def build_module(spec: dict, n_in: int):
    torch = _require_torch()
    nn = torch.nn
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU, "sigmoid": nn.Sigmoid}[spec["activation"]]
    layers: list = []
    width = n_in
    for h in spec["hidden"]:
        layers += [nn.Linear(width, h), act()]
        if spec["dropout"]:
            layers.append(nn.Dropout(spec["dropout"]))
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _matrix(rows: list[dict], cols: list[str]) -> list[list[float]]:
    return [[float(r[c]) for c in cols] for r in rows]


def usable_rows(rows: list[dict], spec: dict) -> list[dict]:
    need = spec["inputs"] + [spec["target"]]
    return [r for r in rows if all(isinstance(r.get(c), (int, float)) and math.isfinite(r[c]) for c in need)]


def split(rows: list[dict], seed: int, val_fraction: float, test_fraction: float):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_test = max(1, int(round(len(rows) * test_fraction)))
    n_val = max(1, int(round(len(rows) * val_fraction)))
    test = [rows[i] for i in idx[:n_test]]
    val = [rows[i] for i in idx[n_test:n_test + n_val]]
    train = [rows[i] for i in idx[n_test + n_val:]]
    return train, val, test


def _stats(m: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(m)
    k = len(m[0])
    mean = [sum(r[j] for r in m) / n for j in range(k)]
    std = [math.sqrt(sum((r[j] - mean[j]) ** 2 for r in m) / n) or 1.0 for j in range(k)]
    return mean, std


def regression_metrics(y: list[float], p: list[float]) -> dict:
    n = len(y)
    err = [pi - yi for yi, pi in zip(y, p)]
    mae = sum(abs(e) for e in err) / n
    rmse = math.sqrt(sum(e * e for e in err) / n)
    ybar = sum(y) / n
    ss_tot = sum((yi - ybar) ** 2 for yi in y) or 1e-12
    r2 = 1 - sum(e * e for e in err) / ss_tot
    return {"n": n, "mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6),
            "max_abs_error": round(max(abs(e) for e in err), 6)}


def least_squares(xtr: list[list[float]], ytr: list[float]):
    """Ordinary least squares with an intercept (numpy), used as the linear baseline."""
    import numpy as np
    X = np.hstack([np.asarray(xtr, dtype=float), np.ones((len(xtr), 1))])
    coef, *_ = np.linalg.lstsq(X, np.asarray(ytr, dtype=float), rcond=None)
    return lambda xs: (np.hstack([np.asarray(xs, dtype=float), np.ones((len(xs), 1))]) @ coef).tolist()


def train(spec: dict, rows: list[dict], train_cfg: dict | None = None) -> dict:
    """Train a network. Returns {"state_dict", "normaliser", "metrics", "history", ...}."""
    spec = validate_spec(spec)
    tc = validate_train(train_cfg or {})
    torch = _require_torch()
    data = usable_rows(rows, spec)
    if len(data) < LIMITS["min_rows"]:
        raise MLError("not_enough_data", f"{len(data)} usable rows; at least {LIMITS['min_rows']} are needed.",
                      usable_rows=len(data))
    tr, va, te = split(data, tc["seed"], tc["val_fraction"], tc["test_fraction"])
    cols, tgt = spec["inputs"], spec["target"]
    xtr, xva, xte = _matrix(tr, cols), _matrix(va, cols), _matrix(te, cols)
    ytr = [r[tgt] for r in tr]
    yva = [r[tgt] for r in va]
    yte = [r[tgt] for r in te]
    xm, xs = _stats(xtr)
    ym, ys = _stats([[v] for v in ytr])
    ym, ys = ym[0], ys[0]

    device = pick_device(tc["device"])
    torch.manual_seed(tc["seed"])
    random.seed(tc["seed"])

    def tens(m, mean, std):
        return torch.tensor([[(v - mean[j]) / std[j] for j, v in enumerate(r)] for r in m], dtype=torch.float32,
                            device=device)
    # All data is moved to the device ONCE and stays there. Only each batch's (small) index vector crosses
    # from CPU to device during training.
    Xtr, Xva, Xte = tens(xtr, xm, xs), tens(xva, xm, xs), tens(xte, xm, xs)
    Ytr = torch.tensor([[(v - ym) / ys] for v in ytr], dtype=torch.float32, device=device)
    Yva = torch.tensor([[(v - ym) / ys] for v in yva], dtype=torch.float32, device=device)
    n_train, bs = Xtr.shape[0], tc["batch_size"]

    net = build_module(spec, len(cols)).to(device)
    w_init = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    opt = torch.optim.Adam(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    loss_fn = torch.nn.MSELoss()
    # Shuffling: the permutation is generated on the CPU with a seeded generator (reproducible on every device).
    g = torch.Generator(device="cpu").manual_seed(tc["seed"])
    best = (float("inf"), None, 0)
    history = []
    grads_finite = True
    t0 = time.perf_counter()
    bad = 0
    for epoch in range(1, tc["epochs"] + 1):
        net.train()
        perm = torch.randperm(n_train, generator=g)               # CPU
        tot = torch.zeros((), device=device)                       # loss summed on the device, read once per epoch
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs].to(device, non_blocking=True)     # only this batch's indices are transferred
            # index_select, NOT advanced indexing (X[idx]): on Intel Arc + torch 2.15.0+xpu (Level Zero driver
            # 1.17.39894) the aten::index gather kernel fails with UR_RESULT_ERROR_UNKNOWN / DEVICE_LOST whenever
            # another process (e.g. LM Studio) is using the GPU; index_select is reliable in the same conditions.
            # See reports/XPU_DIAGNOSIS.md and scripts/xpu_stress.py.
            xb, yb = torch.index_select(Xtr, 0, idx), torch.index_select(Ytr, 0, idx)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.detach() * xb.shape[0]                     # one accumulation per batch (no host sync)
        # one device->host read per epoch; also confirms the last batch's gradients are finite
        train_loss = float(tot.item()) / n_train
        last_grads_ok = all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None)
        grads_finite = grads_finite and last_grads_ok
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xva), Yva).item())
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(vl, 6)})
        if not (math.isfinite(train_loss) and math.isfinite(vl) and last_grads_ok):
            raise MLError("training_diverged", f"Non-finite loss or gradients at epoch {epoch} "
                          f"(train {train_loss}, val {vl}, gradients finite: {last_grads_ok}). Try a lower lr.",
                          epoch=epoch)
        if vl < best[0] - 1e-7:
            best = (vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch)
            bad = 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    seconds = time.perf_counter() - t0
    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        pte = (net(Xte).cpu().flatten() * ys + ym).tolist()
    weight_change = math.sqrt(sum(float(((best[1][k] - w_init[k]) ** 2).sum()) for k in w_init))
    ols = least_squares(xtr, ytr)
    metrics = {
        "test": regression_metrics(yte, pte),
        "baseline_mean": regression_metrics(yte, [ym] * len(yte)),
        "baseline_linear": regression_metrics(yte, ols(xte)),
    }
    metrics["beats_mean_baseline"] = metrics["test"]["rmse"] < metrics["baseline_mean"]["rmse"]
    metrics["beats_linear_baseline"] = metrics["test"]["rmse"] < metrics["baseline_linear"]["rmse"]
    n_params = sum(p.numel() for p in net.parameters())
    return {"spec": spec, "train_config": tc, "device": device,
            "state_dict": {k: v.tolist() for k, v in best[1].items()},
            "normaliser": {"x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys},
            "metrics": metrics, "history": history, "best_epoch": best[2], "epochs_run": len(history),
            "seconds": round(seconds, 3), "parameters": n_params,
            "checks": {"gradients_finite": grads_finite, "weight_change_l2": round(weight_change, 6),
                       "first_train_loss": history[0]["train_loss"], "best_val_loss": round(best[0], 6)},
            "split": {"train": len(tr), "val": len(va), "test": len(te)},
            "test_examples": [{"inputs": {c: r[c] for c in cols}, "actual": r[tgt], "predicted": round(p, 5)}
                              for r, p in list(zip(te, pte))[:8]]}


class Predictor:
    """A trained network ready for inference (CPU). Loaded from a stored model."""

    def __init__(self, model: dict):
        torch = _require_torch()
        self.spec = model["spec"]
        self.norm = model["normaliser"]
        self.net = build_module(self.spec, len(self.spec["inputs"]))
        self.net.load_state_dict({k: torch.tensor(v) for k, v in model["state_dict"].items()})
        self.net.eval()
        self._torch = torch

    def predict_rows(self, rows: list[dict]) -> list[float]:
        cols, n = self.spec["inputs"], self.norm
        missing = sorted({c for r in rows for c in cols if not isinstance(r.get(c), (int, float))})
        if missing:
            raise MLError("invalid_input", f"Missing or non-numeric input(s): {missing}.", required=cols)
        x = self._torch.tensor([[(float(r[c]) - n["x_mean"][j]) / n["x_std"][j] for j, c in enumerate(cols)]
                                for r in rows], dtype=self._torch.float32)
        with self._torch.no_grad():
            y = self.net(x).flatten() * n["y_std"] + n["y_mean"]
        return [round(float(v), 6) for v in y]

    def predict(self, inputs: dict) -> float:
        return self.predict_rows([inputs])[0]


def summarise(result: dict) -> dict[str, Any]:
    """The headline numbers of a training run (no weights)."""
    m = result["metrics"]
    return {"name": result["spec"]["name"], "target": result["spec"]["target"],
            "inputs": result["spec"]["inputs"], "hidden": result["spec"]["hidden"],
            "parameters": result["parameters"], "device": result["device"], "seconds": result["seconds"],
            "epochs_run": result["epochs_run"], "best_epoch": result["best_epoch"], "split": result["split"],
            "test": m["test"], "baseline_mean": m["baseline_mean"], "baseline_linear": m["baseline_linear"],
            "beats_mean_baseline": m["beats_mean_baseline"], "beats_linear_baseline": m["beats_linear_baseline"]}
