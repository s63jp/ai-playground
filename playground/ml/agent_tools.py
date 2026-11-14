"""Agent-facing neural-network tools.

These tools let an agent build datasets from their own throws, train networks, predict outcomes,
and evaluate models - all within their session and scoped to their own data. Results with
`is_prediction=True` are flagged as model outputs, not engine measurements, so they cannot be
used as evidence in experiments or discoveries.

Contract 3.3 from ML_INTERFACE.md: learning tools nn_list, nn_make_dataset, nn_train, nn_predict,
nn_evaluate; category 'learning'; program_ok False; rows only from this agent's own engine-recorded
throws; CPU-only training; torch-missing -> clean ToolError, never crash.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import MLError, torch_status
from .data import dataset_from_session_events
from .features import INPUT_COLUMNS, TARGET_COLUMNS
from .nets import LIMITS, train, validate_spec, validate_train, Predictor, regression_metrics

if TYPE_CHECKING:
    from ..interface.environment import Environment


def _require_torch_for_agent() -> None:
    """Check torch availability and raise a clean ToolError if missing."""
    st = torch_status()
    if not st["available"]:
        from ..interface.schema import ToolError
        raise ToolError("torch_missing", st["reason"])


def _agent_registry_path(memory_dir: Path) -> Path:
    """Return the path to an agent's model registry directory."""
    from .registry import agent_models_dir
    p = agent_models_dir(memory_dir, memory_dir.name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dataset_path(memory_dir: Path) -> Path:
    """Return the path to an agent's datasets directory."""
    from .registry import agent_datasets_dir
    p = agent_datasets_dir(memory_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ nn_list
def nn_list(env: "Environment") -> dict:
    """List the agent's own trained models and available datasets."""
    mem_dir = env.memory.dir
    models_dir = _agent_registry_path(mem_dir)
    datasets_dir = _dataset_path(mem_dir)

    # List models
    models = []
    for p in sorted(models_dir.glob("*/v*.json")):
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
            m = body.get("metrics", {})
            models.append({
                "name": body["spec"]["name"],
                "version": body["version"],
                "target": body["spec"]["target"],
                "inputs": body["spec"]["inputs"],
                "test_rmse": m.get("test", {}).get("rmse"),
                "created_at": body.get("created_at"),
            })
        except Exception:
            continue

    # List datasets
    datasets = []
    for p in sorted(datasets_dir.glob("*.json")):
        try:
            ds = json.loads(p.read_text(encoding="utf-8"))
            man = ds.get("manifest", {})
            datasets.append({
                "name": p.stem,
                "rows": man.get("rows"),
                "source": man.get("source"),
                "created_at": man.get("created_at"),
            })
        except Exception:
            continue

    return {
        "models": models,
        "model_count": len(models),
        "datasets": datasets,
        "dataset_count": len(datasets),
        "available_inputs": sorted(INPUT_COLUMNS.keys()),
        "available_targets": sorted(TARGET_COLUMNS.keys()),
    }


# ------------------------------------------------------------------ nn_make_dataset
def nn_make_dataset(env: "Environment", name: str, min_rows: int = 20, kind: str = "throw") -> dict:
    """Build a dataset from the agent's own engine records in this and past sessions.

    Args:
        name: Name for the dataset (1-40 chars, letters/digits/_/-).
        min_rows: Minimum number of usable rows required.
        kind: What to learn. "throw" (default) = throw outcomes. Other behaviours: "climb", "jump",
            "locomotion", "manipulation", "placement", "affordance" (the cross-action self-model) —
            see nn_list for the catalogue. Lets an agent learn about more than throwing.

    Returns:
        Dataset info including row count and provenance hash.
    """
    from .registry import NAME_RE
    from ..interface.schema import ToolError

    if not NAME_RE.match(name):
        raise ToolError("invalid_name", "Dataset names are 1-40 characters of letters, digits, '_' or '-'.")

    agent_sessions = env.db.q("SELECT id FROM sessions WHERE agent_id = ? ORDER BY id", (env.agent_id,))
    session_ids = [s["id"] for s in agent_sessions]
    if not session_ids:
        raise ToolError("not_enough_data", "No sessions found for this agent.")

    if kind == "throw":
        all_events = []
        for sid in session_ids:
            all_events.extend(env.db.events(sid, types=["throw_release", "flight_end"], limit=10**6))
        ds = dataset_from_session_events(all_events, env.cfg)
    else:
        from .datasets import DATASETS, build_dataset
        if kind not in DATASETS:
            raise ToolError("unknown_kind", f"Unknown dataset kind '{kind}'.",
                            available=["throw"] + sorted(DATASETS))
        actions = []
        for sid in session_ids:
            actions.extend(env.db.actions(sid, limit=10**6))
        ds = build_dataset(actions, kind)

    if ds["manifest"]["rows"] < min_rows:
        raise ToolError("not_enough_data",
                        f"Only {ds['manifest']['rows']} usable '{kind}' rows found; at least {min_rows} required.",
                        rows_found=ds["manifest"]["rows"])

    ds["manifest"]["source"] = f"agent:{env.agent_id}"
    ds["manifest"]["sessions"] = session_ids
    ds["manifest"]["created_at"] = time.time()
    if "rows_sha256" not in ds["manifest"]:
        ds["manifest"]["rows_sha256"] = _hash(ds["rows"])

    p = _dataset_path(env.memory.dir) / f"{name}.json"
    p.write_text(json.dumps(ds), encoding="utf-8")

    return {
        "name": name,
        "kind": kind,
        "rows": ds["manifest"]["rows"],
        "rows_sha256": ds["manifest"]["rows_sha256"],
        "sessions": len(session_ids),
        "path": str(p),
        "saved": True,
    }


# ------------------------------------------------------------------ nn_train
def nn_train(env: "Environment", dataset_name: str, spec: dict, train_config: dict | None = None) -> dict:
    """Train a neural network on one of the agent's datasets.

    Args:
        dataset_name: Name of a dataset created with nn_make_dataset.
        spec: Model specification dict with keys: name, inputs, target, hidden (optional), activation (optional).
        train_config: Optional training config (epochs, lr, batch_size, patience, etc.). Device forced to CPU.

    Returns:
        Training result with metrics, but not the full state_dict.
    """
    from ..interface.schema import ToolError

    _require_torch_for_agent()

    # Load dataset
    ds_path = _dataset_path(env.memory.dir) / f"{dataset_name}.json"
    if not ds_path.exists():
        available = [p.stem for p in _dataset_path(env.memory.dir).glob("*.json")]
        raise ToolError("unknown_dataset", f"No dataset named '{dataset_name}'.",
                        available=available)

    ds = json.loads(ds_path.read_text(encoding="utf-8"))

    # The dataset carries its own column vocabulary (throw datasets use the default). This lets an agent
    # train climb/jump/affordance/etc. models through the same path as throws (audit 2026-11-13).
    from .datasets import vocab_from_manifest
    in_cols, tgt_cols = vocab_from_manifest(ds.get("manifest", {}))

    # Validate spec
    try:
        spec = validate_spec(spec, in_cols, tgt_cols)
    except MLError as e:
        raise ToolError(e.code, e.message, **e.details) from None

    # Force CPU training
    tc = train_config or {}
    tc["device"] = "cpu"

    try:
        tc = validate_train(tc)
    except MLError as e:
        raise ToolError(e.code, e.message, **e.details) from None

    # Train
    try:
        result = train(spec, ds["rows"], tc, in_cols, tgt_cols)
    except MLError as e:
        raise ToolError(e.code, e.message, **e.details) from None

    # Save model
    models_dir = _agent_registry_path(env.memory.dir)
    model_dir = models_dir / spec["name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # Find next version
    versions = sorted(int(p.stem[1:]) for p in model_dir.glob("v*.json")) if model_dir.exists() else []
    version = (versions[-1] + 1) if versions else 1

    body = {
        "owner": env.agent_id,
        "version": version,
        "created_at": time.time(),
        "dataset": {
            "name": dataset_name,
            "rows": ds["manifest"]["rows"],
            "rows_sha256": ds["manifest"]["rows_sha256"],
        },
        **{k: result[k] for k in ("spec", "train_config", "device", "state_dict", "normaliser",
                                  "metrics", "history", "best_epoch", "epochs_run", "seconds",
                                  "parameters", "split", "test_examples")},
    }
    body["content_sha256"] = _hash({k: v for k, v in body.items() if k != "content_sha256"})

    p = model_dir / f"v{version}.json"
    p.write_text(json.dumps(body), encoding="utf-8")

    # Return summary (no weights)
    m = result["metrics"]
    return {
        "name": spec["name"],
        "version": version,
        "saved": True,
        "path": str(p),
        "target": spec["target"],
        "inputs": spec["inputs"],
        "hidden": spec["hidden"],
        "parameters": result["parameters"],
        "epochs_run": result["epochs_run"],
        "best_epoch": result["best_epoch"],
        "seconds": result["seconds"],
        "test_rmse": m["test"]["rmse"],
        "test_r2": m["test"]["r2"],
        "beats_linear_baseline": m["beats_linear_baseline"],
        "dataset_rows": ds["manifest"]["rows"],
    }


# ------------------------------------------------------------------ nn_predict
def nn_predict(env: "Environment", model_name: str, inputs: list[dict], version: int | None = None) -> dict:
    """Make predictions using one of the agent's trained models.

    Args:
        model_name: Name of a model trained with nn_train.
        inputs: List of input row dicts, each containing the model's input columns.
        version: Specific version to use (default: latest).

    Returns:
        Predictions with `is_prediction=True` flag.
    """
    from ..interface.schema import ToolError

    _require_torch_for_agent()

    # Load model
    models_dir = _agent_registry_path(env.memory.dir) / model_name
    if not models_dir.exists():
        available = [p.name for p in _agent_registry_path(env.memory.dir).iterdir() if p.is_dir()]
        raise ToolError("unknown_model", f"No model named '{model_name}'.",
                        available=available)

    versions = sorted(int(p.stem[1:]) for p in models_dir.glob("v*.json"))
    if not versions:
        raise ToolError("unknown_model", f"Model '{model_name}' has no versions.")

    v = version if version is not None else versions[-1]
    p = models_dir / f"v{v}.json"
    if not p.exists():
        raise ToolError("unknown_model", f"Model '{model_name}' has no version {v}.",
                        versions=versions)

    body = json.loads(p.read_text(encoding="utf-8"))

    # Verify integrity
    if _hash({k: x for k, x in body.items() if k != "content_sha256"}) != body.get("content_sha256"):
        raise ToolError("model_tampered", f"Model {model_name} v{v} does not match its content hash.")

    # Predict
    try:
        pred = Predictor(body)
        preds = pred.predict_rows(inputs)
    except MLError as e:
        raise ToolError(e.code, e.message, **e.details) from None

    return {
        "model": model_name,
        "version": v,
        "target": body["spec"]["target"],
        "predictions": preds,
        "count": len(preds),
        "is_prediction": True,
        "note": "These are model predictions, not engine measurements. They cannot be used as evidence.",
    }


# ------------------------------------------------------------------ nn_evaluate
def nn_evaluate(env: "Environment", model_name: str, version: int | None = None) -> dict:
    """Evaluate a model against fresh throw data from this session.

    Args:
        model_name: Name of a model trained with nn_train.
        version: Specific version to evaluate (default: latest).

    Returns:
        Evaluation metrics computed from this session's throws.
    """
    from ..interface.schema import ToolError

    _require_torch_for_agent()

    # Load model
    models_dir = _agent_registry_path(env.memory.dir) / model_name
    if not models_dir.exists():
        available = [p.name for p in _agent_registry_path(env.memory.dir).iterdir() if p.is_dir()]
        raise ToolError("unknown_model", f"No model named '{model_name}'.",
                        available=available)

    versions = sorted(int(p.stem[1:]) for p in models_dir.glob("v*.json"))
    if not versions:
        raise ToolError("unknown_model", f"Model '{model_name}' has no versions.")

    v = version if version is not None else versions[-1]
    p = models_dir / f"v{v}.json"
    if not p.exists():
        raise ToolError("unknown_model", f"Model '{model_name}' has no version {v}.",
                        versions=versions)

    body = json.loads(p.read_text(encoding="utf-8"))

    # Get this session's throws
    events = env.db.events(env.sid, types=["throw_release", "flight_end"], limit=10**6)
    ds = dataset_from_session_events(events, env.cfg)

    if ds["manifest"]["rows"] < 1:
        raise ToolError("not_enough_data", "No throws found in this session to evaluate against.")

    # Filter to usable rows
    from .nets import usable_rows
    spec = body["spec"]
    rows = usable_rows(ds["rows"], spec)

    if not rows:
        raise ToolError("not_enough_data",
                        f"No throws in this session have all required inputs: {spec['inputs']}")

    # Predict
    try:
        pred = Predictor(body)
        preds = pred.predict_rows(rows)
    except MLError as e:
        raise ToolError(e.code, e.message, **e.details) from None

    # Compute metrics
    actuals = [r[spec["target"]] for r in rows]
    metrics = regression_metrics(actuals, preds)

    return {
        "model": model_name,
        "version": v,
        "target": spec["target"],
        "eval_rows": len(rows),
        "metrics": metrics,
        "is_prediction": True,
        "note": "Evaluation uses this session's engine-recorded throws.",
    }
