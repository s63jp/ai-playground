"""Model registry: datasets and trained networks on disk, with provenance.

Layout (under <experiments>/ml/):
    datasets/<name>.json          rows + manifest (SHA-256 of rows, seed, physics config hash)
    models/<owner>/<name>/v<k>.json  spec, weights, normaliser, metrics, dataset hash, physics hash

`owner` is "researcher" for models built from the CLI / dashboard, or an agent id for networks an
agent trained during a session (those also live in that agent's memory folder, see memory/).
Stored models are content-hashed; loading checks the hash so a tampered model is refused.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from ..config import config_hash, resolve_path
from . import MLError

NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def ml_root(cfg: dict) -> Path:
    db = resolve_path(cfg, cfg["database"])
    return db.parent / "ml"


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class Registry:
    def __init__(self, cfg: dict, root: Path | None = None):
        self.cfg = cfg
        self.root = Path(root) if root else ml_root(cfg)
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ datasets
    def dataset_path(self, name: str) -> Path:
        if not NAME_RE.match(name):
            raise MLError("invalid_name", "Dataset names are 1-40 characters of letters, digits, '_' or '-'.")
        return self.root / "datasets" / f"{name}.json"

    def save_dataset(self, name: str, ds: dict) -> dict:
        p = self.dataset_path(name)
        p.write_text(json.dumps(ds), encoding="utf-8")
        return {"name": name, "path": str(p), **{k: ds["manifest"].get(k) for k in ("rows", "rows_sha256", "source")}}

    def load_dataset(self, name: str) -> dict:
        from .data import load_dataset
        p = self.dataset_path(name)
        if not p.exists():
            raise MLError("unknown_dataset", f"No dataset named '{name}'.", available=self.list_datasets())
        return load_dataset(p)

    def list_datasets(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "datasets").glob("*.json"))

    def dataset_info(self) -> list[dict]:
        out = []
        for name in self.list_datasets():
            m = json.loads(self.dataset_path(name).read_text(encoding="utf-8"))["manifest"]
            out.append({"name": name, **{k: m.get(k) for k in ("source", "rows", "seed", "rows_sha256",
                                                               "physics_config_hash", "created_at")}})
        return out

    # ------------------------------------------------------------ models
    def _model_dir(self, owner: str, name: str) -> Path:
        if not NAME_RE.match(owner) or not NAME_RE.match(name):
            raise MLError("invalid_name", "Owner and model names are 1-40 characters of letters, digits, '_' or '-'.")
        return self.root / "models" / owner / name

    def save_model(self, owner: str, result: dict, dataset: dict, extra: dict | None = None) -> dict:
        d = self._model_dir(owner, result["spec"]["name"])
        d.mkdir(parents=True, exist_ok=True)
        version = 1 + max([int(p.stem[1:]) for p in d.glob("v*.json")] or [0])
        body = {"owner": owner, "version": version, "created_at": time.time(),
                "physics_config_hash": config_hash(self.cfg),
                "dataset": {k: dataset["manifest"].get(k) for k in ("source", "rows", "rows_sha256", "seed")},
                **{k: result[k] for k in ("spec", "train_config", "device", "state_dict", "normaliser", "metrics",
                                          "history", "best_epoch", "epochs_run", "seconds", "parameters", "split",
                                          "test_examples")},
                **(extra or {})}
        body["content_sha256"] = _hash({k: v for k, v in body.items() if k != "content_sha256"})
        p = d / f"v{version}.json"
        p.write_text(json.dumps(body), encoding="utf-8")
        return {"owner": owner, "name": result["spec"]["name"], "version": version, "path": str(p),
                "content_sha256": body["content_sha256"]}

    def load_model(self, owner: str, name: str, version: int | None = None) -> dict:
        d = self._model_dir(owner, name)
        versions = sorted(int(p.stem[1:]) for p in d.glob("v*.json")) if d.exists() else []
        if not versions:
            raise MLError("unknown_model", f"No model '{name}' owned by '{owner}'.")
        v = versions[-1] if version is None else int(version)
        p = d / f"v{v}.json"
        if not p.exists():
            raise MLError("unknown_model", f"Model '{name}' has no version {v}.", versions=versions)
        body = json.loads(p.read_text(encoding="utf-8"))
        if _hash({k: x for k, x in body.items() if k != "content_sha256"}) != body.get("content_sha256"):
            raise MLError("model_tampered", f"Model {owner}/{name} v{v} does not match its content hash.")
        return body

    def list_models(self, owner: str | None = None) -> list[dict]:
        out = []
        base = self.root / "models"
        for p in sorted(base.glob("*/*/v*.json")):
            ow, nm = p.parent.parent.name, p.parent.name
            if owner and ow != owner:
                continue
            b = json.loads(p.read_text(encoding="utf-8"))
            m = b["metrics"]
            out.append({"owner": ow, "name": nm, "version": b["version"], "target": b["spec"]["target"],
                        "inputs": b["spec"]["inputs"], "hidden": b["spec"]["hidden"], "parameters": b["parameters"],
                        "test_rmse": m["test"]["rmse"], "test_r2": m["test"]["r2"],
                        "linear_rmse": m["baseline_linear"]["rmse"], "beats_linear": m["beats_linear_baseline"],
                        "dataset_rows": b["dataset"].get("rows"), "created_at": b["created_at"],
                        "closed_loop": b.get("closed_loop")})
        return out

    def attach(self, owner: str, name: str, version: int, key: str, value) -> None:
        """Record an evaluation result inside a stored model (re-hashed)."""
        d = self._model_dir(owner, name)
        p = d / f"v{version}.json"
        body = self.load_model(owner, name, version)
        body[key] = value
        body["content_sha256"] = _hash({k: v for k, v in body.items() if k != "content_sha256"})
        p.write_text(json.dumps(body), encoding="utf-8")
