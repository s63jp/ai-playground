"""Configuration loading. One YAML file drives the whole laboratory."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """Load the default config, then an optional file, then in-memory overrides."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    w = cfg["world"]
    if w["blocks"]["count"] < 1:
        raise ValueError("world.blocks.count must be at least 1")
    if cfg["rules"]["ring_generation"]["charges"] > w["blocks"]["reserve"]:
        raise ValueError("rules.ring_generation.charges cannot exceed world.blocks.reserve")
    if w["timestep"] <= 0 or w["timestep"] > 0.01:
        raise ValueError("world.timestep must be in (0, 0.01]")
    if cfg["agents"]["count"] < 1:
        raise ValueError("agents.count must be at least 1")
    ids = [p["id"] for p in w["platforms"]] + [w["spire"]["id"], w["rock"]["id"], w["ring"]["id"]]
    if len(ids) != len(set(ids)):
        raise ValueError("world object ids must be unique")


def config_hash(cfg: dict, sections: tuple[str, ...] = ("world", "agent_body", "rules")) -> str:
    """Hash of the parts of the config that define the experiment's physics."""
    payload = {k: cfg[k] for k in sections}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resolve_path(cfg: dict, value: str) -> Path:
    """Resolve a config path relative to the project (or PLAYGROUND_HOME) root."""
    p = Path(value)
    if p.is_absolute():
        return p
    root = Path(cfg.get("_root") or os.environ.get("PLAYGROUND_HOME") or PROJECT_ROOT)
    return root / p


def agent_ids(cfg: dict) -> list[str]:
    fmt = cfg["agents"]["id_format"]
    return [fmt.format(i) for i in range(1, cfg["agents"]["count"] + 1)]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
