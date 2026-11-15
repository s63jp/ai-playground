"""Persistent model leaderboard (Nov 2026).

Every finished train job and model-comparison run records one profile entry into a single JSON file
under the researcher ML root, so the dashboard can show a sortable per-model matrix with trends instead
of disconnected one-off comparison tables. Entries are append-only and capped (newest 500 kept).
"""
from __future__ import annotations

import json
import threading
import time

from .registry import ml_root

_LOCK = threading.Lock()
_MAX_ENTRIES = 500


def _path(cfg: dict):
    return ml_root(cfg) / "leaderboard.json"


def _load(path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (ValueError, OSError):
        return []


def record(cfg: dict, entry: dict) -> dict:
    """Append one profile entry and return it (with a timestamp)."""
    from ..memory.store import _write_atomic
    entry = dict(entry)
    entry.setdefault("ts", time.time())
    if "source" not in entry:
        entry["source"] = "train"
    with _LOCK:
        path = _path(cfg)
        entries = _load(path) + [entry]
        entries = entries[-_MAX_ENTRIES:]
        _write_atomic(path, json.dumps(entries, indent=1))
    return entry


def entries(cfg: dict, source: str | None = None, limit: int = 200) -> list[dict]:
    all_entries = _load(_path(cfg))
    if source:
        all_entries = [e for e in all_entries if e.get("source") == source]
    return all_entries[-limit:]


def by_model(cfg: dict) -> list[dict]:
    """Aggregates per (owner, name): best accuracy metrics, trend vs the previous train entry and the
    latest behavioural compare metrics if any."""
    latest: dict[tuple, dict] = {}
    for e in _load(_path(cfg)):
        key = (e.get("owner", "researcher"), e.get("name", "?"))
        d = latest.setdefault(key, {"train_metrics": [], "compare_metrics": []})
        (d["compare_metrics"] if e.get("source") == "compare" else d["train_metrics"]).append(e)
    out = []
    for (owner, name), d in sorted(latest.items()):
        trains = d["train_metrics"]
        r2s = [t.get("test_r2") for t in trains if t.get("test_r2") is not None]
        rmses = [t.get("test_rmse") for t in trains if t.get("test_rmse") is not None]
        beats = sum(1 for t in trains if t.get("beats_linear"))
        trend = None
        if len(r2s) >= 2 and r2s[-1] is not None and r2s[-2] is not None:
            trend = round(r2s[-1] - r2s[-2], 4)
        latest_compare = d["compare_metrics"][-1] if d["compare_metrics"] else None
        out.append({
            "owner": owner, "name": name,
            "train_count": len(trains), "compare_count": len(d["compare_metrics"]),
            "best_test_r2": max(r2s) if r2s else None,
            "best_test_rmse": min(rmses) if rmses else None,
            "beats_linear": beats,
            "trend_r2": trend,
            "last_ts": max([t.get("ts", 0) for t in trains + d["compare_metrics"]] or [0]),
            "latest_compare": latest_compare,
        })
    return out
