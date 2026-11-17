"""Read-only progress telemetry.

Two uses:

1. The live session's own `progress.jsonl` — written by the on_iteration
   hook (same record schema the B3 trainer emits: one JSON object per PPO
   iteration).
2. The RUNNING M6-B3 background run's `progress.jsonl` — READ-ONLY tailing;
   never opened for writing, never moved, never truncated.

All readers tolerate a partially written trailing line (the trainer flushes
per iteration, but a reader can always arrive between flush and newline).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


def read_records(path, limit: int | None = None) -> list[dict]:
    """Full read (small files). Returns the last `limit` records when given."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn line: the next full read supersedes it
    return out[-limit:] if limit else out


class ProgressTail:
    """Incremental JSONL tail with byte offsets; tolerates torn lines."""

    def __init__(self, path):
        self.path = Path(path)
        self._offset = 0
        self._partial = ""

    def read_new(self) -> list[dict]:
        """Records appended since the last call (including a torn final line)."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self._offset)
            raw = f.read()
            if not raw:
                # Nothing new — unless the file shrank (replaced/rotated by
                # whatever owns it); then start over. Never this module's
                # choice: the trainer only ever appends.
                try:
                    size = self.path.stat().st_size
                except OSError:
                    return []
                if size < self._offset:
                    self._offset = 0
                    self._partial = ""
                    return self.read_new()
                return []
        text = (self._partial + raw) if self._partial else raw
        lines = text.split("\n")
        if raw.endswith("\n"):
            self._partial = ""
        else:
            self._partial = lines.pop() if lines else ""
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        # offset tracks FILE bytes only — the held partial was already counted
        # in the call that read it.
        self._offset += len(raw)
        return out

    def last_record(self) -> dict | None:
        recs = self.read_new()
        return recs[-1] if recs else None


class B3Telemetry:
    """Read-only follower of the RUNNING B3 background run's progress file.

    One refresh per `interval` seconds while SSE subscribers exist; JSON
    polls use the stateless reader. This object never writes anywhere —
    opens are read-only, paths are never created or moved.
    """

    def __init__(self, path, bus=None, interval=5.0):
        self.path = Path(path) if path else None
        self.bus = bus
        self.interval = interval
        self._tail = ProgressTail(self.path) if self.path and self.path.exists() else None
        self._lock = threading.Lock()
        self._last_refresh = 0.0

    def refresh(self) -> int:
        """Emit new B3 iteration records as bus events. Returns count."""
        if not (self.bus and self._tail and self.path and self.path.exists()):
            return 0
        now = time.monotonic()
        with self._lock:
            if now - self._last_refresh < self.interval:
                return 0
            self._last_refresh = now
        n = 0
        for rec in self._tail.read_new():
            self.bus.publish_event("b3_iteration", rec)
            n += 1
        return n

    def snapshot(self, limit: int = 30) -> dict:
        if not (self.path and self.path.exists()):
            return {"available": False, "path": str(self.path) if self.path else None}
        recs = read_records(self.path, limit=limit)
        return {"available": True, "path": str(self.path),
                "records": recs, "latest": recs[-1] if recs else None,
                "note": "read-only tail of the running B3 experiment "
                        "(experiments/rl/b3_run1); never written by this "
                        "server"}
