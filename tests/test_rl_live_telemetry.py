"""Progress telemetry: read-only record readers and the incremental tail."""
from __future__ import annotations

import json

from playground.rl_live.telemetry import ProgressTail, read_records


def _write(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(i):
    return {"iteration": i, "total_valid_steps": i * 2048, "stage": "standing"}


def test_read_records_basic(tmp_path):
    p = tmp_path / "progress.jsonl"
    assert read_records(p) == []
    _write(p, [_rec(i) for i in range(1, 6)])
    assert [r["iteration"] for r in read_records(p)] == [1, 2, 3, 4, 5]
    assert [r["iteration"] for r in read_records(p, limit=2)] == [4, 5]


def test_read_records_skips_torn_lines(tmp_path):
    p = tmp_path / "progress.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_rec(1)) + "\n")
        f.write('{"iteration": 2, "broken')        # torn write, no newline
    recs = read_records(p)
    assert [r["iteration"] for r in recs] == [1]


def test_progress_tail_incremental(tmp_path):
    p = tmp_path / "progress.jsonl"
    tail = ProgressTail(p)
    assert tail.read_new() == []                   # file missing → empty
    _write(p, [_rec(1), _rec(2)])
    got = tail.read_new()
    assert [r["iteration"] for r in got] == [1, 2]
    assert tail.read_new() == []                   # nothing new
    # append in two chunks; the second lands mid-line first
    full3 = json.dumps(_rec(3))
    with p.open("a", encoding="utf-8") as f:
        f.write(full3[:12])
    assert tail.read_new() == []                   # torn line is held, not lost
    with p.open("a", encoding="utf-8") as f:
        f.write(full3[12:] + "\n")
    got = tail.read_new()
    assert [r["iteration"] for r in got] == [3]


def test_progress_tail_last_record(tmp_path):
    p = tmp_path / "progress.jsonl"
    tail = ProgressTail(p)
    assert tail.last_record() is None
    _write(p, [_rec(9)])
    assert tail.last_record()["iteration"] == 9
