"""SQLite experimental record.

What lives here is the *researcher's* record: what the physics engine measured,
what tools were called, what the agent said, what was independently verified.
The agent's own memory lives in plain files (see memory/store.py).

`events` and `actions` are append-only (enforced by triggers) and hash-chained per
session, so any later edit to the history is detectable with verify_chain().
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from ..config import canonical_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL, mode TEXT NOT NULL,
  session_number INTEGER, label TEXT, adapter TEXT, model TEXT, config_hash TEXT, config_json TEXT,
  memory_dir TEXT, status TEXT NOT NULL, end_reason TEXT, started_at REAL, ended_at REAL,
  sim_time REAL DEFAULT 0, physics_steps INTEGER DEFAULT 0, action_count INTEGER DEFAULT 0,
  wall_seconds REAL DEFAULT 0, budget_json TEXT, final_state_json TEXT, summary_json TEXT,
  benchmark_run_id INTEGER, task_id TEXT, condition TEXT);
CREATE INDEX IF NOT EXISTS ix_sessions_agent ON sessions(agent_id, mode, session_number);

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, n INTEGER NOT NULL,
  source TEXT NOT NULL, t REAL, step INTEGER, type TEXT NOT NULL, action_idx INTEGER,
  data_json TEXT NOT NULL, hidden_json TEXT, prev_hash TEXT NOT NULL, hash TEXT NOT NULL,
  UNIQUE(session_id, n));
CREATE INDEX IF NOT EXISTS ix_events_type ON events(session_id, type);

CREATE TABLE IF NOT EXISTS actions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, idx INTEGER NOT NULL,
  tool TEXT NOT NULL, category TEXT, args_json TEXT NOT NULL, result_json TEXT NOT NULL,
  success INTEGER NOT NULL, error_code TEXT, source TEXT NOT NULL, program TEXT, turn INTEGER,
  t_start REAL, t_end REAL, n_from INTEGER, n_to INTEGER, wall_ts REAL, wall_ms REAL,
  UNIQUE(session_id, idx));

CREATE TABLE IF NOT EXISTS frames(
  session_id INTEGER NOT NULL, i INTEGER NOT NULL, t REAL, key INTEGER, data_json TEXT NOT NULL,
  PRIMARY KEY(session_id, i));

CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, turn INTEGER, role TEXT,
  content TEXT, wall_ts REAL);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS records(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, agent_id TEXT NOT NULL, kind TEXT NOT NULL,
  key TEXT, action_idx INTEGER, t REAL, data_json TEXT NOT NULL, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_records ON records(agent_id, kind, session_id);

CREATE TABLE IF NOT EXISTS checkpoints(
  session_id INTEGER NOT NULL, action_idx INTEGER NOT NULL, state_json TEXT NOT NULL, wall_ts REAL,
  PRIMARY KEY(session_id, action_idx));

CREATE TABLE IF NOT EXISTS benchmark_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, kind TEXT, created_at REAL, params_json TEXT,
  summary_json TEXT);
CREATE TABLE IF NOT EXISTS benchmark_results(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, session_id INTEGER, agent_id TEXT,
  grp TEXT, condition TEXT, task_id TEXT, metrics_json TEXT NOT NULL, verified_json TEXT NOT NULL,
  created_at REAL);

CREATE TABLE IF NOT EXISTS memory_audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL, ts REAL, actor TEXT, op TEXT,
  item_id TEXT, before_json TEXT, after_json TEXT, reason TEXT);

CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS actions_no_update BEFORE UPDATE ON actions
  BEGIN SELECT RAISE(ABORT, 'actions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS actions_no_delete BEFORE DELETE ON actions
  BEGIN SELECT RAISE(ABORT, 'actions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON memory_audit
  BEGIN SELECT RAISE(ABORT, 'the memory audit trail is immutable'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON memory_audit
  BEGIN SELECT RAISE(ABORT, 'the memory audit trail is immutable'); END;
"""

GENESIS = "0" * 64
# Record kinds kept in the generic `records` table.
RECORD_KINDS = ("experiment", "discovery", "mistake", "goal", "diary", "label", "claim",
                "measurement", "behaviour", "program", "integrity_flag", "memory_use")


def event_hash(prev: str, session_id: int, n: int, source: str, t, step, etype: str, action_idx,
               data_json: str, hidden_json: str | None) -> str:
    payload = canonical_json([prev, session_id, n, source, t, step, etype, action_idx, data_json, hidden_json])
    return hashlib.sha256(payload.encode()).hexdigest()


def _j(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=60000")
            self.conn.executescript(SCHEMA)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # ------------------------------------------------------------ primitives
    def q(self, sql: str, params: Iterable = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable = ()) -> dict | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def x(self, sql: str, params: Iterable = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, tuple(params))
            return cur.lastrowid

    def tx(self):
        return _Tx(self)

    # ------------------------------------------------------------ sessions
    def create_session(self, **f) -> int:
        f.setdefault("status", "running")
        f.setdefault("started_at", time.time())
        for k in ("config_json", "budget_json"):
            if k in f and not isinstance(f[k], (str, type(None))):
                f[k] = _j(f[k])
        cols = ",".join(f)
        return self.x(f"INSERT INTO sessions({cols}) VALUES({','.join('?' * len(f))})", f.values())

    def update_session(self, sid: int, **f) -> None:
        for k in ("final_state_json", "summary_json", "budget_json"):
            if k in f and not isinstance(f[k], (str, type(None))):
                f[k] = _j(f[k])
        sets = ",".join(f"{k}=?" for k in f)
        self.x(f"UPDATE sessions SET {sets} WHERE id=?", [*f.values(), sid])

    def session(self, sid: int) -> dict | None:
        return self.one("SELECT * FROM sessions WHERE id=?", (sid,))

    def next_session_number(self, agent_id: str, mode: str = "exploration") -> int:
        r = self.one("SELECT COALESCE(MAX(session_number),0) AS m FROM sessions WHERE agent_id=? AND mode=?",
                     (agent_id, mode))
        return int(r["m"]) + 1

    # ------------------------------------------------------------ append-only log
    def last_hash(self, sid: int) -> tuple[int, str]:
        r = self.one("SELECT n, hash FROM events WHERE session_id=? ORDER BY n DESC LIMIT 1", (sid,))
        return (int(r["n"]), r["hash"]) if r else (0, GENESIS)

    def append_events(self, sid: int, rows: list[dict], start_n: int, prev_hash: str) -> tuple[int, str]:
        """rows: dicts with source, t, step, type, action_idx, data, hidden. Returns (last_n, last_hash)."""
        n, prev = start_n, prev_hash
        out = []
        for r in rows:
            n += 1
            dj = _j(r.get("data") or {})
            hj = _j(r["hidden"]) if r.get("hidden") else None
            h = event_hash(prev, sid, n, r["source"], r.get("t"), r.get("step"), r["type"], r.get("action_idx"), dj, hj)
            out.append((sid, n, r["source"], r.get("t"), r.get("step"), r["type"], r.get("action_idx"), dj, hj, prev, h))
            prev = h
        if out:
            with self.lock:
                self.conn.executemany(
                    "INSERT INTO events(session_id,n,source,t,step,type,action_idx,data_json,hidden_json,prev_hash,hash)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)", out)
        return n, prev

    def append_action(self, sid: int, **f) -> None:
        f["args_json"] = _j(f.pop("args"))
        f["result_json"] = _j(f.pop("result"))
        cols = ",".join(f)
        self.x(f"INSERT INTO actions(session_id,{cols}) VALUES(?,{','.join('?' * len(f))})", [sid, *f.values()])

    def append_frames(self, sid: int, frames: list[dict]) -> None:
        if frames:
            with self.lock:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO frames(session_id,i,t,key,data_json) VALUES(?,?,?,?,?)",
                    [(sid, f["i"], f["t"], 1 if f.get("key") else 0, _j(f)) for f in frames])

    def verify_chain(self, sid: int) -> dict:
        prev, count = GENESIS, 0
        for r in self.q("SELECT * FROM events WHERE session_id=? ORDER BY n", (sid,)):
            count += 1
            if r["n"] != count:
                return {"ok": False, "events": count, "error": f"sequence gap at n={r['n']}"}
            if r["prev_hash"] != prev:
                return {"ok": False, "events": count, "error": f"broken link at n={r['n']}"}
            h = event_hash(prev, sid, r["n"], r["source"], r["t"], r["step"], r["type"], r["action_idx"],
                           r["data_json"], r["hidden_json"])
            if h != r["hash"]:
                return {"ok": False, "events": count, "error": f"content altered at n={r['n']}"}
            prev = h
        return {"ok": True, "events": count, "head": prev}

    # ------------------------------------------------------------ readers
    def actions(self, sid: int, after: int = 0, limit: int = 100000, tools: list[str] | None = None) -> list[dict]:
        sql = "SELECT * FROM actions WHERE session_id=? AND idx>?"
        params: list = [sid, after]
        if tools:
            sql += f" AND tool IN ({','.join('?' * len(tools))})"
            params += tools
        rows = self.q(sql + " ORDER BY idx LIMIT ?", [*params, limit])
        for r in rows:
            r["args"] = json.loads(r.pop("args_json"))
            r["result"] = json.loads(r.pop("result_json"))
            r["success"] = bool(r["success"])
        return rows

    def action(self, sid: int, idx: int) -> dict | None:
        rows = self.q("SELECT * FROM actions WHERE session_id=? AND idx=?", (sid, idx))
        if not rows:
            return None
        r = rows[0]
        r["args"] = json.loads(r.pop("args_json"))
        r["result"] = json.loads(r.pop("result_json"))
        r["success"] = bool(r["success"])
        return r

    def events(self, sid: int, after: int = 0, limit: int = 100000, types: list[str] | None = None,
               include_hidden: bool = True) -> list[dict]:
        sql = "SELECT n,source,t,step,type,action_idx,data_json,hidden_json FROM events WHERE session_id=? AND n>?"
        params: list = [sid, after]
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            params += types
        rows = self.q(sql + " ORDER BY n LIMIT ?", [*params, limit])
        for r in rows:
            r["data"] = json.loads(r.pop("data_json"))
            h = r.pop("hidden_json")
            if include_hidden and h:
                r["hidden"] = json.loads(h)
        return rows

    def frames(self, sid: int, start: int = 0, limit: int = 2000) -> list[dict]:
        return [json.loads(r["data_json"]) for r in
                self.q("SELECT data_json FROM frames WHERE session_id=? AND i>=? ORDER BY i LIMIT ?",
                       (sid, start, limit))]

    def frame_count(self, sid: int) -> int:
        r = self.one("SELECT COUNT(*) AS c, COALESCE(MAX(i),-1) AS m FROM frames WHERE session_id=?", (sid,))
        return int(r["m"]) + 1

    # ------------------------------------------------------------ generic records
    def add_record(self, agent_id: str, kind: str, data: dict, session_id: int | None = None,
                   key: str | None = None, action_idx: int | None = None, t: float | None = None) -> int:
        assert kind in RECORD_KINDS, kind
        return self.x("INSERT INTO records(session_id,agent_id,kind,key,action_idx,t,data_json,created_at)"
                      " VALUES(?,?,?,?,?,?,?,?)",
                      (session_id, agent_id, kind, key, action_idx, t, _j(data), time.time()))

    def records(self, kind: str | None = None, agent_id: str | None = None, session_id: int | None = None,
                limit: int = 100000) -> list[dict]:
        sql, params = "SELECT * FROM records WHERE 1=1", []
        for col, val in (("kind", kind), ("agent_id", agent_id), ("session_id", session_id)):
            if val is not None:
                sql += f" AND {col}=?"
                params.append(val)
        rows = self.q(sql + " ORDER BY id LIMIT ?", [*params, limit])
        for r in rows:
            r["data"] = json.loads(r.pop("data_json"))
        return rows

    def delete_records(self, kind: str, session_id: int) -> None:
        self.x("DELETE FROM records WHERE kind=? AND session_id=?", (kind, session_id))

    def add_message(self, sid: int, turn: int, role: str, content: str) -> None:
        self.x("INSERT INTO messages(session_id,turn,role,content,wall_ts) VALUES(?,?,?,?,?)",
               (sid, turn, role, content, time.time()))

    def messages(self, sid: int, after_id: int = 0, limit: int = 100000) -> list[dict]:
        return self.q("SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id LIMIT ?", (sid, after_id, limit))

    def save_checkpoint(self, sid: int, action_idx: int, state: dict, keep: int = 3) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO checkpoints(session_id,action_idx,state_json,wall_ts)"
                              " VALUES(?,?,?,?)", (sid, action_idx, _j(state), time.time()))
            self.conn.execute(
                "DELETE FROM checkpoints WHERE session_id=? AND action_idx NOT IN "
                "(SELECT action_idx FROM checkpoints WHERE session_id=? ORDER BY action_idx DESC LIMIT ?)",
                (sid, sid, keep))

    def latest_checkpoint(self, sid: int) -> dict | None:
        r = self.one("SELECT * FROM checkpoints WHERE session_id=? ORDER BY action_idx DESC LIMIT 1", (sid,))
        if r:
            r["state"] = json.loads(r.pop("state_json"))
        return r

    def audit(self, agent_id: str, actor: str, op: str, item_id: str | None, before: Any, after: Any,
              reason: str | None) -> None:
        self.x("INSERT INTO memory_audit(agent_id,ts,actor,op,item_id,before_json,after_json,reason)"
               " VALUES(?,?,?,?,?,?,?,?)",
               (agent_id, time.time(), actor, op, item_id, _j(before) if before is not None else None,
                _j(after) if after is not None else None, reason))


class _Tx:
    def __init__(self, db: Database):
        self.db = db

    def __enter__(self):
        self.db.lock.acquire()
        self.db.conn.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type, exc, tb):
        try:
            self.db.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.db.lock.release()
        return False
