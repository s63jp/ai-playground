#!/usr/bin/env python
"""Audit LM Studio server logs: what was actually sent to the model, and what came back.

Reads LM Studio's DEBUG server logs (one JSON request body + one JSON prediction per call), and
writes one row per chat-completion request with:
  * composition of the assembled request in characters: system prompt, tool schemas, opening
    message (personality situation + memory briefing), assistant text, assistant tool calls,
    tool results, runner nudges;
  * the model's own token accounting (prompt_tokens, completion_tokens, reasoning_tokens);
  * what the model produced (content length, reasoning_content length, tool calls, finish reason);
  * the context window the model was loaded with at that moment (n_ctx_slot / n_slots);
  * context-overflow errors.
Rows are then attributed to playground sessions by time window and model.

Read-only with respect to the playground: nothing in experiments/ is modified.

Usage:
  python scripts/audit_lmstudio_logs.py --logs ~/.lmstudio/server-logs/2026-11 \
      --since 2026-11-12 --out reports/audit_2026_11_13
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
REQ_MARK = "Received request: POST to /v1/chat/completions with body {"
PRED_MARK = "Generated prediction: {"
LOAD_PATH_RE = re.compile(r"load_model: loading model '([^']+)'")
NCTX_RE = re.compile(r"n_slots = (\d+), n_ctx_slot = (\d+)")
OVERFLOW_RE = re.compile(r"Context size has been exceeded")


def _log_files(folder: Path, since: str | None) -> list[Path]:
    def key(p: Path):
        m = re.match(r"(\d{4}-\d{2}-\d{2})\.(\d+)\.log$", p.name)
        return (m.group(1), int(m.group(2))) if m else (p.name, 0)
    files = sorted((p for p in folder.glob("*.log") if re.match(r"\d{4}-\d{2}-\d{2}\.\d+\.log$", p.name)), key=key)
    if since:
        files = [p for p in files if key(p)[0] >= since]
    return files


def _epoch(ts: str) -> float:
    # LM Studio logs local time; the host runs GMT (UTC+0) in November.
    return dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def iter_blocks(files: list[Path]):
    """Yield ('request'|'prediction'|'load'|'nctx'|'overflow', ts, payload) in log order."""
    collecting = None          # (kind, ts, [lines])
    last_ts = None
    for f in files:
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = TS_RE.match(line)
                if m:
                    last_ts = m.group(1)
                if collecting is not None:
                    kind, ts, buf = collecting
                    if m and not line.startswith("}"):
                        # A new log record began before the JSON closed: give up on this block.
                        collecting = None
                    else:
                        buf.append(line)
                        if line.rstrip("\r\n") == "}":
                            try:
                                yield kind, ts, json.loads("".join(buf))
                            except json.JSONDecodeError:
                                pass
                            collecting = None
                        continue
                if REQ_MARK in line:
                    collecting = ("request", last_ts, ["{\n"])
                elif PRED_MARK in line:
                    collecting = ("prediction", last_ts, ["{\n"])
                else:
                    lm = LOAD_PATH_RE.search(line)
                    if lm:
                        yield "load", last_ts, lm.group(1)
                    nm = NCTX_RE.search(line)
                    if nm:
                        yield "nctx", last_ts, (int(nm.group(1)), int(nm.group(2)))
                    if OVERFLOW_RE.search(line):
                        yield "overflow", last_ts, None


def compose(body: dict) -> dict:
    msgs = body.get("messages") or []
    out = {"n_messages": len(msgs), "system": 0, "tools": len(json.dumps(body.get("tools") or [])),
           "n_tools": len(body.get("tools") or []), "opening": 0, "assistant_text": 0,
           "assistant_tool_calls": 0, "tool_results": 0, "nudges": 0, "trim_marker": 0}
    first_user_seen = False
    for msg in msgs:
        role = msg.get("role")
        content = msg.get("content")
        clen = len(content) if isinstance(content, str) else len(json.dumps(content or ""))
        if role == "system":
            out["system"] += clen
        elif role == "user":
            if not first_user_seen:
                out["opening"] += clen
                first_user_seen = True
            elif "[Earlier turns" in str(content):
                out["trim_marker"] += clen
            else:
                out["nudges"] += clen
        elif role == "assistant":
            out["assistant_text"] += clen
            out["assistant_tool_calls"] += len(json.dumps(msg.get("tool_calls") or []))
        elif role == "tool":
            out["tool_results"] += clen
    out["total_chars"] = sum(out[k] for k in ("system", "tools", "opening", "assistant_text",
                                                "assistant_tool_calls", "tool_results", "nudges", "trim_marker"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--since", default=None)
    ap.add_argument("--db", default="experiments/playground.db")
    ap.add_argument("--out", default="reports/audit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = _log_files(Path(args.logs).expanduser(), args.since)
    pending: dict[str, list[dict]] = defaultdict(list)   # model -> FIFO of requests awaiting a prediction
    rows: list[dict] = []
    loads: list[dict] = []
    overflows: list[float] = []
    cur_ctx: dict[str, tuple[int, int]] = {}            # model path stem -> (n_slots, n_ctx_slot)
    last_load_path = None
    for kind, ts, payload in iter_blocks(files):
        t = _epoch(ts) if ts else None
        if kind == "load":
            last_load_path = payload
        elif kind == "nctx" and last_load_path:
            loads.append({"ts": ts, "t": t, "path": last_load_path, "n_slots": payload[0], "n_ctx_slot": payload[1]})
        elif kind == "overflow":
            overflows.append(t)
        elif kind == "request":
            if not isinstance(payload.get("messages"), list):
                continue
            r = {"ts": ts, "t": t, "model": payload.get("model"), **compose(payload)}
            # Was this request from the playground? Its tool list contains the playground's tool names.
            names = {((x.get("function") or {}).get("name")) for x in payload.get("tools") or []}
            r["playground"] = {"look_around", "end_session", "record_experiment"} <= names
            pending[r["model"]].append(r)
            rows.append(r)
        elif kind == "prediction":
            model = payload.get("model")
            q = pending.get(model)
            if not q:
                continue
            r = q.pop(0)
            u = payload.get("usage") or {}
            ch = (payload.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            r.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                     reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     finish_reason=ch.get("finish_reason"), content_len=len(msg.get("content") or ""),
                     reasoning_len=len(msg.get("reasoning_content") or ""),
                     tool_calls=[(tc.get("function") or {}).get("name") for tc in msg.get("tool_calls") or []])
    rows = [r for r in rows if r["playground"]]

    # Attribute to sessions (by time window and the model the session was actually running).
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    sess = [dict(zip(("id", "agent_id", "model", "started_at", "ended_at"), x)) for x in
            db.execute("SELECT id, agent_id, model, started_at, ended_at FROM sessions WHERE mode='exploration'")]
    for r in rows:
        cands = [s for s in sess if s["model"] == r["model"] and s["started_at"] - 5 <= r["t"] <= (s["ended_at"] or 9e12) + 5]
        r["session_id"] = cands[-1]["id"] if len(cands) >= 1 else None
        r["session_ambiguous"] = len(cands) > 1
        # Context window in force: the latest load of a model file whose name matches.
        stem = (r["model"] or "").lower().replace(" ", "")
        ctx = None
        for ld in loads:
            if ld["t"] is not None and ld["t"] <= r["t"]:
                p = Path(ld["path"]).parent.name.lower().replace(" ", "")
                if p == stem:
                    ctx = ld
        r["n_ctx_slot"] = ctx["n_ctx_slot"] if ctx else None
        r["n_slots"] = ctx["n_slots"] if ctx else None

    with (out / "lmstudio_requests.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out / "lmstudio_loads.json").write_text(json.dumps(loads, indent=1), encoding="utf-8")

    # Summary per session.
    by = defaultdict(list)
    for r in rows:
        by[r["session_id"]].append(r)
    summary = []
    for sid, rs in sorted(by.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        pt = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens")]
        first = rs[0]
        def share(key):
            tot = sum(r["total_chars"] for r in rs) or 1
            return round(100 * sum(r[key] for r in rs) / tot, 1)
        summary.append({
            "session_id": sid, "model": first["model"], "requests": len(rs),
            "n_ctx_slot": sorted({r["n_ctx_slot"] for r in rs if r["n_ctx_slot"]}),
            "n_slots": sorted({r["n_slots"] for r in rs if r["n_slots"]}),
            "first_request_prompt_tokens": first.get("prompt_tokens"),
            "first_request_chars": {k: first[k] for k in ("system", "tools", "opening")},
            "prompt_tokens_median": statistics.median(pt) if pt else None, "prompt_tokens_max": max(pt) if pt else None,
            "char_share_pct": {k: share(k) for k in ("system", "tools", "opening", "assistant_text",
                                                      "assistant_tool_calls", "tool_results", "nudges")},
            "turns_with_empty_content": sum(1 for r in rs if r.get("content_len") == 0),
            "turns_with_reasoning": sum(1 for r in rs if (r.get("reasoning_len") or 0) > 0),
            "turns_without_tool_call": sum(1 for r in rs if r.get("finish_reason") is not None and not r.get("tool_calls")),
        })
    (out / "lmstudio_session_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"files {len(files)} | playground requests {len(rows)} | with usage {sum(1 for r in rows if r.get('prompt_tokens'))}"
          f" | model loads {len(loads)} | overflow errors {len(overflows)}")
    for s in summary:
        print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
