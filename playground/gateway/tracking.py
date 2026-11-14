"""Request observability: redaction, token accounting and the in-flight registry.

Everything that must never leak a secret or an unbounded prompt into the request log
lives here, in one place:

  * redact()  - strips credential-shaped keys and truncates long strings before any
                request/response body is stored or returned to the monitor.
  * token accounting - prefers the provider's own `usage` block (authoritative) and
                falls back to a clearly-labelled character-based estimate.
  * RequestMonitor - owns the DB writes, the idempotency guard against accidental
                duplicate submissions, and cooperative cancellation flags.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

# Keys whose values must never be written to the log, matched case-insensitively.
_SECRET_KEYS = {"authorization", "api_key", "apikey", "api-key", "x-api-key", "key",
                "token", "access_token", "password", "secret", "bearer"}
_MAX_STR = 2000
_MAX_LIST = 64


def redact(obj: Any, max_str: int = _MAX_STR, _depth: int = 0) -> Any:
    """Return a copy safe to store/serve: credential-shaped keys dropped, long strings
    truncated, long lists capped. Never mutates the input."""
    if _depth > 12:
        return "…[too deep]"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in _SECRET_KEYS:
                out[k] = "***redacted***"
            else:
                out[k] = redact(v, max_str, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        items = [redact(v, max_str, _depth + 1) for v in list(obj)[:_MAX_LIST]]
        if len(obj) > _MAX_LIST:
            items.append(f"…[{len(obj) - _MAX_LIST} more items omitted]")
        return items
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + f"…[truncated {len(obj) - max_str} chars]"
    return obj


def _char_len(messages: Any) -> int:
    total = 0
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):          # content parts (OpenAI vision-style)
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"])
    return total


def estimate_prompt_tokens(body: dict) -> int:
    """~4 chars/token over the messages. Deliberately rough; always labelled estimated."""
    return max(1, _char_len(body.get("messages")) // 4)


def estimate_completion_tokens(text: str | None) -> int:
    return max(0, len(text or "") // 4)


def tokens_from_response(data: dict, request_body: dict, completion_text: str | None) -> dict:
    """Prefer the provider's reported usage; otherwise estimate and say so.

    Returns {prompt_tokens, completion_tokens, total_tokens, token_source}.
    token_source is 'reported' only when the provider supplied a usage block.
    """
    usage = (data or {}).get("usage") or {}
    pt, ct, tt = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
    if pt is not None or ct is not None or tt is not None:
        pt = pt if pt is not None else 0
        ct = ct if ct is not None else 0
        return {"prompt_tokens": pt, "completion_tokens": ct,
                "total_tokens": tt if tt is not None else pt + ct, "token_source": "reported"}
    pt = estimate_prompt_tokens(request_body)
    ct = estimate_completion_tokens(completion_text)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
            "token_source": "estimated"}


class RequestMonitor:
    """DB-backed request lifecycle + idempotency guard + cancellation flags.

    One instance per gateway (held on app.state.gateway). Thread-safe: requests run in
    FastAPI's threadpool concurrently.
    """

    def __init__(self, db: Any):
        self.db = db
        self._lock = threading.RLock()
        self._inflight: dict[str, str] = {}      # idempotency_key -> request_id
        self._cancel: set[str] = set()           # request_ids asked to cancel

    @staticmethod
    def new_request_id() -> str:
        return "req_" + uuid.uuid4().hex[:20]

    # -------------------------------------------------- idempotency (duplicate guard)
    def duplicate_of(self, idempotency_key: str | None) -> str | None:
        if not idempotency_key:
            return None
        with self._lock:
            return self._inflight.get(idempotency_key)

    def mark_inflight(self, idempotency_key: str | None, request_id: str) -> None:
        if idempotency_key:
            with self._lock:
                self._inflight[idempotency_key] = request_id

    def clear_inflight(self, idempotency_key: str | None) -> None:
        if idempotency_key:
            with self._lock:
                self._inflight.pop(idempotency_key, None)

    # -------------------------------------------------- cancellation
    def request_cancel(self, request_id: str) -> bool:
        with self._lock:
            self._cancel.add(request_id)
        row = self.db.gateway_request(request_id)
        if row is None:
            return False
        if row["status"] in ("queued", "running"):
            self.db.gateway_request_update(request_id, status="cancelled", cancelled=1,
                                           ended_at=time.time())
        return True

    def is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancel

    # -------------------------------------------------- lifecycle writes
    def create(self, **fields) -> None:
        self.db.gateway_request_create(**fields)

    def update(self, request_id: str, **fields) -> None:
        self.db.gateway_request_update(request_id, **fields)
