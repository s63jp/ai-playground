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


def redact_text(text: str | None, limit: int = _MAX_STR) -> str | None:
    """Scrub a free-form string (e.g. an upstream provider error body) before it is stored
    or returned: strip bearer tokens and api-key-shaped secrets, then truncate. Provider
    error bodies can echo the Authorization header or a key, so they must never be logged raw.
    """
    if not text:
        return text
    import re
    scrubbed = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer ***redacted***", text)
    scrubbed = re.sub(r"\bsk-[A-Za-z0-9._\-]{6,}", "sk-***redacted***", scrubbed)
    scrubbed = re.sub(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)[A-Za-z0-9._\-]{6,}", r"\1***redacted***", scrubbed)
    if len(scrubbed) > limit:
        scrubbed = scrubbed[:limit] + f"…[truncated {len(scrubbed) - limit} chars]"
    return scrubbed


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
    if isinstance(obj, str):
        # Scrub credentials that appear inside a string VALUE (e.g. a provider error message
        # echoing "Bearer sk-…"), not just those under a credential-named key, then truncate.
        return redact_text(obj, max_str)
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
    if not isinstance(data, dict):     # a provider may return a non-object JSON (list/string)
        data = {}
    usage = data.get("usage") or {}
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
    def claim_inflight(self, idempotency_key: str | None, request_id: str) -> str | None:
        """Atomic check-and-claim. If a request is already in flight for this key, returns
        its request_id (the caller should reject as a duplicate). Otherwise claims the key for
        request_id and returns None. One lock acquisition covers both the check and the claim,
        so two racing duplicate submissions cannot both pass."""
        if not idempotency_key:
            return None
        with self._lock:
            existing = self._inflight.get(idempotency_key)
            if existing is not None:
                return existing
            self._inflight[idempotency_key] = request_id
            return None

    def duplicate_of(self, idempotency_key: str | None) -> str | None:
        if not idempotency_key:
            return None
        with self._lock:
            return self._inflight.get(idempotency_key)

    def mark_inflight(self, idempotency_key: str | None, request_id: str) -> None:
        if idempotency_key:
            with self._lock:
                self._inflight[idempotency_key] = request_id

    def clear_inflight(self, idempotency_key: str | None, request_id: str | None = None) -> None:
        """Release the key, but only if this request still owns it — so a completing request
        cannot evict a *different* in-flight request that reused the key after it."""
        if not idempotency_key:
            return
        with self._lock:
            if request_id is None or self._inflight.get(idempotency_key) == request_id:
                self._inflight.pop(idempotency_key, None)

    # -------------------------------------------------- cancellation
    def request_cancel(self, request_id: str) -> bool:
        """Mark a request cancelled. The DB transition is a single status-guarded UPDATE so it
        cannot race with a concurrent completion (returns whether the row exists)."""
        with self._lock:
            self._cancel.add(request_id)
        return self.db.gateway_request_cancel(request_id)["found"]

    def is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancel

    # -------------------------------------------------- lifecycle writes
    def create(self, **fields) -> None:
        self.db.gateway_request_create(**fields)

    def update(self, request_id: str, **fields) -> None:
        self.db.gateway_request_update(request_id, **fields)

    def finish(self, request_id: str, **fields) -> None:
        """Terminal write that will not overwrite a row already moved to 'cancelled'."""
        self.db.gateway_request_finish(request_id, **fields)
