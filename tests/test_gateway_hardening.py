"""Regression tests for the 13 findings from the adversarial gateway review (stages A+B).

Each test pins one confirmed defect so it cannot regress: non-JSON 200 handling, the atomic
idempotency guard, owner-guarded release, upstream-error scrubbing, streaming error status,
cancel/finish ordering, defensive body parsing, optional auth, and the resource-tag honesty.
"""
from __future__ import annotations

import time

import httpx
import pytest
from fastapi.testclient import TestClient

from playground.config import load_config
from playground.dashboard import create_app
from playground.gateway.providers import Provider
from playground.gateway.resources import gpus
from playground.gateway.router import Route
from playground.gateway.tracking import RequestMonitor, redact_text, tokens_from_response

LM_BASE = "http://127.0.0.1:1234/v1"


class FakeResp:
    def __init__(self, status=200, payload=None, text="", raise_json=False):
        self.status_code = status
        self._payload = payload
        self.text = text
        self._raise = raise_json

    def json(self):
        if self._raise:
            raise ValueError("no json")
        return self._payload


class FakeStream:
    def __init__(self, status=200, chunks=None, body=b""):
        self.status_code = status
        self._chunks = chunks or []
        self._body = body
        self.closed = False

    def read(self):
        return self._body

    def iter_raw(self):
        yield from self._chunks

    def close(self):
        self.closed = True


def _completion(usage=True):
    d = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}
    if usage:
        d["usage"] = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    return d


@pytest.fixture
def app_db(cfg, db, monkeypatch):
    monkeypatch.setattr("playground.gateway.providers.list_openai_models",
                        lambda base, *a, **k: [{"id": "test-model", "state": "loaded"}] if "1234" in base else [])
    return create_app(cfg, db), db


# ---------- finding 1 & 3: a 200 with a non-JSON / non-object body must not strand the row
def test_non_json_200_becomes_502_not_stranded(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    monkeypatch.setattr(gw.client, "post", lambda url, json, headers: FakeResp(200, raise_json=True, text="<html>"))
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": "test-model", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 502
    row = db.gateway_requests()[0]
    assert row["status"] == "error" and row["http_status"] == 502      # never left 'running'


def test_non_object_json_200_handled(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    monkeypatch.setattr(gw.client, "post", lambda url, json, headers: FakeResp(200, payload=["not", "an", "object"]))
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": "test-model", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 502
    assert db.gateway_requests()[0]["status"] == "error"


def test_tokens_from_response_tolerates_non_dict():
    assert tokens_from_response(["x"], {"messages": [{"content": "abcd"}]}, "ab")["token_source"] == "estimated"


# ---------- finding 2: atomic claim-and-check idempotency guard
def test_claim_inflight_is_atomic(db):
    m = RequestMonitor(db)
    assert m.claim_inflight("k", "req_a") is None       # first claim wins
    assert m.claim_inflight("k", "req_b") == "req_a"     # second sees the owner (would be 409)


def test_clear_inflight_is_owner_guarded(db):
    m = RequestMonitor(db)
    m.claim_inflight("k", "req_a")
    m.clear_inflight("k", "req_b")                       # a different request must not evict A
    assert m.duplicate_of("k") == "req_a"
    m.clear_inflight("k", "req_a")                       # the owner can
    assert m.duplicate_of("k") is None


# ---------- finding 4 & security: upstream error text is scrubbed before storage/return
def test_redact_text_scrubs_credentials():
    s = redact_text('error: Authorization: Bearer sk-abc123SECRET and api_key="sk-zzz999"')
    assert "sk-abc123SECRET" not in s and "sk-zzz999" not in s and "redacted" in s


def test_upstream_error_body_scrubbed(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    leaky = {"error": {"message": "bad key Bearer sk-LEAK1234567 rejected"}}
    monkeypatch.setattr(gw.client, "post", lambda url, json, headers: FakeResp(400, payload=leaky))
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": "test-model", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400
    row = db.gateway_request(db.gateway_requests()[0]["request_id"])
    assert "sk-LEAK1234567" not in str(row.get("response"))        # stored body scrubbed


# ---------- finding 5: optional gateway auth (open-relay guard)
def test_gateway_auth_enforced_when_configured(cfg, db, monkeypatch):
    monkeypatch.setenv("GW_TOKEN", "s3cret")
    c = {**cfg, "gateway": {"auth_token_env": "GW_TOKEN"}, "server": {**cfg["server"], "host": "0.0.0.0"}}
    client = TestClient(create_app(c, db))
    assert client.post("/v1/chat/completions", json={"model": "m", "messages": []}).status_code == 401
    # a request bearing the token passes auth (it then fails later for an unknown model, not 401)
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": []},
                    headers={"X-Gateway-Key": "s3cret"})
    assert r.status_code != 401


# ---------- finding 6 & 7: streaming provider error surfaces as a real HTTP error, not 200
def test_streaming_provider_error_is_real_status(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    monkeypatch.setattr(gw.client, "send",
                        lambda req, stream: FakeStream(status=503, body=b'{"error":{"message":"overloaded"}}'))
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": "test-model", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 503                      # not a 200 event-stream hiding an error
    assert db.gateway_requests()[0]["status"] == "error"


def test_streaming_success_passthrough(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    monkeypatch.setattr(gw.client, "send", lambda req, stream: FakeStream(status=200, chunks=chunks))
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": "test-model", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and b"[DONE]" in r.content
    row = db.gateway_requests()[0]
    assert row["status"] == "done" and row["token_source"] == "estimated"


# ---------- finding 8 & 9: cancel vs completion ordering
def test_cancel_is_atomic_and_finish_wont_overwrite(db):
    db.gateway_request_create(request_id="r1", status="running", model="m")
    assert db.gateway_request_cancel("r1") == {"found": True, "cancelled": True}
    assert db.gateway_request("r1")["status"] == "cancelled"
    db.gateway_request_finish("r1", status="done", http_status=200)      # must NOT resurrect
    assert db.gateway_request("r1")["status"] == "cancelled"
    # cancelling an already-terminal row: found but nothing changed
    db.gateway_request_create(request_id="r2", status="done", model="m")
    assert db.gateway_request_cancel("r2") == {"found": True, "cancelled": False}
    assert db.gateway_request_cancel("nope") == {"found": False, "cancelled": False}


def test_finish_rechecks_cancellation(app_db):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    rm = app.state.gateway["request_monitor"]
    db.gateway_request_create(request_id="rc", status="running", model="m", started_at=time.time())
    rm._cancel.add("rc")                                 # cancel arrived mid-flight
    route = Route(provider=Provider(name="lmstudio", kind="openai_compat", base_url=LM_BASE),
                  base_url=LM_BASE, model="m", api_key=None, api_key_env=None)
    out = gw._finish("rc", {"messages": []}, route, FakeResp(200, _completion(True)), time.time())
    assert out.status_code == 499                        # honoured, not overwritten with done
    assert db.gateway_request("rc")["status"] == "cancelled"


# ---------- finding 10: malformed body -> OpenAI-shaped 400 (not FastAPI 422)
def test_malformed_body_is_openai_400(app_db):
    app, _ = app_db
    client = TestClient(app)
    r = client.post("/v1/chat/completions", content=b"not json at all",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r2 = client.post("/v1/chat/completions", json=["an", "array"])     # valid JSON, not an object
    assert r2.status_code == 400


# ---------- finding 11 & 12: resource-tag honesty
def test_gpu_utilization_tagged_estimated_and_vram_clamped():
    for g in gpus():
        u = g["utilization_percent"]
        # a derived composite is never labelled 'measured'
        assert u["source"] in ("estimated", "unknown")
        if g.get("vram_percent") and g["vram_percent"]["value"] is not None:
            assert 0.0 <= g["vram_percent"]["value"] <= 100.0
