"""Stage B: the /v1/chat/completions gateway, request monitor, failover and cancellation.

Routing discovery (list_openai_models) and the outbound provider call (the gateway's httpx
client) are mocked so these run offline and deterministically. A live test exercises a real
round-trip through LM Studio when it is reachable.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from playground.config import load_config
from playground.dashboard import create_app
from playground.gateway.tracking import RequestMonitor, redact, tokens_from_response

LM_BASE = "http://127.0.0.1:1234/v1"


def _lmstudio_up() -> bool:
    try:
        return httpx.get(f"{LM_BASE}/models", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001
        return False


live = pytest.mark.skipif(not _lmstudio_up(), reason="LM Studio not reachable at 127.0.0.1:1234")


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def _completion(usage=True):
    data = {"id": "chatcmpl-x", "object": "chat.completion", "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"},
                         "finish_reason": "stop"}]}
    if usage:
        data["usage"] = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
    return data


@pytest.fixture
def app_db(cfg, db, monkeypatch):
    """App whose provider discovery reports one loaded model, with no real network."""
    monkeypatch.setattr("playground.gateway.providers.list_openai_models",
                        lambda base, *a, **k: [{"id": "test-model", "state": "loaded"}] if "1234" in base else [])
    app = create_app(cfg, db)
    return app, db


# --------------------------------------------------------------------------- pure helpers
def test_tokens_prefer_reported_over_estimate():
    r = tokens_from_response(_completion(usage=True), {"messages": [{"content": "x" * 400}]}, "hi there")
    assert r == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14, "token_source": "reported"}


def test_tokens_estimate_when_absent():
    body = {"messages": [{"content": "x" * 400}]}
    r = tokens_from_response(_completion(usage=False), body, "y" * 40)
    assert r["token_source"] == "estimated"
    assert r["prompt_tokens"] == 100 and r["completion_tokens"] == 10   # 400/4, 40/4


def test_redact_strips_secrets_and_truncates():
    body = {"api_key": "sk-leak", "authorization": "Bearer sk-leak",
            "messages": [{"role": "user", "content": "z" * 5000}]}
    red = redact(body)
    assert red["api_key"] == "***redacted***" and red["authorization"] == "***redacted***"
    assert "sk-leak" not in str(red)
    assert "truncated" in red["messages"][0]["content"]


def test_request_monitor_duplicate_guard(db):
    m = RequestMonitor(db)
    assert m.duplicate_of("k1") is None
    m.mark_inflight("k1", "req_a")
    assert m.duplicate_of("k1") == "req_a"
    m.clear_inflight("k1")
    assert m.duplicate_of("k1") is None


# --------------------------------------------------------------------------- round-trip
def test_chat_roundtrip_records_and_attributes(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    monkeypatch.setattr(gw.client, "post", lambda url, json, headers: FakeResp(200, _completion(True)))
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"X-Playground-Agent": "agent_007", "X-Playground-App": "hermes"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi there"
    assert r.headers["x-request-id"].startswith("req_")
    assert r.headers["x-provider"] == "lmstudio"
    assert r.headers["x-token-source"] == "reported"
    rows = db.gateway_requests()
    assert len(rows) == 1
    row = rows[0]
    assert row["agent"] == "agent_007" and row["app"] == "hermes"
    assert row["status"] == "done" and row["total_tokens"] == 14 and row["token_source"] == "reported"
    assert row["latency_ms"] is not None
    # correlation id threads through
    full = db.gateway_request(row["request_id"])
    assert full["correlation_id"] == row["request_id"]
    assert full["response"] is not None            # redacted body kept for the inspector


def test_chat_identity_unknown_without_headers(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    monkeypatch.setattr(gw.client, "post", lambda url, json, headers: FakeResp(200, _completion(False)))
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    assert r.headers["x-token-source"] == "estimated"     # provider gave no usage
    row = db.gateway_requests()[0]
    assert row["agent"] == "unknown" and row["app"] == "unknown"


def test_unknown_model_is_404_and_logged(app_db):
    app, db = app_db
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    json={"model": "no-such-model", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"
    row = db.gateway_requests()[0]
    assert row["status"] == "error" and row["http_status"] == 404


def test_missing_model_is_400(app_db):
    app, _ = app_db
    r = TestClient(app).post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_duplicate_idempotency_key_rejected(app_db):
    app, _ = app_db
    rm = app.state.gateway["request_monitor"]
    rm.mark_inflight("dupe-key", "req_existing")       # pretend one is already in flight
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": "test-model", "messages": [{"role": "user", "content": "x"}]},
                             headers={"Idempotency-Key": "dupe-key"})
    assert r.status_code == 409
    assert r.json()["error"]["existing_request_id"] == "req_existing"


def test_cancel_endpoint(app_db):
    app, db = app_db
    rm = app.state.gateway["request_monitor"]
    db.gateway_request_create(request_id="req_run", status="running", model="test-model")
    r = TestClient(app).post("/api/gateway/requests/req_run/cancel")
    assert r.status_code == 200 and r.json()["cancelled"] is True
    assert db.gateway_request("req_run")["status"] == "cancelled"
    assert TestClient(app).post("/api/gateway/requests/nope/cancel").status_code == 404


def test_failover_to_second_provider(cfg, db, monkeypatch):
    # Two providers both offer the model; the first is unreachable, the second answers.
    monkeypatch.setattr("playground.gateway.providers.list_openai_models",
                        lambda base, *a, **k: [{"id": "shared", "state": "loaded"}])
    two = {"adapters": {
        "primary": {"type": "openai_compat", "base_url": LM_BASE, "model": "shared"},
        "backup": {"type": "openai_compat", "base_url": "http://127.0.0.1:5555/v1", "model": "shared"}}}
    merged = {**cfg, "adapters": two["adapters"]}
    app = create_app(merged, db)
    gw = app.state.gateway["chat_gateway"]

    def flaky_post(url, json, headers):
        if url.startswith(LM_BASE):
            raise httpx.ConnectError("primary down")
        return FakeResp(200, _completion(True))
    monkeypatch.setattr(gw.client, "post", flaky_post)
    r = TestClient(app).post("/v1/chat/completions",
                             json={"model": "shared", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    row = db.gateway_requests()[0]
    assert row["status"] == "done"
    assert row["provider"] == "backup"             # failed over to the reachable provider


def test_monitor_list_filters(app_db, monkeypatch):
    app, db = app_db
    gw = app.state.gateway["chat_gateway"]
    monkeypatch.setattr(gw.client, "post", lambda url, json, headers: FakeResp(200, _completion(True)))
    client = TestClient(app)
    for agent in ("agent_001", "agent_002"):
        client.post("/v1/chat/completions",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "x"}]},
                    headers={"X-Playground-Agent": agent})
    r = client.get("/api/gateway/requests", params={"agent": "agent_002"})
    assert r.status_code == 200
    body = r.json()
    assert all(x["agent"] == "agent_002" for x in body["requests"])
    assert body["stats"]["total"] >= 2


# --------------------------------------------------------------------------- live
@live
def test_live_chat_through_gateway(cfg, db):
    """Real round-trip through LM Studio: routed, answered, token-accounted, logged."""
    # Discover a loaded model (there must be one for a real completion).
    from playground.gateway.providers import ProviderRegistry
    loaded = ProviderRegistry(cfg).get("lmstudio").loaded_models()
    if not loaded:
        pytest.skip("no model loaded in LM Studio")
    model = loaded[0]
    client = TestClient(create_app(cfg, db))
    r = client.post("/v1/chat/completions",
                    json={"model": model, "max_tokens": 64,
                          "messages": [{"role": "user", "content": "Reply with the single word: ok"}]},
                    headers={"X-Playground-Agent": "test_live"})
    assert r.status_code == 200, r.text
    data = r.json()
    # The gateway's job is to route, preserve the OpenAI shape, and account tokens — not to
    # dictate what the model says (a reasoning model may put text elsewhere), so we assert the
    # envelope and the accounting, not the specific content string.
    assert "choices" in data and data["choices"]
    assert "message" in data["choices"][0]
    assert r.headers["x-provider"] == "lmstudio"
    row = db.gateway_request(r.headers["x-request-id"])
    assert row["status"] == "done" and row["agent"] == "test_live"
    assert row["token_source"] == "reported" and row["total_tokens"] > 0
    assert row["latency_ms"] > 0
