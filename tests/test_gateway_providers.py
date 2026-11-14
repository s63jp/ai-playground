"""Stage A: provider registry, capability discovery and the gateway discovery endpoints.

Unit tests are fully offline (httpx / model listing are monkeypatched). Tests marked
`live` run only when LM Studio is actually reachable and are skipped otherwise, so the
suite never depends on a running server but does verify the real integration when present.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from playground.config import load_config
from playground.dashboard import create_app
from playground.gateway.providers import Provider, ProviderRegistry, _derive_name, _longest_common_prefix

LM_BASE = "http://127.0.0.1:1234/v1"


def _lmstudio_up() -> bool:
    try:
        return httpx.get(f"{LM_BASE}/models", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001
        return False


live = pytest.mark.skipif(not _lmstudio_up(), reason="LM Studio not reachable at 127.0.0.1:1234")


# --------------------------------------------------------------------------- naming
def test_longest_common_prefix():
    assert _longest_common_prefix(["lmstudio", "lmstudio_unsloth4b", "lmstudio_qwen9b"]) == "lmstudio"
    assert _longest_common_prefix([]) == ""
    assert _longest_common_prefix(["solo"]) == "solo"


def test_derive_name_groups_lmstudio():
    names = ["lmstudio", "lmstudio_unsloth4b", "lmstudio_qwen9b"]
    assert _derive_name(names, LM_BASE) == "lmstudio"
    assert _derive_name(["unsloth"], "http://127.0.0.1:8888/v1") == "unsloth"
    # no usable common prefix -> falls back to host
    assert _derive_name(["a_x", "b_y"], "http://10.0.0.5:9/v1") == "10_0_0_5"


# --------------------------------------------------------------------------- registry build
def test_registry_groups_adapters_by_endpoint():
    cfg = load_config()
    reg = ProviderRegistry(cfg)
    names = {p.name for p in reg.list()}
    # the ~12 lmstudio_* adapters collapse to a single provider at one endpoint
    assert "lmstudio" in names
    lm = reg.get("lmstudio")
    assert lm.base_url == LM_BASE
    assert lm.kind == "openai_compat"
    assert len(lm.adapters) >= 5           # many config adapters map here
    # anthropic provider exists and is a different kind
    assert any(p.kind == "anthropic" for p in reg.list())


def test_registry_merges_runtime_providers(db):
    cfg = load_config()
    db.upsert_provider("my-vllm", "openai_compat", base_url="http://127.0.0.1:9000/v1",
                       default_model="mymodel")
    reg = ProviderRegistry(cfg, db)
    p = reg.get("my-vllm")
    assert p is not None and p.source == "runtime"
    assert p.base_url == "http://127.0.0.1:9000/v1"
    assert db.delete_provider("my-vllm") == 1


# --------------------------------------------------------------------------- discovery (mocked)
def test_discover_models_normalises(monkeypatch):
    fake = [{"id": "m-a", "state": "loaded", "loaded_context_length": 4096, "max_context_length": 8192},
            {"id": "m-b", "state": "not-loaded", "max_context_length": 2048}]
    monkeypatch.setattr("playground.gateway.providers.list_openai_models", lambda *a, **k: fake)
    p = Provider(name="x", kind="openai_compat", base_url=LM_BASE)
    models = p.discover_models(use_cache=False)
    assert {m["id"] for m in models} == {"m-a", "m-b"}
    a = next(m for m in models if m["id"] == "m-a")
    assert a["state"] == "loaded" and a["context"] == 4096 and a["discovered"] is True
    assert p.loaded_models() == ["m-a"]
    caps = p.capabilities()
    assert caps["report_load_state"] is True and caps["report_context_window"] is True


def test_discover_models_empty_when_unreachable(monkeypatch):
    monkeypatch.setattr("playground.gateway.providers.list_openai_models", lambda *a, **k: [])
    p = Provider(name="x", kind="openai_compat", base_url="http://127.0.0.1:1/v1")
    assert p.discover_models(use_cache=False) == []
    assert p.loaded_models() == []


def test_anthropic_discovery_uses_defaults_without_network():
    p = Provider(name="claude", kind="anthropic", base_url="https://api.anthropic.com",
                 default_models=["claude-sonnet-4-5"])
    models = p.discover_models()
    assert models == [{"id": "claude-sonnet-4-5", "state": None, "context": None,
                       "discovered": False, "provider": "claude"}]
    assert p.capabilities()["discover_models"] is False


# --------------------------------------------------------------------------- health (mocked)
def test_health_reachable_and_down(monkeypatch):
    class R:
        status_code = 200
        def json(self):
            return {"data": [{"id": "m-a"}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    p = Provider(name="x", kind="openai_compat", base_url=LM_BASE)
    h = p.health(use_cache=False)
    assert h["reachable"] is True and h["model_count"] == 1 and h["latency_ms"] is not None

    def boom(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", boom)
    h2 = p.health(use_cache=False)
    assert h2["reachable"] is False and "ConnectError" in h2["error"]


# --------------------------------------------------------------------------- secret safety
def test_public_dict_never_leaks_key(monkeypatch):
    monkeypatch.setenv("SECRET_KEY_ENV", "super-secret-value")
    p = Provider(name="x", kind="openai_compat", base_url=LM_BASE, api_key_env="SECRET_KEY_ENV")
    d = p.public_dict(include_models=False, include_health=False)
    assert d["api_key_env"] == "SECRET_KEY_ENV"
    assert d["key_present"] is True
    assert "super-secret-value" not in str(d)


# --------------------------------------------------------------------------- routing
def test_provider_for_model_prefers_loaded(monkeypatch):
    def models_for(base):
        if "1234" in base:
            return [{"id": "shared", "state": "loaded"}]
        return [{"id": "shared", "state": "not-loaded"}]
    monkeypatch.setattr("playground.gateway.providers.list_openai_models",
                        lambda base, *a, **k: models_for(base))
    reg = ProviderRegistry({"adapters": {
        "loaded_srv": {"type": "openai_compat", "base_url": LM_BASE, "model": "shared"},
        "cold_srv": {"type": "openai_compat", "base_url": "http://127.0.0.1:5555/v1", "model": "shared"},
    }})
    chosen = reg.provider_for_model("shared")
    assert chosen is not None and chosen.base_url == LM_BASE      # the loaded one wins
    assert reg.provider_for_model("nonexistent") is None


# --------------------------------------------------------------------------- endpoints (offline)
def test_gateway_endpoints_mounted(cfg, db):
    app = create_app(cfg, db)
    client = TestClient(app)
    # health=false keeps this offline (no provider probing)
    r = client.get("/api/gateway/providers", params={"health": "false", "models": "false"})
    assert r.status_code == 200
    body = r.json()
    assert any(p["name"] == "lmstudio" for p in body["providers"])
    # no secret values anywhere in the serialised providers
    assert "key_present" in body["providers"][0]
    h = client.get("/api/gateway/health")
    assert h.status_code == 200 and h.json()["ok"] is True


# --------------------------------------------------------------------------- live LM Studio
@live
def test_live_discovery_and_v1_models(cfg, db):
    reg = ProviderRegistry(cfg, db)
    models = reg.get("lmstudio").discover_models(use_cache=False)
    assert models, "LM Studio returned no models"
    ids = {m["id"] for m in models}
    assert ids, "expected at least one model id"
    # the unified OpenAI-compatible endpoint lists them
    client = TestClient(create_app(cfg, db))
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert any(m["owned_by"] == "lmstudio" for m in data["data"])
