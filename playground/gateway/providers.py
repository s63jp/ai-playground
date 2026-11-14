"""Provider registry and capability discovery.

A *provider* is a distinct model-serving endpoint: an OpenAI-compatible server
(LM Studio, Unsloth, Ollama, vLLM, an OpenAI-compatible cloud) or the Anthropic
Messages API. Providers are derived from the existing `adapters` section of the
config (so nothing is duplicated and existing provider-neutral behaviour is kept)
and may be extended at runtime from the `providers` table.

Design rules honoured here:
  * Capabilities are DISCOVERED, never assumed. Two OpenAI-compatible servers can
    differ (LM Studio reports per-model load state via /api/v0/models; a bare
    llama.cpp server does not). Each probe records what it actually saw.
  * Secrets never leave this module. Only the NAME of the api-key environment
    variable and whether it is currently set are ever returned; the key value is
    not stored on the object and not serialised.
  * Nothing raises on a dead provider: an unreachable endpoint yields a health
    record with reachable=False, never an exception, so one bad provider cannot
    break discovery of the others.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from ..agents.adapters import list_openai_models

# Provider kinds that actually serve models over a network (the gateway's concern).
# scripted / random / neural / external adapters are agent brains, not providers.
NETWORK_KINDS = ("openai_compat", "anthropic")
_HEALTH_TTL = 5.0  # seconds; dashboards poll, so cache probes briefly


def _longest_common_prefix(names: list[str]) -> str:
    """LCP of a set of strings equals the LCP of its lexicographic min and max."""
    if not names:
        return ""
    lo, hi = min(names), max(names)
    i = 0
    while i < len(lo) and i < len(hi) and lo[i] == hi[i]:
        i += 1
    return lo[:i]


def _derive_name(adapter_names: list[str], base_url: str) -> str:
    """A stable, friendly provider name for a group of adapters sharing an endpoint."""
    if len(adapter_names) == 1:
        return adapter_names[0]
    prefix = _longest_common_prefix(sorted(adapter_names)).rstrip("_-")
    if len(prefix) >= 3:
        return prefix
    host = urlparse(base_url).hostname or "provider"
    return host.replace(".", "_")


@dataclass
class Provider:
    """One model-serving endpoint and what has been discovered about it."""
    name: str
    kind: str                                   # "openai_compat" | "anthropic"
    base_url: str
    api_key_env: str | None = None
    default_models: list[str] = field(default_factory=list)
    adapters: list[str] = field(default_factory=list)   # config adapter names mapped here
    source: str = "config"                      # "config" | "runtime"
    enabled: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- secrets
    def key_present(self) -> bool:
        """Whether the api-key env var this provider names is currently set.

        Never returns or logs the key itself — only that one is (or is not) present.
        """
        return bool(self.api_key_env and os.environ.get(self.api_key_env))

    # ---------------------------------------------------------------- probes
    def _is_lmstudio(self, timeout: float = 3.0) -> bool:
        """LM Studio exposes a richer /api/v0/models with load state; detect it."""
        if self.kind != "openai_compat":
            return False
        root = self.base_url.rstrip("/")
        host = root[:-3] if root.endswith("/v1") else root
        try:
            r = httpx.get(f"{host}/api/v0/models", timeout=timeout)
            return r.status_code == 200 and isinstance(r.json().get("data"), list)
        except (httpx.HTTPError, ValueError):
            return False

    def discover_models(self, timeout: float = 4.0, use_cache: bool = True) -> list[dict]:
        """Every model the endpoint is currently offering, normalised.

        openai_compat: queries the live server (LM Studio's /api/v0/models first, then
        the standard /v1/models) via the shared list_openai_models helper.
        anthropic: cannot be listed without a paid call, so returns the configured
        default model(s) with discovered=False.
        Returns [] (never raises) when the server is unreachable.
        """
        now = time.time()
        if use_cache and (c := self._cache.get("models")) and now - c["ts"] < _HEALTH_TTL:
            return c["data"]
        out: list[dict]
        if self.kind == "anthropic":
            out = [{"id": m, "state": None, "context": None, "discovered": False,
                    "provider": self.name} for m in self.default_models]
        else:
            raw = list_openai_models(self.base_url, self.api_key_env, timeout=timeout)
            out = [{"id": m.get("id"),
                    "state": m.get("state"),
                    "context": m.get("loaded_context_length") or m.get("max_context_length"),
                    "max_context": m.get("max_context_length"),
                    "loaded_context": m.get("loaded_context_length"),
                    "type": m.get("type"),
                    "discovered": True,
                    "provider": self.name}
                   for m in raw if m.get("id")]
        self._cache["models"] = {"ts": now, "data": out}
        return out

    def loaded_models(self) -> list[str]:
        """Model ids the endpoint reports as currently loaded.

        Only meaningful where the server reports load state (LM Studio). Where it does
        not, returns [] and capabilities()['report_load_state'] is False — callers must
        not read an empty list as "nothing loaded" without checking that flag.
        """
        return [m["id"] for m in self.discover_models() if m.get("state") == "loaded"]

    def capabilities(self) -> dict:
        """What this specific endpoint can actually do, by observation.

        Nothing here is assumed from the kind alone: report_load_state and
        report_context_window reflect what the live server returned.
        """
        models = self.discover_models()
        reports_state = any(m.get("state") is not None for m in models)
        reports_ctx = any(m.get("context") is not None for m in models)
        return {
            "discover_models": self.kind == "openai_compat",
            "report_load_state": reports_state,
            "report_context_window": reports_ctx,
            "token_usage_reported": True,       # OpenAI/Anthropic both return a usage block
            "load_unload": False,               # not implemented yet (honest; see follow-up)
            "is_lmstudio": self._is_lmstudio() if self.kind == "openai_compat" else False,
        }

    def health(self, timeout: float = 4.0, use_cache: bool = True) -> dict:
        """A quick reachability + latency probe. Never raises."""
        now = time.time()
        if use_cache and (c := self._cache.get("health")) and now - c["ts"] < _HEALTH_TTL:
            return c["data"]
        rec: dict[str, Any] = {"name": self.name, "reachable": False, "latency_ms": None,
                               "model_count": 0, "error": None}
        if self.kind == "anthropic":
            # Pinging Anthropic costs money; report credential presence instead of a live call.
            rec.update({"reachable": None, "note": "no-cost health check unavailable for anthropic",
                        "key_present": self.key_present()})
        else:
            root = self.base_url.rstrip("/")
            t0 = time.time()
            try:
                headers = {}
                if self.key_present():
                    headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
                r = httpx.get(f"{root}/models", headers=headers, timeout=timeout)
                rec["latency_ms"] = round((time.time() - t0) * 1000, 1)
                rec["reachable"] = r.status_code == 200
                if r.status_code == 200:
                    try:
                        rec["model_count"] = len(r.json().get("data") or [])
                    except ValueError:
                        rec["model_count"] = 0
                else:
                    rec["error"] = f"HTTP {r.status_code}"
            except (httpx.HTTPError, OSError) as exc:
                rec["latency_ms"] = round((time.time() - t0) * 1000, 1)
                rec["error"] = f"{type(exc).__name__}: {exc}"
        self._cache["health"] = {"ts": now, "data": rec}
        return rec

    # ---------------------------------------------------------------- serialisation
    def public_dict(self, include_models: bool = False, include_health: bool = False) -> dict:
        """Safe view for the API/dashboard. Contains NO secret — only api_key_env's
        name and whether a key is present."""
        d = {"name": self.name, "kind": self.kind, "base_url": self.base_url,
             "api_key_env": self.api_key_env, "key_present": self.key_present(),
             "default_models": list(self.default_models), "adapters": list(self.adapters),
             "source": self.source, "enabled": self.enabled}
        if include_health:
            d["health"] = self.health()
            d["capabilities"] = self.capabilities()
        if include_models:
            d["models"] = self.discover_models()
            d["loaded_models"] = self.loaded_models()
        return d


class ProviderRegistry:
    """All known providers: those derived from config adapters plus runtime additions.

    The registry is the single place that answers "what providers exist, what do they
    offer, and which one serves model X" for the gateway and the dashboard.
    """

    def __init__(self, cfg: dict, db: Any | None = None):
        self.cfg = cfg
        self.db = db
        self._providers: dict[str, Provider] = {}
        self.reload()

    # ---------------------------------------------------------------- build
    def reload(self) -> None:
        self._providers = self._from_config()
        for p in self._from_db():
            self._providers[p.name] = p            # runtime rows override config of same name

    def _from_config(self) -> dict[str, Provider]:
        table = self.cfg.get("adapters", {}) or {}
        groups: dict[tuple, list[tuple[str, dict]]] = {}
        for name, spec in table.items():
            kind = spec.get("type")
            if kind not in NETWORK_KINDS:
                continue
            key = (kind, spec.get("base_url", ""), spec.get("api_key_env"))
            groups.setdefault(key, []).append((name, spec))
        providers: dict[str, Provider] = {}
        for (kind, base_url, api_key_env), members in groups.items():
            names = [n for n, _ in members]
            pname = _derive_name(names, base_url)
            default_models = []
            for _, spec in members:
                m = spec.get("model")
                if m and m not in default_models:
                    default_models.append(m)
            providers[pname] = Provider(
                name=pname, kind=kind,
                base_url=base_url or ("https://api.anthropic.com" if kind == "anthropic" else ""),
                api_key_env=api_key_env, default_models=default_models,
                adapters=sorted(names), source="config")
        return providers

    def _from_db(self) -> list[Provider]:
        if self.db is None or not hasattr(self.db, "list_providers"):
            return []
        out = []
        for row in self.db.list_providers():
            out.append(Provider(
                name=row["name"], kind=row["kind"], base_url=row.get("base_url") or "",
                api_key_env=row.get("api_key_env"),
                default_models=[row["default_model"]] if row.get("default_model") else [],
                adapters=[], source="runtime", enabled=bool(row.get("enabled", 1))))
        return out

    # ---------------------------------------------------------------- queries
    def list(self, enabled_only: bool = False) -> list[Provider]:
        ps = [p for p in self._providers.values() if p.enabled or not enabled_only]
        return sorted(ps, key=lambda p: p.name)

    def get(self, name: str) -> Provider | None:
        return self._providers.get(name)

    def all_models(self, discover: bool = True) -> list[dict]:
        """Every model across every enabled provider, attributed to its provider.

        Deduplicates by (provider, id). Unreachable providers simply contribute
        nothing — discovery of the rest is unaffected.
        """
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for p in self.list(enabled_only=True):
            models = p.discover_models() if discover else [
                {"id": m, "provider": p.name, "discovered": False} for m in p.default_models]
            for m in models:
                k = (p.name, m.get("id"))
                if m.get("id") and k not in seen:
                    seen.add(k)
                    out.append(m)
        return out

    def provider_for_model(self, model_id: str, prefer_loaded: bool = True) -> Provider | None:
        """Choose which provider should serve `model_id`.

        Routing policy (used by the gateway in stage B): among providers that offer the
        model, prefer one that has it LOADED (ready, no load latency) and reachable;
        then any reachable provider that lists it; then a provider that names it as a
        configured default even if discovery is unavailable. Returns None if no enabled
        provider offers the model.
        """
        offering: list[tuple[int, Provider]] = []
        for p in self.list(enabled_only=True):
            score = 0
            try:
                models = {m.get("id"): m for m in p.discover_models()}
            except Exception:  # noqa: BLE001 - discovery must never break routing
                models = {}
            if model_id in models:
                score = 2
                if prefer_loaded and models[model_id].get("state") == "loaded":
                    score = 3
            elif model_id in p.default_models:
                score = 1
            if score:
                offering.append((score, p))
        if not offering:
            return None
        offering.sort(key=lambda sp: sp[0], reverse=True)
        return offering[0][1]
