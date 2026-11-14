"""Gateway HTTP surface, mounted onto the existing dashboard FastAPI app.

No second server and no extra proxy process: `register_gateway_routes(app, cfg, db)`
adds the gateway's routes to the app the dashboard already runs, so internal agents and
external OpenAI-compatible clients share one endpoint. Stages:

  A (here): discovery + monitoring — GET /v1/models, /api/gateway/providers,
            /api/gateway/resources, /api/gateway/health.
  B: POST /v1/chat/completions passthrough + request monitor.
  C: resource-manager queue views.

The shared registry and monitor live on app.state.gateway so later stages extend the
same instances rather than rebuilding them.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException

from .providers import ProviderRegistry
from .resources import ResourceMonitor


def register_gateway_routes(app, cfg: dict, db: Any) -> dict:
    registry = ProviderRegistry(cfg, db)
    monitor = ResourceMonitor()
    state = {"registry": registry, "monitor": monitor, "started_at": time.time()}
    app.state.gateway = state

    # ------------------------------------------------------------ unified discovery
    @app.get("/v1/models")
    def v1_models():
        """OpenAI-compatible model list, unified across every provider. An external client
        can point its OpenAI SDK at this base_url and discover what is available without
        knowing the internal provider layout."""
        data = []
        for m in registry.all_models(discover=True):
            data.append({"id": m.get("id"), "object": "model",
                         "owned_by": m.get("provider"),
                         "playground": {k: m.get(k) for k in ("state", "context", "provider")}})
        return {"object": "list", "data": data}

    # ------------------------------------------------------------ provider management (read)
    @app.get("/api/gateway/providers")
    def gateway_providers(models: bool = False, health: bool = True):
        return {"providers": [p.public_dict(include_models=models, include_health=health)
                              for p in registry.list()]}

    @app.get("/api/gateway/providers/{name}")
    def gateway_provider(name: str):
        p = registry.get(name)
        if p is None:
            raise HTTPException(404, f"unknown provider '{name}'")
        return p.public_dict(include_models=True, include_health=True)

    @app.get("/api/gateway/providers/{name}/models")
    def gateway_provider_models(name: str):
        p = registry.get(name)
        if p is None:
            raise HTTPException(404, f"unknown provider '{name}'")
        return {"provider": name, "models": p.discover_models(),
                "loaded": p.loaded_models(), "capabilities": p.capabilities()}

    # ------------------------------------------------------------ resource monitoring
    @app.get("/api/gateway/resources")
    def gateway_resources(gpu: bool = True):
        return monitor.snapshot(include_gpu=gpu)

    # ------------------------------------------------------------ gateway health/overview
    @app.get("/api/gateway/health")
    def gateway_health():
        provs = registry.list()
        reachable = sum(1 for p in provs if (p.health().get("reachable") is True))
        return {"ok": True, "providers": len(provs), "providers_reachable": reachable,
                "uptime_seconds": round(time.time() - state["started_at"], 1)}

    # Stage B: the OpenAI-compatible passthrough (/v1/chat/completions) + request monitor.
    from .gateway import register_chat_routes
    register_chat_routes(app, cfg, db, state)

    return state
