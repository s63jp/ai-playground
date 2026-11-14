"""Model-id -> provider routing with failover.

The router turns "the client asked for model X" into "send it to this endpoint with this
key", preferring a provider that has the model loaded and reachable, and offering the
remaining candidates for failover. The resolved api key is carried only to make the
outbound call; it is never logged or serialised (see tracking.redact).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .providers import Provider, ProviderRegistry


class RouteError(Exception):
    """No enabled provider offers the requested model."""


@dataclass
class Route:
    provider: Provider
    base_url: str
    model: str
    api_key: str | None            # resolved from env for the outbound call; NEVER logged
    api_key_env: str | None

    @property
    def name(self) -> str:
        return self.provider.name


class Router:
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    def _route_for(self, provider: Provider, model: str) -> Route:
        key = os.environ.get(provider.api_key_env) if provider.api_key_env else None
        return Route(provider=provider, base_url=provider.base_url, model=model,
                     api_key=key, api_key_env=provider.api_key_env)

    def resolve(self, model: str) -> Route:
        """Primary route for a model. Raises RouteError if nothing serves it."""
        p = self.registry.provider_for_model(model)
        if p is None:
            known = sorted({m.get("id") for m in self.registry.all_models()} - {None})
            raise RouteError(f"no provider offers model '{model}'. Known models: {known[:20]}"
                             + (" ..." if len(known) > 20 else ""))
        return self._route_for(p, model)

    def candidates(self, model: str) -> list[Route]:
        """All routes that can serve the model, best first (primary + failover)."""
        return [self._route_for(p, model) for p in self.registry.providers_for_model(model)]
