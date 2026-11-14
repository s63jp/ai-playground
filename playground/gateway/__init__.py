"""Model-gateway and local-AI-infrastructure layer.

This package is the infrastructure AI Playground provides *to* agents (its own
autonomous agents and external OpenAI-compatible clients such as Hermes, Claude
Code or Codex) rather than anything that reaches inside an agent. It never touches
an agent's conversation, memory, session history or tool execution: it manages the
model providers those agents connect to.

Pieces, added in stages:
  providers   - provider registry + capability discovery (what each endpoint offers)
  resources   - hardware-aware resource monitoring (CPU/RAM/GPU/VRAM), measured vs estimated
  router      - model-id -> provider routing with failover (stage B)
  gateway     - OpenAI-compatible /v1/chat/completions passthrough + request tracking (stage B)
  concurrency - backend-enforced concurrency limits + request queue (stage C)
  api         - FastAPI routes mounted on the existing dashboard app (no second process)

Nothing here imports the physics simulator; the simulator never imports this.
"""
from __future__ import annotations

from .providers import Provider, ProviderRegistry
from .resources import ResourceMonitor

__all__ = ["Provider", "ProviderRegistry", "ResourceMonitor"]
