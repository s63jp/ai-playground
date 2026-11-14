# Model Gateway & Local-AI-Infrastructure Layer

Status of the model-gateway upgrade on branch `feature/model-gateway`. The gateway is an
**agent-agnostic** infrastructure layer added as `playground/gateway/`, mounted on the existing
dashboard app (one process, no extra proxy). It never touches an agent's conversation, memory,
session history or tool execution — it manages the providers agents connect to.

## Done (committed, tested, live-verified against LM Studio)

| Stage | What | Commit |
|-------|------|--------|
| 0 | WIP checkpoint + backup tag `backup/pre-gateway-2026-11-14` (non-destructive) | `2d5666b` |
| A | Provider registry + capability discovery + hardware-aware resource monitor | `7bc3dfe` |
| B | OpenAI-compatible `/v1/chat/completions` gateway + request monitor | `b7bd89a` |
| — | Hardening: 13 adversarial-review findings fixed + regression tests | `f1a8141` |
| C | Backend-enforced concurrency limits + fair priority queue | `59bf0ec` |

Baseline before this work: **254 tests passing**. Now: **254 + 58 gateway tests**, full suite green,
0 regressions. Live-verified on the workstation: Intel Arc A770 (16 GB) measured via `torch.xpu` +
Windows perf counters; a real `qwen/qwen3.5-9b` round-trip through the gateway with provider-reported
token accounting; concurrency limit proven to serialise at the provider.

### Endpoints
- `POST /v1/chat/completions` — OpenAI-compatible passthrough (routing, failover, request tracking,
  token accounting, cancellation, duplicate guard, streaming, optional auth).
- `GET /v1/models` — unified model discovery across providers.
- `GET /api/gateway/providers[/{name}[/models]]`, `/resources`, `/health` — discovery + monitoring.
- `GET /api/gateway/requests[/{id}]`, `POST .../{id}/cancel` — the API monitor.
- `GET /api/gateway/queue`, `GET|POST /api/gateway/concurrency` — resource management.

### Verify
- `./.venv/Scripts/python -m pytest -q` — full suite.
- `python scripts/gateway_smoke.py` — live end-to-end diagnostic (needs LM Studio up).
- `playground serve` → the gateway endpoints above are live on the dashboard app.

## Remaining work (not yet done — next coherent stages)

1. **Stage D — Dashboard tabs.** Backend is complete and serving data; the dashboard UI
   (`playground/dashboard/static/`) still needs the Infrastructure Overview, Providers, upgraded
   API Monitor, and Resource Management / queue tabs wired to the endpoints above.
2. **Agent orchestration reliability** (Phase 4). Investigate and fix the "Agents 2 & 3 both ended
   `interrupted`" issue in `session/manager.run_agents` + `session/runner.py`: on Windows a
   `KeyboardInterrupt`/`CTRL_C_EVENT` can reach multiple `ThreadPoolExecutor` workers and share a
   cancellation path, so stopping one agent can stop another. Replace the process-wide interrupt with
   per-run cancellation tokens so independent agent execution is preserved. Add agent registration /
   lifecycle visibility / restart handling on top.
3. **Provider management** CRUD (add/edit/remove/test/refresh) over the `providers` table +
   provider performance comparison. The table and read paths exist; write/UI remain.
4. **Benchmark expansion** (Phase 8). Measure through the gateway: time-to-first-token, prompt vs
   generation speed, concurrent-request performance; persist results for historical comparison.
5. **Security hardening** (Phase 11). Optional gateway auth token is in place (`gateway.auth_token_env`,
   enforced + warned off-loopback); still to do: broader remote-access controls, rate limiting, and
   audit of sensitive-log exposure.

## Notes
- A pre-existing untracked/modified file `assets/humanoid_mujoco.xml` (a humanoid test regenerates it)
  is intentionally left to the owner — it is not part of the gateway work.
- Branch is local only — nothing pushed and no PR opened.
