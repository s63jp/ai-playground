"""Live end-to-end smoke test for the model gateway.

Exercises the real gateway against whatever providers are up (notably LM Studio), with no
mocks: provider discovery, unified /v1/models, the resource monitor, a real
/v1/chat/completions round-trip, and the request monitor's per-request inspector.

Usage:
    python scripts/gateway_smoke.py                 # in-process app (still hits real providers)
    python scripts/gateway_smoke.py --url http://127.0.0.1:8765   # a running `playground serve`
    python scripts/gateway_smoke.py --model "qwen/qwen3.5-9b"     # force a model

Exit code 0 on a successful round-trip, 1 otherwise. This is a diagnostic, not a unit test;
it makes real calls and reports exactly what happened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.config import load_config  # noqa: E402


def _client(url: str | None):
    """Either an httpx client bound to a running server, or an in-process TestClient."""
    if url:
        import httpx
        base = url.rstrip("/")

        class C:
            def get(self, path, **kw):
                return httpx.get(base + path, timeout=30, **kw)

            def post(self, path, **kw):
                return httpx.post(base + path, timeout=300, **kw)
        return C(), f"live server {base}"
    from fastapi.testclient import TestClient
    from playground.dashboard import create_app
    cfg = load_config()
    return TestClient(create_app(cfg)), "in-process app (providers still real)"


def _hr(title):
    print(f"\n=== {title} ===")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live smoke test for the model gateway")
    ap.add_argument("--url", default=None, help="target a running server instead of in-process")
    ap.add_argument("--model", default=None, help="force a model id (default: first loaded LM Studio model)")
    ap.add_argument("--prompt", default="Reply with the single word: ok")
    args = ap.parse_args(argv)
    client, where = _client(args.url)
    print(f"Gateway smoke test - {where}")

    _hr("health")
    h = client.get("/api/gateway/health").json()
    print(json.dumps(h, indent=2))

    _hr("providers")
    provs = client.get("/api/gateway/providers", params={"health": "true"}).json()["providers"]
    for p in provs:
        hs = p.get("health", {})
        print(f"  {p['name']:14s} {p['kind']:13s} reachable={hs.get('reachable')} "
              f"latency_ms={hs.get('latency_ms')} models={hs.get('model_count')} key_present={p['key_present']}")

    _hr("resources (measured vs estimated)")
    res = client.get("/api/gateway/resources").json()
    print(f"  CPU {res['cpu'].get('percent')}% [{res['cpu']['source']}]  "
          f"RAM {res['memory'].get('percent')}% [{res['memory']['source']}]")
    for g in res.get("gpus", []):
        util = g["utilization_percent"]
        vram = g.get("vram_percent", {})
        print(f"  GPU {g['name']} util={util['value']}% [{util['source']}] "
              f"vram={vram.get('value')}% [{vram.get('source','unknown')}]")

    _hr("unified /v1/models")
    models = client.get("/v1/models").json().get("data", [])
    print(f"  {len(models)} model(s) across providers; e.g. " +
          ", ".join(m["id"] for m in models[:5]) + (" ..." if len(models) > 5 else ""))

    # pick a model: explicit, else a loaded LM Studio one, else any listed
    model = args.model
    if not model:
        loaded = [m for m in models if (m.get("playground") or {}).get("state") == "loaded"]
        model = (loaded[0]["id"] if loaded else (models[0]["id"] if models else None))
    if not model:
        print("\nNo models available from any provider — start LM Studio and load a model.")
        return 1

    _hr(f"chat round-trip via /v1/chat/completions (model={model})")
    r = client.post("/v1/chat/completions", json={
        "model": model, "max_tokens": 64,
        "messages": [{"role": "user", "content": args.prompt}]},
        headers={"X-Playground-Agent": "smoke_test", "X-Playground-App": "gateway_smoke"})
    if r.status_code != 200:
        print(f"  FAILED: HTTP {r.status_code}: {r.text[:300]}")
        return 1
    rid = r.headers.get("x-request-id")
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    print(f"  request_id   = {rid}")
    print(f"  provider     = {r.headers.get('x-provider')}")
    print(f"  token_source = {r.headers.get('x-token-source')}")
    print(f"  reply        = {text!r}")

    _hr("request inspector /api/gateway/requests/{id}")
    row = client.get(f"/api/gateway/requests/{rid}").json()
    for k in ("agent", "app", "provider", "model", "status", "http_status", "latency_ms",
              "prompt_tokens", "completion_tokens", "total_tokens", "token_source"):
        print(f"  {k:16s} = {row.get(k)}")
    print("\nOK - gateway routed, answered, accounted tokens and logged the request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
