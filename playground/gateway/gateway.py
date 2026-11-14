"""The OpenAI-compatible model gateway: POST /v1/chat/completions passthrough.

An external client (Hermes, Claude Code, Codex, any OpenAI SDK) or an internal app points
at this one endpoint and gets routed to the right provider without knowing the provider
layout. The gateway preserves the OpenAI request/response shape (existing clients are not
broken), records every request for the monitor with a correlation id, accounts tokens
(provider-reported where available, else a labelled estimate), applies a duplicate-submit
guard, supports cancellation, and fails over to another provider on a provider-side failure.

It never takes over anything inside the calling agent — it only forwards the one request.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx
from fastapi import Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .router import RouteError, Router
from .tracking import RequestMonitor, redact, tokens_from_response, estimate_prompt_tokens

ENDPOINT = "/v1/chat/completions"

# Request headers the gateway reads for identity/correlation (all optional).
H_AGENT = "x-playground-agent"
H_APP = "x-playground-app"
H_REQID = "x-request-id"
H_CORR = "x-correlation-id"
H_IDEM = "idempotency-key"


def _identity(headers: Any) -> dict:
    """Attribute a request to an agent/app where the caller identifies itself; 'unknown'
    otherwise (never invented)."""
    get = headers.get
    return {
        "agent": get(H_AGENT) or "unknown",
        "app": get(H_APP) or get("x-app") or "unknown",
        "request_id": get(H_REQID),
        "correlation_id": get(H_CORR),
        "idempotency_key": get(H_IDEM) or get("x-idempotency-key"),
    }


def _completion_text(data: dict) -> str | None:
    try:
        return data["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return None


class ChatGateway:
    def __init__(self, cfg: dict, db: Any, registry, monitor: RequestMonitor):
        self.cfg = cfg
        self.db = db
        self.router = Router(registry)
        self.monitor = monitor
        gw = (cfg.get("gateway") or {})
        self.timeout = float(gw.get("request_timeout", 300.0))
        self.client = httpx.Client(timeout=self.timeout)

    def _provider_headers(self, route) -> dict:
        h = {"Content-Type": "application/json"}
        if route.api_key:                       # the provider's own key; never logged
            h["Authorization"] = f"Bearer {route.api_key}"
        return h

    # ------------------------------------------------------------ non-streaming
    def handle(self, body: dict, headers: Any) -> JSONResponse:
        model = body.get("model")
        ident = _identity(headers)
        request_id = ident["request_id"] or self.monitor.new_request_id()
        correlation_id = ident["correlation_id"] or request_id
        idem = ident["idempotency_key"]

        # Duplicate-submit guard: an identical idempotency key already in flight is rejected
        # rather than silently sent twice.
        dup = self.monitor.duplicate_of(idem)
        if dup:
            return JSONResponse(status_code=409, content={
                "error": {"message": "duplicate request in flight for this idempotency-key",
                          "type": "duplicate_request", "existing_request_id": dup}},
                headers={"X-Request-Id": dup})

        if not model:
            return JSONResponse(status_code=400, content={
                "error": {"message": "field 'model' is required", "type": "invalid_request_error"}})

        base = {"request_id": request_id, "correlation_id": correlation_id, "agent": ident["agent"],
                "app": ident["app"], "model": model, "endpoint": ENDPOINT,
                "idempotency_key": idem, "stream": 1 if body.get("stream") else 0,
                "request_json": redact(body)}

        candidates = self.router.candidates(model)
        if not candidates:
            msg = f"no provider offers model '{model}'"
            self.monitor.create(status="error", http_status=404, error=msg,
                                started_at=time.time(), ended_at=time.time(), **base)
            return JSONResponse(status_code=404, content={
                "error": {"message": msg, "type": "model_not_found"}},
                headers={"X-Request-Id": request_id})

        started = time.time()
        self.monitor.create(status="running", started_at=started, provider=candidates[0].name, **base)
        self.monitor.mark_inflight(idem, request_id)
        try:
            return self._forward(request_id, body, candidates, started)
        finally:
            self.monitor.clear_inflight(idem)

    def _forward(self, request_id: str, body: dict, candidates: list, started: float) -> JSONResponse:
        last_err: str | None = None
        for route in candidates:
            if self.monitor.is_cancelled(request_id):
                self.monitor.update(request_id, status="cancelled", cancelled=1, ended_at=time.time(),
                                    latency_ms=round((time.time() - started) * 1000, 1))
                return JSONResponse(status_code=499, content={
                    "error": {"message": "request cancelled", "type": "cancelled"}},
                    headers={"X-Request-Id": request_id})
            url = route.base_url.rstrip("/") + "/chat/completions"
            try:
                r = self.client.post(url, json=body, headers=self._provider_headers(route))
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                continue                         # provider unreachable -> try next (failover)
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                continue                         # provider-side failure -> failover
            return self._finish(request_id, body, route, r, started)
        # every candidate failed
        latency = round((time.time() - started) * 1000, 1)
        self.monitor.update(request_id, status="error", http_status=502, error=last_err,
                            ended_at=time.time(), latency_ms=latency,
                            provider=candidates[-1].name if candidates else None)
        return JSONResponse(status_code=502, content={
            "error": {"message": f"all providers failed for this model: {last_err}",
                      "type": "provider_unavailable"}},
            headers={"X-Request-Id": request_id})

    def _finish(self, request_id: str, body: dict, route, r: httpx.Response, started: float) -> JSONResponse:
        latency = round((time.time() - started) * 1000, 1)
        hdrs = {"X-Request-Id": request_id, "X-Provider": route.name, "X-Model": route.model}
        if r.status_code != 200:
            try:
                data = r.json()
            except ValueError:
                data = {"error": {"message": r.text[:500], "type": "provider_error"}}
            self.monitor.update(request_id, status="error", http_status=r.status_code,
                                provider=route.name, error=f"HTTP {r.status_code}", ended_at=time.time(),
                                latency_ms=latency, response_json=redact(data))
            return JSONResponse(status_code=r.status_code, content=data, headers=hdrs)
        data = r.json()
        text = _completion_text(data)
        usage = tokens_from_response(data, body, text)
        hdrs["X-Token-Source"] = usage["token_source"]
        self.monitor.update(request_id, status="done", http_status=200, provider=route.name,
                            ended_at=time.time(), latency_ms=latency,
                            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
                            total_tokens=usage["total_tokens"], token_source=usage["token_source"],
                            response_json=redact({"choices": data.get("choices"), "usage": data.get("usage")}))
        return JSONResponse(status_code=200, content=data, headers=hdrs)

    # ------------------------------------------------------------ streaming passthrough
    def handle_stream(self, body: dict, headers: Any) -> StreamingResponse:
        model = body.get("model")
        ident = _identity(headers)
        request_id = ident["request_id"] or self.monitor.new_request_id()
        correlation_id = ident["correlation_id"] or request_id
        try:
            route = self.router.resolve(model)
        except RouteError as e:
            self.monitor.create(status="error", http_status=404, error=str(e), request_id=request_id,
                                correlation_id=correlation_id, agent=ident["agent"], app=ident["app"],
                                model=model, endpoint=ENDPOINT, stream=1,
                                started_at=time.time(), ended_at=time.time())
            return StreamingResponse(iter([json.dumps(
                {"error": {"message": str(e), "type": "model_not_found"}}).encode()]),
                status_code=404, media_type="application/json")

        started = time.time()
        self.monitor.create(status="running", request_id=request_id, correlation_id=correlation_id,
                            agent=ident["agent"], app=ident["app"], provider=route.name, model=model,
                            endpoint=ENDPOINT, stream=1, started_at=started, request_json=redact(body))
        url = route.base_url.rstrip("/") + "/chat/completions"
        monitor, db = self.monitor, self.db

        def gen():
            chars = 0
            status, http_status, err = "done", 200, None
            try:
                with self.client.stream("POST", url, json=body, headers=self._provider_headers(route)) as resp:
                    http_status = resp.status_code
                    if resp.status_code != 200:
                        status, err = "error", f"HTTP {resp.status_code}"
                    for chunk in resp.iter_raw():
                        if monitor.is_cancelled(request_id):
                            status = "cancelled"
                            break
                        chars += len(chunk)
                        yield chunk
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                status, http_status, err = "error", 502, f"{type(exc).__name__}: {exc}"
                yield json.dumps({"error": {"message": err, "type": "provider_unavailable"}}).encode()
            finally:
                latency = round((time.time() - started) * 1000, 1)
                # Streamed responses rarely carry a usage block; estimate and label it.
                pt = estimate_prompt_tokens(body)
                ct = max(0, chars // 4)
                monitor.update(request_id, status=status, http_status=http_status, error=err,
                               cancelled=1 if status == "cancelled" else 0, ended_at=time.time(),
                               latency_ms=latency, prompt_tokens=pt, completion_tokens=ct,
                               total_tokens=pt + ct, token_source="estimated")

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"X-Request-Id": request_id, "X-Provider": route.name})


def register_chat_routes(app, cfg: dict, db: Any, state: dict) -> None:
    """Register the gateway passthrough + request-monitor routes on the existing app."""
    from .tracking import RequestMonitor
    request_monitor = RequestMonitor(db)
    gateway = ChatGateway(cfg, db, state["registry"], request_monitor)
    state["request_monitor"] = request_monitor
    state["chat_gateway"] = gateway

    @app.post(ENDPOINT)
    def chat_completions(request: Request, body: dict = Body(...)):
        if body.get("stream"):
            return gateway.handle_stream(body, request.headers)
        return gateway.handle(body, request.headers)

    # ------------------------------------------------------------ API monitor
    @app.get("/api/gateway/requests")
    def gateway_requests(provider: str | None = None, model: str | None = None,
                         agent: str | None = None, app_: str | None = Query(None, alias="app"),
                         status: str | None = None, failed: bool = False,
                         since: float | None = None, limit: int = 200):
        rows = db.gateway_requests(provider=provider, model=model, agent=agent, app=app_,
                                   status=status, failed=failed, since=since, limit=limit)
        return {"requests": rows, "stats": db.gateway_stats(since=since)}

    @app.get("/api/gateway/requests/{request_id}")
    def gateway_request(request_id: str):
        row = db.gateway_request(request_id)
        if row is None:
            raise HTTPException(404, f"no request '{request_id}'")
        return row

    @app.post("/api/gateway/requests/{request_id}/cancel")
    def gateway_cancel(request_id: str):
        found = request_monitor.request_cancel(request_id)
        if not found:
            raise HTTPException(404, f"no request '{request_id}'")
        return {"request_id": request_id, "cancelled": True}
