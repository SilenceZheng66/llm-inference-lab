"""llm-inference-lab 网关入口。

链路：auth → registry(healthy) → strategy.select → breaker/proxy → metrics/cost。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .auth import require_api_key
from .config import Settings, load_settings
from .observability.cost import cost_usd
from .observability.metrics import (
    COST,
    IN_FLIGHT,
    ITL,
    PREFIX_HITS,
    REQ_LATENCY,
    REQ_TOTAL,
    TOKENS,
    TTFT,
)
from .observability.tracing import TraceMiddleware
from .proxy import UpstreamError, forward_non_stream, forward_stream
from .registry import BackendState, Registry
from .router import STRATEGIES

log = logging.getLogger("lab")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    strategy = STRATEGIES.get(settings.strategy) or STRATEGIES["round_robin"]
    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
    registry = Registry(settings, client)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await registry.check_all()
        await registry.start_health_loop()
        yield
        await registry.stop()

    app = FastAPI(title="llm-inference-lab", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.add_middleware(TraceMiddleware)

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "strategy": strategy.name,
            "backends": {
                name: {
                    "healthy": s.healthy,
                    "in_flight": s.in_flight,
                    "breaker": s.breaker.state,
                    "base_url": s.spec.base_url,
                }
                for name, s in registry.states.items()
            },
        }

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    async def _stream_gen(state: BackendState, payload: dict, t0: float):
        model = state.spec.model
        first = True
        last: float | None = None
        usage: dict = {}
        try:
            async for chunk in forward_stream(client, state.spec.base_url, payload):
                now = time.monotonic()
                if first:
                    TTFT.labels(state.spec.name, model).observe(now - t0)
                    first = False
                elif last is not None:
                    ITL.labels(state.spec.name, model).observe(now - last)
                last = now
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            state.breaker.record_success()
            REQ_TOTAL.labels(state.spec.name, model, "ok").inc()
            if usage:
                TOKENS.labels(state.spec.name, model, "prompt").inc(usage.get("prompt_tokens", 0))
                TOKENS.labels(state.spec.name, model, "completion").inc(usage.get("completion_tokens", 0))
                COST.labels(state.spec.name, model).inc(
                    cost_usd(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                )
        except UpstreamError as e:
            state.breaker.record_failure()
            REQ_TOTAL.labels(state.spec.name, model, "error").inc()
            yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
        finally:
            state.in_flight -= 1
            IN_FLIGHT.labels(state.spec.name).dec()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        require_api_key(request, settings)
        payload = await request.json()

        prefix_key = ""
        for m in payload.get("messages") or []:
            if m.get("role") == "user":
                prefix_key = str(m.get("content", ""))[:64]
                break
        ctx = {"prefix_key": prefix_key}
        stream = bool(payload.get("stream"))

        candidates = registry.healthy_states()
        last_err: Exception | None = None
        while candidates:
            state = strategy.select(candidates, ctx)
            if state is None:
                break
            PREFIX_HITS.labels(state.spec.name, strategy.name).inc()
            state.in_flight += 1
            IN_FLIGHT.labels(state.spec.name).inc()
            t0 = time.monotonic()

            if stream:
                return StreamingResponse(
                    _stream_gen(state, payload, t0),
                    media_type="text/event-stream",
                    headers={"x-lab-backend": state.spec.name},
                )

            try:
                data = await forward_non_stream(client, state.spec.base_url, payload)
                REQ_LATENCY.labels(state.spec.name, state.spec.model).observe(time.monotonic() - t0)
                usage = data.get("usage") or {}
                TOKENS.labels(state.spec.name, state.spec.model, "prompt").inc(usage.get("prompt_tokens", 0))
                TOKENS.labels(state.spec.name, state.spec.model, "completion").inc(usage.get("completion_tokens", 0))
                COST.labels(state.spec.name, state.spec.model).inc(
                    cost_usd(
                        state.spec.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )
                )
                state.breaker.record_success()
                REQ_TOTAL.labels(state.spec.name, state.spec.model, "ok").inc()
                resp = JSONResponse(data)
                resp.headers["x-lab-backend"] = state.spec.name
                return resp
            except UpstreamError as e:
                state.breaker.record_failure()
                REQ_TOTAL.labels(state.spec.name, state.spec.model, "error").inc()
                last_err = e
                candidates = [c for c in candidates if c is not state]
            finally:
                if not stream:
                    state.in_flight -= 1
                    IN_FLIGHT.labels(state.spec.name).dec()

        return JSONResponse(
            {"error": {"message": f"no healthy backend available: {last_err}"}},
            status_code=502,
        )

    return app


app = create_app()
