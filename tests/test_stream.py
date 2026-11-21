"""SSE 转发解析单测：直接对 mock 后端 ASGI 应用发流式请求。"""
from __future__ import annotations

import httpx

from app.proxy import forward_stream
from backends.mock_openai import app as mock_app


def make_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=mock_app)
    return httpx.AsyncClient(transport=transport, base_url="http://mock/v1")


async def test_stream_parses_chunks_and_usage():
    async with make_client() as client:
        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "你好 世界"}],
            "stream": True,
        }
        chunks = [c async for c in forward_stream(client, "http://mock/v1", payload)]
    assert chunks, "should receive chunks"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["completion_tokens"] > 0


async def test_non_stream_roundtrip():
    async with make_client() as client:
        r = await client.post(
            "/chat/completions",
            json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"].startswith("(mock)")
        assert data["usage"]["prompt_tokens"] >= 1
