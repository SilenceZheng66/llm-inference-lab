"""OpenAI 兼容转发：非流式 JSON 与流式 SSE。"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class UpstreamError(Exception):
    pass


async def forward_non_stream(
    client: httpx.AsyncClient, base_url: str, payload: dict
) -> dict:
    r = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
    if r.status_code >= 400:
        raise UpstreamError(f"upstream {r.status_code}: {r.text[:200]}")
    return r.json()


async def forward_stream(
    client: httpx.AsyncClient, base_url: str, payload: dict
) -> AsyncIterator[dict]:
    """逐块转发 SSE，yield 解析后的 chunk dict；调用方负责计时与收尾。"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    async with client.stream("POST", url, json=payload) as r:
        if r.status_code >= 400:
            body = (await r.aread()).decode("utf-8", "replace")
            raise UpstreamError(f"upstream {r.status_code}: {body[:200]}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
