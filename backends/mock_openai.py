"""OpenAI 兼容 mock 后端：本地无 GPU 也能跑通全链路。

启动：uvicorn backends.mock_openai:app --port 8001
"""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock-openai")
STATE = {"in_flight": 0}


@app.get("/v1/models")
@app.get("/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": "mock-model", "object": "model", "owned_by": "lab"}],
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    model = payload.get("model", "mock-model")
    messages = payload.get("messages") or []
    user_text = next(
        (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    reply = f"(mock) you said: {user_text}"

    if payload.get("stream"):
        return StreamingResponse(_stream(model, reply), media_type="text/event-stream")

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, len(user_text) // 4),
                "completion_tokens": max(1, len(reply) // 4),
            },
        }
    )


async def _stream(model: str, reply: str):
    STATE["in_flight"] += 1
    try:
        cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        def _chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
            obj: dict = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage:
                obj["usage"] = usage
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        yield _chunk({"role": "assistant", "content": ""})
        words = reply.split(" ")
        for w in words:
            await asyncio.sleep(0.02)
            yield _chunk({"content": w + " "})
        yield _chunk(
            {},
            finish="stop",
            usage={"prompt_tokens": 8, "completion_tokens": len(words) + 1},
        )
        yield "data: [DONE]\n\n"
    finally:
        STATE["in_flight"] -= 1
