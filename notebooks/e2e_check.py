#!/usr/bin/env python
"""端到端自检：对网关发流式/非流式请求，校验响应与 TTFT。

用法：python notebooks/e2e_check.py --url http://127.0.0.1:8000 [--no-stream]
"""
from __future__ import annotations

import argparse
import json
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="sk-lab-demo-key")
    ap.add_argument("--model", default=None, help="不填则由路由层决定（vllm 优先）")
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.set_defaults(stream=True)
    args = ap.parse_args()

    payload: dict = {
        "messages": [{"role": "user", "content": "用一句话介绍 KV Cache"}],
        "stream": args.stream,
    }
    if args.model:
        payload["model"] = args.model

    headers = {"Authorization": f"Bearer {args.key}"}
    t0 = time.monotonic()
    with httpx.Client(timeout=120) as client:
        if args.stream:
            with client.stream(
                "POST", f"{args.url}/v1/chat/completions", json=payload, headers=headers
            ) as r:
                r.raise_for_status()
                backend = r.headers.get("x-lab-backend", "?")
                ttft = None
                parts: list[str] = []
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    obj = json.loads(data)
                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
            print(f"[stream] backend={backend} ttft={ttft and f'{ttft*1000:.0f}ms'}")
            print("reply:", "".join(parts)[:200])
        else:
            r = client.post(f"{args.url}/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            backend = r.headers.get("x-lab-backend", "?")
            print(
                f"[non-stream] backend={backend} wall={(time.monotonic()-t0)*1000:.0f}ms "
                f"usage={data.get('usage')}"
            )
            print("reply:", data["choices"][0]["message"]["content"][:200])
    print("E2E OK")


if __name__ == "__main__":
    main()
