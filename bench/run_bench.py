"""并发压测：输出 TTFT / TPOT / 吞吐 / 错误率。

用法：
  python bench/run_bench.py --url http://127.0.0.1:8000/v1/chat/completions \
      --key sk-lab-demo-key --model mock-model --concurrency 8 --n 50 --stream
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def run_one(
    client: httpx.AsyncClient, url: str, key: str, model: str, prompt: str, stream: bool
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.monotonic()
    ttft: float | None = None
    itls: list[float] = []
    last_content_at: float | None = None
    usage: dict = {}
    try:
        if not stream:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage") or {}
            return {
                "ok": True,
                "ttft": time.monotonic() - t0,
                "tpot": None,
                "out_tokens": usage.get("completion_tokens", 0),
                "wall": time.monotonic() - t0,
            }
        async with client.stream("POST", url, json=payload, headers=headers) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - (last_content_at or now))
                obj = json.loads(data)
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    last_content_at = now
        out_tokens = usage.get("completion_tokens") or max(1, len(itls) + 1)
        tpot = statistics.mean(itls) if itls else None
        return {"ok": True, "ttft": ttft, "tpot": tpot, "out_tokens": out_tokens, "wall": time.monotonic() - t0}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e), "wall": time.monotonic() - t0}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--key", default="sk-lab-demo-key")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n", type=int, default=50, help="总请求数")
    ap.add_argument("--stream", action="store_true", default=True)
    args = ap.parse_args()

    results: list[dict] = []
    async with httpx.AsyncClient(timeout=120) as client:
        sem = asyncio.Semaphore(args.concurrency)

        async def worker(i: int):
            async with sem:
                results.append(
                    await run_one(client, args.url, args.key, args.model, f"压测请求 {i}: 介绍一下你自己", args.stream)
                )

        wall_start = time.monotonic()
        await asyncio.gather(*(worker(i) for i in range(args.n)))
        wall = time.monotonic() - wall_start

    ok = [r for r in results if r["ok"]]
    err = [r for r in results if not r["ok"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = [r["tpot"] for r in ok if r["tpot"]]
    out_tokens = sum(r["out_tokens"] for r in ok)

    def pct(xs, p):
        if not xs:
            return float("nan")
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    print("=" * 60)
    print(f"requests={len(results)} ok={len(ok)} err={len(err)} wall={wall:.2f}s")
    if ttfts:
        print(f"TTFT  avg={statistics.mean(ttfts)*1000:.1f}ms  p50={pct(ttfts,0.5)*1000:.1f}ms  p95={pct(ttfts,0.95)*1000:.1f}ms")
    if tpots:
        print(f"TPOT  avg={statistics.mean(tpots)*1000:.1f}ms  p95={pct(tpots,0.95)*1000:.1f}ms")
    if wall > 0:
        print(f"吞吐  {out_tokens / wall:.1f} tok/s  ({out_tokens} tokens)")
    if err:
        print(f"错误率 {len(err)/len(results)*100:.1f}%  示例: {err[0].get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
