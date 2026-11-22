#!/usr/bin/env python
"""训后评估：base vs tuned 在同一 eval 集上的量化对比。

指标：
- ROUGE-L（自实现 LCS，零额外依赖）：衡量与参考答案的重叠
- 要点命中率：参考答案中的"关键项"（JSON 键名 / 列表项首词）被生成答案覆盖的比例——
  对"格式类"微调目标更敏感，优于整体 rouge 的模糊性

用法：
  python eval_compare.py --config config.yaml --model-mode base   --out eval_base.json
  python eval_compare.py --config config.yaml --model-mode tuned --out eval_tuned.json
  python eval_compare.py --compare eval_base.json eval_tuned.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import httpx

DEFAULT_CFG = Path(__file__).with_name("config.yaml")


# ---------------------------------------------------------------- 指标（纯函数，可单测）

def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    """LCS F1（简化 ROUGE-L）：句子级分词后计算。"""
    tok = lambda s: re.findall(r"\w+|[^\w\s]", s.lower())
    a, b = tok(pred), tok(ref)
    if not a or not b:
        return 0.0
    l = lcs_len(a, b)
    if l == 0:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


def key_points(ref: str) -> list[str]:
    """抽取参考答案要点：JSON 键名 + markdown 列表项。"""
    pts = set(re.findall(r'"([\w\u4e00-\u9fff]+)"\s*:', ref))
    pts |= {m.strip(" -*\t") for m in re.findall(r"^\s*[-*]\s+(.+)$", ref, flags=re.M)}
    return [p for p in pts if len(p) >= 2]


def coverage(pred: str, ref: str) -> float:
    pts = key_points(ref)
    if not pts:
        return -1.0                                  # 无要点样本，不计入
    hit = sum(1 for p in pts if p.lower() in pred.lower())
    return hit / len(pts)


# ---------------------------------------------------------------- 生成

def generate(cfg: dict, model_mode: str, eval_path: Path, out_path: Path, limit: int = 50) -> None:
    import yaml
    c = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
    v = c["vllm"]
    model = v["adapter_name"] if model_mode == "tuned" else "qwen2.5-1.5b"
    url = f"http://127.0.0.1:{v['port']}/v1/chat/completions"
    rows = [json.loads(l) for l in eval_path.read_text(encoding="utf-8").splitlines()][:limit]

    results, tts = [], 0.0
    with httpx.Client(timeout=120) as cli:
        for i, r in enumerate(rows):
            msgs = r["messages"][:-1]                       # 去掉参考答案，只留 user 轮
            ref = r["messages"][-1]["content"]
            t0 = time.time()
            resp = cli.post(url, headers={"Authorization": "Bearer sk-lab-demo-key"}, json={
                "model": model, "messages": msgs, "temperature": 0,
                "max_tokens": 512})
            dt = time.time() - t0
            pred = resp.json()["choices"][0]["message"]["content"]
            tts += dt
            results.append({"i": i, "rouge_l": round(rouge_l(pred, ref), 4),
                            "coverage": round(coverage(pred, ref), 4),
                            "latency_s": round(dt, 2),
                            "pred_head": pred[:80], "ref_head": ref[:80]})
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(rows)}] {model}")

    scored = [r for r in results if r["coverage"] >= 0]
    summary = {
        "model": model, "n": len(results),
        "rouge_l_avg": round(sum(r["rouge_l"] for r in results) / len(results), 4),
        "coverage_avg": round(sum(r["coverage"] for r in scored) / len(scored), 4) if scored else None,
        "coverage_n": len(scored),
        "latency_avg_s": round(tts / len(results), 3),
    }
    out_path.write_text(json.dumps({"summary": summary, "rows": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def compare(a: Path, b: Path) -> None:
    sa = json.loads(a.read_text(encoding="utf-8"))["summary"]
    sb = json.loads(b.read_text(encoding="utf-8"))["summary"]
    rows = [k for k in ("model", "n", "rouge_l_avg", "coverage_avg", "latency_avg_s")]
    w = max(len(r) for r in rows) + 2
    for k in rows:
        print(f"{k:<{w}} {sa.get(k)!s:>12} {sb.get(k)!s:>12}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("run")
    g.add_argument("--model-mode", choices=["base", "tuned"], required=True)
    g.add_argument("--eval-file", default=None)
    g.add_argument("--out", required=True)
    g.add_argument("--limit", type=int, default=50)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs=2)
    args = ap.parse_args()
    if args.cmd == "run":
        import yaml
        d = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["data"]
        eval_path = Path(args.eval_file or (d["out_dir"] + "/eval.jsonl"))
        generate(args.config, args.model_mode, eval_path, Path(args.out), args.limit)
    else:
        compare(Path(args.files[0]), Path(args.files[1]))


if __name__ == "__main__":
    main()
