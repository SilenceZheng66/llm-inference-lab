#!/usr/bin/env python
"""数据准备：下载/筛选/清洗/切分，产出 messages JSONL + manifest（md5 可追溯）。

设计要点（对应 JD "实验可追溯"）：
- 固定 seed 抽样，manifest 记录来源、条数、md5，任何一次实验的数据都可复现核对；
- task_hint 筛选：优先保留"格式类"任务样本（JSON 输出等），使训后 eval 差异可显性度量，
  而不是依赖泛泛的 rouge 波动。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

DEFAULT_CFG = Path(__file__).with_name("config.yaml")

JSON_FENCE = re.compile(r"```(json)?", re.IGNORECASE)


def looks_structured(answer: str) -> bool:
    """判断回复是否为结构化/格式类任务样本（JSON/列表/键值对），用于定向筛选。"""
    a = answer.strip()
    if a.startswith("{") and a.rstrip().endswith("}"):
        return True
    if JSON_FENCE.search(a):
        return True
    if re.search(r"^\s*[-*]\s+\S+", a, flags=re.M) and len(re.findall(r"^\s*[-*]\s+", a, flags=re.M)) >= 3:
        return True
    return False


def to_messages(instruction: str, inp: str, output: str) -> dict:
    """alpaca 三字段 -> chat messages。非空 input 并入 user 轮。"""
    user = instruction.strip()
    extra = (inp or "").strip()
    if extra:
        user = f"{user}\n\n输入：\n{extra}"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": output.strip()},
    ]}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(dataset_id: str, n_need: int, task_hint: bool):
    """优先 ModelScope（dataset_id 为 ModelScope 仓库名），失败时回退 HF Hub。

    注意：HF 的 load_dataset 只认 HF Hub 上的仓库；AI-ModelScope/... 这类
    ModelScope 专属 id 在 HF 上不存在，会以 ConnectionError 失败而非清晰报错，
    因此 ModelScope id 必须走 MsDataset，这里用双源兜底并打印实际来源。
    """
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(dataset_id, split="train")
        print("[info] dataset source = modelscope")
    except Exception as e:  # noqa: BLE001 - 双源兜底需要捕获任意加载失败
        print(f"[info] ModelScope 加载失败（{type(e).__name__}），回退 HF Hub：{e}")
        try:
            from datasets import load_dataset
        except ImportError as e2:
            sys.exit(f"缺少依赖：请先 pip install datasets modelscope（原因：{e2}）")
        ds = load_dataset(dataset_id, split="train")
        print("[info] dataset source = hf")
    rows = []
    for r in ds:
        out = (r.get("output") or "").strip()
        if len(out) < 8 or len(out) > 900:      # 过滤过短/超长，控制序列长度分布
            continue
        if task_hint and not looks_structured(out):
            continue
        rows.append(r)
        if len(rows) >= n_need:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="准备 SFT 数据集")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--train-samples", type=int, default=None)
    ap.add_argument("--eval-samples", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["data"]
    dataset_id = args.dataset_id or cfg["dataset_id"]
    n_train = args.train_samples or cfg["train_samples"]
    n_eval = args.eval_samples or cfg["eval_samples"]
    out_dir = Path(args.out_dir or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_raw(dataset_id, n_train + n_eval, task_hint=bool(cfg.get("task_hint")))
    if len(rows) < n_train + n_eval:
        print(f"[warn] 筛选后样本不足：{len(rows)} < {n_train + n_eval}，按实际数量切分")

    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:n_eval + n_train]

    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    for path, part in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(to_messages(r["instruction"], r.get("input", ""), r["output"]),
                                   ensure_ascii=False) + "\n")

    manifest = {
        "dataset_id": dataset_id,
        "seed": cfg["seed"],
        "task_hint": bool(cfg.get("task_hint")),
        "train": {"path": train_path.name, "n": len(train_rows), "md5": md5_of(train_path)},
        "eval": {"path": eval_path.name, "n": len(eval_rows), "md5": md5_of(eval_path)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
