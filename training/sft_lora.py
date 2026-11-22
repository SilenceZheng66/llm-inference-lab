#!/usr/bin/env python
"""手写 LoRA SFT 训练循环（transformers + peft，不依赖 Trainer）。

刻意不用 Trainer 的原因：面试与代码评审需要能逐段解释——
assistant loss mask、bf16 autocast 边界、梯度累积与有效 batch、
检查点恢复语义（数据快进 + RNG 状态）、profiler 挂载点。

Resume 语义：数据顺序由 seed 预先固定（epochs 拼接为一条序列），检查点记录
consumed_samples；恢复时重建同一序列并用 islice 快进，保证不重不漏。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np

DEFAULT_CFG = Path(__file__).with_name("config.yaml")


# ---------------------------------------------------------------- tokenization

def build_supervised_example(tokenizer, messages: list[dict], max_len: int) -> dict:
    """单轮对话 -> {input_ids, labels}。labels 仅在 assistant 回复段生效。

    前缀 = apply_chat_template(历史轮, add_generation_prompt=True)，即以 "<|im_start|>assistant\\n"
    结尾；目标 = 回复文本 + <|im_end|>（模板轮尾）。前缀段 labels=-100，目标段为 token id。
    """
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    reply = messages[-1]["content"].strip()
    target = reply + "<|im_end|>"

    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    input_ids = prefix_ids + target_ids
    if len(input_ids) > max_len:            # 左截断保住回复段（简化处理：截前缀）
        overflow = len(input_ids) - max_len
        prefix_ids = prefix_ids[overflow:]
        input_ids = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    """右侧 padding；labels pad -100；attention_mask 标记真实 token。"""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


# ---------------------------------------------------------------- checkpoint

def save_ckpt(run_dir: Path, model, optimizer, scheduler, global_step: int,
              consumed: int, rng_state: dict) -> None:
    ck = run_dir / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ck)               # 仅 LoRA adapter（PEFT 包装下）
    torch_ckpt = {
        "global_step": global_step,
        "consumed_samples": consumed,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state,
    }
    torch_save = ck / "trainer_state.pt"
    import torch
    torch.save(torch_ckpt, torch_save)
    (ck / "DONE").write_text(str(global_step), encoding="utf-8")


def capture_rng() -> dict:
    import torch
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# ---------------------------------------------------------------- main loop

def build_optimizer(model, lr: float, wd: float):
    import torch
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim == 1 or "bias" in n) else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--resume", action="store_true", help="从 run_dir/checkpoint 恢复")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    import torch
    import yaml
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mcfg, tcfg = cfg["model"], cfg["train"]
    run_dir = Path(tcfg["out_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    profile = args.profile or tcfg.get("profile", False)

    # 可复现性：三库 RNG 固定
    random.seed(tcfg.get("seed", 42)); np.random.seed(tcfg.get("seed", 42))
    torch.manual_seed(tcfg.get("seed", 42))

    tok = AutoTokenizer.from_pretrained(mcfg["base_model"])
    pad_id = tok.pad_token_id or tok.eos_token_id

    # ---- 数据：固定顺序序列（预 shuffle + epochs 拼接），resume 精确快进 ----
    data_dir = Path(cfg["data"]["out_dir"])
    rows = [json.loads(l) for l in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(cfg["data"]["seed"])
    rng.shuffle(rows)
    seq = rows * int(tcfg["epochs"])
    examples = [build_supervised_example(tok, m, tcfg["max_seq_len"]) for m in seq]
    bs, accum = tcfg["per_device_batch_size"], tcfg["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(rows) / (bs * accum))
    total_steps = steps_per_epoch * int(tcfg["epochs"])
    print(f"[data] samples={len(rows)} seq={len(examples)} total_steps={total_steps}")

    # ---- 模型：bf16 权重 + LoRA 注入 + gradient checkpointing（显存换算力可关）----
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"], dtype=torch.bfloat16, device_map="cuda")
    lcfg = LoraConfig(
        r=mcfg["lora_rank"], lora_alpha=mcfg["lora_alpha"],
        lora_dropout=mcfg["lora_dropout"], task_type="CAUSAL_LM",
        target_modules=mcfg["target_modules"] if mcfg["target_modules"] != "all-linear"
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    optimizer = build_optimizer(model, tcfg["lr"], tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(int(total_steps * tcfg["warmup_ratio"]), total_steps))

    global_step, consumed = 0, 0
    if args.resume:
        ck = run_dir / "checkpoint"
        state = torch.load(ck / "trainer_state.pt", map_location="cpu", weights_only=False)
        from peft import set_peft_model_state_dict  # noqa: F401 (adapter 由 save_pretrained 加载)
        global_step, consumed = state["global_step"], state["consumed_samples"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        print(f"[resume] step={global_step} consumed={consumed}")

    collate = Collator(pad_id)
    examples = examples[consumed:]          # 快进：不重不漏
    micro_batches = [examples[i:i + bs] for i in range(0, len(examples), bs)]
    print(f"[resume-skip] {consumed} 样本，剩余 micro-batch {len(micro_batches)}")

    prof = None
    if profile:
        p0, p1 = (int(x) for x in str(tcfg.get("profile_steps", "10:30")).split(":"))
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=max(1, p1 - p0)),
            on_trace_ready=lambda p: p.export_chrome_trace(str(run_dir / "trace.json")),
            record_shapes=True, profile_memory=True, with_stack=False)

    model.train()
    t0 = time.time(); tokens_seen = 0; running = []
    if prof: prof.start()
    for mb in micro_batches:
        batch = collate(mb)
        batch = {k: torch.tensor(v, device="cuda") for k, v in batch.items()}
        n_tok = int(batch["attention_mask"].sum())
        tokens_seen += n_tok

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = out.loss / accum
        loss.backward()

        running.append(out.loss.item()); consumed += bs
        if consumed % (bs * accum) != 0:
            continue
        # ---- 每 accum 个 micro-batch：step ----
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       tcfg["grad_clip"])
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % tcfg["log_every"] == 0:
            dt = time.time() - t0
            rec = {"step": global_step, "loss": round(sum(running) / len(running), 4),
                   "lr": scheduler.get_last_lr()[0], "tok_s": int(tokens_seen / max(dt, 1e-9)),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            print(json.dumps(rec, ensure_ascii=False))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            running.clear(); t0 = time.time(); tokens_seen = 0

        if prof: prof.step()
        if global_step % tcfg["ckpt_every"] == 0 or global_step >= total_steps:
            save_ckpt(run_dir, model, optimizer, scheduler, global_step, consumed, capture_rng())
            print(f"[ckpt] step={global_step} saved")

    if prof:
        prof.stop()
        (run_dir / "profile_table.txt").write_text(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), encoding="utf-8")
    save_ckpt(run_dir, model, optimizer, scheduler, global_step, consumed, capture_rng())
    print(f"[done] total_steps={global_step}, adapter at {(run_dir / 'checkpoint')}")


if __name__ == "__main__":
    main()
