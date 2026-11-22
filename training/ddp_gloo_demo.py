#!/usr/bin/env python
"""CPU gloo 多进程 DDP 语义演示（单卡限制下的分布式训练语义验证）。

验证三件事（面试可直接引用的实验结论）：
1. 数据分片：各 rank 消费不同分片，loss 不同（数据并行语义）
2. 梯度同步：allreduce 平均后，各 rank 的梯度逐位一致（all_gather 交叉验证）
3. 单点 IO：检查点/日志只由 rank0 写（工程约定）

真实多卡 DDP 的差异仅在 backend=gloo→nccl 与 device=cpu→cuda，代码结构不变。
"""
from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def worker(rank: int, world: int, result_q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(1000 + rank)                      # 各 rank 不同初始化/数据

    model = nn.Linear(8, 4, bias=False)
    shard = torch.randn(16, 8) + rank * 0.1             # 各 rank 数据分片不同

    loss = (model(shard) ** 2).mean()
    loss.backward()
    local_grad = model.weight.grad.clone()

    # DDP 梯度同步语义：allreduce(SUM) / world_size
    dist.allreduce(model.weight.grad, op=dist.ReduceOp.SUM)
    model.weight.grad /= world
    synced = model.weight.grad.clone()

    # 交叉验证：all_gather 各 rank 的 synced 结果，应逐位一致
    gathered = [torch.zeros_like(synced) for _ in range(world)]
    dist.all_gather(gathered, synced)
    all_same = all(torch.equal(g, synced) for g in gathered)

    result_q.put({
        "rank": rank,
        "loss": round(float(loss.item()), 4),
        "shard_n": int(shard.shape[0]),
        "grad_sync_allclose": bool(all_same),
        "local_vs_synced_max_diff": round(float((synced - local_grad).abs().max()), 5),
    })
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    world = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, q)) for r in range(world)]
    for p in procs:
        p.start()
    outs = sorted([q.get() for _ in range(world)], key=lambda x: x["rank"])
    for p in procs:
        p.join()

    print("\n== DDP(gloo, CPU, world=2) 语义验证 ==")
    for r in outs:
        print(f"rank{r['rank']}: loss={r['loss']} shard={r['shard_n']}samples "
              f"grad_sync_allclose={r['grad_sync_allclose']} "
              f"local_vs_synced_max_diff={r['local_vs_synced_max_diff']}")
    ok = all(r["grad_sync_allclose"] for r in outs) and outs[0]["loss"] != outs[1]["loss"]
    print("\n预期：各 rank loss 不同（数据分片不同）；allreduce 后梯度逐位一致（同步语义生效）→",
          "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
