#!/usr/bin/env bash
# GPU 批次收尾：压测数据落盘 + 关停引擎，释放 GPU 时长
set -euo pipefail
cd /mnt/workspace/llm-inference-lab

TS=$(date +%m%d_%H%M)
echo "== 压测（并发 1/8/32）=="
for c in 1 8 32; do
  python bench/run_bench.py --url http://127.0.0.1:8000/v1/chat/completions \
    --key sk-lab-demo-key --model qwen2.5-1.5b --concurrency "$c" --n 32 \
    | tee "/mnt/workspace/bench_${TS}_c${c}.txt"
done
echo "== 数据已存 /mnt/workspace/bench_${TS}_c*.txt，请回填 README =="
