#!/usr/bin/env bash
# vLLM LoRA 服务化：同一 vLLM 实例同时暴露 base 与 tuned(adapter) 两个 model
# 用法：bash serve_vllm_lora.sh [adapter_dir]
set -euo pipefail

ADAPTER_DIR="${1:-/mnt/workspace/runs/sft-activity/checkpoint}"
CFG="$(dirname "$0")/config.yaml"

# 从 config.yaml 读端口/adapter 名/最大 rank（无 yq 时用 python）
read_cfg() { python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['vllm']['$1'])"; }
PORT=$(read_cfg port)
NAME=$(read_cfg adapter_name)
RANK=$(read_cfg max_lora_rank)

BASE_MODEL=/mnt/workspace/models/qwen2.5-1.5b

echo "== 停旧实例 =="
pkill -f "vllm serve" 2>/dev/null || true; sleep 3

echo "== 启动 vLLM（base + LoRA adapter 同进程）=="
nohup vllm serve "$BASE_MODEL" \
  --served-model-name qwen2.5-1.5b \
  --enable-lora \
  --lora-modules "${NAME}=${ADAPTER_DIR}" \
  --max-lora-rank "$RANK" \
  --max-loras 2 --max-cpu-loras 2 \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  > /mnt/workspace/vllm_lora.log 2>&1 &

echo "等待就绪..."
for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "$NAME"; then
    echo "== models 列表 =="
    curl -s "http://127.0.0.1:${PORT}/v1/models" | python -c \
      "import json,sys;[print(' -',m['id']) for m in json.load(sys.stdin)['data']]"
    exit 0
  fi
  sleep 5
done
echo "[FAIL] vLLM 未在 300s 内就绪，查看 /mnt/workspace/vllm_lora.log"
exit 1
