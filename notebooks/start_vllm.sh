#!/usr/bin/env bash
# GPU 实例：下载 Qwen2.5-1.5B 并用 vLLM 起真后端（模型缓存持久化到 /mnt/workspace）
set -euo pipefail

MODEL_DIR=/mnt/workspace/models/Qwen/Qwen2.5-1.5B-Instruct
PORT="${PORT:-8001}"
SERVED_NAME="${SERVED_NAME:-qwen2.5-1.5b}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "== 下载模型（首次较慢，持久化后不再重复）=="
  pip install -q modelscope
  python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", cache_dir="/mnt/workspace/models")
PY
fi

echo "== 安装 vLLM =="
pip install -q vllm

echo "== 启动 vLLM (port=$PORT) =="
nohup vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  >/tmp/vllm.log 2>&1 &

echo "等待 vLLM 就绪…"
for i in $(seq 1 120); do
  if curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready"; exit 0
  fi
  sleep 5
done
echo "vLLM 未在 10 分钟内就绪，请查看 /tmp/vllm.log"; exit 1
