#!/usr/bin/env bash
# 空闲关停后的一键恢复：模型已持久化，直接重启引擎与网关
set -euo pipefail

cd /mnt/workspace/llm-inference-lab

echo "== 1/3 重启 vLLM 后端 =="
bash notebooks/start_vllm.sh

echo "== 2/3 重启网关（真实后端配置）=="
export LAB_CONFIG=config/gpu.yaml
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >/tmp/gateway.log 2>&1 &
sleep 3
curl -s localhost:8000/healthz

echo
echo "== 3/3 端到端自检 =="
python notebooks/e2e_check.py --url http://127.0.0.1:8000 --stream
