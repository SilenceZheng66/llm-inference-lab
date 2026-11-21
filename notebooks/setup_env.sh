#!/usr/bin/env bash
# ModelScope Notebook（CPU 实例）环境初始化 + mock 全链路启动
# 用法：在 /mnt/workspace 下执行 bash setup_env.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/<OWNER>/llm-inference-lab.git}"
cd /mnt/workspace

if [ ! -d llm-inference-lab ]; then
  git clone "$REPO_URL"
fi
cd llm-inference-lab

pip install -q fastapi "uvicorn[standard]" httpx pyyaml prometheus-client pytest pytest-asyncio

echo "== 单测 =="
python -m pytest -q

echo "== 启动 mock 后端 ×2 =="
nohup python -m uvicorn backends.mock_openai:app --host 127.0.0.1 --port 8001 >/tmp/mock-a.log 2>&1 &
nohup python -m uvicorn backends.mock_openai:app --host 127.0.0.1 --port 8002 >/tmp/mock-b.log 2>&1 &

echo "== 启动网关 =="
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >/tmp/gateway.log 2>&1 &

sleep 3
echo "== 健康检查 =="
curl -s localhost:8000/healthz
echo
echo "== 端到端（流式）=="
python notebooks/e2e_check.py --url http://127.0.0.1:8000 --stream
