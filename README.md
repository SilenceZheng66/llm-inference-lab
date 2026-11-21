# llm-inference-lab

生产风格的大模型推理服务**全链路参考实现**（学习/转岗项目，非生产系统）：

```
OpenAI 兼容客户端
      │  Bearer API Key
      ▼
┌─────────────────────────────┐
│  Gateway (FastAPI, :8000)   │
│  auth → registry → router   │
│  → breaker → proxy          │
│  metrics / tracing / cost   │
└───────┬──────────┬──────────┘
        │          │
   vLLM SLM     vLLM LLM    (ModelScope GPU / Ollama CPU / mock)
```

## 定位

- 广度项目：把「接入 → 鉴权 → 注册表 → 智能路由 → 引擎池 → 熔断降级 → 可观测/成本」完整跑通一遍。
- 后端即插即用：vLLM / SGLang / Ollama / llama.cpp server（均 OpenAI 兼容）+ 本地 mock（无 GPU 也能全链路演示）。
- 实测数据进 README（TTFT / TPOT / 吞吐 / 附加延迟），作为面试展开的核心谈资。
- 姊妹项目 [`infer-router`](../infer-router)：把本项目的路由层用 Go 重写为高性能独立组件。

## 快速开始（本地 CPU，无需 GPU）

```bash
# 1. 起 mock 后端 ×2
uvicorn backends.mock_openai:app --port 8001 &
uvicorn backends.mock_openai:app --port 8002 &

# 2. 起网关
uvicorn app.main:app --port 8000

# 3. 请求
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-lab-demo-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

配置在 `config/backends.yaml`：API keys、路由策略（`round_robin | least_queue | prefix_hash | weighted`）、后端清单与健康检查/熔断参数。

## 模块地图

| 模块 | 文件 | 说明 |
|---|---|---|
| 鉴权 | `app/auth.py` | Bearer API Key 校验 |
| 注册表 | `app/registry.py` | 后端健康探测（`GET {base}/models`）、在途计数 |
| 路由 | `app/router.py` | RR / least-queue / prefix-hash / weighted |
| 熔断 | `app/breaker.py` | closed → open → half-open，fallback 顺序跳过 open 后端 |
| 转发 | `app/proxy.py` | 流式 SSE 解析与非流式透传 |
| 观测 | `app/observability/` | Prometheus 指标、trace_id、TTFT/ITL、成本归因 |
| mock | `backends/mock_openai.py` | OpenAI 兼容 mock（流式+usage） |
| 压测 | `bench/run_bench.py` | 并发梯度输出 TTFT/TPOT/吞吐/错误率 |

## 指标（`/metrics`）

- `lab_requests_total{backend,model,status}` 请求计数
- `lab_request_latency_seconds` 端到端延迟
- `lab_ttft_seconds` 首 token 延迟 / `lab_itl_seconds` token 间延迟
- `lab_tokens_total{direction}` token 用量 / `lab_cost_usd_total` 成本归因
- `lab_in_flight_requests` 后端在途数

## 路线图

- **P0**（当前）：鉴权、RR 路由、流式/非流式转发、熔断 fallback、mock 全链路、compose 观测栈
- **P1**：least-queue/prefix-hash 实战调优、限流、前缀命中率统计、ModelScope GPU 上 vLLM/SGLang 真后端压测报告
- **P2（不做）**：多租户 RBAC、计费治理、云 provider 聚合、MCP

## ModelScope 免费算力用法

见 `notebooks/`：CPU 实例跑 mock 链路与单测；GPU 实例（约 36h 额度）分批起 vLLM 跑 Qwen2.5 真后端，模型缓存放 `/mnt/workspace` 持久目录，`resume.sh` 一键重启。

> 诚实口径：本项目为学习/作品集项目，未接入真实生产流量，不声明 SLA / 高可用 / 大规模并发能力；性能结论仅来自本地与免费算力环境的实测。
