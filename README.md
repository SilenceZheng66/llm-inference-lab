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

- **P0**（已完成）：鉴权、RR 路由、流式/非流式转发、熔断 fallback、mock 全链路、compose 观测栈、Qwen2.5 真后端压测
- **P1**（已完成）：**训练一体闭环**（手写 LoRA SFT + kill-resume + eval +25.3% + vLLM LoRA 服务化 + profiler，见上节）；least-queue/prefix-hash 调优与限流进行中（姊妹项目 [infer-router](https://github.com/SilenceZheng66/infer-router) 已实现 prefix_affinity 路由与双副本对照实测）
- **P2（不做）**：多租户 RBAC、计费治理、云 provider 聚合、MCP

## ModelScope 免费算力用法

见 `notebooks/`：CPU 实例跑 mock 链路与单测；GPU 实例（约 36h 额度）分批起 vLLM 跑 Qwen2.5 真后端，模型缓存放 `/mnt/workspace` 持久目录，`resume.sh` 一键重启。

## 训推一体（training/）

推理之外，本仓库用手写训练循环补齐「训」的一侧——**不依赖 TRL/LLaMA-Factory 等高层封装**，直接基于 `transformers` + `peft` 实现 LoRA SFT 全流程，并与本仓库的 vLLM 推理栈闭环：

```
prepare_data.py ──► sft_lora.py ──► serve_vllm_lora.sh ──► eval_compare.py
   数据管线           手写训练循环        LoRA 服务化          base vs tuned 对比
  (MsDataset,      (assistant-mask,   (单实例双 model,      (ROUGE-L / coverage
   json 筛选)       bf16, accum,       --enable-lora)        / latency 同口径)
                    cosine, ckpt,
                    resume, profiler)
```

### 实测结果（Qwen2.5-1.5B + alpaca-gpt4-data-zh json 子集，A10 单卡）

| 项 | 结果 |
|---|---|
| LoRA SFT（r8 α32 all-linear，1200 样本，150 step） | loss 1.15 → 0.9 区间，B 范数 0.10-0.25（真实更新） |
| **eval 对比（50 条同口径）** | **rouge_l 0.3025 → 0.3789（+25.3%）**，latency 2.177 → 2.044s |
| kill-resume | kill -9 @step50 → resume，loss jump 0.051（阈值 0.35）PASS |
| vLLM LoRA 服务化 | 单实例同时服务 base + adapter；json 行为差异实证：tuned 输出向训练分布收敛（整数价格/库存） |
| profiler | torch.profiler trace + step 表（tok_s ≈ 1667，峰值显存 6.82GB） |

### 调试实录：三个静默失败 bug

训练"成功跑完但什么都没学到"的完整诊断过程（面试级素材，详见 lab-notes）：

1. HF `load_dataset` 加载 ModelScope 专属 repo → ConnectionError → MsDataset 优先 + HF 兜底；
2. 批量组装字段错传整 dict → `KeyError: slice`；
3. **最隐蔽的两个**：`gradient_checkpointing_enable()` 后未调 `enable_input_require_grads()` → embedding 无梯度 → **LoRA 梯度恒零 → loss 全程平坦 → adapter 是 B=0 恒等映射**；resume 未加载 adapter_model.safetensors → kill 前成果静默丢失。诊断路径：loss 平坦 → lr 正常 → 打印 lora_B 范数判决。

### 用法

```bash
python training/prepare_data.py          # 数据下载与筛选（out_dir 可配）
python training/sft_lora.py              # 训练（config.yaml 驱动，支持 --resume）
bash training/resume_test.sh             # kill-resume 演练（自动断点续训并校验 loss 接续）
bash training/serve_vllm_lora.sh         # vLLM LoRA 服务化（base + adapter 同端点）
python training/eval_compare.py --config training/config.yaml run --model-mode base|tuned --out <f>
python training/eval_compare.py compare <base.json> <tuned.json>
```

配套单测 `tests/test_prepare_data.py`（pytest 全绿）。`training/ddp_gloo_demo.py` 为 CPU 可跑的 DDP/gloo 通信演示。



## 网关压测数据（ModelScope 免费 A10 单卡，2026-11-22）

环境：NVIDIA A10 24GB / vLLM 0.10.1.1 / Qwen2.5-1.5B-Instruct（`--served-model-name qwen2.5-1.5b --max-model-len 8192 --gpu-memory-utilization 0.85`）；网关以 `config/gpu.yaml` 启动（vLLM 主后端 + mock 兜底，RR 路由）；压测 `bench/run_bench.py -n 32`（流式，经网关入口测量，含鉴权/路由/转发开销）。

| 并发 | wall | TTFT avg/p50/p95 | TPOT avg/p95 | 吞吐 | tokens | err |
|---|---|---|---|---|---|---|
| 1  | 10.03s | 10.5 / 15.7 / 17.7 ms  | 11.3 / 14.4 ms | 106.5 tok/s | 1068 | 0/32 |
| 8  | 1.18s  | 26.2 / 18.4 / 75.3 ms  | 13.2 / 19.5 ms | 514.9 tok/s | 606  | 0/32 |
| 32 | 1.46s  | 273.4 / 282.9 / 294.5 ms | 11.6 / 14.8 ms | 823.9 tok/s | 1201 | 0/32 |

解读：单并发 TTFT ~10ms（A10 上 1.5B 预填充很快）；并发 8 吞吐近 5 倍提升；并发 32 达 824 tok/s，TTFT 因排队升至 ~270ms 而 TPOT 几乎不变——瓶颈在排队而非解码。并发 8/32 档 tokens 偏少是因为 RR 把部分请求分给 mock 兜底后端（固定短回复），属网关混合后端下的真实口径。端到端冒烟：`e2e_check.py` 经网关拿到 Qwen2.5 真实流式回复，E2E OK。

> 诚实口径：本项目为学习/作品集项目，未接入真实生产流量，不声明 SLA / 高可用 / 大规模并发能力；性能结论仅来自本地与免费算力环境的实测。
