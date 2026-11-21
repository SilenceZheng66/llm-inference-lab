"""成本归因：计价表 × token 用量（仅本地实验口径，非真实账单）。"""
from __future__ import annotations

# USD per 1M tokens（输入, 输出）—— 免费算力环境下的参考计价
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "qwen2.5-7b": (0.10, 0.20),
    "qwen2.5-1.5b": (0.02, 0.04),
    "qwen2.5-vl-3b": (0.05, 0.10),
    "mock": (0.0, 0.0),
}
DEFAULT_PRICE = (0.05, 0.10)


def _match_key(model: str) -> str:
    m = model.lower()
    for key in PRICE_TABLE:
        if key in m:
            return key
    return m


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICE_TABLE.get(_match_key(model), DEFAULT_PRICE)
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout
