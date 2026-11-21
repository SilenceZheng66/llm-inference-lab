"""鉴权与成本归因单测。"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.auth import require_api_key
from app.config import Settings
from app.observability.cost import cost_usd


def test_auth_valid_key_passes():
    s = Settings(api_keys=["sk-lab-demo-key"])
    req = SimpleNamespace(headers={"authorization": "Bearer sk-lab-demo-key"})
    require_api_key(req, s)  # 不抛异常即通过


def test_auth_invalid_key_401():
    s = Settings(api_keys=["sk-lab-demo-key"])
    for headers in (
        {"authorization": "Bearer wrong"},
        {},
        {"authorization": "wrong-format"},
    ):
        req = SimpleNamespace(headers=headers)
        with pytest.raises(HTTPException) as ei:
            require_api_key(req, s)
        assert ei.value.status_code == 401


def test_cost_zero_for_mock():
    assert cost_usd("mock-model", 100, 100) == 0.0


def test_cost_matches_price_table():
    # qwen2.5-7b: (0.10, 0.20) USD / 1M tokens
    got = cost_usd("Qwen2.5-7B-Instruct", 1_000_000, 500_000)
    assert math.isclose(got, 0.10 + 0.10, rel_tol=1e-9)


def test_cost_default_fallback():
    got = cost_usd("unknown-model-x", 1_000_000, 0)
    assert math.isclose(got, 0.05, rel_tol=1e-9)
