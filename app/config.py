"""应用配置：YAML 加载 + 环境变量覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Backend:
    name: str
    base_url: str  # 形如 http://127.0.0.1:8001/v1
    model: str
    weight: int = 1


@dataclass
class Settings:
    api_keys: list[str] = field(default_factory=lambda: ["sk-lab-demo-key"])
    strategy: str = "round_robin"
    health_interval_s: float = 5.0
    breaker_fail_threshold: int = 3
    breaker_cooldown_s: float = 20.0
    backends: list[Backend] = field(default_factory=list)


def load_settings(path: str | None = None) -> Settings:
    p = Path(path or os.getenv("LAB_CONFIG", "config/backends.yaml"))
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Settings(
        api_keys=list(data.get("api_keys") or Settings().api_keys),
        strategy=str(data.get("strategy") or "round_robin"),
        health_interval_s=float(data.get("health_interval_s") or 5.0),
        breaker_fail_threshold=int(data.get("breaker_fail_threshold") or 3),
        breaker_cooldown_s=float(data.get("breaker_cooldown_s") or 20.0),
        backends=[Backend(**b) for b in data.get("backends") or []],
    )
