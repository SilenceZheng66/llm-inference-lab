"""路由策略：RoundRobin / LeastQueue / PrefixHash / Weighted。"""
from __future__ import annotations

import hashlib
import itertools
import random
from abc import ABC, abstractmethod

from .registry import BackendState


class Strategy(ABC):
    name = "base"

    @abstractmethod
    def select(self, candidates: list[BackendState], ctx: dict) -> BackendState | None: ...


class RoundRobin(Strategy):
    name = "round_robin"

    def __init__(self) -> None:
        self._i = itertools.count()

    def select(self, candidates: list[BackendState], ctx: dict) -> BackendState | None:
        if not candidates:
            return None
        return candidates[next(self._i) % len(candidates)]


class LeastQueue(Strategy):
    """选中在途请求最少的后端；并列时 RR 决胜。"""

    name = "least_queue"

    def __init__(self) -> None:
        self._rr = RoundRobin()

    def select(self, candidates: list[BackendState], ctx: dict) -> BackendState | None:
        if not candidates:
            return None
        m = min(s.in_flight for s in candidates)
        tied = [s for s in candidates if s.in_flight == m]
        return self._rr.select(tied, ctx)


class PrefixHash(Strategy):
    """按首条用户消息前缀哈希选择后端：稳定亲和，配合引擎 Prefix Cache 提升命中率。"""

    name = "prefix_hash"

    def select(self, candidates: list[BackendState], ctx: dict) -> BackendState | None:
        if not candidates:
            return None
        key = str(ctx.get("prefix_key") or "")
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
        return candidates[h % len(candidates)]


class Weighted(Strategy):
    name = "weighted"

    def select(self, candidates: list[BackendState], ctx: dict) -> BackendState | None:
        if not candidates:
            return None
        weights = [max(1, s.spec.weight) for s in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]


STRATEGIES: dict[str, Strategy] = {
    cls().name: cls() for cls in (RoundRobin, LeastQueue, PrefixHash, Weighted)
}
