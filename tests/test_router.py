"""路由策略单测：RR / LeastQueue / PrefixHash / Weighted。"""
from __future__ import annotations

from app.config import Backend
from app.registry import BackendState
from app.router import PrefixHash, LeastQueue, RoundRobin, Weighted


def make_backends(specs: list[tuple[str, int]]) -> list[BackendState]:
    return [
        BackendState(spec=Backend(name=n, base_url=f"http://{n}/v1", model="mock-model", weight=w))
        for n, w in specs
    ]


def test_round_robin_cycles():
    rr = RoundRobin()
    cands = make_backends([("a", 1), ("b", 1), ("c", 1)])
    picked = [rr.select(cands, {}).spec.name for _ in range(6)]
    assert picked == ["a", "b", "c", "a", "b", "c"]


def test_round_robin_empty():
    rr = RoundRobin()
    assert rr.select([], {}) is None


def test_least_queue_picks_min_in_flight():
    lq = LeastQueue()
    cands = make_backends([("a", 1), ("b", 1), ("c", 1)])
    cands[0].in_flight = 5
    cands[2].in_flight = 3
    assert lq.select(cands, {}).spec.name == "b"


def test_least_queue_tie_break_round_robin():
    lq = LeastQueue()
    cands = make_backends([("a", 1), ("b", 1)])
    assert lq.select(cands, {}).spec.name == "a"
    assert lq.select(cands, {}).spec.name == "b"


def test_prefix_hash_stable_affinity():
    ph = PrefixHash()
    cands = make_backends([("a", 1), ("b", 1)])
    first = ph.select(cands, {"prefix_key": "系统提示词X"})
    for _ in range(20):
        assert ph.select(cands, {"prefix_key": "系统提示词X"}) is first


def test_prefix_hash_distributes():
    ph = PrefixHash()
    cands = make_backends([("a", 1), ("b", 1)])
    seen = {ph.select(cands, {"prefix_key": str(i)}).spec.name for i in range(50)}
    assert seen == {"a", "b"}


def test_weighted_proportion():
    w = Weighted()
    cands = make_backends([("a", 3), ("b", 1)])
    counts = {"a": 0, "b": 0}
    for _ in range(400):
        counts[w.select(cands, {}).spec.name] += 1
    ratio = counts["a"] / (counts["a"] + counts["b"])
    assert 0.55 < ratio < 0.95, counts
