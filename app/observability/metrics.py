"""Prometheus 指标定义。"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQ_TOTAL = Counter(
    "lab_requests_total", "Proxied chat requests", ["backend", "model", "status"]
)
REQ_LATENCY = Histogram(
    "lab_request_latency_seconds",
    "End-to-end request latency",
    ["backend", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
TTFT = Histogram(
    "lab_ttft_seconds",
    "Time to first token (streaming)",
    ["backend", "model"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
ITL = Histogram(
    "lab_itl_seconds",
    "Inter-token latency (streaming)",
    ["backend", "model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
TOKENS = Counter(
    "lab_tokens_total", "Token usage", ["backend", "model", "direction"]
)
COST = Counter(
    "lab_cost_usd_total", "Estimated cost (lab price table)", ["backend", "model"]
)
IN_FLIGHT = Gauge(
    "lab_in_flight_requests", "In-flight upstream requests", ["backend"]
)
PREFIX_HITS = Counter(
    "lab_prefix_route_total", "Routing decisions by strategy/backend", ["backend", "strategy"]
)
