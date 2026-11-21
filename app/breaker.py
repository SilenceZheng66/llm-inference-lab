"""每后端熔断器：closed → open → half_open → closed。"""
from __future__ import annotations

import time


class Breaker:
    def __init__(self, fail_threshold: int = 3, cooldown_s: float = 20.0):
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.state = "closed"
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.monotonic() - self.opened_at >= self.cooldown_s:
                self.state = "half_open"
                return True
            return False
        return True  # half_open：放行一次探测

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.fail_threshold:
            self.state = "open"
            self.opened_at = time.monotonic()
