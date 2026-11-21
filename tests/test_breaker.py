"""熔断器状态机单测。"""
from __future__ import annotations

import time

from app.breaker import Breaker


def test_opens_after_threshold():
    b = Breaker(fail_threshold=3, cooldown_s=60)
    assert b.allow()
    b.record_failure()
    b.record_failure()
    assert b.state == "closed"
    assert b.allow()
    b.record_failure()
    assert b.state == "open"
    assert not b.allow()


def test_half_open_after_cooldown_then_reopen_on_fail():
    b = Breaker(fail_threshold=1, cooldown_s=0.05)
    b.record_failure()
    assert b.state == "open"
    time.sleep(0.06)
    assert b.allow()
    assert b.state == "half_open"
    b.record_failure()
    assert b.state == "open"


def test_half_open_recovers_on_success():
    b = Breaker(fail_threshold=1, cooldown_s=0.05)
    b.record_failure()
    time.sleep(0.06)
    assert b.allow()
    b.record_success()
    assert b.state == "closed"
    assert b.failures == 0
    assert b.allow()
