"""后端注册表：健康探测 + 在途计数 + 熔断挂载。"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from .breaker import Breaker
from .config import Backend, Settings

log = logging.getLogger("lab.registry")


@dataclass
class BackendState:
    spec: Backend
    healthy: bool = True
    in_flight: int = 0
    last_check: float = 0.0
    breaker: Breaker = field(init=False)

    def __post_init__(self) -> None:
        self.breaker = Breaker()


class Registry:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.states: dict[str, BackendState] = {b.name: BackendState(spec=b) for b in settings.backends}
        self._client = client or httpx.AsyncClient(timeout=5.0)
        self._task: asyncio.Task | None = None

    def healthy_states(self) -> list[BackendState]:
        return [s for s in self.states.values() if s.healthy and s.breaker.allow()]

    async def _probe_once(self, state: BackendState) -> bool:
        url = state.spec.base_url.rstrip("/") + "/models"
        try:
            r = await self._client.get(url)
            return r.status_code == 200
        except Exception:
            return False

    async def check_all(self) -> None:
        for state in self.states.values():
            ok = await self._probe_once(state)
            state.healthy = ok
            state.last_check = time.monotonic()
            if ok:
                # 健康探测成功视为后端可用，重置熔断
                state.breaker.record_success()

    async def start_health_loop(self) -> None:
        async def _loop() -> None:
            while True:
                try:
                    await self.check_all()
                except Exception:  # noqa: BLE001
                    log.exception("health loop error")
                await asyncio.sleep(self.settings.health_interval_s)

        self._task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()
