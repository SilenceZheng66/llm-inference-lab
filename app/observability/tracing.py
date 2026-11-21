"""trace_id 中间件 + 结构化访问日志。"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("lab.access")


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            log.error(
                {"event": "request", "trace_id": trace_id, "path": request.url.path, "status": 500},
                extra={},
            )
            raise
        response.headers["x-trace-id"] = trace_id
        log.info(
            "event=request trace_id=%s path=%s status=%s ms=%.1f",
            trace_id,
            request.url.path,
            response.status_code,
            (time.monotonic() - t0) * 1000,
        )
        return response
