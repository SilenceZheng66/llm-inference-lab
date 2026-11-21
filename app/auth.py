"""Bearer API Key 鉴权。"""
from __future__ import annotations

from fastapi import HTTPException, Request

from .config import Settings


def require_api_key(request: Request, settings: Settings) -> None:
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not key or key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="invalid api key")
