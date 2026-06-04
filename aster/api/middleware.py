from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from aster.core.config import APISettings

PUBLIC_PATHS = frozenset({"/health", "/ready", "/metrics"})


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = max(0, requests_per_minute)
        self.window_seconds = 60.0
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute > 0

    def check(self, client_id: str, *, now: float | None = None) -> tuple[bool, int]:
        if not self.enabled:
            return True, 0

        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            requests = self._requests[client_id]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self.requests_per_minute:
                retry_after = int(requests[0] + self.window_seconds - current) + 1
                return False, max(1, retry_after)
            requests.append(current)
        return True, 0


def install_api_middleware(app: FastAPI, settings: APISettings) -> None:
    limiter = RateLimiter(settings.rate_limit_per_minute)
    api_key = settings.api_key

    @app.middleware("http")
    async def api_guard(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if path not in PUBLIC_PATHS:
            if api_key is not None:
                auth_header = request.headers.get("Authorization", "")
                scheme, _, token = auth_header.partition(" ")
                if scheme.lower() != "bearer" or not token:
                    return _error_response(401, "api_key_required", "API key required")
                if not secrets.compare_digest(token, api_key):
                    return _error_response(401, "invalid_api_key", "Invalid API key")

            if limiter.enabled:
                client_id = request.headers.get("Authorization") or (
                    request.client.host if request.client else "unknown"
                )
                allowed, retry_after = limiter.check(client_id)
                if not allowed:
                    return _error_response(
                        429,
                        "rate_limit_exceeded",
                        f"Rate limit exceeded. Retry after {retry_after} seconds.",
                        headers={"Retry-After": str(retry_after)},
                    )

        return await call_next(request)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": code, "message": message, "details": {}}},
        headers=headers,
    )

