from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from time import monotonic
from typing import Any

from aster.core.errors import AsterError

DISCONNECT_POLL_SECONDS = 0.5
_MAX_CYCLE_SEARCH_DEPTH = 8


def _find_uvicorn_cycle(
    obj: object, *, depth: int = 0, visited: set[int] | None = None
) -> Any | None:
    if depth > _MAX_CYCLE_SEARCH_DEPTH:
        return None
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    disconnected = getattr(obj, "disconnected", None)
    if isinstance(disconnected, bool):
        return obj

    bound_self = getattr(obj, "__self__", None)
    if bound_self is not None:
        if cycle := _find_uvicorn_cycle(bound_self, depth=depth + 1, visited=visited):
            return cycle

    inner_receive = getattr(obj, "_receive", None)
    if inner_receive is not None:
        if cycle := _find_uvicorn_cycle(inner_receive, depth=depth + 1, visited=visited):
            return cycle

    closure = getattr(obj, "__closure__", None)
    if closure:
        for cell in closure:
            with suppress(ValueError):
                if cycle := _find_uvicorn_cycle(
                    cell.cell_contents, depth=depth + 1, visited=visited
                ):
                    return cycle

    return None


def is_client_disconnected(raw_request: Any) -> bool:
    if getattr(raw_request, "_is_disconnected", False):
        return True

    cycle = getattr(raw_request, "_uvicorn_cycle", None)
    if cycle is None and not getattr(raw_request, "_uvicorn_cycle_checked", False):
        receive = getattr(raw_request, "_receive", None)
        if receive is not None:
            cycle = _find_uvicorn_cycle(receive)
        raw_request._uvicorn_cycle = cycle
        raw_request._uvicorn_cycle_checked = True

    if cycle is not None and getattr(cycle, "disconnected", False):
        raw_request._is_disconnected = True
        return True
    return False


def _client_disconnected_error() -> AsterError:
    return AsterError(
        code="client_disconnected",
        message="Client disconnected before inference completed",
        status_code=499,
    )


async def await_with_disconnect[T](
    awaitable: Awaitable[T],
    raw_request: Any,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = DISCONNECT_POLL_SECONDS,
    on_error: Callable[[AsterError], None] | None = None,
) -> T:
    task = asyncio.ensure_future(awaitable)
    started = monotonic()

    async def wait_disconnect() -> None:
        while True:
            await asyncio.sleep(poll_interval_seconds)
            if is_client_disconnected(raw_request):
                return

    disconnect_task = asyncio.create_task(wait_disconnect())
    try:
        done, _ = await asyncio.wait(
            {task, disconnect_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
            elapsed = monotonic() - started
            exc = AsterError(
                code="request_timeout",
                message="Inference request timed out",
                status_code=504,
                details={"timeout_seconds": timeout_seconds, "elapsed_seconds": round(elapsed, 4)},
            )
            if on_error is not None:
                on_error(exc)
            raise exc
        if task in done:
            try:
                return task.result()
            except asyncio.CancelledError:
                if is_client_disconnected(raw_request):
                    exc = _client_disconnected_error()
                    if on_error is not None:
                        on_error(exc)
                    raise exc from None
                raise

        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
        exc = _client_disconnected_error()
        if on_error is not None:
            on_error(exc)
        raise exc
    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
        if not task.done():
            task.cancel()


async def stream_with_disconnect[T](
    generator: AsyncIterator[T],
    raw_request: Any,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = DISCONNECT_POLL_SECONDS,
    heartbeat_interval_seconds: float | None = None,
    heartbeat_factory: Callable[[], T] | None = None,
    on_error: Callable[[AsterError], None] | None = None,
) -> AsyncIterator[T]:
    started = monotonic()
    aiter = generator.__aiter__()

    async def wait_disconnect() -> None:
        while True:
            await asyncio.sleep(poll_interval_seconds)
            if is_client_disconnected(raw_request):
                return

    disconnect_task = asyncio.create_task(wait_disconnect())
    anext_task: asyncio.Task[T] | None = None
    try:
        while True:
            elapsed = monotonic() - started
            remaining_timeout = (
                None if timeout_seconds is None else max(timeout_seconds - elapsed, 0.0)
            )
            if remaining_timeout is not None and remaining_timeout <= 0:
                if on_error is not None:
                    on_error(
                        AsterError(
                            code="request_timeout",
                            message="Inference request timed out",
                            status_code=504,
                            details={
                                "timeout_seconds": timeout_seconds,
                                "elapsed_seconds": round(elapsed, 4),
                            },
                        )
                    )
                break

            if anext_task is None:
                anext_task = asyncio.ensure_future(aiter.__anext__())

            wait_timeout = heartbeat_interval_seconds
            if remaining_timeout is not None:
                wait_timeout = (
                    remaining_timeout
                    if wait_timeout is None
                    else min(wait_timeout, remaining_timeout)
                )
            done, _ = await asyncio.wait(
                {anext_task, disconnect_task},
                timeout=wait_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                if on_error is not None:
                    on_error(_client_disconnected_error())
                break
            if anext_task in done:
                try:
                    item = anext_task.result()
                except StopAsyncIteration:
                    break
                anext_task = None
                yield item
                continue
            if heartbeat_factory is not None:
                yield heartbeat_factory()
    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
        if anext_task is not None:
            if not anext_task.done():
                anext_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await anext_task
        with suppress(Exception):
            await aiter.aclose()
