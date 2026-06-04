from __future__ import annotations

import asyncio

import pytest

from aster.api.disconnect import (
    await_with_disconnect,
    is_client_disconnected,
    stream_with_disconnect,
)
from aster.core.errors import AsterError


class FakeRequest:
    _is_disconnected = False


def test_is_client_disconnected_reads_uvicorn_cycle_from_receive_closure() -> None:
    class FakeCycle:
        disconnected = True

    cycle = FakeCycle()

    async def receive() -> dict[str, object]:
        return {"disconnected": cycle.disconnected}

    request = FakeRequest()
    request._receive = receive

    assert is_client_disconnected(request) is True
    assert request._is_disconnected is True


def test_await_with_disconnect_returns_completed_result() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        result = await await_with_disconnect(
            asyncio.sleep(0, result="done"),
            request,
            timeout_seconds=1,
            poll_interval_seconds=0.001,
        )

        assert result == "done"

    asyncio.run(scenario())


def test_await_with_disconnect_converts_cancelled_work_after_disconnect() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        request._is_disconnected = True
        errors: list[AsterError] = []

        async def work() -> str:
            await asyncio.Event().wait()
            return "unreachable"

        task = asyncio.create_task(work())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with pytest.raises(AsterError) as exc_info:
            await await_with_disconnect(
                task,
                request,
                timeout_seconds=1,
                poll_interval_seconds=0.001,
                on_error=errors.append,
            )

        assert exc_info.value.code == "client_disconnected"
        assert exc_info.value.status_code == 499
        assert [error.code for error in errors] == ["client_disconnected"]

    asyncio.run(scenario())


def test_await_with_disconnect_preserves_cancelled_work_without_disconnect() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        errors: list[AsterError] = []

        async def work() -> str:
            await asyncio.Event().wait()
            return "unreachable"

        task = asyncio.create_task(work())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with pytest.raises(asyncio.CancelledError):
            await await_with_disconnect(
                task,
                request,
                timeout_seconds=1,
                poll_interval_seconds=0.001,
                on_error=errors.append,
            )

        assert errors == []

    asyncio.run(scenario())


def test_await_with_disconnect_cancels_work_on_disconnect() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        errors: list[AsterError] = []

        async def work() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return "unreachable"

        task = asyncio.create_task(
            await_with_disconnect(
                work(),
                request,
                timeout_seconds=1,
                poll_interval_seconds=0.001,
                on_error=errors.append,
            )
        )
        await started.wait()
        request._is_disconnected = True

        with pytest.raises(AsterError) as exc_info:
            await task

        assert exc_info.value.code == "client_disconnected"
        assert exc_info.value.status_code == 499
        assert [error.code for error in errors] == ["client_disconnected"]
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_await_with_disconnect_times_out_and_cancels_work() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        cancelled = asyncio.Event()
        errors: list[AsterError] = []

        async def work() -> str:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return "unreachable"

        with pytest.raises(AsterError) as exc_info:
            await await_with_disconnect(
                work(),
                request,
                timeout_seconds=0.001,
                poll_interval_seconds=0.001,
                on_error=errors.append,
            )

        assert exc_info.value.code == "request_timeout"
        assert exc_info.value.status_code == 504
        assert [error.code for error in errors] == ["request_timeout"]
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_stream_with_disconnect_closes_generator_on_disconnect() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        started = asyncio.Event()
        closed = asyncio.Event()
        errors: list[AsterError] = []

        async def stream():
            try:
                started.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        guarded = stream_with_disconnect(
            stream(),
            request,
            timeout_seconds=1,
            poll_interval_seconds=0.001,
            heartbeat_interval_seconds=0.5,
            on_error=errors.append,
        )
        task = asyncio.create_task(anext(guarded))
        await started.wait()
        request._is_disconnected = True

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(task, timeout=1)

        assert closed.is_set()
        assert [error.code for error in errors] == ["client_disconnected"]

    asyncio.run(scenario())


def test_stream_with_disconnect_drains_finished_anext_task_on_disconnect() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        request._is_disconnected = True
        errors: list[AsterError] = []
        leaked_contexts: list[dict[str, object]] = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: leaked_contexts.append(context))

        async def stream():
            raise AsterError(code="boom", message="boom", status_code=500)
            yield "unreachable"

        try:
            guarded = stream_with_disconnect(
                stream(),
                request,
                poll_interval_seconds=0,
                heartbeat_interval_seconds=1,
                on_error=errors.append,
            )

            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(guarded), timeout=1)
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        assert [error.code for error in errors] == ["client_disconnected"]
        assert leaked_contexts == []

    asyncio.run(scenario())


def test_stream_with_disconnect_reports_timeout_and_closes_generator() -> None:
    async def scenario() -> None:
        request = FakeRequest()
        closed = asyncio.Event()
        errors: list[AsterError] = []

        async def stream():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        guarded = stream_with_disconnect(
            stream(),
            request,
            timeout_seconds=0.001,
            poll_interval_seconds=0.001,
            on_error=errors.append,
        )

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(guarded), timeout=1)

        assert closed.is_set()
        assert [error.code for error in errors] == ["request_timeout"]
        assert errors[0].status_code == 504

    asyncio.run(scenario())


def test_stream_with_disconnect_emits_heartbeat_while_waiting() -> None:
    async def scenario() -> None:
        request = FakeRequest()

        async def stream():
            await asyncio.Event().wait()
            yield {"data": "unreachable"}

        guarded = stream_with_disconnect(
            stream(),
            request,
            timeout_seconds=1,
            poll_interval_seconds=0.001,
            heartbeat_interval_seconds=0.001,
            heartbeat_factory=lambda: {"comment": "heartbeat"},
        )

        assert await asyncio.wait_for(anext(guarded), timeout=1) == {"comment": "heartbeat"}
        await guarded.aclose()

    asyncio.run(scenario())
