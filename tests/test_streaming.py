from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from aster.api.feature_emulation import FeaturePlan
from aster.api.interaction_loop import stream_live_tool_interaction
from aster.api.provider_gateway import LocalProviderRequest, encode_provider_stream
from aster.api.streaming import (
    SSE_HEARTBEAT_SECONDS,
    SSE_SEND_TIMEOUT_SECONDS,
    iter_chat_tool_sse_events,
    iter_responses_tool_sse_events,
    to_chat_sse,
    to_completion_sse,
)
from aster.core.errors import AsterError
from aster.inference.decode_engine import DecodeChunk
from aster.inference.engine import InferenceRequest


async def _finished_chunks():
    yield DecodeChunk(token="", index=0, finished=True)


class _RawRequest:
    _is_disconnected = False


def test_chat_sse_uses_fast_heartbeat_and_send_timeout() -> None:
    async def scenario() -> None:
        response = await to_chat_sse(
            _finished_chunks(),
            "dummy-model",
            response_id="chat-sse-settings",
        )

        assert response.ping_interval == SSE_HEARTBEAT_SECONDS
        assert response.send_timeout == SSE_SEND_TIMEOUT_SECONDS

    asyncio.run(scenario())


def test_completion_sse_uses_fast_heartbeat_and_send_timeout() -> None:
    async def scenario() -> None:
        response = await to_completion_sse(
            _finished_chunks(),
            "dummy-model",
            response_id="completion-sse-settings",
        )

        assert response.ping_interval == SSE_HEARTBEAT_SECONDS
        assert response.send_timeout == SSE_SEND_TIMEOUT_SECONDS

    asyncio.run(scenario())


def test_chat_sse_wraps_raw_request_with_disconnect_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_request = object()
    errors: list[AsterError] = []
    on_error = errors.append
    captured: dict[str, object] = {}

    def fake_stream_with_disconnect(
        events: AsyncIterator[dict[str, str]],
        request: object,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, str]]:
        captured["events"] = events
        captured["request"] = request
        captured.update(kwargs)
        return events

    monkeypatch.setattr("aster.api.streaming.stream_with_disconnect", fake_stream_with_disconnect)

    async def scenario() -> None:
        response = await to_chat_sse(
            _finished_chunks(),
            "dummy-model",
            response_id="chat-disconnect-settings",
            raw_request=raw_request,
            on_error=on_error,
        )

        assert response.ping_interval == SSE_HEARTBEAT_SECONDS
        assert captured["request"] is raw_request
        assert captured["heartbeat_interval_seconds"] == SSE_HEARTBEAT_SECONDS
        assert captured["on_error"] is on_error

    asyncio.run(scenario())


def test_completion_sse_wraps_raw_request_with_disconnect_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_request = object()
    errors: list[AsterError] = []
    on_error = errors.append
    captured: dict[str, object] = {}

    def fake_stream_with_disconnect(
        events: AsyncIterator[dict[str, str]],
        request: object,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, str]]:
        captured["events"] = events
        captured["request"] = request
        captured.update(kwargs)
        return events

    monkeypatch.setattr("aster.api.streaming.stream_with_disconnect", fake_stream_with_disconnect)

    async def scenario() -> None:
        response = await to_completion_sse(
            _finished_chunks(),
            "dummy-model",
            response_id="completion-disconnect-settings",
            raw_request=raw_request,
            on_error=on_error,
        )

        assert response.ping_interval == SSE_HEARTBEAT_SECONDS
        assert captured["request"] is raw_request
        assert captured["heartbeat_interval_seconds"] == SSE_HEARTBEAT_SECONDS
        assert captured["on_error"] is on_error

    asyncio.run(scenario())


def test_chat_sse_disconnect_closes_source_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aster.api.streaming.SSE_DISCONNECT_POLL_SECONDS", 0.001)

    async def scenario() -> None:
        request = _RawRequest()
        closed = asyncio.Event()
        errors: list[AsterError] = []

        async def chunks():
            try:
                yield DecodeChunk(token="a", index=0, finished=False)
                await asyncio.Event().wait()
            finally:
                closed.set()

        response = await to_chat_sse(
            chunks(),
            "dummy-model",
            response_id="chat-disconnect-close",
            raw_request=request,
            on_error=errors.append,
        )
        events = response.body_iterator

        role_event = await anext(events)
        token_event = await anext(events)
        request._is_disconnected = True

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(events), timeout=1)

        assert '"role": "assistant"' in role_event["data"]
        assert '"content": "a"' in token_event["data"]
        assert closed.is_set()
        assert [error.code for error in errors] == ["client_disconnected"]

    asyncio.run(scenario())


def test_completion_sse_disconnect_closes_source_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aster.api.streaming.SSE_DISCONNECT_POLL_SECONDS", 0.001)

    async def scenario() -> None:
        request = _RawRequest()
        closed = asyncio.Event()
        errors: list[AsterError] = []

        async def chunks():
            try:
                yield DecodeChunk(token="a", index=0, finished=False)
                await asyncio.Event().wait()
            finally:
                closed.set()

        response = await to_completion_sse(
            chunks(),
            "dummy-model",
            response_id="completion-disconnect-close",
            raw_request=request,
            on_error=errors.append,
        )
        events = response.body_iterator

        token_event = await anext(events)
        request._is_disconnected = True

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(events), timeout=1)

        assert '"text": "a"' in token_event["data"]
        assert closed.is_set()
        assert [error.code for error in errors] == ["client_disconnected"]

    asyncio.run(scenario())


def test_provider_sse_uses_fast_heartbeat_and_send_timeout() -> None:
    parsed = LocalProviderRequest(
        provider="openai",
        api_family="chat_completions",
        model="dummy-model",
        inference_request=InferenceRequest(messages=[]),
        request_id="provider-sse-settings",
        stream=True,
        feature_plan=FeaturePlan(),
    )

    response = encode_provider_stream(parsed, _finished_chunks())

    assert response.ping_interval == SSE_HEARTBEAT_SECONDS
    assert response.send_timeout == SSE_SEND_TIMEOUT_SECONDS


def test_live_tool_sse_uses_fast_heartbeat_and_send_timeout() -> None:
    async def scenario() -> None:
        parsed = LocalProviderRequest(
            provider="openai",
            api_family="chat_completions",
            model="dummy-model",
            inference_request=InferenceRequest(messages=[]),
            request_id="live-tool-sse-settings",
            stream=True,
            feature_plan=FeaturePlan(mode="tools"),
        )

        response = await stream_live_tool_interaction(object(), parsed)

        assert response.ping_interval == SSE_HEARTBEAT_SECONDS
        assert response.send_timeout == SSE_SEND_TIMEOUT_SECONDS

    asyncio.run(scenario())


def test_responses_tool_sse_emits_failed_event_on_aster_error() -> None:
    async def failing_chunks():
        yield DecodeChunk(token="partial", index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
        )

    async def scenario() -> None:
        events: list[tuple[str | None, dict[str, object]]] = []
        async for event in iter_responses_tool_sse_events(
            failing_chunks(),
            "dummy-model",
            response_id="resp_tool_error",
        ):
            events.append((event.get("event"), json.loads(event["data"])))

        event_names = [name for name, _ in events]
        failed = events[-1][1]["response"]

        assert "response.completed" not in event_names
        assert event_names[-1] == "response.failed"
        assert isinstance(failed, dict)
        assert failed["status"] == "failed"
        assert failed["output"][0]["status"] == "incomplete"
        assert failed["output_text"] == "partial"
        assert failed["error"] == {
            "code": "request_timeout",
            "message": "Inference request timed out",
            "type": "request_timeout",
        }

    asyncio.run(scenario())


def test_responses_tool_sse_marks_partial_function_call_incomplete_on_length() -> None:
    async def partial_tool_chunks():
        yield DecodeChunk(
            token='<tool_call>{"name":"lookup_weather","arguments":{"city":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 1, "finish_reason": "length"},
        )

    async def scenario() -> None:
        events: list[tuple[str | None, dict[str, object]]] = []
        async for event in iter_responses_tool_sse_events(
            partial_tool_chunks(),
            "dummy-model",
            response_id="resp_tool_partial",
        ):
            events.append((event.get("event"), json.loads(event["data"])))

        completed = events[-1][1]["response"]

        assert isinstance(completed, dict)
        assert completed["status"] == "incomplete"
        assert completed["incomplete_details"] == {"reason": "max_output_tokens"}
        assert completed["output_text"] == ""
        assert [item["type"] for item in completed["output"]] == ["function_call"]
        function_call = completed["output"][0]
        assert function_call["status"] == "incomplete"
        assert function_call["name"] == "lookup_weather"
        assert function_call["arguments"] == '{"city":'

    asyncio.run(scenario())


def test_chat_tool_sse_reports_length_for_partial_function_call() -> None:
    async def partial_tool_chunks():
        yield DecodeChunk(
            token='<tool_call>{"name":"lookup_weather","arguments":{"city":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 1, "finish_reason": "length"},
        )

    async def scenario() -> None:
        payloads: list[dict[str, object]] = []
        async for event in iter_chat_tool_sse_events(
            partial_tool_chunks(),
            "dummy-model",
            response_id="chatcmpl_partial_tool",
        ):
            data = event.get("data")
            if data != "[DONE]":
                payloads.append(json.loads(data))

        chunks = [payload for payload in payloads if payload.get("object") == "chat.completion.chunk"]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        finish_reasons = [
            chunk["choices"][0]["finish_reason"]
            for chunk in chunks
            if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
        ]

        assert any(delta.get("tool_calls") for delta in deltas)
        assert finish_reasons == ["length"]

    asyncio.run(scenario())
