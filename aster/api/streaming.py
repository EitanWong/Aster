from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import time
from typing import Any

from sse_starlette.sse import EventSourceResponse

from aster.api.disconnect import stream_with_disconnect
from aster.api.feature_emulation import (
    FeaturePlan,
    StreamingJsonFenceStripper,
    validate_structured_output_text,
)
from aster.core.errors import AsterError
from aster.inference.decode_engine import DecodeChunk
from aster.inference.parser_pipeline import ParsedGenerationDelta, ToolCallDelta
from aster.inference.reasoning_parsers import AutoReasoningParser
from aster.inference.tool_parsers import AutoToolParser
from aster.telemetry.logging import get_logger

_logger = get_logger(__name__)

SSE_HEARTBEAT_SECONDS = 5
SSE_DISCONNECT_POLL_SECONDS = 0.5
SSE_SEND_TIMEOUT_SECONDS = 30.0


def _usage_from_stats(stats: dict[str, object] | None) -> dict[str, int] | None:
    if not stats:
        return None
    prompt_tokens = _safe_int(stats.get("prompt_tokens"))
    completion_tokens = _safe_int(stats.get("completion_tokens"))
    if prompt_tokens is None and completion_tokens is None:
        return None
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def responses_usage_payload(
    input_tokens: int,
    output_tokens: int,
    *,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def responses_usage_from_stats(stats: dict[str, object] | None) -> dict[str, Any] | None:
    usage = _usage_from_stats(stats)
    if usage is None:
        return None
    cached_tokens = _safe_int(stats.get("cached_tokens")) if stats else None
    reasoning_tokens = _safe_int(stats.get("reasoning_tokens")) if stats else None
    return responses_usage_payload(
        usage["prompt_tokens"],
        usage["completion_tokens"],
        cached_tokens=cached_tokens or 0,
        reasoning_tokens=reasoning_tokens or 0,
    )


def responses_finish_reason_from_stats(stats: dict[str, object] | None) -> str | None:
    return _optional_finish_reason_from_stats(stats)


def responses_status_fields_from_finish_reason(finish_reason: str | None) -> dict[str, Any]:
    if finish_reason == "length":
        return {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
    return {"status": "completed"}


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _finish_reason_from_stats(stats: dict[str, object] | None, *, default: str = "stop") -> str:
    if stats and isinstance(stats.get("finish_reason"), str):
        return str(stats["finish_reason"])
    return default


def _optional_finish_reason_from_stats(stats: dict[str, object] | None) -> str | None:
    if stats and isinstance(stats.get("finish_reason"), str):
        return str(stats["finish_reason"])
    return None


def _chat_chunk(
    *,
    response_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"delta": delta, "index": 0, "finish_reason": finish_reason}],
    }


def chat_sse_event(
    *,
    response_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    created: int | None = None,
) -> dict[str, str]:
    return {
        "data": json.dumps(
            _chat_chunk(
                response_id=response_id,
                model=model,
                created=created if created is not None else int(time()),
                delta=delta,
                finish_reason=finish_reason,
            )
        )
    }


def chat_usage_sse_event(
    *,
    response_id: str,
    model: str,
    usage: dict[str, int],
    created: int | None = None,
) -> dict[str, str]:
    return {
        "data": json.dumps(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created if created is not None else int(time()),
                "model": model,
                "choices": [],
                "usage": usage,
            }
        )
    }


def _chat_tool_delta(tool_call: ToolCallDelta) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": tool_call.arguments_delta}
    if tool_call.name is not None:
        function["name"] = tool_call.name
    payload: dict[str, Any] = {
        "index": tool_call.index,
        "type": "function",
        "function": function,
    }
    if tool_call.call_id is not None:
        payload["id"] = tool_call.call_id
    return payload


def _tool_call_name(tool_call: ToolCallDelta) -> str:
    return tool_call.name or "function"


def _tool_call_id(tool_call: ToolCallDelta) -> str:
    return tool_call.call_id or f"call_{tool_call.index}"


def _completion_chunk(
    *,
    response_id: str,
    model: str,
    created: int,
    text: str,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _stream_error_payload(exc: AsterError) -> dict[str, Any]:
    return {
        "message": exc.message,
        "type": exc.code,
        "code": exc.code,
        "details": exc.details or {},
    }


def stream_error_event(exc: AsterError) -> dict[str, str]:
    return {"data": json.dumps({"error": _stream_error_payload(exc)})}


def _stream_error_event(exc: AsterError) -> dict[str, str]:
    return stream_error_event(exc)


def _stream_failed_error() -> AsterError:
    return AsterError(
        code="stream_failed",
        message="Streaming response failed",
        status_code=500,
    )


def responses_output_text(output: list[dict[str, Any]]) -> str:
    fragments: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            fragments.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                fragments.append(part["text"])
    return "".join(fragments)


def responses_message_item(item_id: str, text: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def responses_reasoning_item(item_id: str, text: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "content": [{"type": "reasoning_text", "text": text}],
    }


def responses_function_call_item(
    item_id: str,
    *,
    call_id: str,
    name: str,
    arguments: str,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def responses_error_payload(exc: AsterError) -> dict[str, Any]:
    return {"code": exc.code, "message": exc.message, "type": exc.code}


@dataclass(slots=True)
class ResponsesItemRef:
    item_id: str
    output_index: int


@dataclass(frozen=True, slots=True)
class StreamedToolCall:
    index: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    arguments_text: str


class ToolCallDeltaAccumulator:
    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}
        self._emitted: set[int] = set()

    def add(self, delta: ToolCallDelta) -> StreamedToolCall | None:
        state = self._calls.setdefault(delta.index, {"arguments": ""})
        if delta.call_id:
            state["call_id"] = delta.call_id
        if delta.name:
            state["name"] = delta.name
        if delta.arguments_delta:
            state["arguments"] = f"{state.get('arguments', '')}{delta.arguments_delta}"
        if delta.index in self._emitted or not delta.finished:
            return None

        name = state.get("name")
        arguments_text = str(state.get("arguments") or "")
        if not isinstance(name, str) or not name:
            return None
        try:
            arguments = json.loads(arguments_text) if arguments_text else {}
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None

        self._emitted.add(delta.index)
        call_id = state.get("call_id")
        return StreamedToolCall(
            index=delta.index,
            call_id=call_id if isinstance(call_id, str) and call_id else f"call_{delta.index}",
            name=name,
            arguments=arguments,
            arguments_text=arguments_text,
        )


class ResponsesSseLifecycle:
    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.response_metadata = dict(response_metadata or {})
        self.created_at = int(time())
        self.sequence_number = 0
        self.next_output_index = 0

    def created_event(self) -> dict[str, str]:
        return self._event(
            "response.created",
            {"response": self.response_object(status="in_progress", output=[], usage=None)},
        )

    def in_progress_event(self) -> dict[str, str]:
        return self._event(
            "response.in_progress",
            {"response": self.response_object(status="in_progress", output=[], usage=None)},
        )

    def start_text_item(self, *, item_id: str | None = None) -> tuple[ResponsesItemRef, list[dict[str, str]]]:
        item_ref = self._new_item_ref(item_id or f"msg_{self.response_id}")
        item = {
            "id": item_ref.item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        events = [
            self.output_item_added_event(output_index=item_ref.output_index, item=item),
            self.content_part_added_event(
                item_id=item_ref.item_id,
                output_index=item_ref.output_index,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": []},
            ),
        ]
        return item_ref, events

    def text_delta_event(self, item_ref: ResponsesItemRef, delta: str) -> dict[str, str]:
        return self._event(
            "response.output_text.delta",
            {
                "item_id": item_ref.item_id,
                "output_index": item_ref.output_index,
                "content_index": 0,
                "delta": delta,
                "logprobs": [],
            },
        )

    def finish_text_item(
        self,
        item_ref: ResponsesItemRef,
        text: str,
        *,
        status: str = "completed",
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        item = responses_message_item(item_ref.item_id, text, status=status)
        part = item["content"][0]
        events = [
            self._event(
                "response.output_text.done",
                {
                    "item_id": item_ref.item_id,
                    "output_index": item_ref.output_index,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                },
            ),
            self.content_part_done_event(
                item_id=item_ref.item_id,
                output_index=item_ref.output_index,
                content_index=0,
                part=part,
            ),
            self.output_item_done_event(output_index=item_ref.output_index, item=item),
        ]
        return item, events

    def start_reasoning_item(self, *, item_id: str | None = None) -> tuple[ResponsesItemRef, list[dict[str, str]]]:
        item_ref = self._new_item_ref(item_id or f"rs_{self.response_id}")
        item = {
            "id": item_ref.item_id,
            "type": "reasoning",
            "status": "in_progress",
            "content": [],
        }
        events = [
            self.output_item_added_event(output_index=item_ref.output_index, item=item),
            self.content_part_added_event(
                item_id=item_ref.item_id,
                output_index=item_ref.output_index,
                content_index=0,
                part={"type": "reasoning_text", "text": ""},
            ),
        ]
        return item_ref, events

    def reasoning_delta_event(self, item_ref: ResponsesItemRef, delta: str) -> dict[str, str]:
        return self._event(
            "response.reasoning_text.delta",
            {
                "item_id": item_ref.item_id,
                "output_index": item_ref.output_index,
                "content_index": 0,
                "delta": delta,
            },
        )

    def finish_reasoning_item(
        self,
        item_ref: ResponsesItemRef,
        text: str,
        *,
        status: str = "completed",
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        item = responses_reasoning_item(item_ref.item_id, text, status=status)
        part = item["content"][0]
        events = [
            self._event(
                "response.reasoning_text.done",
                {
                    "item_id": item_ref.item_id,
                    "output_index": item_ref.output_index,
                    "content_index": 0,
                    "text": text,
                },
            ),
            self.content_part_done_event(
                item_id=item_ref.item_id,
                output_index=item_ref.output_index,
                content_index=0,
                part=part,
            ),
            self.output_item_done_event(output_index=item_ref.output_index, item=item),
        ]
        return item, events

    def function_call_events(
        self,
        *,
        call_id: str,
        name: str,
        arguments: str,
        item_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        item_ref, events = self.start_function_call_item(
            call_id=call_id,
            name=name,
            item_id=item_id,
        )
        if arguments:
            events.append(self.function_call_arguments_delta_event(item_ref, arguments))
        item, finish_events = self.finish_function_call_item(
            item_ref,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )
        events.extend(finish_events)
        return item, events

    def start_function_call_item(
        self,
        *,
        call_id: str,
        name: str,
        item_id: str | None = None,
    ) -> tuple[ResponsesItemRef, list[dict[str, str]]]:
        item_ref = self._new_item_ref(item_id or f"fc_{call_id}")
        item = responses_function_call_item(
            item_ref.item_id,
            call_id=call_id,
            name=name,
            arguments="",
            status="in_progress",
        )
        return item_ref, [self.output_item_added_event(output_index=item_ref.output_index, item=item)]

    def function_call_arguments_delta_event(self, item_ref: ResponsesItemRef, delta: str) -> dict[str, str]:
        return self._event(
            "response.function_call_arguments.delta",
            {
                "item_id": item_ref.item_id,
                "output_index": item_ref.output_index,
                "delta": delta,
            },
        )

    def finish_function_call_item(
        self,
        item_ref: ResponsesItemRef,
        *,
        call_id: str,
        name: str,
        arguments: str,
        status: str = "completed",
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        item = responses_function_call_item(
            item_ref.item_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
            status=status,
        )
        return item, [self.output_item_done_event(output_index=item_ref.output_index, item=item)]

    def output_item_added_event(self, *, output_index: int, item: dict[str, Any]) -> dict[str, str]:
        return self._event(
            "response.output_item.added",
            {
                "output_index": output_index,
                "item": item,
            },
        )

    def output_item_done_event(self, *, output_index: int, item: dict[str, Any]) -> dict[str, str]:
        return self._event(
            "response.output_item.done",
            {
                "output_index": output_index,
                "item": item,
            },
        )

    def content_part_added_event(
        self,
        *,
        item_id: str,
        output_index: int,
        content_index: int,
        part: dict[str, Any],
    ) -> dict[str, str]:
        return self._event(
            "response.content_part.added",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": part,
            },
        )

    def content_part_done_event(
        self,
        *,
        item_id: str,
        output_index: int,
        content_index: int,
        part: dict[str, Any],
    ) -> dict[str, str]:
        return self._event(
            "response.content_part.done",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": part,
            },
        )

    def completed_event(
        self,
        *,
        output: list[dict[str, Any]],
        usage: dict[str, Any] | None,
        finish_reason: str | None = None,
    ) -> dict[str, str]:
        status_fields = responses_status_fields_from_finish_reason(finish_reason)
        return self._event(
            "response.completed",
            {
                "response": self.response_object(
                    status=str(status_fields["status"]),
                    output=output,
                    usage=usage,
                    incomplete_details=status_fields.get("incomplete_details"),
                )
            },
        )

    def failed_event(
        self,
        *,
        output: list[dict[str, Any]],
        error: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        return self._event(
            "response.failed",
            {
                "response": self.response_object(
                    status="failed",
                    output=output,
                    usage=usage,
                    error=error,
                )
            },
        )

    def response_object(
        self,
        *,
        status: str,
        output: list[dict[str, Any]],
        usage: dict[str, Any] | None,
        error: dict[str, Any] | None = None,
        incomplete_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "model": self.model,
            **self.response_metadata,
            "status": status,
            "output": output,
            "output_text": responses_output_text(output),
        }
        if usage is not None:
            response["usage"] = usage
        if error is not None:
            response["error"] = error
        if incomplete_details is not None:
            response["incomplete_details"] = incomplete_details
        return response

    def _new_item_ref(self, item_id: str) -> ResponsesItemRef:
        item_ref = ResponsesItemRef(item_id=item_id, output_index=self.next_output_index)
        self.next_output_index += 1
        return item_ref

    def _event(self, event_type: str, payload: dict[str, Any]) -> dict[str, str]:
        self.sequence_number += 1
        payload = {"type": event_type, "response_id": self.response_id, "sequence_number": self.sequence_number, **payload}
        return {"event": event_type, "data": json.dumps(payload)}


async def iter_chat_sse_events(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    include_debug_summary: bool = False,
    include_usage: bool = False,
    on_error: Callable[[AsterError], None] | None = None,
) -> AsyncIterator[dict[str, str]]:
    created = int(time())
    sent_done = False
    cancelled = False
    reasoning_parser = AutoReasoningParser()
    try:
        yield chat_sse_event(
            response_id=response_id,
            model=model,
            created=created,
            delta={"role": "assistant"},
        )
        async for chunk in chunks:
            if chunk.finished:
                usage = _usage_from_stats(chunk.stats)
                yield chat_sse_event(
                    response_id=response_id,
                    model=model,
                    created=created,
                    delta={},
                    finish_reason=_finish_reason_from_stats(chunk.stats),
                )
                if include_usage and usage is not None:
                    yield chat_usage_sse_event(
                        response_id=response_id,
                        model=model,
                        created=created,
                        usage=usage,
                    )
                if include_debug_summary and chunk.stats is not None:
                    summary_payload = {
                        "object": "aster.stream.summary",
                        "model": model,
                        "aster": chunk.stats,
                    }
                    yield {"data": json.dumps(summary_payload)}
                yield {"data": "[DONE]"}
                sent_done = True
                break
            parsed = reasoning_parser.parse_delta(chunk.token)
            if parsed.reasoning_delta:
                yield chat_sse_event(
                    response_id=response_id,
                    model=model,
                    created=created,
                    delta={"reasoning_content": parsed.reasoning_delta},
                )
            if not parsed.content_delta:
                continue
            yield chat_sse_event(
                response_id=response_id,
                model=model,
                created=created,
                delta={"content": parsed.content_delta},
            )
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "chat_sse_stream_aster_error",
            extra={"response_id": response_id, "code": exc.code},
        )
        if on_error is not None:
            on_error(exc)
        yield _stream_error_event(exc)
    except Exception:
        _logger.exception("chat_sse_stream_failed", extra={"response_id": response_id})
        exc = _stream_failed_error()
        if on_error is not None:
            on_error(exc)
        yield _stream_error_event(exc)
    finally:
        if not sent_done and not cancelled:
            yield {"data": "[DONE]"}


async def iter_chat_tool_sse_events(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    include_usage: bool = False,
    emit_role: bool = True,
    emit_done: bool = True,
    created: int | None = None,
    on_delta: Callable[[ParsedGenerationDelta], None] | None = None,
    on_tool_call: Callable[[StreamedToolCall], None] | None = None,
    on_finish: Callable[[dict[str, object] | None, bool], None] | None = None,
    on_error: Callable[[AsterError], None] | None = None,
    tool_request: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, str]]:
    created_at = created if created is not None else int(time())
    sent_done = False
    cancelled = False
    saw_tool_call = False
    completed_tool_call = False
    tool_accumulator = ToolCallDeltaAccumulator()
    reasoning_parser = AutoReasoningParser()
    tool_parser = AutoToolParser()
    tool_parser.configure_request(tool_request)

    try:
        if emit_role:
            yield chat_sse_event(
                response_id=response_id,
                model=model,
                created=created_at,
                delta={"role": "assistant"},
            )
        async for chunk in chunks:
            if chunk.finished:
                flush_delta = tool_parser.flush_delta()
                if flush_delta.content_delta or flush_delta.reasoning_delta or flush_delta.tool_call_deltas:
                    if on_delta is not None:
                        on_delta(flush_delta)
                    if _collect_streamed_tool_calls(flush_delta, tool_accumulator, on_tool_call):
                        completed_tool_call = True
                async for event in _chat_tool_events_from_delta(
                    flush_delta,
                    response_id=response_id,
                    model=model,
                    created=created_at,
                ):
                    yield event
                if flush_delta.tool_call_deltas:
                    saw_tool_call = True
                usage = _usage_from_stats(chunk.stats)
                if on_finish is not None:
                    on_finish(chunk.stats, saw_tool_call)
                yield chat_sse_event(
                    response_id=response_id,
                    model=model,
                    created=created_at,
                    delta={},
                    finish_reason=_chat_tool_finish_reason(
                        chunk.stats,
                        saw_tool_call=saw_tool_call,
                        completed_tool_call=completed_tool_call,
                    ),
                )
                if include_usage and usage is not None:
                    yield chat_usage_sse_event(
                        response_id=response_id,
                        model=model,
                        created=created_at,
                        usage=usage,
                    )
                if emit_done:
                    yield {"data": "[DONE]"}
                sent_done = True
                break

            parsed = reasoning_parser.parse_delta(chunk.token)
            if parsed.reasoning_delta:
                if on_delta is not None:
                    on_delta(ParsedGenerationDelta(reasoning_delta=parsed.reasoning_delta, raw_delta=chunk.token))
                yield chat_sse_event(
                    response_id=response_id,
                    model=model,
                    created=created_at,
                    delta={"reasoning_content": parsed.reasoning_delta},
                )
            if not parsed.content_delta:
                continue
            tool_delta = tool_parser.parse_delta(parsed.content_delta)
            if tool_delta.tool_call_deltas:
                saw_tool_call = True
            if tool_delta.content_delta or tool_delta.reasoning_delta or tool_delta.tool_call_deltas:
                if on_delta is not None:
                    on_delta(tool_delta)
                if _collect_streamed_tool_calls(tool_delta, tool_accumulator, on_tool_call):
                    completed_tool_call = True
            async for event in _chat_tool_events_from_delta(
                tool_delta,
                response_id=response_id,
                model=model,
                created=created_at,
            ):
                yield event
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "chat_tool_sse_stream_aster_error",
            extra={"response_id": response_id, "code": exc.code},
        )
        if on_error is not None:
            on_error(exc)
        yield stream_error_event(exc)
    except Exception:
        _logger.exception("chat_tool_sse_stream_failed", extra={"response_id": response_id})
        exc = _stream_failed_error()
        if on_error is not None:
            on_error(exc)
        yield stream_error_event(exc)
    finally:
        if emit_done and not sent_done and not cancelled:
            yield {"data": "[DONE]"}


def _collect_streamed_tool_calls(
    delta: ParsedGenerationDelta,
    accumulator: ToolCallDeltaAccumulator,
    on_tool_call: Callable[[StreamedToolCall], None] | None,
) -> bool:
    completed = False
    for tool_delta in delta.tool_call_deltas:
        streamed_call = accumulator.add(tool_delta)
        if streamed_call is not None:
            completed = True
            if on_tool_call is not None:
                on_tool_call(streamed_call)
    return completed


def _chat_tool_finish_reason(
    stats: dict[str, object] | None,
    *,
    saw_tool_call: bool,
    completed_tool_call: bool,
) -> str:
    finish_reason = _finish_reason_from_stats(stats)
    if not saw_tool_call:
        return finish_reason
    return "tool_calls" if completed_tool_call else finish_reason


async def _chat_tool_events_from_delta(
    delta: ParsedGenerationDelta,
    *,
    response_id: str,
    model: str,
    created: int,
) -> AsyncIterator[dict[str, str]]:
    if delta.content_delta:
        yield chat_sse_event(
            response_id=response_id,
            model=model,
            created=created,
            delta={"content": delta.content_delta},
        )
    for tool_call in delta.tool_call_deltas:
        yield chat_sse_event(
            response_id=response_id,
            model=model,
            created=created,
            delta={"tool_calls": [_chat_tool_delta(tool_call)]},
        )


async def iter_chat_structured_sse_events(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    feature_plan: FeaturePlan,
    include_usage: bool = False,
) -> AsyncIterator[dict[str, str]]:
    created = int(time())
    sent_done = False
    cancelled = False
    reasoning_parser = AutoReasoningParser()
    fence_stripper = StreamingJsonFenceStripper()
    text_fragments: list[str] = []
    try:
        yield chat_sse_event(
            response_id=response_id,
            model=model,
            created=created,
            delta={"role": "assistant"},
        )
        async for chunk in chunks:
            if chunk.finished:
                flush = fence_stripper.finalize()
                if flush:
                    text_fragments.append(flush)
                    yield chat_sse_event(
                        response_id=response_id,
                        model=model,
                        created=created,
                        delta={"content": flush},
                    )
                try:
                    validate_structured_output_text("".join(text_fragments), feature_plan, allow_repair=False)
                except AsterError as exc:
                    yield stream_error_event(exc)
                else:
                    usage = _usage_from_stats(chunk.stats)
                    yield chat_sse_event(
                        response_id=response_id,
                        model=model,
                        created=created,
                        delta={},
                        finish_reason=_finish_reason_from_stats(chunk.stats),
                    )
                    if include_usage and usage is not None:
                        yield chat_usage_sse_event(
                            response_id=response_id,
                            model=model,
                            created=created,
                            usage=usage,
                        )
                yield {"data": "[DONE]"}
                sent_done = True
                break

            parsed = reasoning_parser.parse_delta(chunk.token)
            if parsed.reasoning_delta:
                yield chat_sse_event(
                    response_id=response_id,
                    model=model,
                    created=created,
                    delta={"reasoning_content": parsed.reasoning_delta},
                )
            if not parsed.content_delta:
                continue
            content = fence_stripper.feed(parsed.content_delta)
            if not content:
                continue
            text_fragments.append(content)
            yield chat_sse_event(
                response_id=response_id,
                model=model,
                created=created,
                delta={"content": content},
            )
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "chat_structured_sse_stream_aster_error",
            extra={"response_id": response_id, "code": exc.code},
        )
        yield stream_error_event(exc)
    except Exception:
        _logger.exception("chat_structured_sse_stream_failed", extra={"response_id": response_id})
        yield stream_error_event(_stream_failed_error())
    finally:
        if not sent_done and not cancelled:
            yield {"data": "[DONE]"}


async def iter_responses_sse_events(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    response_metadata: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, str]]:
    lifecycle = ResponsesSseLifecycle(
        response_id=response_id,
        model=model,
        response_metadata=response_metadata,
    )
    text_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    output: list[dict[str, Any]] = []
    text_ref: ResponsesItemRef | None = None
    reasoning_ref: ResponsesItemRef | None = None
    reasoning_parser = AutoReasoningParser()
    sent_terminal = False
    cancelled = False
    finalized = False

    async def finish_stream(stats: dict[str, object] | None) -> AsyncIterator[dict[str, str]]:
        nonlocal finalized, text_ref, reasoning_ref
        if finalized:
            return
        usage = responses_usage_from_stats(stats)
        if reasoning_ref is not None:
            reasoning_item, events = lifecycle.finish_reasoning_item(reasoning_ref, "".join(reasoning_fragments))
            reasoning_ref = None
            output.append(reasoning_item)
            for event in events:
                yield event
        if text_ref is None:
            text_ref, events = lifecycle.start_text_item()
            for event in events:
                yield event
        text_item, events = lifecycle.finish_text_item(text_ref, "".join(text_fragments))
        output.append(text_item)
        for event in events:
            yield event
        finalized = True
        yield lifecycle.completed_event(
            output=output,
            usage=usage,
            finish_reason=_optional_finish_reason_from_stats(stats),
        )

    async def finish_failed(
        exc: AsterError,
        usage: dict[str, Any] | None,
    ) -> AsyncIterator[dict[str, str]]:
        nonlocal finalized, text_ref, reasoning_ref
        if finalized:
            return
        if reasoning_ref is not None:
            reasoning_item, events = lifecycle.finish_reasoning_item(
                reasoning_ref,
                "".join(reasoning_fragments),
            )
            reasoning_ref = None
            output.append(reasoning_item)
            for event in events:
                yield event
        if text_ref is not None:
            text_item, events = lifecycle.finish_text_item(
                text_ref,
                "".join(text_fragments),
                status="incomplete",
            )
            text_ref = None
            output.append(text_item)
            for event in events:
                yield event
        finalized = True
        yield lifecycle.failed_event(
            output=output,
            error=responses_error_payload(exc),
            usage=usage,
        )

    try:
        yield lifecycle.created_event()
        yield lifecycle.in_progress_event()
        async for chunk in chunks:
            if chunk.finished:
                async for event in finish_stream(chunk.stats):
                    yield event
                sent_terminal = True
                break

            delta = reasoning_parser.parse_delta(chunk.token)
            if delta.reasoning_delta:
                if reasoning_ref is None:
                    reasoning_ref, events = lifecycle.start_reasoning_item()
                    for event in events:
                        yield event
                reasoning_fragments.append(delta.reasoning_delta)
                yield lifecycle.reasoning_delta_event(reasoning_ref, delta.reasoning_delta)
            if not delta.content_delta:
                continue
            if text_ref is None:
                text_ref, events = lifecycle.start_text_item()
                for event in events:
                    yield event
            text_fragments.append(delta.content_delta)
            yield lifecycle.text_delta_event(text_ref, delta.content_delta)
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "responses_sse_stream_aster_error",
            extra={"response_id": response_id, "code": exc.code},
        )
        async for event in finish_failed(exc, None):
            yield event
        sent_terminal = True
    except Exception:
        _logger.exception("responses_sse_stream_failed", extra={"response_id": response_id})
        exc = AsterError(
            code="stream_failed",
            message="Streaming response failed",
            status_code=500,
        )
        async for event in finish_failed(exc, None):
            yield event
        sent_terminal = True
    finally:
        if not sent_terminal and not cancelled:
            async for event in finish_stream(None):
                yield event


async def iter_responses_structured_sse_events(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    feature_plan: FeaturePlan,
    response_metadata: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, str]]:
    lifecycle = ResponsesSseLifecycle(
        response_id=response_id,
        model=model,
        response_metadata=response_metadata,
    )
    text_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    output: list[dict[str, Any]] = []
    text_ref: ResponsesItemRef | None = None
    reasoning_ref: ResponsesItemRef | None = None
    reasoning_parser = AutoReasoningParser()
    fence_stripper = StreamingJsonFenceStripper()
    sent_terminal = False
    cancelled = False
    finalized = False

    async def finish_reasoning() -> AsyncIterator[dict[str, str]]:
        nonlocal reasoning_ref
        if reasoning_ref is None:
            return
        reasoning_item, events = lifecycle.finish_reasoning_item(reasoning_ref, "".join(reasoning_fragments))
        reasoning_ref = None
        output.append(reasoning_item)
        for event in events:
            yield event

    async def finish_success(stats: dict[str, object] | None) -> AsyncIterator[dict[str, str]]:
        nonlocal finalized, text_ref
        if finalized:
            return
        usage = responses_usage_from_stats(stats)
        async for event in finish_reasoning():
            yield event
        if text_ref is None:
            text_ref, events = lifecycle.start_text_item()
            for event in events:
                yield event
        text_item, events = lifecycle.finish_text_item(text_ref, "".join(text_fragments))
        output.append(text_item)
        for event in events:
            yield event
        finalized = True
        yield lifecycle.completed_event(
            output=output,
            usage=usage,
            finish_reason=_optional_finish_reason_from_stats(stats),
        )

    async def finish_failed(exc: AsterError, usage: dict[str, Any] | None) -> AsyncIterator[dict[str, str]]:
        nonlocal finalized, text_ref
        if finalized:
            return
        async for event in finish_reasoning():
            yield event
        if text_ref is not None:
            text_item, events = lifecycle.finish_text_item(text_ref, "".join(text_fragments), status="incomplete")
            text_ref = None
            output.append(text_item)
            for event in events:
                yield event
        finalized = True
        yield lifecycle.failed_event(
            output=output,
            error=responses_error_payload(exc),
            usage=usage,
        )

    try:
        yield lifecycle.created_event()
        yield lifecycle.in_progress_event()
        async for chunk in chunks:
            if chunk.finished:
                flush = fence_stripper.finalize()
                if flush:
                    if text_ref is None:
                        text_ref, events = lifecycle.start_text_item()
                        for event in events:
                            yield event
                    text_fragments.append(flush)
                    yield lifecycle.text_delta_event(text_ref, flush)
                try:
                    validate_structured_output_text("".join(text_fragments), feature_plan, allow_repair=False)
                except AsterError as exc:
                    async for event in finish_failed(exc, responses_usage_from_stats(chunk.stats)):
                        yield event
                else:
                    async for event in finish_success(chunk.stats):
                        yield event
                sent_terminal = True
                break

            delta = reasoning_parser.parse_delta(chunk.token)
            if delta.reasoning_delta:
                if reasoning_ref is None:
                    reasoning_ref, events = lifecycle.start_reasoning_item()
                    for event in events:
                        yield event
                reasoning_fragments.append(delta.reasoning_delta)
                yield lifecycle.reasoning_delta_event(reasoning_ref, delta.reasoning_delta)
            if not delta.content_delta:
                continue
            content = fence_stripper.feed(delta.content_delta)
            if not content:
                continue
            if text_ref is None:
                text_ref, events = lifecycle.start_text_item()
                for event in events:
                    yield event
            text_fragments.append(content)
            yield lifecycle.text_delta_event(text_ref, content)
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "responses_structured_sse_stream_aster_error",
            extra={"response_id": response_id, "code": exc.code},
        )
        async for event in finish_failed(exc, None):
            yield event
        sent_terminal = True
    except Exception:
        _logger.exception("responses_structured_sse_stream_failed", extra={"response_id": response_id})
        exc = AsterError(
            code="stream_failed",
            message="Streaming response failed",
            status_code=500,
        )
        async for event in finish_failed(exc, None):
            yield event
        sent_terminal = True
    finally:
        if not sent_terminal and not cancelled:
            try:
                validate_structured_output_text("".join(text_fragments), feature_plan, allow_repair=False)
            except AsterError as exc:
                async for event in finish_failed(exc, None):
                    yield event
            else:
                async for event in finish_success(None):
                    yield event


async def iter_responses_tool_sse_events(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    tool_request: dict[str, Any] | None = None,
    response_metadata: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, str]]:
    lifecycle = ResponsesSseLifecycle(
        response_id=response_id,
        model=model,
        response_metadata=response_metadata,
    )
    text_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    output: list[dict[str, Any]] = []
    text_ref: ResponsesItemRef | None = None
    reasoning_ref: ResponsesItemRef | None = None
    reasoning_parser = AutoReasoningParser()
    tool_parser = AutoToolParser()
    tool_parser.configure_request(tool_request)
    tool_call_refs: dict[int, ResponsesItemRef] = {}
    tool_call_ids: dict[int, str] = {}
    tool_call_names: dict[int, str] = {}
    tool_call_arguments: dict[int, list[str]] = {}
    sent_terminal = False
    cancelled = False
    finalized = False

    async def finish_stream(stats: dict[str, object] | None) -> AsyncIterator[dict[str, str]]:
        nonlocal finalized, reasoning_ref, text_ref
        if finalized:
            return
        usage = responses_usage_from_stats(stats)
        flush_delta = tool_parser.flush_delta()
        async for event in emit_tool_delta(flush_delta):
            yield event
        item_status = (
            "incomplete" if _optional_finish_reason_from_stats(stats) == "length" else "completed"
        )
        for index in list(tool_call_refs):
            item, events = lifecycle.finish_function_call_item(
                tool_call_refs.pop(index),
                call_id=tool_call_ids.pop(index),
                name=tool_call_names.pop(index),
                arguments="".join(tool_call_arguments.pop(index, [])),
                status=item_status,
            )
            output.append(item)
            for event in events:
                yield event
        if reasoning_ref is not None:
            reasoning_item, events = lifecycle.finish_reasoning_item(reasoning_ref, "".join(reasoning_fragments))
            reasoning_ref = None
            output.append(reasoning_item)
            for event in events:
                yield event
        if text_ref is not None:
            text_item, events = lifecycle.finish_text_item(text_ref, "".join(text_fragments))
            output.append(text_item)
            for event in events:
                yield event
        finalized = True
        yield lifecycle.completed_event(
            output=output,
            usage=usage,
            finish_reason=_optional_finish_reason_from_stats(stats),
        )

    async def finish_failed(
        exc: AsterError,
        usage: dict[str, Any] | None,
    ) -> AsyncIterator[dict[str, str]]:
        nonlocal finalized, reasoning_ref, text_ref
        if finalized:
            return
        flush_delta = tool_parser.flush_delta()
        async for event in emit_tool_delta(flush_delta):
            yield event
        for index in list(tool_call_refs):
            item, events = lifecycle.finish_function_call_item(
                tool_call_refs.pop(index),
                call_id=tool_call_ids.pop(index),
                name=tool_call_names.pop(index),
                arguments="".join(tool_call_arguments.pop(index, [])),
                status="incomplete",
            )
            output.append(item)
            for event in events:
                yield event
        if reasoning_ref is not None:
            reasoning_item, events = lifecycle.finish_reasoning_item(
                reasoning_ref,
                "".join(reasoning_fragments),
                status="incomplete",
            )
            reasoning_ref = None
            output.append(reasoning_item)
            for event in events:
                yield event
        if text_ref is not None:
            text_item, events = lifecycle.finish_text_item(
                text_ref,
                "".join(text_fragments),
                status="incomplete",
            )
            text_ref = None
            output.append(text_item)
            for event in events:
                yield event
        finalized = True
        yield lifecycle.failed_event(
            output=output,
            error=responses_error_payload(exc),
            usage=usage,
        )

    async def emit_tool_delta(delta: ParsedGenerationDelta) -> AsyncIterator[dict[str, str]]:
        nonlocal text_ref
        if delta.content_delta:
            if text_ref is None:
                text_ref, events = lifecycle.start_text_item()
                for event in events:
                    yield event
            text_fragments.append(delta.content_delta)
            yield lifecycle.text_delta_event(text_ref, delta.content_delta)
        for tool_call in delta.tool_call_deltas:
            index = tool_call.index
            if index not in tool_call_refs:
                call_id = _tool_call_id(tool_call)
                name = _tool_call_name(tool_call)
                tool_call_ids[index] = call_id
                tool_call_names[index] = name
                tool_call_arguments[index] = []
                tool_call_refs[index], events = lifecycle.start_function_call_item(
                    call_id=call_id,
                    name=name,
                )
                for event in events:
                    yield event
            if tool_call.name is not None:
                tool_call_names[index] = tool_call.name
            if tool_call.call_id is not None:
                tool_call_ids[index] = tool_call.call_id
            if tool_call.arguments_delta:
                tool_call_arguments[index].append(tool_call.arguments_delta)
                yield lifecycle.function_call_arguments_delta_event(tool_call_refs[index], tool_call.arguments_delta)
            if tool_call.finished:
                item, events = lifecycle.finish_function_call_item(
                    tool_call_refs.pop(index),
                    call_id=tool_call_ids.pop(index),
                    name=tool_call_names.pop(index),
                    arguments="".join(tool_call_arguments.pop(index, [])),
                )
                output.append(item)
                for event in events:
                    yield event

    try:
        yield lifecycle.created_event()
        yield lifecycle.in_progress_event()
        async for chunk in chunks:
            if chunk.finished:
                async for event in finish_stream(chunk.stats):
                    yield event
                sent_terminal = True
                break

            parsed = reasoning_parser.parse_delta(chunk.token)
            if parsed.reasoning_delta:
                if reasoning_ref is None:
                    reasoning_ref, events = lifecycle.start_reasoning_item()
                    for event in events:
                        yield event
                reasoning_fragments.append(parsed.reasoning_delta)
                yield lifecycle.reasoning_delta_event(reasoning_ref, parsed.reasoning_delta)
            if not parsed.content_delta:
                continue
            tool_delta = tool_parser.parse_delta(parsed.content_delta)
            async for event in emit_tool_delta(tool_delta):
                yield event
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "responses_tool_sse_stream_aster_error",
            extra={"response_id": response_id, "code": exc.code},
        )
        async for event in finish_failed(exc, None):
            yield event
        sent_terminal = True
    except Exception:
        _logger.exception("responses_tool_sse_stream_failed", extra={"response_id": response_id})
        exc = AsterError(
            code="stream_failed",
            message="Streaming response failed",
            status_code=500,
        )
        async for event in finish_failed(exc, None):
            yield event
        sent_terminal = True
    finally:
        if not sent_terminal and not cancelled:
            async for event in finish_stream(None):
                yield event


async def to_chat_sse(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    include_debug_summary: bool = False,
    include_usage: bool = False,
    headers: dict[str, str] | None = None,
    raw_request: Any | None = None,
    on_error: Callable[[AsterError], None] | None = None,
) -> EventSourceResponse:
    events = iter_chat_sse_events(
        chunks,
        model,
        response_id=response_id,
        include_debug_summary=include_debug_summary,
        include_usage=include_usage,
        on_error=on_error,
    )
    if raw_request is not None:
        events = stream_with_disconnect(
            events,
            raw_request,
            poll_interval_seconds=SSE_DISCONNECT_POLL_SECONDS,
            heartbeat_interval_seconds=SSE_HEARTBEAT_SECONDS,
            heartbeat_factory=lambda: {"comment": "heartbeat"},
            on_error=on_error,
        )
    return EventSourceResponse(
        events,
        ping=SSE_HEARTBEAT_SECONDS,
        send_timeout=SSE_SEND_TIMEOUT_SECONDS,
        headers=headers,
    )


async def iter_completion_sse_events(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    include_debug_summary: bool = False,
    on_error: Callable[[AsterError], None] | None = None,
) -> AsyncIterator[dict[str, str]]:
    created = int(time())
    sent_done = False
    cancelled = False
    try:
        async for chunk in chunks:
            if chunk.finished:
                yield {
                    "data": json.dumps(
                        _completion_chunk(
                            response_id=response_id,
                            model=model,
                            created=created,
                            text="",
                            finish_reason=_finish_reason_from_stats(chunk.stats),
                            usage=_usage_from_stats(chunk.stats),
                        )
                    )
                }
                if include_debug_summary and chunk.stats is not None:
                    summary_payload = {
                        "object": "aster.stream.summary",
                        "model": model,
                        "aster": chunk.stats,
                    }
                    yield {"data": json.dumps(summary_payload)}
                yield {"data": "[DONE]"}
                sent_done = True
                break
            yield {
                "data": json.dumps(
                    _completion_chunk(
                        response_id=response_id,
                        model=model,
                        created=created,
                        text=chunk.token,
                    )
                )
            }
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "completion_sse_stream_aster_error",
            extra={"response_id": response_id, "code": exc.code},
        )
        if on_error is not None:
            on_error(exc)
        yield _stream_error_event(exc)
    except Exception:
        _logger.exception("completion_sse_stream_failed", extra={"response_id": response_id})
        exc = _stream_failed_error()
        if on_error is not None:
            on_error(exc)
        yield _stream_error_event(exc)
    finally:
        if not sent_done and not cancelled:
            yield {"data": "[DONE]"}


async def to_completion_sse(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    response_id: str,
    include_debug_summary: bool = False,
    headers: dict[str, str] | None = None,
    raw_request: Any | None = None,
    on_error: Callable[[AsterError], None] | None = None,
) -> EventSourceResponse:
    events = iter_completion_sse_events(
        chunks,
        model,
        response_id=response_id,
        include_debug_summary=include_debug_summary,
        on_error=on_error,
    )
    if raw_request is not None:
        events = stream_with_disconnect(
            events,
            raw_request,
            poll_interval_seconds=SSE_DISCONNECT_POLL_SECONDS,
            heartbeat_interval_seconds=SSE_HEARTBEAT_SECONDS,
            heartbeat_factory=lambda: {"comment": "heartbeat"},
            on_error=on_error,
        )
    return EventSourceResponse(
        events,
        ping=SSE_HEARTBEAT_SECONDS,
        send_timeout=SSE_SEND_TIMEOUT_SECONDS,
        headers=headers,
    )


async def to_sse(
    chunks: AsyncIterator[DecodeChunk],
    model: str,
    *,
    include_debug_summary: bool = False,
) -> EventSourceResponse:
    return await to_chat_sse(
        chunks,
        model,
        response_id=f"chatcmpl-{int(time())}",
        include_debug_summary=include_debug_summary,
    )
