from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from time import perf_counter, time
from typing import Any

from sse_starlette.sse import EventSourceResponse

from aster.api.disconnect import stream_with_disconnect
from aster.api.feature_emulation import (
    DecodedLocalOutput,
    ToolCallResult,
    ToolChoice,
    tool_request_from_plan,
    validate_tool_call_arguments,
)
from aster.api.provider_gateway import (
    LocalProviderRequest,
    encode_provider_decoded_response,
    responses_replay_output_messages,
)
from aster.api.streaming import (
    SSE_DISCONNECT_POLL_SECONDS,
    SSE_HEARTBEAT_SECONDS,
    SSE_SEND_TIMEOUT_SECONDS,
    ResponsesItemRef,
    ResponsesSseLifecycle,
    StreamedToolCall,
    ToolCallDeltaAccumulator,
    chat_sse_event,
    chat_usage_sse_event,
    iter_chat_tool_sse_events,
    responses_error_payload,
    responses_finish_reason_from_stats,
    responses_usage_from_stats,
    responses_usage_payload,
    stream_error_event,
)
from aster.core.errors import AsterError
from aster.inference.engine import InferenceResponse
from aster.inference.parser_pipeline import ParsedGenerationDelta
from aster.inference.reasoning_parsers import AutoReasoningParser
from aster.inference.tool_parsers import AutoToolParser
from aster.runtime.tools import ToolExecutionContext
from aster.telemetry.logging import get_logger

MAX_TOOL_ROUNDS = 4
MAX_PARALLEL_TOOL_CALLS = 5
_logger = get_logger(__name__)


@dataclass(slots=True)
class ExecutedToolResult:
    tool_call: ToolCallResult
    result: Any
    is_error: bool = False


@dataclass(slots=True)
class InteractionRound:
    result: InferenceResponse
    decoded: DecodedLocalOutput
    executed_tools: list[ExecutedToolResult] = field(default_factory=list)


@dataclass(slots=True)
class InteractionTrace:
    rounds: list[InteractionRound]

    @property
    def final_round(self) -> InteractionRound:
        return self.rounds[-1]

    @property
    def final_result(self) -> InferenceResponse:
        return self.final_round.result

    @property
    def final_decoded(self) -> DecodedLocalOutput:
        return self.final_round.decoded


def responses_replay_messages_from_trace(
    parsed: LocalProviderRequest,
    trace: InteractionTrace,
) -> list[dict[str, Any]]:
    replay_messages = list(parsed.response_replay_messages)
    for round_state in trace.rounds:
        if round_state.executed_tools:
            replay_messages = _append_replay_tool_results(
                replay_messages,
                round_state.decoded,
                round_state.executed_tools,
            )
            continue
        replay_messages.extend(responses_replay_output_messages(round_state.decoded))
    return replay_messages


async def run_interaction(container: Any, parsed: LocalProviderRequest) -> InteractionTrace:
    current_messages = list(parsed.inference_request.messages or [])
    current_request = parsed.inference_request
    current_parsed = parsed
    rounds: list[InteractionRound] = []

    for _ in range(MAX_TOOL_ROUNDS):
        result = await container.inference_engine.submit(current_request)
        decoded = _decode(current_parsed, result)
        round_state = InteractionRound(result=result, decoded=decoded)
        rounds.append(round_state)

        if current_parsed.feature_plan.mode != "tools" or not decoded.tool_calls:
            return InteractionTrace(rounds)
        if not all(container.tool_registry.has(tool_call.name) for tool_call in decoded.tool_calls):
            return InteractionTrace(rounds)

        executed_tools = await _execute_tools(container, current_parsed, decoded.tool_calls)
        round_state.executed_tools.extend(executed_tools)
        current_messages = _append_tool_results(current_messages, decoded, executed_tools)
        current_request = replace(current_request, messages=current_messages)
        relaxed_plan = replace(current_parsed.feature_plan, tool_choice=ToolChoice())
        current_parsed = replace(
            current_parsed,
            inference_request=current_request,
            feature_plan=relaxed_plan,
        )

    raise AsterError(
        code="tool_loop_exceeded",
        message="Tool execution exceeded the maximum number of rounds.",
        status_code=422,
    )


async def stream_interaction(
    container: Any,
    parsed: LocalProviderRequest,
    *,
    raw_request: Any | None = None,
) -> EventSourceResponse:
    trace = await run_interaction(container, parsed)
    events = _simulated_stream_events(parsed, trace)
    if raw_request is not None:
        events = stream_with_disconnect(
            events,
            raw_request,
            poll_interval_seconds=SSE_DISCONNECT_POLL_SECONDS,
            heartbeat_interval_seconds=SSE_HEARTBEAT_SECONDS,
            heartbeat_factory=lambda: {"comment": "heartbeat"},
            on_error=lambda exc: container.metrics.errors.labels(code=exc.code).inc(),
        )
    return EventSourceResponse(
        events,
        ping=SSE_HEARTBEAT_SECONDS,
        send_timeout=SSE_SEND_TIMEOUT_SECONDS,
        headers={"X-Request-Id": parsed.request_id},
    )


async def stream_live_tool_interaction(
    container: Any,
    parsed: LocalProviderRequest,
    *,
    raw_request: Any | None = None,
    on_responses_complete: Callable[[list[dict[str, Any]]], None] | None = None,
    timeout_seconds: float | None = None,
) -> EventSourceResponse:
    events = (
        _live_responses_tool_events(
            container,
            parsed,
            on_complete=on_responses_complete,
            timeout_seconds=timeout_seconds,
        )
        if parsed.api_family == "responses"
        else _live_chat_tool_events(container, parsed, timeout_seconds=timeout_seconds)
    )
    if raw_request is not None:
        events = stream_with_disconnect(
            events,
            raw_request,
            poll_interval_seconds=SSE_DISCONNECT_POLL_SECONDS,
            heartbeat_interval_seconds=SSE_HEARTBEAT_SECONDS,
            heartbeat_factory=lambda: {"comment": "heartbeat"},
            on_error=lambda exc: container.metrics.errors.labels(code=exc.code).inc(),
        )
    if parsed.api_family == "responses":
        return EventSourceResponse(
            events,
            ping=SSE_HEARTBEAT_SECONDS,
            send_timeout=SSE_SEND_TIMEOUT_SECONDS,
            headers={"X-Request-Id": parsed.request_id},
        )
    return EventSourceResponse(
        events,
        ping=SSE_HEARTBEAT_SECONDS,
        send_timeout=SSE_SEND_TIMEOUT_SECONDS,
        headers={"X-Request-Id": parsed.request_id},
    )


def _decode(parsed: LocalProviderRequest, result: InferenceResponse) -> DecodedLocalOutput:
    from aster.api.provider_gateway import decode_local_output

    return decode_local_output(result.text, parsed.feature_plan)


async def _execute_tools(
    container: Any,
    parsed: LocalProviderRequest,
    tool_calls: list[ToolCallResult],
    *,
    timeout_seconds: float | None = None,
) -> list[ExecutedToolResult]:
    if len(tool_calls) <= 1 or not parsed.feature_plan.allow_parallel_tool_calls:
        return [
            await _execute_single_tool_result(
                container,
                parsed,
                tool_call,
                timeout_seconds=timeout_seconds,
            )
            for tool_call in tool_calls
        ]

    semaphore = asyncio.Semaphore(min(MAX_PARALLEL_TOOL_CALLS, len(tool_calls)))

    async def execute_bounded(tool_call: ToolCallResult) -> ExecutedToolResult:
        async with semaphore:
            return await _execute_single_tool_result(
                container,
                parsed,
                tool_call,
                timeout_seconds=timeout_seconds,
            )

    return list(await asyncio.gather(*(execute_bounded(tool_call) for tool_call in tool_calls)))


async def _execute_single_tool_result(
    container: Any,
    parsed: LocalProviderRequest,
    tool_call: ToolCallResult,
    *,
    timeout_seconds: float | None,
) -> ExecutedToolResult:
    started = perf_counter()
    try:
        validate_tool_call_arguments(tool_call, parsed.feature_plan)
        result = await _execute_tool_with_timeout(
            container,
            parsed,
            tool_call,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        status = _tool_metric_status(exc)
        _record_tool_metrics(container, tool_call.name, status, perf_counter() - started)
        message = _tool_error_message(tool_call, exc)
        _logger.warning(
            "tool_execution_failed",
            extra={
                "request_id": parsed.request_id,
                "provider": parsed.provider,
                "api_family": parsed.api_family,
                "tool_name": tool_call.name,
                "error": exc.message if isinstance(exc, AsterError) else str(exc),
                "error_code": exc.code if isinstance(exc, AsterError) else None,
            },
        )
        return ExecutedToolResult(tool_call=tool_call, result=message, is_error=True)
    _record_tool_metrics(container, tool_call.name, "success", perf_counter() - started)
    return ExecutedToolResult(tool_call=tool_call, result=result)


async def _execute_tool_with_timeout(
    container: Any,
    parsed: LocalProviderRequest,
    tool_call: ToolCallResult,
    *,
    timeout_seconds: float | None,
) -> Any:
    execution = container.tool_registry.execute(
        tool_call.name,
        tool_call.arguments,
        ToolExecutionContext(
            request_id=parsed.request_id,
            provider=parsed.provider,
            api_family=parsed.api_family,
            model=parsed.model,
        ),
    )
    if timeout_seconds is None:
        return await execution
    try:
        return await asyncio.wait_for(execution, timeout=timeout_seconds)
    except TimeoutError:
        raise AsterError(
            code="tool_timeout",
            message="Tool execution timed out",
            status_code=504,
            details={"timeout_seconds": timeout_seconds},
        ) from None


def _tool_error_message(tool_call: ToolCallResult, exc: Exception) -> str:
    if isinstance(exc, AsterError):
        detail = exc.message
    else:
        detail = str(exc)
    if not detail:
        detail = exc.__class__.__name__
    return f"Error: Tool '{tool_call.name}' execution failed: {detail}"


def _tool_metric_status(exc: Exception) -> str:
    if isinstance(exc, AsterError) and exc.code == "tool_timeout":
        return "timeout"
    return "error"


def _record_tool_metrics(
    container: Any,
    tool_name: str,
    status: str,
    elapsed_seconds: float,
) -> None:
    metrics = getattr(container, "metrics", None)
    if metrics is None:
        return
    metrics.tool_executions.labels(tool_name=tool_name, status=status).inc()
    metrics.tool_execution_latency.labels(tool_name=tool_name, status=status).observe(
        max(elapsed_seconds, 0.0)
    )


def _append_tool_results(
    current_messages: list[dict[str, str]],
    decoded: DecodedLocalOutput,
    executed_tools: list[ExecutedToolResult],
) -> list[dict[str, str]]:
    messages = list(current_messages)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "assistant_text": decoded.assistant_text,
                    "tool_calls": [
                        {
                            "id": item.tool_call.call_id,
                            "name": item.tool_call.name,
                            "arguments": item.tool_call.arguments,
                        }
                        for item in executed_tools
                    ],
                },
                ensure_ascii=True,
            ),
        }
    )
    for item in executed_tools:
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "tool_call_id": item.tool_call.call_id,
                        "tool_name": item.tool_call.name,
                        "result": item.result,
                    },
                    ensure_ascii=True,
                ),
            }
        )
    return messages


def _append_replay_tool_results(
    current_messages: list[dict[str, Any]],
    decoded: DecodedLocalOutput,
    executed_tools: list[ExecutedToolResult],
) -> list[dict[str, Any]]:
    messages = list(current_messages)
    messages.extend(responses_replay_output_messages(decoded))
    for item in executed_tools:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": item.tool_call.call_id,
                "content": _tool_result_content(item.result),
            }
        )
    return messages


def _tool_result_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=True)


async def _live_chat_tool_events(
    container: Any,
    parsed: LocalProviderRequest,
    *,
    timeout_seconds: float | None = None,
) -> AsyncIterator[dict[str, str]]:
    created = int(time())
    current_messages = list(parsed.inference_request.messages or [])
    current_request = parsed.inference_request
    current_parsed = parsed
    sent_done = False
    cancelled = False

    try:
        for round_index in range(MAX_TOOL_ROUNDS):
            content_fragments: list[str] = []
            reasoning_fragments: list[str] = []
            streamed_tool_calls: list[ToolCallResult] = []
            finish_stats: dict[str, object] | None = None
            stream_error: AsterError | None = None

            def remember_delta(
                delta: ParsedGenerationDelta,
                content_fragments: list[str] = content_fragments,
                reasoning_fragments: list[str] = reasoning_fragments,
            ) -> None:
                if delta.content_delta:
                    content_fragments.append(delta.content_delta)
                if delta.reasoning_delta:
                    reasoning_fragments.append(delta.reasoning_delta)

            def remember_tool_call(
                tool_call: StreamedToolCall,
                streamed_tool_calls: list[ToolCallResult] = streamed_tool_calls,
            ) -> None:
                streamed_tool_calls.append(
                    ToolCallResult(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )

            def remember_finish(stats: dict[str, object] | None, _saw_tool_call: bool) -> None:
                nonlocal finish_stats
                finish_stats = stats

            def remember_error(exc: AsterError) -> None:
                nonlocal stream_error
                stream_error = exc

            async for event in iter_chat_tool_sse_events(
                _stream_chunks_with_timeout(
                    container.inference_engine.stream(current_request),
                    timeout_seconds=timeout_seconds,
                ),
                current_parsed.model,
                response_id=current_parsed.response_id,
                include_usage=False,
                emit_role=round_index == 0,
                emit_done=False,
                created=created,
                on_delta=remember_delta,
                on_tool_call=remember_tool_call,
                on_finish=remember_finish,
                on_error=remember_error,
                tool_request=tool_request_from_plan(current_parsed.feature_plan),
            ):
                yield event

            if stream_error is not None:
                yield {"data": "[DONE]"}
                sent_done = True
                return

            if current_parsed.feature_plan.mode != "tools" or not streamed_tool_calls:
                if parsed.include_stream_usage:
                    usage = _usage_from_stream_stats(finish_stats)
                    if usage is not None:
                        yield chat_usage_sse_event(
                            response_id=parsed.response_id,
                            model=parsed.model,
                            created=created,
                            usage=usage,
                        )
                yield {"data": "[DONE]"}
                sent_done = True
                return

            if not all(
                container.tool_registry.has(tool_call.name) for tool_call in streamed_tool_calls
            ):
                if parsed.include_stream_usage:
                    usage = _usage_from_stream_stats(finish_stats)
                    if usage is not None:
                        yield chat_usage_sse_event(
                            response_id=parsed.response_id,
                            model=parsed.model,
                            created=created,
                            usage=usage,
                        )
                yield {"data": "[DONE]"}
                sent_done = True
                return

            decoded = DecodedLocalOutput(
                mode="tools",
                raw_text="".join(content_fragments),
                assistant_text="".join(content_fragments) or None,
                reasoning_text="".join(reasoning_fragments) or None,
                tool_calls=streamed_tool_calls,
            )
            executed_tools = await _execute_tools(
                container,
                current_parsed,
                streamed_tool_calls,
                timeout_seconds=timeout_seconds,
            )
            current_messages = _append_tool_results(current_messages, decoded, executed_tools)
            current_request = replace(current_request, messages=current_messages)
            relaxed_plan = replace(current_parsed.feature_plan, tool_choice=ToolChoice())
            current_parsed = replace(
                current_parsed,
                inference_request=current_request,
                feature_plan=relaxed_plan,
            )

        raise AsterError(
            code="tool_loop_exceeded",
            message="Tool execution exceeded the maximum number of rounds.",
            status_code=422,
        )
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "live_tool_stream_aster_error",
            extra={
                "request_id": parsed.request_id,
                "provider": parsed.provider,
                "api_family": parsed.api_family,
                "code": exc.code,
            },
        )
        yield stream_error_event(exc)
    except Exception:
        _logger.exception(
            "live_tool_stream_failed",
            extra={
                "request_id": parsed.request_id,
                "provider": parsed.provider,
                "api_family": parsed.api_family,
            },
        )
        yield stream_error_event(
            AsterError(
                code="stream_failed",
                message="Streaming response failed",
                status_code=500,
            )
        )
    finally:
        if not sent_done and not cancelled:
            yield {"data": "[DONE]"}


async def _live_responses_tool_events(
    container: Any,
    parsed: LocalProviderRequest,
    *,
    on_complete: Callable[[list[dict[str, Any]]], None] | None = None,
    timeout_seconds: float | None = None,
) -> AsyncIterator[dict[str, str]]:
    lifecycle = ResponsesSseLifecycle(
        response_id=parsed.response_id,
        model=parsed.model,
        response_metadata=parsed.response_metadata,
    )
    output: list[dict[str, Any]] = []
    current_messages = list(parsed.inference_request.messages or [])
    current_request = parsed.inference_request
    current_parsed = parsed
    replay_messages = list(parsed.response_replay_messages)
    completed = False
    cancelled = False
    active_failure_finalizer: Callable[[], AsyncIterator[dict[str, str]]] | None = None

    try:
        yield lifecycle.created_event()
        yield lifecycle.in_progress_event()

        for round_index in range(MAX_TOOL_ROUNDS):
            text_fragments: list[str] = []
            reasoning_fragments: list[str] = []
            streamed_tool_calls: list[ToolCallResult] = []
            text_ref = None
            reasoning_ref = None
            finish_stats: dict[str, object] | None = None
            round_finished = False
            reasoning_parser = AutoReasoningParser()
            tool_parser = AutoToolParser()
            tool_parser.configure_request(tool_request_from_plan(current_parsed.feature_plan))
            tool_accumulator = ToolCallDeltaAccumulator()
            tool_call_refs: dict[int, ResponsesItemRef] = {}
            tool_call_ids: dict[int, str] = {}
            tool_call_names: dict[int, str] = {}
            tool_call_arguments: dict[int, list[str]] = {}

            async def emit_tool_delta(
                delta: ParsedGenerationDelta,
                round_index: int = round_index,
                text_fragments: list[str] = text_fragments,
                streamed_tool_calls: list[ToolCallResult] = streamed_tool_calls,
                tool_accumulator: ToolCallDeltaAccumulator = tool_accumulator,
                tool_call_refs: dict[int, ResponsesItemRef] = tool_call_refs,
                tool_call_ids: dict[int, str] = tool_call_ids,
                tool_call_names: dict[int, str] = tool_call_names,
                tool_call_arguments: dict[int, list[str]] = tool_call_arguments,
            ) -> AsyncIterator[dict[str, str]]:
                nonlocal text_ref
                if delta.content_delta:
                    if text_ref is None:
                        text_ref, events = lifecycle.start_text_item(
                            item_id=f"msg_{parsed.response_id}_{round_index}"
                        )
                        for event in events:
                            yield event
                    text_fragments.append(delta.content_delta)
                    yield lifecycle.text_delta_event(text_ref, delta.content_delta)
                for tool_delta in delta.tool_call_deltas:
                    index = tool_delta.index
                    if index not in tool_call_refs:
                        call_id = tool_delta.call_id or f"call_{index}"
                        name = tool_delta.name or "function"
                        tool_call_ids[index] = call_id
                        tool_call_names[index] = name
                        tool_call_arguments[index] = []
                        tool_call_refs[index], events = lifecycle.start_function_call_item(
                            call_id=call_id,
                            name=name,
                        )
                        for event in events:
                            yield event
                    if tool_delta.name is not None:
                        tool_call_names[index] = tool_delta.name
                    if tool_delta.call_id is not None:
                        tool_call_ids[index] = tool_delta.call_id
                    if tool_delta.arguments_delta:
                        tool_call_arguments[index].append(tool_delta.arguments_delta)
                        yield lifecycle.function_call_arguments_delta_event(
                            tool_call_refs[index],
                            tool_delta.arguments_delta,
                        )
                    streamed_call = tool_accumulator.add(tool_delta)
                    if streamed_call is not None:
                        streamed_tool_calls.append(
                            ToolCallResult(
                                call_id=streamed_call.call_id,
                                name=streamed_call.name,
                                arguments=streamed_call.arguments,
                            )
                        )
                    if tool_delta.finished and index in tool_call_refs:
                        item, events = lifecycle.finish_function_call_item(
                            tool_call_refs.pop(index),
                            call_id=tool_call_ids.pop(index),
                            name=tool_call_names.pop(index),
                            arguments="".join(tool_call_arguments.pop(index, [])),
                        )
                        output.append(item)
                        for event in events:
                            yield event

            async def finish_round(
                stats: dict[str, object] | None = None,
                round_index: int = round_index,
                reasoning_fragments: list[str] = reasoning_fragments,
                text_fragments: list[str] = text_fragments,
                streamed_tool_calls: list[ToolCallResult] = streamed_tool_calls,
                tool_parser: AutoToolParser = tool_parser,
                tool_call_refs: dict[int, ResponsesItemRef] = tool_call_refs,
                tool_call_ids: dict[int, str] = tool_call_ids,
                tool_call_names: dict[int, str] = tool_call_names,
                tool_call_arguments: dict[int, list[str]] = tool_call_arguments,
            ) -> AsyncIterator[dict[str, str]]:
                nonlocal reasoning_ref, text_ref
                flush_delta = tool_parser.flush_delta()
                suppress_empty_text_item = tool_parser.suppressed_tool_protocol
                async for event in emit_tool_delta(flush_delta):
                    yield event
                item_status = (
                    "incomplete"
                    if responses_finish_reason_from_stats(stats) == "length"
                    else "completed"
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
                    reasoning_item, events = lifecycle.finish_reasoning_item(
                        reasoning_ref, "".join(reasoning_fragments)
                    )
                    output.append(reasoning_item)
                    reasoning_ref = None
                    for event in events:
                        yield event
                has_function_call_output = any(item.get("type") == "function_call" for item in output)
                if text_ref is not None or (
                    not suppress_empty_text_item
                    and not streamed_tool_calls
                    and not has_function_call_output
                ):
                    if text_ref is None:
                        text_ref, events = lifecycle.start_text_item(
                            item_id=f"msg_{parsed.response_id}_{round_index}"
                        )
                        for event in events:
                            yield event
                    text_item, events = lifecycle.finish_text_item(
                        text_ref, "".join(text_fragments)
                    )
                    output.append(text_item)
                    text_ref = None
                    for event in events:
                        yield event

            async def finish_failed_round(
                reasoning_fragments: list[str] = reasoning_fragments,
                text_fragments: list[str] = text_fragments,
                tool_parser: AutoToolParser = tool_parser,
                tool_call_refs: dict[int, ResponsesItemRef] = tool_call_refs,
                tool_call_ids: dict[int, str] = tool_call_ids,
                tool_call_names: dict[int, str] = tool_call_names,
                tool_call_arguments: dict[int, list[str]] = tool_call_arguments,
            ) -> AsyncIterator[dict[str, str]]:
                nonlocal reasoning_ref, text_ref
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
                    output.append(reasoning_item)
                    reasoning_ref = None
                    for event in events:
                        yield event
                if text_ref is not None:
                    text_item, events = lifecycle.finish_text_item(
                        text_ref,
                        "".join(text_fragments),
                        status="incomplete",
                    )
                    output.append(text_item)
                    text_ref = None
                    for event in events:
                        yield event

            active_failure_finalizer = finish_failed_round

            async for chunk in _stream_chunks_with_timeout(
                container.inference_engine.stream(current_request),
                timeout_seconds=timeout_seconds,
            ):
                if chunk.finished:
                    finish_stats = chunk.stats
                    async for event in finish_round(finish_stats):
                        yield event
                    round_finished = True
                    break

                parsed_delta = reasoning_parser.parse_delta(chunk.token)
                if parsed_delta.reasoning_delta:
                    if reasoning_ref is None:
                        reasoning_ref, events = lifecycle.start_reasoning_item(
                            item_id=f"rs_{parsed.response_id}_{round_index}"
                        )
                        for event in events:
                            yield event
                    reasoning_fragments.append(parsed_delta.reasoning_delta)
                    yield lifecycle.reasoning_delta_event(
                        reasoning_ref, parsed_delta.reasoning_delta
                    )
                if not parsed_delta.content_delta:
                    continue
                tool_delta = tool_parser.parse_delta(parsed_delta.content_delta)
                async for event in emit_tool_delta(tool_delta):
                    yield event
            if not round_finished:
                async for event in finish_round(finish_stats):
                    yield event

            decoded = DecodedLocalOutput(
                mode="tools" if streamed_tool_calls else "plain",
                raw_text="".join(text_fragments),
                assistant_text="".join(text_fragments) or None,
                reasoning_text="".join(reasoning_fragments) or None,
                tool_calls=streamed_tool_calls,
            )

            if current_parsed.feature_plan.mode != "tools" or not streamed_tool_calls:
                if on_complete is not None:
                    on_complete(
                        [
                            *replay_messages,
                            *responses_replay_output_messages(decoded),
                        ]
                    )
                yield lifecycle.completed_event(
                    output=output,
                    usage=responses_usage_from_stats(finish_stats),
                    finish_reason=responses_finish_reason_from_stats(finish_stats),
                )
                completed = True
                return

            if not all(
                container.tool_registry.has(tool_call.name) for tool_call in streamed_tool_calls
            ):
                if on_complete is not None:
                    on_complete(
                        [
                            *replay_messages,
                            *responses_replay_output_messages(decoded),
                        ]
                    )
                yield lifecycle.completed_event(
                    output=output,
                    usage=responses_usage_from_stats(finish_stats),
                    finish_reason=responses_finish_reason_from_stats(finish_stats),
                )
                completed = True
                return

            executed_tools = await _execute_tools(
                container,
                current_parsed,
                streamed_tool_calls,
                timeout_seconds=timeout_seconds,
            )
            replay_messages = _append_replay_tool_results(
                replay_messages,
                decoded,
                executed_tools,
            )
            current_messages = _append_tool_results(current_messages, decoded, executed_tools)
            current_request = replace(current_request, messages=current_messages)
            relaxed_plan = replace(current_parsed.feature_plan, tool_choice=ToolChoice())
            current_parsed = replace(
                current_parsed,
                inference_request=current_request,
                feature_plan=relaxed_plan,
            )

        raise AsterError(
            code="tool_loop_exceeded",
            message="Tool execution exceeded the maximum number of rounds.",
            status_code=422,
        )
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    except AsterError as exc:
        _logger.warning(
            "live_responses_tool_stream_aster_error",
            extra={
                "request_id": parsed.request_id,
                "provider": parsed.provider,
                "api_family": parsed.api_family,
                "code": exc.code,
            },
        )
        if active_failure_finalizer is not None:
            async for event in active_failure_finalizer():
                yield event
        yield lifecycle.failed_event(output=output, error=responses_error_payload(exc), usage=None)
        completed = True
    except Exception:
        _logger.exception(
            "live_responses_tool_stream_failed",
            extra={
                "request_id": parsed.request_id,
                "provider": parsed.provider,
                "api_family": parsed.api_family,
            },
        )
        exc = AsterError(
            code="stream_failed",
            message="Streaming response failed",
            status_code=500,
        )
        if active_failure_finalizer is not None:
            async for event in active_failure_finalizer():
                yield event
        yield lifecycle.failed_event(output=output, error=responses_error_payload(exc), usage=None)
        completed = True
    finally:
        if not completed and not cancelled:
            yield lifecycle.completed_event(output=output, usage=None)


async def _stream_chunks_with_timeout(
    chunks: AsyncIterator[Any],
    *,
    timeout_seconds: float | None,
) -> AsyncIterator[Any]:
    if timeout_seconds is None:
        async for chunk in chunks:
            yield chunk
        return

    started = perf_counter()
    aiter = chunks.__aiter__()
    try:
        while True:
            elapsed = perf_counter() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise AsterError(
                    code="request_timeout",
                    message="Inference request timed out",
                    status_code=504,
                    details={
                        "timeout_seconds": timeout_seconds,
                        "elapsed_seconds": round(elapsed, 4),
                    },
                )
            try:
                chunk = await asyncio.wait_for(aiter.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                return
            except TimeoutError:
                elapsed = perf_counter() - started
                raise AsterError(
                    code="request_timeout",
                    message="Inference request timed out",
                    status_code=504,
                    details={
                        "timeout_seconds": timeout_seconds,
                        "elapsed_seconds": round(elapsed, 4),
                    },
                ) from None
            yield chunk
    finally:
        with suppress(Exception):
            await aiter.aclose()


async def _simulated_stream_events(
    parsed: LocalProviderRequest,
    trace: InteractionTrace,
) -> AsyncIterator[dict[str, str]]:
    if parsed.provider == "openai" and parsed.api_family == "chat_completions":
        created = int(time())
        yield chat_sse_event(
            response_id=parsed.response_id,
            model=parsed.model,
            created=created,
            delta={"role": "assistant"},
        )
        for index, round_state in enumerate(trace.rounds):
            is_last_round = index == len(trace.rounds) - 1
            for chunk in _chunk_text(round_state.decoded.reasoning_text or ""):
                yield chat_sse_event(
                    response_id=parsed.response_id,
                    model=parsed.model,
                    created=created,
                    delta={"reasoning_content": chunk},
                )
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield chat_sse_event(
                    response_id=parsed.response_id,
                    model=parsed.model,
                    created=created,
                    delta={"content": chunk},
                )
            if round_state.decoded.tool_calls:
                finish_reason = (
                    "tool_calls" if is_last_round or round_state.executed_tools else None
                )
                yield chat_sse_event(
                    response_id=parsed.response_id,
                    model=parsed.model,
                    created=created,
                    finish_reason=finish_reason,
                    delta={
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": tool_call.call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=True),
                                },
                            }
                            for tool_index, tool_call in enumerate(round_state.decoded.tool_calls)
                        ]
                    },
                )
        if not trace.final_decoded.tool_calls:
            yield chat_sse_event(
                response_id=parsed.response_id,
                model=parsed.model,
                created=created,
                delta={},
                finish_reason=_result_finish_reason(trace.final_result),
            )
        if parsed.include_stream_usage:
            yield chat_usage_sse_event(
                response_id=parsed.response_id,
                model=parsed.model,
                created=created,
                usage={
                    "prompt_tokens": trace.final_result.prompt_tokens,
                    "completion_tokens": trace.final_result.completion_tokens,
                    "total_tokens": trace.final_result.prompt_tokens
                    + trace.final_result.completion_tokens,
                },
            )
        yield {"data": "[DONE]"}
        return

    if parsed.provider == "openai" and parsed.api_family == "responses":
        lifecycle = ResponsesSseLifecycle(
            response_id=parsed.response_id,
            model=parsed.model,
            response_metadata=parsed.response_metadata,
        )
        output: list[dict[str, Any]] = []
        yield lifecycle.created_event()
        yield lifecycle.in_progress_event()
        for round_index, round_state in enumerate(trace.rounds):
            if round_state.decoded.reasoning_text:
                reasoning_ref, events = lifecycle.start_reasoning_item(
                    item_id=f"rs_{parsed.response_id}_{round_index}"
                )
                for event in events:
                    yield event
                for chunk in _chunk_text(round_state.decoded.reasoning_text):
                    yield lifecycle.reasoning_delta_event(reasoning_ref, chunk)
                reasoning_item, events = lifecycle.finish_reasoning_item(
                    reasoning_ref, round_state.decoded.reasoning_text
                )
                output.append(reasoning_item)
                for event in events:
                    yield event
            if round_state.decoded.tool_calls:
                for tool_call in round_state.decoded.tool_calls:
                    arguments = json.dumps(tool_call.arguments, ensure_ascii=True)
                    item, events = lifecycle.function_call_events(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=arguments,
                    )
                    output.append(item)
                    for event in events:
                        yield event
                continue
            text = round_state.decoded.assistant_text or ""
            text_ref, events = lifecycle.start_text_item(
                item_id=f"msg_{parsed.response_id}_{round_index}"
            )
            for event in events:
                yield event
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield lifecycle.text_delta_event(text_ref, chunk)
            text_item, events = lifecycle.finish_text_item(text_ref, text)
            output.append(text_item)
            for event in events:
                yield event
        usage = responses_usage_payload(
            trace.final_result.prompt_tokens,
            trace.final_result.completion_tokens,
        )
        yield lifecycle.completed_event(
            output=output,
            usage=usage,
            finish_reason=_result_finish_reason(trace.final_result),
        )
        return

    if parsed.provider == "anthropic" and parsed.api_family == "messages":
        yield {
            "event": "message_start",
            "data": json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": parsed.request_id,
                        "type": "message",
                        "role": "assistant",
                        "model": parsed.model,
                        "content": [],
                    },
                }
            ),
        }
        content_index = 0
        for round_state in trace.rounds:
            if round_state.decoded.reasoning_text:
                yield {
                    "event": "content_block_start",
                    "data": json.dumps(
                        {
                            "type": "content_block_start",
                            "index": content_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }
                    ),
                }
                for chunk in _chunk_text(round_state.decoded.reasoning_text):
                    yield {
                        "event": "content_block_delta",
                        "data": json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": content_index,
                                "delta": {"type": "thinking_delta", "thinking": chunk},
                            }
                        ),
                    }
                yield {
                    "event": "content_block_stop",
                    "data": json.dumps({"type": "content_block_stop", "index": content_index}),
                }
                content_index += 1
            if round_state.decoded.tool_calls:
                for tool_call in round_state.decoded.tool_calls:
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": content_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_call.call_id,
                                    "name": tool_call.name,
                                    "input": {},
                                },
                            }
                        ),
                    }
                    yield {
                        "event": "content_block_delta",
                        "data": json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": content_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(
                                        tool_call.arguments, ensure_ascii=True
                                    ),
                                },
                            }
                        ),
                    }
                    yield {
                        "event": "content_block_stop",
                        "data": json.dumps({"type": "content_block_stop", "index": content_index}),
                    }
                    content_index += 1
                continue
            yield {
                "event": "content_block_start",
                "data": json.dumps(
                    {
                        "type": "content_block_start",
                        "index": content_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                ),
            }
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": content_index,
                            "delta": {"type": "text_delta", "text": chunk},
                        }
                    ),
                }
            yield {
                "event": "content_block_stop",
                "data": json.dumps({"type": "content_block_stop", "index": content_index}),
            }
            content_index += 1
        yield {
            "event": "message_delta",
            "data": json.dumps(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use" if trace.final_decoded.tool_calls else "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": trace.final_result.prompt_tokens,
                        "output_tokens": trace.final_result.completion_tokens,
                    },
                }
            ),
        }
        yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}
        return

    if parsed.provider == "gemini":
        for round_state in trace.rounds:
            if round_state.decoded.tool_calls:
                for tool_call in round_state.decoded.tool_calls:
                    yield {
                        "data": json.dumps(
                            {
                                "candidates": [
                                    {
                                        "index": 0,
                                        "content": {
                                            "role": "model",
                                            "parts": [
                                                {
                                                    "functionCall": {
                                                        "name": tool_call.name,
                                                        "args": tool_call.arguments,
                                                    }
                                                }
                                            ],
                                        },
                                    }
                                ]
                            }
                        )
                    }
                continue
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield {
                    "data": json.dumps(
                        {
                            "candidates": [
                                {
                                    "index": 0,
                                    "content": {"role": "model", "parts": [{"text": chunk}]},
                                }
                            ]
                        }
                    )
                }
        return

    if parsed.provider == "cohere":
        yield {"data": json.dumps({"type": "message-start", "id": parsed.request_id})}
        for round_state in trace.rounds:
            if round_state.decoded.tool_calls:
                for tool_call in round_state.decoded.tool_calls:
                    yield {
                        "data": json.dumps(
                            {
                                "type": "tool-call-start",
                                "id": tool_call.call_id,
                                "name": tool_call.name,
                            }
                        )
                    }
                    yield {
                        "data": json.dumps(
                            {
                                "type": "tool-call-delta",
                                "id": tool_call.call_id,
                                "name": tool_call.name,
                                "delta": json.dumps(tool_call.arguments, ensure_ascii=True),
                            }
                        )
                    }
                continue
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield {
                    "data": json.dumps(
                        {
                            "type": "content-delta",
                            "delta": {"message": {"content": {"text": chunk}}},
                        }
                    )
                }
        finish_reason = "TOOL_CALL" if trace.final_decoded.tool_calls else "COMPLETE"
        yield {"data": json.dumps({"type": "message-end", "finish_reason": finish_reason})}
        return

    yield {
        "data": json.dumps(
            json.loads(
                encode_provider_decoded_response(
                    parsed,
                    trace.final_result,
                    trace.final_decoded,
                ).body.decode()
            )
        )
    }


def _chunk_text(text: str, size: int = 64) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def _result_finish_reason(result: object) -> str:
    finish_reason = getattr(result, "finish_reason", None)
    return finish_reason if isinstance(finish_reason, str) else "stop"


def _usage_from_stream_stats(stats: dict[str, object] | None) -> dict[str, int] | None:
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
