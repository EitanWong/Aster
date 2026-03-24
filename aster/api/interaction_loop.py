from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any

from sse_starlette.sse import EventSourceResponse

from aster.api.feature_emulation import DecodedLocalOutput, ToolCallResult
from aster.api.provider_gateway import LocalProviderRequest, encode_provider_response
from aster.core.errors import AsterError
from aster.inference.engine import InferenceResponse
from aster.runtime.tools import ToolExecutionContext


MAX_TOOL_ROUNDS = 4


@dataclass(slots=True)
class ExecutedToolResult:
    tool_call: ToolCallResult
    result: Any


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


async def run_interaction(container: Any, parsed: LocalProviderRequest) -> InteractionTrace:
    current_messages = list(parsed.inference_request.messages or [])
    current_request = parsed.inference_request
    rounds: list[InteractionRound] = []

    for _ in range(MAX_TOOL_ROUNDS):
        result = await container.scheduler.submit(current_request)
        decoded = _decode(parsed, result)
        round_state = InteractionRound(result=result, decoded=decoded)
        rounds.append(round_state)

        if parsed.feature_plan.mode != "tools" or not decoded.tool_calls:
            return InteractionTrace(rounds)
        if not all(container.tool_registry.has(tool_call.name) for tool_call in decoded.tool_calls):
            return InteractionTrace(rounds)

        executed_tools = await _execute_tools(container, parsed, decoded.tool_calls)
        round_state.executed_tools.extend(executed_tools)
        current_messages = _append_tool_results(current_messages, decoded, executed_tools)
        current_request = replace(current_request, messages=current_messages)

    raise AsterError(code="tool_loop_exceeded", message="Tool execution exceeded the maximum number of rounds.", status_code=422)


async def stream_interaction(container: Any, parsed: LocalProviderRequest) -> EventSourceResponse:
    trace = await run_interaction(container, parsed)
    return EventSourceResponse(_simulated_stream_events(parsed, trace))


def _decode(parsed: LocalProviderRequest, result: InferenceResponse) -> DecodedLocalOutput:
    from aster.api.provider_gateway import decode_local_output

    return decode_local_output(result.text, parsed.feature_plan)


async def _execute_tools(
    container: Any,
    parsed: LocalProviderRequest,
    tool_calls: list[ToolCallResult],
) -> list[ExecutedToolResult]:
    executed: list[ExecutedToolResult] = []
    for tool_call in tool_calls:
        try:
            result = await container.tool_registry.execute(
                tool_call.name,
                tool_call.arguments,
                ToolExecutionContext(
                    request_id=parsed.request_id,
                    provider=parsed.provider,
                    api_family=parsed.api_family,
                    model=parsed.model,
                ),
            )
        except Exception as exc:
            raise AsterError(
                code="tool_execution_failed",
                message=f"Tool '{tool_call.name}' execution failed.",
                status_code=500,
                details={"tool_name": tool_call.name, "error": str(exc)},
            ) from exc
        executed.append(ExecutedToolResult(tool_call=tool_call, result=result))
    return executed


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
                        {"id": item.tool_call.call_id, "name": item.tool_call.name, "arguments": item.tool_call.arguments}
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


async def _simulated_stream_events(
    parsed: LocalProviderRequest,
    trace: InteractionTrace,
) -> AsyncIterator[dict[str, str]]:
    if parsed.provider == "openai" and parsed.api_family == "chat_completions":
        for index, round_state in enumerate(trace.rounds):
            is_last_round = index == len(trace.rounds) - 1
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield {
                    "data": json.dumps(
                        {
                            "id": parsed.request_id,
                            "object": "chat.completion.chunk",
                            "model": parsed.model,
                            "choices": [{"delta": {"content": chunk}, "index": 0, "finish_reason": None}],
                        }
                    )
                }
            if round_state.decoded.tool_calls:
                finish_reason = "tool_calls" if is_last_round and not round_state.executed_tools else None
                yield {
                    "data": json.dumps(
                        {
                            "id": parsed.request_id,
                            "object": "chat.completion.chunk",
                            "model": parsed.model,
                            "choices": [
                                {
                                    "delta": {
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
                                    "index": 0,
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                    )
                }
        yield {"data": "[DONE]"}
        return

    if parsed.provider == "openai" and parsed.api_family == "responses":
        yield {
            "event": "response.created",
            "data": json.dumps(
                {
                    "type": "response.created",
                    "response": {"id": parsed.request_id, "object": "response", "model": parsed.model, "status": "in_progress"},
                }
            ),
        }
        for round_state in trace.rounds:
            if round_state.decoded.tool_calls:
                for tool_call in round_state.decoded.tool_calls:
                    arguments = json.dumps(tool_call.arguments, ensure_ascii=True)
                    yield {
                        "event": "response.function_call_arguments.delta",
                        "data": json.dumps(
                            {
                                "type": "response.function_call_arguments.delta",
                                "response_id": parsed.request_id,
                                "item_id": tool_call.call_id,
                                "name": tool_call.name,
                                "delta": arguments,
                            }
                        ),
                    }
                continue
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield {
                    "event": "response.output_text.delta",
                    "data": json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "response_id": parsed.request_id,
                            "item_id": f"msg_{parsed.request_id}",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": chunk,
                        }
                    ),
                }
        final_payload = encode_provider_response(parsed, trace.final_result).body.decode()
        yield {"event": "response.completed", "data": json.dumps({"type": "response.completed", "response": json.loads(final_payload)})}
        return

    if parsed.provider == "anthropic" and parsed.api_family == "messages":
        yield {
            "event": "message_start",
            "data": json.dumps(
                {
                    "type": "message_start",
                    "message": {"id": parsed.request_id, "type": "message", "role": "assistant", "model": parsed.model, "content": []},
                }
            ),
        }
        content_index = 0
        for round_state in trace.rounds:
            if round_state.decoded.tool_calls:
                for tool_call in round_state.decoded.tool_calls:
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": content_index,
                                "content_block": {"type": "tool_use", "id": tool_call.call_id, "name": tool_call.name, "input": {}},
                            }
                        ),
                    }
                    yield {
                        "event": "content_block_delta",
                        "data": json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": content_index,
                                "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_call.arguments, ensure_ascii=True)},
                            }
                        ),
                    }
                    yield {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": content_index})}
                    content_index += 1
                continue
            yield {
                "event": "content_block_start",
                "data": json.dumps({"type": "content_block_start", "index": content_index, "content_block": {"type": "text", "text": ""}}),
            }
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps({"type": "content_block_delta", "index": content_index, "delta": {"type": "text_delta", "text": chunk}}),
                }
            yield {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": content_index})}
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
                                    {"index": 0, "content": {"role": "model", "parts": [{"functionCall": {"name": tool_call.name, "args": tool_call.arguments}}]}}
                                ]
                            }
                        )
                    }
                continue
            for chunk in _chunk_text(round_state.decoded.assistant_text or ""):
                yield {"data": json.dumps({"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": chunk}]}}]})}
        return

    if parsed.provider == "cohere":
        yield {"data": json.dumps({"type": "message-start", "id": parsed.request_id})}
        for round_state in trace.rounds:
            if round_state.decoded.tool_calls:
                for tool_call in round_state.decoded.tool_calls:
                    yield {"data": json.dumps({"type": "tool-call-start", "id": tool_call.call_id, "name": tool_call.name})}
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
                yield {"data": json.dumps({"type": "content-delta", "delta": {"message": {"content": {"text": chunk}}}})}
        finish_reason = "TOOL_CALL" if trace.final_decoded.tool_calls else "COMPLETE"
        yield {"data": json.dumps({"type": "message-end", "finish_reason": finish_reason})}
        return

    yield {"data": json.dumps(json.loads(encode_provider_response(parsed, trace.final_result).body.decode()))}


def _chunk_text(text: str, size: int = 16) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]
