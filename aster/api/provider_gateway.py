from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from aster.api.feature_emulation import (
    DecodedLocalOutput,
    FeaturePlan,
    ToolChoice,
    ToolSpec,
    apply_feature_plan,
    build_structured_plan,
    build_tool_plan,
    decode_local_output,
    parse_anthropic_tool_choice,
    parse_anthropic_tools,
    parse_bedrock_tools,
    parse_cohere_tools,
    parse_gemini_structured_schema,
    parse_gemini_tool_config,
    parse_gemini_tools,
    parse_openai_responses_text_format,
    parse_openai_tool_choice,
    parse_openai_tools,
    parse_structured_schema,
)
from aster.core.errors import AsterError
from aster.inference.decode_engine import DecodeChunk
from aster.inference.engine import InferenceRequest, InferenceResponse


@dataclass(slots=True)
class LocalProviderRequest:
    provider: str
    api_family: str
    model: str
    inference_request: InferenceRequest
    request_id: str
    stream: bool
    feature_plan: FeaturePlan


def build_provider_request(
    *,
    provider: str,
    api_family: str,
    body: Mapping[str, Any],
    model: str | None = None,
    request_id: str | None = None,
) -> LocalProviderRequest:
    assigned_request_id = request_id or str(uuid.uuid4())
    parser = _PARSERS[(provider, api_family)]
    return parser(body, model=model, request_id=assigned_request_id)


def encode_provider_response(parsed: LocalProviderRequest, result: InferenceResponse) -> JSONResponse:
    encoder = _FINAL_ENCODERS[(parsed.provider, parsed.api_family)]
    decoded = decode_local_output(result.text, parsed.feature_plan)
    return JSONResponse(encoder(parsed, result, decoded))


def encode_provider_stream(
    parsed: LocalProviderRequest,
    chunks: AsyncIterator[DecodeChunk],
) -> EventSourceResponse:
    if parsed.feature_plan.mode != "plain":
        raise _unsupported(parsed.provider, parsed.api_family, "Streaming tool-calling and structured outputs are not yet supported.")
    encoder = _STREAM_ENCODERS[(parsed.provider, parsed.api_family)]
    return EventSourceResponse(encoder(parsed, chunks))


def provider_error_response(provider: str, api_family: str, exc: AsterError, request_id: str | None = None) -> JSONResponse:
    payload_builder = _ERROR_ENCODERS.get((provider, api_family)) or _openai_error_payload
    headers = {"X-Request-Id": request_id} if request_id else None
    return JSONResponse(status_code=exc.status_code, content=payload_builder(exc), headers=headers)


def _parse_openai_chat(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    messages = [_normalize_openai_like_message(item) for item in _list(body.get("messages"))]
    feature_plan = _feature_plan_openai_chat(body)
    return _build_local_request(
        provider="openai",
        api_family="chat_completions",
        model=model or _str(body.get("model")) or "local-model",
        request_id=request_id,
        messages=messages,
        stream=bool(body.get("stream", False)),
        max_tokens=_int(body.get("max_tokens"), default=256),
        temperature=_float(body.get("temperature"), default=0.7),
        top_p=_float(body.get("top_p"), default=0.95),
        feature_plan=feature_plan,
    )


def _parse_openai_responses(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    _reject_truthy(body, "background", "previous_response_id")
    messages = _normalize_openai_responses_input(body.get("input"))
    feature_plan = _feature_plan_openai_responses(body)
    return _build_local_request(
        provider="openai",
        api_family="responses",
        model=model or _str(body.get("model")) or "local-model",
        request_id=request_id,
        messages=messages,
        stream=bool(body.get("stream", False)),
        max_tokens=_int(body.get("max_output_tokens"), default=256),
        temperature=_float(body.get("temperature"), default=0.7),
        top_p=_float(body.get("top_p"), default=0.95),
        feature_plan=feature_plan,
    )


def _parse_anthropic_messages(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    _reject_truthy(body, "thinking", "container")
    messages: list[dict[str, str]] = []
    system_text = _extract_anthropic_system(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for item in _list(body.get("messages")):
        if not isinstance(item, Mapping):
            continue
        messages.append({"role": _anthropic_role(item.get("role")), "content": _extract_anthropic_content(item.get("content"))})
    tools, _, _ = parse_anthropic_tools(body.get("tools"))
    tool_choice = parse_anthropic_tool_choice(body.get("tool_choice"))
    feature_plan = build_tool_plan(tools=tools, tool_choice=tool_choice) if tools else FeaturePlan()
    return _build_local_request(
        provider="anthropic",
        api_family="messages",
        model=model or _str(body.get("model")) or "local-model",
        request_id=request_id,
        messages=messages,
        stream=bool(body.get("stream", False)),
        max_tokens=_int(body.get("max_tokens"), default=256),
        temperature=_float(body.get("temperature"), default=0.7),
        top_p=_float(body.get("top_p"), default=0.95),
        feature_plan=feature_plan,
    )


def _parse_gemini_generate_content(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    generation_config = body.get("generationConfig")
    if isinstance(generation_config, Mapping) and generation_config.get("thinkingConfig"):
        raise _unsupported("gemini", "generate_content", "Gemini thinking controls are not yet supported by the local runtime.")
    messages: list[dict[str, str]] = []
    system_instruction = body.get("systemInstruction")
    if isinstance(system_instruction, Mapping):
        system_text = _extract_gemini_parts(system_instruction.get("parts"))
        if system_text:
            messages.append({"role": "system", "content": system_text})
    for item in _list(body.get("contents")):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, Mapping):
            continue
        messages.append({"role": "assistant" if _str(item.get("role")) == "model" else "user", "content": _extract_gemini_parts(item.get("parts"))})
    feature_plan = _feature_plan_gemini(body)
    return _build_local_request(
        provider="gemini",
        api_family="generate_content",
        model=model or "local-model",
        request_id=request_id,
        messages=messages,
        stream=False,
        max_tokens=_int_from_mapping(generation_config, "maxOutputTokens", 256),
        temperature=_float_from_mapping(generation_config, "temperature", 0.7),
        top_p=_float_from_mapping(generation_config, "topP", 0.95),
        feature_plan=feature_plan,
    )


def _parse_gemini_stream_generate_content(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    parsed = _parse_gemini_generate_content(body, model=model, request_id=request_id)
    parsed.stream = True
    parsed.inference_request.stream = True
    parsed.api_family = "stream_generate_content"
    return parsed


def _parse_cohere_chat(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    messages: list[dict[str, str]] = []
    for item in _list(body.get("messages")):
        if not isinstance(item, Mapping):
            continue
        messages.append({"role": _str(item.get("role")) or "user", "content": _extract_cohere_content(item.get("content"))})
    feature_plan = _feature_plan_cohere(body)
    return _build_local_request(
        provider="cohere",
        api_family="chat_v2",
        model=model or _str(body.get("model")) or "local-model",
        request_id=request_id,
        messages=messages,
        stream=bool(body.get("stream", False)),
        max_tokens=_int(body.get("max_tokens"), default=256),
        temperature=_float(body.get("temperature"), default=0.7),
        top_p=_float(body.get("p"), default=0.95),
        feature_plan=feature_plan,
    )


def _parse_bedrock_converse(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    _reject_truthy(body, "guardrailConfig", "additionalModelRequestFields")
    messages: list[dict[str, str]] = []
    for item in _list(body.get("system")):
        if isinstance(item, Mapping):
            text = _str(item.get("text"))
            if text:
                messages.append({"role": "system", "content": text})
    for item in _list(body.get("messages")):
        if isinstance(item, Mapping):
            messages.append({"role": _str(item.get("role")) or "user", "content": _extract_bedrock_content(item.get("content"))})
    tools, tool_choice, allow_parallel = parse_bedrock_tools(body.get("toolConfig"))
    feature_plan = build_tool_plan(tools=tools, tool_choice=tool_choice, allow_parallel_tool_calls=allow_parallel) if tools else FeaturePlan()
    inference_config = body.get("inferenceConfig")
    return _build_local_request(
        provider="bedrock",
        api_family="converse",
        model=model or "local-model",
        request_id=request_id,
        messages=messages,
        stream=False,
        max_tokens=_int_from_mapping(inference_config, "maxTokens", 256),
        temperature=_float_from_mapping(inference_config, "temperature", 0.7),
        top_p=_float_from_mapping(inference_config, "topP", 0.95),
        feature_plan=feature_plan,
    )


def _parse_xai_chat(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    parsed = _parse_openai_chat(body, model=model, request_id=request_id)
    parsed.provider = "xai"
    parsed.api_family = "chat_completions"
    return parsed


def _parse_xai_responses(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    parsed = _parse_openai_responses(body, model=model, request_id=request_id)
    parsed.provider = "xai"
    parsed.api_family = "responses"
    return parsed


def _parse_mistral_chat(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
) -> LocalProviderRequest:
    parsed = _parse_openai_chat(body, model=model, request_id=request_id)
    parsed.provider = "mistral"
    parsed.api_family = "chat_completions"
    return parsed


def _build_local_request(
    *,
    provider: str,
    api_family: str,
    model: str,
    request_id: str,
    messages: list[dict[str, str]],
    stream: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    feature_plan: FeaturePlan,
) -> LocalProviderRequest:
    augmented_messages = apply_feature_plan(messages, feature_plan)
    return LocalProviderRequest(
        provider=provider,
        api_family=api_family,
        model=model,
        request_id=request_id,
        stream=stream,
        feature_plan=feature_plan,
        inference_request=InferenceRequest(
            messages=augmented_messages,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            trace_id=request_id,
        ),
    )


def _openai_chat_payload(parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput) -> dict[str, Any]:
    if decoded.tool_calls:
        return {
            "id": parsed.request_id,
            "object": "chat.completion",
            "created": 0,
            "model": parsed.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": decoded.assistant_text,
                        "tool_calls": [
                            {
                                "id": tool_call.call_id,
                                "type": "function",
                                "function": {"name": tool_call.name, "arguments": json.dumps(tool_call.arguments, ensure_ascii=True)},
                            }
                            for tool_call in decoded.tool_calls
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": _openai_usage(result),
        }
    return {
        "id": parsed.request_id,
        "object": "chat.completion",
        "created": 0,
        "model": parsed.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": decoded.assistant_text or ""}, "finish_reason": "stop"}],
        "usage": _openai_usage(result),
    }


def _openai_responses_payload(parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput) -> dict[str, Any]:
    if decoded.tool_calls:
        output: list[dict[str, Any]] = []
        for tool_call in decoded.tool_calls:
            output.append(
                {
                    "id": f"fc_{tool_call.call_id}",
                    "type": "function_call",
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=True),
                    "status": "completed",
                }
            )
        return {
            "id": parsed.request_id,
            "object": "response",
            "status": "completed",
            "model": parsed.model,
            "output": output,
            "usage": _responses_usage(result),
        }
    return {
        "id": parsed.request_id,
        "object": "response",
        "status": "completed",
        "model": parsed.model,
        "output": [
            {
                "id": f"msg_{parsed.request_id}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": decoded.assistant_text or "", "annotations": []}],
            }
        ],
        "usage": _responses_usage(result),
    }


def _anthropic_payload(parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput) -> dict[str, Any]:
    if decoded.tool_calls:
        return {
            "id": parsed.request_id,
            "type": "message",
            "role": "assistant",
            "model": parsed.model,
            "content": [
                {"type": "tool_use", "id": tool_call.call_id, "name": tool_call.name, "input": tool_call.arguments}
                for tool_call in decoded.tool_calls
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": result.prompt_tokens, "output_tokens": result.completion_tokens},
        }
    return {
        "id": parsed.request_id,
        "type": "message",
        "role": "assistant",
        "model": parsed.model,
        "content": [{"type": "text", "text": decoded.assistant_text or ""}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": result.prompt_tokens, "output_tokens": result.completion_tokens},
    }


def _gemini_payload(parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput) -> dict[str, Any]:
    parts: list[dict[str, Any]]
    if decoded.tool_calls:
        parts = [{"functionCall": {"name": tool_call.name, "args": tool_call.arguments}} for tool_call in decoded.tool_calls]
    else:
        parts = [{"text": decoded.assistant_text or ""}]
    return {
        "candidates": [{"index": 0, "content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": result.prompt_tokens,
            "candidatesTokenCount": result.completion_tokens,
            "totalTokenCount": result.prompt_tokens + result.completion_tokens,
        },
        "modelVersion": parsed.model,
    }


def _cohere_payload(parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": decoded.assistant_text or ""}]}
    finish_reason = "COMPLETE"
    if decoded.tool_calls:
        message = {
            "role": "assistant",
            "content": [],
            "tool_calls": [{"id": tool_call.call_id, "name": tool_call.name, "arguments": tool_call.arguments} for tool_call in decoded.tool_calls],
        }
        finish_reason = "TOOL_CALL"
    return {
        "id": parsed.request_id,
        "message": message,
        "finish_reason": finish_reason,
        "usage": {"tokens": {"input_tokens": result.prompt_tokens, "output_tokens": result.completion_tokens}},
    }


def _bedrock_payload(parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput) -> dict[str, Any]:
    content = [{"text": decoded.assistant_text or ""}]
    stop_reason = "end_turn"
    if decoded.tool_calls:
        content = [{"toolUse": {"toolUseId": tool_call.call_id, "name": tool_call.name, "input": tool_call.arguments}} for tool_call in decoded.tool_calls]
        stop_reason = "tool_use"
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": result.prompt_tokens,
            "outputTokens": result.completion_tokens,
            "totalTokens": result.prompt_tokens + result.completion_tokens,
        },
    }


async def _openai_chat_stream(parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]) -> AsyncIterator[dict[str, str]]:
    async for chunk in chunks:
        if chunk.finished:
            yield {"data": "[DONE]"}
            break
        yield {"data": json.dumps({"id": parsed.request_id, "object": "chat.completion.chunk", "model": parsed.model, "choices": [{"delta": {"content": chunk.token}, "index": 0, "finish_reason": None}]})}


async def _openai_responses_stream(parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]) -> AsyncIterator[dict[str, str]]:
    yield {"event": "response.created", "data": json.dumps({"type": "response.created", "response": {"id": parsed.request_id, "object": "response", "model": parsed.model, "status": "in_progress"}})}
    text_fragments: list[str] = []
    async for chunk in chunks:
        if chunk.finished:
            stats = chunk.stats or {}
            prompt_tokens = _int(stats.get("prompt_tokens"), default=0)
            completion_tokens = _int(stats.get("completion_tokens"), default=len(text_fragments))
            yield {
                "event": "response.completed",
                "data": json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": parsed.request_id,
                            "object": "response",
                            "model": parsed.model,
                            "status": "completed",
                            "output": [{"id": f"msg_{parsed.request_id}", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "".join(text_fragments), "annotations": []}]}],
                            "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
                        },
                    }
                ),
            }
            break
        text_fragments.append(chunk.token)
        yield {"event": "response.output_text.delta", "data": json.dumps({"type": "response.output_text.delta", "response_id": parsed.request_id, "item_id": f"msg_{parsed.request_id}", "output_index": 0, "content_index": 0, "delta": chunk.token})}


async def _anthropic_stream(parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]) -> AsyncIterator[dict[str, str]]:
    yield {"event": "message_start", "data": json.dumps({"type": "message_start", "message": {"id": parsed.request_id, "type": "message", "role": "assistant", "model": parsed.model, "content": []}})}
    yield {"event": "content_block_start", "data": json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})}
    async for chunk in chunks:
        if chunk.finished:
            stats = chunk.stats or {}
            yield {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": 0})}
            yield {"event": "message_delta", "data": json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"input_tokens": _int(stats.get("prompt_tokens"), default=0), "output_tokens": _int(stats.get("completion_tokens"), default=0)}})}
            yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}
            break
        yield {"event": "content_block_delta", "data": json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk.token}})}


async def _gemini_stream(parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]) -> AsyncIterator[dict[str, str]]:
    async for chunk in chunks:
        if chunk.finished:
            break
        yield {"data": json.dumps({"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": chunk.token}]}}]})}


async def _cohere_stream(parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]) -> AsyncIterator[dict[str, str]]:
    yield {"data": json.dumps({"type": "message-start", "id": parsed.request_id})}
    async for chunk in chunks:
        if chunk.finished:
            yield {"data": json.dumps({"type": "message-end", "finish_reason": "COMPLETE"})}
            break
        yield {"data": json.dumps({"type": "content-delta", "delta": {"message": {"content": {"text": chunk.token}}}})}


def _openai_error_payload(exc: AsterError) -> dict[str, Any]:
    return {"error": {"message": exc.message, "type": exc.code, "code": exc.code}}


def _anthropic_error_payload(exc: AsterError) -> dict[str, Any]:
    return {"type": "error", "error": {"type": exc.code, "message": exc.message}}


def _gemini_error_payload(exc: AsterError) -> dict[str, Any]:
    return {"error": {"code": exc.status_code, "message": exc.message, "status": exc.code.upper()}}


def _cohere_error_payload(exc: AsterError) -> dict[str, Any]:
    return {"message": exc.message}


def _bedrock_error_payload(exc: AsterError) -> dict[str, Any]:
    return {"message": exc.message, "__type": exc.code}


def _feature_plan_openai_chat(body: Mapping[str, Any]) -> FeaturePlan:
    tools, _, _ = parse_openai_tools(body.get("tools"))
    tool_choice = parse_openai_tool_choice(body.get("tool_choice"))
    parallel = bool(body.get("parallel_tool_calls", True))
    response_format = body.get("response_format")
    return _build_feature_plan(
        tools=tools,
        tool_choice=tool_choice,
        parallel=parallel,
        structured=_openai_response_format_schema(response_format),
    )


def _feature_plan_openai_responses(body: Mapping[str, Any]) -> FeaturePlan:
    tools, _, _ = parse_openai_tools(body.get("tools"))
    tool_choice = parse_openai_tool_choice(body.get("tool_choice"))
    parallel = bool(body.get("parallel_tool_calls", True))
    text_payload = body.get("text")
    structured: tuple[dict[str, Any], str | None] | None = None
    if isinstance(text_payload, Mapping) and isinstance(text_payload.get("format"), Mapping):
        structured = parse_openai_responses_text_format(text_payload)
    return _build_feature_plan(tools=tools, tool_choice=tool_choice, parallel=parallel, structured=structured)


def _feature_plan_gemini(body: Mapping[str, Any]) -> FeaturePlan:
    tools, _, _ = parse_gemini_tools(body.get("tools"))
    tool_choice = parse_gemini_tool_config(body.get("toolConfig"))
    generation_config = body.get("generationConfig")
    structured: tuple[dict[str, Any], str | None] | None = None
    if isinstance(generation_config, Mapping) and (
        generation_config.get("responseMimeType") == "application/json" or isinstance(generation_config.get("responseSchema"), Mapping)
    ):
        if isinstance(generation_config.get("responseSchema"), Mapping):
            structured = parse_gemini_structured_schema(generation_config)
        else:
            structured = ({"type": "object"}, None)
    return _build_feature_plan(tools=tools, tool_choice=tool_choice, parallel=True, structured=structured)


def _feature_plan_cohere(body: Mapping[str, Any]) -> FeaturePlan:
    tools, _, _ = parse_cohere_tools(body.get("tools"))
    tool_choice = parse_openai_tool_choice(body.get("tool_choice"))
    response_format = body.get("response_format")
    structured: tuple[dict[str, Any], str | None] | None = None
    if isinstance(response_format, Mapping):
        if isinstance(response_format.get("json_schema"), Mapping):
            structured = (response_format["json_schema"], None)
        elif response_format.get("type") == "json_object":
            structured = ({"type": "object"}, None)
    return _build_feature_plan(tools=tools, tool_choice=tool_choice, parallel=True, structured=structured)


def _openai_response_format_schema(response_format: Any) -> tuple[dict[str, Any], str | None] | None:
    if not isinstance(response_format, Mapping):
        return None
    if response_format.get("type") == "json_object":
        return {"type": "object"}, None
    return parse_structured_schema(dict(response_format))


def _build_feature_plan(
    *,
    tools: list[ToolSpec],
    tool_choice: ToolChoice,
    parallel: bool,
    structured: tuple[dict[str, Any], str | None] | None,
) -> FeaturePlan:
    if tools and structured:
        raise AsterError(code="feature_combination_unsupported", message="Combining tools and structured outputs is not yet supported by the local runtime.", status_code=400)
    if tools:
        return build_tool_plan(tools=tools, tool_choice=tool_choice, allow_parallel_tool_calls=parallel)
    if structured:
        return build_structured_plan(structured[0], name=structured[1])
    return FeaturePlan()


def _normalize_openai_like_message(item: Mapping[str, Any]) -> dict[str, str]:
    role = _str(item.get("role")) or "user"
    if role == "developer":
        role = "system"
    if role == "function":
        role = "tool"
    return {"role": role, "content": _extract_openai_content(item.get("content"))}


def _normalize_openai_responses_input(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages: list[dict[str, str]] = []
    for item in _list(value):
        if not isinstance(item, Mapping):
            continue
        messages.append({"role": _str(item.get("role")) or "user", "content": _extract_openai_content(item.get("content"))})
    return messages


def _extract_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    fragments: list[str] = []
    for part in _list(content):
        if not isinstance(part, Mapping):
            continue
        part_type = _str(part.get("type"))
        if part_type in {None, "text", "input_text", "output_text"}:
            text = _str(part.get("text")) or _str(part.get("input_text")) or _str(part.get("content"))
            if text:
                fragments.append(text)
            continue
        if part_type == "image_url":
            fragments.append("[image]")
            continue
        if part_type == "input_audio":
            fragments.append("[audio]")
            continue
        raise _unsupported("openai", "content", f"Non-text content part '{part_type}' is not yet supported by the local runtime.")
    return "\n".join(fragment for fragment in fragments if fragment).strip()


def _extract_anthropic_system(content: Any) -> str:
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for item in _list(content):
        if isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str):
            fragments.append(item["text"])
    return "\n".join(fragments).strip()


def _extract_anthropic_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for item in _list(content):
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            fragments.append(item["text"])
            continue
        raise _unsupported("anthropic", "messages", f"Anthropic content block '{item.get('type')}' is not yet supported by the local runtime.")
    return "\n".join(fragments).strip()


def _extract_gemini_parts(parts: Any) -> str:
    fragments: list[str] = []
    for item in _list(parts):
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            fragments.append(item["text"])
            continue
        raise _unsupported("gemini", "generate_content", "Only text Gemini parts are currently supported by the local runtime.")
    return "\n".join(fragments).strip()


def _extract_cohere_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for item in _list(content):
        if isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str):
            fragments.append(item["text"])
            continue
        raise _unsupported("cohere", "chat_v2", "Only text Cohere content parts are currently supported by the local runtime.")
    return "\n".join(fragments).strip()


def _extract_bedrock_content(content: Any) -> str:
    fragments: list[str] = []
    for item in _list(content):
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            fragments.append(item["text"])
            continue
        raise _unsupported("bedrock", "converse", "Only text Bedrock content blocks are currently supported by the local runtime.")
    return "\n".join(fragments).strip()


def _anthropic_role(value: Any) -> str:
    role = _str(value) or "user"
    return "assistant" if role == "assistant" else "user"


def _openai_usage(result: InferenceResponse) -> dict[str, int]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _responses_usage(result: InferenceResponse) -> dict[str, int]:
    return {
        "input_tokens": result.prompt_tokens,
        "output_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _reject_truthy(body: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        value = body.get(key)
        if value in (None, False, "", [], {}):
            continue
        raise _unsupported("local", key, f"Field '{key}' is not yet supported by the local runtime.")


def _unsupported(provider: str, api_family: str, message: str) -> AsterError:
    return AsterError(code=f"{provider}_{api_family}_unsupported", message=message, status_code=400)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: Any, *, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _float(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _int_from_mapping(mapping: Any, key: str, default: int) -> int:
    if isinstance(mapping, Mapping):
        return _int(mapping.get(key), default=default)
    return default


def _float_from_mapping(mapping: Any, key: str, default: float) -> float:
    if isinstance(mapping, Mapping):
        return _float(mapping.get(key), default=default)
    return default


_PARSERS: dict[tuple[str, str], Any] = {
    ("openai", "chat_completions"): _parse_openai_chat,
    ("openai", "responses"): _parse_openai_responses,
    ("anthropic", "messages"): _parse_anthropic_messages,
    ("gemini", "generate_content"): _parse_gemini_generate_content,
    ("gemini", "stream_generate_content"): _parse_gemini_stream_generate_content,
    ("cohere", "chat_v2"): _parse_cohere_chat,
    ("bedrock", "converse"): _parse_bedrock_converse,
    ("xai", "chat_completions"): _parse_xai_chat,
    ("xai", "responses"): _parse_xai_responses,
    ("mistral", "chat_completions"): _parse_mistral_chat,
}

_FINAL_ENCODERS: dict[tuple[str, str], Any] = {
    ("openai", "chat_completions"): _openai_chat_payload,
    ("openai", "responses"): _openai_responses_payload,
    ("anthropic", "messages"): _anthropic_payload,
    ("gemini", "generate_content"): _gemini_payload,
    ("cohere", "chat_v2"): _cohere_payload,
    ("bedrock", "converse"): _bedrock_payload,
    ("xai", "chat_completions"): _openai_chat_payload,
    ("xai", "responses"): _openai_responses_payload,
    ("mistral", "chat_completions"): _openai_chat_payload,
}

_STREAM_ENCODERS: dict[tuple[str, str], Any] = {
    ("openai", "chat_completions"): _openai_chat_stream,
    ("openai", "responses"): _openai_responses_stream,
    ("anthropic", "messages"): _anthropic_stream,
    ("gemini", "stream_generate_content"): _gemini_stream,
    ("cohere", "chat_v2"): _cohere_stream,
    ("xai", "chat_completions"): _openai_chat_stream,
    ("xai", "responses"): _openai_responses_stream,
    ("mistral", "chat_completions"): _openai_chat_stream,
}

_ERROR_ENCODERS: dict[tuple[str, str], Any] = {
    ("openai", "chat_completions"): _openai_error_payload,
    ("openai", "responses"): _openai_error_payload,
    ("anthropic", "messages"): _anthropic_error_payload,
    ("gemini", "generate_content"): _gemini_error_payload,
    ("gemini", "stream_generate_content"): _gemini_error_payload,
    ("cohere", "chat_v2"): _cohere_error_payload,
    ("bedrock", "converse"): _bedrock_error_payload,
    ("xai", "chat_completions"): _openai_error_payload,
    ("xai", "responses"): _openai_error_payload,
    ("mistral", "chat_completions"): _openai_error_payload,
}
