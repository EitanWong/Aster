from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from aster.api.content_parts import is_multimodal_content_type
from aster.api.disconnect import stream_with_disconnect
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
    tool_request_from_plan,
)
from aster.api.streaming import (
    SSE_DISCONNECT_POLL_SECONDS,
    SSE_HEARTBEAT_SECONDS,
    SSE_SEND_TIMEOUT_SECONDS,
    iter_chat_sse_events,
    iter_chat_structured_sse_events,
    iter_chat_tool_sse_events,
    iter_responses_sse_events,
    iter_responses_structured_sse_events,
    iter_responses_tool_sse_events,
    responses_output_text,
    responses_status_fields_from_finish_reason,
    responses_usage_payload,
)
from aster.core.errors import AsterError
from aster.inference.decode_engine import DecodeChunk
from aster.inference.engine import InferenceRequest, InferenceResponse
from aster.inference.reasoning_parsers import AutoReasoningParser
from aster.inference.tool_parsers import AutoToolParser


def _empty_response_metadata() -> dict[str, Any]:
    return {}


def _empty_response_replay_messages() -> list[dict[str, Any]]:
    return []


@dataclass(slots=True)
class LocalProviderRequest:
    provider: str
    api_family: str
    model: str
    inference_request: InferenceRequest
    request_id: str
    stream: bool
    feature_plan: FeaturePlan
    include_stream_usage: bool = False
    response_id: str = ""
    response_metadata: dict[str, Any] = field(default_factory=_empty_response_metadata)
    response_replay_messages: list[dict[str, Any]] = field(
        default_factory=_empty_response_replay_messages
    )

    def __post_init__(self) -> None:
        if not self.response_id:
            self.response_id = self.request_id


def build_provider_request(
    *,
    provider: str,
    api_family: str,
    body: Mapping[str, Any],
    model: str | None = None,
    request_id: str | None = None,
    previous_response_messages: list[dict[str, Any]] | None = None,
) -> LocalProviderRequest:
    assigned_request_id = request_id or str(uuid.uuid4())
    parser = _PARSERS[(provider, api_family)]
    if api_family == "responses" and provider in {"openai", "xai"}:
        return parser(
            body,
            model=model,
            request_id=assigned_request_id,
            previous_response_messages=previous_response_messages,
        )
    return parser(body, model=model, request_id=assigned_request_id)


def encode_provider_response(
    parsed: LocalProviderRequest, result: InferenceResponse
) -> JSONResponse:
    encoder = _FINAL_ENCODERS[(parsed.provider, parsed.api_family)]
    decoded = decode_local_output(result.text, parsed.feature_plan)
    return JSONResponse(
        encoder(parsed, result, decoded),
        headers={"X-Request-Id": parsed.request_id},
    )


def encode_provider_decoded_response(
    parsed: LocalProviderRequest,
    result: InferenceResponse,
    decoded: DecodedLocalOutput,
) -> JSONResponse:
    encoder = _FINAL_ENCODERS[(parsed.provider, parsed.api_family)]
    return JSONResponse(
        encoder(parsed, result, decoded),
        headers={"X-Request-Id": parsed.request_id},
    )


def responses_replay_output_messages(decoded: DecodedLocalOutput) -> list[dict[str, Any]]:
    tool_calls = [
        {
            "id": tool_call.call_id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.arguments, ensure_ascii=True),
            },
        }
        for tool_call in decoded.tool_calls
    ]
    assistant_text = decoded.assistant_text or ""
    if not assistant_text and not tool_calls:
        return []
    message: dict[str, Any] = {"role": "assistant", "content": assistant_text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return [message]


def encode_provider_stream(
    parsed: LocalProviderRequest,
    chunks: AsyncIterator[DecodeChunk],
    *,
    raw_request: Any | None = None,
) -> EventSourceResponse:
    if parsed.feature_plan.mode == "tools" and not supports_provider_live_tool_stream(parsed):
        raise _unsupported(
            parsed.provider,
            parsed.api_family,
            "Streaming tool-calling is only supported for OpenAI-like local provider streams.",
        )
    if parsed.feature_plan.mode == "structured" and not supports_provider_structured_stream(parsed):
        raise _unsupported(
            parsed.provider,
            parsed.api_family,
            "Streaming structured outputs are only supported for OpenAI-like local provider streams.",
        )
    if parsed.feature_plan.mode not in {"plain", "tools", "structured"}:
        raise _unsupported(
            parsed.provider, parsed.api_family, "Streaming feature mode is not supported."
        )
    encoder = _STREAM_ENCODERS[(parsed.provider, parsed.api_family)]
    events = encoder(parsed, chunks)
    if raw_request is not None:
        metrics = raw_request.app.state.container.metrics
        events = stream_with_disconnect(
            events,
            raw_request,
            poll_interval_seconds=SSE_DISCONNECT_POLL_SECONDS,
            heartbeat_interval_seconds=SSE_HEARTBEAT_SECONDS,
            heartbeat_factory=lambda: {"comment": "heartbeat"},
            on_error=lambda exc: metrics.errors.labels(code=exc.code).inc(),
        )
    return EventSourceResponse(
        events,
        ping=SSE_HEARTBEAT_SECONDS,
        send_timeout=SSE_SEND_TIMEOUT_SECONDS,
        headers={"X-Request-Id": parsed.request_id},
    )


def supports_provider_live_tool_stream(parsed: LocalProviderRequest) -> bool:
    return parsed.feature_plan.mode == "tools" and (parsed.provider, parsed.api_family) in {
        ("openai", "chat_completions"),
        ("openai", "responses"),
        ("xai", "chat_completions"),
        ("xai", "responses"),
        ("mistral", "chat_completions"),
    }


def supports_provider_structured_stream(parsed: LocalProviderRequest) -> bool:
    return parsed.feature_plan.mode == "structured" and (parsed.provider, parsed.api_family) in {
        ("openai", "chat_completions"),
        ("openai", "responses"),
        ("xai", "chat_completions"),
        ("xai", "responses"),
    }


def provider_error_response(
    provider: str, api_family: str, exc: AsterError, request_id: str | None = None
) -> JSONResponse:
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
    chat_template_kwargs = _chat_template_kwargs(body.get("chat_template_kwargs"))
    enable_thinking = _openai_chat_enable_thinking(body, chat_template_kwargs)
    return _build_local_request(
        provider="openai",
        api_family="chat_completions",
        model=model or _str(body.get("model")) or "local-model",
        request_id=request_id,
        messages=messages,
        stream=bool(body.get("stream", False)),
        max_tokens=_openai_chat_max_tokens(body),
        temperature=_float(body.get("temperature"), default=0.7),
        top_p=_float(body.get("top_p"), default=0.95),
        top_k=_int(body.get("top_k"), default=0),
        min_p=_float(body.get("min_p"), default=0.0),
        presence_penalty=_float(body.get("presence_penalty"), default=0.0),
        frequency_penalty=_float(body.get("frequency_penalty"), default=0.0),
        repetition_penalty=_float(body.get("repetition_penalty"), default=1.0),
        stop=_stop_sequences(body.get("stop")),
        stop_token_ids=_stop_token_ids(body.get("stop_token_ids")),
        timeout_seconds=_positive_float(body.get("timeout")),
        enable_thinking=enable_thinking,
        chat_template_kwargs=chat_template_kwargs,
        thinking_token_budget=_positive_int(body.get("thinking_token_budget")),
        include_stream_usage=_include_stream_usage(body.get("stream_options")),
        feature_plan=feature_plan,
    )


def _parse_openai_responses(
    body: Mapping[str, Any],
    *,
    model: str | None,
    request_id: str,
    previous_response_messages: list[dict[str, Any]] | None = None,
) -> LocalProviderRequest:
    _reject_truthy(body, "background")
    current_messages = _normalize_openai_responses_input(body.get("input"))
    messages = [*list(previous_response_messages or ())]
    instructions = _str(body.get("instructions"))
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.extend(current_messages)
    feature_plan = _feature_plan_openai_responses(body)
    response_metadata = _openai_responses_metadata(body, instructions=instructions)
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
        top_k=_int(body.get("top_k"), default=0),
        min_p=_float(body.get("min_p"), default=0.0),
        presence_penalty=_float(body.get("presence_penalty"), default=0.0),
        frequency_penalty=_float(body.get("frequency_penalty"), default=0.0),
        repetition_penalty=_float(body.get("repetition_penalty"), default=1.0),
        stop=_stop_sequences(body.get("stop")),
        stop_token_ids=_stop_token_ids(body.get("stop_token_ids")),
        timeout_seconds=_positive_float(body.get("timeout")),
        response_metadata=response_metadata,
        response_replay_messages=[
            *list(previous_response_messages or ()),
            *current_messages,
        ],
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
        messages.append(
            {
                "role": _anthropic_role(item.get("role")),
                "content": _extract_anthropic_content(item.get("content")),
            }
        )
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
        top_k=_int(body.get("top_k"), default=0),
        stop=_stop_sequences(body.get("stop_sequences")),
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
        raise _unsupported(
            "gemini",
            "generate_content",
            "Gemini thinking controls are not yet supported by the local runtime.",
        )
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
        messages.append(
            {
                "role": "assistant" if _str(item.get("role")) == "model" else "user",
                "content": _extract_gemini_parts(item.get("parts")),
            }
        )
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
        top_k=_int_from_mapping(generation_config, "topK", 0),
        stop=_stop_sequences(
            generation_config.get("stopSequences")
            if isinstance(generation_config, Mapping)
            else None
        ),
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
        messages.append(
            {
                "role": _str(item.get("role")) or "user",
                "content": _extract_cohere_content(item.get("content")),
            }
        )
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
        top_k=_int(body.get("k"), default=0),
        stop=_stop_sequences(body.get("stop_sequences")),
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
            messages.append(
                {
                    "role": _str(item.get("role")) or "user",
                    "content": _extract_bedrock_content(item.get("content")),
                }
            )
    tools, tool_choice, allow_parallel = parse_bedrock_tools(body.get("toolConfig"))
    feature_plan = (
        build_tool_plan(
            tools=tools, tool_choice=tool_choice, allow_parallel_tool_calls=allow_parallel
        )
        if tools
        else FeaturePlan()
    )
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
        stop=_stop_sequences(
            inference_config.get("stopSequences") if isinstance(inference_config, Mapping) else None
        ),
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
    previous_response_messages: list[dict[str, Any]] | None = None,
) -> LocalProviderRequest:
    parsed = _parse_openai_responses(
        body,
        model=model,
        request_id=request_id,
        previous_response_messages=previous_response_messages,
    )
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
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    stop: str | list[str] | None = None,
    stop_token_ids: tuple[int, ...] = (),
    timeout_seconds: float | None = None,
    enable_thinking: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    thinking_token_budget: int | None = None,
    include_stream_usage: bool = False,
    response_metadata: dict[str, Any] | None = None,
    response_replay_messages: list[dict[str, Any]] | None = None,
) -> LocalProviderRequest:
    augmented_messages = apply_feature_plan(messages, feature_plan)
    resolved_chat_template_kwargs = dict(chat_template_kwargs or {})
    resolved_chat_template_kwargs["enable_thinking"] = enable_thinking
    response_id = _provider_response_id(
        provider=provider,
        api_family=api_family,
        request_id=request_id,
    )
    return LocalProviderRequest(
        provider=provider,
        api_family=api_family,
        model=model,
        request_id=request_id,
        stream=stream,
        feature_plan=feature_plan,
        include_stream_usage=include_stream_usage,
        response_id=response_id,
        response_metadata=dict(response_metadata or {}),
        response_replay_messages=list(response_replay_messages or ()),
        inference_request=InferenceRequest(
            messages=augmented_messages,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            stop=stop,
            stop_token_ids=stop_token_ids,
            parser_stop_sequences=_parser_stop_sequences(feature_plan),
            trace_id=request_id,
            request_aliases=(response_id,) if stream and response_id != request_id else (),
            timeout_seconds=timeout_seconds,
            enable_thinking=enable_thinking,
            chat_template_kwargs=resolved_chat_template_kwargs,
            thinking_token_budget=thinking_token_budget,
            structured_output_schema=feature_plan.structured_schema
            if feature_plan.mode == "structured"
            else None,
        ),
    )


def _provider_response_id(*, provider: str, api_family: str, request_id: str) -> str:
    if api_family == "chat_completions" and provider in {"openai", "xai", "mistral"}:
        return f"chatcmpl-{uuid.uuid4().hex[:8]}"
    if api_family == "responses" and provider in {"openai", "xai"}:
        return f"resp_{uuid.uuid4().hex}"
    return request_id


def _parser_stop_sequences(feature_plan: FeaturePlan) -> tuple[str, ...]:
    if feature_plan.mode != "tools":
        return ()
    return tuple(AutoToolParser.extra_stop_tokens)


def _openai_chat_payload(
    parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput
) -> dict[str, Any]:
    if decoded.tool_calls:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": decoded.assistant_text,
            "tool_calls": [
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=True),
                    },
                }
                for tool_call in decoded.tool_calls
            ],
        }
        if decoded.reasoning_text:
            message["reasoning_content"] = decoded.reasoning_text
        return {
            "id": parsed.response_id,
            "object": "chat.completion",
            "created": 0,
            "model": parsed.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": _openai_usage(result),
        }
    message = {"role": "assistant", "content": decoded.assistant_text or ""}
    if decoded.reasoning_text:
        message["reasoning_content"] = decoded.reasoning_text
    return {
        "id": parsed.response_id,
        "object": "chat.completion",
        "created": 0,
        "model": parsed.model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": _openai_finish_reason(result)}
        ],
        "usage": _openai_usage(result),
    }


def _openai_responses_payload(
    parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput
) -> dict[str, Any]:
    if decoded.tool_calls:
        output: list[dict[str, Any]] = []
        if decoded.reasoning_text:
            output.append(_responses_reasoning_item(parsed, decoded.reasoning_text))
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
        return _openai_responses_response(
            parsed,
            output=output,
            usage=_responses_usage(result),
            finish_reason=_openai_finish_reason(result),
        )
    output = []
    if decoded.reasoning_text:
        output.append(_responses_reasoning_item(parsed, decoded.reasoning_text))
    output.append(
        {
            "id": f"msg_{parsed.response_id}",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": decoded.assistant_text or "", "annotations": []}
            ],
        }
    )
    return _openai_responses_response(
        parsed,
        output=output,
        usage=_responses_usage(result),
        finish_reason=_openai_finish_reason(result),
    )


def _openai_responses_response(
    parsed: LocalProviderRequest,
    *,
    output: list[dict[str, Any]],
    usage: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    response = {
        "id": parsed.response_id,
        "object": "response",
        "created_at": 0,
        "model": parsed.model,
        **parsed.response_metadata,
    }
    response.update(responses_status_fields_from_finish_reason(finish_reason))
    response.update(
        {
            "output": output,
            "output_text": responses_output_text(output),
            "usage": usage,
        }
    )
    return response


def _anthropic_payload(
    parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput
) -> dict[str, Any]:
    if decoded.tool_calls:
        content: list[dict[str, Any]] = []
        if decoded.reasoning_text:
            content.append({"type": "thinking", "thinking": decoded.reasoning_text})
        content.extend(
            {
                "type": "tool_use",
                "id": tool_call.call_id,
                "name": tool_call.name,
                "input": tool_call.arguments,
            }
            for tool_call in decoded.tool_calls
        )
        return {
            "id": parsed.request_id,
            "type": "message",
            "role": "assistant",
            "model": parsed.model,
            "content": content,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
            },
        }
    content = []
    if decoded.reasoning_text:
        content.append({"type": "thinking", "thinking": decoded.reasoning_text})
    content.append({"type": "text", "text": decoded.assistant_text or ""})
    return {
        "id": parsed.request_id,
        "type": "message",
        "role": "assistant",
        "model": parsed.model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(result),
        "stop_sequence": _stop_sequence_from_result(result),
        "usage": {"input_tokens": result.prompt_tokens, "output_tokens": result.completion_tokens},
    }


def _responses_reasoning_item(parsed: LocalProviderRequest, reasoning_text: str) -> dict[str, Any]:
    return {
        "id": f"rs_{parsed.response_id}",
        "type": "reasoning",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": reasoning_text}],
    }


def _gemini_payload(
    parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput
) -> dict[str, Any]:
    parts: list[dict[str, Any]]
    if decoded.tool_calls:
        parts = [
            {"functionCall": {"name": tool_call.name, "args": tool_call.arguments}}
            for tool_call in decoded.tool_calls
        ]
    else:
        parts = [{"text": decoded.assistant_text or ""}]
    return {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": parts},
                "finishReason": _gemini_finish_reason(result),
            }
        ],
        "usageMetadata": {
            "promptTokenCount": result.prompt_tokens,
            "candidatesTokenCount": result.completion_tokens,
            "totalTokenCount": result.prompt_tokens + result.completion_tokens,
        },
        "modelVersion": parsed.model,
    }


def _cohere_payload(
    parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"type": "text", "text": decoded.assistant_text or ""}],
    }
    finish_reason = _cohere_finish_reason(result)
    if decoded.tool_calls:
        message = {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {"id": tool_call.call_id, "name": tool_call.name, "arguments": tool_call.arguments}
                for tool_call in decoded.tool_calls
            ],
        }
        finish_reason = "TOOL_CALL"
    return {
        "id": parsed.request_id,
        "message": message,
        "finish_reason": finish_reason,
        "usage": {
            "tokens": {
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
            }
        },
    }


def _bedrock_payload(
    parsed: LocalProviderRequest, result: InferenceResponse, decoded: DecodedLocalOutput
) -> dict[str, Any]:
    content = [{"text": decoded.assistant_text or ""}]
    stop_reason = _bedrock_stop_reason(result)
    if decoded.tool_calls:
        content = [
            {
                "toolUse": {
                    "toolUseId": tool_call.call_id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
            }
            for tool_call in decoded.tool_calls
        ]
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


async def _openai_chat_stream(
    parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]
) -> AsyncIterator[dict[str, str]]:
    if parsed.feature_plan.mode == "tools":
        async for event in iter_chat_tool_sse_events(
            chunks,
            parsed.model,
            response_id=parsed.response_id,
            include_usage=parsed.include_stream_usage,
            tool_request=tool_request_from_plan(parsed.feature_plan),
        ):
            yield event
        return
    if parsed.feature_plan.mode == "structured":
        async for event in iter_chat_structured_sse_events(
            chunks,
            parsed.model,
            response_id=parsed.response_id,
            include_usage=parsed.include_stream_usage,
            feature_plan=parsed.feature_plan,
        ):
            yield event
        return
    async for event in iter_chat_sse_events(
        chunks,
        parsed.model,
        response_id=parsed.response_id,
        include_usage=parsed.include_stream_usage,
    ):
        yield event


async def _openai_responses_stream(
    parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]
) -> AsyncIterator[dict[str, str]]:
    if parsed.feature_plan.mode == "tools":
        async for event in iter_responses_tool_sse_events(
            chunks,
            parsed.model,
            response_id=parsed.response_id,
            tool_request=tool_request_from_plan(parsed.feature_plan),
            response_metadata=parsed.response_metadata,
        ):
            yield event
        return
    if parsed.feature_plan.mode == "structured":
        async for event in iter_responses_structured_sse_events(
            chunks,
            parsed.model,
            response_id=parsed.response_id,
            feature_plan=parsed.feature_plan,
            response_metadata=parsed.response_metadata,
        ):
            yield event
        return
    async for event in iter_responses_sse_events(
        chunks,
        parsed.model,
        response_id=parsed.response_id,
        response_metadata=parsed.response_metadata,
    ):
        yield event


async def _anthropic_stream(
    parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]
) -> AsyncIterator[dict[str, str]]:
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
    reasoning_parser = AutoReasoningParser()
    thinking_started = False
    text_started = False
    thinking_index = 0
    text_index = 1
    async for chunk in chunks:
        if chunk.finished:
            stats = chunk.stats or {}
            if thinking_started:
                yield {
                    "event": "content_block_stop",
                    "data": json.dumps({"type": "content_block_stop", "index": thinking_index}),
                }
            if text_started:
                yield {
                    "event": "content_block_stop",
                    "data": json.dumps(
                        {
                            "type": "content_block_stop",
                            "index": text_index if thinking_started else 0,
                        }
                    ),
                }
            yield {
                "event": "message_delta",
                "data": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": _anthropic_stop_reason_from_stats(stats),
                            "stop_sequence": _stop_sequence_from_stats(stats),
                        },
                        "usage": {
                            "input_tokens": _int(stats.get("prompt_tokens"), default=0),
                            "output_tokens": _int(stats.get("completion_tokens"), default=0),
                        },
                    }
                ),
            }
            yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}
            break
        delta = reasoning_parser.parse_delta(chunk.token)
        if delta.reasoning_delta:
            if not thinking_started:
                thinking_started = True
                yield {
                    "event": "content_block_start",
                    "data": json.dumps(
                        {
                            "type": "content_block_start",
                            "index": thinking_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }
                    ),
                }
            yield {
                "event": "content_block_delta",
                "data": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": thinking_index,
                        "delta": {"type": "thinking_delta", "thinking": delta.reasoning_delta},
                    }
                ),
            }
        if delta.content_delta:
            current_text_index = text_index if thinking_started else 0
            if not text_started:
                text_started = True
                yield {
                    "event": "content_block_start",
                    "data": json.dumps(
                        {
                            "type": "content_block_start",
                            "index": current_text_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                    ),
                }
            yield {
                "event": "content_block_delta",
                "data": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": current_text_index,
                        "delta": {"type": "text_delta", "text": delta.content_delta},
                    }
                ),
            }


async def _gemini_stream(
    parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]
) -> AsyncIterator[dict[str, str]]:
    async for chunk in chunks:
        if chunk.finished:
            stats = chunk.stats or {}
            yield {
                "data": json.dumps(
                    {
                        "candidates": [
                            {"index": 0, "finishReason": _gemini_finish_reason_from_stats(stats)}
                        ],
                        "usageMetadata": {
                            "promptTokenCount": _int(stats.get("prompt_tokens"), default=0),
                            "candidatesTokenCount": _int(stats.get("completion_tokens"), default=0),
                            "totalTokenCount": _int(stats.get("prompt_tokens"), default=0)
                            + _int(stats.get("completion_tokens"), default=0),
                        },
                    }
                )
            }
            break
        yield {
            "data": json.dumps(
                {
                    "candidates": [
                        {"index": 0, "content": {"role": "model", "parts": [{"text": chunk.token}]}}
                    ]
                }
            )
        }


async def _cohere_stream(
    parsed: LocalProviderRequest, chunks: AsyncIterator[DecodeChunk]
) -> AsyncIterator[dict[str, str]]:
    yield {"data": json.dumps({"type": "message-start", "id": parsed.request_id})}
    async for chunk in chunks:
        if chunk.finished:
            yield {
                "data": json.dumps(
                    {
                        "type": "message-end",
                        "finish_reason": _cohere_finish_reason_from_stats(chunk.stats or {}),
                    }
                )
            }
            break
        yield {
            "data": json.dumps(
                {"type": "content-delta", "delta": {"message": {"content": {"text": chunk.token}}}}
            )
        }


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
    return _build_feature_plan(
        tools=tools, tool_choice=tool_choice, parallel=parallel, structured=structured
    )


def _openai_responses_metadata(
    body: Mapping[str, Any], *, instructions: str | None
) -> dict[str, Any]:
    raw_max_output_tokens = body.get("max_output_tokens")
    max_output_tokens = (
        raw_max_output_tokens
        if isinstance(raw_max_output_tokens, int) and not isinstance(raw_max_output_tokens, bool)
        else None
    )
    raw_metadata = body.get("metadata")
    metadata = (
        dict(cast(Mapping[str, Any], raw_metadata)) if isinstance(raw_metadata, Mapping) else {}
    )
    raw_text = body.get("text")
    text = (
        dict(cast(Mapping[str, Any], raw_text))
        if isinstance(raw_text, Mapping)
        else {"format": {"type": "text"}}
    )
    raw_store = body.get("store")
    return {
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": instructions,
        "max_output_tokens": max_output_tokens,
        "max_tool_calls": None,
        "metadata": metadata,
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "previous_response_id": _str(body.get("previous_response_id")),
        "text": text,
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": _list(body.get("tools")),
        "top_p": _float(body.get("top_p"), default=0.95),
        "temperature": _float(body.get("temperature"), default=0.7),
        "truncation": _str(body.get("truncation")) or "disabled",
        "user": _str(body.get("user")),
        "store": raw_store if isinstance(raw_store, bool) else True,
    }


def _feature_plan_gemini(body: Mapping[str, Any]) -> FeaturePlan:
    tools, _, _ = parse_gemini_tools(body.get("tools"))
    tool_choice = parse_gemini_tool_config(body.get("toolConfig"))
    generation_config = body.get("generationConfig")
    structured: tuple[dict[str, Any], str | None] | None = None
    if isinstance(generation_config, Mapping) and (
        generation_config.get("responseMimeType") == "application/json"
        or isinstance(generation_config.get("responseSchema"), Mapping)
    ):
        if isinstance(generation_config.get("responseSchema"), Mapping):
            structured = parse_gemini_structured_schema(generation_config)
        else:
            structured = ({"type": "object"}, None)
    return _build_feature_plan(
        tools=tools, tool_choice=tool_choice, parallel=True, structured=structured
    )


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
    return _build_feature_plan(
        tools=tools, tool_choice=tool_choice, parallel=True, structured=structured
    )


def _openai_response_format_schema(
    response_format: Any,
) -> tuple[dict[str, Any], str | None] | None:
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
        raise AsterError(
            code="feature_combination_unsupported",
            message="Combining tools and structured outputs is not yet supported by the local runtime.",
            status_code=400,
        )
    if tools:
        return build_tool_plan(
            tools=tools, tool_choice=tool_choice, allow_parallel_tool_calls=parallel
        )
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


def _normalize_openai_responses_input(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages: list[dict[str, Any]] = []
    for item in _list(value):
        if not isinstance(item, Mapping):
            continue
        item_type = _str(item.get("type"))
        if item_type == "function_call":
            call_id = _str(item.get("call_id")) or f"call_{uuid.uuid4().hex}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": _str(item.get("name")) or "",
                                "arguments": _str(item.get("arguments")) or "",
                            },
                        }
                    ],
                }
            )
            continue
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _str(item.get("call_id")) or "",
                    "content": _str(item.get("output")) or "",
                }
            )
            continue
        if item_type == "reasoning":
            reasoning_text = _extract_responses_reasoning_content(item.get("content"))
            if reasoning_text:
                messages.append({"role": "assistant", "content": reasoning_text})
            continue
        role = _str(item.get("role")) or "user"
        if role == "developer":
            role = "system"
        messages.append(
            {
                "role": role,
                "content": _extract_openai_content(item.get("content")),
            }
        )
    return messages


def _extract_responses_reasoning_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for part in _list(content):
        if isinstance(part, Mapping):
            text = _str(part.get("text")) or _str(part.get("summary_text"))
            if text:
                fragments.append(text)
    return "\n".join(fragments).strip()


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
            text = (
                _str(part.get("text")) or _str(part.get("input_text")) or _str(part.get("content"))
            )
            if text:
                fragments.append(text)
            continue
        if is_multimodal_content_type(part_type):
            raise AsterError(
                code="multimodal_not_supported",
                message=(
                    "Multimodal content must be handled by the media request model; "
                    "the local text runtime no longer flattens media into placeholders."
                ),
                status_code=400,
            )
        raise _unsupported(
            "openai",
            "content",
            f"Non-text content part '{part_type}' is not yet supported by the local runtime.",
        )
    return "\n".join(fragment for fragment in fragments if fragment).strip()


def _extract_anthropic_system(content: Any) -> str:
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for item in _list(content):
        if (
            isinstance(item, Mapping)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
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
        raise _unsupported(
            "anthropic",
            "messages",
            f"Anthropic content block '{item.get('type')}' is not yet supported by the local runtime.",
        )
    return "\n".join(fragments).strip()


def _extract_gemini_parts(parts: Any) -> str:
    fragments: list[str] = []
    for item in _list(parts):
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            fragments.append(item["text"])
            continue
        raise _unsupported(
            "gemini",
            "generate_content",
            "Only text Gemini parts are currently supported by the local runtime.",
        )
    return "\n".join(fragments).strip()


def _extract_cohere_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for item in _list(content):
        if (
            isinstance(item, Mapping)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            fragments.append(item["text"])
            continue
        raise _unsupported(
            "cohere",
            "chat_v2",
            "Only text Cohere content parts are currently supported by the local runtime.",
        )
    return "\n".join(fragments).strip()


def _extract_bedrock_content(content: Any) -> str:
    fragments: list[str] = []
    for item in _list(content):
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            fragments.append(item["text"])
            continue
        raise _unsupported(
            "bedrock",
            "converse",
            "Only text Bedrock content blocks are currently supported by the local runtime.",
        )
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


def _responses_usage(result: InferenceResponse) -> dict[str, Any]:
    return responses_usage_payload(result.prompt_tokens, result.completion_tokens)


def _openai_finish_reason(result: InferenceResponse) -> str:
    return _finish_reason_value(getattr(result, "finish_reason", None))


def _anthropic_stop_reason(result: InferenceResponse) -> str:
    return _anthropic_stop_reason_value(_openai_finish_reason(result))


def _gemini_finish_reason(result: InferenceResponse) -> str:
    return _gemini_finish_reason_value(_openai_finish_reason(result))


def _cohere_finish_reason(result: InferenceResponse) -> str:
    return _cohere_finish_reason_value(_openai_finish_reason(result))


def _bedrock_stop_reason(result: InferenceResponse) -> str:
    return _bedrock_stop_reason_value(_openai_finish_reason(result))


def _finish_reason_from_stats(stats: Mapping[str, Any]) -> str:
    return _finish_reason_value(stats.get("finish_reason"))


def _finish_reason_value(value: Any) -> str:
    return value if isinstance(value, str) and value else "stop"


def _stop_sequence_from_result(result: InferenceResponse) -> str | None:
    stop_sequence = getattr(result, "stop_sequence", None)
    return stop_sequence if isinstance(stop_sequence, str) and stop_sequence else None


def _stop_sequence_from_stats(stats: Mapping[str, Any]) -> str | None:
    stop_sequence = stats.get("stop_sequence")
    return stop_sequence if isinstance(stop_sequence, str) and stop_sequence else None


def _anthropic_stop_reason_from_stats(stats: Mapping[str, Any]) -> str:
    return _anthropic_stop_reason_value(_finish_reason_from_stats(stats))


def _anthropic_stop_reason_value(finish_reason: str) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "stop_sequence":
        return "stop_sequence"
    if finish_reason in {"content_filter", "safety", "refusal"}:
        return "refusal"
    return "end_turn"


def _gemini_finish_reason_from_stats(stats: Mapping[str, Any]) -> str:
    return _gemini_finish_reason_value(_finish_reason_from_stats(stats))


def _gemini_finish_reason_value(finish_reason: str) -> str:
    if finish_reason == "length":
        return "MAX_TOKENS"
    if finish_reason in {"content_filter", "safety"}:
        return "SAFETY"
    if finish_reason == "recitation":
        return "RECITATION"
    if finish_reason == "blocklist":
        return "BLOCKLIST"
    if finish_reason == "prohibited_content":
        return "PROHIBITED_CONTENT"
    if finish_reason == "spii":
        return "SPII"
    if finish_reason == "malformed_function_call":
        return "MALFORMED_FUNCTION_CALL"
    if finish_reason in {"stop", "stop_sequence", "tool_calls"}:
        return "STOP"
    return "OTHER"


def _cohere_finish_reason_from_stats(stats: Mapping[str, Any]) -> str:
    return _cohere_finish_reason_value(_finish_reason_from_stats(stats))


def _cohere_finish_reason_value(finish_reason: str) -> str:
    if finish_reason == "length":
        return "MAX_TOKENS"
    if finish_reason == "tool_calls":
        return "TOOL_CALL"
    if finish_reason == "stop_sequence":
        return "STOP_SEQUENCE"
    if finish_reason in {"content_filter", "safety"}:
        return "ERROR"
    return "COMPLETE"


def _bedrock_stop_reason_value(finish_reason: str) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "stop_sequence":
        return "stop_sequence"
    if finish_reason == "content_filter":
        return "content_filtered"
    if finish_reason in {"safety", "guardrail_intervened"}:
        return "guardrail_intervened"
    return "end_turn"


def _reject_truthy(body: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        value = body.get(key)
        if value in (None, False, "", [], {}):
            continue
        raise _unsupported(
            "local", key, f"Field '{key}' is not yet supported by the local runtime."
        )


def _unsupported(provider: str, api_family: str, message: str) -> AsterError:
    return AsterError(code=f"{provider}_{api_family}_unsupported", message=message, status_code=400)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _stop_sequences(value: Any) -> str | list[str] | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        stops = [item for item in value if isinstance(item, str)]
        return stops or None
    return None


def _stop_token_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))


def _include_stream_usage(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("include_usage") is True


def _int(value: Any, *, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _openai_chat_max_tokens(body: Mapping[str, Any]) -> int:
    if body.get("max_tokens") is not None:
        return _int(body.get("max_tokens"), default=256)
    return _int(body.get("max_completion_tokens"), default=256)


def _chat_template_kwargs(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _openai_chat_enable_thinking(
    body: Mapping[str, Any], chat_template_kwargs: Mapping[str, Any]
) -> bool:
    requested = body.get("enable_thinking")
    if isinstance(requested, bool):
        return requested
    template_requested = chat_template_kwargs.get("enable_thinking")
    return template_requested if isinstance(template_requested, bool) else False


def _float(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _positive_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


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
