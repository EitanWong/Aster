from __future__ import annotations

from typing import Any

from aster.core.canonical import (
    CanonicalRequest,
    ContentPartType,
    MessageRole,
    ProviderExtensionData,
    ProviderHttpRequest,
    ProviderRef,
    ProviderRequestContext,
)
from aster.core.provider_errors import UnsupportedProviderFeatureError
from aster.providers._shared import join_text_parts
from aster.providers.openai.tool_mapper import build_chat_tools, build_responses_tools


def build_responses_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    body: dict[str, Any] = {
        "model": request.model.model_id,
        "input": [_message_to_responses_input(message) for message in request.messages],
        "stream": request.stream,
    }
    if request.tools:
        body["tools"] = build_responses_tools(request.tools)
    if request.max_output_tokens is not None:
        body["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop"] = request.stop
    if request.store is not None:
        body["store"] = request.store
    if request.background is not None:
        body["background"] = request.background
    if request.previous_response_id:
        body["previous_response_id"] = request.previous_response_id
    if request.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = request.parallel_tool_calls
    if request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    if request.metadata:
        body["metadata"] = request.metadata
    if request.reasoning:
        body["reasoning"] = request.reasoning
    if request.structured_output_schema is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.structured_output_name or "structured_output",
                "schema": request.structured_output_schema,
            }
        }
    body.update(request.provider_options)
    headers = {
        "Authorization": f"Bearer {context.api_key}" if context and context.api_key else "",
        "Content-Type": "application/json",
    }
    return ProviderHttpRequest(
        method="POST",
        path="/v1/responses",
        headers=_clean_headers(headers, context),
        json_body=body,
        provider_extensions=ProviderExtensionData(values={"provider": provider.name.value}),
    )


def build_chat_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    if request.previous_response_id:
        raise UnsupportedProviderFeatureError.build(
            provider=provider,
            category="unsupported_feature",
            code="openai_chat_no_previous_response_id",
            message="OpenAI Chat Completions does not support previous_response_id response chaining.",
        )
    if request.background:
        raise UnsupportedProviderFeatureError.build(
            provider=provider,
            category="unsupported_feature",
            code="openai_chat_no_background_mode",
            message="OpenAI Chat Completions does not provide Responses-style background execution.",
        )
    body: dict[str, Any] = {
        "model": request.model.model_id,
        "messages": [_message_to_chat_message(message) for message in request.messages],
        "stream": request.stream,
    }
    if request.tools:
        body["tools"] = build_chat_tools(request.tools)
    if request.max_output_tokens is not None:
        body["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop"] = request.stop
    if request.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = request.parallel_tool_calls
    if request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    if request.metadata:
        body["metadata"] = request.metadata
    if request.structured_output_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.structured_output_name or "structured_output",
                "schema": request.structured_output_schema,
            },
        }
    body.update(request.provider_options)
    headers = {
        "Authorization": f"Bearer {context.api_key}" if context and context.api_key else "",
        "Content-Type": "application/json",
    }
    return ProviderHttpRequest(
        method="POST",
        path="/v1/chat/completions",
        headers=_clean_headers(headers, context),
        json_body=body,
        provider_extensions=ProviderExtensionData(values={"provider": provider.name.value}),
    )


def _message_to_responses_input(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": [_content_part_to_responses_part(part) for part in message.content],
    }
    if message.role is MessageRole.TOOL and "tool_call_id" in message.provider_extensions.values:
        payload["tool_call_id"] = message.provider_extensions.values["tool_call_id"]
    return payload


def _message_to_chat_message(message: Any) -> dict[str, Any]:
    if all(part.type in {ContentPartType.TEXT, ContentPartType.INPUT_TEXT, ContentPartType.OUTPUT_TEXT} for part in message.content):
        content: str | list[dict[str, Any]] = join_text_parts(message.content)
    else:
        content = [_content_part_to_chat_part(part) for part in message.content]
    payload: dict[str, Any] = {"role": message.role.value, "content": content}
    if message.name:
        payload["name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments_json or "",
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.role is MessageRole.TOOL and "tool_call_id" in message.provider_extensions.values:
        payload["tool_call_id"] = message.provider_extensions.values["tool_call_id"]
    return payload


def _content_part_to_responses_part(part: Any) -> dict[str, Any]:
    if part.type in {ContentPartType.TEXT, ContentPartType.INPUT_TEXT, ContentPartType.OUTPUT_TEXT}:
        return {"type": "input_text", "text": part.text or ""}
    if part.type is ContentPartType.INPUT_IMAGE:
        if part.image_url:
            return {"type": "input_image", "image_url": part.image_url}
        if "openai_native" in part.provider_extensions.values:
            return dict(part.provider_extensions.values["openai_native"])
    if part.type is ContentPartType.INPUT_AUDIO and part.data and part.audio_format:
        return {"type": "input_audio", "input_audio": {"data": part.data, "format": part.audio_format}}
    if part.type is ContentPartType.FILE and part.file_id:
        return {"type": "input_file", "file_id": part.file_id}
    if part.type is ContentPartType.JSON:
        return {"type": "input_text", "text": str(part.json_value)}
    if "openai_native" in part.provider_extensions.values:
        return dict(part.provider_extensions.values["openai_native"])
    raise UnsupportedProviderFeatureError.build(
        provider=None,
        category="unsupported_feature",
        code="openai_responses_content_part_unsupported",
        message=f"OpenAI Responses content part '{part.type.value}' requires provider-native extension data.",
    )


def _content_part_to_chat_part(part: Any) -> dict[str, Any]:
    if part.type in {ContentPartType.TEXT, ContentPartType.INPUT_TEXT, ContentPartType.OUTPUT_TEXT}:
        return {"type": "text", "text": part.text or ""}
    if part.type is ContentPartType.INPUT_IMAGE and part.image_url:
        return {"type": "image_url", "image_url": {"url": part.image_url}}
    if "openai_native" in part.provider_extensions.values:
        return dict(part.provider_extensions.values["openai_native"])
    raise UnsupportedProviderFeatureError.build(
        provider=None,
        category="unsupported_feature",
        code="openai_chat_content_part_unsupported",
        message=f"OpenAI Chat content part '{part.type.value}' requires provider-native extension data.",
    )


def _clean_headers(headers: dict[str, str], context: ProviderRequestContext | None) -> dict[str, str]:
    merged = dict(headers)
    if context is not None:
        merged.update(context.extra_headers)
    return {key: value for key, value in merged.items() if value}
