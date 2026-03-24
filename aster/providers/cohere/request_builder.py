from __future__ import annotations

from typing import Any

from aster.core.canonical import CanonicalRequest, ContentPartType, MessageRole, ProviderHttpRequest, ProviderRef, ProviderRequestContext
from aster.core.provider_errors import UnsupportedProviderFeatureError
from aster.providers.cohere.tool_mapper import build_tools


def build_chat_request(provider: ProviderRef, request: CanonicalRequest, context: ProviderRequestContext | None) -> ProviderHttpRequest:
    body: dict[str, Any] = {
        "model": request.model.model_id,
        "messages": [_message_to_cohere(message) for message in request.messages],
        "stream": request.stream,
    }
    if request.tools:
        body["tools"] = build_tools(request.tools)
    if request.max_output_tokens is not None:
        body["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["p"] = request.top_p
    if request.stop is not None:
        body["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if request.structured_output_schema is not None:
        body["response_format"] = {
            "type": "json_object",
            "json_schema": request.structured_output_schema,
        }
    if request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    body.update(request.provider_options)
    headers = {
        "Authorization": f"Bearer {context.api_key}" if context and context.api_key else "",
        "Content-Type": "application/json",
    }
    if context:
        headers.update(context.extra_headers)
    return ProviderHttpRequest(method="POST", path="/v2/chat", headers={key: value for key, value in headers.items() if value}, json_body=body)


def _message_to_cohere(message: Any) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in message.content:
        if part.type in {ContentPartType.TEXT, ContentPartType.INPUT_TEXT, ContentPartType.OUTPUT_TEXT}:
            content.append({"type": "text", "text": part.text or ""})
            continue
        if "cohere_native" in part.provider_extensions.values:
            content.append(dict(part.provider_extensions.values["cohere_native"]))
            continue
        raise UnsupportedProviderFeatureError.build(
            provider=None,
            category="unsupported_feature",
            code="cohere_content_part_unsupported",
            message=f"Cohere content part '{part.type.value}' requires cohere_native extension data.",
        )
    payload = {"role": message.role.value, "content": content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments or {},
            }
            for tool_call in message.tool_calls
        ]
    if message.role is MessageRole.TOOL or message.tool_results:
        payload["role"] = "tool"
        payload["tool_results"] = [
            {
                "call_id": result.call_id,
                "outputs": [{"text": str(result.output)}],
                "is_error": result.is_error,
            }
            for result in message.tool_results
        ]
    return payload
