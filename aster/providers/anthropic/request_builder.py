from __future__ import annotations

from typing import Any

from aster.core.canonical import (
    CanonicalRequest,
    ContentPartType,
    MessageRole,
    ProviderHttpRequest,
    ProviderRef,
    ProviderRequestContext,
)
from aster.core.provider_errors import UnsupportedProviderFeatureError
from aster.providers._shared import non_system_messages, system_message_text
from aster.providers.anthropic.tool_mapper import build_tools, tool_result_block


def build_messages_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    if request.structured_output_schema is not None:
        raise UnsupportedProviderFeatureError.build(
            provider=provider,
            category="unsupported_feature",
            code="anthropic_messages_structured_output_unsupported",
            message="Anthropic Messages does not offer a direct lossless equivalent of canonical JSON-schema structured output.",
        )
    if request.background:
        raise UnsupportedProviderFeatureError.build(
            provider=provider,
            category="unsupported_feature",
            code="anthropic_messages_background_unsupported",
            message="Anthropic Messages does not support OpenAI-style background execution.",
        )
    body: dict[str, Any] = {
        "model": request.model.model_id,
        "max_tokens": request.max_output_tokens or 1024,
        "messages": [_message_to_anthropic(message) for message in non_system_messages(request.messages)],
        "stream": request.stream,
    }
    system_prompt = system_message_text(request.messages)
    if system_prompt:
        body["system"] = system_prompt
    if request.tools:
        body["tools"] = build_tools(request.tools)
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if request.tool_choice is not None:
        body["tool_choice"] = _tool_choice(request.tool_choice)
    if request.metadata:
        body["metadata"] = request.metadata
    body.update(request.provider_options)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": context.api_key if context and context.api_key else "",
        "anthropic-version": context.anthropic_version if context and context.anthropic_version else "2023-06-01",
    }
    if context and context.anthropic_beta:
        headers["anthropic-beta"] = ",".join(context.anthropic_beta)
    if context:
        headers.update(context.extra_headers)
    return ProviderHttpRequest(
        method="POST",
        path="/v1/messages",
        headers={key: value for key, value in headers.items() if value},
        json_body=body,
    )


def _message_to_anthropic(message: Any) -> dict[str, Any]:
    if message.role is MessageRole.TOOL or message.tool_results:
        blocks = [tool_result_block(result) for result in message.tool_results]
        if not blocks and message.provider_extensions.values.get("tool_use_id"):
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.provider_extensions.values["tool_use_id"],
                    "content": "\n".join(part.text or "" for part in message.content).strip(),
                    "is_error": message.provider_extensions.values.get("is_error", False),
                }
            ]
        return {"role": "user", "content": blocks}
    content = []
    for part in message.content:
        if part.type in {ContentPartType.TEXT, ContentPartType.INPUT_TEXT, ContentPartType.OUTPUT_TEXT}:
            content.append({"type": "text", "text": part.text or ""})
            continue
        if part.type is ContentPartType.INPUT_IMAGE and "anthropic_native" in part.provider_extensions.values:
            content.append(dict(part.provider_extensions.values["anthropic_native"]))
            continue
        if "anthropic_native" in part.provider_extensions.values:
            content.append(dict(part.provider_extensions.values["anthropic_native"]))
            continue
        raise UnsupportedProviderFeatureError.build(
            provider=None,
            category="unsupported_feature",
            code="anthropic_content_part_unsupported",
            message=f"Anthropic content part '{part.type.value}' requires anthropic_native extension data.",
        )
    for tool_call in message.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.call_id,
                "name": tool_call.name,
                "input": tool_call.arguments or {},
            }
        )
    role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
    return {"role": role, "content": content}


def _tool_choice(choice: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(choice, dict):
        return choice
    return {"type": choice}
