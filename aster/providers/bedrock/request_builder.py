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
from aster.providers.bedrock.tool_mapper import build_tool_config, tool_result_block
from aster.providers.openai.request_builder import build_chat_request, build_responses_request


def build_converse_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    if request.structured_output_schema is not None:
        raise UnsupportedProviderFeatureError.build(
            provider=provider,
            category="unsupported_feature",
            code="bedrock_converse_structured_output_requires_native_model_fields",
            message="Bedrock Converse structured output must be expressed through provider-specific model request fields.",
        )
    body: dict[str, Any] = {
        "messages": [_message_to_bedrock(message) for message in non_system_messages(request.messages)],
    }
    system_prompt = system_message_text(request.messages)
    if system_prompt:
        body["system"] = [{"text": system_prompt}]
    inference_config: dict[str, Any] = {}
    if request.max_output_tokens is not None:
        inference_config["maxTokens"] = request.max_output_tokens
    if request.temperature is not None:
        inference_config["temperature"] = request.temperature
    if request.top_p is not None:
        inference_config["topP"] = request.top_p
    if request.stop is not None:
        inference_config["stopSequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if inference_config:
        body["inferenceConfig"] = inference_config
    if request.tools:
        body["toolConfig"] = build_tool_config(request.tools, request.tool_choice)
    if request.metadata:
        body["requestMetadata"] = request.metadata
    body.update(request.provider_options)
    path = f"/model/{request.model.model_id}/{'converse-stream' if request.stream else 'converse'}"
    headers = {"Content-Type": "application/json"}
    if context:
        headers.update(context.extra_headers)
    return ProviderHttpRequest(
        method="POST",
        path=path,
        headers=headers,
        json_body=body,
        requires_sigv4=True,
    )


def build_openai_chat_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    http_request = build_chat_request(provider, request, context)
    http_request.path = "/openai/v1/chat/completions"
    http_request.requires_sigv4 = True
    http_request.headers.pop("Authorization", None)
    return http_request


def build_openai_responses_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    http_request = build_responses_request(provider, request, context)
    http_request.path = "/openai/v1/responses"
    http_request.requires_sigv4 = True
    http_request.headers.pop("Authorization", None)
    return http_request


def _message_to_bedrock(message: Any) -> dict[str, Any]:
    if message.role is MessageRole.TOOL or message.tool_results:
        return {"role": "user", "content": [tool_result_block(result) for result in message.tool_results]}
    content: list[dict[str, Any]] = []
    for part in message.content:
        if part.type in {ContentPartType.TEXT, ContentPartType.INPUT_TEXT, ContentPartType.OUTPUT_TEXT}:
            content.append({"text": part.text or ""})
            continue
        if "bedrock_native" in part.provider_extensions.values:
            content.append(dict(part.provider_extensions.values["bedrock_native"]))
            continue
        raise UnsupportedProviderFeatureError.build(
            provider=None,
            category="unsupported_feature",
            code="bedrock_content_part_unsupported",
            message=f"Bedrock content part '{part.type.value}' requires bedrock_native extension data.",
        )
    for tool_call in message.tool_calls:
        content.append(
            {
                "toolUse": {
                    "toolUseId": tool_call.call_id,
                    "name": tool_call.name,
                    "input": tool_call.arguments or {},
                }
            }
        )
    role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
    return {"role": role, "content": content}
