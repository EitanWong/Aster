from __future__ import annotations

from typing import Any

from aster.core.canonical import CanonicalRequest, ProviderHttpRequest, ProviderRef, ProviderRequestContext
from aster.core.provider_errors import UnsupportedProviderFeatureError
from aster.providers.mistral.tool_mapper import build_tools, normalize_tool_choice
from aster.providers.openai.request_builder import build_chat_request


def build_chat_completions_request(provider: ProviderRef, request: CanonicalRequest, context: ProviderRequestContext | None) -> ProviderHttpRequest:
    if request.previous_response_id:
        raise UnsupportedProviderFeatureError.build(
            provider=provider,
            category="unsupported_feature",
            code="mistral_chat_previous_response_unsupported",
            message="Mistral chat/completions does not use Responses-style previous_response_id chaining.",
        )
    http_request = build_chat_request(provider, request, context)
    http_request.path = "/v1/chat/completions"
    if request.tools:
        http_request.json_body["tools"] = build_tools(request.tools)
    if request.tool_choice is not None:
        http_request.json_body["tool_choice"] = normalize_tool_choice(request.tool_choice)
    return http_request


def build_conversations_request(provider: ProviderRef, request: CanonicalRequest, context: ProviderRequestContext | None) -> ProviderHttpRequest:
    body: dict[str, Any] = {
        "model": request.model.model_id,
        "messages": build_chat_completions_request(provider, request, context).json_body["messages"],
    }
    if request.conversation_id:
        body["conversation_id"] = request.conversation_id
    if request.tools:
        body["tools"] = build_tools(request.tools)
    if request.max_output_tokens is not None:
        body["max_tokens"] = request.max_output_tokens
    if request.stream:
        body["stream"] = True
    body.update(request.provider_options)
    headers = {
        "Authorization": f"Bearer {context.api_key}" if context and context.api_key else "",
        "Content-Type": "application/json",
    }
    if context:
        headers.update(context.extra_headers)
    return ProviderHttpRequest(method="POST", path="/v1/conversations", headers={key: value for key, value in headers.items() if value}, json_body=body)
