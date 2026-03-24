from __future__ import annotations

from aster.core.canonical import CanonicalRequest, ProviderHttpRequest, ProviderRef, ProviderRequestContext
from aster.providers.openai.request_builder import build_chat_request as build_openai_chat_request
from aster.providers.openai.request_builder import build_responses_request as build_openai_responses_request
from aster.providers.xai.tool_mapper import build_chat_tools, build_responses_tools


def build_responses_request(provider: ProviderRef, request: CanonicalRequest, context: ProviderRequestContext | None) -> ProviderHttpRequest:
    http_request = build_openai_responses_request(provider, request, context)
    http_request.path = "/v1/responses"
    if request.tools:
        http_request.json_body["tools"] = build_responses_tools(request.tools)
    http_request.json_body.update(request.provider_options)
    return http_request


def build_chat_request(provider: ProviderRef, request: CanonicalRequest, context: ProviderRequestContext | None) -> ProviderHttpRequest:
    http_request = build_openai_chat_request(provider, request, context)
    http_request.path = "/v1/chat/completions"
    if request.tools:
        http_request.json_body["tools"] = build_chat_tools(request.tools)
    if request.store is not None:
        http_request.json_body["store"] = request.store
    if request.reasoning:
        http_request.json_body["reasoning"] = request.reasoning
    http_request.json_body.update(request.provider_options)
    return http_request
