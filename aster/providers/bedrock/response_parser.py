from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalFinalResponse, CanonicalMessage, MessageRole, ModelRef, ProviderName, ProviderRef, RawProviderEnvelope
from aster.providers._shared import text_part
from aster.providers.bedrock.extensions import converse_extensions
from aster.providers.bedrock.tool_mapper import parse_tool_use
from aster.providers.openai.response_parser import parse_chat_response, parse_responses_response


def parse_converse_response(
    provider: ProviderRef,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> CanonicalFinalResponse:
    output = payload.get("output", {})
    if not isinstance(output, Mapping):
        output = {}
    message = output.get("message", {})
    if not isinstance(message, Mapping):
        message = {}
    content_parts = []
    tool_calls = []
    text_fragments: list[str] = []
    for block in message.get("content", []):
        if not isinstance(block, Mapping):
            continue
        if isinstance(block.get("text"), str):
            text_fragments.append(block["text"])
            content_parts.append(text_part(block["text"], provider_extensions=dict(block)))
        elif "toolUse" in block:
            tool_calls.append(parse_tool_use(block))
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    return CanonicalFinalResponse(
        provider=provider,
        model=ModelRef.from_values(provider.name, provider.api_family, _model_id(headers)),
        response_id=_request_id(headers) or "bedrock-converse",
        status="completed",
        output=[CanonicalMessage(role=MessageRole(message.get("role", "assistant")), content=content_parts, tool_calls=tool_calls)],
        output_text="".join(text_fragments),
        tool_calls=tool_calls,
        usage={
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
        },
        finish_reason=payload.get("stopReason") if isinstance(payload.get("stopReason"), str) else None,
        request_id=_request_id(headers),
        provider_extensions=converse_extensions(payload).merged({"metrics": payload.get("metrics")}),
        raw_provider_envelope=RawProviderEnvelope(
            provider=ProviderName.BEDROCK,
            api_family=provider.api_family,
            headers=dict(headers or {}),
            payload=dict(payload),
        ),
    )


def parse_openai_chat_response(provider: ProviderRef, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
    return parse_chat_response(provider, payload, headers=headers)


def parse_openai_responses_response(provider: ProviderRef, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
    return parse_responses_response(provider, payload, headers=headers)


def _request_id(headers: Mapping[str, str] | None) -> str | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    return lowered.get("x-amzn-requestid")


def _model_id(headers: Mapping[str, str] | None) -> str:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    return lowered.get("x-amzn-bedrock-model-id", "")
