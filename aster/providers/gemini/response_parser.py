from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalFinalResponse,
    CanonicalMessage,
    MessageRole,
    ModelRef,
    ProviderName,
    ProviderRef,
    RawProviderEnvelope,
)
from aster.providers._shared import text_part
from aster.providers.gemini.extensions import response_extensions
from aster.providers.gemini.tool_mapper import parse_function_call
from aster.providers.openai.response_parser import parse_chat_response


def parse_generate_content_response(
    provider: ProviderRef,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> CanonicalFinalResponse:
    candidates = payload.get("candidates", [])
    first_candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
    if not isinstance(first_candidate, Mapping):
        first_candidate = {}
    content = first_candidate.get("content", {})
    if not isinstance(content, Mapping):
        content = {}
    content_parts = []
    tool_calls = []
    text_fragments: list[str] = []
    for part in content.get("parts", []):
        if not isinstance(part, Mapping):
            continue
        if isinstance(part.get("text"), str):
            text_fragments.append(part["text"])
            content_parts.append(text_part(part["text"], provider_extensions=dict(part)))
        elif "functionCall" in part:
            tool_calls.append(parse_function_call(part))
    usage_payload = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), Mapping) else {}
    usage = {
        "input_tokens": usage_payload.get("promptTokenCount"),
        "output_tokens": usage_payload.get("candidatesTokenCount"),
        "total_tokens": usage_payload.get("totalTokenCount"),
        "provider_extensions": {"values": {key: value for key, value in usage_payload.items() if key not in {"promptTokenCount", "candidatesTokenCount", "totalTokenCount"}}},
    }
    model_id = payload.get("modelVersion") if isinstance(payload.get("modelVersion"), str) else ""
    return CanonicalFinalResponse(
        provider=provider,
        model=ModelRef.from_values(provider.name, provider.api_family, model_id),
        response_id=model_id or "gemini-response",
        status="completed",
        output=[CanonicalMessage(role=MessageRole.ASSISTANT, content=content_parts, tool_calls=tool_calls)],
        output_text="".join(text_fragments),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=first_candidate.get("finishReason") if isinstance(first_candidate.get("finishReason"), str) else None,
        provider_extensions=response_extensions(payload),
        raw_provider_envelope=RawProviderEnvelope(
            provider=ProviderName.GEMINI,
            api_family=provider.api_family,
            headers=dict(headers or {}),
            payload=dict(payload),
        ),
    )


def parse_openai_chat_response(
    provider: ProviderRef,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> CanonicalFinalResponse:
    return parse_chat_response(provider, payload, headers=headers)
