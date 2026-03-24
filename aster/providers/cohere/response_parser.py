from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalFinalResponse, CanonicalMessage, MessageRole, ModelRef, ProviderName, ProviderRef, RawProviderEnvelope
from aster.providers._shared import function_call_to_canonical, text_part
from aster.providers.cohere.extensions import chat_extensions


def parse_chat_response(provider: ProviderRef, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
    message = payload.get("message", {})
    if not isinstance(message, Mapping):
        message = {}
    content = message.get("content", [])
    parts = []
    output_text = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                output_text.append(item["text"])
                parts.append(text_part(item["text"], provider_extensions=dict(item)))
    tool_calls = []
    if isinstance(message.get("tool_calls"), list):
        for item in message["tool_calls"]:
            if isinstance(item, Mapping):
                tool_calls.append(
                    function_call_to_canonical(
                        call_id=str(item.get("id") or "tool_call"),
                        name=str(item.get("name") or "tool"),
                        arguments_json="{}",
                    )
                )
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    tokens = usage.get("tokens") if isinstance(usage.get("tokens"), Mapping) else {}
    return CanonicalFinalResponse(
        provider=provider,
        model=ModelRef.from_values(provider.name, provider.api_family, str(payload.get("id") or "")),
        response_id=str(payload.get("id") or ""),
        status="completed",
        output=[CanonicalMessage(role=MessageRole.ASSISTANT, content=parts, tool_calls=tool_calls)],
        output_text="".join(output_text),
        tool_calls=tool_calls,
        usage={
            "input_tokens": tokens.get("input_tokens"),
            "output_tokens": tokens.get("output_tokens"),
            "total_tokens": tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0) if isinstance(tokens.get("input_tokens"), int) and isinstance(tokens.get("output_tokens"), int) else None,
        },
        finish_reason=payload.get("finish_reason") if isinstance(payload.get("finish_reason"), str) else None,
        provider_extensions=chat_extensions(payload),
        raw_provider_envelope=RawProviderEnvelope(provider=ProviderName.COHERE, api_family=provider.api_family, headers=dict(headers or {}), payload=dict(payload)),
    )
