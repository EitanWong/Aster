from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalFinalResponse,
    CanonicalMessage,
    MessageRole,
    ModelRef,
    ProviderExtensionData,
    ProviderName,
    ProviderRef,
    RawProviderEnvelope,
)
from aster.providers._shared import text_part
from aster.providers.anthropic.extensions import response_extensions
from aster.providers.anthropic.tool_mapper import parse_tool_use


def parse_messages_response(
    provider: ProviderRef,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> CanonicalFinalResponse:
    content_parts = []
    tool_calls = []
    output_text = []
    for block in payload.get("content", []):
        if not isinstance(block, Mapping):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            output_text.append(block["text"])
            content_parts.append(text_part(block["text"], provider_extensions=dict(block)))
        elif block.get("type") == "tool_use":
            tool_calls.append(parse_tool_use(block))
    usage_payload = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    usage = {
        "input_tokens": usage_payload.get("input_tokens"),
        "output_tokens": usage_payload.get("output_tokens"),
        "provider_extensions": {"values": {key: value for key, value in usage_payload.items() if key not in {"input_tokens", "output_tokens"}}},
    }
    return CanonicalFinalResponse(
        provider=provider,
        model=ModelRef.from_values(provider.name, provider.api_family, str(payload.get("model") or "")),
        response_id=str(payload.get("id") or ""),
        status="completed",
        output=[
            CanonicalMessage(
                role=MessageRole(payload.get("role", "assistant")),
                content=content_parts,
                tool_calls=tool_calls,
            )
        ],
        output_text="".join(output_text),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=payload.get("stop_reason") if isinstance(payload.get("stop_reason"), str) else None,
        request_id=_request_id(headers),
        provider_version=_header(headers, "anthropic-version"),
        provider_beta_headers=_beta_headers(headers),
        provider_extensions=response_extensions(payload),
        raw_provider_envelope=RawProviderEnvelope(
            provider=ProviderName.ANTHROPIC,
            api_family=provider.api_family,
            headers=dict(headers or {}),
            payload=dict(payload),
        ),
    )


def _request_id(headers: Mapping[str, str] | None) -> str | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    return lowered.get("request-id")


def _header(headers: Mapping[str, str] | None, key: str) -> str | None:
    lowered = {name.lower(): value for name, value in (headers or {}).items()}
    return lowered.get(key.lower())


def _beta_headers(headers: Mapping[str, str] | None) -> list[str]:
    beta = _header(headers, "anthropic-beta")
    return beta.split(",") if beta else []
