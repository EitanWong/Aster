from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalFinalResponse,
    CanonicalMessage,
    ContentPartType,
    MessageRole,
    ModelRef,
    ProviderExtensionData,
    ProviderName,
    ProviderRef,
    RawProviderEnvelope,
)
from aster.providers._shared import text_part, usage_from_token_payload
from aster.providers.openai.extensions import chat_extensions, response_extensions
from aster.providers.openai.tool_mapper import parse_chat_tool_call, parse_response_output_tool_call


def parse_responses_response(
    provider: ProviderRef,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> CanonicalFinalResponse:
    messages: list[CanonicalMessage] = []
    tool_calls = []
    output_text_parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "message":
            content_parts = []
            for content in item.get("content", []):
                if not isinstance(content, Mapping):
                    continue
                content_type = content.get("type")
                if content_type in {"output_text", "text"} and isinstance(content.get("text"), str):
                    output_text_parts.append(content["text"])
                    content_parts.append(text_part(content["text"], provider_extensions=dict(content)))
                elif content_type == "refusal" and isinstance(content.get("refusal"), str):
                    content_parts.append(
                        text_part(
                            content["refusal"],
                            provider_extensions={"type": ContentPartType.REFUSAL.value, **dict(content)},
                        )
                    )
            messages.append(
                CanonicalMessage(
                    role=MessageRole(item.get("role", "assistant")),
                    content=content_parts,
                    provider_extensions=ProviderExtensionData(values={key: value for key, value in item.items() if key not in {"type", "role", "content"}}),
                )
            )
            continue
        if item_type == "function_call":
            tool_calls.append(parse_response_output_tool_call(item))
    usage = usage_from_token_payload(
        payload.get("usage"),
        input_key="input_tokens",
        output_key="output_tokens",
        total_key="total_tokens",
        extra_keys=("input_tokens_details", "output_tokens_details"),
    )
    return CanonicalFinalResponse(
        provider=provider,
        model=ModelRef.from_values(provider.name, provider.api_family, str(payload.get("model") or "")),
        response_id=str(payload.get("id") or ""),
        status=_str_or_none(payload.get("status")),
        output=messages,
        output_text="".join(output_text_parts),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_finish_reason_from_responses(payload),
        request_id=_request_id(headers),
        rate_limit_headers=_rate_limit_headers(headers),
        provider_extensions=response_extensions(payload),
        raw_provider_envelope=RawProviderEnvelope(
            provider=ProviderName.OPENAI,
            api_family=provider.api_family,
            headers=dict(headers or {}),
            payload=dict(payload),
        ),
    )


def parse_chat_response(
    provider: ProviderRef,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> CanonicalFinalResponse:
    choices = payload.get("choices", [])
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(first_choice, Mapping):
        first_choice = {}
    message_data = first_choice.get("message", {})
    if not isinstance(message_data, Mapping):
        message_data = {}
    content = message_data.get("content")
    content_parts = [text_part(content, provider_extensions=dict(message_data))] if isinstance(content, str) and content else []
    tool_calls = [
        parse_chat_tool_call(item)
        for item in message_data.get("tool_calls", [])
        if isinstance(item, Mapping)
    ]
    usage = usage_from_token_payload(
        payload.get("usage"),
        input_key="prompt_tokens",
        output_key="completion_tokens",
        total_key="total_tokens",
        extra_keys=("prompt_tokens_details", "completion_tokens_details"),
    )
    return CanonicalFinalResponse(
        provider=provider,
        model=ModelRef.from_values(provider.name, provider.api_family, str(payload.get("model") or "")),
        response_id=str(payload.get("id") or ""),
        status="completed",
        output=[
            CanonicalMessage(
                role=MessageRole(message_data.get("role", "assistant")),
                content=content_parts,
                tool_calls=tool_calls,
                provider_extensions=ProviderExtensionData(values={key: value for key, value in message_data.items() if key not in {"role", "content", "tool_calls"}}),
            )
        ],
        output_text=content if isinstance(content, str) else "",
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_str_or_none(first_choice.get("finish_reason")),
        request_id=_request_id(headers),
        rate_limit_headers=_rate_limit_headers(headers),
        provider_extensions=chat_extensions(payload),
        raw_provider_envelope=RawProviderEnvelope(
            provider=ProviderName.OPENAI,
            api_family=provider.api_family,
            headers=dict(headers or {}),
            payload=dict(payload),
        ),
    )


def _finish_reason_from_responses(payload: Mapping[str, Any]) -> str | None:
    incomplete = payload.get("incomplete_details")
    if isinstance(incomplete, Mapping) and isinstance(incomplete.get("reason"), str):
        return incomplete["reason"]
    return _str_or_none(payload.get("status"))


def _request_id(headers: Mapping[str, str] | None) -> str | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    return lowered.get("x-request-id")


def _rate_limit_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    return {key: value for key, value in (headers or {}).items() if "ratelimit" in key.lower()}


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
