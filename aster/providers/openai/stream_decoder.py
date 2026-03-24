from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalResponseChunk,
    ProviderExtensionData,
    ProviderRef,
    ProviderStreamEvent,
    RawProviderEnvelope,
)
from aster.providers._shared import function_call_to_canonical, usage_from_token_payload


def decode_responses_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    data = event.data if isinstance(event.data, Mapping) else {}
    event_type = _event_type(event, data)
    chunk = CanonicalResponseChunk(
        provider=provider,
        response_id=_string(data.get("response_id")) or _string(data.get("id")),
        event_type=event_type,
        sequence_number=event.sequence_number,
        provider_extensions=ProviderExtensionData(values={key: value for key, value in data.items() if key not in {"type", "response_id", "delta", "usage"}}),
        raw_provider_envelope=RawProviderEnvelope(
            provider=provider.name,
            api_family=provider.api_family,
            event_name=event_type,
            payload=data if data else event.raw,
        ),
    )
    if event_type == "response.output_text.delta":
        chunk.delta_text = _string(data.get("delta"))
    elif event_type == "response.function_call_arguments.delta":
        chunk.tool_call = function_call_to_canonical(
            call_id=_string(data.get("item_id")) or _string(data.get("call_id")) or "tool_call",
            name=_string(data.get("name")) or "function",
            arguments_json=_string(data.get("delta")),
            provider_extensions={"partial": True},
        )
    elif event_type == "response.completed":
        usage_payload = data.get("response", {}).get("usage") if isinstance(data.get("response"), Mapping) else data.get("usage")
        if isinstance(usage_payload, Mapping):
            chunk.usage = usage_from_token_payload(
                usage_payload,
                input_key="input_tokens",
                output_key="output_tokens",
                total_key="total_tokens",
            )
    return [chunk]


def decode_chat_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    if isinstance(event.data, str) and event.data.strip() == "[DONE]":
        return [
            CanonicalResponseChunk(
                provider=provider,
                event_type="done",
                sequence_number=event.sequence_number,
            )
        ]
    data = event.data if isinstance(event.data, Mapping) else {}
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return []
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        return []
    delta = first_choice.get("delta", {})
    if not isinstance(delta, Mapping):
        delta = {}
    chunk = CanonicalResponseChunk(
        provider=provider,
        response_id=_string(data.get("id")),
        event_type="chat.completion.chunk",
        delta_text=_string(delta.get("content")),
        finish_reason=_string(first_choice.get("finish_reason")),
        output_index=0,
        sequence_number=event.sequence_number,
        raw_provider_envelope=RawProviderEnvelope(
            provider=provider.name,
            api_family=provider.api_family,
            event_name="chat.completion.chunk",
            payload=data,
        ),
    )
    tool_calls = delta.get("tool_calls", [])
    if isinstance(tool_calls, list) and tool_calls:
        first_tool = tool_calls[0]
        if isinstance(first_tool, Mapping):
            function_data = first_tool.get("function", {})
            if not isinstance(function_data, Mapping):
                function_data = {}
            chunk.tool_call = function_call_to_canonical(
                call_id=_string(first_tool.get("id")) or "tool_call",
                name=_string(function_data.get("name")) or "function",
                arguments_json=_string(function_data.get("arguments")),
                provider_extensions={"partial": True, "index": first_tool.get("index")},
            )
    return [chunk]


def _event_type(event: ProviderStreamEvent, data: Mapping[str, Any]) -> str:
    if isinstance(data.get("type"), str):
        return data["type"]
    if event.event:
        return event.event
    return "unknown"


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
