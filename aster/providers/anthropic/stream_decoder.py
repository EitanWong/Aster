from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalResponseChunk, ProviderRef, ProviderStreamEvent, RawProviderEnvelope
from aster.providers._shared import function_call_to_canonical


def decode_messages_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    data = event.data if isinstance(event.data, Mapping) else {}
    event_type = _event_type(event, data)
    chunk = CanonicalResponseChunk(
        provider=provider,
        response_id=_string(data.get("message", {}).get("id")) if isinstance(data.get("message"), Mapping) else None,
        event_type=event_type,
        sequence_number=event.sequence_number,
        raw_provider_envelope=RawProviderEnvelope(
            provider=provider.name,
            api_family=provider.api_family,
            event_name=event_type,
            payload=data if data else event.raw,
        ),
    )
    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        if isinstance(delta, Mapping):
            if delta.get("type") == "text_delta":
                chunk.delta_text = _string(delta.get("text"))
            elif delta.get("type") == "input_json_delta":
                chunk.tool_call = function_call_to_canonical(
                    call_id=_string(data.get("index")) or "tool_use",
                    name="tool",
                    arguments_json=_string(delta.get("partial_json")),
                    provider_extensions={"partial": True},
                )
    elif event_type == "message_delta":
        delta = data.get("delta", {})
        if isinstance(delta, Mapping) and isinstance(delta.get("stop_reason"), str):
            chunk.finish_reason = delta["stop_reason"]
        usage = data.get("usage")
        if isinstance(usage, Mapping):
            chunk.usage = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
    return [chunk]


def _event_type(event: ProviderStreamEvent, data: Mapping[str, Any]) -> str:
    if event.event:
        return event.event
    if isinstance(data.get("type"), str):
        return data["type"]
    return "unknown"


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
