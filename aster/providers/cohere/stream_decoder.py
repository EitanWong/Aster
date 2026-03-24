from __future__ import annotations

from collections.abc import Mapping

from aster.core.canonical import CanonicalResponseChunk, ProviderRef, ProviderStreamEvent, RawProviderEnvelope
from aster.providers._shared import function_call_to_canonical


def decode_chat_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    data = event.data if isinstance(event.data, Mapping) else {}
    event_type = data.get("type") if isinstance(data.get("type"), str) else (event.event or "unknown")
    chunk = CanonicalResponseChunk(
        provider=provider,
        event_type=event_type,
        sequence_number=event.sequence_number,
        raw_provider_envelope=RawProviderEnvelope(provider=provider.name, api_family=provider.api_family, event_name=event_type, payload=data if data else event.raw),
    )
    if event_type == "content-delta" and isinstance(data.get("delta"), Mapping):
        delta = data["delta"]
        if isinstance(delta.get("message"), Mapping) and isinstance(delta["message"].get("content"), Mapping):
            content = delta["message"]["content"]
            if isinstance(content.get("text"), str):
                chunk.delta_text = content["text"]
    elif event_type in {"tool-call-start", "tool-call-delta"}:
        chunk.tool_call = function_call_to_canonical(
            call_id=str(data.get("id") or "tool_call"),
            name=str(data.get("name") or "tool"),
            arguments_json=str(data.get("delta") or "{}"),
            provider_extensions={"partial": event_type == "tool-call-delta"},
        )
    elif event_type == "message-end":
        chunk.finish_reason = data.get("finish_reason") if isinstance(data.get("finish_reason"), str) else None
    return [chunk]
