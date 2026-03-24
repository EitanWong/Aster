from __future__ import annotations

from collections.abc import Mapping

from aster.core.canonical import CanonicalResponseChunk, ProviderRef, ProviderStreamEvent, RawProviderEnvelope
from aster.providers._shared import function_call_to_canonical
from aster.providers.openai.stream_decoder import decode_chat_stream


def decode_generate_content_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    data = event.data if isinstance(event.data, Mapping) else {}
    candidates = data.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return []
    first_candidate = candidates[0]
    if not isinstance(first_candidate, Mapping):
        return []
    content = first_candidate.get("content", {})
    if not isinstance(content, Mapping):
        content = {}
    parts = content.get("parts", [])
    chunks: list[CanonicalResponseChunk] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        chunk = CanonicalResponseChunk(
            provider=provider,
            event_type="candidate",
            sequence_number=event.sequence_number,
            finish_reason=first_candidate.get("finishReason") if isinstance(first_candidate.get("finishReason"), str) else None,
            raw_provider_envelope=RawProviderEnvelope(
                provider=provider.name,
                api_family=provider.api_family,
                event_name="candidate",
                payload=data,
            ),
        )
        if isinstance(part.get("text"), str):
            chunk.delta_text = part["text"]
            chunks.append(chunk)
        elif "functionCall" in part and isinstance(part.get("functionCall"), Mapping):
            function_call = part["functionCall"]
            chunk.tool_call = function_call_to_canonical(
                call_id=str(function_call.get("id") or function_call.get("name") or "function_call"),
                name=str(function_call.get("name") or "function"),
                arguments_json="{}",
                provider_extensions={"partial": True},
            )
            chunks.append(chunk)
    return chunks


def decode_openai_chat_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    return decode_chat_stream(provider, event)
