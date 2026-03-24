from __future__ import annotations

from collections.abc import Mapping

from aster.core.canonical import CanonicalResponseChunk, ProviderRef, ProviderStreamEvent, RawProviderEnvelope
from aster.providers._shared import function_call_to_canonical
from aster.providers.openai.stream_decoder import decode_chat_stream, decode_responses_stream


def decode_converse_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    data = event.data if isinstance(event.data, Mapping) else {}
    chunks: list[CanonicalResponseChunk] = []
    if "contentBlockDelta" in data and isinstance(data["contentBlockDelta"], Mapping):
        delta = data["contentBlockDelta"]
        inner = delta.get("delta", {})
        if isinstance(inner, Mapping) and isinstance(inner.get("text"), str):
            chunks.append(
                CanonicalResponseChunk(
                    provider=provider,
                    event_type="contentBlockDelta",
                    delta_text=inner["text"],
                    sequence_number=event.sequence_number,
                    raw_provider_envelope=RawProviderEnvelope(provider=provider.name, api_family=provider.api_family, event_name="contentBlockDelta", payload=data),
                )
            )
    if "contentBlockStart" in data and isinstance(data["contentBlockStart"], Mapping):
        start = data["contentBlockStart"]
        inner = start.get("start", {})
        if isinstance(inner, Mapping) and isinstance(inner.get("toolUse"), Mapping):
            tool_use = inner["toolUse"]
            chunks.append(
                CanonicalResponseChunk(
                    provider=provider,
                    event_type="contentBlockStart",
                    tool_call=function_call_to_canonical(
                        call_id=str(tool_use.get("toolUseId") or "tool_use"),
                        name=str(tool_use.get("name") or "tool"),
                        arguments_json="{}",
                        provider_extensions={"partial": True},
                    ),
                    sequence_number=event.sequence_number,
                    raw_provider_envelope=RawProviderEnvelope(provider=provider.name, api_family=provider.api_family, event_name="contentBlockStart", payload=data),
                )
            )
    if "metadata" in data and isinstance(data["metadata"], Mapping):
        metadata = data["metadata"]
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), Mapping) else {}
        chunks.append(
            CanonicalResponseChunk(
                provider=provider,
                event_type="metadata",
                usage={
                    "input_tokens": usage.get("inputTokens"),
                    "output_tokens": usage.get("outputTokens"),
                    "total_tokens": usage.get("totalTokens"),
                },
                sequence_number=event.sequence_number,
                raw_provider_envelope=RawProviderEnvelope(provider=provider.name, api_family=provider.api_family, event_name="metadata", payload=data),
            )
        )
    if "messageStop" in data and isinstance(data["messageStop"], Mapping):
        stop = data["messageStop"]
        chunks.append(
            CanonicalResponseChunk(
                provider=provider,
                event_type="messageStop",
                finish_reason=stop.get("stopReason") if isinstance(stop.get("stopReason"), str) else None,
                sequence_number=event.sequence_number,
                raw_provider_envelope=RawProviderEnvelope(provider=provider.name, api_family=provider.api_family, event_name="messageStop", payload=data),
            )
        )
    return chunks


def decode_openai_chat_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    return decode_chat_stream(provider, event)


def decode_openai_responses_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    return decode_responses_stream(provider, event)
