from __future__ import annotations

from aster.core.canonical import CanonicalResponseChunk, ProviderRef, ProviderStreamEvent
from aster.providers.openai.stream_decoder import decode_chat_stream as decode_openai_chat_stream
from aster.providers.openai.stream_decoder import decode_responses_stream as decode_openai_responses_stream


def decode_responses_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    return decode_openai_responses_stream(provider, event)


def decode_chat_stream(provider: ProviderRef, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
    return decode_openai_chat_stream(provider, event)
