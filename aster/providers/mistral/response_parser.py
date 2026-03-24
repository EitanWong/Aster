from __future__ import annotations

from aster.core.canonical import CanonicalFinalResponse, ProviderRef
from aster.providers.openai.response_parser import parse_chat_response as parse_openai_chat_response


def parse_chat_response(provider: ProviderRef, payload: dict[str, object], headers: dict[str, str] | None = None) -> CanonicalFinalResponse:
    return parse_openai_chat_response(provider, payload, headers=headers)
