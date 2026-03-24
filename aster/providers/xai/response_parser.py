from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalFinalResponse, ProviderRef
from aster.providers.openai.response_parser import parse_chat_response as parse_openai_chat_response
from aster.providers.openai.response_parser import parse_responses_response as parse_openai_responses_response
from aster.providers.xai.extensions import chat_extensions, response_extensions


def parse_responses_response(provider: ProviderRef, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
    response = parse_openai_responses_response(provider, payload, headers=headers)
    response.provider_extensions = response_extensions(payload).merged({"citations": payload.get("citations"), "reasoning": payload.get("reasoning")})
    return response


def parse_chat_response(provider: ProviderRef, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
    response = parse_openai_chat_response(provider, payload, headers=headers)
    response.provider_extensions = chat_extensions(payload).merged({"citations": payload.get("citations")})
    return response
