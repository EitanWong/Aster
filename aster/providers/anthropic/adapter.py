from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalError,
    CanonicalFinalResponse,
    CanonicalRequest,
    CanonicalResponseChunk,
    ProviderCapabilities,
    ProviderHttpRequest,
    ProviderName,
    ProviderRef,
    ProviderRequestContext,
    ProviderStreamEvent,
)
from aster.core.contracts import ProviderAdapter
from aster.providers.anthropic.capabilities import messages_capabilities
from aster.providers.anthropic.errors import map_error
from aster.providers.anthropic.models import MESSAGES_DOCS
from aster.providers.anthropic.request_builder import build_messages_request
from aster.providers.anthropic.response_parser import parse_messages_response
from aster.providers.anthropic.stream_decoder import decode_messages_stream


class AnthropicMessagesAdapter(ProviderAdapter):
    provider_name = ProviderName.ANTHROPIC.value
    api_family = "messages"
    docs_urls = MESSAGES_DOCS
    provider_ref = ProviderRef(name=ProviderName.ANTHROPIC, api_family="messages")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return messages_capabilities()

    def build_request(
        self,
        request: CanonicalRequest,
        context: ProviderRequestContext | None = None,
    ) -> ProviderHttpRequest:
        return build_messages_request(self.provider_ref, request, context)

    def parse_response(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalFinalResponse:
        return parse_messages_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_messages_stream(self.provider_ref, event)

    def map_error(
        self,
        status_code: int,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)
