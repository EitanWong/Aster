from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalError, CanonicalFinalResponse, CanonicalRequest, CanonicalResponseChunk, ProviderCapabilities, ProviderHttpRequest, ProviderName, ProviderRef, ProviderRequestContext, ProviderStreamEvent
from aster.core.contracts import ProviderAdapter
from aster.providers.cohere.capabilities import chat_v2_capabilities
from aster.providers.cohere.errors import map_error
from aster.providers.cohere.models import COHERE_DOCS
from aster.providers.cohere.request_builder import build_chat_request
from aster.providers.cohere.response_parser import parse_chat_response
from aster.providers.cohere.stream_decoder import decode_chat_stream


class CohereChatV2Adapter(ProviderAdapter):
    provider_name = ProviderName.COHERE.value
    api_family = "chat_v2"
    docs_urls = COHERE_DOCS
    provider_ref = ProviderRef(name=ProviderName.COHERE, api_family="chat_v2")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return chat_v2_capabilities()

    def build_request(self, request: CanonicalRequest, context: ProviderRequestContext | None = None) -> ProviderHttpRequest:
        return build_chat_request(self.provider_ref, request, context)

    def parse_response(self, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
        return parse_chat_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_chat_stream(self.provider_ref, event)

    def map_error(self, status_code: int, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None = None) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)
