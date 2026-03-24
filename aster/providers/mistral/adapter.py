from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalError, CanonicalFinalResponse, CanonicalRequest, CanonicalResponseChunk, ProviderCapabilities, ProviderHttpRequest, ProviderName, ProviderRef, ProviderRequestContext, ProviderStreamEvent
from aster.core.contracts import ProviderAdapter
from aster.providers.mistral.capabilities import chat_capabilities, conversations_capabilities
from aster.providers.mistral.errors import map_error
from aster.providers.mistral.models import MISTRAL_DOCS
from aster.providers.mistral.request_builder import build_chat_completions_request, build_conversations_request
from aster.providers.openai.response_parser import parse_chat_response as parse_openai_chat_response
from aster.providers.openai.stream_decoder import decode_chat_stream as decode_openai_chat_stream


class MistralChatCompletionsAdapter(ProviderAdapter):
    provider_name = ProviderName.MISTRAL.value
    api_family = "chat_completions"
    docs_urls = MISTRAL_DOCS
    provider_ref = ProviderRef(name=ProviderName.MISTRAL, api_family="chat_completions")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return chat_capabilities()

    def build_request(self, request: CanonicalRequest, context: ProviderRequestContext | None = None) -> ProviderHttpRequest:
        return build_chat_completions_request(self.provider_ref, request, context)

    def parse_response(self, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
        return parse_openai_chat_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_openai_chat_stream(self.provider_ref, event)

    def map_error(self, status_code: int, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None = None) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)


class MistralConversationsAdapter(ProviderAdapter):
    provider_name = ProviderName.MISTRAL.value
    api_family = "conversations"
    docs_urls = MISTRAL_DOCS
    provider_ref = ProviderRef(name=ProviderName.MISTRAL, api_family="conversations")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return conversations_capabilities()

    def build_request(self, request: CanonicalRequest, context: ProviderRequestContext | None = None) -> ProviderHttpRequest:
        return build_conversations_request(self.provider_ref, request, context)

    def parse_response(self, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
        return parse_openai_chat_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_openai_chat_stream(self.provider_ref, event)

    def map_error(self, status_code: int, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None = None) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)
