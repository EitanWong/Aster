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
from aster.providers.gemini.capabilities import generate_content_capabilities, openai_chat_capabilities
from aster.providers.gemini.errors import map_error
from aster.providers.gemini.models import GENERATE_CONTENT_DOCS
from aster.providers.gemini.request_builder import build_generate_content_request, build_openai_chat_request
from aster.providers.gemini.response_parser import parse_generate_content_response, parse_openai_chat_response
from aster.providers.gemini.stream_decoder import decode_generate_content_stream, decode_openai_chat_stream


class GeminiGenerateContentAdapter(ProviderAdapter):
    provider_name = ProviderName.GEMINI.value
    api_family = "generate_content"
    docs_urls = GENERATE_CONTENT_DOCS
    provider_ref = ProviderRef(name=ProviderName.GEMINI, api_family="generate_content")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return generate_content_capabilities()

    def build_request(
        self,
        request: CanonicalRequest,
        context: ProviderRequestContext | None = None,
    ) -> ProviderHttpRequest:
        return build_generate_content_request(self.provider_ref, request, context)

    def parse_response(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalFinalResponse:
        return parse_generate_content_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_generate_content_stream(self.provider_ref, event)

    def map_error(
        self,
        status_code: int,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)


class GeminiOpenAIChatAdapter(ProviderAdapter):
    provider_name = ProviderName.GEMINI.value
    api_family = "openai_chat"
    docs_urls = GENERATE_CONTENT_DOCS
    provider_ref = ProviderRef(name=ProviderName.GEMINI, api_family="openai_chat")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return openai_chat_capabilities()

    def build_request(
        self,
        request: CanonicalRequest,
        context: ProviderRequestContext | None = None,
    ) -> ProviderHttpRequest:
        return build_openai_chat_request(self.provider_ref, request, context)

    def parse_response(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalFinalResponse:
        return parse_openai_chat_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_openai_chat_stream(self.provider_ref, event)

    def map_error(
        self,
        status_code: int,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)
