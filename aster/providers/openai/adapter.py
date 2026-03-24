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
from aster.providers.openai.capabilities import chat_capabilities, responses_capabilities
from aster.providers.openai.errors import map_error
from aster.providers.openai.models import CHAT_DOCS, RESPONSES_DOCS
from aster.providers.openai.request_builder import build_chat_request, build_responses_request
from aster.providers.openai.response_parser import parse_chat_response, parse_responses_response
from aster.providers.openai.stream_decoder import decode_chat_stream, decode_responses_stream


class OpenAIResponsesAdapter(ProviderAdapter):
    provider_name = ProviderName.OPENAI.value
    api_family = "responses"
    docs_urls = RESPONSES_DOCS
    provider_ref = ProviderRef(name=ProviderName.OPENAI, api_family="responses")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return responses_capabilities()

    def build_request(
        self,
        request: CanonicalRequest,
        context: ProviderRequestContext | None = None,
    ) -> ProviderHttpRequest:
        return build_responses_request(self.provider_ref, request, context)

    def parse_response(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalFinalResponse:
        return parse_responses_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_responses_stream(self.provider_ref, event)

    def map_error(
        self,
        status_code: int,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)


class OpenAIChatCompletionsAdapter(ProviderAdapter):
    provider_name = ProviderName.OPENAI.value
    api_family = "chat_completions"
    docs_urls = CHAT_DOCS
    provider_ref = ProviderRef(name=ProviderName.OPENAI, api_family="chat_completions")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return chat_capabilities()

    def build_request(
        self,
        request: CanonicalRequest,
        context: ProviderRequestContext | None = None,
    ) -> ProviderHttpRequest:
        return build_chat_request(self.provider_ref, request, context)

    def parse_response(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalFinalResponse:
        return parse_chat_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_chat_stream(self.provider_ref, event)

    def map_error(
        self,
        status_code: int,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)
