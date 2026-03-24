from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalError, CanonicalFinalResponse, CanonicalRequest, CanonicalResponseChunk, ProviderCapabilities, ProviderHttpRequest, ProviderName, ProviderRef, ProviderRequestContext, ProviderStreamEvent
from aster.core.contracts import ProviderAdapter
from aster.providers.bedrock.capabilities import converse_capabilities, openai_compat_capabilities
from aster.providers.bedrock.errors import map_error
from aster.providers.bedrock.models import CONVERSE_DOCS
from aster.providers.bedrock.request_builder import build_converse_request, build_openai_chat_request, build_openai_responses_request
from aster.providers.bedrock.response_parser import parse_converse_response, parse_openai_chat_response, parse_openai_responses_response
from aster.providers.bedrock.stream_decoder import decode_converse_stream, decode_openai_chat_stream, decode_openai_responses_stream


class BedrockConverseAdapter(ProviderAdapter):
    provider_name = ProviderName.BEDROCK.value
    api_family = "converse"
    docs_urls = CONVERSE_DOCS
    provider_ref = ProviderRef(name=ProviderName.BEDROCK, api_family="converse")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return converse_capabilities()

    def build_request(self, request: CanonicalRequest, context: ProviderRequestContext | None = None) -> ProviderHttpRequest:
        return build_converse_request(self.provider_ref, request, context)

    def parse_response(self, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
        return parse_converse_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_converse_stream(self.provider_ref, event)

    def map_error(self, status_code: int, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None = None) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)


class BedrockOpenAIResponsesAdapter(ProviderAdapter):
    provider_name = ProviderName.BEDROCK.value
    api_family = "openai_responses"
    docs_urls = CONVERSE_DOCS
    provider_ref = ProviderRef(name=ProviderName.BEDROCK, api_family="openai_responses")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return openai_compat_capabilities("openai_responses")

    def build_request(self, request: CanonicalRequest, context: ProviderRequestContext | None = None) -> ProviderHttpRequest:
        return build_openai_responses_request(self.provider_ref, request, context)

    def parse_response(self, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
        return parse_openai_responses_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_openai_responses_stream(self.provider_ref, event)

    def map_error(self, status_code: int, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None = None) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)


class BedrockOpenAIChatAdapter(ProviderAdapter):
    provider_name = ProviderName.BEDROCK.value
    api_family = "openai_chat"
    docs_urls = CONVERSE_DOCS
    provider_ref = ProviderRef(name=ProviderName.BEDROCK, api_family="openai_chat")

    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        return openai_compat_capabilities("openai_chat")

    def build_request(self, request: CanonicalRequest, context: ProviderRequestContext | None = None) -> ProviderHttpRequest:
        return build_openai_chat_request(self.provider_ref, request, context)

    def parse_response(self, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> CanonicalFinalResponse:
        return parse_openai_chat_response(self.provider_ref, payload, headers=headers)

    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        return decode_openai_chat_stream(self.provider_ref, event)

    def map_error(self, status_code: int, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None = None) -> CanonicalError:
        return map_error(self.provider_ref, status_code, payload, headers=headers)
