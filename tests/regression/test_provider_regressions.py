from __future__ import annotations

from aster.core.canonical import ProviderStreamEvent
from aster.providers.anthropic import AnthropicMessagesAdapter
from aster.providers.gemini import GeminiGenerateContentAdapter
from aster.providers.openai import OpenAIResponsesAdapter
from aster.providers.xai import XAIResponsesAdapter
from aster.testing.fixtures import load_fixture


def test_openai_response_parser_preserves_unknown_native_fields() -> None:
    adapter = OpenAIResponsesAdapter()
    response = adapter.parse_response(load_fixture("providers", "openai", "responses_response.json"))

    assert response.provider_extensions.values["native_extra"]["preserved"] is True


def test_anthropic_parser_preserves_tool_use_finish_reason() -> None:
    adapter = AnthropicMessagesAdapter()
    response = adapter.parse_response(load_fixture("providers", "anthropic", "messages_response.json"))

    assert response.finish_reason == "tool_use"
    assert response.tool_calls[0].name == "lookup_weather"


def test_xai_parser_preserves_citations_as_provider_extensions() -> None:
    adapter = XAIResponsesAdapter()
    response = adapter.parse_response(load_fixture("providers", "xai", "responses_response.json"))

    assert response.provider_extensions.values["citations"][0]["url"] == "https://example.com"


def test_gemini_stream_decoder_handles_empty_payload_defensively() -> None:
    adapter = GeminiGenerateContentAdapter()
    chunks = adapter.decode_stream_event(ProviderStreamEvent(data={}))

    assert chunks == []
