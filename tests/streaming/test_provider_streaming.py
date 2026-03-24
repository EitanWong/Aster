from __future__ import annotations

from aster.core.canonical import ProviderStreamEvent
from aster.providers.anthropic import AnthropicMessagesAdapter
from aster.providers.bedrock import BedrockConverseAdapter
from aster.providers.cohere import CohereChatV2Adapter
from aster.providers.gemini import GeminiGenerateContentAdapter
from aster.providers.openai import OpenAIResponsesAdapter
from aster.testing.fixtures import load_fixture


def test_openai_responses_stream_delta_decodes_text() -> None:
    adapter = OpenAIResponsesAdapter()
    event = ProviderStreamEvent(data=load_fixture("providers", "openai", "responses_stream_event.json"))

    chunks = adapter.decode_stream_event(event)

    assert chunks[0].delta_text == "Hel"


def test_anthropic_stream_tool_delta_preserves_partial_tool_state() -> None:
    adapter = AnthropicMessagesAdapter()
    event = ProviderStreamEvent(event="content_block_delta", data=load_fixture("providers", "anthropic", "stream_tool_delta.json"))

    chunks = adapter.decode_stream_event(event)

    assert chunks[0].tool_call is not None
    assert chunks[0].tool_call.provider_extensions.values["partial"] is True


def test_gemini_stream_chunk_decodes_candidate_text() -> None:
    adapter = GeminiGenerateContentAdapter()
    event = ProviderStreamEvent(data=load_fixture("providers", "gemini", "stream_chunk.json"))

    chunks = adapter.decode_stream_event(event)

    assert chunks[0].delta_text == "Gem"


def test_bedrock_stream_metadata_decodes_usage() -> None:
    adapter = BedrockConverseAdapter()
    event = ProviderStreamEvent(data=load_fixture("providers", "bedrock", "converse_stream.json"))

    chunks = adapter.decode_stream_event(event)

    assert chunks[0].usage.input_tokens == 9


def test_cohere_stream_tool_delta_decodes_partial_tool_call() -> None:
    adapter = CohereChatV2Adapter()
    event = ProviderStreamEvent(data=load_fixture("providers", "cohere", "stream_tool_delta.json"))

    chunks = adapter.decode_stream_event(event)

    assert chunks[0].tool_call is not None
    assert chunks[0].tool_call.name == "lookup_weather"
