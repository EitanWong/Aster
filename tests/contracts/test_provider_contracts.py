from __future__ import annotations

import pytest

from aster.core.canonical import (
    CanonicalContentPart,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalToolDefinition,
    ContentPartType,
    MessageRole,
    ModelRef,
)
from aster.core.provider_errors import UnsupportedProviderFeatureError
from aster.providers.anthropic import AnthropicMessagesAdapter
from aster.providers.bedrock import BedrockConverseAdapter
from aster.providers.gemini import GeminiGenerateContentAdapter
from aster.providers.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from aster.providers.xai import XAIResponsesAdapter


def _request(model: ModelRef) -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=[
            CanonicalMessage(role=MessageRole.SYSTEM, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="You are helpful.")]),
            CanonicalMessage(role=MessageRole.USER, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="Hello")]),
        ],
        tools=[CanonicalToolDefinition(name="lookup_weather", input_schema={"type": "object"})],
        structured_output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        structured_output_name="answer_schema",
        max_output_tokens=128,
    )


def test_openai_responses_request_shape_contains_response_fields() -> None:
    adapter = OpenAIResponsesAdapter()
    request = _request(ModelRef.from_values("openai", "responses", "gpt-5.1"))

    http_request = adapter.build_request(request)

    assert http_request.path == "/v1/responses"
    assert "input" in http_request.json_body
    assert http_request.json_body["text"]["format"]["type"] == "json_schema"


def test_openai_chat_rejects_previous_response_id() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    request = _request(ModelRef.from_values("openai", "chat_completions", "gpt-4.1"))
    request.previous_response_id = "resp_old"

    with pytest.raises(UnsupportedProviderFeatureError):
        adapter.build_request(request)


def test_anthropic_request_extracts_system_and_required_headers() -> None:
    adapter = AnthropicMessagesAdapter()
    request = CanonicalRequest(
        model=ModelRef.from_values("anthropic", "messages", "claude-sonnet-4-5"),
        messages=[
            CanonicalMessage(role=MessageRole.SYSTEM, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="Be concise.")]),
            CanonicalMessage(role=MessageRole.USER, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="Hi")]),
        ],
        max_output_tokens=64,
    )

    http_request = adapter.build_request(request)

    assert http_request.path == "/v1/messages"
    assert http_request.json_body["system"] == "Be concise."
    assert "anthropic-version" in http_request.headers


def test_gemini_native_request_uses_generation_config_schema() -> None:
    adapter = GeminiGenerateContentAdapter()
    request = _request(ModelRef.from_values("gemini", "generate_content", "gemini-2.5-flash"))

    http_request = adapter.build_request(request)

    assert http_request.path.endswith(":generateContent")
    assert http_request.json_body["generationConfig"]["responseSchema"]["type"] == "object"


def test_bedrock_converse_uses_sigv4_and_native_config() -> None:
    adapter = BedrockConverseAdapter()
    request = CanonicalRequest(
        model=ModelRef.from_values("bedrock", "converse", "anthropic.claude-3-5-sonnet"),
        messages=[
            CanonicalMessage(role=MessageRole.SYSTEM, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="System")]),
            CanonicalMessage(role=MessageRole.USER, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="Hi")]),
        ],
        max_output_tokens=32,
    )

    http_request = adapter.build_request(request)

    assert http_request.requires_sigv4 is True
    assert http_request.json_body["system"][0]["text"] == "System"


def test_xai_preserves_native_web_search_tool() -> None:
    adapter = XAIResponsesAdapter()
    request = CanonicalRequest(
        model=ModelRef.from_values("xai", "responses", "grok-4"),
        messages=[CanonicalMessage(role=MessageRole.USER, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="Search the news")])],
        tools=[
            CanonicalToolDefinition(
                name="web_search",
                provider_extensions={"values": {"type": "web_search", "max_results": 5}},
            )
        ],
    )

    http_request = adapter.build_request(request)

    assert http_request.json_body["tools"][0]["type"] == "web_search"
