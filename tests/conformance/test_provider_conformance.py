from __future__ import annotations

from aster.core.canonical import (
    CanonicalContentPart,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalToolResult,
    ContentPartType,
    MessageRole,
    ModelRef,
)
from aster.providers.anthropic import AnthropicMessagesAdapter
from aster.providers.bedrock import BedrockConverseAdapter
from aster.providers.gemini import GeminiGenerateContentAdapter
from aster.providers.openai import OpenAIResponsesAdapter


def test_same_canonical_system_prompt_maps_differently_by_provider() -> None:
    request = CanonicalRequest(
        model=ModelRef.from_values("openai", "responses", "gpt-5.1"),
        messages=[
            CanonicalMessage(role=MessageRole.SYSTEM, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="System prompt")]),
            CanonicalMessage(role=MessageRole.USER, content=[CanonicalContentPart(type=ContentPartType.TEXT, text="Hello")]),
        ],
    )

    openai_body = OpenAIResponsesAdapter().build_request(request).json_body

    anthropic_request = request.model_copy(deep=True)
    anthropic_request.model = ModelRef.from_values("anthropic", "messages", "claude-sonnet-4-5")
    anthropic_body = AnthropicMessagesAdapter().build_request(anthropic_request).json_body

    gemini_request = request.model_copy(deep=True)
    gemini_request.model = ModelRef.from_values("gemini", "generate_content", "gemini-2.5-flash")
    gemini_body = GeminiGenerateContentAdapter().build_request(gemini_request).json_body

    assert openai_body["input"][0]["role"] == "system"
    assert anthropic_body["system"] == "System prompt"
    assert gemini_body["systemInstruction"]["parts"][0]["text"] == "System prompt"


def test_same_tool_result_maps_to_provider_native_shapes() -> None:
    base_request = CanonicalRequest(
        model=ModelRef.from_values("anthropic", "messages", "claude-sonnet-4-5"),
        messages=[
            CanonicalMessage(
                role=MessageRole.TOOL,
                tool_results=[CanonicalToolResult(call_id="call_1", output={"temp_c": 22})],
            )
        ],
    )

    anthropic_body = AnthropicMessagesAdapter().build_request(base_request).json_body

    gemini_request = base_request.model_copy(deep=True)
    gemini_request.model = ModelRef.from_values("gemini", "generate_content", "gemini-2.5-flash")
    gemini_body = GeminiGenerateContentAdapter().build_request(gemini_request).json_body

    bedrock_request = base_request.model_copy(deep=True)
    bedrock_request.model = ModelRef.from_values("bedrock", "converse", "anthropic.claude-3-5-sonnet")
    bedrock_body = BedrockConverseAdapter().build_request(bedrock_request).json_body

    assert anthropic_body["messages"][0]["content"][0]["type"] == "tool_result"
    assert "functionResponse" in gemini_body["contents"][0]["parts"][0]
    assert "toolResult" in bedrock_body["messages"][0]["content"][0]
