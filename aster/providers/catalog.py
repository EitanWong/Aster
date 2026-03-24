from __future__ import annotations

from aster.providers.anthropic import AnthropicMessagesAdapter
from aster.providers.bedrock import BedrockConverseAdapter, BedrockOpenAIChatAdapter, BedrockOpenAIResponsesAdapter
from aster.providers.cohere import CohereChatV2Adapter
from aster.providers.gemini import GeminiGenerateContentAdapter, GeminiOpenAIChatAdapter
from aster.providers.mistral import MistralChatCompletionsAdapter, MistralConversationsAdapter
from aster.providers.openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from aster.providers.registry import ProviderRegistry
from aster.providers.xai import XAIChatCompletionsAdapter, XAIResponsesAdapter


def build_default_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    for adapter in (
        OpenAIResponsesAdapter(),
        OpenAIChatCompletionsAdapter(),
        AnthropicMessagesAdapter(),
        GeminiGenerateContentAdapter(),
        GeminiOpenAIChatAdapter(),
        BedrockConverseAdapter(),
        BedrockOpenAIResponsesAdapter(),
        BedrockOpenAIChatAdapter(),
        MistralChatCompletionsAdapter(),
        MistralConversationsAdapter(),
        CohereChatV2Adapter(),
        XAIResponsesAdapter(),
        XAIChatCompletionsAdapter(),
    ):
        registry.register(adapter)
    return registry
