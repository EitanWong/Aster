from __future__ import annotations

from aster.core.capabilities import build_capabilities, feature_flags
from aster.core.canonical import CapabilitySupport, ProviderName, ProviderRef
from aster.providers.openai.models import CHAT_DOCS, RESPONSES_DOCS


def responses_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.OPENAI, api_family="responses")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="responses",
        features=feature_flags(
            multimodal_input=CapabilitySupport.SUPPORTED,
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
            built_in_tools=CapabilitySupport.SUPPORTED,
            parallel_tool_calls=CapabilitySupport.SUPPORTED,
            conversation_state=CapabilitySupport.SUPPORTED,
            background_mode=CapabilitySupport.BETA,
            reasoning_controls=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["Authorization"],
        supported_stop_reasons=["completed", "max_output_tokens", "tool_calls", "content_filter"],
        extension_fields=["text", "reasoning", "background", "metadata", "store"],
        docs_urls=RESPONSES_DOCS,
    )


def chat_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.OPENAI, api_family="chat_completions")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="chat_completions",
        features=feature_flags(
            multimodal_input=CapabilitySupport.SUPPORTED,
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
            parallel_tool_calls=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["Authorization"],
        supported_stop_reasons=["stop", "length", "tool_calls", "content_filter", "function_call"],
        extension_fields=["response_format", "metadata"],
        notes=["Chat Completions is retained as a compatibility surface; Responses is the primary modern OpenAI path."],
        docs_urls=CHAT_DOCS,
    )
