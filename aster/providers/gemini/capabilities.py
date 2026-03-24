from __future__ import annotations

from aster.core.capabilities import build_capabilities, feature_flags
from aster.core.canonical import CapabilitySupport, ProviderName, ProviderRef
from aster.providers.gemini.models import GENERATE_CONTENT_DOCS


def generate_content_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.GEMINI, api_family="generate_content")
    return build_capabilities(
        provider=provider,
        auth_scheme="x-goog-api-key",
        endpoint_family="generate_content",
        features=feature_flags(
            multimodal_input=CapabilitySupport.SUPPORTED,
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
            reasoning_controls=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["x-goog-api-key"],
        supported_stop_reasons=["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"],
        extension_fields=["generationConfig", "toolConfig", "systemInstruction", "promptFeedback"],
        docs_urls=GENERATE_CONTENT_DOCS,
    )


def openai_chat_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.GEMINI, api_family="openai_chat")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="openai_chat",
        features=feature_flags(
            multimodal_input=CapabilitySupport.SUPPORTED,
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["Authorization"],
        notes=["This is Gemini's OpenAI compatibility surface, not the canonical Gemini adapter path."],
        docs_urls=GENERATE_CONTENT_DOCS,
    )
