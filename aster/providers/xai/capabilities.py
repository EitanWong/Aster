from __future__ import annotations

from aster.core.capabilities import build_capabilities, feature_flags
from aster.core.canonical import CapabilitySupport, ProviderName, ProviderRef
from aster.providers.xai.models import XAI_DOCS


def responses_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.XAI, api_family="responses")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="responses",
        features=feature_flags(
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
            built_in_tools=CapabilitySupport.SUPPORTED,
            reasoning_controls=CapabilitySupport.PROVIDER_SPECIFIC,
        ),
        required_headers=["Authorization"],
        extension_fields=["citations", "reasoning", "store", "truncation"],
        notes=["xAI supports provider-native tools such as web search; do not treat these as generic OpenAI built-ins."],
        docs_urls=XAI_DOCS,
    )


def chat_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.XAI, api_family="chat_completions")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="chat_completions",
        features=feature_flags(
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
            built_in_tools=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["Authorization"],
        notes=["Legacy chat/completions remains separate from xAI's preferred modern inference style."],
        docs_urls=XAI_DOCS,
    )
