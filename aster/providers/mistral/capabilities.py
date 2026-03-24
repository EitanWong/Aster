from __future__ import annotations

from aster.core.capabilities import build_capabilities, feature_flags
from aster.core.canonical import CapabilitySupport, ProviderName, ProviderRef
from aster.providers.mistral.models import MISTRAL_DOCS


def chat_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.MISTRAL, api_family="chat_completions")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="chat_completions",
        features=feature_flags(
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["Authorization"],
        docs_urls=MISTRAL_DOCS,
    )


def conversations_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.MISTRAL, api_family="conversations")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="conversations",
        features=feature_flags(
            streaming=CapabilitySupport.BETA,
            user_defined_tools=CapabilitySupport.SUPPORTED,
            conversation_state=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["Authorization"],
        notes=["The agents/conversations surface is modeled separately from chat/completions because lifecycle semantics differ."],
        docs_urls=MISTRAL_DOCS,
    )
