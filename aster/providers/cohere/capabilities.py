from __future__ import annotations

from aster.core.capabilities import build_capabilities, feature_flags
from aster.core.canonical import CapabilitySupport, ProviderName, ProviderRef
from aster.providers.cohere.models import COHERE_DOCS


def chat_v2_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.COHERE, api_family="chat_v2")
    return build_capabilities(
        provider=provider,
        auth_scheme="bearer",
        endpoint_family="chat_v2",
        features=feature_flags(
            streaming=CapabilitySupport.SUPPORTED,
            structured_outputs=CapabilitySupport.PROVIDER_SPECIFIC,
            user_defined_tools=CapabilitySupport.SUPPORTED,
        ),
        required_headers=["Authorization"],
        docs_urls=COHERE_DOCS,
    )
