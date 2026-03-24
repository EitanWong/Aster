from __future__ import annotations

from aster.core.capabilities import build_capabilities, feature_flags
from aster.core.canonical import CapabilitySupport, ProviderName, ProviderRef
from aster.providers.anthropic.models import MESSAGES_DOCS


def messages_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.ANTHROPIC, api_family="messages")
    return build_capabilities(
        provider=provider,
        auth_scheme="x-api-key",
        endpoint_family="messages",
        features=feature_flags(
            multimodal_input=CapabilitySupport.SUPPORTED,
            streaming=CapabilitySupport.SUPPORTED,
            user_defined_tools=CapabilitySupport.SUPPORTED,
            reasoning_controls=CapabilitySupport.PROVIDER_SPECIFIC,
        ),
        required_headers=["x-api-key", "anthropic-version"],
        beta_headers=["anthropic-beta"],
        versioning_strategy="required_header",
        supported_stop_reasons=["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"],
        extension_fields=["anthropic-beta", "metadata", "thinking"],
        notes=["System prompt text is separate from the messages array.", "Tool results are represented as user content blocks, not a native tool role."],
        docs_urls=MESSAGES_DOCS,
    )
