from __future__ import annotations

from aster.core.capabilities import build_capabilities, feature_flags
from aster.core.canonical import CapabilitySupport, ProviderName, ProviderRef
from aster.providers.bedrock.models import CONVERSE_DOCS


def converse_capabilities() -> object:
    provider = ProviderRef(name=ProviderName.BEDROCK, api_family="converse")
    return build_capabilities(
        provider=provider,
        auth_scheme="sigv4",
        endpoint_family="converse",
        features=feature_flags(
            multimodal_input=CapabilitySupport.MODEL_DEPENDENT,
            streaming=CapabilitySupport.MODEL_DEPENDENT,
            user_defined_tools=CapabilitySupport.MODEL_DEPENDENT,
            conversation_state=CapabilitySupport.UNSUPPORTED,
        ),
        versioning_strategy="aws_model_catalog",
        supported_stop_reasons=["end_turn", "tool_use", "max_tokens", "guardrail_intervened", "content_filtered"],
        extension_fields=["additionalModelRequestFields", "additionalModelResponseFields", "guardrailConfig", "performanceConfig"],
        notes=["Capabilities depend on the underlying hosted model family."],
        docs_urls=CONVERSE_DOCS,
    )


def openai_compat_capabilities(api_family: str) -> object:
    provider = ProviderRef(name=ProviderName.BEDROCK, api_family=api_family)
    return build_capabilities(
        provider=provider,
        auth_scheme="sigv4",
        endpoint_family=api_family,
        features=feature_flags(
            multimodal_input=CapabilitySupport.MODEL_DEPENDENT,
            streaming=CapabilitySupport.MODEL_DEPENDENT,
            structured_outputs=CapabilitySupport.MODEL_DEPENDENT,
            user_defined_tools=CapabilitySupport.MODEL_DEPENDENT,
        ),
        notes=["This is Bedrock's OpenAI-compatible surface; native Bedrock semantics remain separate in the Converse adapter."],
        docs_urls=CONVERSE_DOCS,
    )
