from __future__ import annotations

from aster.core.canonical import (
    CapabilitySupport,
    ProviderCapabilities,
    ProviderFeatureFlags,
    ProviderRef,
)


def feature_flags(
    **overrides: CapabilitySupport,
) -> ProviderFeatureFlags:
    return ProviderFeatureFlags(**overrides)


def build_capabilities(
    *,
    provider: ProviderRef,
    auth_scheme: str,
    endpoint_family: str,
    features: ProviderFeatureFlags,
    required_headers: list[str] | None = None,
    beta_headers: list[str] | None = None,
    versioning_strategy: str | None = None,
    supported_stop_reasons: list[str] | None = None,
    notes: list[str] | None = None,
    model_constraints: list[str] | None = None,
    extension_fields: list[str] | None = None,
    docs_urls: tuple[str, ...] = (),
) -> ProviderCapabilities:
    return ProviderCapabilities(
        provider=provider,
        auth_scheme=auth_scheme,
        endpoint_family=endpoint_family,
        features=features,
        required_headers=required_headers or [],
        beta_headers=beta_headers or [],
        versioning_strategy=versioning_strategy,
        supported_stop_reasons=supported_stop_reasons or [],
        notes=notes or [],
        model_constraints=model_constraints or [],
        extension_fields=extension_fields or [],
        docs_urls=docs_urls,
    )
