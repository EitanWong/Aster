from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalError, ErrorCategory, ProviderRef, RawProviderEnvelope


def map_error(
    provider: ProviderRef,
    status_code: int,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str] | None = None,
) -> CanonicalError:
    message = "Gemini request failed"
    details = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(details, Mapping) and isinstance(details.get("message"), str):
        message = details["message"]
    return CanonicalError(
        provider=provider,
        category=_category(status_code),
        code=str(details.get("status") if isinstance(details, Mapping) else "provider_error"),
        message=message,
        status_code=status_code,
        retryable=status_code in {408, 429} or status_code >= 500,
        headers=dict(headers or {}),
        raw_provider_envelope=RawProviderEnvelope(
            provider=provider.name,
            api_family=provider.api_family,
            status_code=status_code,
            headers=dict(headers or {}),
            payload=dict(payload or {}),
        ),
    )


def _category(status_code: int) -> ErrorCategory:
    if status_code in {401, 403}:
        return ErrorCategory.AUTH
    if status_code == 429:
        return ErrorCategory.RATE_LIMIT
    if status_code == 408:
        return ErrorCategory.TIMEOUT
    if status_code == 400:
        return ErrorCategory.VALIDATION
    if status_code >= 500:
        return ErrorCategory.TRANSIENT
    return ErrorCategory.PROVIDER
