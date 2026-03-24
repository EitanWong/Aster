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
    error = payload.get("error", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(error, Mapping):
        error = {}
    error_type = str(error.get("type") or "provider_error")
    return CanonicalError(
        provider=provider,
        category=_category(status_code, error_type),
        code=error_type,
        message=str(error.get("message") or "Anthropic request failed"),
        status_code=status_code,
        retryable=status_code in {408, 409, 429} or status_code >= 500,
        request_id=_request_id(headers),
        headers=dict(headers or {}),
        raw_provider_envelope=RawProviderEnvelope(
            provider=provider.name,
            api_family=provider.api_family,
            status_code=status_code,
            headers=dict(headers or {}),
            payload=dict(payload or {}),
        ),
    )


def _category(status_code: int, error_type: str) -> ErrorCategory:
    if status_code in {401, 403}:
        return ErrorCategory.AUTH
    if status_code == 429:
        return ErrorCategory.RATE_LIMIT
    if status_code == 408:
        return ErrorCategory.TIMEOUT
    if status_code >= 500 or error_type in {"api_error", "overloaded_error"}:
        return ErrorCategory.TRANSIENT
    if status_code == 400:
        return ErrorCategory.VALIDATION
    return ErrorCategory.PROVIDER


def _request_id(headers: Mapping[str, str] | None) -> str | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    return lowered.get("request-id")
