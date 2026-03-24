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
    message = str(error.get("message") or "OpenAI request failed")
    category = _category(status_code, error_type)
    return CanonicalError(
        provider=provider,
        category=category,
        code=str(error.get("code") or error_type),
        message=message,
        status_code=status_code,
        retryable=category in {ErrorCategory.RATE_LIMIT, ErrorCategory.TRANSIENT, ErrorCategory.TIMEOUT},
        request_id=_request_id(headers),
        headers=dict(headers or {}),
        provider_extensions={"values": {"error_type": error_type}},
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
    if status_code == 408:
        return ErrorCategory.TIMEOUT
    if status_code == 429:
        return ErrorCategory.RATE_LIMIT
    if status_code >= 500 or "server" in error_type:
        return ErrorCategory.TRANSIENT
    if status_code == 400:
        return ErrorCategory.VALIDATION
    return ErrorCategory.PROVIDER


def _request_id(headers: Mapping[str, str] | None) -> str | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    return lowered.get("x-request-id")
