from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import CanonicalError, ErrorCategory, ProviderRef, RawProviderEnvelope


def map_error(provider: ProviderRef, status_code: int, payload: Mapping[str, Any] | None, headers: Mapping[str, str] | None = None) -> CanonicalError:
    code = "provider_error"
    message = "Bedrock request failed"
    if isinstance(payload, Mapping):
        for key in ("message", "__type", "code"):
            if isinstance(payload.get(key), str):
                if key == "message":
                    message = payload[key]
                else:
                    code = payload[key]
    return CanonicalError(
        provider=provider,
        category=_category(status_code),
        code=code,
        message=message,
        status_code=status_code,
        retryable=status_code in {408, 429} or status_code >= 500,
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


def _request_id(headers: Mapping[str, str] | None) -> str | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    return lowered.get("x-amzn-requestid")
