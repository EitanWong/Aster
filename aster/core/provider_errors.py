from __future__ import annotations

from typing import Any

from aster.core.canonical import CanonicalError, ErrorCategory, ProviderRef, RawProviderEnvelope


class ProviderContractError(Exception):
    def __init__(self, error: CanonicalError) -> None:
        super().__init__(error.message)
        self.error = error

    @classmethod
    def build(
        cls,
        *,
        provider: ProviderRef | None,
        category: ErrorCategory,
        code: str,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
        request_id: str | None = None,
        headers: dict[str, str] | None = None,
        raw_payload: dict[str, Any] | str | None = None,
        provider_extensions: dict[str, Any] | None = None,
    ) -> "ProviderContractError":
        envelope = None
        if provider is not None and (raw_payload is not None or headers or status_code is not None):
            envelope = RawProviderEnvelope(
                provider=provider.name,
                api_family=provider.api_family,
                status_code=status_code,
                headers=headers or {},
                payload=raw_payload,
            )
        return cls(
            CanonicalError(
                provider=provider,
                category=category,
                code=code,
                message=message,
                status_code=status_code,
                retryable=retryable,
                request_id=request_id,
                headers=headers or {},
                provider_extensions={"values": provider_extensions or {}},
                raw_provider_envelope=envelope,
            )
        )


class UnsupportedProviderFeatureError(ProviderContractError):
    pass


class ProviderPayloadValidationError(ProviderContractError):
    pass


class ProviderStreamDecodeError(ProviderContractError):
    pass
