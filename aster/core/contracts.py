from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalError,
    CanonicalFinalResponse,
    CanonicalRequest,
    CanonicalResponseChunk,
    ProviderCapabilities,
    ProviderHttpRequest,
    ProviderRef,
    ProviderRequestContext,
    ProviderStreamEvent,
)


class ProviderAdapter(ABC):
    provider_name: str
    api_family: str
    docs_urls: tuple[str, ...]

    @property
    def provider_ref(self) -> ProviderRef:
        raise NotImplementedError

    @property
    def adapter_id(self) -> str:
        return self.provider_ref.adapter_id

    @abstractmethod
    def capabilities(self, model_id: str | None = None) -> ProviderCapabilities:
        raise NotImplementedError

    @abstractmethod
    def build_request(
        self,
        request: CanonicalRequest,
        context: ProviderRequestContext | None = None,
    ) -> ProviderHttpRequest:
        raise NotImplementedError

    @abstractmethod
    def parse_response(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalFinalResponse:
        raise NotImplementedError

    @abstractmethod
    def decode_stream_event(self, event: ProviderStreamEvent) -> list[CanonicalResponseChunk]:
        raise NotImplementedError

    @abstractmethod
    def map_error(
        self,
        status_code: int,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalError:
        raise NotImplementedError

    def merge_headers(
        self,
        base_headers: dict[str, str],
        context: ProviderRequestContext | None,
    ) -> dict[str, str]:
        if context is None:
            return base_headers
        merged = dict(base_headers)
        merged.update(context.extra_headers)
        return merged

    def rate_limit_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        if headers is None:
            return {}
        return {key: value for key, value in headers.items() if "ratelimit" in key.lower()}

    def request_id(
        self,
        headers: Mapping[str, str] | None,
        *keys: str,
    ) -> str | None:
        if headers is None:
            return None
        lowered = {key.lower(): value for key, value in headers.items()}
        for key in keys:
            value = lowered.get(key.lower())
            if value:
                return value
        return None
