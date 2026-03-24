from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalFinalResponse,
    CanonicalRequest,
    CanonicalResponseChunk,
    ProviderHttpRequest,
    ProviderRequestContext,
    ProviderStreamEvent,
)
from aster.runtime.provider_registry import ProviderRegistry
from aster.runtime.routing import ProviderRouter


class ProviderDispatcher:
    def __init__(self, registry: ProviderRegistry) -> None:
        self.router = ProviderRouter(registry)

    def prepare(
        self,
        request: CanonicalRequest,
        context: ProviderRequestContext | None = None,
    ) -> ProviderHttpRequest:
        adapter = self.router.resolve(request.model)
        return adapter.build_request(request, context=context)

    def parse_response(
        self,
        request: CanonicalRequest,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> CanonicalFinalResponse:
        adapter = self.router.resolve(request.model)
        return adapter.parse_response(payload, headers=headers)

    def decode_stream(
        self,
        request: CanonicalRequest,
        event: ProviderStreamEvent,
    ) -> list[CanonicalResponseChunk]:
        adapter = self.router.resolve(request.model)
        return adapter.decode_stream_event(event)
