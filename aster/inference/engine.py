from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from aster.cache.paged_kv_cache import PagedKVCache
from aster.cache.prefix_cache import PrefixCache
from aster.core.config import RuntimeSettings
from aster.inference.backends import build_inference_backend
from aster.inference.embedding_backends import build_embedding_backend
from aster.inference.contracts import InferenceRequest, InferenceResponse
from aster.inference.decode_engine import DecodeChunk
from aster.scheduler.policy_engine import PolicyEngine
from aster.telemetry.metrics import MetricsRegistry


class InferenceEngine:
    def __init__(
        self,
        settings: RuntimeSettings,
        metrics: MetricsRegistry,
        kv_cache: PagedKVCache,
        prefix_cache: PrefixCache,
        policy_engine: PolicyEngine,
    ) -> None:
        self.settings = settings
        self.metrics = metrics
        self.policy_engine = policy_engine
        self.kv_cache = kv_cache
        self.prefix_cache = prefix_cache
        self.backend = build_inference_backend(settings, metrics, kv_cache, prefix_cache, policy_engine)
        self.embedding_backend = build_embedding_backend(settings)

    def health(self) -> bool:
        return self.backend.health()

    def supports_concurrent_dispatch(self) -> bool:
        return self.backend.capabilities.supports_concurrent_dispatch

    def supports_embeddings(self) -> bool:
        return self.embedding_backend.supports_embeddings()

    def configured_embedding_model(self) -> str | None:
        return self.embedding_backend.configured_model()

    async def aclose(self) -> None:
        await self.backend.aclose()
        await self.embedding_backend.aclose()

    async def warmup(self) -> None:
        """Pre-warm expensive resources (tokenizer, etc.) in background threads."""
        if hasattr(self.backend, "warmup"):
            await self.backend.warmup()

    async def embeddings(self, *, model: str | None, input_data: str | list[str]) -> dict[str, Any]:
        return await self.embedding_backend.embeddings(model=model, input_data=input_data)

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        return await self.backend.infer(request)

    async def stream(self, request: InferenceRequest) -> AsyncIterator[DecodeChunk]:
        async for chunk in self.backend.stream(request):
            yield chunk


__all__ = ["InferenceEngine", "InferenceRequest", "InferenceResponse"]
