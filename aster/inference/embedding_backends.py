from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx
from mlx_lm import load as load_mlx_lm

from aster.core.config import EmbeddingsSettings, RuntimeSettings
from aster.core.errors import AsterError, ConfigurationError
from aster.inference.vllm_mlx_client import VLLMMLXClient
from aster.telemetry.logging import get_logger

class EmbeddingBackend(ABC):
    @abstractmethod
    def supports_embeddings(self) -> bool:
        pass

    @abstractmethod
    def configured_model(self) -> str | None:
        pass

    @abstractmethod
    async def embeddings(self, *, model: str | None, input_data: str | list[str]) -> dict[str, Any]:
        pass

    @abstractmethod
    async def aclose(self) -> None:
        pass


class DisabledEmbeddingBackend(EmbeddingBackend):
    def supports_embeddings(self) -> bool:
        return False

    def configured_model(self) -> str | None:
        return None

    async def embeddings(self, *, model: str | None, input_data: str | list[str]) -> dict[str, Any]:
        del model, input_data
        raise AsterError(
            code="embeddings_disabled",
            message="Embeddings are disabled in the current Aster configuration",
            status_code=501,
        )

    async def aclose(self) -> None:
        return None


class MLXEmbeddingBackend(EmbeddingBackend):
    def __init__(self, settings: EmbeddingsSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded_model_ref: str | None = None

    def supports_embeddings(self) -> bool:
        return self.settings.enabled

    def configured_model(self) -> str | None:
        return self.settings.model

    async def aclose(self) -> None:
        return None

    async def embeddings(self, *, model: str | None, input_data: str | list[str]) -> dict[str, Any]:
        texts = [input_data] if isinstance(input_data, str) else list(input_data)
        if not texts:
            raise AsterError(
                code="empty_embedding_input",
                message="Embedding input cannot be empty",
                status_code=400,
            )
        return await asyncio.to_thread(self._embed_sync, model, texts)

    def _embed_sync(self, requested_model: str | None, texts: list[str]) -> dict[str, Any]:
        model_ref = requested_model or self.settings.model_path or self.settings.model
        if not model_ref:
            raise AsterError(
                code="embedding_model_not_configured",
                message="No MLX embedding model configured",
                status_code=400,
            )

        with self._lock:
            model, tokenizer = self._ensure_loaded(model_ref)
            enc = tokenizer._tokenizer(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.settings.max_length,
            )
            input_ids = mx.array(enc["input_ids"])
            attention_mask = enc["attention_mask"]
            hidden = model.model(input_ids)

            lengths = attention_mask.sum(axis=1) - 1
            pooled = []
            for index, last_idx in enumerate(lengths.tolist()):
                pooled.append(hidden[index, int(last_idx), :])
            pooled_array = mx.stack(pooled)
            norms = mx.sqrt(mx.sum(pooled_array * pooled_array, axis=1, keepdims=True))
            embeddings = (pooled_array / norms).tolist()
            prompt_tokens = int(attention_mask.sum())

        model_name = requested_model or self.settings.model
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": idx, "embedding": vector}
                for idx, vector in enumerate(embeddings)
            ],
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    def _ensure_loaded(self, model_ref: str) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None or self._loaded_model_ref != model_ref:
            self.logger.info(
                "mlx_embedding_model_loading",
                extra={"model_ref": model_ref},
            )
            self._model, self._tokenizer = load_mlx_lm(model_ref, lazy=False)
            self._loaded_model_ref = model_ref
            self.logger.info(
                "mlx_embedding_model_loaded",
                extra={
                    "model_ref": model_ref,
                    "configured_model": self.settings.model,
                    "dimensions": self.settings.dimensions,
                },
            )
        return self._model, self._tokenizer


class VLLMMLXEmbeddingBackend(EmbeddingBackend):
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.client = VLLMMLXClient(settings.vllm_mlx)

    def supports_embeddings(self) -> bool:
        return self.settings.embeddings.enabled and bool(
            self.settings.embeddings.model_path
            or self.settings.embeddings.model
        )

    def configured_model(self) -> str | None:
        return self.settings.embeddings.model

    async def embeddings(self, *, model: str | None, input_data: str | list[str]) -> dict[str, Any]:
        resolved_model = self._resolve_model_name(model)
        if not resolved_model:
            raise AsterError(
                code="embedding_model_not_configured",
                message="No embedding model configured for vllm_mlx backend",
                status_code=400,
            )
        return await self.client.embeddings(model=resolved_model, input_data=input_data)

    async def aclose(self) -> None:
        await self.client.aclose()

    def _resolve_model_name(self, requested_model: str | None) -> str | None:
        configured_path = self.settings.embeddings.model_path
        configured_alias = self.settings.embeddings.model
        requested = (requested_model or "").strip()

        if not requested or requested == "default":
            return configured_path or configured_alias

        if configured_alias and requested == configured_alias:
            return configured_path or configured_alias

        if configured_path and requested == configured_path:
            return configured_path

        return requested


def build_embedding_backend(settings: RuntimeSettings) -> EmbeddingBackend:
    if not settings.embeddings.enabled:
        return DisabledEmbeddingBackend()
    if settings.embeddings.backend == "mlx":
        return MLXEmbeddingBackend(settings.embeddings)
    if settings.embeddings.backend == "vllm_mlx":
        return VLLMMLXEmbeddingBackend(settings)
    raise ConfigurationError(
        code="unsupported_embedding_backend",
        message=f"Unsupported embeddings backend: {settings.embeddings.backend}",
    )


__all__ = [
    "EmbeddingBackend",
    "DisabledEmbeddingBackend",
    "MLXEmbeddingBackend",
    "VLLMMLXEmbeddingBackend",
    "build_embedding_backend",
]
