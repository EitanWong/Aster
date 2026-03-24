from __future__ import annotations

import asyncio

from aster.core.config import RuntimeSettings
from aster.inference.embedding_backends import VLLMMLXEmbeddingBackend


def test_vllm_embedding_backend_maps_default_and_alias_to_local_path() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "vllm_mlx": {
                "embedding_model": "/tmp/embed-model",
            },
            "embeddings": {
                "enabled": True,
                "backend": "vllm_mlx",
                "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
                "model_path": "/tmp/embed-model",
            },
        }
    )
    backend = VLLMMLXEmbeddingBackend(settings)
    seen: list[str] = []

    async def fake_embeddings(*, model: str, input_data: str | list[str]) -> dict[str, object]:
        seen.append(model)
        return {"object": "list", "data": [], "model": model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}

    backend.client.embeddings = fake_embeddings  # type: ignore[method-assign]

    async def run_test() -> None:
        try:
            await backend.embeddings(model=None, input_data="hello")
            await backend.embeddings(model="default", input_data="hello")
            await backend.embeddings(model="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", input_data="hello")
            await backend.embeddings(model="/tmp/embed-model", input_data="hello")
        finally:
            await backend.aclose()

    asyncio.run(run_test())

    assert seen == ["/tmp/embed-model", "/tmp/embed-model", "/tmp/embed-model", "/tmp/embed-model"]
