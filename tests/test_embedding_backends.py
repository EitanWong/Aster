from __future__ import annotations

from aster.core.config import RuntimeSettings
from aster.inference.embedding_backends import MLXEmbeddingBackend, build_embedding_backend


def test_legacy_vllm_embedding_backend_is_normalized_to_mlx() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "embeddings": {
                "enabled": True,
                "backend": "mlx",
                "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
                "model_path": "/tmp/embed-model",
            },
        }
    )
    backend = build_embedding_backend(settings)
    assert isinstance(backend, MLXEmbeddingBackend)
    assert backend.configured_model() == "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
