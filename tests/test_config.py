from __future__ import annotations

from pathlib import Path

from aster.core.config import load_settings


def test_load_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: DEBUG\n")
    settings = load_settings(str(path))
    assert settings.logging.level == "DEBUG"
    assert settings.model.runtime == "mlx"
    assert settings.audio.asr_backend == "mlx"
    assert settings.audio.tts_backend == "mlx"
    assert settings.embeddings.backend == "mlx"
    assert settings.embeddings.model == "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


def test_load_settings_with_vllm_mlx_runtime(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  runtime: vllm_mlx",
                "vllm_mlx:",
                "  base_url: http://127.0.0.1:9000",
            ]
        )
    )
    settings = load_settings(str(path))
    assert settings.model.runtime == "vllm_mlx"
    assert settings.vllm_mlx.base_url == "http://127.0.0.1:9000"


def test_load_settings_with_mixed_audio_backends(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "audio:",
                "  asr_backend: vllm_mlx",
                "  tts_backend: mlx",
            ]
        )
    )
    settings = load_settings(str(path))
    assert settings.audio.asr_backend == "vllm_mlx"
    assert settings.audio.tts_backend == "mlx"


def test_load_settings_with_vllm_embeddings_backend(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "embeddings:",
                "  backend: vllm_mlx",
                "  model: local-embedder",
            ]
        )
    )
    settings = load_settings(str(path))
    assert settings.embeddings.backend == "vllm_mlx"
    assert settings.embeddings.model == "local-embedder"
