from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aster.core.config import RuntimeSettings, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: DEBUG\n")
    settings = load_settings(str(path))
    assert settings.logging.level == "DEBUG"
    assert settings.model.runtime == "mlx"
    assert settings.engine.runtime_kernel == "manual"
    assert settings.api.api_key is None
    assert settings.api.rate_limit_per_minute == 0
    assert settings.api.responses_store_max_entries == 1000
    assert settings.audio.asr.backend == "mlx"
    assert settings.audio.tts.backend == "mlx"
    assert settings.embeddings.backend == "mlx"
    assert settings.embeddings.model == "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


def test_example_config_enables_decode_aware_prefill_budget() -> None:
    settings = load_settings(str(PROJECT_ROOT / "configs" / "config.yaml.example"))

    assert settings.engine.decode_active_prefill_token_budget == 512
    assert settings.engine.decode_tensorized_logprobs_enabled is False
    assert settings.engine.decode_stage_observer_max_events == 0
    assert settings.engine.snapshot_reservation_trace_max_events == 64


def test_snapshot_reservation_trace_capacity_is_bounded() -> None:
    assert RuntimeSettings().engine.snapshot_reservation_trace_max_events == 64
    assert (
        RuntimeSettings.model_validate(
            {"engine": {"snapshot_reservation_trace_max_events": 0}}
        ).engine.snapshot_reservation_trace_max_events
        == 0
    )
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"snapshot_reservation_trace_max_events": -1}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"snapshot_reservation_trace_max_events": 257}})


def test_decode_stage_observer_is_opt_in_and_bounded() -> None:
    assert RuntimeSettings().engine.decode_stage_observer_max_events == 0
    assert (
        RuntimeSettings.model_validate(
            {"engine": {"decode_stage_observer_max_events": 256}}
        ).engine.decode_stage_observer_max_events
        == 256
    )
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"decode_stage_observer_max_events": -1}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"decode_stage_observer_max_events": 257}})


def test_load_settings_reads_responses_store_capacity(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "api:",
                "  responses_store_max_entries: 3",
            ]
        )
    )

    settings = load_settings(str(path))

    assert settings.api.responses_store_max_entries == 3


def test_batch_generator_lane_limit_defaults_to_one_and_is_bounded() -> None:
    assert RuntimeSettings().engine.batch_generator_max_lanes == 1
    assert RuntimeSettings().engine.batch_generator_lane_admission_window_ms == 0.0
    assert RuntimeSettings().engine.batch_generator_lane_target_size == 0
    assert RuntimeSettings().engine.batch_generator_longest_lane_step_quanta == 1
    assert RuntimeSettings().engine.batch_generator_lane_streams is False
    assert RuntimeSettings().engine.chat_prompt_cache_max_entries == 32
    assert RuntimeSettings().engine.snapshot_max_chat_reuse_points == 8
    assert RuntimeSettings().engine.snapshot_chat_reuse_sparse_points == 4
    assert RuntimeSettings().engine.snapshot_chat_reuse_sparse_min_tokens == 2048
    assert RuntimeSettings().engine.snapshot_skip_full_prompt_on_prefix_hit is True
    assert (
        RuntimeSettings.model_validate(
            {
                "engine": {
                    "batch_generator_max_lanes": 2,
                    "batch_generator_lane_admission_window_ms": 200,
                    "batch_generator_lane_target_size": 3,
                    "batch_generator_longest_lane_step_quanta": 2,
                    "batch_generator_lane_streams": True,
                    "chat_prompt_cache_max_entries": 8,
                }
            }
        ).engine.batch_generator_max_lanes
        == 2
    )
    assert (
        RuntimeSettings.model_validate(
            {"engine": {"batch_generator_lane_admission_window_ms": 200}}
        ).engine.batch_generator_lane_admission_window_ms
        == 200
    )
    assert (
        RuntimeSettings.model_validate(
            {"engine": {"batch_generator_lane_target_size": 3}}
        ).engine.batch_generator_lane_target_size
        == 3
    )
    assert (
        RuntimeSettings.model_validate(
            {"engine": {"batch_generator_longest_lane_step_quanta": 2}}
        ).engine.batch_generator_longest_lane_step_quanta
        == 2
    )
    assert (
        RuntimeSettings.model_validate(
            {"engine": {"batch_generator_lane_streams": True}}
        ).engine.batch_generator_lane_streams
        is True
    )
    assert (
        RuntimeSettings.model_validate(
            {"engine": {"chat_prompt_cache_max_entries": 8}}
        ).engine.chat_prompt_cache_max_entries
        == 8
    )

    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"batch_generator_max_lanes": 0}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"batch_generator_max_lanes": 2}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"batch_generator_lane_admission_window_ms": -1}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"batch_generator_lane_target_size": -1}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"batch_generator_longest_lane_step_quanta": 0}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"chat_prompt_cache_max_entries": -1}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"snapshot_max_chat_reuse_points": -1}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"snapshot_chat_reuse_sparse_points": -1}})
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"engine": {"snapshot_chat_reuse_sparse_min_tokens": -1}})


def test_load_settings_preserves_legacy_runtime_as_ignored_metadata(tmp_path: Path) -> None:
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
    assert settings.engine.prefill_token_budget == 1024
    assert settings.deprecation_warnings == (
        "model.runtime is legacy metadata and is ignored by Aster's built-in "
        "vllm-mlx-compatible runtime.",
        "vllm_mlx.* settings are deprecated and only used as engine shims.",
    )


def test_load_settings_normalizes_legacy_audio_backends(tmp_path: Path) -> None:
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
    assert settings.audio.asr.backend == "mlx"
    assert settings.audio.tts.backend == "mlx"


def test_load_settings_normalizes_legacy_embeddings_backend(tmp_path: Path) -> None:
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
    assert settings.embeddings.backend == "mlx"
    assert settings.embeddings.model == "local-embedder"
