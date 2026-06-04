from __future__ import annotations

import pytest

from aster.__main__ import build_parser, settings_from_args, settings_from_serve_args


def test_serve_cli_maps_vllm_mlx_startup_flags_to_settings() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "/models/qwen",
            "--served-model-name",
            "qwen",
            "--host",
            "0.0.0.0",
            "--port",
            "18000",
            "--max-num-seqs",
            "8",
            "--prefill-batch-size",
            "2",
            "--completion-batch-size",
            "4",
            "--chunked-prefill-tokens",
            "512",
            "--stream-interval",
            "3",
            "--max-request-tokens",
            "2048",
            "--timeout",
            "42",
            "--disable-prefix-cache",
            "--embedding-model",
            "embed-model",
        ]
    )

    settings = settings_from_serve_args(args)

    assert settings.model.name == "qwen"
    assert settings.model.path == "/models/qwen"
    assert settings.model.context_length == 16384
    assert settings.model.enable_thinking is True
    assert settings.api.host == "0.0.0.0"
    assert settings.api.port == 18000
    assert settings.api.request_timeout_seconds == 42
    assert settings.api.max_request_tokens == 2048
    assert settings.engine.max_active_requests == 8
    assert settings.engine.max_decode_batch == 4
    assert settings.engine.prefill_token_budget == 512
    assert settings.engine.stream_interval_tokens == 3
    assert settings.engine.prefix_cache_enabled is False
    assert settings.cache.prefix_cache_enabled is False
    assert settings.batch.prefill_batch_size == 2
    assert settings.batch.decode_batch_size == 4
    assert settings.embeddings.enabled is True
    assert settings.embeddings.model == "embed-model"


def test_serve_cli_max_request_tokens_does_not_reduce_context_length() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "/models/qwen",
            "--max-request-tokens",
            "128",
        ]
    )

    settings = settings_from_serve_args(args)

    assert settings.api.max_request_tokens == 128
    assert settings.model.context_length == 16384


def test_serve_cli_default_chat_template_kwargs_can_disable_thinking() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "/models/qwen",
            "--default-chat-template-kwargs",
            '{"enable_thinking": false}',
        ]
    )

    settings = settings_from_serve_args(args)

    assert settings.model.enable_thinking is False


def test_serve_cli_accepts_unimplemented_vllm_mlx_flags_with_warning() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "/models/qwen",
            "--continuous-batching",
            "--enable-metrics",
            "--tool-call-parser",
            "qwen",
            "--mllm",
        ]
    )

    settings = settings_from_serve_args(args)

    assert settings.model.path == "/models/qwen"
    assert settings.deprecation_warnings
    assert "continuous_batching" in settings.deprecation_warnings[0]
    assert "enable_metrics" in settings.deprecation_warnings[0]
    assert "tool_call_parser" in settings.deprecation_warnings[0]
    assert "mllm" in settings.deprecation_warnings[0]


def test_legacy_config_path_still_supported(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: config-model\n")

    parser = build_parser()
    args = parser.parse_args(["--config", str(config_path)])
    settings = settings_from_args(args)

    assert settings.model.name == "config-model"


def test_cli_requires_serve_model_or_config() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    with pytest.raises(SystemExit):
        settings_from_args(args)
