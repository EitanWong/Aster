from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from aster.core.errors import ConfigurationError


class APISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    max_queue_depth: int = 128
    request_timeout_seconds: float = 180.0
    max_request_tokens: int = 32768
    api_key: str | None = None
    rate_limit_per_minute: int = 0
    responses_store_max_entries: int = Field(default=1000, ge=1)


class ModelSettings(BaseModel):
    name: str = "Qwen3.5-9B"
    path: str = "models/qwen3.5-9b"
    runtime: Literal["mlx", "vllm_mlx"] = "mlx"
    context_length: int = 16384
    enable_thinking: bool = False


class SpeculativeSettings(BaseModel):
    enabled: bool = False
    draft_name: str = "Qwen3.5-0.8B"
    draft_path: str = "models/qwen3.5-0.8b"
    max_draft_tokens: int = 0
    min_acceptance_rate: float = 0.45
    min_speedup_ratio: float = 1.05
    auto_disable_on_regression: bool = True


class EngineSettings(BaseModel):
    engine_type: Literal["manual", "batched"] = "manual"
    runtime_kernel: Literal["manual", "batch_generator"] = "manual"
    max_active_requests: int = 16
    batch_generator_max_lanes: int = Field(default=1, ge=1)
    batch_generator_lane_admission_window_ms: float = Field(default=0.0, ge=0.0)
    batch_generator_lane_target_size: int = Field(default=0, ge=0)
    batch_generator_longest_lane_step_quanta: int = Field(default=1, ge=1)
    chat_prompt_cache_max_entries: int = Field(default=32, ge=0)
    max_decode_batch: int = 4
    prefill_token_budget: int = 1024
    idle_prefill_token_limit: int = 4096
    pressure_prefill_token_budget: int = 512
    admission_retry_limit: int = 16
    snapshot_budget_bytes: int = 8 * 1024 * 1024 * 1024
    snapshot_min_prefix_tokens: int = 32
    snapshot_chunk_checkpoint_max_tokens: int = 0
    snapshot_max_entries: int = 256
    snapshot_max_chat_reuse_points: int = Field(default=8, ge=0)
    snapshot_chat_reuse_sparse_points: int = Field(default=4, ge=0)
    snapshot_chat_reuse_sparse_min_tokens: int = Field(default=2048, ge=0)
    kv_cache_step_tokens: int = 2048
    prefix_cache_enabled: bool = True
    prefix_cache_persist_path: str | None = None
    prefix_cache_load_on_warmup: bool = True
    prefix_cache_save_on_shutdown: bool = True
    warm_prompts_path: str | None = None
    warm_prompts_max_tokens: int = 1
    warm_prompts_concurrency: int = 1
    stream_interval_tokens: int = 1
    memory_headroom_ratio: float = 0.10
    paged_cache_block_size: int = 64
    paged_cache_max_blocks: int = 1000
    paged_cache_enabled: bool = False
    paged_cache_direct_attention_enabled: bool = False

    @model_validator(mode="after")
    def validate_batch_generator_lanes(self) -> EngineSettings:
        if self.batch_generator_max_lanes > 1 and self.batch_generator_lane_admission_window_ms <= 0:
            raise ValueError(
                "batch_generator_lane_admission_window_ms must be positive when "
                "batch_generator_max_lanes is greater than one"
            )
        return self


class CacheSettings(BaseModel):
    kv_page_tokens: int = 128
    kv_max_pages: int = 8192
    prefix_cache_enabled: bool = True
    prefix_cache_max_entries: int = 256
    prefix_cache_max_bytes: int = 8 * 1024 * 1024 * 1024
    eviction_policy: Literal["lru"] = "lru"


class BatchSettings(BaseModel):
    max_batch_size: int = 8
    prefill_batch_size: int = 4
    decode_batch_size: int = 8
    min_batch_window_ms: float = 1.5
    max_batch_window_ms: float = 10.0
    latency_target_ms: float = 250.0
    scheduler_mode: Literal["adaptive", "throughput", "latency"] = "adaptive"


class AutotuneSettings(BaseModel):
    enabled: bool = False
    startup_warmup: bool = False
    profile_path: str = "./configs/autotune_profile.json"
    benchmark_prompt_tokens: list[int] = Field(default_factory=lambda: [4096, 8192, 16384])
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 2, 4])


class TelemetrySettings(BaseModel):
    json_logs: bool = True
    metrics_namespace: str = "aster"


class LoggingSettings(BaseModel):
    level: str = "INFO"


class EmbeddingsSettings(BaseModel):
    enabled: bool = True
    backend: Literal["mlx", "vllm_mlx"] = "mlx"
    model: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    model_path: str | None = None
    dimensions: int = 1024
    max_length: int = 512


class ASRSettings(BaseModel):
    enabled: bool = False
    backend: Literal["mlx", "vllm_mlx"] = "mlx"
    model: str = "mlx-community/whisper-large-v3-turbo"
    model_path: str = "models/qwen3-asr-0.6b"


class TTSSettings(BaseModel):
    enabled: bool = False
    backend: Literal["mlx", "vllm_mlx"] = "mlx"
    model: str = "mlx-community/Kokoro-82M-bf16"
    model_path: str = "models/qwen3-tts-0.6b-base"
    custom_voice_model: str | None = None
    custom_voice_path: str | None = None
    default_voice: str = "af_heart"
    cache_enabled: bool = True
    cache_max_entries: int = 128


class AudioSettings(BaseModel):
    asr: ASRSettings = Field(default_factory=ASRSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    max_audio_upload_mb: int = 25
    max_tts_input_chars: int = 4096


class RuntimeSettings(BaseModel):
    api: APISettings = Field(default_factory=APISettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    speculative: SpeculativeSettings = Field(default_factory=SpeculativeSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    autotune: AutotuneSettings = Field(default_factory=AutotuneSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    deprecation_warnings: tuple[str, ...] = ()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _ensure_block(data: dict[str, Any], key: str) -> dict[str, Any]:
    current = data.get(key)
    if isinstance(current, dict):
        return current
    block: dict[str, Any] = {}
    data[key] = block
    return block


def _normalize_legacy_config(data: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    normalized = copy.deepcopy(data)
    warnings: list[str] = []

    engine = _ensure_block(normalized, "engine")
    cache = _ensure_block(normalized, "cache")
    batch = _ensure_block(normalized, "batch")
    model = _ensure_block(normalized, "model")
    embeddings = _ensure_block(normalized, "embeddings")
    audio = _ensure_block(normalized, "audio")

    if model.get("runtime") == "vllm_mlx":
        warnings.append(
            "model.runtime is legacy metadata and is ignored by Aster's built-in "
            "vllm-mlx-compatible runtime."
        )

    legacy_vllm = normalized.get("vllm_mlx")
    if isinstance(legacy_vllm, dict):
        if "stream_interval" in legacy_vllm and "stream_interval_tokens" not in engine:
            engine["stream_interval_tokens"] = int(legacy_vllm["stream_interval"])
        if (
            "chunked_prefill_tokens" in legacy_vllm
            and int(legacy_vllm["chunked_prefill_tokens"]) > 0
            and "prefill_token_budget" not in engine
        ):
            engine["prefill_token_budget"] = int(legacy_vllm["chunked_prefill_tokens"])
        warnings.append("vllm_mlx.* settings are deprecated and only used as engine shims.")

    if "decode_batch_size" in batch and "max_decode_batch" not in engine:
        engine["max_decode_batch"] = int(batch["decode_batch_size"])
    elif "max_batch_size" in batch and "max_decode_batch" not in engine:
        engine["max_decode_batch"] = int(batch["max_batch_size"])

    if "max_batch_size" in batch and "max_active_requests" not in engine:
        engine["max_active_requests"] = int(batch["max_batch_size"])

    if "prefix_cache_max_bytes" in cache and "snapshot_budget_bytes" not in engine:
        engine["snapshot_budget_bytes"] = int(cache["prefix_cache_max_bytes"])
    if "prefix_cache_max_entries" in cache and "snapshot_max_entries" not in engine:
        engine["snapshot_max_entries"] = int(cache["prefix_cache_max_entries"])
    if "prefix_cache_enabled" in cache and "prefix_cache_enabled" not in engine:
        engine["prefix_cache_enabled"] = bool(cache["prefix_cache_enabled"])

    if "backend" in embeddings and embeddings["backend"] == "vllm_mlx":
        embeddings["backend"] = "mlx"
        warnings.append("embeddings.backend=vllm_mlx is deprecated and is treated as mlx.")

    asr = _ensure_block(audio, "asr")
    tts = _ensure_block(audio, "tts")

    if "asr_enabled" in audio and "enabled" not in asr:
        asr["enabled"] = bool(audio["asr_enabled"])
    if "tts_enabled" in audio and "enabled" not in tts:
        tts["enabled"] = bool(audio["tts_enabled"])
    if "asr_backend" in audio and "backend" not in asr:
        asr["backend"] = audio["asr_backend"]
    if "tts_backend" in audio and "backend" not in tts:
        tts["backend"] = audio["tts_backend"]

    if asr.get("backend") == "vllm_mlx":
        asr["backend"] = "mlx"
        warnings.append("audio.asr backend vllm_mlx is deprecated and is treated as mlx.")
    if tts.get("backend") == "vllm_mlx":
        tts["backend"] = "mlx"
        warnings.append("audio.tts backend vllm_mlx is deprecated and is treated as mlx.")

    return normalized, tuple(warnings)


def load_settings(config_path: str) -> RuntimeSettings:
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(code="config_not_found", message=f"Missing config: {path}")
    raw_data: Any = yaml.safe_load(path.read_text())
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    env_override = os.getenv("ASTER_CONFIG_OVERRIDE")
    if env_override:
        env_data: Any = yaml.safe_load(env_override)
        override_dict: dict[str, Any] = env_data if isinstance(env_data, dict) else {}
        data = _deep_merge(data, override_dict)
    normalized, warnings = _normalize_legacy_config(data)
    settings = RuntimeSettings.model_validate(normalized)
    return settings.model_copy(update={"deprecation_warnings": warnings})
