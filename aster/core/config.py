from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from aster.core.errors import ConfigurationError


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class APISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    max_queue_depth: int = 128
    request_timeout_seconds: float = 180.0


# ---------------------------------------------------------------------------
# Model (primary / target model only)
# ---------------------------------------------------------------------------

class ModelSettings(BaseModel):
    name: str = "Qwen3.5-9B"
    path: str = "models/qwen3.5-9b"
    runtime: Literal["mlx", "vllm_mlx"] = "mlx"
    context_length: int = 16384
    # Default thinking mode for this model; per-request `enable_thinking`
    # overrides this value when explicitly provided by the caller.
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Speculative decoding (draft model lives here, not in ModelSettings)
# ---------------------------------------------------------------------------

class SpeculativeSettings(BaseModel):
    enabled: bool = True
    draft_name: str = "Qwen3.5-0.8B"
    draft_path: str = "models/qwen3.5-0.8b"
    max_draft_tokens: int = 6
    min_acceptance_rate: float = 0.45
    min_speedup_ratio: float = 1.05
    auto_disable_on_regression: bool = True


# ---------------------------------------------------------------------------
# vLLM-MLX sidecar (connection + sidecar behaviour only, no model paths)
# ---------------------------------------------------------------------------

class VLLMMLXSettings(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    timeout_seconds: float = 300.0
    # Reasoning parser enables vLLM-MLX's built-in thinking-token splitter.
    # When set, Aster renders prompts via apply_chat_template(enable_thinking=…)
    # and uses /v1/completions so vLLM-MLX doesn't re-apply the template.
    # The model path is taken from model.path — no extra field needed.
    reasoning_parser: Literal["qwen3", "deepseek_r1", "gpt_oss", "harmony"] | None = None
    # Batching / memory knobs for the sidecar process
    continuous_batching: bool = True
    use_paged_cache: bool = True
    enable_prefix_cache: bool = True
    chunked_prefill_tokens: int = 2048
    cache_memory_percent: float = 0.2
    stream_interval: int = 1


# ---------------------------------------------------------------------------
# KV-cache (Aster-side page allocator, shared by both runtimes)
# ---------------------------------------------------------------------------

class CacheSettings(BaseModel):
    kv_page_tokens: int = 128
    kv_max_pages: int = 8192
    prefix_cache_enabled: bool = True
    prefix_cache_max_entries: int = 256
    prefix_cache_max_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GB
    eviction_policy: Literal["lru"] = "lru"


# ---------------------------------------------------------------------------
# Scheduler / batching
# ---------------------------------------------------------------------------

class BatchSettings(BaseModel):
    max_batch_size: int = 8
    prefill_batch_size: int = 4
    decode_batch_size: int = 8
    min_batch_window_ms: float = 1.5
    max_batch_window_ms: float = 10.0
    latency_target_ms: float = 250.0
    scheduler_mode: Literal["adaptive", "throughput", "latency"] = "adaptive"


# ---------------------------------------------------------------------------
# Autotune
# ---------------------------------------------------------------------------

class AutotuneSettings(BaseModel):
    enabled: bool = True
    startup_warmup: bool = True
    profile_path: str = "./configs/autotune_profile.json"
    benchmark_prompt_tokens: list[int] = Field(default_factory=lambda: [4096, 8192, 16384])
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 2, 4])


# ---------------------------------------------------------------------------
# Workers / supervision
# ---------------------------------------------------------------------------

class WorkerSettings(BaseModel):
    restart_limit: int = 5
    heartbeat_timeout_seconds: float = 30.0
    degraded_after_restarts: int = 2


# ---------------------------------------------------------------------------
# Telemetry & logging
# ---------------------------------------------------------------------------

class TelemetrySettings(BaseModel):
    json_logs: bool = True
    metrics_namespace: str = "aster"


class LoggingSettings(BaseModel):
    level: str = "INFO"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingsSettings(BaseModel):
    enabled: bool = True
    backend: Literal["mlx", "vllm_mlx"] = "mlx"
    model: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    # Local filesystem path; falls back to `model` (HF repo id) when absent.
    model_path: str | None = None
    dimensions: int = 1024
    max_length: int = 512


# ---------------------------------------------------------------------------
# Audio (ASR + TTS each as a coherent sub-block)
# ---------------------------------------------------------------------------

class ASRSettings(BaseModel):
    enabled: bool = True
    backend: Literal["mlx", "vllm_mlx"] = "mlx"
    model: str = "mlx-community/whisper-large-v3-turbo"
    model_path: str = "models/qwen3-asr-0.6b"


class TTSSettings(BaseModel):
    enabled: bool = True
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


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class RuntimeSettings(BaseModel):
    api: APISettings = Field(default_factory=APISettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    speculative: SpeculativeSettings = Field(default_factory=SpeculativeSettings)
    vllm_mlx: VLLMMLXSettings = Field(default_factory=VLLMMLXSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    autotune: AutotuneSettings = Field(default_factory=AutotuneSettings)
    workers: WorkerSettings = Field(default_factory=WorkerSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def load_settings(config_path: str) -> RuntimeSettings:
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(code="config_not_found", message=f"Missing config: {path}")
    raw_data: Any = yaml.safe_load(path.read_text())
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}  # type: ignore[assignment]
    env_override = os.getenv("ASTER_CONFIG_OVERRIDE")
    if env_override:
        env_data: Any = yaml.safe_load(env_override)
        override_dict: dict[str, Any] = env_data if isinstance(env_data, dict) else {}  # type: ignore[assignment]
        data = _deep_merge(data, override_dict)
    return RuntimeSettings.model_validate(data)
