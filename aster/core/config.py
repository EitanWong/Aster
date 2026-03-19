from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from aster.core.errors import ConfigurationError


class APISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    max_queue_depth: int = 128
    request_timeout_seconds: float = 180.0


class ModelSettings(BaseModel):
    name: str = "Qwen3.5-9B"
    path: str = "models/qwen3.5-9b"
    draft_name: str = "Qwen3.5-0.8B"
    draft_path: str = "models/qwen3.5-0.8b"
    runtime: Literal["mlx"] = "mlx"
    context_length: int = 16384


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


class SpeculativeSettings(BaseModel):
    enabled: bool = True
    max_draft_tokens: int = 6
    min_acceptance_rate: float = 0.45
    min_speedup_ratio: float = 1.05
    auto_disable_on_regression: bool = True


class AutotuneSettings(BaseModel):
    enabled: bool = True
    startup_warmup: bool = True
    profile_path: str = "./configs/autotune_profile.json"
    benchmark_prompt_tokens: list[int] = Field(default_factory=lambda: [4096, 8192, 16384])
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 2, 4])


class WorkerSettings(BaseModel):
    restart_limit: int = 5
    heartbeat_timeout_seconds: float = 30.0
    degraded_after_restarts: int = 2


class TelemetrySettings(BaseModel):
    json_logs: bool = True
    metrics_namespace: str = "aster"


class LoggingSettings(BaseModel):
    level: str = "INFO"


class RuntimeSettings(BaseModel):
    api: APISettings = Field(default_factory=APISettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    speculative: SpeculativeSettings = Field(default_factory=SpeculativeSettings)
    autotune: AutotuneSettings = Field(default_factory=AutotuneSettings)
    workers: WorkerSettings = Field(default_factory=WorkerSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


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
