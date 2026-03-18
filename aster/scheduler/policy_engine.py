from __future__ import annotations

from dataclasses import dataclass, replace

from aster.core.config import RuntimeSettings


@dataclass(slots=True)
class RuntimePolicy:
    speculative_enabled: bool = True
    speculative_draft_tokens: int = 4
    speculative_acceptance_hint: float = 0.7
    prefix_cache_enabled: bool = True
    batch_window_ms: float = 3.0
    max_batch_size: int = 4
    stream_flush_ms: float = 10.0
    scheduler_mode: str = "adaptive"


class PolicyEngine:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._policy = RuntimePolicy(
            speculative_enabled=settings.speculative.enabled,
            speculative_draft_tokens=min(settings.speculative.max_draft_tokens, 4),
            prefix_cache_enabled=settings.cache.prefix_cache_enabled,
            batch_window_ms=settings.batch.min_batch_window_ms,
            max_batch_size=settings.batch.max_batch_size,
            stream_flush_ms=12.0,
            scheduler_mode=settings.batch.scheduler_mode,
        )

    def current(self) -> RuntimePolicy:
        return self._policy

    def update(self, **kwargs: object) -> RuntimePolicy:
        self._policy = replace(self._policy, **kwargs)
        return self._policy

    def should_use_speculative(self, request_class: str) -> bool:
        if request_class == "latency_critical" and self._policy.scheduler_mode == "latency":
            return False
        return self._policy.speculative_enabled
