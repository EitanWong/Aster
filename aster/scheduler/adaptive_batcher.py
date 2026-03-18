from __future__ import annotations

from dataclasses import dataclass

from aster.core.config import BatchSettings
from aster.scheduler.policy_engine import RuntimePolicy


@dataclass(slots=True)
class BatchDecision:
    batch_size: int
    window_ms: float


class AdaptiveBatcher:
    def __init__(self, settings: BatchSettings) -> None:
        self.settings = settings

    def decide(self, queue_depth: int, avg_prompt_tokens: int, policy: RuntimePolicy) -> BatchDecision:
        if queue_depth <= 1:
            return BatchDecision(batch_size=1, window_ms=self.settings.min_batch_window_ms)
        if avg_prompt_tokens > 8000:
            return BatchDecision(batch_size=min(2, policy.max_batch_size), window_ms=policy.batch_window_ms)
        batch_size = min(policy.max_batch_size, max(1, queue_depth))
        window = min(self.settings.max_batch_window_ms, self.settings.min_batch_window_ms + queue_depth)
        return BatchDecision(batch_size=batch_size, window_ms=window)
