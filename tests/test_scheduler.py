"""Tests for scheduler modules."""

from __future__ import annotations

from aster.core.config import BatchSettings
from aster.scheduler.adaptive_batcher import AdaptiveBatcher
from aster.scheduler.policy_engine import RuntimePolicy


def test_adaptive_batcher_initialization() -> None:
    """Test adaptive batcher initialization."""
    settings = BatchSettings(
        min_batch_window_ms=10,
        max_batch_size=4
    )
    batcher = AdaptiveBatcher(settings)
    assert batcher is not None


def test_adaptive_batcher_batch_formation() -> None:
    """Test batch formation logic."""
    settings = BatchSettings(
        min_batch_window_ms=10,
        max_batch_size=4
    )
    batcher = AdaptiveBatcher(settings)
    policy = RuntimePolicy(max_batch_size=4, batch_window_ms=10.0)

    # Batcher should handle empty queue
    decision = batcher.decide(queue_depth=0, avg_prompt_tokens=100, policy=policy)
    assert decision is not None
    assert decision.batch_size >= 1
