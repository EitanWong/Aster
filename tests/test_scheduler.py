"""Tests for scheduler modules."""

from aster.core.config import BatchSettings
from aster.scheduler.adaptive_batcher import AdaptiveBatcher


def test_adaptive_batcher_initialization():
    """Test adaptive batcher initialization."""
    settings = BatchSettings(
        window_ms=10,
        max_batch_size=4,
        min_batch_size=1
    )
    batcher = AdaptiveBatcher(settings)
    assert batcher is not None


def test_adaptive_batcher_batch_formation():
    """Test batch formation logic."""
    settings = BatchSettings(
        window_ms=10,
        max_batch_size=4,
        min_batch_size=1
    )
    batcher = AdaptiveBatcher(settings)

    # Batcher should handle empty queue
    batch = batcher.form_batch([])
    assert batch is not None or batch == []
