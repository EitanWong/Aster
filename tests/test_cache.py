"""Tests for cache modules."""

from __future__ import annotations

from aster.cache.cache_keys import prefix_hash
from aster.cache.paged_kv_cache import PagedKVCache
from aster.core.config import CacheSettings
from aster.telemetry.metrics import MetricsRegistry


def test_prefix_hash_deterministic() -> None:
    """Test that prefix hashing is deterministic."""
    tokens = [1, 2, 3, 4, 5]
    hash1 = prefix_hash("test_model", tokens)
    hash2 = prefix_hash("test_model", tokens)
    assert hash1 == hash2


def test_prefix_hash_different_inputs() -> None:
    """Test that different inputs produce different hashes."""
    hash1 = prefix_hash("test_model", [1, 2, 3])
    hash2 = prefix_hash("test_model", [1, 2, 4])
    assert hash1 != hash2


def test_paged_kv_cache_allocation() -> None:
    """Test KV cache page allocation."""
    settings = CacheSettings(kv_max_pages=10, kv_page_tokens=512)
    metrics = MetricsRegistry()
    cache = PagedKVCache(settings, metrics)

    # Allocate pages for a request
    request_id = "test_request_1"
    pages = cache.allocate(request_id, tokens=5)
    assert len(pages) == 1

    # Check that pages are valid
    for page_id in pages:
        assert page_id >= 0
        assert page_id < settings.kv_max_pages


def test_paged_kv_cache_release() -> None:
    """Test KV cache page release."""
    settings = CacheSettings(kv_max_pages=10, kv_page_tokens=512)
    metrics = MetricsRegistry(namespace="test")
    cache = PagedKVCache(settings, metrics)

    # Allocate and release
    request_id = "test_request_2"
    cache.allocate(request_id, tokens=5)
    cache.release(request_id)

    # Should be able to allocate again
    new_request_id = "test_request_3"
    new_pages = cache.allocate(new_request_id, tokens=5)
    assert len(new_pages) == 1
