"""Tests for cache modules."""

import pytest

from aster.cache.cache_keys import prefix_hash
from aster.cache.paged_kv_cache import PagedKVCache
from aster.core.config import CacheSettings


def test_prefix_hash_deterministic():
    """Test that prefix hashing is deterministic."""
    tokens = [1, 2, 3, 4, 5]
    hash1 = prefix_hash(tokens)
    hash2 = prefix_hash(tokens)
    assert hash1 == hash2


def test_prefix_hash_different_inputs():
    """Test that different inputs produce different hashes."""
    hash1 = prefix_hash([1, 2, 3])
    hash2 = prefix_hash([1, 2, 4])
    assert hash1 != hash2


def test_paged_kv_cache_allocation():
    """Test KV cache page allocation."""
    settings = CacheSettings(max_pages=10, page_size=512)
    cache = PagedKVCache(settings)
    
    # Allocate pages
    pages = cache.allocate(5)
    assert len(pages) == 5
    
    # Check that pages are valid
    for page_id in pages:
        assert page_id >= 0
        assert page_id < settings.max_pages


def test_paged_kv_cache_release():
    """Test KV cache page release."""
    settings = CacheSettings(max_pages=10, page_size=512)
    cache = PagedKVCache(settings)
    
    # Allocate and release
    pages = cache.allocate(5)
    cache.release(pages)
    
    # Should be able to allocate again
    new_pages = cache.allocate(5)
    assert len(new_pages) == 5
