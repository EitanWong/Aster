from __future__ import annotations

from pathlib import Path

from aster.inference.prefix_store import PrefixStore


class _Array:
    shape = (1, 1, 5, 1)


class _RewindableCacheLayer:
    def __init__(self, offset: int) -> None:
        self.offset = offset
        self.keys = _Array()
        self.values = _Array()


class _NoSliceList(list[tuple[int, ...]]):
    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        if isinstance(index, slice):
            raise AssertionError("prefix lookup must not copy the candidate index")
        return super().__getitem__(index)


def test_prefix_store_returns_longest_matching_snapshot() -> None:
    store = PrefixStore(
        budget_bytes=1024 * 1024,
        max_entries=16,
        min_prefix_tokens=2,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "short"},
        approx_bytes=128,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[1, 2, 3, 4, 5],
        cache_token_count=4,
        prompt_cache={"name": "long"},
        approx_bytes=256,
    )

    hit = store.lookup("test-model", [1, 2, 3, 4, 5, 6])

    assert hit is not None
    assert list(hit.prefix_tokens) == [1, 2, 3, 4, 5]
    assert hit.cache_token_count == 4
    assert hit.match_type == "prefix"
    assert store.last_match_type == "prefix"


def test_prefix_store_prefix_lookup_does_not_slice_candidate_index() -> None:
    store = PrefixStore(
        budget_bytes=1024 * 1024,
        max_entries=64,
        min_prefix_tokens=2,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "prefix"},
        approx_bytes=128,
    )
    for branch in range(32):
        store.store(
            model_name="test-model",
            prefix_tokens=[1, 2, 3, branch, 100 + branch],
            cache_token_count=4,
            prompt_cache={"name": f"branch-{branch}"},
            approx_bytes=128,
        )
    namespace = ("test-model", None)
    store._sorted_tokens_by_namespace[namespace] = _NoSliceList(
        store._sorted_tokens_by_namespace[namespace]
    )

    hit = store.lookup("test-model", [1, 2, 3, 99, 999])

    assert hit is not None
    assert list(hit.prefix_tokens) == [1, 2, 3]
    assert hit.match_type == "prefix"


def test_prefix_store_respects_budget_with_size_aware_eviction() -> None:
    store = PrefixStore(
        budget_bytes=200,
        max_entries=16,
        min_prefix_tokens=1,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "older"},
        approx_bytes=128,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[4, 5, 6],
        cache_token_count=2,
        prompt_cache={"name": "newer"},
        approx_bytes=128,
    )

    assert store.entry_count == 1
    remaining = store.lookup("test-model", [4, 5, 6, 7])
    assert remaining is not None
    assert list(remaining.prefix_tokens) == [4, 5, 6]


def test_prefix_store_keeps_other_same_length_entries_indexed_after_eviction() -> None:
    store = PrefixStore(
        budget_bytes=260,
        max_entries=4,
        min_prefix_tokens=1,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "first"},
        approx_bytes=120,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[4, 5, 6],
        cache_token_count=2,
        prompt_cache={"name": "second"},
        approx_bytes=120,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[7, 8, 9, 10],
        cache_token_count=3,
        prompt_cache={"name": "third"},
        approx_bytes=120,
    )

    remaining = store.lookup("test-model", [4, 5, 6, 99])

    assert remaining is not None
    assert list(remaining.prefix_tokens) == [4, 5, 6]
    assert store.stats.evictions == 1


def test_prefix_store_partitions_entries_by_model_fingerprint() -> None:
    store = PrefixStore(
        budget_bytes=1024 * 1024,
        max_entries=16,
        min_prefix_tokens=2,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        model_fingerprint="fingerprint-a",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "a"},
        approx_bytes=128,
    )

    miss = store.lookup(
        "test-model",
        [1, 2, 3, 4],
        model_fingerprint="fingerprint-b",
    )
    hit = store.lookup(
        "test-model",
        [1, 2, 3, 4],
        model_fingerprint="fingerprint-a",
    )

    assert miss is None
    assert hit is not None
    assert hit.model_fingerprint == "fingerprint-a"
    assert hit.match_type == "prefix"


def test_prefix_store_returns_lcp_match_only_for_rewindable_cache() -> None:
    store = PrefixStore(
        budget_bytes=1024 * 1024,
        max_entries=16,
        min_prefix_tokens=2,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[1, 2, 3, 4, 5],
        cache_token_count=4,
        prompt_cache=[_RewindableCacheLayer(offset=4)],
        approx_bytes=128,
    )

    hit = store.lookup("test-model", [1, 2, 99])

    assert hit is not None
    assert list(hit.prefix_tokens) == [1, 2]
    assert hit.cache_token_count == 1
    assert hit.match_type == "lcp"
    assert store.last_match_type == "lcp"
    assert store.stats.lcp_hits == 1
    assert store.stats.tokens_saved == 1


def test_prefix_store_skips_lcp_for_unsafe_cache() -> None:
    store = PrefixStore(
        budget_bytes=1024 * 1024,
        max_entries=16,
        min_prefix_tokens=2,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        prefix_tokens=[1, 2, 3, 4, 5],
        cache_token_count=4,
        prompt_cache={"name": "not-rewindable"},
        approx_bytes=128,
    )

    hit = store.lookup("test-model", [1, 2, 99])

    assert hit is None
    assert store.last_match_type == "miss"
    assert store.stats.unsafe_lcp_skips == 1


def test_prefix_store_persists_entries_to_disk(tmp_path: Path) -> None:
    cache_path = tmp_path / "prefix-cache.pkl"
    store = PrefixStore(
        budget_bytes=1024 * 1024,
        max_entries=16,
        min_prefix_tokens=2,
        enabled=True,
    )
    store.store(
        model_name="test-model",
        model_fingerprint="fingerprint-a",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "persisted"},
        approx_bytes=128,
    )

    saved = store.save_to_disk(cache_path)
    restored = PrefixStore(
        budget_bytes=1024 * 1024,
        max_entries=16,
        min_prefix_tokens=2,
        enabled=True,
    )
    loaded = restored.load_from_disk(
        cache_path,
        model_name="test-model",
        model_fingerprint="fingerprint-a",
    )
    hit = restored.lookup(
        "test-model",
        [1, 2, 3, 4],
        model_fingerprint="fingerprint-a",
    )

    assert saved == 1
    assert loaded == 1
    assert hit is not None
    assert hit.prompt_cache == {"name": "persisted"}
    assert hit.match_type == "prefix"


def test_prefix_store_stats_snapshot_and_clear_preserve_pinned_entries() -> None:
    store = PrefixStore(
        budget_bytes=1024,
        max_entries=10,
        min_prefix_tokens=1,
        enabled=True,
    )
    first = store.store(
        model_name="model",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "pinned"},
        approx_bytes=32,
    )
    store.store(
        model_name="model",
        prefix_tokens=[4, 5, 6],
        cache_token_count=2,
        prompt_cache={"name": "clearable"},
        approx_bytes=48,
    )
    assert first is not None
    store.pin(first.key)

    before = store.stats_snapshot()
    assert before["entries"] == 2
    assert before["pinned_entries"] == 1
    assert before["evictable_entries"] == 1
    assert before["bytes"] == 80
    assert before["pinned_bytes"] == 32
    assert before["evictable_bytes"] == 48
    assert before["cached_tokens"] == 4
    assert before["avg_entry_bytes"] == 40
    assert before["max_entry_bytes"] == 48
    assert before["memory_utilization"] == 80 / 1024

    cleared = store.clear()

    assert cleared["entries_cleared"] == 1
    assert cleared["bytes_cleared"] == 48
    assert cleared["pinned_preserved"] == 1
    after = store.stats_snapshot()
    assert after["entries"] == 1
    assert after["pinned_entries"] == 1
    assert after["evictable_entries"] == 0
    assert after["pinned_bytes"] == 32
    assert after["evictable_bytes"] == 0

    store.unpin(first.key)
    cleared_after_unpin = store.clear()

    assert cleared_after_unpin["entries_cleared"] == 1
    assert store.stats_snapshot()["entries"] == 0


def test_prefix_store_replacing_pinned_entry_preserves_pin() -> None:
    store = PrefixStore(
        budget_bytes=96,
        max_entries=10,
        min_prefix_tokens=1,
        enabled=True,
    )
    first = store.store(
        model_name="model",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "first"},
        approx_bytes=32,
    )
    assert first is not None
    store.pin(first.key)

    replacement = store.store(
        model_name="model",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"name": "replacement"},
        approx_bytes=64,
    )

    assert replacement is not None
    assert replacement.pin_count == 1
    cleared = store.clear()
    assert cleared["entries_cleared"] == 0
    assert cleared["pinned_preserved"] == 1
    snapshot = store.stats_snapshot()
    assert snapshot["entries"] == 1
    assert snapshot["pinned_bytes"] == 64
    assert store.lookup("model", [1, 2, 3]) is not None

    store.unpin(replacement.key)
    assert store.clear()["entries_cleared"] == 1


def test_prefix_store_stats_track_tokens_saved_by_hit_type() -> None:
    store = PrefixStore(
        budget_bytes=1024,
        max_entries=10,
        min_prefix_tokens=2,
        enabled=True,
    )
    store.store(
        model_name="model",
        prefix_tokens=[1, 2],
        cache_token_count=1,
        prompt_cache={"name": "short"},
        approx_bytes=32,
    )
    store.store(
        model_name="model",
        prefix_tokens=[4, 5, 6],
        cache_token_count=2,
        prompt_cache={"name": "long"},
        approx_bytes=64,
    )

    prefix_hit = store.lookup("model", [1, 2, 3])
    exact_hit = store.lookup("model", [4, 5, 6])

    assert prefix_hit is not None
    assert exact_hit is not None
    stats = store.stats_snapshot()
    assert stats["hits"] == 2
    assert stats["tokens_saved"] == 3
