from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from aster.inference.paged_attention_bridge import dispatch_paged_attention
from aster.inference.paged_cache import PagedCacheManager
from aster.inference.paged_kv_adapter import PagedKVCacheLayer


def _array(values: list[float]):
    return mx.array(values, dtype=mx.float32).reshape(1, 1, len(values), 1)


def test_dispatch_falls_back_to_dense_state_for_unsupported_mask() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=4)
    cache = PagedKVCacheLayer(
        manager,
        layer_index=0,
        enable_block_pool=False,
        enable_direct_attention=True,
    )
    cache.update_and_fetch(_array([1, 2, 3]), _array([11, 12, 13]))
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def native_attention(queries, keys, values, **kwargs):
        calls.append((tuple(keys.shape), tuple(values.shape)))
        return "native"

    result = dispatch_paged_attention(
        native_attention,
        _array([7]),
        _array([1]),
        _array([11]),
        cache=cache,
        scale=1.0,
        mask="window",
    )

    assert result == "native"
    assert calls == [((1, 1, 3, 1), (1, 1, 3, 1))]
    cache._pool.release()


def test_direct_layer_stores_only_in_the_pool() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=4)
    cache = PagedKVCacheLayer(
        manager,
        layer_index=0,
        enable_block_pool=False,
        enable_direct_attention=True,
    )

    cache.update_and_fetch(_array([1, 2, 3]), _array([11, 12, 13]))

    assert cache.direct_attention_enabled is True
    assert cache._materialized_keys is not None
    assert cache._pool.nbytes == 0

    cache.prepare_direct_attention()

    assert cache._materialized_keys is None
    assert cache._pool.nbytes > 0
    cache._reset_owned_blocks()
    cache._pool.release()
