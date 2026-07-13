from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from aster.inference.paged_cache import PagedCacheManager
from aster.inference.paged_kv_adapter import (
    PagedKVCacheLayer,
    replace_kv_cache_layers,
)
from mlx_lm.models.cache import KVCache


def _array(values: list[float]):
    return mlx.array(values, dtype=mlx.float32).reshape(1, 1, len(values), 1)


def _values(array) -> list[float]:
    mlx.eval(array)
    return np.asarray(array).reshape(-1).tolist()


def test_paged_layer_materializes_blocks_in_logical_order() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=4)
    layer = PagedKVCacheLayer(manager, layer_index=0)

    keys, values = layer.update_and_fetch(_array([1, 2, 3]), _array([11, 12, 13]))
    assert _values(keys) == [1, 2, 3]
    assert _values(values) == [11, 12, 13]
    assert layer.offset == 3
    assert layer.block_table.block_ids == [1, 2]

    keys, values = layer.update_and_fetch(_array([4]), _array([14]))
    assert _values(keys) == [1, 2, 3, 4]
    assert _values(values) == [11, 12, 13, 14]
    assert layer.offset == 4


def test_fork_copies_a_shared_partial_block_before_writing() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=4, max_blocks=4)
    first = PagedKVCacheLayer(manager, layer_index=0)
    first.update_and_fetch(_array([1, 2]), _array([11, 12]))

    second = first.fork()
    keys, values = second.update_and_fetch(_array([3]), _array([13]))

    assert _values(keys) == [1, 2, 3]
    assert _values(values) == [11, 12, 13]
    assert _values(first.state[0]) == [1, 2]
    assert first.block_table.block_ids != second.block_table.block_ids
    assert manager.stats.cow_copies == 1
    assert manager.stats.shared_blocks == 0


def test_attention_view_exposes_blocks_without_hiding_materialization() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=4)
    layer = PagedKVCacheLayer(manager, layer_index=0)
    layer.update_and_fetch(_array([1, 2, 3]), _array([11, 12, 13]))

    view = layer.attention_view()

    assert view.block_ids == (1, 2)
    assert view.block_size == 2
    assert view.sequence_length == 3
    keys, values = view.materialize()
    assert _values(keys) == [1, 2, 3]
    assert _values(values) == [11, 12, 13]


@pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal is required")
def test_attention_view_can_dispatch_block_indexed_metal_attention() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=4)
    layer = PagedKVCacheLayer(manager, layer_index=0)
    keys = mlx.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    values = mlx.array([[[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]])
    layer.update_and_fetch(keys, values)
    query = mlx.array([[[[0.5, 0.25]]]])
    view = layer.attention_view()

    actual = view.attention(query, scale=0.5)
    expected = mlx.fast.scaled_dot_product_attention(
        query, keys, values, scale=0.5, mask=None
    )
    mlx.eval(actual, expected)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=2e-3)


def test_materialized_view_is_compatible_with_native_mlx_attention() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=4)
    paged = PagedKVCacheLayer(manager, layer_index=0)
    native = KVCache()
    keys = mlx.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    values = mlx.array([[[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]])
    query = mlx.array([[[[0.5, 0.25]]]])

    native_keys, native_values = native.update_and_fetch(keys, values)
    paged_keys, paged_values = paged.update_and_fetch(keys, values)
    native_output = mlx.fast.scaled_dot_product_attention(
        query, native_keys, native_values, scale=0.5
    )
    paged_output = mlx.fast.scaled_dot_product_attention(
        query, paged_keys, paged_values, scale=0.5
    )
    mlx.eval(native_output, paged_output)

    np.testing.assert_allclose(np.asarray(paged_output), np.asarray(native_output))


def test_replacement_preserves_non_sequence_cache_layers() -> None:
    manager = PagedCacheManager(num_layers=3, block_size=2, max_blocks=4)
    native_cache = [KVCache(), object(), KVCache()]

    replaced = replace_kv_cache_layers(
        native_cache,
        manager,
        kv_cache_type=KVCache,
        request_id="replacement-test",
    )

    assert isinstance(replaced[0], PagedKVCacheLayer)
    assert replaced[1] is native_cache[1]
    assert isinstance(replaced[2], PagedKVCacheLayer)
    assert replaced[0].block_table is replaced[2].block_table
