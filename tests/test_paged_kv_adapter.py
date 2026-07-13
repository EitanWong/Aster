from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from aster.inference.paged_cache import PagedCacheManager
from aster.inference.paged_kv_adapter import (
    PagedKVCacheBundle,
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


def test_cow_copies_each_layer_pool_after_one_shared_table_split() -> None:
    manager = PagedCacheManager(num_layers=2, block_size=4, max_blocks=8)
    table = manager.create_table("source-bundle")
    source_k0 = PagedKVCacheLayer(manager, layer_index=0, block_table=table)
    source_k1 = PagedKVCacheLayer(manager, layer_index=1, block_table=table)
    source_k0.update_and_fetch(_array([1, 2]), _array([11, 12]))
    source_k1.update_and_fetch(_array([101, 102]), _array([111, 112]))

    fork_table = manager.fork_table(table, "fork-bundle")
    fork_k0 = PagedKVCacheLayer(manager, layer_index=0, block_table=fork_table, pool=source_k0._pool)
    fork_k1 = PagedKVCacheLayer(manager, layer_index=1, block_table=fork_table, pool=source_k1._pool)
    fork_k0.offset = fork_k1.offset = 2
    fork_k0.update_and_fetch(_array([3]), _array([13]))
    fork_k1.update_and_fetch(_array([103]), _array([113]))

    assert _values(source_k0.state[0]) == [1, 2]
    assert _values(source_k1.state[0]) == [101, 102]
    assert _values(fork_k0.state[0]) == [1, 2, 3]
    assert _values(fork_k1.state[0]) == [101, 102, 103]
    assert manager.stats.cow_copies == 1


def test_bundle_release_reclaims_pool_after_table_release() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=8)
    bundle = PagedKVCacheBundle.from_prompt_cache(
        [KVCache()], manager, kv_cache_type=KVCache, request_id="release-source"
    )
    layer = bundle.layers[0]
    layer.update_and_fetch(_array([1, 2, 3]), _array([11, 12, 13]))
    pool = layer._pool

    assert pool.nbytes > 0
    assert manager.stats.allocated_blocks == 2

    bundle.release()
    bundle.release()

    assert manager.get_table("release-source") is None
    assert manager.stats.allocated_blocks == 0
    assert pool.nbytes == 0


def test_bundle_fork_keeps_shared_pool_alive_until_last_release() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=4, max_blocks=8)
    source = PagedKVCacheBundle.from_prompt_cache(
        [KVCache()], manager, kv_cache_type=KVCache, request_id="fork-source"
    )
    source_layer = source.layers[0]
    source_layer.update_and_fetch(_array([1, 2]), _array([11, 12]))
    pool = source_layer._pool

    fork = source.fork("fork-child")
    fork.layers[0].update_and_fetch(_array([3]), _array([13]))
    assert _values(source_layer.state[0]) == [1, 2]
    assert _values(fork.layers[0].state[0]) == [1, 2, 3]

    fork.release()
    assert pool.nbytes > 0
    assert manager.get_table("fork-source") is not None

    source.release()
    assert pool.nbytes == 0


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


def test_attention_view_reuses_a_persistent_physical_block_pool() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=2, max_blocks=8)
    layer = PagedKVCacheLayer(manager, layer_index=0)
    layer.update_and_fetch(_array([1, 2, 3]), _array([11, 12, 13]))

    first_pool, _, first_indices = layer.attention_view().block_pool()
    layer.update_and_fetch(_array([4]), _array([14]))
    second_pool, _, second_indices = layer.attention_view().block_pool()
    mlx.eval(first_pool, second_pool, second_indices)

    assert first_pool is second_pool
    assert first_pool.shape[0] >= max(layer.block_table.block_ids) + 1
    assert np.asarray(second_indices).tolist() == layer.block_table.block_ids
    assert _values(second_pool[layer.block_table.block_ids[0], ..., :2, :]) == [1, 2]


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
