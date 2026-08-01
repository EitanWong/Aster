from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache  # noqa: E402

from aster.inference.paged_cache import PagedCacheManager  # noqa: E402
from aster.inference.paged_kv_adapter import (  # noqa: E402
    PagedAttentionView,
    PagedBatchAttentionView,
    PagedKVCacheBundle,
    PagedKVCacheLayer,
)


def _source_bundle(
    manager: PagedCacheManager,
    *,
    prefix_tokens: int,
    seed: int,
) -> PagedKVCacheBundle:
    mx.random.seed(seed)
    bundle = PagedKVCacheBundle.from_prompt_cache(
        [KVCache()],
        manager,
        kv_cache_type=KVCache,
        request_id=f"source-{seed}",
        enable_block_pool=True,
        enable_direct_attention=True,
    )
    keys = mx.random.normal((1, 2, prefix_tokens, 32)).astype(mx.float16)
    values = mx.random.normal((1, 2, prefix_tokens, 32)).astype(mx.float16)
    bundle.layers[0].update_and_fetch(keys, values)
    pool_keys, pool_values = bundle.layers[0]._pool.block_pool()
    mx.eval(pool_keys, pool_values)
    return bundle


def _fork_with_suffixes(
    source: PagedKVCacheBundle,
    suffix_lengths: list[int],
    *,
    seed: int,
) -> list[PagedKVCacheBundle]:
    children: list[PagedKVCacheBundle] = []
    for index, suffix_length in enumerate(suffix_lengths):
        child = source.fork(f"child-{seed}-{index}")
        mx.random.seed(seed + index)
        keys = mx.random.normal((1, 2, suffix_length, 32)).astype(mx.float16)
        values = mx.random.normal((1, 2, suffix_length, 32)).astype(mx.float16)
        child.layers[0].update_and_fetch(keys, values)
        children.append(child)
    pool_keys, pool_values = source.layers[0]._pool.block_pool()
    mx.eval(pool_keys, pool_values)
    return children


def _rowwise_native_attention(
    views: tuple[PagedAttentionView, ...],
    queries,
    *,
    scale: float,
):
    rows = []
    for index, view in enumerate(views):
        keys, values = view.materialize()
        rows.append(
            mx.fast.scaled_dot_product_attention(
                queries[index : index + 1],
                keys,
                values,
                scale=scale,
            )
        )
    output = mx.concatenate(rows, axis=0)
    mx.eval(output)
    return output


def _release_all(source: PagedKVCacheBundle, children: list[PagedKVCacheBundle]) -> None:
    for child in children:
        child.release()
    source.release()


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_shared_prefix_batch_attention_matches_native_without_materialization(
    batch_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PagedCacheManager(num_layers=1, block_size=16, max_blocks=64)
    source = _source_bundle(manager, prefix_tokens=31, seed=8600 + batch_size)
    children = _fork_with_suffixes(source, [1] * batch_size, seed=8610 + batch_size)
    views = tuple(child.layers[0].attention_view() for child in children)
    mx.random.seed(8620 + batch_size)
    queries = mx.random.normal((batch_size, 8, 1, 32)).astype(mx.float16)
    scale = 1.0 / math.sqrt(queries.shape[-1])

    try:
        expected = _rowwise_native_attention(views, queries, scale=scale)

        def fail_materialize(_view: PagedAttentionView):
            raise AssertionError("shared-prefix batch attention materialized K/V")

        def fail_native_merge(_cls, _caches):
            raise AssertionError("shared-prefix batch attention invoked native merge")

        monkeypatch.setattr(PagedAttentionView, "materialize", fail_materialize)
        monkeypatch.setattr(PagedKVCacheLayer, "merge", classmethod(fail_native_merge))

        batch_view = PagedBatchAttentionView.from_views(views)
        actual = batch_view.attention(queries, scale=scale)
        mx.eval(actual)

        assert tuple(actual.shape) == tuple(expected.shape)
        assert tuple(batch_view.block_tables.shape) == (batch_size, 2)
        assert np.asarray(batch_view.sequence_lengths).tolist() == [32] * batch_size
        assert batch_view.metadata_nbytes <= batch_size * 3 * 4
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=3e-3,
            atol=3e-3,
        )
    finally:
        _release_all(source, children)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_shared_prefix_batch_attention_supports_unequal_sequence_lengths() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=16, max_blocks=64)
    source = _source_bundle(manager, prefix_tokens=17, seed=8630)
    children = _fork_with_suffixes(source, [1, 4, 17], seed=8640)
    views = tuple(child.layers[0].attention_view() for child in children)
    mx.random.seed(8650)
    queries = mx.random.normal((3, 8, 1, 32)).astype(mx.float16)
    scale = 1.0 / math.sqrt(queries.shape[-1])

    try:
        expected = _rowwise_native_attention(views, queries, scale=scale)
        batch_view = PagedBatchAttentionView.from_views(views)
        actual = batch_view.attention(queries, scale=scale)
        mx.eval(actual)

        assert np.asarray(batch_view.sequence_lengths).tolist() == [18, 21, 34]
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=3e-3,
            atol=3e-3,
        )
    finally:
        _release_all(source, children)


def test_shared_prefix_batch_view_tracks_suffix_cow_and_borrows_pool_lifetime() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=16, max_blocks=64)
    source = _source_bundle(manager, prefix_tokens=17, seed=8660)
    children = _fork_with_suffixes(source, [1, 1, 1], seed=8670)
    views = tuple(child.layers[0].attention_view() for child in children)
    pool = source.layers[0]._pool

    batch_view = PagedBatchAttentionView.from_views(views)

    assert len({view.block_ids[0] for view in views}) == 1
    assert len({view.block_ids[-1] for view in views}) == len(views)
    assert len({id(view.layer._pool) for view in views}) == 1
    assert pool.nbytes > 0

    _release_all(source, children)

    assert manager.stats.allocated_blocks == 0
    assert pool.nbytes == 0
    with pytest.raises(ValueError, match="empty paged KV pool"):
        batch_view.attention(mx.zeros((3, 8, 1, 32)), scale=1.0)


def test_shared_prefix_batch_view_rejects_independent_pools() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=16, max_blocks=64)
    first = _source_bundle(manager, prefix_tokens=16, seed=8680)
    second = _source_bundle(manager, prefix_tokens=16, seed=8690)

    try:
        with pytest.raises(ValueError, match="share one physical pool"):
            PagedBatchAttentionView.from_views(
                (first.layers[0].attention_view(), second.layers[0].attention_view())
            )
    finally:
        first.release()
        second.release()
