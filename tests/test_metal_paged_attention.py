from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from aster.inference.metal_paged_attention import paged_block_attention


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_block_indexed_attention_matches_causal_mlx_attention() -> None:
    batch = 1
    query_heads = 4
    kv_heads = 2
    query_tokens = 3
    past_tokens = 2
    total_tokens = past_tokens + query_tokens
    block_size = 3
    head_dim = 4
    value_dim = 4

    queries = mx.arange(batch * query_heads * query_tokens * head_dim, dtype=mx.float32)
    queries = queries.reshape(batch, query_heads, query_tokens, head_dim) / 17.0
    pool = mx.arange(2 * batch * kv_heads * block_size * head_dim, dtype=mx.float32)
    pool = pool.reshape(2, batch, kv_heads, block_size, head_dim) / 23.0
    values = mx.arange(2 * batch * kv_heads * block_size * value_dim, dtype=mx.float32)
    values = values.reshape(2, batch, kv_heads, block_size, value_dim) / 29.0
    block_indices = mx.array([1, 0], dtype=mx.uint32)
    keys = mx.concatenate(
        [pool[1].reshape(batch, kv_heads, block_size, head_dim),
         pool[0].reshape(batch, kv_heads, block_size, head_dim)],
        axis=2,
    )[..., :total_tokens, :]
    dense_values = mx.concatenate(
        [values[1].reshape(batch, kv_heads, block_size, value_dim),
         values[0].reshape(batch, kv_heads, block_size, value_dim)],
        axis=2,
    )[..., :total_tokens, :]
    mask = (
        mx.arange(total_tokens)[None, :] <= (past_tokens + mx.arange(query_tokens))[:, None]
    )[None, None, :, :]
    scale = 1.0 / math.sqrt(head_dim)

    expected = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        dense_values,
        scale=scale,
        mask=mask,
    )
    actual = paged_block_attention(
        queries,
        pool,
        values,
        block_indices,
        query_offset=past_tokens,
        total_kv_tokens=total_tokens,
        scale=scale,
    )
    mx.eval(expected, actual)

    # Native MLX and the proof kernel use different reduction orders; the
    # tolerance is still far below a token-selection tie for this smoke.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_block_indexed_attention_supports_float16_decode() -> None:
    mx.random.seed(17)
    batch, query_heads, kv_heads, head_dim = 1, 2, 2, 8
    block_size, total_tokens = 4, 7
    queries = mx.random.normal((batch, query_heads, 1, head_dim)).astype(mx.float16)
    key_pool = mx.random.normal(
        (2, batch, kv_heads, block_size, head_dim)
    ).astype(mx.float16)
    value_pool = mx.random.normal(
        (2, batch, kv_heads, block_size, head_dim)
    ).astype(mx.float16)
    block_indices = mx.array([1, 0], dtype=mx.uint32)
    keys = mx.concatenate([key_pool[1], key_pool[0]], axis=2)[..., :total_tokens, :]
    values = mx.concatenate([value_pool[1], value_pool[0]], axis=2)[..., :total_tokens, :]
    scale = 1.0 / math.sqrt(head_dim)

    expected = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=None
    )
    actual = paged_block_attention(
        queries,
        key_pool,
        value_pool,
        block_indices,
        query_offset=total_tokens - 1,
        total_kv_tokens=total_tokens,
        scale=scale,
    )
    mx.eval(expected, actual)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_tiled_block_indexed_attention_matches_native_attention() -> None:
    mx.random.seed(23)
    batch, query_heads, kv_heads, head_dim = 1, 2, 1, 32
    block_size, total_tokens = 32, 64
    queries = mx.random.normal((batch, query_heads, 1, head_dim)).astype(mx.float16)
    key_pool = mx.random.normal(
        (2, batch, kv_heads, block_size, head_dim)
    ).astype(mx.float16)
    value_pool = mx.random.normal(
        (2, batch, kv_heads, block_size, head_dim)
    ).astype(mx.float16)
    block_indices = mx.array([1, 0], dtype=mx.uint32)
    keys = mx.concatenate([key_pool[1], key_pool[0]], axis=2)[..., :total_tokens, :]
    values = mx.concatenate([value_pool[1], value_pool[0]], axis=2)[..., :total_tokens, :]
    scale = 1.0 / math.sqrt(head_dim)

    expected = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale
    )
    actual = paged_block_attention(
        queries,
        key_pool,
        value_pool,
        block_indices,
        query_offset=total_tokens - 1,
        total_kv_tokens=total_tokens,
        scale=scale,
    )
    mx.eval(expected, actual)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=5e-3, atol=5e-3)
