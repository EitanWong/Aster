from __future__ import annotations

import math
from typing import Any


_KERNEL: Any | None = None


_SOURCE = r"""
    uint elem = thread_position_in_grid.x;
    uint B = queries_shape[0];
    uint Hq = queries_shape[1];
    uint Q = queries_shape[2];
    uint Dk = queries_shape[3];
    uint Hkv = key_pool_shape[2];
    uint block_size = key_pool_shape[3];
    uint Dv = value_pool_shape[4];
    uint query_offset_value = query_offset[0];
    uint total_tokens = total_kv_tokens[0];
    uint gqa = Hq / Hkv;

    uint value_dim = elem % Dv;
    uint q_index = elem / Dv;
    uint query_index = q_index % Q;
    uint head_index = (q_index / Q) % Hq;
    uint batch_index = q_index / (Q * Hq);
    uint kv_head_index = head_index / gqa;

    ulong query_base =
        (((ulong)batch_index * Hq + head_index) * Q + query_index) * Dk;
    float query_position = (float)(query_offset_value + query_index);
    float max_score = -INFINITY;

    for (uint token = 0; token < total_tokens; ++token) {
        if ((float)token > query_position) {
            continue;
        }
        uint logical_block = token / block_size;
        uint block_offset = token % block_size;
        uint physical_block = block_indices[logical_block];
        ulong key_base =
            ((((ulong)physical_block * B + batch_index) * Hkv + kv_head_index)
             * block_size + block_offset) * Dk;
        float score = 0.0f;
        for (uint d = 0; d < Dk; ++d) {
            score += (float)queries[query_base + d] * (float)key_pool[key_base + d];
        }
        score *= scale[0];
        if (score > max_score) {
            max_score = score;
        }
    }

    float denominator = 0.0f;
    float output_value = 0.0f;
    for (uint token = 0; token < total_tokens; ++token) {
        if ((float)token > query_position) {
            continue;
        }
        uint logical_block = token / block_size;
        uint block_offset = token % block_size;
        uint physical_block = block_indices[logical_block];
        ulong key_base =
            ((((ulong)physical_block * B + batch_index) * Hkv + kv_head_index)
             * block_size + block_offset) * Dk;
        ulong value_base =
            ((((ulong)physical_block * B + batch_index) * Hkv + kv_head_index)
             * block_size + block_offset) * Dv;
        float score = 0.0f;
        for (uint d = 0; d < Dk; ++d) {
            score += (float)queries[query_base + d] * (float)key_pool[key_base + d];
        }
        float weight = metal::exp(score * scale[0] - max_score);
        denominator += weight;
        output_value += weight * (float)value_pool[value_base + value_dim];
    }

    out[elem] = (T)(output_value / denominator);
"""


def _get_kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        import mlx.core as mx

        _KERNEL = mx.fast.metal_kernel(
            name="aster_paged_block_attention",
            input_names=[
                "queries",
                "key_pool",
                "value_pool",
                "block_indices",
                "query_offset",
                "total_kv_tokens",
                "scale",
            ],
            output_names=["out"],
            source=_SOURCE,
            compile_options={"math_mode": "safe"},
        )
    return _KERNEL


def paged_block_attention(
    queries: Any,
    key_pool: Any,
    value_pool: Any,
    block_indices: Any,
    *,
    query_offset: int,
    total_kv_tokens: int,
    scale: float,
) -> Any:
    """Run causal attention over a physical block pool without K/V concat.

    Shapes are ``queries=[B,Hq,Q,Dk]`` and
    ``key/value_pool=[P,B,Hkv,block,D]``. ``block_indices`` maps logical blocks
    to physical pool rows. This is a correctness-first proof of the kernel
    contract; it intentionally uses one thread per output element and is not
    yet the production attention implementation.
    """
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Metal is required for paged block attention")
    if len(queries.shape) != 4 or len(key_pool.shape) != 5 or len(value_pool.shape) != 5:
        raise ValueError("Unexpected paged attention tensor rank")
    if len(block_indices.shape) != 1:
        raise ValueError("block_indices must be one-dimensional")
    if key_pool.shape[:4] != value_pool.shape[:4]:
        raise ValueError("Key/value block pools have incompatible shapes")
    if queries.shape[0] != key_pool.shape[1] or queries.shape[1] % key_pool.shape[2] != 0:
        raise ValueError("Query and KV head dimensions are incompatible")
    if queries.shape[3] != key_pool.shape[4]:
        raise ValueError("Query and key head dimensions are incompatible")
    if total_kv_tokens <= 0 or total_kv_tokens > block_indices.shape[0] * key_pool.shape[3]:
        raise ValueError("total_kv_tokens exceeds the supplied block table")
    if query_offset < 0 or query_offset + queries.shape[2] > total_kv_tokens:
        raise ValueError("query_offset and query length exceed the KV sequence")

    output_shape = (
        queries.shape[0],
        queries.shape[1],
        queries.shape[2],
        value_pool.shape[4],
    )
    kernel = _get_kernel()
    outputs = kernel(
        inputs=[
            queries,
            key_pool,
            value_pool,
            block_indices.astype(mx.uint32),
            mx.array([query_offset], dtype=mx.uint32),
            mx.array([total_kv_tokens], dtype=mx.uint32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("T", value_pool.dtype)],
        grid=(math.prod(output_shape), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[value_pool.dtype],
        stream=mx.gpu,
    )
    return outputs[0]


__all__ = ["paged_block_attention"]
