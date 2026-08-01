from __future__ import annotations

import math
from typing import Any

_KERNEL: Any | None = None
_DECODE_KERNEL: Any | None = None
_TILED_KERNEL: Any | None = None
_VECTOR_KERNEL: Any | None = None
_BATCH_VECTOR_KERNEL: Any | None = None


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


_DECODE_SOURCE = r"""
    uint query_id = thread_position_in_grid.x;
    uint B = queries_shape[0];
    uint Hq = queries_shape[1];
    uint Dk = queries_shape[3];
    uint Hkv = key_pool_shape[2];
    uint block_size = key_pool_shape[3];
    uint Dv = value_pool_shape[4];
    uint query_offset_value = query_offset[0];
    uint total_tokens = total_kv_tokens[0];
    uint gqa = Hq / Hkv;
    uint head_index = query_id % Hq;
    uint batch_index = query_id / Hq;
    uint kv_head_index = head_index / gqa;
    ulong query_base = ((ulong)batch_index * Hq + head_index) * Dk;
    float max_score = -INFINITY;
    float accum[256];

    for (uint d = 0; d < Dv; ++d) {
        accum[d] = 0.0f;
    }

    for (uint token = 0; token < total_tokens; ++token) {
        if (token > query_offset_value) {
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
    for (uint token = 0; token < total_tokens; ++token) {
        if (token > query_offset_value) {
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
        for (uint d = 0; d < Dv; ++d) {
            accum[d] += weight * (float)value_pool[value_base + d];
        }
    }

    ulong output_base = ((ulong)batch_index * Hq + head_index) * Dv;
    for (uint d = 0; d < Dv; ++d) {
        out[output_base + d] = (T)(accum[d] / denominator);
    }
"""


_TILED_SOURCE = r"""
    constexpr uint SIMD_WIDTH = 32;
    constexpr uint QK_PER_LANE = D / SIMD_WIDTH;
    constexpr uint V_PER_LANE = V / SIMD_WIDTH;
    uint query_id = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
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
    uint query_index = query_id % Q;
    uint head_index = (query_id / Q) % Hq;
    uint batch_index = query_id / (Q * Hq);
    uint kv_head_index = head_index / gqa;

    thread float query_fragment[QK_PER_LANE];
    thread float output_fragment[V_PER_LANE];
    ulong query_base =
        (((ulong)batch_index * Hq + head_index) * Q + query_index) * Dk;
    ulong output_base =
        (((ulong)batch_index * Hq + head_index) * Q + query_index) * Dv;
    for (uint d = 0; d < QK_PER_LANE; ++d) {
        query_fragment[d] = (float)queries[query_base + lane * QK_PER_LANE + d];
    }
    for (uint d = 0; d < V_PER_LANE; ++d) {
        output_fragment[d] = 0.0f;
    }

    float max_score = -INFINITY;
    float denominator = 0.0f;
    float query_position = (float)(query_offset_value + query_index);
    for (uint token = 0; token < total_tokens; ++token) {
        uint logical_block = token / block_size;
        uint block_offset = token % block_size;
        uint physical_block = block_indices[logical_block];
        ulong key_base =
            ((((ulong)physical_block * B + batch_index) * Hkv + kv_head_index)
             * block_size + block_offset) * Dk;
        float partial_score = 0.0f;
        for (uint d = 0; d < QK_PER_LANE; ++d) {
            partial_score += query_fragment[d] *
                (float)key_pool[key_base + lane * QK_PER_LANE + d];
        }
        float score = simd_sum(partial_score) * scale[0];
        if ((float)token > query_position) {
            score = -INFINITY;
        }
        float factor = 1.0f;
        float exp_score = 0.0f;
        if (lane == 0) {
            float new_max = max(max_score, score);
            factor = metal::exp(max_score - new_max);
            exp_score = metal::exp(score - new_max);
            max_score = new_max;
            denominator = denominator * factor + exp_score;
        }
        factor = simd_broadcast(factor, 0);
        exp_score = simd_broadcast(exp_score, 0);

        ulong value_base =
            ((((ulong)physical_block * B + batch_index) * Hkv + kv_head_index)
             * block_size + block_offset) * Dv;
        for (uint d = 0; d < V_PER_LANE; ++d) {
            output_fragment[d] = output_fragment[d] * factor + exp_score *
                (float)value_pool[value_base + lane * V_PER_LANE + d];
        }
    }

    float denominator_value = simd_broadcast(denominator, 0);
    for (uint d = 0; d < V_PER_LANE; ++d) {
        out[output_base + lane * V_PER_LANE + d] =
            (T)(output_fragment[d] / denominator_value);
    }
"""


_VECTOR_SOURCE = r"""
    constexpr uint BLOCKS = 32;
    constexpr uint HEAD_DIM = 32;
    constexpr uint QK_PER_THREAD = D / HEAD_DIM;
    constexpr uint V_PER_THREAD = V / HEAD_DIM;
    uint query_id = threadgroup_position_in_grid.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
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
    uint query_index = query_id % Q;
    uint head_index = (query_id / Q) % Hq;
    uint batch_index = query_id / (Q * Hq);
    uint kv_head_index = head_index / gqa;

    thread float query_fragment[QK_PER_THREAD];
    thread float key_fragment[QK_PER_THREAD];
    thread float output_fragment[V_PER_THREAD];
    threadgroup float max_scores[BLOCKS];
    threadgroup float sum_exp_scores[BLOCKS];
    ulong query_base =
        (((ulong)batch_index * Hq + head_index) * Q + query_index) * Dk;
    for (uint d = 0; d < QK_PER_THREAD; ++d) {
        query_fragment[d] =
            (float)queries[query_base + lane * QK_PER_THREAD + d] * scale[0];
    }
    for (uint d = 0; d < V_PER_THREAD; ++d) {
        output_fragment[d] = 0.0f;
    }

    float max_score = -INFINITY;
    float denominator = 0.0f;
    float query_position = (float)(query_offset_value + query_index);
    for (uint token = simd_group; token < total_tokens; token += BLOCKS) {
        if (token > query_position) {
            continue;
        }
        uint logical_block = token / block_size;
        uint block_offset = token % block_size;
        uint physical_block = block_indices[logical_block];
        ulong key_base =
            ((((ulong)physical_block * B + batch_index) * Hkv + kv_head_index)
             * block_size + block_offset) * Dk;
        for (uint d = 0; d < QK_PER_THREAD; ++d) {
            key_fragment[d] =
                (float)key_pool[key_base + lane * QK_PER_THREAD + d];
        }
        float partial_score = 0.0f;
        for (uint d = 0; d < QK_PER_THREAD; ++d) {
            partial_score += query_fragment[d] * key_fragment[d];
        }
        float score = simd_sum(partial_score);
        float new_max = max(max_score, score);
        float factor = metal::exp(max_score - new_max);
        float exp_score = metal::exp(score - new_max);
        max_score = new_max;
        denominator = denominator * factor + exp_score;

        ulong value_base =
            ((((ulong)physical_block * B + batch_index) * Hkv + kv_head_index)
             * block_size + block_offset) * Dv;
        for (uint d = 0; d < V_PER_THREAD; ++d) {
            output_fragment[d] = output_fragment[d] * factor + exp_score *
                (float)value_pool[value_base + lane * V_PER_THREAD + d];
        }
    }

    if (lane == 0) {
        max_scores[simd_group] = max_score;
        sum_exp_scores[simd_group] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[lane];
    float new_max = simd_max(max_score);
    float factor = metal::exp(max_score - new_max);
    denominator = simd_sum(sum_exp_scores[lane] * factor);

    threadgroup float partial_outputs[BLOCKS * HEAD_DIM];
    ulong output_base =
        (((ulong)batch_index * Hq + head_index) * Q + query_index) * Dv;
    for (uint d = 0; d < V_PER_THREAD; ++d) {
        partial_outputs[lane * HEAD_DIM + simd_group] = output_fragment[d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        output_fragment[d] = simd_sum(
            partial_outputs[simd_group * HEAD_DIM + lane] * factor
        );
        output_fragment[d] = denominator == 0.0f
            ? output_fragment[d]
            : output_fragment[d] / denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint d = 0; d < V_PER_THREAD; ++d) {
            out[output_base + simd_group * V_PER_THREAD + d] =
                (T)output_fragment[d];
        }
    }
"""


_BATCH_VECTOR_SOURCE = r"""
    constexpr uint BLOCKS = 32;
    constexpr uint HEAD_DIM = 32;
    constexpr uint QK_PER_THREAD = D / HEAD_DIM;
    constexpr uint V_PER_THREAD = V / HEAD_DIM;
    uint query_id = threadgroup_position_in_grid.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint Hq = queries_shape[1];
    uint Q = queries_shape[2];
    uint Dk = queries_shape[3];
    uint pool_batch = key_pool_shape[1];
    uint Hkv = key_pool_shape[2];
    uint block_size = key_pool_shape[3];
    uint Dv = value_pool_shape[4];
    uint max_blocks = block_tables_shape[1];
    uint gqa = Hq / Hkv;
    uint query_index = query_id % Q;
    uint head_index = (query_id / Q) % Hq;
    uint batch_index = query_id / (Q * Hq);
    uint kv_head_index = head_index / gqa;
    uint total_tokens = sequence_lengths[batch_index];
    uint query_offset_value = total_tokens - Q;

    thread float query_fragment[QK_PER_THREAD];
    thread float key_fragment[QK_PER_THREAD];
    thread float output_fragment[V_PER_THREAD];
    threadgroup float max_scores[BLOCKS];
    threadgroup float sum_exp_scores[BLOCKS];
    ulong query_base =
        (((ulong)batch_index * Hq + head_index) * Q + query_index) * Dk;
    for (uint d = 0; d < QK_PER_THREAD; ++d) {
        query_fragment[d] =
            (float)queries[query_base + lane * QK_PER_THREAD + d] * scale[0];
    }
    for (uint d = 0; d < V_PER_THREAD; ++d) {
        output_fragment[d] = 0.0f;
    }

    float max_score = -INFINITY;
    float denominator = 0.0f;
    float query_position = (float)(query_offset_value + query_index);
    for (uint token = simd_group; token < total_tokens; token += BLOCKS) {
        if ((float)token > query_position) {
            continue;
        }
        uint logical_block = token / block_size;
        uint block_offset = token % block_size;
        uint physical_block =
            block_tables[batch_index * max_blocks + logical_block];
        ulong key_base =
            ((((ulong)physical_block * pool_batch) * Hkv + kv_head_index)
             * block_size + block_offset) * Dk;
        for (uint d = 0; d < QK_PER_THREAD; ++d) {
            key_fragment[d] =
                (float)key_pool[key_base + lane * QK_PER_THREAD + d];
        }
        float partial_score = 0.0f;
        for (uint d = 0; d < QK_PER_THREAD; ++d) {
            partial_score += query_fragment[d] * key_fragment[d];
        }
        float score = simd_sum(partial_score);
        float new_max = max(max_score, score);
        float factor = metal::exp(max_score - new_max);
        float exp_score = metal::exp(score - new_max);
        max_score = new_max;
        denominator = denominator * factor + exp_score;

        ulong value_base =
            ((((ulong)physical_block * pool_batch) * Hkv + kv_head_index)
             * block_size + block_offset) * Dv;
        for (uint d = 0; d < V_PER_THREAD; ++d) {
            output_fragment[d] = output_fragment[d] * factor + exp_score *
                (float)value_pool[value_base + lane * V_PER_THREAD + d];
        }
    }

    if (lane == 0) {
        max_scores[simd_group] = max_score;
        sum_exp_scores[simd_group] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[lane];
    float new_max = simd_max(max_score);
    float factor = metal::exp(max_score - new_max);
    denominator = simd_sum(sum_exp_scores[lane] * factor);

    threadgroup float partial_outputs[BLOCKS * HEAD_DIM];
    ulong output_base =
        (((ulong)batch_index * Hq + head_index) * Q + query_index) * Dv;
    for (uint d = 0; d < V_PER_THREAD; ++d) {
        partial_outputs[lane * HEAD_DIM + simd_group] = output_fragment[d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        output_fragment[d] = simd_sum(
            partial_outputs[simd_group * HEAD_DIM + lane] * factor
        );
        output_fragment[d] = denominator == 0.0f
            ? output_fragment[d]
            : output_fragment[d] / denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint d = 0; d < V_PER_THREAD; ++d) {
            out[output_base + simd_group * V_PER_THREAD + d] =
                (T)output_fragment[d];
        }
    }
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


def _get_decode_kernel() -> Any:
    global _DECODE_KERNEL
    if _DECODE_KERNEL is None:
        import mlx.core as mx

        _DECODE_KERNEL = mx.fast.metal_kernel(
            name="aster_paged_block_attention_decode",
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
            source=_DECODE_SOURCE,
            compile_options={"math_mode": "safe"},
        )
    return _DECODE_KERNEL


def _get_tiled_kernel() -> Any:
    global _TILED_KERNEL
    if _TILED_KERNEL is None:
        import mlx.core as mx

        _TILED_KERNEL = mx.fast.metal_kernel(
            name="aster_paged_block_attention_tiled",
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
            source=_TILED_SOURCE,
            compile_options={"math_mode": "safe"},
        )
    return _TILED_KERNEL


def _get_vector_kernel() -> Any:
    global _VECTOR_KERNEL
    if _VECTOR_KERNEL is None:
        import mlx.core as mx

        _VECTOR_KERNEL = mx.fast.metal_kernel(
            name="aster_paged_block_attention_vector",
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
            source=_VECTOR_SOURCE,
            compile_options={"math_mode": "safe"},
        )
    return _VECTOR_KERNEL


def _get_batch_vector_kernel() -> Any:
    global _BATCH_VECTOR_KERNEL
    if _BATCH_VECTOR_KERNEL is None:
        import mlx.core as mx

        _BATCH_VECTOR_KERNEL = mx.fast.metal_kernel(
            name="aster_paged_shared_batch_attention_vector",
            input_names=[
                "queries",
                "key_pool",
                "value_pool",
                "block_tables",
                "sequence_lengths",
                "scale",
            ],
            output_names=["out"],
            source=_BATCH_VECTOR_SOURCE,
            compile_options={"math_mode": "safe"},
        )
    return _BATCH_VECTOR_KERNEL


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
    to physical pool rows. Aligned long sequences use a token-parallel
    simdgroup kernel; short or non-aligned shapes retain correctness-first
    fallback kernels.
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
    tiled = queries.shape[3] % 32 == 0 and value_pool.shape[4] % 32 == 0
    vectorized = tiled and total_kv_tokens >= 32
    kernel = (
        _get_vector_kernel()
        if vectorized
        else _get_tiled_kernel()
        if tiled
        else _get_decode_kernel()
        if queries.shape[2] == 1 and value_pool.shape[4] <= 256
        else _get_kernel()
    )
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
        template=(
            [("T", value_pool.dtype), ("D", int(queries.shape[3])), ("V", int(value_pool.shape[4]))]
            if vectorized or tiled
            else [("T", value_pool.dtype)]
        ),
        grid=(
            queries.shape[0] * queries.shape[1] * queries.shape[2] * 1024
            if vectorized
            else queries.shape[0] * queries.shape[1] * queries.shape[2] * 32
            if tiled
            else math.prod(output_shape),
            1,
            1,
        ),
        threadgroup=(1024, 1, 1) if vectorized else (32, 1, 1) if tiled else (256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[value_pool.dtype],
        stream=mx.gpu,
    )
    return outputs[0]


def paged_batch_block_attention(
    queries: Any,
    key_pool: Any,
    value_pool: Any,
    block_tables: Any,
    sequence_lengths: Any,
    *,
    scale: float,
) -> Any:
    """Run vectorized attention for multiple block tables over one KV pool.

    The pool carries a singleton storage batch while ``block_tables`` and
    ``sequence_lengths`` select the logical sequence for each query row. This
    is the benchmark-only shared-prefix boundary; it never constructs a
    ``[B, Hkv, K, D]`` cache tensor.
    """
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Metal is required for paged batch attention")
    if len(queries.shape) != 4 or len(key_pool.shape) != 5 or len(value_pool.shape) != 5:
        raise ValueError("Unexpected paged batch attention tensor rank")
    if len(block_tables.shape) != 2 or len(sequence_lengths.shape) != 1:
        raise ValueError("Block tables and sequence lengths have incompatible ranks")
    if key_pool.shape[:4] != value_pool.shape[:4]:
        raise ValueError("Key/value block pools have incompatible shapes")
    if key_pool.shape[1] != 1:
        raise ValueError("Shared-prefix paged attention requires a singleton pool batch")
    batch_size = int(queries.shape[0])
    if block_tables.shape[0] != batch_size or sequence_lengths.shape[0] != batch_size:
        raise ValueError("Query rows and paged metadata have incompatible batch sizes")
    if block_tables.shape[1] == 0:
        raise ValueError("Block tables must contain at least one block")
    if queries.shape[1] % key_pool.shape[2] != 0:
        raise ValueError("Query and KV head dimensions are incompatible")
    if queries.shape[3] != key_pool.shape[4]:
        raise ValueError("Query and key head dimensions are incompatible")
    if queries.shape[3] % 32 != 0 or value_pool.shape[4] % 32 != 0:
        raise ValueError("Shared-prefix batch attention requires 32-aligned head dimensions")

    output_shape = (
        batch_size,
        queries.shape[1],
        queries.shape[2],
        value_pool.shape[4],
    )
    outputs = _get_batch_vector_kernel()(
        inputs=[
            queries,
            key_pool,
            value_pool,
            block_tables.astype(mx.uint32),
            sequence_lengths.astype(mx.uint32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", value_pool.dtype),
            ("D", int(queries.shape[3])),
            ("V", int(value_pool.shape[4])),
        ],
        grid=(batch_size * queries.shape[1] * queries.shape[2] * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[value_pool.dtype],
        stream=mx.gpu,
    )
    return outputs[0]


__all__ = ["paged_batch_block_attention", "paged_block_attention"]
