#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

namespace {

constexpr char kLibraryName[] = "aster_paged_primitive_proto_v1";
constexpr char kKernelName[] = "aster_paged_vector_f16_d128_v128";

constexpr char kMetalSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
  uint B;
  uint Hq;
  uint Q;
  uint Dk;
  uint Hkv;
  uint block_size;
  uint Dv;
  uint query_offset;
  uint total_tokens;
  uint gqa;
  uint physical_blocks;
  float scale;
};

[[kernel, max_total_threads_per_threadgroup(1024)]]
void aster_paged_vector_f16_d128_v128(
    device const half* queries [[buffer(0)]],
    device const half* key_pool [[buffer(1)]],
    device const half* value_pool [[buffer(2)]],
    device const uint* block_indices [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BLOCKS = 32;
  constexpr uint HEAD_DIM = 32;
  constexpr uint QK_PER_THREAD = 4;
  constexpr uint V_PER_THREAD = 4;

  const uint query_id = threadgroup_position_in_grid.x;
  const uint query_index = query_id % params.Q;
  const uint head_index = (query_id / params.Q) % params.Hq;
  const uint batch_index = query_id / (params.Q * params.Hq);
  const uint kv_head_index = head_index / params.gqa;

  thread float query_fragment[QK_PER_THREAD];
  thread float key_fragment[QK_PER_THREAD];
  thread float output_fragment[V_PER_THREAD];
  threadgroup float max_scores[BLOCKS];
  threadgroup float sum_exp_scores[BLOCKS];
  threadgroup uint invalid_blocks[BLOCKS];

  const ulong query_base =
      (((ulong)batch_index * params.Hq + head_index) * params.Q + query_index) *
      params.Dk;
  for (uint d = 0; d < QK_PER_THREAD; ++d) {
    query_fragment[d] =
        (float)queries[query_base + lane * QK_PER_THREAD + d] * params.scale;
  }
  for (uint d = 0; d < V_PER_THREAD; ++d) {
    output_fragment[d] = 0.0f;
  }

  float max_score = -INFINITY;
  float denominator = 0.0f;
  bool invalid_block = false;
  const uint query_position = params.query_offset + query_index;
  for (uint token = simd_group; token < params.total_tokens; token += BLOCKS) {
    if (token > query_position) {
      continue;
    }
    const uint logical_block = token / params.block_size;
    const uint block_offset = token % params.block_size;
    const uint physical_block = block_indices[logical_block];
    if (physical_block >= params.physical_blocks) {
      invalid_block = true;
      continue;
    }
    const ulong key_base =
        ((((ulong)physical_block * params.B + batch_index) * params.Hkv +
          kv_head_index) *
             params.block_size +
         block_offset) *
        params.Dk;
    for (uint d = 0; d < QK_PER_THREAD; ++d) {
      key_fragment[d] =
          (float)key_pool[key_base + lane * QK_PER_THREAD + d];
    }
    float partial_score = 0.0f;
    for (uint d = 0; d < QK_PER_THREAD; ++d) {
      partial_score += query_fragment[d] * key_fragment[d];
    }
    const float score = simd_sum(partial_score);
    const float new_max = max(max_score, score);
    const float factor = metal::exp(max_score - new_max);
    const float exp_score = metal::exp(score - new_max);
    max_score = new_max;
    denominator = denominator * factor + exp_score;

    const ulong value_base =
        ((((ulong)physical_block * params.B + batch_index) * params.Hkv +
          kv_head_index) *
             params.block_size +
         block_offset) *
        params.Dv;
    for (uint d = 0; d < V_PER_THREAD; ++d) {
      output_fragment[d] = output_fragment[d] * factor +
          exp_score * (float)value_pool[value_base + lane * V_PER_THREAD + d];
    }
  }

  if (lane == 0) {
    max_scores[simd_group] = max_score;
    sum_exp_scores[simd_group] = denominator;
    invalid_blocks[simd_group] = invalid_block ? 1 : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool invalid_any = simd_max(invalid_blocks[lane]) != 0;
  max_score = max_scores[lane];
  const float new_max = simd_max(max_score);
  const float factor = metal::exp(max_score - new_max);
  denominator = simd_sum(sum_exp_scores[lane] * factor);

  threadgroup float partial_outputs[BLOCKS * HEAD_DIM];
  const ulong output_base =
      (((ulong)batch_index * params.Hq + head_index) * params.Q + query_index) *
      params.Dv;
  for (uint d = 0; d < V_PER_THREAD; ++d) {
    partial_outputs[lane * HEAD_DIM + simd_group] = output_fragment[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    output_fragment[d] =
        simd_sum(partial_outputs[simd_group * HEAD_DIM + lane] * factor);
    output_fragment[d] = denominator == 0.0f
        ? output_fragment[d]
        : output_fragment[d] / denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    for (uint d = 0; d < V_PER_THREAD; ++d) {
      out[output_base + simd_group * V_PER_THREAD + d] =
          invalid_any ? (half)NAN : (half)output_fragment[d];
    }
  }
}
)METAL";

struct AttentionParams {
  uint32_t B;
  uint32_t Hq;
  uint32_t Q;
  uint32_t Dk;
  uint32_t Hkv;
  uint32_t block_size;
  uint32_t Dv;
  uint32_t query_offset;
  uint32_t total_tokens;
  uint32_t gqa;
  uint32_t physical_blocks;
  float scale;
};

static_assert(sizeof(AttentionParams) == 48);

bool params_equal(const AttentionParams& left, const AttentionParams& right) {
  return left.B == right.B && left.Hq == right.Hq && left.Q == right.Q &&
      left.Dk == right.Dk && left.Hkv == right.Hkv &&
      left.block_size == right.block_size && left.Dv == right.Dv &&
      left.query_offset == right.query_offset &&
      left.total_tokens == right.total_tokens && left.gqa == right.gqa &&
      left.physical_blocks == right.physical_blocks && left.scale == right.scale;
}

uint32_t checked_u32(int64_t value, const char* name) {
  if (value < 0 ||
      static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) + " is outside uint32 range");
  }
  return static_cast<uint32_t>(value);
}

uint32_t checked_shape_dim(const mx::array& value, int axis, const char* name) {
  const auto dimension = value.shape(axis);
  if (dimension <= 0 ||
      static_cast<uint64_t>(dimension) > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) + " has an invalid dimension");
  }
  return static_cast<uint32_t>(dimension);
}

void require_row_contiguous(const mx::array& value, const char* name) {
  if (!value.flags().row_contiguous) {
    throw std::invalid_argument(std::string(name) + " must be row-contiguous");
  }
}

class PagedAttentionPrimitive final : public mx::Primitive {
 public:
  PagedAttentionPrimitive(mx::Stream stream, AttentionParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("PagedAttentionPrimitive has no CPU path");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& query = inputs[0];
    const auto& key_pool = inputs[1];
    const auto& value_pool = inputs[2];
    const auto& block_indices = inputs[3];
    auto& output = outputs[0];

    output.set_data(mx::allocator::malloc(output.nbytes()));

    auto& stream = this->stream();
    auto& device = mx::metal::device(stream.device);
    auto* library = device.get_library(kLibraryName, [] {
      return std::string(kMetalSource);
    });
    auto* kernel = device.get_kernel(kKernelName, library);
    if (kernel->threadExecutionWidth() != 32 ||
        kernel->maxTotalThreadsPerThreadgroup() < 1024) {
      throw std::runtime_error(
          "Paged attention prototype requires 32-wide SIMD and 1024 threads per group");
    }
    auto& encoder = mx::metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(query, 0);
    encoder.set_input_array(key_pool, 1);
    encoder.set_input_array(value_pool, 2);
    encoder.set_input_array(block_indices, 3);
    encoder.set_output_array(output, 4);
    encoder.set_bytes(params_, 5);
    encoder.dispatch_threadgroups(
        MTL::Size(
            static_cast<NS::UInteger>(params_.B) * params_.Hq * params_.Q,
            1,
            1),
        MTL::Size(1024, 1, 1));
  }

  const char* name() const override {
    return "AsterPagedAttentionPrototype";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& right = static_cast<const PagedAttentionPrimitive&>(other);
    return params_equal(params_, right.params_);
  }

 private:
  AttentionParams params_;
};

mx::array paged_attention(
    const mx::array& query,
    const mx::array& key_pool,
    const mx::array& value_pool,
    const mx::array& block_indices,
    int64_t query_offset,
    int64_t total_tokens,
    float scale,
    mx::StreamOrDevice stream_or_device = {}) {
  if (query.ndim() != 4 || key_pool.ndim() != 5 || value_pool.ndim() != 5) {
    throw std::invalid_argument("Unexpected paged attention tensor rank");
  }
  if (block_indices.ndim() != 1) {
    throw std::invalid_argument("block_indices must be one-dimensional");
  }
  if (query.dtype() != mx::float16 || key_pool.dtype() != mx::float16 ||
      value_pool.dtype() != mx::float16) {
    throw std::invalid_argument("Prototype supports float16 tensors only");
  }
  if (block_indices.dtype() != mx::uint32) {
    throw std::invalid_argument("block_indices must use uint32");
  }
  require_row_contiguous(query, "query");
  require_row_contiguous(key_pool, "key_pool");
  require_row_contiguous(value_pool, "value_pool");
  require_row_contiguous(block_indices, "block_indices");

  const uint32_t B = checked_shape_dim(query, 0, "query");
  const uint32_t Hq = checked_shape_dim(query, 1, "query");
  const uint32_t Q = checked_shape_dim(query, 2, "query");
  const uint32_t Dk = checked_shape_dim(query, 3, "query");
  const uint32_t physical_blocks = checked_shape_dim(key_pool, 0, "key_pool");
  const uint32_t pool_batch = checked_shape_dim(key_pool, 1, "key_pool");
  const uint32_t Hkv = checked_shape_dim(key_pool, 2, "key_pool");
  const uint32_t block_size = checked_shape_dim(key_pool, 3, "key_pool");
  const uint32_t key_dim = checked_shape_dim(key_pool, 4, "key_pool");
  const uint32_t Dv = checked_shape_dim(value_pool, 4, "value_pool");
  const uint32_t offset = checked_u32(query_offset, "query_offset");
  const uint32_t token_count = checked_u32(total_tokens, "total_tokens");

  if (Dk != 128 || key_dim != 128 || Dv != 128) {
    throw std::invalid_argument("Prototype requires Dk=Dv=128");
  }
  if (Q != 1) {
    throw std::invalid_argument("Prototype supports decode query length 1 only");
  }
  if (pool_batch != B || value_pool.shape(0) != physical_blocks ||
      value_pool.shape(1) != pool_batch || value_pool.shape(2) != Hkv ||
      value_pool.shape(3) != block_size) {
    throw std::invalid_argument("Key/value block pools have incompatible shapes");
  }
  if (Hq % Hkv != 0) {
    throw std::invalid_argument("Query and KV head dimensions are incompatible");
  }
  const uint64_t logical_capacity =
      static_cast<uint64_t>(block_indices.size()) * block_size;
  if (token_count == 0 || token_count > logical_capacity) {
    throw std::invalid_argument("total_tokens exceeds the block table capacity");
  }
  if (static_cast<uint64_t>(offset) + Q > token_count) {
    throw std::invalid_argument("query_offset and query length exceed total_tokens");
  }
  if (!std::isfinite(scale) || scale <= 0.0f) {
    throw std::invalid_argument("scale must be finite and positive");
  }

  const AttentionParams params{
      B,
      Hq,
      Q,
      Dk,
      Hkv,
      block_size,
      Dv,
      offset,
      token_count,
      Hq / Hkv,
      physical_blocks,
      scale,
  };
  auto stream = mx::to_stream(stream_or_device);
  return mx::array(
      {static_cast<int>(B), static_cast<int>(Hq), static_cast<int>(Q),
       static_cast<int>(Dv)},
      mx::float16,
      std::make_shared<PagedAttentionPrimitive>(stream, params),
      {query, key_pool, value_pool, block_indices});
}

}  // namespace

NB_MODULE(_aster_paged_primitive, module) {
  module.doc() = "Throwaway MLX Primitive prototype for Aster paged attention";
  module.def(
      "paged_attention",
      &paged_attention,
      "query"_a,
      "key_pool"_a,
      "value_pool"_a,
      "block_indices"_a,
      "query_offset"_a,
      "total_tokens"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
}
