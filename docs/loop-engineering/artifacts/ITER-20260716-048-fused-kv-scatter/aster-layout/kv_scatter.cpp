#include <nanobind/nanobind.h>

#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

namespace {

constexpr char kLibraryName[] = "aster_kv_scatter_proto_v1";
constexpr char kKernelName[] = "aster_fused_kv_scatter_f16";

constexpr char kMetalSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ScatterParams {
  uint B;
  uint H;
  uint T;
  uint Dk;
  uint Dv;
  uint block_size;
  uint block_id;
  uint block_offset;
};

[[kernel]] void aster_fused_kv_scatter_f16(
    device const half* key [[buffer(0)]],
    device const half* value [[buffer(1)]],
    device half* key_cache [[buffer(2)]],
    device half* value_cache [[buffer(3)]],
    constant ScatterParams& params [[buffer(4)]],
    uint token_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const uint batch = token_id / params.T;
  const uint token = token_id % params.T;
  const uint max_width = params.H * max(params.Dk, params.Dv);

  for (uint index = tid; index < max_width; index += threads) {
    if (index < params.H * params.Dk) {
      const uint head = index / params.Dk;
      const uint dim = index % params.Dk;
      const ulong source =
          (((ulong)batch * params.H + head) * params.T + token) * params.Dk + dim;
      const ulong target =
          (((((ulong)params.block_id * params.B + batch) * params.H + head) *
             params.block_size + params.block_offset + token) * params.Dk) + dim;
      key_cache[target] = key[source];
    }
    if (index < params.H * params.Dv) {
      const uint head = index / params.Dv;
      const uint dim = index % params.Dv;
      const ulong source =
          (((ulong)batch * params.H + head) * params.T + token) * params.Dv + dim;
      const ulong target =
          (((((ulong)params.block_id * params.B + batch) * params.H + head) *
             params.block_size + params.block_offset + token) * params.Dv) + dim;
      value_cache[target] = value[source];
    }
  }
}
)METAL";

struct ScatterParams {
  uint32_t B;
  uint32_t H;
  uint32_t T;
  uint32_t Dk;
  uint32_t Dv;
  uint32_t block_size;
  uint32_t block_id;
  uint32_t block_offset;
};

static_assert(sizeof(ScatterParams) == 32);

bool params_equal(const ScatterParams& left, const ScatterParams& right) {
  return left.B == right.B && left.H == right.H && left.T == right.T &&
      left.Dk == right.Dk && left.Dv == right.Dv &&
      left.block_size == right.block_size && left.block_id == right.block_id &&
      left.block_offset == right.block_offset;
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

bool graph_references(const mx::array& root, std::uintptr_t target_id) {
  std::vector<mx::array> pending{root};
  std::unordered_set<std::uintptr_t> visited;
  while (!pending.empty()) {
    auto current = std::move(pending.back());
    pending.pop_back();
    if (!visited.insert(current.id()).second) {
      continue;
    }
    if (current.id() == target_id) {
      return true;
    }
    pending.insert(
        pending.end(), current.inputs().begin(), current.inputs().end());
  }
  return false;
}

bool may_overlap(const mx::array& left, const mx::array& right) {
  const auto& left_data = left.data_shared_ptr();
  const auto& right_data = right.data_shared_ptr();
  return left.id() == right.id() ||
      (left_data && right_data && left_data == right_data) ||
      graph_references(left, right.id());
}

class KVScatterPrimitive final : public mx::Primitive {
 public:
  KVScatterPrimitive(mx::Stream stream, ScatterParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("KVScatterPrimitive has no CPU path");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    outputs[0].copy_shared_buffer(inputs[2]);
    outputs[1].copy_shared_buffer(inputs[3]);

    auto& stream = this->stream();
    auto& device = mx::metal::device(stream.device);
    auto* library = device.get_library(kLibraryName, [] {
      return std::string(kMetalSource);
    });
    auto* kernel = device.get_kernel(kKernelName, library);
    const uint64_t width =
        static_cast<uint64_t>(params_.H) * std::max(params_.Dk, params_.Dv);
    const auto threads = static_cast<uint32_t>(std::min<uint64_t>(width, 256));
    if (threads == 0 || kernel->maxTotalThreadsPerThreadgroup() < threads) {
      throw std::runtime_error("KV scatter threadgroup is unsupported");
    }

    auto& encoder = mx::metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(outputs[0], 2);
    encoder.set_output_array(outputs[1], 3);
    encoder.set_bytes(params_, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(static_cast<NS::UInteger>(params_.B) * params_.T, 1, 1),
        MTL::Size(threads, 1, 1));
  }

  const char* name() const override {
    return "AsterKVScatterPrototype";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto* right = dynamic_cast<const KVScatterPrimitive*>(&other);
    return right != nullptr && params_equal(params_, right->params_);
  }

 private:
  ScatterParams params_;
};

std::vector<mx::array> fused_write_arrays(
    const mx::array& key,
    const mx::array& value,
    const mx::array& key_cache,
    const mx::array& value_cache,
    int64_t block_id,
    int64_t block_offset,
    mx::StreamOrDevice stream_or_device = {}) {
  if (key.ndim() != 4 || value.ndim() != 4 || key_cache.ndim() != 5 ||
      value_cache.ndim() != 5) {
    throw std::invalid_argument("Unexpected KV scatter tensor rank");
  }
  if (key.dtype() != mx::float16 || value.dtype() != mx::float16 ||
      key_cache.dtype() != mx::float16 || value_cache.dtype() != mx::float16) {
    throw std::invalid_argument("Prototype supports float16 tensors only");
  }
  require_row_contiguous(key, "key");
  require_row_contiguous(value, "value");
  require_row_contiguous(key_cache, "key_cache");
  require_row_contiguous(value_cache, "value_cache");
  if (may_overlap(key, key_cache) || may_overlap(key, value_cache) ||
      may_overlap(value, key_cache) || may_overlap(value, value_cache) ||
      may_overlap(key_cache, value_cache) || may_overlap(value_cache, key_cache)) {
    throw std::invalid_argument("KV scatter inputs and caches must not overlap");
  }

  const uint32_t B = checked_shape_dim(key, 0, "key");
  const uint32_t H = checked_shape_dim(key, 1, "key");
  const uint32_t T = checked_shape_dim(key, 2, "key");
  const uint32_t Dk = checked_shape_dim(key, 3, "key");
  const uint32_t Dv = checked_shape_dim(value, 3, "value");
  const uint32_t blocks = checked_shape_dim(key_cache, 0, "key_cache");
  const uint32_t block_size = checked_shape_dim(key_cache, 3, "key_cache");
  const uint32_t selected_block = checked_u32(block_id, "block_id");
  const uint32_t selected_offset = checked_u32(block_offset, "block_offset");

  if (value.shape(0) != B || value.shape(1) != H || value.shape(2) != T) {
    throw std::invalid_argument("Key/value inputs have incompatible shapes");
  }
  if (key_cache.shape(1) != B || key_cache.shape(2) != H ||
      key_cache.shape(4) != Dk || value_cache.shape(0) != blocks ||
      value_cache.shape(1) != B || value_cache.shape(2) != H ||
      value_cache.shape(3) != block_size || value_cache.shape(4) != Dv) {
    throw std::invalid_argument("Key/value caches have incompatible shapes");
  }
  if (selected_block >= blocks) {
    throw std::invalid_argument("block_id exceeds cache capacity");
  }
  if (static_cast<uint64_t>(selected_offset) + T > block_size) {
    throw std::invalid_argument("KV segment exceeds block capacity");
  }
  const uint64_t width =
      static_cast<uint64_t>(H) * std::max<uint32_t>(Dk, Dv);
  if (width > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("KV scatter width exceeds uint32 range");
  }

  const ScatterParams params{
      B, H, T, Dk, Dv, block_size, selected_block, selected_offset};
  auto stream = mx::to_stream(stream_or_device);
  return mx::array::make_arrays(
      {key_cache.shape(), value_cache.shape()},
      {key_cache.dtype(), value_cache.dtype()},
      std::make_shared<KVScatterPrimitive>(stream, params),
      {key, value, key_cache, value_cache});
}

nb::tuple fused_write(
    nb::handle key_handle,
    nb::handle value_handle,
    nb::handle key_cache_handle,
    nb::handle value_cache_handle,
    int64_t block_id,
    int64_t block_offset) {
  nb::object array_type = nb::module_::import_("mlx.core").attr("array");
  auto checked_array = [&array_type](nb::handle handle, const char* name)
      -> const mx::array& {
    auto* expected_type = reinterpret_cast<PyTypeObject*>(array_type.ptr());
    if (!PyObject_TypeCheck(handle.ptr(), expected_type)) {
      PyErr_Format(PyExc_TypeError, "%s must be mlx.core.array", name);
      throw nb::python_error();
    }
    return *nb::inst_ptr<mx::array>(handle);
  };
  auto outputs = fused_write_arrays(
      checked_array(key_handle, "key"),
      checked_array(value_handle, "value"),
      checked_array(key_cache_handle, "key_cache"),
      checked_array(value_cache_handle, "value_cache"),
      block_id,
      block_offset);
  nb::object out_key = array_type(nb::int_(0));
  nb::object out_value = array_type(nb::int_(0));
  nb::inst_ptr<mx::array>(out_key)->overwrite_descriptor(outputs[0]);
  nb::inst_ptr<mx::array>(out_value)->overwrite_descriptor(outputs[1]);
  return nb::make_tuple(out_key, out_value);
}

}  // namespace

NB_MODULE(_aster_kv_scatter, module) {
  module.doc() = "Throwaway fused KV scatter Primitive for Aster layout";
  module.def(
      "fused_write",
      &fused_write,
      "key"_a,
      "value"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_id"_a,
      "block_offset"_a);
}
