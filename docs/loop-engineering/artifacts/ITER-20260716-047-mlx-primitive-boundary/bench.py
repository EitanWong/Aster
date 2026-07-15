from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import mlx.core as mx

PROTOTYPE_ROOT = Path(__file__).resolve().parent
METHODS = ("public_fast", "direct_fast", "guarded_fast", "primitive")
GPU_DEVICE = mx.Device(mx.gpu)
_GUARDED_FAST_KERNEL: Any | None = None
_GUARDED_VECTOR_SOURCE = r"""
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
    uint physical_block_count = physical_blocks[0];
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
    threadgroup uint invalid_blocks[BLOCKS];
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
    bool invalid_block = false;
    uint query_position = query_offset_value + query_index;
    for (uint token = simd_group; token < total_tokens; token += BLOCKS) {
        if (token > query_position) {
            continue;
        }
        uint logical_block = token / block_size;
        uint block_offset = token % block_size;
        uint physical_block = block_indices[logical_block];
        if (physical_block >= physical_block_count) {
            invalid_block = true;
            continue;
        }
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
        invalid_blocks[simd_group] = invalid_block ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool invalid_any = simd_max(invalid_blocks[lane]) != 0;
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
                invalid_any ? (T)NAN : (T)output_fragment[d];
        }
    }
"""


def secure_build_dir() -> Path:
    configured = os.environ.get("ASTER_PRIMITIVE_BUILD_DIR")
    if configured is None:
        return Path(tempfile.mkdtemp(prefix="aster-iter047-build-"))
    candidate = Path(configured).expanduser()
    if candidate.is_symlink():
        raise RuntimeError("ASTER_PRIMITIVE_BUILD_DIR must not be a symlink")
    try:
        candidate.mkdir(mode=0o700)
    except FileExistsError:
        pass
    resolved = candidate.resolve(strict=True)
    metadata = resolved.stat()
    if not resolved.is_dir() or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise RuntimeError("Build directory must be owned by the current user with mode 0700")
    return resolved


BUILD_DIR = secure_build_dir()


@dataclass(frozen=True)
class Case:
    batch: int
    tokens: int
    query_offset: int
    block_size: int = 64
    query_heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    value_dim: int = 128


@dataclass(frozen=True)
class Inputs:
    query: Any
    key_pool: Any
    value_pool: Any
    block_indices: Any
    logical_to_physical: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aster-root", type=Path, default=Path.cwd())
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--verify-large", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_aster_kernel(aster_root: Path) -> Any:
    source = aster_root / "aster/inference/metal_paged_attention.py"
    spec = importlib.util.spec_from_file_location("aster_metal_paged_attention", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_extension() -> Any:
    cmake = Path(sys.executable).with_name("cmake")
    configure = [
        str(cmake),
        "-S",
        str(PROTOTYPE_ROOT),
        "-B",
        str(BUILD_DIR),
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    build = [str(cmake), "--build", str(BUILD_DIR), "--config", "Release", "-j", "4"]
    for command in (configure, build):
        result = subprocess.run(command, capture_output=True, check=False, text=True)
        if result.returncode:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    extension_paths = tuple(BUILD_DIR.rglob("_aster_paged_primitive*.so"))
    if len(extension_paths) != 1:
        raise RuntimeError(f"Expected one built extension, found {len(extension_paths)}")
    spec = importlib.util.spec_from_file_location(
        "_aster_paged_primitive", extension_paths[0]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension {extension_paths[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(case: Case, *, random_data: bool) -> Inputs:
    physical_blocks = math.ceil(case.tokens / case.block_size)
    shift = 1 if physical_blocks > 1 else 0
    mapping = tuple((index + shift) % physical_blocks for index in range(physical_blocks))
    query_shape = (case.batch, case.query_heads, 1, case.head_dim)
    pool_shape = (
        physical_blocks,
        case.batch,
        case.kv_heads,
        case.block_size,
        case.head_dim,
    )
    value_shape = pool_shape[:-1] + (case.value_dim,)
    if random_data:
        query = (mx.random.normal(query_shape) * 0.125).astype(mx.float16)
        key_pool = (mx.random.normal(pool_shape) * 0.125).astype(mx.float16)
        value_pool = (mx.random.normal(value_shape) * 0.125).astype(mx.float16)
    else:
        query = mx.full(query_shape, 0.03125, dtype=mx.float16)
        key_pool = mx.full(pool_shape, 0.015625, dtype=mx.float16)
        value_pool = mx.full(value_shape, 0.0625, dtype=mx.float16)
    block_indices = mx.array(mapping, dtype=mx.uint32)
    mx.eval(query, key_pool, value_pool, block_indices)
    mx.synchronize()
    return Inputs(query, key_pool, value_pool, block_indices, mapping)


def native_reference(case: Case, inputs: Inputs, scale: float) -> Any:
    keys = mx.concatenate(
        [inputs.key_pool[index] for index in inputs.logical_to_physical], axis=2
    )[:, :, : case.query_offset + 1, :]
    values = mx.concatenate(
        [inputs.value_pool[index] for index in inputs.logical_to_physical], axis=2
    )[:, :, : case.query_offset + 1, :]
    repeats = case.query_heads // case.kv_heads
    keys = mx.repeat(keys, repeats, axis=1)
    values = mx.repeat(values, repeats, axis=1)
    scores = mx.matmul(inputs.query, mx.swapaxes(keys, -1, -2)) * scale
    return mx.matmul(mx.softmax(scores, axis=-1), values)


def direct_fast_call(aster: Any, case: Case, inputs: Inputs, scale: float) -> Any:
    kernel = aster._get_vector_kernel()
    scalar_inputs = (
        mx.array([case.query_offset], dtype=mx.uint32),
        mx.array([case.tokens], dtype=mx.uint32),
        mx.array([scale], dtype=mx.float32),
    )
    mx.eval(*scalar_inputs)

    def call() -> Any:
        return kernel(
            inputs=[
                inputs.query,
                inputs.key_pool,
                inputs.value_pool,
                inputs.block_indices,
                *scalar_inputs,
            ],
            template=[("T", mx.float16), ("D", 128), ("V", 128)],
            grid=(case.batch * case.query_heads * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(case.batch, case.query_heads, 1, case.value_dim)],
            output_dtypes=[mx.float16],
            stream=GPU_DEVICE,
        )[0]

    return call


def guarded_fast_call(case: Case, inputs: Inputs, scale: float) -> Any:
    global _GUARDED_FAST_KERNEL
    if _GUARDED_FAST_KERNEL is None:
        _GUARDED_FAST_KERNEL = mx.fast.metal_kernel(
            name="aster_paged_block_attention_guarded_control",
            input_names=[
                "queries",
                "key_pool",
                "value_pool",
                "block_indices",
                "query_offset",
                "total_kv_tokens",
                "physical_blocks",
                "scale",
            ],
            output_names=["out"],
            source=_GUARDED_VECTOR_SOURCE,
            compile_options={"math_mode": "safe"},
        )
    scalar_inputs = (
        mx.array([case.query_offset], dtype=mx.uint32),
        mx.array([case.tokens], dtype=mx.uint32),
        mx.array([inputs.key_pool.shape[0]], dtype=mx.uint32),
        mx.array([scale], dtype=mx.float32),
    )
    mx.eval(*scalar_inputs)

    def call() -> Any:
        return _GUARDED_FAST_KERNEL(
            inputs=[
                inputs.query,
                inputs.key_pool,
                inputs.value_pool,
                inputs.block_indices,
                *scalar_inputs,
            ],
            template=[("T", mx.float16), ("D", 128), ("V", 128)],
            grid=(case.batch * case.query_heads * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(case.batch, case.query_heads, 1, case.value_dim)],
            output_dtypes=[mx.float16],
            stream=GPU_DEVICE,
        )[0]

    return call


def make_calls(aster: Any, extension: Any, case: Case, inputs: Inputs) -> dict[str, Callable[[], Any]]:
    scale = case.head_dim**-0.5

    def public_fast() -> Any:
        return aster.paged_block_attention(
            inputs.query,
            inputs.key_pool,
            inputs.value_pool,
            inputs.block_indices,
            query_offset=case.query_offset,
            total_kv_tokens=case.tokens,
            scale=scale,
        )

    direct_fast = direct_fast_call(aster, case, inputs, scale)
    guarded_fast = guarded_fast_call(case, inputs, scale)

    def primitive() -> Any:
        return extension.paged_attention(
            inputs.query,
            inputs.key_pool,
            inputs.value_pool,
            inputs.block_indices,
            case.query_offset,
            case.tokens,
            scale,
            stream=GPU_DEVICE,
        )

    return {
        "public_fast": public_fast,
        "direct_fast": direct_fast,
        "guarded_fast": guarded_fast,
        "primitive": primitive,
    }


def max_abs(left: Any, right: Any) -> float:
    if not bool(mx.all(mx.isfinite(left)).item()):
        raise AssertionError("Left comparison output contains non-finite values")
    if not bool(mx.all(mx.isfinite(right)).item()):
        raise AssertionError("Right comparison output contains non-finite values")
    delta = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    return float(delta.item())


def verify_correctness(aster: Any, extension: Any) -> dict[str, Any]:
    mx.random.seed(173)
    records: list[dict[str, Any]] = []
    max_cross = 0.0
    max_native = 0.0
    for tokens in (32, 33, 63, 64, 65, 129):
        for query_offset in (0, tokens // 2, tokens - 1):
            case = Case(batch=2 if tokens == 65 else 1, tokens=tokens, query_offset=query_offset)
            inputs = make_inputs(case, random_data=True)
            calls = make_calls(aster, extension, case, inputs)
            outputs = {name: call() for name, call in calls.items()}
            reference = native_reference(case, inputs, case.head_dim**-0.5)
            mx.eval(*outputs.values(), reference)
            mx.synchronize()
            native_errors = {name: max_abs(reference, output) for name, output in outputs.items()}
            cross_errors = {
                "direct_vs_public": max_abs(outputs["public_fast"], outputs["direct_fast"]),
                "guarded_vs_public": max_abs(outputs["public_fast"], outputs["guarded_fast"]),
                "primitive_vs_guarded": max_abs(outputs["guarded_fast"], outputs["primitive"]),
                "primitive_vs_public": max_abs(outputs["public_fast"], outputs["primitive"]),
            }
            cross = max(cross_errors.values())
            native = max(native_errors.values())
            max_cross = max(max_cross, cross)
            max_native = max(max_native, native)
            records.append(
                {
                    "tokens": tokens,
                    "query_offset": query_offset,
                    "batch": case.batch,
                    "cross_max_abs": cross_errors,
                    "native_max_abs": native_errors,
                }
            )
            del inputs, outputs, reference
    if max_cross > 1.3e-4:
        raise AssertionError(f"Primitive/current parity exceeded: {max_cross}")
    if max_native > 2.0e-3:
        raise AssertionError(f"Primitive/native parity exceeded: {max_native}")
    return {"max_cross_abs": max_cross, "max_native_abs": max_native, "cases": records}


def verify_large_shapes(
    aster: Any, extension: Any, batches: list[int], token_counts: list[int]
) -> dict[str, Any]:
    mx.random.seed(0xA57E047)
    records: list[dict[str, Any]] = []
    max_native = 0.0
    for tokens in token_counts:
        for batch in batches:
            case = Case(batch=batch, tokens=tokens, query_offset=tokens - 1)
            inputs = make_inputs(case, random_data=True)
            calls = make_calls(aster, extension, case, inputs)
            outputs = {name: call() for name, call in calls.items()}
            reference = native_reference(case, inputs, case.head_dim**-0.5)
            mx.eval(*outputs.values(), reference)
            native_errors = {name: max_abs(reference, output) for name, output in outputs.items()}
            cross_error = max(
                max_abs(outputs["public_fast"], outputs["direct_fast"]),
                max_abs(outputs["public_fast"], outputs["guarded_fast"]),
                max_abs(outputs["guarded_fast"], outputs["primitive"]),
                max_abs(outputs["public_fast"], outputs["primitive"]),
            )
            case_native = max(native_errors.values())
            if cross_error > 1.3e-4 or case_native > 2.0e-3:
                raise AssertionError(
                    f"Large-shape parity failed for tokens={tokens}, batch={batch}: "
                    f"cross={cross_error}, native={case_native}"
                )
            max_native = max(max_native, case_native)
            records.append(
                {
                    "tokens": tokens,
                    "batch": batch,
                    "cross_max_abs": cross_error,
                    "native_max_abs": native_errors,
                }
            )
            del calls, inputs, outputs, reference
            gc.collect()
            mx.clear_cache()
    return {"max_native_abs": max_native, "cases": records}


def expect_error(call: Callable[[], Any], label: str) -> str:
    try:
        call()
    except (TypeError, ValueError, RuntimeError) as error:
        return f"{type(error).__name__}: {error}"
    raise AssertionError(f"Expected validation failure: {label}")


def verify_validation(extension: Any) -> dict[str, str]:
    case = Case(batch=1, tokens=65, query_offset=64)
    inputs = make_inputs(case, random_data=False)
    scale = case.head_dim**-0.5

    def invoke(query: Any = inputs.query, *, offset: int = 64, total: int = 65) -> Any:
        return extension.paged_attention(
            query,
            inputs.key_pool,
            inputs.value_pool,
            inputs.block_indices,
            offset,
            total,
            scale,
            stream=GPU_DEVICE,
        )

    bad_block_indices = mx.array([0, 2], dtype=mx.uint32)
    bad_inputs = Inputs(
        inputs.query,
        inputs.key_pool,
        inputs.value_pool,
        bad_block_indices,
        inputs.logical_to_physical,
    )
    guarded = extension.paged_attention(
        inputs.query,
        inputs.key_pool,
        inputs.value_pool,
        bad_block_indices,
        64,
        65,
        scale,
        stream=GPU_DEVICE,
    )
    mx.eval(guarded)
    if not bool(mx.all(mx.isnan(guarded)).item()):
        raise AssertionError("Single-group physical block failure did not reach every output")
    guarded_fast_bad = guarded_fast_call(case, bad_inputs, scale)()
    mx.eval(guarded_fast_bad)
    if not bool(mx.all(mx.isnan(guarded_fast_bad)).item()):
        raise AssertionError("Guarded mx.fast control did not propagate the block failure")
    masked_invalid = extension.paged_attention(
        inputs.query,
        inputs.key_pool,
        inputs.value_pool,
        bad_block_indices,
        63,
        65,
        scale,
        stream=GPU_DEVICE,
    )
    mx.eval(masked_invalid)
    if not bool(mx.all(mx.isfinite(masked_invalid)).item()):
        raise AssertionError("Causally masked invalid block affected the output")
    masked_case = Case(batch=1, tokens=65, query_offset=63)
    guarded_fast_masked = guarded_fast_call(masked_case, bad_inputs, scale)()
    mx.eval(guarded_fast_masked)
    if not bool(mx.all(mx.isfinite(guarded_fast_masked)).item()):
        raise AssertionError("Guarded mx.fast control read a causally masked invalid block")

    return {
        "query_dtype": expect_error(lambda: invoke(inputs.query.astype(mx.float32)), "query dtype"),
        "query_offset": expect_error(lambda: invoke(offset=65), "query offset"),
        "token_capacity": expect_error(lambda: invoke(total=129), "token capacity"),
        "head_dim": expect_error(
            lambda: invoke(mx.zeros((1, 16, 1, 64), dtype=mx.float16)), "head dim"
        ),
        "query_length": expect_error(
            lambda: invoke(mx.zeros((1, 16, 2, 128), dtype=mx.float16), offset=63),
            "query length",
        ),
        "physical_block_bounds": "both guarded paths propagated one-group failure",
        "masked_physical_block": "both guarded paths stayed finite outside causal range",
    }


def timed_call(call: Callable[[], Any]) -> float:
    started = time.perf_counter_ns()
    output = call()
    mx.eval(output)
    return (time.perf_counter_ns() - started) / 1_000_000.0


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "stdev_ms": statistics.pstdev(ordered),
        "p10_ms": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def benchmark_case(
    aster: Any,
    extension: Any,
    case: Case,
    *,
    warmups: int,
    iterations: int,
    run_id: int,
) -> dict[str, Any]:
    mx.reset_peak_memory()
    inputs = make_inputs(case, random_data=False)
    calls = make_calls(aster, extension, case, inputs)
    parity_outputs = {name: call() for name, call in calls.items()}
    mx.eval(*parity_outputs.values())
    mx.synchronize()
    parity_max_abs = max(
        max_abs(parity_outputs["public_fast"], parity_outputs["direct_fast"]),
        max_abs(parity_outputs["public_fast"], parity_outputs["guarded_fast"]),
        max_abs(parity_outputs["guarded_fast"], parity_outputs["primitive"]),
        max_abs(parity_outputs["public_fast"], parity_outputs["primitive"]),
    )
    if parity_max_abs > 1.3e-4:
        raise AssertionError(f"Benchmark parity exceeded: {parity_max_abs}")
    del parity_outputs
    for index in range(warmups):
        order = METHODS[index % len(METHODS) :] + METHODS[: index % len(METHODS)]
        for name in order:
            timed_call(calls[name])
    samples = {name: [] for name in METHODS}
    for index in range(iterations):
        shift = (index + run_id) % len(METHODS)
        order = METHODS[shift:] + METHODS[:shift]
        for name in order:
            samples[name].append(timed_call(calls[name]))
    summaries = {name: summarize(values) for name, values in samples.items()}
    public = summaries["public_fast"]["median_ms"]
    direct = summaries["direct_fast"]["median_ms"]
    guarded = summaries["guarded_fast"]["median_ms"]
    primitive = summaries["primitive"]["median_ms"]
    peak_memory_bytes = int(mx.get_peak_memory())
    active_memory_bytes = int(mx.get_active_memory())
    result = {
        "batch": case.batch,
        "tokens": case.tokens,
        "warmups": warmups,
        "iterations": iterations,
        "parity_max_abs": parity_max_abs,
        "methods": summaries,
        "samples_ms": samples,
        "primitive_vs_public_pct": 100.0 * (primitive / public - 1.0),
        "primitive_vs_direct_pct": 100.0 * (primitive / direct - 1.0),
        "primitive_vs_guarded_pct": 100.0 * (primitive / guarded - 1.0),
        "guarded_vs_direct_pct": 100.0 * (guarded / direct - 1.0),
        "direct_vs_public_pct": 100.0 * (direct / public - 1.0),
        "peak_memory_bytes": peak_memory_bytes,
        "active_memory_bytes": active_memory_bytes,
    }
    del calls, inputs
    gc.collect()
    mx.clear_cache()
    result["post_clear_active_memory_bytes"] = int(mx.get_active_memory())
    return result


def environment() -> dict[str, Any]:
    compiler = subprocess.run(
        ["xcrun", "clang++", "--version"], capture_output=True, check=False, text=True
    )
    return {
        "python": platform.python_version(),
        "mlx": metadata.version("mlx"),
        "nanobind": metadata.version("nanobind"),
        "numpy": metadata.version("numpy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": mx.device_info(),
        "pid": os.getpid(),
        "compiler": compiler.stdout.splitlines()[0] if compiler.returncode == 0 else "unavailable",
    }


def aster_baseline_identity(aster_root: Path) -> dict[str, Any]:
    relative_source = Path("aster/inference/metal_paged_attention.py")
    source = aster_root / relative_source
    commit = subprocess.run(
        ["git", "-C", str(aster_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    source_status = subprocess.run(
        ["git", "-C", str(aster_root), "status", "--porcelain", "--", str(relative_source)],
        capture_output=True,
        check=False,
        text=True,
    )
    worktree_status = subprocess.run(
        ["git", "-C", str(aster_root), "status", "--porcelain"],
        capture_output=True,
        check=False,
        text=True,
    )
    if commit.returncode or source_status.returncode or worktree_status.returncode:
        raise RuntimeError("Unable to identify the Aster benchmark baseline")
    return {
        "commit": commit.stdout.strip(),
        "source": str(relative_source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_dirty": bool(source_status.stdout.strip()),
        "worktree_dirty": bool(worktree_status.stdout.strip()),
    }


def source_hashes() -> dict[str, str]:
    names = ("CMakeLists.txt", "aster_paged_ops.cpp", "bench.py", "aggregate.py")
    return {
        name: hashlib.sha256((PROTOTYPE_ROOT / name).read_bytes()).hexdigest() for name in names
    }


def swap_used_bytes() -> int | None:
    result = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, check=False, text=True
    )
    if result.returncode != 0:
        return None
    match = re.search(r"used = ([0-9.]+)([KMGTP])", result.stdout)
    if match is None:
        return None
    multipliers = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }
    return round(float(match.group(1)) * multipliers[match.group(2)])


def main() -> None:
    args = parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("Metal is required")
    if args.warmups < 0 or args.iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    aster = load_aster_kernel(args.aster_root.resolve())
    extension = build_extension()
    swap_before = swap_used_bytes()
    report: dict[str, Any] = {
        "environment": environment(),
        "aster_baseline": aster_baseline_identity(args.aster_root.resolve()),
        "source_hashes": source_hashes(),
        "run_id": args.run_id,
        "correctness": verify_correctness(aster, extension),
        "large_shape_correctness": (
            verify_large_shapes(aster, extension, args.batches, args.tokens)
            if args.verify_large
            else None
        ),
        "validation": verify_validation(extension),
        "results": [],
    }
    if not args.smoke_only:
        cases = [
            Case(batch=batch, tokens=tokens, query_offset=tokens - 1)
            for tokens in args.tokens
            for batch in args.batches
        ]
        random.Random(0xA57E + args.run_id).shuffle(cases)
        for case in cases:
            report["results"].append(
                benchmark_case(
                    aster,
                    extension,
                    case,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    run_id=args.run_id,
                )
            )
    swap_after = swap_used_bytes()
    report["swap"] = {
        "used_bytes_before": swap_before,
        "used_bytes_after": swap_after,
        "delta_bytes": (
            swap_after - swap_before
            if swap_before is not None and swap_after is not None
            else None
        ),
    }
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    if args.output is not None:
        compact_payload = json.dumps(
            report, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        args.output.write_text(compact_payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
