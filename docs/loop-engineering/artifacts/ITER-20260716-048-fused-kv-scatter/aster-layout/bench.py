from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sysconfig
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import nanobind

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT.parent
ASTER_ROOT = Path(os.environ["ASTER_ROOT"]).resolve()
REFERENCE_ROOT = Path(os.environ["VLLM_METAL_REFERENCE_ROOT"]).resolve()
METHODS = ("mlx_scatter", "fused_primitive")


def secure_build_dir() -> Path:
    configured = os.environ.get("ASTER_KV_SCATTER_BUILD_DIR")
    if configured is None:
        return Path(tempfile.mkdtemp(prefix="aster-iter048-build-"))
    candidate = Path(configured).expanduser()
    if candidate.is_symlink():
        raise RuntimeError("ASTER_KV_SCATTER_BUILD_DIR must not be a symlink")
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
    key_dim: int = 128
    value_dim: int = 128
    heads: int = 8
    block_size: int = 64
    blocks: int = 8


def load_extension() -> Any:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    matches = tuple(BUILD_DIR.rglob(f"_aster_kv_scatter*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one built extension, found {len(matches)}")
    spec = importlib.util.spec_from_file_location("_aster_kv_scatter", matches[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load extension {matches[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> tuple[dict[str, Any], str, dict[str, str]]:
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_source = Path(__file__).resolve().relative_to(ARTIFACT_ROOT).as_posix()
    if manifest["artifact_sources"].get(relative_source) != sha256(Path(__file__)):
        raise RuntimeError("Benchmark source does not match manifest")
    toolchain = manifest["toolchain"]
    if platform.python_version() != toolchain["python"]:
        raise RuntimeError("Python version does not match manifest")
    if mx.__version__ != toolchain["mlx"]:
        raise RuntimeError("MLX version does not match manifest")
    if nanobind.__version__ != toolchain["root_nanobind"]:
        raise RuntimeError("nanobind version does not match manifest")

    reference_hashes = manifest["reference"]["source_hashes"]
    sources = {
        "Aster/paged_kv_adapter.py": sha256(ASTER_ROOT / "aster/inference/paged_kv_adapter.py"),
        "vllm-metal/paged_ops.cpp": sha256(REFERENCE_ROOT / "metal/paged_ops.cpp"),
        "vllm-metal/reshape_and_cache.metal": sha256(
            REFERENCE_ROOT / "metal/kernels_v2/reshape_and_cache.metal"
        ),
        "CMakeLists.txt": sha256(ROOT / "CMakeLists.txt"),
        "kv_scatter.cpp": sha256(ROOT / "kv_scatter.cpp"),
        "bench.py": sha256(Path(__file__)),
    }
    expected = {
        "Aster/paged_kv_adapter.py": manifest["repository_sources"][
            "aster/inference/paged_kv_adapter.py"
        ],
        "vllm-metal/paged_ops.cpp": reference_hashes["paged_ops.cpp"],
        "vllm-metal/reshape_and_cache.metal": reference_hashes["reshape_and_cache.metal"],
        "CMakeLists.txt": manifest["artifact_sources"]["aster-layout/CMakeLists.txt"],
        "kv_scatter.cpp": manifest["artifact_sources"]["aster-layout/kv_scatter.cpp"],
        "bench.py": manifest["artifact_sources"][relative_source],
    }
    if sources != expected:
        raise RuntimeError("Benchmark provenance does not match manifest")
    return manifest, sha256(manifest_path), sources


def command_record(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def resource_snapshot() -> dict[str, Any]:
    swap = command_record(["sysctl", "vm.swapusage"])
    match = re.search(r"used = ([0-9.]+)([KMGTP])", swap["stdout"])
    if swap["returncode"] == 0 and match is not None:
        scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
        swap["used_bytes"] = round(float(match.group(1)) * scale[match.group(2)])
    thermal = command_record(["pmset", "-g", "therm"])
    thermal["warning"] = not (
        "No thermal warning level has been recorded" in thermal["stdout"]
        and "No performance warning level has been recorded" in thermal["stdout"]
    )
    return {"swap": swap, "thermal": thermal}


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values),
    }


def make_inputs(case: Case) -> tuple[Any, Any]:
    key = (mx.random.normal((case.batch, case.heads, case.tokens, case.key_dim)) * 0.125).astype(
        mx.float16
    )
    value = (
        mx.random.normal((case.batch, case.heads, case.tokens, case.value_dim)) * 0.125
    ).astype(mx.float16)
    mx.eval(key, value)
    return key, value


def make_caches(case: Case) -> tuple[Any, Any]:
    prefix = (case.blocks, case.batch, case.heads, case.block_size)
    key_cache = mx.zeros((*prefix, case.key_dim), dtype=mx.float16)
    value_cache = mx.zeros((*prefix, case.value_dim), dtype=mx.float16)
    mx.eval(key_cache, value_cache)
    return key_cache, value_cache


def reference_scatter(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    block_id: int,
    block_offset: int,
) -> tuple[Any, Any]:
    end = block_offset + int(key.shape[2])
    key_cache[block_id, ..., block_offset:end, :] = key
    value_cache[block_id, ..., block_offset:end, :] = value
    return key_cache, value_cache


def pool_error(
    reference_key: Any,
    reference_value: Any,
    candidate_key: Any,
    candidate_value: Any,
) -> float:
    mx.eval(reference_key, reference_value, candidate_key, candidate_value)
    return max(
        float(mx.max(mx.abs(reference_key - candidate_key)).item()),
        float(mx.max(mx.abs(reference_value - candidate_value)).item()),
    )


def verify(extension: Any) -> dict[str, Any]:
    cases = [
        Case(batch=1, tokens=1, heads=1, key_dim=64, value_dim=32),
        Case(batch=1, tokens=1),
        Case(batch=2, tokens=16),
        Case(batch=1, tokens=64),
    ]
    records = []
    max_error = 0.0
    for case in cases:
        key, value = make_inputs(case)
        reference_state = list(make_caches(case))
        fused_state = list(make_caches(case))
        original_fused = tuple(fused_state)
        writes = [
            (0, 0),
            (case.blocks - 1, case.block_size - case.tokens),
            (1, 0),
            (0, 0),
        ]
        output_error = 0.0
        original_alias_error = 0.0
        for block_id, block_offset in writes:
            reference_state[:] = reference_scatter(
                key,
                value,
                reference_state[0],
                reference_state[1],
                block_id,
                block_offset,
            )
            fused_state[:] = extension.fused_write(
                key,
                value,
                fused_state[0],
                fused_state[1],
                block_id,
                block_offset,
            )
            output_error = max(
                output_error,
                pool_error(*reference_state, *fused_state),
            )
            original_alias_error = max(
                original_alias_error,
                pool_error(*reference_state, *original_fused),
            )

        del fused_state
        gc.collect()
        after_output_release_error = pool_error(*reference_state, *original_fused)
        case_error = max(
            output_error,
            original_alias_error,
            after_output_release_error,
        )
        max_error = max(max_error, case_error)
        records.append(
            {
                "batch": case.batch,
                "tokens": case.tokens,
                "heads": case.heads,
                "key_dim": case.key_dim,
                "value_dim": case.value_dim,
                "writes": [
                    {"block_id": block_id, "block_offset": block_offset}
                    for block_id, block_offset in writes
                ],
                "output_max_abs": output_error,
                "original_alias_max_abs": original_alias_error,
                "after_output_release_max_abs": after_output_release_error,
                "max_abs": case_error,
            }
        )

    release_case = Case(batch=1, tokens=1)
    key, value = make_inputs(release_case)
    reference_state = list(make_caches(release_case))
    reference_state[:] = reference_scatter(key, value, *reference_state, 0, 0)
    input_key_cache, input_value_cache = make_caches(release_case)
    output_key, output_value = extension.fused_write(
        key, value, input_key_cache, input_value_cache, 0, 0
    )
    del input_key_cache, input_value_cache
    gc.collect()
    after_input_release_error = pool_error(*reference_state, output_key, output_value)
    max_error = max(max_error, after_input_release_error)

    lazy_case = Case(
        batch=1,
        tokens=1,
        heads=1,
        key_dim=8,
        value_dim=8,
        block_size=4,
        blocks=2,
    )
    lazy_key = mx.zeros((1, 1, 1, 8), dtype=mx.float16)
    lazy_value = mx.ones((1, 1, 1, 8), dtype=mx.float16)
    reference_state = [
        mx.zeros((2, 1, 1, 4, 8), dtype=mx.float16),
        mx.zeros((2, 1, 1, 4, 8), dtype=mx.float16),
    ]
    reference_state[:] = reference_scatter(lazy_key, lazy_value, *reference_state, 0, 0)
    reference_state[:] = reference_scatter(lazy_key, lazy_value, *reference_state, 1, 0)
    lazy_state = list(make_caches(lazy_case))
    lazy_state[:] = extension.fused_write(lazy_key, lazy_value, *lazy_state, 0, 0)
    lazy_state[:] = extension.fused_write(lazy_key, lazy_value, *lazy_state, 1, 0)
    lazy_chain_error = pool_error(*reference_state, *lazy_state)
    max_error = max(max_error, lazy_chain_error)
    if max_error != 0.0:
        raise AssertionError(f"Aster-layout fused scatter parity failed: {max_error}")

    invalid_case = Case(batch=1, tokens=1)
    key, value = make_inputs(invalid_case)
    key_cache, value_cache = make_caches(invalid_case)
    overlap_case = Case(batch=1, tokens=64, blocks=2)
    overlap_key_cache, overlap_value_cache = make_caches(overlap_case)
    overlap_shape = (
        overlap_case.batch,
        overlap_case.heads,
        overlap_case.tokens,
        overlap_case.key_dim,
    )
    overlap_key = mx.as_strided(overlap_key_cache, shape=overlap_shape)
    overlap_value = mx.as_strided(overlap_value_cache, shape=overlap_shape)
    mx.eval(overlap_key, overlap_value)

    class SpoofedArray:
        @property
        def __class__(self) -> type[Any]:
            return mx.array

    invalid = {}
    for name, args in {
        "block_id": (key, value, key_cache, value_cache, 8, 0),
        "block_offset": (key, value, key_cache, value_cache, 0, 64),
        "key_dtype": (key.astype(mx.float32), value, key_cache, value_cache, 0, 0),
        "key_rank": (key[0], value, key_cache, value_cache, 0, 0),
        "value_shape": (key, value[..., :64], key_cache, value_cache, 0, 0),
        "wrong_type": (object(), value, key_cache, value_cache, 0, 0),
        "spoofed_type": (SpoofedArray(), value, key_cache, value_cache, 0, 0),
        "overlap": (
            overlap_key,
            overlap_value,
            overlap_key_cache,
            overlap_value_cache,
            0,
            0,
        ),
        "cache_overlap": (key, value, key_cache, key_cache, 0, 0),
    }.items():
        try:
            extension.fused_write(*args)
        except (TypeError, ValueError, RuntimeError) as exc:
            invalid[name] = f"{type(exc).__name__}: {exc}"
        else:
            raise AssertionError(f"Invalid case {name} did not fail")
    return {
        "cases": records,
        "after_input_release_max_abs": after_input_release_error,
        "lazy_chain_max_abs": lazy_chain_error,
        "invalid": invalid,
        "max_abs": max_error,
    }


def benchmark_case(
    extension: Any,
    case: Case,
    *,
    warmups: int,
    iterations: int,
    generator: random.Random,
) -> dict[str, Any]:
    key, value = make_inputs(case)
    reference_state = list(make_caches(case))
    fused_state = list(make_caches(case))

    def call_reference(block_id: int) -> None:
        reference_state[:] = reference_scatter(
            key, value, reference_state[0], reference_state[1], block_id, 0
        )
        mx.eval(*reference_state)

    def call_fused(block_id: int) -> None:
        fused_state[:] = extension.fused_write(
            key, value, fused_state[0], fused_state[1], block_id, 0
        )
        mx.eval(*fused_state)

    calls: dict[str, Callable[[int], None]] = {
        "mlx_scatter": call_reference,
        "fused_primitive": call_fused,
    }
    for index in range(warmups):
        order = list(METHODS)
        generator.shuffle(order)
        for name in order:
            calls[name](index % case.blocks)

    samples: dict[str, list[float]] = {name: [] for name in METHODS}
    mx.reset_peak_memory()
    for index in range(iterations):
        order = list(METHODS)
        generator.shuffle(order)
        block_id = index % case.blocks
        for name in order:
            started = time.perf_counter_ns()
            calls[name](block_id)
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000)

    peak_memory_bytes = int(mx.get_peak_memory())
    post_benchmark_error = pool_error(*reference_state, *fused_state)
    baseline = statistics.median(samples["mlx_scatter"])
    candidate = statistics.median(samples["fused_primitive"])
    return {
        "batch": case.batch,
        "tokens": case.tokens,
        "key_dim": case.key_dim,
        "value_dim": case.value_dim,
        "warmups": warmups,
        "iterations": iterations,
        "methods": {name: summarize(values) for name, values in samples.items()},
        "samples_ms": samples,
        "fused_vs_mlx_pct": 100.0 * (candidate / baseline - 1.0),
        "post_benchmark_max_abs": post_benchmark_error,
        "peak_memory_bytes": peak_memory_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.iterations < 1:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    _, manifest_hash, sources = verify_manifest()
    resource_before = resource_snapshot()
    mx.random.seed(0xA57E048 + args.run_id)
    extension = load_extension()
    correctness = verify(extension)
    cases = [Case(batch=batch, tokens=tokens) for tokens in (1, 16, 64) for batch in (1, 2, 4, 8)]
    results = []
    if not args.smoke_only:
        generator = random.Random(0xA57E048 + args.run_id)
        generator.shuffle(cases)
        results = [
            benchmark_case(
                extension,
                case,
                warmups=args.warmups,
                iterations=args.iterations,
                generator=generator,
            )
            for case in cases
        ]
    resource_after = resource_snapshot()
    payload = {
        "run_id": args.run_id,
        "environment": {
            "device": mx.device_info(),
            "machine": platform.machine(),
            "mlx": mx.__version__,
            "nanobind": nanobind.__version__,
            "pid": os.getpid(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "manifest_sha256": manifest_hash,
        "correctness": correctness,
        "source_hashes": sources,
        "resources": {
            "before": resource_before,
            "after": resource_after,
            "power": {
                "measured": False,
                "reason": "powermetrics requires elevated privileges",
            },
        },
        "results": results,
    }
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
