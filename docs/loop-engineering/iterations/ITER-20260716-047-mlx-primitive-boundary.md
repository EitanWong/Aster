# Iteration 047 - MLX Primitive boundary reproduction

Date: 2026-07-16
Start commit: `22865cd`
End commit: none (no runtime implementation retained)
Status: `ROLLED_BACK`

## Problem and hypothesis

Iteration 046 showed that Aster's existing block-indexed attention math is
already stronger than the reproduced vllm-metal kernel at long context, but
the end-to-end paged path still loses its kernel advantage. This iteration
isolated one proposed cause: Python-side `mx.fast.metal_kernel` graph/launch
metadata versus a native lazy MLX C++ `Primitive`.

The initial hypothesis was that packing scalar metadata into a C++ Primitive
and encoding directly through MLX's Metal `CommandEncoder` would improve the
same `Hq=16/Hkv=8/D=V=128` decode shape by at least 3% without changing
attention math. Failure conditions were any parity failure, non-finite output,
resource regression, private-ABI build instability, or a repeatable gain below
3%.

## Reference comparison

- Current Aster: `aster/inference/metal_paged_attention.py`, especially
  `_VECTOR_SOURCE`, `_get_vector_kernel`, and `paged_block_attention`.
- Apple MLX: `examples/mlx/examples/extensions/axpby/axpby.cpp`,
  `axpby.h`, and `CMakeLists.txt` for Primitive ownership, output allocation,
  `CommandEncoder`, nanobind domain, and build behavior.
- Cider: `examples/cider/csrc/src/w8a8_primitive.mm` and
  `csrc/src/prim_bindings.cpp` for cached pipelines, lazy arrays, scalar
  binding, and input registration.
- OMLX:
  `examples/omlx/omlx/custom_kernels/glm_moe_dsa/csrc/exact_block_attention.cpp`
  for packed attention parameters and threadgroup dispatch.
- vllm-metal: `examples/vllm-metal/vllm_metal/metal/paged_ops.cpp` for the
  original lazy paged-operation integration hypothesis.

The standalone prototype retained Aster's 32-simdgroup online-softmax math,
hard-coded only the tested FP16 decode `Q=1`, `D=V=128` specialization, and
used a packed 48-byte parameter structure. It never entered an Aster runtime
module. Source, setup instructions, raw samples, and aggregate results are
preserved under
`docs/loop-engineering/artifacts/ITER-20260716-047-mlx-primitive-boundary/`.
The authoritative source SHA-256 values are:

- `CMakeLists.txt`: `2a6c4447ef176e06fc289b9ca1e5c120eb901af996eac63b2bda9a798d1b0abb`
- `aster_paged_ops.cpp`: `d8320595389514726d16912ab42a08b2621bdc17f34df6cad297aa6119aee6a3`
- `bench.py`: `eff2e21e7d1c33c1f2571c90b50a103e93f1f2009938d8d3757093d583104490`
- `aggregate.py`: `38b1ca9b05e5b1bb571d0973ab678dd17d2203e6e41a1734dc2bd0fbbb6da1cc`

## Environment

- Apple M5, 24 GB unified memory, macOS 27.0, architecture
  `applegpu_g17g`.
- Python 3.13.12, MLX 0.32.0, NumPy 2.5.1.
- Apple clang 21.0.0 (`clang-2100.3.25.1`).
- nanobind 2.13.0, matching the current MLX build source.
- Start commit `22865cd`; no Aster dependency or virtual environment changed.

An initial nanobind 2.10.2 build linked successfully but could not convert an
`mlx.core.array` through the shared `NB_DOMAIN mlx` type registry. Rebuilding
the same minimal array echo with nanobind 2.13.0 fixed the conversion. This is
an additional packaging cost of the private C++ boundary, not an attention
optimization.

## Correctness and safety gates

The RED step was an executable harness that failed because the extension did
not exist. After the minimal implementation was built, the final harness
checked all four paths against an independent dense MLX reference:

1. current public `paged_block_attention`;
2. direct cached `mx.fast` invocation with pre-created scalar arrays;
3. a direct cached `mx.fast` control with the same physical-block guard and
   invalid-threadgroup reduction as the Primitive;
4. the C++ Primitive with packed parameters.

Eighteen combinations covered 32/33/63/64/65/129 tokens, beginning/middle/end
causal offsets, batch 1/2, and rotated physical block mappings. All cross-path
differences were zero. Maximum absolute difference against native MLX was
`6.103515625e-05`. A second random-data matrix covered every timed 2K/8K batch
1/2/4/8 shape; its cross-path error was zero and native error was at most
`7.62939453125e-06`. Every normal path was required to be finite before
comparison and JSON emission rejected NaN.

Invalid FP32 queries, unsupported head dimensions, invalid offsets, and table
capacity overflow failed before dispatch. Decode query lengths other than one
were explicitly rejected. The Metal guard checked physical block IDs before
pool access and returned NaN; the harness then rejected that output. A case
where only one simdgroup observed the bad ID propagated NaN to the complete
output, while an invalid block outside the causal range left the output
finite. This is a benchmark guard, not a production fail-closed contract.
Pipeline creation rejected devices without 32-wide SIMD or 1,024-thread
threadgroups.

## Benchmark design

Each authoritative process used:

- FP16 `Hq=16`, `Hkv=8`, `D=V=128`, block size 64, decode query length 1;
- 2,048 and 8,192 KV tokens; batch 1/2/4/8;
- 30 warmups and 200 measurements per method per cell;
- rotating method order inside each cell and randomized cell order per
  process;
- five fresh processes;
- median-of-process-medians, process-median p95, raw samples, and a 5,000
  resample outer-process plus paired moving-block bootstrap. The reported 95%
  intervals are exploratory stability estimates, not calibrated significance
  tests.

`public_fast` is the current user-visible wrapper. `direct_fast` removes its
per-call scalar-array construction. `guarded_fast` adds the Primitive's
physical-block guard and invalid-threadgroup reduction while retaining the
same `mx.fast` host boundary. `primitive` changes only the lazy node, parameter
ABI, and resource binding relative to that guarded control. Primitive versus
guarded is therefore the primary boundary comparison; public and direct are
secondary controls that expose wrapper and guard costs.

The artifact `README.md` contains complete setup, correctness, main A/B,
stress, and aggregation commands. The main command was repeated in five
independent processes with run IDs 61 through 65:

```bash
"$PYTHON" "$ARTIFACT/bench.py" \
  --aster-root "$REPO_ROOT" \
  --batches 1 2 4 8 --tokens 2048 8192 \
  --warmups 30 --iterations 200 --run-id 61 \
  --output "$ARTIFACT/results/main-run-1.json"
```

## A/B result

Positive deltas mean the Primitive was slower. The primary interval is
Primitive versus GPU-work-equivalent guarded `mx.fast`.

| KV | Batch | Public | Direct | Guarded | Primitive | vs guarded | vs public | Exploratory 95% vs guarded |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 1 | 0.3206 ms | 0.3145 ms | 0.3178 ms | 0.3070 ms | -3.41% | -4.26% | [-5.15%, -0.10%] |
| 2K | 2 | 0.4064 ms | 0.3990 ms | 0.4046 ms | 0.4002 ms | -1.09% | -1.53% | [-2.42%, -0.13%] |
| 2K | 4 | 0.5400 ms | 0.5310 ms | 0.5361 ms | 0.5301 ms | -1.10% | -1.82% | [-2.54%, -0.05%] |
| 2K | 8 | 0.8130 ms | 0.8224 ms | 0.8173 ms | 0.8185 ms | +0.15% | +0.68% | [-3.31%, +1.98%] |
| 8K | 1 | 0.5394 ms | 0.5387 ms | 0.5522 ms | 0.5456 ms | -1.20% | +1.14% | [-2.56%, -0.28%] |
| 8K | 2 | 0.9669 ms | 0.9145 ms | 0.9765 ms | 0.9310 ms | -4.67% | -3.72% | [-9.46%, -0.93%] |
| 8K | 4 | 1.6582 ms | 1.6398 ms | 1.6492 ms | 1.6535 ms | +0.26% | -0.28% | [-6.00%, +5.32%] |
| 8K | 8 | 3.0227 ms | 3.0077 ms | 3.0554 ms | 3.0597 ms | +0.14% | +1.23% | [-3.17%, +4.44%] |

Five cells excluded zero in favor of the Primitive, but none established a
gain of at least 3%: the closest-to-zero interval bounds ranged from `-0.05%`
to `-0.93%`. The two nominal `>=3%` medians were 2K/batch-1 (`3.41%`) and
8K/batch-2 (`4.67%`). Guarded versus direct medians varied from `-0.63%` to
`+6.78%`; the largest guard cost occurred at 8K/batch-2, where Primitive versus
direct was actually `1.80%` slower and its interval crossed zero. This confirms
that the unguarded path cannot establish a native-boundary win.

P95 was also mixed. Public/guarded/Primitive process-median P95 was
`0.7318/0.6720/0.5658 ms` at 2K/batch-1,
`0.9985/0.9693/1.0322 ms` at 2K/batch-2, and
`3.8901/3.8848/3.7921 ms` at 8K/batch-8. Across main cells, per-process
standard deviation ranged `0.1793~0.5143 ms` for public,
`0.1696~0.4557 ms` for guarded, and `0.1454~0.8086 ms` for Primitive. Global
main-sample min/max was `0.1785/5.4701 ms`, `0.1739/4.8940 ms`, and
`0.1720/8.3882 ms`, respectively. Raw samples remain archived rather than
being promoted as an engine claim.

## Candidate confirmation

The two nominal `>=3%` main cells triggered a confirmation instead of
promotion. Five new processes re-ran the containing 2K/8K x batch-1/2 matrix
with the same 30 warmups and 200 measurements:

- 2K/batch-1 fell to a `2.13%` median gain, with exploratory interval
  `[-9.36%, -0.02%]`;
- 8K/batch-2 reversed to a `3.51%` median regression, with interval
  `[-3.42%, +8.07%]`;
- 2K/batch-2 and 8K/batch-1 remained below the 3% gate.

The main matrix therefore did not reproduce a stable `>=3%` boundary gain.
The confirmation raw samples and aggregate are archived alongside the primary
matrix with run IDs 91 through 95.

## Long-context stress and resources

The same four paths then ran five independent 32K processes (10 warmups, 50
measurements) and five independent 64K processes (10 warmups, 30
measurements). All outputs retained zero cross-path difference. Positive delta
means Primitive was slower; every Primitive-versus-guarded interval crossed
zero.

| KV | Batch | Public | Direct | Guarded | Primitive | vs guarded | vs public | Exploratory 95% vs guarded | Peak MLX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 1 | 1.7156 ms | 1.7081 ms | 1.7109 ms | 1.7808 ms | +4.09% | +3.80% | [-3.43%, +9.25%] | 134.3 MB |
| 32K | 2 | 3.2494 ms | 3.1784 ms | 3.2138 ms | 3.1876 ms | -0.81% | -1.90% | [-4.21%, +3.60%] | 268.5 MB |
| 64K | 1 | 3.0394 ms | 3.1852 ms | 3.2375 ms | 3.1252 ms | -3.47% | +2.82% | [-5.16%, +1.22%] | 268.5 MB |

The prior unguarded 64K comparison was confounded by the physical-block guard.
Primitive versus guarded remained inconclusive in every stress cell. Median
public/guarded/Primitive P95 was `1.9248/2.0126/2.3124 ms`,
`3.9101/4.0260/4.0560 ms`, and `4.3332/3.9219/4.5516 ms` for the three rows.
All 22 archived correctness, main, confirmation, and stress process records
reported zero swap growth. Post-clear active MLX memory was `16 B` in every
measured cell; the maximum probe peak was `268,813,526 B`. Peak values
describe shared benchmark inputs plus all compared paths, not a per-method
memory delta.
`pmset -g therm` reported no recorded thermal or performance warning. Power
was unavailable because `powermetrics` requires elevated privileges.

Cache hit rate, token throughput, TTFT, CPU/GPU utilization counters, and
energy per token are not applicable or unavailable for this isolated
single-kernel graph/launch probe; no end-to-end engine claim is made.

## Repository verification

Commands and results:

```bash
.venv/bin/ruff check \
  docs/loop-engineering/artifacts/ITER-20260716-047-mlx-primitive-boundary/bench.py \
  docs/loop-engineering/artifacts/ITER-20260716-047-mlx-primitive-boundary/aggregate.py
# All checks passed

"$PYTHON" -m py_compile \
  docs/loop-engineering/artifacts/ITER-20260716-047-mlx-primitive-boundary/bench.py \
  docs/loop-engineering/artifacts/ITER-20260716-047-mlx-primitive-boundary/aggregate.py
# passed

.venv/bin/pytest -q
# 458 passed, 9 skipped, 1 failed, 1 warning

git diff --check
# passed

uvx pip-audit --disable-pip --no-deps \
  -r "$ARTIFACT/requirements.txt"
# No known vulnerabilities found (direct pinned artifact dependencies)
```

The archived smoke command rebuilt the standalone CMake/C++ extension from
source and passed. The single full-suite failure belongs to pre-existing,
out-of-scope worktree changes; a scoped diagnostic excluding that test yielded
`458 passed, 9 skipped, 1 deselected, 1 warning`. This iteration did not modify
that runtime or its tests. The project lock audit also reported no known
vulnerabilities. Negative aggregation probes rejected fewer than five
processes and duplicate input files. No user-owned runtime or test file was
changed.

## Root cause and decision

The native Primitive boundary is not a demonstrated bottleneck for this
attention kernel. Packing metadata can shift sub-millisecond host graph
construction and evaluation latency, and it recovered part of the explicit
guard cost in some cells. However, the main matrix did not establish a >=3%
interval against GPU-work-equivalent guarded `mx.fast`. More importantly, its
two nominal >=3% medians failed the independent confirmation: one fell to
2.13% and the other reversed to a 3.51% regression. Every stress interval
crossed zero. The private C++ ABI therefore has no demonstrated compensating
performance case.

Do not add a native attention Primitive, nanobind/CMake packaging, or a private
MLX C++ ABI to Aster. No runtime change exists to roll back. Keep the current
`mx.fast` attention path and its existing fallback behavior.

## Next priority

Separate the remaining vllm-metal hypothesis: measure Aster's current K/V pool
write path, then prototype fused K/V scatter alone against the same physical
layout. Start with an `mx.fast` Metal boundary to avoid private C++ ABI cost.
A repeatable >=3% isolated write-path win only qualifies the candidate for a
real-model matrix. Final adoption still requires >=3% end-to-end improvement
or a material memory reduction, plus exact block/COW parity, batch and boundary
corners, bounded memory, and zero swap growth.

## Fixed loop output

LOOP ITERATION: 047
STATUS: ROLLED_BACK
START COMMIT: 22865cd
END COMMIT: none
FOCUS: Lazy MLX C++ Primitive boundary for Aster block-indexed attention
ROOT CAUSE: Guard work explains part of the control delta; nominal boundary gains fail the >=3% confirmation gate
CHANGES: Archived standalone Primitive, GPU-work-equivalent guarded control, main/confirmation/stress benchmarks, hardened aggregator, and raw results; no Aster runtime code retained
TESTS: 18 boundary corners, all timed 2K/8K shapes versus native, seven invalid-input/guard outcomes, five-process confirmation, 32K/64K stress, build/lint, and repository suite
BENCHMARK: Main nominal >=3% cells re-tested as a 2.13% gain and a 3.51% regression; every stress guarded interval crossed zero
MEMORY_POWER: <=268,813,526 B probe peak, 16 B post-clear active, zero swap across 22 process records; power unavailable
REGRESSION: No runtime change; one unrelated user-worktree test remains failing
REFERENCE_PROJECTS: MLX extensions, Cider, OMLX, vllm-metal, and Aster current Metal kernel
DECISION: Reject the native attention Primitive and private ABI cost
NEXT PRIORITY: Isolate and reproduce fused K/V scatter without bundling an attention Primitive
