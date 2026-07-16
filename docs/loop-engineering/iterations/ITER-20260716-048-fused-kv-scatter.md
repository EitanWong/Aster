# Iteration 048 - Fused K/V scatter transfer

Date: 2026-07-16
Start commit: `82a760d`
End commit: none (no runtime implementation retained)
Status: `ROLLED_BACK`

## Problem and hypothesis

Iteration 047 rejected a C++ Primitive around Aster's attention kernel, but
left vllm-metal's fused K/V scatter as a separate hypothesis. Aster's
`_PagedKVBlockPool.write()` updates key and value pools with two MLX slice
assignments. vllm-metal replaces two slot-mapping scatters with one Metal
dispatch whose two outputs alias the existing cache buffers.

The admission gate was deliberately two-stage:

1. reproduce the pinned reference and require a repeatable >=3% isolated win;
2. transfer the mechanism to Aster's actual layout and require the same gate
   before any real-model or runtime integration.

Any parity failure, private-ABI build instability, memory growth, shape
regression, or Aster-layout result below the gate rejected the candidate.

## CodeGraph and reference comparison

CodeGraph traced the current path as:

`PagedKVCacheLayer.update_and_fetch()` -> `_write_segment()` ->
`_PagedKVBlockPool.write()`.

The ownership boundary is already narrow. The layer splits writes at block
boundaries, obtains a writable physical block, copies a shared COW source when
needed, then appends one contiguous segment. `_PagedKVBlockPool.write()` only
owns the two physical-pool assignments and its written-token high-water mark.
No `engine.py` change is required to experiment at this point.

The cross-check used:

- Aster `aster/inference/paged_kv_adapter.py`, source SHA-256
  `de760744bbaa6bf764cd4a9c13ff2fa26ab154138a6b59fc4ebd034af41ff666`;
- vllm-metal commit `4c18ee0e6e3ce2b594ab114d0a53ca24eafb1d58`,
  `reshape_and_cache.metal`, `paged_ops.cpp`, and its upstream parity test;
- MLX 0.32.0 `backend/metal/custom_kernel.cpp` and Python fast-kernel API.

The MLX source audit found a hard boundary: every `mx.fast.metal_kernel`
output allocates its own buffer. It cannot reproduce vllm-metal's
`copy_shared_buffer()` aliasing without copying a complete cache. An `mx.fast`
full-pool output was therefore rejected before benchmarking.

## Three isolated reproductions

The archived artifact contains three independent probes:

1. `pure-mlx/`: one combined K/V storage scatter, including the real per-call
   `mx.stack()` cost, plus a pre-stacked ceiling;
2. `reference-vllm-metal/`: the unmodified pinned `reshape_and_cache`
   Primitive against its exact two-scatter MLX baseline;
3. `aster-layout/`: a standalone Primitive for Aster's
   `[capacity, B, H, block, D]` pools, with independent key/value dimensions.

No extension binary, virtual environment, build tree, or runtime integration
is retained. A source manifest, hash-locked dependencies, reproduction commands,
raw samples, and compact aggregate records live under
`docs/loop-engineering/artifacts/ITER-20260716-048-fused-kv-scatter/`.

## Environment

- Apple M5, 24 GB unified memory, macOS 27.0, `applegpu_g17g`.
- Python 3.13.12 and MLX 0.32.0.
- vllm-metal reference: nanobind 2.10.2, matching its pinned project metadata.
- Aster-layout prototype: nanobind 2.13.0, matching the working MLX array
  registry established in Iteration 047.
- Apple clang 21.0.0 for the C++ extensions.

The reference was first observed with nanobind 2.13.0, then rebuilt and fully
re-run with its exact 2.10.2 pin. The final archived reference records are the
2.10.2 runs.

## Correctness and boundary gates

The pure MLX probe compared the complete separate and combined pools for seven
FP16 batch/token/head/dimension combinations. Maximum error was zero.

The pinned reference reproduced its upstream test contract:

- FP16, BF16, and FP32;
- `(H,D)=(2,64)` and `(4,128)`;
- sparse, rotated physical slots;
- a negative padding slot that must be ignored.

All reference outputs were byte-identical to two MLX scatters. The upstream
kernel intentionally trusts the scheduler for positive slot upper bounds; the
Aster transfer did not inherit that weakness.

The Aster-layout RED step failed because no extension existed. Later RED probes
showed that arbitrary Python objects and spoofed `__class__` values caused a
native crash and that overlapping source/cache storage was accepted. The
corrected standalone implementation then passed complete-pool parity for:

- batch 1/2;
- 1/16/64-token writes at block start and end;
- `Dk=Dv=128` and `Dk=64,Dv=32`;
- physical block rotation across repeated writes;
- original cache handles after returned aliases were released, and returned
  aliases after input handles were released;
- two lazy Primitive writes chained without an intervening `mx.eval()`.

It rejected out-of-range block IDs, block-offset overflow, FP32 input, invalid
rank, incompatible value/cache shapes, non-MLX Python objects, spoofed
`__class__` values, overlapping source/cache views, and overlapping key/value
caches before dispatch. All normal, lifetime, and lazy-chain cases had zero
maximum absolute error. The standalone Primitive aliases both existing pools;
it does not allocate replacement cache buffers.

## Benchmark design

All authoritative matrices used five fresh processes, randomized cell and
method order, 30 warmups, 200 measurements, raw samples, and a 5,000-resample
paired moving-block/process bootstrap. Each process first produced its own
candidate/baseline ratio; point estimates and process resampling aggregate
those paired effects rather than dividing independent cross-process medians.
Intervals are exploratory stability estimates, not calibrated significance
tests. Positive deltas mean the candidate was slower.

The timed region includes lazy graph construction and `mx.eval()`, but excludes
input/cache allocation, first extension build, and first Metal compilation.

## Pure MLX result

The real combined-storage path never cleared the gate:

| Tokens | Batch | Median delta | Exploratory 95% interval |
| ---: | ---: | ---: | ---: |
| 1 | 1 | +0.03% | [-1.37%, +1.57%] |
| 1 | 2 | -0.40% | [-1.98%, +1.65%] |
| 16 | 1 | +1.77% | [+0.28%, +3.80%] |
| 16 | 2 | -0.35% | [-3.17%, +1.72%] |
| 64 | 1 | +1.32% | [-0.27%, +4.02%] |
| 64 | 2 | +2.95% | [+1.77%, +4.69%] |

No real combined-storage interval established a gain; 16-token batch 1 and
64-token batch 2 instead showed directional regressions. The pre-stacked
ceiling reached -2.73% at single-token batch 2, but its interval
`[-5.23%, -0.71%]` did not establish a >=3% gain. A combined pool would also
require an unequal-dimension fallback and wider layout ownership, so it was
rejected without a runtime patch.

## Pinned reference result

The exact vllm-metal Primitive did demonstrate a real mechanism-level win:

The two latency columns are descriptive medians of per-process medians. The
delta and interval use the paired per-process effects described above, so the
delta is not computed by dividing the two displayed rounded values.

| Tokens | MLX scatter | Fused Primitive | Delta | Exploratory 95% interval |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2419 ms | 0.2268 ms | -5.61% | [-7.42%, -3.91%] |
| 2 | 0.2395 ms | 0.2290 ms | -4.39% | [-5.87%, -1.92%] |
| 4 | 0.2490 ms | 0.2369 ms | -6.32% | [-9.52%, -4.11%] |
| 8 | 0.2455 ms | 0.2262 ms | -7.81% | [-9.82%, -3.67%] |
| 16 | 0.2420 ms | 0.2276 ms | -6.22% | [-7.90%, -3.04%] |
| 64 | 0.2501 ms | 0.2328 ms | -8.35% | [-11.60%, -3.98%] |
| 128 | 0.2575 ms | 0.2288 ms | -11.65% | [-15.40%, -10.15%] |

The 1/4/8/16/64/128-token intervals completely cleared the 3% gate. This
validates vllm-metal's token-contiguous slot-mapping design, not an Aster engine
claim.

## Aster-layout transfer result

The Aster-layout transfer did not reproduce a safe adoption region. Selected
cells show the main and independent confirmation groups:

| Tokens | Batch | Main delta / interval | Confirmation delta / interval |
| ---: | ---: | ---: | ---: |
| 1 | 1 | -3.49% / [-4.95%, -1.88%] | -2.29% / [-3.87%, +0.02%] |
| 1 | 4 | -4.04% / [-7.17%, -1.10%] | -1.93% / [-3.22%, -0.44%] |
| 1 | 8 | -4.84% / [-8.19%, -1.40%] | -1.38% / [-5.83%, -0.38%] |
| 16 | 2 | -4.05% / [-6.41%, -2.29%] | -1.24% / [-3.25%, +1.88%] |
| 64 | 1 | -2.51% / [-4.83%, -1.65%] | -0.85% / [-3.63%, +0.52%] |
| 64 | 4 | +4.24% / [+2.04%, +6.83%] | +7.10% / [+6.31%, +7.96%] |
| 64 | 8 | +5.04% / [+3.04%, +6.89%] | +8.22% / [+6.80%, +9.38%] |

No cell established a >=3% gain in both the main and independent confirmation
groups. Main single-token cells had nominal gains, but all intervals stopped
short of the gate and confirmation weakened them. Single-request 64-token
confirmation was only -0.85% and crossed zero. Conversely, 64-token batch 4/8
regressed in both groups; batch 8 intervals were wholly above 3% in both, while
batch 4 cleared that regression gate in confirmation. A shape router would add
a private ABI and branch complexity for sub-millisecond, unstable wins while
retaining a proven regression surface. It is not admitted.

## Stress and resources

The Aster-layout matrix then ran 1,000 measurements per method in every 1/16/64
token x batch 1/2/4/8 cell. It completed with exact parity, a maximum MLX peak
of `52,428,824 B`, and deltas from `-4.93%` to `+6.24%`. Every cell archived
zero complete-pool error after all timed writes. The record also captures
`vm.swapusage` at `0 B` before and after and the full `pmset -g therm` output,
which reported no thermal or performance warning. Power was unavailable because
`powermetrics` requires elevated privileges.

Cache hit rate, TTFT, token throughput, energy per token, and model output hash
are not applicable to this isolated cache-write probe. No end-to-end claim is
made.

## Repository verification

All eight archived Python sources passed Ruff and `py_compile`; the artifact's
six focused aggregation tests passed. The 28 result records plus source manifest
parsed as JSON objects. Every final process uses CPython 3.13.12, carries the
same manifest hash, and retains exact pre/post-benchmark parity. A fresh
out-of-tree CMake build of the archived Aster-layout extension completed and its
expanded smoke matrix passed. No binary, object, CMake cache, personal path, or
credential material is archived.

Negative evidence probes rejected four-process and duplicate-file groups,
duplicate cells, shortened samples, altered summaries, stale source hashes, and
a tampered reference export. Native RED/GREEN probes changed arbitrary-object
and spoofed-`__class__` handling from signal 11 to `TypeError`, preserved two
lazy back-to-back writes, and changed a shared `as_strided` cache view from
accepted to a pre-dispatch `ValueError`.

Both environments were installed from full distribution-hash lock files through
a fixed PyPI index with uv 0.11.15. Its official release checksum and GitHub
attestation were verified; lock audits reported no known vulnerabilities.

Independent Metal/C++, security, and code/evidence re-reviews confirmed that
their earlier medium findings were resolved and reported no remaining HIGH or
MEDIUM finding.

The repository suite reported `458 passed, 9 skipped, 1 failed, 1 warning`.
The single failure belongs to pre-existing out-of-scope user-worktree runtime
changes; excluding that test yielded `458 passed, 9 skipped, 1 deselected, 1
warning`. This iteration did not modify the affected runtime or test.

## Decision

Retain Aster's current two MLX pool assignments. Do not introduce combined K/V
storage, an Aster fused-scatter Primitive, nanobind/CMake packaging, or a
shape-specific dispatch router.

The iteration nevertheless preserves one useful conclusion: vllm-metal's
fused scatter is a genuine optimization for its scheduler-owned slot mapping
and token-contiguous layout. That result must not be copied across a different
layout and ownership model without a local transfer gate.

## Next priority

Stop optimizing scatter in isolation. Profile the complete paged
`update_and_fetch` and attention graph inside a real model to determine where
the previously observed kernel advantage is actually lost. The next candidate
must remove a measured end-to-end boundary rather than another sub-millisecond
operator. Uzu's command-buffer ownership and explicit GPU timing remain the
next native-runtime ceiling reference; DFlash stays deferred until cache and
batch-state parity are stable.

## Fixed loop output

LOOP ITERATION: 048
STATUS: ROLLED_BACK
START COMMIT: 82a760d
END COMMIT: none
FOCUS: vllm-metal fused K/V scatter transfer to Aster's physical pool layout
ROOT CAUSE: Reference slot-mapping fusion wins, but no Aster cell repeats a >=3% gain across both groups and large-chunk batches regress
CHANGES: Archived hash-bound pure MLX, pinned reference, and Aster-layout C++ reproductions; no runtime code retained
TESTS: Three pre/post parity matrices, nine native invalid/boundary families, lazy chaining, two Aster five-process groups, and 1,000-iteration stress
BENCHMARK: Reference 64/128 tokens -8.35%/-11.65%; Aster 64-token batch-1 confirmation -0.85% and batch-4/8 +7.10%/+8.22%
MEMORY_POWER: <=52,428,824 B probe peak, zero stress swap growth; power unavailable
REGRESSION: No runtime change; one unrelated user-worktree test remains failing
REFERENCE_PROJECTS: vllm-metal 4c18ee0, MLX 0.32.0 custom-kernel allocator, Aster paged pool
DECISION: Reject fused-scatter transfer and retain current MLX writes
NEXT PRIORITY: Profile the real paged update/attention graph against Uzu-style native command ownership
