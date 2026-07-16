# Iteration 049 - Real-model paged graph and TurboQuant reproduction

Date: 2026-07-17
Start commit: `6265962`
End commit: none (no runtime implementation retained)
Status: `ROLLED_BACK`

## Problem and hypothesis

Iteration 048 showed that isolated fused K/V writes do not transfer to Aster's
physical-pool layout. The next required boundary was the complete paged
`update_and_fetch` plus direct-attention graph in a real model. In parallel,
the 2026 Open-TQ-Metal and BaseRT results made compressed-domain attention and
native command ownership the highest-value frontier mechanisms to falsify.

The hypotheses were:

1. the opt-in direct paged graph might recover its synthetic kernel advantage
   in Qwen3.5-0.8B after pool packing had already been removed;
2. if it did not, method-level instrumentation would identify the next
   material boundary instead of selecting another isolated operator;
3. TurboQuant's fused compressed-domain decode could beat both Aster paged
   FP16 and the default MLX FP16 path while materially reducing cache storage;
4. any lossy candidate also had to preserve deterministic greedy output,
   teacher-forced top-1, and perplexity before runtime admission.

The benefit gate was either a repeatable 3% speed improvement whose
process-bootstrap lower bound also cleared 3%, or at least 1.5x complete-cache
reduction. A capacity-only candidate still required the no-regression gate.
Quality required exact greedy tokens, at least 99% teacher top-1 agreement, at
most 0.5% absolute PPL change, no decode regression above 1%, finite outputs,
zero swap growth, and a verified fallback. Failure of any applicable gate
rejected runtime integration.

## CodeGraph and current graph

CodeGraph was used before source reads and traced:

`Qwen3.5 attention` -> `PagedKVCacheLayer.update_and_fetch()` ->
`_PagedKVBlockPool.write()` -> `PagedAttentionView.attention()` ->
`paged_block_attention()`.

The bridge only dispatches direct block attention for causal, sink-free,
single-request `Q<=8` calls. Long prefill remains native MLX SDPA. Decode calls
the model, synchronizes the sampled token through `_sample_token().item()`, and
then evaluates retained cache state in `_eval_cache()`.

Across five instrumented 2K native profiles, the median of per-process decode
medians was `8.082 ms`. `sample_token_sync` consumed `6.974 ms`, model graph
construction only `0.671 ms`, and cache evaluation was `0.364 ms`. The
cache-eval aggregate also contains three prefill completions,
including a roughly 200 ms maximum, so its total must not be added to the
nested decode total. The material default-path opportunity is synchronization
and graph ownership, not Python attention wrapper time.

## Reference and frontier intake

The source comparison used fixed versions and primary artifacts:

- Aster paged attention at source SHA-256
  `b7b4bea2ead78057d4d4759d99fc1de62f674a6ba3dd603b0d7233bc8bbd8796`;
- OMLX commit `e3a4fe4691b76b56935963d563909cba1eab486f`, including its
  TurboQuant dispatcher and hybrid-cache conversion policy;
- vllm-metal commit `4c18ee0e6e3ce2b594ab114d0a53ca24eafb1d58` and
  `kernels_v2/turboquant.metal`;
- Uzu commit `15b8e73c83aca4c305e297310600c06add786080`, especially explicit
  command-buffer state, commit/wait, and GPU timing;
- local `gemma4metal` commit
  `0f09466b7fde772a4876bf7bee3ccdeb34313304`, MIT, retained as an ignored
  read-only reference;
- [Open-TQ-Metal](https://arxiv.org/abs/2604.16957),
  [BaseRT](https://arxiv.org/abs/2607.00501), and
  [DFlash](https://arxiv.org/abs/2602.06036).

Open-TQ-Metal's public C++/Metal source was configured against local MLX
`4367c73`, but the native build stopped at 1% because the host lacks Apple's
separately downloadable Metal Toolchain component. No system component was
installed. The same mechanism was therefore reproduced through the public
`mx.fast.metal_kernel` implementation pinned by OMLX and mlx-vlm.

The Open-TQ repository's current test is only `N=10`, MHA, `D=256`, and accepts
maximum error below `0.1`, despite a README claim below `1e-5`. It was treated
as a research artifact, not a sufficient quality gate. OMLX's broader suite
passed `51/51` and covered MSE codecs, cache round trips, batching, sinks, long
prefill, mixed bit widths, multi-row verification, and `D=512`.

The DFlash references were also cross-checked. A minimum valid reproduction
needs a target-compatible DFlash checkpoint, block verification, KV injection,
acceptance telemetry, and rollback parity. No Qwen3.5-9B DFlash checkpoint was
available locally, and the 0.8B target is not a compatible drafter, so DFlash
remains deferred rather than being approximated with an invalid model pair.

Agent Reach `v1.5.0` is installed at `~/.local/bin/agent-reach`; its GitHub,
web, RSS, Exa, V2EX, and basic Bilibili checks pass. Exa is registered in the
home mcporter config for subsequent frontier intake.

## Real-model direct-paged baseline

Qwen3.5-0.8B-4bit ran greedy, prefix-off, batch-1 requests with 64 completion
tokens and eight warmup tokens. Ten fresh control processes per variant were
formed from the original five runs plus a separate five-run confirmation.
Five additional profile processes per variant captured method timings. Every
native/direct pair retained token IDs and text SHA, and every swap delta was
zero.

| Prompt tokens | Metric | Native | Direct paged | Paired direct delta |
| ---: | --- | ---: | ---: | ---: |
| 2,229 | elapsed median | 1.1005 s | 1.1044 s | +0.37%, 95% `[-1.24%, +1.48%]` |
| 2,229 | generation | 118.83 tok/s | 118.64 tok/s | -0.58%, 95% `[-1.80%, +1.79%]` |
| 2,229 | maximum peak | 768.21 MB | 759.09 MB | -9.12 MB |
| 8,373 | elapsed median | 2.5783 s | 2.5838 s | +0.20%, 95% `[-0.30%, +0.50%]` |
| 8,373 | generation | 109.25 tok/s | 111.00 tok/s | +1.94%, 95% `[-0.83%, +4.02%]` |
| 8,373 | maximum peak | 960.12 MB | 954.02 MB | -6.09 MB |

The separately seeded confirmation leaves both contexts in the noise region;
neither latency nor throughput establishes a 3% gain. Direct paged attention
remains opt-in.

Host-side direct decode work was small. Across the five profiled direct runs,
total paged attention enqueue was about `4.7-4.8 ms`, K/V pool writes about
`1.3 ms`, and cache-update wrapper time about `3.9-4.0 ms` over all 64
decode tokens. These cannot explain or repair a multi-percent end-to-end
regression in isolation.

## Compressed-domain kernel reproduction

An isolated Python 3.13.12 environment pinned MLX 0.32.0, mlx-lm 0.31.3, and
mlx-vlm 0.6.3 at commit `78b96eb`. Five fresh processes measured four methods
at the Qwen3.5 shape `B=1,Hq=8,Hkv=2,Q=1,D=256`: default MLX FP16, Aster paged
FP16, TurboQuant fused 4-bit, and TurboQuant dequantize-then-SDPA. Each context
used 30 warmups and 200 Latin-square-interleaved calls per method, for 16,000
timed calls. Raw samples, p95, source hashes, memory, thermal state, and swap
are archived.

Positive comparison values mean lower candidate latency.

| KV tokens | MLX FP16 | Aster paged | TQ fused | vs Aster paged | vs MLX default | Cache reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.288 ms | 0.358 ms | 0.291 ms | +18.07% `[15.70,19.41]` | -3.47% `[-3.59,-1.10]` | 3.94x |
| 8,192 | 0.395 ms | 0.698 ms | 0.384 ms | +44.72% `[38.92,45.56]` | +1.37% `[-1.64,5.33]` | 3.94x |
| 32,768 | 0.886 ms | 2.533 ms | 1.164 ms | +54.03% `[36.72,67.88]` | -34.13% `[-60.78,8.05]` | 3.94x |
| 65,536 | 1.627 ms | 5.021 ms | 2.020 ms | +59.78% `[57.65,63.16]` | -25.08% `[-29.97,-15.50]` | 3.94x |

All outputs were finite. Aster versus MLX maximum absolute error stayed at or
below `6.10e-5`; TurboQuant fused versus its dequantized reference stayed at or
below `1.53e-5`. K/V dequantized cosine similarity remained above `0.9952`,
but MSE was about `0.0094`. All swap deltas were zero. TurboQuant therefore
beats Aster's experimental single-channel paged kernel, but it does not beat
the production MLX default across the context curve.

## Full-model quality and performance

The decisive matrix loaded the local Qwen3.5-0.8B-4bit model and WikiText-2
test corpus SHA-256
`d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0`.
Prefill stayed FP16, then only the six full-attention `KVCache` layers were
converted, matching OMLX's hybrid `ArraysCache + KVCache` policy. Two contexts,
two variants, and five paired runs used 20 fresh processes. Every cell first
performed a same-length warmup, then 64 greedy and 64 teacher-forced tokens.
Runs used five distinct corpus offsets while each FP16/TurboQuant pair shared
the same window. Every loaded safetensors, tokenizer, config, and chat-template
input is hash-bound in each record.

| Context | Greedy parity | Teacher top-1 | PPL delta | Decode change | End-to-end change | Whole-cache reduction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 3/5 exact; min prefix 4 | 89.06% | +0.20% median; 7.49% max abs | 5.22% slower `[3.00,6.71]` | 4.76% slower `[2.40,7.04]` | 1.72x |
| 8,192 | 3/5 exact; min prefix 11 | 93.75% | +1.28% median; 3.38% max abs | 5.72% slower `[5.04,9.56]` | 1.76% slower `[0.90,3.74]` | 2.67x |

TTFT also regressed by `0.84%` and `0.47%`. Median generation changed from
`133.01 -> 122.61 tok/s` at 2K and `119.18 -> 112.63 tok/s` at 8K. Model
weights dominate allocator peak, so total peak changed by only about +0.03%
despite the retained-cache reduction; all 20 swap deltas were zero. Median RSS
was effectively flat, but each context's process-bootstrap lower bound allowed
about a `2.69%` regression, so the strict 1% RSS no-regression gate also fails.

The candidate fails deterministic, teacher top-1, PPL, and decode regression
gates. The memory benefit is real but is not admitted at the cost of these
regressions. The Aster-runtime fallback gate remains explicitly false because
the candidate was falsified before runtime integration.

## Reproduction commands

Use new empty result directories because each runner rejects occupied cells.
The isolated environment used Python 3.13.12, MLX 0.32.0, mlx-lm 0.31.3 at
`ab1806e`, mlx-vlm 0.6.3 at `78b96eb`, pytest 9.0.2, requests 2.32.5, and
Pillow 12.1.1.

```bash
ART=docs/loop-engineering/artifacts/ITER-20260717-049-real-model-paged-graph

.venv/bin/python "$ART/run_matrix.py" \
  --python .venv/bin/python \
  --config configs/config.yaml \
  --model models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --results /tmp/iter049-real-main \
  --modes control profile --contexts 2048 8192 \
  --runs 1 2 3 4 5 --max-tokens 64 --warmup-tokens 8 --seed 49017

.venv/bin/python "$ART/run_matrix.py" \
  --python .venv/bin/python \
  --config configs/config.yaml \
  --model models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --results /tmp/iter049-real-confirmation \
  --modes control --contexts 2048 8192 \
  --runs 11 12 13 14 15 --max-tokens 64 --warmup-tokens 8 --seed 49018

/tmp/aster-iter049-tq-env/bin/python "$ART/turboquant_run_matrix.py" \
  --python /tmp/aster-iter049-tq-env/bin/python \
  --results /tmp/iter049-turboquant \
  --runs 5 --tokens 2048 8192 32768 65536 \
  --iterations 200 --warmups 30 --seed 49117

curl -fL \
  https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt \
  -o /tmp/aster-iter049-wikitext2-test.txt

/tmp/aster-iter049-tq-env/bin/python "$ART/turboquant_model_run_matrix.py" \
  --python /tmp/aster-iter049-tq-env/bin/python \
  --model models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --corpus /tmp/aster-iter049-wikitext2-test.txt \
  --results /tmp/iter049-turboquant-model \
  --contexts 2048 8192 --runs 5 \
  --teacher-tokens 64 --generation-tokens 64 \
  --prefill-step 1024 --offset 1024 --offset-stride 16384 \
  --bits 4.0 --seed 49217

PYTHONPATH=examples/omlx /tmp/aster-iter049-tq-env/bin/python -m pytest -q \
  examples/omlx/tests/test_turboquant.py
```

## Artifacts and verification

The artifact directory contains the real-model profiler, strict aggregators,
fresh-process runners, raw records, execution manifests, source hashes, the
OMLX JUnit report, and regression tests. Negative evidence checks reject
duplicate PID reuse, duplicate cells, shortened or altered sample summaries,
source drift, incomplete context matrices, and mismatched model/dataset
provenance. The two real-paged execution manifests bind all 60 outputs to the
actual dirty runtime source, YAML configuration, safetensors, tokenizer, and
chat-template hashes; strict tests recompute all three aggregates from raw
records and compare them with the archived output.

Artifact tests pass `23/23`; Ruff and `py_compile` pass for every Iteration 049
Python file. OMLX's pinned TurboQuant suite passes `51/51`. The broader Aster
suite reports `458 passed, 9 skipped, 1 failed, 1 warning`; the single failure
is the pre-existing user-worktree expectation for
`InferenceEngine._snapshot_budget_for_state`. Excluding that unrelated test
yields `458 passed, 9 skipped, 1 deselected, 1 warning`. This iteration does
not modify the affected runtime or test.

## Decision

Retain no TurboQuant or direct-paged runtime change. Keep native MLX attention
as the production path and direct paged attention as opt-in experimental code.
Preserve the benchmark and all negative evidence so a later 6/8-bit,
model-specific, or capacity-only proposal starts from the measured quality
curve rather than the isolated 4-bit kernel claim.

Open-TQ-Metal, BaseRT, DFlash, Uzu, vllm-metal, OMLX, and gemma4metal remain
reference inputs. None is promoted from paper or microkernel results alone.

## Next priority

Measure whether the post-sample `_eval_cache()` traversal is redundant after
the sampled token has already forced the model graph. Its decode median is
about `0.364 ms`, enough to matter if graph provenance proves all cache states
materialized. Build a 10,000-step RAW/WAW cache-state stress test first, cover
hybrid recurrent/full-attention state, COW/trim/cancellation, and compare exact
tokens and cache arrays before changing the call. Uzu's explicit command
ownership and vllm-metal's lazy Primitive provenance are the reference models.

If that gate fails, test a Qwen3.5-specific two-pass 512-token partition only
as a paged-path experiment. Do not reuse the previously rejected general
vllm-metal occupancy threshold and do not promote a paged-only win over a
faster MLX default.

## Fixed loop output

LOOP ITERATION: 049
STATUS: ROLLED_BACK
START COMMIT: 6265962
END COMMIT: none
FOCUS: Real-model paged graph profiling and frontier compressed-domain attention reproduction
ROOT CAUSE: Direct paged host wrappers are small; sample synchronization dominates decode, and 4-bit TurboQuant loses default-path speed and model quality despite cache reduction
CHANGES: Added hash-bound profilers, kernel/model matrices, strict evidence validation, and frontier records; no runtime code retained
TESTS: Artifact 23/23, OMLX TurboQuant 51/51, 60 manifest-bound real-paged records, 5 kernel processes, 20 model processes
BENCHMARK: Direct +0.37%/+0.20% elapsed at 2K/8K; TQ model decode 5.22%/5.72% slower and only 3/5 greedy windows matched at each context
MEMORY_POWER: Direct saved 9.12/6.09 MB peak; TQ cache reduced 1.72x/2.67x model-wide, RSS gate inconclusive, zero swap; power unavailable
REGRESSION: No runtime change; 4-bit TQ rejected by deterministic, top-1, PPL, and speed gates
REFERENCE_PROJECTS: OMLX e3a4fe4, vllm-metal 4c18ee0, Uzu 15b8e73, gemma4metal 0f09466, Open-TQ-Metal, BaseRT, DFlash
DECISION: Keep native MLX default; preserve TurboQuant as rejected 4-bit capacity research evidence
NEXT PRIORITY: Prove or falsify redundant post-sample cache evaluation with hybrid-state RAW/WAW stress
