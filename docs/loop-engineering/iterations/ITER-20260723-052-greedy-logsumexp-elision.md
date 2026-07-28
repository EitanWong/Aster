# Iteration 052: Profile Greedy Logsumexp Elision

- **Date:** 2026-07-23
- **Start reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`
- **End reference:** same working-tree reference; no production source change was admitted
- **Machine:** macOS 27.0, `Mac17,2`, 10 logical CPUs, 24 GiB unified memory
- **Runtime:** Python 3.14.5, MLX 0.32.0, MLX-LM 0.31.3
- **Model:** Qwen3.5-0.8B 4-bit MLX, fixed 409-token prompt, native cache

## Problem and Hypothesis

Iteration 051 leaves a per-row `logsumexp` before every sampler. MLX-LM's
`temperature == 0` sampler is an argmax, for which subtracting a constant does
not change the selected token. The hypothesis was that explicitly tagging that
sampler at benchmark setup and passing raw logits would reduce decode time.
The tag is an Aster-owned semantic marker, not an inspection of MLX-LM's
anonymous function identity.

Success required exact token/text/cache parity, zero swap growth, and at least a
3% decode improvement in the paired workload cells. Any production marker would
also have to retain the normalized-logprob contract for non-greedy samplers and
host-driven structured processors.

## Benchmark Design

`candidate_benchmark.py` reuses the Iteration 051 adjacent paired harness. Each
process has two cloned runners, alternates baseline/candidate call order, and
alternates which physical runner receives each policy by run ID. The baseline is
the current `ModelRunner._decode_batch`; the candidate only skips normalization
when the sampler was created for `temperature == 0`. All other rows use the
existing normalized path. Each run used 32 pair-warmup steps and 256 measured
steps with 16-step blocks.

The screen contains two fresh processes for each cell:

| Workload | Batch | Candidate speed change | Exact | Swap growth |
| --- | ---: | ---: | :---: | ---: |
| greedy | 2 | `+0.53%` median (`+0.46%` to `+0.59%`) | yes | 0 |
| greedy | 4 | `+0.58%` median (`+0.20%` to `+0.96%`) | yes | 0 |
| greedy | 8 | `+0.50%` median (`+0.24%` to `+0.77%`) | yes | 0 |
| penalties | 4 | `+0.60%` median (`+0.40%` to `+0.80%`) | yes | 0 |
| mixed | 4 | `+1.27%` median (`+1.22%` to `+1.32%`) | yes | 0 |

The ranges above are observed two-process screen ranges, not an admission
interval. The raw payloads and their source/model hashes are retained under
`docs/loop-engineering/artifacts/ITER-20260723-052-greedy-logsumexp-elision/`.

## Operator Profile

`operator_profile.py` evaluates one real model logits tensor first, then
alternates `argmax(logits - logsumexp(logits))` and `argmax(logits)` on the same
rows. The vocabulary is 248,320 and logits are BF16:

| Batch | Normalized argmax median | Raw argmax median | Graph delta | Delta / normalized |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 275.7 us | 253.4 us | 22.3 us | 8.1% |
| 2 | 282.4 us | 269.3 us | 13.1 us | 4.6% |
| 4 | 299.0 us | 268.6 us | 30.4 us | 10.2% |
| 8 | 317.3 us | 264.0 us | 53.3 us | 16.8% |

The isolated graph delta is real, but the measured decode steps were roughly
6.5 ms (B2), 7.9 ms (B4), and 12.4 ms (B8); the operator therefore exposes
only a sub-percent end-to-end opportunity. The mixed cell confirms that only
the temperature-zero rows are elided (`288` direct rows and `864` normalized
rows per candidate process).

## Correctness and Verification

- Candidate artifact tests: `4 passed`.
- Ruff: passed for all Iteration 052 scripts and tests.
- Every screen payload reported exact token IDs, text hashes, and cache digests.
- Every screen payload reported zero swap growth.
- The production runtime and its existing tests were not changed by this
  iteration; the prior Iteration 051 focused/full-suite results remain the
  applicable runtime regression evidence.

## Failed/Rejected Experiments

- A 16-step B2 smoke was slightly slower (`-0.87%` candidate throughput); it
  was treated as a smoke result, not a decision.
- The first profile invocation referenced `_settings` at the wrong module
  layer. The command failed before model execution, the call site was corrected,
  and the successful profile was rerun and hash-recorded.
- No production marker or `ModelRunner` change was retained because no cell
  approached the required 3% core improvement. The candidate remains a
  reproducible profiling harness rather than a serving optimization.

## Decision and Next Step

**Decision: reject for production, retain the evidence.** The raw-logit path is
mathematically valid for the tagged greedy sampler and exact in the tested
workloads, but its end-to-end effect is below the loop's performance gate. The
next investigation should isolate processor-specific graph costs and evaluate
homogeneous tensorized groups only where request order, RNG order, and
structured-parser ownership can be proven unchanged.

Rollback is deleting the Iteration 052 artifact directory; no serving rollback
is required because no production file was modified.

## Reproduction

```bash
ART=docs/loop-engineering/artifacts/ITER-20260723-052-greedy-logsumexp-elision
.venv/bin/python "$ART/operator_profile.py" --repeats 8 \
  --output "$ART/operator-profile.json"
.venv/bin/pytest -q "$ART/test_artifacts.py"
```

The paired screen commands are the `candidate_benchmark.py` invocations
recorded in the artifact directory; each output is self-describing and binds
the current model inputs and source hashes.
