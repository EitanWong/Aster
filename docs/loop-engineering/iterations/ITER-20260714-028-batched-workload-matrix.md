# Iteration 028: BatchedEngine Workload Matrix and Profile Guard

Date: 2026-07-14

## Scope

Starting commit: `1dd3fba`.

The previous prefix-cache implementation showed strong exact-reuse gains, but
an initial matrix exposed different greedy output hashes when heterogeneous
prompt lengths and cache states were mixed in one `BatchGenerator` batch. The
same discrepancy reproduced with prefix caching disabled, proving that the
problem was BatchGenerator's hybrid `ArraysCache + KVCache` batching boundary,
not only cache restore.

## Root cause and design

Qwen3.5's recurrent `ArraysCache` state does not have a proven offset contract
when requests with different prompt lengths or cache profiles are merged. A
prototype with `prefill_batch_size=1` fixed only part of the discrepancy;
decode merging still diverged. The engine now admits only requests with the
same prompt length, cache/no-cache mode, and cached-token offset into the
active BatchGenerator profile. Other requests remain queued until that profile
drains. This preserves exact token parity while retaining batching for
homogeneous reuse/long workloads.

The matrix also found two independent BatchedEngine correctness issues:

- structured schema arguments were passed to `build_json_logits_processor` in
  the wrong order;
- tokenizer EOS tokens were not in the effective stop set, and final/stream
  decoding included special tokens.

Both were fixed and covered by tests. Structured output now returns valid JSON
and stops cleanly.

## Changes

- `aster/inference/batched_engine.py`: profile admission, effective stop IDs,
  structured processor argument order, and special-token filtering.
- `scripts/dev/benchmark_batched_engine.py`: reproducible cache on/off matrix
  with p50/p95, output hashes, MLX/RSS/swap, structured validity, prefix stats,
  and cancellation probe.
- Tests cover profile admission, structured processor arguments, effective stop
  IDs, special-token decoding, and benchmark helpers.

## Commands and results

```text
python scripts/dev/benchmark_batched_engine.py \
  --config /tmp/aster-qwen08-native.yaml \
  --workloads reuse,mixed,reuse-divergent,staggered,long \
  --concurrency-levels 2,4,8 --rounds 2 --long-prompt-words 512 \
  --prefix-cache on --output .../prefix-on.json

python scripts/dev/benchmark_batched_engine.py ... --prefix-cache off

pytest -q
ruff check ...
python -m compileall -q aster scripts tests
uv lock --check
pip check
```

Full regression: `427 passed, 9 skipped, 1 warning`; Ruff, compileall, lock,
package compatibility, and diff checks passed.

The corrected 0.8B matrix contained 30 matched on/off records. Every record
had zero request errors, zero swap delta, and exact response-hash parity. On
the second warm round, cache-on elapsed improvement versus cache-off was:

| Workload | C=2 | C=4 | C=8 |
| --- | ---: | ---: | ---: |
| `reuse` | -9.4% | -10.2% | -11.0% |
| `mixed` | -18.1% | -16.1% | -16.0% |
| `reuse-divergent` | -7.1% | -9.6% | -9.4% |
| `staggered` | -10.1% | -12.9% | -9.1% |
| `long` | -24.9% | -31.9% | -33.5% |

The mixed cache-on peak was `1.517 GB` versus `1.472 GB` off (`+3.1%`),
while long C=8 was `1.546 GB` versus `2.582 GB` off. All swap deltas were
zero. The profile guard intentionally serializes some heterogeneous requests;
these numbers are not a claim of unrestricted continuous-batching parity.

Structured output at concurrency 2/4 produced valid JSON for every request,
with `stop` finish reasons and zero errors. The cancellation probe at
concurrency 8 completed 7 requests, cancelled 1, accepted a follow-up, and
left zero running requests and zero pinned prefix entries.

Raw records:

- `artifacts/ITER-20260714-028-batched-workload-matrix/prefix-on.json`
- `artifacts/ITER-20260714-028-batched-workload-matrix/prefix-off.json`
- `artifacts/ITER-20260714-028-batched-workload-matrix/structured.json`

## Decision

Keep `17f20ee` and the profile guard in the experimental `engine_type=batched`
path. Do not enable the separate `BatchGeneratorRuntimeKernel` or change the
manual production default. Rollback is `git revert 17f20ee`.

## Next priority

Recover mixed-workload parallelism without weakening parity: evaluate
per-profile queues or separate BatchGenerator lanes, then compare their
memory and scheduling overhead against the current conservative guard.
