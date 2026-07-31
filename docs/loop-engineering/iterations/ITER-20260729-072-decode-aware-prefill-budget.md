# Iteration 072: Decode-Aware Prefill Budget

- **Date:** 2026-07-29
- **Phase:** admitted
- **Scope:** a reversible scheduler default for continuing prefill while decode
  work is queued; the configured value is 512 tokens.

## Problem

I071 reproduced a long-prefill/interactive-decode interference case twice. A
short MT-Bench request released 50 ms after a 10,334-token QMSUM request began
waited about 1.2 s for admission to prefill, then spent 9.35-9.61 s in an
8-token decode while the long request continued in 1,024-token prefill chunks.
The existing decode-first scheduler selects decode before prefill between
chunks, but a full runner-thread prefill call still blocks the next decode step.

## Hypothesis

When a decode queue is nonempty, cap the next continuing prefill chunk at the
already-configured 512-token pressure budget. This should shorten the blocking
runner calls and improve the staggered short request's decode and end-to-end
latency, while leaving the default (no opt-in budget) behavior unchanged.

## Candidate and Control

- **Control:** `decode_active_prefill_token_budget = null`; existing 1,024-token
  prefill chunks are retained.
- **Candidate:** `decode_active_prefill_token_budget = 512`; only prefill steps
  run while a decode request remains queued are capped. Idle prefill, initial
  admission, prefix matching, cache ownership, cancellation, and decode order
  are not changed.

## Gates

1. Add focused tests that prove the opt-in budget applies only with queued
   decode work and never exceeds the ordinary/pressure memory budget.
2. Preserve exact token-ID/text hashes and finish reason for the long and short
   deterministic requests; cancellation cleanup remains zero-active-state.
3. Run at least three independently started source-bound staggered B2 pairs in
   balanced control/candidate order, with warmup and the same workload/lock.
4. Candidate needs at least a 3% median improvement in short-request
   end-to-end latency or decode duration, no material long-request latency
   regression, and no worse resource/cancellation outcome. Swap growth is a
   blocking observation until explained, not a number to optimize around.

## Rollback

The new setting defaults to `null`; omitting it retains I071 behavior. If any
parity, lifecycle, long latency, or resource gate fails, leave the setting
disabled and record the rejection.

## Implementation

- Added nullable `engine.decode_active_prefill_token_budget`; its default is
  `null` and `configs/config.yaml` does not enable it.
- `_prefill_budget()` now takes the minimum of the ordinary/pressure budget and
  this setting only while decode work is queued. The memory-pressure budget
  remains an upper bound.
- The public arrival/load harness accepts and records
  `--decode-active-prefill-budget`, keeping the control path explicit.

## Source-Bound Screen

Four independently started staggered QMSUM-plus-MT-Bench pairs used two
candidate-first and two control-first orders. All eight long and short requests
had exact token-ID hash, text hash, and `length` finish parity. Paired medians
for the 512-token opt-in candidate were:

- short decode duration: `-55.204%`;
- short end-to-end latency: `-48.276%`;
- long end-to-end latency: `-9.320%`;
- long prefill model time: `-7.986%`.

The candidate raised long prefill from 9 to 12 steps, as intended by the
smaller active-decode chunk bound. Its cancellation path was also clean:
accepted cancellation, one checkpoint, zero active/pending state after cleanup,
a completed deterministic follow-up, and zero swap growth.

## Resource Gate

The process-level swap data does not have a stable candidate direction. Control
samples include `+585,236,480` and `+311,296,000` bytes; candidate samples
include `+1,043,791,872`, `+273,743,872`, `0`, and `-25,165,824` bytes. The
large candidate sample exceeds every control sample in this screen. It cannot
be attributed safely to the policy or ignored as a benchmark artifact.

## Resource Attribution Plan

The harness now records `before_engine_create`, `after_engine_create`,
`after_engine_start`, `after_warmup`, `before_workload`, `after_workload`, and
`after_close` snapshots. These snapshots are outside the request timing path
and retain the original workload-only RSS/swap fields for compatibility.

The attribution screen uses the same locked staggered QMSUM/MT-Bench B2 case,
greedy 8-token completion, and 512-token candidate value in four fresh
processes:

1. prefix cache on: control then candidate;
2. prefix cache off: candidate then control.

It is a classification screen, not a new latency admission. A candidate-only
workload-stage swap increase that survives prefix-cache-off is evidence against
the policy. A swap increase in create/start/warmup/close, or one shared by the
paired control, remains lifecycle/host evidence and keeps the default disabled.
Only a predeclared, repeatable resource boundary can reopen default admission.

## Historical Decision

Do not enable the candidate by default. The setting remains a null-default,
reversible experiment because it has repeatable latency and correctness gains,
but its resource admission is blocked. The next measurement must separate
model/process lifecycle swap from scheduler-policy memory before revisiting a
default change.

## Resource Attribution Result

All four predeclared fresh-process runs retained the same long and short
token-ID hash, text hash, and length finish reason. The lifecycle snapshots
show no positive candidate-specific swap movement:

| Cache state | Variant | Create | Start | Warmup | Workload | Close | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 512 candidate | 0 | 0 | 0 | -8,388,608 | 0 | -8,388,608 |
| off | null control | 0 | 0 | 0 | 0 | 0 | 0 |
| on | 512 candidate | 0 | 0 | 0 | 0 | 0 | 0 |
| on | null control | 0 | 0 | 0 | 0 | 0 | 0 |

The cache-on runs each retain one 390,103,040-byte snapshot, while cache-off
runs retain none. This isolates that retained snapshot allocation from the
previous positive process-level swap observation. The original +1,043,791,872
candidate sample is not repeatable in either cache state or any measured
lifecycle stage. That clears the candidate-specific resource gate; it does not
claim a global host-memory root cause.

## Current-Source Cross-Engine Smoke

Aster and direct MLX-LM ran two locked MT-Bench records with the current 9B
model and tokenizer, greedy 8-token completion, and fresh isolated processes.
The workload, source lock, model/tokenizer fingerprint, generation contract,
and shared harness sources match. Both records have exact token IDs, text
hashes, and length finishes, and both engines report zero swap delta. This is a
current-source compatibility smoke only, not a heterogeneous timing ranking or
a replacement for the complete I066/I067 matrices.

## Superseding Admission

Enable engine.decode_active_prefill_token_budget at 512 in the tracked
configuration template and the local production configuration used for the
screen. The setting still applies only while decode work is queued and remains
reversible by setting it to null. The source-bound latency, parity,
cancellation, resource-attribution, and current-source compatibility evidence
is compactly bound in decode-aware-prefill-budget-admission.json.

## Verification

- Post-admission focused runtime, arrival/load, and config tests -> 69 passed.
- Current-source Aster/direct-MLX-LM smoke -> exact two-record output identity
  with zero swap in both isolated processes.

- `uv run pytest -q` -> `536 passed, 9 skipped, 1 warning`
- Candidate-focused tests -> `20 passed` across arrival, runtime, config, and
  CLI coverage.
- `uv run ruff check ...` and `git diff --check` -> passed.
