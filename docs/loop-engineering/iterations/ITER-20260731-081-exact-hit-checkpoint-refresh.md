# Iteration 081: Exact-Hit Checkpoint Refresh

- Date: 2026-07-31
- Phase: admitted-for-8gib-validation
- Baseline commit: `d69557e1b1801cf47619b2bbe2978d36e356e661` plus the
  SHA-bound I077-I080 checkpoint recorded in Git history
- Scope: exact-hit snapshot lifecycle only; no snapshot-budget default,
  general eviction policy, persistence, or representation change.

## Problem

I080 rejected the temporary 4 GiB budget because three of four disjoint
six-key windows evicted another snapshot during the final exact replay. The
replay lookup itself succeeded with zero prefill steps and exact terminal
output in every row. The eviction happened later, while decode activation
reserved and stored the same full logical prefix again.

Current source explains the candidate boundary:

1. `PrefixStore.lookup` records an exact hit, touches the retained entry, and
   the engine pins that key before cloning it into request state.
2. A new request starts with an empty `checkpoints_created` set. Exact hits do
   not add the matched logical prefix to that set.
3. `_should_store_full_prompt_checkpoint` skips only a strict prefix hit, not
   an exact hit, so `_activate_decode` calls `_store_checkpoint` again.
4. Capacity reservation runs before `PrefixStore.store` replaces the same
   key. Because the hit entry is pinned, an unrelated unpinned snapshot can
   be evicted to make room for a duplicate clone.

This is a source-traced hypothesis. I081 must prove the intended lifecycle
with tests before changing production behavior.

## Hypothesis

Treating the exact matched full prefix as already checkpointed will preserve
the lookup's existing LRU touch and pin ownership while avoiding a duplicate
reservation, cache clone, and same-key store. Exact output, cleanup, strict
prefix append, miss, cancellation, and persistence behavior will remain
unchanged. The three previously failing 4 GiB windows will then retain all
six entries with zero reservation/store eviction.

## Predeclared Work

1. Add focused failing tests that reproduce exact-hit decode activation and
   assert no second full-prefix reservation, clone, or store. Also assert that
   the hit entry remains touched and pinned until terminal cleanup.
2. Add negative controls for a cache miss, strict-prefix append, cancellation,
   disabled cache, and a matched entry whose logical length does not equal the
   request length. These paths must retain their current checkpoint behavior.
3. Implement the smallest request-lifecycle change at exact lookup or the
   full-prompt checkpoint predicate. Do not add a second cache index or change
   `PrefixStore.store` replacement semantics globally.
4. Run the affected config/engine/prefix/arrival tests and the full suite.
5. In fresh processes, rerun only I080's failing offsets 6, 18, and 24 at the
   temporary 4 GiB budget, using the locked plans and execution contract.

## Success Gates

- The new exact-hit regression test fails before implementation and passes
  afterward; all negative controls pass.
- Exact hits perform zero duplicate reservation/clone/store work while keeping
  one exact hit, the existing LRU touch, correct pin count, and terminal unpin.
- Miss and strict-prefix append requests still create the required checkpoint;
  cancellation and all terminal paths leave zero active/pinned state.
- All three fresh 4 GiB rows match I080 source, plan, execution, and terminal
  identity; retain six snapshots at or below budget; record zero evictions,
  preflight skips, and dropped/forbidden trace events; and preserve exact
  zero-prefill replay.
- A failed gate rolls back the runtime change. Passing the screen admits the
  lifecycle fix for a wider 8 GiB no-regression validation, not a 4 GiB
  production default.

## Non-Goals

- Do not lower the configured 8 GiB snapshot budget in I081.
- Do not change LRU victim selection, pin protection, clone reserve, snapshot
  max entries, persistence, serialization, or cache representation.
- Do not infer throughput, memory savings, sustained-session behavior, or a
  cross-engine ranking from three bounded retention rows.

## Rollback

Revert the exact-hit lifecycle change and its tests. The I080 harness offset,
raw evidence, rejection artifact, and configured 8 GiB default remain valid.

## Result

The predeclared red tests reproduced the duplicate exact-hit store, replay
store growth, and cancellation store growth. The explicit
`snapshot_skip_full_prompt_on_prefix_hit=false` rollback control continued to
refresh the exact checkpoint. The one-line predicate change then passed the
four focused tests and the affected 97-test cache/engine/config/persistence/
arrival suite. The full suite is green at `554 passed, 9 skipped, 1 warning`;
touched-file Ruff and `git diff --check` also pass.

In the default path, an exact hit now performs the lookup clone only. It keeps
the lookup LRU touch and request pin until cleanup, and does not emit a second
reservation event or call `PrefixStore.store`. Misses, strict-prefix appends,
cancellation, persistence, and the disabled-cache controls retain their
existing behavior. The compatibility switch remains an immediate rollback
without a code revert.

## Four-GiB Screen

Fresh processes reran I080's failing offsets 6, 18, and 24 with the locked
plans and unchanged source. All three rows match their I080 source, plan, and
seven-request terminal identities. Each retains six entries, performs six
stores, records one exact hit, zero evictions, zero preflight skips, six
prompt-free trace events, zero dropped events, zero replay prefill steps, and
zero active/pinned state after cleanup. The raw rows and paired hashes are
bound in `docs/loop-engineering/artifacts/ITER-20260731-081-exact-hit-checkpoint-refresh/exact-hit-checkpoint-refresh-screen.json`.

This clears the bounded lifecycle screen and admits the candidate to the next
configured-8-GiB validation. It does not change the production budget, claim a
memory or throughput gain, or establish a cross-engine ranking.

## Compatibility Smoke

The current source was also run in fresh Aster and direct MLX-LM processes on
the two-record locked MT-Bench smoke. Workload/source/model fingerprints,
greedy generation settings, output token IDs, text hashes, length finishes,
and zero swap delta match across both engines. This is output-compatibility
evidence only; the complete I066/I067 matrices remain the timing boundary.

## Next Iteration

I082 runs the same exact-hit lifecycle at the configured 8 GiB budget across
fresh, order-balanced windows, then exercises the wider cancellation and
persistence branches. The predicate remains an uncommitted candidate until
those production-budget gates pass; any failure rolls back to the explicit
refresh switch and leaves the 8 GiB default unchanged.
