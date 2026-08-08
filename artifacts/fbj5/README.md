# fbj5 — retained-handle barrier mutation reproducer

Plan Task 12c (`docs/plans/2026-08-02-sycl-merge-readiness.md`), tracker `llama.cpp-fbj5`.
Written by `impl-fbj5` **before** any of it was executed: everything under
"Pre-registered predictions" is a prediction, not a transcript.

| | |
|---|---|
| test | `tests/test-sycl-retained-handoff-contended.cpp` |
| registration | `ggml/src/ggml-sycl/CMakeLists.txt` (next to `test-sycl-retained-handoff-barrier`) |
| mutation | `artifacts/fbj5/guard-mutation.diff` (1 file, 1 hunk, 1 line) |
| historical evidence | `artifacts/oze0/repro-drain-barrier*.cpp`, `artifacts/oze0/repro-drain-barrier-output.txt` (untouched) |

## The guard

`drain_retained_handles(true)` in `ggml/src/ggml-sycl/mem-handle.cpp` waits on

```cpp
return state.queue.empty() && state.active == 0 && state.publishers == 0;
```

The mutation deletes the third clause, `&& state.publishers == 0`, and nothing else.

**Why that clause is *the* ownership guard for this scenario.** `state.queue` and
`state.active` only ever describe handles that have *already* been handed to
`retain_handles_until_event()`. A thread that owns an allocation and has not reached
that call yet appears in neither, so before `1f84f1a95` it was invisible to the
barrier — which is exactly what `artifacts/oze0/repro-drain-barrier-output.txt`
measured: 12 of 12 runs, 79–99 % of boundary crossings returned "all clear" with
another thread demonstrably holding.

`1f84f1a95` closed that window by counting the owners themselves.
`begin_retained_handle_publish()` increments `state.publishers` under the state mutex
*before* the owner touches its allocation (`getrows.cpp:244`, after the allocate and
resolve, before any use); the ticket is then moved into `retain_handles_until_event()`,
whose by-value parameter destructor decrements it only *after* the handles are already
in `state.queue`. There is therefore no instant at which the owner is unaccounted for,
and `state.publishers == 0` is the clause that makes the barrier consult that fact.

Deleting it restores the pre-`1f84f1a95` predicate byte for byte. The counter, the
ticket type, and every call site stay exactly as they are — this is removal of the
guard, not injection of a bug elsewhere.

The other candidates were considered and rejected as the primary mutation:

- **`graph_lifetime_retention_active()` routing** — a real guard, but it is already
  pinned by `test-sycl-graph-retention-scope`, and removing it changes *where* handles
  are parked rather than whether an owner is visible mid-hold.
- **the drain worker's `record.event.wait_and_throw()`** — device-completion, not
  ownership; device-free it waits on an already-complete event and removing it is a
  no-op for this test.
- **`mem_handle` refcount release ordering** — not reachable as a one-clause removal.

## What the test asserts, and why it is not a coin flip

Owners run the production shape from `getrows.cpp`: take a ticket, publish a hold,
sleep `kHoldWindow`, leave the hold, then hand the handle to
`retain_handles_until_event()` with the ticket. The boundary (the main thread) polls
each owner's hold state, calls `drain_retained_handles(true, 50)`, and re-reads.

A **violation** is: the drain reported clear, and some owner's hold state is *bit
identical* before and after with the holding bit set — the owner never left that hold.
Because the ticket is live across the whole of that hold, `state.publishers` cannot
have reached 0 during it, so with the clause present a violation is **structurally
impossible**, not merely unlikely. Only the RED side depends on timing.

`kHoldWindow` is 50 µs deliberately. The oze0 note records that a 200-cycle window —
far shorter than a real `stream_dma` submission — dropped the original reproduction
rate to 1 in 5, i.e. it flattered the barrier. Do not shrink it.

**Self-check against a vacuous pass.** Each scenario must observe `kMinCrossings` (50)
*crossings*: drains that began while an owner was mid-hold and went on to report clear.
Those are precisely the events that become violations when the guard is gone, so a run
that cannot reach 50 of them has tested nothing and **fails** rather than passing. The
run also fails if any handle was parked for command-graph lifetime, which would mean
the event-bound path was not the one exercised.

Scenarios run at 1, 2 and 3 concurrent owners, mirroring
`repro-drain-barrier-multi.cpp`'s finding that the holder count is a distribution
bounded by the number of contexts rather than a constant.

## Pre-registered predictions

### GREEN — HEAD, unmutated

Exit code **0**. stdout carries one stats line and one PASS line per scenario, then a
final PASS:

```
owners=1 crossings=<N, at least 50> contended=<C> clears=<K> violations=0 graph_parked=0
PASS: owners=1 -- drain never reported clear across <N> contended crossings
owners=2 ... violations=0 graph_parked=0
PASS: owners=2 -- ...
owners=3 ... violations=0 graph_parked=0
PASS: owners=3 -- ...
PASS: retained-handle publisher barrier holds against concurrent owners
```

`crossings` is ≥ 50 and may exceed it in the multi-owner scenarios (one clear drain can
close out up to one crossing per owner). `contended` and `clears` are free-running
counters — any non-zero values are fine. `violations` and `graph_parked` **must** be 0.
Expected wall time: well under 10 s for all three scenarios; the registration allows 180 s.

### RED — with `guard-mutation.diff` applied

Exit code **1**, and stderr contains the literal string

```
retained-handle barrier violated
```

in full:

```
FAIL: retained-handle barrier violated: drain reported clear while owner 0 held an unpublished handle (owners=1 hold_state=<odd> crossings=<small>)
```

preceded on stdout by a stats line with `violations=1`. The failure is fail-fast, so
expect it in the **`owners=1`** scenario, within milliseconds, with `crossings` small
(often 0 — the very first contended drain is usually the violating one).

Confidence that it flips: the oze0 measurement puts a violation at 79–99 % *per
crossing*, and the mutated drain returns in microseconds against a 50 µs hold, so the
per-crossing probability here is at the top of that range. The test would have to miss
on 50 consecutive crossings to pass, which is below 1e-30. If it does **not** flip,
that is a finding, not noise — report it back rather than retrying.

**Expected collateral, not a bug:** the single-threaded sibling
`test-sycl-retained-handoff-barrier` also goes red under this mutation. That is the
positive control that the mutation landed at all. It does not make this test redundant —
the sibling proves the counter is *consulted*, while this one constrains the *ordering*
(ticket acquired before ownership begins, released only after the handle is queued).
A mutation that merely moved the ticket release before the queue push would leave the
sibling green and turn this one red.

## Commands for the lead

Build first — the mutation is inside `libggml-sycl`, so building the test target alone
would link against a stale library. Confirm `build/bin/libggml-sycl.so*` postdates the
patch before trusting either verdict.

```bash
source /opt/intel/oneapi/setvars.sh --force

# GREEN (HEAD)
./scripts/sycl-build.sh
ctest --test-dir build -R '^test-sycl-retained-handoff-contended$' --output-on-failure
# expect: exit 0, "PASS: retained-handle publisher barrier holds against concurrent owners"

# RED (one guard removed)
git apply artifacts/fbj5/guard-mutation.diff
./scripts/sycl-build.sh
ctest --test-dir build -R '^test-sycl-retained-handoff-contended$' --output-on-failure
# expect: exit 1, stderr contains "retained-handle barrier violated"

# restore
git apply -R artifacts/fbj5/guard-mutation.diff
./scripts/sycl-build.sh
```

Anchor the `-R` pattern as written. `ctest -R`/`-E` take unanchored substrings, and
`test-sycl-retained-handoff-barrier` is the sibling next door — an unanchored
`-R test-sycl-retained-handoff` runs both, which is fine to do deliberately but should
not happen by accident when you are reading one verdict.

No GPU, no model, no device is involved; the test is safe at any ctest parallelism.

## Interpreting a failure that is neither of the above

`FAIL: contended window never exercised: <n>/50 crossings in 20s` means the boundary
never managed both to catch an owner mid-hold and to see a legitimate clear — a
scheduling problem on the host, not a barrier violation. It is deliberately a failure
rather than a green pass, because a run with no crossings proves nothing either way.
If it appears, send it back: the fix is a timing constant (`kIdleWindow`), not a
verdict about the guard.
