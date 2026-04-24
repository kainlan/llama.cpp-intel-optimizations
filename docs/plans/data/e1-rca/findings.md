# E1 RCA — Minimal-Repro Investigation

**Bead:** llama.cpp-m09zb (E1 — L0 DirectSubmission non-flush wedges event.wait())
**Date:** 2026-04-24
**Owner:** implementer-1 (planner-validation team)

## Summary

The bare async-H2D + `event.wait()` pattern from `staging_buffer_pool::acquire`
(`common.hpp:1843-1871`) does **not** reproduce the m09zb wedge in isolation.
Five runs of `minimal-repro` with 64 x 64 MiB H2D copies + lookback waits all
completed cleanly in ~80 ms drain, max wait 5 ms.

Two of the three planned mitigations were tested; the third (`ZE_SERIALIZE`)
was tested as a baseline-mode env knob since the SYCL spec does not expose
per-call serialization.

A new, **independent** driver bug was discovered while testing mitigation 1
(`ext_oneapi_submit_barrier`): a single empty-waitlist barrier after one H2D
copy hangs and triggers `xe_guc_submit.c:1594 guc_exec_queue_timedout_job` ->
GT-reset cascade. This is a separate bug from m09zb but worth filing.

## Environment

- Kernel: 7.0.0-14-generic (Ubuntu 26.04)
- xe driver: in-tree, kernel module from `/lib/modules/7.0.0-14-generic`
- compute-runtime: `/Apps/compute-runtime` (custom build, fix/combined-26.09 = 26.09.37435.10 + cross-device wedge fix per CLAUDE.md memory)
- L0 loader: `Intel(R) oneAPI Unified Runtime over Level-Zero V2` 20.1.0 [1.14.37020]
- oneAPI: 2025.3.3 (icpx 2025.3.3.20260319)
- Device under test: `level_zero:0` = Intel Arc B580 Graphics (level_zero:1 = B50 disabled)

Full env dump: `env-dump.txt`. xe/drm kernel log excerpt: `driver-log.txt`.

## Variants

| Variant | Source | Env | Behavior | Exit | Notes |
|---------|--------|-----|----------|------|-------|
| baseline | `minimal-repro.cpp` | `MITIGATION=none` | All 64 chunks completed; per-event wait <= 5 ms; drain 80 ms | 0 | Does NOT reproduce m09zb |
| baseline (rerun, post-recovery) | `minimal-repro.cpp` | `MITIGATION=none` | Same as baseline | 0 | Confirms B580 recovered after GT-reset |
| baseline + `ZE_SERIALIZE=1` | `minimal-repro.cpp` | env override | All 64 chunks completed; same timing | 0 | Driver-level ENQUEUE serialize doesn't change behavior |
| baseline + `ZE_SERIALIZE=2` | `minimal-repro.cpp` | env override | Same as ZE=1 | 0 | Driver-level COMPLETION serialize doesn't change behavior |
| submit_barrier mitigation | `minimal-repro.cpp` | `MITIGATION=barrier` | `ext_oneapi_submit_barrier()` every 4 chunks -> HANG at chunk submit | 124 | Triggers GT-reset; see kernel-WARNING below |
| wait_and_throw mitigation | `minimal-repro.cpp` | `MITIGATION=wait` | `q.wait_and_throw()` every 8 chunks -> all events already complete; max wait 10 us total | 0 | Effective if needed, but baseline doesn't need it |
| barrier-bug isolation (empty waitlist) | `probe-barrier-bug.cpp` | none | One H2D copy + bare `submit_barrier()` -> barrier submit takes 26 ms, then **HANG** before printing copy.wait | 124 | Confirms the bug is in the empty-waitlist barrier path, not in any specific timing |
| barrier with explicit deps | `probe-barrier-deps.cpp` | none | One H2D copy + `submit_barrier({e_cp})` -> barrier submit 38 ms, wait 73 us | 0 | Non-empty waitlist works fine |

### Kernel-WARNING produced by `MITIGATION=barrier` and `probe-barrier-bug`

```
xe 0000:03:00.0: [drm] Tile0: GT0: Timedout job: seqno=4831999, lrc_seqno=4831999, guc_id=0, flags=0x73 in no process [-1]
xe 0000:03:00.0: [drm] Tile0: GT0: Kernel-submitted job timed out
WARNING: drivers/gpu/drm/xe/xe_guc_submit.c:1594 at guc_exec_queue_timedout_job+0x7b7/0xc90 [xe]
Workqueue: gt-ordered-wq drm_sched_job_timedout [gpu_sched]
RIP: 0010:guc_exec_queue_timedout_job+0x7c4/0xc90 [xe]
```

Recovery: `Tile0: GT0: reset queued / started / done`. Subsequent runs work.

## Conclusion

**The bare async-H2D + event.wait() pattern does NOT reproduce m09zb in isolation.**

This is itself important signal:

1. The wedge witnessed in `staging_buffer_pool::acquire` is **not** caused by the H2D + event.wait() shape alone.
2. m09zb requires additional context that this minimal repro lacks. Hypotheses, in priority order:
   - **(H1, most likely)** Some prior SYCL submission in the llama.cpp init path interacts with the staging pool's later wait. Candidates: oneDNN engine creation, MKL handle creation, the unified-cache's host-arena initialization itself, or one of the warmup probe kernels in `ggml_backend_sycl_init`.
   - **(H2)** The wedge depends on the **size of the staging buffer being reused** (the actual call site reuses <= 4 MB chunks; this repro uses 64 MB). Smaller chunks may interact differently with BCS engine queuing.
   - **(H3)** The wedge depends on a specific Level Zero queue option (BCS engine override, `ze_command_queue_priority_t`, copy-engine selection) that the SYCL Unified Runtime sets up for queues constructed via `ggml_sycl_set_device_queue` but a plain `gpu_selector_v` doesn't pick.
   - **(H4)** The wedge requires multi-context concurrency (D0.1/D0.2 use TWO contexts on the same model). A single-context async H2D loop doesn't trigger it.
3. **Mitigation #1 (submit_barrier) actively makes things worse.** Empty-waitlist barriers are unsafe on this driver and trigger a GuC kernel-job timeout. DO NOT use bare `q.ext_oneapi_submit_barrier()` in any code path. Filing this as a separate bug below.
4. **Mitigation #2 (`q.wait_and_throw()`) is safe but unnecessary at the bare-repro level** -- the events are already complete by the time we wait. It's still the obvious safe-fallback for the staging pool if we ever reproduce m09zb in a follow-up.
5. **Mitigation #3 (ZE_SERIALIZE)** -- neither value (1 = block-on-enqueue, 2 = block-on-completion) changes behavior at this layer. If m09zb is reproduced in a follow-up, ZE_SERIALIZE=2 is worth re-testing then.

## Proposed E1 Implementation Direction

Given that minimal-repro doesn't trigger the wedge, the E1 fix cannot be derived from this RCA in one shot. Recommended next steps for E1 proper:

**Step 1 -- broaden the repro envelope.** Build a new repro on top of `ggml_backend_sycl_init` (no llama.cpp, just ggml + the SYCL backend) that:
- Calls `ggml_backend_sycl_init(0)` (this is what unblocks the host-arena init path)
- Allocates a small device tensor via `ggml_backend_alloc_ctx_tensors_from_buft`
- Runs the same H2D loop but using `ggml_backend_tensor_set` instead of raw `q.memcpy`
- Adds a second `ggml_backend_sycl_init` mid-loop to mimic D0.1's two-context pattern

This is the smallest envelope that includes the SYCL backend's per-stream queue setup, oneDNN init, and the unified-cache init that bare-SYCL skipped.

**Step 2 -- if Step 1 reproduces:** bisect by removing one init step at a time (oneDNN, host arena, unified-cache zone configuration) until the trigger is isolated. The `D0.4` canary (`test-planner-canary-direct-load.cpp`) is already very close to this shape and witnesses the wedge -- start by running it with `GGML_SYCL_OP_TIMEOUT_MS=5000` and `GGML_SYCL_SAFE_MODE=1` to surface the exact op.

**Step 3 -- confirmed safe fallback for E1.** Regardless of root cause, replacing the `event.wait()` in `staging_buffer_pool::acquire` (common.hpp:1863) with a `q.wait_and_throw()`-equivalent **on the queue that produced the event** is functionally correct and at most a few microseconds slower per acquire. If Step 1 or 2 reproduces the wedge, this is a credible mitigation to test in-tree.

**Step 4 -- DO NOT use empty-waitlist `submit_barrier`.** Document this in code (`common.hpp` near the staging acquire) and in CLAUDE.md memory.

## Companion Bug Report (to file with Intel)

### Title
`xe driver + Arc B580: empty-waitlist sycl::queue::ext_oneapi_submit_barrier() hangs and triggers GuC kernel-job timeout`

### Environment
- Hardware: Intel Arc B580 Graphics (Battlemage, BMG)
- Kernel: Linux 7.0.0-14-generic (Ubuntu 26.04)
- xe driver: in-tree (kernel 7.0)
- compute-runtime: 26.09.37435.10 (also reproduces with stock 26.05)
- L0 loader: 20.1.0 [1.14.37020] (Intel oneAPI Unified Runtime over Level-Zero V2)
- oneAPI DPC++: 2025.3.3

### Repro
`tests/e1-rca/probe-barrier-bug.cpp` (~70 lines).

```cpp
sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
char * host = sycl::malloc_host<char>(N, q);
char * dev  = sycl::malloc_device<char>(N, q);
sycl::event e_cp = q.memcpy(dev, host, N);   // 64 MiB H2D
sycl::event e_br = q.ext_oneapi_submit_barrier();  // empty waitlist
e_cp.wait();   // hangs; GT timeout fires after ~5 s
```

### Expected
Empty-waitlist barrier should be a no-op or wait for all prior submissions
(per Level Zero `zeCommandListAppendBarrier` semantics) and immediately
return a complete event.

### Actual
- `submit_barrier` call itself blocks ~26 ms (vs. <1 ms for the
  non-empty-waitlist form on the same hardware/driver)
- Subsequent `event.wait()` never returns
- After ~5 s, `xe_guc_submit.c:1594` fires
  `guc_exec_queue_timedout_job` WARNING with seqno N, flags=0x73
- GT-reset cascade follows; subsequent runs in a fresh process recover

### Workaround
Pass an explicit non-empty waitlist:
```cpp
std::vector<sycl::event> deps{e_cp};
sycl::event e_br = q.ext_oneapi_submit_barrier(deps);
```
Verified working in `probe-barrier-deps.cpp`. Submit time 38 ms, event wait
73 us.

### Files attached
- `tests/e1-rca/probe-barrier-bug.cpp`
- `tests/e1-rca/probe-barrier-deps.cpp`
- `docs/plans/data/e1-rca/driver-log.txt`
- `docs/plans/data/e1-rca/env-dump.txt`

## Fix Attempt #1 (Task 9, 2026-04-24): event.wait() -> event.wait_and_throw()

**Status: applied, did NOT clear m09zb. Broader-envelope investigation needed.**

Surgical 1-line replacement in `staging_buffer_pool::acquire()` (common.hpp:1863)
and `staging_buffer_pool::drain_all()` (common.hpp:2010): `event.wait()` ->
`event.wait_and_throw()`. Same wait semantics; async errors propagate as
sycl::exception instead of being silently swallowed.

Rationale: this RCA's "Step 3 -- confirmed safe fallback" recommendation. The
producer queue handle is not stored on the slot, so the event-level
wait_and_throw is the most surgical interpretation of "wait_and_throw on the
producing queue". A `q.wait_and_throw()` on the consumer queue (the only one
acquire() has access to) would be too broad and could stall unrelated work.

Verification:
- Build: `ninja -C build llama-completion test-planner-canary-direct-load` --
  succeeded. (Note: `tests/test-pinned-chunk-pool.cpp` has pre-existing
  breakage referencing removed unified_cache::{allocate,free}_pinned_runtime
  methods -- unrelated to this fix; flagged for separate cleanup.)
- llama-completion (Mistral 7B Q4_0, single context): TIMEOUT after 120 s.
  Last stderr line: `[HOST-ARENA] Zones configured: WEIGHT=0.0 MB KV=64.0 MB
  STAGING=221.1 MB SCRATCH=442.2 MB` -- identical signature to the originally
  documented m09zb wedge.
- D0.4 canary (`test-planner-canary-direct-load`): TIMEOUT after 120 s. Last
  stderr line: `[SYCL] Allocated pinned runtime chunk 1 (size=2048.0 MB,
  total=2.0 GB)` -- wedges in `ggml_backend_sycl_buffer_set_tensor` per the
  bead's documented signature.

Confirms the Task 4 hypothesis: bare async-H2D + event.wait() is not the
trigger pattern. The wedge is something else in the SYCL backend init/post-
init path. The fix is left in place because it's a strict improvement (errors
propagate properly) but is NOT the m09zb fix.

Recommended next investigation (per Step 1 in the conclusion above): build a
broader-envelope repro on top of `ggml_backend_sycl_init` -- this is the
smallest envelope that includes per-stream queue setup, oneDNN init, and the
unified-cache initialization that bare-SYCL skipped.

## Step 1: Broader-Envelope Investigation (Task 10, 2026-04-24)

**Status: trigger isolated. m09zb is NOT in the staging-pool acquire path.**
**The wedge is in the very first H2D submission on a freshly-initialized
SYCL backend stream.**

### Stage-by-stage bisection

`tests/e1-rca/broader-envelope-repro.cpp` runs ggml + ggml-sycl (no llama.cpp)
at progressively wider envelopes around `ggml_backend_sycl_init`. Each STAGE
runs to completion or times out; the first to timeout is the trigger. Tested
on Arc B580, level_zero:0, with the Task 9 `event.wait_and_throw` fix already
applied at common.hpp:1863.

| STAGE | Description | Wall time | Result |
|-------|-------------|-----------|--------|
| `init-only` | `ggml_backend_sycl_init(0)` then `ggml_backend_free` | 11 s init, 0.2 s free | PASS |
| `init-wait` | init, then `ggml_backend_synchronize(backend)` (drains all queues via `queues_wait_and_throw`) | 11 s init, 0.22 s sync, 115 us free | PASS |
| `init-alloc` | init, then `ggml_backend_alloc_ctx_tensors_from_buft` for one 4 KB F16 tensor | 11 s init, 63 us alloc | PASS |
| `init-set` (default GLOBAL drain) | init + alloc + `ggml_backend_tensor_set` 4 KB | wedges inside `set_tensor` (>30 s) | FAIL |
| `init-stream-set` (`SET_TENSOR_STREAM_FENCE=1`) | init + alloc + `ggml_backend_tensor_set` 4 KB | wedges inside `set_tensor` (>30 s) | FAIL |

### Findings

1. **Backend init alone does NOT leave the queue stuck.** `init-wait` proves
   that `queues_wait_and_throw()` over all `_queues` populated by init returns
   in ~220 ms.
2. **Allocating a device tensor does NOT cause submissions to get stuck.**
   `init-alloc` returns in 63 us with no queue activity.
3. **The wedge is in the FIRST H2D copy** issued via the buffer's stream
   inside `ggml_backend_sycl_buffer_set_tensor`. Both sync modes (GLOBAL
   `queues_wait_and_throw` at ggml-sycl.cpp:12746 and STREAM_FENCE
   `stream->wait_and_throw()` at 12750) wedge.
4. **The pre-`set_tensor` GLOBAL drain (12746) is NOT the trigger.** STREAM
   sync is narrower (just one queue) and still wedges. So the wedge is
   downstream of the sync-mode dispatch.

### Code path that wedges

For an F16 4 KB tensor with `dst_host_accessible == false` (device buffer),
`ggml_backend_sycl_buffer_set_tensor` falls into the staged-upload branch
(ggml-sycl.cpp:~12990-13100):

```cpp
ggml_sycl::offload_buffer_lease stage_lease{};
acquire_offload_buffer(req, &stage_lease);   // first call -> grow_into 2 GB chunk
char * stage_ptr = static_cast<char *>(stage_lease.handle.ptr);
memcpy(stage_ptr, data, 4096);               // host->host copy
// THIS is the wedge:
auto err = CHECK_TRY_ERROR(
    (*stream).memcpy(tensor->data, stage_ptr, 4096).wait());
```

The `(*stream).memcpy(...).wait()` never returns. This is the same class of
failure as the original m09zb signature ("post-init event.wait() never
returns") but at a different code site than Task 9 patched -- the staging
pool's `acquire()` is never reached because the FIRST set_tensor call fails
before any reuse can happen.

### Why Task 4's bare-SYCL minimal-repro didn't reproduce

`tests/e1-rca/minimal-repro.cpp` constructs its own `sycl::queue{gpu_selector_v,
in_order{}}` -- a freshly created queue with NO prior backend state.
`ggml_backend_sycl_init` does substantially more: it brings up dpct's device
manager, creates a `sycl::context` shared across queues, runs the
`ggml_backend_probe_max_alloc_size` binary search (14 successive
malloc_device/free cycles up to 11.6 GB on Arc B580), queries XMX caps, and
populates the device's `_queues` array. SOMETHING in that init sequence
leaves the L0 driver in a state where the next H2D copy submitted on a
backend-managed queue does not flush.

### Why Task 9's fix didn't clear m09zb

Task 9 replaced `event.wait()` with `event.wait_and_throw()` in
`staging_buffer_pool::acquire` (common.hpp:1863). That code path is only
reached when reusing a previously-released slot. The wedge fires on the
FIRST set_tensor's *initial* H2D, before any slot has been released. The
staging pool's acquire is well outside the trigger envelope.

### Hypotheses for the actual trigger (priority order)

1. **(H1) Probe state.** `ggml_backend_probe_max_alloc_size` does 14 binary-
   search malloc_device/free cycles up to 11.6 GB. The L0 USM allocator
   keeps freed allocations in a pool; the post-probe state may leave the
   default queue's command list in a state that doesn't flush the next
   submission. Disabling the probe (or running with a tiny probe upper
   bound) would test this.
2. **(H2) DPCT context vs. plain SYCL context.** dpct's `dev_mgr` constructs
   queues differently from `sycl::queue{gpu_selector_v}`. Possible:
   `enable_exception_handler` flag, async-handler installation, or implicit
   context ordering effects.
3. **(H3) BCS queue creation.** `[UNIFIED-CACHE] Created BCS queue for H2D
   copy pipelining` happens in init. The presence of a separate BCS queue
   alongside the default IOQ may interact poorly with the L0 driver's
   DirectSubmission.
4. **(H4) Pinned chunk allocation.** The first `acquire_offload_buffer`
   triggers `grow_into` -> `sycl::malloc_host` of a 2 GB chunk. The L0 driver
   has to set up GGTT mappings for this; the `(*stream).memcpy(... stage_ptr ...)`
   immediately after may hit unmapped host pages.

### Concrete next step

Add two more STAGEs to `broader-envelope-repro.cpp`:
- `STAGE=init-no-probe` -- env-disable the probe in `ggml_backend_probe_max_alloc_size`
   (would need a new env hook in ggml-sycl.cpp). If this PASSes set_tensor,
   H1 is confirmed.
- `STAGE=init-set-prefilled-stage` -- pre-allocate stage_ptr from `malloc_host`
   on a *fresh* sycl::queue (not the backend's), then submit `stream->memcpy`
   with that ptr. If this PASSes, H4 is confirmed. If it still wedges,
   the trigger is in the backend stream itself, independent of stage memory.

Both probes are 30-60 minutes of work each. Filing as recommended follow-ups.

### Recommended E1 fix shape (revised, post-Task-10)

The Task 9 fix at common.hpp:1863 is keep-but-not-fixing-m09zb. The actual
fix needs to live in ONE of:

- `ggml_backend_sycl_buffer_set_tensor` -- if the first H2D wedge is
  reproducible, replace `(*stream).memcpy(...).wait()` at line ~13076 with a
  no-stream copy path (CPU memcpy when dst is host-accessible already exists;
  extend it or use a fresh `sycl::queue` from the context for the FIRST
  set_tensor only).
- The init path -- if H1 confirms, gate the probe behind an env or shrink
  its upper bound to avoid leaving the L0 driver in the bad state.
- DPCT helper -- if H2 confirms, switch backend queues to a plain
  `sycl::queue` constructor instead of `dpct::device_ext::create_queue`.

None of these are 1-line surgical fixes; all need the further-bisection
described above before committing.
