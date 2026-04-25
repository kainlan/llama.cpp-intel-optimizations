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

## Step 2: H1/H4 Bisection Probes (Task 11, 2026-04-24)

**Status: H1 CONFIRMED. The 14-step alloc probe in `ggml_backend_sycl_init` is the m09zb trigger.**

### Two new STAGEs added to broader-envelope-repro.cpp

| STAGE | What it tests | Hypothesis |
|-------|---------------|------------|
| `init-no-probe` | Same as `init-set` but exports `GGML_SYCL_E1_RCA_DISABLE_ALLOC_PROBE=1` BEFORE calling `ggml_backend_sycl_init`. The env-gate (added in ggml-sycl.cpp:~9345 for RCA only, default OFF) skips `ggml_backend_probe_max_alloc_size`. PASS confirms H1. | H1: alloc probe state |
| `init-set-prefilled-stage` | After init + alloc, submits a 4 KB H2D on a FRESH `sycl::queue{gpu_selector_v, in_order{}}` BEFORE `ggml_backend_tensor_set`. PASS confirms H4. | H4: GGTT mapping / first-H2D priming |

### Results

| STAGE | Wall time | Result | Interpretation |
|-------|-----------|--------|----------------|
| `init-no-probe` (run 1) | init=1.1 s, set_tensor=224 ms, readback PASS | **PASS** | H1 confirmed: probe is the trigger |
| `init-no-probe` (run 2, reproducibility) | set_tensor=225 ms, readback PASS | **PASS** | reproducible |
| `init-set-prefilled-stage` | wedges in `warmup_h2d_fresh_queue` (>60 s) | **FAIL** | H4 refuted — but see below |
| `init-set` (baseline rerun) | wedges as before (>30 s) | FAIL | confirms baseline still reproduces |

### Findings

1. **H1 CONFIRMED.** Skipping the binary-search alloc probe in `ggml_backend_sycl_init` makes set_tensor work. The probe — 14 successive `malloc_device` / `free` cycles binary-searching from `max_alloc_size` (~11.6 GB) down to a "safe" size — leaves the L0 driver in a state where the next H2D copy submitted on **any** queue does not flush.
2. **H4 refuted but with a stronger corollary.** The warmup H2D on a fresh `sycl::queue{gpu_selector_v, in_order{}}` ALSO wedges after init runs the probe. So the wedge is **not** specific to backend-managed queues. Once the probe has run, no queue on the device can complete an H2D. The bad state is at the L0 driver / device level.
3. **Init time is 10x faster without the probe.** `init=11.0 s` (with probe) vs `init=1.1 s` (without). The probe alone is ~10 s of init time on Arc B580 — itself notable.
4. **Probe accuracy is preserved by the existing fallback.** When `safe_alloc == 0` (probe skipped or probe failed), the code already falls back to `floor(probe_upper * safety_margin)` = 95% of `max_alloc_size`. With the probe disabled, `safe_max_alloc_size` was reported as 11024.9 MB vs 11024.8 MB with the probe — within 0.1 MB. The probe's binary search is finding essentially the same answer the simple multiplication produces.

### Recommended E1 fix shape (post-Task 11)

**Primary direction: replace the binary-search alloc probe with `floor(max_alloc_size * 0.95)`.**

The probe is 14 expensive malloc_device/free cycles with a known-bad side effect (m09zb), and produces a result that's within ~100 KB of the trivial calculation. Replacing it removes ~10 s of init time AND clears m09zb in one change.

Alternative: keep the probe but reset/recover the L0 state after it. The post-probe stack pattern (`malloc_device` allocations linger in the L0 USM pool) suggests an explicit pool drain or a `queue.wait_and_throw()` followed by `sycl::context` re-init might recover the state. This is more invasive and less predictable; recommend the primary direction unless the probe is empirically known to produce a more accurate cap on hardware where 95% is wrong.

The earlier fix-shape candidates are deprecated by this finding:
- (A) "Use a fresh sycl::queue for first set_tensor H2D" -- refuted by `init-set-prefilled-stage` failure. Fresh queues wedge too.
- (C) "Switch dpct to plain SYCL queue construction" -- not relevant; the wedge is at the device/driver level, not the queue construction path.

### Code changes added by Task 11

1. **ggml-sycl.cpp env-gate** (RCA-only, default OFF): `GGML_SYCL_E1_RCA_DISABLE_ALLOC_PROBE=1` skips the probe and uses the simple-multiplication fallback. Left in place because it's the same control the fix would use; production behavior unchanged when env not set.
2. **broader-envelope-repro.cpp**: two new STAGEs (`init-no-probe`, `init-set-prefilled-stage`) and a `<sycl/sycl.hpp>` include for the fresh-queue warmup.
3. **CMake**: `e1-broader-envelope` now compiles with `-fsycl` (was previously linking ggml only — sycl::queue usage requires the SYCL frontend).

### Recommended Task 12 (the actual E1 fix)

Replace the probe call site at ggml-sycl.cpp:~9345 with the trivial 95% calculation, behind the same env-gate so it can be reverted for diagnostics. Verify against the same two probes used in Task 9: model load + D0.4 canary should both PASS. Phase C unblocks.

## Step 3: Task 12 + Task 13 Negative Results (2026-04-24)

**Status: Task 11 fix-shape recommendation REFUTED. The probe is not safely
removable on this driver. Phase C remains gated; treating as upstream bug.**

### Task 12 attempt: replace probe with `floor(probe_upper * 0.95)`

Implemented exactly as Task 11 recommended: skip
`ggml_backend_probe_max_alloc_size`, set `safe_alloc =
floor(probe_upper * 0.95)`. Built clean. Validated:

- `test-planner-canary-direct-load` (D0.4): **PASS** -- m09zb cleared,
  tensor_set_us=223376 (~223 ms), readback ✓.
- `llama-completion -m mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3, 4, 5,'
  -n 15 --seed 42 --temp 0`: **FAIL** -- exit 134, aborted with
  `level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)`
  inside `ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>` at
  ggml-sycl.cpp:21339 on the FIRST graph compute, before producing any
  tokens. Reproducible across multiple runs.

The simple substitution clears m09zb on D0.4 but breaks Mistral 7B
inference. Trade was NET-NEGATIVE -- swapping a documented P0 (m09zb wedge)
for an undocumented P0 (Mistral 7B SIGABRT). Reverted; not committed.

### Task 13 bisection: one-shot warmup at varying sizes

Hypothesis: the probe's side effect was implicitly priming the L0 USM pool;
a single alloc/free at `safe_alloc` would prime it without the 14-step
binary-search trigger envelope.  Implemented and bisected with the
`GGML_SYCL_E1_WARMUP_BYTES` env override:

| Warmup size | m09zb wedge? | OOR at first MUL_MAT? | Mistral 7B outcome |
|-------------|--------------|------------------------|---------------------|
| 256 MB      | NO           | YES                    | SIGABRT (exit 134)  |
| 512 MB      | NO           | YES                    | SIGABRT (exit 134)  |
| 768 MB      | NO           | YES                    | SIGABRT (exit 134)  |
| 1024 MB     | NO           | YES                    | SIGABRT (exit 134)  |
| **1536 MB** | **YES**      | n/a (wedged earlier)   | TIMEOUT (exit 124)  |
| 2 GB        | YES          | n/a                    | TIMEOUT             |
| 4 / 6 / 8 GB| YES          | n/a                    | TIMEOUT             |
| 11024 MB (Task 12 size) | YES | n/a                | TIMEOUT             |

### Critical conclusion

**There is no warmup size on Arc B580 + xe + oneAPI 2025.3 that satisfies
both constraints simultaneously.**

- The wedge threshold lives between 1024 MB and 1536 MB.  Below that, m09zb
  does not fire.
- The runtime-priming threshold lives somewhere above 1024 MB.  Below that,
  the runtime hits OUT_OF_RESOURCES on the first MUL_MAT compute.
- These thresholds **do not overlap**.  Empty window.

This refutes ALL of the originally proposed fix shapes:
- (A) "fresh sycl::queue for first set_tensor H2D" -- already refuted by
  Task 10's `init-set-prefilled-stage` FAIL.
- (B) "gate or shrink the alloc probe" -- the simple replacement (Task 12)
  causes OOR; the warmup workaround (Task 13) cannot prime without wedging.
- (C) "switch dpct queue construction" -- already refuted by Task 11; the
  bad state is at the L0 device level, not queue-specific.

### Recommendation (superseded — see Step 4)

The Step 3 recommendation, written 2026-04-24, was to accept m09zb as
upstream-blocked and file a comprehensive driver bug. **That conclusion
no longer reflects the current state**: the patched compute-runtime
build at `/Apps/compute-runtime` (branch `fix/combined-26.09`,
libze_intel_gpu.so 1.14.37435) has been installed as the system default
on 2026-04-25 and resolves m09zb. See **Step 4** below for the
verifying evidence.

The Task 9 fix at common.hpp:1863 (`event.wait()` -> `event.wait_and_throw()`)
remains in place as a strict improvement -- async errors propagate properly
even if it is not the m09zb fix.

## Step 4: Patched runtime resolves m09zb (2026-04-25)

**Status: m09zb is upstream-FIXED in our compute-runtime fork; that
fork is now the system-default L0 GPU library via `dpkg-divert` +
`ldconfig`.**

### Paired test (stock vs patched libze)

| Configuration | libze_intel_gpu.so version | Behavior | safe_max_alloc_size |
|---|---|---|---|
| Stock (pre-2026-04-25) | 1.14.37020 (Ubuntu/oneAPI default) | First H2D after `ggml_backend_sycl_init` wedges (m09zb signature) | 11024 MB (oversized, accepted by stock libze even though L0 cannot actually flush copies of this size) |
| Patched (2026-04-25 onward) | 1.14.37435 (`fix/combined-26.09`) | First H2D after init returns cleanly; D0.4 canary PASSES | 1593 MB (correct ceiling — patched libze rejects oversized allocations rather than accepting and wedging) |

The probe under stock libze claimed 11024 MB was allocatable; patched
libze correctly reports 1593 MB. The probe was always reporting what
libze told it; the bug was in stock libze's accepting allocations it
could not actually flush copies on. Patched libze enforces the real
ceiling, so the probe (and downstream `ggml_backend_sycl_init`) lands
in a workable state.

### D0.4 canary verification

Under the patched runtime:
- `tests/test-planner-canary-direct-load` runs to completion.
- `tensor_set_us = 282422` (~282 ms for the canary's direct mmap →
  device load — no hang).
- Pre-patch: same canary timed out at 60 s.

This validates A7's "direct mmap → device `ggml_backend_tensor_set`
within 60 s with byte-identical readback" acceptance for the unified
memory placement plan epic.

### New wedge identified (Task 14)

A separate wedge surfaced under the patched runtime: a single 4 GB KV
buffer allocation exceeds the patched runtime's 1.5 GB single-alloc
ceiling. This is a different failure mode (alloc rejection, not
silent wedge) and is tracked separately as Task 14. It does **not**
re-open m09zb.

### Conclusion

m09zb is **resolved**. No llama.cpp code change beyond Task 9's
`event.wait_and_throw` strict improvement is required. The fix lives
in our compute-runtime fork at `/Apps/compute-runtime` branch
`fix/combined-26.09`, system-installed via `dpkg-divert` so every
process on the host loads it transparently as
`libze_intel_gpu.so.1`.

Phase C tasks (5/6/7) of the planner-validation epic are now unblocked
on the m09zb axis. The remaining gating item is Task 14 (KV alloc
ceiling) for the subset of Phase C that allocates > 1.5 GB single
buffers; smaller-context probes can run today.

## Step 4: Patched compute-runtime + KV-clear sibling wedge (Task 14, 2026-04-24)

**Status: m09zb proper is upstream-fixed in patched compute-runtime.
Steps 1-3 above measured stock libze (1.14.37020) behavior. A different
sibling wedge surfaces under patched libze (1.14.37435) and is filed as
follow-up bead llama.cpp-zhzbp.**

### Important context correction

All Step-1/2/3 results above were collected against the stock system libze
(`/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so` -> 1.14.37020). The patched
compute-runtime at `/Apps/compute-runtime/build-26.09` (1.14.37435) is now
system-installed via dpkg-divert, so future tests automatically use it.
The "upstream-blocked" recommendation in Step 3 was incorrect: m09zb
proper IS fixed in our patched runtime, just under a different code path
than my Tasks 12/13 attempted.

### Patched runtime evidence (Task 14)

| Test | Stock libze 1.14.37020 | Patched libze 1.14.37435 |
|------|-------------------------|----------------------------|
| Probe `safe_max_alloc_size` | 11024.8 MB (overestimate) | **1593.1 MB** (real cap) |
| Total VRAM reported | 11605.2 MB | 12216.0 MB |
| D0.4 canary | wedge | **PASS** (282 ms) |
| Mistral 7B first H2D | wedge at `[HOST-ARENA] Zones configured` | OK; gets past zones, allocates KV |
| Mistral 7B KV clear | (never reached) | **wedge at `tiered_kv_buffer_clear`** |

The probe correctly converges to the real per-allocation cap on patched
runtime (~1.5 GB on Arc B580). Stock returned a fictitious 11 GB cap whose
allocations subsequently wedged.

### New sibling wedge (filed as llama.cpp-zhzbp)

Mistral 7B llama-completion now hangs at `ggml_backend_buffer_clear(buf, 0)`
calling `tiered_kv_buffer_clear` (ggml-sycl.cpp:14802). The function
iterates 32 transformer layers x 2 (k+v) calling
`stream->memset(la.ptr, value, la.size).wait()` and one of the early
calls hangs. **Not size-dependent** -- bisected n_ctx values:

| n_ctx | KV size | Outcome |
|-------|---------|---------|
| 512   | 64 MiB  | wedge   |
| 2048  | 256 MiB | wedge   |
| 8192  | 1024 MiB| wedge   |
| 16384 | 2048 MiB| wedge   |
| default| 4096 MiB| wedge  |

Even 2 MiB per layer hangs. Quick-fix attempt: `ctx->stream->wait_and_throw()`
drain inserted before the per-layer memset loop -- did not clear the wedge
(reverted).

### Recommended fix shape (follow-up, NOT in this task's scope)

Replace the per-layer `stream->memset(...).wait()` loop with a single
contiguous memset across the arena-allocated KV region. The vmem_pool path
at line 14807 already does this; extend to the non-vmem path when
`alloc_base_is_arena == true`. Skip per-layer iteration when the layer
allocations are contiguous sub-ranges of one arena alloc.

### Implications for Phase C

- Phase C tasks 5 / 7 do NOT run inference -- unaffected by zhzbp.
- Phase C task 6 (D0.4 re-run) PASSES under patched runtime.
- The "upstream-blocked" recommendation in Step 3 should NOT be applied as
  the standing conclusion. The real conclusion is: m09zb proper is fixed,
  one sibling wedge remains as a tracked P1 follow-up, Phase C is NOT
  blocked on it.
