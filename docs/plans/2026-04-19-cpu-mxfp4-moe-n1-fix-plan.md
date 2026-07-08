# Narrow fix: SYCL MoE CPU-TG activation staging alloc failure → uninitialized dst

**Status**: PLAN — not yet implemented
**Branch**: `feature/sycl-coalescing`
**Priority**: P0 — blocks 8 test-backend-ops MUL_MAT_ID MXFP4 CI failures + 20B Gate 4

## 1. Reproduction

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops -o MUL_MAT_ID
```

8 MXFP4 failures, ALL n=1:

| n_mats | n_used | b | m   | n | k   | ERR  |
|--------|--------|---|-----|---|-----|------|
| 4      | 1      | 0 | 512 | 1 | 256 | 76.2 |
| 4      | 1      | 1 | 512 | 1 | 256 | 76.8 |
| 4      | 2      | 1 | 512 | 1 | 256 | 71.7 |
| 4      | 4      | 1 | 512 | 1 | 256 | 76.8 |
| 8      | 1      | 0 | 512 | 1 | 256 | 74.7 |
| 8      | 1      | 1 | 512 | 1 | 256 | 73.7 |
| 8      | 2      | 1 | 512 | 1 | 256 | 81.6 |
| 8      | 4      | 1 | 512 | 1 | 256 | 66.7 |

With `GGML_SYCL_CPU_EXPERT_TG=0` all 69 MXFP4 MUL_MAT_ID tests pass.

20B canary:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p "1, 2, 3, 4, 5," -n 30 --seed 42 --temp 0
```
Currently produces garbled output. Gate 4.

## 2. Root Cause

**Primary bug — `ggml/src/ggml-sycl/ggml-sycl.cpp:34384-34390`**

In `ggml_sycl_mul_mat_id` (line 34080), when `ne12==1` (TG batch=1) and host weights are
detected, the moe-fusion fast-path at line 34304 is entered. It tries to stage the activation
token into a thread-local managed host-pinned staging buffer:

```cpp
// line 34384
if (!tl_first_act.ensure(*stream, ctx.device, act_bytes,
                          ggml_sycl::alloc_role::EXPERT_STAGING,
                          ggml_sycl::runtime_category::HOST_COMPUTE,
                          "moe_fusion_first_act")) {
    GGML_LOG_ERROR("[MOE-P4] Failed host activation staging alloc for layer %d\n", ...);
    moe_fusion_erase(cur_layer_fast);
    return;  // dst NEVER WRITTEN -- GARBAGE
}
```

Under `test-backend-ops`, the managed arena holds ~370 MB after the test fixture reserves
budget for probe (512 MB slack), scratch (256 MB), and oneDNN (256 MB). The
`ensure()` call for the 1 KB activation staging buffer returns false. The code erases
fusion state and **returns without writing to `dst`**. The output tensor holds uninitialized
allocation bytes → ERR 66-82 (garbage-level error vs 0.0005 threshold).

**Why n≥4 passes**: `ne12 > 1` skips the fusion fast-path. The unfused `dispatch_cpu_compute`
lambda at line 36595 uses `g_pinned_buffer_pools[ctx.device]` — a pre-allocated pool seeded
at model-load time. Pool allocs succeed even under test-fixture budget starvation.

**Contradicts the initial ggml-cpu-backend hypothesis**: The bug is NOT in ggml-cpu's MXFP4
implementation. The CPU backend's MUL_MAT_ID code path is correct. The bug is in SYCL's
dispatch wrapper silently skipping the op when a managed alloc fails.

**Companion site**: `ggml-sycl.cpp:36582-36587` has a similar `s_act_staging.ensure()` call
in the `dispatch_cpu_compute` lambda's shared-activation optimization. That path already
degrades gracefully (sets `act_on_host=false` → per-expert D2H branch at 36676-36690).
No change needed there.

## 3. Chosen Fix — Fallthrough to unfused dispatch

When `tl_first_act.ensure()` fails, instead of silent early return, fall through to the
unfused `ne12==1` hybrid dispatch path (line 36518+). The unfused path uses
`dispatch_cpu_compute` which takes a different alloc path (pool-backed via
`g_pinned_buffer_pools`).

**Implementation**: add a boolean `fusion_alloc_failed = false` flag at the top of the
ne12==1 block (around line 34231). Set it true when `ensure()` fails (erase fusion state
but don't return). The code then falls through; subsequent GPU-dispatch attempts will
also fail (no staging), eventually reaching the host-routing path (line 36016), then
`ne12==1` hybrid (line 36518), then `dispatch_cpu_compute` (line 36595). The pool-backed
alloc in `dispatch_cpu_compute` succeeds → activation D2H completes → MXFP4 vec_dot runs
correctly.

Alternative: `goto unfused_ne12_eq_1` label. Works but less clean than the flag.

## 4. Candidate Fixes Considered

### A. Fallthrough to unfused dispatch (CHOSEN)

Described above. Minimal control-flow change. No perf regression on success path. Fallback
only fires under arena starvation.

### B. Fall back to direct `sycl::malloc_host` in `ensure()`

Add a raw `sycl::malloc_host` fallback when the pool returns null. **Rejected**: wider
change, introduces allocations bypassing zone accounting, violates unified-cache invariants.

### C. Pre-size the managed staging pool

Reserve a dedicated tiny EXPERT_STAGING mini-zone at model-load time. **Rejected**: requires
knowing minimum required staging size at load time; papers over the underlying issue
(silent skip on alloc failure).

## 5. Files To Modify

- `ggml/src/ggml-sycl/ggml-sycl.cpp` — lines 34384-34390: replace early return with
  `fusion_alloc_failed = true; moe_fusion_erase(cur_layer_fast);`. Add `fusion_alloc_failed`
  flag at the top of the ne12==1 block (line 34231+). Ensure control flow actually reaches
  the unfused path when the flag is true.

3-10 line diff depending on how cleanly the existing control flow accepts fallthrough.

## 6. Gate Plan

| Gate | Criterion | Pass condition |
|------|-----------|----------------|
| 1 | Mistral canonical | output == `6, 7, 8, 9, 10` |
| 2 | Mistral perf | PP512 ≥ 1700, TG128 ≥ 81 |
| 3 | test-backend-ops MUL_MAT_ID MXFP4 | 0 failures (was 8) |
| 4 | 20B coherent -n 30 | no garble (stretch; may need l144i §8B follow-up) |
| 5 | 20B bench 3/3 | PP≥54, TG≥15 |

Gates 1-3 required for bead close. Gate 4 is a stretch goal.

## 7. Independence From Epic

This fix is complete and shippable regardless of whether
`docs/plans/2026-04-19-unified-memory-epic.md` is implemented. The unified-memory epic
reduces memory use by eliminating a redundant copy; this fix eliminates a silent
skip-without-compute that produces wrong output. They don't overlap in files or logic.

## 8. References

- docs/plans/2026-04-19-l144i-rootcause.md — diagnostic that narrowed n=1 failure
- docs/plans/2026-04-19-cache-expert-invariant-investigation.md — n04bq findings confirming
  SYCL side is consistent; bug is not in placement/dispatch gap
- Commit ceb8cbf95 (skgik), c2c31d7b6 (lj6p0) — CpuExpertPool + arena allocator context
