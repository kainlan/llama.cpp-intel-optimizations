# Plan: Unify MoE Multi-GPU Under the Unified Cache

## Progress

### Completed
- **Deadlock fix**: `get_effective_mode()` was calling `ggml_sycl_info().total_gpu_count` inside cache init (inside `ggml_sycl_init()` static guard → re-entry deadlock). Fixed with `g_total_gpu_count` global + `set_total_gpu_count()`.
- **Planner gate removed**: `compute_and_store_plan_for_inventory()` now activates when `total_gpu_count >= 2` without requiring `GGML_SYCL_MOE_MULTI_GPU`.
- **MoE auto-enable**: `g_moe_multi_gpu_active` auto-enables when 2+ GPUs visible (with env var override for opt-out).
- **Phase 1 DONE**: `g_secondary_queues[]` replaced with `init_shared_context_queues()` / `get_shared_context_queue()`. All 17 references replaced. `g_gpu1_queue` removed. Single-GPU verified PP512=1840, TG128=81.
- **Phase 2 DONE**: Eliminated greedy bin-packing in `moe_hybrid_init_once()`. Replaced with planner-driven expert upload using `g_placement_plan.expert_device[layer][expert] = device` map. The unified cache planner's multi-device plan now drives which GPU each expert uploads to. Removed ~200 lines of duplicate placement logic. Single-GPU verified PP512=1840, TG128=81.
- **Phase 3 DONE**: `g_moe_multi_gpu_active` in `moe_hybrid_init_once()` now uses `g_has_placement_plan && g_placement_plan.multi_device`. Forward declarations added.
- **Phase 6 DONE**: Removed 16 MoE-specific env vars that were replaced by the unified cache/planner:
  - `GGML_SYCL_PHASE2_MAX_UPLOAD`, `GGML_SYCL_PHASE2_TIMEOUT_SEC`, `GGML_SYCL_B50_EXPERT_SLOTS` (Phase 2 eliminated)
  - `GGML_SYCL_B50_LOCAL_AGG` (always enabled), `GGML_SYCL_PIPELINE_MOE` (always disabled)
  - `GGML_SYCL_MOE_MULTI_GPU` (planner auto-detects), `GGML_SYCL_MOE_HYBRID` (always enabled, planner overrides), `GGML_SYCL_MOE_GPU_PROBE` (auto when profiling)
  - `GGML_SYCL_EXPERT_PREFETCH_DEPTH`, `GGML_SYCL_EXPERT_PREDICT`, `GGML_SYCL_EXPERT_PREDICT_DEPTH` (cache manages)
  - `GGML_SYCL_EXPERT_MISS_PRECISION`, `GGML_SYCL_EXPERT_MISS_BURST_THRESHOLD` (hardcoded MIXED/3)
  - `GGML_SYCL_EXPERT_RESERVE_MULT` (hardcoded 2x), `GGML_SYCL_EXPERT_DEFER` (always enabled), `GGML_SYCL_MOE_DEFER_COUNT` (hardcoded 1)
  - Kept 12+ tuning/debugging env vars (MOE_WARMUP_TOKENS, MOE_STATS, MOE_PROFILE, XMX_MOE, etc.)
  - Single-GPU verified PP512=1840, TG128=81.

### Remaining (requires multi-GPU testing)
- Phase 4: Replace `expert_ptrs_host` / `update_moe_ptr_table` with `mem_handle` lookups (HIGH risk — touches hot dispatch path)
- Phase 5: Consolidate `g_expert_prefetchers[]` into unified cache prefetch (MEDIUM risk)
- End-to-end multi-GPU validation (blocked — no multi-GPU test runs allowed)


## Problem Statement

The SYCL backend has ~35 MoE-specific env vars, 148+ MoE-specific global state references,
and a parallel MoE dispatch infrastructure (`g_secondary_queues`, `g_moe_multi_gpu_active`,
Phase 1/Phase 2 bin-packing, `expert_ptrs_host`, etc.) that exists OUTSIDE the unified cache.

This violates the unified cache's design philosophy:
- **Unified cache owns all memory** — but MoE Phase 2 uploads bypass it with raw `sycl::malloc_device`
- **All allocations flow through the cache** — but `g_secondary_queues` creates separate queues
- **Everything uses `mem_handle`** — but expert dispatch uses raw `void*` device pointers
- **Intelligent device/host placement** — but MoE has its own bin-packing that ignores the planner

The result: DEVICE_LOST errors on multi-GPU, because the MoE dispatch path doesn't go through
the unified cache's device-aware allocation and copy infrastructure.

## Current Architecture (The Mess)

### MoE-Specific Globals (in ggml-sycl.cpp)
```
g_secondary_queues[GGML_SYCL_MAX_DEVICES]  -- separate SYCL queues per GPU
g_gpu1_queue                                -- alias for g_secondary_queues[1]
g_moe_multi_gpu_active                     -- flag: is MoE multi-GPU enabled?
g_moe_n_experts_total                      -- expert count
g_moe_n_experts_used                       -- used expert count
g_moe_expert_total_bytes                   -- total expert weight bytes
g_expert_prefetchers[GGML_SYCL_MAX_DEVICES] -- per-device prefetch workers
g_moe_hybrid_init_flags[GGML_SYCL_MAX_DEVICES] -- std::call_once guards
```

### MoE-Specific Code Paths (17 branches on g_moe_multi_gpu_active)
1. **moe_hybrid_init_once()** (~800 lines): Separate init for MoE, creates secondary queues,
   Phase 1/Phase 2 expert upload, warmup, SOA reordering
2. **dispatch_experts_secondary_gpu_impl()**: Uses `g_secondary_queues[target_device]` directly
3. **Expert weight upload**: `direct_stage_expert()` per-device, bypassing planner
4. **CPU-TG expert fallback**: Checks `g_moe_multi_gpu_active` to decide CPU vs GPU
5. **MMVQ/fast-path selection**: Branches on multi-GPU for kernel selection

### 35 MoE-Specific Env Vars
```
GGML_SYCL_MOE_MULTI_GPU       -- enable/disable (now auto, but env override exists)
GGML_SYCL_B50_EXPERT_SLOTS    -- cap secondary GPU expert cache
GGML_SYCL_B50_LOCAL_AGG       -- local aggregation mode
GGML_SYCL_PHASE2_MAX_UPLOAD   -- cap Phase 2 uploads
GGML_SYCL_PHASE2_TIMEOUT_SEC  -- Phase 2 timeout
GGML_SYCL_EXPERT_PREFETCH_DEPTH  -- prefetch depth
GGML_SYCL_EXPERT_PREDICT_DEPTH   -- prediction depth
GGML_SYCL_EXPERT_PREDICT         -- enable prediction
GGML_SYCL_MOE_STATS              -- enable stats
GGML_SYCL_MOE_STATS_INTERVAL     -- stats interval
GGML_SYCL_MOE_WARMUP_TOKENS      -- warmup token count
GGML_SYCL_MOE_RERANK_INTERVAL    -- reranking interval
GGML_SYCL_MOE_PERSISTENT         -- persistent MoE kernel
GGML_SYCL_MOE_HYBRID             -- hybrid dispatch mode
GGML_SYCL_MOE_FUSE               -- kernel fusion
GGML_SYCL_MOE_DEBUG              -- debug mode
GGML_SYCL_MOE_DEFER_COUNT        -- deferred expert count
GGML_SYCL_MOE_GPU_PROBE          -- GPU probing
GGML_SYCL_MOE_PROFILE / _PP_PROFILE  -- profiling
GGML_SYCL_MOE_PROB_THRESHOLD     -- probability threshold
GGML_SYCL_XMX_MOE / _TILED / _FUSED  -- XMX kernel variants
GGML_SYCL_CPU_EXPERT_TG / _THREADS   -- CPU expert dispatch
GGML_SYCL_EXPERT_DEFER            -- deferred expert execution
GGML_SYCL_EXPERT_MISS_PRECISION   -- cache miss precision
GGML_SYCL_EXPERT_MISS_BURST_THRESHOLD -- burst threshold
GGML_SYCL_EXPERT_RESERVE_MULT     -- reserve multiplier
GGML_SYCL_MAX_BLIND_PRELOAD_EXPERTS -- blind preload cap
GGML_SYCL_FUSED_MOE_MAX_BATCH     -- fused MoE batch size
GGML_SYCL_PIPELINE_MOE            -- pipeline MoE
GGML_SYCL_BATCH_EXPERTS           -- batched expert launch
GGML_SYCL_DISABLE_FUSED_MOE       -- disable fused MoE
GGML_SYCL_MMVQ_MOE_PP             -- MMVQ prompt processing
```

## Target Architecture (Unified Cache Owns Everything)

### Design Principles
1. **Unified cache is the ONLY allocator** — no raw `sycl::malloc_device` for weights
2. **mem_handle is the ONLY pointer type** — no raw `void*` device pointers in dispatch
3. **Planner drives ALL placement** — no separate Phase 2 bin-packing
4. **Cache handles ALL device transfers** — copy_to_device_async with proper cross-device events
5. **Expert residency is a cache query** — `is_expert_resident(device, layer, expert)` replaces pointer tables
6. **No MoE-specific multi-GPU flag** — the planner's `multi_device` flag drives everything

### Key Refactoring Steps

#### Phase 1: Shared-Context Queue Pool (replaces g_secondary_queues)

**CRITICAL CONSTRAINT**: On Level Zero, cross-device `depends_on()` events REQUIRE matching
`ze_context_handle_t`. Device default queues (from `dpct::dev_mgr`) each have their own
context. `g_secondary_queues` were explicitly created with device 0's shared context
(line 3886: `dev0.default_queue().get_context()`). Same pattern used for `g_pipeline_copy_queue`
(line 8753: `dev0.default_queue().get_context()`).

`ctx.stream(d, 0)` is **NOT SAFE** for cross-device dispatch — it returns the device's
default queue which has its OWN context, causing DEVICE_LOST on Level Zero cross-device ops.

**Approach**: Create a `shared_context_queue_pool` in the unified cache that manages
per-device queues with device 0's shared SYCL context:

```cpp
// In unified-cache.hpp:
class shared_context_queue_pool {
    sycl::queue * queues[GGML_SYCL_MAX_DEVICES] = {};
    sycl::context shared_ctx;  // device 0's context
public:
    void init();  // creates queues for all visible GPUs with shared_ctx
    sycl::queue * get(int device);  // returns queue or nullptr
    bool is_initialized() const;
};
```

- Move queue creation from `moe_hybrid_init_once()` and `compute_placement_plan_early()`
  into the pool's `init()`
- All cross-device dispatch uses `pool.get(target_device)` instead of `g_secondary_queues[d]`
- `g_secondary_queues[]` and `g_gpu1_queue` globals are removed
- The pool is owned by the unified cache (not a global), so it's available to all dispatch code

#### Phase 2: Eliminate Phase 2 Bin-Packing — Use Planner's expert_device Map
- Remove the Phase 2 expert upload code from `moe_hybrid_init_once()`
- Instead, the planner's `compute_multi_device_plan()` already populates `expert_device[layer][expert] = device`
- The unified cache's `direct_stage_expert()` already handles per-device staging
- At weight upload time, consult `g_placement_plan.expert_device` to decide which device each expert goes to
- The cache handles the actual device pointer resolution — no manual `sycl::malloc_device`

#### Phase 3: Replace g_moe_multi_gpu_active with Planner Queries
- Remove `g_moe_multi_gpu_active` global
- Replace all 17 branch points with `g_has_placement_plan && g_placement_plan.multi_device`
- This unifies MoE and non-MoE multi-GPU: the planner decides, not a MoE-specific flag

#### Phase 4: Eliminate Expert Pointer Tables — Use mem_handle
- Replace `expert_ptrs_host` and `update_moe_ptr_table()` with `mem_handle`-based lookups
- Each expert weight gets a `mem_handle` from the unified cache
- `mem_handle.resolve(device)` returns the device-local pointer, handling cross-device copies automatically
- This eliminates raw `void*` device pointers and the manual pointer synchronization

#### Phase 5: Consolidate Expert Prefetching into Cache Prefetch
- Remove `g_expert_prefetchers[]` — separate per-device prefetch workers
- Replace with unified cache's existing prefetch infrastructure
- The cache already knows which device each weight is on and can prefetch accordingly

#### Phase 6: Consolidate Env Vars
- Keep only the genuinely useful tuning knobs (MOE_WARMUP_TOKENS, MOE_PROB_THRESHOLD, MOE_STATS)
- Remove all env vars that exist only because of the parallel MoE infrastructure:
  - GGML_SYCL_MOE_MULTI_GPU (auto-detected now)
  - GGML_SYCL_B50_EXPERT_SLOTS (cache budget handles this)
  - GGML_SYCL_PHASE2_MAX_UPLOAD (no more Phase 2)
  - GGML_SYCL_PHASE2_TIMEOUT_SEC (no more Phase 2)
  - GGML_SYCL_EXPERT_PREFETCH_DEPTH (cache prefetch handles this)
  - GGML_SYCL_EXPERT_PREDICT / _DEPTH (cache prediction)
  - GGML_SYCL_CPU_EXPERT_TG / _THREADS (cache handles CPU offload)

### Migration Strategy
Each phase is independently deployable:
- Phase 1 can be done first without changing expert dispatch behavior
- Phase 2 can follow once Phase 1 ensures queues work
- Phase 3 removes the flag but keeps the same dispatch paths
- Phase 4 is the big refactor — pointer tables to mem_handle
- Phase 5 and 6 are cleanup

### What Changes for Multi-GPU
After all phases:
- Model loads → planner computes multi-device plan automatically
- Expert weights → cache stages them on the right device (per planner's expert_device map)
- Expert dispatch → uses `ctx.stream(target_device, 0)` for cross-device compute
- Pointer resolution → `mem_handle.resolve(device)` handles device locality
- No env vars needed for basic multi-GPU — it just works

### Level Zero Context Constraint (Critical)

On Intel Arc GPUs with Level Zero, cross-device SYCL operations (events, `depends_on()`,
USM transfers between devices) REQUIRE that both queues share the same `ze_context_handle_t`.
Each device's default queue (from `dpct::dev_mgr`) gets its own context. The `g_secondary_queues`
were explicitly created with device 0's context to satisfy this requirement.

Evidence from the codebase:
- Line 3883-3885: "All secondary queues MUST share device 0's sycl::context so that
  cross-device depends_on() works on Level Zero."
- Line 8751-8752: "Create in-order copy queue with device 0's context so cross-device
  depends_on() works on Level Zero (events require matching ze_context)."
- Line 33921-33922: Cross-device dispatch captures both contexts but doesn't validate match.

Any replacement for `g_secondary_queues` MUST create queues with device 0's shared context.
Using device default queues will cause DEVICE_LOST or silent corruption.

### Risk Assessment
- Phase 1 (queue replacement): LOW RISK — ctx.stream() already works for single-device
- Phase 2 (eliminate Phase 2): MEDIUM RISK — changes weight upload path
- Phase 3 (remove flag): LOW RISK — mechanical replacement
- Phase 4 (mem_handle): HIGH RISK — touches hot path, all expert dispatch
- Phase 5 (prefetch consolidation): MEDIUM RISK — affects performance tuning
- Phase 6 (env var cleanup): LOW RISK — removal only, no behavior change

## Immediate Next Step
Phase 1: Replace `g_secondary_queues[d]` with `ctx.stream(d, 0)` in the expert dispatch path.
This is the prerequisite for multi-GPU compute to work — currently DEVICE_LOST happens because
`g_secondary_queues[d]` queues don't have proper execution context for the backend scheduler.