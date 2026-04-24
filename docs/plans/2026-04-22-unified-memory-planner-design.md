# Unified Memory Placement Planner — Design

**Date:** 2026-04-22
**Status:** Design (approved, awaiting implementation plan)
**Branch:** `feature/sycl-coalescing`
**Scope:** SYCL backend only (fork-local; upstream compatibility is not a constraint)

---

## Motivation

The SYCL backend currently has three distinct planning surfaces:

1. **Weights**: `placement_plan` + priority bin-packer (at model load).
2. **KV cache**: `placement_kv_info` + per-layer SWA bin-packer (at context creation).
3. **Compute buffers, staging pools, oneDNN scratchpad, per-op dispatch**: not planned — sized from hardcoded defaults, dispatched via ad-hoc switch statements that infer placement from raw buffer pointers or the weight plan.

This split-planning approach produced a repeating bug pattern: the weight plan says one thing, runtime dispatch infers something different from pointer values, the two disagree, and execution fails or corrupts data. The most recent example is the GPT-OSS 20B `attn_sinks` case — the weight plan correctly placed `attn_sinks` on host-pinned memory for budget-constrained layers, but `ggml_sycl_op_is_planned_on_host` applied the weight plan signal to `FLASH_ATTN_EXT` (which shouldn't be governed by the weight plan since FA has weight sources only in some models), causing FA to be forced to CPU, causing a device mismatch with the KV cache, causing global FA disable, causing the attention score matrix to materialize at 16 GB.

The fix to that single bug added a targeted per-op staging guard. That pattern is not scalable — every future bug of this shape needs its own guard, and the surface area grows faster than we can audit it.

## Principle

**The plan is the source of truth for every byte of memory the SYCL backend touches.** No inference-time allocation is made without consulting it. No op dispatch is made without consulting it. Nothing is excluded.

## Planning phases

### Phase 0 — auto-detection (once per process)

Before any planning begins:

- Enumerate SYCL devices, query VRAM totals, compute throughput, Level Zero per-allocation caps.
- Query host RAM, TTM pinned-pool cap, CPU topology.
- Read the model's default `n_ctx` from GGUF; parse user's cmdline (`-c`, `-ub`, `-b`, `GGML_SYCL_VRAM_BUDGET_PCT`, `GGML_SYCL_SPLIT_RATIO`, `GGML_SYCL_MULTI_GPU_MODE`, `GGML_SYCL_VISIBLE_DEVICES`).
- Resolve final values: user override wins, model default otherwise, with validation (refuse `ctx > model_max`).

The resolved values are the planner's inputs. Env vars become overrides on top of auto-detected defaults, not primary inputs.

### Phase 1 — weight plan at model load (shared across contexts)

Called from `llama_model_load(path, load_params)` where `load_params` now carries `ctx`, `ubatch`, `batch`, `budget`, device topology.

1. Open a **throwaway mini context** with the resolved ctx/ubatch/batch.
2. Build the real graph via `llama_build_graph`, run `ggml_backend_sched_reserve(no_alloc=true)` per backend.
3. This yields **ground-truth** per-backend compute-buffer demand. No estimation.
4. Planner reserves compute-buffer capacity per backend in their VRAM arenas **first** (compute wins over weights when VRAM is tight, since a forward pass must fit).
5. Planner reserves KV arena capacity per the `placement_kv_info` math for the declared ctx.
6. Planner places weights in remaining VRAM by priority (`NORM_EMBED` > `ATTENTION` > `FFN` > `MOE_DOWN` > `MOE_UP` > `MOE_GATE_PROJ`). Overflow spills to host-pinned arena.
7. Planner walks the graph once more and emits a per-op dispatch table: `op → { backend, input staging requirements }`. **Every op must have an entry. Missing entry → hard assert at planner emit time.**
8. Destroys the throwaway mini context.
9. Plan is installed into `unified_cache` as the source of truth.

Weight load then materializes tensors directly at planned locations — no preload staging copy. If the source is mmap, copy mmap → arena; otherwise, allocate directly in the arena.

### Phase 2 — compute plan at each context creation (per-context)

Each real `llama_context` instance generates its own compute plan against the shared weight plan:

1. Weight plan is already installed; weight arena is already populated.
2. Build the actual graph for this context's ctx/ubatch.
3. Walk the graph and validate every op against the dispatch table emitted in Phase 1.
4. Materialize KV arena per the weight plan's KV reservation for this context.
5. Materialize compute pools per the weight plan's compute reservation, one pool per `{device, tier, purpose}`.
6. Materialize staging pools per op-dispatch requirements.

Concurrent contexts share the weight arenas (read-only) but each get their own KV + compute + staging.

## Enforcement

- `unified_cache::allocate(name, ...)` refuses allocations that don't match the plan's tier/device for plan-tracked tensors.
- `ggml_sycl_resolve_tensor_ptr` consults the plan, not inference-time heuristics on buffer types or pointer ranges.
- Op dispatch (`ggml_backend_sycl_device_supports_op`, `ggml_sycl_op_is_planned_on_host`, `dispatch_cpu_compute`, etc.) consults `plan.dispatch_for(op)` rather than inferring from buffer-type or pointer-value.
- Ephemeral tensors sub-allocate from plan-sized pools; their placement is the pool's placement.
- No "best-effort" path. No fallback. Runtime asserts if the plan doesn't answer.

## Failure mode

- If the plan can't fit (weights + KV + compute exceed VRAM + host-pinned), `llama_model_load` fails with an explicit error:

  > *Plan infeasible: need X MB, have Y MB VRAM + Z MB host-pinned. Reduce ctx (current N) or raise `GGML_SYCL_VRAM_BUDGET_PCT` (current M%).*

- Never auto-shrink ctx. Never partial-load.
- If the user doesn't pass `-c` and the model default is infeasible (e.g., GPT-OSS 20B's 131072), fail at load with the same error — user learns to pass `-c`.

## Multi-device policy

- Default: planner-autonomous.
- Weight split: proportional to VRAM + compute throughput per device.
- MoE experts: split by priority × layer stride (existing heuristic, now encoded in the planner).
- Overrides: env vars (`GGML_SYCL_SPLIT_RATIO`, `GGML_SYCL_MULTI_GPU_MODE`, `GGML_SYCL_VISIBLE_DEVICES`) inject before auto-detection. If set, they win.

## Mutability

- Weight plan is immutable for the model's lifetime.
- Compute plan is immutable for the context's lifetime.
- Evicting a plan-placed weight under pressure is **forbidden** — the plan already honored the budget. If pressure exists post-load, it is a bug elsewhere (e.g., a non-planned allocation leaked through).
- Concurrent contexts each have their own compute plan; they share the weight plan and the weight arenas (read-only).

## Components

### `planner.hpp/cpp` — data structures + generator

```cpp
namespace ggml_sycl {

struct plan_tensor_entry {
    std::string name;
    size_t      size_bytes;
    int         target_device;   // -1 = host
    tier        target_tier;     // VRAM, HOST_PINNED, MMAP
    zone_id     target_zone;     // KV, WEIGHT, RUNTIME, SCRATCH, ONEDNN
    size_t      target_offset;   // within zone
    layout_mode layout;          // AOS, SOA, COALESCED, XMX_TILED
    placement_priority priority;
};

struct plan_pool_entry {
    std::string name;
    int         device;
    tier        target_tier;
    size_t      max_bytes;
    std::vector<std::string> tenant_tensor_names;  // who lives here
};

struct plan_op_entry {
    int         op_id;          // index into graph node array
    std::string op_name;
    ggml_op     op_type;
    int         execution_device;
    std::vector<input_staging_directive> staging;  // per-src staging if needed
};

class memory_plan {
  public:
    // Queries
    const plan_tensor_entry * lookup_tensor(const std::string & name) const;
    const plan_pool_entry   * lookup_pool(const std::string & name) const;
    const plan_op_entry     * lookup_op(int op_id) const;

    // Enforcement helpers
    int  execution_device_for(const ggml_tensor * op) const;  // asserts if missing
    tier tier_for_tensor(const std::string & name) const;

    // Diagnostics
    void dump(std::ostream &) const;
    size_t estimated_vram_per_device(int device) const;
    size_t estimated_host_pinned() const;

  private:
    std::vector<plan_tensor_entry> tensors_;
    std::vector<plan_pool_entry>   pools_;
    std::vector<plan_op_entry>     ops_;
    // ... indexing maps
};

class planner {
  public:
    struct inputs {
        const llama_model * model;
        uint32_t            ctx, ubatch, batch;
        size_t              vram_budget_pct;
        device_topology     topology;
        // ... resolved env-var overrides
    };

    std::unique_ptr<memory_plan> generate(const inputs &);
};

}  // namespace ggml_sycl
```

### Hooks

- `llama_model_load(path, load_params)` — accept `ctx/ubatch/batch/budget/topology`.
- `llama_model_load` installs the plan into `unified_cache` via a new call like `unified_cache_install_plan(std::unique_ptr<memory_plan>)`.
- `llama_context` constructor materializes KV + compute + staging pools per the plan, refuses to proceed if any pool is missing from the plan.

### Throwaway mini-context utility

A small helper that:
- Constructs a minimal `llama_context` with target params (no KV, no real buffers).
- Runs `llama_build_graph` with a synthetic batch.
- Runs `ggml_backend_sched_reserve(no_alloc=true)` for each backend.
- Returns the discovered compute demand per backend + the full op graph.
- Destroys itself cleanly (every resource owned by the throwaway is freed before returning).

### Consumer migration

All of these sites currently infer placement or dispatch from non-plan sources. Each must be rewritten to consult `plan.*`:
- `ggml_backend_sycl_device_supports_op`
- `ggml_sycl_op_is_planned_on_host`
- `ggml_sycl_layer_plan_applies_to_op` (likely removed entirely — replaced by plan.dispatch_for(op))
- `depends_on_planned_host_weight` (removed — plan covers every op)
- `dispatch_cpu_compute` (now reads plan.staging for input placement)
- `ggml_sycl_resolve_tensor_ptr` (now reads plan.lookup_tensor)
- All direct `sycl::malloc_device/host/shared` sites → route through `unified_cache`
- `g_tl_fattn_weight_stage` (15li2 per-op FA staging) — removed; replaced by plan-driven staging

### Plan dump utility

- Text dump: tensors by zone with offsets, pools with tenants, ops with dispatch assignments.
- Diff two plans (useful for regression detection across model-load runs).

## Relationship to existing beads

This epic **implements** `llama.cpp-mubmt` (EPIC: Make unified_cache the sole owner of SYCL memory and relocatable residency handles).

Subsumes:
- `llama.cpp-8gz7y` (query scheduler for compute buffer sizes) — the throwaway mini-context approach supersedes this.
- `llama.cpp-tyoc2` (host compute buffer through SCRATCH zone) — handled by the compute-pool materialization in Phase 2.
- `llama.cpp-01mcl` (remove `must_device=true`) — handled by the unified allocation path.
- `llama.cpp-15li2` (FA `attn_sinks` per-op staging) — the per-op staging hack is removed once dispatch is plan-driven.

Stepping stones (already landed, retained):
- `llama.cpp-w1rxh` (arena-aware `get_max_size`) — structural cleanup that works either way; kept.
- `llama.cpp-ioua6` (budget=30 load failure) — resolved by w1rxh + 15li2, would be trivially correct under the new plan.

## Out of scope

- CUDA, Vulkan, Metal backends. Fork-local for SYCL only.
- Upstream merge compatibility.
- llama.cpp's ggml-alloc internals. We constrain ggml-alloc's behavior via pool sizing + device binding; we don't modify ggml-alloc.
- Runtime plan mutation (e.g., adaptive eviction). Plan is immutable once made.
- Non-SYCL backends' reaction to the changed `llama_model_load` API. If we break CUDA/Vulkan/Metal, we fix them in the same patch or add a compat shim.

## Success criteria

1. GPT-OSS 20B at `VRAM_BUDGET_PCT=30` with default ctx loads and runs.
2. Mistral 7B Q4_0 regression: PP512 ≥ 1700, TG128 ≥ 80, canonical completion correct.
3. `grep -c 'sycl::malloc_device\|sycl::malloc_host\|sycl::malloc_shared' ggml/src/ggml-sycl/*.cpp` → zero (all routed through unified_cache).
4. `grep -c 'layer_plan_applies_to_op\|depends_on_planned_host_weight\|g_tl_fattn_weight_stage' ggml/src/ggml-sycl/*.cpp` → zero (heuristics replaced by plan queries).
5. `plan.dump()` output for Mistral 7B and GPT-OSS 20B committed as reference artifacts in `docs/plans/data/` — golden diff tests verify deterministic plan generation.
6. All existing models/tests that pass today still pass.
