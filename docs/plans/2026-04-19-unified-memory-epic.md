# [EPIC] Unified memory: single allocation for host-planned MoE expert weights

**Status**: PLAN
**Branch**: `feature/sycl-coalescing`
**Priority**: P1

## 1. Key Insight (from Phase A exploration)

`tensor->data` for `moe_expert_host_pinned=true` tensors **already** points into the SYCL
unified cache's host-pinned WEIGHT zone. The backend-buffer wiring is correct:

- `src/llama-model.cpp:3583-3589` — routes `moe_expert_host_pinned` tensors to
  `ggml_backend_sycl_host_buffer_type()`
- `ggml/src/ggml-sycl/ggml-sycl.cpp:17999-18052` — this buffer type's `alloc_buffer`
  goes through `ggml_sycl::unified_alloc` with `role=WEIGHT, use_pinned_pool=true`
- `tensor->data` is a flat allocation inside the unified cache host zone

The **split-brain** is that `ggml_sycl_preload_model_weights` at
`ggml-sycl.cpp:11681-11700` and `12017-12021` makes a REDUNDANT SECOND allocation from the
same zone and `memcpy`s each per-expert slice into it. That copy is then what
`register_host_expert(key, ptr)` records. SYCL dispatch reads the copy via
`lookup_expert(key)`; CPU backend reads the original via `tensor->data`. Same data, two
allocations. Epic removes the copy.

## 2. Target State

S1-PRELOAD calls `register_host_expert(key, tensor->data + e*nb02, size, GGML_LAYOUT_AOS)`
directly — no second allocation, no `memcpy`. CPU backend reads `tensor->data` as before
(unchanged). SYCL dispatch reads `lookup_expert(key)` which now returns the same memory.
Eviction / refcount / chunk-lease discipline from `0k543` covers both implicitly.

## 3. Current State — File-Level Map

| File:line | Role | Status |
|-----------|------|--------|
| `src/llama-model.cpp:3583-3589` | `moe_expert_host_pinned=true` → `ggml_backend_sycl_host_buffer_type()` | Correct — routes to unified cache host WEIGHT zone |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:17999-18052` | `alloc_buffer` via `unified_alloc(role=WEIGHT, use_pinned_pool=true)` | Correct — one contiguous allocation per tensor, in unified cache |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:11673` | `expert_aos = src0->data + e * expert_size` | Correct pointer into the existing unified-cache allocation |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:11681-11700` | `alloc _host_req2` + `memcpy(host_ptr, expert_aos)` + `register_host_expert(key, host_ptr)` | **REDUNDANT** — second allocation from same zone |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:12017-12021` | `memcpy(arena_ptr, expert_aos)` + `register_host_expert(key, arena_ptr)` in `all_weights_host` branch | **REDUNDANT** — same issue, different branch |
| `ggml/src/ggml-cpu/ggml-cpu.c:1620` | `src0_cur = src0->data + cur_a * nb02` | Unchanged after epic — already correct |

## 4. Non-Goals

- CUDA, Vulkan, Metal unification (scope is SYCL+CPU only)
- Changing the CPU backend in any way
- Changing `ggml_backend_sycl_host_buffer_type`'s public API
- Changing how dense (non-MoE) weights are loaded
- Device-side (VRAM) weight paths: SOA/COALESCED staging unchanged

## 5. Transition Plan

### Phase 1 — Audit: verify `unified_alloc` routes through `pinned_chunk_pool`

**AUDIT COMPLETE (2026-04-19, bead llama.cpp-devce)**

#### Correction to premise

The epic's §3 table and §5 heading both claim the caller always uses
`use_pinned_pool=true`. This is incorrect. At
`ggml-sycl.cpp:18018`:

```cpp
req.intent.constraints.use_pinned_pool = (sycl_cache && sycl_cache->host_zones_configured());
```

`use_pinned_pool` is `true` only when host zones are already configured. During early
model-load (before `configure_zones` runs), it is `false`.

#### Exact call path (file:symbol:line)

```
ggml-sycl.cpp:18013    must_host_pinned = true
ggml-sycl.cpp:18018    use_pinned_pool  = host_zones_configured()   ← conditional
ggml-sycl.cpp:18020    unified_alloc(req, &alloc)
  unified-cache.cpp:5542  tier forced to HOST_PINNED (must_host_pinned=true)
  unified-cache.cpp:5657  enters HOST_PINNED block
  unified-cache.cpp:5659  get_unified_cache_for_device() → ucache
  unified-cache.cpp:5692  try_zone_alloc_contiguous lambda defined
    unified-cache.cpp:5693    if (!zones_configured && use_pinned_pool) → return nullptr
    unified-cache.cpp:5698    ucache->host_zone_alloc(zone, …)   [WEIGHT zone]
      unified-cache.cpp:8008   host_arena_->zone_alloc(zone, …)
        pinned-pool.cpp          TLSF sub-allocator inside pinned_chunk_pool
  unified-cache.cpp:5721  if (use_pinned_pool) → try_zone_alloc_contiguous(WEIGHT)
  unified-cache.cpp:5730  else if (zones_configured) → try_zone_alloc_contiguous(zone)
  unified-cache.cpp:5742  else (zones NOT configured, !use_pinned_pool):
                               ucache->host_pool_alloc()
                                 → host_arena_->allocate_runtime()
                                 → pinned_chunk_pool::allocate_runtime()
  unified-cache.cpp:5746  if (!ptr) fallback:
    unified-cache.cpp:5753  role==WEIGHT only → ggml_sycl_malloc_host() ← DIRECT sycl::malloc_host
    unified-cache.cpp:5756  else → GGML_LOG_ERROR + return false
```

`unified_cache::host_arena_` is always a `pinned_chunk_pool`
(unified-cache.cpp:1284: `host_arena_ = std::make_unique<pinned_chunk_pool>(...)`).

#### YES/NO: does this path always route through `pinned_chunk_pool`?

**NO — one fallback path bypasses `pinned_chunk_pool`.**

There are four sub-paths through the HOST_PINNED block:

| Sub-path | Condition | Backing allocator |
|----------|-----------|-------------------|
| A | `use_pinned_pool=true` + zones configured → zone_alloc succeeds | `pinned_chunk_pool` WEIGHT TLSF zone |
| B | `use_pinned_pool=false` + zones configured → zone_alloc succeeds | `pinned_chunk_pool` zone TLSF (same pool) |
| C | `use_pinned_pool=false` + zones NOT configured → `host_pool_alloc` succeeds | `pinned_chunk_pool::allocate_runtime()` (same pool) |
| D | Any of A/B/C return `nullptr` AND `role==WEIGHT` | Direct `sycl::malloc_host` — **outside `pinned_chunk_pool`** |

Sub-path D triggers when:
- `use_pinned_pool=true` but zones are not yet configured (line 5693-5696 returns `nullptr`
  immediately from `try_zone_alloc_contiguous`), **AND** the `host_pool_alloc` fallback is
  also skipped because `use_pinned_pool=true` bypassed the `else` branch at 5738 —
  so D fires immediately after A's zone miss with no runtime-pool retry.
- OR zone is configured but exhausted AND `grow_zone` fails (budget or phase gate).
- The non-WEIGHT role path (non-D, role!=WEIGHT) fails hard (`return false`) rather than
  falling through to `malloc_host`.

In practice for model-load time (early load, zones not yet configured):
`use_pinned_pool=false` → path C → `allocate_runtime` in `pinned_chunk_pool`. Path D is
the rare exhaustion/budget fallback.

#### Implication for Phase 4 lease strategy

Because path D can produce a pointer **not** inside any `pinned_chunk_pool` chunk,
`mem_handle::from_chunk_ptr(expert_aos)` will return kind=DIRECT for those allocations
(the `find_chunk_handle` lookup misses, `acquire_chunk_lease` returns
`INVALID_CHUNK_HANDLE`).

DIRECT kind is still correct for lifetime (the buffer's `ggml_backend_buffer_t` keeps the
memory alive for the full model lifetime), but the chunk-refcount that guards async tasks
will be zero, which defeats the defense-in-depth goal of dyhdl.

**Recommended Phase 4 lease strategy: union of both.**

In `register_host_expert` (after Phase 2/3 eliminate the copy):

1. Call `from_chunk_ptr(ptr)` → if kind == `CHUNK_LEASE`, store the chunk handle in the
   weight_entry and release it at unregister/eviction. (Covers paths A/B/C.)
2. If kind == `DIRECT` (path D / rare `malloc_host` fallback): acquire an entry-level
   `acquire_weight_lease(key)` instead. This is the correct approach for memory not owned
   by `pinned_chunk_pool`.

This union approach handles both pool-backed and direct allocations correctly without
requiring a separate code path. The `from_chunk_ptr` → DIRECT fallback is not an error
condition; it is the designed-in graceful path when `pinned_chunk_pool` is bypassed.

**Acceptance**: CLOSED — audit paragraph added, Phase 4 lease strategy updated.


### Phase 2 — Eliminate redundant copy in placement-plan host branch

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp:11681-11700`

Replace:
```cpp
ggml_sycl::alloc_request _host_req2{};
_host_req2.device = device;
_host_req2.size   = expert_size;
_host_req2.intent.role = ggml_sycl::alloc_role::WEIGHT;
_host_req2.intent.constraints.must_host_pinned = true;
_host_req2.intent.constraints.use_pinned_pool  = true;
void * host_ptr = ggml_sycl::unified_allocate(_host_req2).resolve().ptr;
if (!host_ptr) { failed++; continue; }
std::memcpy(host_ptr, expert_aos, expert_size);
cache->register_host_expert(key, host_ptr, expert_size, GGML_LAYOUT_AOS);
```

With:
```cpp
// expert_aos already lives in the unified cache host-pinned WEIGHT zone
// (allocated by ggml_backend_sycl_host_buffer_type_alloc_buffer at model load).
// Register in-place — no copy needed. src0->buffer owns the backing memory for
// its lifetime.
cache->register_host_expert(key,
                             const_cast<void *>(static_cast<const void *>(expert_aos)),
                             expert_size, GGML_LAYOUT_AOS);
host_registered++;
```

Optional: gate behind `GGML_SYCL_UNIFIED_MEM_PHASE2=1` env var for first deployment; remove
after one clean gate cycle.

**Gate**: Mistral gates 1-2 unchanged; 20B bench gate 3 unchanged; host-pinned memory
footprint for MoE expert weights reduced by ~50%.

### Phase 3 — Eliminate the `all_weights_host` branch entirely (not just its memcpy)

**Architectural principle** (per user, reinforcing commit `445028753` "remove
feature-blocking MoE placement booleans"): when the unified cache is the source of truth,
code should NOT need branching booleans that duplicate placement-plan state. `all_weights_host`
is one of those booleans — a secondary state variable derivable from
`cache->has_placement_plan()` + per-tensor plan lookup.

**Current state**: `ggml-sycl.cpp:12017-12021` lives inside an `if (all_weights_host)` block
(or similar). The boolean is set elsewhere (21 occurrences of `all_weights_host` in
`ggml-sycl.cpp`). The whole branch duplicates the placement-plan branch but with different
allocator wiring.

**Fix**: unify the two expert-registration paths into ONE loop that, per expert:
1. Queries the unified cache / placement plan for this expert's planned location
2. If host-planned: `register_host_expert(key, expert_aos, size, layout)` in-place
3. If device-planned: existing device-promotion path
4. No `all_weights_host` boolean — the decision is per-expert, not a global flag

The refactor eliminates the branch duplication AND the redundant memcpy at 12017-12021 in one
move. Includes removing every use-site of `all_weights_host` (21 sites) that can be derived
from the per-expert cache/plan query.

**Gate**: Mistral 1-2; 20B coherent -n 30 (stretch). No regression in any offload mode
(`GGML_SYCL_VRAM_BUDGET_PCT=30` Mistral stays functional).

**Non-goal for this phase**: other legacy booleans (`host_weights_fast`, `cpu_expert_tg_active`)
stay for now — they can be folded into a follow-up phase once Phase 3 pattern is validated.

### Phase 4 — Lease discipline at every pointer-assignment / pointer-free point

**Architectural principle** (per user): SYCL backend owns the allocations AND the refcount;
other backends get lifetime protection implicitly by the refcount held while the memory is
"in use" (assigned to a tensor). Two complementary hook points:

#### 4a — Buffer-level lease (protects CPU-backend reads for tensor lifetime)

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp:17981-18053` (`ggml_backend_sycl_host_buffer_type`
callbacks)

The buffer already tracks the `alloc_handle` in `sycl_host_buf_ctx { ptr, size, alloc }`.
Add a `mem_handle buffer_lease` field that holds a lease on the allocation for the entire
buffer lifetime:

- `alloc_buffer` (line 17999): after `unified_alloc` succeeds, call
  `buffer_lease = mem_handle::from_chunk_ptr(ptr, device, GGML_LAYOUT_AOS, false)`.
  If lease is DIRECT (path D — malloc_host fallback), use
  `acquire_weight_lease(buffer_key)` instead (requires synthesizing a buffer-level cache
  key; see design below).
- `free_buffer` (line 17981): the `buffer_lease` field destructor runs automatically when
  `delete ctx` executes. That releases the lease, allowing `unified_free` to actually
  reclaim the memory.

Result: while any `ggml_backend_buffer_t` holds the allocation (i.e., while any tensor is
using `tensor->data` from it), the lease refcount is > 0 and eviction cannot free the
memory. Protects CPU-backend reads without any CPU-backend awareness — the buffer IS the
CPU backend's view of the memory.

#### 4b — Per-entry lease on register_host_expert (defense in depth for async tasks)

**File**: `ggml/src/ggml-sycl/unified-cache.cpp` — `register_host_expert` impl

For entries registered without copy (Phase 2-3), acquire a `from_chunk_ptr` chunk lease on
the per-expert pointer, stored in the `weight_entry`. Release at `unregister_host_expert`
/ entry eviction. This adds per-entry refcount on top of 4a's buffer-level lease.

Path D fallback: if `from_chunk_ptr` returns DIRECT, use `acquire_weight_lease(key)`
instead (per P1 audit §9 Q1 union strategy).

**Cumulative refcount** on the allocation = 1 (buffer lease, 4a) + N (per-expert entries,
4b). Memory survives as long as ANY of these is held. CPU-backend reads are safe for the
whole buffer lifetime. In-flight SYCL async tasks are safe for the entry lifetime.

#### 4c — Eviction guard assert (concrete guard rail)

In `evict_one` (or the WEIGHT-entry eviction site), assert that the target entry's lease
refcount is zero before calling `unified_free`. If a WEIGHT-zone entry has lease > 0, log
with full context (entry key, refcount value, caller pattern) and either skip the eviction
or assert-with-detail. Any future drift that would silently free in-use memory surfaces
loudly.

**Gate**: `GGML_SYCL_A7L5W_INSTRUMENT=1` canary — 0 probe aborts on 20B -n 30. Runtime
evict_one calls: non-zero lease refcount never reaches `unified_free`.

### Phase 5 — Remove dead `_host_req2` code path and update comments

After Phase 2-4 stable. Remove the `alloc_request _host_req2` block. Update comments.

### Phase 6 — Audit remaining MoE routing booleans and eliminate derivable ones

**Architectural principle** (per user): legacy branching booleans should not exist when
the unified cache + placement plan carry the state. Commit `445028753` already removed
`host_resident_moe_weights`, `early_cpu_expert_tg`, `skip_gpu_moe_fast_paths`. New ones
have sprouted since.

**Audit target**: every boolean in `ggml_sycl_mul_mat_id` (and related callers) that
captures MoE expert placement. Candidates:
- `host_weights_fast` (3 use-sites) — derivable from `lookup_expert` per-expert residency
- `cpu_expert_tg_active` (10 use-sites) — derivable from placement plan + op shape
- Any other `_fast`, `_host`, `_active` suffixed boolean that mirrors placement state

For each boolean: is it derivable from the cache + plan + op metadata alone? If yes,
eliminate it and replace with the direct query. If no (genuine local state, e.g.
thermal throttle flag), document why it must stay.

**Gate**: No regression on gates 1-5 across all offload modes. Code reads cleanly: routing
decisions are pointer-location queries against the cache, not boolean gates.

Close epic on Phase 6 pass.

## 6. Risk Analysis

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `from_chunk_ptr` returns DIRECT (pool miss) | Medium | Phase 1 audit resolves; DIRECT is still correct for lifetime |
| Lifetime hole: tensor buffer freed mid-inference | Very Low | CPU backend never frees tensor buffers mid-inference |
| Phase 2 breaks `all_weights_host` path (different alloc origin) | Medium | Phase 3 separate; gated by alloc-origin audit |
| Mistral 7B regression (dense weight path) | Very Low | Dense tensors don't go through the 11681-11700 or 12017-12021 expert copy paths |
| mmap-backed tensors | Low | `llama-model.cpp:3632` already allows `host_buft_with_mmap=true` for `moe_expert_host_pinned` — no change |

## 7. Interaction With Narrow Fix

The narrow fix (`2026-04-19-cpu-mxfp4-moe-n1-fix-plan.md`) is independent of all epic phases.
After the epic lands, `dispatch_cpu_compute` still needs its pool-backed activation staging
buffer. The narrow fix's fallthrough path remains valid.

## 8. Interaction With Other Epics

- **`project_tensor_access_redesign`**: This epic delivers the precondition — `tensor->data`
  IS the canonical arena pointer. Raw `->data` reads from CPU backend become safe (they read
  the unified cache allocation, not a dangling mmap).
- **`llama.cpp-4oi3i`** (ADD_ID bias crash): CLOSED (`6226d2a74`). No interaction.
- **`mxfp4-compute-parity`** (`llama.cpp-tlcjr`): CLOSED. 20B at 30% VRAM works. This epic
  benefits the 100% VRAM path by halving host memory use for the `moe_expert_host_pinned`
  case.
- **`0k543` C3 lease migration**: Phase 4 uses `from_chunk_ptr` from 0k543. Dependency.

## 9. Open Questions (need decision before Phase 2 starts)

1. **Does `unified_alloc(use_pinned_pool=true)` always route through `pinned_chunk_pool`?**
   **ANSWERED (Phase 1 audit)**: NO — there is a direct `sycl::malloc_host` fallback for
   role=WEIGHT when the pool is exhausted or zones are not yet configured. Phase 4 uses a
   union strategy: `from_chunk_ptr` for pool-backed pointers, `acquire_weight_lease(key)` for
   direct-malloc fallback. See Phase 1 audit section above.
2. **Environment gate for Phase 2?** Recommend `GGML_SYCL_UNIFIED_MEM_PHASE2=1` for first
   deployment; remove after one clean gate cycle.
3. **Does the `all_weights_host` branch allocate from the same pool?** Phase 3 audit answers.
   May split Phase 3 into two substeps if origins differ.

## 10. Success Metrics

- Host-pinned memory for MoE expert weights: ~50% reduction (one allocation instead of two)
- 20B at default VRAM budget: same or better throughput (less WEIGHT zone pressure)
- All gates 1-5 pass (no regression)
- `GGML_SYCL_A7L5W_INSTRUMENT=1` canary: 0 probe aborts (chunk-lease integrity)
- `_host_req2` code path removed from S1-PRELOAD (dead code eliminated)
