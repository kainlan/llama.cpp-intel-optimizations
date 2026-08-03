# Canonical SYCL Memory Architecture Contract

*Status: in force since 2026-04-27; last updated 2026-08-04. All new SYCL backend
code must comply. Migration tasks for existing non-conforming sites are tracked
in `llama.cpp-32dg8`. Section 12 is the lifecycle authority: it states both the
target contract and the non-conforming current implementation explicitly.*

---

## 1. Three Primitives

The SYCL backend has exactly **three canonical memory primitives**. Every memory
decision in the backend reduces to one of these three.

### 1.1 Planner (`compute_placement_plan` / `placement_plan`)

**File:** `ggml/src/ggml-sycl/unified-cache.hpp`
**Key types:** `placement_plan`, `placement_entry`, `placement_kv_info`
**Entry point:** `ggml_sycl::compute_placement_plan(...)`,
                 `ggml_sycl::compute_multi_device_plan(...)`

The planner is the **sole authority for deciding where memory lives** before any
allocation occurs. It runs once at model-load time and produces a `placement_plan`
that covers:

| Category | `placement_plan` field(s) |
|---|---|
| Dense weights (each layer) | `entries[]`, `layer_device` |
| MoE experts | `entries[]`, `expert_device` |
| KV cache (per layer) | `kv_device`, `kv_vram_bytes`, `kv_host_bytes` |
| oneDNN reorder scratch | `onednn_reorder_bytes` → Zone: SCRATCH |
| oneDNN scratchpad | `onednn_scratchpad_bytes` → Zone: ONEDNN |
| MoE Q8 workspace | `moe_q8_workspace_bytes` → Zone: SCRATCH |
| MoE routing IDs | `moe_routing_ids_bytes` → Zone: RUNTIME |
| MoE expert pointer tables | `moe_expert_ptrs_bytes` → Zone: RUNTIME |
| DMA staging pool | `dma_staging_pool_bytes` → Zone: RUNTIME |
| PP pipeline scratch | `pp_pipeline_scratch_bytes` → Zone: RUNTIME |
| CPU quant buffers | `cpu_quant_buffer_bytes` → Zone: HOST |
| Graph metadata | `graph_metadata_bytes` → Zone: HOST |
| TP FFN / attn buffers | `tp_ffn_buffer_bytes`, `tp_attn_buffer_bytes` → Zone: RUNTIME |
| TP staging | `tp_staging_buffer_bytes` → Zone: STAGING |

**What the planner does NOT own:**
- The actual allocations (unified cache owns those).
- Per-token or per-request scratch that is reset between inferences (SCRATCH zone).
- Raw pointer values — the planner records bytes and device IDs, never pointers.

**Invariant:** Dispatch code must never make host/device placement decisions by
inspecting `ggml_tensor::data` or calling `ggml_backend_buffer_is_host()` directly.
All placement decisions are delegated to the planner-derived handles at graph-build time.

### 1.2 Unified Cache (`ggml_sycl::unified_cache`)

**File:** `ggml/src/ggml-sycl/unified-cache.hpp`, `unified-cache.cpp`
**Key class:** `ggml_sycl::unified_cache`
**Key free functions:** `unified_alloc`, `unified_allocate`, `unified_free`,
`unified_free_ptr`, `unified_lookup`, `unified_cache_raw_malloc_device`,
`unified_cache_raw_malloc_host`, `unified_cache_raw_malloc_shared`

The unified cache is the **sole allocator and materializer for all SYCL backend
memory**. It manages five named zones:

| Zone (`vram_zone_id`) | Purpose | Allocation lifetime |
|---|---|---|
| `KV` | KV cache buffers | Per context; reset at context free |
| `WEIGHT` | Model weight cache | Process-scoped; evicted under pressure |
| `ONEDNN` | oneDNN FP16 scratch | Fixed region per model load |
| `RUNTIME` | Compute buffers, MoE pools, TP buffers | Per model load; some per-token |
| `SCRATCH` | Per-token scratch; reset between tokens | Token-scoped |

Plus a host-pinned tier managed by `host_zone_id` (`pinned-pool.hpp:56`), which
has four named zones distinct from `vram_zone_id`:

| Zone (`host_zone_id`) | Purpose |
|---|---|
| `WEIGHT` | Host-resident model weights (overflow from VRAM) |
| `KV` | KV cache host fallback |
| `STAGING` | DMA staging / expert bias D2H |
| `SCRATCH` | Host-side quantization / CPU compute scratch |

Host-pinned allocations use `unified_cache_host_zone_alloc(host_zone_id zone, ...)` /
`unified_cache_zone_alloc`. The `→ Zone: HOST` shorthand in §1.1 refers to any of
these four `host_zone_id` zones.

**What the unified cache does NOT own:**
- Placement decisions (the planner owns those).
- Kernel dispatch logic (the dispatch router owns that).
- Memory allocated inside tests (see §8).
- ggml buffer allocations at the public `ggml_backend` API boundary (see §8).

**Authority is not optional.** Commit `9a0670712` removed optional cache
enablement and the cache enable/disable branches, including the
`unified_cache_enabled()` check of the now-unread `GGML_SYCL_UNIFIED_CACHE`
opt-out. `GGML_SYCL_UNIFIED_CACHE_MODE` is reserved for topology selection
(`auto`, `global`, or `per_device`); it does not disable the cache, and no
replacement enable/disable control exists. See §9.3 for the migration rule.

Reset does not override ownership. Weight reclaim is ownership- and mode-aware:
`reset_model_weight_entries()` preserves entries with active `mem_handle` leases,
entries owned by any live model even when `in_use_count == 0`, and unattributed
entries when the current reclaim mode and live-model mask require it. It reclaims
only entries for which `weight_entry_reclaimable()` returns true. Outside
`MID_LOAD_REPLAN`, an attributed entry that remains leased with no live model
owner is diagnosed separately as an ownerless lease and may abort when
`GGML_SYCL_STRICT_LEASES=1`; that ownerless warning and strict abort are
suppressed during `MID_LOAD_REPLAN`, and the state is not treated as legitimate
concurrent-model ownership. Whole-zone resets cannot preserve a single
allocation, so `host_zone_reset()` and `zone_reset()` refuse the entire
reset while the target zone contains live registered allocations (`4afdb6d9f`).
Callers must release the owning handles and retry; they must not purge ownership
records or force reclamation.

### 1.3 `mem_handle` (`ggml_sycl::mem_handle`)

**File:** `ggml/src/ggml-sycl/mem-handle.hpp`, `mem-handle.cpp`
**Key type:** `ggml_sycl::mem_handle`
**Resolution type:** `ggml_sycl::resolved_ptr` (holds `void * ptr`, `ggml_layout_mode layout`, `bool on_device`)

`mem_handle` is the **sole pointer abstraction that may cross dispatch boundaries**
in the SYCL backend. Raw `void *` pointers must not be passed between subsystems;
they must be wrapped in a handle.

A handle encodes one of six kinds (`mem_handle_kind`):

| Kind | Backing | Lifetime protection |
|---|---|---|
| `WEIGHT` | Cache-managed entry | `in_use_count` refcount; eviction blocked while handle alive |
| `DIRECT` | Raw pointer (no cache) | Caller's responsibility; handle provides no protection |
| `ARENA_RUNTIME` | Offset into RUNTIME zone | Zone reset must not occur while handle alive |
| `ARENA_SCRATCH` | Offset into SCRATCH zone | Zone reset must not occur while handle alive |
| `ARENA_ONEDNN` | Offset into ONEDNN zone | Zone must outlive handle |
| `CHUNK_LEASE` | Arena chunk lease (host or VRAM) | `sycl::free` of backing chunk blocked while handle alive |

**Resolution:** call `handle.resolve()` to get a `resolved_ptr`. This is a ~3 ns hot
path for `DIRECT` and generation-current `WEIGHT` handles; otherwise it re-queries
the cache.

**What `mem_handle` does NOT own:**
- Allocation — use `unified_alloc` / `unified_allocate` to create memory; the
  resulting `alloc_handle` can then be wrapped in a `mem_handle`.
- Placement decisions — `from_weight()` records a device ID from the planner,
  but does not decide placement.
- SYCL event dependencies — handles are value types. Event tracking belongs at the
  call site (see §4).

---

## 2. Dispatch Router

The dispatch router is the **consumer of the three primitives above**. Given an op
and its `mem_handle` operands, it selects and invokes the correct kernel
(XMX / ESIMD / MMVQ / MMQ / DMMV / CPU / oneDNN). It is not a memory primitive; it
reads placement decisions from the planner, resolves pointers from handles, and
allocates nothing.

**Current entry point:** `ggml_sycl_compute_forward` in `ggml/src/ggml-sycl/ggml-sycl.cpp:39002`
**Current MUL_MAT entry point:** `ggml_sycl::ggml_sycl_mul_mat_unified` in `ggml/src/ggml-sycl/dispatch.hpp:418`
**Kernel selector:** `ggml_sycl::select_kernel_type` in `ggml/src/ggml-sycl/dispatch.hpp:186`

**Current shape:** `ggml_sycl_compute_forward` dispatches by `ggml_op` enum and
calls into per-op helpers. Those helpers currently branch on caller-side host/device
predicates (`ggml_backend_buffer_is_host`, `ggml_sycl_is_host_resident_weight`,
`has_placement_plan`) to choose between GPU and CPU paths. This is the legacy
pattern that T8 replaces.

**Target shape (TBD — tracked in `llama.cpp-32dg8.9`):** A canonical router whose
inputs are `ggml_op`, `mem_handle` operands, op metadata, device ID, and SYCL event
dependencies. Its signature shape is:

```cpp
// Target API — not yet implemented (llama.cpp-32dg8.9)
sycl::event dispatch_op(
    ggml_op                        op,
    std::span<const mem_handle>    srcs,
    const mem_handle &             dst,
    const OperationContext &       ctx,  // see note below
    std::span<const sycl::event>   deps);
```

*Note on `OperationContext`:* Two independent definitions exist today —
`op-context.hpp:32` and `dispatch.hpp:353`. T8 (`llama.cpp-32dg8.9`) must
disambiguate (either unify them or pick one canonical definition) before this
target API can be implemented.

The router determines residency from `handle.resolve().on_device` and `handle.device()`,
selects the kernel, and returns an event — without the caller performing any
host/device predicate check.

**What the dispatch router does NOT own:**
- Placement decisions (the planner owns those).
- Allocation (the unified cache owns that).
- Pointer resolution (the handle owns that).

---

## 3. Allocator Entry Points (Allowlist)

Only the following functions may allocate SYCL memory. All other code must call
these or be migrated (see §9 for temporary allowlisted sites):

| Function | Purpose | Returns |
|---|---|---|
| `unified_alloc(req, out)` | Primary allocator; routes by zone/tier | `bool` + `alloc_handle` |
| `unified_allocate(req)` | Handle-returning wrapper around `unified_alloc` | `mem_handle` |
| `unified_cache_allocate(device, size, category, queue)` | Bulk weight/arena slot allocator | `unified_alloc_result` (`unified-cache.hpp:2374`) |
| `unified_cache_zone_alloc(device_id, zone, size, align = 256)` | Named VRAM zone allocation (`unified-cache.hpp:2856`) | `void *` |
| `unified_cache_host_zone_alloc(zone, size, align)` | **Deprecated** — host-pinned zone allocation; migrate to `unified_allocate()` with `must_host_pinned` + `use_pinned_pool` | `void *` |
| `unified_cache_arena_alloc` | **Deprecated** — migrate to `unified_allocate(..., prefer_vram_zone=SCRATCH)` | `void *` |
| `unified_cache_raw_malloc_device(size, queue)` | Raw `sycl::malloc_device` wrapper — call only from inside `unified_cache` internals | `void *` |
| `unified_cache_raw_malloc_host(size, queue/ctx)` | Raw `sycl::malloc_host` wrapper — call only from inside `unified_cache` internals | `void *` |
| `unified_cache_raw_malloc_shared(size, queue)` | Raw `sycl::malloc_shared` wrapper — call only from inside `unified_cache` internals | `void *` |

**Rule:** `sycl::malloc_device`, `sycl::malloc_host`, and `sycl::malloc_shared` must
never be called outside `unified-cache.cpp`. All other files must use the wrappers
above.

**Deallocation:** `unified_free(handle)`, `unified_free_ptr(ptr, device)`,
`unified_cache_zone_free(device_id, zone, ptr)`, `unified_cache_arena_free(device_id, ptr, size)` (deprecated).

---

## 4. Pointer-Resolution Entry Points (Allowlist)

Only the following functions may hand out resolved raw pointers from a handle:

| Function | When to use |
|---|---|
| `mem_handle::resolve()` | Normal dispatch: returns `resolved_ptr{ptr, layout, on_device}` |
| `unified_lookup(void * ptr, alloc_handle * out)` | Reverse-look up an allocation by raw pointer (`unified-cache.hpp:2345`) |
| `unified_cache::memset(h, val, size, queue)` | Fill memory behind a handle (void, no event returned — **deprecated API shape**; see §6) |
| `unified_cache::memcpy(dst, src, size, queue)` | Copy between handles (void, no event returned — **deprecated API shape**; see §6) |

**Rule:** Callers must not store or pass the `void *` from `resolve()` across async
boundaries. The handle that produced it must remain alive for the duration of any
SYCL work that uses the pointer.

---

## 5. Process-Scoped vs Context/Slot-Scoped State

A process may switch models sequentially, keep multiple `llama_model` objects
alive, and create multiple contexts. Those object lifetimes are distinct from
execution support: current planner globals are process-wide rather than keyed by
model, and per-device arena state is not keyed by context. The graph-compute
eviction guard and graph-compute mutex are separate process-global state. Live
ownership must therefore be safe across coexisting objects, but concurrent
inference on the same SYCL device is not supported. The mutex does not
universally serialize submission or device execution; §5.3 defines the mixed
path taxonomy and the resulting overlap. The contract distinguishes:

### 5.1 Process-scoped (model) state

Associated with the currently active model load; immutable after that load
completes and shared by its contexts:

| Global / struct | Current location | Correct scope |
|---|---|---|
| `g_tensor_inventory` | `ggml-sycl.cpp:6169` | Model-scoped: populated at set_tensor_data, read-only during inference |
| `g_placement_plan` | `ggml-sycl.cpp:6194` | Model-scoped: computed once, immutable |
| `g_has_placement_plan` | `ggml-sycl.cpp:6195` | Model-scoped flag |
| `g_placement_kv_info` | `ggml-sycl.cpp:6178` | Model-scoped: per-layer KV sizing |
| `g_model_n_layer` | `ggml-sycl.cpp:6177` | Model-scoped |
| `unified_cache` WEIGHT zone | `unified-cache.cpp` | Model-scoped: weight entries survive across requests |
| `placement_plan::entries` | `unified-cache.hpp` | Model-scoped |
| `placement_plan::layer_device`, `expert_device`, `kv_device` | `unified-cache.hpp` | Model-scoped |

**Invariant:** No context-level operation (graph compute, KV reset, slot eviction)
may modify model-scoped globals. A load or reset must preserve weight entries
owned by every still-live model object. These process-wide globals are not
model-keyed concurrency state, and read-only access does not make same-device
concurrent graph compute safe.

### 5.2 Context/slot-scoped state

Owned per inference context; reset between requests or at context free:

| State | Zone / location | Reset boundary |
|---|---|---|
| KV cache buffers | `vram_zone_id::KV` | Context free or `llama_kv_cache_clear` |
| KV host fallback buffers | Host zone / `kv_host_bytes` | Context free |
| RUNTIME compute buffers | `vram_zone_id::RUNTIME` | `arena_reserve` at graph compute boundary |
| SCRATCH per-token buffers | `vram_zone_id::SCRATCH` | Each graph compute step (`ggml-sycl.cpp:41455`) |
| oneDNN scratch | `vram_zone_id::ONEDNN` | Acquired/released per graph compute |
| `g_layer_on_cpu` | `ggml-sycl.cpp:6201` | Recomputed at each graph build |
| MoE routing buffers | RUNTIME zone | Per-inference reset |
| Staging / DMA buffers | HOST / RUNTIME | Per-weight-stream event |

**Invariant:** `arena_reserve` requests resets of KV + RUNTIME + HOST zones only
for the owning device's cache. A reset proceeds only when the target zone has no
live registered allocations; otherwise the reset is refused and existing
allocations are preserved. It must not reset another context's zones. Until T1
(`llama.cpp-32dg8.2`) adds explicit context ownership keys, callers must ensure
single-active-context per device.

### 5.3 Multiple models, contexts, and server slots

Keep these cases separate:
- `llama-bench` with multiple `-m` values performs sequential model switches:
  it frees the old model before loading the next one. It tests teardown and
  reload, not simultaneous model or context execution.
- Multiple model or context objects may remain alive. Their ownership records
  and leases must coexist without one model load or teardown reclaiming another
  live model's weights.
- Object coexistence does not imply execution concurrency. Each server slot
  would need a distinct context and context-keyed KV/RUNTIME arena reservation;
  that ownership is not implemented yet (tracked in `llama.cpp-32dg8.15.10`).
  `unified_cache_set_graph_compute_active(bool)` has no device argument and sets
  the process-global `g_graph_compute_active` eviction guard
  (`unified-cache.cpp:303`). It is not per-device or per-context state.
- Independently, the process-global `g_sycl_graph_compute_mutex` is acquired at
  the current graph-compute entry point (`ggml-sycl.cpp:91438`) but does not
  universally serialize submission. Direct/fallback paths explicitly release it
  before `compute_impl` submission (`ggml-sycl.cpp:91627-91630`). In contrast,
  persistent-TG and deferred-copy paths submit while it remains held
  (`ggml-sycl.cpp:91978`, `92084`, `92101`, `92159`), as do command-graph
  record/replay paths (`ggml-sycl.cpp:92700`, `93161`, `93188`, `93298`).
  Completion may still outlive the lock where a path permits deferred exit.
  Thus host submission can overlap across graph-compute calls on direct/fallback
  paths, and device execution may overlap across calls and devices; pure-GPU
  decode may also return with kernels still in flight. Do not infer supported
  concurrent inference or cache safety from either overlap. The distinct
  same-device limitation remains: same-device concurrent inference is
  unsupported until context-keyed KV/RUNTIME arena ownership exists.

---

## 6. Memory Operation Helpers

Memory operations between handles must use the canonical helpers, not raw
`sycl::queue::memcpy` or `std::memcpy`:

### 6.1 Current API (void, no event returned — pre-canonical)

```cpp
// unified_cache instance methods:
void unified_cache::memset(const mem_handle & h, int value, size_t size, sycl::queue & stream);
void unified_cache::memcpy(const mem_handle & dst, const mem_handle & src, size_t size, sycl::queue & stream);
```

These are the current production API. They do not return events and do not accept
dependency lists. They are correct for use inside synchronous paths but are
**not safe for graph recording** (a graph-recording context must not call `.wait()`
implicitly or make blocking decisions).

### 6.2 Required canonical form (target — tracked in `llama.cpp-32dg8.8` / `llama.cpp-32dg8.15.13`)

The canonical memory op API must:
- Accept `std::span<sycl::event>` dependencies.
- Return `sycl::event`.
- Dispatch H2H, H2D, D2H, same-device D2D, cross-device D2D as separate code paths.
- Be graph-safe (no implicit host-wait during graph recording).
- Be exposed as free functions so callers do not need a `unified_cache` instance.

Until `llama.cpp-32dg8.8` lands, callers may continue to use the instance-method
forms above with the understanding that they are scheduled for replacement.

---

## 7. Single-GPU / Multi-GPU / Multi-Context Invariants

### 7.1 Single GPU

- One `unified_cache` instance per device.
- `compute_placement_plan(...)` called once at model load with `device_id = 0`.
- All zones live on device 0 or host pinned.
- `mem_handle::device()` always returns 0 for weight handles.

### 7.2 Multi-GPU

- One `unified_cache` instance per device (`get_unified_cache_for_device(dev)`).
- `compute_multi_device_plan(...)` called at model load; produces a single
  `placement_plan` with `multi_device = true` and populated `devices`,
  `per_device_vram`, `layer_device`, `expert_device`, `kv_device` maps.
- Each `placement_entry` has `target_device` set explicitly. No entry may have
  `target_device = -1` unless placement is `HOST`.
- As of bead `llama.cpp-32dg8.15.11`, `from_direct` requires an explicit `device`
  parameter (defaulting to `HOST_DEVICE = -1`); device-resident handles must pass
  their owning device, host pointers pass `HOST_DEVICE`.
- Cross-device transfers use explicit BCS memcpy (OOQ) between device-pinned
  buffers. Hardware P2P is not supported on current Arc hardware; all transfers
  stage via host.
- The `mem_handle::device()` value is authoritative for which cache instance to
  query. Dispatch must not assume handles from different devices are interchangeable.
- Multi-GPU layout: `layer_device` maps dense layers, `expert_device` maps MoE
  experts. KV cache co-locates with the dense weights of each layer (`kv_device`).

### 7.3 Multi-context / Multi-server slot

The lifetime and execution distinctions in §5.3 apply. Contexts sharing a model
may share its read-only `placement_plan` and WEIGHT entries, but the
process-global mutex does not make host submission or device execution globally
serial. Separately, same-device concurrent inference remains unsupported until
`llama.cpp-32dg8.15.10` delivers explicit context-keyed KV/RUNTIME arena
ownership.

---

## 8. Out of Scope

This contract does not govern:

1. **Test-internal allocations** — `sycl::malloc_device` / `sycl::malloc_host`
   called directly inside test files under `ggml/src/ggml-sycl/tests/` are exempt.
   Tests may use raw SYCL allocation APIs to construct fixtures without routing
   through `unified_alloc`.

2. **`ggml_backend` buffer API boundary** — `ggml_backend_buffer_t` allocations
   driven by the public `ggml_backend_alloc_ctx_tensors` / `ggml_backend_buffer_type_alloc_buffer`
   path are governed by the ggml buffer type abstraction, not by this contract.
   The SYCL buffer type implementation calls `unified_cache_raw_malloc_device` /
   `unified_cache_raw_malloc_host` internally, which is an allowed raw-malloc site.

3. **Non-SYCL backends** — CUDA, Metal, Vulkan, CPU backends have their own memory
   management; this contract is SYCL-only.

4. **Host-side ggml tensor `.data` allocation** — CPU tensors backed by normal
   `malloc` are not SYCL-managed and are outside this contract.

---

## 9. Temporary Allowlists (Migration Inventory)

The following raw allocation or host-residency-check sites are temporarily allowed
pending migration. Each entry names the owning bead and deletion criteria.

### 9.1 Direct SYCL allocation sites (non-`unified_cache` internals)

The full inventory (711 `malloc_device|malloc_host|malloc_shared` hits across `ggml/src/ggml-sycl/`, per `grep -rE 'malloc_device|malloc_host|malloc_shared'`) is being
built as part of `llama.cpp-32dg8.15.14`. Until that inventory is complete, sites
are grouped by subsystem:

| Subsystem / file | Approximate site count | Owner bead | Deletion criteria |
|---|---|---|---|
| `dense-scheduler.cpp` | ~5 | `llama.cpp-32dg8.6` | Migrate to `unified_allocate` with RUNTIME zone |
| `mmvq.cpp`, `mmq.cpp` | ~10 | `llama.cpp-32dg8.6` | Migrate to `unified_allocate` |
| `convert.cpp`, `dmmv.cpp`, `getrows.cpp`, `cpy.cpp` | ~20 | `llama.cpp-32dg8.6` | Migrate to `unified_allocate` |
| `kv-offload.cpp` | ~8 | `llama.cpp-32dg8.5` | Migrate to planner-sized KV zone |
| `ccl-comm.cpp` | ~5 | `llama.cpp-32dg8.11` | Migrate after multi-GPU validation |
| `fused-moe-esimd.hpp`, `gpu-sampler.hpp` | ~10 | `llama.cpp-32dg8.6` | Migrate to `unified_allocate` |
| `ggml-sycl.cpp` MoE + TP paths | ~30 | `llama.cpp-32dg8.6` | Migrate to planner + `unified_allocate` |
| `unified-cache.cpp` internals | all | — | Allowed permanently as raw-malloc gateway |
| Test files under `tests/` | all | — | Allowed permanently (see §8) |

### 9.2 Host-residency predicates (caller-side "is on host?" checks)

129 sites across `ggml/src/ggml-sycl/` branch on `ggml_backend_buffer_is_host`,
`ggml_sycl_is_host_resident_weight`, `ggml_sycl_weight_is_planned_on_host`, or
`has_placement_plan` (101 in `ggml-sycl.cpp` alone; per
`grep -rn 'ggml_backend_buffer_is_host|ggml_sycl_is_host_resident|ggml_sycl_weight_is_planned_on_host|has_placement_plan\b'`).
These are temporarily allowed; migration to handle-based
dispatch is tracked in:

| Owner bead | Scope |
|---|---|
| `llama.cpp-32dg8.9` | Build dispatch router; replace MUL_MAT / MUL_MAT_ID host-predicate branches |
| `llama.cpp-32dg8.10` | Migrate remaining op call sites |
| `llama.cpp-32dg8.15.17` | Build host-fallback coverage matrix, file per-op blockers |

### 9.3 Removed cache opt-out

The authority and topology contract is stated in §1.2. Migration must not
reintroduce optional cache enablement or cache enable/disable branches;
`GGML_SYCL_UNIFIED_CACHE_MODE` remains topology-only. Remaining memory-routing
branches should express concrete conditions such as weight eviction,
placement-plan state, or host accessibility rather than cache enablement.

| Owner bead | Action |
|---|---|
| `llama.cpp-32dg8.4` | Removed optional unified-cache mode branches |
| `llama.cpp-32dg8.3` | Delete legacy host-weight fallback registration (precondition for T3) |

### 9.4 `mem_handle::from_direct` with implicit device 0

All `from_direct(ptr, layout, on_device)` calls omit device identity. These are
temporarily allowed but must be audited for multi-GPU callers:

| Owner bead | Action |
|---|---|
| `llama.cpp-32dg8.15.11` | Add device-aware DIRECT constructors; define wrong-device policy |
| `llama.cpp-32dg8.7` | Expand `mem_handle` coverage; replace remaining DIRECT escapes |

---

## 10. Cross-References: Child Beads to Contract Sections

| Bead | Title | Contract sections |
|---|---|---|
| `llama.cpp-32dg8.1` | T0 — this document | all |
| `llama.cpp-32dg8.2` | T1 — S1-PRELOAD consumes `placement_plan` entries directly | §1.1, §5.1 |
| `llama.cpp-32dg8.3` | T2 — Delete legacy host-weight fallback registration | §9.3 |
| `llama.cpp-32dg8.4` | T3 — Removed optional unified-cache mode branches | §9.3 |
| `llama.cpp-32dg8.5` | T4 — Planner covers all SYCL memory domains | §1.1, §5, §9.1 |
| `llama.cpp-32dg8.6` | T5 — Unified cache as sole allocator | §1.2, §3, §9.1 |
| `llama.cpp-32dg8.7` | T6 — Expand `mem_handle` into universal handle | §1.3, §4, §9.4 |
| `llama.cpp-32dg8.8` | T7 — Canonical `mem_handle` memory operations | §6.2 |
| `llama.cpp-32dg8.9` | T8 — Unified dispatch router over `mem_handle` operands | §2, §4, §9.2 |
| `llama.cpp-32dg8.10` | T9 — Migrate op call sites off raw residency checks | §9.2 |
| `llama.cpp-32dg8.11` | T10 — Multi-GPU validation | §7.2, §9.1 |
| `llama.cpp-32dg8.12` | T11 — Multi-user / multi-context validation | §5.3, §7.3 |
| `llama.cpp-32dg8.13` | T12 — Final audit gates | all |
| `llama.cpp-32dg8.15` | PROOF gate | §1–§9 (all proof P1–P8 invariants) |
| `llama.cpp-32dg8.15.1` | P1 — Prove model-scoped vs context-scoped memory ownership | §5.1, §5.2, §5.3 |
| `llama.cpp-32dg8.15.2` | P2 — Prove multi-GPU `mem_handle` ownership and wrong-device behavior | §1.3, §7.2, §9.4 |
| `llama.cpp-32dg8.15.3` | P3 — Prove in-flight `mem_handle` lease lifetime through SYCL events | §1.3, §4 |
| `llama.cpp-32dg8.15.4` | P4 — Prototype canonical `mem_handle` memory operations | §6.2 |
| `llama.cpp-32dg8.15.5` | P5 — Audit every SYCL allocation site before sole-allocator migration | §3, §9.1 |
| `llama.cpp-32dg8.15.6` | P6 — Prove dispatch-router shape with MUL_MAT and MUL_MAT_ID | §2, §4, §9.2 |
| `llama.cpp-32dg8.15.7` | P7 — Prove planner can predict runtime and scratch allocation demand | §1.1, §5, §6 |
| `llama.cpp-32dg8.15.8` | P8 — Prove host-resident fallback coverage for spilled subgraphs | §9.2 |
| `llama.cpp-32dg8.15.9` | P9 — Make 32dg8 implementation beads junior-ready | all |
| `llama.cpp-32dg8.15.10` | P1-FIX — model vs context ownership fixes | §5.1, §5.2, §5.3 |
| `llama.cpp-32dg8.15.11` | P2-FIX — `mem_handle` device identity | §1.3, §7.2, §9.4 |
| `llama.cpp-32dg8.15.12` | P3-FIX — in-flight handle lease lifetime | §1.3, §4 |
| `llama.cpp-32dg8.15.13` | P4-FIX — canonical event-returning memory ops | §6.2 |
| `llama.cpp-32dg8.15.14` | P5-FIX — allocation site inventory | §3, §9.1 |
| `llama.cpp-32dg8.15.15` | P6-FIX — dispatch router vertical slice | §2, §4, §9.2 |
| `llama.cpp-32dg8.15.16` | P7-FIX — plan-vs-actual auditor | §1.1, §5, §6 |
| `llama.cpp-32dg8.15.17` | P8-FIX — host fallback coverage matrix | §9.2 |

---

## 11. Execution Chunks and Checkpoints

The remaining migration must land as bounded checkpoints. Do not start a later
chunk while an earlier proof chunk is unresolved.

| Chunk | Beads | Scope | Checkpoint |
|---|---|---|---|
| C1 | `32dg8.16`, `32dg8.1`, `32dg8.15` and proof children | Proof gate and junior-ready foundation | All proof tasks close with evidence or concrete fix beads; `test-thread-safety` GPU smoke proof recorded |
| C2 | `32dg8.17`, `32dg8.2`-`32dg8.6` | Planner and unified-cache allocator migration | Preload, placement, and allocation decisions flow through planner plus unified cache; legacy optional paths are gone |
| C3 | `32dg8.18`, `32dg8.7`-`32dg8.10` | `mem_handle` operations and dispatch router migration | Common memory operations use event-returning handle helpers; migrated ops dispatch from handles |
| C4 | `32dg8.19`, `32dg8.11`-`32dg8.13` | Multi-GPU, multi-user, and final audit gates | Single-GPU, dual-GPU, server-slot, and final grep/audit gates pass |

### 11.1 C1 Bounded Proof Result

On 2026-04-28, the current implementation passed a bounded SYCL multi-context
smoke test:

```bash
source /opt/intel/oneapi/setvars.sh --force >/tmp/oneapi-setvars.log 2>&1
export LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:${LD_LIBRARY_PATH}
export ONEAPI_DEVICE_SELECTOR=level_zero:1
./build-sycl/bin/test-thread-safety \
  -m /models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 -p '1, 2, 3,' -n 4 -c 128 -ub 32 -np 2
```

Observed result: `All threads finished without errors.` The run selected the
Intel Arc Pro B50 and exercised two concurrent contexts per loaded model.

This is a smoke proof only. It does not close `32dg8.15.10` because it does not
prove server slots with different context sizes, explicit context-keyed arena
ownership, or context-keyed resets for KV/RUNTIME/HOST zones. C1 remains open
until those ownership keys or equivalent guards exist and the test matrix covers
same-model contexts with different `n_ctx`/slot lifetimes.

---

## 12. Model lifecycle and asynchronous ownership contract

### 12.1 Status and vocabulary

This section is an **enforceable target contract**, not a description of APIs
that exist today. A target name in backticks is a required identity or concept,
not a claim that a C++ type or public API with that spelling has landed. Until
all phases in §12.10 pass, current SYCL code is non-conforming where called out
below and must retain the conservative restrictions in §5.

| Identity | Required meaning and validity rule | Required owner/key |
|---|---|---|
| `ModelId` | Process-unique, monotonic identity for one model object; never a pointer and never reused during the process | All model-semantic plans, inventories, weights, diagnostics, and teardown |
| `SlotGeneration` | Monotonic generation incremented whenever a bounded numeric slot is reserved; `(slot, generation)` is the only valid slot identity | Transitional cache owner masks/slot tables; a bare slot is never sufficient |
| `LoadTxnId` | Process-unique identity for one **outermost** model-load attempt | Every load scratch write, allocation, registration, commit, and rollback |
| `ContextId` | Process-unique identity for one inference context | KV, RUNTIME, SCRATCH, oneDNN, staging, and context teardown |
| `SessionId` | Identity for one logical server sequence/session, namespaced by `ContextId`; not an allocation owner by itself | KV sequence rows and request diagnostics |
| `GraphEpoch` | Monotonic counter namespaced by `ContextId`; incremented on record, invalidation, and clear so stale callbacks cannot affect a replacement graph | Recorded graph state, pointer tables, retained handles, and terminal events |

Zero/wildcard/invalid identities must fail closed at mutating entry points. IDs
must appear in debug ownership records and lifecycle logs. Address identity,
`last_completed`, current-thread state, and device number are not substitutes.
All ID counters use checked, non-wrapping increment. Exhausted `ModelId`,
`LoadTxnId`, `ContextId`, `SessionId`, or `GraphEpoch` space permanently returns
the typed `ID_EXHAUSTED` error before mutation; it never wraps or reuses zero.
An exhausted `SlotGeneration` permanently retires that numeric slot. A bounded
slot may be reused only after the old generation is DEAD and all its leases have
drained; stale `(slot, generation)` operations are rejected.

Slot reservation is a typed, side-effect-free precondition to entering LOADING:
conceptually `result<slot_reservation, lifecycle_error>`, where the 33rd
simultaneous reservation returns `SLOT_EXHAUSTED`. Failure creates no `ModelId`
or `LoadTxnId`, changes no masks/generations/`last_completed`, allocates nothing,
and runs no load reset, planner, callback, or logging hook that mutates lifecycle
state. The current unattributed fallback is forbidden. Reservation plus checked
ID allocation is one atomic lifecycle-registry operation; only its success may
construct the owner tuple and transition ABSENT → LOADING.

**Current implementation (audited source, 2026-08-04):**
`ggml-sycl.cpp:8986-9199` has 32 bare ownership slots, process globals for one
loading slot and one last-completed slot, and an atomic nesting depth. It has no
slot generation, `ModelId`, or `LoadTxnId`; `ggml_backend_sycl_model_slot_current()`
returns last completed rather than caller identity. Planner/inventory scratch at
`ggml-sycl.cpp:9736-10096` is process-global. Pending KV masks at
`ggml-sycl.cpp:9244-9273` are keyed only by device and FIFO order. Therefore
those mechanisms are evidence of the migration need, not compliant identity APIs.

### 12.2 Lifecycle state machines

Required states and the only legal transitions are:

```text
model:   ABSENT --reserve+begin--> LOADING --commit--> LIVE
                     |                 `--abort--> ABORTING --> DEAD
                     `--typed error (no state change)
         LIVE --teardown--> TEARING_DOWN --> DEAD

txn:     ACTIVE --explicit outer success--> COMMITTED
         ACTIVE --failure/cancel/protocol error/missing success--> POISONED
         POISONED --outer unwind--> ROLLING_BACK --> ABORTED

context: ABSENT --create(ModelId)--> ALLOCATING --publish--> LIVE
         ALLOCATING --failure--> DRAINING --> DEAD
         LIVE --free/model teardown--> DRAINING --events drained--> DEAD

session: ABSENT --open(ContextId)--> ACTIVE
         ACTIVE --reset--> RESETTING --reset complete--> ACTIVE
         ACTIVE/RESETTING --close/context drain--> CLOSING --> CLOSED

graph:   NONE --record(ContextId,new GraphEpoch)--> RECORDING --> ACTIVE
         RECORDING --failure--> RETIRING --terminal event--> RETIRED
         ACTIVE --invalidate/replace/clear--> RETIRING --terminal event--> RETIRED
```

A load transaction is terminal exactly once. Nested load calls carry the same
`LoadTxnId`, increment/decrement its checked depth, and cannot commit. Wrong
transaction, depth underflow/overflow, cancellation, or an inner failure poisons
the outer transaction. The outermost successful exit may publish LIVE only
after all planned state and ownership records are complete. **Abort is the
default:** missing explicit success also poisons. Outermost unwind rolls back
only that transaction's allocations and scratch, releases its reservation, and
publishes neither `last_completed` nor LIVE. Rollback is idempotent: repeated
abort/unwind returns the same terminal result and performs no additional free,
callback, generation change, or owner removal. A failure while another model is
LIVE leaves that model byte-for-byte usable.

A context cannot publish LIVE until its owner model is LIVE. A session cannot
be ACTIVE unless its context is LIVE. Context drain first prevents new sessions
and graph epochs, then drains their events outside locks. Session reset is
owner-targeted and cannot reset another session. IDs are never resurrected after
DEAD/CLOSED/RETIRED.

At graph replacement, the old epoch becomes RETIRING and a new epoch may become
current. Completion of the retiring epoch releases **only resources retained by
that old `(ContextId, GraphEpoch)`** and may mark that old epoch RETIRED. It must
not clear, reset, publish, invalidate, decrement, or otherwise mutate current
context/session/graph state. Every completion callback compares the complete
epoch identity; a stale callback has no current-state side effects.

Teardown blocks new device tokens for the target `ModelId`, marks it
TEARING_DOWN, retires its contexts/epochs, and waits **without locks** for their
terminal events. It releases only matching owner records and finally marks DEAD.
A repeated teardown of the same DEAD identity is an idempotent no-op; an unknown
or stale identity is a typed error, never a global sweep.

**Current limitation:** the bool load boundary at `ggml-sycl.cpp:9093` clamps
underflow and treats the outermost `false` as successful; it cannot identify or
roll back a failed nested transaction. This behavior must not be documented as
meeting the target state machines.

### 12.3 Supported lifecycle matrix

| Scenario | Object lifecycle target | Execution target |
|---|---|---|
| Sequential `A -> B -> A` (fresh A object on return) | Required; no semantic state from either prior load may be observed unless content-addressed and owner-safe | Required serially |
| A and B both LIVE | Required, including deduplicated weights with both owners | Same-device overlap is not implied |
| Multiple contexts for one model | Required, with distinct `ContextId` arenas/KV | Same-device overlap only under the execution-lease rule below |
| Multiple server sessions | Required, with `(ContextId, SessionId)` KV attribution | Must not alias another session's reset |
| Different-device execution | Permitted only after all touched process globals are model/context keyed | Event dependencies still required |
| Same-device concurrent inference | **Target is serialization/rejection by an event-held device execution lease**, not overlapping execution | Unsupported today; no optimistic overlap |

There is exactly one top-level exclusive execution token per logical SYCL
device. Its owner is `(device, ModelId, ContextId, GraphEpoch, invocation_id)`.
The graph-compute entry acquires it **before** any context arena mutation or
submission. Every kernel/copy/record/replay submit receives a ref-counted copy;
the final join event retains the root copy. Releasing a child event copy cannot
make the device available: the token becomes free only after the final join is
complete and every copy is gone, including deferred copies and pure-GPU decode
after the host call returns.

Acquisition policy is explicit:

- same owner reentrancy within one top-level invocation returns another copy;
  a nested call with a different invocation or epoch is not reentrant;
- incompatible `try_acquire` returns typed `DEVICE_BUSY` without touching arena,
  queue, epoch, or completion state;
- blocking `acquire` waits only outside all ranked locks and cancellation returns
  typed `CANCELLED`; implementations must not spin while holding partial state;
- a multi-device graph atomically reserves all device IDs in ascending order or
  none. On contention it releases the reservation set, then waits/retries outside
  locks. Each device's submits retain its copy and the one final multi-device
  join event retains all roots;
- error after partial submission still forms/uses a terminal join for submitted
  work. Error before submission releases all roots immediately.

Dropping `g_sycl_graph_compute_mutex` or any host mutex after enqueue does not
release this token. Different contexts may share immutable model weights, but
never mutable KV/RUNTIME/SCRATCH ownership. The test diagnostics must record the
physical device UUID/index for every acquisition so “same device” is asserted,
not inferred from the selector.

### 12.4 Owner-targeted reset, graph invalidation, and teardown

Every reset/clear/free operation must take an explicit owner tuple and match it:

| Resource | Minimum reset key | Required behavior |
|---|---|---|
| Model load scratch | `LoadTxnId` | Clear/rollback only the attempt; committed model state is immutable |
| Weight/cache owner | `ModelId` plus `(slot, SlotGeneration)` during slot migration | Remove only that owner; shared/deduplicated entries survive while any owner/lease remains |
| KV and context arenas | `ContextId` (and `SessionId` for sequence rows) | Never reset every context on the device |
| Recorded graph | `(ContextId, GraphEpoch)` | Clear only the matching epoch; stale clear/callback is ignored/rejected |
| Device execution | terminal event for the exact execution lease | Lease releases only after completion, not after submission |

`sycl_exec_graph_clear_active()` currently clears a supplied backend context, but
model teardown calls an all-device graph-lease sweep because retained graph
handles lack model attribution (`ggml-sycl.cpp:85621-85670`). That sweep is a
known non-conformance: target teardown must not invalidate another live model's
graph. Device-only FIFO ownership of pending KV masks is likewise forbidden.
Whole-zone reset remains a refusal, not a force-free, if any non-target or
in-flight allocation would be affected.

### 12.5 Locking and waiting

The target lock inventory and mandatory order is concrete:

| Rank | Target/current lock inventory | Same-rank tie-break |
|---|---|---|
| L1 | lifecycle/ID/slot/load registry; current `g_sycl_model_slot_mutex` | one process lock |
| L2 | per-device execution-token registry; transitional `g_sycl_graph_compute_mutex` | ascending stable device ID |
| L3 | model/context/session/graph registries; current `sycl_ctx->graph_mutex` and pending-KV registry lock | `(ModelId, ContextId, SessionId, GraphEpoch)` lexicographic, absent fields zero |
| L4 | unified-cache metadata; current per-device `unified_cache::rw_mutex_` and global cache registry lock | ascending device ID, then cache instance ID |
| L5 | arena/zone/chunk allocators, staging/pool locks | device ID, zone enum, then allocation address/ordinal |
| isolated C | completion-queue mutex used only to push/pop completed identity records | never co-held with L1-L5 or another C lock |

Code may skip ranks but never acquire a lower-numbered rank while holding a
higher one. Two same-rank locks require the table's stable key order; pointer
address is not a stable owner key except the final allocator-address tie-break.
The execution-token lock protects acquisition bookkeeping only; it is not held
for device execution. The completion lock protects queue mechanics only; pop a
record, unlock, then compare identities/release resources.

Owner snapshots are taken under L1-L3 and acted on after release. No code may,
while holding **any** listed lock: wait on a SYCL event/queue/future/condition
variable; perform a blocking allocation or device call; invoke a completion or
user callback; destroy a final `mem_handle`, execution-token copy, or arena
backing lease; or call teardown/reset code that can acquire another listed lock.
Event callbacks only enqueue an identity-tagged completion under isolated C and
return. Lock-rank instrumentation must expose acquire/release, wait, callback,
blocking-call, and final-destructor hooks so H8/M7 test every prohibition.

### 12.6 `mem_handle` and event-held leases

Resolving a pointer requires a live `mem_handle`, but lexical handle lifetime is
not enough for asynchronous work. Every submit must create an event lease that
owns copies of all input/output/sidecar/pointer-table handles and the top-level
execution-token copy through the terminal event. Fan-out retains until the join
event; graph recording retains by `(ContextId, GraphEpoch)` through the final
replay event; error/cancel paths retain until submitted work completes. A
completion handler may release only an exact identity/epoch match. No
`sycl::free`, cache eviction, owner reset, slot reuse, or graph replacement may
invalidate those handles first.

A bare `DIRECT` handle is forbidden in any asynchronous event lease because it
has no backing lifetime. It is allowed only for a synchronous ABI scope whose
work completes before return. Async callers must use a lease-bearing handle
(`WEIGHT`/`CHUNK_LEASE`) or pair the DIRECT view with an explicit backing-owner
lease whose range, device, and generation cover the view; the submit validator
rejects a missing/mismatched owner. `ARENA_RUNTIME`, `ARENA_SCRATCH`, and
`ARENA_ONEDNN` event leases must also retain an arena/chunk backing lease and
arena generation, preventing reset/rebuild until the terminal event. An offset
handle without that backing lease is rejected rather than assumed safe.

The current `mem_handle` refcount protects some allocations, but DIRECT and ARENA
shapes, APIs that return no event, and global graph lease clearing do not by
themselves prove this rule. They remain migration sites, not exemptions.

### 12.7 Tier verdict is reporting only

A load's `planned_host_bytes`, `actual_host_bytes`, and tier verdict are immutable
records keyed by `ModelId`/`LoadTxnId`. The verdict may drive logs, telemetry,
and API reporting **only**. Dispatch, placement, allocation, reset, and teardown
must use the placement plan and resolved handles, never a process-global
"current model has host placement" boolean. An unknown/partial/aborted verdict
must report UNKNOWN and cannot silently inherit the previous model's value.
`g_current_model_planner_host_placement` at `ggml-sycl.cpp:9766-9769` is current
load scratch and is not a compliant multi-model routing authority.

### 12.8 Child DAG, path ownership, and acceptance

No child may claim closure from a log-only or grep-only check. The dependency DAG
is normative; in particular graph epoch, execution-token, and allocation event
leases land before teardown consumes them:

```text
nn6z (model/load/slot foundation; owns G1)
  ├──> nlww (context/session ownership)
  │      └──> vbeb (GraphEpoch + top-level device token)
  │             └──> h5m4 (allocation/backing event leases)
  │                    └──> y36c (owner-targeted reset/teardown)
  └──> x3ou (reporting-only tier verdict)

t5nq consumes the lock inventories from nn6z/nlww/vbeb/h5m4 and must precede y36c.
(all implementation children) ──> hcyp final census
```

| Child | Exclusive code-path ownership | Required deliverable/evidence |
|---|---|---|
| `nn6z` | slot/ID registry; model-load begin/nesting/commit/abort; planner publication | checked nonwrapping IDs, side-effect-free slot exhaustion, rollback; H1-H4, H10, **G1**, M1-M3 |
| `nlww` | context/session create/free; KV pending-mask attribution; context arena owner keys | context/session state machines and exact reset keys; H5-H6, G3-G4 |
| `vbeb` | graph epoch registry; top-level per-device token acquisition/copy/join; record/replay epoch plumbing | retiring-epoch isolation, busy/wait/reentrant/multi-device behavior; H7, H11, G5-G6 |
| `h5m4` | submit event-lease payload and completion retention for handles, DIRECT backing, ARENA backing, sidecars/pointer tables | allocation lifetime through final event; H7, H12, G5-G6, M6 |
| `t5nq` | lock-rank declarations/instrumentation and completion queue mechanics only | concrete inventory/tie-break and every no-under-lock prohibition; H8, M7 |
| `y36c` | model/context/session reset and teardown orchestration; cache owner removal | consumes prior identities/events; no graph-token or handle redesign; H3-H6, G2-G4, M4-M5 |
| `x3ou` | tier-verdict record/public reporting and reader audit | no routing/reset readers; H9, M8 |
| `hcyp` | generated census/prose reconciliation only | regenerate at final HEAD and make `--check` clean |

Overlapping legacy work is superseded for lifecycle acceptance as follows; it
may supply prerequisites/history but must not edit a path concurrently with its
new exclusive owner:

| Legacy scope | Superseding owner / boundary |
|---|---|
| `32dg8.15.10` model/context ownership and §5 single-context guard | `nlww` owns context/session keys; `y36c` owns reset/teardown |
| `32dg8.15.12` in-flight handle lifetime | `h5m4` owns allocation/backing event payload; `vbeb` owns device token/epoch |
| `32dg8.15.13` event-returning memory ops | remains a prerequisite API shape; `h5m4` owns lifecycle retention integration |
| current bare-slot ownership/reclaim (`0qlw` history) | `nn6z` owns generated identity; `y36c` only consumes it |
| current all-device graph cleanup (`2wv5` history) | `vbeb` owns epoch attribution; `y36c` removes the sweep after that lands |
| current load scratch reset (`k7b0` history) | retained as mitigation until `nn6z` transaction rollback supersedes it |

Cross-child changes require an explicit handoff commit from the exclusive owner;
no duplicate “temporary” token, epoch, reset, or event-lease implementation is
permitted.

### 12.9 Exact verification and mutation matrix

The implementation children must add the named tests below. Names are **target
test names, not claims that they exist now**.

| ID | Class | Exact command | Required assertions |
|---|---|---|---|
| H1 | host | `ctest --test-dir build -R '^sycl-lifecycle-load-txn$' --output-on-failure` | nested success commits exactly once |
| H2 | host | `for c in inner-failure wrong-txn depth-underflow cancel rollback-idempotence; do ./build/bin/test-sycl-lifecycle-load-txn --case "$c" || exit 1; done` | each case poisons/aborts, publishes no LIVE owner; idempotence case proves one free/callback/generation change |
| H3 | host | `ctest --test-dir build -R '^sycl-lifecycle-multi-model$' --output-on-failure` | A and B LIVE; B load/abort cannot change A |
| H4 | host | `ctest --test-dir build -R '^sycl-lifecycle-multi-model$' --output-on-failure` | sequential A→B→A gets fresh IDs/generations and identical A plan |
| H5 | host | `ctest --test-dir build -R '^sycl-lifecycle-owner-reset$' --output-on-failure` | reset/teardown touches only target model/context/session/epoch |
| H6 | host | `ctest --test-dir build -R '^sycl-lifecycle-owner-reset$' --output-on-failure` | stale slot generation and stale/retiring graph callback cannot mutate current state |
| H7 | host | `ctest --test-dir build -R '^sycl-lifecycle-event-lease$' --output-on-failure` | mocked incomplete event retains every handle and top-level token through final join |
| H8 | host | `ctest --test-dir build -R '^sycl-lifecycle-lock-order$' --output-on-failure` | inversion/tie-break plus wait, callback, blocking call, final handle/token destruction under locks fail |
| H9 | host | `ctest --test-dir build -R '^sycl-lifecycle-tier-reporting$' --output-on-failure` | changing verdict changes report only, never route/reset |
| H10 | host | `./build/bin/test-sycl-lifecycle-load-txn --case slot-exhaustion --case id-overflow --case generation-overflow` | 33rd slot returns `SLOT_EXHAUSTED` before LOADING with registry snapshot unchanged; IDs never wrap; exhausted slot retires |
| H11 | host | `./build/bin/test-sycl-lifecycle-event-lease --case reentrant --case busy --case wait-cancel --case multi-device --case final-join` | exactly one token/device; reentrancy tuple, no-side-effect busy, unlocked wait, all-or-none sorted multi-device, final join |
| H12 | host | `./build/bin/test-sycl-lifecycle-event-lease --case bare-direct-rejected --case direct-with-owner --case arena-backing` | async bare DIRECT rejected; matching backing accepted; ARENA reset blocked until terminal event |
| G1 | GPU | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 GGML_SYCL_LIFECYCLE_TEST_DEVICE=0 ctest --test-dir build -R '^sycl-lifecycle-gpu-sequential$' --output-on-failure` | `nn6z` owner; one process runs A→B→A on logged/asserted same physical device |
| G2 | GPU | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 GGML_SYCL_LIFECYCLE_TEST_DEVICE=0 ctest --test-dir build -R '^sycl-lifecycle-gpu-multi-live$' --output-on-failure` | A remains runnable after B load, injected failed load, and B teardown on same device |
| G3 | GPU | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 GGML_SYCL_LIFECYCLE_TEST_DEVICE=0 ctest --test-dir build -R '^sycl-lifecycle-gpu-context-reset$' --output-on-failure` | clearing context/session 1 preserves context/session 2 KV on same device |
| G4 | GPU | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 GGML_SYCL_LIFECYCLE_TEST_DEVICE=0 ctest --test-dir build -R '^sycl-lifecycle-gpu-context-reset$' --output-on-failure` | unequal `n_ctx`, interleaved slot lifetime, no cross-reset |
| G5 | GPU | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 GGML_SYCL_LIFECYCLE_TEST_DEVICE=0 ctest --test-dir build -R '^sycl-lifecycle-gpu-event-lease$' --output-on-failure` | delayed terminal event returns BUSY/blocks second same-device owner and teardown |
| G6 | GPU | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 GGML_SYCL_LIFECYCLE_TEST_DEVICE=0 ctest --test-dir build -R '^sycl-lifecycle-gpu-event-lease$' --output-on-failure` | retiring epoch releases old resources but cannot mutate replacement epoch |

The lifecycle GPU CMake registrations must require a reproducible
`test-sycl-lifecycle-models` fixture. Fixture A is the existing downloaded
`${CMAKE_BINARY_DIR}/tinyllamas/stories15M-q4_0.gguf`, SHA-256
`66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`.
Fixture B is a byte-identical copy at
`${CMAKE_BINARY_DIR}/sycl-lifecycle-fixtures/stories15M-copy-q4_0.gguf` with the
same verified hash. The duplicate is deliberate: distinct model objects/paths
exercise deduplicated shared-weight ownership. Reproduce fixture setup exactly:

```sh
ctest --test-dir build -R '^test-download-model$' --output-on-failure
cmake -E make_directory build/sycl-lifecycle-fixtures
cmake -E copy_if_different build/tinyllamas/stories15M-q4_0.gguf \
  build/sycl-lifecycle-fixtures/stories15M-copy-q4_0.gguf
printf '%s  %s\n%s  %s\n' \
  66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739 build/tinyllamas/stories15M-q4_0.gguf \
  66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739 build/sycl-lifecycle-fixtures/stories15M-copy-q4_0.gguf | sha256sum -c -
```

Every G CTest registration passes
`--model-a <A> --model-b <B> --prompt "1, 2, 3, 4, 5," --seed 42 --temp 0 --n-predict 8 --device 0`.
The test asserts the logical device equals
`GGML_SYCL_LIFECYCLE_TEST_DEVICE`, records the queue's physical device UUID for
every phase, asserts all recorded UUIDs are identical, and fails if work spills
to a second device. G1/G2 require identical generated token-ID vectors for A1,
B, and A2 (the fixtures are byte-identical); G3/G4 compare each context against
its isolated serial reference. GPU commands run once, serially by the lead
session under repository safety rules; exit 77/SKIP is not a pass. Identity and
owner diagnostics are assertions, not merely log output.

Mutation support is test-build-only: `scoped_lifecycle_mutation` activates one
named hook in one subprocess; production builds contain no environment-controlled
fault path. `tests/test-sycl-lifecycle-mutations.sh build Mx` must (1) run the
listed baseline green, (2) activate the exact hook/patch, (3) require nonzero
from the named test and the exact assertion marker below, (4) destroy the RAII
hook/restart the subprocess, and (5) rerun the baseline green. Thus restoration
is executable, not a manual source revert.

| Mutation | Exact executable command | Hook and temporary patch | Required failing assertion / restored baseline |
|---|---|---|---|
| M1 | `tests/test-sycl-lifecycle-mutations.sh build M1` | `slot_reserve_before_generation`: skip generation increment once | `stale slot generation accepted`; H4/H6 green after restore |
| M2 | `tests/test-sycl-lifecycle-mutations.sh build M2` | `nested_exit_before_commit_guard`: publish nested child once | `nested load committed`; H1 green after restore |
| M3 | `tests/test-sycl-lifecycle-mutations.sh build M3` | `txn_poison_before_outer_decision`: clear poison for each H2 case | `poisoned transaction published LIVE`; H2/G2 negative control green after restore |
| M4 | `tests/test-sycl-lifecycle-mutations.sh build M4` | `context_reset_before_owner_filter`: substitute wildcard owner | `non-target context changed`; H5/G3 green after restore |
| M5 | `tests/test-sycl-lifecycle-mutations.sh build M5` | `graph_clear_before_epoch_filter`: substitute all-device/all-epoch clear | `current/non-target graph changed`; H5/G2/G6 green after restore |
| M6 | `tests/test-sycl-lifecycle-mutations.sh build M6` | `event_lease_before_terminal_join`: drop token, handles and DIRECT/ARENA backing immediately after submit | `resource released before terminal event`; H7/H12/G5/G6 green after restore |
| M7 | `for v in wait callback blocking-call final-handle-destroy final-token-destroy; do for r in L1 L2 L3 L4 L5 C; do tests/test-sycl-lifecycle-mutations.sh build M7-$v-$r || exit 1; done; done` | `lock_probe_before_unlock($v,$r)`: perform exactly that operation while that rank/isolated completion lock is held | exact marker `M7:$v:forbidden-under:$r`; H8 green after every restored subprocess |
| M8 | `tests/test-sycl-lifecycle-mutations.sh build M8` | `tier_report_before_dispatch`: invert route when report verdict flips | `reporting verdict changed dispatch`; H9 green after restore |

The mutation runner fails if the mutant unexpectedly passes, the marker is
absent, an unrelated assertion fires, restoration leaves a hook active, or the
post-restoration baseline fails. The final `hcyp` gate is exact:

```sh
python3 scripts/audit-sycl-static-storage.py --self-test
python3 scripts/audit-sycl-static-storage.py
python3 scripts/audit-sycl-static-storage.py --check
```

The regenerated audit must record current source commit/hash/counts, reconcile
all new lifecycle statics, and contain no unowned mutable model/context/session/
graph state. The historical 5793 census is not accepted as final evidence.

### 12.10 Phased code-path plan (not implemented by this document)

| Phase | Code paths to migrate | Exit criterion |
|---|---|---|
| P0 inventory | slot/load globals, planner scratch, pending KV FIFO, graph clear, cache owner masks | Every mutable site assigned one exclusive owner in §12.8 |
| P1 transactions | model load begin/nested exit/preload/failure; slot/ID registry | `nn6z`, including G1, exhaustion/overflow, and rollback mutations pass |
| P2 context ownership | context/session creation/drain, KV/RUNTIME/SCRATCH reservation, pending masks | `nlww` state-machine and exact-owner tests pass; no teardown sweep yet |
| P3 graph/execution foundation | graph compute, device token, record/replay, retiring epochs | `vbeb` token/epoch tests pass before teardown changes |
| P4 allocation event foundation | memcpy/memset, direct/fallback, persistent-TG, pointer tables/sidecars, DIRECT/ARENA backing | `h5m4` H7/H12/G5/G6 and M6 pass |
| P5 locks then teardown | `t5nq` ranks/completion queue, then `y36c` model/context teardown and cache reclaim | all lock mutations pass; owner-targeted teardown consumes P1-P4 foundations |
| P6 reporting | planner/tier API and all readers | `x3ou` proves report-only behavior |
| P7 final audit | parser census and source-anchor review at final HEAD | `hcyp` regeneration and `--check` pass; all H/G tests green |

No phase may expose a target identity API as supported current behavior before
its owning paths and tests land end-to-end. Compatibility shims must fail closed
and carry the full identity tuple; a new ID wrapped around process-global state
is not completion.
