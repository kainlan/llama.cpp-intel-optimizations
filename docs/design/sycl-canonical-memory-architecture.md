# Canonical SYCL Memory Architecture Contract

*Status: in-force as of 2026-04-27. All new SYCL backend code must comply. Migration
tasks for existing non-conforming sites are tracked in `llama.cpp-32dg8`.*

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

Plus a host-pinned zone (`HOST`) for CPU-accessible buffers managed via
`unified_cache_host_zone_alloc` / `unified_cache_zone_alloc`.

**What the unified cache does NOT own:**
- Placement decisions (the planner owns those).
- Kernel dispatch logic (the dispatch router owns that).
- Memory allocated inside tests (see §7).
- ggml buffer allocations at the public `ggml_backend` API boundary (see §7).

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
  call site (see §3).

### 1.4 Dispatch Router

**Current entry point:** `ggml_sycl_compute_forward` in `ggml/src/ggml-sycl/ggml-sycl.cpp:39001`
**Current MUL_MAT entry point:** `ggml_sycl::ggml_sycl_mul_mat_unified` in `ggml/src/ggml-sycl/dispatch.hpp:418`
**Kernel selector:** `ggml_sycl::select_kernel_type` in `ggml/src/ggml-sycl/dispatch.hpp:186`

The dispatch router is the component that, given an op and its operands, selects
and invokes the correct kernel (XMX / ESIMD / MMVQ / MMQ / DMMV / CPU / oneDNN).

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
    const OperationContext &       ctx,
    std::span<const sycl::event>   deps);
```

The router determines residency from `handle.resolve().on_device` and `handle.device()`,
selects the kernel, and returns an event — without the caller performing any
host/device predicate check.

**What the dispatch router does NOT own:**
- Placement decisions (the planner owns those).
- Allocation (the unified cache owns that).
- Pointer resolution (the handle owns that).

---

## 2. Allocator Entry Points (Allowlist)

Only the following functions may allocate SYCL memory. All other code must call
these or be migrated (see §8 for temporary allowlisted sites):

| Function | Purpose | Returns |
|---|---|---|
| `unified_alloc(req, out)` | Primary allocator; routes by zone/tier | `bool` + `alloc_handle` |
| `unified_allocate(req)` | Handle-returning wrapper around `unified_alloc` | `mem_handle` |
| `unified_cache_allocate(size, device, ...)` | Bulk weight slot allocator | internal |
| `unified_cache_zone_alloc(zone, size, ...)` | Named zone allocation | `void *` |
| `unified_cache_host_zone_alloc(size, align)` | Host-pinned zone allocation | `void *` |
| `unified_cache_arena_alloc` | **Deprecated** — migrate to `unified_allocate(..., prefer_vram_zone=SCRATCH)` | `void *` |
| `unified_cache_raw_malloc_device(size, queue)` | Raw `sycl::malloc_device` wrapper — call only from inside `unified_cache` internals | `void *` |
| `unified_cache_raw_malloc_host(size, queue/ctx)` | Raw `sycl::malloc_host` wrapper — call only from inside `unified_cache` internals | `void *` |
| `unified_cache_raw_malloc_shared(size, queue)` | Raw `sycl::malloc_shared` wrapper — call only from inside `unified_cache` internals | `void *` |

**Rule:** `sycl::malloc_device`, `sycl::malloc_host`, and `sycl::malloc_shared` must
never be called outside `unified-cache.cpp`. All other files must use the wrappers
above.

**Deallocation:** `unified_free(handle)`, `unified_free_ptr(ptr, device)`,
`unified_cache_zone_free(zone, ptr)`, `unified_cache_arena_free(ptr)` (deprecated).

---

## 3. Pointer-Resolution Entry Points (Allowlist)

Only the following functions may hand out resolved raw pointers from a handle:

| Function | When to use |
|---|---|
| `mem_handle::resolve()` | Normal dispatch: returns `resolved_ptr{ptr, layout, on_device}` |
| `unified_lookup(key, device, out)` | Look up a weight by cache key without holding a handle |
| `unified_cache::memset(h, val, size, queue)` | Fill memory behind a handle (void, no event returned — **deprecated API shape**; see §5) |
| `unified_cache::memcpy(dst, src, size, queue)` | Copy between handles (void, no event returned — **deprecated API shape**; see §5) |

**Rule:** Callers must not store or pass the `void *` from `resolve()` across async
boundaries. The handle that produced it must remain alive for the duration of any
SYCL work that uses the pointer.

---

## 4. Process-Scoped vs Context/Slot-Scoped State

A single process may load one model (process-scoped) and serve multiple concurrent
inference contexts or server slots (context/slot-scoped). The contract distinguishes:

### 4.1 Process-scoped (model) state

Owned by one model load; immutable after load completes; shared across all contexts:

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
may modify model-scoped globals. Read-only access is safe across threads.

### 4.2 Context/slot-scoped state

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

**Invariant:** `arena_reserve` resets KV + RUNTIME + HOST zones only for the owning
device's cache. It must not reset another context's zones. Until T1 (`llama.cpp-32dg8.2`)
adds explicit context ownership keys, callers must ensure single-active-context per
device.

### 4.3 Multi-user / server concurrency

For multi-server-slot operation:
- Model-scoped globals (§4.1) are read-only after load — safe for concurrent readers.
- Each server slot must have a distinct ggml context and dedicated KV/RUNTIME arena
  reservation. Sharing an arena between slots is **not safe** under the current
  implementation (tracked in `llama.cpp-32dg8.15.10`).
- The `unified_cache_set_graph_compute_active` flag is per-device, not per-context;
  concurrent graph compute on the same device is not supported.

---

## 5. Memory Operation Helpers

Memory operations between handles must use the canonical helpers, not raw
`sycl::queue::memcpy` or `std::memcpy`:

### 5.1 Current API (void, no event returned — pre-canonical)

```cpp
// unified_cache instance methods:
void unified_cache::memset(const mem_handle & h, int value, size_t size, sycl::queue & stream);
void unified_cache::memcpy(const mem_handle & dst, const mem_handle & src, size_t size, sycl::queue & stream);
```

These are the current production API. They do not return events and do not accept
dependency lists. They are correct for use inside synchronous paths but are
**not safe for graph recording** (a graph-recording context must not call `.wait()`
implicitly or make blocking decisions).

### 5.2 Required canonical form (target — tracked in `llama.cpp-32dg8.8` / `llama.cpp-32dg8.15.13`)

The canonical memory op API must:
- Accept `std::span<sycl::event>` dependencies.
- Return `sycl::event`.
- Dispatch H2H, H2D, D2H, same-device D2D, cross-device D2D as separate code paths.
- Be graph-safe (no implicit host-wait during graph recording).
- Be exposed as free functions so callers do not need a `unified_cache` instance.

Until `llama.cpp-32dg8.8` lands, callers may continue to use the instance-method
forms above with the understanding that they are scheduled for replacement.

---

## 6. Single-GPU / Multi-GPU / Multi-Context Invariants

### 6.1 Single GPU

- One `unified_cache` instance per device.
- `compute_placement_plan(...)` called once at model load with `device_id = 0`.
- All zones live on device 0 or host pinned.
- `mem_handle::device()` always returns 0 for weight handles.

### 6.2 Multi-GPU

- One `unified_cache` instance per device (`get_unified_cache_for_device(dev)`).
- `compute_multi_device_plan(...)` called at model load; produces a single
  `placement_plan` with `multi_device = true` and populated `devices`,
  `per_device_vram`, `layer_device`, `expert_device`, `kv_device` maps.
- Each `placement_entry` has `target_device` set explicitly. No entry may have
  `target_device = -1` unless placement is `HOST`.
- A `mem_handle` created by `from_direct(...)` currently sets `device_ = 0`
  unconditionally — this is a known gap (`llama.cpp-32dg8.15.11`). Callers
  using `from_direct` in multi-GPU contexts must pass the correct device until
  the fix lands.
- Cross-device transfers use explicit BCS memcpy (OOQ) between device-pinned
  buffers. Hardware P2P is not supported on current Arc hardware; all transfers
  stage via host.
- The `mem_handle::device()` value is authoritative for which cache instance to
  query. Dispatch must not assume handles from different devices are interchangeable.
- Multi-GPU layout: `layer_device` maps dense layers, `expert_device` maps MoE
  experts. KV cache co-locates with the dense weights of each layer (`kv_device`).

### 6.3 Multi-context / Multi-server slot

- Each llama context that shares a model sees the same `placement_plan` and the
  same WEIGHT zone entries (read-only).
- KV and RUNTIME zones must be reserved separately per context (see §4.2).
- Concurrent inference on the same device is not safe until `llama.cpp-32dg8.15.10`
  delivers explicit context-keyed arena ownership.

---

## 7. Out of Scope

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

## 8. Temporary Allowlists (Migration Inventory)

The following raw allocation or host-residency-check sites are temporarily allowed
pending migration. Each entry names the owning bead and deletion criteria.

### 8.1 Direct SYCL allocation sites (non-`unified_cache` internals)

The full inventory (545 raw SYCL alloc patterns in `ggml/src/ggml-sycl`) is being
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
| Test files under `tests/` | all | — | Allowed permanently (see §7) |

### 8.2 Host-residency predicates (caller-side "is on host?" checks)

78 sites in `ggml/src/ggml-sycl` branch on `ggml_backend_buffer_is_host`,
`ggml_sycl_is_host_resident_weight`, `ggml_sycl_weight_is_planned_on_host`, or
`has_placement_plan`. These are temporarily allowed; migration to handle-based
dispatch is tracked in:

| Owner bead | Scope |
|---|---|
| `llama.cpp-32dg8.9` | Build dispatch router; replace MUL_MAT / MUL_MAT_ID host-predicate branches |
| `llama.cpp-32dg8.10` | Migrate remaining op call sites |
| `llama.cpp-32dg8.15.17` | Build host-fallback coverage matrix, file per-op blockers |

### 8.3 `unified_cache_enabled()` / `weights_evictable()` branches

54 sites branch on `unified_cache_enabled()` or equivalent guards. These are
temporarily allowed as the optional-cache mode is removed:

| Owner bead | Action |
|---|---|
| `llama.cpp-32dg8.4` | Remove optional unified-cache mode branches (`unified_cache_enabled()` checks) |
| `llama.cpp-32dg8.3` | Delete legacy host-weight fallback registration (precondition for T3) |

### 8.4 `mem_handle::from_direct` with implicit device 0

All `from_direct(ptr, layout, on_device)` calls omit device identity. These are
temporarily allowed but must be audited for multi-GPU callers:

| Owner bead | Action |
|---|---|
| `llama.cpp-32dg8.15.11` | Add device-aware DIRECT constructors; define wrong-device policy |
| `llama.cpp-32dg8.7` | Expand `mem_handle` coverage; replace remaining DIRECT escapes |

---

## 9. Cross-References: Child Beads to Contract Sections

| Bead | Title | Contract sections |
|---|---|---|
| `llama.cpp-32dg8.1` | T0 — this document | all |
| `llama.cpp-32dg8.2` | T1 — S1-PRELOAD consumes `placement_plan` entries directly | §1.1, §4.1 |
| `llama.cpp-32dg8.3` | T2 — Delete legacy host-weight fallback registration | §8.3 |
| `llama.cpp-32dg8.4` | T3 — Remove optional unified-cache mode branches | §8.3 |
| `llama.cpp-32dg8.5` | T4 — Planner covers all SYCL memory domains | §1.1, §4, §8.1 |
| `llama.cpp-32dg8.6` | T5 — Unified cache as sole allocator | §1.2, §2, §8.1 |
| `llama.cpp-32dg8.7` | T6 — Expand `mem_handle` into universal handle | §1.3, §3, §8.4 |
| `llama.cpp-32dg8.8` | T7 — Canonical `mem_handle` memory operations | §5.2 |
| `llama.cpp-32dg8.9` | T8 — Unified dispatch router over `mem_handle` operands | §1.4, §3, §8.2 |
| `llama.cpp-32dg8.10` | T9 — Migrate op call sites off raw residency checks | §8.2 |
| `llama.cpp-32dg8.11` | T10 — Multi-GPU validation | §6.2, §8.1 |
| `llama.cpp-32dg8.12` | T11 — Multi-user / multi-context validation | §4.3, §6.3 |
| `llama.cpp-32dg8.13` | T12 — Final audit gates | all |
| `llama.cpp-32dg8.15` | PROOF gate | §1–§8 (all proof P1–P8 invariants) |
| `llama.cpp-32dg8.15.1` | P1 — Prove model-scoped vs context-scoped memory ownership | §4.1, §4.2, §4.3 |
| `llama.cpp-32dg8.15.2` | P2 — Prove multi-GPU `mem_handle` ownership and wrong-device behavior | §1.3, §6.2, §8.4 |
| `llama.cpp-32dg8.15.3` | P3 — Prove in-flight `mem_handle` lease lifetime through SYCL events | §1.3, §3 |
| `llama.cpp-32dg8.15.4` | P4 — Prototype canonical `mem_handle` memory operations | §5.2 |
| `llama.cpp-32dg8.15.5` | P5 — Audit every SYCL allocation site before sole-allocator migration | §2, §8.1 |
| `llama.cpp-32dg8.15.6` | P6 — Prove dispatch-router shape with MUL_MAT and MUL_MAT_ID | §1.4, §3, §8.2 |
| `llama.cpp-32dg8.15.7` | P7 — Prove planner can predict runtime and scratch allocation demand | §1.1, §4, §5 |
| `llama.cpp-32dg8.15.8` | P8 — Prove host-resident fallback coverage for spilled subgraphs | §8.2 |
| `llama.cpp-32dg8.15.9` | P9 — Make 32dg8 implementation beads junior-ready | all |
| `llama.cpp-32dg8.15.10` | P1-FIX — model vs context ownership fixes | §4.1, §4.2, §4.3 |
| `llama.cpp-32dg8.15.11` | P2-FIX — `mem_handle` device identity | §1.3, §6.2, §8.4 |
| `llama.cpp-32dg8.15.12` | P3-FIX — in-flight handle lease lifetime | §1.3, §3 |
| `llama.cpp-32dg8.15.13` | P4-FIX — canonical event-returning memory ops | §5.2 |
| `llama.cpp-32dg8.15.14` | P5-FIX — allocation site inventory | §2, §8.1 |
| `llama.cpp-32dg8.15.15` | P6-FIX — dispatch router vertical slice | §1.4, §3, §8.2 |
| `llama.cpp-32dg8.15.16` | P7-FIX — plan-vs-actual auditor | §1.1, §4, §5 |
| `llama.cpp-32dg8.15.17` | P8-FIX — host fallback coverage matrix | §8.2 |
