# SYCL Backend Memory Design (unified cache + mem_handle)

**This is the key design constraint of this fork's SYCL backend. Do not forget it.**

Every GPU, host-pinned, staging, scratch, graph-temporary, KV, oneDNN, and
weight-layout allocation in the SYCL backend flows through **one allocator**
(the unified cache) and is owned by **one lifetime token** (`mem_handle`). No
code in the backend calls `sycl::malloc_device` / `sycl::malloc_host` /
`sycl::free` directly, keys anything by a raw device pointer, or stores a raw
`void*` as the source of truth for an allocation.

This doc is the narrative onboarding for that design. The authoritative,
enforceable version — with the exact allowlist of permitted allocation and
pointer-resolution entry points, and the migration inventory of not-yet-compliant
sites — is
[`docs/design/sycl-canonical-memory-architecture.md`](../design/sycl-canonical-memory-architecture.md).
Code lives in `ggml/src/ggml-sycl/{unified-cache.hpp,unified-cache.cpp,mem-handle.hpp,mem-handle.cpp}`.

## Why

GPU memory on this hardware is scarce and multi-tiered (device VRAM → pinned
host → mmap). Weights, KV, and scratch compete for it, and the cache must be
free to **move an allocation between tiers** (evict a weight to host, promote it
back) at any time. If any consumer held a raw VRAM pointer, that move would
leave it dangling — `DEVICE_LOST` or silent corruption. So the invariant is:

> The cache owns **placement**. The handle owns **lifetime**. A raw pointer is
> only a **transient view**, resolved from a handle for one immediate use
> (a kernel submit, a oneDNN call, a scoped CPU access) and never stored.

## The three primitives

Every memory decision in the backend reduces to one of these three (see the
canonical contract §1 for the formal version):

1. **Planner** (`compute_placement_plan` → `placement_plan`) — the sole
   authority for *deciding where memory lives*. Runs once at model-load time and
   produces a plan covering dense weights, MoE experts, KV cache (per layer),
   and oneDNN scratch, assigning each to a device and a VRAM zone.

2. **Unified cache** (`ggml_sycl::unified_cache`) — the sole *allocator and
   owner* of backend memory. Holds the tiered weight cache (device VRAM / pinned
   host / mmap, LRU eviction), the VRAM arena and its zones, the host pinned
   pool, and all runtime/scratch/KV allocations. Enforces the VRAM budget
   (`min(total*pct, free_at_init)`) and does ref-counted eviction.

3. **`mem_handle`** (`ggml_sycl::mem_handle`) — the *ownership and lifetime
   token*. A lightweight, copyable, ref-counted handle that resolves to the
   current pointer on dereference. Holding a handle guarantees the backing
   allocation cannot be freed or evicted underneath you.

## The one allocation entry point

All runtime/scratch/staging/KV/compute allocation goes through a single
function:

```cpp
// ggml/src/ggml-sycl/unified-cache.hpp
mem_handle unified_allocate(const alloc_request & req);   // <-- use this

struct alloc_request {
    sycl::queue * queue  = nullptr;
    int           device = -1;
    size_t        size   = 0;
    bool          suppress_failure_log = false;
    alloc_intent  intent;                 // role/category/tier hints for routing
};
```

The cache reads `req.intent`, selects a tier (`alloc_tier`: DEVICE_VRAM /
pinned host / …) and a VRAM zone (`vram_zone_id`: KV, WEIGHT, ONEDNN, RUNTIME,
SCRATCH), performs the allocation (arena/zone TLSF sub-allocation or a raw
`sycl::malloc` *inside the cache implementation*), and hands back a
`mem_handle`. The handle's destructor releases the allocation — callers never
call a free function for handle-owned memory.

Typical call site (the pattern you'll see across `binbcast.cpp`, `cpy.cpp`,
`dmmv.cpp`, `convert.cpp`, `compute-buffer-manager.cpp`, `set_rows.cpp`, …):

```cpp
ggml_sycl::alloc_request req{ &stream, device, bytes, false, intent };
ggml_sycl::mem_handle owner = ggml_sycl::unified_allocate(req);   // owns the memory
void * ptr = owner.resolve().ptr;                                  // transient view for this submit
// ... enqueue kernel using ptr, with `owner` kept alive until the work is done ...
// no free() — ~mem_handle reclaims through the cache
```

The older `unified_alloc(req, &alloc_handle)` / `unified_free(handle)` pair
(explicit `alloc_handle`, manual free) still exists and backs the same
machinery; `unified_allocate` is the smart-pointer front that most callers
should use. `alloc_handle::as_mem_handle()` bridges the two.

## Weights: cache-managed WEIGHT handles

Weights aren't allocated ad-hoc — they're materialized into the cache per the
placement plan and handed out as **WEIGHT-kind** `mem_handle`s keyed by
`ggml_sycl_cache_id` (tensor identity), not by pointer. A WEIGHT handle:

- **resolves lazily and re-resolves on staleness.** A single global generation
  counter is bumped whenever a pointer could have moved (evict, promote, flush).
  `resolve()` is a ~3 ns compare-and-return when the generation matches; on a
  miss it calls `resolve_slow()`, which re-queries the cache for the current
  location. This is what lets the cache migrate a weight VRAM↔host transparently.
- **holds a lease.** While the handle is alive it has incremented the cache
  entry's `in_use_count`. **Eviction may only remove entries with
  `in_use_count == 0`.** If the cache can't evict because leases are still held,
  that's a missing release to fix — never force eviction.

Other `mem_handle` kinds: `DIRECT` (raw pointer wrapper for buffers the cache
never moves — always returns its cached pointer), `ARENA_RUNTIME/SCRATCH/ONEDNN`
(views into fixed VRAM zones), and `CHUNK_LEASE` (a raw pointer plus a lease on
its backing arena chunk, so the chunk can't be `sycl::free`'d while the pointer
is in use).

## The rules that fall out of this

These are the practical do/don'ts (the CLAUDE.md "SYCL Memory Ownership" section
is the short form; this is the why):

- **Never store a raw `void*` from the cache.** It becomes dangling the moment
  the cache evicts to host. Hold a `mem_handle` and `resolve()` at point of use.
- **Never key a table by a raw device pointer.** Pointer tables and dispatch
  caches key on the handle's stable identity (`stable_identity_hash`), not the
  transient address. If a kernel ABI table must hold raw pointers, retain the
  corresponding handles for at least the lifetime of the queued work / graph.
- **Keep the handle alive until the work is done** — the CPU thread, SYCL event,
  command graph, or pointer table that uses the allocation must hold the handle
  (or an object that owns one) until it's finished. Async submission outlives the
  enclosing scope; a handle that dies too early frees live memory.
- **Never add forced eviction / forced reap / zone-reset to reclaim memory that
  still has a live handle.** A live allocation at cleanup means a leaked
  reference or stale owner — fix that, don't force the free.
- **Host-resident weights dispatch on CPU, not via GPU "zero-copy."** Feeding a
  host-pinned pointer to a GPU kernel is slower (measured 1.6–2.6×) *and* breaks
  the tier abstraction. Let `resolve()` report residency and route accordingly.

## See also

- [`docs/design/sycl-canonical-memory-architecture.md`](../design/sycl-canonical-memory-architecture.md)
  — enforceable contract: allocator allowlist, pointer-resolution allowlist,
  dispatch router, migration inventory.
- Source: `ggml/src/ggml-sycl/unified-cache.hpp` (allocator + planner types),
  `ggml/src/ggml-sycl/mem-handle.hpp` (handle kinds, resolution, lease
  semantics — the header comments are the primary spec).
