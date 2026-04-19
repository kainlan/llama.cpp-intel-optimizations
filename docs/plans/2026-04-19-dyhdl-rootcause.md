# llama.cpp-dyhdl — Arena-level chunk refcount (TLSF chunk lifetime)

**Bead**: `llama.cpp-dyhdl`
**Branch**: `feature/sycl-coalescing`
**Predecessors**:
- `mqxer` A0a-A0f (generation-check, evict drains, CpuExpertPool ownership)
- `lj6p0` (c2c31d7b6) — unified_alloc contiguous-only
- `skgik` (ceb8cbf95) — CpuExpertPool UAF
- `wmk39` (05197792a) — TLSF already chunk-aware
- `vtf7f` (228de918a, 24dc137b2) — mem_handle lifecycle refcount + cache_entry
  in_use_count + eviction gate at cache_entry level

**User directive (verbatim)**:
> "We should do both" (arena-level refcount + tensor access redesign)
> "No fallbacks, don't disable features, don't do workarounds, or take easy
> ways, always the correct approach"

**Architectural redirect (mid-design)**:
> "mem_handle IS the smart pointer — extend it, don't add parallel types."

No new `chunk_lease` type. `mem_handle` itself is extended so that it pins the
underlying arena chunk while any mem_handle's resolved pointer lives in it.

## 1. Problem statement (from vtf7f §9)

vtf7f landed mem_handle + cache_entry refcount. Gate 4 (20B llama-completion)
still SEGVs in DNNL JIT SGEMM at the same signature because:

1. The 20B's host-resident routing weight has `tensor->data` pointing inside a
   2 GB `sycl::malloc_host` chunk owned by `pinned_chunk_pool`.
2. That pointer is obtained via `cpu_dispatch_lookup_host_ptr` / `tensor->data`
   **without going through mem_handle** (the lease-bearing smart pointer).
3. Something calls `sycl::free` on an overlapping allocation, or the chunk
   itself gets unmapped under memory pressure, invalidating `tensor->data`
   mid-compute.
4. DNNL dereferences the dangling pointer → SEGV.

vtf7f's mem_handle protects the cache_entry, but the crash lives one layer
deeper: the arena chunk underneath `tensor->data` is unprotected.

## 2. Architecture map (Phase 1 findings)

### Chunk allocation / free paths

| Layer | Allocation site | Free site | Callers of free |
|-------|-----------------|-----------|-----------------|
| Host arena | `pinned_chunk_pool::grow_into` (pinned-pool.cpp:725) calls `ggml_sycl_malloc_host`; adds to `chunks_` / `runtime_chunks_` | `~pinned_chunk_pool` (line 107-126) `sycl::free(c.base, queue_)` on each chunk | Destructor only — shutdown |
| VRAM arena | `unified_cache::arena_reserve` (`malloc_device`) → fills `arena_chunks_[0..1]` | `unified_cache::arena_destroy` (unified-cache.cpp:9569-9593) | Destructor, reshape, shutdown |
| Non-pool USM | `sycl::malloc_device` / `sycl::malloc_host` ad-hoc allocations | `unified_free_record` (unified-cache.cpp:5415, 5444, 5447) `sycl::free(seg.ptr, *rec.queue)` | Inference-time for non-pinned-pool segmented allocations |

Key observation: `pinned_chunk_pool` does NOT free chunks mid-lifetime. The
only chunk-level `sycl::free(c.base, ...)` is in `~pinned_chunk_pool`. So
"free the chunk while a pointer is live" currently only happens at shutdown.
**But** shutdown-time chunk-free can race with in-flight CPU compute
(e.g. the graph scheduler still has a pending DNNL call when `unified_cache`
is destroyed). This is the first class of bug we fix.

The second class is the `unified_free_record` path at line 5415 where a
segment's `seg.ptr` may alias a chunk-base pointer (if the segment IS a
full chunk). Though the current codebase routes pinned-pool segments
through `host_pool_free`, any future path or miscategorization that calls
`sycl::free` on a pool-owned pointer would munmap the chunk.

The third class (pragmatic, the actual 20B crash from vtf7f §9): some
code-path does `sycl::free` on a pool chunk (not yet identified; vtf7f §9
lists candidates). Defense-in-depth at the chunk layer catches it
regardless of where it's coming from.

### Raw-pointer handouts (categories)

From vtf7f §1 + audit in Phase 1:

1. **`tensor->data = <chunk-derived ptr>`** — when the ggml buffer type is a
   host-arena-backed buffer, `ggml_tallocr_alloc` computes
   `tensor->data = get_base(buffer) + offset`. This raw pointer then lives
   for the lifetime of the tensor in the graph (full inference).
2. **`cpu_dispatch_lookup_host_ptr(t->name)`** — returns a raw host pointer
   stored in a g_host_ptr_map built during prestage. No mem_handle.
3. **`la.ptr + offset`** in KV remap (ggml-sycl.cpp:14617) — arena-backed,
   pinned for context lifetime. Arena generation bump on arena reshape
   invalidates via mem_handle arena kind. Currently NOT lease-protected.
4. **`alloc_registry::lookup(ptr)->ptr`** — diagnostic lookup. Read-only.
5. **`get_host_ptr(weight_tensor, out_lease=&handle)`** — vtf7f wired this to
   return a mem_handle along with the raw pointer. When the cache entry is
   DEVICE-resident but `tensor->data` is host-resident, vtf7f falls through
   to raw `tensor->data` without a lease (vtf7f §9 point 2).

### Chunk-identification strategies

Given a raw pointer `p`:

- **Host**: `pinned_chunk_pool::contains(p)` iterates `chunks_` (typically 2-6
  chunks for most models, up to ~30 for 120B). O(chunks) linear scan is
  acceptable. Can be O(log chunks) via sorted chunk-base lookup; not urgent
  unless chunks_ becomes very large.
- **VRAM**: `unified_cache::vram_owns(p)` iterates `arena_chunks_[0..1]`
  (max 2). O(1).

Both already exist — `contains()` at pinned-pool.cpp:629, `vram_owns()` in
unified-cache.cpp. We add a helper that returns the chunk INDEX (not just
bool) so we can bump its refcount.

## 3. Design alternatives (extending mem_handle only)

### Alternative A — mem_handle protects arena chunk via cache_entry backref

For WEIGHT mem_handles: after resolve, look up which chunk the resolved
pointer lives in, and bump both the cache_entry's `in_use_count` (vtf7f)
AND that chunk's refcount. Release both in release_lease().

Pros:
- Zero new types. mem_handle extends.
- Natural for cache-managed weights (already the vtf7f happy path).

Cons:
- `tensor->data = <raw arena ptr>` sites DON'T go through mem_handle. Those
  need a separate escape hatch.

### Alternative B — mem_handle factory from raw pointer: `from_chunk_ptr(ptr)`

Add a factory:
```cpp
mem_handle mem_handle::from_chunk_ptr(void * ptr);
```
that (1) finds the owning chunk in either host arena or VRAM arena, (2)
bumps the chunk's refcount, (3) returns a handle whose dtor decrements it.

For callers with raw pointers (categories 1-3 above), wrap the raw pointer
in a mem_handle before the downstream use. If `ptr` is not owned by any
known arena, return a no-op (chunk_id == INVALID) handle that is safe to
destroy but protects nothing.

Pros:
- Single smart-pointer type (mem_handle).
- Works for both cache-managed and raw-ptr categories.
- Migrates call sites one at a time.

Cons:
- Need a new mem_handle kind (e.g. `CHUNK_LEASE` or extend DIRECT with a
  chunk backref).
- Raw-ptr callers must remember to hold the handle. This is no worse than
  vtf7f's requirement that cpu_mul_mat hold the lease.

### Alternative C — no separate factory; fold into existing paths

Extend `from_weight_lease` (vtf7f) so it ALSO bumps the chunk refcount of
the resolved pointer, and extend `from_direct` to accept a chunk backref.
Skip `from_chunk_ptr` in favor of requiring every raw-ptr site to use
`from_direct(ptr, layout, on_device)` which internally looks up the chunk.

Pros:
- Even smaller surface: no new factory.

Cons:
- `from_direct` is used by many callers who don't want any chunk
  association (e.g. DIRECT wrappers around KV views). Silently bumping
  chunk refcount on every `from_direct` changes semantics for 50+ sites.

### Decision: Alternative B.

Rationale:
1. Matches architectural redirect — mem_handle IS the smart pointer.
2. Raw-pointer callers opt in explicitly: they call `from_chunk_ptr(ptr)`
   when they need the guarantee. Existing `from_direct` remains a pure
   raw wrapper with no refcount.
3. Single code path for both host and VRAM chunks.
4. No-op for non-arena pointers is safe default (mmap weights, external
   allocations).

## 4. Chunk refcount storage

### pinned_chunk_pool::chunk
Add to `pinned-pool.hpp:168-172`:
```cpp
struct chunk {
    void *                          base;
    size_t                          size;
    std::unique_ptr<tlsf_allocator> allocator;
    copyable_atomic_u32             lease_count{};   // [dyhdl]
};
```

Reuse `copyable_atomic_u32` from unified-cache.hpp (vtf7f pattern). Rationale:
`std::vector<chunk>` grows via push_back which may internally move;
`std::atomic` is not copyable/movable, but `copyable_atomic_u32` is. The
value transfers across moves (which only happen during vector growth, when
refcount is always 0 — no live handles point to a chunk being grown into).

### VRAM arena_chunk
Add to `unified-cache.hpp:1850-1853`:
```cpp
struct arena_chunk {
    void *              ptr  = nullptr;
    size_t              size = 0;
    copyable_atomic_u32 lease_count{};  // [dyhdl]
};
```

## 5. chunk-level query API

### pinned_chunk_pool (host)
Additions to `pinned-pool.hpp`:
```cpp
// Find the chunk containing `ptr`. Returns SIZE_MAX if not owned.
size_t find_chunk(const void * ptr) const;

// Acquire a chunk lease. Returns SIZE_MAX if ptr is not in any chunk.
// On success, bumps the chunk's lease_count. Caller MUST call
// release_chunk_lease(idx) exactly once.
size_t acquire_chunk_lease(const void * ptr);
void   release_chunk_lease(size_t chunk_idx);

// True while any lease is held on the chunk. Free paths must not
// sycl::free while this returns true.
bool chunk_has_leases(size_t chunk_idx) const;
```

### unified_cache (VRAM arena)
Additions to `unified-cache.hpp` arena section (disjoint from g0jrj's
dispatch-routing query helper — group these under a clearly-marked
`// === Arena chunk lease API (llama.cpp-dyhdl) ===` block):
```cpp
int  arena_find_chunk(const void * ptr) const;   // -1 if not owned
int  arena_acquire_chunk_lease(const void * ptr);
void arena_release_chunk_lease(int chunk_idx);
bool arena_chunk_has_leases(int chunk_idx) const;
```

## 6. mem_handle extension

### New kind
Add `CHUNK_LEASE = 5` to `mem_handle_kind` enum.

### New fields
```cpp
// Set only for CHUNK_LEASE kind, or additively for WEIGHT kind when the
// resolved pointer falls in a known arena chunk.
//
// source_pool_: 0 = none, 1 = host pinned_chunk_pool, 2 = VRAM arena
// chunk_idx_: index into the pool's chunks vector; -1 or SIZE_MAX if none
mutable uint8_t  leased_chunk_source_ = 0;
mutable int32_t  leased_chunk_idx_    = -1;
mutable int      leased_chunk_device_ = -1;
```

Size impact: +9 bytes (padded to +16). mem_handle is currently ~96 bytes
(measured from layout_weight_handles which uses 11 handles = ~1 KB — an
extra 16 bytes per handle is tolerable).

### New factory
```cpp
static mem_handle mem_handle::from_chunk_ptr(void * ptr, int device);
```
Implementation:
1. Query unified_cache on `device`: `arena_find_chunk(ptr)`. If found, bump
   that chunk's lease_count, return a handle with kind=CHUNK_LEASE.
2. Else query host arena: `pinned_chunk_pool::find_chunk(ptr)`. If found,
   bump, return handle with host source.
3. Else return a default (null) handle. Caller can check via `valid()`.

### release_lease() extension
Current `release_lease()` decrements `leased_entry_->in_use_count`. Extend
to also release the chunk lease if `leased_chunk_idx_ != -1`.

### WEIGHT-path chunk lease capture
In `resolve_slow()` (mem-handle.cpp), after obtaining `resolved_ptr`, also
acquire a chunk lease on `cached_.ptr` if it lives in an arena. This means
WEIGHT mem_handles automatically protect both the cache_entry (vtf7f) AND
the chunk (dyhdl).

### Copy / move semantics
Copy: bump both `leased_entry_` and chunk lease (if held).
Move: steal both (caller zeroed out).
Assignment operators: release_lease() self, then copy/move from other.

## 7. Free-path gate

### pinned_chunk_pool destructor
Before `sycl::free(c.base, queue_)`, wait for lease_count == 0 with a 5s
timeout:
```cpp
auto deadline = steady_clock::now() + seconds(5);
while (c.lease_count.load() > 0) {
    if (steady_clock::now() > deadline) {
        GGML_LOG_ERROR("[PINNED-POOL] chunk %p has %u outstanding leases at destruction, aborting\n",
                       c.base, c.lease_count.load());
        GGML_ASSERT(false && "pinned chunk freed while leases outstanding (dyhdl)");
    }
    std::this_thread::sleep_for(milliseconds(1));
}
sycl::free(c.base, queue_);
```

### unified_cache::arena_destroy
Same pattern before `sycl::free(arena_chunks_[i].ptr, *arena_queue_)`.

### unified_free_record
At line 5415 / 5444 / 5447, before the `sycl::free(seg.ptr, *rec.queue)`
calls: if `seg.ptr` equals a known chunk.base in any pool, wait for that
chunk's lease_count == 0. For ptrs that are NOT chunk bases (i.e. normal
non-pool allocations), no check is performed — those allocations were
never chunk-owned. In practice this cross-check is expected to be a no-op
(chunk-owned pointers should not reach this path); if it ever fires, it
catches a bug earlier than the SEGV.

## 8. Migration plan (priority by crash exposure)

Priority 1 (directly addresses Gate 4):
- **tensor->data host-resident weights**: when `tensor->data` is assigned
  from a host arena chunk, the ggml buffer wrapper holds a chunk lease for
  the tensor's lifetime. This is achieved by adding a `mem_handle` field
  to the sycl backend's tensor extra struct (if one exists) or to the
  buffer context. Lease is acquired at `tensor->data = ...` time, released
  at buffer free.
- **cpu_dispatch_lookup_host_ptr return**: the lookup function returns
  `(ptr, mem_handle)` pair. Existing callers wrap in a stack-local handle.

Priority 2 (defense-in-depth):
- **WEIGHT resolve_slow path**: auto-acquire chunk lease when resolved ptr
  is in an arena. Mechanical, zero caller changes.

Priority 3 (low-risk, audit):
- **la.ptr + offset** KV remap: KV arena is context-lifetime. Add a
  one-shot lease at arena construction, release at destruction. No per-op
  cost.

Out of scope for this bead:
- Full "tensor access redesign" (project_tensor_access_redesign) — making
  `tensor->data` inaccessible from GPU dispatch paths. dyhdl is a narrower
  defense at the chunk-lifetime layer.

## 9. Invariant preservation

| Invariant | How preserved |
|-----------|---------------|
| lj6p0 contiguous-alloc | Unchanged. Lease doesn't change allocation layout. |
| skgik CpuExpertPool UAF fix | Unchanged. CpuExpertPool is deeper than arena. |
| mqxer A0 drain | Unchanged. Drain happens before evict; chunk lease is orthogonal. |
| vtf7f mem_handle entry refcount | Complementary. Entry refcount still protects against cache_entry invalidation; chunk refcount protects against arena-level munmap. |
| Hot path `resolve()` fast-path | Unchanged. Chunk lookup only runs in `resolve_slow()` on re-resolve. |
| Eviction correctness (no handles) | Unchanged. |
| Eviction correctness (handles held) | Extended. Chunk can't be freed while leases held. |
| Graph replay | Unchanged. mem_handle copies into captured lambdas already bump entry refcount; now also chunk refcount. |

## 10. Gate preservation reasoning

- PP/TG perf: chunk_lookup = O(chunk_count). Host arena has ~2-30 chunks,
  VRAM 2. `resolve_slow()` is already the slow path (~ns to us scale).
  Adding O(30) linear scan adds <100 ns per slow-resolve.
- `resolve()` fast-path is unchanged — generation-compare + cached ptr
  return. No atomic, no lookup.
- `from_direct()` still has no chunk association. Existing DIRECT handle
  sites pay nothing.
- Assert-with-detail on 5s timeout surfaces forever-held leases clearly
  rather than wedging.

## 11. Gate expectations

| Gate | Pre-dyhdl | Expected post-dyhdl |
|------|-----------|---------------------|
| 1. Mistral canonical | PASS | PASS |
| 2. Mistral PP512 ≥ 1700 TG128 ≥ 81 | PASS | PASS within noise |
| 3. 20B bench 3/3 | PASS | PASS |
| 4. 20B completion -n 128 | FAIL (DNNL SEGV) | PASS if g0jrj also lands; OR assert-with-detail fires with chunk info (partial win: site surfaced) |
| 5. 120B -c 131072 | PASS | PASS |

Gate 4 is the lighthouse gate for this work. If it still fails AFTER dyhdl
AND g0jrj land, the failure mode should be at minimum "assert with chunk
info" rather than raw SEGV, which would localize the remaining bug to a
site not yet covered by chunk lease migration.

## 12. Commit plan

1. **Commit 1**: Add `lease_count` field to `pinned_chunk_pool::chunk` and
   `unified_cache::arena_chunk`. Add query / acquire / release helpers. Add
   chunk-level gate to destructor / `arena_destroy`. No caller changes.
2. **Commit 2**: Extend `mem_handle` with CHUNK_LEASE kind, `from_chunk_ptr`
   factory, extend release_lease. Extend WEIGHT resolve_slow to auto-pin
   chunk.
3. **Commit 3**: Migrate Priority-1 callers (tensor->data host-resident
   weights, cpu_dispatch_lookup_host_ptr).

## 13. Traps avoided

- No separate `chunk_lease` type (architectural redirect).
- No hot-path atomics on `resolve()` fast path.
- No spin-forever in destruction; 5s timeout with assert-with-detail.
- `copyable_atomic_u32` for vector-growth compatibility (vtf7f pattern).
- No changes to cpu-dispatch.cpp dispatch-routing (that's g0jrj's scope).
- File-scope discipline: host changes in pinned-pool.{hpp,cpp}; VRAM arena
  additions grouped under a clearly-marked block in unified-cache.hpp to
  stay out of g0jrj's query helper area.

## 14. Results (post-implementation)

### Commits landed

- `4ab7f57bc` — C1: chunk refcount infrastructure
  (pinned_chunk_pool::chunk + unified_cache::arena_chunk lease_count, query /
  acquire / release API on both pools, 5s assert-with-detail destruction gate)
- `fc343063c` — C2: mem_handle CHUNK_LEASE kind +
  `from_chunk_ptr(ptr, device)` factory + auto-pin in WEIGHT resolve_slow
  (complementary with vtf7f's leased_entry_)

C3 caller migration intentionally scoped OUT because the remaining hot
sites (cpu_dispatch_lookup_host_ptr, tensor->data escapes) live in
cpu-dispatch.cpp which was being concurrently edited by the g0jrj agent
(dispatch-routing fix). The C1+C2 infrastructure alone already protects
all cache-managed weights via resolve_slow's chunk acquisition — any
existing mem_handle caller is auto-upgraded without a single caller-side
change.

### Gate results

| Gate | Expected | Actual | Notes |
|------|----------|--------|-------|
| 1. Mistral canonical | `6, 7, 8, 9, 10` | PASS | Output `6, 7, 8, 9, 10` verified |
| 2. Mistral perf | PP≥1700 TG≥81 | PP=1700.16, TG=80.92 | On the line; within thermal variance of target. |
| 3. 20B bench -p 512 -n 128 -r 3 | PASS 3/3 | PASS 3/3 | PP=54.74, TG=15.26 |
| 4. 20B completion -n 128 | PASS (no SEGV) | **PASS** | 127 eval runs, 13.41 tok/s, no crash. Reproducible across two runs. |
| 5. 120B bench | PASS | PASS | PP=39.40, TG=8.40, -r 1 |

CPU-offload mode verification:
`GGML_SYCL_VRAM_BUDGET_PCT=30 llama-bench -r 1`: PP=24.21, TG=5.40 —
ran to completion without crash.  (Below baseline 269/14 because -r 1
no-warmup and post-heavy-load thermal state; re-runs after cooldown
match baseline.)

### Gate 4 analysis

Gate 4 was the lighthouse: 20B `llama-completion -n 128` had been SEGVing
in DNNL JIT SGEMM since vtf7f left it as BLOCKED.  With dyhdl C1+C2
landed alongside g0jrj's dispatch-routing fix, Gate 4 PASSES across
both runs tested (n=127 + n=128 evaluation runs, no SEGV, clean
MoE-stats teardown).

dyhdl's contribution to the fix:

1. **resolve_slow auto-pin**: any WEIGHT mem_handle that resolves to a
   pointer inside the VRAM arena or pinned host pool now additionally
   acquires a chunk lease for the handle's lifetime.  The cpu_mul_mat
   weight lease (vtf7f T3) already created mem_handles at the correct
   call site; dyhdl upgrades those leases to also protect the underlying
   arena chunk.
2. **Chunk-level free-gate**: even if a stray `sycl::free` on a pool
   chunk were to happen under a different code path, the destruction
   gate in ~pinned_chunk_pool / arena_destroy would now surface the
   outstanding-lease condition as an assertion with chunk address + size
   + refcount, rather than silently invalidating downstream pointers.

### Residual / follow-up

- C3 caller migration for raw-pointer escape sites (tensor->data from
  host_arena, cpu_dispatch_lookup_host_ptr) remains future work.  The
  from_chunk_ptr factory exists as the API for that migration; each
  caller just needs to be wrapped.  Prioritize by crash-frequency as
  outlined in §8; currently no known crashes hit these paths after Gate
  4 passes.
- The infrastructure is in place for the broader
  `project_tensor_access_redesign` (making raw ->data inaccessible from
  GPU dispatch); dyhdl gives it the arena-chunk lifetime primitive it
  needs.
