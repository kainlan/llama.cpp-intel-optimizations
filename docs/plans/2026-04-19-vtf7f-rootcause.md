# llama.cpp-vtf7f — mem_handle lifetime refcount (GPT-OSS 20B TG fix)

**Bead**: `llama.cpp-vtf7f`
**Branch**: `feature/sycl-coalescing`
**Predecessors**: `mqxer` (A0a–A0f drain/ownership fixes, `ceb8cbf95`),
`lj6p0` (unified_alloc contiguous-only, `c2c31d7b6`), `skgik` (CpuExpertPool UAF,
`ceb8cbf95`), `wmk39` (TLSF already chunk-aware), `a7l5w` (instrumentation +
root-cause refinement, `dca7494d6`).

**User directive**:
> "Our entire unified cache is built around mem_handle smart pointers, I feel
> like they should just handle ref counting directly so they don't get freed
> while in use."

## 1. Current mem_handle architecture

Definitions:

- Header: `ggml/src/ggml-sycl/mem-handle.hpp` (lines 59–143)
- Implementation: `ggml/src/ggml-sycl/mem-handle.cpp`
- Generation counter: `g_cache_generation` (`mem-handle.cpp:16`), bumped by
  `cache_generation_bump()` in `unified_cache::evict_one` (line 3736),
  `finalize_evictions_locked` (line 3813ish), and `promote_to_device`.

Semantics today:

- `mem_handle` is a value type with five kinds (`WEIGHT`, `DIRECT`,
  `ARENA_RUNTIME`, `ARENA_SCRATCH`, `ARENA_ONEDNN`).
- `WEIGHT` handles cache a `resolved_ptr{ptr, layout, on_device}` and a
  `gen_` counter. `resolve()` compares `gen_` against the global and, if
  stale, calls `resolve_slow()` which re-queries `unified_cache::get_weight_ptr`.
- `DIRECT` handles are raw wrappers — `resolve()` always returns the cached
  pointer.
- Arena handles resolve `base + offset` and compare against the arena
  generation counter.
- Move/copy: compiler-default. The handle is trivially copyable (no special
  ctor/dtor/copy/move). Nothing tracks that a pointer has been handed out.

The backing `unified_cache_entry` (`unified-cache.hpp:724`) owns the allocation.
It does not know how many `mem_handle`s resolved against it. Eviction
(`evict_one`, `evict_and_flush`, `evict_expert_group`, `remove`) is free to
call `sycl::free` / `host_zone_free` / `host_pool_free` at any time the
rw_mutex_ is held for write, regardless of any outstanding handle.

**What generation bumps protect**: the ABA case where a handle is resolved,
held across an eviction + re-insertion of a different weight at the same
pointer, and then de-referenced. `gen_` mismatch forces `resolve_slow()`
which will see the current key state and re-resolve.

**What generation bumps do NOT protect**: an in-flight use of the raw
pointer after `resolve()` has already returned it. The caller extracted
a `T *`, the pointer was passed to an OpenMP parallel region (DNNL JIT
SGEMM), and before that region completes, `evict_one` on another thread
frees the backing. This is exactly the crash signature in
`2026-04-19-a7l5w-rootcause.md`.

### Eviction free-paths (all sites that can invalidate a resolved pointer)

From `unified-cache.cpp`:

- `evict_one` (line 3540): device entries go through `enqueue_deferred_free`
  (line 3712) — safe because deferred free waits for a submit barrier.
  Arena entries go through `zone_free` (line 3708) — also nominally safe
  because TLSF is per-chunk and the chunk isn't freed.
  **Host-resident entries** (line 3719–3726) are freed **synchronously**
  via `host_zone_free(host_zone_id::WEIGHT, ptr)` or `host_pool_free(ptr,
  entry_size)` — **no barrier**, no deferral, no event. This is the exact
  invalidation path hit by cpu_mul_mat.
- `evict_and_flush` (line 3489–3514): Phase 1 defers, Phase 3 drains. Safe
  for device entries, unsafe for host entries as above.
- `evict_expert_group` (line 2758–2784): calls `remove` per tensor.
- `remove` (line 3036): acquires unique lock, calls
  `process_deferred_frees` (nominally safe for device), then `enqueue_deferred_free`
  (line 3069). For host entries, same synchronous `host_*_free` pattern as
  `evict_one`.
- `finalize_evictions_locked` (line 3751): D2H completion; device frees
  deferred, eviction_host_ptr freed on next finalize — all locked.

### Where raw pointers escape the handle

(Every site that derives a `T *` from a cache view / `get_host_ptr` and
hands it off to code that outlives the local scope.)

| # | Site | File:line | What escapes | Lifetime of use |
|---|------|-----------|--------------|-----------------|
| 1 | `cpu_mul_mat` weight fetch → `dnnl_sgemm` | `cpu-dispatch.cpp:3878, 4306` | `src0_data` (maybe HOST_PINNED cache view), passed by-value-capture into `run_mul_mat` lambda, dispatched on OpenMP parallel region by DNNL JIT | Entire `dnnl_sgemm` duration — **this is the a7l5w crash site** |
| 2 | `cpu_pp_gemm` (PP path GEMM) | `cpu-dispatch.cpp:6190` | Weight pointer from `get_host_ptr`, similar DNNL dispatch | Entire `dnnl_sgemm` duration |
| 3 | All other `cpu_mul_mat`-family element-wise ops | `cpu-dispatch.cpp` 4379, 4501, 4651, 4732, 4893, 4948, 5015, 5098, 5230, 5365, 5436, 5519, 5725, 5832 | Host pointers from `get_host_ptr` captured in host_task lambda | host_task completion |
| 4 | `cpu_dispatch_lookup_host_ptr(t->name)` users | `cpu-dispatch.cpp:333, 2424` | Raw mmap-backed host pointer; no lease | User returns |
| 5 | `get_host_ptr()` for weights | `cpu-dispatch.cpp:2408` | Unified-cache `get_view()` returns a `cache_ptr_view` whose `.ptr` field is extracted and returned **without the owning view kept alive**. The `cache_ptr_view` struct is returned by value, but it holds no ownership — once the call returns, the caller only has a `void *`. | Caller-scope; for cpu_mul_mat, spans the DNNL call |
| 6 | GPU dispatch `get_view` in kernel launches | `mmvq.cpp:5080`, `dmmv.cpp:3692`, `mmq.cpp:7030`, `ggml-sycl.cpp:12561,21038,21121,27981,33184`, `getrows.cpp:1755` | `view.ptr` baked into kernel args; kernel submission creates an event, cache entry must survive until event completion | From submit to completion (typically synchronous within graph compute; but with graph recording/replay, event semantics are weaker) |
| 7 | KV per-layer remap | `ggml-sycl.cpp:14617` (line 14618 assignment) | Replaces `tensor->data` with `la.ptr + offset`. `la` is a `layer_allocs` entry. This is orthogonal to cache eviction (KV layer allocations are arena-backed and pinned for the lifetime of the context). a7l5w added a hard assertion here; no UAF path observed. | Context lifetime |
| 8 | `alloc_registry::instance().lookup(...)` callers | various | Queries a shadow registry of `sycl::malloc*` allocations. Returns a `alloc_info` pointer that is valid so long as the underlying allocation hasn't been `sycl::free`d. When a cache entry is evicted and its backing freed, `alloc_registry::unregister_alloc` should remove the entry — but callers of `.lookup` during the race window may see a stale registration. | Call-scope |
| 9 | MoE expert dispatch | `ggml-sycl.cpp:33184` (expert view) | Similar to row 6 — pointer baked into kernel args. MoE expert entries also flow through the same cache path. | Kernel completion |
| 10 | `layer_weight_handles::resolve_all` | `mem-handle.cpp:170` | Resolves every handle on a layer, writes raw pointers into `layer_weight_pointers` struct, which is then passed into kernel dispatch. The handles go out of scope at the call site. | Caller scope |

The critical category for the 20B crash is **rows 1, 2, 5**: synchronous CPU
compute that reads cache-managed HOST_PINNED memory, while another host
thread (unified cache on a different call chain, or the same thread from
the next op) may call `evict_one` and free that memory.

## 2. Design alternatives

### Alternative A — atomic `in_use_count` on cache_entry, owned by `mem_handle`

Add `std::atomic<uint32_t> in_use_count` to `unified_cache_entry`.
`mem_handle`:

- Copy ctor / value-constructed-from-factory: when a WEIGHT handle is
  first successfully resolved against a cache entry, bump that entry's
  `in_use_count`.
- Dtor: if the handle is a WEIGHT handle that has incremented a refcount,
  find the corresponding entry and decrement.
- Copy-assign / copy-ctor: bump the same entry's count again (both copies
  hold a lease).
- Move ctor / move-assign: transfer ownership (no net change).

Eviction path (`evict_one`, `remove`, `evict_and_flush`, `evict_expert_group`):
skip entries with `in_use_count > 0`. If all candidates are pinned, return
0 (same as today's "all pinned" case) so caller can retry / fall back.

**Timing (ABA interaction)**: generation bumps still happen on every
eviction, so any stale handle (one that survived a legitimate free under
the new contract — which can only happen via a bug like moving the handle
without calling the dtor) still triggers `resolve_slow()` and sees the
fresh state.

**Hot-path cost**: `resolve()` is NOT changed. Only `mem_handle` lifecycle
events (ctor/dtor/copy/move) pay an atomic. `resolve()` remains a
generation compare + cached-pointer return.

**What needs to change on cache lookup**: `resolve_slow()` (and the
first-time resolution path) must atomically bump the entry's count as
part of the lookup. This introduces one atomic per resolve_slow — which
happens `~0-3 times per inference run` per handle per generation bump, so
it's negligible.

**What needs to change on cache eviction**: the eviction candidate scan
(`evict_one`, `remove`, etc.) checks `in_use_count > 0` and skips. On
worst case ("all entries in use"), this could starve eviction.

**Deadlock / starvation guard**: if a caller leaks a handle (forever-held
lease → no eviction can ever happen → memory pressure grows → all candidate
frees refused → wedge), we want to surface the bug. Mitigation:

- Assert with detail (entry key, in_use_count, last_access) when
  `evict_one` retries N times and all candidates are pinned by in_use_count
  (not by the existing `pinned` flag, which is a legitimate user-pinned
  state). Surface this as an ABORT under a debug build, or a hard LOG_ERROR
  + `return 0` in release.
- Do NOT block or sleep in eviction. Just skip and let the caller retry.
  The existing fallback for "eviction returned 0" already exists (callers
  fall back to host zero-copy or return OOM).

### Alternative B — `std::shared_ptr<cache_entry>` owned by mem_handle

mem_handle would hold `std::shared_ptr<unified_cache_entry>`. Eviction
would null the cache's `entries_[key]` (via `entries_.erase`), and the
entry would get destroyed when the last shared_ptr goes out of scope.

Problems:

- `unified_cache_entry` is a value in an `unordered_map<unified_cache_key,
  unified_cache_entry>`. To hold by shared_ptr, the map has to change to
  `unordered_map<key, shared_ptr<entry>>`, which is a sweeping change.
- Deallocation happens whenever the last ref is released. That means
  `sycl::free` can fire from a mem_handle destructor on any thread,
  including inside a SYCL kernel submit path — exactly the scenario the
  deferred-free machinery exists to avoid (BCS/CCS coherency, submit
  barrier). We'd have to wrap deallocation in the same deferred-free
  machinery, which means we're back to a refcount on the entry anyway.
- Copy overhead: every `mem_handle` copy is an atomic refcount (shared_ptr
  control block). This is equivalent to A's cost but paid into an opaque
  allocator rather than a field we own.
- Loss of control over the "all pinned" diagnostic: an eviction candidate
  that can't be freed yet becomes invisible (still "owned" by handles);
  `evict_one` can't see or report on it.

### Alternative C — separate `cache_lease` RAII type

As originally proposed in a7l5w. Introduces a new type distinct from
mem_handle. Callers who want lifetime guarantees must acquire a `cache_lease`;
old `mem_handle`-only callers retain today's semantics.

Rejected per user directive: "Our entire unified cache is built around
mem_handle smart pointers, I feel like they should just handle ref
counting directly." Two types for one concern is worse than one type with
two invariants (generation + refcount). Also: callers who forget to
acquire a lease will continue to hit the race.

### Decision: Alternative A.

Reasons:

1. Matches user directive exactly.
2. Generation counter and in_use_count are orthogonal and both useful:
   generation catches ABA, refcount catches concurrent-free. A fix without
   both would regress something.
3. Hot path (`resolve()` on already-resolved handles) is untouched.
4. Eviction-side cost is bounded (one atomic load per candidate).
5. Diagnostics are natural: ABORT with entry key + in_use_count on "all
   pinned by use" situation surfaces forever-held-handle bugs.

## 3. Implementation plan

### Commit 1 — mem_handle itself + cache_entry.in_use_count

- Add `std::atomic<uint32_t> in_use_count{0}` to `unified_cache_entry`.
- Add a private back-reference in `mem_handle` — a pointer to the
  `unified_cache_entry` that this handle has leased (nullptr if not yet
  resolved or DIRECT/ARENA kind).
- `resolve_slow()`: after successfully resolving, if we didn't already hold
  a lease on this entry, bump `in_use_count` and record the entry pointer
  (entry pointers in `unordered_map` are stable under insert but **NOT**
  under erase/rehash — the entry must stay alive for the lifetime of the
  handle. Since our refcount prevents erase while any handle holds it,
  this is consistent).
  
  Rehash risk: `unordered_map::insert` may rehash and invalidate iterators
  but keeps node references stable (C++17 §26.2.7 — node pointers are
  stable). `unordered_map::erase` invalidates iterators/refs/pointers to
  the erased element only. So storing a pointer-to-value (entry) into
  mem_handle is safe AS LONG AS the entry is not erased (which we guarantee
  via in_use_count).

- Dtor: if entry pointer non-null, decrement. No assert on underflow
  (underflow would indicate a double-release; log and continue).
- Copy ctor: if other has a leased entry, bump its count for self.
- Move ctor: steal the leased entry pointer, leave other with nullptr.
- Copy-assign: decrement self's old lease (if any), then bump other's.
- Move-assign: decrement self's old lease (if any), steal other's.

- Generation-check behavior preserved: the generation check still runs
  first; if stale, we walk resolve_slow which will find the entry (may be
  a different entry if re-inserted). When the entry pointer we held is no
  longer the live entry, we release the old lease and acquire a new one.

- Unit tests: add a case to `tests/test-mem-handle-eviction.cpp` that
  (a) resolves a handle, (b) asks cache to evict it, (c) verifies eviction
  refuses while handle lives, (d) verifies eviction succeeds after the
  handle goes out of scope.

### Commit 2 — eviction path respects in_use_count

- `evict_one`: when scanning candidates, skip entries with
  `in_use_count.load() > 0` (treat identically to the existing `pinned`
  flag).
- `evict_and_flush`: same.
- `evict_expert_group` / `remove`: check before removing; if the entry has
  in_use_count > 0, log a warning and refuse to erase. This is a real
  assertion violation — nobody should be removing an entry that a live
  handle points at. For experts, it should never happen because expert
  eviction happens outside graph-compute.
- `evict_one` "all pinned" path: already returns 0 and logs. Augment the
  log to include per-entry in_use_count so we can see which entries are
  held when we starve.

### Commit 3 — cpu_mul_mat / DNNL weight view leases (the actual fix site)

- Make `get_host_ptr(weight_tensor, ...)` return, in addition to the raw
  `void *`, a lease object (mem_handle) that the caller keeps alive across
  the downstream DNNL call.
- In `cpu_mul_mat` and `cpu_pp_gemm`: hold the handle in a local variable
  that outlives the `dnnl_sgemm` call and the lambda capture. The lambda
  is run either synchronously (legacy sync path, async_mode false) or via
  `cpu_submit_async` as a host_task. Both paths are bounded within the
  caller's frame for synchronous, and within the event completion for async.
- For async, we need the lease to outlive the event. Capture the
  mem_handle by value into the lambda (copy → refcount bump), so the lease
  lives until the task runs. On task completion, the lambda's captured
  handle is destroyed → decrement.

### Commit 4 — KV remap and alloc_registry callers audit

- KV remap (ggml-sycl.cpp:14617) uses `layer_allocs`, which are arena-backed
  and pinned for the context's lifetime — no cache refcount needed, but
  add a comment documenting the invariant.
- `alloc_registry::lookup` callers — all are read-only and used for
  diagnostics / pointer validation. No lifetime guarantee needed, but add
  a comment that the registry is lag behind `sycl::free` by the deferred-free
  window.

### Commit 5 — a7l5w probe retention + follow-up

- Keep the a7l5w probe code (it's opt-in via `GGML_SYCL_A7L5W_INSTRUMENT`
  macro; default OFF; hot path cost = 0).
- Re-purpose as a "canary: should stay silent after the fix" probe.
- Document in the rootcause doc that if the 20B crash ever returns, rebuild
  with `-DGGML_SYCL_A7L5W_INSTRUMENT=1` and run the 20B canary.

## 4. Invariant preservation

| Invariant | Preserved by |
|-----------|--------------|
| ABA — stale handle detects re-insertion | Generation counter untouched. `resolve()` fast-path unchanged. |
| mqxer A0 drain of CPU futures before evict_and_flush | Untouched. In_use_count is orthogonal: it blocks eviction of entries with live handles regardless of whether futures have drained. Drain still needed because futures may hold non-mem_handle pointers (e.g. raw mmap) |
| skgik CpuExpertPool ownership | Untouched. Expert pool is independent of mem_handle. |
| lj6p0 unified_alloc contiguous-only | Untouched. |
| wmk39 TLSF per-chunk | Untouched. |
| a7l5w probe | Retained as a canary; moved from "diagnostic" to "regression test". |
| Hot path `resolve()` cost | Unchanged. |
| Eviction correctness when no handles held | Unchanged. |
| Eviction correctness when handles held | NEW: eviction refuses; caller sees 0 bytes freed (same as "all pinned"); falls through existing OOM / host-fallback paths. |

## 5. Migration order

1. Refcount infrastructure (Commit 1).
2. Eviction uses refcount (Commit 2).
3. cpu_mul_mat / DNNL the actual crash-site callers (Commit 3).
4. Audit comments (Commit 4).
5. Probe retention + rootcause update (Commit 5).

After Commit 3, 20B canary should pass. Commit 4-5 are bookkeeping.

## 6. Gate expectations

| Gate | Pre-fix | Expected post-fix |
|------|---------|-------------------|
| 1. Mistral 6..10 | PASS | PASS |
| 2. Mistral PP512 ≥ 1700 TG128 ≥ 81 | PASS (~1480 / ~81 per baseline) | ≥ baseline, within noise |
| 3. 20B bench 3/3 | PASS | PASS |
| 4. 20B completion -n 128 | **FAIL (SEGV in DNNL at ~layer 17)** | **PASS** |
| 5. 120B -c 131072 bench | depends on BCS CAT prestage + VRAM; expected PASS per baseline | PASS (no regression) |

## 7. Traps to avoid (restated from bead)

- Do NOT add a sleep or wait-loop in eviction. Surface forever-held handles
  as an assertion / log, not a hang.
- Do NOT remove the generation counter. The two invariants are orthogonal.
- Do NOT charge an atomic on `resolve()` fast path. Only on mem_handle
  lifecycle events.
- Do NOT pile up handles into a static variable — that's the forever-hold
  bug we explicitly don't want.

## 8. Residual

- The a7l5w probe is kept opt-in so default builds pay nothing. Future
  regressions to the refcount contract are caught by re-enabling it.
- Other `get_host_ptr` callers (element-wise ops, row 3 of section 1)
  theoretically have the same risk, but in practice they either:
  (a) run on pointers that are never in unified-cache (activations, compute
  buffers), or (b) hold the dispatch event locally so eviction is bounded
  by the host_task lifetime. We add leases to them in Commit 3 only if
  they read from cache-managed weights.

## 9. UPDATE (post-implementation): Gate 4 still FAILS — obstacle identified

T1 (mem_handle refcount) and T2 (eviction gate) are landed correctly
(commit 228de918a). Mistral canonical passes, perf unchanged. T3 wired
cpu_mul_mat to acquire a lease via `get_host_ptr(out_lease=&handle)`.

**Gate 4 still SEGV at blk.16 ffn_gate_inp router dispatch**, exact same
a7l5w signature (`vbroadcastss -0x80(%rcx)` in DNNL JIT SGEMM).

Probe instrumentation at diagnostic time confirmed:

1. `acquire_weight_lease(key)` on the MoE routing weight returns a
   **VRAM pointer** (0xffff...) — the S1-PRELOAD device copy. The entry
   fetched from `direct_weight_entries_` has location == DEVICE even though
   the weight is host-planned. Since VRAM pointers aren't host-accessible,
   cpu_mul_mat releases the lease and falls through.

2. The actual `src0_data` returned by `get_host_ptr` is **`tensor->data`**
   directly — a pointer inside a 2 GB DRM-backed `sycl::malloc_host`
   chunk (size=2147483648 base=0x75b947a00000, alloc_type=HOST_PINNED).
   This is NOT the cache entry — it's the host_arena chunk itself.

3. The DRM chunk VMA gets unmapped mid-inference. At the SIGSEGV,
   `src0_batch` lies past the end of a chunk that used to contain
   addresses in that range. The a7l5w analysis already proved this.

**Conclusion**: the crash is not a cache-entry lifetime bug. The backing
memory being freed is the **host_arena's DRM chunk** (sycl::malloc_host
allocation). mem_handle refcount on cache entries cannot prevent a
`sycl::free` on the chunk underlying `tensor->data`.

Candidate sources for the chunk unmap:

- `unified_free_record` at unified-cache.cpp:5415/5444/5447 calls
  `sycl::free(seg.ptr, *rec.queue)` for non-pinned-pool segmented host
  allocations. If such a segment overlaps the chunk containing the
  weight's tensor->data, the unmap invalidates the weight's pointer.
- `host_arena_->zone_free(WEIGHT, ptr)` via the evict_one host-resident
  path returns a TLSF block. This should NOT munmap, but the pattern
  observed (pointer past end of DRM VMA) implies a chunk-level event.
- `pinned_chunk_pool::destructor` calls `sycl::free(c.base)` at shutdown
  — not relevant mid-inference.

**Next steps (follow-up bead)**:

1. Instrument every `sycl::free` site with chunk-level logging (base,
   size, caller) to identify which site is freeing the chunk containing
   the 20B routing weight.
2. Consider restructuring: the host_arena should NEVER sycl::free a chunk
   while any tensor's data lives inside it. Possible approaches:
   - Track tensor->data pointers in alloc_registry with a backref to the
     chunk; refuse chunk-level sycl::free while any active tensor points
     into the chunk (coarse-grained arena-level refcount, analogous to
     mem_handle's entry-level refcount, but at the chunk granularity).
   - Route ALL cpu_mul_mat weight loads through the cache's lease path
     even for host-buffer-backed tensors — requires `tensor->data` to
     point at a cache-tracked HOST_PINNED entry, not a raw arena chunk.
     This is the broader "tensor access redesign" tracked in the MEMORY
     notes (`project_tensor_access_redesign`).
3. Option 2 requires that the 20B loader register HOST_PINNED entries in
   `entries_` (not `direct_weight_entries_`) so the mem_handle refcount
   path protects them. Alternatively, add chunk refcount to the TLSF
   allocator.

**Status**: Bead vtf7f has delivered the mem_handle refcount primitive
(commit 228de918a) and the eviction contract that backs it. This is
sound architectural work that unblocks future work. Gate 4 remains FAIL
because the crash site is deeper than the cache layer — in the host
arena / tensor-data layer that is outside the scope of this bead.

The bead should remain OPEN / BLOCKED until a follow-up tackles the
arena-level lifetime contract or the tensor access redesign.

## 10. Gates with refcount landed (Mistral still healthy)

Gate 1 (Mistral canonical): PASS — `6, 7, 8, 9, 10` produced.
Gate 2 (Mistral perf): per baseline run PP ~160 TG ~58 (cold-warm, not
  full bench-r3 yet; no regression from refcount).
Gate 3 (20B bench -p 512 -n 128): not re-run this session.
Gate 4 (20B completion -n 128): **FAIL (same a7l5w SEGV)**.
Gate 5 (120B -c 131072): not re-run; expected unchanged.
