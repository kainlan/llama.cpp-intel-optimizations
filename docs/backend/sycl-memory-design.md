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

## Epoch-refcounted transient zones (llama.cpp-2757 / iiff Option C)

**Background.** `llama.cpp-iiff` set out to delete the SYCL backend's bulk
zone resets (`unified_cache_zone_reset`, `unified_cache_host_zone_reset`,
`unified_cache_reset_scratch_pool`) outright, on the owner's ruling that
"memory is freed only when references reach zero; the zones should not reset
at all." `llama.cpp-37ba`'s audit (see its tracker comments) found that
mandate cannot be executed as a blanket removal: the Phase-0 escape
inventory's "0 live at reset" reading proves the *registry* is empty, not
that the *physical bytes* were ever individually returned to the allocator.
For a subset of zones — the VRAM scratch bump pool covered here, plus host
SCRATCH/STAGING and the oneDNN scratch pointers covered by later steps — the
codebase's own comments say outright that the bulk reset is the *only*
reclaim path (`return_scratch()`: "we don't actually free individual
allocations"; `unified_free_record()`: "SCRATCH/STAGING host zones:
reset-only by design — freed by host_zone_reset()"). Deleting the reset for
those zones without first giving them a real per-allocation release path
would not produce correctness — it would produce the exact unbounded memory
growth this document's "never add forced eviction ... to reclaim memory that
still has a live handle" rule exists to prevent, just approached from the
opposite direction (no reclaim at all, rather than reclaiming too eagerly).

**The design: keep the bump allocator, refcount the epoch, rewind on zero.**
Rather than replace the allocator or replace the reset, give each reset-only
zone (or pool) a small ring of equal-sized **regions**, and a **live-handle
counter per region** — the region's *epoch*:

- **Each region is sized at the caller's full requested capacity, not a
  fraction of it.** `reserve_scratch_pool(pool_bytes)` gives every region
  `pool_bytes` (total footprint = `kScratchPoolRegionCount × pool_bytes`),
  rather than splitting `pool_bytes` across the ring. This was a deliberate
  fix during step 1's review: `pool_bytes` is the sizing contract callers
  already plan and unit-test against as one epoch's full capacity (e.g.
  `unified_cache_reserve_moe_q8_1_scratch()`'s Q8_1 demand sizing) — dividing
  it by the region count would silently halve what a caller asked for the
  moment a production caller exists, turning "ring adds rotation headroom"
  into "ring steals half the requested capacity." The ring is *additional*
  memory bought for the ability to linger a still-live epoch, not a
  reslicing of the capacity the caller already sized for. The tradeoff is
  paid explicitly, as a larger real allocation, rather than paid silently as
  reduced usable capacity.
- The bump allocator itself is unchanged: allocation is still a lock-free
  `fetch_add` on an offset, same as before.
- Every allocation from a region increments that region's counter; every
  release (`return_scratch()` and its future host-zone/oneDNN equivalents)
  decrements it. This is the literal reading of the owner's ruling — "memory
  is freed only when references reach zero" — applied at **region
  granularity** rather than per-byte-range, which is exactly the *RC
  regions* pattern from Gay & Aiken (PLDI'01): a region is freed as a whole
  once its reference count returns to zero, giving O(1) release cost with no
  per-object bookkeeping. 4 bytes (or here, one `std::atomic` counter) per
  region is the whole overhead.
- At the point the old code called the bulk reset (still the same call
  site — see Migration below), the region's counter decides the outcome:
  - **counter == 0**: every allocation from this epoch was already
    released. Rewind the region's offset to 0 in place. This *is* the
    reset now — a consequence of the last release reaching zero, firing at
    the same instant the old unconditional reset used to fire, with
    identical behavior in the steady-state case (which the Phase-0 audit
    proved is the overwhelming majority: single-model workloads showed 0
    live at every measured boundary).
  - **counter != 0**: something outlived this epoch. The region is left
    **untouched** — any pointer still held into it stays valid, because
    nothing physically happened to its backing memory — and the *next*
    region in the ring becomes current for the new epoch. This is the
    **VMA linear-allocator pattern** (`vmaCreateVirtualBlock` release-all,
    or the simpler "linear pool" idiom): reset normally means "rewind to
    the start," but a still-referenced allocation forces a fresh pool
    instead of corrupting the old one.
  - **every region in the ring is still live**: refuse the rotation,
    loudly, in exactly the same style as the existing `host_zone_settle()`
    / `zone_settle()` "refusing ..." guards (named `host_zone_reset()` /
    `zone_reset()` before `llama.cpp-37ba`'s naming split) — never wrap
    onto a region that is still referenced, which would be reclaiming a
    live handle by another name.
- The assert-only Phase-0 audit machinery (`zone_reset_audit_*`,
  `zone_audit_site_visit`) is unchanged and is exactly what names the
  leaking site when a region lingers: the audit already distinguishes
  "visited and clean" from "visited and live," and a lingering region is
  visible in its `visits_with_live` / cohort inventory the same way a
  registry-based zone's live entries are. No new instrumentation is
  needed — the region counter *is* what the audit reads at that boundary.
- **Deferred GPU work — event-retained epochs.** A region can outlive its
  graph not because of a bug but because its last consumer is still an
  in-flight SYCL event (a kernel that hasn't completed, an async copy still
  draining). The existing `retain_handles_until_event()` mechanism is the
  hook for this: an epoch whose only remaining "release" is a pending event
  registers itself there instead of releasing synchronously, and the region
  decrements (and becomes reclaimable) when the event actually completes.
  This is the same shape as a GPU frame-arena recycled on a fence in a
  double/triple-buffered renderer — the region is a "frame," and it can't be
  reused until the GPU is provably done with it, which is exactly why the
  ring needs a *minimum* of two regions (current + linger) rather than one.

**The failure-mode inversion this produces is the point, not a side
effect.** The scheduled reset's failure mode was silent, forced
invalidation: it reclaimed a zone's memory on a clock, and a raw pointer
still aliasing that zone became a use-after-free the instant the reset fired
— the exact bug class (`llama.cpp-oze0`, `skgik`, `mqxer`, `FaultLevel=4`)
that motivated this epic in the first place. Under epoch refcounting, the
same escape produces the opposite failure mode: the memory is never
force-freed out from under a live pointer (it lingers), and the escape shows
up as a WARN in the audit/rotation-refusal path — a diagnosable alarm
instead of a silent corruption. This is strictly the tradeoff the parent
epic asks for: "an abort is a bad outcome that tells you; a leak is a bad
outcome that does not" — and epoch refcounting turns the leak into a *told*
outcome, because a lingering region is observable (it shows as still-live in
the audit) rather than truly unbounded and invisible growth.

**The policy generalizes; the implementation, deliberately, does not.** What
step 1 (below) actually validated is a *policy*: per-epoch live counter,
rewind-on-zero, rotate-on-live, refuse-on-exhaustion, never force-reclaim.
That policy applies unchanged to every reset-only population this epic
covers. `scratch_pool_region` — a contiguous ring of equal-sized regions
addressed by `off`, located by address-range membership in
`scratch_pool_region_of()` — is not that policy; it is one *shape* the
policy can take, and it is specific to a single contiguous bump-pool
allocation. The other two reset-only populations this document's migration
order covers have different shapes, and forcing them through
`scratch_pool_region`'s shape would be wrong, not just inconvenient:

- **Host SCRATCH/STAGING zones (step 2).** These are TLSF-arena-backed, with
  every allocation already individually registered — `host_zone_settle()`
  (named `host_zone_reset()` before `llama.cpp-37ba`'s rename; see "Why"
  above and the canonical contract) walks `g_runtime_alloc_registry`
  per zone, not a bump offset. There is no single contiguous span to carve
  into address-range regions and no `off` to rewind: the epoch tag belongs
  on the *registry record* for each allocation (an epoch id alongside the
  existing `alloc_id`/`cohort`/`role`/`category` fields), and "rewind" means
  something different for a TLSF allocator than it does for a bump pointer.
- **oneDNN scratch pointers (step 3).** `onednn_weights_scratch_` and
  `onednn_activations_scratch_` are exactly two named pointers, not a pool of
  interchangeable allocations at all. Epoch tracking here is a live count
  *per pointer* (or, equivalently, folding the same counter into the
  `mem_handle` these pointers should hold once converted off the raw-pointer
  pattern this document's earlier section already flags) — there is no ring
  to rotate through and no address-range lookup to perform.

A shared `epoch_region`-style struct is therefore **deliberately not
extracted** out of step 1's implementation. Sharing a struct across three
populations that disagree on "where do allocations live" and "how is an
allocation's region found" would either force two of the three into an
address-range model they do not have, or bloat the struct with fields only
one shape uses. What step 2 and step 3 inherit from step 1 is the *policy*
above, expressed in whatever data structure each zone's existing allocation
tracking already uses — not `scratch_pool_region` itself, and not a common
base type it should be refactored into.

**Migration order (each step lead-verified on hardware before the next).**
llama.cpp-37ba's per-symbol audit is the map of which reset-only populations
exist and how large their blast radius is if converted incorrectly;
these steps convert them from smallest/safest to largest/most load-bearing:

1. **VRAM scratch bump pool** (`llama.cpp-2757`, this doc's section origin) —
   `unified_cache::get_scratch()` / `return_scratch()` /
   `scratch_pool_epoch_boundary()` (named `reset_scratch_pool()` before
   `llama.cpp-37ba`'s rename) in `unified-cache.cpp`. Chosen first because it
   has no production caller
   today (only `tests/test-sycl-moe-q8-scratch.cpp` exercises it) — the
   mechanism is validated on the lowest-stakes zone before it is trusted with
   a zone that real inference traffic depends on.
2. **Host SCRATCH/STAGING zones** (`llama.cpp-lbm3`) — the zones
   `unified_free_record()` documents as "reset-only by design," with real
   production callers (MoE CPU-expert pools, staging buffers, `get_rows`
   indices). This is where the mechanism has to handle actual escape
   cohorts, not just a test harness.
3. **oneDNN scratch pointers** (`llama.cpp-67c2`) — `onednn_weights_scratch_`
   / `onednn_activations_scratch_`, which today are raw pointers managed
   entirely outside `mem_handle` (no registry entry at all), converted to
   real epoch-tracked handles so the existing `zone_boundary_check(vram_zone_id::ONEDNN)`
   call site (`zone_reset(vram_zone_id::ONEDNN)` before `llama.cpp-37ba`'s
   rename) in the scratch-reservation path stops being the *sole* reclaim
   mechanism for their own previous allocation.
4. **Reconcile the reset call sites** (`llama.cpp-37ba`) — originally planned
   as literal deletion of the reset functions, keeping only the boundary
   asserts. Executed instead as a per-symbol audit once steps 1–3 landed: see
   "Step 4: reconciling the reset call sites" below for what that audit
   found. Summary, because it changes what "done" means for this step: every
   remaining call site is *already* a provable no-op for the zones this epic
   targets (steps 1–3's own conversions made the reclaim happen elsewhere, so
   the call is now the assert Phase 3 asks for, not a scheduled reclaim) —
   there is nothing left to delete without also deleting the observability
   Phase 3 requires keeping. The zones with a call site that is *not* a
   no-op (KV, RUNTIME) were never in this epic's audited scope to begin with.

See [`docs/design/sycl-canonical-memory-architecture.md`](../design/sycl-canonical-memory-architecture.md)
for the enforceable allocator/pointer-resolution contract this mechanism
must continue to satisfy — in particular, a region's bump pointer is never
exposed as a stored raw pointer outside the immediate `get_scratch()` /
`return_scratch()` pair, matching the "raw pointer is only a transient view"
rule above.

### C2 implementation: host SCRATCH/STAGING zones (`llama.cpp-lbm3`)

Step 2 converts `unified_free_record()`'s "SCRATCH/STAGING host zones:
reset-only by design" fall-through — every SCRATCH allocation with
`role != EXPERT_STAGING` and every STAGING allocation with
`cohort != staging_buffer_pool` — the same defect class step 1 fixed for the
VRAM scratch bump pool, but on TLSF rather than a bump pointer. This
subsection settles the three design questions the ticket posed, and states
explicitly which parts of step 1's *implementation* (not its policy) do not
carry over, per the "policy generalizes; the implementation, deliberately,
does not" rule above.

**Q1 — what is "rewind" for a TLSF-backed zone, and is release batched or
immediate?** Immediate, per-record `host_zone_free()` at the instant a
`mem_handle` releases — not batched to "epoch death" the way a candidate
design might defer individual frees until a zone's last outstanding
allocation clears. The two options were weighed explicitly:

- **Per-record free at release** (chosen). Every SCRATCH/STAGING allocation
  now takes the same `host_zone_free()` path the WEIGHT/KV/EXPERT_STAGING/
  staging_buffer_pool carve-outs already used pre-C2 — the fix is simply
  widening that carve-out to cover the rest of the population, in a third
  `unified_free_record()` branch kept textually separate from the untouched
  carve-out — the existing WEIGHT/KV/EXPERT_STAGING-in-SCRATCH/
  staging_buffer_pool-in-STAGING code path is left exactly as it was, and the
  new population-widening logic lives entirely in its own branch alongside it,
  rather than folding the two together.
- **Batched-at-epoch-death** (rejected). This would mirror step 1 more
  literally — hold released allocations until a whole epoch's live count
  reaches zero, then free them as a group — preserving whatever locality
  bump-pointer-style batching buys.

The deciding fact is structural, not a performance tradeoff: **step 1's bump
pool needs batching/rotation because a live allocation blocks the *offset*
behind it** — rewinding while anything is still outstanding would hand out
already-referenced bytes again, so C1 opens a fresh region rather than touch
the live one. **TLSF has no such conflict.** A new `host_zone_alloc()` call
is satisfied from whatever the zone's free list currently holds, completely
independent of which allocations from which epochs are still outstanding
elsewhere in the same zone. There is no address range for a live allocation
to "block," so there is nothing for batching to protect. Given that, batching
only adds a deferred bookkeeping structure (which released-but-not-yet-freed
records belong to which epoch) with no corresponding benefit — and per
`llama.cpp-2757` comment `c-6ngo`, per-call TLSF free is ~350 ns, noise
against graph execution time, and these zones do not fragment under many
small frees (9 of 11 zone/capture pairs measured `largest_free`
`first == last == min`). Per-record free is simultaneously the purer literal
reading of "freed only when references reach zero" *and* the simpler
implementation — the two considerations point the same way.

One consequence follows directly: because release already reclaims
unconditionally, `host_zone_settle(zone)`'s prior bulk
`host_arena_->zone_reset(zone)` call is retired for SCRATCH/STAGING, not
merely gated. In the clean case (`live_allocations == 0`) there is nothing
left in the zone for a bulk reinitialization to reclaim — every byte was
already individually returned by the record that used to occupy it.
"Rewind-on-zero" for TLSF is therefore the empty action, not a rewritten
one: `host_zone_settle()`'s remaining job is exactly the refusal check it
already had (`live_allocations > 0` → refuse, unchanged), which is now its
*entire* job for these two zones, reached through `host_zone_boundary_check()`
(`llama.cpp-37ba`'s naming split). KV — the only other zone this function
resets — is unaffected: it was already in the individually-freeing carve-out
before this ticket, and still reaches a real `host_arena_->zone_reset(KV)`
call, through `host_zone_reclaim()`.

**Q2 — epoch granularity: per-zone or per-zone-per-graph?** Per-zone,
advancing at the same graph-boundary call sites step 1's regions rotate at
— `host_zone_settle(zone)` bumps a single monotonic `g_host_zone_epoch[zone]`
counter on every call, refused or clean, exactly mirroring where C1 opens a
new region. The counter is **global, not per-`unified_cache`-instance**,
because `host_zone_settle()`'s own live scan already has no device filter
(host zones are shared across devices in this codebase, unlike the VRAM
zones) — matching `g_runtime_alloc_registry`'s existing device-agnostic
scope rather than introducing a new per-device split the surrounding code
doesn't have.

Unlike step 1, this counter **does not gate anything**. C1's region epoch
decides which physical region is "current" for new allocations to land in;
TLSF allocation never routes by epoch at all (see Q1), so there is nothing
for the counter to decide. `alloc_handle::epoch_id`, stamped from the
counter at allocation time, exists purely so a stuck refusal can report how
many graph boundaries the offending allocation has survived
(`oldest live epoch=... boundaries stale=...` in the WARN) — a diagnostic
step 1's design doc asked for explicitly ("the epoch tag belongs on the
registry record... alongside the existing `alloc_id`/`cohort`/`role`/
`category` fields") that step 1 itself had no use for, since C1's regions
are identified by address range, not a stamped id. No new audit-structure
field was added: `zone_audit_live_entry`/`zone_audit_site_visit` are
untouched, matching this document's "no new instrumentation is needed"
guidance for the mechanism generally — the epoch age is text in the
existing refusal `GGML_LOG_WARN`, not a new structured/tested column.

**Q3 — how many concurrent epochs before refusal, and what bounds growth
while one lingers?** No bound is enforced, and none is needed, for a
structural reason rather than a policy choice: step 1's ring needed an
exhaustion refusal ("every region in the ring is still live") because a
bump pool has a *fixed number of interchangeable regions* and running out
of clean ones means the next allocation has nowhere to go. TLSF has no such
resource to exhaust — an allocation from a zone with ten lingering epochs
and an allocation from a zone with zero both resolve identically, against
whatever the free list currently holds. So "how many epochs can linger" is
answered the same way it always was pre-C1: by the zone's actual configured
capacity (`host_zone_grow`/budget), not by a ring size. A lingering epoch is
exactly a lingering `mem_handle` — the same condition this document's
ownership rules already require diagnosing and fixing at its source, not
capping. What C2 adds is that the (now epoch-attributed) refusal WARN makes
a stuck epoch's *age* directly legible, which the pre-C2 code — reporting
only a live count, with no notion of "since when" — could not.

**Event-retained epochs.** Traced rather than assumed: `mem-handle.cpp`'s
`retain_handles_until_event()` (both its background-drain-worker path and
its `graph_lifetime_retention_active()` command-graph path) defers only the
`mem_handle` vector's destruction — which is what eventually reaches
`unified_free_record()` — until the retained event completes or
`release_graph_retained_handles()` runs. Since C2's reclaim IS the
individual free (unlike C1, nothing is decremented separately from it —
see Q1), and that free only ever executes when the handle's real lifetime
ends, in-flight BCS DMA staging already respects the epoch correctly
through this existing machinery with no additional integration required
— there is no separate "epoch" state
for `retain_handles_until_event()` to know about or preserve.

**What this means for the shared `epoch_region`-style struct question.**
Step 1 already declared that no shared struct should be extracted across
the three populations. C2 reinforces why: it needed no ring, no
address-range lookup, and no rotation-refusal at all — three of C1's four
structural elements (region, `region_of()`, ring) don't exist here, and the
fourth (the live counter) exists in a different form (per-zone atomic plus
the pre-existing registry scan, not a `std::atomic<int64_t> live` per
region). The only thing step 2 inherited from step 1, as promised, was the
policy sentence, not any code.

### C3 implementation: oneDNN scratch pointers (`llama.cpp-67c2`)

Step 3 converts `onednn_weights_scratch_` / `onednn_activations_scratch_`
(`unified-cache.hpp`, alongside their `_size_`/`_owner_` siblings), the
reservation logic in `unified_cache::reserve_onednn_scratch()`
(`unified-cache.cpp`), and the two `zone_reset(vram_zone_id::ONEDNN)` call
sites (renamed to `zone_boundary_check(vram_zone_id::ONEDNN)` by
`llama.cpp-37ba` — one of the two was retired outright, see below) that used
to be that reservation's sole reclaim path for its own previous allocation.
**Mechanism name: point release** — distinct from C1's
*epoch decrement* (many interchangeable ring regions, address-range lookup)
and C2's *per-record free* (many independently-registered TLSF records). C3
has neither a ring nor a population: it is exactly two named pointers, each
with a live count of at most 1, so "the epoch" is simply "does this pointer
still reference the block this reservation itself allocated" — release is a
single named `zone_free()` on a single named pointer, not a lookup into
anything.

**The two cases behave completely differently, and only one needed a fix.**
`reserve_onednn_scratch()` places each buffer in one of two ways depending on
whether the pre-reserved VRAM arena's ONEDNN zone is large enough for the
request:

- **Arena-owned** (the common case): `zone_alloc(vram_zone_id::ONEDNN, ...)`
  sub-allocates from the ONEDNN TLSF zone and stores the raw pointer directly
  in `onednn_weights_scratch_` / `onednn_activations_scratch_` — no
  `mem_handle` at all. This was the defect: the *only* place that ever
  reclaimed these bytes was an unconditional `zone_reset(vram_zone_id::ONEDNN)`
  at the top of the reservation-replacement branch, which the code's own
  comment called reclaiming "any previous allocation" — i.e. the bulk-reset
  dependency this epic's C1/C2 sections describe, just with the reservation
  call site itself playing the role C1/C2's periodic graph-boundary reset
  played for their zones.
- **Direct** (grown outside the zone, when a path-scoped sizing predicate
  under-estimated the ONEDNN zone for this model): `allocate_direct_scratch()`
  calls `unified_alloc()` and wraps the result with
  `mem_handle::from_owned_alloc()`, storing the handle in
  `onednn_weights_scratch_owner_` / `onednn_activations_scratch_owner_`. This
  **was already correct** — verified by tracing `release_direct_scratch()`
  (the reservation-replacement cleanup for this case): assigning `owner = {}`
  drops the handle's `shared_ptr<alloc_handle>`, and since this cache field is
  the sole owner (`get_onednn_scratch()` hands callers the raw `void*`, never a
  handle copy), that drop is always the last reference, so
  `release_owned_alloc_handle()`'s `unified_free()` runs synchronously right
  there. No conversion needed for this case; the `_owner_` fields the ticket
  flagged for verification were already doing the job.

**The fix, scoped to the arena case only.** Before doing anything else —
before even reading `zone_capacity()` — `reserve_onednn_scratch()` now
individually `zone_free()`s whichever of the two OLD pointers is currently
arena-owned (`vram_owns(ptr)`), then nulls that field. A pointer that is
instead a DIRECT leftover from an earlier growth episode is left untouched by
this step — it is already a real lease, and the existing
`release_direct_scratch()` cleanup a few lines later (unchanged) still frees
it. On a partial failure (the weights half allocates but the activations
half does not, or vice versa), only the half that this attempt itself
allocated is freed — the old pair was already individually reclaimed up
front, so there is nothing else in the zone to touch.

**The point-release must run unconditionally, before the capacity check —
not inside the "zone is big enough" branch.** The first version of this step
put the release inside `if (total_needed <= zone_cap)`, after the
`total_needed > zone_cap` branch's `ensure_planned_arena_zones()` re-plan
attempt. Spec review caught the consequence: `ensure_planned_arena_zones()`'s
`has_live_scratch` refusal check
(`onednn_weights_scratch_ != nullptr || onednn_activations_scratch_ != nullptr
|| ...`) treats a non-null pointer as still-live *regardless of who owns
it* — including the reservation's own predecessor, about to be replaced by
the very call that is checking. So on exactly the path that most needed the
fix — an existing arena reservation growing into a bigger one — the old
pointers were still set when the refusal check ran, the re-plan refused
itself every time, `total_needed <= zone_cap` stayed false after the
refusal, the point-release block was skipped entirely, and control fell
through to the shared direct-allocation cleanup further down. That cleanup's
arena-owned branch only nulled the fields (it predates this step and was
never in scope for the original design) — orphaning the TLSF bytes until
whole-arena teardown. Because the refusal predicate and the leak precondition
are the same non-null fields, **every growth attempt following a prior arena
reservation leaked**, and the in-place growth path had plausibly never
actually succeeded. The fix moves the point-release to the top of the
`arena_active()` branch, unconditional on the capacity check: this closes the
leak and, as a direct consequence, lets `ensure_planned_arena_zones()`
actually succeed when nothing else is live, since the reservation's own
predecessor no longer counts against itself. The shared direct-allocation
cleanup's arena-owned branch is now provably unreachable (whenever
`arena_active()` was true, the point-release already ran; whenever it is
false, `vram_owns()` is always false too, since it tests `arena_base_`, which
every `arena_destroy()` caller already required these fields null before
calling) — it is now an assert/log rather than a silent null, so a future
regression that reopens this leak announces itself instead of orphaning
memory quietly again. A CPU-only source-contract check
(`tests/test-sycl-zone-reset-audit-source.py`, "the oneDNN point-release
precedes the growth-path re-plan") pins the ordering the same way the
existing hook-before-early-return checks pin theirs, and is confirmed to fail
against the pre-fix commit.

**Nothing else lives in the ONEDNN VRAM zone — grep-verified, not assumed.**
The only `zone_alloc(vram_zone_id::ONEDNN, ...)` call sites in the entire
backend are the four inside `reserve_onednn_scratch()` itself (two on the
sub-allocation path just described, two on a `use_arena_zone` growth path
that the surrounding code comment already documents as unreachable given the
function's current control flow). So, now that the release runs
unconditionally rather than only on the zone-was-big-enough path, freeing
exactly the two pointers this reservation previously handed out is a
*complete* reclaim of the zone on every path that reaches
`reserve_onednn_scratch()`'s arena branch, not only the ones where the zone
happened to already be big enough — there is no third occupant to enumerate
or worry about. This also explains the Phase-0 audit's one genuinely-moving
`zone_largest_free` reading (256 → 143.88 MB, no recovery across the captured
run): on Mistral, `onednn_reorder` — the largest oneDNN reorder buffer, see
"Path-scoped zone sizing" below — is 112.0 MB, and 256 − 112 ≈ 144 MB matches
the observed dip to the byte. The "occupancy" was always these two pointers;
nothing else was ever a candidate.

**The call site stays, but its role ends — mirroring C2's treatment of
`host_zone_boundary_check(SCRATCH|STAGING)`.** (`llama.cpp-37ba` later
renamed this call from `zone_reset(vram_zone_id::ONEDNN)` to
`zone_boundary_check(vram_zone_id::ONEDNN)`, matching the same-named
renaming/dispatch split it applied throughout this document — see this
document's "Step 4" subsection.) Unlike C1/C2, there is no *separate*
periodic reset call outside this function: the reservation-replacement
branch was always the only caller of this checkpoint in the codebase, so
there is no
graph-boundary checkpoint elsewhere to leave alone. The call is kept at the
same position, now placed *after* the point-release frees above it, so by
construction it can only ever observe an empty zone. This is deliberately not
deleted: the Phase-0 audit's `device-zone-reset/ONEDNN` cohort reads exactly
this call site, and dropping it would make the site silently stop being
visited rather than continuing to report a truthfully empty zone — the same
"retiring the reset must not retire the observability" rule C1/C2 already
follow. The second (former) call site — a bulk reset on partial
sub-allocation failure — is not retained in any form: it already had no
external observability to preserve (it existed purely as this function's own
cleanup), and the point release now performs that exact cleanup precisely
(`zone_free()` the half that was actually allocated) rather than
approximately (reset everything, including a still-good direct-scratch
sibling that was never at risk).

**In-flight safety — adjudicated from source, not assumed.** The concern was
whether the old bulk reset provided any delay ("wait for the current oneDNN
primitive to finish") that a same-instant individual free would lose. Tracing
the actual call chain (`acquire_onednn_pp_scratch` in `ggml-sycl.cpp` →
`unified_cache_get_onednn_scratch` → `acquire_onednn_scratch_reservation`,
which waits on `onednn_scratch_cv_` for `onednn_scratch_refcount_ == 0` before
handing a caller the buffer, and `release_onednn_scratch_reservation`, which
decrements it back to 0 when the caller's `onednn_pp_scratch_guard` goes out
of scope) shows that this CV/refcount pair is the *only* mechanism that ever
serializes a reservation against an in-flight consumer — and
**`reserve_onednn_scratch()` itself never consults it.** It takes
`onednn_scratch_mutex_` and reallocates unconditionally, whether the old
buffer's last CV-observed refcount was 0 or not, because a completely
separate call can enter `reserve_onednn_scratch()` through the same mutex
while another thread's reservation is still outstanding (the mutex is
released for the whole window between an `acquire` and its matching
`release`). **This is a pre-existing race, identical in the old bulk-reset
code and unchanged by this conversion** — the old `zone_reset()` was exactly
as immediate and exactly as unconditional as the new `zone_free()` calls,
gated by the same single mutex and nothing else. There was no delay to lose,
so no event-deferral was added: doing so here would be inventing a safety
property the code never had, on a hazard this step's scope does not cover.
The direct (non-arena) case carries the identical gap — `release_direct_scratch()`
frees synchronously with no event wait either, and always has. Filed as a
known pre-existing gap for a future ticket, not fixed here: closing it needs
`reserve_onednn_scratch()` to either take the CV wait itself or fold
replacement into the same critical section as acquisition, which is a
concurrency-control change to the reservation protocol, not a reclaim-path
change.

The growth-path leak fix (below) moves the point-release earlier within this
same function — to the top of the `arena_active()` branch instead of inside
the `total_needed <= zone_cap` branch — but this does not change the analysis
above. Both the old position and the new one are inside the single
`std::lock_guard<std::mutex> lock(onednn_scratch_mutex_);` that spans the
entire function body; moving the release earlier within that one critical
section neither adds a new lock nor consults the CV/refcount pair that was
already the only thing missing. The race described above is exactly as
present, and exactly as unchanged, at the new call site as it was at the old
one.

**The two graph-boundary drains, re-derived from source (carried from the
37ba gate adjudication, `c-634z`, citing `m72w`'s "two graph drains separated
by pool-retained release, not proven redundant").** The graph-boundary block
in `ggml_backend_sycl_graph_compute` (`ggml-sycl.cpp`, immediately before the
`unified_cache_scratch_pool_epoch_boundary` / `unified_cache_host_zone_boundary_check(STAGING)`
/ `unified_cache_host_zone_boundary_check(SCRATCH)` call sites C1/C2 target,
renamed by `llama.cpp-37ba` — see this document's "Step 4" subsection)
contains exactly this shape: `ggml_sycl_cpu_staging_drain()` (WEDGE-48330, waits on
`g_cpu_staging`'s host_task/compute completion events), then a pool-retained
release (`ggml_sycl_cpu_staging_release()`, which drops `g_cpu_staging`'s
leases back to the offload pool with **no wait of its own**), then a second
drain (`ggml_sycl_staging_pool().release_all_idle(...)`, whose own body calls
`drain_all()` — waiting on `staging_buffer_pool`'s BCS DMA events — before
releasing its slots). Re-derived with C1+C2+C3 landed: these two drains guard
two *different* structures (`g_cpu_staging`'s bank/slot array vs.
`staging_buffer_pool`'s slots), neither of which C1/C2/C3 touched — both sit
above the TLSF-zone reclaim layer those steps converted, as a caching layer
that reuses freed buffers rather than returning them to the zone on every
release. Since `ggml_sycl_cpu_staging_release()` performs no wait, the first
drain is load-bearing for it specifically, and remains so regardless of what
happens to the zone-level reset below. **Not proven redundant — confirmed,
not merely re-affirmed as unresolved.** Left in place; noted here for the
37ba finale rather than acted on in this step, per that ticket's scope.

### Step 4: reconciling the reset call sites (`llama.cpp-37ba`)

With C1, C2, and C3 landed, this step re-derived — per-symbol, from source,
not from the epic's original (pre-Option-C) framing — what was actually left
to remove. **The headline finding: nothing was literally deletable.** Every
remaining call site was either (a) already a pure liveness/audit checkpoint
that cannot force-reclaim a live handle, because C1/C2/C3 already made the
real reclaim happen elsewhere, or (b) a genuinely load-bearing on-demand
reclaim for a zone the epic never targeted. Deleting a call site would have
destroyed the audit's ability to notice a future regression at that
boundary, contradicting CLAUDE.md's "retiring the reset must not retire the
observability."

**Owner ruling: split by semantics rather than keep the old names.** An
earlier version of this section proposed keeping the three names and
reconciling the acceptance criterion's wording instead (a rename was judged
disproportionate — cosmetic gain, real risk, no implementer-side hardware
verification). The owner overruled that: acceptance criterion 1 is honored
*literally*, by making every name say what it does, splitting each dispatcher
by semantics so no symbol lies about which of the two things it does. Final
vocabulary, all landed:

| Old name | Zones | New name(s) |
|---|---|---|
| `unified_cache_reset_scratch_pool(device_id)` | VRAM scratch bump pool (C1) | `unified_cache_scratch_pool_epoch_boundary(device_id)` |
| `unified_cache_host_zone_reset(zone)` | SCRATCH/STAGING (liveness check) | `unified_cache_host_zone_boundary_check(zone)` |
| `unified_cache_host_zone_reset(zone)` | KV (real reclaim, out of scope) | `unified_cache_host_zone_reclaim(zone)` |
| `unified_cache::zone_reset(zone)` (private) | ONEDNN/SCRATCH (liveness check) | `unified_cache::zone_boundary_check(zone)` / `unified_cache_zone_boundary_check(device_id, zone)` |
| `unified_cache::zone_reset(zone)` (private) | KV/RUNTIME (real reclaim, out of scope) | `unified_cache::zone_reclaim(zone)` / `unified_cache_zone_reclaim(device_id, zone)` |

Each pair shares one private, genuinely-internal implementation
(`unified_cache::host_zone_settle(zone)` / `unified_cache::zone_settle(zone)`
— unchanged logic, renamed from `host_zone_reset()`/`zone_reset()`) that the
two truthfully-named callers dispatch into after an assert restricting which
zones each may pass:

- `host_zone_boundary_check(zone)` asserts `zone == SCRATCH || zone == STAGING`.
- `host_zone_reclaim(zone)` asserts `zone == KV`.
- `zone_boundary_check(zone)` asserts `zone == ONEDNN || zone == SCRATCH`.
- `zone_reclaim(zone)` asserts `zone == KV || zone == RUNTIME`.

**Internal TLSF-layer primitives kept their names, per the ruling's explicit
exemption** ("genuinely internal ... not reachable as backend policy" — the
criterion governs the backend's *policy surface*, not every private helper
that happens to contain the word "reset"): `pinned_chunk_pool::zone_reset()`
(`pinned-pool.hpp`/`.cpp`, the host-zone TLSF primitive `host_zone_settle()`
still calls via `host_arena_->zone_reset(zone)`) and
`scratch_pool_reset_regions()` (`unified-cache.hpp`). Neither is reachable
from `ggml-sycl.cpp` or any free-function wrapper; both are pure allocator
internals.

**Not renamed: the audit environment variable and the audit site-name
strings**, per the ruling's explicit exclusion. `GGML_SYCL_ZONE_RESET_AUDIT`
is an external interface referenced by every committed capture and every
piece of prior documentation — renaming it would break nothing at the type
level but would orphan every historical capture's instructions. The audit
site-name strings (`"device-zone-reset"`, `"host-zone-reset"`,
`"scratch-pool-reset"`, `"weight-reclaim"`, passed as the first argument to
`zone_audit_site_visit`) are baseline-comparison identifiers across captures
01–14 cited throughout this document and the tracker; renaming them would
silently break every prior-capture comparison that keys on those strings. Both
are stable historical identifiers now, independent of the current C++ function
names they originated from — read them as such, not as a naming
inconsistency.

**Per-symbol disposition (unchanged conclusions, now expressed with the new
names):**

- **`unified_cache_scratch_pool_epoch_boundary(device_id)`** (C1, VRAM
  scratch bump pool, `scratch_pool_*`). One production call site
  (`ggml-sycl.cpp`, graph boundary). Internally rewind-on-zero /
  rotate-on-live / refuse-on-exhaustion (C1) — the bulk-reclaim semantics
  were already gone from the *function*, not just hidden behind a guard.
  `get_scratch()`/`return_scratch()` still have zero other production
  callers (re-confirmed by grep), so in every measured production workload
  this call observes an empty pool and does nothing observable. No KV-style
  dual-purpose need existed for this symbol, so it took a single honest
  rename rather than a two-way split.
- **`unified_cache_host_zone_boundary_check(zone)`** / **`unified_cache_host_zone_reclaim(zone)`**
  (C2, host SCRATCH/STAGING vs. KV). Four production call sites: two at the
  graph boundary (`ggml-sycl.cpp`, STAGING/SCRATCH → `boundary_check`) and
  two inside `arena_reserve()` (`unified-cache.cpp`, context-switch: KV →
  `reclaim`, STAGING → `boundary_check`). Both dispatch into the shared
  `host_zone_settle(zone)`, whose `epoch_tracked` branch
  (`zone == SCRATCH || zone == STAGING`) still returns *before* reaching
  `host_arena_->zone_reset(zone)` — the real bulk call is reached only
  through `host_zone_reclaim()`, i.e. for KV.
- **`unified_cache::zone_boundary_check(vram_zone_id::ONEDNN)`** (C3,
  internal to `reserve_onednn_scratch()`) and
  **`unified_cache::zone_boundary_check(vram_zone_id::SCRATCH)`** (the
  pool_leg compute-arena checkpoint in `arena_reset()`, found by this step —
  see below). Both call sites, not externally wrapped for these two zones
  today (a free-function `unified_cache_zone_boundary_check(device_id, zone)`
  exists for future/test callers, but no production caller uses it).

**Out of the epic's scope, and always was — KV and RUNTIME.** The Phase-0/
Phase-1 audit inventory that drove C1–C3 covers exactly eight sites
(`device-zone-reset/{SCRATCH,ONEDNN}`, `host-zone-reset/{SCRATCH,STAGING}`,
`scratch-pool-reset/bump`, `weight-reclaim/{load-boundary,mid-load-replan,
model-teardown}` — matching the counts in every Phase-0 capture cited in the
C1–C3 subsections). No `device-zone-reset/KV`, `/RUNTIME`, `/WEIGHT`, or
`host-zone-reset/KV` row appears anywhere in that inventory: these zones were
never measured, never audited, and never part of C1/C2/C3's mandate.
Grep-verified they are also still genuinely load-bearing, not merely
unmeasured:

- **VRAM KV** (`unified_cache::zone_reclaim(vram_zone_id::KV)`, reached via
  `unified_cache_zone_reclaim(device_id, KV)`): called from `ggml-sycl.cpp`'s
  KV-buffer-type allocation fallback (device VRAM KV request exceeds
  `zone_available(KV)` → try a reclaim, then re-check) and from
  `arena_reserve()`'s context-switch path ("same model, new context —
  reclaim ephemeral zones so KV/runtime space from the previous context is
  reclaimable"). Both are on-demand reclaim triggered by an actual capacity
  need, not a scheduled sweep, and `zone_settle()`'s own
  shared-KV+WEIGHT-arena refusal guard (unaffected by C1–C3) still protects
  live weight entries.
- **VRAM RUNTIME** (`unified_cache::zone_reclaim(vram_zone_id::RUNTIME)`):
  same `arena_reserve()` context-switch call. RUNTIME allocations (ggml
  compute buffers) route through `unified_alloc()` with
  `prefer_vram_zone = RUNTIME` (`ggml-sycl.cpp`, `unified-cache.cpp`), which
  — unlike the internal `zone_alloc()`/`zone_free()` pairs C1/C3 replaced —
  registers into `g_runtime_alloc_registry`. So this zone's live-check is not
  vacuous-by-construction the way ONEDNN's was: it is a real, context-lifetime
  reclaim need, structurally the same shape as KV.
- **Host KV** (`unified_cache::host_zone_reclaim(host_zone_id::KV)`):
  `!epoch_tracked`, reaches the real `host_arena_->zone_reset(zone)`
  unconditionally when clean. Called from `arena_reserve()`'s context-switch
  path alongside host STAGING's `host_zone_boundary_check()`.
- **WEIGHT** (both VRAM and host): hard-refused unconditionally
  (`GGML_ASSERT(zone != host_zone_id::WEIGHT ...)` and `zone_settle()`'s own
  `zone == vram_zone_id::WEIGHT` early return, both predating this epic).
  Never actually resets; already in the epic's explicit KEEP list ("every
  `refusing ...` guard"). Neither `zone_boundary_check()` nor
  `zone_reclaim()`'s assert admits WEIGHT — a caller that tried would fail
  the assert before ever reaching `zone_settle()`'s own refusal.

**Recommendation, standing: KV, RUNTIME, and WEIGHT are out of
`llama.cpp-iiff`'s scope.** They are context/session-lifetime zones with a
genuine on-demand reclaim need at context-switch or capacity-pressure
boundaries, not reset-only zones whose sole reclaim path was a scheduled
sweep — the defect class this epic exists to fix. Converting them (if ever
warranted) is a different, future ticket, not a step of this one.

**A fourth, previously-unenumerated population — found, and already
compliant.** `unified_cache::arena_alloc()` / `arena_free()` / `arena_reset()`
(`unified-cache.cpp`) implement the "pool_leg" per-op compute scratch
allocator — a *different* mechanism from C1's `scratch_pool_*` bump pool,
sharing only the English word "scratch." Two modes:

- **`arena_active()` (the default, tested, documented configuration):**
  `arena_alloc()`/`arena_free()`/`arena_reset()` delegate directly to
  `zone_alloc(vram_zone_id::SCRATCH, ...)` / `zone_free(vram_zone_id::SCRATCH,
  ptr)` / `zone_boundary_check(vram_zone_id::SCRATCH)` — i.e. every pool_leg
  allocation already gets a real, individual TLSF free the moment its caller
  releases it (`arena_free()`'s arena-active branch calls `zone_free()`
  unconditionally; there is no watermark/no-op path in this mode). The
  boundary check in `arena_reset()` is therefore *already*, structurally, the
  same "checkpoint over a zone that individual frees already emptied"
  pattern C1–C3 established elsewhere — it was never touched by this epic
  and did not need to be: it arrived at the target state independently,
  because pool_leg was built on `zone_alloc`/`zone_free` from the start
  rather than a raw bump offset. This is also why it uses
  `zone_boundary_check()`, not `zone_reclaim()`, despite living in
  `arena_reset()`'s "reset" name.
- **`arena_active() == false` (fallback, non-default):** a raw atomic bump
  allocator (`compute_arena_off_`) with a genuine reset-only defect —
  `arena_free()`'s non-arena branch does watermark-only reclaim (only the
  block currently at the bump-pointer top is reclaimed; the code's own
  comment states "Non-watermark free: no-op (space reclaimed at
  arena_reset)"). This is architecturally the same defect class C1 fixed for
  `scratch_pool_*`, but it is a **different, unaudited, unconverted
  population** — never measured by Phase-0 (arena-active is the default, so
  none of the cited captures exercised this fallback), never in C1's scope
  (C1 explicitly targeted `scratch_pool_*`, not `compute_arena_`'s own
  allocator), and not touched here. **Flagged as a new finding for a future
  ticket**, not fixed as part of 37ba: fixing it would mean converting a
  fourth, previously-unknown population under a ticket scoped to closing out
  three known ones, and the non-arena-active configuration is not the
  production path this fork ships.

**Acceptance criterion 1 is now literally satisfied, machine-checked.**
`tests/test-sycl-zone-reset-audit-source.py`'s "the three old reset/boundary
dispatcher names have zero production call sites" check scans
`unified-cache.cpp`, `unified-cache.hpp`, `ggml-sycl.cpp`, and `common.cpp`
(the same four sources every other check in that file reads) for
`zone_reset`, `host_zone_reset`, and `reset_scratch_pool` — both the bare
member-function spellings and their `unified_cache_*` free-function wrapper
spellings — as real call/definition syntax (name immediately followed by
`(`), with a lookbehind exemption for the two internal TLSF primitives named
above. Comments never reach the scan at all (`strip_comments()` runs before
the check does), so the historical "named `host_zone_reset()` before
`llama.cpp-37ba`'s rename" narration scattered through this codebase for
readers' benefit does not trip it. Verified as a genuine positive control: a
mutant reintroducing a single stray `unified_cache_zone_reset(0, zone)` call
makes the check fail and print exactly which file and which old name
survived.

## Lifecycle identity and async lease boundary

**Target invariants (not current APIs or current behavior).** The enforceable
contract is canonical §12: `ModelId`, `(slot, SlotGeneration)`, `LoadTxnId`,
`ContextId`, `(ContextId, SessionId, SessionResetEpoch)`, `(ContextId,
GraphEpoch)`, and `InvocationId`. IDs never wrap; slot 33 fails before LOADING.
Loads abort by default on missing success, wrong txn, depth error, cancellation,
or failure. Reset/teardown uses exact typed tickets, including reset epochs that
prevent ABA.

Allocation identity, semantic owner, and asynchronous use are distinct. One
exclusive top-level token per device is copied into submits. Each invocation binds exactly one `ContextId`/`GraphEpoch` across its devices;
a cross-context/epoch submit fails before side effects. Lifecycle authority is
an aggregate with separate root retention and terminal-event sets per device.
Each device is OPEN while producers/submits register, SEALED only after
registration closes and every producer seals, then COMPLETE only when every
registered slot is terminal; uncertain submission becomes QUARANTINED. A fast
terminal while OPEN cannot release anything. Join creation failure drains known
events outside locks; quarantine retains roots/backing until quiescence is
proven. Same-owner reentrancy copies the exact InvocationId; busy/wait and
multi-device all-or-none rules remain explicit.

Every async pointer has a backing lifetime. Bare `DIRECT` requires a validated
owner/backing lease and ARENA handles retain arena/chunk generation. A retiring
`GraphEpoch` completion releases only old-epoch resources, never replacement
state. Locks follow exhaustive L1 lifecycle → L2 execution → L3 owner registries
→ L4 cache/queue registry → L5 allocator/work ordering; the canonical table
includes current oneDNN scratch, MoE buffer, pipeline/block copy-queue,
backend-context, and `control_host_allocs_mutex` locks. The target deletes the
oneDNN global `unique_lock` registry. Exclusive foundation owner
`32dg8.15.12` deletes it and freezes a logical
`{device, generation, reservation_id}` API: brief same-thread keyed-mutex
transitions increment/decrement reservation refcounts, while event payloads carry
only logical reservation/backing handles. Cross-thread completion takes/releases
the keyed mutex on that completion thread; no mutex ownership is retained. Global/transitional same-rank co-holding is forbidden and
completion/diagnostic locks C/D are isolated. No wait, blocking allocation/device
call, queue create/destroy, callback, or final handle/token/backing destruction
occurs under a listed lock. Tier verdicts are reporting-only.

**Current exceptions during migration.** Current code still has bare slot masks,
process-global load/planner scratch, device-only pending-KV FIFO,
`g_sycl_graph_compute_mutex`, graph cleanup without model/epoch attribution,
DIRECT/ARENA shapes without universal async backing retention, and void memory
ops without terminal events. Current oneDNN scratch keeps keyed locks in a
global same-rank registry, and current control-host cleanup clears owning
`mem_handle`s under its context mutex. The sole target teardown order is: begin
drain → wait for terminal context events outside locks → extract/move the control
batch under L4 → unlock → destroy batch → finish drain. These are
non-conformances to migrate, not licensed exceptions to preserve or descriptions
of supported concurrency.

Canonical §12.8-§12.10 assigns foundations only to `viu2` (model/load), `1q72`
(context/session/GraphEpoch/tokens), `32dg8.15.12` (exclusive async backing/event
leases/oneDNN/locks), and `o6jx` (owner-targeted teardown). Exact foundation
edges are `viu2 → 1q72` and `{1q72, 32dg8.15.13} → 32dg8.15.12 → o6jx`;
`tudj` is a closed duplicate with no ownership or edge. Existing focused IDs retain
their actual scopes: `nn6z` MoE discovery/popularity, `nlww` MoE bias/activation,
`vbeb` layer streaming, `y36c` pending KV masks, and `x3ou` diagnostics;
`x3ou` consumes `viu2`/`1q72`/`32dg8.15.12`/`o6jx`. Closed `h5m4` remains the
TLS-reset proof gate. `t5nq` is a closed merged packed-K-sidecar gate; `udpi`
completed its live GPU failpoint/retry/teardown matrix, and `32dg8.15.12`
strengthened consumer lifetime with leased snapshots. `otry`, not `t5nq`,
revalidates packed-K guarantees after foundations. “All
foundations/focused children → otry” is the lifecycle transitive-closure
projection, not the exact live edge list. Exact direct `otry` dependencies are
`nlww`, `h5m4`, `nn6z`, `y36c`, `vbeb`, `x3ou`, `t5nq`, and `o6jx`;
foundation/organizational edges are transitive. Exact tail edges are `{otry,
hcyp (closed)} → jwy4` and `{jwy4, awcp (closed)} → k7b0`; final `k7b0` closure
is blocked by `jwy4`. `{1q72, .15.13} → .15.12 → o6jx` is preserved. `jwy4`, not `hcyp`, owns the final script/fixtures/CSV/prose
census refresh. The fixed teardown order, H1-H14/G1-G7, fixtures, split
mutations, and lock controls remain canonical.

## Path-scoped zone sizing

`populate_host_zone_sizing` (`ggml/src/ggml-sycl/unified-cache.cpp`) once sized
every arena zone from a single global `max_tensor_bytes` — the largest tensor in
the model. Several consumers can never hold that tensor: the oneDNN matmul
scratchpad, its per-layer reorder buffer, the CPU quantization slots and the DMA
weight-stream staging pool all operate on per-layer weights, while the global
maximum is the vocabulary embedding or the LM head. Each consumer was therefore
over-provisioned by the difference, and two of those figures are summed a second
time into `host_zone_scratch_bytes`.

Measured (`docs/plans/2026-07-25-zone-sizing-findings.md`, Task 1): on GPT-OSS
20B MXFP4 the global maximum is `output.weight` at **586.8 MB** while the
largest per-layer weight is **134.5 MB**; on Mistral 7B Q4_0 it is
`output.weight` at **102.5 MB** against **31.5 MB**.

### The classifier is structural, never by name

`zone-sizing.hpp` classifies a tensor by how often its `(type, ne[0..3])` key
repeats in the inventory:

```
key  = (type, ne[0], ne[1], ne[2], ne[3])
zone_is_per_layer_weight(t)  <=>  t.has_shape && freq[key(t)] >= k_zone_per_layer_min_group   // 4
```

A per-layer weight family repeats once per block; the embedding and the LM head
are singletons, or a pair when they happen to share type and shape. **Names are
a diagnostic field only — no decision path may branch on one.** Tensor names are
a GGUF convention, not a guarantee, and a name predicate that matches nothing
fails *silently*: every maximum degrades back to the global one, which looks
exactly like the reclaim genuinely being zero. Task 1 made that concrete —
**there is no `lm_head` tensor in either reference model** (llama.cpp names the
LM head `output.weight`), so the originally-planned `lm_head` clause would have
matched nothing and nobody would have noticed.

The threshold, the histograms it was chosen from, and the `zone_*` (pure) versus
`zone_sizing_*` (global state or log output) naming contract are all documented
in `zone-sizing.hpp`; read it there rather than restating it here.

### The maxima

`zone_scoped_maxima(inventory)` returns:

| field | meaning |
|---|---|
| `any_tensor` | the legacy global maximum; use only when the path genuinely accepts any tensor |
| `onednn_eligible` | largest tensor that can be a oneDNN matmul reorder subject, **as stored** |
| `onednn_reorder` | largest oneDNN reorder buffer, i.e. that same eligible set **dequantized to f16** |
| `cpu_quant_eligible` | largest tensor the CPU quantization slots can hold |
| `dma_streamed` | largest tensor the host→device weight stream carries |

**`onednn_reorder` is not `onednn_eligible` times a constant, and it is not
bounded by `any_tensor`.** Both surprises are load-bearing:

- The oneDNN matmul weights reorder holds a **dequantized f16 copy**, so its size
  is the element count × 2 — a per-type expansion (Q4_0 3.56×, Q8_0 1.88×,
  MXFP4 3.76×), not one factor. Across a mixed-quantization model the largest
  *stored* eligible tensor and the largest *expanded* one are different tensors,
  so the maximum must be taken over the expanded sizes. Scaling the stored
  winner by its own factor under-sizes whenever a lower-bit-rate tensor with
  more elements exists — a worked case is Case 2b in
  `ggml/src/ggml-sycl/tests/test-zone-sizing.cpp` (23% under).
- It **legitimately exceeds `any_tensor`**. On Mistral 7B Q4_0 the largest tensor
  in the model is 102.5 MB and the largest reorder buffer is 112.0 MB. Do not
  add an `onednn_reorder <= any_tensor` assertion, budget clamp, or collapse
  check — it would fire on a healthy model. `is_monotonic` in the unit test
  deliberately omits it and says so.

The expansion is computed in `unified_cache_adapt_zone_inventory`, the only
party with ggml's type traits, and carried as `zone_tensor_desc::reorder_size`.
The classifier TU never derives it — that is the same rule that keeps it from
deriving a size from `ne`, and it matters for the same reason: `ne[2]` is the
expert count on MoE weights, so a 2-D-only product under-states a 3-D tensor
by 32×.

**Rule for adding a consumer:** pick the maximum matching your path. Reach for
`any_tensor` only when the path really does accept anything, and say why in a
comment — an unjustified `any_tensor` reintroduces exactly the over-provision
this exists to remove.

Two consumers are deliberately left on `any_tensor`, and both comments say why:
`moe_q8_workspace_bytes` is derived as `max_tensor_bytes / n_experts` and is
therefore already a per-expert figure, and `s1_per_inflight_bytes` backs the S1
preload, which streams *every* tensor including the embeddings. Narrowing that
second one would be this bug in reverse.

### There are TWO oneDNN sizing sites, not one

This is the trap most likely to waste a future change:

- `plan.onednn_scratchpad_bytes`, set in `populate_host_zone_sizing`, is the
  plan-level figure and the one the `[SYCL-PLAN]` line reports. **It does not
  size the VRAM ONEDNN zone.**
- `g_tensor_inventory_onednn_scratchpad_bytes`, set in
  `populate_inventory_globals` (`ggml/src/ggml-sycl/ggml-sycl.cpp`), is what
  actually sizes it: it flows through
  `unified_cache_set_planned_onednn_scratchpad_bytes` into
  `g_planned_onednn_scratchpad_bytes[device]`, which `ensure_planned_arena_zones`
  reads when it lays out the arena.

Both now narrow through the same `unified_cache_adapt_zone_inventory` +
`zone_scoped_maxima` pair, and both compute the identical
`onednn_reorder + onednn_eligible` sum. **A future consumer that changes only
one of the two will appear to work and change nothing** — the plan figure moves
in the log while the zone stays exactly as large as it was. Note the second site
lives in `ggml-sycl.cpp`, where codescout's index is blind; `search_text` for a
symbol there returns the `unified-cache.cpp` occurrences and silently omits it.
Verify with `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n '<pattern>'`.

`plan.onednn_reorder_bytes` is a third consumer of the same maxima — it is the
reorder buffer alone (no activations half) and feeds the minimum-zone-size sum
in `populate_host_zone_sizing`. It takes `onednn_reorder` for the same reason
the weights half does.

### What the narrowing actually reclaimed

Measured before/after (`57db1e693` → `b36bb603b`), Task 8. **VRAM and host are
separate budgets; never sum them.**

| budget | model | before | after |
|---|---|---:|---:|
| VRAM ONEDNN zone (both cards) | GPT-OSS 20B | 1173.6 MB | 268.9 MB |
| VRAM arena weight zone, B50 | GPT-OSS 20B | 13348.4 MB | 14253.1 MB |
| VRAM arena weight zone, B70 | GPT-OSS 20B | 29578.1 MB | 30482.9 MB |
| host SCRATCH zone | GPT-OSS 20B | 2644.1 MB | 1287.0 MB |
| host SCRATCH zone | Mistral 7B | 442.2 MB | 229.0 MB |

**Only the VRAM half converts into granted MoE layers.** The 904.7 MB the ONEDNN
zone gives up is picked up exactly by the arena weight zone, and on the B50 the
MoE down-i8 layout pass turns it into **4 more granted layers, 6/24 → 10/24** at
~261 MB per tensor. The B70 already granted all 24 in both builds, so there the
same 904.7 MB is idle headroom. The host SCRATCH reclaim (−1357.1 MB on GPT-OSS,
−213.2 MB on Mistral) is host memory and moves no VRAM budget at all.

`cpu_quant_buffer_bytes` shrinks too, but it has **no reader anywhere** — it is a
diagnostic figure, not a provision. Do not add it to a reclaim total.

Mistral reclaims **zero VRAM**, and that is the floor working rather than a
failure: both the before estimate (205.1 MB) and the after estimate (63.0 MB)
sit under the 256 MB ONEDNN zone floor, so the zone is 256 MB either way.

### When a predicate under-estimates

A request larger than the planned zone grows it **through the unified cache** —
never a direct allocation — and the existing refusal to rebuild the arena while
live leases are held is preserved, not routed around. A wrong predicate
therefore costs a reallocation, not a crash.

`reserve_onednn_scratch` reaches its growth path for two causes that are logged
distinctly and **must not be conflated**:

- `planned zone under-estimated` — the request exceeded the planned zone. This
  is a sizing defect, and the only case
  `zone_sizing_record_underestimate("onednn", …)` counts.
- `zone fragmented` — the zone was large enough but the sub-allocation failed
  anyway. An allocator condition; counting it would blame the predicate for
  something it did not do.

`GGML_SYCL_DEBUG=1` prints `[SYCL-PLAN] zone sizing coverage: onednn
observations=N underestimates=M` once per plan. The *observation* count is what
distinguishes "the path was never entered" from "the path was entered and the
predicate held" — both leave the under-estimate count at zero and they mean
opposite things.

**What to do when the warning fires:** identify the tensor whose structure
defeated the predicate, correct the predicate in `zone-sizing.cpp`, and add the
case to `ggml/src/ggml-sycl/tests/test-zone-sizing.cpp`. **Do not raise the
estimate back to `any_tensor` to silence it** — persistent growth is worse than
the original over-provision, but so is reinstating it.

### Known limits (load-bearing — read before changing any of this)

1. **The ONEDNN scratchpad's two halves are in different units, deliberately.**
   The formula is `onednn_reorder + onednn_eligible` — expanded weights plus a
   *stored-bytes placeholder* for the activations. That asymmetry is the current
   state of knowledge, not drift:

   - **The weights half is exact.** It is the f16 reorder buffer, and the size
     matches what the consumers request to the byte.
   - **The activations half is a placeholder that happens to cover.** The real
     buffer is `batch_tokens × K × sizeof(f16)` — measured 14.0 MB on Mistral 7B
     Q4_0 at `-p 512`, against the 31.5 MB `onednn_eligible` reserves for it.
     Sizing it properly needs a batch bound the planner does not have at that
     point: `planner_n_ctx` is the context length, not `n_ubatch`, and using it
     would inflate this half to the weights half's size for no gain. **If you
     revisit it, find a real bound — do not substitute `onednn_reorder`**, which
     would over-provision ~78% and start pushing models past the floor.

   This replaces the earlier defect (`llama.cpp-2wgg`, fixed): the weights half
   used to read `onednn_eligible`, sizing a dequantized buffer from a quantized
   byte count. Mistral 7B Q4_0 planned **63.0 MB against a measured 126.0 MB
   peak** — exactly 2.00× under — and only the 256 MB floor kept it working.
   The classifier was picking the right tensor throughout; the multiplier
   applied to it ignored format expansion.

   **Three ratios were in play in the original report and they are not the same
   quantity.** If you find them quoted elsewhere, keep them apart: the *expansion*
   is 3.5556× (Q4_0's 4.5 bits/weight → f16's 16); the *sizing error* was 2.00×
   (126.0 / 63.0, weights and activations being live simultaneously — an earlier
   ~1.78× figure compared a single 112.x MB request against the plan and
   understated it); and *how much the floor can hide* is `256 MB / planned`, so
   it is model-dependent rather than a fixed factor.

   **The floor is no longer masking a known defect, but it is still a floor.**
   Mistral now plans 143.5 MB, still under 256 MB, so `underestimate_count`
   remains an insensitive instrument for this path on that model. Compare
   planned-vs-observed in the logs — `[SYCL-PLAN] oneDNN scratchpad` against
   `[UNIFIED-CACHE] Runtime breakdown … ONEDNN_ZONE` — rather than reading the
   counter. Reproduce with:

   ```bash
   ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_DEBUG=1 GGML_SYCL_ARENA_PP_PROFILE=1 \
     ./build/bin/llama-bench -m …/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 0 -r 1 -v
   ```

   `GGML_SYCL_ARENA_PP_PROFILE=1` adds `[ARENA-PP-ONEDNN] … reserve_req_mb=W/A`,
   the summed weights/activations requests — the only log that reports what was
   actually asked for.
2. **GPT-OSS never enters `reserve_onednn_scratch`** (observations = 0 on every
   run, including prompt processing) — its MoE work goes through a separate
   PP-MoE oneDNN ring. The grow path and its counters are exercised only by dense
   models, so a clean GPT-OSS run is not evidence the oneDNN predicate is right.
3. **`llama-bench` needs `-v`.** It installs a null log callback, so every
   `GGML_LOG_INFO` — all `[SYCL-PLAN]`, `[VRAM-ARENA]`, `[HOST-ARENA]` and
   `[MOE-LAYOUT]` lines — is silently discarded without it. An empty capture is
   indistinguishable from "the zone did not change"; this voided captures
   repeatedly while the work was being measured.
4. **`-p 0 -n 4` yields observations = 0 on every model and card.** A capture
   without prompt processing cannot detect an under-estimate at all. Use it for
   zone figures, never for coverage.

## See also

- [`docs/design/sycl-canonical-memory-architecture.md`](../design/sycl-canonical-memory-architecture.md)
  — enforceable contract: allocator allowlist, pointer-resolution allowlist,
  dispatch router, migration inventory.
- Source: `ggml/src/ggml-sycl/unified-cache.hpp` (allocator + planner types),
  `ggml/src/ggml-sycl/mem-handle.hpp` (handle kinds, resolution, lease
  semantics — the header comments are the primary spec).
- `ggml/src/ggml-sycl/zone-sizing.hpp` — the path-scoped sizing predicates,
  their threshold, and the `zone_*` / `zone_sizing_*` naming contract.
- `docs/plans/2026-07-25-zone-sizing-findings.md` — the measurements every byte
  figure in "Path-scoped zone sizing" above is quoted from (Tasks 1 and 8).
- `docs/plans/2026-07-25-sycl-path-scoped-zone-sizing.md` — the plan those tasks
  belong to; read its Amendments 1 and 2 first, as they supersede the body.
