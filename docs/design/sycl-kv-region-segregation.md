# llama.cpp-moua: planned, lifetime-segregated layout for the shared KV+WEIGHT zone

Design, revision 2. Author: impl-moua, 2026-09-26. This revision answers two reviews:
- design review r1 (design-moua-r1: 3 Critical, 7 Important, 9 Minor), recorded in §6.1;
- the principles audit's moua section (audit-mem-b: 5 Important, 4 Minor), recorded in §6.2.

Both reviews read revision 1.

Revisions cited:
- `master` = `401ff76cc`, the base of `task/moua`.
- `jehw` = `task/jehw` HEAD `4d41db5c8`. `u1bb` = `task/u1bb` HEAD `f01b3e86c`. Neither is in master.
- `fe6c` = `fe6c9356d`, the revision the A2 logs were taken on.

Every file:line below names its revision.

## 0. Summary

- **Root cause of A2** (B50, PCT=60, Mistral Q4_0, `-c 32768`): the TLSF allocator that KV and
  WEIGHT share in single-chunk mode is fragmented by staging order (§1). The revision leaves
  this unchanged.
- **Design:** the shared zone gets a planned layout, and a block's side is set by its
  *lifetime class*, which is derived from its role:
  - weights front-carve from offset 0;
  - optional (yieldable) tenants are staged after all load-time weight staging, at the weight
    frontier, and yield as a strict address-ordered prefix;
  - every context-lifetime and transient tenant top-carves from the high end.
- **KV regions:** each context's device KV is a **region**, registered under
  `(ContextId, device)`.
  - The region is reserved idempotently at the runtime-context transaction.
  - A context's buffers find it through a llama-side, ContextId-keyed **region scope**.
  - The fit and the reservation run the same pure function over the same geometry,
    re-checked under the lock.
- **Error path:** a device-planned KV layer that does not land in its region is a planner
  bug. It is refused with `[KV-PLAN-BUG]` at ERROR, and aborts under
  `GGML_SYCL_STRICT_KV_PLAN=1`. It is never admitted and then spilled.
- **Holes:** weight-side holes and buried optional tenants are usable region extents
  (regions are multi-extent). Only holes smaller than one slot are a limit (§2.9).
- **Landing:** L1 is done (`eab1ebeb6`, `9e0a708dc`, with the review fixes in `97315421b`). L2 is folded into L6, so no interim
  guard turns today's spills into refusals. L3-L7 need jehw and u1bb merged.

## 0.1 Acceptance conditions: the four principles

Every step of this design, and its reviews, are held to these. Each row names the mechanism
that enforces the principle and the check that shows it.

| | principle | enforced by | shown by |
|---|---|---|---|
| **P1** | The unified cache is the only allocator. There is no out-of-arena fallback for planned KV. | KV regions are carved from the arena TLSF (§2.4). The tiered device branch issues no `unified_alloc` (§2.6). A region reservation never falls through to `unified_cache_malloc_device_tracked` (§2.8). | H7a/b; C1 counts **zero** KV-role EXT-ALLOC lines, and total EXT-ALLOC bytes no higher than the base run (§3.3). |
| **P2** | `mem_handle` / `alloc_owner` are the only ownership surfaces. | A region is an owner-first `CACHE_SUBALLOCATION`. The registry, each KV buffer, and each layer's view hold its `mem_handle`. Slots are views on that handle, not allocations. Retained holes have no owner and are never handed out (§2.3). | G1 checks `arena_owns` and the handle refcounts. H7f checks that no raw pointer is stored as region state. |
| **P3** | Placement decides the executor. | Residency is still decided per layer by the planner. Host-tier layers keep executing on the CPU exactly as today, and only the capacity input changes (§2.5). | C2/C3 gates; the demotion WARN names the layers. |
| **P4** | Plan == reality. A KV region that does not fit is a planner bug: refused, never admitted and then spilled. | One function, `kv_region_fit`, decides "fits" for the planner, the reservation (re-run under the lock), the ring admission and the `-c` hint. Every second source of "fits" is deleted or rederived (§2.2). The carve commits exactly the extents the fit named. | H2 property test (fit ⇔ carve), H7d (one source), G1 forced mismatch ⇒ `[KV-PLAN-BUG]`. |

## 1. Root cause (the 32K case)

### 1.1 The path, file:line at fe6c9356d

1. **KV and WEIGHT share one TLSF in single-chunk mode.**
   - `unified_cache::arena_reserve` lays the chunk out as `[KV/WEIGHT shared][ONEDNN][RUNTIME][SCRATCH]`
     with `kz.start = wz.start = 0` and `kz.size = wz.size = shared_bytes`
     (`unified-cache.cpp:22326-22335`).
   - It gives WEIGHT a null allocator: "WEIGHT delegates to KV's allocator in single-chunk
     mode" (`unified-cache.cpp:22362`).
   - `zone_alloc(WEIGHT)` uses the KV allocator (`unified-cache.cpp:22810-22813`).
   - B50 at PCT=60 is single-chunk. The arena is about 8152 MB (6872 shared + 1280 tail),
     well under the 14714 MB safe cap (`[SYCL] Device 0 alloc caps` in a2_long.err:2), so
     `try_single_chunk` holds (`unified-cache.cpp:22279`).
2. **Every dense weight and every WOQ copy is a WEIGHT-zone allocation.**
   `direct_stage_weight` → `unified_cache_zone_allocate_owner(cache_device, vram_zone_id::WEIGHT, dst_size)`
   (`unified-cache.cpp:6034-6040`).
3. **Each copy is staged right after its own primary.** In the S1-PRELOAD dense loop the
   primary is staged at `ggml-sycl.cpp:34511`. Its WOQ copy follows in the same iteration,
   at `ggml-sycl.cpp:34554-34603`, before the next tensor's primary.
4. **The TLSF carves from the front, so address order equals allocation order.**
   `split_block` keeps the front of the chosen block and returns the remainder to the free
   list (`tlsf-allocator.hpp:228-258`). During a load there is one free block, so the zone
   fills `P1 C1 P2 C2 ... Pn Cn [tail ≈ 84 MB]`.
5. **The yield frees bytes, not an extent.**
   - The runtime-context transaction yields the copies (`ggml-sycl.cpp:17862-17888`).
   - It re-reads `unified_cache_kv_vram_available` (`unified-cache.cpp:21265-21280`),
     which returns `zone_available(KV)`.
   - `zone_available(KV)` returns `tlsf_allocator::available()`, i.e. `total_size_ - used_`
     (`tlsf-allocator.hpp:479`).
   - The re-fit therefore saw 2954.1 MB = 84.4 + 2869.8 and admitted 23 × 128 MB = 2944 MB
     (a2_long.err:6-7).
   - The real free blocks were one-copy holes, each fenced by primaries.
6. **The per-layer allocation misses the zone and spills.** `tiered_kv_buft_alloc_buffer`
   issues `unified_alloc` with `role=KV`, `must_device`, `prefer_vram_zone=KV`
   (`ggml-sycl.cpp:39060-39072`). In `unified_alloc` it goes:
   - `zone_alloc(KV)` fails and falls through, because `forbid_vram_zone_spill` is unset
     (`unified-cache.cpp:15031-15043`);
   - the P5 retry fails, with a DEBUG-only "trying raw device allocation"
     (`unified-cache.cpp:15045-15066`);
   - `ptr = unified_cache_malloc_device_tracked(...)` succeeds (`unified-cache.cpp:15089`),
     and the `[EXT-ALLOC]` trace prints `prefer_vram_zone=0`, which is `vram_zone_id::KV`
     (`unified-cache.cpp:15090-15099`);
   - `valid_device_kv_handle` accepts it, because it is DEVICE on the right device
     (`ggml-sycl.cpp:39031`).
   The only guard in front of it is the physical-VRAM overcommit check
   (`unified-cache.cpp:14963-15018`). That check uses `unified_alloc_total_vram`, not the
   PCT budget.
7. **The mismatch backstop cannot see it.** `kv_admission_mismatch` compares byte totals
   (`ggml-sycl.cpp:38767`), and bytes agree.

### 1.2 Why a zone mismatch is excluded

The copies and the KV are the same physical allocator (items 1-2). A zone mismatch would
need two allocators. The only other nullptr cause in `zone_alloc` is a closed allocator
group (`group.state != OPEN || !arena_zone_open`). Nothing closes the KV/WEIGHT group
between the yield and the KV allocation, and the WEIGHT-side staging in the same group ran
normally. The arithmetic in §1.3 predicts the observed 0-of-23 exactly without it.

### 1.3 Off-GPU reproduction with the real allocator

`scratchpad/moua-sim/sim.cpp` includes fe6c9356d's `tlsf-allocator.hpp` unmodified. It uses:
- a 6872 MiB zone;
- a 64 MiB load-time KV reserve, the planner's n_ctx=512;
- Mistral Q4_0 tensor bytes in S1 plan-rank order, with primaries totalling 3917.9 MiB;
- the S1 staging guard from `ggml-sycl.cpp:34571-34580`.

```
[interleaved(today)] primaries=3917.9 MiB copies=194 (2870.0 MiB staged) free_before_yield=84.1 largest=84.1
  after yield: free=2954.1 MiB largest_free=102.5 MiB -> 128 MiB KV layers that land: 0 (byte count admits 23)
[two-pass]           primaries=3917.9 MiB copies=194 (2870.0 MiB staged) free_before_yield=84.1 largest=84.1
  after yield: free=2954.1 MiB largest_free=2954.1 MiB -> 128 MiB KV layers that land: 23 (byte count admits 23)
```

The hardware showed 0 of 23 in the arena, i.e. 23 EXT-ALLOC lines. The sim picks 194
copies where the planner picked 182, because the selection order differs slightly. The
property does not depend on which copies are chosen. **Staging order alone decides it.**
jehw independently reached the same root cause (task comment c-r939). Its `4d41db5c8`
handles it by simulating KV placement on a snapshot of the TLSF and holding the re-fit to
what lands. That is correct, but it demotes all 32 layers, where the design below keeps 23
on the device.

## 2. Design

### 2.1 Layout and lifetime classes (single chunk)

```
offset 0                                                                        shared_bytes
[ WEIGHT side: primaries, late weights -> | OPT -> ] [   gap   ] [ <- CONTEXT side ]
                                           ^ weight frontier     ^ context frontier
```

A request's **lifetime class** is derived from `alloc_role`. The zone it names does not
decide it, because many non-weight requests name `vram_zone_id::WEIGHT` only to stay out of
the B50 tail zones (r1 C3). Examples: master `ggml-sycl.cpp:46120`, the per-op MMVQ scratch,
whose comment reads *"Keep tiny MMVQ Q8 activation scratch out of arena tail zones on B50"*;
and `ggml-sycl.cpp:97965`, the SOA-graph Q8_1.

| class | who | side | placement |
|---|---|---|---|
| `WEIGHT` | `alloc_role::WEIGHT`, including `EXPERT_CACHE` (vram-pool.cpp:81, unified-cache.cpp:18484: on-demand expert rows, which are real weights) | weight | TLSF `allocate` (today's behaviour), tag `TAG_WEIGHT`; after the optional ladder exists, first a weight-side hole, then the gap front (§2.3.3) |
| `OPTIONAL` | optional layout copies (jehw's `optional_layout`) | weight frontier | `allocate_gap_front`, tag `TAG_OPTIONAL` |
| `KV_REGION` | the per-(ContextId, device) KV region | context | the extents chosen by `kv_region_fit` (§2.4) |
| `CONTEXT` | u1bb ring KV-zone slots, `backend-buffer-kv-zone` compute overflow | context | `context_side_place` (§2.3.2) |
| `TRANSIENT` | every **non-WEIGHT role** that names `WEIGHT` or `KV`: COMPUTE/STAGING scratch (ggml-sycl.cpp:46120, :97965; mmvq.cpp:16863; unified-cache.cpp:21455; common.hpp:6658), persistent buffers (unified-kernel.cpp:4890), scratch_pool (unified-cache.cpp:20629), and the fattn packed-K/sidecar KV-role requests (fattn.cpp:569, :1640) | context | `context_side_place`, bounded by the planned transient reserve (§2.4.3) |

The TRANSIENT class stays in the shared chunk, so it keeps the reason those callers chose
`WEIGHT` (the chunk is not a tail zone). Moving it off the weight side fixes three of r1's
C3 consequences:
- **C3(i), burying:** it can no longer front-carve above the optional ladder and bury it.
- **C3(iii), churn:** per-op churn never touches the weight side. The generation counter it
  used to bump is gone anyway (§2.4.2).
- **C3(ii), missed allocations:** its bytes are a planned term of the fit (§2.4.3), so the
  region cannot consume the last of the gap and push these requests out of the arena.

**Staging order (C3(iv)).** Optional tenants are staged in a pass that runs after **all**
S1-PRELOAD weight staging: the dense loop *and* the MoE expert/DPAS staging (master
`ggml-sycl.cpp` ~34700-34760). Running it after the dense primaries alone would not be
enough.

**What can still bury the ladder.** A `WEIGHT`-class allocation made after the optional pass
that fits no weight-side hole: a runtime expert-cache fill, a re-layout, or a second model's
load. That front-carves above the ladder, and `frontier_walk` then stops at it.
- **This is not a P4 hazard.** The fit reads the actual block list, so what it reports is
  what lands.
- **It costs little capacity.** Buried lease-free optional tenants still yield when a
  context needs the room, as hole extents (§2.9, audit I5). What burying costs is
  contiguity: the room comes back as a separate extent, not as frontier growth.
- **Only a leased buried tenant is lost to KV.** The fit reports that, and the §2.9 WARN
  covers it.

### 2.2 One fact: every source of "fits"

Every site that decides whether device KV fits is listed here with what it becomes. H7d
enforces this list, with a mutation witness per row (r1 I1).

| site (master unless marked) | today | becomes |
|---|---|---|
| transaction re-fit, `ggml-sycl.cpp` ~17752-17932 (jehw ~17766+) | bytes (fe6c); jehw: `fit_capacity` from `kv_zone_snapshot` | `kv_region_fit` on the geometry snapshot, re-run under the lock at reservation (§2.4) |
| tiered `kv_admission_mismatch(planned_device_bytes, kv_vram_cap)`, `ggml-sycl.cpp:38675` | live `unified_cache_kv_vram_available` at buffer-alloc time | **deleted for arena devices.** Once the region is reserved, the live available excludes it, so this check would refuse every device-planned buffer. The region claim (§2.6) replaces it. It stays on the no-arena budget path. |
| `configure_with_weights(device, n_layers, kv_vram_cap, kv_slice)`, `:38701` | the same live cap | arena devices: the cap is the region's slot-table capacity from the registry |
| `kv_device_budget` byte path, `:38815-38965` (per-layer `total_device + layer_size <= kv_device_budget` at `:38965`) | free VRAM minus the compute reserve | arena devices: residency comes from the region slot table and is not consulted here. No-arena: unchanged |
| u1bb ring contiguity, u1bb `ggml-sycl.cpp:17605` (`zone_largest_free(KV)`) | the TLSF-wide largest free block, weight-side holes included | `context_side_place` on the same snapshot, with the region already reserved |
| `-c` hint, `ggml_sycl_largest_fitting_n_ctx_live` | bytes | `kv_region_fit` |
| the demotion WARN's "free for KV" | bytes | the region capacity the fit computed |
| `largest_free_block()` anywhere in a fit decision | approximate (the head of the highest SL list) | never; fits read `frontier_walk` / `gap_below`, which are exact (L1, `9e0a708dc`) |

### 2.3 The context side (unified-cache, L4)

#### 2.3.1 Structure and locking

Each shared TLSF has one `context_side`: an address-ordered list of blocks
`{offset, size, state LIVE|RETAINED, class, owner}`. Its **anchor** is the lowest
LIVE-or-RETAINED block.

Every read and write of `context_side` and of the TLSF runs under
`unified_cache::arena_allocator_group_mutex(zone)`. That includes the geometry snapshot
(r1 044k; L1 documents the contract at `9e0a708dc`). The snapshot copies out a POD
`shared_zone_geometry` and releases the lock. The pure fit then runs on the copy.

Lock order, in canonical contract §12.5's ranks:

```
L3 kv_region_mutex_ (new; per device)  ->  L4 direct_stage_mutex_, rw_mutex_  ->  L5 arena_allocator_group_mutex(zone)
```

- `kv_region_mutex_` is a new lock. §12.5 says *"an unclassified lock is a failing census
  item"*, so it is ranked **L3**: a keyed-registry lock beside `g_pending_kv_layer_masks_mutex`
  and the planned keyed `context_graph_mutex[ContextId]`, with ascending device ID as the
  tie-break.
- L4 adds it to the §12.5 table in the same commit that introduces it.

- The existing order `cache locks -> group lock` (staging calls `zone_alloc` under the cache
  locks) is preserved.
- `kv_region_mutex_` serialises region reservations and registry edits on a device. It is
  never taken by the yield path, `zone_free` or staging, so it adds no cycle.
- **No code holds the group mutex while calling the yield** (r1 C2, §2.4.2).

#### 2.3.2 Placement

- **The KV region:** `carve_kv_region(extents)` carves exactly the extents the fit returned
  (§2.4). There is no search.
- **CONTEXT/TRANSIENT:** `context_side_place(g, size)` is pure. Both the admission of a
  planned CONTEXT tenant (the ring) and every CONTEXT/TRANSIENT allocation call it, so the
  admission and the placement use one policy (r1 I4, second half). The policy:
  1. best fit over RETAINED runs;
  2. else `allocate_below(anchor)`, or `allocate_top` when the side is empty.
  It never front-carves.
- **Reusing a retained run (r1 M3)** is one critical section under the group lock:
  1. `free(run)`; the run's neighbours are allocated, so it cannot coalesce onto the weight
     side;
  2. carve the new block from the run's top with `allocate_below(upper neighbour)`, or
     `allocate_top` when the run is the physically last block;
  3. immediately re-carve the remainder with `allocate_below(new block)`, which takes the
     whole run when it is too small to split, and record it RETAINED again.
  No free block ever exists on the context side outside this section, so the weight side's
  first-fit `allocate` can never see a context-side hole.

  L1 has no split-allocated primitive. This free/carve/re-carve sequence is how L4 composes
  one from L1's calls, and H4 pins it.
- **Free:** freeing the anchor block returns it, plus every RETAINED run now reachable from
  the frontier, to the TLSF, where they coalesce into the gap. Freeing an interior block
  marks it RETAINED, and adjacent RETAINED runs merge.
- **A retained run has no owner and no registration (audit I3).** A retained run stays
  TLSF-allocated but is not a live allocation. Three rules, all run under the group lock:
  1. **At retain time,** the freeing owner's exact record and `allocation_id` are removed
     (`arena_unregister_exact`) *before* the block is marked RETAINED. This is the same order
     a normal `zone_free` follows. A stale pointer into the run then resolves to nothing in
     `unified_lookup`, never to the dead owner.
  2. **Reuse is owner-first.** The M3 sequence above mints the new block's control (a fresh
     `alloc_owner`, class derived from the request, §2.10) before the carve. It registers the
     carved extent with `arena_register_exact` under the new `allocation_id`. The re-retained
     remainder is registered to no one.
  3. **Rebuild and destroy treat retained runs as free.** The arena rebuild/destroy and
     settle paths decide "live" by registered allocations, which retained runs never are.
     Every place that reads a shared TLSF's `used()` as "live bytes" subtracts
     `context_side.retained_bytes`: `zone_used`/`zone_available` reports, the settle
     precondition, and leak and quiescence checks. L4 enumerates those readers, and H4
     asserts them.

#### 2.3.3 Weight-side placement after the optional pass

`zone_alloc(WEIGHT)` works as follows once an optional ladder is live on that TLSF:
1. first-fit into a weight-side hole: a free block that is **not** the gap block;
2. only then the gap front, with the burying WARN of §2.1.

This needs one new L1-level primitive, `allocate_excluding(anchor, size, align, tag)`. It
takes the gap block out of the free lists, calls `allocate`, and reinserts the gap. It lands
in L4 as an L1 follow-up, with its own host test.

#### 2.3.4 Reset, settle, and the dead KV reclaim

- **zone_settle / TLSF reset (r1 M2).** Any path that `reset()`s a shared TLSF clears that
  TLSF's `context_side` in the same critical section. Retained runs are free space, so
  dropping them is correct. The settle's existing precondition (no live registered
  allocations) already excludes live regions.
- **`zone_reclaim(KV)`.**
  - The bulk reclaim in `arena_reserve` (fe6c `unified-cache.cpp:22171`) is removed.
  - The second caller, master `ggml-sycl.cpp:37301`, sits in the
    `mem_policy == GGML_SYCL_MEM_POLICY_KV_AUTO` branch (`:37163`). That branch is **dead**:
    every buffer type is constructed with `GGML_SYCL_MEM_POLICY_STATIC` (`:37060`, `:37661`,
    `:37690`, `:42468`, `:42578`), and nothing assigns `KV_AUTO`.
  - L4 removes that branch too (r1 M4).

#### 2.3.5 N-chunk arenas

There is one `context_side` per TLSF (each weight chunk plus the tail chunk's KV TLSF). A
region may span extents on several TLSFs, and within one TLSF it may use several extents
(§2.4.1). The fit packs slots across all of them with the same greedy rule, so N-chunk is
not a separate policy.

C0 decides on hardware whether the B70 exercises this path. B50 at PCT≤100 is single-chunk
(§1.1).

### 2.4 The fit and the reservation

#### 2.4.1 `kv_region_fit` (pure, SYCL-free, `kv-runtime-demotion.{hpp,cpp}`, L3)

```
kv_region_fit(const shared_zone_geometry & g, const kv_region_request & r) -> kv_region_fit_result
```

**Inputs.** The geometry `g`, per TLSF of the device:
- `gap`;
- `optional_ladder`: the `frontier_walk` output in LIFO order, truncated at the first tenant
  that fails jehw's lease predicate, because skipping a leased tenant would leave a hole;
- `retained_runs`;
- `weight_holes`: free blocks that are neither the gap nor part of the frontier walk. They
  are **usable as region extents** (audit I5, below). They were reported-only in revision 1;
- `buried_optional`: lease-free `TAG_OPTIONAL` blocks outside the frontier walk, i.e. below a
  weight that buried them, each with the free run their release would form. This needs a
  whole-TLSF block census from the snapshot, which is a second L4 read primitive beside
  `frontier_walk`;
- `self_extents`: this ContextId's existing region on this device, when the request is a
  same-shape republish (§2.5).

The request `r`:
- the slot sizes, from `kv_layer_alloc_bytes()` (jehw's single definition), grouped
  full-attention first, then SWA, each group in layer order;
- the planned `transient_reserve` (§2.4.3).

**Output.**
- `fits`;
- the per-layer residency (device or host) after the demotion loop;
- the strict LIFO `yield_prefix` count per TLSF;
- the chosen `extents` (an ordered list of `{tlsf, offset, size}`);
- each device layer's `(extent, slot_offset)`.

**Rules.**
- **Extents (r1 I4; audit I5).** A region may use several extents within one TLSF, of three
  kinds:
  - the **frontier** extent: the gap grown by the yield prefix;
  - **retained** runs on the context side;
  - **hole** extents on the weight side: a weight hole, or the run formed by releasing
    buried optional tenants.

  Slots are packed greedily in slot-table order: best-fit retained and hole extents first,
  then the frontier. A slot is never split.
  - A hole extent is carved from the top of its run with `allocate_below(the allocated block
    above it)`. Only the buried optional tenants that intersect the carved range are
    released.
  - When the region is freed, a hole extent is returned straight to the TLSF, where it
    becomes a weight hole again. Only frontier and retained extents follow the context-side
    free rule of §2.3.2.
  - A hole extent is never inside the frontier walk, so it cannot truncate the ladder.
  - **One owner per extent (audit I2).** Each extent is its own owner-first
    `CACHE_SUBALLOCATION` control and `mem_handle`. The registry entry, each KV buffer and
    each layer view hold the handles of the extents they touch. A slot is
    `extent_handle.slice(slot_offset, slot_size)` (mem-handle.hpp:400) of its own extent's
    handle. This was already forced by N-chunk regions, and it is now the only form. Revision
    1 wrote "one region handle".
  - The only cost of more extents is one clear per extent instead of one memset.
    `alloc_base_is_arena` holds per extent.
  - This replaces revision 1's "one extent per TLSF". Under that rule a retained run smaller
    than a whole region was useless to the next context.
- **Carve mirroring (r1 M1).**
  - The frontier extent is top-carved below the anchor. When the gap left over after the
    extent is below `MIN_BLOCK_SIZE`, L1's `carve_gap` takes the whole gap, and the extent's
    base is then the gap start, which is up to 255 B below `anchor - size`.
  - The fit computes offsets with the same rule, so its slot offsets are the carve's.
  - Slot *sizes* are 512 B multiples, and slot offsets are relative to the extent base.
    Absolute alignment is therefore 256 B, which is what P5 promises today (MIN_BLOCK_SIZE),
    and nothing claims more.
- **The demotion loop.** It is unchanged in order and meaning: latest full-attention layers
  first, then (never shrink context) latest SWA layers, to the host tier. Each step asks
  `fits` over the extents instead of comparing bytes.
- **The yield prefix is strict (r1 M7).**
  - jehw HEAD `4d41db5c8` already selects copies by address, from the top, but it *prunes*
    copies whose release would not help.
  - A pruned interior copy fences the freed ones above it and leaves a hole, so this design
    needs the strict prefix: the minimal count `k` such that releasing the top `k` ladder
    entries makes the frontier extent large enough.
  - This is a smaller change to jehw's selection than revision 1 framed: drop the pruning
    and keep the address order.
- **Transient reserve.** `fits` requires `frontier extent after the region ≥ transient_reserve`,
  so the region never consumes the gap that TRANSIENT/CONTEXT tenants need (§2.4.3).

#### 2.4.2 `reserve_kv_region` and the transaction step (L4 + L6)

For each device `d` of a context `c` in publish mode, with `kv_region_mutex_(d)` held:

1. **Idempotence (r1 C1).**
   - If the registry has `(c, d)` with the **same shape key** (n_ctx, n_seq_max, kv_unified,
     swa_full, type_k/type_v, the slot table), return it. There is no re-fit and no side
     effect. This is today's `admitted_kv` rule, *"a same-shape republish by an admitted
     context keeps the published residency"* (jehw `ggml-sycl.cpp:17785` block, master
     `:17771`), made physical.
   - It makes the split case safe: every backend's run of the transaction re-fits every
     device (master `:17771-17772`, *"each re-fits, because admission is per backend"*).
     The first run to reach `(c, d)` reserves, and every later one finds it.
   - It also makes the auto micro-batch ladder safe: master `llama-context.cpp` republishes
     via `sycl_resync_runtime_context_flash_attn()` per candidate (`:1571`) and at the settle
     (`:1936`), with only `n_ubatch` changed, which does not change the KV shape.
2. **Plan, with no lock held.**
   - Snapshot the geometry: copy it out under the group mutex, then release.
   - Run `kv_region_fit`. That yields the residency, the yield prefix and the extents.
3. **Yield, with no lock held (r1 C2).**
   - Call `yield_optional_prefix(d, k)`. This is jehw's `yield_optional_layouts` (jehw
     `unified-cache.cpp:7737-7880`) with a strict-prefix selection.
   - It takes `direct_stage_mutex_` and `rw_mutex_` itself, waits on its barrier with no lock
     held, and drops the deferred rows with no lock held. That drop is what reaches
     `zone_free` and the group mutex.
   - `kv_region_mutex_` is still held, but the yield never takes it, so there is no cycle.
   - The rewrite keeps jehw's reader barrier gate and the optional-layout epoch bump
     unchanged (audit m4). It only changes *which* copies are picked: the strict frontier
     prefix plus the buried tenants the fit chose.
   - A yielded copy is never lazily re-staged at dispatch. Dispatch reads the primary's
     materialized layout (P3, "the loaded layout is the answer"). H7 carries a source check
     for this.
   - Inherited, not widened: jehw's oneDNN WOQ copy readers take no lease, and recorded exec
     graphs bake the copies' pointers (610516bd7's own message). That is a P2 migration site
     moua inherits. The barrier gate and epoch bump are what make the release safe today,
     and moua keeps both.
4. **Commit, under the group mutex.**
   - Re-snapshot, and re-run `kv_region_fit` on the live geometry.
   - If the result equals step 2's (same residency, same extents), carve each extent
     owner-first (one control per extent) and register
     `(c, d) -> {extent mem_handles, shape key, layout}`.
   - If it differs, release the lock and go back to step 2. This replaces revision 1's
     global `geometry_generation`, which every per-op scratch alloc/free would have bumped
     (r1 I5, C3(iii)). Staleness is decided by re-running the fit and comparing its result,
     not by a counter.
5. **Shortfall is a runtime outcome, not a planner bug (r1 C2).**
   - The yield may return `freed < picked`, via jehw's `pending_bytes`, or when its barrier
     wait fails (the frees stay queued).
   - The re-fit in step 4 then sees less space, and the loop re-plans with the real geometry:
     it demotes more layers.
   - Demotion always terminates, because all layers on the host always fit. The loop is
     bounded at 3 iterations. After that the transaction refuses with a named cause
     (`region geometry kept moving`), never PLAN_BUG.
6. **PLAN_BUG is only this:** the step-4 re-fit said `fits`, and carving those extents
   failed. That is an allocator bug by construction. §2.8 covers it.

**Multi-device atomicity (r1 I5).**
- Devices are reserved in device order. When device `j` refuses, the transaction erases every
  registry entry it created for `c` in this call. No buffer references those regions yet, so
  erasing drops the last reference and returns the extents. Entries found in step 1
  (pre-existing) are left alone.
- Yields already performed are **not** undone. They released optional tenants, which is a
  prompt-processing performance cost, never a correctness one. The yield WARN names them.
- A retry within the loop counts `c`'s own just-created regions through `self_extents`, so a
  retry never competes with itself.

**Shape change.**
- A different shape key for an existing `(c, d)` erases the old entry. The old region stays
  alive through the old KV buffers' references and is released when llama frees them.
- The fit does **not** count it as a self extent, because it is still physically occupied.
  The existing WARN ("the re-fit counts still-allocated KV as used") stays.

**Probe mode** runs step 2 only, with no side effects.

**Teardown.** The exec-context close path erases `c`'s registry entries. That path is master
`llama-context.cpp`'s `llama_context_sycl_exec_drain_and_close` and its backend drain; the
buffers themselves go with llama's memory. The last `mem_handle` reference frees the region
through `zone_free`, which runs the §2.3.2 free rule.

#### 2.4.3 The transient reserve

The TRANSIENT and CONTEXT tenants (§2.1) have no plan today. llama.cpp-zhcn covers the
compute overflow, and llama.cpp-23mk the other non-KV out-of-arena allocations. This design
does not plan them; it **reserves room** for them so KV cannot starve them:
- `transient_reserve(d)` is a named pure function in L3. It returns the u1bb ring's admitted
  KV-zone bytes, plus u1bb's compute-overflow estimate (`k_pp_moe_ring_compute_reserve_bytes_per_row`
  × rows, the existing interim number), plus a fixed per-device scratch floor.
  - The floor is initially the largest single TRANSIENT request observed in the A2 and
    GPT-OSS gate logs, rounded up. L3 records the measurement it came from.
- The fit leaves at least that much in the frontier extent.
- A TRANSIENT request that still misses behaves as it does today; that is 23mk's scope.
  C1 checks that moua does not make it worse (§3.3).

### 2.5 The region registry and the region scope (r1 I2; decision (d) revised)

Revision 1 carried the region on the KV-mask `handoff_id`. That is withdrawn for four
reasons, each verified:
- The mask handoff correlates on `ModelToken`, not the context. `test-thread-safety` creates
  same-model contexts concurrently (master `tests/test-thread-safety.cpp:209`,
  `threads.emplace_back`), so FIFO pops can cross-wire regions.
- Its id is minted by `llama_kv_cache`'s push (master `src/llama-kv-cache.cpp:410`).
  That is after the transaction, and only when the mask is non-empty.
- It is one pop per buffer, while iSWA puts two buffers on one region.
- A layer whose planned owner is not the buffer's device must reach the *owner's* region.

**Identity.** The context's existing SYCL exec context id, `ggml_sycl_exec_context_id`, is
used as the key. Master `llama-context.cpp` creates it and binds it to every SYCL backend
(`:796`, `:803`) *before* the first `sycl_resync_runtime_context_flash_attn()` (`:810`) and before
`memory.reset(model.create_memory(...))` (`:869`). So every transaction run sees
`ctx->execution_context_id` (master `ggml-sycl.cpp:11852`), and it is fixed for the
context's lifetime.

**Registry** (`unified_cache`, per device, under `kv_region_mutex_`):
`ContextId -> {mem_handle region, shape key, kv_region_layout}`. The transaction writes it
(§2.4.2), the claim reads it, and teardown erases it.

**Scope (the llama-side change).** Two new exported procs, resolved the same way as the other
`llama_context_sycl_*` procs (DL-safe, absent means no-op):
- `ggml_backend_sycl_kv_region_scope_begin(ggml_sycl_exec_context_id)`;
- `ggml_backend_sycl_kv_region_scope_end()`.

`llama_context`'s constructor wraps `memory.reset(model.create_memory(...))` in an RAII
guard, so the scope ends on a throw as well. The backend keeps the scope in a
**`thread_local`**:
- The memory is created synchronously on the constructing thread.
- Concurrent contexts on different threads therefore cannot see each other's scope.
- A nested begin on one thread is an error (GGML_ABORT with both ids), because it has no
  meaning.

**Attach semantics.**
- Every tiered KV buffer allocated inside the scope attaches to the registry, with no pop and
  no count. iSWA's FULL_ATTN_ONLY and SWA_ONLY buffers both attach to the same region.
- For each device-planned layer `l` with planned owner `o`, the claim looks up
  `registry(o)[ContextId]` and takes slot `l`. It reaches the owner's region even when the
  buffer's device differs.
- The KV-mask handoff is unchanged and still carries llama's residency view. The claim checks
  **mask == slot table**, and a mismatch is `[KV-PLAN-BUG]` naming both. So the mask is
  verified against the one source, not trusted as a second one.

**No scope.** An arena device with device-planned layers and no open scope is a caller
contract violation. It is refused with `[KV-PLAN-BUG] ... no KV region scope`. L6
enumerates the tests that allocate tiered KV buffers directly (the
`test-sycl-lifecycle-*` family and the kv-layer-sizing source gates) and gives them a scope
through a `GGML_SYCL_PRIVATE_TESTING` hook.

### 2.6 `tiered_kv_buft_alloc_buffer` (L6)

The device-planned branch (master `ggml-sycl.cpp` ~38590-39200):
1. **Resolve the scope, and for each owner device the registry entry.** Check that every
   device-planned layer has a slot whose size equals `kv_layer_alloc_bytes(layer)`, and that
   mask == slot table.
2. **Set `layer_allocs[l]` to a slice.** `kv_layer_alloc::set_owner` takes a legacy
   `alloc_handle` today (master `ggml-sycl.cpp:37727`). L6 adds a `mem_handle` overload
   (audit m1):
   - it stores the slot slice `extent_handle.slice(slot_offset, slot_size)` as
     `zone_handle`/`chunk_lease`;
   - it takes `ptr` from `slice.resolve()`, never from a separately computed base+offset,
     which would give the pointer a second source.
   The buffer holds every extent handle it touches, and the last reference releases each
   extent.
3. **Clear per extent.** When the buffer's slots are one extent, `alloc_base` is that
   sub-range and `alloc_base_is_arena` keeps the single memset (llama.cpp-zhzbp). Otherwise
   `tiered_kv_buffer_clear` loops over the extents.
   - Each fill's event lease holds copies of the slices it writes until the fill event
     completes (contract §12.6; audit m3). The buffer being freed first cannot release them
     under a queued fill.
4. **VMEM: the region takes precedence (audit I4; supersedes r1 M5's disposition).**
   - The opt-in `GGML_SYCL_VMEM_KV=1` branch (master `:38846-38925`) runs only under an
     active arena, runs before the per-layer path, and returns early. It maps KV in physical
     pages outside the unified cache's accounting.
   - Revision 2's "skip the region when vmem applies" would have kept planned device KV on
     that out-of-accounting path, which is a P1 violation.
   - Instead, under an active arena, device-planned KV **always** takes the region claim.
     The vmem branch is skipped, and `GGML_SYCL_VMEM_KV=1` produces one WARN per process:
     `[SYCL] GGML_SYCL_VMEM_KV is ignored while the VRAM arena is active: planned KV is
     reserved in the arena (llama.cpp-moua); pattern #2 (llama.cpp-1oxa) replaces vmem-kv`.
   - With no arena, vmem-kv behaves as today. 1oxa deletes it.

No device layer issues a per-layer `unified_alloc` any more. The host-tier branch is
unchanged. The arena-device uses of `kv_admission_mismatch`, `kv_vram_cap` and
`kv_device_budget` are deleted per §2.2.

### 2.7 u1bb ring, MMID pools, RUNTIME, compute overflow (r1 M9 corrected)

- **The ring's KV-zone slots** are CONTEXT-class and placed by `context_side_place`. The
  ring's admission reads the same function (§2.2, row "u1bb ring contiguity").
- **Revision 1's claim that ring slots "cascade because they are lowest" was false.** The
  compute-buffer overflow (`backend-buffer-kv-zone`, master `ggml-sycl.cpp:37313-37342`) is
  allocated by `sched_reserve` *after* the publish, so it sits below the ring. When the
  auto-ubatch ladder republishes, u1bb releases and re-admits the ring while the previous
  candidate's overflow is still live below it. The ring's old slots become RETAINED runs.
- What happens next, and the bound:
  - A re-admitted ring of the same or smaller size reuses the run by best fit (§2.3.2), with
    the remainder re-retained.
  - A larger ring goes below the overflow.
  - The context side can therefore accumulate at most one RETAINED run per ladder candidate.
    The ladder is bounded (master `llama-context.cpp` "tried %s" list), and the settle's
    final `sched_reserve` frees the last overflow.
  - These runs are context-side only, never weight-side, and they are visible to the fit as
    retained runs. Multi-extent regions (§2.4.1) keep them usable.
  - H4b replays the ladder sequence and pins the bound.
- **MMID pools and the ring's weight slot** stay in RUNTIME, a separate TLSF. No change.
- **ONEDNN and SCRATCH tail zones:** no change.

### 2.8 The error path (decision (a), scoped per r1 I3)

Refuse is the default; `GGML_SYCL_STRICT_KV_PLAN=1` aborts instead, mirroring
`GGML_SYCL_STRICT_LEASES`. The rule applies **only at region-backed sites**:

1. **`reserve_kv_region` step 6:** the fit said `fits` under the lock, and the carve failed.
   ```
   [KV-PLAN-BUG] device 0 ctx 7: the region fit placed 23 slots (2944.0 MB) in 1 extent(s)
     [gap 84.4 MB + 21 optional tenant(s) 2869.8 MB] and the carve failed at extent 0
     (offset 0x..., 2944.0 MB): the plan and the allocator disagree -- refusing
     (no out-of-arena allocation, no demotion). llama.cpp-moua
   ```
2. **The tiered claim:**
   - no scope;
   - no registry entry for a device-planned owner;
   - a device-planned layer without a slot;
   - a slot of the wrong size;
   - mask != slot table.
   ```
   [KV-PLAN-BUG] device 0 layer 17 ctx 7: device-planned KV layer has no slot in the context's
     region (region 2944.0 MB, 23 slots; mask says device, slot table says host) -- refusing
   ```
   It returns nullptr, and context creation fails loudly. The existing DIVERGENCE
   diagnostic (master ~39170-39192) stays for the host-vs-device plan check.

**Why `unified_alloc` itself is no longer an ERROR site (r1 I3).** A guard keyed on
`role == KV` would catch the fattn packed-K/sidecar requests (master `fattn.cpp:569`,
`:1640`). Those are lazy, unplanned and KV-role. The guard would then label them *"the plan
admitted it"* on every dispatch, which is false, and STRICT would abort inference.

Those requests are TRANSIENT (§2.1). They get `forbid_vram_zone_spill`, set silently, so a
miss returns failure into fattn's existing non-packed fallback
(`!allocation || tier != DEVICE_VRAM`). That is P1-compliant. It is a behaviour change only
when the context side is full, and C6 is there to catch any throughput effect.

**L2 is folded into L6.** Landing a refusal before the region path exists would turn
today's working spills into refused contexts (r1 I3: u1bb's ring slots, allocated in the
transaction before the KV buffers, can take blocks jehw's snapshot assigned to KV layers).
There is no interim WARN-only step, because nothing lands between jehw/u1bb and L6 that
needs one.

### 2.9 Weight holes and buried optional tenants (decision (c) revised; r1 I7, audit I5)

Revision 1 made weight-side holes unusable for KV, and it stopped counting optional tenants
once a later weight buried them. Two consequences followed, and both reviews flagged them:
- **r1 I7:** a server model swap (load the new model, then free the old one) produces exactly
  those holes, and it is the common case.
- **audit I5:** after model 2's primaries bury model 1's optional copies, KV demotes to the
  host *while those copies still hold VRAM*. That breaks the owner's rule that KV wins over
  optional layouts.

Revision 1 needed the limit only because a region was one extent. Now that regions are
multi-extent (r1 I4, §2.4.1), both go away:
- **Weight holes are region extents.** The fit packs whole slots into any weight hole that
  holds at least one. This is safe: the hole extent returns to the weight side when the
  context ends, and it never sits inside the frontier walk.
- **Buried optional tenants yield like frontier ones.** The fit treats a lease-free buried
  `TAG_OPTIONAL` block as releasable, together with its free neighbours, as a hole extent.
  The yield releases only the ones the chosen extents intersect, through the same
  barrier-gated path. A leased buried tenant is not releasable, and its run is split
  around it.

**The fix proposed instead, and why it was not chosen.** The alternative was "yield model 1's
optional tenants before model 2's primaries stage". It would release model 1's copies on
every second-model load, whether or not any context ever needs the space. That costs model
1's prompt-processing layouts for nothing. The lazy form above releases them only when a
context's KV actually needs the room, which is exactly "KV beats optional layouts".

**What remains a limit, and its diagnostic.** A hole smaller than one slot cannot hold KV;
slots are never split. Whenever the demotion loop demotes a layer while the geometry holds
weight-side free bytes, log at WARN:
```
[SYCL-PLAN] KV overflow on device 0 with 38 MB of weight-side free space in holes smaller
  than one 128 MB slot (pattern #1 never splits a slot; llama.cpp-1oxa removes this limit)
```
L7 documents this limit, and pattern #2 remains the remedy.

**This supersedes lead decision (c)** ("accept the limit"). It needs the lead's
confirmation, not an owner ruling: it only removes a limit, and it adds no policy.

### 2.10 The canonical memory contract

- **§3 allocator allowlist.**
  - Each region extent is an owner-first `CACHE_SUBALLOCATION` through `unified_cache`, and
    slots are `slice()`s of their extent's handle.
  - **The new entry points are added to §3's allowlist** (audit m2): `reserve_kv_region`,
    `zone_alloc_optional`, and `context_side_place`'s allocating wrapper.
  - The allocation class of each is **derived from the request**
    (`role` plus `prefer_vram_zone`, plus the §2.1 lifetime class) by the existing
    classifier. It is never hand-set at the call site.
  - No new raw allocation site is added, and no new `CACHE_BACKING` mint.
  - The one route by which planned KV reached `unified_cache_malloc_device_tracked` under an
    arena (the per-layer tiered `unified_alloc`) is removed.
- **§1.1 planner authority.** The planner decides residency per layer, as today. Its capacity
  input is the allocator's geometry, and an admission is a reservation, so "planned device"
  means physically reserved.
- **`mem_handle` lifetime.**
  - A region is freed only when its last handle drops: the registry entry plus the KV buffers
    and layer views.
  - There is no forced eviction. The yield takes only optional tenants that pass jehw's lease
    predicate, and a leased tenant truncates the ladder.
  - Retained runs are TLSF-allocated storage with no owner and no registration (§2.3.2,
    audit I3). They are kept off the TLSF's free lists so that weights cannot take them.
    They are not leaked handles, and every "live bytes" reader subtracts them.
- **§12.5 locking.** The new `kv_region_mutex_` is ranked L3, and the order is L3 → L4 → L5
  (§2.3.1). No L5 lock is held across a wait or across the L4 yield (§2.4.2).
- **§12.6 event leases.** The KV clear's fill events retain the slices they write (§2.6).
- **§5.2/§5.3.** KV becomes context-keyed per (ContextId, device), which is half of the
  "context-keyed KV/RUNTIME arena reservation" §5.2 lists as missing. RUNTIME stays per
  device. Same-device concurrent inference remains unsupported (§5.3).
- **Docs (L7).**
  - `docs/backend/sycl-memory-design.md` gets a "Shared-zone lifetime segregation" section:
    the classes, the registry and scope, the limits of §2.9, and the lock order.
  - The `unified-cache.hpp:68-78` arena comment is rewritten.
  - Contract §3 notes that planned KV never spills under an arena, and §5.2/§5.3 are revised.
  - This design doc moves under `docs/design/` history once L7 lands.

### 2.11 Out of scope

- Pattern #2, the VM-backed arena, is llama.cpp-1oxa. The tag policies above stay in the USM
  backing, behind the zone allocate path, so an `arena_backing` interface can separate them.
  1oxa's VM backing gives each class its own VA sub-range instead.
- Pre-existing out-of-arena paths are tracked elsewhere, not here:
  - llama.cpp-23mk covers `onednn_weights_scratch` (113.5 MB in A2), the `cohort=?` 4-byte
    STAGING allocations, and `backend-buffer-kv-zone` without `forbid_vram_zone_spill`
    (master `ggml-sycl.cpp:37235`). Its comment c-m2jh adds `backend-buffer-runtime-zone` and
    the tiered KV raw fallback, which moua's L6 removes.
  - **llama.cpp-gxur** (filed for this revision) covers the mechanism itself: with an arena
    active, `unified_alloc`'s raw `unified_cache_malloc_device_tracked` fallback (master
    `unified-cache.cpp:14921`) is fail-open by default for every `must_device` role without
    `forbid_vram_zone_spill`. gxur flips that default after 23mk and moua L6 land.
- The fattn-onednn.cpp:1011 `< 2^47` host-pointer heuristic is 1oxa's phase 1. moua's
  regions are USM device memory and are unaffected.

## 3. RED-first test plan

### 3.1 Host tests (no GPU; subagents can run them)

- **H1 `test-tlsf-allocator`** (L1, done at `eab1ebeb6`). Registered as `hostonly`, with 22
  cases.
  - The A2 shape on the real header lands **0/23 interleaved and 23/23 segregated**.
  - L4 adds `allocate_excluding` cases.
  - `TAG_OPTIONAL != 0` is a `static_assert` where the tags are defined (r1 M6), because
    `frontier_walk` and `tag_at` use 0 to mean untagged.
- **H2 `test-kv-runtime-demotion`: A2 fixture** (L3). RED restated for the post-jehw base
  (r1 I6):
  - jehw `4d41db5c8` already holds `fit_capacity` to what lands, so the "admitted 23, landed 0"
    RED of revision 1 does not reproduce there. The fixture yields 0 admitted and 0 landed,
    and fit == land passes vacuously.
  - The RED is therefore a **capacity** RED: on the interleaved fixture, *expected 23
    device-resident layers, got 0*, in jehw's pipeline, which demotes all 32.
  - GREEN, on the segregated fixture through `kv_region_fit`: 23 device, 9 host, and a
    yield prefix equal to the minimal strict prefix covering `2944 MiB + transient_reserve −
    84.1 MiB`.
  - **Property test (P4):** seeded random sequences of WEIGHT/OPTIONAL/TRANSIENT/region
    requests and frees, on the real TLSF plus the `context_side` model. For every region
    request, `kv_region_fit(snapshot).fits` must hold exactly when the carve succeeds, and
    the carved extents and slot offsets must equal the fit's. jehw's `kv_zone_snapshot`
    simulation is the second oracle for the byte totals.
- **H3 heterogeneous slots.** SWA plus full layers; per-layer truth (gemma-like); split K/V
  dims. Checks:
  - the slot table groups full attention first, then SWA;
  - demotion takes latest-full first, then latest-SWA, asking the fit at each step;
  - multi-extent packing never splits a slot.
- **H4 context side** (L4, on a host model of the shared TLSF plus `context_side`). The
  scenario is consistent this time (r1 M8): contexts A, B and C reserve in that order, so A
  is highest, B is in the middle, and C is lowest (the anchor).
  - Free B (interior): it becomes a RETAINED run, and the TLSF sees no new free block.
  - Reserve D, which fits the run: D reuses it by best fit, and the remainder is re-retained
    in the same critical section, so no free block ever appears on the context side (M3).
  - Reserve E, which does not fit the run: E top-carves below C and becomes the new anchor.
  - Free E, then C: each cascade returns the anchor plus every retained run reachable from
    the frontier to the gap.
  - Throughout, tags show no context-side bytes on the weight side.
  - Registration hygiene (audit I3), checked on the host model of the registry:
    - retaining B removes B's exact record before the RETAINED mark;
    - a lookup of B's old base resolves to nothing;
    - D's reuse registers D under a fresh `allocation_id`;
    - `live_bytes = used() − retained_bytes` at every step.
- **H4b auto-ubatch ladder replay (r1 M9).** Replays `publish(ring_k) -> sched_reserve(overflow_k)`
  for the candidates, then the settle. It asserts:
  - the RETAINED run count stays at or below the number of candidates;
  - after the settle, fit == carve for a fresh context.
- **H5 multi-model and burying (r1 C3, I7).**
  - Model 2's weights and a runtime EXPERT_CACHE fill land in weight holes first
    (`allocate_excluding`). Only when none fits do they bury the ladder.
  - The ladder then truncates at `TAG_WEIGHT`, and the burying WARN fires once.
  - A TRANSIENT request never front-carves (the class routing is pinned).
  - **Hole extents and buried optional tenants (audit I5).**
    - Model 2 buries model 1's optional copies. A context whose KV needs the room must then
      get device slots in the run formed by releasing only the intersecting buried copies.
      The capacity RED on revision 1's fit is "expected N device layers, got 0 while M
      optional bytes stay resident".
    - Out-of-order unload leaves a weight hole that holds two slots, and the region uses it.
    - A hole smaller than one slot stays unused, and the sub-slot WARN fires.
    - A leased buried copy splits its run and is never released.
- **H6 N-chunk.** Two weight TLSFs plus the tail KV TLSF, checking greedy packing across
  TLSFs and extents, and fit == carve.
- **H7 source gates** (python; the kv-layer-sizing family plus a new
  `test-sycl-kv-region-source.py`, each check with a mutation witness):
  - (a) the region reservation never reaches `unified_cache_malloc_device_tracked`;
  - (b) the tiered device-planned branch issues no per-layer `unified_alloc`;
  - (c) the optional pass runs after all S1 staging, dense *and* expert/DPAS;
  - (d) **one source**: every row of the §2.2 table reads `kv_region_fit` /
    `context_side_place`, and the deleted arena-device uses of `kv_admission_mismatch`,
    `kv_vram_cap` and `kv_device_budget` stay deleted;
  - (e) `[KV-PLAN-BUG]` is logged at ERROR at both sites, and STRICT aborts;
  - (f) no raw pointer is stored as region state;
  - (g) `reserve_kv_region` does not hold the group mutex across the yield call (C2);
  - (h) no dispatch path re-stages a yielded optional copy. Dispatch reads the primary's
    materialized layout (audit m4);
  - (i) the slot view's pointer comes from `slice().resolve()`, and `set_owner` has the
    `mem_handle` overload (audit m1);
  - (j) the new entry points appear in contract §3's allowlist, and their allocation class
    is derived, not hand-set (audit m2).
  RED: every check fires on the pre-change tree, and the count is recorded.
- **H8 region scope under concurrency (r1 I6, third point).**
  - The registry and scope logic is factored into a SYCL-free header, `kv-region-registry.hpp`.
  - Four threads each open a scope for a distinct ContextId on the *same* model and device,
    and attach two buffers each (iSWA shape).
  - A test hook, `g_test_block_next_region_attach`, modelled on the existing
    `g_test_block_next_kv_push` (master `ggml-sycl.cpp:13360-13405`), holds one thread
    mid-attach while the others proceed.
  - Assert no cross-wiring: each buffer's slots belong to its own ContextId's region.
  - The existing `g_test_block_next_kv_push` hook drives the KV-mask handoff, which this
    design no longer uses to find regions. The new hook plays the same role for the scope.
  - Positive control: a variant that shares one global (non-thread-local) scope must fail.

### 3.2 GPU test binary (the lead runs it, pinned selector, once)

**G1 `test-sycl-kv-region`** (label `cache`, `GGML_SYCL_PRIVATE_TESTING`). It:
- reserves a region on a real device cache and claims its slots under a scope;
- asserts every pointer is inside the arena (`arena_owns`) and the handle refcounts are as
  expected;
- **forces a mismatch** with `unified_cache_test_fail_next_kv_region_carve()`, then asserts
  that `[KV-PLAN-BUG]` reached the log callback, that the reservation returned failure, and
  that the external-bytes counter did not grow (a test accessor, not a stderr grep);
- checks STRICT in a subprocess: exit 134, and the message is present;
- **retained-run registration (audit I3):** frees an interior region, then asserts that
  `unified_lookup` of a pointer into the retained run resolves to no owner, that
  reusing the run registers a fresh owner, and that the arena rebuild/settle precondition
  does not count the retained bytes as live;
- **per-extent owners (audit I2):** forces a two-extent region (a retained run plus the
  frontier), claims slots across both, and asserts that each slot's pointer lies inside its
  own extent and that each extent is released only when its last slice drops;
- **VMEM precedence (audit I4):** with `GGML_SYCL_VMEM_KV=1` under the arena, the region is
  used, no vmem pages are mapped (vmem pool empty), and the ignored-flag WARN reaches the
  log callback exactly once.

Command:
`ONEAPI_DEVICE_SELECTOR=level_zero:1 ctest --test-dir build -R '^test-sycl-kv-region$' --output-on-failure`.
Sample `Shmem`/`MemAvailable` before the run and about 5 s after.

### 3.3 Lead-run GPU acceptance (serial, pinned selectors)

Pre-check: `grep -E '^GGML_SYCL:' build/CMakeCache.txt` and
`ldd build/bin/llama-completion | grep -cE 'libggml-sycl|libsycl'`.

- **C0 arena mode per card** (once, informational): `llama-bench ... -v | grep 'VRAM-ARENA\] (Reserved|Reserving)'`
  for `level_zero:0` and `level_zero:1`.
- **C1 the A2 rerun, the primary acceptance for P1 and P4.** Metrics restated per r1 I6. The
  `cohort=kv-tier-layer` count would be vacuous after L6, because that cohort is no longer
  emitted.
  ```
  GGML_SYCL_EXT_ALLOC_TRACE=1 GGML_SYCL_VRAM_BUDGET_PCT=60 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
    ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf -c 32768 \
    -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > c1.out 2> c1.err
  cat c1.err | grep 'EXT-ALLOC' | grep -c 'role=2 '       # KV role: must be 0
  cat c1.err | grep 'EXT-ALLOC' | tail -1                 # total_external: compare with base
  cat c1.err | grep -c 'KV-PLAN-BUG'                      # must be 0
  cat c1.err | grep 'KV overflow re-placed'               # predicted: 9 layer(s) demoted
  ```
  - **Positive control.** The same command on the base (post-jehw/u1bb master) prints
    `role=2` EXT-ALLOC lines, or demotes all 32 layers under jehw. The fe6c run printed 23
    such lines. That shows the trace is armed.
  - **Non-KV EXT-ALLOC bytes** (total minus the KV role) must not exceed the base run's.
    That is the check for r1 C3(ii).
  - Predicted: 23 device layers in the region, 9 on the host, and output starting
    `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`.
- **C2 Mistral gate:** B50 and B70, the CLAUDE.md command.
- **C3 GPT-OSS B50 gate:** `llama-cli ... -c 4096`, scored `grep -cx '1, 2, 3, 4, 5'` = 1. It
  is also run once with u1bb's `-ub 1024` acceptance form, to cover the ring and the ladder.
- **C4 G1** (§3.2).
- **C5 multi-context:** `test-thread-safety` with `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`,
  ONE run, only if it is green on the base first. It is known-crashy (oze0), so compare
  against the base. H8 is the primary coverage, and C5 is corroboration.
- **C6 perf sanity:** B50 Mistral pp512/tg128 and GPT-OSS pp512, ABBA ×3 against the base.
  - The expected delta is 0.
  - The places to look if it is not zero:
    - the TRANSIENT reclassification: scratch addresses move from the weight side to the
      context side, in the same chunk;
    - fattn sidecar `forbid_vram_zone_spill` falling back to the non-packed kernel when the
      context side is full.

## 4. Decomposition, effort, landing order

| id | work | files | effort | depends on | lands |
|----|------|-------|--------|------------|-------|
| L1 | TLSF placement primitives, tags, frontier walk; H1 | `tlsf-allocator.hpp`, `tests/test-tlsf-allocator.cpp`, CMake | high | none | **done:** `eab1ebeb6`, `9e0a708dc` (in review) |
| L3 | pure `kv_region_fit` (multi-extent, self extents, strict prefix, carve mirroring, transient reserve), `context_side_place`, `transient_reserve`, `kv-region-registry.hpp` (registry and scope logic); H2, H3, H6, H8 | `kv-runtime-demotion.{hpp,cpp}`, `kv-region-registry.hpp`, their tests | xhigh | L1, jehw merged | after jehw |
| L4 | `context_side` (classes by role, retained runs, atomic reuse, settle reset), `allocate_excluding` and a whole-TLSF block census (L1 follow-ups), retained-run registration hygiene and the `live_bytes` readers, per-extent owner-first carve, `kv_region_mutex_` added to contract §12.5 (L3), locked geometry snapshot, `reserve_kv_region` with the §2.4.2 lock discipline, strict-prefix `yield_optional_prefix`, removal of the dead `KV_AUTO` reclaim and of the `arena_reserve` KV reclaim, N-chunk; H4, H4b, H5 | `unified-cache.{hpp,cpp}`, `tlsf-allocator.hpp` (one primitive) | xhigh | L1, L3, jehw merged | after jehw |
| L5 | the optional pass after all S1 staging (dense + expert/DPAS); `zone_alloc_optional` | `ggml-sycl.cpp` S1 block | medium | L4 | with L4/L6 |
| L6 | transaction step (idempotent registry, plan/yield/commit loop, multi-device rollback, VMEM skip), llama-side scope procs + RAII guard, tiered claim + `set_owner(mem_handle)` slice views + per-extent clear with event-held slices, VMEM_KV precedence, second sources deleted (§2.2), both ERROR sites + `GGML_SYCL_STRICT_KV_PLAN`, TRANSIENT routing incl. fattn sidecars, ring admission via `context_side_place`; H7, G1. **Absorbs revision 1's L2.** | `ggml-sycl.cpp`, `unified-cache.cpp`, `fattn.cpp`, `src/llama-context.cpp`, tests | xhigh | L3, L4, L5, **u1bb merged** | last |
| L7 | docs: memory-design section, contract §3/§5.2/§5.3, arena comment, limits (§2.9), lock order | docs, `unified-cache.hpp` comment | medium | L6 | with L6 |

- L3 through L7 form one series in one worktree, so there is one first build and the
  follow-ups reuse it.
- Every step is RED → GREEN with its host test.
- G1 and C0-C6 are the lead's runs at the end of L6.
- **Seams to coordinate before L3/L4:**
  - **jehw:** retire `select_optional_layout_yield`'s pruning in favour of the strict prefix,
    retire `kv_zone_snapshot`/`fit_capacity` from production (they stay as a test oracle),
    and reuse `kv_layer_alloc_bytes`.
  - **u1bb:** the ring admission reads `context_side_place` instead of `zone_largest_free(KV)`,
    and its KV-zone slots become CONTEXT-class.

## 5. Decisions and open questions

- **(a) Refuse by default, and abort under `GGML_SYCL_STRICT_KV_PLAN=1`:** adopted, at
  region-backed sites only (§2.8).
- **(b) Yield order:** a strict address-ordered prefix, highest first (§2.4.1). This is a
  smaller change to jehw HEAD than revision 1 framed (r1 M7). The two-pass staging lands in
  the same series (L5).
- **(c) Weight holes:** **revised, pending the lead's confirmation.** With multi-extent
  regions, weight holes and buried optional tenants become region extents, and only
  sub-slot holes remain a limit, with a WARN (§2.9). This is the fix for audit I5.
- **(d) Handoff:** **revised.** A dedicated ContextId-keyed registry plus a thread-local
  region scope, which is a llama-side change (§2.5). The KV-mask handoff is kept and
  verified against the slot table.
- **Open (owner):** the transient reserve's fixed scratch floor (§2.4.3) is an estimate until
  zhcn/23mk plan those tenants. Is an estimate acceptable in the interim, as u1bb's
  1 MiB/row is?

## 6. Review dispositions

### 6.1 Design review r1

| id | finding | disposition |
|----|---------|-------------|
| C1 | a split reserves twice per device; a same-shape republish re-reserves | **Changed.** The registry is keyed `(ContextId, device)`, reservation is idempotent on the same shape key, and every backend's run finds the entry (§2.4.2 step 1). The same-shape republish keeps the region, matching `admitted_kv`. |
| C2 | yield under the group lock deadlocks (self-lock and ABBA); `freed < retained` misreported as PLAN_BUG | **Changed.** Plan with no lock, yield with no lock, then commit under the group mutex with a re-fit. The lock order is written down, and a shortfall re-plans and demotes (§2.4.2 steps 2-5; H7g). |
| C3 | many non-weight requests target WEIGHT: burying, raw misses, churn, MoE ordering | **Changed.** The lifetime class comes from the role (§2.1): TRANSIENT goes to the context side, budgeted by `transient_reserve` (§2.4.3). The optional pass runs after all S1 staging. Residual burying by real weights is exact for the fit, reduced by `allocate_excluding`, and WARNed. The generation counter is removed. |
| I1 | live second sources of "fits" | **Changed.** §2.2 enumerates every site with its disposition, and H7d enforces it. |
| I2 | the KV-mask handoff is the wrong carrier | **Changed** (decision (d) revised). ContextId registry plus thread-local scope with attach semantics, owner-device lookup, and mask == slot-table check (§2.5). |
| I3 | L2 before L6 turns spills into refusals; the role==KV guard false-labels fattn | **Changed.** L2 is folded into L6. The ERROR sites are region-backed only, and the fattn sidecars are TRANSIENT with a silent `forbid_vram_zone_spill` (§2.8). |
| I4 | one extent per TLSF wastes retained runs; first-fit vs best-fit | **Changed.** Multi-extent regions within a TLSF (§2.4.1). Placement is one policy (`context_side_place`, or the fit's extents), and allocation never searches independently (§2.3.2). |
| I5 | multi-device commit not atomic; generation too coarse | **Changed.** Registry rollback of this call's entries, self extents on retry, and a documented non-undoable yield. Staleness is decided by re-fit comparison (§2.4.2). |
| I6 | H2 RED does not reproduce post-jehw; C1 grep vacuous; no concurrent-creation host test | **Changed.** Capacity RED (H2), C1 on KV-role EXT-ALLOC plus total bytes against the base, and the H8 thread test with a blocking hook and a negative control. |
| I7 | out-of-order unload understated | **Changed, and further by audit I5.** Weight holes and buried optional tenants are now region extents, and only sub-slot holes remain a limit, with a WARN (§2.9). |
| M1 | alignment wording; whole-gap take moves the base | **Changed.** The wording is fixed (256 B absolute, 512 B slot sizes), and the fit mirrors L1's whole-gap rule (§2.4.1). |
| M2 | zone_settle vs retained runs | **Changed.** The context side is reset in the same critical section (§2.3.4). |
| M3 | retained-run reuse must stay atomic | **Changed.** free/carve/re-carve in one group-lock section, spelled out (§2.3.2), with H4 pinning it. |
| M4 | dead `KV_AUTO` `zone_reclaim(KV)` caller | **Changed.** Verified dead (only STATIC is constructed); L4 removes it (§2.3.4). |
| M5 | VMEM path double-charges | **Changed, then superseded by audit I4.** The region takes precedence under an arena, and `GGML_SYCL_VMEM_KV` is ignored there with a WARN. Revision 2's "skip the region" would have kept planned KV outside accounting (§2.6). |
| M6 | assert `TAG_OPTIONAL != 0` | **Changed.** A `static_assert` at the tag definitions (§3.1 H1). |
| M7 | jehw already selects by address with pruning | **Accepted.** (b) restated as strict prefix vs pruned subset (§2.4.1, §5). |
| M8 | H4 scenario inconsistent | **Changed.** Rewritten with a consistent order (§3.1 H4). |
| M9 | ring slots are not lowest once overflow sits below | **Changed.** The claim is withdrawn; the actual behaviour is bounded and tested (§2.7, H4b). |

### 6.2 Principles audit (audit-mem-b, moua section)

| id | finding | disposition |
|----|---------|-------------|
| I1 | the yield under the group lock is an L4-under-L5 inversion, a wait under a lock, and a self-deadlock through zone_free | **Already changed in revision 2** (r1 C2): plan with no lock, yield with no lock, commit under L5 with a re-fit comparison. A STALE result retries by re-fitting, not by a generation counter (§2.4.2). The lock ranks are now named against contract §12.5, and the new `kv_region_mutex_` is classified L3 (§2.3.1). |
| I2 | an N-chunk region needs a control per extent | **Changed.** One owner-first `CACHE_SUBALLOCATION` per extent. A slot is `slice()` of its own extent's handle, and buffers and views hold every extent handle they touch. This now applies to all multi-extent regions, not only N-chunk (§2.4.1, §2.6). |
| I3 | retained holes need registration hygiene | **Changed.** Retain unregisters the exact record first, reuse is owner-first with a fresh registration, and rebuild, destroy and settle count retained runs as free through `live_bytes` (§2.3.2). Checked by H4 and G1. |
| I4 | VMEM_KV double-charges and keeps planned KV outside accounting | **Changed.** Under an arena the region takes precedence and `GGML_SYCL_VMEM_KV` is ignored with one WARN. Without an arena, vmem-kv is unchanged until 1oxa deletes it (§2.6). |
| I5 | buried optional tenants make KV demote while the copies hold VRAM | **Changed. A fix is proposed; no owner ruling needed, but the lead should confirm, since it revises decision (c).** Buried lease-free optional tenants yield lazily and their runs become region extents, as do weight holes (§2.9). The eager alternative ("yield model 1's copies before model 2 stages") was rejected, because it releases them even when no context needs the room. |
| m1 | `set_owner` needs a `mem_handle` overload | **Changed.** The overload stores the slice, and `ptr` comes from `slice.resolve()` (§2.6). |
| m2 | allowlist the new entry points and derive their class | **Changed** (§2.10, H7j). |
| m3 | slot slices must outlive the clear's fill event | **Changed.** The fill's event lease holds the slices (§2.6, §2.10). |
| m4 | keep the barrier and epoch; never lazily re-stage a yielded copy | **Changed** (§2.4.2 step 3, H7h). The inherited P2 site (WOQ copy readers take no lease; recorded graphs bake pointers) is named and not widened. |
| pre-existing | the raw fallback, backend-buffer-kv-zone, onednn_weights_scratch and STAGING EXT-ALLOCs are named but not tracked | **Tracked.** llama.cpp-23mk already covers backend-buffer-kv-zone, onednn_weights_scratch and STAGING (plus c-m2jh). The fail-open raw fallback mechanism is new ticket **llama.cpp-gxur**, cross-referenced on 23mk (§2.11). |
