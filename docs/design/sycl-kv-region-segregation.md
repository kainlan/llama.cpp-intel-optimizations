# llama.cpp-moua: planned, lifetime-segregated layout for the shared KV+WEIGHT zone

Design, revision 4. Author: impl-moua, 2026-09-26. The revisions answer three reviews:
- design review r1 (design-moua-r1: 3 Critical, 7 Important, 9 Minor), recorded in §6.1;
- the principles audit's moua section (audit-mem-b: 5 Important, 4 Minor), recorded in §6.2;
- design review r2 (design-moua-r2: 1 Critical, 11 Important, 10 Minor), recorded in §6.3;
- design-moua-r2's addendum on `1972b32b0` (2 Important, 5 conditions, 2 Minor), recorded in
  §6.4, together with the owner's ruling on the transient reserve.

r1 and the audit read revision 1. r2 read revision 2 at `19f7f08bf`, before the audit fold
(`1972b32b0`) reached it; §6.3 re-checks each audit row it names against the folded text.

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
  - Inside the transaction (which already holds L1), the order is:
    1. release the u1bb ring;
    2. plan;
    3. yield;
    4. run the byte-budget steps;
    5. commit-carve;
    6. re-admit the ring.

    A guard declared before L1 rolls back on any refusal, and the registry entry is published
    only after the plan's CAS (§2.4.2).
- **Context-side tenants are planned exactly (owner ruling 2026-09-26).** The room the fit
  leaves for CONTEXT/TRANSIENT tenants is the sum of their planned demands, which
  llama.cpp-zhcn and llama.cpp-23mk produce. There is no estimate and no floor (§2.4.3).
- **Error path:** a device-planned KV layer that does not land in its region is a planner
  bug. It is refused with `[KV-PLAN-BUG]` at ERROR, and aborts under
  `GGML_SYCL_STRICT_KV_PLAN=1`. It is never admitted and then spilled.
- **Holes:** weight-side holes and buried optional tenants are usable region extents
  (regions are multi-extent). Only holes smaller than one slot are a limit (§2.9).
- **KV shape (r2 N-C1):** slot sizes come from the shape llama will actually allocate
  (`type_k`, `type_v`, `v_trans`, and the per-context KV layer set), which llama sends through
  a new publish entry point. One function computes a layer's bytes for the fit, the
  reservation and the tiered claim (§2.4.4).
- **Landing:** L1 is done and approved (`eab1ebeb6`, `9e0a708dc`, review fixes `97315421b`
  and `456650c01`). L2 is folded into L6, so no interim guard turns today's spills into
  refusals. L3-L7 need jehw and u1bb merged.

## 0.1 Acceptance conditions: the four principles

Every step of this design, and its reviews, are held to these. Each row names the mechanism
that enforces the principle and the check that shows it.

| | principle | enforced by | shown by |
|---|---|---|---|
| **P1** | The unified cache is the only allocator. There is no out-of-arena fallback for planned KV. | KV regions are carved from the arena TLSF (§2.4). The tiered device branch issues no `unified_alloc` (§2.6). A region reservation never falls through to `unified_cache_malloc_device_tracked` (§2.8). | H7a/b; C1 counts **zero** KV-role EXT-ALLOC lines, and total EXT-ALLOC bytes no higher than the base run (§3.3). |
| **P2** | `mem_handle` / `alloc_owner` are the only ownership surfaces. | Each region **extent** is an owner-first `CACHE_SUBALLOCATION` with its own `mem_handle`. The registry, each KV buffer, and each layer's view hold the extent handles they touch. A slot is `extent_handle.slice()`, not an allocation. Retained holes have no owner and are never handed out (§2.3). Final handle drops never happen under a listed lock (§2.4.2). | G1 checks `arena_owns` and the handle refcounts. H7f checks that no raw pointer is stored as region state. |
| **P3** | Placement decides the executor. | Residency is still decided per layer by the planner. Host-tier layers keep executing on the CPU exactly as today, and only the capacity input changes (§2.5). | C2/C3 gates; the demotion WARN names the layers. |
| **P4** | Plan == reality. A KV region that does not fit is a planner bug: refused, never admitted and then spilled. | One function, `kv_region_fit`, decides "fits" for the planner, the reservation (re-run under the lock), the ring admission and the `-c` hint. One function, `kv_layer_tensor_bytes`, sizes a layer for the fit and for the claim, from the shape llama allocates (§2.4.4). Llama's residency answers come from the registry (§2.5). Every second source of "fits" is deleted or rederived (§2.2), and the capacity primitives are allowlisted (H7d). The carve commits exactly the extents the fit named. | H2 property test (fit ⇔ carve), H2/H3 shape cases (q8_0, `v_trans`, MTP), H7d (one source), G1 forced mismatch ⇒ `[KV-PLAN-BUG]`. |

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

A request's **lifetime class** is carried explicitly (r2 m3). L4 adds a
`shared_zone_lifetime lifetime` field to `alloc_constraints`, with UNSET as its default, and threads it
into `zone_alloc`. `zone_alloc` has no role parameter today (master `unified-cache.cpp:22479`).
- When the field is UNSET, the class is derived from `alloc_role`: WEIGHT gives WEIGHT, and
  every other role gives TRANSIENT.
- The call sites that know better set it: the optional pass sets OPTIONAL; the
  KV region carve sets KV_REGION; the u1bb ring and `backend-buffer-kv-zone` set CONTEXT.
  CONTEXT and TRANSIENT are both `alloc_role::COMPUTE`, so the role alone cannot separate them.
- The zone a request names does not decide its class. Many non-weight requests name
  `vram_zone_id::WEIGHT` only to stay out of the B50 tail zones (r1 C3). Examples: master
  `ggml-sycl.cpp:46120`, the per-op MMVQ scratch (*"Keep tiny MMVQ Q8 activation scratch out
  of arena tail zones on B50"*), and `ggml-sycl.cpp:97965`, the SOA-graph Q8_1.

| class | who | side | placement |
|---|---|---|---|
| `WEIGHT` | `alloc_role::WEIGHT`. That includes the runtime expert-cache fills, which are `alloc_role::WEIGHT` with `runtime_category::EXPERT_CACHE` (vram-pool.cpp:81, unified-cache.cpp:18484): on-demand expert rows, which are real weights. It also includes a `SYCL<n>` weight buffer that overflows RUNTIME into the shared zone (r2 m2, below). | weight | TLSF `allocate` (today's behaviour), tag `TAG_WEIGHT`; after the optional ladder exists, first a weight-side hole, then the gap front (§2.3.3) |
| `OPTIONAL` | optional layout copies (jehw's `optional_layout`) | weight frontier | `allocate_gap_front`, tag `TAG_OPTIONAL` |
| `KV_REGION` | the per-(ContextId, device) KV region | context | the extents chosen by `kv_region_fit` (§2.4) |
| `CONTEXT` | u1bb ring KV-zone slots; the `backend-buffer-kv-zone` overflow of a **compute** buffer | context | `context_side_place` (§2.3.2) |
| `TRANSIENT` | every **non-WEIGHT role** that names `WEIGHT` or `KV`: COMPUTE/STAGING scratch (ggml-sycl.cpp:46120, :97965; mmvq.cpp:16863; unified-cache.cpp:21455; common.hpp:6658), persistent buffers (unified-kernel.cpp:4890), scratch_pool (unified-cache.cpp:20629), and the fattn packed-K/sidecar KV-role requests (fattn.cpp:569, :1640) | context | `context_side_place`, bounded by the planned transient reserve (§2.4.3) |

**`backend-buffer-kv-zone` keeps its buffer's role (r2 m2).** Standard `SYCL<n>` buffers carry
both model weights and scheduler compute buffers. `alloc_role` is WEIGHT for a weight buffer
(master `ggml-sycl.cpp:37164-37166`, `should_use_runtime` at `:37184`). The KV-zone fallback
then hard-labels every request `role=COMPUTE` (`:37231-37236`). Under class-from-role, that
would put a model's weights on the context side for the model's lifetime. L4 makes the
fallback pass the buffer's own `alloc_role`: a weight buffer is WEIGHT-class, and a compute
buffer is CONTEXT-class.

**The B50 tail-zone quirk (r2 N-I8, corrected).** Revision 2 said TRANSIENT "keeps the reason
those callers chose `WEIGHT` (the chunk is not a tail zone)". That was false. In single-chunk
mode the ONEDNN, RUNTIME and SCRATCH tail zones are **in the same chunk**, at high offsets
(master `unified-cache.cpp:22124-22160`). The quirk is positional, and it is uncharacterised.
The workaround's own comment grounds it in the low end: *"Weight-zone pointers are already
exercised by S1-preloaded weights"* (master `common.hpp:6655-6657`).
- This design moves TRANSIENT, and the KV regions, to the **top** of the shared zone,
  directly below ONEDNN. Those pages are not the S1-exercised ones. Today a small model at
  the default PCT never touches them.
- It is therefore an acceptance item, not an argument: **C2a, B50 MMVQ/STAGING first submit**
  (§3.3). It is the Mistral B50 gate on `level_zero:1` at the default PCT, where the region
  and the per-op scratch sit at the top of a mostly empty shared zone.
- **The fallback lever**, if C2a fails. The two call sites that carry the B50 comment
  (master `ggml-sycl.cpp:46120` MMVQ Q8 scratch, and `common.hpp:6658` STAGING) set
  `lifetime = WEIGHT_SIDE_TRANSIENT`. That places them first-fit from offset 0, as today, into
  the pages the S1 weights exercise.
  - Its cost is today's C3(i): they can bury the ladder.
  - The class exists only for those two sites and is gated by H7 (no third site), so the rest
    of TRANSIENT stays on the context side.
  - L6 lands the lever dark, with an env override `GGML_SYCL_B50_SCRATCH_WEIGHT_SIDE=1` to test
    it. It becomes the default for those two sites only if C2a fails.

Moving TRANSIENT off the weight side fixes three of r1's C3 consequences:
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
- **On GPT-OSS this is the main path, not a corner (r2 m1).** Runtime expert-cache fills are
  WEIGHT-class and arrive after the optional pass. They take weight-side holes first
  (§2.3.3), and bury the ladder only when none fits. H5 covers it with an EXPERT_CACHE fill.
- **This is not a P4 hazard.** The fit reads the actual block list, so what it reports is
  what lands.
- **It costs little capacity.** Buried lease-free optional tenants still yield when a
  context needs the room, as hole extents (§2.9, audit I5). What burying costs is
  contiguity: the room comes back as a separate extent, not as frontier growth.
- **Only a leased buried tenant is lost to KV.** The fit reports that, and the §2.9 WARN
  covers it.

### 2.2 One fact: every source of "fits"

Every site that decides whether device KV fits, or which layers are device-resident, is
listed here with what it becomes (r1 I1; r2 N-I4 added the rows marked r2).

A list of rows fails open to any site nobody listed, so H7d does not check rows. It
**forbids the capacity primitives themselves** outside an allowlist:
- the primitives are `unified_cache_kv_vram_available`, `zone_available(KV|WEIGHT)`,
  `zone_largest_free`, `largest_free_block`, `ggml_sycl_kv_capacity_live`,
  `kv_zone_snapshot`/`fit_capacity`, and the new `live_bytes()` (§2.3.2);
- the allowlist holds their definitions, the no-arena branches (each marked with a comment
  H7d matches), log-only readers that feed no decision, and test code
  (`kv_zone_snapshot`/`fit_capacity` survive only as a test oracle, §4 seams).

Every other occurrence in `ggml-sycl.cpp`, `unified-cache.cpp` and `fattn.cpp` fails the gate.
Each allowlist entry and each forbidden primitive has a mutation witness.

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
| (r2) MMID budget demotion: `ggml_sycl_try_demote_runtime_kv` on `BUDGET_EXCEEDED` / `GROWTH_BUDGET_EXCEEDED`, master `:17919-17960`, and its `-c` hint via `ggml_sycl_largest_fitting_n_ctx_live` | byte budget, after the re-fit, so it can demote layers the region already holds slots for | it keeps its constraint, which is a different fact (the device's total VRAM budget, including RUNTIME demand), but it runs **before the carve**. The commit re-fit starts from its residency and can only demote further (§2.4.2 step 5). Its `-c` hint reads `kv_region_fit`. |
| (r2) `rebuild_runtime_per_device_vram` (`:17868`) and `moe_mmid_reaccount_replacement` (`:17881`) | byte accounting refusals | arena devices: their KV term is the fit's region bytes (Σ slot sizes), the same number the carve takes. Their refusals roll back through the transaction guard (§2.4.2). |
| (r2) u1bb ring admission: `kv_capacity_bytes = ggml_sycl_kv_capacity_live(...)` and `kv_bytes = ggml_sycl_device_kv_bytes_with_slack(...)` (u1bb `:18334-18335`; the function at `:16958-16970`); u1bb's re-fit capacity (`:18009`) | live bytes, counting the live ring's KV-zone bytes as free | the ring is released before the fit (§2.4.2 step 2) and re-admitted by `context_side_place` on the post-region geometry. Both u1bb reads are deleted for arena devices, and the re-fit reads `kv_region_fit`. |
| (r2) `ggml_backend_sycl_kv_layer_on_device_from_dev` (master `:107332-107343`), llama's residency hook | the process-global plan snapshot | under an open region scope, the registry entry for this ContextId (§2.5, r2 N-I5). Outside a scope, as today. |
| (r2) `GGML_SYCL_BLOCK_EXEC_CANDIDATE_KV` (master `:38623-38649`, opt-in) | reassigns `kv_device` at buffer-alloc time | arena devices: ignored with one WARN per process, like `GGML_SYCL_VMEM_KV` (§2.6). If it is ever wanted there, the reassignment moves into the fit's input. |
| (r2 N-C1) the tiered slice's per-layer size, `kv_slice` (master `:38593-38612`), and `update_runtime_kv_sizes` at alloc time (`:38625`) | the actual buffer size divided evenly across its layers; the plan re-sized from the geometry | arena devices: the slot table, sized by `kv_layer_tensor_bytes` from the published shape (§2.4.4). The claim checks the buffer size against the table's sum. |

### 2.3 The context side (unified-cache, L4)

#### 2.3.1 Structure and locking

Each shared TLSF has one `context_side`: an address-ordered list of blocks
`{offset, size, state LIVE|RETAINED, class, owner}`. Its **anchor** is the lowest
LIVE-or-RETAINED block.

Every read and write of `context_side` and of the TLSF runs under
`unified_cache::arena_allocator_group_mutex(zone)`. That includes the geometry snapshot
(r1 044k; L1 documents the contract at `9e0a708dc`). The snapshot copies out a POD
`shared_zone_geometry` and releases the lock. The pure fit then runs on the copy.

**Locks, in canonical contract §12.5's ranks (r2 N-I1).**

What already runs under L1:
- The runtime-context transaction takes `g_tensor_inventory_mutex` (L1) and holds it for
  its whole body (jehw `ggml-sycl.cpp:17687` to the function's end at `:18401`; master
  `:17673` to `:18306`).
- On jehw that body includes the optional-layout yield. It includes the yield's reader-barrier
  wait (jehw `unified-cache.cpp:7835`, `readers_done.wait_and_throw`) and its final drop of the
  deferred-free handles (`:7860`, `released.clear()`).
- Both are things §12.5 forbids under **any** listed lock. They are pre-existing on jehw, and
  moua does not widen them: it adds no wait and no final-handle drop under L1, or under its
  own lock.
- The consequence for moua: region **reservations are already globally serialised by L1**,
  because every reservation happens inside the transaction. `kv_region_mutex_` exists only
  because two paths touch the registry without L1: the claim (inside `create_memory`) and
  teardown (the exec-context close, §2.4.2).

**`kv_region_mutex_`** (new, one per device):
- **Rank L3**, beside `g_pending_kv_layer_masks_mutex`, with ascending device ID as the
  same-rank tie-break. So L1 → L3 is legal, and it sits below L4 `direct_stage_mutex_`.
  L4 adds it to the §12.5 table in the commit that introduces it.
- **Never co-held with `g_pending_kv_layer_masks_mutex` (addendum).** Both are L3, and §12.5
  forbids co-holding a global/transitional lock with another lock of the same rank. The
  tiered claim uses both: it pops the KV mask under `g_pending_kv_layer_masks_mutex`,
  releases it, and only then reads the registry under `kv_region_mutex_`. The two are
  **sequential, never nested**, and H7k checks that neither scope contains the other.
- **Strictly leaf**, in the same sense as `moe_discovery_registry::mutex_`: it is never held
  while any other lock is acquired, and never across a wait, an allocation, a callback or
  the destruction of a final `mem_handle`.
- Every registry access is a short copy-in or copy-out:
  - lookups copy handles out (a refcount increment, never a final drop);
  - inserts move handles in;
  - erases **move the entry out, unlock, and only then drop it**: on rollback, on a shape
    change, and on teardown.
- It is never held across the yield, the carve or the ring re-admission (§2.4.2).

**The group mutex** `arena_allocator_group_mutex(zone)` is L5. It is held only for the
geometry snapshot copy-out and for the carve, and never across the yield or a wait.
- The existing order `cache locks -> group lock` (staging calls `zone_alloc` under the cache
  locks) is preserved.
- No code holds the group mutex while calling the yield (r1 C2, §2.4.2).

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
  2. **Reuse is owner-first, through `zone_alloc`'s own protocol (r2 m5).** No new minting
     code:
     - The new block's `allocation_id` is minted **before** the group lock, as `zone_alloc`
       already does (master `unified-cache.cpp:22490-22494`: *"Mint before taking the
       physical allocator-group lock"*).
     - The carve and `arena_register_exact` then run under that lock, with the existing
       failure path that frees the block if registration fails (`:22543-22546`).
     - The context-side placements (the M3 sequence, the region carve) are new branches
       inside `zone_alloc`'s locked section. They are not a separate hand-minted path.
     - The re-retained remainder is registered to no one.
  3. **Rebuild, destroy and quiescence treat retained runs as not live, but capacity does
     not treat them as free (addendum NEW-1).** These are two separate numbers, and each
     has its own reader:
     - **`zone_available` is unchanged**: it is `tlsf_allocator::available()`, which counts
       retained runs as used. They sit off the free lists, and only `context_side_place` can
       use them. A plain `allocate` never can, so a byte reader that counted them as free
       would be reading bytes it cannot place, which is A2's "bytes, not extents" defect.
       Revision 2's text subtracted them here, and that is withdrawn.
     - **A new `live_bytes()`** = `used() − context_side.retained_bytes`. It is read only by
       the arena rebuild/destroy decision, the settle precondition, and the leak and
       quiescence checks. Those paths already decide "live" by registered allocations, which
       retained runs never are.
     - No fit decision reads either one. H7d forbids `zone_available` in any fit decision
       (§2.2), and `live_bytes` is on H7d's primitive list too, so it cannot become the next
       byte-based fit.
     - L4 enumerates the `live_bytes` readers, and H4 asserts both numbers at every step.

#### 2.3.3 Weight-side placement after the optional pass

`zone_alloc(WEIGHT)` works as follows once an optional ladder is live on that TLSF:
1. first-fit into a weight-side hole: a free block that is **not** the gap block;
2. only then the gap front, with the burying WARN of §2.1.

Both steps skip free blocks that intersect a pending reservation range (§2.4.2 step 4,
addendum (b)).

This needs one new L1-level primitive, `allocate_excluding(excluded ranges, size, align,
tag)`. It takes every free block that intersects an excluded range (the gap block, and any
pending ranges) out of the free lists, calls `allocate`, and reinserts them. It lands
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

**Which context side takes a CONTEXT or TRANSIENT request (r2 m7).** The zone the request
names picks the TLSF:
- `WEIGHT` goes to the **last** weight chunk's TLSF. Its frontier is the one the S1 staging
  reached last, so it is the least likely to hold a ladder a later weight would bury.
- `KV` goes to the tail chunk's KV TLSF.

`transient_reserve` (§2.4.3) is computed per TLSF from the same routing, and the fit keeps
each TLSF's share free on that TLSF. H6 pins the routing.

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
  whole-TLSF block census, a second L4 read primitive beside `frontier_walk`. Like
  `frontier_walk` (L1's 044k contract, `9e0a708dc`), it runs **only under the group mutex**,
  inside the snapshot copy-out (addendum (d));
- `self_extents`: the extents this ContextId carved earlier in the **same** transaction call,
  on a retry of the commit loop (§2.4.2 step 6). A same-shape republish never re-fits (step 1),
  and a shape change's old region is still occupied, so neither is a self extent.

The request `r`:
- the slot sizes, `kv_layer_alloc_bytes(kv_layer_tensor_bytes(...))` over the published KV
  shape (§2.4.4), for the layers the shape says hold KV. They are grouped full-attention
  first, then SWA, each group in layer order;
- `forced_host`: layers an earlier step of the same transaction already demoted (the MMID
  budget demotion, §2.4.2 step 5). The fit may demote further but never promotes them;
- the planned `transient_reserve` per TLSF (§2.4.3).

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

  **The pack is cost-ordered (addendum NEW-2).** Revision 2 packed "best-fit retained and
  hole extents first, then the frontier". Hole extents included runs formed by releasing
  buried optional tenants, so that order yielded buried copies even when the gap alone would
  have held the slots. That contradicted §2.9's "only when KV needs the room". It also put KV
  on the weight side while the gap was free.

  Slots are placed in slot-table order, each into the cheapest tier that still has room.
  Within a tier, best fit decides. A slot is never split. The tiers, cheapest first:
  1. **RETAINED runs** on the context side: no cost.
  2. **The frontier gap, with no yield**: no cost, and segregation is kept.
  3. **Free weight-side holes**: no layout is lost, but the context's KV pins a weight-side
     hole for its lifetime (§2.9 (c)). Hence tier 3, after the gap.
  4. **Yields, ranked by optional-layout bytes lost per slot gained.** The candidates are the
     next strict-prefix step at the frontier (the smallest `k` more ladder entries that gains a
     slot) and each buried-tenant run (only the tenants that intersect the smallest carved
     range). The cheapest candidate goes first, and a tie goes to the frontier, which keeps
     segregation. Repeat until the slots fit or no candidate is left. Then the demotion loop
     takes over.

  So a buried tenant is released only when tiers 1-3 cannot hold the slot, and only when it
  costs less layout per slot than growing the frontier prefix. This is the ordering that the
  owner-visible line on llama.cpp-moua states (§2.9 (e)).
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

The runtime-context transaction runs under L1 (§2.3.1). For a context `c` in publish mode,
the region work sits inside it as follows.

**The transaction guard (r2 N-I3).** L6 declares a `kv_region_txn` guard **before** the
transaction's L1 `lock_guard`. Reverse destruction order therefore runs the guard's
destructor after L1 is released, which is the same technique contract §12.5 records for the
zone-reset audit's site-visit guard. The guard owns everything this call has done and not
yet published:
- every extent handle this call carved;
- every pending registry edit: new entries, and old entries replaced by a shape change;
- the u1bb ring's pre-transaction `n_ubatch`, when step 2 released the ring.

On **any** return other than a committed publish, the destructor rolls all of it back. That
covers a `refuse()`, a `busy()`, or an exception, from any step. The pending edits are
discarded, the carved handles are dropped (outside L1 and outside `kv_region_mutex_`, so
their `zone_free` runs lock-clean), and the ring is re-admitted at its old `n_ubatch` with
u1bb's existing rollback call.
- Yields already performed are **not** undone. They released optional tenants, which costs
  prompt-processing performance, never correctness, and the yield WARN names them.

**Steps.**

1. **Idempotence (r1 C1; r2 N-I9, m10).**
   - For each device `d`, look up `(c, d)`: copy it out under `kv_region_mutex_`, then unlock.
     If an entry exists with the **same shape key**, reuse it: no fit, no side effect. If every
     device matches, steps 2-6 are skipped, and the transaction continues at step 7 exactly as
     today.
   - **The key is built from request inputs only, never from fit outputs.** It is
     `(n_ctx, n_seq_max, kv_unified, swa_full, the published KV shape's digest)`. The digest
     covers `type_k`, `type_v`, `v_trans`, `no_alloc` and the per-layer descriptors (§2.4.4).
     It holds no slot table and no residency.
   - **`n_ubatch` is excluded.** SWA slot bytes depend on it (the SWA branch of
     `kv_layer_bytes_for_kind`, master `unified-cache.hpp:580`), but llama allocates the KV
     once, at `create_memory` (master `llama-context.cpp:869`), with the `n_ubatch` of the
     constructor's first publish (`:810`). The auto micro-batch ladder republishes with other
     values after the KV exists (`:1039` → `:1571`, `:1936`). A region's slot sizes are
     therefore **frozen at the reservation's `n_ubatch`**, which is the value llama used. A
     ladder republish matches the key and never reserves a second region. That is the B50
     GPT-OSS `-ub 1024` case in C3.
   - This is today's `admitted_kv` rule, *"a same-shape republish by an admitted context keeps
     the published residency"* (master `ggml-sycl.cpp:17738-17747`, jehw `:17752-17760`), made
     physical.
   - It makes the split case safe. Every backend's run of the transaction re-fits every device
     (*"each re-fits, because admission is per backend"*, the comment block that follows). The
     first run to reach `(c, d)` reserves and publishes the entry. L1 serialises the runs, and
     every later one finds it.
2. **Release the ring first (r2 N-I6).**
   - When a reservation is needed on `d` and the u1bb PP MoE oneDNN ring holds KV-zone slots
     there, release those slots **before** the snapshot. L6 splits u1bb's
     `ggml_sycl_replan_pp_moe_onednn_ring` into its release half and its admit half.
   - `RELEASE_REFUSED` (a slot claimed by an in-flight dispatch) returns `busy` here, before
     anything is carved.
   - This is u1bb's own order, *"KV wins over them, and that transaction re-admits the ring
     after KV"* (u1bb `ggml-sycl.cpp:16950-16953`), made physical. The ring's blocks are then
     simply free in the geometry, and nothing counts them twice.
   - On a same-shape republish (step 1 matched), the ring is not released here. u1bb's own
     tail re-plan handles an `n_ubatch` change, exactly as today.
3. **Plan.** Snapshot the geometry (a copy-out under the group mutex, then release), and run
   `kv_region_fit`. That yields the residency, the yield prefix, and the extents.
4. **Yield.** Neither the group mutex nor `kv_region_mutex_` is held; L1 is held, as it
   already is on jehw.
   - Call `yield_optional_prefix(d, k)`. This is jehw's `yield_optional_layouts` (jehw
     `unified-cache.cpp:7737-7880`) with a strict-prefix selection, plus the buried tenants the
     fit chose (§2.9).
   - It keeps jehw's reader barrier gate unchanged.
   - **Released ranges are held for this reservation (addendum (b)).** Before the yield, the
     transaction records the **extents the fit chose** as `pending_ranges(c, d)` on the TLSF,
     under the group mutex. Those are the exact ranges step 6 will carve: the top of the gap
     below the anchor, grown by the prefix, and the carved part of each chosen buried-tenant
     run. Freed bytes outside them, such as the gap front, stay available to weights as today.
     - Until step 6 carves them, or the guard rolls back, weight-side placement skips any
       free block that intersects a pending range. That is `zone_alloc(WEIGHT)` both in a
       hole and at the gap front (§2.3.3).
     - Without this, a concurrent WEIGHT allocation (an expert-cache fill, or another model's
       staging, neither of which takes L1) would take the freed block first. The commit
       re-fit would then differ, the yielded layout would be lost for nothing, and after
       three iterations the context would be refused.
     - A WEIGHT request that fits **only** inside a pending range gets the zone miss it would
       get a moment later, once the region is carved there. That is the post-commit state, so
       this adds no new behaviour class.
     - Pending ranges are cleared by the commit carve, and by the guard's rollback. The guard
       owns them, like everything else it owns, so no exit path leaves one behind.
   - **The optional-layout epoch bump stays where jehw has it: in the transaction, not in the
     yield** (audit m4). After a yield that retired anything, the transaction calls
     `ggml_sycl_optional_layouts_retired()` (jehw `ggml-sycl.cpp:17904`, which bumps
     `g_ggml_sycl_optional_layout_epoch`, `:17606-17609`). Graph compute compares against that
     epoch before replaying (`:105557-105565`).
   - **A yielded copy is never lazily re-staged.** Dispatch reads the primary's materialized
     layout (P3, "the loaded layout is the answer"), and H7h gates it.
   - Inherited, not widened:
     - jehw's oneDNN WOQ copy readers take no lease, and recorded exec graphs bake the
       copies' pointers (610516bd7's own message). That is a P2 migration site moua inherits.
       The barrier gate and the epoch bump are what make the release safe today, and moua
       keeps both.
     - The yield's wait and final drops under L1 (§2.3.1).
5. **The byte-budget steps, in their existing order (r2 N-I4).** These are
   `rebuild_runtime_per_device_vram`, `moe_mmid_reaccount_replacement`, and the MMID re-plan
   with its `ggml_sycl_try_demote_runtime_kv` fallback.
   - Under an arena, their KV term is the fit's region bytes.
   - A demotion here only removes device layers. It becomes the fit's `forced_host` input, so
     the carve reflects it, and the carve never holds slots for a layer this step demoted.
   - A refusal here rolls back through the guard. So far there is nothing to roll back
     except the ring.
6. **Commit the carve, under the group mutex (r2 N-I7).**
   - Re-snapshot, and re-run `kv_region_fit` on the live geometry with `forced_host`.
   - If it fits **with no further yield**, carve its extents owner-first (one control per
     extent, through `zone_alloc`'s mint-before-lock protocol, §2.3.2). The handles go into the
     guard. The extents may differ from step 3's, because live TRANSIENT churn below the anchor
     moves them, and that is fine.
     - If its residency differs from step 3's with no further yield needed (a shortfall),
       commit that residency too. Re-planning on the same geometry would compute the same
       answer, so looping would change nothing.
   - If it needs **more yield**, release the group mutex and go back to step 3.
   - The loop is bounded at 3 iterations. After that the transaction refuses with a named
     cause (`region geometry kept moving`), never `[KV-PLAN-BUG]`.
   - Across devices, carves run in device order. A later device's refusal rolls back the
     earlier ones through the guard. On a retry, the fit counts `c`'s own carves from this call
     through `self_extents`, so a retry never competes with itself.
7. **Re-admit the ring, and the rest of the transaction.**
   - If step 2 released the ring, its admit half runs through `context_side_place` on the
     post-region geometry. Otherwise u1bb's own tail re-plan runs as today, placing through
     the same function. Failure is u1bb's existing refusal (*"PP MoE oneDNN scratch ring does
     not fit"*), and the guard rolls the region back.
   - The remaining steps run as today, and any refusal among them rolls back the same way:
     the non-FA scratch check, the MMID workspace materialization, and the publication CAS
     (master `:18223-18258`).
8. **Publish.** Only after the publication CAS succeeds does the guard commit. Commit cannot
   fail, and it is the last step:
   - under `kv_region_mutex_`, insert the new entries and move out the shape-replaced ones;
   - unlock;
   - the moved-out handles stay in the guard and drop in its destructor, after L1.
   - This is where a shape change takes effect. The old region stays alive through the old KV
     buffers' references until llama frees them. The fit never counted it as a self extent,
     because it is still physically occupied. The existing WARN ("the re-fit counts
     still-allocated KV as used") stays.

**Shortfall is a runtime outcome, not a planner bug (r1 C2).** The yield may return
`freed < picked`, through jehw's `pending_bytes`, or because its barrier wait failed (the frees
stay queued). Step 6's re-fit then sees less space: it either asks for more yield (loop) or
demotes more (commit). Demotion always terminates, because all layers on the host always fit.

**`[KV-PLAN-BUG]` is only this:** step 6's re-fit said `fits`, and carving exactly those
extents under the same lock failed. That is an allocator bug by construction; §2.8 covers it.

**Probe mode** runs steps 1 and 3 only: no ring release, no yield, no carve, no side effects.

**`no_alloc` contexts (r2 m6).** When the published shape has `no_alloc` set (llama's
dummy-buffer contexts, master `llama-kv-cache.cpp:389-393`), the transaction publishes the
residency and reserves nothing, and the claim is skipped for their size-0 buffers.

**Teardown, in both close paths (r2 N-I3(b)).** The registry erase lives in the backend,
at the one point both close paths reach: `ggml_sycl_execution_clear_bindings_for_context`,
which runs on success in both of the following:
- `ggml_backend_sycl_execution_context_close_if_idle` (master `ggml-sycl.cpp:15152-15161`),
  which is the construction-unwind path (llama `llama-context.cpp:741-753`), including a
  `[KV-PLAN-BUG]` refusal inside `create_memory`;
- `ggml_backend_sycl_execution_context_finish_drain` (`:15117-15138`), which is the end of
  `llama_context_sycl_exec_drain_and_close`.

The erase follows the leaf rule: under `kv_region_mutex_`, move every `(c, *)` entry out,
unlock, then drop. The last `mem_handle` reference (the registry's, or the KV buffers',
whichever goes last) frees each extent through `zone_free`, which runs the §2.3.2 free rule.

#### 2.4.3 Planned context-side demand (owner ruling 2026-09-26: "plan them exactly first")

**The ruling.** The owner ruled on the question revision 3 left open: **plan them exactly
first**. There is no interim estimate, no fixed floor and no fallback constant.
- llama.cpp-zhcn (the compute buffer, and the fattn K/V conversion buffers, sized from the
  graph shape) and llama.cpp-23mk (the remaining out-of-arena cohorts, routed through planned
  zones) produce the numbers.
- Both block moua L4 onward (tracker deps, moua comment c-2yg4).
- Revision 3's estimate-plus-floor formula, its measured floor and its once-per-device miss
  WARN are all withdrawn.

**The interface (proposed to impl-zhcn and impl-23mk on 2026-09-26, pending their answers).**
Every planned tenant that can land on a shared TLSF's context side publishes one record per
device:
```
struct context_side_demand {
    int                  device;
    vram_zone_id         zone;      // WEIGHT or KV: which TLSF's context side (§2.3.5 routing)
    shared_zone_lifetime lifetime;  // CONTEXT or TRANSIENT (§2.1)
    const char *         cohort;    // the cohort_id its allocations carry
    size_t               bytes;     // planned PEAK concurrently-live bytes on this device
};
```
- **Producers** are pure, host-testable functions, one per owning ticket, called by the
  runtime-context transaction:
  - zhcn: the compute buffer, which is the `backend-buffer-kv-zone` overflow, and the fattn
    K/V conversion buffers;
  - 23mk: every census cohort it routes into a shared zone's context side;
  - a cohort routed into a tail zone (ONEDNN/RUNTIME/SCRATCH are separate TLSFs) is outside
    this interface.
- **The consumer** is `transient_reserve(d, tlsf)` in L3:
  `Σ max(0, demand.bytes − live_bytes(cohort, d, tlsf))` over the demands routed to that
  TLSF. Bytes a cohort already holds live are in the geometry, so they are never charged
  twice (r2's statistic point).
  - The u1bb ring is not a demand. It is released before the fit and re-admitted after the
    carve, because KV wins over it (§2.4.2 steps 2 and 7).
- **Timing.** Every number must exist when the transaction runs, which is under L1, before
  `create_memory` and before `sched_reserve`. So a size that ggml only learns at
  `sched_reserve` must be derived beforehand from the graph shape. The other option, a
  republish after `sched_reserve` that re-fits, is less attractive: it would add a
  post-memory re-reservation to §2.4.2's idempotence rule. The interface asks zhcn which of
  the two applies.
- **Plan == reality at allocation time.** `context_side` meters live bytes per
  `(device, TLSF, cohort)`:
  - An allocation that takes its cohort past the planned bytes is a plan violation. It logs
    `[KV-PLAN-BUG]`-class ERROR once per `(device, cohort)`, with planned vs actual, and
    aborts under `GGML_SYCL_STRICT_KV_PLAN=1`.
  - A context-side miss for a planned cohort is the same error.
  - An allocation on the context side whose cohort has **no** demand record is a census
    failure. H7p gates it in source, and at runtime it logs the same ERROR.
  - The meter is exposed through a `GGML_SYCL_PRIVATE_TESTING` accessor, and G1 reads it.
- **Scope gap, raised with 23mk and the lead.** The in-arena TRANSIENT scratch that already
  lands in the shared zone today appears in neither ticket's description:
  - the per-op MMVQ Q8 scratch (master `ggml-sycl.cpp:46120`) and the SOA Q8_1 (`:97965`);
  - STAGING (`common.hpp:6658`, `unified-cache.cpp:21455`, `mmvq.cpp:16863`);
  - persistent buffers (`unified-kernel.cpp:4890`) and `scratch_pool` (`unified-cache.cpp:20629`);
  - the opt-in fattn packed-K sidecar and materialize requests (`fattn.cpp:569`, `:1640`), which
    are KV-role and TRANSIENT (§2.8). These sit closest to zhcn's fattn conversion work.

  23mk's census (200 request sites) may already cover them. If it does not, moua L3 takes
  them, with exact demand functions of the shapes known at context time. Any of them that
  has no exact function is escalated to the lead as a blocker, never floored.
- The fit leaves each TLSF's `transient_reserve` in that TLSF's frontier extent.

#### 2.4.4 The KV shape: one per-layer byte function (r2 N-C1)

**The defect in revision 2.** Revision 2 sized slots with the planner's shape, and that
shape is not what llama allocates:
- `kv_layer_bytes_for_kind` hard-codes `sizeof(ggml_fp16_t)` (master `unified-cache.hpp:582`,
  `:591`).
- The publish passes only `n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full, flash_attn`
  (master `llama-context.cpp:1147-1148`).
- The actual per-layer size is decided elsewhere, from the actual buffer: *"The slice is the
  ONE place the per-layer KV size is decided"* (master `ggml-sycl.cpp:38593-38612`).

Once a region physically reserves a slot, three configurations that work today would break:
- `-ctk`/`-ctv q8_0` (slot ≠ actual, so `[KV-PLAN-BUG]`);
- FA-off models with variable V width, which llama pads to `n_embd_v_gqa_max()` under
  `v_trans` (master `llama-kv-cache.cpp:226-229`, `:277-279`; actual > slot);
- contexts whose layer set is not the model's. These are the MTP filter, the GEMMA4
  assistant's shared layers, and GEMMA3N/GEMMA4 reuse (master `llama-model.cpp` create_memory,
  its `filter`/`reuse`/`mem_other` lambdas). Such a context would reserve slots it never
  claims.

**The shape crosses the ABI once, from llama, which is the side that knows it.**
- A new publish entry point, `ggml_backend_sycl_set_runtime_context_for_model_kv(backend,
  token, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full, flash_attn, const
  ggml_sycl_runtime_kv_shape * shape)`. It is resolved by proc address like its sibling, and
  the old entry point stays.
- `ggml_sycl_runtime_kv_shape` is
  `{ uint32_t struct_size; uint32_t layer_desc_size; int32_t type_k; int32_t type_v;
  uint8_t v_trans; uint8_t no_alloc; uint32_t n_layer; const ggml_sycl_kv_layer_desc * layers; }`.
- Each `ggml_sycl_kv_layer_desc` is `{ uint32_t n_embd_k_gqa; uint32_t n_embd_v_gqa; uint8_t has_kv;
  uint8_t is_swa; }`:
  - the widths are exactly the ones llama passes to `ggml_new_tensor_3d` (master
    `llama-kv-cache.cpp:347-348`). `n_embd_v_gqa` is taken after the `[TAG_V_CACHE_VARIABLE]`
    padding, and is 0 when the model has no V (MLA);
  - `has_kv = 0` marks a filtered, shared or reused layer.
- The backend reads `layers[i]` at the stride `layer_desc_size`, and refuses an unknown
  `struct_size`. Both libraries are built together, but this array crosses the dlopen
  boundary, where an appended field would otherwise silently move the stride.
- If libllama publishes through the old entry point while an arena device has device-planned
  KV, the transaction refuses with a named cause (`no KV shape published: libllama and
  libggml-sycl are out of step`). It does not guess the shape.

**On the llama side, one function produces the shape.**
- `llama_kv_layer_shapes(model, params_mem, cparams)` is factored out of two places:
  `create_memory`'s filter/reuse/share decisions and `llama_kv_cache`'s per-layer width
  decision. Both of them, and the publish, call it. It covers iSWA's two caches, and the
  attention half of the hybrid memories. Recurrent state does not use the tiered KV buft and
  is out of scope.
- **Frozen once computed.** llama computes it at the constructor's first publish (master
  `llama-context.cpp:810`, before `create_memory` at `:869`), stores it in `llama_context`, and
  every later republish sends the stored copy: the ladder (`:1571`, `:1936`) and the FA recheck.
  This matters because auto-FA resolves after the memory exists (`:1283`), and a recomputed
  `v_trans` would change the key (§2.4.2 step 1) for a KV that did not change.

**On the backend side, one function sizes a layer.**
- It is `kv_layer_tensor_bytes(desc, type_k, type_v, cells)`:
  - K's bytes are `ggml_row_size(type_k, n_embd_k_gqa) × cells`, and V's are the same with
    `type_v` and `n_embd_v_gqa`;
  - each is padded by the rule ggml's context allocator applies for the tiered buft (its
    alignment and `get_alloc_size`).
- `cells` comes from `kv_layer_cells(kind, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full,
  n_swa)`, the cell arithmetic factored out of `kv_layer_bytes_for_kind` unchanged.
- The fit's slot size is `kv_layer_alloc_bytes(kv_layer_tensor_bytes(...))`.
- The tiered claim computes each claimed layer's bytes with the same call on the same stored
  shape. It checks that the buffer's size equals the sum over its layers, and that each slot
  equals its layer's bytes. For arena devices this replaces the even division at master
  `:38593-38612`.
- `kv_layer_bytes_for_kind` keeps only the **load-time** estimate. No context exists at load
  time, so it calls the same function with an f16 shape. It never sizes a region.

**Tests.**
- H2/H3 add these cases:
  - K and V at q8_0;
  - `v_trans` with variable V width, and MLA (no V);
  - an MTP context and a GEMMA4-assistant context, whose layer sets exclude layers (no slot
    for them);
  - GEMMA3N reuse;
  - a same-shape ladder republish, which reuses the region. Its n_ubatch-only key change must
    not re-reserve (r2 N-I9).
- L6 adds a CPU-buft-only llama test. For synthetic hparams covering those cases, the
  function's descriptors must equal the tensors `llama_kv_cache` creates (`ggml_nbytes` per
  layer).

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

**Registry** (`unified_cache`, per device, under `kv_region_mutex_`, a leaf lock, §2.3.1):
`ContextId -> {extent mem_handles[], shape key, frozen n_ubatch, kv_region_layout}`, where
`kv_region_layout` maps each device layer to `(extent index, slot_offset, slot_size)`. There
is one `mem_handle` per extent, each its own owner-first control (r2 N-I2; §2.4.1). The
transaction publishes entries at its commit step (§2.4.2 step 8), the claim and the residency
hook read them, and teardown erases them.

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
- The KV-mask handoff is unchanged. The claim checks **mask == slot table**, and a mismatch
  is `[KV-PLAN-BUG]` naming both.

**Llama's residency answer comes from the registry (r2 N-I5).**
`llama_kv_cache` decides host vs device per layer through
`ggml_backend_sycl_kv_layer_on_device_from_dev` (master `ggml-sycl.cpp:107332-107343`), and
that builds the mask. Today the hook reads the **process-global** plan snapshot (*"the placement
plan is process-global"*). The transaction and `create_memory` are not atomic together, so
another context's transaction, or another model's publish, can land between them:
A's transaction, then B's (B's fit sees A's region and demotes more), then A's `create_memory`
reads B's `kv_device`. A's mask then disagrees with A's slot table, and A is refused.

Under an open region scope, the hook therefore answers from the registry: layer `il` is
device-resident iff some device's entry for the scope's ContextId has a slot for `il`. The
mask and the slot table then have one source, and the mask check becomes a check of llama's
mask assembly, not of two plans. Outside a scope (no arena, or a device with no reservation),
the hook reads the plan as today.
- H8 adds this interleaving: publish A, publish B, then create A, with the shared plan
  mutated in between. A's mask must equal A's slot table.
- What stays process-global is pre-existing and unchanged: compute-time readers of the plan,
  such as `ggml_sycl_op_is_planned_on_host`, and the single published plan per model. KV
  *allocation* becomes context-keyed. Contexts with different residency on one model and
  device remain the contract's §5.3 limitation.

**No scope.** An arena device with device-planned layers and no open scope is a caller
contract violation. It is refused with `[KV-PLAN-BUG] ... no KV region scope`. L6
enumerates the tests that allocate tiered KV buffers directly (the
`test-sycl-lifecycle-*` family and the kv-layer-sizing source gates) and gives them a scope
through a `GGML_SYCL_PRIVATE_TESTING` hook.

### 2.6 `tiered_kv_buft_alloc_buffer` (L6)

The device-planned branch (master `ggml-sycl.cpp` ~38590-39200):
1. **Resolve the scope, and for each owner device the registry entry.** Check that every
   device-planned layer has a slot whose size equals
   `kv_layer_alloc_bytes(kv_layer_tensor_bytes(...))` for that layer, from the entry's stored
   shape (§2.4.4). Check that the buffer's size equals the sum of its layers' bytes, and that
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
   - **After L6, `GGML_SYCL_VMEM_KV=1` is effectively inert everywhere (addendum, VMEM wording).**
     Revision 2 said "with no arena, vmem-kv behaves as today", which was false. The vmem
     branch is itself gated on `ggml_sycl::vram_arena_enabled()` (master `ggml-sycl.cpp:38856`),
     so it never runs without an arena.
     - The one case left is a device where the arena is **enabled but not active**, because
       its reservation failed. There vmem-kv still runs, as today.
     - L6 keeps that residual branch rather than widening its scope. llama.cpp-1oxa deletes
       vmem-kv outright (pattern #2 replaces it), so moua does not carry a second deletion.
5. **`GGML_SYCL_BLOCK_EXEC_CANDIDATE_KV` is ignored under an arena (r2 N-I4).** Its buffer-time
   `kv_device` reassignment (master `:38623-38649`) would be a second residency source beside
   the slot table. It gets the same treatment as VMEM_KV: one WARN per process, and the region
   decides.

No device layer issues a per-layer `unified_alloc` any more. The host-tier branch is
unchanged. The arena-device uses of `kv_admission_mismatch`, `kv_vram_cap` and
`kv_device_budget` are deleted per §2.2.

### 2.7 u1bb ring, MMID pools, RUNTIME, compute overflow (r1 M9 corrected)

- **The ring's KV-zone slots** are CONTEXT-class and placed by `context_side_place`. The
  ring's admission reads the same function (§2.2, rows "u1bb ring").
  - When a region is reserved, the ring is released before the fit and re-admitted after the
    carve (§2.4.2 steps 2 and 7). That is u1bb's own "KV wins, then re-admit the ring" order,
    made physical.
- **Revision 1's claim that ring slots "cascade because they are lowest" was false.** The
  compute-buffer overflow (`backend-buffer-kv-zone`, master `ggml-sycl.cpp:37235`) is
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

1. **§2.4.2 step 6, the commit carve:** the fit said `fits` under the lock, and the carve failed.
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
(`!allocation || tier != DEVICE_VRAM`). That is P1-compliant.
- **Production never reaches them (r2 m4).** The sidecar is opt-in only:
  `GGML_SYCL_PACKED_K_SIDECAR`, or `GGML_SYCL_FA_FORCE_PATH=split-packed` (master
  `fattn.cpp:151-157`). So the routing matters only under those flags, and C6 has no fattn
  clause.
- **A latent defect becomes more reachable.** When an earlier update's sidecar allocation
  missed, a later `set_rows` can create the sidecar, zero-fill it, and pack only its own rows
  (`fattn.cpp:560-590`). The lookup (`:412-444`) has no completeness check, so it serves stale
  packed K. Forbid-spill makes the miss-then-create sequence more likely under those flags.
  This is filed as **llama.cpp-cxgg**: invalidate the sidecar on any update miss. It is
  pre-existing, and moua does not fix it. It should land before anyone relies on the flag.

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
- **Weight holes are region extents, as a last resort before any yield.** The fit packs
  whole slots into a weight hole only after the context side (retained runs and the free gap)
  cannot hold them (tier 3 of §2.4.1's cost-ordered pack). The hole extent returns to the
  weight side when the context ends, and it never sits inside the frontier walk.
  - **Its cost, stated plainly (addendum (c)).** A hole extent pins that weight-side hole for
    the context's lifetime, which is lifetime mixing on the weight side. The server-swap
    pattern shows it: model 1 is unloaded, a model-2 context's KV takes model 1's old hole,
    then model 3 loads. Model 3's weights cannot use the hole, so they bury the ladder (they
    take the gap front).
  - The fit stays exact, so P4 holds, and the burying WARN fires. But the segregation the
    design exists for is relaxed for as long as that context lives. H5 replays this second
    swap and asserts the fit is still exact and the WARN still fires.
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

The r2 reviewer offered a second alternative (N-I10): a WEIGHT-class allocation that would bury
the ladder yields the bottom ladder tenants it needs instead. It keeps the ladder contiguous,
but it pays a yield (and a barrier wait) on every burying load, including every runtime
expert-cache fill on GPT-OSS, whether or not any context's KV ever needs the room. It also
moves a yield into the weight staging path, under that path's cache locks. The lazy form
yields only at a reservation, where the transaction already yields, so it is kept. If C1-C3
ever show the buried-tenant census costing measurable fit time, the eager form is the
fallback.

**What remains a limit, and its diagnostic.** A hole smaller than one slot cannot hold KV;
slots are never split. Whenever the demotion loop demotes a layer while the geometry holds
weight-side free bytes, log at WARN:
```
[SYCL-PLAN] KV overflow on device 0 with 38 MB of weight-side free space in holes smaller
  than one 128 MB slot (pattern #1 never splits a slot; llama.cpp-1oxa removes this limit)
```
L7 documents this limit, and pattern #2 remains the remedy.

**This supersedes lead decision (c)** ("accept the limit"). The lead provisionally agrees.
- **It does relax the owner's segregation invariant, so the owner sees it (addendum (e)).**
  Letting context-lifetime KV into weight-side holes is a policy change. Revision 2 said it
  "adds no policy", and that was wrong.
- With the cost-ordered pack it is a strict last resort before any yield. The one line for
  the owner is on llama.cpp-moua: *"KV may occupy a weight-side hole only when the context
  side (retained runs and the free gap) cannot hold it; optional copies are yielded only
  after that, cheapest layout loss first, and a buried copy only when it is cheaper than
  growing the frontier prefix."*

### 2.10 The canonical memory contract

- **§3 allocator allowlist.**
  - Each region extent is an owner-first `CACHE_SUBALLOCATION` through `unified_cache`, and
    slots are `slice()`s of their extent's handle.
  - **The new entry points are added to §3's allowlist** (audit m2): `reserve_kv_region`,
    `zone_alloc_optional`, and `context_side_place`'s allocating wrapper.
  - The allocation class of each is **derived from the request** (its `role`,
    `prefer_vram_zone`, and the §2.1 `lifetime` field) by the existing classifier. It is never
    hand-set at the call site.
  - The new publish entry point (§2.4.4) allocates nothing. It carries the shape only.
  - No new raw allocation site is added, and no new `CACHE_BACKING` mint.
  - The one route by which planned KV reached `unified_cache_malloc_device_tracked` under an
    arena (the per-layer tiered `unified_alloc`) is removed.
- **§1.1 planner authority.** The planner decides residency per layer, as today. Its capacity
  input is the allocator's geometry, its size input is the shape llama allocates (§2.4.4), and
  an admission is a reservation, so "planned device" means physically reserved.
- **`mem_handle` lifetime.**
  - Each region extent is freed only when its last handle drops: the registry entry's, the
    KV buffers', the layer views', or a fill event's slices.
  - There is no forced eviction. The yield takes only optional tenants that pass jehw's lease
    predicate, and a leased tenant truncates the ladder.
  - Retained runs are TLSF-allocated storage with no owner and no registration (§2.3.2,
    audit I3). They are kept off the TLSF's free lists so that weights cannot take them.
    They are not leaked handles, and every "live bytes" reader subtracts them.
- **§12.5 locking.** The new `kv_region_mutex_` is ranked L3 and is strictly leaf (§2.3.1).
  - Every final region-handle drop happens after every listed lock is released. The
    transaction guard drops after L1, and the registry erase drops after `kv_region_mutex_`.
  - No L5 lock is held across a wait or across the yield (§2.4.2).
  - The yield's wait and final drops under L1 are pre-existing on jehw and are not widened.
    §2.3.1 states them, for jehw's owner to resolve.
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

- **H1 `test-tlsf-allocator`** (L1, done and approved: `eab1ebeb6` … `456650c01`). Registered
  as `hostonly`, with 24 cases, including `allocate(SIZE_MAX)`'s refusal.
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
  - **Commit under churn (r2 N-I7).** Between plan and commit, TRANSIENT tenants are
    allocated below the anchor. The commit re-fit must succeed with moved extents and no
    further yield, and must not loop. A variant that needs more yield loops once, then
    commits. A variant that keeps moving refuses after 3 with the named cause.
  - **Shape cases (r2 N-C1, N-I9):** see §2.4.4 (q8_0, `v_trans`, MLA, MTP, the assistant,
    reuse, and the ladder republish).
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
    - `live_bytes() = used() − retained_bytes` and `zone_available() = available()` (retained
      counted as used) at every step (addendum NEW-1).
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
  - **Cost-ordered pack (addendum NEW-2), RED first:** a gap that fits the region, plus a
    lease-free buried tenant. The buried tenant must **stay resident** and nothing may be
    yielded. On revision 2's pack order this fails: the buried tenant is released. Further cases:
    - a free weight hole is used only when the gap cannot hold the slot;
    - when both yield candidates exist, the one with less layout bytes lost per slot goes
      first, and a tie goes to the frontier.
  - **Second server swap (addendum (c)):** model 1 unloads, a model-2 context's KV takes model
    1's hole, then model 3 loads. Model 3's weights bury the ladder, the WARN fires, and the
    fit stays exact.
  - **Pending ranges (addendum (b)):** between the yield and the carve, a WEIGHT allocation
    that would fit the pending extent is placed elsewhere or misses. The commit re-fit
    succeeds with no loop, and a rollback clears the ranges.
- **H6 N-chunk.** Two weight TLSFs plus the tail KV TLSF, checking greedy packing across
  TLSFs and extents, fit == carve, and the TRANSIENT routing: WEIGHT-naming requests go to the
  last weight chunk, KV-naming ones to the KV TLSF, each TLSF keeping its own reserve (r2 m7).
- **H7 source gates** (python; the kv-layer-sizing family plus a new
  `test-sycl-kv-region-source.py`, each check with a mutation witness):
  - (a) the region reservation never reaches `unified_cache_malloc_device_tracked`;
  - (b) the tiered device-planned branch issues no per-layer `unified_alloc`;
  - (c) the optional pass runs after all S1 staging, dense *and* expert/DPAS;
  - (d) **one source, by primitive (r2 N-I4)**: the capacity primitives listed in §2.2 occur
    only at allowlisted sites. The deleted arena-device uses of `kv_admission_mismatch`,
    `kv_vram_cap` and `kv_device_budget` stay deleted. The one byte function
    `kv_layer_tensor_bytes` is the only per-layer KV sizer that feeds the fit or the claim;
  - (e) `[KV-PLAN-BUG]` is logged at ERROR at both sites, and STRICT aborts;
  - (f) no raw pointer is stored as region state;
  - (g) `reserve_kv_region` does not hold the group mutex across the yield call (C2);
  - (h) no dispatch path re-stages a yielded optional copy. Dispatch reads the primary's
    materialized layout (audit m4);
  - (i) the slot view's pointer comes from `slice().resolve()`, and `set_owner` has the
    `mem_handle` overload (audit m1);
  - (j) the new entry points appear in contract §3's allowlist, and their allocation class
    is derived, not hand-set (audit m2);
  - (k) `kv_region_mutex_` is leaf: no lock is acquired, and no `mem_handle` is destroyed,
    inside its scopes, and no `g_pending_kv_layer_masks_mutex` scope contains one (r2 N-I1,
    addendum);
  - (l) the transaction's `kv_region_txn` guard is declared before its L1 `lock_guard`, and
    the registry commit follows the publication CAS (r2 N-I3);
  - (m) the registry erase sits in `ggml_sycl_execution_clear_bindings_for_context`, which
    both close paths reach (r2 N-I3(b));
  - (n) `WEIGHT_SIDE_TRANSIENT` appears at exactly the two B50-commented sites (r2 N-I8);
  - (o) llama's later publishes send the stored KV shape, never a recomputed one (r2 N-I9);
  - (p) every context-side allocation site carries a cohort that has a `context_side_demand`
    producer, and no estimate constant or floor exists in `transient_reserve` (owner ruling,
    §2.4.3).
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
  - **Plan race (r2 N-I5):** publish A, publish B with a different residency (the shared
    plan changes), then run A's `create_memory` attach. A's residency hook answers and A's
    mask must equal A's slot table. The positive control answers from the shared plan and
    must fail.
- **H9 transaction guard (r2 N-I3; SYCL-free, in `kv-region-registry.hpp`).** A host model of
  the §2.4.2 steps, with a failpoint at every refusing step (after the ring release, after
  the carve of device 0 of 2, at the MMID step, at the ring re-admit, and at the CAS). At
  each failpoint it asserts:
  - the registry is unchanged;
  - every carved extent's handle is dropped, and dropped only after the instrumented L1 and
    registry locks are released;
  - the ring re-admit callback ran with the old `n_ubatch`.

  After a success, a teardown through either close path empties `(c, *)`.

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
  log callback exactly once;
- **rollback and teardown on a real cache (r2 N-I3):**
  - a forced refusal after the carve leaves `arena_owns` false for the extents, and the
    zone's live bytes back at their pre-call value;
  - `close_if_idle` on a never-run context erases its entries and frees its extents;
- **the per-cohort meter** (§2.4.3): it moves when per-op scratch is allocated. A forced
  over-plan allocation and a forced context-side miss each log the plan-violation ERROR once
  per `(device, cohort)`, and STRICT aborts in a subprocess.

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
- **C2a B50 MMVQ/STAGING first submit (r2 N-I8), a named acceptance item.** The C2 B50 run,
  on `level_zero:1` at the **default** PCT (no `GGML_SYCL_VRAM_BUDGET_PCT`), is where the
  region and the per-op MMVQ/STAGING scratch sit at the top of a mostly empty shared zone,
  directly below ONEDNN.
  - Pass: rc=0, the digit output, zero aborts, and no `UR_RESULT_ERROR` in stderr.
  - Its positive control is the base tree, where those pages are untouched. On a failure,
    rerun with `GGML_SYCL_B50_SCRATCH_WEIGHT_SIDE=1` (§2.1's lever). A pass there makes the
    lever the default for the two B50 sites.
- **C3 GPT-OSS B50 gate:** `llama-cli ... -c 4096`, scored `grep -cx '1, 2, 3, 4, 5'` = 1. It
  is also run once with u1bb's `-ub 1024` acceptance form, to cover the ring and the ladder.
  - With `GGML_SYCL_KV_REGION_TRACE=1`, a WARN line per reservation that is off by default,
    the ladder run must show exactly **one** reservation for the device, however many ladder
    candidates republish (r2 N-I9).
  - C3 is also the default-PCT half of C2a for STAGING.
- **C4 G1** (§3.2).
- **C5 multi-context:** `test-thread-safety` with `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`,
  ONE run, only if it is green on the base first. It is known-crashy (oze0), so compare
  against the base. H8 is the primary coverage, and C5 is corroboration.
- **C6 perf sanity:** B50 Mistral pp512/tg128 and GPT-OSS pp512, ABBA ×3 against the base.
  - It adds a **no-replay decode arm** (r2 m8): Mistral tg128 with
    `GGML_SYCL_DISABLE_GRAPH=1`. In that mode every per-op TRANSIENT allocation pays a
    best-fit walk of the retained runs plus up to 3 TLSF operations under the group mutex,
    where replay hides it.
  - The expected delta is 0 on every arm.
  - If it is not zero, look at the TRANSIENT reclassification: scratch addresses move from
    the weight side to the context side, and each placement costs the walk above.
  - There is no fattn clause: the sidecar is opt-in only (§2.8, r2 m4).

## 4. Decomposition, effort, landing order

| id | work | files | effort | depends on | lands |
|----|------|-------|--------|------------|-------|
| L1 | TLSF placement primitives, tags, frontier walk; H1 | `tlsf-allocator.hpp`, `shared-zone-tags.hpp`, `tests/test-tlsf-allocator.cpp`, CMake | high | none | **done, approved:** `eab1ebeb6`, `9e0a708dc`, `97315421b`, `456650c01` |
| L3 | pure `kv_region_fit` (multi-extent, self extents, `forced_host`, strict prefix, carve mirroring, per-TLSF transient reserve), `context_side_place`, `transient_reserve` summing the `context_side_demand` records (no constant), the cost-ordered pack, `kv_layer_cells` + `kv_layer_tensor_bytes` (the one byte function), `kv-region-registry.hpp` (registry, scope, residency answer, `kv_region_txn` guard model); H2, H3, H6, H8, H9 | `kv-runtime-demotion.{hpp,cpp}`, `kv-region-registry.hpp`, `unified-cache.hpp` (`kv_layer_bytes_for_kind` delegates), their tests | xhigh | L1, jehw merged, the zhcn/23mk demand interface agreed | after jehw |
| L4 | `context_side` (explicit `lifetime` field threaded into `zone_alloc`, retained runs, atomic reuse, settle reset, the per-cohort meter and plan-violation ERROR, pending reservation ranges, `live_bytes()` beside an unchanged `zone_available`), `allocate_excluding` and a whole-TLSF block census (L1 follow-ups), retained-run registration hygiene and the `live_bytes` readers, per-extent owner-first carve inside `zone_alloc`'s mint-before-lock protocol, leaf `kv_region_mutex_` added to contract §12.5 (L3), locked geometry snapshot, `reserve_kv_region`, strict-prefix `yield_optional_prefix`, `backend-buffer-kv-zone` passing its buffer's role, removal of the dead `KV_AUTO` reclaim and of the `arena_reserve` KV reclaim, N-chunk routing; H4, H4b, H5 | `unified-cache.{hpp,cpp}`, `tlsf-allocator.hpp` (one primitive), `ggml-sycl.cpp` (the kv-zone fallback's role) | xhigh | L1, L3, jehw merged, **llama.cpp-zhcn and llama.cpp-23mk** (owner ruling; comment c-2yg4) | after jehw, zhcn, 23mk |
| L5 | the optional pass after all S1 staging (dense + expert/DPAS); `zone_alloc_optional` | `ggml-sycl.cpp` S1 block | medium | L4 | with L4/L6 |
| L6 | llama side: `llama_kv_layer_shapes` factored out of `create_memory`/`llama_kv_cache`, stored at the first publish, the new `..._for_model_kv` publish, and the scope procs with an RAII guard. Backend side: the transaction steps of §2.4.2 (guard, idempotent key without `n_ubatch`, u1bb ring release/re-admit split, plan/yield/byte-budget/commit order, commit after the CAS), teardown erase in `clear_bindings_for_context`, the residency hook answering from the registry, the tiered claim with `set_owner(mem_handle)` slice views, the per-extent clear with event-held slices, VMEM_KV and BLOCK_EXEC_CANDIDATE_KV ignored under an arena, the second sources deleted (§2.2), both ERROR sites plus `GGML_SYCL_STRICT_KV_PLAN`, TRANSIENT routing including fattn sidecars, the dark B50 lever, `GGML_SYCL_KV_REGION_TRACE`; H7, the CPU-buft llama shape test, G1. **Absorbs revision 1's L2.** | `ggml-sycl.cpp`, `ggml-sycl.h`, `unified-cache.cpp`, `fattn.cpp`, `common.hpp`, `src/llama-context.{h,cpp}`, `src/llama-model.cpp`, `src/llama-kv-cache.{h,cpp}`, tests | xhigh | L3, L4, L5, **u1bb merged** | last |
| L7 | docs: memory-design section, contract §3/§5.2/§5.3/§12.5, arena comment, limits (§2.9), lock order, the `context_side_demand` interface, the owner-visible weight-hole line | docs, `unified-cache.hpp` comment | medium | L6 | with L6 |

- L3 through L7 form one series in one worktree, so there is one first build and the
  follow-ups reuse it.
- Every step is RED → GREEN with its host test.
- G1 and C0-C6 are the lead's runs at the end of L6.
- **Seams to coordinate before L3/L4:**
  - **jehw:** retire `select_optional_layout_yield`'s pruning in favour of the strict prefix,
    retire `kv_zone_snapshot`/`fit_capacity` from production (they stay as a test oracle),
    and reuse `kv_layer_alloc_bytes`.
  - **u1bb:** the ring admission reads `context_side_place` instead of `zone_largest_free(KV)`
    and `ggml_sycl_kv_capacity_live`. Its KV-zone slots become CONTEXT-class, and
    `ggml_sycl_replan_pp_moe_onednn_ring` splits into release and admit halves, so the
    transaction can release before the fit and re-admit after the carve (§2.4.2 steps 2 and 7).
  - **jehw, for its owner (not a moua change):** the yield's reader-barrier wait and its
    final-handle drops run under L1 (§2.3.1). moua's plan/yield/commit structure is the
    snapshot/revalidate shape that would let the yield run outside L1. But releasing L1
    mid-transaction lets concurrent context creations interleave and return BUSY, so it
    needs its own decision.

## 5. Decisions and open questions

- **(a) Refuse by default, and abort under `GGML_SYCL_STRICT_KV_PLAN=1`:** adopted, at
  region-backed sites only (§2.8).
- **(b) Yield order:** a strict address-ordered prefix, highest first (§2.4.1). This is a
  smaller change to jehw HEAD than revision 1 framed (r1 M7). The two-pass staging lands in
  the same series (L5).
- **(c) Weight holes:** **revised; the lead provisionally agrees.** With multi-extent
  regions, weight holes and buried optional tenants become region extents, and only
  sub-slot holes remain a limit, with a WARN (§2.9).
  - This is the fix for audit I5 (r2 N-I10): KV beats buried optional copies lazily, only when
    a context needs the room.
  - The reviewer's alternative, a burying WEIGHT allocation yielding the bottom ladder tenants
    it needs instead, is weighed in §2.9. It pays the yield on every burying load, whether
    or not KV ever needs the room.
- **(d) Handoff:** **revised.** A dedicated ContextId-keyed registry plus a thread-local
  region scope, which is a llama-side change (§2.5). The KV-mask handoff is kept and
  verified against the slot table.
- **(e) Decided (owner, 2026-09-26): "Plan them exactly first."** There is no interim
  estimate and no floor.
  - `transient_reserve` sums the planned `context_side_demand` records that zhcn and 23mk
    produce (§2.4.3).
  - zhcn and 23mk block moua L4+ (c-2yg4).
  - The interface is proposed to impl-zhcn and impl-23mk and is pending their answers. The
    open points are when each size is known relative to the publish, which cohorts land in
    the shared zone, and who owns the in-arena per-op scratch that neither ticket names.
- **(f) The KV shape crosses the ABI (r2 N-C1).** This is a llama-side ABI addition: a new
  publish entry point with a versioned struct, and the old entry point kept (§2.4.4). It is
  the lead's call, not the owner's, since it only threads facts llama already has.

## 6. Review dispositions

### 6.1 Design review r1

| id | finding | disposition |
|----|---------|-------------|
| C1 | a split reserves twice per device; a same-shape republish re-reserves | **Changed.** The registry is keyed `(ContextId, device)`, reservation is idempotent on the same shape key, and every backend's run finds the entry (§2.4.2 step 1). The same-shape republish keeps the region, matching `admitted_kv`. |
| C2 | yield under the group lock deadlocks (self-lock and ABBA); `freed < retained` misreported as PLAN_BUG | **Changed.** Plan with no lock, yield with no lock, then commit under the group mutex with a re-fit. The lock order is written down, and a shortfall re-plans and demotes (§2.4.2 steps 3-6, renumbered in revision 3; H7g). |
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
| m4 | keep the barrier and epoch; never lazily re-stage a yielded copy | **Changed** (§2.4.2 step 4 as renumbered in revision 3, H7h). The inherited P2 site (WOQ copy readers take no lease; recorded graphs bake pointers) is named and not widened. |
| pre-existing | the raw fallback, backend-buffer-kv-zone, onednn_weights_scratch and STAGING EXT-ALLOCs are named but not tracked | **Tracked.** llama.cpp-23mk already covers backend-buffer-kv-zone, onednn_weights_scratch and STAGING (plus c-m2jh). The fail-open raw fallback mechanism is new ticket **llama.cpp-gxur**, cross-referenced on 23mk (§2.11). |

### 6.3 Design review r2 (design-moua-r2, read `19f7f08bf`)

**Audit rows the reviewer found missing, re-checked against `1972b32b0`** (the audit fold it
did not see):

| audit id | r2 said | in `1972b32b0` | revision 3 |
|---|---|---|---|
| I1 | partial (same as C2) | locks named, but the L1 facts were missing | **Changed** under N-I1 below. |
| I2 | not resolved | one owner per extent in §2.4.1 and §2.6. But §0.1 P2 still said "a region is an owner-first CACHE_SUBALLOCATION", and §2.5 said `{mem_handle region}` | **Fixed.** Both are now per-extent handles. The carve goes through `zone_alloc`'s protocol (m5). |
| I3 | partial | unregister at retain, a fresh owner and exact registration on reuse, and rebuild/destroy/settle counting retained runs as free: all present (§2.3.2) | **Kept**, and minting now goes through `zone_alloc`'s protocol (m5). |
| I4 | partial | the region takes precedence under an arena, and VMEM_KV is ignored with a WARN, so the fit decides "fits" and no vmem pages exist (§2.6) | **Kept.** BLOCK_EXEC_CANDIDATE_KV gets the same treatment (N-I4). |
| I5 | not resolved | lazy yield of buried optional tenants, and their runs as hole extents (§2.9, §6.2) | **Kept**, and the reviewer's eager alternative is weighed (§2.9, N-I10). |
| m1 | not resolved | `set_owner(mem_handle)` stores `extent_handle.slice()`, and `ptr` comes from `slice.resolve()` (§2.6) | **Kept.** |
| m2 | not resolved | the new entry points are in the §3 allowlist, with a derived class (§2.10, H7j) | **Kept.** |
| m3 | not resolved | fill-event leases hold the slices (§2.6 step 3) | **Kept.** |
| m4 | partial | the barrier and "no lazy re-stage" were present, but the epoch bump was described as part of the yield | **Fixed.** The bump stays in the transaction (jehw `:17904`), not the yield (§2.4.2 step 4). |

**New findings:**

| id | finding | disposition |
|----|---------|-------------|
| N-C1 | slot sizes come from the planner's fp16 shape; the publish carries no `type_k`/`type_v`/`v_trans`/layer set | **Changed.** New publish entry point with a versioned shape struct. `llama_kv_layer_shapes` is factored out and frozen at the first publish. One backend byte function (`kv_layer_tensor_bytes`) serves the fit and the claim, replacing the even slice division for arena devices. H2/H3 add q8_0, `v_trans`, MLA, MTP, assistant and reuse cases, plus a CPU-buft llama test (§2.4.4). |
| N-I1 | the transaction holds L1 across the yield's wait and final drops; `kv_region_mutex_` unranked and held across them | **Changed.** The L1 facts are stated (§2.3.1). `kv_region_mutex_` is L3 and strictly leaf, never held across the yield, a carve or a drop. Erases move out, unlock, then drop. The guard drops after L1. The jehw wait under L1 is named for its owner (§4 seams). H7k. |
| N-I2 | one owner cannot own N extents | **Confirmed** from `1972b32b0`, and the two stale sentences are fixed (audit I2 row above). |
| N-I3 | rollback covers only the device loop; construction unwind leaks via `close_if_idle` | **Changed.** A `kv_region_txn` guard, declared before L1, rolls back on any non-committed return, including the byte-budget, MMID, ring and CAS refusals. Registry commit is the last step, after the CAS. The erase sits in `clear_bindings_for_context`, which both close paths reach (§2.4.2). H9, H7l/m, G1. |
| N-I4 | §2.2 misses the MMID demotion, the accounting refusals, u1bb's `kv_capacity_live`/`with_slack`, `kv_layer_on_device` and BLOCK_EXEC_CANDIDATE_KV; H7d row-based | **Changed.** Six rows added (§2.2). H7d now forbids the capacity primitives outside an allowlist. The MMID demotion runs before the carve and feeds `forced_host`. |
| N-I5 | the residency hook reads the global plan; a racing transaction makes A's mask disagree with A's slot table | **Changed.** Under a scope, the hook answers from the registry for the scope's ContextId (§2.5). H8 adds the interleaving with a positive control. The global plan's remaining readers are named as pre-existing (§5.3). |
| N-I6 | the ring is live at the carve, and the reserve double-counts it | **Changed.** The ring is released before the fit and re-admitted after the carve (§2.4.2 steps 2 and 7), and it is not in the reserve (§2.4.3). u1bb seam: release/admit split. |
| N-I7 | "same extents or loop" turns TRANSIENT churn into refusals | **Changed.** Commit whenever the re-fit needs no further yield, even with moved extents or more demotion. Loop only for more yield (§2.4.2 step 6). H2 churn case. |
| N-I8 | the B50 rationale is false: the tail zones share the chunk at high offsets | **Changed.** The rationale is withdrawn (§2.1). C2a is a named acceptance item at the default PCT on `level_zero:1`, with a documented dark lever (`WEIGHT_SIDE_TRANSIENT` for the two B50 sites, `GGML_SYCL_B50_SCRATCH_WEIGHT_SIDE=1`). H7n. |
| N-I9 | the key includes `n_ubatch`-dependent slot bytes; the SWA ladder double-reserves | **Changed.** The key is built from request inputs, without `n_ubatch`. Slot sizes are frozen at the reserving `n_ubatch`, and later publishes send llama's stored shape (§2.4.2 step 1, §2.4.4). C3 counts one reservation. H7o. |
| N-I10 | audit I5 still open | **Fixed** in `1972b32b0` (§2.9, §6.2). Revision 3 weighs the reviewer's eager alternative (§2.9, §5 (c)). |
| N-I11 | §6 had no audit rows | **Fixed** in `1972b32b0` (§6.2), and re-checked above. |
| m1 | EXPERT_CACHE is a category, not a role; it is GPT-OSS's main burying path | **Changed.** Table wording, and a note that it is the main path (§2.1). H5. |
| m2 | `backend-buffer-kv-zone` hard-labels weight buffers COMPUTE | **Changed.** It passes its buffer's own role, so a weight buffer is WEIGHT-class and a compute buffer is CONTEXT-class (§2.1, L4). |
| m3 | `zone_alloc` has no role; CONTEXT vs TRANSIENT is not role-derivable | **Changed.** An explicit `lifetime` field in `alloc_constraints`, threaded into `zone_alloc` (§2.1). |
| m4 | the fattn sidecar is env-gated; a latent stale sidecar | **Changed.** C6's fattn clause is removed. The latent bug is filed as **llama.cpp-cxgg** (§2.8). |
| m5 | hand-minting the region owner | **Changed.** The mint and registration go through `zone_alloc`'s mint-before-lock protocol (§2.3.2). |
| m6 | `no_alloc` contexts would reserve | **Changed.** `no_alloc` is in the shape, and no region is reserved (§2.4.2). |
| m7 | N-chunk TRANSIENT routing unspecified | **Changed.** WEIGHT-naming requests go to the last weight chunk and KV-naming ones to the KV TLSF, with a per-TLSF reserve (§2.3.5). H6. |
| m8 | per-op TRANSIENT cost under the group mutex | **Changed.** C6 adds a no-replay decode arm. |
| m9 | citation drift | **Fixed.** `admitted_kv` is at master `:17738-17747` (jehw `:17752-17760`), and `backend-buffer-kv-zone` is at `:37235`. |
| m10 | the key must come from request inputs only | **Changed** (§2.4.2 step 1). |
| (c) | transient floor, with conditions | **Superseded by the owner's ruling** ("plan them exactly first"): no floor. `transient_reserve` consumes zhcn/23mk's planned demands (§2.4.3, §5 (e)). |
| M9 (r1) citation | `:37313-37342` | **Fixed** to `:37235` (§2.7). |

### 6.4 design-moua-r2 addendum on `1972b32b0`, and the owner's ruling

The addendum reviewed `1972b32b0` and did not see revision 3 (`0eba7875b`). The "rev 3" column
says what revision 3 already covered.

| id | finding | rev 3 (`0eba7875b`) | revision 4 |
|----|---------|---------------------|------------|
| audit I1 | L1 and `kv_region_mutex_` held across the jehw wait | **covered**: L1 facts stated, and `kv_region_mutex_` is leaf and never held across the yield (§2.3.1) | kept |
| L3 co-hold | `kv_region_mutex_` must never be co-held with `g_pending_kv_layer_masks_mutex` | not covered | **Changed.** Sequential, never nested: pop the mask, release, then read the registry (§2.3.1, H7k). |
| audit I4 wording | "with no arena, vmem-kv behaves as today" is false: the branch is arena-gated (`:38856`) | not covered | **Fixed.** Inert after L6, except where the arena is enabled but not active. 1oxa deletes it (§2.6). |
| NEW-1 | subtracting retained runs from `zone_available` counts unplaceable bytes as free | not covered (rule 3 still said it) | **Changed.** `zone_available` is unchanged (retained counted as used). A separate `live_bytes()` serves only rebuild, destroy, settle and leak checks. Both are H7d primitives, and H4 asserts both (§2.3.2). |
| NEW-2 | the pack released buried tenants before using the free gap | not covered | **Changed.** The pack is cost-ordered: retained, then the gap, then free weight holes, then yields ranked by layout bytes lost per slot, ties to the frontier (§2.4.1). H5 RED: a fitting gap keeps the buried tenant resident. The order puts the gap before free holes, a deliberate change from the addendum's literal list, so that (e)'s rule, holes only when the context side cannot hold it, holds. |
| (a) | buried yields must be a last resort ranked by cost | not covered | **Changed** (NEW-2). |
| (b) | a yield-to-carve race hands the freed block to a concurrent WEIGHT allocation | step-4 extent equality relaxed (N-I7), but the race remained | **Changed.** The fit's extents are recorded as pending ranges before the yield, and weight placement skips them until the carve or a rollback (§2.4.2 step 4, §2.3.3, H5). |
| (c) | state the weight-hole lifetime cost; replay a second swap | not covered | **Changed** (§2.9, H5). |
| (d) | leased buried tenants split runs; the census runs under the group mutex | the split was covered; the lock was not stated | **Changed** (§2.4.1 inputs). |
| (e) | the owner should see one line: KV in weight holes only when the frontier cannot hold it | not covered ("adds no policy") | **Changed.** The claim is withdrawn, and the line is on llama.cpp-moua as comment **c-2vv1** (§2.9). |
| r2 N-C1, N-I1..N-I9 | still open at `1972b32b0` | **all covered** in revision 3 (§6.3) | kept |

**The owner's ruling on §5 (e), 2026-09-26: "Plan them exactly first."** Revision 3's
estimate-plus-floor reserve is withdrawn. `transient_reserve` sums the planned
`context_side_demand` records from zhcn and 23mk, with no constant. An over-plan or a missed
planned cohort is a plan-violation ERROR, and it aborts under STRICT (§2.4.3). zhcn and
23mk block L4+. The interface is proposed and pending both implementers' answers (§5 (e)).
