# llama.cpp-moua: planned, lifetime-segregated layout for the shared KV+WEIGHT zone

Design, revision 7.6. Author: impl-moua, 2026-09-26. The revisions answer eight reviews:
- design review r1 (design-moua-r1: 3 Critical, 7 Important, 9 Minor), recorded in §6.1;
- the principles audit's moua section (audit-mem-b: 5 Important, 4 Minor), recorded in §6.2;
- design review r2 (design-moua-r2: 1 Critical, 11 Important, 10 Minor), recorded in §6.3;
- design-moua-r2's addendum on `1972b32b0` (2 Important, 5 conditions, 2 Minor), recorded in
  §6.4, together with the owner's ruling on the transient reserve;
- design review r3 (design-moua-r3 on `456650c01..1dfc63531`: 2 Critical, 8 Important,
  12 Minor) and the lead's five rulings on it, recorded in §6.5;
- design review r4 (design-moua-r4 on `f34acb398`: 0 Critical, 10 Important, 14 Minor), the
  lead's rulings on it, the single tenant protocol agreed with impl-zhcn, the r4 addendum,
  and the fold-ins queued during r4 (jehw's reclaim review, 23mk's sidecar, 1oxa's dump),
  recorded in §6.6;
- design review r5, twice: on `b021c9629` (0 Critical, 6 Important, 9 Minor) and on
  `c2613a688` (0 Critical, 7 Important, 13 Minor; it supersedes the first for `c2613a688`),
  the lead's rulings on both, the post-r5 queue (M1 single site, llama.cpp-uwlx, zhcn's exact
  wording, the host pool lock and phase gate, the four gaps from zhcn r3's list), zhcn rev 4's
  twelve requests, impl-23mk's two items, and the lead's rulings on revision 7's flags (the
  reap, ruling "B"), recorded in §6.7. Revision 7 is three commits on top of `c2613a688`:
  `99fd614da` (the `b021c9629` verdict and the queue), `a402c15af` (the `c2613a688` verdict,
  zhcn rev 4, 23mk) and revision 7.1 (`f2e5606bc`, the reap and the flag rulings), then
  `b30321a6f` (the reap's store mutex and the per-slot read);
- design review r6 (design-moua-r6 on `c2613a688..b30321a6f`: 0 Critical, 6 Important,
  10 Minor), the lead's rulings on it, design-zhcn-r4's corrections that reach this design (the
  GA figures, the reap's conditions, the cached slot table) and the 23mk review addendum,
  recorded in §6.8. Revision 7.2 is one commit on top of `b30321a6f`. Revision 7.3 answers
  design-moua-r6's rewrite of its verdict (0 Critical, 7 Important, 11 Minor) and adopts the
  lead's rulings file, recorded in §6.9. Revision 7.4 makes L0 process-global (rulings §E.2),
  and revision 7.5 withdraws step 8′ (rulings §B.2) and adopts §D15 and §D16, all recorded in
  §6.9;
- design review r7 (design-moua-r7 on `5925f3fe1`: 0 Critical, 7 Important, 11 Minor) and the
  lead's rulings on it (§L0R, §M7, the tightened §D15), recorded in §6.10. Revision 7.6 is one
  commit on top of `5925f3fe1`.

**The lead's rulings file.** The rulings shared by zhcn, moua, 1oxa, 23mk and jehw/uwlx are in
one file, `lead-rulings-2026-09-26.md` (sections §B, §B.1 (superseded), §B.2, §R, §RING, §E,
§E.1, §E.2, §L0R, §M7, §REC, §T, §L6, §GA, §FM, §STRICT, §D15, §D16, §Z52). This document
cites it as "rulings §X". **Where this document paraphrases a ruling
and differs from the file, the file wins.**

Revisions cited:
- **Current master is `3d9414c8c`, which contains jehw and u1bb** (jehw landed). Revision 7.6
  re-pins to it the text that r7 checked: the header, §2.4.2's L0 block, step (s), step (a),
  the §RING removal sites, and every line new in 7.6. Those cite `3d9414c8c` explicitly.
- Older citations keep the revision they name. `master` without a sha means `401ff76cc`, the
  base of `task/moua`.
- `jehw` = `task/jehw` HEAD `c41fed119` (its parent `2ad2e0f0e` included). Revisions 3 and 4 cited
  `4d41db5c8`, which r3 found superseded (§6.5 C2). Where `4d41db5c8` is still meant, it is named.
- `u1bb` = `task/u1bb` HEAD `ddee53ed5`. r3 read `f01b3e86c`; every u1bb line below was
  re-checked at `ddee53ed5`.
- jehw and u1bb are both in master `3d9414c8c`. A `c41fed119` (jehw) line is not a `3d9414c8c`
  line: jehw merged master before landing, so its lines moved.
  `fe6c` = `fe6c9356d`, the revision the A2 logs were taken on.
- Lines cited as `11faace69` (the record-mode park, `common.hpp`'s exec-graph slots,
  `llama-context.cpp`'s teardown paths) were read at current master `11faace69`.

Every file:line below names its revision.

## 0. Summary

- **Root cause of A2** (B50, PCT=60, Mistral Q4_0, `-c 32768`): the TLSF allocator that KV and
  WEIGHT share in single-chunk mode is fragmented by staging order (§1). The revision leaves
  this unchanged.
- **Design:** the shared zone gets a planned layout, and a block's side is set by its
  *lifetime class*, which is derived from its role unless the call site states it:
  - weights front-carve from offset 0;
  - optional (yieldable) tenants are staged after all load-time weight staging, at the weight
    frontier, and yield as a strict address-ordered prefix;
  - every context-lifetime and transient tenant sits on the context side, at the high end.
- **KV regions:** each context's device KV is a **region**, registered under
  `(ContextId, device)`.
  - The region is reserved idempotently at the runtime-context transaction.
  - A context's buffers find it through a llama-side, ContextId-keyed **region scope**.
  - The fit and the reservation run the same pure function over the same geometry,
    re-checked at the commit.
- **Context-side tenants are planned exactly, and their room is held (owner ruling
  2026-09-26; r3 C1, I2, I7; r4 I1-I4).** Every context-side tenant publishes a demand record:
  an **indexed list of slots**, one per allocation that can be live at once, under a scope
  (a context, or the device for the u1bb ring). The fit places those slots first, as **head
  slots**, and KV demotes around them. The commit carves each one as a **reserved slot**: an
  owner-first `CACHE_SUBALLOCATION` whose `mem_handle` the owner holds (the context's registry
  entry, or the device cache for the ring). An allocation of that cohort **claims its slot by
  index** as a slice lease, and slot reuse is **event-chained**: a claim is released at
  submission with its completion event, and the next claim of that index depends on it. So
  non-LIFO or event-deferred frees cannot fragment planned room, and nothing waits on the host.
  This is the one tenant protocol shared with zhcn (§2.3.2; agreed 2026-09-26 with
  zhcn's amendments A1-A4, §6.6). The ring,
  the compute buffer and fattn slot (zhcn), the per-op scratch (beni), the MXFP4 MoE TG caches
  and fattn workspaces (jzvq) and the recurrent state (moua) are all such tenants. There is no
  estimate, no floor and no per-row constant (§2.4.3).
- **Inside the transaction** (§2.4.2), the order is:
  1. match the idempotence key (a matched key takes the tenant-only path, which never
     re-fits KV and never yields);
  2. plan: reconcile the demand records, snapshot, fit;
  3. run the byte-accounting steps, whose demotions become the fit's `forced_host`;
  4. make every predictable refusal, so a refused candidate never yields;
  5. record **every** placement the fit made as a pending range, then yield (begun under L1,
     finished with L1 released);
  6. commit-carve, re-fitting only inside this call's own pending ranges, at exact offsets;
  7. run the MMID materialization;
  8. check the ring's generation and record the contribution, run the publication CAS, commit
     (install the ring's slots), and only then, after L1 is released, drop the slots the new
     plan superseded.

  In this path nothing held before the transaction is released before its publish, so a
  rollback never has to re-acquire room (r4 I5, I6). The tenant-only path's pre-L1 release is
  the one exception, and it is zhcn's step (i). A two-phase guard declared before L1 rolls
  back on any refusal: its L1 phase clears the pending ranges and discards the registry
  insert, and its second phase drops this call's new handles with no lock held (§2.4.2).
- **Error path:** a device-planned KV layer that does not land in its region, and a claim
  that exceeds its slot or names no slot, are plan violations. They are logged at ERROR, return
  an error status that reaches the graph (never a skipped op), and abort under
  `GGML_SYCL_STRICT_PLAN=1`, the one STRICT variable. There is no fallback of any kind: a
  context-side miss never reaches a raw allocation, and never takes unplanned room (§2.4.3).
- **Holes:** weight-side holes and buried optional tenants are usable region extents
  (regions are multi-extent). Only holes smaller than one slot are a limit (§2.9).
- **Optional copies held by recorded graphs.** On jehw HEAD a recorded graph leases the WOQ
  copies it reads for the graph's life, so jehw's predicate calls them not yieldable, and KV
  demotes while they hold VRAM. The fit plans with exactly that, and a WARN names the leased
  copies and the demoted layers. Getting the room back is **llama.cpp-423j** (owner impl-jehw)
  on §2.9's terms: a retire-on-request under L1, issued only when this transaction demoted a
  layer those copies would have covered, once per transaction; no context ever touches another
  context's graphs (r4 addendum). A demoted layer stays on the host for the context's life.
- **One publish descriptor (lead ruling 4).** llama sends one versioned, `struct_size`-headed
  descriptor whose layout moua owns: a KV-shape section and a recurrent-state section (moua),
  and a measured-tenant section (zhcn, carrying beni's and jzvq's demands too). One function
  computes a KV layer's bytes for the fit, the reservation and the claim (§2.4.4).
- **Landing (lead ruling, r4 I10):** jehw → moua L1-L3 → zhcn → beni producers → moua L4-L7 →
  beni site conversions. 23mk core lands after jehw, and llama.cpp-jzvq closes before moua L4.
  L1 is done and approved (`eab1ebeb6`, `9e0a708dc`, review fixes `97315421b` and
  `456650c01`). L2 is folded into L6, so no interim guard turns today's spills into refusals.
  Until beni converts a site, that site keeps today's placement and is not context-side (the
  transition rule, §2.4.3).

## 0.1 Acceptance conditions: the four principles

Every step of this design, and its reviews, are held to these. Each row names the mechanism
that enforces the principle and the check that shows it.

| | principle | enforced by | shown by |
|---|---|---|---|
| **P1** | The unified cache is the only allocator. There is no out-of-arena fallback for planned KV or for a planned context-side cohort. | KV regions and reserved slots are carved from the arena TLSF (§2.4). The tiered device branch issues no `unified_alloc` (§2.6). A region reservation never falls through to `unified_cache_malloc_device_tracked` (§2.8). A context-side claim allocates nothing: it is a slice of a reserved slot, and a miss returns an error status with no fallback (§2.4.3). | H7a/b/p; C1 counts **zero** KV-role EXT-ALLOC lines, and total EXT-ALLOC bytes no higher than the base run (§3.3). |
| **P2** | `mem_handle` / `alloc_owner` are the only ownership surfaces. | Each region **extent** and each **reserved slot** is an owner-first `CACHE_SUBALLOCATION` with its own `mem_handle`, held by its owner: the context's registry entry, or the device cache for the ring. The registry, each KV buffer, and each layer's view hold the extent handles they touch. A KV slot and a tenant's claim are `slice()`s, not allocations, so a reserved slot outlives its last claim by refcount and there is no orphaned state (r4 I3). Retained runs have no owner and are never handed out as ownership (§2.3.2). Final handle drops never happen under a listed lock (§2.4.2, §2.10). | G1 checks `arena_owns` and the handle refcounts. H7f checks that no raw pointer is stored as region state. H9 and H7t check where the drops run. |
| **P3** | Placement decides the executor. | Residency is still decided per layer by the planner. Host-tier layers keep executing on the CPU exactly as today, and only the capacity input changes (§2.5). | C2/C3 gates; the demotion WARN names the layers. |
| **P4** | Plan == reality. A KV region that does not fit is a planner bug: refused, never admitted and then spilled. A planned tenant always has its room. | One function, `kv_region_fit`, decides "fits" for the planner, the reservation (re-run at the commit), the ring and the other head slots, and the `-c` hint. One function, `kv_layer_tensor_bytes`, sizes a layer for the fit and for the claim, from the shape llama allocates (§2.4.4). Head slots are held as reserved slots from the commit until their owner ends, and each is claimed by index with event-chained reuse, so a claim in plan always has its slot (§2.3.2, §2.4.3). Llama's residency answers come from the registry (§2.5). Every second source of "fits" is deleted or rederived (§2.2), and the capacity primitives are allowlisted (H7d). The yield and the fit decide reclaimability with jehw's one predicate (§2.4.1). | H2 property test (fit ⇔ carve), H4 non-LIFO and deferred-release churn property test with the {100, 50} best-fit RED, H2/H3 shape cases (q8_0, `v_trans`, MTP), H5 reserve cases, H7d/q (one source), G1 forced mismatch ⇒ `[KV-PLAN-BUG]`. |

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
- A context-side call site does not allocate at all once its producer exists: it **claims**
  a reserved slot by `(owner, cohort, slot index)`, where the owner is its ContextId (read from
  the backend context's `execution_context_id`) or the device for the ring (§2.3.2, r4 I1). The
  class matters only for requests that still allocate: WEIGHT, OPTIONAL, the region carve,
  the reserved-slot carve, and the sites beni has not converted yet (§2.4.3's transition
  rule).
- The zone a request names does not decide its class. Many non-weight requests name
  `vram_zone_id::WEIGHT` only to stay out of the B50 tail zones (r1 C3). Examples: master
  `ggml-sycl.cpp:46120`, the per-op MMVQ scratch (*"Keep tiny MMVQ Q8 activation scratch out
  of arena tail zones on B50"*), and `ggml-sycl.cpp:97965`, the SOA-graph Q8_1.

| class | who | side | placement |
|---|---|---|---|
| `WEIGHT` | `alloc_role::WEIGHT`. That includes the runtime expert-cache fills, which are `alloc_role::WEIGHT` with `runtime_category::EXPERT_CACHE` (vram-pool.cpp:81, unified-cache.cpp:18484): on-demand expert rows, which are real weights. It also includes a `SYCL<n>` weight buffer that overflows RUNTIME into the shared zone (r2 m2, below). | weight | TLSF `allocate` (today's behaviour), tag `TAG_WEIGHT`; after the optional ladder exists, first a weight-side hole, then the gap front (§2.3.3) |
| `OPTIONAL` | optional layout copies (jehw's `optional_layout`) | weight frontier | `allocate_gap_front`, tag `TAG_OPTIONAL` |
| `KV_REGION` | the per-(ContextId, device) KV region, including each layer's persistent packed-K sidecar as a companion slot when the sidecar is enabled (§2.4.1) | context | the extents chosen by `kv_region_fit` (§2.4) |
| `CONTEXT` | u1bb ring KV-zone slots; zhcn's compute chunks and fattn slot; the recurrent state (§2.4.4, r4 I9); the context-lifetime cohorts beni routes here (the oneDNN activation half, `graph_input_stage`, the oneDNN Graph scratch) | context | a reserved slot, claimed by index (§2.3.2, §2.4.3) |
| `TRANSIENT` | every **non-WEIGHT role** that names `WEIGHT` or `KV`: COMPUTE/STAGING scratch (ggml-sycl.cpp:46120, :97965; mmvq.cpp:16863; common.hpp:6658), jzvq's MXFP4 MoE TG caches and fattn workspaces, persistent buffers (unified-kernel.cpp:4890, refused under an arena per 23mk Q5), and the forced-split fattn packed-K request (fattn.cpp:1640). `unified-cache.cpp:21455` and `scratch_pool` (`:20629`) are dead code that 23mk deletes | context | a reserved slot, claimed by index; the demand functions are beni's and jzvq's (§2.4.3) |

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
  - **Where their demand goes, and what protects it (r3 m5; r4 m4).** Under the lever, those
    two cohorts' demand records carry `lifetime = WEIGHT_SIDE_TRANSIENT`. Their slots are still
    head slots of the fit and are still carved as reserved slots at the commit, with the same
    claim, miss and plan-violation rules (§2.4.3). The only difference is where they are
    placed: the fit places them on the weight side, at the lowest free offsets of the weight
    side's free blocks in address order (today's pages), and records each chosen
    `{tlsf, offset, size}`. The commit carves exactly those offsets with `allocate_at`
    (§2.3.3), so fit == carve holds for them as for every other slot. Revision 5 said
    "first-fit through `allocate_excluding`", which TLSF's size-class search cannot make
    predictable. They are not part of the context side's accounting, and a weight allocation
    cannot take them, because a reserved slot is an allocated TLSF block.

Moving TRANSIENT off the weight side fixes three of r1's C3 consequences:
- **C3(i), burying:** it can no longer front-carve above the optional ladder and bury it.
- **C3(iii), churn:** per-op churn never touches the weight side. The generation counter it
  used to bump is gone anyway (§2.4.2).
- **C3(ii), missed allocations:** its slots are head slots of the fit, held as reserved slots
  from the commit (§2.4.3), so neither a region nor a later weight can take the room these
  requests need and push them out of the arena.

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

No fit, admission or gate here reads the driver's `free_memory`: a reading taken after any
destroy is stale-low, because the driver credits a destroy lazily (rulings §FM). The fit's one
source is the TLSF geometry.

A list of rows fails open to any site nobody listed, so H7d does not check rows. It
**forbids the capacity primitives themselves** outside an allowlist:
- the primitives are `unified_cache_kv_vram_available`, `zone_available(KV|WEIGHT)`,
  `zone_largest_free`, `largest_free_block`, `ggml_sycl_kv_capacity_live`,
  `kv_zone_snapshot`/`fit_capacity`, the new TLSF-wide `live_bytes()` (§2.3.2),
  `optional_layout_bytes`, `ggml_sycl_device_vram_budget_room` in any in-arena head-slot
  admission (r4 I7), and any per-row reserve constant (r3 I8). Revision 5's per-pair
  `reserved_slot_occupancy()` is no longer a capacity input (slots never move, §2.4.3), so it
  leaves this list and becomes the test accessor `reserved_slot_claims`;
- the allowlist holds their definitions, the no-arena branches (each marked with a comment
  H7d matches), log-only readers that feed no decision, and test code
  (`kv_zone_snapshot`/`fit_capacity` survive only as a test oracle, §4 seams).

Every other occurrence in `ggml-sycl.cpp`, `unified-cache.cpp` and `fattn.cpp` fails the gate.
Each allowlist entry and each forbidden primitive has a mutation witness.

| site (master unless marked) | today | becomes |
|---|---|---|
| transaction re-fit, `ggml-sycl.cpp` ~17752-17932 (jehw ~17830-17960) | bytes (fe6c); jehw: `fit_capacity` from `kv_zone_snapshot`, and `in.yieldable` from `unified_cache_optional_layout_bytes` (jehw `:17846`) | `kv_region_fit` on the geometry snapshot, re-run under the lock at reservation (§2.4) |
| tiered `kv_admission_mismatch(planned_device_bytes, kv_vram_cap)`, `ggml-sycl.cpp:38675` | live `unified_cache_kv_vram_available` at buffer-alloc time | **deleted for arena devices.** Once the region is reserved, the live available excludes it, so this check would refuse every device-planned buffer. The region claim (§2.6) replaces it. It stays on the no-arena budget path. |
| `configure_with_weights(device, n_layers, kv_vram_cap, kv_slice)`, `:38701` | the same live cap | arena devices: the cap is the region's slot-table capacity from the registry |
| `kv_device_budget` byte path, `:38815-38965` (per-layer `total_device + layer_size <= kv_device_budget` at `:38965`) | free VRAM minus the compute reserve | arena devices: residency comes from the region slot table and is not consulted here. No-arena: unchanged |
| u1bb ring contiguity, u1bb `ggml-sycl.cpp:17625-17626` (`zone_largest_free(KV)`) | the TLSF-wide largest free block, weight-side holes included | deleted: the ring's KV-zone part is head slots of the one fit (§2.4.3, lead ruling 2) |
| (r3 I8) u1bb ring compute reserve, `k_pp_moe_ring_compute_reserve_bytes_per_row` = 1 MiB/row (u1bb `:17462-17473`, fed at `:17629`) | an estimate of the compute buffer, by the constant's own comment *"An estimate, not a plan"* | **deleted by zhcn** (lead ruling 3): the compute buffer is zhcn's demand record, a head slot in the same fit. H7p/H7r check that no context-side admission keeps a per-row constant. |
| (r3) u1bb ring budget room, `budget_room_bytes = ggml_sycl_device_vram_budget_room(...)` (u1bb `:18362`, from `32136b16e`) | the plan's VRAM budget room | **deleted for in-arena head slots (r4 I7; lead ruling).** Revision 5 kept it as "a different fact", but it decided the same question the fit decides, "does this head slot fit", from a second source with no demotion lever: a head slot that passed the fit's geometry could fail the budget room where demoting one KV layer satisfies both, and u1bb ran it inside the ring admit, after the yield. Under an arena the budget authority already fixed the arena's size at load, so the geometry is the budget's physical form. The fit is the one source; nothing re-checks after the yield. Every carved head slot and KV extent is still **charged** to `vram_bytes`/`per_device_vram[dev]`, at exactly one site, the commit (§2.4.2 step 6), for accounting; no admission reads that charge for an in-arena slot. No-arena devices keep the check. |
| `-c` hint, `ggml_sycl_largest_fitting_n_ctx_live` | bytes | `kv_region_fit` |
| the demotion WARN's "free for KV" | bytes | the region capacity the fit computed |
| `largest_free_block()` anywhere in a fit decision | approximate (the head of the highest SL list) | never; fits read `frontier_walk` / `gap_below`, which are exact (L1, `9e0a708dc`) |
| (r2) MMID budget demotion: `ggml_sycl_try_demote_runtime_kv` on `BUDGET_EXCEEDED` / `GROWTH_BUDGET_EXCEEDED`, master `:17919-17960`, and its `-c` hint via `ggml_sycl_largest_fitting_n_ctx_live` | byte budget, after the re-fit, so it can demote layers the region already holds slots for | it keeps its constraint, which is a different fact (the device's total VRAM budget, including RUNTIME demand), but it runs **before the carve**. The commit re-fit starts from its residency and can only demote further (§2.4.2 step 3, before the yield). Its `-c` hint reads `kv_region_fit`. |
| (r2) `rebuild_runtime_per_device_vram` (`:17868`) and `moe_mmid_reaccount_replacement` (`:17881`) | byte accounting refusals | arena devices: their KV term is the fit's region bytes (Σ slot sizes), the same number the carve takes. Their refusals roll back through the transaction guard (§2.4.2). |
| (r2) u1bb ring admission: `kv_capacity_bytes = ggml_sycl_kv_capacity_live(...)` and `kv_bytes = ggml_sycl_device_kv_bytes_with_slack(...)` (u1bb `:18356-18357`; the functions at `:16958` and `:17504`); u1bb's re-fit capacity (`:18031`) | live bytes, counting the live ring's KV-zone bytes as free | the ring's KV-zone part is a demand record whose slots are head slots of `kv_region_fit` (lead ruling 2), and the ring is admitted into its reserved slots (§2.4.2 step 8 (c)). Both u1bb reads are deleted for arena devices, and the re-fit reads `kv_region_fit`. |
| (r2) `ggml_backend_sycl_kv_layer_on_device_from_dev` (master `:107332-107343`), llama's residency hook | the process-global plan snapshot | under an open region scope, the registry entry for this ContextId (§2.5, r2 N-I5). Outside a scope, as today. |
| (r2) `GGML_SYCL_BLOCK_EXEC_CANDIDATE_KV` (master `:38623-38649`, opt-in) | reassigns `kv_device` at buffer-alloc time | arena devices: ignored with one WARN per process, like `GGML_SYCL_VMEM_KV` (§2.6). If it is ever wanted there, the reassignment moves into the fit's input. |
| (r2 N-C1) the tiered slice's per-layer size, `kv_slice` (master `:38593-38612`), and `update_runtime_kv_sizes` at alloc time (`:38625`) | the actual buffer size divided evenly across its layers; the plan re-sized from the geometry | arena devices: the slot table, sized by `kv_layer_tensor_bytes` from the published shape (§2.4.4). The claim checks the buffer size against the table's sum. |

### 2.3 The context side (unified-cache, L4)

#### 2.3.1 Structure and locking

Each shared TLSF has one `context_side`: an address-ordered list of blocks
`{offset, size, state LIVE|RETAINED, class, slot}`. Its **anchor** is the lowest block in the
list. `slot` is set on a block that is a reserved slot: `{scope, owner, cohort, index}`. A
reserved slot is an ordinary LIVE, owner-first allocation whose handle its owner holds; a
claim into it is a slice lease and changes nothing here (§2.3.2, r4 I3). Revision 5's
RESERVED state, an allocated block with no owner, is gone.

Every read and write of `context_side`, of the pending ranges and of the TLSF runs under
`unified_cache::arena_allocator_group_mutex(zone)`. That includes the geometry snapshot
(r1 044k; L1 documents the contract at `9e0a708dc`). The snapshot copies out a POD
`shared_zone_geometry` and releases the lock. The pure fit then runs on the copy.

**What the transaction holds, on jehw HEAD (r3 C2(a)).** Revisions 3 and 4 described
`4d41db5c8`, where L1 was held for the whole body. That is no longer the code:
- The transaction is jehw `ggml-sycl.cpp:17620-18434`. It takes `g_tensor_inventory_mutex`
  (L1) as a `std::unique_lock` at `:17689`.
- The yield is split (`c41fed119`). `unified_cache_yield_optional_layouts_begin` runs under
  L1 (`:17896`). If it retired anything, the epoch is bumped and `lock.unlock()` runs
  (`:17900-17903`). `unified_cache_yield_optional_layouts_finish` then runs with L1 released
  (`:17904-17910`). It waits on the reader barrier (jehw `unified-cache.cpp:7865`), drops the
  withdrawn mirrors (`:7853`) and the deferred-free rows (`:7894`). Then `lock.lock()` runs, and
  the transaction returns `busy` if the published plan changed meanwhile (`:17911-17915`).
- So the wait and the final drops now run with no listed lock held. moua inherits a
  conforming yield and keeps it that way (§2.10).

**Consequences for moua:**
- **Reservations are serialised by L0, not by L1.** Between the plan (step 2) and the commit
  (step 6), L1 is released for the yield window, but no other transaction or load can run in
  it, because the process-global re-plan mutex (L0, rulings §E.2; §2.4.2) is held for the whole
  transaction. Runtime weight allocations (expert-cache fills, lazy MoE layout
  materialization), which take neither lock, can still allocate at any point, window or not,
  so every room a transaction plans to use is a pending range before any of them could see it
  free (item 1 below, and the tenant-only path's (0), rulings §M7 I-5(a)).
- **Three things keep A's plan valid across that window:**
  1. A's **pending ranges** are recorded under the group mutex **before** `lock.unlock()`
     (§2.4.2 step 5). They cover **every** extent and slot A's fit placed, not only the
     yielded run. Every other placement treats them as occupied: weight placement
     (`allocate_excluding` whenever any pending range exists on the TLSF, §2.3.3) and every
     other transaction's geometry snapshot, whose fit sees them as allocated blocks. So B
     cannot plan into A's room, and a weight cannot take it.
  2. The relock's plan check (jehw `:17911-17915`; master `3d9414c8c` `:18122`) returns `busy`
     on master when a commit intervened. Under L0 no other publish can intervene anywhere in the
     process, so here that return is `[CONTEXT-PLAN-BUG]` (rulings §E.2), kept as a check, and
     A's guard rolls back (§2.4.2).
  3. The commit re-fits **only inside A's own pending ranges** and carves at exact offsets
     (§2.4.2 step 6, r4 I8), so nothing that runs between A's re-fit and A's carve can take
     what the re-fit chose.
- **`kv_region_mutex_` is still needed, for a different reason than revision 4 gave.**
  Revision 4 said L1 serialised every reservation, so the mutex only covered the claim and
  teardown. The registry is read by the claim (inside `create_memory`), by the residency hook,
  and by teardown, none of which holds L1. The transaction writes it only at the commit and
  publish steps, under L1. The registry map is protected by `kv_region_mutex_`, and the
  geometry by the group mutex. Neither lock is L1's job.

**`kv_region_mutex_`** (new, one per device):
- **Rank L3**, beside `g_pending_kv_layer_masks_mutex`. So L1 → L3 is legal, and it sits
  below L4 `direct_stage_mutex_`. L4 adds it to the §12.5 table in the commit that introduces
  it. It needs **no same-rank tie-break**, because it is strictly leaf (below): no two
  `kv_region_mutex_` instances, and no other lock, are ever held with it. Revision 5 wrote
  "ascending device ID", which is not the contract's L3 tie-break `(ModelId, ContextId, …)`
  and was never needed (r4 m8).
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
  - erases **move the entry out, unlock, and only then drop it**: on rollback and on
    teardown. Teardown's drop also runs outside `g_execution_backend_binding_mutex` (§2.4.2,
    r3 I6).
- It is never held across the yield, the carve or the ring re-admission (§2.4.2).

**The group mutex** `arena_allocator_group_mutex(zone)` is L5. It is held only for the
geometry snapshot copy-out, the pending-range edits and the carve, and never across the yield
or a wait. A claim does not take it (§2.3.2).
- The existing order `cache locks -> group lock` (staging calls `zone_alloc` under the cache
  locks) is preserved.
- **The snapshot takes both, in that order (r3 C2(b)).** It takes the cache's
  `direct_stage_mutex_` and `rw_mutex_` shared (L4), exactly as jehw's
  `optional_layout_bytes` does (jehw `unified-cache.cpp:7677-7688`), then the group mutex. Under
  both it evaluates the yield predicate for every `TAG_OPTIONAL` block and copies out the TLSF
  geometry. So the fit's view of what is yieldable and its view of where it is are one state.
- No code holds the group mutex while calling the yield (r1 C2, §2.4.2).

**The execution-binding lock (r3 I6; r4 m8).** `g_execution_backend_binding_mutex` (master
`ggml-sycl.cpp:11931`) is missing from contract §12.5's table, which the contract itself counts
as a failing census item. Its classification must account for what nests under it: jehw's
`set_runtime_context_for_model` holds it while taking `backend_ctx->execution_state_mutex` and
calling `ggml_sycl::execution::global_registry().extract/attach_root`, which take the execution
registry's own lock (jehw `ggml-sycl.cpp:18648-18690`). Neither of those is in §12.5 either.
L7's §12.5 edit therefore classifies the chain as one census entry:
- `g_execution_backend_binding_mutex` at **L3** (a process-global binding registry, beside
  `g_backend_context_by_device_mutex`);
- `execution_state_mutex` and the execution registry's lock at **L4** (per-backend and
  registry metadata), in that order, so the nesting is L3 → L4 → L4 with the L4 tie-break
  "listed lock ordinal". Placing them at L3 would break the contract's rule against
  co-holding a global L3 lock with another L3 lock.
- The same edit must confirm, with a source gate, that no other L3 lock is held when the
  binding lock is taken.

This is census work for the contract owner to ratify. moua takes none of these locks and adds
no work under them; its only requirement is that the region extract never runs under the
binding lock (§2.4.2 "Teardown").

#### 2.3.2 Placement, and the tenant protocol

This subsection is the **normative specification of the reserved-slot and claim protocol**
for every context-side tenant, zhcn's included. zhcn's design cites it and does not restate it
(r4 I3; agreed with impl-zhcn 2026-09-26, with zhcn's amendments A1-A4; record in §6.6).

- **The KV region:** `carve_kv_region(extents)` carves exactly the extents the fit returned
  (§2.4), at their offsets. There is no search.
- **Reserved slots are held handles (r3 C1, I7; r4 I3).** A planned tenant's room is carved at
  the commit as one owner-first `CACHE_SUBALLOCATION` per demand slot, at the offset the fit
  chose, through `zone_alloc`'s mint-before-lock protocol (rule 2 below; §2.10 for the lock
  sequence). Its `mem_handle` is held by the slot's owner:
  - CONTEXT scope: the context's registry entry (§2.5), for the context's life;
  - DEVICE scope (the u1bb ring only, §2.7): the device cache's ring record.

  It is an allocated, registered TLSF block, so no weight allocation, no other owner and no
  other transaction can take it. Revision 5 held reserved room as unowned RESERVED blocks keyed
  by an integer owner and released by explicit calls, with an "orphaned" state for an occupied
  slot whose owner had ended. That was an ownership surface that is neither `mem_handle` nor
  `alloc_owner`, and it is withdrawn: here the parent handle outlives its claims by refcount,
  so an owner ending while a claim is live simply leaves the last claim to free the block.
- **Claim by index (r4 I1).** A producer's slot list is **indexed**, and every claim names its
  index. There is no best-fit search. The index is the producer's:
  - zhcn's compute chunks, device and host: `c` = the number of buffer objects of this
    `(context, buft)` currently alive (zhcn §3.4). gallocr frees the whole vbuffer before a
    realloc and allocates chunks in index order, so the count restarts at 0. That is the only
    reason the count is an index, so zhcn's claim scope asserts it (zhcn A1): within one ALLOC,
    per `(context, buft)`, the indexes run 0..n-1 in order and n is at most the number of
    measured chunks; a violation is `[CONTEXT-PLAN-BUG]`, with the plan-violation disposition
    of §2.4.3;
  - zhcn's fattn slot: index 0;
  - beni's and jzvq's per-op slabs: the role index their demand function assigns (an op that
    needs two slabs of one cohort at once has two indexes);
  - the recurrent state: one slot per device RS buffer, index 0 (§2.4.4);
  - the ring: its ring slot index.

  A claim `claim_slot(owner, cohort, index, size)` requires `size ≤ cap[index]` and returns
  `slot_handle.slice(0, size)` plus the event it must depend on (next item). Revision 5 took
  "the best-fitting vacant slot whose size is at least the request". r4's counterexample shows
  that is not exact once slots differ in size: slots {100, 50}; X=40 takes 50; Y=45 takes 100;
  X vacates; Z=90 arrives while Y is live, and the live set {45, 90} is in plan, but the only
  vacant slot is 50. A named index cannot make that mistake, because the producer, which knows
  which allocation is which, decided it. H4 carries that sequence as a RED against best fit.
- **Event-chained reuse (r4 I2; the owner's no-host-waits rule).** Lifetime and occupancy are
  separate facts:
  - **Lifetime** is the slice lease. The tenant retains it until its completion event, through
    `retain_handles_until_event` as today (master `ggml-sycl.cpp:46141-46152`), so the memory
    is never freed under queued work (P2).
  - **Occupancy** is logical. The tenant releases its claim **at submission**, with the event
    of the last kernel that uses the slot: `release_claim(owner, cohort, index, event)`. The slot
    is then vacant, and the next `claim_slot` of that index returns that event, which the
    claimant passes to `depends_on`. So a pipelined decode reuses the slot on the device's own
    ordering, with no host wait and no queue-depth slot count.
  - For a gallocr buffer, "release at submission" is the buffer object's free, done by a
    gallocr realloc or the scheduler's teardown, including the K-shift allocation and the
    post-update reserve inside `memory_update` (zhcn A4). Those frees always follow
    `ggml_backend_synchronize`, so the release event is the queue's last event and is already
    complete; zhcn still passes it to `depends_on`, so the rule is uniform and adds no wait.
    The free **takes** the queue's last event (for example `ext_oneapi_get_last_event`); it
    submits no barrier, so it publishes no retention newer than the step-(s) synchronize of
    §2.4.2 (r6 m-2).
    zhcn's fattn slot chains on its SDPA's own completion event.
  - It also removes 23mk's c-4vlt old+new overlap (the oneDNN scratch held old and new at once
    behind its event-deferred release): a resize within `cap[index]` reuses the same index.
  - A claim of an index that is still claimed (occupied, not released) is a plan violation
    (§2.4.3). It can only mean the producer listed fewer concurrent allocations than exist.
  - **Per execution mode (r5 I-C; lead ruling).** The rules above are the **eager** rule. SYCL
    graph recording is a third execution mode, and there a kernel's event is a recorded node's
    event, which is not a valid dependency for an eager submission, and every replay re-runs the
    recorded kernels on the baked slot address with no claim at all. So:
    - **In record mode a claim lasts for the life of the executable graph that recorded it.** A
      claim released while recording stores no event; it is handed to that executable graph,
      which holds it (and its slice) with its retained handles, because every replay reuses the
      slot. The slot stays claimed for as long as the graph can be replayed.
    - **It vacates when that graph is destroyed**, chained on an **eager** event taken after
      the last replay's submission: the event `ext_oneapi_graph` returned for that replay
      (refreshed at every replay submission), or a marker submitted after it on the same queue
      at the destruction. Never a graph-node event. The marker is a claim-state event only: it
      retains nothing, so it adds no retention after the step-(s) synchronize (r6 m-2).
    - **Master's unmarked record-mode release park is converted to this rule.** In
      `ggml_sycl_pool_leg::free`'s arena branch (master `11faace69` `ggml-sycl.cpp:43378-43381`),
      a free while `ggml_sycl_graph_recording_active()` pushes the handle into
      `graph_retained_handles` with no event and returns. Under an arena that park becomes the
      record-mode claim hand-off above, and the graph's destruction releases it with the eager
      event; H7 gates that no other record-mode release path parks a claim without it.
    - **The consequence: one index per recorded use (rulings §REC; the 23mk review
      addendum).** A recorded claim lasts for the graph's life, so an index held by a live
      executable graph is unavailable to eager claims, to other graphs' recordings, and to a
      second use of the same cohort inside the same recording (which eager execution serves by
      releasing and re-claiming one index). So:
      - recorded claims draw from the cohort's dedicated **record-mode index set**, with **one
        index per recorded use**: two uses of one cohort in one recording take two indexes,
        where a shared index would be claimed twice and report a false `[CONTEXT-PLAN-BUG]`;
      - an eager claim never shares an index with a live recording. A context keeps one
        executable graph on master's whole-graph regime (`common.hpp:6073`, "one exec_graph slot
        per context") and replays it by default, so an eager claim after a replay on a shared
        index would be a false `[CONTEXT-PLAN-BUG]` too;
      - the MoE dispatch/block graphs (`common.hpp:6137-6169`) can keep several graphs alive,
        and each live graph that records the cohort needs its own set.

      The producers' demand functions (beni, jzvq, zhcn's fattn slot) count one index per
      recorded use per live recording graph, and G1 prints the bytes that adds; §5 (m) records
      the cost and the alternative the reviewer offered (refresh the slot's release event at
      every replay submission, so replays and eager claims share one index), which is not
      chosen.
    - The eager rule is unchanged. Step (i)'s own-graph invalidation (§2.4.2) releases every
      record-mode claim of the context before its claimed-slot check.
- **Where the claim state lives.** Claims take neither the group mutex nor `kv_region_mutex_`,
  because they are on the dispatch path. The published slot table of a context is never mutated
  in place; each backend context of the llama context caches a `shared_ptr` to it at the
  publish, and the registry entry holds another. On the tenant-only path it is **taken at (i)(c)
  and dropped at (i)(e), and the new table is installed at the commit** (rulings §B step 8; r6
  I-3); the full path's commit installs the first one. Each slot carries an atomic claimed flag
  and its last release event under a per-slot leaf spin lock (`mem_handle_spin_lock` class).
  Nothing allocates and nothing logs under that spin lock: a claim copies the event out and
  unlocks, and a violation's `[CONTEXT-PLAN-BUG]` line is formatted and logged after the unlock
  (zhcn r3 item 1; lead ruling). The ring's slots keep u1bb's per-slot state
  (`g_pp_moe_onednn_scratch_slot_state[device]`, L5) as their claim state, converted to this
  protocol (§2.7).
- **Release.** A slot is released only by dropping its owner's handle:
  - at the publish that supersedes it (§2.4.2 step 8, r4 I5), never earlier in the full
    transaction;
  - in the tenant-only path's pre-L1 step (i), zhcn's form (§2.4.2);
  - at context teardown, when the registry entry drops (§2.4.2 "Teardown");
  - for the ring, when its last contributor's entry drops (§2.7).

  The block is freed when the last reference drops: the owner's, or a claim still retained by
  an in-flight event. There is no separate release-on-vacate site to forget (r4 I4(d)).
- **Charging: one charge, one uncharge (r4 I7; zhcn A2; lead ruling).** Each reserved slot and
  region extent is charged to `vram_bytes` and `per_device_vram[dev]` exactly once, at the
  step-6 carve, by the carved TLSF block size including alignment. It is uncharged exactly
  once, when that block is really freed: the last handle or retained slice dropping, after its
  event (the free rule above). A claim or a release never charges or uncharges, and no
  admission reads the charge for an in-arena slot; the fit's slot sizes are a pure input.
  u1bb's ring charging site is deleted for arena devices. A host-tier slot follows the same
  rule against the host inventory (§2.4.3). H7y checks that nothing else adds or subtracts a
  context-side cohort's bytes.
- **No `context_side_place` (r4 I3).** Revision 5 let a plan violation take unreserved
  context-side room when not under STRICT. zhcn's rule, "no fallback in scope", is the agreed
  one (and the lead's ruling, 23mk's TRANSIENT cohorts included), so that path and its
  placement function are deleted: a claim either lands in its slot or fails (§2.4.3).
- **Why slots, not a byte budget (r3 I7).** Revision 4 bounded the context side by planned
  peak live bytes and placed each request with a best fit over retained runs, else below the
  anchor. With event-deferred, non-LIFO frees, the span that policy occupies exceeds the peak
  live bytes: allocate A small, then B large below it; free A, which is interior and becomes a
  RETAINED run too small for C; C then goes below B, so the span is A+B+C while only B+C is
  live. An indexed slot per concurrently-live allocation makes the room exact.
- **Carving into a retained run (r1 M3).** This covers a region extent and a reserved slot that
  the fit put in a retained run. Each is one critical section under the group lock:
  1. `free(run)`; the run's neighbours are allocated, so it cannot coalesce onto the weight
     side;
  2. carve the new block at the fit's offset with `allocate_at` (§2.3.3);
  3. immediately re-carve each remainder (below and above the new block) with `allocate_at`,
     and record it RETAINED again.
  No free block ever exists on the context side outside this section, so the weight side's
  first-fit `allocate` can never see a context-side hole. L1 has no split-allocated primitive;
  this sequence is how L4 composes one, and H4 pins it.
- **Free:** freeing the anchor block returns it, plus every RETAINED run now reachable from
  the frontier, to the TLSF, where they coalesce into the gap. Freeing an interior block
  marks it RETAINED, and adjacent RETAINED runs merge. "Freeing" here means a region extent's
  or a reserved slot's last handle dropping.
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
     - The context-side placements (the M3 sequence, the region and slot carves) are new
       branches inside `zone_alloc`'s locked section. They are not a separate hand-minted path.
     - The re-retained remainder is registered to no one.
  3. **Rebuild, destroy and quiescence treat retained runs as not live, but capacity does not
     treat them as free (addendum NEW-1).** These are two separate numbers, and each has its
     own reader:
     - **`zone_available` is unchanged**: it is `tlsf_allocator::available()`, which counts
       retained runs and reserved slots as used. They sit off the free lists, and only the
       context side's own carve can use a retained run. A byte reader that counted them as free
       would be reading bytes it cannot place, which is A2's "bytes, not extents" defect.
     - **A new `live_bytes()`** = `used() − context_side.retained_bytes`. It is read only by the
       arena rebuild/destroy decision, the settle precondition, and the leak and quiescence
       checks. A reserved slot is live in this sense: it is a registered allocation owned by a
       live owner, and a reserved slot whose owner is gone is a leak those checks should see.
     - No fit decision reads either one. H7d forbids `zone_available` in any fit decision
       (§2.2), and `live_bytes` is on H7d's primitive list too.
     - L4 enumerates the `live_bytes` readers, and H4 asserts both numbers at every step.

#### 2.3.3 Weight-side placement after the optional pass

`zone_alloc(WEIGHT)` works as follows once an optional ladder is live on that TLSF:
1. first-fit into a weight-side hole: a free block that is **not** the gap block;
2. only then the gap front, with the burying WARN of §2.1.

**Whenever any pending range exists on a TLSF, every allocation on that TLSF goes through
`allocate_excluding`, ladder or not (r4 m1)**: WEIGHT on a shared TLSF, and any RUNTIME-zone
allocation on the RUNTIME TLSF, which holds pending ranges for the ring's RUNTIME half (§2.7).
Revision 5 said "once an optional ladder is live", which left a hole: with no ladder,
`zone_alloc(WEIGHT)` used plain TLSF `allocate`, which could front-carve into a pending range while
the transaction that recorded it held L1 (weights never take L1). A TLSF with no pending range keeps
plain `allocate`, so the common path pays nothing (addendum (b); §2.4.2 step 5).

This needs one new L1-level primitive, `allocate_excluding(excluded ranges, size, align,
tag)`. **It excludes ranges, not blocks (r3 m3).** Revision 4 took every free block that
intersects an excluded range off the free lists. The gap block always intersects the pending
frontier range, so that removed the whole gap, which contradicted step 5's "the gap front stays
available to weights". The semantics are:
- it returns a block `[off, off + size)` that is disjoint from every excluded range, or
  `SIZE_MAX`;
- a free block that an excluded range cuts is usable in its parts outside the ranges. The
  primitive **chooses an offset** inside one free block such that `[off, off + size)` lies
  outside every range, and carves exactly that with `allocate_at` semantics (r5 m-c). The
  remainders stay whole free blocks, coalesced with their neighbours as TLSF requires, and may
  straddle a range; the exclusion is enforced at allocation time, not by splitting free blocks.
  Revision 6's first draft split a free block at the range boundary and returned both parts,
  which leaves two physically adjacent free blocks, exactly what L1's `check_invariants`
  rejects ("free blocks %d and %d are adjacent", `tlsf-allocator.hpp:760`);
- step 1's "not the gap block" is expressed the same way, as the gap's whole extent passed as
  one more excluded range.

It lands in L4 as an L1 follow-up. H1 adds the cases: a gap whose top is a pending range still
serves a gap-front allocation below the range; a request that fits only across the range
misses; `check_invariants()` holds after every excluded allocation, with a RED on the
split-at-boundary form (two adjacent free blocks); and a free block straddling a range stays one
block.

**A second L1 follow-up, `allocate_at(offset, size, tag)` (r4 I8, m4).** It allocates exactly
`[offset, offset + size)` from the free block that contains it, splitting that block's front and
back remainders back onto the free lists (a remainder below `MIN_BLOCK_SIZE` is absorbed, by the
same rule `carve_gap` uses, which the fit mirrors). It returns `SIZE_MAX` if the range is not
wholly inside one free block. It is how every planned carve works: region extents, reserved
slots (context side and the B50 lever's weight-side ones), and the retained-run carve of
§2.3.2. The fit chooses offsets, and the carve takes exactly those offsets, so fit == carve is a
property of the primitive rather than of matching two search policies. H1 adds its cases:
exact interior, exact at either end, a range that crosses two blocks (refused), and a
sub-`MIN_BLOCK_SIZE` remainder.

#### 2.3.4 Reset, settle, and the dead KV reclaim

- **zone_settle / TLSF reset (r1 M2).** Any path that `reset()`s a shared TLSF clears that
  TLSF's `context_side` in the same critical section. Retained runs are free space, so
  dropping them is correct. The settle's existing precondition (no live registered
  allocations) already excludes live regions and reserved slots.
  - Reserved slots are registered allocations held by their owners (§2.3.2), so the
    precondition already sees every one of them, vacant or claimed, including the ring's.
    Revision 5's unregistered vacant slots needed a special rule; held handles do not.
  - **L4 adds two refusals (r4 m10):** a settle or reset is refused while any pending range
    exists on the TLSF (a transaction is between its step 5 and its commit or rollback), since
    the ranges would otherwise survive a reset that invalidates their offsets. The ring's
    DEVICE-scope slots are covered by the registered-allocation rule above.
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

A demand record names its zone, so its head slots are placed on that TLSF by the same
routing, and each slot is carved there (§2.4.3). H6 pins the routing.

C0 decides on hardware whether the B70 exercises this path. B50 at PCT≤100 is single-chunk
(§1.1).

### 2.4 The fit and the reservation

#### 2.4.1 `kv_region_fit` (pure, SYCL-free, `kv-runtime-demotion.{hpp,cpp}`, L3)

```
kv_region_fit(const shared_zone_geometry & g, const kv_region_request & r) -> kv_region_fit_result
```

**Inputs.** The geometry `g`, per TLSF of the device:
- `gap`;
- `optional_ladder`: the `frontier_walk` output in LIFO order, each tenant classified by jehw's
  predicate (below) as yieldable or not yieldable (two-way, r4 addendum D). The ladder is
  truncated at the first tenant that is not yieldable, because skipping it would leave a hole;
- **the predicate is jehw's, not a second one (r3 C2(b)).** Every classification of an optional
  tenant in the geometry, for the ladder and for the buried census, calls
  `unified_cache::optional_layout_yieldable_locked` (jehw `unified-cache.cpp:7660`), which decides
  through `weight_entry_reclaimable(..., OPTIONAL_LAYOUT_YIELD, ...)` (`:12945`, the mode at
  `:12954`). The snapshot evaluates it under the cache locks it needs, in the same section as the
  geometry copy (§2.3.1). The yield's retire re-checks the same predicate (jehw `:7804-7811`), so
  a tenant that stopped being yieldable between the plan and the yield is skipped, and the
  commit sees a shortfall (§2.4.2). H7q forbids any other reclaimability test on an optional
  tenant;
- **leased copies (r3 C2(c); r4 addendum).** A tenant leased beyond the cache's own mirror,
  by a recorded graph's sink or by an in-flight reader (the veto cannot tell them apart), is
  not yieldable and truncates the ladder, as on jehw HEAD. A copy that 423j's request has
  retired but whose last lease is still live is an allocated block like any other: not
  yieldable, and the strict prefix truncates at it (§2.9; H5). Revision 5's third class,
  "yieldable after a graph drop", is withdrawn with the graph drop it depended on;
- `retained_runs`;
- `reservations`: every reserved slot on the TLSF with its `{scope, owner, cohort, index,
  size}`. They are allocated blocks, and the fit never moves or frees one. The only use the fit
  makes of the list is **reuse in place**: a head slot of this request whose owner already holds
  the same `(cohort, index)` with **capacity at least the planned size** is placed on that slot,
  and nothing is carved for it (r5 I-B; lead ruling). A larger slot serves a smaller claim
  (§2.3.2, `size ≤ cap[index]`), so a ring shrink is never a carve and can never demote another
  context. A slot the new plan does not reuse (a growth) stays allocated in the fit's view,
  because it is released only after the publish (r4 I5): the overlap of old and new is real and
  priced here, and the layers it demotes are labelled `ring-growth` (Output, below);
- `weight_holes`: free blocks that are neither the gap nor part of the frontier walk. They
  are **usable as region extents** (audit I5, below). They were reported-only in revision 1;
- `buried_optional`: lease-free `TAG_OPTIONAL` blocks outside the frontier walk, i.e. below a
  weight that buried them, each with the free run their release would form. This needs a
  whole-TLSF block census, a second L4 read primitive beside `frontier_walk`. Like
  `frontier_walk` (L1's 044k contract, `9e0a708dc`), it runs **only under the group mutex**,
  inside the snapshot copy-out (addendum (d));
- `pending_ranges`: other transactions' pending ranges, which the fit treats as allocated
  (§2.3.1). At the commit re-fit, this call's own ranges are passed separately as `own_ranges`,
  and the re-fit may place **only inside them** (§2.4.2 step 6, r4 I8);
- `self_extents`: the extents and slots this call carved earlier in the **same** transaction,
  on an earlier device of a multi-device commit (§2.4.2 step 6). A same-key republish never
  re-fits KV (step 1), so its region is never a self extent.

The request `r`:
- the slot sizes, `kv_layer_alloc_bytes(kv_layer_tensor_bytes(...))` over the published KV
  shape (§2.4.4), for the layers the shape says hold KV. They are grouped full-attention
  first, then SWA, each group in layer order;
- **the packed-K sidecar as a companion slot (23mk, agreed; lead ruling).** When the persistent
  sidecar is enabled for the context (a flag in the KV-shape section, §2.4.4), each layer's
  slot is `align(kv bytes) + align(sidecar bytes)`: the sidecar is a slice of the same slot,
  at `sidecar_offset = align(kv bytes)`, so it is placed, demoted and freed with its layer, and
  is never a separate record. The sidecar bytes come from 23mk's function over
  `kv_layer_cells` and `n_stream`. The forced-split packed-K stays a TRANSIENT record of its
  own (it is per-dispatch, not per-layer);
- `forced_host`: layers an earlier step of the same transaction already demoted (the MMID
  budget demotion, §2.4.2 step 3). The fit may demote further but never promotes them;
- **the head slots (r3 C1, I5; lead ruling 2; r4 I4, I9).** The indexed slots of every demand
  record this transaction (re)plans: the context's CONTEXT-scope records (zhcn's tenants,
  beni's and jzvq's cohorts, the recurrent state), and the device's ring record, sized as the
  max over the live contributors including this context (§2.7). Each head slot names its zone,
  cohort and index. Reuse in place follows the `reservations` rule above. There is no MODEL
  scope (r4 m12: it had no producer);
- **The byte budget is not a second fit input (r4 I7, lead ruling).** Under an arena the
  budget authority fixed the arena's size at load, so the geometry is the budget's physical
  form, and a head slot fits exactly when the geometry says so. The charge to `vram_bytes` is
  accounting only (§2.2).

**Output.**
- `fits`;
- the per-layer residency (device or host) after the demotion loop;
- the strict LIFO `yield_prefix` count per TLSF;
- the chosen `extents` (an ordered list of `{tlsf, offset, size}`);
- each device layer's `(extent, slot_offset)`;
- each head slot's placement: a reused reserved slot, or a new `{tlsf, offset, size}`;
- the superseded slots, which the publish releases (§2.4.2 step 8);
- **`free_after_full_kv`, per TLSF (zhcn GA; r4 I10(e)).** A signed byte count: the room the
  fit can place into on that TLSF after the head slots, minus the bytes of every KV slot
  (companion sidecars included) with every KV layer device-resident. A negative value is the
  deficit the demotion loop covers; zhcn scores GA against it and prints the `-ub` WARN from it;
- **the demotion cause, per demoted layer**, tested in this order:
  - `ring-growth` when the layer would have stayed on the device with the superseded ring
    slots counted free (r5 I-B; lead ruling). The old and new ring coexist until the publish, so
    the growth's overlap is a real, priced cost; the WARN names these layers with the old and
    new ring bytes. Revision 6's first draft labelled them `capacity`;
  - `head_slot` when the layer would have stayed on the device with this request's head slots
    removed (the explicit `-ub` case, where "KV demotes, with a WARN" is the ruling);
  - else `capacity`.
  The WARN names each cause's layers separately. A demoted layer stays on the host for the
  context's life (§2.9 item 5), including a `ring-growth` one whose room is freed milliseconds
  later; no release-before-carve scheme is added (lead ruling);
- on failure, the refusal names the head slots that did not fit, with each TLSF's free space
  (zhcn's agreed tenants-alone message).

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

  **Head slots are placed first, then KV slots in slot-table order** (lead ruling 2). A head
  slot is mandatory: the demotion loop only moves KV layers, so if a head slot cannot be
  placed even with every KV layer on the host, the fit fails and the transaction refuses,
  naming the tenant. Otherwise KV demotes until both fit. Each slot goes into the cheapest tier
  that still has room. Within a tier, best fit decides. A slot is never split. The tiers,
  cheapest first:
  1. **RETAINED runs** on the context side: no cost.
  2. **The frontier gap, with no yield**: no cost, and segregation is kept. Its capacity for a
     KV slot is what the head slots left, because they were placed first. There is no reserve
     post-check any more, so a greedy pack cannot fill the gap with KV and then fail a reserve
     that a tier-3 hole would have satisfied at no cost (r3 m1).
  3. **Free weight-side holes**: no layout is lost, but the context's KV pins a weight-side
     hole for its lifetime (§2.9 (c)). Hence tier 3, after the gap.
  4. **Yields, ranked by optional-layout bytes lost per slot gained.** The candidates are the
     next strict-prefix step at the frontier (the smallest `k` more ladder entries that gains a
     slot) and each buried-tenant run (only the tenants that intersect the smallest carved
     range). The ranking is lexicographic: layout bytes lost per slot gained, then the
     frontier before a buried run, which keeps segregation. Repeat until the slots fit or no
     candidate is left. Then the demotion loop takes over.

  So a buried tenant is released only when tiers 1-3 cannot hold the slot, and only when it
  costs less layout per slot than growing the frontier prefix. This is the ordering that the
  owner-visible line on llama.cpp-moua states (§2.9).
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
- **Demotion order and granularity (agreed with zhcn).** The loop is unchanged in order and
  meaning, and this states it exactly, because zhcn pre-registers numbers against it:
  - the unit is **one whole layer, K and V together** (and its sidecar companion, when
    enabled). A layer's K and V are never split across tiers;
  - full-attention layers go first, **the highest layer index first** (the latest layer);
  - SWA layers go only after every full-attention layer is on the host, again highest index
    first (never shrink context: a SWA layer's slot is small, and moving it gains little);
  - each step asks `fits` over the extents; it never compares bytes;
  - the recurrent state and every other head slot are never demoted (they are mandatory;
    a head slot that does not fit with every KV layer on the host refuses the transaction).

  Worked prediction (zhcn GA, B50 GPT-OSS `-c 65536 -ub 1024`; r5 m-g: one number, from one
  function, cited by both designs; corrected by design-zhcn-r4, §6.8). **The fit reads actual
  allocations**, because its geometry is the live TLSF (lead ruling). The breakdown, from
  `kv_layer_bytes_for_kind` at `n_ubatch` = 1024 (f16 K and V, 8 KV heads × head dim 64, so
  2 KiB per cell per layer, K and V together):
  - room for KV and head slots, after the actual weights (11510.9 MiB): 1827.1 MiB;
  - full-attention KV: 12 layers × 65536 cells × 2 KiB = 12 × 128.0 = 1536.0 MiB;
  - SWA KV: 12 layers × 1280 cells × 2 KiB = 12 × 2.5 = 30.0 MiB, where 1280 =
    PAD(n_swa + n_ubatch, 256) = PAD(128 + 1024, 256);
  - room after full KV: 1827.1 − 1566.0 = 261.1 MiB;
  - head slots: 988.0 MiB (zhcn's 808.0 compute plus the ring's 180.0 context-side half at
    `ring_depth` = 1, §2.7);
  - `free_after_full_kv` = 261.1 − 988.0 = **−726.9 MiB** (on the planning weights, −752.2).

  Withdrawn: −714.9 (revisions 7 and 7.1, and zhcn rev 4.1), which took the SWA term at
  `-ub 512`'s PAD(128 + 512, 256) = 768 cells (18.0 MiB), and revision 6's −728.4, which no
  breakdown reproduces. Full-attention slots are 128 MiB; 5 × 128 = 640 is short by 86.9 MiB
  and 6 × 128 = 768 covers the deficit with 41.1 MiB to spare (⌈726.9 / 128⌉ = 6), so the fit
  demotes **exactly the 6 highest-indexed full-attention layers and no SWA layer**. Head-slot
  alignment rounding adds under 512 B per slot (sizes are 512 B multiples), far inside both
  margins (rulings §GA). **These figures are hand arithmetic, and the score does not use them
  (r6 I-7):** GA's pre-registered numbers are the output of H2's run of `kv_region_fit` on this
  geometry, recorded before the lead's run. The −714.9 error was a stale `-ub 512` SWA term that
  a run of the function would not have made. It demotes 7 only if weight-side holes smaller
  than one 128 MiB slot fragment the
  free room, in which case the §2.9 sub-slot WARN names them. **The 988.0 assumes
  `ring_depth` = 1 and no record-mode per-op index sets (r6 m-6):** if L4 finds the PP MoE
  oneDNN path reached while recording, the ring's depth counts recorded holders (§2.7), and GA
  is re-scored with the depth the plan line prints and the index-set bytes G1 prints.
- **The yield prefix is strict (r1 M7).**
  - jehw (since `4d41db5c8`, unchanged at `c41fed119`: `select_optional_layout_yield`, called at
    `unified-cache.cpp:7771`) already selects copies by address, from the top, but it *prunes*
    copies whose release would not help.
  - A pruned interior copy fences the freed ones above it and leaves a hole, so this design
    needs the strict prefix: the minimal count `k` such that releasing the top `k` ladder
    entries makes the frontier extent large enough.
  - This is a smaller change to jehw's selection than revision 1 framed: drop the pruning
    and keep the address order.
- **No reserve inequality (r3 C1).** Revision 4 required `frontier extent after the region ≥
  transient_reserve`. That constrained only the fit, and nothing held the room afterwards. The
  head slots replace it: their placements are part of `extents`, recorded as pending ranges
  before the yield, and carved as reserved slots at the commit (§2.4.2).

#### 2.4.2 `reserve_kv_region` and the transaction step (L4 + L6)

The runtime-context transaction takes L1 and releases it only inside the yield window
(§2.3.1). For a context `c` in publish mode, the region work sits inside it as follows. The
steps were renumbered in revision 6: revision 5's step 2, "release the ring if its demand
changed", is gone, because nothing held before the transaction is released before its publish
(r4 I5, I6).

**The re-plan transaction mutex: no `busy` from any re-plan contention (rulings §E, §E.1,
§E.2, §L0R).** llama-server treats a `llama_decode` return of −2 as fatal, so no return code
may mean "busy, retry" (rulings §E). The published plan is one process-global atomic
(`g_placement_publication`, master `3d9414c8c` `:2638`, read by
`ggml_sycl_global_plan_snapshot` at `:2676`), and every re-plan revalidates against it. So
every contention between re-plans, loads and unloads is replaced by serialization on one lock,
L0, and a failed revalidation under L0 is a bug, not a race.

- **What it is (rulings §E.2, §L0R).** One process-global `std::mutex`, `g_replan_txn_mutex`,
  ranked **L0**: taken before L1 and never under any L1-L5 lock. It is taken only through one
  RAII token type, `ggml_sycl_replan_token`, backed by a thread-local held flag:
  - an acquire on a thread that already holds L0 is a **nested hold** and does not lock;
  - an acquire on another thread blocks;
  - the outermost token unlocks.

  There is one mutex, so there is no device order. zhcn's gate 31 function-local token is this
  same type.
- **Where it is taken (rulings §L0R): at the top of every public entry point that can publish
  the plan or prepare a live update,** before the module admission guard and before any ticket.
  Located at master `3d9414c8c`, the publishers are the call sites of
  `ggml_sycl_publish_plan_locked`, `ggml_sycl_publish_prepared_plan_locked`,
  `lifecycle_replace_placement_plan` and `Registry::prepare_live_update`. Their public entry
  points:
  - `ggml_backend_sycl_set_runtime_context_for_model` (`:18840`), the wrapper around the
    transaction: its live-update ticket (`:18867`) and its publish (`:18955`) are under L0, and
    so is the inner transaction body (`:17852-18648`);
  - the probe wrapper (`:18761`) and the FA recheck (`:19041`);
  - `ggml_backend_sycl_activate_model_plan` (`:15228`, publish at `:15259`);
  - model unload, `ggml_backend_sycl_model_unloaded_token` (`:12283`), with its failure and
    exception republishes (`:12308`, `:12329` → `ggml_sycl_publish_restored_plan` `:12547`) and
    its latest-live republish (`:12599`), and the quarantine reaper's retry of it;
  - the model-load entry points (below);
  - the tenant-only path's llama-held scope, `ggml_backend_sycl_replan_scope`, which llama opens
    before (s) on the growth path. The probe and (ii) that run inside it are nested holds;
  - the teardown release proc, and every other ring mutator outside graph compute.
- **The load's span (m-9).** A load takes L0 at the top of each public backend load entry, not
  across the whole load: `ggml_backend_sycl_model_load_begin` (`:12728`), the plan computation
  (`ggml_backend_sycl_compute_placement_plan_early`, `:16473`, which runs
  `populate_inventory_globals`), `ggml_sycl_set_tensor_inventory_impl` (`:16491`), the weight
  preload (`ggml_sycl_preload_model_weights`, whose republish is `:33769`), and
  `ggml_backend_sycl_model_load_end` (`:13096`, publish at `:13205`). llama's loader code and
  the application's `progress_callback` therefore never run under L0, and the deadlock rule
  below needs no clause for callbacks into application code. Between two load entries another
  re-plan can run. That is safe because the load's own state is its bound candidate
  (`ggml_sycl_bound_load_candidate`, `:2681`), not the published plan, and its publish at
  `load_end` is under L0. Loads never nest inside a transaction, and a transaction never
  triggers a load.
- **The allowlist: decode's identity-preserving republishes (rulings §L0R).** Four sites
  republish the current snapshot from inside graph compute and do **not** take L0:
  `ggml_sycl_republish_current_plan()` (`:2996`) at `:5887` and `:5908` (the MoE
  secondary-queue setup, `:5852`), `:60125` (the lazy MoE layout materialization, `:60062`) and
  `:74882` (`ggml_sycl_mul_mat_id`). They are allowed only because they publish the
  same pointer they read, under `g_tensor_inventory_mutex`, so the plan identity cannot change.
  A gate (H7ai) proves it: the republish helper takes no snapshot argument, and a debug check
  compares the identity before and after. A change of identity there is `[CONTEXT-PLAN-BUG]`.
  The same helper's fifth caller, `:33769`, is in the weight preload, a load entry under L0.
- **Who never takes it (rulings §L0R; r7 m-7).** "Decode never takes L0" means **graph compute
  and dispatch never take it**. `llama_decode` can reach a re-plan legitimately through
  `sched_reserve` (the resync at `llama-context.cpp:1571`, and the FA recheck through
  `resolve_fused_ops` at `:1298`). That is a re-plan, and it takes L0 like any other. The
  accepted cost is that re-plans, loads and unloads on different devices and models serialize;
  they are rare (rulings §E.2).
- **The probe and the FA recheck validate against their own model's plan (rulings §L0R).** On
  master both compare the model token with the one global snapshot (`:18761-18767`,
  `:19041-19056`). With two models loaded, B's load or re-plan publishes B's plan between two of
  A's L0 holds, and A's probe then answers STALE_IDENTITY ("not the published model"), which
  llama turns into a refusal or a throw (`llama-context.cpp:1531`, `:1195-1200`). L0 serializes;
  it does not re-bind. So both read `lifecycle_select_placement_plan(model)`, the plan of the
  model they are called for, and validate against that. The FA recheck has a second defect of
  the same family: its headroom predicate reads **live** device free memory
  (`ggml_backend_sycl_get_device_memory()`, the pvjr comment at `:19058-19075`), a reading taken
  after this process's own destroys, which rulings §FM forbids for any sizing or gate. On an
  arena device it is a second source of "fits" (P4; §2.2), so it reads the fit's ledger (the
  registry's residency answer) instead. This was found while re-pinning the recheck and is
  reported to the lead with its path.
- **What changes.** Under L0 no other re-plan, release proc, load or unload runs concurrently,
  so every return below can fire only if a mutator skipped L0. Each is `[CONTEXT-PLAN-BUG]`
  (an abort under `GGML_SYCL_STRICT_PLAN=1`), never `busy` and never a retryable refusal:
  - the two ring returns: another transaction's RELEASING at step 2, and step 8 (a)'s
    generation mismatch;
  - jehw's five in the transaction body, at master `3d9414c8c`: `:17867` "stale identity",
    `:17881` the live-update lease, `:17886` "plan changed while acquiring the transaction
    lock", `:18122` the yield's relock, and `:18646` the lost publication CAS (a `refuse`, which
    is PLAN_REJECTED and fatal on the server);
  - the wrapper's own `BUSY` returns: module admission (`:18848-18849`), the ticket
    (`:18867-18870`), an open invocation or graph (`:18925-18929`), and `attach_root`
    (`:18934-18935`).

  The ring's own `busy` at `:18515` ("scratch ring claimed by an in-flight dispatch") is gone on
  arena devices (r7 m-8): the full path no longer releases the ring, and on the tenant-only path
  a claimed slot after (s) is step 7's occupancy `[CONTEXT-PLAN-BUG]` (rulings §B step 7). On a
  device with no arena, u1bb's ring re-plan and that return stay as they are, outside this
  design. llama's `BUSY` sleep-backoff loops (`llama-context.cpp:1154-1158`, `:1522-1526`) are
  deleted (rulings §L0R); the deletion is in zhcn's scope, and this design depends on it.
- **The pins.** RELEASING, `ring_plan_gen` and the step-2 copies and pins stay (rulings §RING)
  as the checked invariants. Pins come only from a transaction guard's step 2 (rulings §M7
  I-3). Under L0 no other guard is alive at a move-out, and (0) takes no pins, so the P
  snapshotted at the move-out is 0 on every path. (e) still bounds by the snapshotted P, and
  reports a nonzero P as `[CONTEXT-PLAN-BUG]` (r7 m-3).
- **Held across waits.** It may be held across the transaction's own step (s) synchronize and
  its unlocked yield or driver window (1oxa's create/map). That is not a GPU wait on another
  party's work.
- **The deadlock rule, and teardown's path.** A thread holding L0 never waits on anything that
  needs L0 on another thread.
  - Everything that runs under L0 is backend code, llama's re-plan sequence (`sched.reset()` and
    the backend calls inside `ggml_backend_sycl_replan_scope`), or the release proc. None of it
    runs application callbacks, destroys a context, or waits on another thread's llama call.
    Graph compute, which other threads may be in, never takes L0, and queue waits are GPU
    progress.
  - Nested holds on one thread are legal and do not lock: the growth path's probe and (ii) run
    inside llama's replan scope (r7 I-2).
  - Teardown takes L0 only in the release proc, which runs from `sycl_plan_guard`'s destructor
    (§2.4.2 "Teardown"). That destructor runs in `~llama_context`, or in the constructor's
    unwind after the transaction's backend call has returned (the `create_memory` refusal at
    `:869` throws after the transaction at `:810` has closed its scope). So the release proc is
    never reached on a thread that already holds L0. Its token is an **outermost-only** kind: a
    debug check aborts if the thread's held flag is already set when it enters. That is
    distinct from the transaction entries' legal nesting, and H9 has an arm for each.
- **Gates.** H7ai checks every publish and `prepare_live_update` call site: each is under a
  token taken at the top of its public entry point, or is one of the four allowlisted
  identity-preserving republishes; graph compute and dispatch take no token; the token is
  taken before L1 and never under L1-L5. H9 runs:
  - two contexts re-planning concurrently, on one device and on two devices, a re-plan racing a
    load on the other device, and a re-plan racing an unload;
  - A in a transaction, then B loads, then A's probe and FA recheck, with zero STALE_IDENTITY;
  - the growth path's nested holds (positive), and a release proc entered under a held token
    (negative, the debug abort).

  Each asserts zero `busy` returns and zero lost CASes.

**The transaction guard: two phases (r2 N-I3; r3 I4; r4 I5, m14).** L6 declares a
`kv_region_txn` guard **before** the transaction's L1 `std::unique_lock` (jehw `:17689`) and
before any pre-L1 work (the host-tier allocation, the tenant-only path's (s), (0) and (i)).
Reverse destruction order therefore runs the guard's destructor after the lock is released,
on every exit path. The guard owns everything this call has made and not yet published:
- the owner-first controls it pre-minted before L1 (§2.10), used or not;
- every extent handle and every new reserved-slot handle this call carved, and the host
  reservation it allocated before L1 at a first publish (§2.4.3);
- **copies of the ring's current slot handles**, taken in step 2's ring-lock section before the
  snapshot, each counted in the ring record's `pinned[slot]` (r5 I-A; r6 I-4, I-5), so the blocks
  the plan reuses cannot be freed underneath it;
- this call's pending ranges;
- the pending registry insert, this call's tentative ring contribution (step 8), and a
  `RELEASING` mark this call set (the tenant-only path's step (i));
- after a commit, **the superseded handles** (step 8; r5 m-a): device slots, the ring's old
  slots, and any mirror handles 423j's retire withdrew (§2.9), all dropped after L1 is
  released. Host slots are never superseded within the plan (§2.4.3).

It owns nothing that existed before the call. The ring's current slots and the context's
current tenant slots stay with their owners until step 8. So on any return other than a
committed publish (a `refuse()`, a `[CONTEXT-PLAN-BUG]` return, or an exception, from any step;
there is no `busy()` under L0), the destructor
rolls back in two phases, and never has to re-acquire room:
1. **Metadata, with one leaf-class lock at a time.** Clear this call's pending ranges on each
   TLSF, under that TLSF's group mutex (L5, taken alone). Discard the pending registry insert
   (local state). Then, under the ring record's lock (L5, taken alone, never nested with the
   group mutex): remove this call's tentative contribution if step 8 recorded one, clear
   `RELEASING` if this call set it, and bump `ring_plan_gen` for either (r5 I-A(d)). This phase
   needs no L1: revision 5 re-took L1 only to restore the ring, and there is no restore any
   more.
2. **With no lock held.** Drop this call's new handles, the ring-handle copies, the unused
   pre-minted controls, and on a commit the superseded handles, so each `zone_free` runs
   lock-clean. Then take the ring record's lock alone and decrement `pinned[slot]` for each copy
   dropped. The order is drop first, then decrement, so a slot's `use_count()` never exceeds 1
   plus the pins a check has snapshotted (§2.4.2 (i), "The ring"). Phase 2 runs on every exit,
   the commit included; phase 1 runs on every exit that did not publish.

Every old slot is untouched, so after a refusal at the MMID step or the CAS the ring is exactly
where it was, in its original slots, and a claim in flight on it is unaffected (r4 I5; H9).

The destructor is `noexcept`. `std::mutex::lock` can throw `std::system_error`; the destructor
catches it and aborts with a named message (`kv_region_txn rollback could not take the group
mutex`), because a rollback that cannot clear its ranges cannot be completed later and must not
propagate out of a destructor (r4 m14).

Yields already performed are **not** undone. They released optional tenants, which costs
prompt-processing performance, never correctness, and the yield WARN names them.

**Steps.**

1. **Idempotence (r1 C1; r2 N-I9, m10; r3 I3, m11).**
   - For each device `d`, look up `(c, d)`: copy it out under `kv_region_mutex_`, then unlock.
   - **The key is built from request inputs only, never from fit outputs.** It is
     `(n_ctx, n_seq_max, kv_unified, swa_full, the KV-shape section's digest, the recurrent
     section's digest)`. The KV digest covers `type_k`, `type_v`, `v_trans`, `no_alloc`, the
     per-layer descriptors, the sidecar-enabled flag and `n_stream` (§2.4.4). It holds no slot
     table and no residency.
   - **`n_ubatch` is excluded.** SWA slot bytes depend on it (the SWA branch of
     `kv_layer_bytes_for_kind`, master `unified-cache.hpp:580`), but llama allocates the KV
     once, at `create_memory` (master `llama-context.cpp:869`), with the `n_ubatch` of the
     constructor's first publish (`:810`). The auto micro-batch ladder republishes with other
     values after the KV exists (`:1039` → `:1571`, `:1936`). A region's slot sizes are
     therefore **frozen at the reservation's `n_ubatch`**, which is the value llama used.
   - **A matched key takes the tenant-only path (below), never steps 2-8 (r3 I3).** A matched
     key whose tenant key is also equal is an OK no-op that returns the published residency
     (r5 m-j), which is what every later backend's run of one publish finds in the split case.
   - **A matched ContextId with a different key is refused (r3 m11)**, as `context republished a
     different KV shape`, aborting under `GGML_SYCL_STRICT_PLAN=1`. The key is frozen at the
     first publish, and every input to it is fixed for a context's life.
   - This is today's `admitted_kv` rule, *"a same-shape republish by an admitted context keeps
     the published residency"* (master `ggml-sycl.cpp:17738-17747`), made physical.
   - Step 1 takes only `kv_region_mutex_`, so on the full path it runs before L1, and the
     host-tier allocation that follows it ("before step 2") is still before L1 (r5 I-I(3)).
   - It makes the split case safe. Every backend's run of the transaction re-fits every device
     (*"each re-fits, because admission is per backend"*). The first run to reach `(c, d)`
     reserves and publishes the entry, and every later one matches it.
2. **Plan.**
   - **Reconcile the demand records (§2.4.3).** This context's CONTEXT-scope records come from
     the descriptor. The ring's device record is the max over its live contributors' rings,
     with this context contributing its ring at the reservation's `n_ubatch` (§2.7).
   - **The ring-lock section comes first, before the snapshot (r5 I-A; r6 I-5; lead ruling).**
     Under the ring record's lock: read RELEASING (the rule below), copy `ring_plan_gen`,
     and copy the record's current slots (control identity, offset, size, zone) and their
     handles into the guard, incrementing `pinned[slot]` for each copy (r6 I-4). The fit takes
     the ring's current slots **from this copy, never from TLSF tags**, so every slot it plans as
     reuse in place is one the guard holds. `ring_plan_gen` is bumped by **every** ring-record
     mutation: a contribution recorded or removed, a publish that swaps slots, RELEASING set or
     cleared, and the teardown release proc's last-contributor drop. So a mutation after this
     section changes the generation, and step 8 (a) sees it, while the copies keep the planned
     blocks allocated. Revision 7.1 copied after the snapshot: a release proc's step 4 landing
     between the two let B plan reuse of blocks the record no longer held, copy nothing, match
     the new generation, and publish a ring nothing held (H9).
   - Snapshot the geometry (§2.3.1: cache locks, then the group mutex, then release; the
     RUNTIME TLSF is included for the ring's RUNTIME half, §2.7), and run `kv_region_fit`. That
     yields the residency, the yield prefix, the extents, the head-slot placements (the ring's
     RUNTIME/KV-zone split among them), the superseded slots and `free_after_full_kv`.
   - If a head slot cannot be placed even with every KV layer on the host, the transaction
     refuses, naming the tenant.
   - **RELEASING owned by another ContextId is `[CONTEXT-PLAN-BUG]` (rulings §E.1, §E.2).**
     Under the re-plan mutex (L0) no other transaction or release proc can be mid-release, so a
     RELEASING mark this call does not own means a mutator skipped the mutex. It refuses with
     the BUG line, before anything is recorded or snapshotted, and aborts under STRICT; it is
     never `busy`. **RELEASING has an owner (r5 I-A, sharpened):** it is `{owner ContextId,
     ring_plan_gen}`, and the owner's own (ii) passes this step and step 8 (a). The owner's
     guard clears it on every exit: publish, refusal or exception.
3. **The byte-accounting steps, in their existing order, before any yield (r2 N-I4; r3 m4).**
   These are `rebuild_runtime_per_device_vram`, `moe_mmid_reaccount_replacement`, and the MMID
   re-plan with its `ggml_sycl_try_demote_runtime_kv` fallback (jehw `:18056`).
   - They run on step 2's residency. Under an arena, their KV term is the fit's region bytes.
     **The in-arena head-slot bytes are excluded from the BUDGET_EXCEEDED demotion input**
     (jehw `:18118-18127`, which routes to `ggml_sycl_try_demote_runtime_kv` at `:18047-18056`;
     r5 m-b; lead ruling): the arena's own TLSF is the one source for in-arena bytes, and the
     fit already placed them. Revision 6's first draft fed them in from the charge function,
     which made the byte budget a second fit input, the shape r4 I7 removed. They are still
     charged at the commit, for accounting only (§2.3.2 "Charging"). The MMID re-plan's budget
     is RUNTIME growth, a different fact from the shared zone's geometry, so it keeps its
     demotion.
   - A demotion here only removes device layers. It becomes the fit's `forced_host`, and the
     fit is re-run (it is pure, so this has no side effect). The yield is therefore sized for
     the final residency.
4. **Every predictable refusal, before any yield.** After this step, only a runtime shortfall
   or a lost race can change the outcome.
   - The non-FA scratch check (jehw `:18187-18246`) and the publication-ID check (jehw `:18351`)
     move here. The non-FA check, with its shape recording and its SCRATCH raise, runs only for
     a context whose tenants are not planned (`!tenants_planned`, a flag only zhcn's MEASURE
     sets). Under a plan the KQ chunk and the `context-nonfa-stage` slots are already head slots
     of step 2's fit, so the check would count them twice (zhcn §3.8 row 21, its gate 32).
   - The ring has no separate "does not fit" refusal (u1bb `:18388`) and no budget-room check
     (u1bb `:18362`, deleted, r4 I7): its slots are head slots of step 2's fit.
   - Nothing after the yield can fail for a runtime reason (rulings §M7 I-5, §E.2). The MMID
     workspace is planned and carved at step 6, so step 7 allocates nothing, and under L0 the
     CAS cannot lose. What is left is a shortfall, which step 6's re-fit demotes, and
     `[CONTEXT-PLAN-BUG]`.
   - **Probe mode ends here.** A probe runs steps 1-4 with no side effects: no pending range, no
     yield, no carve, and no ring-lock copies or pins (rulings §M7 I-3). On a matched key it
     runs the tenant-only path's fit only, with the context's own tenant slots, and a sole
     contributor's old ring when its ring must grow, counted free by arithmetic (zhcn's step
     (0); §2.4.2 "(0) Probe").
   - **Probes no longer see another transaction's pending ranges (r4 m13; rulings §E.2).** A
     probe holds L0, so no other transaction can be mid-yield anywhere while the probe runs. The
     spurious refusal that r4 m13 accepted is now unreachable.
5. **Record the pending ranges, then yield (addendum (b); r3 C2(a); r4 I8).**
   - **Before the yield, and before `lock.unlock()`**, record **every** placement the fit made
     (all KV extents and all new head-slot placements, not only the yielded run) as
     `pending_ranges(c, d)` on each TLSF, under the group mutex.
   - Until step 6 carves them, or the guard's first phase clears them, every other placement
     treats them as occupied: `zone_alloc(WEIGHT)` through `allocate_excluding` (§2.3.3), and
     every other transaction's snapshot (§2.3.1). This holds whether or not a yield happens:
     jehw unlocks only if something was retired (jehw `:17899-17903`), but weights never take
     L1, so the ranges are needed with L1 held too.
   - A WEIGHT request that fits **only** inside a pending range gets the zone miss it would get
     a moment later, once the range is carved. That is the post-commit state, so this adds no
     new behaviour class.
   - Then the yield, as jehw HEAD runs it (jehw `:17873-17915`), with one change: the picks are
     the fit's strict prefix plus the buried tenants it chose, passed as an explicit pick list
     through **llama.cpp-uwlx** (owner impl-jehw, after jehw lands; proposed by impl-1oxa):
     `yield_optional_layouts_begin` takes the picks, one group per pick for moua, identified by
     key plus entry generation (no raw pointer, no caller-supplied size), re-checks each under
     L1 with the one predicate, and retires all-or-none per group. A skipped group is a
     shortfall, which step 6's re-fit demotes. On arena devices `select_optional_layout_yield`
     and `kv_zone_snapshot` stop being the pick source, and jehw's `kv_layers_allocatable` hold
     stays until L4's fit is exact. So:
     - begin under L1: retire the picks, and submit the reader barrier;
     - if anything was retired, bump the optional-layout epoch
       (`ggml_sycl_optional_layouts_retired()`, jehw `:17610-17612`) and `lock.unlock()`;
     - finish with L1 released: wait on the barrier, drop the withdrawn mirrors and the freed
       rows;
     - `lock.lock()`. A changed plan here (jehw's relock, master `3d9414c8c` `:18122`) is
       `[CONTEXT-PLAN-BUG]` under L0 (rulings §E.2), never `busy`; the guard's first phase then
       clears the pending ranges.
   - **A yielded copy is never lazily re-staged.** Dispatch reads the primary's materialized
     layout (P3, "the loaded layout is the answer"), and H7h gates it.
6. **Commit the carve (r2 N-I7; r4 I8, m3).**
   - **Re-fit only inside this call's own pending ranges.** Re-snapshot, and re-run
     `kv_region_fit` with `forced_host` and `own_ranges` = this call's ranges, as the only room
     it may place into. Nothing else can have entered them: weights exclude them, and every
     other transaction's fit treats them as allocated. So the only change the re-fit can see is
     a **shortfall**: a pick that was not retired (jehw's retire re-check, `:7804-7811`) or whose
     frees stayed queued still occupies part of a range, or a weight allocation landed in the
     planned room between step 2's snapshot and step 5's recording (weights never take L1, and
     the ranges do not exist yet at that point). All three look the same to the re-fit: part of
     a range is not free.
   - On a shortfall the re-fit **demotes**, inside the ranges, and commits that residency. It
     never asks for more yield, so there is no commit loop and no "geometry kept moving"
     refusal. Demotion always terminates, because all layers on the host always fit, and the
     head slots were placed in the ranges first.
   - **Why this, and not one critical section (r4 I8, lead ruling "less lock surface").** The
     other option was to run re-snapshot, fit and carve as one section under the cache locks and
     the group mutex. That puts the fit under L5, co-held with L4, and blocks every allocation
     on the device, including other models' weight staging, for the fit's duration. Restricting
     the re-fit to exclusive ranges needs no new co-hold at all: the snapshot and the carve each
     take the group mutex briefly, as in revision 5, and the ranges make the gap between them
     harmless. The cost is that a shortfall cannot opportunistically use room outside the
     ranges, which errs toward demotion, never toward a miss.
   - **The carve.** Under each TLSF's group mutex, in one section: `allocate_at` each extent and
     each new head slot at the re-fit's offsets, register each with one of the pre-minted
     controls (§2.10), and clear this call's ranges on that TLSF. The handles go into the guard.
     Across devices, carves run in device order, and a later device's refusal rolls back the
     earlier ones through the guard (`self_extents` covers the earlier devices' carves in the
     later devices' fits).
   - **Re-run the accounting on the committed residency (r4 m3).** If the re-fit demoted,
     `rebuild_runtime_per_device_vram` and the charge run again on the final residency before
     step 7. They can only decrease, so they cannot refuse. The published `per_device_vram` then
     matches what was carved.
   - The carve runs under L1, because the transaction body does. That is allocation work under
     a registry lock; §2.10 states the exact sequence and classifies it (lead ruling: a
     classified exception pending ratification; r4 m7).
7. **The MMID workspace, from planned room only (rulings §M7 I-5(b)).** On master
   (`3d9414c8c` `:18612-18620`) `ggml_sycl_materialize_published_mmid_workspaces` allocates the
   new plan's MMID pools at runtime, after the yield. That makes step 7 a runtime refusal that
   another context's RUNTIME or host allocation can cause, mapped to PLAN_REJECTED. Instead:
   - step 2's fit places the new plan's MMID device pool (`moe_mmid_device_pool_bytes`) as a
     head slot on the RUNTIME TLSF, the way it places the ring's RUNTIME half (§2.7), and step 6
     carves it with the other head slots;
   - its host pool (`moe_mmid_host_pool_bytes`) is a view of a held host carve. The MMID
     workspaces belong to the model (`unified_cache_materialize_moe_mmid_workspaces` is keyed
     by the model token), not to a context, so the carve is the model's: at the first
     materialization, the wrapper's bind at context creation (`:18980`, where host growth is
     still allowed), the MMID registry entry allocates and holds host room for the plan's
     largest MMID host pool (the workspace at the plan's top `n_ubatch` rung), the same rule
     §D15 sets for the tenants' host room (rulings §D15). Later workspace plans take views of
     it, and a host pool beyond it is a candidate refusal by arithmetic at step 4, before
     anything is released. The carve is released with the model's MMID entry, at unload;
   - step 7 materializes into those two carves and allocates nothing. An allocation there, or a
     carve too small, is `[CONTEXT-PLAN-BUG]`, never a refusal.

   A stable MMID plan (`ggml_sycl_same_mmid_workspace_plan`) plans nothing new. Any refusal
   still rolls back through the guard, which leaves the old ring and the old slots untouched.
8. **Ring check, publish, then release what the new plan superseded (r4 I5; r5 I-A, m-a).**
   - **(a) The ring check, before the CAS (lead ruling).** Under the ring record's lock:
     `ring_plan_gen` must equal the value step 2 copied, and `RELEASING` must be clear or owned
     by this call. Under the re-plan mutex a mismatch means a mutator skipped it, so it is
     `[CONTEXT-PLAN-BUG]` (a STRICT abort), never `busy`; the guard rolls back (rulings §E.1,
     §E.2). On a match, record this context's contribution now, bump the generation, and unlock.
     The contribution is tentative and owned by the guard, whose first phase removes it on any
     later refusal. From here on this context is a contributor, so neither a teardown release
     nor a sole-contributor step (i) can drop the ring under it. The check sits immediately
     before the CAS rather than after it, because a mismatch found after the CAS could be
     answered only by un-publishing.
   - **(b) The publication CAS** (jehw `:18386`), as today. A lost CAS rolls back through the
     guard.
   - **(c) Commit, which has no refusing step** (nothing in it returns `busy` or a plan error):
     - under `kv_region_mutex_`: insert the new entry, holding the extent handles, this
       context's tenant-slot handles, the host reservation and its host-slot views (a move of
       the pre-L1 handles); on the tenant-only path, install the new table in the existing
       entry, with the device slots (c) kept and (ii) carved and the unchanged host-slot views
       (rulings §D15); unlock;
     - under the ring record's lock: if the ring's slots changed (a growth), swap in the new slot
       handles, bump `ring_plan_gen`, clear `RELEASING` if this call set it, and move the
       superseded handles **into the guard**; unlock. The ring admit then installs the slot
       handles for u1bb's dispatch, per the split step 2's fit recorded (r4 I6: no live
       `zone_available(RUNTIME)` read). Its slots were reused in place or carved at step 6, so
       it allocates nothing;
     - still under L1, if the fit reported a KV demotion that releasing lease-vetoed copies
       would have covered: log the WARN naming those copies, their bytes and the demoted
       layers, and, once llama.cpp-423j has landed, issue its retire request for them, once
       (§2.9). This is the only site, and it runs once per committed transaction.
   - **(d) The superseded handles drop after L1 is released**, in the guard's second phase
     (r5 m-a). Revision 6's first draft dropped them "with no lock held" while the transaction
     still held L1, which contradicted P2 and H7t. A claim still retained by an in-flight event
     keeps its block until the event completes (P2).

**The tenant-only path (a matched key; zhcn's protocol, §3.1 of its design).** A
same-key republish, which is how the auto-ubatch ladder, the setters, encode and a republish
after `memory_update` arrive, re-plans only this context's head slots and its ring
contribution. It never re-fits KV and never yields.
- **A matched tenant key is an OK no-op (zhcn r3 I-5; r5 m-j; lead ruling).** If the candidate's
  tenant section digest (§2.4.4) also equals the published tenant key, the backend's coverage
  query (below) answers EQUAL and llama keeps the published residency: no probe, no step (i), no
  registry change, no L0 and no L1. The server's per-request setters (a sampler set, a LoRA
  apply) change the graph set but usually not one slot's bytes, and they must not pay a drain
  and a fit, or open a refusal window, for nothing. On the llama side zhcn recreates the
  scheduler over the same slots (zhcn §3.1 step 2), which reuses the same indexes through the
  event chain and touches no backend state. The graph-set digest serves only zhcn's staleness
  seal.
- **Reuse in place: the covered path (r5 I-B; zhcn rev 5; lead ruling; r6 m-10 for the host
  tier; r7 m-11, m-1).** A candidate is **covered** when both of these hold:
  - every candidate tenant slot fits the published slot at the same `(cohort, index, device)`
    (`slot_bytes ≤ cap`), host slots included; and
  - its ring contribution (§2.7) does not grow. The ring contribution is not part of the tenant
    key (the digest covers the tenant section only, §2.4.4), so a candidate whose tenant slots
    are all covered but whose ring contribution grows (a larger `n_ubatch`) is **not** covered:
    it takes the growth path below (r7 m-11).

  A covered candidate takes **no L0 and makes no registry change** (zhcn §3.1 step 3; r7 m-1):
  no step (i), no fit, no allocation, no release, and no write to the ring record. The published
  table, its caps, the published tenant key, any published index the candidate no longer uses,
  and a ring contribution larger than the candidate needs all stay as they are, as held planned
  room (like the ring's excess, §2.7). Claims keep reading the same table, and a claim's
  `slot_bytes ≤ cap` is what makes the smaller need safe. Because the key is not re-recorded, a
  repeat of the same candidate takes the covered check again, which is arithmetic over the
  context's own table under the `kv_region_mutex_` leaf and nothing else. Only this context's
  thread re-plans this context, and no other transaction writes its table, so the check needs no
  L0. zhcn rev 5 states the rule for its tenants; this design applies it to the host tier too
  (r6 m-10), so a covered republish neither allocates host bytes nor refuses. Only a candidate
  with a slot that outgrows its published cap, a new index, or a growing ring contribution takes
  the path below.
- **The coverage query, and L0 on the tenant-only path (rulings §E.2, §L0R; zhcn rev 5.1; r7
  I-2, m-1).** llama asks first, through a read-only backend query,
  `ggml_backend_sycl_tenant_coverage(ctx, candidate)`, which answers EQUAL, COVERED or GROWTH
  from the context's own table under the `kv_region_mutex_` leaf. It neither publishes nor
  prepares a live update, so it is not an L0 entry point (§2.4.2 "The re-plan transaction
  mutex"), and EQUAL and COVERED end there with no L0. On GROWTH, llama opens
  `ggml_backend_sycl_replan_scope`, which takes L0 **before (s)**, and holds it to the guard's
  second phase. The backend transaction wrapper it then calls takes the same token at its top as
  a nested hold (rulings §L0R). No path writes the ring record without L0: the covered path does
  not write it at all.
- **The guard is declared before (s)**, so steps (s), (0), (i) and (ii) all run inside its
  lifetime (r5 I-A(d)). Its first phase is where RELEASING is cleared on any exit that did not
  publish.
- **(s) Synchronize (rulings §B step 1; zhcn step 1; r6 I-1, m-2).** Before (0), zhcn's step 1
  synchronizes **every queue that can hold work on this context's slots**, and zhcn rev 5
  carries the list (lines re-pinned to master `3d9414c8c`; r7 m-4):
  - each device's execution queue. `ggml_backend_sycl_synchronize` (`ggml-sycl.cpp:84293-84356`)
    waits `stream(device, 0)` only, and on the deferred-decode path only `last_graph_event`
    (`:84306-84307`), so the step waits the queue itself;
  - the split queues: secondary, merge and coord (`g_split_secondary_queue_owner`,
    `g_split_merge_queue_owner`, `g_split_coord_queue_owner`, `:62351-62353`);
  - the MoE shared-context queues (`ggml_sycl_ensure_moe_secondary_queues_for_plan`, `:5852`);
  - the cache's queues (`cache->get_queue()` and `get_bcs_queue()`, waited at `:4788-4790` and
    `:4803-4805` today);
  - the MMID exact queue needs no wait of its own: it is `ctx.stream()` (`:75151`), so the
    execution-queue wait covers its submits and host tasks (`:75254-75264`);
  - the CPU-dispatch queue (`ggml_sycl_get_cpu_queue`, `cpu-dispatch.cpp:4771`, submitted
    through `cpu_submit_async`, `:252-266`);
  - each device's TP queue when TP is on (`ggml_sycl_get_tp_queue`, `common.cpp:194`), and the
    TP device-1 worker's own queue (`:48143`);
  - each device's PP pipeline copy queue when `GGML_SYCL_PP_PIPELINE` is on
    (`g_pipeline_copy_queue`, `:23431`).

  zhcn rev 5.1's list lacks three of these: the cache's `get_bcs_queue()`, the TP worker's
  queue, and the PP pipeline copy queue. zhcn is asked to add them (each submits on a queue the
  device wait does not cover; r7 m-1). **The invariant (rulings §Z52 "Order"):** MEASURE and the
  coverage query are read-only and run before (s); (s) runs only on the growth path, after L0;
  and it completes before (c)'s occupancy check and move-out and before any reap or release.

  The list is zhcn's gate 30 (the queue census). The between-graph scatter lists,
  `g_pending_secondary_scatter` and `g_pipeline_scatter`, hold compute slices, but zhcn flushes
  them at every `graph_compute` exit and drains them in its step 3 (zhcn C2t), so neither needs
  a queue wait of its own. A queue added later must join the list. If one is missed, (d)'s
  backstop makes the miss a loud `[CONTEXT-PLAN-BUG]`, never a free under queued work. **Nothing
  between (s) and (d) publishes a retention whose event postdates (s) (r6 m-2):** (a)'s
  own-context clear submits no device work (its per-context input-staging reset,
  `graph_input_staging_clear`, ignores its queue: `common.hpp:6457-6460` at `3d9414c8c`); A4's
  gallocr free takes the queue's last event and submits no barrier (§2.3.2); and the record-mode
  vacate's marker is a claim-state event that retains nothing (§2.3.2).
- **(0) Probe, then hold its placements (rulings §M7 I-3(b), I-5(a)).** zhcn's measure pass has
  already sized the candidate's tenants. A fit places the candidate's head slots on the live
  geometry, with two kinds of room counted free **by arithmetic only**: the context's current
  tenant slots, and, when this context is the ring's sole contributor and its ring must grow,
  the ring's old slots (§2.7). Their extents are copied out under the `kv_region_mutex_` leaf,
  and the ring's under the ring record's lock, each then released. (0) takes **no ring-lock
  copies and no pins**: it is not step 2's ring-lock section, and "no side effects" (step 4,
  "Probe mode") excludes both. So a growth that fits only in the old ring's room passes (0), and
  nothing (0) did keeps the old ring allocated into (ii). **The probe checks the host tier by
  arithmetic too (rulings §D15; r5 I-I(3)):** the candidate's host slots against the context's
  held host reservation (§2.4.3). A refusal stops here with nothing released. zhcn's probe runs
  for every device before any (i), so a predictable refusal is atomic across devices and the
  host tier.

  **If (0) fits, its placements become this call's pending ranges at once** (rulings §M7
  I-5(a)), recorded under the group mutex on each TLSF before anything is released. They cover
  the part of the freed room the candidate will use and any free room the fit used beside it.
  From here until (ii) carves or the guard's first phase clears them, `allocate_excluding` keeps
  every weight allocation, expert-cache fill and MoE layout materialization out of them
  (§2.3.3), and none of those takes L0. A range that overlaps a still-allocated old slot is
  harmless: the block is allocated anyway, and when (f) frees it the room is already excluded.
  So the window between (f) and (ii), in which a fill that takes no lock could land in the freed
  room, no longer exists.
- **Host slots on this path (rulings §D15; r6 m-10; r7 I-6).** The context's host slots were
  sub-carved at the plan's maximum caps from its held host reservation at the first publish
  (§2.4.3), so every candidate the plan admits is covered on the host tier: its host slots are
  reused in place, nothing is allocated, and nothing is released. A candidate whose host need
  exceeds the reservation is refused **arithmetically** at (0), against the reservation and
  never against live free room (rulings §B step 4). (c) moves the host slots from the table into
  the registry entry, and they go into the new table at the commit.
- **Device slots are reused per slot on this path too (rulings §B step 4).** A device slot at
  the same `(cohort, index, device)` whose cap covers its new need is not released by (i): (c)
  moves it into the registry entry, not the batch, and (ii)'s fit places that head slot on it
  (`reservations`, §2.4.1) and carves nothing for it. Only a slot that outgrows its cap, or an
  index the candidate no longer uses, is reaped and dropped. So a candidate that grows one
  cohort frees and re-carves only that cohort's slots.
- **(i) Before L1: release, by a synchronous targeted reap (zhcn r3 I-3; r5 I-G; lead ruling
  "B", 2026-09-26; r6).** The sequence is llama's, because it interleaves the scheduler's
  destruction; this design supplies the backend steps and the rules they keep. There is no
  poll, no sleep and no timeout anywhere in it. The only wait is (d)'s backstop, on an event
  that (s) should already have completed.
  - **(a) This context clears its own executable graphs, and nothing else (zhcn r3 I-3; r6 I-2;
    lead ruling).** On its own thread, with no lock held, outside `graph_compute`, and only if
    this context has a recorded graph, a new own-context clear runs per backend. It destroys
    that backend's executable graphs and releases its `graph_retained_handles` and its pools'
    graph retentions, which in record mode releases the claims those graphs hold (§2.3.2 "Per
    execution mode"). Without it, after any recorded decode the old slot blocks stay alive
    through `graph_retained_handles` (master `3d9414c8c` `ggml-sycl.cpp:97064`), which the drain
    does not wait on (`mem-handle.cpp:88-91`), and the probe's "free by arithmetic" would be
    false.
    - It is **not** `sycl_exec_graph_clear_active` (master `3d9414c8c`
      `ggml-sycl.cpp:99310-99359`), which also does two process-global things:
      - `release_graph_retained_handles()` (`:99280`; `mem-handle.cpp:2060-2070`) swaps out the
        whole `graph_unwaitable` list, every context's entries included, and those can back
        another context's live executable graph (`mem-handle.cpp:92-103`);
      - `ggml_sycl_cpu_staging_cache_clear()` (`:99355`) clears `g_leaf_staging_cache`
        (`cpu-dispatch.cpp:2491-2496`), an unordered map with no mutex that another context's
        compute thread may be using while (a) runs.
    - It also calls `graph_unpin_moe_experts` and `graph_unpin_weights` (`:99356-99357`). A
      per-context re-plan never unpins MoE experts or weights (rulings §B step 5; oze0).
    - None of these is reachable from (i) (H7af). This context's own slices that reached
      `graph_unwaitable` are removed by (d)'s owner-filtered scan, so neither this design nor
      zhcn's depends on the global release.
    - The gate on a recorded graph keeps (a) off the other tenant-key changes: the dkw0 comment
      (`:99301-99309`) records that the existing clear is not side-effect-free even when
      nothing is recorded.
    - This is not the cross-context graph clear the r4 addendum withdrew (§2.9): a context acts
      only on its own graphs, on the thread that drives it, at a point where it is not
      computing.
  - **(b) llama destroys the old scheduler** (`sched.reset()`), which frees every gallocr buffer
    object and so releases each claim with its event (A4).
  - **(c) The occupancy check and the move-out**, under `kv_region_mutex_`. The check reads
    each slot's **atomic claimed flag only**, and never takes the per-slot spin lock under the
    leaf (r6 m-8). A current tenant slot that is still claimed is `[CONTEXT-PLAN-BUG]` (a
    buffer outlived its scheduler). There is no legitimate in-flight occupier (zhcn's question,
    answered): queued eager work holds a slot's **lifetime** through retained slices, never its
    **claim**, which ends at submission. A record-mode claim does last for its graph's life
    (rulings §REC; §2.3.2 "Per execution mode"), but (a) destroyed this context's executable
    graphs, which vacated those claims on an eager event, so none is left at (c). The one state
    on master that looked
    like an in-flight occupier, u1bb's ring slot kept `busy` until its `done_event` completes, is
    vacant-with-event under §2.7's conversion. Otherwise (rulings §B steps 7-8; zhcn §3.1 step
    5(c); r7 m-1):
    - **take every holder of the published slot table (rulings §B step 8; r6 I-3):** move the
      registry entry's `shared_ptr` into the local batch, and reset each backend context's
      cached pointer (§2.3.2). This is legal because (c) runs on the context's own thread, which
      is not computing, and (b) released every claim; a claim that finds no table is
      `[CONTEXT-PLAN-BUG]` with an error status;
    - **check the table, then open it:** its `use_count()` must be 1, the batch's own
      reference, else `[CONTEXT-PLAN-BUG]` naming the table's holder (the holder scan, (e)).
      Only then is it opened: each old **device** slot that the candidate reuses in place and
      each old host slot move into the registry entry; each growing device slot and each index
      the candidate no longer uses move into the batch; the emptied table is dropped;
    - clear the tenant key; unlock.

    The reap's owner list (d) is the batch's device handles, which nobody else can reach once
    the table is taken. zhcn opens the table at the same point, so the two designs agree (r7
    m-1).
  - **No step 8′ (rulings §B.2, superseding §B.1).** Revision 7.4 had a step (c′) here that
    erased this context's entries from four backend caches. It is deleted: every one of those
    caches only compares identity (`stable_identity_equal`, `common.hpp:6706` and `:7040` at
    master `3d9414c8c`), so an owning handle held there is a holder to fix at the producer, not
    to exempt. The MMVQ and MoE q8 `cached_src_handle` fields and the `moe_ids_cache` keys
    become a non-owning `mem_handle_identity`, `g_data_ptr_cache` keys on identity plus the
    tenant publish generation, and every `runtime_tensor_extras` publisher restores at scope
    exit; all of that is zhcn's scope, specified in rulings §B.2 and not restated here. So (d)
    follows (c) directly, and the three §B holder classes stand (e).
  - **The holder census beyond §B.2's list (rulings §M7 I-7; r7 I-7).** No step 8′ means the
    census carries the whole §B.2 load, so it must name every container that can own a tenant
    slice between graphs. Three more do, at master `3d9414c8c`:
    - **`graph_input_staging`** (`common.hpp:6338-6343`, used at `:6375-6428`): a per-context
      map keyed by a raw `ggml_tensor *`, whose entries own a `mem_handle` and persist across
      graphs; it is cleared only by `graph_input_staging_clear` (`:6457`), that is only inside
      (a), which runs only when something is recorded. After beni converts
      `graph_input_stage` to a CONTEXT tenant cohort (§2.4.3), every growth republish of a
      context with nothing recorded would trip (c) or (e). The fix holds on the eager path as
      well as in record mode: the map keeps only a non-owning `mem_handle_identity` and the
      capacity, and the staging storage is claimed by index from the `graph_input_stage` slot
      each graph, so its slice leaves graph compute through the claim and the retained store
      like any other tenant slice. A recorded graph's claim lives in that graph's container
      (class 3, rulings §REC). zhcn's §3.1.1 lists the map as "not a holder"; it is one until
      this fix lands, and the two designs now classify it the same way (told to zhcn);
    - **`g_moe_ids_d2h_cache`** (`ggml-sycl.cpp:19457`) and **`g_moe_prompt_admission_cache`**
      (`:19513`): `thread_local` maps keyed by `moe_ids_cache_key`, whose `handle` field owns
      the ids tensor's compute slice (`common.hpp:5713-5736`). They are cleared only at the next
      graph start (`:19985`, `:19998`) or at model teardown (`:33767`), so between graphs they
      hold that slice. §B.2's fix is on the **key type**, so it covers both maps and the context
      member (`common.hpp:5750`) at once: the key's `handle` becomes a non-owning
      `mem_handle_identity`. The entry values' own `device_handle` and `staging_handle` are the
      cache's allocations, not tenant slices; if beni converts either to a tenant claim, it
      falls under the same rule;
    - **`g_data_ptr_cache`** (`:19339`), a `thread_local` map from `(tensor, device)` to an
      owning `mem_handle`, cleared at graph start (`:19342`). §B.2 binds: it keys on identity
      plus the tenant publish generation and holds no owning handle. zhcn's clear at
      `graph_compute` exit is **in addition** to that key fix, never instead of it (rulings §M7
      I-7): a clear on one thread's exit does not reach another thread's `thread_local`
      instance.

    Each is either a non-owning identity key or holds no owning handle of a tenant slice past
    `graph_compute` exit. H7ag names all four, each with a mutation witness.
  - **(d) The reap, with no L1-L5 lock held (lead ruling "B"; r6 I-1, I-2, m-3, m-5; zhcn rev
    5).** zhcn's mem-handle call, shared with llama.cpp-uwlx's yield (jehw), is
    `retained_reap_result release_retained_referencing(const retained_reap_request &)`. The
    request is `{const mem_handle * owners; size_t n_owners; retained_reap_precondition pre;
    const char * reason; bool * owner_pending}` (rulings §R). Its owners are the batch's device
    handles (the old slots the candidate does not reuse), plus the ring's moved-out slots; a
    reused slot is not an owner, because it is not freed. It is built with
    `pre = RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER` and `owner_pending = nullptr`. The other
    precondition, `RETAINED_REAP_QUERY_EVENT_STATUS`, is uwlx's: it keeps what is pending and
    marks `owner_pending`. The result is, in order, `{size_t entries_dropped; size_t
    entries_pending; size_t pending_bytes; size_t in_hand_yields; size_t unwaitable_dropped}`
    (rulings §R). `entries_pending` counts entries; `pending_bytes` sums the distinct owner
    allocations that have a kept entry, each counted once. In COMPLETE mode both are 0 on a
    correct run. Under the retained-store mutex only, it scans three places:
    - the queued records, whole entries, by owner-control identity;
    - the drain worker's in-hand record. The worker publishes it in the same critical section as
      its pop, with an immutable copy of the entry's control identities taken at the pop. It
      clears `in_hand` only after the record's handles are gone: cleared outside the mutex (then
      `in_hand` is cleared at today's `--active` point), or parked into `graph_unwaitable` in
      the same critical section that clears `in_hand`. The reap reads only that copy, never the
      live handles vector that the worker clears outside the mutex, and yields (a cv wait, which
      releases the mutex) until the worker clears it: at most one completed-event return and one
      drop or park. Today the worker pops, waits, clears and decrements in separate sections
      (`mem-handle.cpp:126-177`, the same at `11faace69` and `3d9414c8c`); the change lands with
      the reap's implementation, zhcn's or jehw's, whichever lands first;
    - after the yield, the process-global `graph_unwaitable` list, per handle, by owner-control
      identity. So a copy the worker parked during the yield, through its `"command graph"`
      catch (`mem-handle.cpp:143-162`), is caught (lead ruling), and the reap does not depend on
      (a)'s global release, which it no longer has (r6 I-2).

    Matches are moved out under the mutex and destroyed after it is released; nothing is
    destroyed under it (§2.10). A `graph_unwaitable` match counts in both `entries_dropped` and
    `unwaitable_dropped`. **The backstop (r6 I-1; lead ruling).** After the unlock, each
    moved-out queued record's event status is queried. A complete event, the precondition
    holding, costs no wait: the query blocks only on an incomplete event. An incomplete event
    means a queue that (s) did not cover. The record is then **waited, never freed early**, so
    there is no use-after-free, and the miss is `[CONTEXT-PLAN-BUG]`: a WARN by default, an
    abort under `GGML_SYCL_STRICT_PLAN=1`. The BUG line carries the count of records the
    backstop waited, and a `GGML_SYCL_PRIVATE_TESTING` counter accumulates it; §R's five-field
    result is unchanged (rulings §D16; r7 m-6). It is not re-queued and reported pending,
    because that would turn a missed queue into a silent pending. A `graph_unwaitable` match
    carries no event; it is dropped on the precondition that (s) and (a) destroyed every
    executable graph that could use it (zhcn rev 5). Retained handles are otherwise released
    only by the background worker (`mem-handle.cpp:2039-2052`), which is why the step exists.

    **The ring's moved-out retentions are dropped here too (rulings §M7 I-3(a)).** The
    last-generation `retained_owners[slot]` that the ring's move-out took (below) are slices of
    the old ring slots and share their controls (`mem-handle.hpp:70-72`, `:84`), so each would
    count in that slot's `use_count()`. They are not in the retained store, so the reap does not
    see them; (d) handles them itself, after the reap and before (e): each one's
    `done_events[slot]` takes the same backstop (query; an incomplete event is waited, counted
    in the same BUG line and counter, never freed early), and then the retention is dropped.
    They are never held in the batch across (e).

    **The request's owner vector dies at the end of (d)** (zhcn §3.8 row 20). Its `owners` are
    `mem_handle`s, one reference each, so it is destroyed after the reap and the backstop and
    before (e). If it lived longer, every (e) `use_count() == 1` check would read 2.
  - **(e) The use-count check (rulings §B step 10; r6 I-3; rulings §M7 I-3).** With no lock
    held, on the local batch (the table was checked and opened at (c)):
    - each moved-out CONTEXT tenant handle must have `use_count() == 1`, the batch's own
      reference;
    - each old ring slot must have `use_count() ≤ 1 + P[slot]`, the guard pins snapshotted at
      the ring's move-out ("The ring", below). Pins come only from a transaction guard's step 2,
      and under L0 no other guard is alive at the move-out while this call's (0) takes none, so
      P is 0 on every path and the check is `use_count() == 1`. A nonzero snapshotted P is
      itself `[CONTEXT-PLAN-BUG]` (a guard outlived its transaction, or a mutator skipped L0;
      r7 m-3).

    Anything else is `[CONTEXT-PLAN-BUG]`, aborting under `GGML_SYCL_STRICT_PLAN=1`. The BUG
    line is followed by a holder scan over the census containers (zhcn rev 5 row 13), so it
    names who holds the reference. Without STRICT, the call logs, drops the batch and continues
    (zhcn rev 5 row 17): the extra holder keeps its block allocated through its own reference,
    and (ii)'s live-TLSF fit counts that block as allocated, so the worst case is a refusal,
    never an overlap. After (d)
    it means a reference held **outside the retained store**, which the protocol does not
    allow: a tenant slice may be retained only by a claim, by the retained store, by a
    per-context graph container that (a) clears (rulings §B, §B.2; H7ag; r6 I-3). Two kinds of
    holder are design errors to fix in their producers, never to exempt:
    - a per-op cache that keeps an owning handle of a slice past submission, such as the oneDNN
      Graph scratch park, the four containers the holder census names (above), or any park
      beni's conversions leave. A cache that only compares
      identity holds a non-owning `mem_handle_identity` instead (rulings §B.2);
    - a `host_task` lambda that captures a tenant-slot handle instead of publishing its
      retention through the store (zhcn rev 5).

    It is never `busy` and never a timeout.
  - **(f) Drop the batch**, with no lock held, after (e). The blocks are really free before
    (ii), so (ii)'s live-TLSF fit never sees a graph- or event-held tenant block; its live-TLSF
    reading stays as the belt (below). The room they free that the candidate uses has been this
    call's pending range since (0) (rulings §M7 I-5(a)), so no allocation that takes no L0 can
    land in it before (ii) carves.
  - **The ring (rulings §RING, §M7 I-3; r4 I6; r5 I-A; r6 I-4).** If this context is the ring's
    **sole** contributor and its ring must grow, then right after (c)'s unlock (never nested
    with the leaf `kv_region_mutex_`), under the ring record's lock:
    - mark the record `RELEASING` with this context as its owner, and bump `ring_plan_gen`;
    - move out the slot handles into the batch, **and, separately, each slot's slot-state
      retention of its last generation** (`retained_owners[slot]` with `done_events[slot]`;
      master `3d9414c8c` `ggml-sycl.cpp:1526`, set at the record, `:1722`). That retention is
      not in the retained store, so the reap cannot see it, and without the move it keeps every
      ring slot PP MoE has used allocated until that index is next claimed. (d) drops it
      through the backstop before (e); it never sits in the batch across (e);
    - snapshot `P[slot] = pinned[slot]`;
    - a claimed ring slot is `[CONTEXT-PLAN-BUG]`, as for any tenant.

    The old slots go through the same reap as the tenant slots. The bound
    `use_count() ≤ 1 + P[slot]` is sound: once RELEASING is set and the handles have left the
    record, no new pin can be taken (every other transaction waits on L0, and the owner's own
    (ii) finds no handles to copy), and pins only fall. P is 0 under L0, and a nonzero P is
    `[CONTEXT-PLAN-BUG]` ((e)); any excess over the bound is `[CONTEXT-PLAN-BUG]` too, and
    nothing is exempt. (0) made the growth possible without any of this: it counted the old
    ring free by arithmetic and took no copy (rulings §M7 I-3(b)).
    - Revision 7.1 exempted the old ring slots from (e) outright. That hid the slot-state
      retention: the sole-contributor release freed nothing, (ii) refused the growth without
      demoting and without saying why, and a stray holder of an old ring slot went unreported
      (r6 I-4).
    - A ring that does not grow is reused in place (§2.7), so nothing is released. With other
      contributors, the ring is not released here: its new slots are carved at the commit beside
      the old ones, and the old ones go after the publish.
  - **RELEASING is always cleared by this call (r5 I-A(d)).** The publish at step 8 clears it
    and bumps the generation. On any exit that does not publish, including (ii)'s refusal, the
    guard's first phase clears it under the ring lock and bumps the generation, leaving an
    empty ring record. So a refused sole-contributor republish never leaves a stale mark for the
    next transaction on the device, which would otherwise be `[CONTEXT-PLAN-BUG]`.
  - **After a refused (ii) (r6 m-9).** The context is still a contributor, but it now holds no
    ring slots (a sole contributor's ring left at the move-out), and it holds only the device
    tenant slots kept for reuse (the others left at (e)). Its next PP MoE dispatch finds an
    empty ring: that claim is `[CONTEXT-PLAN-BUG]` with an error status, never a silent skip,
    and so is any tenant claim. zhcn's ladder revert re-carves both from empty. The tenant key
    was cleared at (c), so the revert does not match the equal-key no-op, which would return OK
    with nothing carved; it takes this path, and its (ii) carves the previous candidate's slots
    and the ring. If the revert is refused too, the decode fails with that refusal, naming the
    tenant bytes.
- **(ii) Under L1.** Steps 2-4 run for the head slots only, with the region fixed; a RELEASING
  mark this call owns passes its own step 2. The fit is restricted to this call's own pending
  ranges, recorded at (0), exactly as step 6's re-fit is (`own_ranges`): nothing else can have
  entered them, because every allocation on those TLSFs honours them and no other re-plan runs
  under L0, and (f) freed what they overlapped. It reads the **live** TLSF, so a block whose
  release has not completed counts as allocated by construction (the belt; zhcn's row 4a): the
  worst case is a refusal, never an overlap. On a correct run (ii) therefore places exactly what
  (0) placed, step 6 carves it, and steps 7-8 run; step 5 records nothing new and nothing
  yields. If (ii) does not fit, some old block is still allocated, which means a holder that (e)
  already reported as `[CONTEXT-PLAN-BUG]` (an abort under STRICT). Without STRICT the candidate
  is then refused with the tenants-alone message, **with no demotion**, and the ladder moves on;
  a setter or encode surfaces the refusal as a decode error naming the tenant bytes. It is not a
  runtime race: no allocation that skips L0 can take the room (rulings §M7 I-5; r7 I-5), and
  zhcn's ladder revert republishes the previous candidate.
- The ladder's reservations therefore stay one KV region per `(c, d)`, which C3's trace counts.

**Why "re-fit after `sched_reserve`" is ruled out (r3 I3).** A republish after `sched_reserve`
that re-fits would change the region after the KV is allocated and would break the idempotent
key. Every demand must therefore be computable before its publish, for every ladder
`n_ubatch`, and zhcn's measure pass provides that.

**`[KV-PLAN-BUG]` is only this:** step 6's `allocate_at` failed for an extent or slot that the
re-fit placed inside this call's own pending ranges. Nothing can legitimately have entered
those ranges, so that is an allocator bug by construction, and the name is honest (r4 I8);
§2.8 covers it.

**The rollback's second phase and a concurrent fit (r4 m2), accepted.** B's guard drops B's
extents with no lock held, after B released L1. On B's device this no longer overlaps
anything: B's guard runs both phases inside L0 (rulings §E.2), so no other re-plan can
snapshot while B's extents are doomed. A weight allocation can still
see them as allocated for that instant, which errs toward the weight's own spill, never toward
a miss. Marking doomed extents in phase 1 would add a third block state to remove a transient
pessimization, so it is not done.

**`no_alloc` contexts (r2 m6).** When the published shape has `no_alloc` set (llama's
dummy-buffer contexts, master `llama-kv-cache.cpp:389-393`), the transaction publishes the
residency and reserves nothing, and the claim is skipped for their size-0 buffers.

**Teardown: one release, on every path, whatever the close returned (r2 N-I3(b); r3 I6; r4
m9).** Revision 5 ran `kv_region_registry_extract` at the two close sites, and only on their
success paths: `close_if_idle` calls `clear_bindings_for_context` only when the execution
registry's close returns OK, and `finish_drain` returns early on a non-OK `finish_drain` or on a
failed `sycl_module_mutation_guard` (BUSY) (master `ggml-sycl.cpp:15117-15161`). A non-OK close
left the context's registry entry and tenant slots with no reclaim path, for the process
lifetime (the execution registry's own stranding on that path is llama.cpp-34hr's).

Revision 6 decouples the two. The region release is a new exported proc,
`ggml_backend_sycl_kv_region_release(ggml_sycl_exec_context_id)`, resolved like the other
`llama_context_sycl_*` procs. It is **idempotent**: a second call, or a call for a context that
never reserved, finds nothing. The backend's close functions no longer reach it.

**llama calls it from exactly one place (zhcn M1; lead ruling):** the destructor of an RAII
member `sycl_plan_guard` of `llama_context`, declared **before** `sched` (zhcn's rev 3 checked
the member order: `memory` at `llama-context.h:335`, `sched` at `:390`).
- A member's destructor runs after `~llama_context`'s body, so after the execution close
  whatever it returned and after `synchronize()`, and after `sched`, so the scheduler's
  compute-buffer slices are already dropped.
- It covers the paths revision 6's first draft missed (r5 I-F). `drain_and_close` returns
  early on exactly the non-OK paths r4 m9 is about (master `11faace69` `llama-context.cpp`
  `:162-164` id 0, `:166-173` no drain sequence, `:177-183` quiesce, `:185-191` begin,
  `:208-219` extract), so a call "at its end" never ran there. And the construction-unwind
  scope's `sycl_exec_close_if_idle` zeroes `sycl_exec_context` (`:752`) before that scope's
  destructor finishes (`:760-764`), so a release keyed on that member passes id 0. A throw after
  the transaction (`:810`), including §2.8's own `create_memory` refusal (`:869`), took that
  path and leaked the region. An initialized member is destroyed when the constructor throws,
  so the guard covers it.
- **The guard holds its own copy of the ContextId**, captured when `create_exec` returns
  (`:796-801`), and never reads `sycl_exec_context` at destruction (lead ruling on the r5 I-F
  residual). An unset copy (no SYCL device, `vocab_only`) makes the destructor a no-op.
- H7m checks that the proc is called only from `sycl_plan_guard`'s destructor, that the guard
  reads its captured id, and that it never runs under BINDING. Mutation witnesses: revision 4's
  erase inside `clear_bindings_for_context`, revision 5's call guarded by `rc == OK`, a call
  placed at the end of `drain_and_close` (which the early returns skip), and a guard that reads
  the zeroed member.

Two properties zhcn's teardown relies on (zhcn T1, T2):
- **No live-lease assertion.** Even after `sched` is gone, a slot can still be referenced by
  an in-flight event's retained slice or a claim not yet released. The release only drops the
  registry's references; the last reference frees each block by refcount, outside every lock
  (§2.3.2). It never asserts that a slot is unclaimed or unleased.
- **No BINDING nesting.** The guard passes the ContextId it captured, and the proc is entered
  with no lock held, so the extract under `kv_region_mutex_` never runs under
  `g_execution_backend_binding_mutex`, and there is nothing to rank.

With no lock held on entry, it takes the process-global re-plan mutex (L0, rulings §E.2) and
holds it to the end. Then
it runs (r5 I-A(a); lead ruling: the proper path, L1 plus RELEASING, never a lock-free ring
drop):
1. under `kv_region_mutex_`: move every `(c, *)` entry out into a local batch; unlock;
2. take L1, then, per device, the ring record's lock: remove `c`'s contribution and bump
   `ring_plan_gen`. If `c` was the last contributor, also mark the record `RELEASING` and move
   the ring's handles into the batch, together with each slot's slot-state retention of its last
   generation (r6 I-4). That retention goes to `retain_handles_until_event(done_events[slot])`
   after the unlock (§2.7, r6 I-6), so the teardown frees each old ring block after its last
   event rather than leaving it held until a later ring reuses the index. Unlock the ring lock,
   then release L1. Under L0 no transaction is between its step 2
   and its step 8 while this runs; the generation bump and the copies (rulings §RING) remain as
   the checked invariants;
3. with no lock held: drop the batch. The last `mem_handle` reference (the registry's, the KV
   buffers', a transaction guard's copy, or a claim still retained by an event, whichever goes
   last) frees each extent and slot through `zone_free`, which runs the §2.3.2 free rule;
4. under each ring lock it marked: clear `RELEASING` and bump `ring_plan_gen`, leaving an empty
   record.

L1 is taken only for step 2, which is a short copy-out; the proc is `noexcept`, and a lock
failure aborts, as for the guard (r4 m14). Dropping the registry's references is safe whatever
state the execution registry is in, because it is only a reference drop: anything still queued
holds its own leases (§2.3.2).

#### 2.4.3 Planned context-side demand (owner ruling 2026-09-26: "plan them exactly first")

**The ruling.** The owner ruled on the question revision 3 left open: **plan them exactly
first**. There is no interim estimate, no fixed floor and no fallback constant. Revision 3's
estimate-plus-floor formula, its measured floor and its once-per-device miss WARN are all
withdrawn.

**Who produces what (lead rulings 2026-09-26; r3 I1; r4 I9, I10).** moua L3 consumes demand
records; the only record moua produces is the recurrent state's.
- **llama.cpp-zhcn:** the device compute chunks (per SYCL device buft, one slot per measured
  gallocr chunk, the K-shift and post-update graphs included) and the fattn K/V materialize
  slot, cohorts `context-compute` and `context-fattn-materialize`, CONTEXT scope, and the host
  compute buffer, cohort `context-compute-host` (below); and the `context-graph-stage` cohort
  (zhcn §2.9, 1oxa's W7 staging tail, assigned to zhcn by the lead), a device head-slot index
  set with eager and record-mode (rulings §REC) sets, which (ii) carves like the other tenants
  (zhcn §3.8 row 28). The sizes come from zhcn's pre-publish measure pass and travel in the
  descriptor's tenant section (§2.4.4). zhcn also deletes u1bb's
  `k_pp_moe_ring_compute_reserve_bytes_per_row` (lead ruling 3).
- **llama.cpp-beni** (split from 23mk, lead ruling on 23mk's Q3; ticket comment c-khaj): the
  `graph_input_stage` per-context slab, the oneDNN activation scratch, `mmvq_q8_slab` (merging
  the per-op mint at master `ggml-sycl.cpp:46128`, the SOA prealloc at `:97906` and the q8
  activation cache at `common.hpp:6648`), the MoE Q8_1 prealloc (`mmvq.cpp:16856`), the
  forced-split packed-K slot (`fattn.cpp:1636`, TRANSIENT, max over layers), and **the oneDNN
  Graph scratch** (`unified_cache::onednn_graph_scratch_alloc`, reached through
  `make_engine_with_allocator`, shaped by `n_kv` and `n_q`; r4 I10(d)). Every beni demand is
  computed by a visitor in zhcn's measure walker, before KV admission, and travels in zhcn's
  tenant section. The llama.cpp-cxgg sidecar fix is folded into beni.
- **llama.cpp-jzvq:** the fattn thread-local device workspaces (`fattn.cpp:747` and its
  callers, the `:1800` XMX split workspace) and the **MXFP4 MoE token-generation caches**
  (`g_mxfp4_moe_tg_reuse`, `mmvq.cpp:1744`, and `g_mxfp4_stored_gemm_ksplit_scratch`,
  `mxfp4-stored-gemm.cpp:242`; ticket comment c-jv0r). The latter are on the **default GPT-OSS
  decode path** (four raw `role=6` EXT-ALLOC lines on the B50 master baseline), so without jzvq
  they would be unplanned claims on every GPT-OSS decode, a STRICT abort. jzvq moves them to
  per-(ContextId, device) ownership with an exact demand function, and **closes before moua
  L4** (r4 I10(c)).
- **llama.cpp-23mk core** (lands after jehw): zones and `forbid_vram_zone_spill` on every
  out-of-arena row, the trace, the dead-code deletions (`scratch_pool`,
  `unified_cache_allocate_moe_q8_1_graph_scratch`), the oneDNN weights scratch reserved once at
  its max in the ONEDNN tail. The opt-in persistent-TG buffers (`unified-kernel.cpp:4890`) are
  **refused under an arena**, with one WARN (lead ruling on 23mk's Q5), so they need no record.
- **The persistent packed-K sidecar** is not a record: it is each layer's companion slot in the
  KV region (§2.4.1), sized by 23mk's function over `kv_layer_cells` and `n_stream`.
- **moua:** the recurrent state, one CONTEXT-scope slot per device RS buffer (§2.4.4, r4 I9).
- **llama.cpp-u1bb** (in master): the ring, DEVICE scope, one contribution per context at that
  context's reservation `n_ubatch`, sized by u1bb's pure `pp_moe_onednn_admit_ring` (§2.7).
- A cohort routed into a tail zone (ONEDNN/RUNTIME/SCRATCH are separate TLSFs, laid out at
  load) is outside this interface, except the ring's RUNTIME half, which the fit places (§2.7).

**The host-pinned tier: `context-compute-host` (zhcn A3; lead ruling: one section, this one).**
zhcn's SYCL_Host compute buffer (61.65 MiB on GPT-OSS) is a tenant of the same protocol on the
host-pinned tier, and this paragraph is its only specification; zhcn supplies the measured
`slot_bytes` in the tenant section and cites it. What is the same: CONTEXT scope, the handles
held by the registry entry, owner-first allocations, claim by index (the SYCL host buft's
live-object count, with A1's assertion), release at submission with an event (A4), release by
handle drop at the §2.3.2 sites, one charge and one uncharge (§2.3.2), here against the host
inventory. What differs, because it is not in the device geometry:
- **It is not a head slot of `kv_region_fit`.** No VRAM TLSF holds it, so it neither competes
  with KV nor demotes anything; demoting a KV layer would add host bytes, not free them. A
  host-tier shortfall refuses the candidate with the tenants-alone message.
- **It is allocated before L1, owner-first, never carved under L1 (zhcn M2; lead ruling; closes
  §5 (i)).** The host arena is a lazily grown `pinned_chunk_pool` (`unified-cache.cpp:4235` at
  `2c4f5e45d`, `:4253` at `11faace69`; created with `committed=0.0 GB`), so a host-zone
  allocation can need a new pinned chunk, which is a USM call, and a USM call must not run under
  L1. The transaction therefore allocates the context's host reservation (below) before L1,
  after step 1 and before step 2 of the full transaction at the context's first publish, through
  the unified cache's owner-first surface, **`unified_allocate_owner`**
  (`unified-cache.cpp:15815` at `11faace69`; `:15797` at `2c4f5e45d`), never a bare USM call
  outside unified-cache code. The request (r5 I-I(1)): `must_host_pinned` and `use_pinned_pool`
  set, category `HOST_COMPUTE`, cohort `context-compute-host`, `require_host_usm_base` false,
  and `forbid_host_zone_growth` false, since it runs only at the first publish (below); the
  pattern is master `11faace69` `unified-cache.cpp:6515-6527`. `unified_allocate_owner_impl`
  classifies it as `CACHE_SUBALLOCATION` and mints the control first (`:15729-15812`), and the
  result is handed to the registry through `mem_handle::from_owned_alloc`. **Inside a claim
  scope**, the SYCL_Host buft's `alloc_buffer` becomes a claim of these slots and no longer
  allocates through the legacy `unified_alloc` → `from_legacy_owned_alloc` path
  (`ggml-sycl.cpp:42466-42475`). **Outside a claim scope it keeps that path (r6 m-11):** llama's
  output buffer (`11faace69` `src/llama-context.cpp:3492-3499`) and the LoRA and control-vector
  tensors also allocate on the SYCL_Host buft, and zhcn keeps them out of every claim scope
  (zhcn T10, its gate 20). The path is `unified_allocate_owner_impl` (the control is minted
  first, `:15711`), `unified_alloc` (`:14699`), the contiguous host-zone allocation
  (`:15050-15092`), and on a miss `host_zone_grow` → `pinned_chunk_pool::grow_zone` →
  `grow_into`, under the pool's own lock (§2.10's census row). The reservation is a real
  allocation from the start, so it needs no pending range. The guard holds it; a failed host
  allocation is a refusal before L1; a rollback drops it in the guard's second phase, with no
  lock held; and **the commit only installs** the handle into the registry entry, a move, not an
  allocation. The full path runs only at a context's first publish, and the tenant-only path
  allocates nothing on the host tier (below).
- **The first publish reserves AND HOLDS the host-tier room (rulings §D15; r7 I-6; lead ruling
  on the pool's phase gate).** `grow_into` WARNs, and at `GGML_SYCL_HOST_ALLOC_PHASE_GATE` ≥ 2
  asserts, when the pool grows during a PP or TG phase (`pinned-pool.cpp:865-882`), so the one
  host allocation happens at the **first** publish, at context creation, outside any inference
  phase:
  - **One held carve.** The first publish allocates a single owner-first carve, the context's
    **host reservation**, through the request above, and holds it in the registry entry for the
    context's life. "Reserves" means this carve holds the bytes: a flag such as
    `forbid_host_zone_growth` reserves nothing, and revision 7.5's "records that capacity"
    reserved nothing either (r7 I-6).
  - **Its size is the plan's maximum tenant host demand:** for each host `(cohort, index)`, the
    largest `slot_bytes` over every candidate the plan admits (each ladder rung up to the top),
    summed. That envelope is at least any single candidate's total.
  - **Host slots are sub-carved from it**, at those maximum caps, as offset views that share the
    reservation's control (the slice form §2.3.2's claims use). So every candidate the plan
    admits is covered on the host tier, a republish never allocates host bytes, and no host slot
    is ever released and re-allocated within the plan.
  - **A request beyond it is refused by arithmetic, against the reservation**, never against
    live free room: a **candidate refusal** with the tenants-alone message at rulings §B step 4
    ((0) on the tenant-only path), before anything is released, never a growth under the gate
    and never a transient. The setters and warmup are not ladder candidates, so they can reach
    this refusal; it is deterministic in the plan.
  - Expert-cache host fills and host weight staging take no L0 and allocate from the same pinned
    pool (zhcn T7). Between publishes they can take any free host room, but not the
    reservation, which is allocated. They are themselves planned capacity (P4; 1oxa r3 I5); host
    growth the plan does not account for is a plan bug, not a race to tolerate.

  §3.1 H4 carries the cases, including the RED where a concurrent host fill takes the room.

**The record (r3 I2, I7; r4 I1, I4, m12).**
```
enum class demand_scope : uint8_t { CONTEXT, DEVICE };

struct context_side_demand {
    int                  device;
    vram_zone_id         zone;      // WEIGHT, KV or RUNTIME (ring only): which TLSF (§2.3.5 routing)
    shared_zone_lifetime lifetime;  // CONTEXT or TRANSIENT (WEIGHT_SIDE_TRANSIENT under §2.1's lever)
    demand_scope         scope;     // whose lifetime the reservation follows
    uint64_t             owner;     // the ContextId; for DEVICE, the contributing ContextId
    const char *         cohort;    // the cohort_id its claims carry
    std::vector<size_t>  slots;     // slots[i] = cap of slot index i, one per allocation that can be live at once
};
```
- **Indexed slots, not a byte peak (r3 I7; r4 I1).** The producer lists, by index, the
  allocations that can be live at once, and each claim names its index (§2.3.2).
- **Scopes (r4 I4).** MODEL scope is deleted: it had no producer and no release site (r4 m12).
  **Every per-op cohort is CONTEXT scope** (beni's ticket already says "scope/owner per
  context"). Revision 5 let the per-op scratch be DEVICE scope, sized from one transaction's
  plan, which could not see other owners: model 2's context could shrink slots model 1's
  running contexts still used, two concurrently executing contexts would share one slot, and a
  dropped slot that was claimed had no release. Per context, each context's slots are its own,
  and each context executing concurrently (C5; the overlapping host submission of CLAUDE.md
  §5) claims its own. **Only the ring is DEVICE scope**, with the explicit rule of §2.7.
- **Zone.** Every record carries its zone (23mk, agreed), so its slots are placed on that
  TLSF by §2.3.5's routing.

**Reconciliation, per transaction.** A full transaction plans this context's CONTEXT records
and the ring's record (the max over the live contributors, this context included). A slot is
reused in place when its owner already holds the same `(cohort, index)` with a capacity at
least the new need (r5 I-B; r6 m-1);
every other slot is carved new, and the old one it supersedes is released at the publish
(§2.4.2 step 8). Claimed slots are never moved.

**Timing (r3 I3; r4 I10(d)).** Every record exists before the transaction that consumes it.
zhcn's measure pass runs before each publish, for each ladder candidate, and computes its own,
beni's and (where they are graph-shaped) jzvq's demands with its visitors. The ring's and the
recurrent state's are pure over the plan, the model, `n_ubatch` and the descriptor. There is no
re-fit after `sched_reserve` (§2.4.2).

**Plan == reality at claim time.** A context-side site claims `(owner, cohort, index, size)`
(§2.3.2).
- **In plan:** the index exists, `size ≤ cap[index]`, and the index is not claimed. The claim
  returns the slice and the event to depend on.
- **Plan violation:** the size exceeds the cap, the index is out of range or still claimed, or
  the cohort has no record for that owner (unplanned; H7p gates it in source).
- **The disposition: no fallback, never raw, never a skipped op (r3 C1; r4 I3, m11).** It is
  zhcn's rule, agreed as the one rule for every tenant:
  - log `[CONTEXT-PLAN-BUG]` at ERROR, once per `(device, owner, cohort)`, with the planned
    slots and the request;
  - **return an error status that reaches the graph.** A scheduler buffer claim returns NULL
    from `alloc_buffer`, and `ggml_backend_sched_alloc_graph` returns false (zhcn adds the
    missing `ggml_gallocr_reserve_n` return check). A per-op claim makes its op return a failure
    status out of `graph_compute`, so the decode fails with an error. An op is never skipped
    (beni c-khaj: "a miss returns an error status (never a skipped op)");
  - abort under `GGML_SYCL_STRICT_PLAN=1`.

  Revision 5 let a non-STRICT violation take unreserved context-side room through
  `context_side_place`. That is deleted (§2.3.2): room nobody planned is exactly what the
  owner's ruling excludes, and zhcn's rule was already "no fallback in scope". H7p checks that
  every claim site's failure branch returns an error status and reaches no allocator.
- **On 1oxa's VM backing** there is no unreserved context-side room at all, so the same rule
  is also the only possible one there (1oxa rev 4 cites this revision's §2.3.2 for its chunks).
- The claim state is exposed through a `GGML_SYCL_PRIVATE_TESTING` accessor
  (`reserved_slot_claims`), and G1 reads it. It is no longer a fit input, because slots never
  move and the fit reads only which blocks are allocated.

**The transition rule (r4 I10(b)).** The landing order puts beni's site conversions after moua
L4-L7. So when L4 lands, most TRANSIENT sites still allocate as today. The rule:
- a site becomes context-side **only in the commit that converts it to a claim** (zhcn's,
  beni's or jzvq's). Until then it keeps its current placement: the zone and
  `forbid_vram_zone_spill` that 23mk core gave it. It is not a context-side request, it is
  never classified as a plan violation, and STRICT does not abort on it;
- H7p carries an explicit **unconverted-site list**, each entry naming its site and its
  converting ticket. At L4, H7p passes when every site is either converted (claims, with a
  producer) or on the list. The list may only shrink, and a mutation that adds a site fails
  the gate. beni's last conversion commit empties it, and from then on H7p requires it empty;
- jzvq closes before L4, so its sites are never on the list.

This keeps H7p meaningful at every landing: nothing can be context-side without a producer,
and nothing can silently stay unconverted.

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

**The shape crosses the ABI once, from llama, which is the side that knows it, in one
descriptor (lead ruling 4, D4).** zhcn also needs to publish its measured tenants. Two
descriptors would be two versioned structs for one publish, so there is one, and moua owns its
layout.
- A new publish entry point, `ggml_backend_sycl_set_runtime_context_desc(backend, token, n_ctx,
  n_ubatch, n_seq_max, kv_unified, swa_full, flash_attn, const ggml_sycl_runtime_context_desc *
  desc)`. It is resolved by proc address like its sibling, and the old entry point stays.
- The descriptor:
  ```
  struct ggml_sycl_runtime_context_desc {
      uint32_t struct_size;       // sizeof as the publisher built it; the reader gates every field on it
      uint32_t version;           // bumped on any change of meaning; append-only fields otherwise
      // KV-shape section (moua)
      int32_t  type_k;
      int32_t  type_v;
      uint8_t  v_trans;
      uint8_t  no_alloc;
      uint8_t  sidecar;           // persistent packed-K sidecar enabled (companion slots, §2.4.1)
      uint32_t n_stream;
      uint32_t n_layer;
      uint32_t layer_desc_size;
      const ggml_sycl_kv_layer_desc * layers;
      // measured-tenant section (zhcn fills, with beni's and jzvq's demands; element layout is zhcn's)
      uint32_t n_tenants;
      uint32_t tenant_desc_size;
      const ggml_sycl_context_tenant_desc * tenants;
      // recurrent-state section (moua, r4 I9); n_rs_layer = 0 for a model with no recurrent state
      uint32_t n_rs_layer;
      uint32_t rs_layer_desc_size;
      const ggml_sycl_rs_layer_desc * rs_layers;
  };
  ```
- **The layout rules, which moua owns.** Fields are only appended. A reader treats a field
  beyond the publisher's `struct_size` as absent, and refuses a `version` it does not know.
  Arrays are read at their own element stride (`layer_desc_size`, `tenant_desc_size`), each
  element also gated by its size. A section's owner defines its element struct in the same
  header and adds a row to H7o's layout check.
- `ggml_sycl_context_tenant_desc` is one slot (rulings §T), agreed with zhcn (zhcn's proposal
  plus `device`; accepted by zhcn rev 4.1 `4bb0436`, which withdrew its superset carrying `tier`
  and `scope`):
  ```
  struct ggml_sycl_context_tenant_desc {
      uint32_t struct_size;  // element stride gate, as for every section
      uint32_t cohort;       // the cohort id; zone, lifetime, scope and tier are fixed per cohort
      uint32_t slot_index;   // the claim index (§2.3.2)
      int32_t  device;       // SYCL device index; -1 for the host-pinned tier
      uint64_t slot_bytes;   // the slot's cap
  };
  ```
  Zone, lifetime, scope and tier come from one static cohort table in the same header, never
  from the element, so they have one source; H7o checks the table covers every cohort id. A
  record of §2.4.3 is the set of elements with one `(device, cohort)`. The fattn slot is its
  own cohort. beni's and jzvq's demands are elements of this section too, filled by the
  visitors in zhcn's measure walker (r4 I10(d)). The tenant key (the matched-key path, §2.4.2)
  is the digest of this section's `(device, cohort, slot_index, slot_bytes)` tuples.
  **`device = -1` names a tier, not an owner (r5 I-I(2)):** its slots are allocated in, held
  by, and claimed through the cache and the registry entry of the SYCL_Host buft's device,
  `ggml_sycl_device_id_from_backend_dev(buft->device)` (device 0 today; the buft is bound to
  reg device 0 at master `11faace69` `ggml-sycl.cpp:42535-42536` and routes to that device's
  cache and host arena, `:42445-42451`). The registry key is `(ContextId, that device)`. If
  that device holds none of the context's KV layers, the entry exists anyway, with no extents,
  and holds only the host slots; the release proc drops it like any other.
- **The recurrent-state section (r4 I9; lead ruling: recurrent state is moua's, zhcn T6/D2).**
  Each `ggml_sycl_rs_layer_desc` is `{ uint32_t il; int32_t type_r; int32_t type_s; uint32_t
  n_embd_r; uint32_t n_embd_s; uint32_t n_rows; }`, exactly the arguments
  `llama_memory_recurrent` passes to `ggml_new_tensor_2d` for `r_l`/`s_l` (master
  `llama-memory-recurrent.cpp:117-120`, with `n_rows = mem_size × (1 + n_rs_seq)`), for the
  layers its filter keeps and whose device is an arena device. The backend sizes each device's
  RS buffer with the same per-tensor rule as `kv_layer_tensor_bytes` (row size × rows, padded
  by the tiered buft's alignment and `get_alloc_size`), summed in layer order the way
  `ggml_backend_alloc_ctx_tensors_from_buft` lays one context's tensors out.
- Each `ggml_sycl_kv_layer_desc` is `{ uint32_t n_embd_k_gqa; uint32_t n_embd_v_gqa; uint32_t
  n_head_kv; uint32_t n_embd_head_k; uint8_t has_kv; uint8_t is_swa; }`:
  - the widths are exactly the ones llama passes to `ggml_new_tensor_3d` (master
    `llama-kv-cache.cpp:347-348`). `n_embd_v_gqa` is taken after the `[TAG_V_CACHE_VARIABLE]`
    padding, and is 0 when the model has no V (MLA);
  - `has_kv = 0` marks a filtered, shared or reused layer;
  - `n_head_kv` and `n_embd_head_k` are the layer's KV head count and K head dim
    (`hparams.n_head_kv(il)`, `hparams.n_embd_head_k(il)`), added for 23mk's
    `packed_k_sidecar_bytes(ℓ)`, which wraps `ggml_sycl_fattn_xmx_compute_packed_k_bytes(n_kv,
    H_kv, batch)` and needs the head dim and `type_k` to decide whether the layer takes the
    packed-K path at all (impl-23mk's question; `n_embd_k_gqa` alone cannot be factored).
    These two fields block beni's sidecar conversion, not 23mk core (lead ruling).
- The backend reads `layers[i]` at the stride `layer_desc_size`. Both libraries are built
  together, but these arrays cross the dlopen boundary, where an appended field would otherwise
  silently move the stride.
- If libllama publishes through the old entry point while an arena device has device-planned
  KV, the transaction refuses with a named cause (`no runtime context descriptor published:
  libllama and libggml-sycl are out of step`). It does not guess the shape or the tenants.

**On the llama side, one function produces the shape.**
- `llama_kv_layer_shapes(model, params_mem, cparams)` is factored out of two places:
  `create_memory`'s filter/reuse/share decisions and `llama_kv_cache`'s per-layer width
  decision. Both of them, and the publish, call it. It covers iSWA's two caches, and the
  attention half of the hybrid memories.
- `llama_rs_layer_shapes(model, params_mem, cparams)` does the same for the recurrent state,
  factored out of `llama_memory_recurrent`'s constructor, and fills the recurrent section. It
  honours the constructor's `offload` flag (r5 m-f): with `offload = false` the state lives on
  the CPU buft, so `rs_layers` lists **only the offloaded layers on arena devices**, and a
  non-offloaded layer plans nothing.

**The recurrent state is planned and claimed (r4 I9).** Revision 5 called it out of scope on
the ground that it does not use the tiered KV buft. It does not, but that made it worse, not
irrelevant: `r_l`/`s_l` are allocated by `ggml_backend_alloc_ctx_tensors_from_buft` on the
plain device buft (master `llama-memory-recurrent.cpp:98-129`; the SYCL hook
`llama_recurrent_sycl_kv_buft` returns nullptr *"until recurrent placement is designed"*,
`:17-25`), so it went down the unplanned backend-buffer chain, which zhcn's GH gate
(Qwen3.8-Flash-Next under STRICT) forbids. This is that design:
- **Fit.** Each arena device's RS buffer is one CONTEXT-scope head slot, cohort
  `context-recurrent-state`, index 0, sized as above. It is mandatory, like every head slot:
  attention KV demotes around it, and if it cannot be placed with every KV layer on the host the
  transaction refuses, naming it. Per-layer recurrent demotion is not offered, because the
  recurrent layer's device is decided by `model.dev_layer(i)` at load, not by the KV planner,
  and moving it would be a placement change (P3), not a capacity answer.
- **Claim.** `llama_recurrent_sycl_kv_buft` returns a new recurrent-state buft for an arena
  device. Inside the region scope (which `create_memory`, and so the recurrent constructor,
  already runs in, §2.5), its `get_max_size` returns the planned slot size, so
  `ggml_backend_alloc_ctx_tensors_from_buft` makes exactly one buffer, and its `alloc_buffer`
  claims slot 0 of that cohort (§2.3.2) and returns a buffer holding the slice. A size mismatch
  is a plan violation (§2.4.3). A non-arena device keeps the plain buft.
- **Key.** The recurrent section's digest is in the idempotence key (§2.4.2 step 1).
- **Tests.** H2 adds a hybrid shape (attention and recurrent layers on one device) where the RS
  slot is placed first and attention KV demotes around it; H3 checks the section's sizes against
  the tensors `llama_memory_recurrent` creates, in the CPU-buft llama test; zhcn's GH is the
  device acceptance (§3.3 C7).
- **Frozen once computed.** llama computes it at the constructor's first publish (master
  `llama-context.cpp:810`, before `create_memory` at `:869`), stores it in `llama_context`, and
  every later republish sends the stored copy of the KV-shape section: the ladder (`:1571`,
  `:1936`) and the FA recheck. The tenant section is re-measured per ladder candidate by zhcn.
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
`ContextId -> {extent mem_handles[], shape key, frozen n_ubatch, kv_region_layout, tenant
slots, tenant key}`, where:
- `kv_region_layout` maps each device layer to `(extent index, slot_offset, slot_size)`, plus
  `(sidecar_offset, sidecar_size)` when the sidecar is enabled (the companion slot's second
  slice, 23mk);
- `tenant slots` is the published slot table: per `(cohort, index)`, the reserved slot's
  `mem_handle`, its cap and its claim state (§2.3.2), behind a `shared_ptr` that each backend
  context caches for claims, and that the tenant-only path's (i)(c) takes back from all of
  them (§2.4.2; r6 I-3);
- `tenant key` is the tenant section's digest (the tenant-only path, §2.4.2).

There is one `mem_handle` per extent and per reserved slot, each its own owner-first control
(r2 N-I2; §2.4.1). The transaction publishes entries at its publish step (§2.4.2 step 8), the
claims and the residency hook read them, and the release proc removes them (§2.4.2
"Teardown"). **The registry entry is the owner of the context's reserved slots (r4 I3):** this
is zhcn's "held carve", `{KV extents, KV key, tenant chunk handles, tenant key, layouts}`.

**Sidecar claims are keyed `(ContextId, device, layer)` (23mk, agreed).** The fattn sidecar
site asks the registry for its layer's companion slice, with a hard ceiling of
`sidecar_size`; the KV claim checks the buffer against the **KV-only** sum of its layers, so
the companion never inflates the KV buffer's size check.

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

Under an open region scope, the hook therefore answers from the registry. The mask and the
slot table then have one source, and the mask check becomes a check of llama's mask assembly,
not of two plans. The answer for layer `il`, whose planned device is `d`:
- **`d` reserves regions** (an active arena): device-resident iff `registry(d)[ContextId]` has a
  slot for `il`.
- **`d` does not** (r3 m9): no arena, an arena that is enabled but not active, or a mixed device
  set where only some devices have an active arena. The registry holds nothing for `d`, so
  reading it would answer "host" for a layer the plan put on `d`. For such a device the hook
  reads the plan, as today. H8 adds a mixed-set case.

Outside a scope, the hook reads the plan as today.
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
   device-planned layer has a slot whose KV part equals
   `kv_layer_alloc_bytes(kv_layer_tensor_bytes(...))` for that layer, from the entry's stored
   shape (§2.4.4). Check that the buffer's size equals the **KV-only** sum of its layers' bytes
   (a sidecar companion is claimed separately by the fattn site, §2.5), and that mask == slot
   table.
2. **Set `layer_allocs[l]` to a slice.** `kv_layer_alloc::set_owner` takes a legacy
   `alloc_handle` today (master `ggml-sycl.cpp:37727`). L6 adds a `mem_handle` overload
   (audit m1):
   - it stores the slot slice `extent_handle.slice(slot_offset, kv_size)` as
     `zone_handle`/`chunk_lease` (the KV part only; the sidecar companion is the fattn site's
     slice);
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
   - **The recurrent-state buft's clear does the same (r5 m-f).** `llama_memory_recurrent`
     clears its buffer right after allocating it (`ggml_backend_buffer_clear(buf, 0)`, after
     `alloc_ctx_tensors_from_buft`). The recurrent-state buft's clear retains the slot slice it
     writes until its fill event completes, exactly as the tiered KV clear does.
4. **VMEM: the region takes precedence (audit I4; supersedes r1 M5's disposition). The refusal
   itself is carried by llama.cpp-23mk core (lead ruling), which lands before moua L4-L7:** 23mk
   refuses vmem-kv under `arena_active()` with one policy WARN (23mk rev 3.1 §6.6), so L6 adds
   no vmem code. The facts below are why the refusal is needed, and G1 still checks it.
   - The opt-in `GGML_SYCL_VMEM_KV=1` branch (master `:38846-38925`) runs only under an
     active arena, runs before the per-layer path, and returns early. It maps KV in physical
     pages outside the unified cache's accounting.
   - Revision 2's "skip the region when vmem applies" would have kept planned device KV on
     that out-of-accounting path, which is a P1 violation.
   - Instead, under an active arena, device-planned KV **always** takes the region claim
     once L6 has landed. Before L6 there is no region claim: between 23mk core and L6, the
     KV that 23mk's refusal sends past the vmem branch takes the existing per-layer path, whose
     last step is the raw `unified_alloc` fallback (`11faace69` `ggml-sycl.cpp:39267`), which
     23mk's rows 54/55 cite as the interim KV path. L6 replaces that path with the claim.
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

### 2.7 u1bb ring, MMID pools, RUNTIME, compute overflow (r1 M9 corrected; r3 I5, I8; r4 I4, I6, I7)

- **The whole ring is a planned head slot, both halves (lead ruling 2; r4 I6).** u1bb keeps
  the ring's weight slots, and any ubatch-scaled slot kind the RUNTIME zone can also hold, in
  the RUNTIME TLSF, and sends the rest to the shared KV zone (u1bb `:17596-17620`), deciding
  the split from live `zone_available(RUNTIME)` (`:17603`). Revision 5 made only the KV-zone half
  a head slot, and released the whole physical ring before the yield
  (`release_pp_moe_onednn_scratch_ring`, `:17581`), so for the whole L1-released window the
  RUNTIME half was unreserved, and compute buffers, which try RUNTIME first, could take it.
  Revision 6:
  - the fit's geometry includes the RUNTIME TLSF, and the fit places every ring slot, deciding
    the split by u1bb's own rule (weights in RUNTIME; an ubatch-scaled kind in RUNTIME if it
    fits there, else in the shared zone) from the snapshot. The split is part of the fit's
    output and is frozen: step 8's commit installs per the recorded split and reads no live
    `zone_available`;
  - the ring is **never released before the publish** in a full transaction: a changed ring
    gets new slots at the commit, beside the old ones, and the old ones go at the publish (§2.4.2
    step 8). So the ring is never physically absent, and the guard never has to restore it;
  - the only early releases are the tenant-only path's step (i) for a sole contributor whose
    ring must grow, and the release proc's last-contributor drop. Both mark the record
    `RELEASING` and bump the generation, under L0, so no other transaction can observe the mark
    or be past step 2 at the time; either would be `[CONTEXT-PLAN-BUG]` (rulings §E.2).
    RELEASING is cleared by the releasing call's publish, by its guard on any other exit, or by
    the release proc once its drop is done (r5 I-A). This closes r4's race in which B, running
    in A's window, published a ring A had released, and r5's three further routes to the same
    absent-ring publish (§3.1 H9).
  - **A model load never releases or re-sizes the ring (llama.cpp-r7fz; rulings §M7 I-4).** On
    master, `populate_inventory_globals` (`3d9414c8c` `ggml-sycl.cpp:15891`, reached from
    `ggml_backend_sycl_compute_placement_plan_early` at model load) overwrites the device's
    planned ring sizes and depth (`unified_cache_set_planned_pp_moe_onednn_scratch`,
    `:15967-15970`), releases the physical ring whenever it holds KV-zone bytes
    (`release_pp_moe_onednn_scratch_ring()`, `:15983-16000`; its own comment says this "also
    releases a ring that another still-loaded model's live context admitted"), clears the
    KV-zone slot flags (`:16002`), and overwrites the per-row bytes and the planned `n_ubatch`
    (`:16013-16015`). On an arena device the KV-zone half is always fit-placed, so that release
    would fire on every load that follows a live PP MoE context: with model A live on device 0,
    B's load (an L0 holder, so serialized) releases A's ring, and A's next PP MoE claim finds an
    empty ring, which is `[CONTEXT-PLAN-BUG]` with an error status, a −2 that is fatal on the
    server. On arena devices, therefore:
    - a load neither releases the ring nor writes the ring record's sizes, depth, split flags,
      per-row bytes or `n_ubatch`. The ring record changes only through contributions: a
      context's contribution is computed from **its own model's** plan (its per-row bytes and
      depth, carried in that model's `placement_plan`, not in device globals), recorded at its
      commit, and removed by its release proc;
    - storage is released only by the release proc's last-contributor drop and by the
      sole-contributor move-out (above);
    - a load only adds: its model's ring demand enters the device's plan when a context of that
      model contributes.

    A device with no arena keeps master's behaviour, outside this design. **Gate (H7z):**
    `release_pp_moe_onednn_scratch_ring` and the planned-ring setters
    (`unified_cache_set_planned_pp_moe_onednn_scratch`, `_kv_zone_slots`, `_row_bytes` and
    `_n_ubatch`) are reachable, on the arena path, only from the release proc, the
    sole-contributor move-out and the commit's install (§2.4.2 step 8); the mutation witness
    restores the `populate_inventory_globals` release. **Test (H9):** "load B while A
    contributes" (§3.1).
- **The KV-zone half's size: one source (zhcn r3 item 9; r5 m-n; lead ruling).** The ring's
  context-side (KV-zone) half is `unified_cache_get_planned_pp_moe_onednn_kv_zone_bytes(device)`
  (`unified-cache.cpp:2222` at `2c4f5e45d`, `:2240` at `11faace69`): the activation and output
  slots routed to the KV zone, **times `ring_depth`**
  (`unified_cache_get_planned_pp_moe_onednn_ring_depth`). On master that routing is u1bb's own
  split; here the fit decides the split (above), so for arena devices the planned
  `*_in_kv_zone` flags are set from the fit's recorded split at step 8 (c)'s install, and the
  function returns the recorded split × depth. The fit reads the ring's per-half slot demand,
  never this function, so the function is a reader of the fit's output, not a second input.
  zhcn's G2/GA "180.0" is its value at `ring_depth` = 1, and the plan line prints the depth
  beside it so the score cannot silently assume 1.
- **The ring record is declared state (r4 I6).** Per device, under its own lock (the existing
  L5 `g_pp_moe_onednn_scratch_slot_state[device].mutex`): the slot handles (the ring's owner is
  the device cache), each slot's claim state, the contributions `{ContextId → ring size}`,
  `ring_plan_gen` and the RELEASING flag. **The generation is read, not just written (r5 I-A(c)):**
  step 2 copies it, step 8 (a) compares it, and every mutation of the record bumps it: a
  contribution recorded or removed, a slot swap at a publish, RELEASING set or cleared, and the
  release proc's last-contributor drop.
  It is not u1bb's per-slot `pp_moe_onednn_next_generation_` (unified-cache.cpp:17897), which
  stays what it is, a claim generation.
- **Multi-context rule (r4 I4(c)).** Revision 5 carried u1bb's device-global semantics forward:
  the ring was sized at the latest reservation's `n_ubatch`, so a later context with a smaller
  `-ub` shrank a ring an earlier context still needed. Now:
  - each context contributes the ring at its reservation's `n_ubatch`; the ring is sized as the
    **max over the live contributions**;
  - it grows at the transaction whose contribution exceeds it: new slots beside the old, the
    old released after the publish. **The overlap is real and is priced (r5 I-B; lead ruling).**
    The growing transaction's fit places its KV around both the old ring (still allocated) and
    the new one, so it can demote up to the old ring's bytes worth of layers for room freed just
    after its publish. Those layers are labelled `ring-growth` with the old and new ring bytes
    (§2.4.1), never `capacity`, and they stay on the host for the context's life. No
    release-before-carve scheme is added; that would reopen the absent-ring window r4 I6 closed;
  - it **never shrinks by carving**. A smaller need is served by the existing slots in place
    (reuse needs capacity at least the planned size, §2.4.1), so a shrink never carves, never
    needs room, and never demotes or refuses another context. When a contributor leaves, its
    contribution is removed at the release proc and the ring keeps its size; the excess over the
    live max is held, planned room that every fit reads as allocated, bounded by the largest
    departed contribution, until the last contributor leaves;
  - it is released when its last contributor's entry drops, through the release proc's L1 and
    RELEASING path (§2.4.2 "Teardown").
- **Claims.** A ring slot is claimed by its ring slot index, with event-chained reuse (§2.3.2). On
  master the state is occupancy-until-completion: `pp_moe_onednn_claim_scratch_slot` round-robins, a
  slot stays `busy` until its recorded `done_event` completes, and when every slot is busy the claim
  host-waits (`wait_event.wait_and_throw()`, master `2c4f5e45d` `ggml-sycl.cpp:1648`; the drain's at
  `:1772` is teardown). Revision 6 converts it: a slot is *claimed* from the claim to
  `pp_moe_onednn_record_scratch_slot_event`, which is the submission; the recorded `done_event`
  becomes the slot's release event; the next claim of the slot, still chosen round-robin (the slots
  are uniform), returns that event for `depends_on` instead of waiting; `retained_owners` stays the
  slot's lifetime retention. So a slot whose work is still queued is **vacant with an event**, not
  claimed. Two contexts running PP MoE on one device therefore serialise on the device's own event
  order at the ring, with no host wait (same-device concurrent inference stays unsupported, §2.10
  §5.3; this only keeps the overlap that exists today safe).
  - **Two generations on one slot (r5 I-K; lead ruling).** Master's per-slot state holds one
    generation: `pp_moe_onednn_record_scratch_slot_event` overwrites `retained_owners[slot]`
    (`2c4f5e45d` `ggml-sycl.cpp:1722`), and the claim swaps the owners out and calls
    `release_pp_moe_onednn_scratch_slot(slot, generation)` only after the previous `done_event`
    completed (`:1625`, `:1635`), which the host wait at `:1648` guaranteed. Without the wait, a
    re-claim of a slot whose previous work is still queued must not drop that retention. So, at
    the re-claim, the previous generation's `retained_owners` are moved out, and after the
    slot-state mutex is unlocked they go to `retain_handles_until_event(previous done_event)`,
    before the new generation's record; they drop only after that event. The cache-side
    per-generation refcount (`claim_pp_moe_onednn_scratch_slot` /
    `release_pp_moe_onednn_scratch_slot`, `2c4f5e45d` `unified-cache.cpp:18037-18096`) is
    **deleted for arena devices**, because the reserved-slot handles make it redundant: each
    claim takes a slice of the slot's handle, the slice travels in `retained_owners`, and a slot
    superseded by a ring growth (§2.4.2 step 8) keeps its block until the last such slice drops
    after its event. H7z's mutation witness adds the overwrite restored; H9 carries the
    supersession case.
  - **Every other removal of the retention hands off too (rulings §RING; r6 I-6).** With the
    refcount deleted, `retained_owners` is the only lifetime holder of a superseded ring slot.
    Master drops it with no event, under the L5 slot-state mutex, at five more sites
    (`ggml-sycl.cpp`, the same lines at `11faace69` and `3d9414c8c`):
    - `pp_moe_onednn_reset_slot_state_locked` (`:1551-1559`, `retained_owners.assign(ring_depth,
      {})`), reached from the claim, bind, record and release paths whenever the depth changes
      (`:1602-1604`, `:1693-1695`, `:1712-1714`, `:1733-1735`). The record-mode depth rule below
      makes a depth change reachable;
    - the generation-0 branch of the claim (`:1617-1621`);
    - release-unused and rollback (`:1585-1588`, `:1738-1742`);
    - **the bind (r7 m-5):** `pp_moe_onednn_bind_scratch_slot_generation` clears
      `retained_owners[slot]` (`:1698`). It is benign only while the claim before it has
      already moved the previous generation out (above), so the slot is empty at the bind. It
      gets the same hand-off, and a debug check that the vector it moves out is empty; the
      witness plants a retention at the bind and requires it to be handed off, not dropped.

    Each of these would be a free under queued work, and each is a `mem_handle` destruction
    under a listed lock (H7t). Each site now moves the vector out and, after unlocking the
    slot-state mutex, hands it to `retain_handles_until_event(done_events[slot])`. A plain clear
    remains only where the generation is 0 and nothing was recorded. H7z covers all six sites,
    and H9's two-generation arm adds the depth change.
  - **Record mode (r5 I-C, for the ring).** A ring claim made while recording follows §2.3.2's
    record-mode rule: the recorded graph holds the slot for its life, and the slot vacates at
    the graph's destruction on an eager event; no node event is stored as a ring slot's release
    event. The ring is device-shared, so a recorded holder takes a slot from every context on
    that device, and the ring's depth must count record-mode holders the way a per-op cohort's
    index sets do (§5 (m)). L4 checks whether the PP MoE oneDNN path is reached while recording
    at all; if it is not, the rule is vacuous for the ring and H7ab gates that it stays so.
- **What the head slot fixes (r3 I5).** Revision 4 carried u1bb's order: KV took the gap, and
  the ring was admitted afterwards against whatever was left, refusing the context when nothing
  was left (*"PP MoE oneDNN scratch ring does not fit"*, u1bb `:18388`), where demoting one KV
  layer would have fit both. H2 covers it.
- **The ring-held yield limitation, closed by L4+ (lead ruling, from jehw's merge).** On the
  jehw-merge order, the yield models the KV zone while the ring's KV-zone slots are still held
  (the ring is released and re-admitted after KV, inside `ggml_sycl_replan_pp_moe_onednn_ring`).
  So on a device holding both WOQ copies and ring slots, jehw's fit places fewer KV layers than
  the room allows: an error toward extra demotion, never an out-of-arena allocation, and only
  on a MoE model with Q4_0 dense weights (which have copies). Revision 6 counts the ring once,
  as its head slot, with its existing slots reused in place, so the limitation is gone. H2
  carries the case with a RED on the jehw-merge order (§3.1).
- **The budget-room check is deleted for arena devices (r4 I7).** The ring's growth is no
  longer checked against `budget_room_bytes` (u1bb `:18362`): the fit is the single source of
  "a head slot fits" (§2.2).
- **The ladder.** Only a candidate's growth competes for leftover room: the tenant-only path
  places the larger ring contribution and compute slots without re-fitting KV, and a candidate
  whose growth does not fit is refused, so the ladder moves on (§2.4.2).
- **The per-row estimate is deleted (r3 I8).** u1bb's admission subtracted
  `k_pp_moe_ring_compute_reserve_bytes_per_row` = 1 MiB/row (u1bb `:17462-17473`, fed at
  `:17629`) as a stand-in for the compute buffer. The compute buffer is zhcn's record in the
  same fit, so zhcn deletes the constant (lead ruling 3), and H7p/H7r check that no
  context-side admission keeps a per-row constant.
- **Retained runs.** A superseded slot released at a publish may leave a RETAINED run when it is
  interior. H4b keeps revision 4's bound (at most one per ladder candidate) and adds that
  fit == carve after the settle.
- **MMID pools** stay in RUNTIME. The fit reads the live ones as allocated blocks of the RUNTIME
  TLSF, and places a new plan's device pool as a head slot there, carved at step 6; the host
  pool is a view of the model's held MMID host carve. So step 7 allocates nothing (§2.4.2 step
  7; rulings §M7 I-5(b)).
- **ONEDNN and SCRATCH tail zones:** no change.

### 2.8 The error path (decision (a), scoped per r1 I3)

Refuse is the default; `GGML_SYCL_STRICT_PLAN=1` aborts instead, mirroring
`GGML_SYCL_STRICT_LEASES`. It is the one STRICT variable for every plan violation in this
design and in zhcn's, beni's and jzvq's, which consume it (lead ruling 4, D3). There is no
alias: `GGML_SYCL_STRICT_KV_PLAN` never existed in any tree, zhcn's rev 2 alias sentence is
withdrawn by agreement (§6.6), and H7e gates that the name never appears. Every site below is
a **terminal** refusal: the fit's demotion order is a planned cascade, not a series of misses,
so nothing in it is counted or aborted as a plan bug (rulings §STRICT). The KV rule applies
**only at region-backed sites**; the context-side rule is §2.4.3's:

1. **§2.4.2 step 6, the commit carve:** `allocate_at` failed inside this call's own pending
   ranges, and only that (r5 m-o). A registration failure after a successful `allocate_at` (a
   container allocation throwing `bad_alloc`, or the runtime registry's duplicate-pointer
   refusal, master `11faace69` `unified-cache.cpp:1399`) is not "the plan and the allocator
   disagree": it is a runtime refusal, logged at ERROR without the `[KV-PLAN-BUG]` tag, and it
   rolls back through the guard like any other refusal.
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

Those requests are planned (§2.4.3): the persistent sidecar is each layer's companion slot in
the region, and the forced-split one is a TRANSIENT slot of beni's. Once beni converts them,
each claims its slot, and the plan-violation rule applies only when a request exceeds what was
planned, which is a true statement, not a false "the plan admitted it". Until beni converts
them (the transition rule, §2.4.3), they keep 23mk core's `forbid_vram_zone_spill`, so a miss
returns failure into fattn's existing non-packed path (`!allocation || tier != DEVICE_VRAM`),
which computes the same result through another kernel. That is P1-compliant.
- **Production never reaches them (r2 m4).** The sidecar is opt-in only:
  `GGML_SYCL_PACKED_K_SIDECAR`, or `GGML_SYCL_FA_FORCE_PATH=split-packed` (master
  `fattn.cpp:151-157`). So the routing matters only under those flags, and C6 has no fattn
  clause.
- **A latent defect becomes more reachable.** When an earlier update's sidecar allocation
  missed, a later `set_rows` can create the sidecar, zero-fill it, and pack only its own rows
  (`fattn.cpp:560-590`). The lookup (`:412-444`) has no completeness check, so it serves stale
  packed K. Forbid-spill makes the miss-then-create sequence more likely under those flags.
  This is filed as **llama.cpp-cxgg**: invalidate the sidecar on any update miss. It is
  pre-existing; its fix is folded into llama.cpp-beni (lead ruling), and moua does not fix it.
  With the sidecar a planned companion slot, the miss-then-create sequence cannot occur once
  beni converts the site.

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
  barrier-gated path. A buried tenant that jehw's predicate calls not yieldable is not
  releasable, and its run is split around it.

**Copies held by recorded graphs: llama.cpp-423j, on retire-on-request terms (r3 C2(c); r4
addendum; lead ruling).** Since `2ad2e0f0e` a WOQ reader leases its copy, and while recording,
that lease lands in the graph's sink for the graph's life (jehw `ggml-sycl.cpp:17598-17607`).
The lease is correct ownership, but its consequence breaks "KV wins over optional tenants":
- once any context has recorded graphs that read copies, those copies are leased for as long
  as the graphs live, which for an idle server slot is indefinitely;
- jehw's predicate vetoes them, the ladder truncates, and a later context's KV demotes while
  the copies keep their VRAM.

**With nothing added, the design is exact and safe, and loses one property.** Exact: every
optional tenant is classified by jehw's predicate (`optional_layout_yieldable_locked`,
`:7660-7673`), which vetoes any lease beyond the mirror; the snapshot copies that veto with the
geometry, and the retire re-checks it (`:7804-7811`), so a pick that became leased is skipped,
the commit sees a shortfall and demotes, and fit == carve holds. Safe: no context touches
another context's graph, and nothing is freed under a live lease. What is lost is "KV beats
optional tenants" for a context created while recorded graphs hold copies. That is a lost
property, not a memory-safety or plan-versus-reality hole, and it is made owner-visible: the
fit reports the copies its ladder truncation left resident, and the transaction logs a WARN
naming those leased copies, their bytes and the layers it demoted.

**Revision 5's reclaim contract is withdrawn (r4 addendum A).** It dropped the holding graphs
from the yield's unlocked half by running `sycl_exec_graph_clear_active` for *another* context
under `g_sycl_graph_compute_mutex` (L2). At jehw `c41fed119` that function waits on the queue
(`:99049`), which is a wait under L2 and a §12.5 violation, and then resets context-local,
thread-affine state (`:99026-99073`) that no current lock serialises against that context's own
compute thread. Its steps must not be implemented, and none of them is: the three-way predicate,
tier 4's "graphs forced to drop" key, and the graph drops in §2.4.2, §2.10, §4 and §5 (h) are
deleted with it.

**423j's terms (this section is their source; the lead's ruling, carried on 423j).** 423j
depends on llama.cpp-uwlx, whose pick-list begin it extends with the retire request (§4):
0. **The entry point is 423j's, on uwlx's pick-list surface (r5 I-J(4)).** jehw HEAD has no
   "retire these named, leased copies" call: `yield_optional_layouts_begin` retires only
   yieldable picks and submits a reader barrier. 423j adds it (this design names no function
   of its own). Its contract: both cache locks held unique; **no barrier** (the leases gate the
   free, so there is nothing to order); the retired copies' mirrors moved out to the caller;
   the caller bumps the optional-layout epoch.
1. **Retire on request, under L1.** The transaction asks jehw's cache to retire the vetoed
   copies, the way the yield's begin already retires its picks: under L1, a retired copy is
   resolved by no lookup (`acquire_entry_lease` refuses retired and non-READY entries,
   `acquire_layout_handle` is lookup-only, and the WOQ gemm resolves the primary on a null
   copy), and the optional-layout epoch is bumped. Bumping the epoch alone would do nothing: a
   holder clears its recorded graphs at the top of its next `graph_compute` (`:105626`, on its
   own thread, which is safe) and re-records in the same call, and without the retire it would
   re-lease the same copy before any transaction could yield it.
   **The retire withdraws the mirror (r5 I-J(1)).** `optional_layout_yieldable_locked` counts
   the cache's own direct-stage mirror as a lease (`own_leases = 1` when mirrored, jehw
   `unified-cache.cpp:7660-7673`), and a retired entry is finalized only at `in_use_count == 0`
   (`finalize_retired_entries_locked`, `:12532-12538`). A retire that left the mirror in
   `direct_weight_entries_` would never free the copy: resident, unusable and unyieldable until
   model teardown, strictly worse than no request. So the retire withdraws and remaps the
   mirror as the yield's begin does (`:7826-7834`), and the withdrawn mirror handles go into
   the transaction guard and drop after L1 is released (step 8 (d); the yield's finish drops
   its mirrors with no L1 the same way, `:7853`). Dropping them under L1 would violate H7t.
2. **The storage comes back at the next finalize pass (r5 I-J(2)).** A lease drop only decrements
   `in_use_count` (jehw `mem-handle.cpp:1115-1129`). The block returns to the TLSF at the next
   finalize pass on that device after the last lease goes. The pass that reliably follows a holder's
   clear is the deferred-free pass at the end of `graph_compute`, gated on
   `has_pending_deferred_frees`, i.e. `retired_pending_count_ != 0` (jehw
   `ggml-sycl.cpp:84063-84068`, `unified-cache.cpp:12766-12768`). So the room returns at the end of
   the holder's next compute; a lease held only by an in-flight reader, with no graph, frees at the
   next pass on that device, whenever one runs. There is no bypass and no cross-thread clear, so P2
   holds as it is.
3. **A narrow trigger.** The request is issued only when this transaction demoted a KV layer
   that releasing the vetoed copies would have covered, and at most once per transaction: at
   step 8, after the publish, so it runs once per committed transaction. **"Vetoed" means
   lease-only (r5 I-J(3); lead ruling):** a copy whose sole veto is a lease beyond the mirror.
   The predicate also rejects non-READY or IN_PROGRESS copies, host-resident or non-DEVICE
   copies, copies already retired, and copies vetoed by `weight_entry_reclaimable`'s live-owner
   terms; none of those is named, because retiring an IN_PROGRESS staging copy is a different
   lifecycle. It names every lease-vetoed copy the fit would have used, because the veto cannot
   tell a sink lease from an in-flight reader's; a retired copy that an in-flight reader holds
   is still correct, since its free waits for that lease. The narrowness matters, because the
   retire costs holders the layout at once while the room comes back only after their next
   compute, and an idle holder never computes, so its retired copy stays resident and unusable:
   strictly worse than today for that copy. **The accepted cost (r5 m-m):** a request naming a
   prefix held by several holders can cost the active holders their layout while an idle
   holder's copy keeps the prefix fenced, so no room returns. It is not restricted further,
   because "idle" is not observable at step 8; the WARN names each retired copy, so a retire
   that returned no room is visible.
4. **A retired-but-leased block is allocated and not yieldable** until it is freed. The
   geometry is the live TLSF, so the fit already counts it that way, and the strict-prefix
   ladder truncates at it (H5).
5. **Who benefits.** The context whose transaction issued the request keeps its demotion: a
   demoted layer stays on the host for the context's life. moua has no re-promotion path, and
   adding one would contradict decided parts of this design (KV is allocated once at
   `create_memory`, the key and the residency are frozen at the first publish, the tenant-only
   path never re-fits KV, and a re-fit after `sched_reserve` is ruled out): it would be live KV
   migration mid-context, host to device under event chaining, with a republished residency
   and executor (P3), a re-record, and a region reservation outside the frozen key. If the
   owner wants it, it is its own ticket. The room the retire returns serves **later**
   contexts, and only after the holders have computed.

jehw's follow-up note also stands: u1bb's reconcile is reordered around begin/finish, with no
lock or signature change.

**Tests.** H5 carries three pure cases: (a) a vetoed copy truncates the ladder, the layer
demotes, and the WARN names both; (b) the request retires the copy, and a re-recording holder
resolves the primary and cannot re-lease it (jehw's (e), code-free on jehw HEAD), and its mirror
is withdrawn and dropped after L1; (c) after the holder's clear **and a modelled finalize pass**
the block is free, and a **new** context's fit uses it (a model that frees on the lease drop
fails); (d) a negative case: an IN_PROGRESS copy, a host-resident copy and a live-owner-vetoed
copy are never named. G1 is the device form of (a)-(c), keyed on whether 423j has landed
(§3.2).

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
  - Each region extent and each reserved slot is an owner-first `CACHE_SUBALLOCATION` through
    `unified_cache`. KV slots, sidecar companions and tenant claims are `slice()`s of their
    extent's or slot's handle.
  - **The new entry points are added to §3's allowlist** (audit m2): `reserve_kv_region`,
    `zone_alloc_optional`, the reserved-slot carve, the recurrent-state buft's `alloc_buffer`,
    and **the host-tier allocation** (r5 I-I(1)): `unified_allocate_owner` with
    `must_host_pinned`, `use_pinned_pool`, `HOST_COMPUTE`, cohort `context-compute-host` (§2.4.3).
    The SYCL_Host buft's `alloc_buffer` is a claim inside a claim scope, and stays on the list
    of `unified_alloc` callers for its out-of-scope uses (the output buffer, LoRA and control
    vectors; r6 m-11). `claim_slot`/`release_claim` allocate nothing (they return slices), and
    are listed as ownership surfaces, not allocators. Revision 5's `context_side_place` is
    deleted (§2.3.2).
  - The allocation class of each is **derived from the request** (its `role`,
    `prefer_vram_zone`, and the §2.1 `lifetime` field) by the existing classifier. It is never
    hand-set at the call site.
  - The new publish entry point (§2.4.4) and the release proc (§2.4.2) allocate nothing.
  - No new raw allocation site is added, and no new `CACHE_BACKING` mint.
  - The one route by which planned KV reached `unified_cache_malloc_device_tracked` under an
    arena (the per-layer tiered `unified_alloc`) is removed.
- **§1.1 planner authority.** The planner decides residency per layer, as today. Its capacity
  input is the allocator's geometry, its size input is the shape llama allocates (§2.4.4), and
  an admission is a reservation, so "planned device" means physically reserved.
- **`mem_handle` lifetime.**
  - Each region extent and reserved slot is freed only when its last handle drops: the
    registry entry's (or the ring record's), the KV buffers', the layer views', a claim's, or a
    fill or kernel event's retained slice.
  - There is no forced eviction. The yield takes only optional tenants that jehw's predicate
    allows. A copy held by a recorded graph is never yielded while leased: 423j's request only
    retires it (withdrawing its mirror), and its block returns at the next finalize pass after
    the holder's own clear drops the lease, so the lease is released by its holder, never
    bypassed (§2.9).
  - Retained runs are TLSF-allocated storage with no owner and no registration (§2.3.2, audit
    I3). They are kept off the TLSF's free lists so that weights cannot take them, and every
    "live bytes" reader subtracts them. A reserved slot is **not** such storage any more: it is
    a registered allocation owned by a live owner (r4 I3).
- **§12.5 locking.** The new `kv_region_mutex_` is ranked L3 and is strictly leaf (§2.3.1).
  - Every final region- or slot-handle drop happens after every listed lock is released: the
    guard's second phase, the publish's superseded-slot drop, the tenant-only path's step (i),
    and the release proc (§2.4.2).
  - `g_execution_backend_binding_mutex` and the two locks nested under it are classified by L7
    as one census entry (§2.3.1, r4 m8). moua adds no work under them.
  - No L5 lock is held across a wait or across the yield (§2.4.2).
  - On jehw HEAD the yield's wait and final drops already run with L1 released (`c41fed119`).
    moua keeps that. No transaction drops another context's graphs, so no graph-compute
    exclusion appears here (r4 addendum). 423j's retire at step 8 runs under L1, takes only the
    cache locks, submits no barrier, waits on nothing, and moves the withdrawn mirror handles
    into the guard, which drops them after L1 (§2.9; r5 I-J).
  - **Allocation under a registry lock: the step-6 carve (r3 m12; r4 m7). Ratified by the lead
    2026-09-26, llama.cpp-moua r5,** as a classified exception to §12.5 for device carves, on the
    lock sequence below (r5 confirmed it lock-safe on `c2613a688`: L1 → the group mutex → {the arena
    authority's registration, `g_runtime_alloc_mutex`}; `unified-cache.cpp` never takes L1; no
    driver call and no wait under L1) and conditional on r5 m-e, applied here. §12.5 calls
    "allocation/device work under registry locks" non-conforming. The carve runs inside the
    transaction body, under L1. Its exact sequence, per TLSF:
    1. **Before L1** (at the guard's construction): mint an upper bound of owner-first controls
       and their allocation ids, `N` = the device KV layers plus the head slots, both known
       from the descriptor and the demand records before L1. L4 splits `unified_allocate_owner`
       into its existing mint half and a bind half for this, so the mint code is not
       duplicated. Unused controls are dropped by the guard's second phase. (Revision 6's first
       draft also reserved `g_runtime_alloc_registry` capacity before L1 and promised "the
       locked section never rehashes". That reserve covers one of four containers, and other
       threads insert between it and the carve, since weights never take L1, so the promise
       could not be kept; it is withdrawn (r5 m-e). Container growth is heap work with no device
       call, and the exception does not rely on its absence.)
    2. **Under L1, take the TLSF's group mutex** (L1 → L5, a legal order).
    3. For each placement, in one section: `allocate_at` (a TLSF metadata split), then bind a
       pre-minted control, then `arena_register_exact` and the runtime-registry commit, which
       takes `g_runtime_alloc_mutex` inside the group mutex and emplaces one row
       (master `unified-cache.cpp:1354-1380`, reached from `zone_alloc`'s locked section,
       `:22479-22546`). The group mutex → {arena authority, `g_runtime_alloc_mutex`} nesting is
       **pre-existing**: every registered `zone_alloc` already does it (`11faace69`
       `:22572-22650`). All three are L5, so the L5 tie-break ("subsystem ordinal") must order
       the group mutex, the arena authority's lock and `g_runtime_alloc_mutex`; L7's §12.5 edit
       writes that order down, the authority lock included (r5). moua adds no new pair.
    4. Clear this call's pending ranges on the TLSF; release the group mutex.

    What it is not: no USM call, no device submission, no wait, and no final handle drop. The
    remaining heap work under the locks is **four container inserts per registration** (r5):
    `group.allocations.emplace` (`11faace69` `:22531`), `arena_authority::register_allocation`
    (reached at `:22514`), the runtime-registry row, which copies a `runtime_alloc_record`
    holding a `std::string` cohort id (`:1400`), and `g_runtime_cohort_tier[cohort_id]`
    (`:1403`). Each can allocate and rehash: bounded, non-blocking heap work with no device
    call. A failure there is a runtime refusal, not `[KV-PLAN-BUG]` (§2.8, r5 m-o).
    **It never reaches `unified_alloc`**, so `unified_alloc`'s overcommit guard, which can call
    `cache->evict_and_flush()` (master `unified-cache.cpp:14795-14840`), cannot run under L1:
    the carve enters `zone_alloc`'s locked branch directly, as the region carve does, and
    carving inside an arena reserved at load changes no physical VRAM figure, so the guard would
    have nothing to check. H7 gates both facts (§3.1 H7u).

    It must be under L1 because the transaction body is: the carve's result is what the CAS
    publishes, and the CAS needs L1. With the re-fit restricted to this call's own ranges
    (§2.4.2 step 6) the carve no longer needs L1 for correctness, so if the contract owner
    declines the exception, the carve can move before the relock inside the yield window (when
    one exists) at the cost of an unconditional unlock/relock; that alternative is recorded,
    not chosen.
  - **The host-pinned compute tenant is not carved here.** The host arena grows lazily, so its
    allocation can be a USM call; it is therefore made before L1 with no lifecycle lock held
    (§2.4.3, "The host-pinned tier"), and this exception does not cover it.
  - **L7 census row: `pinned_chunk_pool::mutex_` (lead ruling).** The pinned pool's lock
    (`pinned-pool.hpp:313`) is not in §12.5's table, and the table's own rule makes an
    unclassified lock a failing census item. L7 adds it as **L5** (allocation/pool), with a named
    exception: `grow_into` (`pinned-pool.cpp:865`) holds it across `allocate_pinned_chunk_owner`,
    which is `sycl::malloc_host`, and, when `alloc_timeout_ms_` is set, a `future.wait_for`. That
    is a blocking allocation and a wait under an L5 lock, **pre-existing** on master for every
    caller that grows the pool; moua does not introduce it, and its host-tier allocation reaches
    it only with no L1-L5 lock held. The exception retires with **llama.cpp-nrng** (reserve under
    the lock, allocate unlocked and owner-first, install and revalidate under the lock), which is
    pre-existing work and not moua's to implement.
  - **L7 census row: the re-plan transaction mutex (rulings §E.1, §E.2).**
    `g_replan_txn_mutex`, one per process, is **L0**: taken before L1 and never under L1-L5.
    There is one, so there is no device order. Its holders: the full and tenant-only
    transactions, the probe, the teardown release proc, every other ring mutator outside decode,
    model load and jehw's optional-layout pass (on every backing, USM and VM). Loads never nest
    inside a transaction, and a transaction never triggers a load. It is held across the
    transaction, including its own synchronize and its unlocked yield or driver window, which is
    a wait on this transaction's own work only. Decode and graph compute never take it. The
    deadlock rule, and teardown's path to it, are in §2.4.2.
  - **L7 census row: the retained-store mutex (lead ruling "B").** `retained_handle_state::mutex`
    (`mem-handle.cpp:53-64`, the lock of `g_retained_handles_state`) is not in §12.5's table.
    L7 adds it as **L5, leaf, and last in the L5 tie-break** (rulings §R; r6 m-4). The facts
    that ranking rests on, at master `11faace69`:
    - nothing is acquired under it;
    - the cv waits release it while they wait: the drain worker's wait for work (lock at
      `mem-handle.cpp:131`, wait at `:132`), `drain_retained_handles`' wait (`:2048-2049`), and
      the reap's in-hand yield. The drain worker's event wait runs outside it (`:125-176`);
    - one debug `fprintf` runs under it: the park's `GGML_SYCL_DEBUG` line (`:159-161`, inside
      the section at `:154-161`). It prints only under `GGML_SYCL_DEBUG` and takes no lock of
      ours;
    - no `mem_handle` is destroyed under it. The park moves the handles into `graph_unwaitable`
      and its `clear()` (`:158`) destroys only moved-from handles. Under the §R change the
      worker clears its record's handles outside the mutex and only then clears `in_hand` under
      it, and a park clears `in_hand` in the park's own section. The reap destroys what it moved
      out only after unlocking (§2.4.2 (i)(d));
    - it is last among the L5 locks because `publish_handles_until_event` can be reached with
      another L5 lock held, so any L5 lock may be held when it is taken, and none may be taken
      under it.

    The reap takes it with no L1-L5 lock held. It is the only lock held **during the scan**
    (the queue, the in-hand record, `graph_unwaitable`). The destroy after the unlock does take
    allocator locks, through each handle's release, with nothing else held, as any `mem_handle`
    drop does.
  - The ring admit (step 8 (c)) allocates nothing any more: it installs slot handles (§2.7).
    No `mem_handle` is destroyed under the ring record's lock (the L5 slot-state mutex): every
    removal of a slot's retention hands it off after the unlock (§2.7, r6 I-6), and the guard's
    pin decrement follows its copy's drop (§2.4.2).
  - The guard's first phase takes only the group mutex, to clear ranges (§2.4.2).
- **§12.6 event leases.** The KV clear's fill events retain the slices they write (§2.6), and a
  tenant's claim is retained until its completion event (§2.3.2).
- **§5.2/§5.3.** KV and the context-side tenants become context-keyed per (ContextId, device),
  which is most of the "context-keyed KV/RUNTIME arena reservation" §5.2 lists as missing;
  only the ring stays per device (§2.7). Same-device concurrent inference remains
  unsupported (§5.3).
- **Docs (L7).**
  - `docs/backend/sycl-memory-design.md` gets a "Shared-zone lifetime segregation" section:
    the classes, the registry and scope, reserved slots and the claim protocol, demand
    records, the limits of §2.9, and the lock order.
  - The `unified-cache.hpp:68-78` arena comment is rewritten.
  - Contract §3 notes that planned KV never spills under an arena, and §5.2/§5.3 are revised.
  - This design doc moves under `docs/design/` history once L7 lands.

### 2.11 Out of scope

- Pattern #2, the VM-backed arena, is llama.cpp-1oxa. The tag policies above stay in the USM
  backing, behind the zone allocate path, so an `arena_backing` interface can separate them.
  1oxa's VM backing gives each class its own VA sub-range instead, and places this design's
  reserved slots, with the same carve and claim protocol, on its transient chunks (1oxa rev 4,
  citing §2.3.2). On a VM device there is no unreserved context-side room, so §2.4.3's
  disposition (error, never raw) is the only one possible there.
- Pre-existing out-of-arena paths are tracked elsewhere, not here:
  - llama.cpp-23mk covers `onednn_weights_scratch` (113.5 MB in A2), the `cohort=?` 4-byte
    STAGING allocations, and `backend-buffer-kv-zone` without `forbid_vram_zone_spill`
    (master `ggml-sycl.cpp:37235`). Its comment c-m2jh adds `backend-buffer-runtime-zone` and
    the tiered KV raw fallback, which moua's L6 removes.
  - **llama.cpp-gxur** (filed for this revision) covers the mechanism itself: with an arena
    active, `unified_alloc`'s raw `unified_cache_malloc_device_tracked` fallback (master
    `unified-cache.cpp:14921`) is fail-open by default for every `must_device` role without
    `forbid_vram_zone_spill`. gxur flips that default after 23mk and moua L6 land.
- **The non-FA outside-arena reserve (r3 I8 note).** The transaction's non-FA scratch check
  compares against an EMPIRICAL 928 MiB constant,
  `unified_cache_nonfa_attn_outside_arena_reserve_bytes()` (jehw `ggml-sycl.cpp:18187-18246`,
  the constant's comment at `:18198-18224`). It is an estimate of consumers outside the arena,
  not a context-side demand, so it is not `transient_reserve`'s successor and H7p does not cover
  it. It is pre-existing and belongs to 23mk/zhcn (and llama.cpp-k1ev for the unattributed
  part). moua only moves the check before the yield (§2.4.2 step 4), where it runs only for an
  unplanned context (`!tenants_planned`), so the constant serves only unplanned contexts.
- The fattn-onednn.cpp:1011 `< 2^47` host-pointer heuristic is 1oxa's phase 1. moua's
  regions are USM device memory and are unaffected.

## 3. RED-first test plan

### 3.1 Host tests (no GPU; subagents can run them)

- **H1 `test-tlsf-allocator`** (L1, done and approved: `eab1ebeb6` … `456650c01`). Registered
  as `hostonly`, with 24 cases, including `allocate(SIZE_MAX)`'s refusal.
  - The A2 shape on the real header lands **0/23 interleaved and 23/23 segregated**.
  - L4 adds `allocate_excluding` cases, with range (not block) exclusion (r3 m3): a gap whose
    top is a pending range still serves a gap-front allocation below the range; a request that
    fits only across the range misses; the gap passed as an excluded range forces a hole; and
    `check_invariants()` holds after every split.
  - L4 adds `allocate_at` cases (r4 I8, m4): an exact interior range, a range at either end of
    a free block, a range crossing two blocks (refused), and a sub-`MIN_BLOCK_SIZE` remainder
    (absorbed by `carve_gap`'s rule).
  - `TAG_OPTIONAL != 0` is a `static_assert` where the tags are defined (r1 M6), because
    `frontier_walk` and `tag_at` use 0 to mean untagged.
- **H2 `test-kv-runtime-demotion`: A2 fixture** (L3). RED restated for the post-jehw base
  (r1 I6):
  - jehw (since `4d41db5c8`, kept at `c41fed119`) already holds `fit_capacity` to what lands, so the "admitted 23, landed 0"
    RED of revision 1 does not reproduce there. The fixture yields 0 admitted and 0 landed,
    and fit == land passes vacuously.
  - The RED is therefore a **capacity** RED: on the interleaved fixture, *expected 23
    device-resident layers, got 0*, in jehw's pipeline, which demotes all 32.
  - GREEN, on the segregated fixture through `kv_region_fit`, **recomputed from the head
    slots (r3 m8)**. Revision 4's formula (`2944 + transient_reserve − 84.1`) and C1's "23/9"
    predate the ruling. Let `H` be the sum of the head slots on the shared TLSF (ring, compute,
    per-op scratch; the fixture sets them from the producers' functions at A2's shape). After
    the full yield, A2's zone has 2954.1 MiB for KV and head slots, and 23 slots of 128 MiB
    leave 10.1 MiB. So the fixture asserts `device = 23 − ⌈max(0, H − 10.1 MiB) / 128 MiB⌉`,
    `host = 32 − device`, and a yield prefix equal to the minimal strict prefix covering
    `128 MiB × device + H − 84.1 MiB`. It runs at `H = 0` (23/9), at `H = 10 MiB` (23/9), at
    `H = 11 MiB` (22/10) and at the producers' A2 value. **That value is pinned (r4 m5):** H2
    records `H_A2` as a named constant computed from the producers' functions at A2's shape
    (zhcn's host replay for the compute chunks, beni's and jzvq's functions for the rest), and
    C1 scores the device count against `H_A2`, not against the `H` the run prints.
  - **Property test (P4):** seeded random sequences of WEIGHT/OPTIONAL/TRANSIENT/region
    requests and frees, on the real TLSF plus the `context_side` model. For every region
    request, `kv_region_fit(snapshot).fits` must hold exactly when the carve succeeds, and
    the carved extents and slot offsets must equal the fit's. jehw's `kv_zone_snapshot`
    simulation is the second oracle for the byte totals.
  - **Commit under churn (r2 N-I7; r4 I8).** Between the plan and the commit, weight
    allocations and another context's transaction run on the same TLSF. Cases:
    - churn after step 5 never touches the pending ranges, and the commit carves exactly the
      planned offsets;
    - **churn between the re-snapshot and the carve** (r4 I8's H2 case): a weight allocation
      and another transaction's carve run in that gap; the carve still succeeds, because both
      exclude this call's ranges. The RED is revision 5's step 7 (re-fit on the live geometry,
      then carve): the same interleaving makes its carve fail and reports `[KV-PLAN-BUG]`;
    - a weight allocation that lands in the planned room **before** step 5 records the ranges,
      and a pick that is not retired: each is a shortfall, the re-fit demotes inside the
      ranges, and nothing loops;
    - `[KV-PLAN-BUG]` fires only under the forced carve failure.
  - **The ring is a head slot (r3 I5).** A zone where KV plus the ring does not fit, but KV
    minus one layer plus the ring does: the fit demotes one KV layer and places the ring.
    RED: revision 4's order (KV first, the ring admitted after) refuses the context. A zone
    where the ring alone does not fit with every KV layer on the host refuses, naming the ring.
  - **Mandatory head slots (r3 C1).** A head slot is never dropped to fit KV; KV demotes first.
  - **The budget is not a second source (r4 I7).** A head slot that fits the geometry but not
    revision 5's budget room is placed, with no demotion beyond what the geometry needs. RED:
    revision 5's order (the fit, then the ring's budget-room check) refuses it.
  - **The ring-held yield (lead ruling, §2.7).** A Q4_0-dense MoE fixture: optional copies and
    the ring's KV-zone slots on one TLSF. The number of device KV layers equals what the room
    allows with the ring counted once, as its own slots reused in place. RED on the jehw-merge
    order (the ring's slots held while the yield models the KV zone, then re-admitted): it
    places fewer layers.
  - **The ring's RUNTIME half (r4 I6).** The fit places the ring's weight slots on the RUNTIME
    TLSF and records the split; a RUNTIME allocation made in the yield window cannot take them
    (pending ranges on the RUNTIME TLSF); the admit reads the recorded split. RED: revision 5,
    whose RUNTIME half was unreserved in the window.
  - **Demotion order and `free_after_full_kv` (zhcn GA).** A GPT-OSS-shaped fixture at zhcn's GA
    deficit, 726.9 MiB on the actual weights (11510.9 MiB; the breakdown is in §2.4.1, and the
    fixture builds each term of it), against 128 MiB full-attention slots: 5 × 128 = 640 is
    short and 6 × 128 = 768 covers it, with margins of 86.9 and 41.1 MiB, so head-slot rounding
    (under 512 B per slot) cannot flip it (§2.4.1). Exactly the 6 highest-indexed
    full-attention layers demote, K and V together, no SWA layer, `free_after_full_kv` reads the
    deficit, and each demoted layer's cause is `head_slot`. A variant with sub-slot weight holes
    demotes 7 and fires the sub-slot WARN.
  - **Recurrent state (r4 I9).** A hybrid shape: the RS slot is placed first on its device,
    attention KV demotes around it, and an RS slot that cannot fit with every KV layer on the
    host refuses the transaction, naming it.
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
      and reserved counted as used) at every step (addendum NEW-1).
  - **Reserved slots and the claim protocol (r3 C1, I7; r4 I1, I2, I3).**
    - A WEIGHT allocation (first-fit and `allocate_excluding`) never lands in a reserved slot.
    - A claim and its release do no TLSF operation and take neither the group mutex nor
      `kv_region_mutex_` (an instrumented-lock check); the slot's offset and size are unchanged
      across 100 claim/release cycles.
    - **The best-fit RED (r4 I1).** Slots {index 0: 100, index 1: 50}, i.e. the producer's two
      roles "large" and "small". Claims: X=40 (role large, index 0), Y=45 (role small, index 1),
      release X, then Z=90 (role large, index 0) while Y is live. By index the sequence is in plan
      and never misses. The positive control is revision 5's best-fit occupancy, which puts X in
      the 50 slot and Y in the 100 slot, and then has only the 50 slot for Z: the test must fail
      on it.
    - **Deferred-release churn property test (r4 I2).** Seeded random sequences of claims and
      releases within each owner's indexed slots, where each release carries an event that
      completes at a random later step (modelling `retain_handles_until_event` on a pipelined
      queue). A claim of a released index always succeeds and returns the releasing event; the
      model asserts every claim's kernel is ordered after that event, and that no host wait is
      ever issued. A claim of a still-claimed index is a plan violation, never a placement into
      another slot or another owner's slot. The positive control is revision 5's "vacate at
      completion": it reports a plan violation on the healthy pipelined sequence, so the test
      can fail.
    - **Held handles (r4 I3).** Dropping the owner's handle while a claim is live frees nothing
      until the claim's retained slice drops; there is no orphaned state to query. The positive
      control is revision 5's explicit-release model, which leaves the block owned by no one.
    - **Revision 4's span policy** (best fit over RETAINED, else below the anchor, bounded by
      peak live bytes) on A-small, B-large, free A, C-large still exceeds the peak, kept as the
      control for why slots exist at all.
    - **Record and replay (r5 I-C; lead ruling).** The model records a decode graph whose op
      claims index k of a cohort, replays it three times, and interleaves eager ops of the same
      cohort. It asserts that the eager ops claim from the cohort's eager indexes and never
      index k while the graph lives; that an eager claim of k is a plan violation; that no
      replay releases or re-claims k; and that destroying the graph vacates k on an eager event
      (the last replay's submission event, or a marker after it), which the next claim of k
      depends on. No node event is ever stored as a release event. The positive control is
      master's park in `ggml_sycl_pool_leg::free` (`11faace69` `ggml-sycl.cpp:43378-43381`),
      modelled as a vacate at record time: the eager op then claims k while replays still write
      it, and the test must fail.
    - **A graph-held lease at tenant-only step (i) (r5 I-G; queue R8 (4); zhcn step 4; r6).** A
      context whose own recorded graph holds tenant slot k, whose queued-work retention entries
      for k sit in the modelled retained store, and whose two backend contexts each cache the
      published slot table, republishes with k grown. (a) clears its own executable graphs, on
      its own thread with no instrumented lock held; (c) takes the registry's and both cached
      table pointers and sees the table's `use_count() == 1`; (d)'s reap drops the store
      entries referencing the batch's owners; (e) sees `use_count() == 1`; and the republish
      **passes**. No poll or sleep occurs (an instrumented clock). REDs: `c2613a688`'s (i), with
      no invalidation and no reap, whose (ii) refuses; and `f2e5606bc`'s (c), which leaves the
      cached tables, so (e) aborts under STRICT on this healthy run (r6 I-3). Further arms:
      - **no other context is touched (r6 I-2).** The model holds a second context B with a live
        executable graph, its own `graph_unwaitable` entries and a leaf-staging map in use on
        B's compute thread. After A's (i), all three are unchanged. The positive control models
        `sycl_exec_graph_clear_active`'s real effects (the global `graph_unwaitable` swap and the
        staging-map clear) as A's (a), and the arm must fail on it. A context with nothing
        recorded runs no clear at all;
      - a stray holder outside the store (a leaked copy, a per-op cache's park, a `host_task`
        capture) makes (e) report `[CONTEXT-PLAN-BUG]`, never `busy`;
      - the reap yields once to a modelled in-hand entry, reading only its immutable identity
        copy, and still returns;
      - **`graph_unwaitable` (zhcn rev 5's arms, cited):** a parked handle matching a moved-out
        CONTEXT slot is dropped by the COMPLETE reap, `unwaitable_dropped == 1`, and (e) passes;
        a forced `"command graph"` throw while the reap is yielding parks the in-hand entry, and
        the post-yield scan catches it; a non-matching parked handle is untouched;
      - **the backstop (r6 I-1).** A retention of slot k published on a second modelled queue
        that (s) does not cover, with its kernel still running: the reap waits the record, the
        modelled kernel's storage stays intact until the kernel completes, and the miss is
        `[CONTEXT-PLAN-BUG]` (a WARN, and an abort under STRICT, run in a subprocess); the BUG
        line carries the count 1 and the PRIVATE_TESTING backstop counter reads 1. On every
        other GREEN arm of H4 the counter reads 0 (rulings §D16; r7 m-6). RED: a COMPLETE reap
        that drops without the backstop, which frees under the queued kernel;
      - **the ring's bound (r6 I-4; rulings §M7 I-3).** Ring slot k was used once by two claims,
        so its slot-state retention of the last generation holds two slices that share the
        slot's control, and a sole-contributor growth fits only in the old ring's room.
        - **(0) on this case:** (0) passes, counting the old ring free by arithmetic; afterwards
          `pinned[k]` is 0 and the guard holds no ring copy. RED: a (0) that runs step 2's
          ring-lock section, whose copy keeps `use_count` at 2 into (ii), which refuses.
        - **The move-out and (d):** the move-out takes the retention separately from the slot
          handles; (d) backstops its `done_events[k]` and drops it before (e); (e) sees
          `use_count() == 1` with P = 0; (f) frees the blocks; and (ii) places the grown ring.
          REDs: `f2e5606bc`, whose (ii) refuses with no demotion and no reason; and 7.5's order,
          which put the retention into the batch, so (e) sees `1 + 2` and reports
          `[CONTEXT-PLAN-BUG]` on this healthy run.
        - **A nonzero P is a bug (r7 m-3):** a test hook skips L0 for a second modelled
          transaction, whose guard pins slot k; the move-out snapshots P = 1, and (e) reports
          `[CONTEXT-PLAN-BUG]` even though `use_count() ≤ 1 + P` holds. A stray copy with P = 0
          (`use_count` 2) reports it too;
      - **the freed room is held (rulings §M7 I-5(a)).** A growth whose grown slot lands in its
        old slot's room: between (f) and (ii) a modelled expert-cache fill and a modelled lazy
        MoE layout materialization, each through `allocate_excluding` and taking no L0, try to
        allocate exactly that room. Both are placed elsewhere or miss, and (ii) places exactly
        what (0) placed. RED: 7.5's order, which recorded no pending range before (f), so the
        fill takes the room and (ii) refuses a within-plan candidate;
    - **The covered path (r5 I-B; zhcn rev 5; r6 m-10; r7 m-11, m-1).** A candidate whose every
      slot fits its published cap, host slots included, and whose ring contribution does not
      grow, gets COVERED from the coverage query: no L0 (the modelled L0 records no acquire), no
      step (i), no fit, no allocation, no release, no registry or ring-record write, and the
      published tenant key unchanged. A candidate whose tenant slots are covered but whose ring
      contribution grows gets GROWTH and takes L0 and step (i). A candidate with one slot over
      its cap takes step (i); its host slots are reused and nothing is allocated on the host
      tier. REDs: `b30321a6f`, which re-allocates every host slot on any key change; and 7.5's
      covered rule, which sends the ring-growth candidate down the covered path, so its next PP
      MoE claim finds a ring too small.
    - **After a refused (ii) (r6 m-9).** A sole-contributor growth that (ii) refuses leaves the
      ring record empty and the context without slots. The next PP MoE claim reports
      `[CONTEXT-PLAN-BUG]` with an error status and the op is not skipped; the ladder revert,
      whose key no longer matches (it was cleared at (c)), re-carves the previous candidate's
      slots and ring. The positive control keeps the tenant key through (c), so the revert hits
      the equal-key no-op and returns OK with nothing carved.
    - **Host-tier order (r5 I-I(3); rulings §D15).** A republish whose host need exceeds the
      context's host reservation: (0) refuses by arithmetic, with every device tenant slot and
      every host slot still held, and no host allocation is attempted. The concurrent-fill
      variant of earlier revisions is deleted: within the plan there is no host allocation left
      for a fill to race (the next arm). RED: `c2613a688`'s order (host allocation after (i)),
      which releases every device slot first.
    - **The tenant-key no-op (r5 queue R8 (6); r5 m-j).** A sampler change that leaves every
      slot's bytes equal republishes the same tenant key: the coverage query answers EQUAL and
      llama keeps the published residency, with no probe, no step (i), no registry change, no
      L0 and no L1, and the
      count of tenant-only republishes is 0. In a split model, the second backend's run of one
      publish finds the first's entry and is the same no-op. RED: `c2613a688`, which sent every
      matched key to the tenant-only path.
    - **Claim-index order (zhcn A1).** Within one ALLOC per `(ContextId, buft)`, indexes 0, 1, 2
      pass; the orders 0, 2 and 1, 0, and an index at or above the measured chunk count, each
      report `[CONTEXT-PLAN-BUG]`, formatted and logged after the spin lock is released.
    - **The host reservation is held (r5 queue R7b; rulings §D15; r7 I-6).** The first publish
      allocates one host carve sized to the plan's maximum tenant host demand, with growth
      allowed, and sub-carves the host slots from it at the maximum caps. Then, after a modelled
      decode:
      - **a concurrent host fill fails to take the room:** a modelled expert-cache host fill,
        taking no L0, asks the pinned pool for every free byte; it gets the room outside the
        reservation and not one byte of the reservation. A republish at the plan's top rung
        then succeeds, allocates nothing on the host tier, and an instrumented `grow_zone`
        counter stays at 0. RED: revision 7.5's recorded-only reservation (host slots at the
        first candidate's size, later republishes allocating with `forbid_host_zone_growth =
        true`), where the fill takes the room and the top-rung republish refuses on live free
        room;
      - a republish whose host need exceeds the reservation is refused at (0) as a candidate
        refusal naming the tenants, by arithmetic, whatever the pool's live free room (the arm
        runs it with the pool both full and empty, and the answer is the same); `grow_zone`
        stays at 0. RED: the first draft's host path, which let every republish grow the pool.
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
    - **gap before holes, RED against the addendum's literal order (r3 m1):** a gap and a free
      weight hole that can each hold the slot. The slot goes to the gap. An implementation of the
      addendum's literal list (holes before the frontier) fails this case;
    - **the head slots' room (r3 m1):** a gap that holds the region but not the region plus the
      head slots, and a free weight hole that holds one KV slot. The KV slot goes to the hole,
      the head slots stay in the gap, and nothing is yielded. Revision 4's post-check greedy
      filled the gap with KV and then demoted or yielded;
    - when both yield candidates exist, the one with less layout bytes lost per slot goes
      first, and a tie goes to the frontier.
  - **Second server swap (addendum (c)):** model 1 unloads, a model-2 context's KV takes model
    1's hole, then model 3 loads. Model 3's weights bury the ladder, the WARN fires, and the
    fit stays exact.
  - **Pending ranges (addendum (b); r3 C2(a); r4 m1):** between step 5 and the carve, a WEIGHT
    allocation that would fit a pending range is placed elsewhere or misses, **with and
    without an optional ladder on the TLSF** (the no-ladder case is r4 m1's RED: revision 5's
    plain `allocate` front-carves into the range). A second context's fit, run in the yield
    window, sees the ranges as allocated and does not plan into them. A rollback clears the
    ranges, and a settle is refused while any exists (r4 m10).
  - **The reserve holds after the commit (r3 C1):** after the commit, an EXPERT_CACHE fill that
    would take the head slots' room is placed elsewhere or misses, and the compute buffer then
    lands in its reserved slot. An unplanned context-side request cannot take a reserved slot
    either.
  - **Leased copies and retire-on-request (r3 C2(c); r4 addendum; §2.9):**
    - (a) a lease-vetoed copy truncates the ladder, the layer it would have covered demotes,
      and the fit reports the copy and the layer for the WARN;
    - (b) the request retires the copy; a holder that re-records resolves the primary:
      `acquire_entry_lease` refuses the copy, `acquire_layout_handle` finds nothing, and the
      WOQ gemm takes the primary, so the lease is not re-taken (jehw's (e), code-free on jehw
      HEAD). A retired copy still leased is allocated and not yieldable, and the strict prefix
      truncates at it;
    - (c) after the holder's clear drops the last lease, the block is free, and a **new**
      context's fit uses it; the requesting context's demoted layer stays on the host;
    - the request fires at most once per transaction, never on a refused one, and never when
      the transaction demoted nothing the vetoed copies would have covered.
- **H6 N-chunk.** Two weight TLSFs plus the tail KV TLSF, checking greedy packing across
  TLSFs and extents, fit == carve, and the TRANSIENT routing: WEIGHT-naming requests go to the
  last weight chunk, KV-naming ones to the KV TLSF, and each record's head slots are placed and
  carved on its zone's TLSF (r2 m7; the per-TLSF reserve is gone, r3 C1).
- **H7 source gates** (python; the kv-layer-sizing family plus a new
  `test-sycl-kv-region-source.py`, each check with a mutation witness):
  - (a) the region reservation never reaches `unified_cache_malloc_device_tracked`;
  - (b) the tiered device-planned branch issues no per-layer `unified_alloc`;
  - (c) the optional pass runs after all S1 staging, dense *and* expert/DPAS;
  - (d) **one source, by primitive (r2 N-I4)**: the capacity primitives listed in §2.2 occur
    only at allowlisted sites, including `ggml_sycl_device_vram_budget_room` in no in-arena
    head-slot admission (r4 I7). The deleted
    arena-device uses of `kv_admission_mismatch`, `kv_vram_cap` and `kv_device_budget` stay
    deleted. The one byte function `kv_layer_tensor_bytes` is the only per-layer KV sizer that
    feeds the fit or the claim;
  - (e) `[KV-PLAN-BUG]` is logged at ERROR at both sites, `[CONTEXT-PLAN-BUG]` at every claim
    site, and `GGML_SYCL_STRICT_PLAN` aborts at each; `GGML_SYCL_STRICT_KV_PLAN` does not occur,
    in moua's files or zhcn's (no alias, r4 I3);
  - (f) no raw pointer is stored as region state;
  - (g) `reserve_kv_region` does not hold the group mutex across the yield call (C2); the
    pending ranges are recorded before the yield window's `lock.unlock()` (r3 C2(a)) and cover
    every placement (r4 I8); the commit re-fit passes `own_ranges` and the carve uses
    `allocate_at` only (r4 I8);
  - (h) no dispatch path re-stages a yielded **or 423j-retired** optional copy (r5 m-m).
    Dispatch reads the primary's materialized layout (audit m4);
  - (i) the slot view's pointer comes from `slice().resolve()`, and `set_owner` has the
    `mem_handle` overload (audit m1);
  - (j) the new entry points appear in contract §3's allowlist, and their allocation class
    is derived, not hand-set (audit m2);
  - (k) `kv_region_mutex_` is leaf: no lock is acquired, and no `mem_handle` is destroyed,
    inside its scopes, and no `g_pending_kv_layer_masks_mutex` scope contains one (r2 N-I1,
    addendum); the tenant-only path's (i)(c) reads only the atomic claimed flag under it, never
    the per-slot spin lock (r6 m-8);
  - (l) the transaction's `kv_region_txn` guard is declared before its L1 `std::unique_lock`;
    the registry commit and the superseded-slot release follow the publication CAS (r2 N-I3,
    r4 I5); the ring's release half is not called in the full transaction (r4 I6); the guard's
    destructor is `noexcept` and takes no L1 (r4 m14);
  - (m) llama calls `ggml_backend_sycl_kv_region_release` **only** from the destructor of the
    `sycl_plan_guard` member, which is declared before `sched` and passes the ContextId it
    captured at `create_exec`, never the `sycl_exec_context` member; no backend close function
    reaches the proc, it never runs under `g_execution_backend_binding_mutex`, and it contains no
    live-lease assertion (r3 I6; r4 m9; zhcn T1-T3, M1; r5 I-F). Mutation witnesses: revision 4's
    erase inside `clear_bindings_for_context`, revision 5's call guarded by `rc == OK`, revision
    6's second call at the drain-and-close tail, and a guard that reads the zeroed member;
  - (n) `WEIGHT_SIDE_TRANSIENT` appears at exactly the two B50-commented sites (r2 N-I8);
  - (o) llama's later publishes send the stored KV-shape section, never a recomputed one
    (r2 N-I9), and the descriptor's layout rules hold (append-only, `struct_size`-gated,
    per-array element strides) for every section (§2.4.4);
  - (p) **widened (lead ruling 3; r4 I10(b), m11):** every context-side site is either a
    converted claim (its cohort has a `context_side_demand` producer, and it calls
    `claim_slot` with an index) or on the explicit unconverted-site list, which only shrinks and
    must be empty after beni's last conversion (§2.4.3's transition rule); every context-side
    admission (the ring's, the compute buffer's, any cohort's) reads only planned records, with
    no estimate constant, per-row reserve or floor; and every claim site's failure branch
    returns an error status to the graph, reaches no allocator, and never skips the op;
  - (q) the fit's optional census and the yield classify optional tenants only through
    `optional_layout_yieldable_locked`; no other reclaimability test on an optional tenant
    exists in `unified-cache.cpp` or `ggml-sycl.cpp` (r3 C2(b));
  - (r) `k_pp_moe_ring_compute_reserve_bytes_per_row` does not occur (r3 I8);
  - (s) the non-FA check, the budget steps and every other predictable refusal precede the
    yield in the transaction body (r3 m4);
  - (t) no `mem_handle` is destroyed while any listed lock is held by the guard, the publish,
    the tenant-only path's step (i), or the release proc (r3 I4);
  - (u) the step-6 carve never reaches `unified_alloc` or `evict_and_flush`; the controls it
    binds were minted before L1 (r4 m7); no host-tier allocation runs under L1 (§2.4.3);
  - (v) `context_side_place` does not occur, and no claim site retries into unreserved room
    (r4 I3);
  - (w) `claim_slot`/`release_claim` acquire neither the group mutex nor `kv_region_mutex_`
    (r4 I2), and nothing allocates or logs under the per-slot spin lock: the
    `[CONTEXT-PLAN-BUG]` line and the STRICT abort follow its release (r5 m-k);
  - (x) llama's recurrent constructor allocates through `llama_recurrent_sycl_kv_buft`, which
    returns the recurrent-state buft for an arena device (r4 I9);
  - (y) `vram_bytes`/`per_device_vram` change for a context-side cohort only at the carve's
    charge and the real free's uncharge, and the host inventory likewise for
    `context-compute-host`; claim and release sites touch neither, and u1bb's ring charging
    site is gone for arena devices (zhcn A2). Mutation witnesses: an uncharge added at
    `release_claim`, and u1bb's charge restored;
  - (z) the ring's claim contains no host wait: `pp_moe_onednn_claim_scratch_slot` returns the
    slot's release event for `depends_on` (§2.7), a re-claim hands the previous generation's
    `retained_owners` to `retain_handles_until_event` before recording the new one, and the
    cache-side per-generation refcount is gone for arena devices (r5 I-K); **every** removal of
    a slot's `retained_owners`, at the reset, the generation-0 branch, release-unused, rollback
    and the re-claim, moves the vector out and hands it to `retain_handles_until_event` after the
    slot-state mutex is released (r6 I-6); mutation witnesses, master's `wait_and_throw`, master's
    overwrite of `retained_owners`, and master's `assign(ring_depth, {})` in the reset;
  - (aa) **every placement on a TLSF that can carry pending ranges honours them (r5 m-d).**
    Every `allocate`, `allocate_gap_front` and `allocate_below` call on a shared-zone or RUNTIME
    TLSF goes through the range-excluding form whenever a pending range exists: the WEIGHT
    first-fit, `zone_alloc_optional`, `backend-buffer-runtime-zone`, the MMID pools and staging.
    The gate enumerates the call sites of the three primitives in `unified-cache.cpp` and
    `ggml-sycl.cpp` and fails on any site on those TLSFs not on the excluding list. Mutation
    witness: the pending-range test removed from `zone_alloc_optional`;
  - (ab) **no record-mode release parks a claim without the lifetime rule (r5 I-C).** The
    record-mode branch of `ggml_sycl_pool_leg::free` hands the claim to the executable graph,
    and the graph's destruction vacates it on an eager event; no path stores a graph-node event
    as a slot's release event. Mutation witness: master's park (`11faace69`
    `ggml-sycl.cpp:43378-43381`);
  - (ac) **the MMID budget demotion reads no in-arena head-slot bytes (r5 m-b).** jehw's
    `BUDGET_EXCEEDED` demotion (`:18118`) excludes in-arena head-slot bytes from its sum.
    Mutation witness: the head-slot term added back;
  - (ad) **the host tier is reached only through `unified_allocate_owner` (r5 queue R7a, R7b).**
    The `context-compute-host` reservation is allocated only by `unified_allocate_owner`, only
    at the first publish, and no later republish reaches a host allocation (see (an)). Mutation
    witnesses: a direct `pinned_chunk_pool` grow call, and a republish that allocates;
  - (ae) **the tenant-only release is a reap, not a wait (lead ruling "B").** The (i) sequence
    contains no sleep, poll loop or timeout; `release_retained_referencing` is called with the
    moved-out batch's owners before the batch is dropped and with no L1-L5 lock held; and the
    `use_count()` check's failure branch is `[CONTEXT-PLAN-BUG]`, not `busy`; the reap's
    moved-out records pass the backstop (status query, wait on an incomplete event, the BUG
    line) before they are dropped (r6 I-1). Mutation witnesses: rev 7's wait-1 loop, a drop
    placed before the reap, and a COMPLETE drop with no backstop;
  - (af) **(i) reaches no process-global clear (r6 I-2).** Neither
    `release_graph_retained_handles` nor `ggml_sycl_cpu_staging_cache_clear` is reachable from
    the tenant-only path's step (i), and (a)'s own-context clear runs only behind the
    recorded-graph gate. Mutation witness: (a) calling `sycl_exec_graph_clear_active`;
  - (ag) **a tenant slice has three permitted holders (rulings §B, §B.2; r6 I-3).** A CONTEXT
    tenant slice is retained only by a claim, by `retain_handles_until_event` or the graph sink,
    or by a per-context graph container that (a) clears. No `host_task` lambda captures a
    tenant-slot handle (zhcn rev 5). The gate runs over every converted claim site. **Any other
    park that owns a handle of a tenant slice fails** (rulings §B.2; r6 I-3): the oneDNN Graph
    scratch park, any per-op cache beni converts, the four former §B.1 members (the MMVQ and MoE
    q8 `cached_src_handle`, the `moe_ids_cache` keys, `runtime_tensor_extras`) if they still own
    one, and, by name (rulings §M7 I-7), `graph_input_staging` on the eager path and in record
    mode, `g_moe_ids_d2h_cache`, `g_moe_prompt_admission_cache` and `g_data_ptr_cache` (§2.4.2
    "The holder census"). The identity type's no-recycling test and the `runtime_tensor_extras`
    publisher census are zhcn's gate 27 (rulings §B.2). Mutation witnesses: a beni-style owning
    park in a per-op cache, an owning `cached_src_handle` restored, a
    `ggml_backend_sycl_release_buffer_refs` step reintroduced, a `host_task` capture, and one
    per named container with an owning entry restored: an owning `graph_input_staging` handle
    with nothing recorded, an owning `moe_ids_cache_key::handle` in each of the two maps, and an
    owning `g_data_ptr_cache` value with zhcn's exit clear present (the key fix alone must catch
    it);
  - (ah) **step 2's ring-lock section precedes the snapshot (r6 I-5).** The copy of
    `ring_plan_gen` and of the ring's slots, with the `pinned[slot]` increments, is taken before
    the geometry snapshot, and the fit's ring input is that copy. Mutation witness: revision
    7.1's order (snapshot, fit, then the ring-lock section);
  - (ai) **every publish and live-update preparation is under L0 or allowlisted (rulings §E.1,
    §E.2, §L0R; r7 I-1, I-2, m-7).** The census is by call site, not by operation name: every
    call of `ggml_sycl_publish_plan_locked`, `ggml_sycl_publish_prepared_plan_locked`,
    `lifecycle_replace_placement_plan` and `Registry::prepare_live_update` must be reachable
    only under a `ggml_sycl_replan_token` taken at the top of its public entry point (§2.4.2
    "The re-plan transaction mutex" lists them), or be one of the four allowlisted
    identity-preserving republishes (`:5887`, `:5908`, `:60125`, `:74882`), whose helper takes
    no snapshot argument and whose debug identity check stays in. Beside it: the token is the
    only way to take `g_replan_txn_mutex`; it is taken before L1 and never under an L1-L5 lock;
    graph compute and dispatch never take it (a `sched_reserve` re-plan does, legitimately); the
    release proc's token is the outermost-only kind; no wrapper or transaction return maps to
    `BUSY`; llama has no busy sleep loop; the probe and the FA recheck read
    `lifecycle_select_placement_plan(model)`, not the global snapshot; and no per-device re-plan
    mutex exists. Mutation witnesses: a publisher call site without a token, a fifth
    identity-preserving republish that passes a new snapshot, the mutex taken under L1, a
    graph-compute path taking it, a restored wrapper `BUSY` return, a restored llama sleep loop,
    the probe reading the global snapshot, and a per-device mutex array;
  - (aj) **a model load does not touch the ring (llama.cpp-r7fz; rulings §M7 I-4).** On the
    arena path, `release_pp_moe_onednn_scratch_ring` and the planned-ring setters
    (`unified_cache_set_planned_pp_moe_onednn_scratch`, `_kv_zone_slots`, `_row_bytes`,
    `_n_ubatch`) are reachable only from the release proc, the sole-contributor move-out and the
    commit's install. Mutation witness: the `populate_inventory_globals` release restored;
  - (ak) **(0) takes no ring-lock copies or pins, and (d) drops the ring's retentions before (e)
    (rulings §M7 I-3).** Mutation witnesses: (0) calling step 2's ring-lock section, and the
    move-out putting `retained_owners` into the batch;
  - (al) **the freed room is pending before (f) (rulings §M7 I-5(a)).** (0)'s placements are
    recorded as pending ranges before any release, and (ii) re-fits only inside them. Mutation
    witness: the recording moved after (f);
  - (am) **step 7 allocates nothing (rulings §M7 I-5(b)).** The MMID materialization on the
    transaction path takes its device pool from step 6's carve and its host pool from the
    model's held carve; no `unified_alloc`, `unified_allocate_owner` or pool growth is reachable
    from it. Mutation witness: master's `unified_cache_materialize_moe_mmid_workspaces`
    allocation restored at `:18614`;
  - (an) **the host tier allocates only at the first publish, from one held carve (rulings
    §D15; r7 I-6).** The host slots are views of the reservation, sized at the plan's maximum
    caps, and no tenant-only path reaches a host allocation. Mutation witness: 7.5's
    per-republish allocation with `forbid_host_zone_growth = true`.
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
  - **Mixed device set (r3 m9):** a two-device plan where only one device has an active
    arena. A layer planned on the other device reads device-resident from the plan, not host
    from an empty registry.
  - **Two contexts before `sched_reserve` (r3 I2):** A's transaction, then B's, then A's
    compute buffer. A's buffer lands in A's reserved slot, and B's fit never planned into it.
    The positive control keys the meter by `(device, TLSF, cohort)` as revision 4 did, and B's
    reserve then under-counts by A's live compute buffer.
  - **Concurrent per-op claims (r4 I4(b)).** Two contexts on one device, each on its own thread,
    claim the same per-op cohort at the same time: each claims its own CONTEXT slot, and neither
    sees a plan violation. The positive control is revision 5's DEVICE-scope per-op slot, where
    the second claim is over plan.
  - **The ring across contexts (r4 I4(c); r6 m-1).** Context A reserves with `-ub 1024`, B with
    512: the ring stays at A's size. A closes: its contribution is removed and the ring keeps
    A's size, the excess over B's held as planned room (it never shrinks by carving, §2.7), and
    a later transaction on the device carves nothing for it. B closes: B was the last
    contributor, so the ring is released. RED: revision 5's "last re-plan wins", which shrinks it
    at B's reservation, under A.
- **H9 transaction guard (r2 N-I3; r3 I4, I6; r4 I5, I6, m9, m14; SYCL-free, in
  `kv-region-registry.hpp`).** A host model of the §2.4.2 steps, with a failpoint at every
  refusing step (the fit's head-slot refusal, the RELEASING `[CONTEXT-PLAN-BUG]`, the accounting
  step, the non-FA check, the yield's relock (a `[CONTEXT-PLAN-BUG]` under L0, forced by the
  skip-L0 hook below), after the carve of device 0 of 2,
  the MMID step, the ring check of step 8 (a), and the CAS). At each failpoint it asserts:
  - the registry and the ring record are unchanged;
  - no yield ran for a failpoint before step 5;
  - this call's pending ranges are cleared under the instrumented group mutex, with the
    instrumented L1 **not** held (the guard's first phase takes no L1);
  - every handle this call carved, and every unused pre-minted control, is dropped, and dropped
    only after the instrumented L1, registry, group and binding locks are released;
  - **at the MMID and CAS failpoints, the ring is in its original slots** (the same handles,
    offsets and claim state as before the call), and a ring claim taken before the call is
    still valid (r4 I5). RED: revision 5's guard, whose ring slots were released at its step 7.

  **Serialized, with no `busy` (rulings §E.1, §E.2).** The model's process-global re-plan mutex
  (L0) is instrumented, and the model checks the lock order L0 before L1 on every path.
  - **Two contexts re-plan concurrently, on one device and on two devices**, in every pairing of
    a tenant-only republish, a sole-contributor ring release, a full-context publish and a
    teardown release proc, plus a re-plan on one device racing a model load on the other. The
    model's published plan is one global atomic, as jehw's is. Every call completes with **zero
    `busy` returns and zero lost CASes**, and each call observes the other's publish or refusal
    whole. RED: revision 7.2, which had no L0 and returned `busy` from step 2 and step 8 (a);
    and a per-device L0 (revision 7.3's first form), under which the two-device pairings trip
    the plan-identity checks and lose the CAS.
  - **Teardown's path.** The release proc runs from the guard member's destructor while another
    thread holds L0: it waits, then runs. A same-thread re-entry (a release proc
    reached with this thread already holding L0) trips the outermost-only token's debug check
    and aborts, instead of hanging.
  - **Legitimate nesting (rulings §L0R; r7 I-2), positive.** llama's `replan_scope` takes the
    token; inside it the growth path's (0) probe, the transaction wrapper and (ii) each acquire
    it again. The nested acquires do not lock (the instrumented mutex records one lock and one
    unlock), the outermost token unlocks, and another thread's acquire blocks until then. RED: a
    plain `std::mutex` in place of the token, which self-deadlocks (the model detects it with a
    try-lock and reports it).
  - **A in a transaction, then B loads, then A's probe and FA recheck (rulings §L0R; r7 I-1).**
    Two modelled models. A opens a transaction and publishes; B's load entries run and publish
    B's plan; A then runs its probe and its FA recheck. Both validate against A's own plan
    (`lifecycle_select_placement_plan(A)`) and answer OK, with zero STALE_IDENTITY. RED:
    master's probe and recheck, which compare A's token with the one global snapshot and answer
    STALE_IDENTITY.
  - **Load B while A contributes (llama.cpp-r7fz; rulings §M7 I-4).** Model A's context is a
    ring contributor with a claimed-then-vacated ring on device 0; model B loads on device 0.
    After B's load A's ring handles, sizes, depth, split flags and contribution are unchanged,
    and A's next PP MoE claim succeeds. RED: master's `populate_inventory_globals`, which
    releases the ring, so A's claim finds it empty and reports `[CONTEXT-PLAN-BUG]`.
  - **Every publisher under L0 (rulings §L0R).** The model runs, against a parked L0 holder,
    each public entry point §2.4.2 lists (the wrapper, activate, unload with its failure
    republish, quarantine restore, each load entry): each blocks until L0 is released. The four
    allowlisted republishes run against the parked holder without blocking, and each leaves the
    plan identity unchanged; a modelled republish that changes it reports
    `[CONTEXT-PLAN-BUG]`.

  **No absent ring (r4 I6), and the routes that used to answer `busy` (r5 I-A; r6 I-5).** Under
  L0 none of these interleavings can occur, so each is forced by a test hook that lets one
  mutator **skip** L0 (the only way the state can arise). In each, the victim reports
  `[CONTEXT-PLAN-BUG]` (an abort under the STRICT model), never `busy`, and never publishes a
  ring nothing holds:
  - **RELEASING observed at step 2:** a tenant-only sole-contributor release marks the ring
    RELEASING, and a second transaction reaches step 2. RED: revision 5, where B, running in A's
    window with an unchanged demand, published a ring A had released;
  - **teardown between step 2 and step 8:** B plans the ring as reuse in place; A, the last
    contributor, is released by its `sycl_plan_guard` in B's yield window; B's step 8 (a) sees
    the new generation. B's reused-handle copies keep the ring's blocks allocated, so no weight
    can take the room before B's guard drops them. RED: `b021c9629`, where B publishes a ring
    with no slots;
  - **RELEASING raised after a transaction's step 2:** a sole-contributor step (i) runs while B
    is past step 2; B's step 8 (a) sees it;
  - **a release proc between B's ring-lock section and its step 8 (r6 I-5).** The last
    contributor's release proc runs its steps 2-4 at each point after B's ring-lock section:
    between it and the snapshot, between the snapshot and the fit, and in the yield window. B's
    copies keep the planned blocks allocated, and its step 8 (a) sees the new generation. RED:
    `f2e5606bc`'s order, where step 4 landing between B's snapshot and its ring-lock section
    lets B copy nothing, match the new generation and publish an absent ring.

  **A (ii) refusal after a sole-contributor (i)** (no hook needed): the guard's first phase
  clears RELEASING and bumps the generation; the next transaction on the device, once it gets
  L0, passes step 2. RED: the flag stays set and every later transaction reports
  `[CONTEXT-PLAN-BUG]`.

  **Ring growth and shrink (r5 I-B):** a growth fixture whose overlap demotes exactly the layers
  the old ring's bytes cover, each labelled `ring-growth` with the old and new bytes (RED: the
  first draft's `capacity` label); a shrink fixture (a contributor leaves, a later context needs
  less) that carves nothing, demotes nothing and refuses nothing (RED: the same-size reuse rule,
  which carved a smaller ring beside the old one).

  **Superseded drops after L1 (r5 m-a):** at a committed publish, the superseded handles are
  dropped with the instrumented L1 released, including a retire's withdrawn mirrors (r5 I-J).
  RED: the first draft's drop inside step 8 under L1.

  **RELEASING has an owner (r5 I-A, sharpened):** a sole-contributor tenant-only republish that
  marked RELEASING passes its own (ii) step 2 and step 8 (a); a second transaction, anywhere in
  the process, waits on L0 and then finds the mark cleared; the owner's guard clears the mark on
  a refusal and on an injected exception. RED: `c2613a688`'s ownerless flag, on which the owner
  answers `busy` to itself.

  **Two ring generations on one slot (r5 I-K; r6 I-6):** slot k is re-claimed while its previous
  generation's event is incomplete, then a ring growth supersedes the old slots and the
  publish drops them. The previous kernel's storage (modelled) stays allocated until that
  event completes. RED: master's `retained_owners` overwrite, which frees it under the queued
  kernel. A variant changes the ring's depth at the next claim, so the reset path runs: the
  previous generation's slices are handed off, not dropped, and the storage again survives
  until the event. RED: master's `retained_owners.assign(ring_depth, {})` under the slot-state
  mutex.

  **Superseded slots (r4 I5):** a transaction that grows the ring carves new slots beside the
  old ones; the old ones are dropped only after the CAS, and a refusal at the CAS leaves them in
  place and drops the new ones.

  **Different key (r3 m11):** a matched ContextId republishing a different key is refused, and
  nothing is carved or moved out.

  **Tenant-only step (i):** a still-claimed tenant slot is `[CONTEXT-PLAN-BUG]` and nothing is
  released; an unclaimed one is dropped with no instrumented lock held.

  **Teardown at the single site (r4 m9; zhcn M1; r5 I-F):** the modelled `sycl_plan_guard`
  destructor calls the release proc with the ContextId it captured at `create_exec`. After a
  successful close, the proc empties `(c, *)`, removes `c`'s ring contribution under L1 and the
  ring lock (if `c` was the last, it sets RELEASING and moves the ring handles out), drops
  everything with no instrumented lock held, then clears RELEASING and bumps the generation.
  The same holds when the modelled drain-and-close returns early with BUSY, STALE or a
  finish-drain error, and on the construction unwind that zeroes `sycl_exec_context` before a
  `create_memory` refusal. RED: revision 5's extract, which runs only on OK, and revision 6's
  first draft, whose drain-tail call is skipped by the early returns and whose guard read the
  zeroed member. A second release call is a no-op. With a compute-buffer slice still retained,
  the release succeeds without asserting and the block is freed when that slice drops.

  **Destructor lock failure (r4 m14):** a failpoint makes the group mutex's `lock()` throw in the
  destructor; the process aborts with the named message instead of propagating (run in a
  subprocess).

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
- **reserved slots and claims** (§2.3.2, §2.4.3): a per-op claim of its cohort's index moves
  `reserved_slot_claims` and its pointer is inside the slot; two claims of one index in a
  pipelined sequence are ordered by the returned event, with no host wait. A forced over-cap
  claim logs `[CONTEXT-PLAN-BUG]` once per `(device, owner, cohort)`, the op returns a failure
  status that `graph_compute` reports (the decode fails; nothing is skipped), no unreserved room
  is taken, and the external-bytes counter does not grow. `GGML_SYCL_STRICT_PLAN=1` aborts in a
  subprocess;
- **graph-held copies (r3 C2(c), §2.9; r4 m6):** context 1 records and replays a graph that
  reads WOQ copies; context 2 is created with KV that needs the room. In both arms context 2
  demotes, and its WARN names the leased copies, their bytes and the demoted layers (case (a)).
  The rest is **keyed on whether 423j has landed**, read from a `GGML_SYCL_PRIVATE_TESTING`
  accessor (`unified_cache_test_graph_reclaim_available()`), never inferred from the outcome:
  - with 423j: the copies are retired (b); context 1's next compute re-records against the
    primaries and gives the same output as before; after it, the copies' blocks are free, and a
    context 3 created then places the KV layers context 2 demoted (c). Context 2's layers stay
    on the host;
  - without it: nothing is retired, the copies stay leased, and context 3 demotes as context 2
    did.

  Revision 5 "recorded which" arm happened, which passes either way;
- **record/replay claim lifetime (r5 I-C):** a context records a decode graph and replays it,
  interleaved with an eager op of the same cohort. The eager op claims its own index, with
  `[CONTEXT-PLAN-BUG]` = 0 under `GGML_SYCL_STRICT_PLAN=1`; destroying the graph moves
  `reserved_slot_claims` back and returns an eager event (a test accessor reports its kind);
  and a forced eager claim of the graph's index while the graph lives logs
  `[CONTEXT-PLAN-BUG]` and fails the op. It prints the bytes the record-mode index sets add to
  the context's planned context-side demand (lead ruling on the cost), so plan == reality is
  visible per run.

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
  cat c1.err | grep 'KV overflow re-placed'               # predicted: see below
  cat c1.err | grep 'KV-REGION'                           # the head-slot sum H per TLSF
  ```
  - **Positive control.** The same command on the base (post-jehw/u1bb master) prints
    `role=2` EXT-ALLOC lines, or demotes all 32 layers under jehw. The fe6c run printed 23
    such lines. That shows the trace is armed.
  - **Non-KV EXT-ALLOC bytes** (total minus the KV role) must not exceed the base run's.
    That is the check for r1 C3(ii).
  - **Predicted, from a pinned head-slot sum (r3 m8; r4 m5):** `device = 23 − ⌈max(0, H_A2 −
    10.1 MB) / 128 MB⌉` layers in the region and the rest on the host (23/9 only if
    `H_A2 ≤ 10.1 MB`), and output starting `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`. `H_A2` is H2's
    pinned constant, computed on the host from the producers' functions before C1 runs (zhcn's
    replay needs `/models`, so the lead runs it, CPU-only, and records the value on the ticket
    first). C1 passes only if the run's `GGML_SYCL_KV_REGION_TRACE=1` line prints `H == H_A2`
    **and** the device count matches the formula at `H_A2`. Revision 5 scored against the `H`
    the run printed, so an inflated `H` still passed.
- **C2 Mistral gate:** B50 and B70, the CLAUDE.md command.
- **C2a B50 MMVQ/STAGING first submit (r2 N-I8), a named acceptance item.** The C2 B50 run,
  on `level_zero:1` at the **default** PCT (no `GGML_SYCL_VRAM_BUDGET_PCT`), is where the
  region and the per-op MMVQ/STAGING scratch sit at the top of a mostly empty shared zone,
  directly below ONEDNN.
  - Pass: rc=0, the digit output, zero aborts, and no `UR_RESULT_ERROR` in stderr.
  - Its **baseline** is the base tree, where those pages are untouched (r3 m5: a baseline
    run, not a positive control, since nothing in it is known to fail). On a failure, rerun
    with `GGML_SYCL_B50_SCRATCH_WEIGHT_SIDE=1` (§2.1's lever). A pass there makes the lever
    the default for the two B50 sites.
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
  - It adds a **no-replay decode arm** (r2 m8; r4 I2): Mistral tg128 with
    `GGML_SYCL_DISABLE_GRAPH=1`. In that mode every per-op claim runs per dispatch: an atomic
    claim of its index and an event hand-off, with no lock and no TLSF operation. Replay hides
    it. **This arm also scores `[CONTEXT-PLAN-BUG]` = 0 under `GGML_SYCL_STRICT_PLAN=1`**: it is
    the pipelined decode in which revision 5's vacate-at-completion would have reported healthy
    reuse as over plan. The default (replay) arm is scored the same way, for the record-mode
    rule (r5 I-C).
  - The expected delta is 0 on every arm.
  - If it is not zero, look at the TRANSIENT reclassification: scratch addresses move from
    the weight side to the context side.
  - There is no fattn clause: the sidecar is opt-in only (§2.8, r2 m4).
- **C7 zhcn's shared gates, scored against this design's plan fields.** zhcn's GA (B50 GPT-OSS
  `-c 65536 -ub 1024`: exactly the 6 highest-indexed full-attention layers demote, scored against
  `free_after_full_kv` and the demotion causes, §2.4.1) and GH (B50 Qwen3.8-Flash-Next under
  `GGML_SYCL_STRICT_PLAN=1`: no unplanned per-context SYCL buffer, the recurrent state claimed
  from its planned slot, §2.4.4). They are listed in zhcn's §5.3 and run once each, after L6.

## 4. Decomposition, effort, landing order

| id | work | files | effort | depends on | lands |
|----|------|-------|--------|------------|-------|
| L1 | TLSF placement primitives, tags, frontier walk; H1 | `tlsf-allocator.hpp`, `shared-zone-tags.hpp`, `tests/test-tlsf-allocator.cpp`, CMake | high | none | **done, approved:** `eab1ebeb6`, `9e0a708dc`, `97315421b`, `456650c01` |
| L3 | pure `kv_region_fit` (multi-extent, self extents, `forced_host`, `own_ranges`-restricted commit re-fit, strict prefix, carve mirroring, indexed head slots placed first with reuse in place, the ring as max over contributions and its RUNTIME/KV-zone split, the recurrent slot, sidecar companion slots, the cost-ordered pack with the two-way optional classification, pending ranges as allocated, the stated demotion order, `free_after_full_kv` and demotion causes), the `context_side_demand` record (CONTEXT/DEVICE scopes, indexed slots) and its reconciliation, `kv_layer_cells` + `kv_layer_tensor_bytes` (the one byte function) and the RS-buffer size function, `kv-region-registry.hpp` (registry with tenant slots, scope, residency answer with the no-region fallback, the two-phase guard model, the ring record model with RELEASING, the release proc model); H2, H3, H6, H8, H9 | `kv-runtime-demotion.{hpp,cpp}`, `kv-region-registry.hpp`, `unified-cache.hpp` (`kv_layer_bytes_for_kind` delegates), their tests | xhigh | L1, jehw on master, the zhcn protocol (agreed, §6.6) | after jehw |
| L4 | `context_side` (explicit `lifetime` field threaded into `zone_alloc`, LIVE/RETAINED states and the slot tag, the reserved-slot carve as owner-first handles, `claim_slot`/`release_claim` with event-chained reuse and per-slot claim state, atomic retained-run carve, settle refusals for pending ranges, the plan-violation ERROR with an error status, pending ranges on shared and RUNTIME TLSFs, `live_bytes()` beside an unchanged `zone_available`), `allocate_excluding` with range exclusion (used whenever a pending range exists), `allocate_at`, and a whole-TLSF block census (L1 follow-ups), retained-run registration hygiene and the `live_bytes` readers, the per-extent and per-slot owner-first carve inside `zone_alloc`'s locked section with controls pre-minted before L1 (`unified_allocate_owner` split into mint and bind halves), leaf `kv_region_mutex_` added to contract §12.5 (L3), locked geometry snapshot (cache locks then group mutex, with jehw's predicate), `reserve_kv_region`, strict-prefix `yield_optional_prefix`, `backend-buffer-kv-zone` passing its buffer's role, removal of the dead `KV_AUTO` reclaim and of the `arena_reserve` KV reclaim, N-chunk routing, and **1oxa's `GGML_SYCL_PRIVATE_TESTING` dump of `shared_zone_geometry` plus `kv_region_request` at each fit** (step 2 and the tenant-only path; lead-approved, for 1oxa's VM branch to test against); H4, H4b, H5 | `unified-cache.{hpp,cpp}`, `tlsf-allocator.hpp` (two primitives), `ggml-sycl.cpp` (the kv-zone fallback's role) | xhigh | L1, L3, **zhcn landed, beni's producers landed, llama.cpp-jzvq closed** (lead ruling, r4 I10), **llama.cpp-uwlx landed** (the pick-list yield, §2.4.2 step 5; r5 m-g) | after zhcn, beni producers, jzvq, uwlx |
| L5 | the optional pass after all S1 staging (dense + expert/DPAS); `zone_alloc_optional` | `ggml-sycl.cpp` S1 block | medium | L4 | with L4/L6 |
| L6 | llama side: `llama_kv_layer_shapes` and `llama_rs_layer_shapes` factored out and stored at the first publish, the `ggml_sycl_runtime_context_desc` descriptor (KV-shape with sidecar and `n_stream`, recurrent section; zhcn's tenant section filled by zhcn) and its publish entry point, the scope procs with an RAII guard, the one `ggml_backend_sycl_kv_region_release` call site in the `sycl_plan_guard` member's destructor (zhcn M1, with the ContextId captured at `create_exec`), and `llama_recurrent_sycl_kv_buft` returning the recurrent-state buft. Backend side: the transaction steps of §2.4.2 (two-phase guard without L1, idempotent key without `n_ubatch`, the tenant-only path with zhcn's step (i) (the own-context graph clear behind its recorded-graph gate, the slot-table take at (c), the reap call with its backstop, the use-count bound; r6), reuse in place for host slots, the ring record with contributions, `ring_plan_gen`, RELEASING and `pinned[slot]`, the ring-lock section before the snapshot, the slot-state retention moved out at a sole-contributor release and handed off after the unlock at all six removal sites (r6 I-4, I-5, I-6; r7 m-5), the model load's ring release and ring-record writes removed on arena devices (llama.cpp-r7fz), the ring admit per the recorded split, plan / accounting / predictable refusals / pending ranges / yield / restricted re-fit and carve / MMID / ring check / CAS / commit with the ring admit / superseded drops after L1), the registry release proc, `g_execution_backend_binding_mutex` census entry, the residency hook answering from the registry, the tiered claim with `set_owner(mem_handle)` slice views and the KV-only size check, the sidecar companion claim, the recurrent-state buft, the per-extent clear with event-held slices, BLOCK_EXEC_CANDIDATE_KV ignored under an arena (the VMEM_KV refusal is 23mk core's, §2.6), the second sources deleted (§2.2, the budget-room check included), both ERROR sites plus `GGML_SYCL_STRICT_PLAN`, the dark B50 lever, `GGML_SYCL_KV_REGION_TRACE`; H7 with the unconverted-site list, the CPU-buft llama shape tests, G1. **Absorbs revision 1's L2.** | `ggml-sycl.cpp`, `ggml-sycl.h`, `unified-cache.cpp`, `fattn.cpp`, `common.hpp`, `src/llama-context.{h,cpp}`, `src/llama-model.cpp`, `src/llama-kv-cache.{h,cpp}`, `src/llama-memory-recurrent.cpp`, tests | xhigh | L3, L4, L5 | before beni's conversions |
| L7 | docs: memory-design section, contract §3/§5.2/§5.3/§12.5 (the binding-lock chain, the L5 group → `g_runtime_alloc_mutex` order, the step-6 carve exception), arena comment, limits (§2.9), lock order, the tenant protocol and the descriptor's layout rules, the owner-visible weight-hole line | docs, `unified-cache.hpp` comment | medium | L6 | with L6 |

**Landing order (lead ruling; r4 I10).** jehw lands on master first (u1bb already has). Then:
1. moua L1-L3 (pure, host-tested; L1 is done). zhcn and beni need L3's record type,
   reconciliation and the protocol's host model before they produce records.
2. llama.cpp-zhcn: the measure pass, its tenants, the descriptor's tenant section, the deletion
   of the per-row constant.
3. llama.cpp-beni's **producers**: the demand functions and zhcn-walker visitors for its
   cohorts, including the oneDNN Graph scratch.
4. moua L4-L7. L4 needs every producer that exists at its landing to be real, because a
   converted site with no producer fails H7p; sites not yet converted are on H7p's list
   (§2.4.3's transition rule), so L4 does not wait for beni's conversions.
5. llama.cpp-beni's **site conversions**: each commit converts sites to claims and removes them
   from H7p's list; the last one empties it.

Alongside: llama.cpp-23mk core lands after jehw (independent of moua's order), and
llama.cpp-jzvq **closes before L4**, because its MXFP4 MoE TG caches are on GPT-OSS's default
decode path and would otherwise be unplanned at L4 (r4 I10(c)). llama.cpp-uwlx (the pick-list
`yield_optional_layouts_begin`, owner impl-jehw) depends on jehw and lands before L4, which is
its consumer. llama.cpp-423j (retire on request, §2.9) depends on uwlx and moua L1-L3 and can
land at any point after them; until it does, leased copies are not yieldable, and the WARN is
the owner-visible record. So: jehw → uwlx → 423j, and uwlx → moua L4.

- L3 through L7 form one series in one worktree, so there is one first build and the
  follow-ups reuse it. L3 lands first on its own; L4-L7 follow, rebased on zhcn and beni's
  producers.
- Every step is RED → GREEN with its host test.
- G1 and C0-C7 are the lead's runs at the end of L6.
- **Seams (agreed unless marked):**
  - **jehw:** retire `select_optional_layout_yield`'s pruning in favour of the strict prefix;
    retire `kv_zone_snapshot`/`fit_capacity` from production (they stay as a test oracle) and
    reuse `kv_layer_alloc_bytes`; the yield's lock split (`c41fed119`) is kept as it is; u1bb's
    reconcile is reordered around begin/finish with no lock or signature change (jehw's note).
    The yield takes the fit's picks through **llama.cpp-uwlx** (`yield_optional_layouts_begin`
    with an explicit pick list, all-or-none per group, identity by key plus entry generation;
    §2.4.2 step 5). The seam for leased copies is jehw's retire-on-request, **llama.cpp-423j**,
    on §2.9's terms, and it depends on uwlx; no transaction drops another context's graphs.
  - **u1bb (in master):** the ring becomes a DEVICE-scope record with per-context
    contributions; both halves are placed by the fit, and the admit installs the recorded split;
    `release_pp_moe_onednn_scratch_ring` is no longer called by the full transaction; the
    `zone_largest_free(KV)`, `ggml_sycl_kv_capacity_live`, `ggml_sycl_device_kv_bytes_with_slack`
    and budget-room reads are deleted for arena devices; its "does not fit" refusal becomes the
    fit's head-slot refusal.
  - **zhcn (agreed, §6.6):** the measure pass and its walker (with beni's visitors), its tenant
    records and the descriptor's tenant section, the claim by live-object index, step (i)'s
    pre-L1 release, no fallback, charging at moua's commit, `GGML_SYCL_STRICT_PLAN` with no
    alias, the scheduler reset before a candidate publish, the gallocr return check, and the
    deletion of `k_pp_moe_ring_compute_reserve_bytes_per_row`. zhcn cites §2.3.2 and §2.4.4 as
    normative.
  - **beni:** the records of §2.4.3's list, as producers first and site conversions after L4;
    the cxgg fix.
  - **jzvq:** per-(ContextId, device) ownership and exact demand for the fattn workspaces and the
    MXFP4 MoE TG caches, as claims under §2.3.2; closes before L4.
  - **23mk core:** zones and forbid on out-of-arena rows, the trace and its positive control, the
    dead-code deletions, the oneDNN weights scratch at its max in ONEDNN, the persistent-TG
    refusal under an arena; the sidecar size function over `kv_layer_cells` and `n_stream`.
  - **1oxa:** its VM branch places the same reserved slots on its chunks and cites §2.3.2; the
    L4 geometry dump is its test hook.

## 5. Decisions and open questions

- **(a) Refuse by default, and abort under `GGML_SYCL_STRICT_PLAN=1`:** adopted, at
  region-backed sites and at context-side plan violations (§2.8, §2.4.3).
- **(b) Yield order:** a strict address-ordered prefix, highest first (§2.4.1). This is a
  smaller change to jehw HEAD than revision 1 framed (r1 M7). The two-pass staging lands in
  the same series (L5).
- **(c) Weight holes:** **revised; the lead provisionally agrees.** With multi-extent
  regions, weight holes and buried optional tenants become region extents, and only
  sub-slot holes remain a limit, with a WARN (§2.9). The gap-before-holes order is approved
  (r3).
  - This is the fix for audit I5 (r2 N-I10): KV beats buried optional copies lazily, only when
    a context needs the room.
  - The reviewer's alternative, a burying WEIGHT allocation yielding the bottom ladder tenants
    it needs instead, is weighed in §2.9. It pays the yield on every burying load, whether
    or not KV ever needs the room.
- **(d) Handoff:** **revised.** A dedicated ContextId-keyed registry plus a thread-local
  region scope, which is a llama-side change (§2.5). The KV-mask handoff is kept and
  verified against the slot table.
- **(e) Decided (owner, 2026-09-26): "Plan them exactly first."** There is no estimate, no
  floor and no per-row constant.
  - Context-side tenants publish indexed slot lists with a scope; the fit places them as head
    slots, and the commit holds them as reserved slots owned by handles (§2.4.3, §2.3.2).
  - **Producers (lead rulings, r3 and r4):** zhcn the compute chunks and fattn slot; beni every
    other context-shaped cohort, the oneDNN Graph scratch included; jzvq the fattn workspaces and
    the MXFP4 MoE TG caches; moua the recurrent state; u1bb's ring is the one DEVICE-scope
    record. moua L3 consumes the rest.
  - Landing: jehw → moua L1-L3 → zhcn → beni producers → moua L4-L7 → beni conversions; 23mk
    core after jehw; jzvq before L4 (§4).
- **(f) The runtime context crosses the ABI once (r2 N-C1; lead ruling 4, D4).** One versioned,
  `struct_size`-headed descriptor, whose layout moua owns, with a KV-shape section and a
  recurrent section (moua) and a measured-tenant section (zhcn, carrying beni's and jzvq's
  demands). The old entry point is kept (§2.4.4).
- **(g) The STRICT variable (lead ruling 4, D3):** `GGML_SYCL_STRICT_PLAN`, one for moua, zhcn,
  beni and jzvq, with no alias (agreed with zhcn, §6.6).
- **(h) Copies held by recorded graphs (lead ruling on the r4 addendum).** llama.cpp-423j (owner
  impl-jehw, after jehw lands) is jehw's retire-on-request under L1 with the epoch, on §2.9's
  terms. Revision 6's first draft left a choice between a per-context graph exclusion and an
  idle-only try-acquire to 423j's review; both served the cross-context graph drop, which is
  withdrawn, so the choice is gone. Re-promoting a demoted layer is out of scope; it would be
  its own ticket.
- **(i) The step-6 carve under L1 (rulings §L6; r3 m12; r4 m7): ratified by the lead 2026-09-26,
  llama.cpp-moua r5,** for device carves, on r5 m-e's terms (the "never rehashes" promise
  withdrawn). The exact lock sequence is in §2.10. zhcn's host-pinned compute tenant was
  open here; it is now specified in §2.4.3 and allocated before L1, because the host arena
  grows lazily and its allocation can be a USM call. The exception covers device carves only.
- **(j) One tenant protocol (r4 I3; confirmed by impl-zhcn 2026-09-26, with A1-A4 and M1-M2).** Held
  handles, claim by index, event-chained reuse, zhcn's step (i), no fallback, one charge site, one
  STRICT variable. This document is the normative spec (§2.3.2); the record of who changed what is
  §6.6.
- **(k) The commit's race closure (r4 I8; lead ruling: pick the smaller lock surface).** Pending
  ranges cover every placement, and the commit re-fits only inside them, carving at exact
  offsets; the alternative, one critical section under the cache locks and the group mutex, was
  rejected for its lock surface (§2.4.2 step 6).
- **(l) The byte budget (r4 I7; lead ruling).** Under an arena the geometry binds; the budget
  room is not an admission input for in-arena head slots, and every carved byte is still charged
  at one site (§2.2).
- **(m) The record-mode claim lifetime costs index sets (r5 I-C; lead ruling).** A claim made
  while recording lasts for the life of the executable graph and vacates at its destruction on
  an eager event (§2.3.2). The cost: a per-op cohort used both in a recorded decode and in an
  eager pass needs a separate record-mode index set, with one index per recorded use, per live
  executable graph that records it (lead ruling, from the 23mk review addendum). Master keeps
  one exec graph per context (`common.hpp:6073`), and the MoE dispatch/block graphs
  (`common.hpp:6137-6169`) can multiply that. The producers (beni, jzvq, zhcn's fattn slot)
  count it in their demand functions, so the planned context-side bytes grow by those slots.
  **Not chosen:** the reviewer's alternative of refreshing each slot's release event at every
  replay submission, so replays and eager claims share one index. It would put a per-replay
  write to every recorded slot on the replay path, and the bound on concurrent use would then
  rest on the refresh being complete, which no source gate can see, and it couples eager claims
  to replay order. **Accepted by the lead 2026-09-26:** the cost is planned and stated, so plan
  == reality; the alternative stays documented; G1 prints the added bytes.
- **(n) The GA number (zhcn GA; r5 m-g).** The fit reads the live TLSF, so it scores against
  the actual weight allocations (11510.9 MiB), never the planning figure: `free_after_full_kv`
  = −726.9 MiB, and 6 layers demote (§2.4.1, with the breakdown). The −714.9 agreed with zhcn
  rev 4.1 (`4bb0436`) carried the SWA term at `-ub 512`; design-zhcn-r4 found it, and both
  designs now print −726.9 (planning −752.2); revision 6's 728.4 is withdrawn. The
  pre-registration is scored by ⌈−printed / 128⌉ = 6, which the correction does not change.

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
| I3 | retained holes need registration hygiene | **Changed.** Retain unregisters the exact record first, reuse is owner-first with a fresh registration, and rebuild, destroy and settle treat retained runs as **not live** through `live_bytes`, while `zone_available` still counts them as used, so no capacity reader treats them as free (§2.3.2 rule 3; corrected in revision 5, r3 m6). Checked by H4 and G1. |
| I4 | VMEM_KV double-charges and keeps planned KV outside accounting | **Changed.** Under an arena the region takes precedence and `GGML_SYCL_VMEM_KV` is ignored with one WARN. The vmem branch is itself arena-gated, so without an arena it never ran; the only residual case is an arena that is enabled but not active, where it runs as today until 1oxa deletes it (§2.6; corrected in revision 5, r3 m7). |
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

### 6.5 Design review r3 (design-moua-r3, read `456650c01..1dfc63531`), and the lead's rulings

The lead's rulings on r3: (1) rebase onto jehw HEAD, and propose the recorded-graph seam to
impl-jehw; (2) the ring at the reservation's `n_ubatch` is a planned mandatory CONTEXT demand,
a head slot in the one fit; (3) zhcn deletes `k_pp_moe_ring_compute_reserve_bytes_per_row`,
and moua widens H7p to every context-side admission; (4) the zhcn contract is accepted, D3 is
`GGML_SYCL_STRICT_PLAN`, D4 is one versioned descriptor whose layout moua owns; (5) landing
order jehw, then moua L1-L3, zhcn, 23mk, moua L4+. Also: 23mk owns the in-arena per-op
TRANSIENT demand functions, and gap-before-holes is approved.

| id | finding | disposition |
|----|---------|-------------|
| C1 | the transient reserve is a fit-time inequality nothing holds: weights, the ring re-admit and the yield window consume it; the non-STRICT miss disposition is unspecified | **Changed.** Head slots are placed first in the fit, recorded as pending ranges before the yield, and carved as **reserved slots** at the commit: TLSF-allocated blocks held for their `(owner, cohort)`, which no weight, other owner or transaction can take (§2.3.2, §2.4.2, §2.4.3). The ring is one of them, so its re-admit occupies its own slots. The disposition is per cohort: ERROR once, STRICT abort, else unreserved room via `context_side_place`, else failure through `forbid_vram_zone_spill`, never raw (§2.4.3). H4, H5, H7p, G1. |
| C2(a) | jehw HEAD `c41fed119` splits the yield around `lock.unlock()`; §2.3.1, step 4, §2.10, §4 and H7l are stale | **Changed.** Re-cited against `c41fed119` (confirmation requested from impl-jehw). §2.3.1 states that reservations are not serialised end to end, and what keeps a plan valid across the window: pending ranges recorded before the unlock and treated as allocated by every other placement and fit, the relock's busy check, and the commit re-fit. `kv_region_mutex_`'s rationale is restated. §2.10 and §4's seam are rewritten; H7g/l updated. |
| C2(b) | the fit must call the yield's predicate | **Changed.** The snapshot classifies every optional tenant through `optional_layout_yieldable_locked` under the cache locks, in the same section as the geometry copy; the retire re-checks it (§2.3.1, §2.4.1). H7q. |
| C2(c) | recorded graphs lease the WOQ copies for their life, so KV demotes while copies hold VRAM; step 4's "no lease" is obsolete | **Changed.** The inherited-P2 text is removed. §2.9 specifies a reclaim contract for jehw (split lease count, three-way predicate, retire under L1, drop the holding graphs under each backend's graph-compute exclusion in the unlocked half), proposed to impl-jehw. Until it lands, graph-held copies are not yieldable and the fit stays exact. H5 and G1 cases. |
| I1 | the scope ruling is not in the doc or on 23mk | **Changed.** §2.4.3 lists who produces what, §5 (e) records the ruling, the L3 row says consumer only. The widened scope is on llama.cpp-23mk in impl-23mk's comment c-2qrv (its design revision 2, §6b), and §2.4.3's list follows that revision. |
| I2 | demand records have no scope or owner; B's reserve subtracts A's live buffer, and B can take A's outstanding room | **Changed.** Records carry `demand_scope` and `owner`; each owner's slots are its own reserved blocks from its commit, so the outstanding sum is held physically (§2.4.3). H8 two-context case with a positive control. |
| I3 | a same-key republish skips the check at the ladder's larger `n_ubatch` | **Changed.** A matched key takes the tenant-only path: head slots re-planned against the live geometry with the region fixed, no yield, the candidate refused if growth does not fit. Timing option 2 (re-fit after `sched_reserve`) is ruled out (§2.4.2). |
| I4 | the guard's ring restore after L1 can clobber another transaction's ring | **Changed.** Two-phase guard: the ring restore (conditioned on the ring generation), the pending-range clear and the slot release run under a re-taken L1; the handle drops run after every lock. The restore needs no space from the extents, because the ring's slots are released only at a commit (§2.4.2). H9 concurrent re-plan failpoint. |
| I5 | KV takes the gap, then the ring refuses where demoting one layer would fit | **Changed** per lead ruling 2: the ring is a head slot and KV demotes around it (§2.7). H2 case with a RED on revision 4's order. |
| I6 | the teardown drop runs under `g_execution_backend_binding_mutex`, which §12.5 does not classify | **Changed.** The erase moves to the two close sites, after `clear_bindings_for_context` returns, and drops with no lock held; the binding mutex is classified L3 (§2.3.1, §2.4.2). H7m reworded, H9. |
| I7 | planned peak live bytes are not placeable under non-LIFO frees | **Changed.** Demand is a slot list and each allocation occupies a slot of its pair, so no free order fragments planned room (§2.3.2, §2.4.3). H4 non-LIFO property test with revision 4's policy as the positive control. |
| I8 | u1bb's 1 MiB/row compute reserve survives; H7p covers only `transient_reserve` | **Changed** per lead ruling 3: zhcn deletes the constant; H7p is widened to every context-side admission, and H7r forbids the constant (§2.2, §2.7). The 928 MiB non-FA constant is cited in §2.11 as pre-existing, 23mk/zhcn's. |
| m1 | tier 2 needs capacity `gap − reserve`; add the hole case and the RED label | **Changed.** Head slots are placed first, so the gap's KV capacity is what they leave, with no post-check (§2.4.1). H5 adds the case and labels the gap-before-holes case as a RED against the addendum's literal order. |
| m2 | `live_bytes` names two things; allowlist the meter and copy it with the geometry | **Changed.** The per-pair meter is `reserved_slot_occupancy()`, part of the geometry snapshot, on H7d's list and allowlisted (§2.4.3). `live_bytes()` keeps the TLSF-wide meaning. |
| m3 | `allocate_excluding` removes the whole gap block | **Changed.** Range exclusion with splits at range boundaries (§2.3.3). H1 cases. |
| m4 | the byte-budget demotion runs after the yield | **Changed.** It runs on step 3's residency before the yield, together with every predictable refusal (§2.4.2 steps 4-5). H7s. |
| m5 | where the lever's cohorts' demand goes; the C2a "positive control" is a baseline | **Changed.** Their slots are reserved slots carved weight-side first-fit (§2.1); C2a says baseline (§3.3). |
| m6 | §6.2 row I3 says retained runs count as free | **Fixed** (§6.2). |
| m7 | §6.2 row I4 says vmem-kv is unchanged without an arena | **Fixed** (§6.2). |
| m8 | H2's formula and C1's 23/9 predate the ruling | **Changed.** Both are a formula in the head-slot sum `H`, with fixture points at 0, 10 and 11 MiB, and C1 is scored against the traced `H` (§3.1, §3.3). |
| m9 | the hook reads "host" for a device with no region | **Changed.** Such a device falls back to the plan (§2.5). H8 mixed-set case. |
| m10 | citation drift in §2.3.1 | **Fixed**, re-cited against jehw `c41fed119` (wait `unified-cache.cpp:7865`, drops `:7853` and `:7894`). |
| m11 | nothing reaches the "shape change" path | **Deleted.** A different key for a known ContextId is refused as a contract violation, with one H9 case (§2.4.2 step 1). |
| m12 | the step-6 carve and the ring re-admit allocate under L1 | **Classified** (§2.10): the carve is new, metadata only, and must be atomic with the plan; the ring occupy is inherited. Recorded as an exception for the contract owner (§5 (i)). |

### 6.6 Design review r4 (design-moua-r4, read `f34acb398`), the lead's rulings, and the tenant protocol

The lead's rulings on r4: I1 each claim names its slot index; I2 event-chained slot reuse, no
host waits, covering 23mk's c-4vlt overlap; I3 one tenant protocol with zhcn, owner-first held
handles, moua's doc normative, agreed directly with impl-zhcn and recorded here; I4 per-op
cohorts CONTEXT scope, the ring DEVICE scope as the max over live contributions with an explicit
release; I5 superseded slots released only at the publish; I6 the whole ring a head slot held
through the window, its generation declared state, no absent-ring publish; I7 delete the
budget-room check, the fit is the single source; I8 pick the race closure with less lock
surface; I9 recurrent state in moua's scope; I10 the landing order, beni's oneDNN Graph scratch,
a transition rule for H7p, the explicit `-ub` WARN, and `free_after_full_kv`; m7 state the
carve's lock sequence as the §12.5 exception; m9/m12 a reclaim for a non-OK close and no MODEL
scope without a release.

**The tenant protocol: who changed what (confirmed by impl-zhcn 2026-09-26, with amendments
A1-A4 and M1-M2).** moua's rev 5 and
zhcn's rev 2 specified the same two tenants differently (r4 I3's table). moua proposed P1-P11;
zhcn sent five deltas at the same time, which P1-P11 already covered (the messages crossed); the
lead accepted the five deltas; zhcn then agreed to P1-P11 with four amendments, A1-A4, all
accepted here. The agreed protocol is §2.3.2, §2.4.2 and §2.4.3. zhcn's rev 3 (scratch
`abc23c4`) makes the zhcn-side changes and cites this document as normative.

| aspect | moua rev 5 | zhcn rev 2 | rev 6 | who changed |
|---|---|---|---|---|
| holding | unowned RESERVED TLSF blocks keyed by an integer owner, explicit release, "orphaned" state | owner-first carve held by the registry entry, claims as `mem_handle` leases | zhcn's: owner-first handles held by the registry entry (CONTEXT) or the ring record (DEVICE); claims are slices; no orphaned state | **moua** |
| claim | best fit over vacant slots | slot index = live buffer-object count | by index, the producer's (zhcn's live-object count for chunks; role indexes for per-op slabs) | **moua** |
| reuse | vacate at completion | the fattn slot chains on the previous SDPA event | event-chained for every tenant: release at submission with an event, the next claim depends on it (new, r4 I2) | **both** (moua specifies; zhcn's chunks and fattn slot already comply) |
| tenant-only step (i) | wait for vacancy, `busy` if occupied, keep slots to the commit | pre-L1: still leased → `[CONTEXT-PLAN-BUG]`, else move out and drop | zhcn's, plus the ring's sole-contributor rule with RELEASING | **moua** |
| full-transaction release | the ring released before the yield; owner slots released at the commit | — | nothing released before the publish; superseded slots released after the CAS (r4 I5, I6) | **moua** |
| miss | ERROR, then `context_side_place`, else fail | `[CONTEXT-PLAN-BUG]`, decode returns an error, no fallback | zhcn's, for every tenant; an error status reaches the graph, never a skipped op | **moua** |
| charging | only the ring charged to `vram_bytes` | tenants charged "at the ring's charging site" | one charge at the step-6 carve by carved block size, one uncharge at the block's real free, never at a claim or release; u1bb's ring charging site deleted for arena devices; H7y (r4 I7; zhcn A2) | **both** (moua charges all; zhcn's §3.3 cites moua's commit) |
| STRICT | `GGML_SYCL_STRICT_PLAN`, `..._KV_PLAN` must not occur | `GGML_SYCL_STRICT_PLAN` "aliasing `GGML_SYCL_STRICT_KV_PLAN`" | `GGML_SYCL_STRICT_PLAN` only, no alias | **zhcn** |
| scopes | CONTEXT, MODEL, DEVICE (per-op scratch DEVICE) | chunks and fattn slot CONTEXT | CONTEXT for every tenant but the ring; the ring DEVICE; no MODEL (r4 I4, m12) | **moua** |
| specification | both docs specified the slot protocol | — | moua §2.3.2 and §2.4.4 are normative; zhcn cites them and stops restating | **zhcn** (cites) |
| host compute tenant (SYCL_Host) | — | "same protocol on the host-pinned tier", carve "at the same commit" | specified only in moua §2.4.3, cohort `context-compute-host`; allocated with no lock held before L1, because the host arena grows lazily and its allocation can be a USM call; not a head slot of the device fit; charged to the host inventory (zhcn A3; lead ruling: one section) | **moua** (specifies; closes §5 (i)) with **zhcn M2** (owner-first via `unified_allocate_owner`, the commit's install is a move, a failed host carve is a refusal before L1); zhcn supplies `slot_bytes` |
| tenant element | "zhcn's to define: at least {device, zone, lifetime, cohort, index, cap}" | `{struct_size, cohort, slot_index, slot_bytes}` | zhcn's element plus `int32_t device` (-1 = host tier, owned by the host buft's device entry); zone, lifetime, scope and tier from one static cohort table, H7o | **both** (moua adds `device`; accepted by zhcn rev 4.1 `4bb0436`, §6.7) |
| teardown | release at the drain tail and the construction unwind, "exactly once" | — | one call site: the destructor of zhcn's `sycl_plan_guard` member (declared before `sched`), with the ContextId it captured at `create_exec`; idempotent; no live-lease assertion; no BINDING nesting (zhcn T1-T3) | **zhcn M1, lead ruling** (r5 I-F added the captured id) |

**zhcn's amendments to P1-P11, all accepted.**

| id | amendment | where |
|----|-----------|-------|
| A1 | "Within one ALLOC, per (ContextId, buft), the claim scope asserts the slot indices run 0..n-1 in order and n ≤ the measured chunk count. A violation is `[CONTEXT-PLAN-BUG]` (P5 disposition). The live-object-count rule is valid only because gallocr frees the whole vbuffer and allocates in chunk order; this assertion enforces it." | §2.3.2 "Claim by index"; H4 |
| A2 | "The single charge at the commit has a single uncharge, at the block's real free: the last handle or lease drop after its event. It never happens at vacate or at a claim. The H7e family gates that nothing else adds or subtracts a context-side cohort's bytes." (The gate is H7y.) | §2.3.2 "Charging", H7y |
| A3 | "The SYCL_Host compute tenant (cohort `context-compute-host`, slot index = the host buft's scope-claimed live-object count) is specified only here, as the §2.3.2 protocol on the host-pinned tier, charged once to the host inventory. zhcn supplies `slot_bytes` in the tenant section and has no host-tier protocol text." | §2.4.3 "The host-pinned tier" |
| A4 | "For gallocr buffers, release-at-submission is the buffer object's free, which always follows `ggml_backend_synchronize`, so the recorded event is already complete. The next claim still passes `depends_on`. For the fattn slot, the release event is the SDPA completion event." | §2.3.2 "Event-chained reuse" |
| M1 | the one teardown call site, the `sycl_plan_guard` member's destructor (lead ruling) | §2.4.2 "Teardown"; H7m, H9 |
| M2 | the host compute carve before L1, owner-first via `unified_allocate_owner`; the commit only installs (lead ruling) | §2.4.3 "The host-pinned tier"; H7ad |
| P6 note | a RELEASING `busy` is surfaced by every zhcn caller as a refusal, never admitted or skipped. **Superseded by rulings §E.1:** RELEASING under the re-plan mutex is `[CONTEXT-PLAN-BUG]`, and no `busy` reaches a zhcn caller from the ring | zhcn §3.1 step 4 |

**zhcn's teardown requirements (from design-zhcn-r2 m-b, m-c), all accepted:** T1 the release
never asserts "no live lease"; T2 the extract under `kv_region_mutex_` never nests in BINDING;
T3 the release is idempotent. zhcn's M1 then made the `sycl_plan_guard` member's destructor
the only call site (lead ruling; §2.4.2 "Teardown"; H7m, H9).

**The lead's questions answered.** *A legitimate in-flight occupier at step (i)?* None. Queued
work and recorded graphs hold a slot's lifetime through retained slices, never its claim, which
ends at submission. The one master state that looked like one, u1bb's ring slot kept `busy`
until its `done_event` completes (and host-waited on, `ggml-sycl.cpp:1648`), is converted by
§2.7 into vacant-with-event, and its host wait is deleted (H7z). *The SYCL_Host compute
tenant:* one section, §2.4.3.

**The r4 addendum (lead ruling: accepted as written).**

| item | disposition |
|------|-------------|
| A: revision 5's §2.9 steps 1-6 (a cross-context graph clear under L2) are unsafe | **Deleted**, with the three-way predicate, tier 4's "graphs forced to drop" key, and the graph drops in §2.4.2, §2.10, §4 and §5 (h). This also supersedes this revision's first draft, which left "(i) vs (ii)" to 423j's review. |
| B: with the seam absent the design is exact and safe; the P4 property is lost for contexts created after a replay | **Stated** in §2.9, with the owner-visible WARN naming the leased copies and the demoted layers (§2.4.2 step 8), and on the ticket. |
| C1: retire on request under L1 with the epoch, narrowly triggered, once per transaction | **Adopted** as 423j's terms (§2.9 items 1-3); the only site is step 8, so a `busy` retry never issues it. |
| C2: a retired-but-leased block is allocated and not yieldable | **Adopted** (§2.9 item 4, §2.4.1); H5 case. |
| C3: delete "re-promoted eventually" | **Stated as out of scope**: a demoted layer stays on the host for the context's life; re-promotion would be its own ticket (§2.9 item 5, §0). |
| D: the test changes | H5 cases (a)-(c); G1 is their device form keyed on 423j's accessor and does not "record which". |
| r3 C2(c) | **Resolved in part**: exact and safe; the P4 property is lost for contexts created after a recorded replay, recovered only for later contexts, and only after the holders compute. |

**Findings.**

| id | finding | disposition |
|----|---------|-------------|
| I1 | best-fit occupancy of heterogeneous slots misses in-plan requests ({100, 50}) | **Changed.** Every claim names its index, the producer's (§2.3.2). H4 carries the {100, 50} sequence as a RED against best fit. |
| I2 | "live at once" undefined under event-deferred release; a pipelined decode reports healthy reuse as over plan | **Changed.** Lifetime (the slice retained until its event) is separated from occupancy (released at submission with its event; the next claim depends on it). No host wait, no queue-depth counts; c-4vlt's overlap removed (§2.3.2). H4's churn models deferred completion with a positive control; C6's no-replay arm scores `[CONTEXT-PLAN-BUG]` = 0 under STRICT. |
| I3 | moua and zhcn specify the same tenants differently | **Changed**, one protocol agreed with impl-zhcn (P1-P11 plus zhcn's A1-A4), tables above (§2.3.2, §2.4.2, §2.4.3, §2.8). |
| I4 | DEVICE scope cannot see other owners; concurrent contexts share per-op slots; the ring shrinks under a live context; a dropped claimed slot is never released | **Changed.** Per-op cohorts are CONTEXT scope; MODEL scope deleted; the ring is the only DEVICE record, the max over live contributions, never shrinking below a live need, released with its last contributor (§2.4.3, §2.7). Release is by handle drop, so a claimed slot is freed by its last claim (§2.3.2). H8 cases for concurrent claims and the ring across contexts. |
| I5 | the guard's "restore needs no space" is false after the commit; step 8 occupies before the CAS | **Changed.** The full transaction releases nothing before the publish; the commit carves new slots beside the old; the old go after the CAS; the guard drops only this call's handles (§2.4.2). H9 asserts the ring is in its original slots at the MMID and CAS failpoints. |
| I6 | the ring's RUNTIME half is unreserved across the window; the split is read live; B can publish an absent ring; the generation is undeclared | **Changed.** The fit places both halves (the RUNTIME TLSF is in the geometry, with pending ranges) and freezes the split; the ring is never released before the publish; the one early release (tenant-only, sole contributor) sets RELEASING and others return `busy`; the ring record and `ring_plan_gen` are declared (§2.7, §2.4.2). H2 and H9 cases. |
| I7 | u1bb's budget room is a second "fits" source with no demotion lever, run after the yield | **Changed** per the lead's ruling: deleted for in-arena head slots; the geometry binds; one charge site (§2.2, §2.4.1). H2 case with a RED on revision 5's order; H7d. |
| I8 | the step-7 re-fit is not atomic with the carve | **Changed.** Pending ranges cover every placement; the commit re-fits only inside them and carves with `allocate_at`, so nothing can intervene; a shortfall demotes and never loops. Chosen over one critical section for lock surface (§2.4.2 step 6, §5 (k)). H2 case for churn between re-snapshot and carve, RED on revision 5; H7g. |
| I9 | recurrent state ruled into moua but called out of scope; zhcn's GH fails | **Changed.** A recurrent section in the descriptor, the RS buffer as a mandatory CONTEXT head slot, claimed through `llama_recurrent_sycl_kv_buft`'s new recurrent-state buft; H2/H3 cases, H7x, C7 (GH) (§2.4.4). |
| I10 | landing order, beni, jzvq, the transition rule, the `-ub` WARN field | **Changed.** (a) the order is jehw → moua L1-L3 → zhcn → beni producers → moua L4-L7 → beni conversions, 23mk core after jehw (§0, §4); (b) the transition rule with H7p's shrinking unconverted-site list (§2.4.3); (c) jzvq closes before L4, its MXFP4 MoE TG caches named (§2.4.3); (d) beni's oneDNN Graph scratch is a producer, and beni's demands travel in zhcn's tenant section (§2.4.3, §2.4.4); (e) `free_after_full_kv` and per-layer demotion causes are fit outputs, and the demotion order is stated with zhcn's GA prediction (§2.4.1). |
| m1 | pending ranges protected only with a ladder live | **Changed.** `allocate_excluding` whenever any pending range exists on the TLSF (§2.3.3); H5 no-ladder RED. |
| m2 | the rollback's unlocked drop pessimizes a concurrent fit | **Accepted, reasoned** (§2.4.2): the commit re-fit uses only its own ranges, so only a retry's new plan can see the doomed extents, for one instant, erring toward demotion. |
| m3 | the accounting is not re-run after a shortfall demotion | **Changed.** Re-run on the committed residency before step 7; it can only decrease (§2.4.2 step 6). |
| m4 | the lever's first-fit placement is not fit-predictable | **Changed.** The fit chooses explicit weight-side offsets and the carve uses `allocate_at` (§2.1, §2.3.3). |
| m5 | C1 scored against the printed `H` is circular | **Changed.** `H_A2` is pinned from the producers on the host before C1; C1 requires the printed `H` to equal it (§3.1 H2, §3.3 C1). |
| m6 | G1's graph-held case passes either way | **Changed.** The expected arm is keyed on a 423j-presence accessor; the no-seam arm asserts the demotion WARN names the graph-held copies (§3.2). |
| m7 | "metadata only" needs its lock and work chain | **Changed.** The exact sequence, the pre-existing group → `g_runtime_alloc_mutex` L5 nesting, the pre-minted controls, the map reserve, and the bypass of `unified_alloc`'s `evict_and_flush` guard are stated (§2.10); H7u. zhcn's host-pinned tenant is allocated before L1, because the host arena grows lazily (§2.4.3; §5 (i) closed). |
| m8 | the binding lock's nested locks; the invented `kv_region_mutex_` tie-break | **Changed.** The binding chain is one census entry (binding L3; execution state and execution registry L4, ordered) for L7 (§2.3.1). `kv_region_mutex_` is strictly leaf, so it needs no tie-break, and the device-ID one is deleted. |
| m9 | a non-OK close leaves the entry with no reclaim | **Changed.** One unconditional release proc called by llama on both teardown paths whatever the close returned; the backend's close functions no longer reach it (§2.4.2 "Teardown"). H7m, H9. Superseded by zhcn M1's single call site (§6.7). |
| m10 | settle should refuse while pending ranges exist or DEVICE slots are occupied | **Changed.** Pending ranges refuse a settle; every reserved slot, the ring's included, is a registered allocation that the existing precondition sees (§2.3.4). |
| m11 | a miss must return an error status, never a skipped op | **Changed** (§2.4.3); H7p checks the failure branch's effect. |
| m12 | MODEL scope has no producer or release | **Deleted** (§2.4.3). |
| m13 | probes see others' pending ranges as allocated | **Accepted, reasoned** (§2.4.2 step 4): errs toward refusal for one yield's length; the ladder or `busy` backoff recovers. |
| m14 | drift: u1bb is in master; the destructor's L1 re-take can throw; the STRICT alias | **Fixed.** The header says u1bb is in master `2c4f5e45d`; the guard's destructor takes no L1, is `noexcept` and aborts on a lock failure; the alias is withdrawn by zhcn (§2.8, table above). |

**Queued fold-ins applied in this revision (lead-approved during r4).**

| item | where |
|---|---|
| landing order with llama.cpp-beni, never "23mk-b"; the "if Q3 accepted" sentence removed | §0, §2.4.3, §4, §5 (e) |
| §2.9 becomes llama.cpp-423j's acceptance; jehw's (e) as a code-free H5 case. Superseded by the r4 addendum (table above): jehw's N1-N5 and the (i)/(ii) choice served the withdrawn graph drop, and 423j is now retire-on-request | §2.9, §2.4.1, §3.1 H5, §3.2 G1 |
| 23mk's sidecar as a per-layer companion slot (`align(kv) + align(sidecar)`), the registry's second slice, claims by `(ContextId, device, layer)` with a hard ceiling, the KV-only claim sum, the sidecar flag and `n_stream` in the key; the forced-split packed-K a TRANSIENT record; cxgg's fix in beni; persistent-TG refused under an arena; every record carries a zone | §2.1, §2.4.1, §2.4.3, §2.4.4, §2.5, §2.6, §2.8 |
| the demotion-order paragraph and `free_after_full_kv`, with zhcn's GA prediction (6 layers; 7 only with sub-slot holes) | §2.4.1, §3.1 H2, §3.3 C7 |
| 1oxa's `GGML_SYCL_PRIVATE_TESTING` dump of `shared_zone_geometry` plus `kv_region_request` at each fit | §4 L4 |
| 1oxa's VM note: no unreserved context-side room, so the disposition there is error, never raw | §2.4.3, §2.11 |
| the ring-held yield limitation, named and closed by L4+, with an H2 RED on the jehw-merge order | §2.7, §3.1 H2 |
| the step-7 (now step-6) carve exception, ruled "classified, pending ratification"; ratified in r5 (§6.7) | §2.10, §5 (i) |

### 6.7 Design review r5 (design-moua-r5 on `b021c9629`, then on `c2613a688`), the lead's rulings, the post-r5 queue, zhcn rev 4 and 23mk

**Ratification (lead, 2026-09-26).** The step-6 carve under L1 is a §12.5 exception **ratified
by the lead 2026-09-26, llama.cpp-moua r5**, for device carves, conditional on m-e, which is
applied (§2.10, §5 (i)). Host carves are outside L1 and outside the exception (§2.4.3).

**The one consequence to flag (I-C).** The record-mode lifetime rule makes an index held by a
live executable graph unavailable to eager claims, so a per-op cohort that is both recorded and
run eagerly needs a record-mode index set per live exec graph. That grows the planned
context-side bytes by one slot per such cohort per context that records (more with the MoE
graphs). §5 (m) records the cost and the alternative that was not chosen.

**Findings.**

| id | finding | disposition |
|----|---------|-------------|
| I-A | the ring can be published absent: teardown drops the last contributor's ring with no L1 or RELEASING; RELEASING checked only at step 2; `ring_plan_gen` never read; nothing clears RELEASING after a (ii) refusal | **Changed** as ruled. Step 2 copies `ring_plan_gen` and the reused ring handles into the guard; step 8 (a), before the CAS, re-reads the generation and RELEASING under the ring lock and records a tentative contribution; every ring mutation bumps the generation; the teardown proc takes L1 and the ring lock and sets RELEASING while it drops the last contributor's handles; the guard's first phase clears RELEASING and bumps the generation on any non-publish (§2.4.2 steps 2 and 8, "Teardown", §2.7). H9: three RED routes (teardown in the window, RELEASING after step 2, the livelock). |
| I-B | old and new ring slots side by side demote the new context's KV permanently and mislabel the cause; a ring shrink is a carve | **Changed.** Reuse in place whenever the capacity covers the need, so a shrink never carves; the growth overlap is real, priced and labelled `ring-growth` with the old and new bytes; the causes are ordered `ring-growth`, `head_slot`, `capacity`; excess ring capacity is held until the last contributor leaves (§2.4.1, §2.7). H9 growth and shrink fixtures. |
| I-C | event-chained reuse covers eager submission only; recorded releases carry node events; master parks record-mode releases with no event; replays reuse slots unclaimed | **Changed** per the lead's ruling: a claim made in record mode lasts for the life of its exec graph and vacates at the graph's destruction on an eager event (the replay submission event or a marker after it), never a node event; master's park (`11faace69` `ggml-sycl.cpp:43378-43381`) is converted (§2.3.2 "Per execution mode"). Consequence stated above and in §5 (m). H4 record/replay model, H7ab, G1 replay case, C6's replay arm under STRICT. |
| I-D | the r4 addendum not applied at `b021c9629` | **Resolved in `c2613a688`** (r5 confirmed); unchanged here. |
| I-E | zhcn's A1-A4 missing at `b021c9629` | **Resolved in `c2613a688`**; zhcn's exact wording now quoted in §6.6. |
| I-F | M1 and M2 confirmed against code; M1's real holes are drain-and-close's early returns (`llama-context.cpp:162-219`) and the unwind zeroing (`:752`) before `create_memory` (`:869`); residual: capture the id | **Changed.** One call site, the `sycl_plan_guard` destructor, with the ContextId it captured at `create_exec` (`:796-801`); the drain-tail call is deleted (§2.4.2 "Teardown"; H7m, H9). M2 as in `c2613a688`, with zhcn's wording (§2.4.3). |
| m-a | step 8 drops superseded handles under L1 | **Changed.** The superseded handles move into the guard at step 8 (c) and drop in its second phase after L1 is released (§2.4.2 step 8 (d)); H9 case. |
| m-b | step 3 feeds head-slot bytes into jehw's `BUDGET_EXCEEDED` demotion (`:18118`) | **Changed.** In-arena head-slot bytes are excluded from that sum; the fit is the only admission for them (§2.4.2 step 3); H7ac. |
| m-c | `allocate_excluding` leaves two adjacent free blocks, which `check_invariants` rejects (`tlsf-allocator.hpp:760`) | **Changed.** It chooses an offset and carves with `allocate_at`'s semantics, so the remainders stay whole and coalesced (§2.3.3); H1 runs `check_invariants` after every call. |
| m-d | every allocation on a TLSF with pending ranges must honour them | **Changed.** H7aa enumerates the placement primitives' call sites on shared and RUNTIME TLSFs, with a mutation witness. |
| m-e | the pre-L1 registry reserve cannot guarantee "never rehashes" | **Changed.** The promise is withdrawn; a rehash is heap work with no device call, and the exception does not rely on its absence (§2.10). The ratification above is conditional on this. |
| m-f | the RS buffer's clear needs event-held slices; `rs_layers` must honour `offload` | **Changed.** The recurrent-state buft's clear holds per-extent slices on its event like the tiered clear (§2.6); `rs_layers` lists only offloaded layers on arena devices (§2.4.4). |
| m-g | the GA deficit differs across the designs; llama.cpp-uwlx missing from §4 and L4 | **Changed.** One number from one function on the live geometry, 6 layers (§2.4.1, H2, §5 (n)); zhcn rev 4 (`4388d34`) printed the same figure. **Corrected in revision 7.2** from −714.9 to −726.9 MiB (design-zhcn-r4: the SWA term at `-ub 1024` is 30.0 MiB, not 18.0; §6.8). uwlx is in step 5, §4's jehw seam, the landing text (jehw → uwlx → 423j; uwlx → L4) and L4's depends-on. |
| m-h | 1oxa rev 4 cites rev 5's over-plan fallback | Not moua's text; relayed by the lead. The fallback does not exist here (§2.3.2 "No `context_side_place`"). |
| m-i | citation drift in text `c2613a688` deleted | Recorded; none of those lines is cited here. |

**The post-r5 queue (lead-approved), applied.**

| item | disposition | where |
|---|---|---|
| M1 single site | the release proc is called only from the `sycl_plan_guard` destructor | §2.4.2 "Teardown", H7m, H9, §6.6 |
| M2 wording | `unified_allocate_owner`; "the commit only installs", a move; failure is a refusal before L1 | §2.4.3, §6.6 |
| uwlx | cited, with its dependencies | §2.4.2 step 5, §2.9, §4 |
| zhcn wording | A1-A4 quoted exactly; status "confirmed by impl-zhcn 2026-09-26, with amendments A1-A4 and M1-M2" | §6.6 |
| R7a | census row: `pinned_chunk_pool::mutex_` (`pinned-pool.hpp:313`) is L5, with a named exception for `grow_into` (`malloc_host` plus the timed future wait), retiring with llama.cpp-nrng (not moua's) | §2.10 |
| R7b | the first publish reserves host-tier headroom with growth allowed; later republishes pass `forbid_host_zone_growth = true`; a shortfall is a candidate refusal | §2.4.3, H4, H7ad |
| R8 (1) | no allocation and no logging under the per-slot spin lock; `[CONTEXT-PLAN-BUG]` is logged after it is released | §2.3.2 |
| R8 (4) | step (i) invalidates only this context's own exec graphs, on its own thread, no lock, outside `graph_compute`, distinct from the withdrawn cross-context clear; (ii) re-fits on the live TLSF, so a leased block counts as allocated | §2.4.2 tenant-only path, H4 |
| R8 (6) | a tenant-key match is an OK no-op | §2.4.2 tenant-only path, H4 |
| R8 (8) | the fit reads actual allocations; the GA pre-registration states the rounding and its margins (74.9 / 53.1 MiB then; 86.9 / 41.1 MiB after revision 7.2's correction, §6.8). zhcn's 2.9 MiB margin belongs to its I8-GE row (`-ub 512`: −130.9 against 128), which this design does not pre-register | §2.4.1, H2 |
| R8 (9) | the ring's context-side half is `unified_cache_get_planned_pp_moe_onednn_kv_zone_bytes` (`unified-cache.cpp:2222`) = slot × `ring_depth`, depth 1 in GA | §2.7 |

**The `c2613a688` verdict (supersedes the `b021c9629` one for `c2613a688`).** Carried items
(I-A, I-B, I-C, m-a to m-h) are dispositioned in the table above; the new and sharpened ones:

| id | finding | disposition |
|----|---------|-------------|
| (b) | the step-6 device carve is lock-safe; ratify, with text conditions | **Applied.** The lock order names the arena authority's registration beside `g_runtime_alloc_mutex`, and L7's tie-break orders all three L5 locks; the heap work is the four container inserts per registration; no "never rehashes"; a registration failure is not `[KV-PLAN-BUG]`; 423j's step-8 retire takes only the cache locks, submits no barrier and drops its withdrawn mirrors after L1 (§2.10, §2.8, §2.9). |
| I-A (sharpened) | RELEASING has no owner, so the releasing transaction's own (ii) returns `busy` against itself | **Changed.** RELEASING = `{owner ContextId, ring_plan_gen}`; the owner's (ii) is exempt at step 2 and step 8 (a); its guard clears the mark on every exit (§2.4.2 step 2, tenant-only path). H9 case. |
| I-C (ring) | the ring's release event goes stale under record/replay | **Changed.** §2.7 states the record-mode rule for the ring; a recorded holder takes a device-shared slot, so the depth counts it; L4 checks whether the ring is reached while recording (§2.7 "Claims"). |
| I-G | the tenant-only path counts graph-held blocks as free | **Changed** as ruled: (i) (a) invalidates this context's own executable graphs on its own thread with no lock, then the scheduler destroy, the occupancy check and move-out, zhcn's synchronous targeted reap `release_retained_referencing` (lead ruling "B"), a `use_count() == 1` check (`[CONTEXT-PLAN-BUG]` otherwise), and the drop, so every released block is really free before (ii); (ii)'s live TLSF is the belt (§2.4.2). H4 case, RED on `c2613a688`; H7ae. |
| I-I | the host tier: entry point, identity, ordering | **Changed.** (1) `unified_allocate_owner` with `must_host_pinned`, `use_pinned_pool`, `HOST_COMPUTE`, cohort `context-compute-host`, `require_host_usm_base` false, handed over by `from_owned_alloc`; on §2.10's allowlist; the SYCL_Host buft's `alloc_buffer` becomes a claim. (2) `device = -1` slots live in the cache and registry entry of the host buft's device (device 0 today), an entry with no extents if that device holds no layer. (3) New host slots are allocated before (i), old ones stay until the CAS and drop after L1, (0) checks the host tier, and step 1 is stated as pre-L1 (§2.4.2, §2.4.3, §2.4.4). H4 case. |
| I-J | the 423j terms are incomplete | **Changed.** (1) the retire withdraws and remaps the mirror, whose handles drop after L1; (2) the room returns at the next finalize pass (the `graph_compute`-end deferred-free pass), not at the lease drop; (3) "vetoed" means lease-only, with the other predicate vetoes excluded; (4) the entry point is 423j's on uwlx's surface, with its contract (both cache locks, no barrier, mirrors out, caller bumps the epoch) (§2.9). H5 (b)-(d), with a finalize-pass model and a negative case. |
| I-K | the vacant-with-event conversion drops the previous generation's retention on a re-claim | **Changed.** At a re-claim the previous `retained_owners` go to `retain_handles_until_event(previous done_event)`; the cache-side per-generation refcount is deleted for arena devices, because each claim's slice keeps a superseded slot's block (§2.7). H7z witness; H9 supersession case. |
| m-j | no backend-side equal-key rule | **Changed.** A matched key with an equal tenant key returns OK with the published residency, and the split case's later runs are that no-op (§2.4.2 step 1, tenant-only path). H4. |
| m-k | log after the claim spin lock | **Changed** (§2.3.2); H7w checks it. |
| m-l | the guard's own ContextId | Already in `99fd614da` (§2.4.2 "Teardown"). |
| m-m | 423j-retired copies and the fenced prefix | **Changed.** H7h covers 423j-retired copies; the fenced-prefix cost is stated as accepted, since "idle" is not observable at step 8, and the WARN makes a no-room retire visible (§2.9 item 3). |
| m-n | the ring's KV-zone function is a second source beside the fit's split | **Changed.** For arena devices the function returns the fit's recorded split × depth, set at step 8 (c)'s install; the fit never reads it (§2.7). |
| m-o | a registration failure is not `[KV-PLAN-BUG]` | **Changed** (§2.8 site 1, §2.10). |
| m-p | `host_arena_`'s line | **Fixed:** `:4235` at `2c4f5e45d`, `:4253` at `11faace69` (§2.4.3). |

**zhcn rev 4's requests (`4388d34` §3.8, sent after rev 7's first commit), and rev 4.1.**

| # | request | disposition |
|---|---|---|
| 1 | A1 recorded | Done (§6.6, quoted). |
| 2 | A4 recorded | Done (§6.6, quoted). |
| 3 | A2 recorded, gate in the H7e family | Done; the gate is H7y. |
| 4 | (i) in rev 4's step-4 order | **Superseded by the lead's ruling "B"**: a synchronous targeted reap replaces both waits and both `busy`-on-timeout states (§2.4.2 (i) (a)-(f)); the ring's old slots go through the same reap and were exempt from the use-count check, an exemption revision 7.2 replaces with r6 I-4's bound (§6.8). Rev 7's wait text (`a402c15af`) is withdrawn. |
| 5 | unreleased tenant blocks count as allocated | **Met by construction**: (ii) fits the live TLSF, where such a block is still allocated (§2.4.2 (ii)). |
| 6 | a per-slot reference read | **Kept as the step-(e) check** (lead ruling), applied to the moved-out batch after the reap; the batch is local, so it takes no lock. The reap's control set is the same batch (row 4b). |
| 7 | M2 | Done; the host allocation now precedes (i), per r5 I-I(3) (zhcn to mirror: "after (0), before (i)"). |
| 8 | the equal-key skip | Done; backend no-op, llama recreates the scheduler over the same slots (§2.4.2). |
| 9 | the element superset | **Closed by zhcn rev 4.1 (`4bb0436`)**, which accepts this design's element with `int32_t device` (-1 = host) and the cohort table; the tenant key digests `(device, cohort, slot_index, slot_bytes)` (§2.4.4). |
| 10 | M1 | Done in `99fd614da`. |
| 11 | the printed `free_after_full_kv` on live geometry | Done (§2.4.1); zhcn rev 4.1 agreed on −714.9 MiB, **corrected by both designs to −726.9 in revision 7.2** (§6.8), and I8-GE carries its own 2.9 MiB margin. |
| 12 | cite the KV-zone function | Done, with both master line numbers and m-n's one-source rule (§2.7). |

**impl-23mk (lead ruling on the second).**
- The KV layer descriptor gains `n_head_kv` and `n_embd_head_k` per layer, for
  `packed_k_sidecar_bytes(ℓ)` (§2.4.4). They block beni's sidecar conversion, not 23mk core.
- The vmem-kv refusal under an arena is carried by 23mk core, which lands first; L6 drops it
  and §2.6 cites 23mk (its gate reads `vram_arena_enabled()` today, not `arena_active()`,
  which is 23mk's to align).

**The lead's rulings on rev 7's flags (2026-09-26).** FLAG 1: the record-mode index sets are
accepted as planned cost; the per-replay refresh stays the documented alternative; G1 prints
the added bytes (§5 (m), §3.2). FLAG 2: `tier` and `scope` stay out of the element (the cohort
table is the one source); the waits are replaced by the reap (ruling "B"); the backend-side
equal-key call is an OK no-op, and llama may rebuild its scheduler over the same slots without
a republish (zhcn's to state).

**The `device` field's type (lead ruling 2026-09-26):** `int32_t`, -1 for the host tier, as
`a402c15af` and zhcn rev 4.1 have it; `99fd614da`'s `UINT32_MAX` is withdrawn. zhcn has
mirrored the host order ("after (0), before (i)"). **FLAG B accepted:** the ring's depth counts
a recorded holder's slot, and L4 checks whether the PP MoE oneDNN path is reached under
recording (§2.7). **The reap's store mutex** is ranked in §2.10's census list (L5, leaf).

### 6.8 Design review r6 (design-moua-r6 on `c2613a688..b30321a6f`), the lead's rulings, and the r6 queue

design-moua-r6 read revision 7.1 (`f2e5606bc`) and noted `b30321a6f`, which fixed m-4 in part.
Verdict: REVISE, 0 Critical, 6 Important, 10 Minor. It found no lock-order inversion (its
question 3) and confirmed the step order matches zhcn's ruled order. Revision 7.2 answers every
item, on top of `b30321a6f`.

| item | finding | disposition |
|---|---|---|
| I-1 | COMPLETE mode trusted a premise (synchronize waits every queue) that holds only for the device execution queue: a missed queue would be a silent free under queued work | **Changed (lead ruling, combined with zhcn's).** A named step (s) synchronizes every queue that can reach the slots, the CPU-dispatch and TP queues included, and zhcn rev 5 carries the list. After the unlock, the reap backstops each moved-out record: an incomplete event is waited, never freed early, and reported as `[CONTEXT-PLAN-BUG]` (a WARN, an abort under STRICT). It is not re-queued as pending (§2.4.2 (s), (i)(d); H4, H7ae). |
| I-2 | (a) named `sycl_exec_graph_clear_active`, which swaps the global `graph_unwaitable` list and clears the unsynchronized global leaf-staging map; the design relied on the first | **Changed (lead ruling).** (a) is a new own-context clear, gated on a recorded graph, that never reaches `release_graph_retained_handles` or `ggml_sycl_cpu_staging_cache_clear`. The reap scans `graph_unwaitable` by owner after its in-hand yield, so neither design depends on the global release. H4's "no other context touched" arm now models the real clear as its positive control (§2.4.2 (i)(a), (d); H4, H7af). |
| I-3 | (e)'s `use_count() == 1` contradicted the cached slot table the design itself specifies | **Accepted.** (c) takes the registry's and every backend context's table pointer, checks the table's own count, then moves the handles out. H7ag gates the three permitted holders of a tenant slice (a claim, the retained store, a graph container (a) clears); beni-style parks and `host_task` captures are errors to fix, not to exempt (§2.3.2, §2.4.2 (i)(c), (e); H4). |
| I-4 | the ring's blanket exemption from (e) hid the slot-state retention of the last generation, so the sole-contributor release freed nothing and (ii) refused silently | **Accepted: the narrower rule replaces the exemption.** The move-out takes each slot's `retained_owners` and `done_events`; guard pins are counted in `pinned[slot]` (step 2 increments, guard phase 2 decrements after its drop); (e) requires `use_count() ≤ 1 + P` with P snapshotted at the move-out. H4 carries the growth that now succeeds (RED on `f2e5606bc`) and the pin-plus-stray BUG. zhcn mirrors the rule (§2.4.2 "The ring"). |
| I-5 | the generation was copied after the fit's snapshot, so a release proc's step 4 in between let B publish an absent ring | **Changed (lead ruling: the first option).** Step 2's ring-lock section (RELEASING, the generation, the slot identities and handles, the pins) runs before the snapshot, and the fit takes the ring's slots from that copy. No new `busy` state is added. H9 carries the route, and H7ah gates the order (§2.4.2 step 2). |
| I-6 | the slot state's other `retained_owners` drops (reset on a depth change, generation 0, release-unused, rollback) free under queued work once the refcount is deleted, and destroy under an L5 lock | **Accepted.** Every removal moves the vector out and hands it to `retain_handles_until_event(done_events[slot])` after the unlock (§2.7). H7z covers all five sites, and H9 adds the depth change. |
| m-1 | "same size" and "shrinks it to B's" residues | **Fixed** (§2.4.3 reconciliation: capacity at least the need; H8's ring case keeps the held excess). |
| m-2 | the sync step was unnamed; retentions published after it | **Fixed.** Step (s) is named and listed. Nothing between (s) and (d) publishes a newer retention: (a) submits nothing (`graph_input_staging_clear` ignores its queue), the gallocr free takes the last event, and the vacate marker retains nothing (§2.3.2, §2.4.2 (s)). |
| m-3 | the in-hand publish's two conditions | **Fixed** (§2.4.2 (i)(d); zhcn rev 5's worker condition, cited). `in_hand` is published in the pop's section with an immutable copy of the control identities, and cleared after the drop or in the park's own section. A park during the yield is caught by the post-yield scan, not reported as a BUG. |
| m-4 | the store mutex's rank; "nothing waits under it" | **Fixed** (§2.10): it is last in the L5 tie-break, and the cv waits on it release it. |
| m-5 | the cited request omitted `owner_pending` | **Fixed**: the request and result are quoted as zhcn gave them, with zhcn rev 5 authoritative (§2.4.2 (i)(d)). |
| m-6 | GA's 988.0 assumes depth 1 and no record-mode index sets | **Fixed** (§2.4.1): GA is re-scored if L4 finds the path reached while recording. |
| m-7 | `unified_allocate_owner` cited without a sha | **Fixed**: `:15815` at `11faace69` (`:15797` at `2c4f5e45d`), §2.4.3. |
| m-8 | (c)'s claimed-slot read under the leaf | **Fixed**: the atomic flag only (§2.4.2 (i)(c); H7k). |
| m-9 | the state after a refused sole-contributor (ii) | **Fixed** (§2.4.2 "After a refused (ii)"; H4): an empty-ring claim is `[CONTEXT-PLAN-BUG]` with an error status, and the revert re-carves from empty because the key was cleared. |
| m-10 | host slots re-allocated on every key change | **Adopted (lead ruling):** reuse in place when the published cap covers the need, consistent with the device rule (§2.4.2, §2.4.3; H4). |

**The r6 queue (lead, zhcn, 23mk review addendum).**
- **GA (design-zhcn-r4).** The SWA term at `-ub 1024` is PAD(128 + 1024, 256) = 1280 cells =
  30.0 MiB, not 18.0. So `free_after_full_kv` = −726.9 MiB (not −714.9), and the margins are
  86.9 / 41.1 MiB (zhcn's G2 spare is 41.1). The 6-layer result holds. The full breakdown is in
  §2.4.1, and every figure in this document is corrected: §2.4.1, H2, §5 (n), and §6.7's rows
  m-g, R8 (8) and 11. Revision 6's −728.4 is withdrawn: no breakdown reproduces it.
- **The reap's names (lead ruling; zhcn):** `RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER` and
  `RETAINED_REAP_QUERY_EVENT_STATUS`; the request and result as quoted in §2.4.2 (i)(d).
- **The reap's conditions (design-zhcn-r4, via the lead), mirrored by citation:** the backstop
  after the unlock (with the lead's disposition, I-1 above); no destruction under the store
  mutex; the worker publishes its in-hand record's control identities; `host_task` lambdas
  publish retention through the store (H7ag). zhcn's step 1 enumerates every queue, and (s)
  cites it.
- **The scan (lead ruling, from this design's trace of the park route):** the reap scans
  `graph_unwaitable` by owner, after the yield. zhcn withdrew its "parked entry → BUG" arm; the
  arms are zhcn rev 5's, cited in H4.
- **zhcn confirmed** that 4a reaches `sycl_exec_graph_clear_active`. That path is replaced here
  by the own-context clear (I-2).
- **The cached slot table (design-zhcn-r4, via the lead):** a holder to drop before (e) (I-3).
- **23mk review addendum:** (1) revision 6's L6 row had listed VMEM_KV; `a402c15af` had already
  corrected it. §2.6 now also names the interim KV path before L6 (the raw fallback at
  `11faace69` `ggml-sycl.cpp:39267`, 23mk's rows 54/55). (2) Record mode takes one index per
  recorded use (§2.3.2, §5 (m)). (3) The descriptor's `n_head_kv` and `n_embd_head_k` were
  already named. They block beni's sidecar conversion, not 23mk core (§2.4.4).

**Open for the lead (resolved in revision 7.3).** Revision 7.2 adds no `busy` state: I-5 takes
the ring-lock-first option, not "else busy". The two `busy` returns that r5 I-A ruled in were
left unchanged here: step 2's RELEASING and step 8 (a)'s generation mismatch. How a `busy`
surfaces on llama-server is design-zhcn-r4's I-E. Rulings §E.1 retired both returns; §6.9
records the change.

### 6.9 design-moua-r6's rewritten verdict (on `c2613a688..b30321a6f`), and the lead's rulings file

design-moua-r6 rewrote its verdict to cover `b30321a6f`: 0 Critical, 7 Important, 11 Minor. It
scores `b30321a6f`, so most of its items were already answered by revision 7.2 (§6.8). The lead
ruled on the new items and put every shared ruling in one file, `lead-rulings-2026-09-26.md`,
which this document now cites by section (the header). Revision 7.3 is one commit on top of
revision 7.2 (`a5e195b57`).

| item | finding | disposition |
|------|---------|-------------|
| I-1 | the COMPLETE precondition rested on a synchronize that waits one queue | **Held as ruled (rulings §B step 1, §R "Backstop").** (s) lists every queue by name and line: the device queue, the split secondary queue (`:62143`), the MoE shared-context queues (`:5852`), the cache's queues, the MMID exact queue, the CPU-dispatch queue and the TP queues. An incomplete event after the unlock is waited, never freed early, and reported as `[CONTEXT-PLAN-BUG]`. Nothing is re-queued (§2.4.2 (s), (i)(d)). |
| I-2 | (a) was the process-global clear | **Held** from revision 7.2, and extended (rulings §B step 5): (a) also never unpins MoE experts or weights (`:99148-99149`; H7af). |
| I-3 | (e) contradicted by the cached slot table and by non-store parks | **Changed (rulings §B step 8).** (c) takes the registry's table pointer and every backend context's cached pointer into the batch. (e) checks the table's own `use_count()` first, and only then opens it and checks each slot handle's. §2.3.2 now says the table is dropped at (i)(c) and the new one is installed at the commit, not "immutable until the next publish". H7ag now fails any non-store park of a tenant slice: the oneDNN Graph scratch park, the q8 activation cache, and anything beni converts. |
| I-4, I-5, I-6 | the ring's slot-state retention, the step-2 order, the five drop sites | **Held** from revision 7.2; now cited as rulings §RING. |
| I-7 | GA wrong at six sites | **Held** from revision 7.2 (all six corrected; rulings §GA). Added: the pre-registered numbers are H2's run of the one function, not hand arithmetic (§2.4.1). |
| m-4 | the store-mutex census wording | **Fixed (§2.10).** The cv waits release the mutex (`:131-132`, `:2048-2049`, the reap's yield); one debug `fprintf` runs under it (`:159-161`); "the only lock held **during the scan**", because the destroy after the unlock takes allocator locks with nothing held; last in L5. |
| m-11 | every SYCL_Host `alloc_buffer` made a claim | **Fixed (§2.4.3, §2.10).** It is a claim only inside a claim scope. llama's output buffer (`llama-context.cpp:3492-3499`) and the LoRA and cvec tensors stay outside every scope and keep the existing path. |
| other Minor | m-1 to m-3, m-5 to m-10 | **Held** from revision 7.2 (§6.8). |

**Rulings adopted in the same revision, beyond the verdict.**
- **Rulings §E and §E.1: no `busy` from the ring.** A re-plan transaction mutex, rank L0,
  serializes every re-plan, probe, release proc, load and optional-layout pass (§2.4.2 "The
  re-plan transaction mutex"). bbae3a703 made it per device; rulings §E.2 made it one
  process-global mutex in revision 7.4 (below). The two ring `busy` returns are retired: a
  RELEASING mark this call does not own, and a step 8 (a) generation mismatch, are both
  `[CONTEXT-PLAN-BUG]`. RELEASING, the generation and the pins stay as checked invariants, and P
  is always 0 under L0. Teardown's path to the mutex is stated, with a debug check for
  same-thread re-entry. Gates: H7ai (every ring mutator outside decode takes it) and H9's
  two-context concurrent re-plan with zero `busy`. The H9 absent-ring routes are now forced
  through a test hook that skips L0, and must report the bug.
- **Rulings §B step 4: reuse per slot, device and host.** On the (i) path a device slot whose
  cap covers its new need is kept, not reaped, and (ii)'s fit reuses it (§2.4.2).
- **Rulings §R: the result carries `pending_bytes`**, in the ruled order (§2.4.2 (i)(d)). The
  reap's owners are only the slots not reused.
- **Rulings §FM and §STRICT** are cited where this design sizes from free memory (§2.2) and
  counts refusals (§2.8).

**Raised for the lead, resolved by rulings §E.2 (revision 7.4).** Per-device L0 does not close
the plan-identity checks on the transaction path, because the published plan is **one
process-global atomic** (`g_placement_publication`, read at jehw `c41fed119`
`ggml-sycl.cpp:2676`). A publish on any device, or any load, trips them:

| site (jehw `c41fed119`) | return | reached under per-device L0 by |
|---|---|---|
| `:17672` stale identity | `busy` | a publish by another device's transaction or a load |
| `:17686` live-update lease | `busy` | another live update of the same model, on another device |
| `:17691` plan changed while acquiring L1 | `busy` | as `:17672` |
| `:17914` the yield's relock | `busy` | as `:17672`, during the unlocked window |
| `:18386` publication CAS lost | `refuse` (PLAN_REJECTED, fatal on the server) | as `:17672`, between the plan and the CAS |

The recommendation is that a transaction take L0 on every device the published plan covers, in
ascending order. That closes all five, which then become `[CONTEXT-PLAN-BUG]`, at no measurable
cost, since re-plans and loads are rare. The alternative is to make the plan check per device,
which is jehw's code. The ring's own `busy` at `:18274` is not on this list: after (s), a
claimed ring slot is step 7's occupancy bug.

**zhcn rev 5 (`0089dc6`) §3.8 deltas, checked against `b30321a6f`.**

| row | delta | disposition |
|-----|-------|-------------|
| 6c | covered slots reused in place; only the growing ones released | **Held** (rulings §B step 4; §2.4.2). This design also releases an index the candidate no longer uses on the (i) path, since §B step 4 reuses only a slot the candidate still has. The all-covered fast path keeps unused indexes as held room, as zhcn's does. |
| 8 | GA −726.9, G2 spare 41.1 | **Held** from revision 7.2 (rulings §GA). |
| 12 | (c) empties the table object, then resets the cached pointers | **Differs, and follows rulings §B step 8 and the lead's I-3 wording:** (c) takes the table pointers, and (e) checks the table's `use_count()` before opening it. Emptying a table that another holder may still read would be the mutation r6 I-3 warned about. The end state is the same. |
| 13 | a new step (c′), `ggml_backend_sycl_release_buffer_refs(backend, owners)`, that erases owner-matched entries from backend-context caches, plus a holder scan when (e) fails | **Superseded by rulings §B.2 (revision 7.5).** Revision 7.4 adopted (c′) as §B.1's step 8′; §B.2 withdrew it, because every listed cache only compares identity. The fields become a non-owning `mem_handle_identity` at the producer, which is zhcn's scope (§2.4.2 (i), "No step 8′"). The holder scan is kept (§2.4.2 (e)). |
| 14 | the `graph_unwaitable` scan after the yield, `unwaitable_dropped` | **Held** from revision 7.2 (rulings §R). |
| 15 | (a) gated on recorded state; context-scoped clear | **Held** from revision 7.2, plus rulings §B step 5's "never unpin". |
| 16 | (c)'s "never its claim" contradicts record mode | **Fixed** (§2.4.2 (i)(c)): a record-mode claim lasts for its graph's life (rulings §REC), and (a) vacated it on an eager event before (c). |
| 17 | non-STRICT (e): log, drop the batch, continue | **Adopted** (§2.4.2 (e)); the live-TLSF belt makes it safe. |

**Revision 7.4: L0 is process-global (rulings §E.2).** The lead ruled on the transients above:
L0 is one process-global re-plan mutex, `g_replan_txn_mutex`, not one per device. "Every
device the plan covers" was not closed under a publish that adds a device. Changed:
- §2.4.2 "The re-plan transaction mutex": one mutex, no device order. All seven former returns
  are `[CONTEXT-PLAN-BUG]`: the two ring ones and jehw's five (`:17672`, `:17686`, `:17691`,
  `:17914`, `:18386`). `:18274` is superseded by step 7. The accepted cost is that re-plans and
  loads on different devices serialize; decode and compute never take L0.
- §2.3.1: reservations are serialised end to end by L0; only runtime weight allocations can
  land in a yield window, and the pending ranges still exclude them.
- The teardown release proc, the probe, the RELEASING rules and the §2.10 census row name the
  one mutex.
- H7ai checks every holder takes the one mutex and that no per-device mutex exists. H9's
  concurrent test now covers one device, two devices, and a re-plan racing a load.
- The teardown release proc's retention move-out to `retain_handles_until_event` is confirmed
  by the lead (§2.4.2 "Teardown").
- **Rulings §B.1 (zhcn row 13), in the same revision (superseded in revision 7.5 by §B.2,
  below):** step 8′ is (c′) in §2.4.2 (i), with the closed four-member cache list; (e)'s holder
  rule and H7ag name the fourth class; the §6.9 row 13 disposition is updated. The earlier text
  that called the q8 activation cache (`:6648`) a forbidden park is withdrawn: its
  `cached_src_handle` is on the list.

**Revision 7.5 (rulings §B.2, §D15, §D16; master is now `3d9414c8c`, which contains jehw).**
- **§B.2 supersedes §B.1.** Step 8′ (c′) and the fourth holder class are deleted; the holder
  classes are §B's three (§2.4.2 (i), (e); H7ag). The q8 `cached_src_handle` fields and the
  `moe_ids_cache` keys become a non-owning `mem_handle_identity`, at the producer, in zhcn's
  scope. The withdrawal of "the `:6648` activation cache is a forbidden park" stands, for a new
  reason: it stops being a park because it no longer owns a handle, not because it is on a list.
- **zhcn rows 6c, 16, 17, as resent with 5.1 (`1e3f54b`), adopted; none conflicts with the
  rulings.** 6c: on the growth path only replaced device slots go into the batch, and covered
  device slots go back into the registry entry with the host slots, for the commit's new table
  (§2.4.2 (e), "Device slots are reused per slot"). 16: a record-mode claim lasts for its
  graph's life and (a) vacated it (§2.4.2 (c)). 17: without STRICT, (e) logs, drops the batch
  and continues behind the live-TLSF belt.
- **§D15:** the first publish reserves and holds host-slot room for the plan's maximum tenant
  demand; a need beyond it is a step-4 refusal (§2.4.3).
- **§D16:** the backstop count is carried in the `[CONTEXT-PLAN-BUG]` line and a
  PRIVATE_TESTING counter; §R's five-field result stands (§2.4.2 (i)(d)).
- **Line numbers.** jehw landed, so jehw `c41fed119` citations now map to master `3d9414c8c`.
  The five §E.2 sites are, at `3d9414c8c`: `:17867` (stale identity), `:17881` (live-update
  lease), `:17886` (plan changed while acquiring L1), `:18122` (the relock), `:18646` (the lost
  CAS), and `:18515` is the superseded ring `busy`. The other jehw citations keep their
  `c41fed119` lines, named as such.

### 6.10 Design review r7 (design-moua-r7 on `5925f3fe1`) and the lead's rulings §L0R, §M7, §D15

design-moua-r7 found 0 Critical, 7 Important and 11 Minor. The lead ruled in rulings §L0R and
§M7 and tightened §D15. Revision 7.6 answers every item; none is deferred.

| item | finding | disposition |
|------|---------|-------------|
| I-1 | the L0 census names operations, not the publishers of the one fact L0 protects | **Changed (rulings §L0R).** §2.4.2 "The re-plan transaction mutex" lists every public entry point that reaches a publish or `prepare_live_update` call site at `3d9414c8c`, with lines: the wrapper (its ticket `:18867` and publish `:18955`), the probe and FA recheck, activate (`:15259`), unload and quarantine restore (`:12308`, `:12329` → `:12547`, `:12599`), and each load entry. Decode's four identity-preserving republishes are allowlisted behind a gate. H7ai checks every call site; H9 runs every entry against a parked holder. The probe and the recheck validate against `lifecycle_select_placement_plan(model)`, with an H9 arm (A in a transaction, B loads, A's probe and recheck). |
| I-2 | the tenant-only path's L0 scope contradicts itself and zhcn | **Changed (rulings §L0R).** One RAII token type with a thread-local held flag; a nested acquire is a no-op hold, the release proc's token is outermost-only, and H9 has a positive nesting arm and a negative release-proc arm. Growth takes L0 before (s), in llama's `replan_scope`; the equal-key and covered paths take none, because they end at a new read-only coverage query that publishes nothing (§2.4.2 "The coverage query"). |
| I-3 | §RING's bound fails on the sole-contributor growth, by two routes | **Changed (rulings §M7 I-3).** (a) The moved-out `retained_owners` are backstopped and dropped in (d), before (e), never held in the batch. (b) (0) counts the sole contributor's old ring free by arithmetic and takes no ring-lock copy or pin, so P = 0; (e) reports a nonzero P as `[CONTEXT-PLAN-BUG]`. H4's ring arm is GREEN-reachable, with (0), (d)-order and nonzero-P arms; H7ak gates both. |
| I-4 | a model load releases a ring other contexts contribute to | **Changed (llama.cpp-r7fz, in this design's ring scope; rulings §M7 I-4).** On arena devices `populate_inventory_globals` neither releases nor writes the ring record; contributions carry their own model's per-row bytes and depth (§2.7). Gate H7aj; H9 "load B while A contributes", RED on master. |
| I-5 | transient refusals remain on the re-plan path | **Changed (rulings §M7 I-5).** (a) (0)'s placements are this call's pending ranges before (f), and (ii) re-fits only inside them, so no allocation that skips L0 can take the freed room; H4 "the freed room is held", H7al. (b) Step 7 draws only from planned room: the MMID device pool is a RUNTIME head slot carved at step 6, and its host pool is a view of a carve the model's MMID entry holds at the plan's maximum (§2.4.2 step 7); H7am. No other transient was found on the re-plan path. The FA recheck's live free-memory read (below) is a §FM finding, not a transient refusal. |
| I-6 | the §D15 reservation is recorded, not held | **Changed (rulings §D15).** The first publish allocates one owner-first host carve at the plan's maximum tenant host demand and holds it; host slots are views of it at the maximum caps; a need beyond it is an arithmetic refusal at (0). The old+new host peak text is deleted (§2.4.2, §2.4.3). H4's concurrent-fill arm is deleted; the new arm is RED on 7.5's recorded-only reservation. H7an. |
| I-7 | the holder census misses parks, and zhcn deviates on `g_data_ptr_cache` | **Changed (rulings §M7 I-7).** §2.4.2 "The holder census" names `graph_input_staging` (eager and record mode), `g_moe_ids_d2h_cache`, `g_moe_prompt_admission_cache` (fixed at the key type) and `g_data_ptr_cache` (the §B.2 key fix, with zhcn's exit clear only in addition). H7ag names all four with mutation witnesses. The `graph_input_staging` class is aligned with zhcn. |
| m-1 | zhcn agreement: bcs queue, covered path, table opening | **Changed.** The table is checked and opened at (c), as zhcn does; the covered path takes no L0 and writes nothing; zhcn is asked to add `get_bcs_queue()` with the TP worker and pipeline queues. |
| m-2 | stale "busy" text | **Fixed** at step 5's relock, step 4's CAS, H9's failpoint list, the 423j WARN, §2.3.1 item 2, the guard's rollback sentence, and (ii). |
| m-3 | a nonzero P accepted silently | **Changed:** reported as `[CONTEXT-PLAN-BUG]` ((e); H4). |
| m-4 | header hygiene; five sites at `c41fed119`; (s) at `11faace69` | **Fixed.** The rulings list is complete; master is `3d9414c8c`; the five sites and (s) and (a) are re-pinned in the body. |
| m-5 | the bind's `retained_owners` clear is a sixth removal site | **Changed:** listed in §2.7 with the hand-off, a debug empty-check and a witness; H7z and the L6 row say six. |
| m-6 | §D16's counter missing from (d) and H4 | **Fixed:** (d)'s backstop text and H4's backstop arm (counter 1 there, 0 on every other GREEN arm). |
| m-7 | "decode never takes L0" is too broad | **Fixed:** graph compute and dispatch never take it; a `sched_reserve` re-plan does. |
| m-8 | ":18515 is superseded" holds only on arena devices | **Fixed:** scoped to arena devices; a device with no arena keeps u1bb's ring re-plan. |
| m-9 | the load's L0 span | **Changed:** L0 at the top of each public load entry, not across the load, so neither llama's loader nor `progress_callback` runs under L0; the load's state is its bound candidate. |
| m-10 | llama's BUSY sleep loops survive | **Changed:** deleted (rulings §L0R), in zhcn's scope; H7ai's witness restores one. |
| m-11 | a covered candidate whose ring grows | **Changed:** covered requires a non-growing ring contribution; a ring growth takes the growth path (H4 arm, RED on 7.5's rule). |

**Found while re-pinning, reported to the lead.**
- The FA recheck (`:19041`) gates on a live device free-memory reading (`:19058-19075`), which
  rulings §FM forbids; on arena devices it now reads the registry (§2.4.2 "The probe and the
  FA recheck").
- The wrapper's module-admission `BUSY` (`:18848-18849`) is reachable only when the module is
  not ACTIVE (reactivation or shutdown). Under L0 it becomes `[CONTEXT-PLAN-BUG]` with the
  wrapper's other returns; a legitimate shutdown race, if one exists, would need its own
  answer.
- The load's L0 span is a choice (per entry, not load_begin..load_end). The alternative covers
  the whole load and needs a callback clause in the deadlock rule.
- The MMID host pool is held by the model's MMID entry, not a context's reservation, because
  the workspaces are keyed by the model token.

**zhcn rev 5.2 (`014abd5`) and 5.3 (`b171100`, `6afd110`) §3.8 rows, checked against 7.6.**

| row | delta | disposition |
|-----|-------|-------------|
| 13 | §B.2 supersedes §B.1 | **Present** since 7.5. |
| 19 | §D15 held host reservation | **Present** (§2.4.3): one carve, host slots are slices at the maximum caps, a larger need is refused by arithmetic. zhcn's D20.1 (warmup is an ordinary setter, no warmup term) matches "the setters and warmup are not ladder candidates". |
| 20 | the (d) owner vector dies before (e) | **Adopted** (§2.4.2 (d)). |
| 21 | the moved non-FA check runs only for `!tenants_planned` | **Adopted** (§2.4.2 step 4, §2.11). |
| 22 | cite gate 30; the scatter lists need no wait (C2t) | **Adopted** (§2.4.2 (s)). |
| 24, 25, 27 | §M7 I-3, I-5(a), I-7 | **Present** in 7.6. zhcn aligns to 7.6's covered path (no L0, no writes, the coverage query), the table opening at (c), and (0)'s pending ranges in its r5 round. |
| 28 | the `context-graph-stage` cohort | **Adopted** (§2.4.3 producers): a device head-slot index set with eager and record sets, carved by (ii). |
