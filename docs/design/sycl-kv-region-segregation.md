# llama.cpp-moua: planned, lifetime-segregated layout for the shared KV+WEIGHT zone

Design, revision 7.14h. Author: impl-moua, 2026-09-27. The revisions answer seventeen reviews:
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
  commit on top of `5925f3fe1`;
- design review r8 (design-moua-r8 on `874fe5490`: 0 Critical, 5 Important, 12 Minor), the
  lead's rulings on it (§M8), and the items queued for this round (§M76's conditions 1, 2 and
  5, §M76a, §Z42.3, §ZR5 I-1 to I-3, the llama.cpp-fsgi pointer), recorded in §6.11. Revision
  7.7 is one commit on top of `db609bd15`, the merge of master `76c7f6548` into `task/moua`.
  Revision 7.7a is two commits on top of 7.7 (`307a4ddb1`): `85635feee` (committed as "7.8"
  before the lead named the round 7.7a) and this one. It carries the lead's ruling on the
  allowlisted republish, zhcn 5.4's §3.8 rows 19, 22, 26, 27 and 30, the two cross-design items
  closed with zhcn, the whole-tree census for the three deleted setters, and the pool phase
  gate's provenance rule, all recorded in §6.11;
- design review r9 (design-moua-r9 on `85635feee`: 0 Critical, 4 Important, 11 Minor), the
  lead's rulings on it (§M9), §M77, and the queue that arrived while r9 ran (§Z6 I-C, §Z43.4,
  §Z5 IMP-6, llama.cpp-dhpw, one cite), recorded in §6.12. Revision 7.9 is one commit on top of
  7.7a (`f8a628420`). It is numbered 7.9 because `85635feee` was committed as "7.8".
  Revision 7.10 is one commit on top of 7.9 (`53908bce5`). It answers design-moua-r9's
  addendum on 7.7a (I-5 and m-12 to m-14, rulings §M9a), 23mk's four additions to the
  pending-range primitive (rev 4.4 §8.1) and zhcn 5.5's four relays, recorded in §6.12;
- design review r10 (design-moua-r10 on `85635feee..9d4d826b3`: 0 Critical, 6 Important, 7
  Minor), the lead's rulings on it (§M11), §M10, and design-1oxa-r7's I-3 (§X7, the load's room
  is admitted at the early stage), recorded in §6.13. Revision 7.11 is one commit on top of 7.10
  (`9d4d826b3`). Revision 7.11a is one commit on top of 7.11 (`c445f3c46`): §Z8 I-2 (relayed as
  §M11a: the term filter on the clear and the fit), design-23mk-r6's H1 arm, zhcn 5.7's and
  f6218f3's relays, and the lead's decisions on m-6 and m-5, recorded in §6.13; it is two
  commits, `0fc9f8c5e` and the decisions;
- design review r11 (design-moua-r11 on `9d4d826b3..c66848c26`: 1 Critical, 4 Important, 13
  Minor), the lead's rulings on it (§M12), and the post-r11 queue (§M11b, §M11b-amend, §Z9 I-3's
  always-compiled witnesses, §Z9a's H4 (b) limits), recorded in §6.14. Revision 7.12 is one
  commit on top of 7.11a (`c66848c26`). Revision 7.12a is one commit on top of 7.12
  (`e64dac7db`): the lead's §M13 and §M13a, which fold 23mk rev 4.5's items, the `DEVICE_TERM`
  interim (§Z10.1), 1oxa r8's I-D (§X9), 23mk r7's I-A (§Z11), zhcn `8a58ad4`'s H4h hooks and
  ldvb's ownership of row 648, recorded in §6.14.
- design review r12 (design-moua-r12 on `c66848c26..7f4808729`: 1 Critical, 3 Important, 15
  Minor), the lead's rulings on it (§M14, §M15), §Z13.1, the late-stage string's term slot, zhcn
  5.9's H4h vehicle (`4f8ff34`) and §Z14.2, recorded in §6.15. Revision 7.13 is one commit on
  top of 7.12a (`7f4808729`), and revision 7.13a is two commits on top of 7.13 (`79abac034`):
  `7fb6afc2a` (committed as "7.13, second commit" before the lead named the round 7.13a) and
  this one. They fold §M15 and §Z14.2, which crossed 7.13; this one adds §Z14.2's reason and the
  name.
- design review r13 (design-moua-r13 on `c66848c26..8e1c5c094`: 1 Critical, 6 Important, 10
  Minor; its I-F is the §M17 flag), the lead's rulings on it (§M18), and the queue folded with
  it (§M16, §M16a and §M16b on `replace_within`, §M17 and §M17a on the zone terms, 23mk r8's two
  items, and the zone-term enum proposed to impl-23mk), recorded in §6.16. Revision 7.14 is one
  commit on top of 7.13a (`8e1c5c094`). Revision 7.14a is one commit on top of 7.14
  (`4dfa587e7`): the lead's §M18.3a (PRIMARY and OPTIONAL planned copies), the relay that
  crossed 7.14 (r13 I-F (5), m-f, zhcn r11's H4 (b) literal and route, §Z15), 23mk's table-only
  requests (23mk 4.7b `e81dc2327`, 4.7c `175dcd51b`), §M19, the lead's §M20 on the items first
  held for its fact-finder (row 134, `moe_control`, the RUNTIME figures in exact bytes), §V12
  and 1oxa rev 10a's relay, recorded in §6.17. It is two commits: `def60ce6c` and `9f68c61bd`.
  Revision 7.14b is one commit on top of 7.14a (`9f68c61bd`): the lead's §M21 and §M22 on
  7.14a's points, and the §M20 message that crossed 7.14a, recorded in §6.18.
- design review r14 (design-moua-r14 on `8e1c5c094..44b4b9d66`, judged at 7.14a `9f68c61bd`:
  0 Critical, 6 Important, 11 Minor), the lead's rulings on it (§M25), §M23 and §M24 (23mk
  4.7d's byte mirror and zhcn 5.11's relay) and §V13 m-2, recorded in §6.19, with r14's
  addendum on 7.14b (m-12 to m-16). Revision 7.14c is two commits on top of 7.14b
  (`44b4b9d66`): `f05d67195` and `7b2cb5898`, which folds the addendum. Revision 7.14d is one
  commit on top of 7.14c (`7b2cb5898`): the lead's §M27 on 7.14c's three questions, with its
  (2a) addendum, §M26 I-3 and I-4 where they bind this design, and §M26a I-1, I-4 and m13,
  recorded in §6.20. Revision 7.14e is one commit on top of 7.14d (`877c868e0`): the lead's
  §M28 on 7.14d's three questions, with the code fact that decides the ring's class, §M29 on
  the oneDNN scratchpad (it amends §M19) and §M29a (it amends §M29), §V14 I-E, 1oxa rev 11's
  relays (`f6d3015`) and zhcn 5.12's (`d70c2d2`), recorded in §6.21.
- design review r15 (design-moua-r15 on `44b4b9d66..8fee92a67`: 1 Critical, 6 Important, 9
  Minor), the lead's rulings on it (§M32), §M33 I-G, §V16 I-A, the probe results (§M31, §M31a),
  §M30, §V15 with §V15a, §V16a, §M34 (4), §M35, and the queue held since 7.14e (23mk 4.8's two
  items, 1oxa rev 12's classifier, zhcn 5.13's L4 arm), recorded in §6.22. Revision 7.14f is one
  commit on top of 7.14e (`8fee92a67`). Revision 7.14g is one commit on top of 7.14f
  (`ee484eb5b`): the rulings made on 7.14f and its questions (§G1, §G1a, §V17's items for this
  design, §M36's relays, §M37), 23mk 4.9's names, 1oxa rev 13's relays and zhcn 5.14's two
  questions, recorded in §6.23.
- design review r16 (design-moua-r16 on `8fee92a67..ee484eb5b`: 2 Critical, 5 Important, 3
  Minor), the lead's rulings on it (§M38), §G1b, §Z20 where it binds this design, and the note
  that two of the thirteen scratchpad sites are dead, recorded in §6.24. Revision 7.14h is one
  commit on top of 7.14g (`3d782e264`).

**The lead's rulings file.** The rulings shared by zhcn, moua, 1oxa, 23mk and jehw/uwlx are in
one file, `lead-rulings-2026-09-26.md` (sections §B, §B.1 (superseded), §B.2, §R, §RING, §E,
§E.1, §E.2, §L0R, §M7, §REC, §T, §L6, §GA, §FM, §STRICT, §D15, §D16, §Z3, §Z42, §Z52, §D20
(superseded), §D20.1, §M76, §M76a, §ZR5, §M8, §Z43, §Z5, §Z6, §M77, §M9, §M9a, §M10, §Z6x, §X7,
§M11, §M11b, §Z8, §Z9, §Z9a, §M12, §Z10, §X9, §Z11, §M13, §M13a, §Z12, §Z13, §M14, §Z14, §M15,
§V11, §M16, §M16a, §M16b, §M17, §M17a, §M18, §Z15, §M18.3a, §M19, §M20, §V12, §M21, §M22, §M23,
§M24, §M25, §V13, §M26, §M27, §M26a, §M28, §M29, §M29a, §V14, §M30, §M31, §M31a, §M32, §M33,
§V15, §V15a, §V16, §V16a, §M34, §M35, §G1, §G1a, §V17, §M36, §M37, §Z20, §M38, §G1b). §M11a is a
relay line inside §Z8, not a section, and is cited as §Z8 I-2 (r12 m-14). This document cites it
as "rulings §X".
**Where this document paraphrases a ruling and differs from the file, the file wins.**

Revisions cited:
- **Current master is `3d9414c8c`, which contains jehw and u1bb** (jehw landed). Revision 7.6
  re-pins to it the text that r7 checked: the header, §2.4.2's L0 block, step (s), step (a),
  the §RING removal sites, and every line new in 7.6. Those cite `3d9414c8c` explicitly.
  Master has since moved to `76c7f6548` (vuy0, build files only), which `task/moua` merged at
  `db609bd15`. vuy0 touched no cited source line: `git diff 3d9414c8c db609bd15` over `ggml/`,
  `src/`, `common/` and `include/` shows only the build files and this task's own L1 files, so
  every `3d9414c8c` citation also holds in the worktree. Every line new in 7.7 was checked with
  `git show 3d9414c8c:<path>` (r8 item 0).
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
  (a context; the ring's rows are the context's too, rulings §M32 I-1). The fit places those
  slots first, as **head slots**, and KV demotes around them. The commit carves each one as a
  **reserved slot**: an owner-first `CACHE_SUBALLOCATION` whose `mem_handle` the owner holds
  (the context's registry entry). An allocation of that cohort **claims its slot by
  index** as a slice lease, and slot reuse is **event-chained**: a claim is released at
  submission with its completion event, and the next claim of that index depends on it. So
  non-LIFO or event-deferred frees cannot fragment planned room, and nothing waits on the host.
  This is the one tenant protocol shared with zhcn (§2.3.2; agreed 2026-09-26 with
  zhcn's amendments A1-A4, §6.6). The ring's rows,
  the compute buffer and fattn slot (zhcn), the per-op scratch (beni), the MXFP4 MoE TG caches
  and fattn workspaces (jzvq) and the recurrent state (moua) are all such tenants. There is no
  estimate, no floor and no per-row constant (§2.4.3).
- **Inside the transaction** (§2.4.2), the order is:
  1. match the idempotence key (a matched key takes the tenant-only path, which never
     re-fits KV and never yields);
  2. plan: reconcile the demand records, snapshot, fit;
  3. run the byte-accounting steps, which on an arena device size the MMID workspace and
     demote nothing (the fit is the one source of "fits");
  4. make every predictable refusal, so a refused candidate never yields;
  5. record **every** placement the fit made as a pending range, then yield (begun under L1,
     finished with L1 released);
  6. commit-carve, re-fitting only inside this call's own pending ranges, at exact offsets;
  7. materialize the context's MMID workspaces, where the route is reachable, into their planned
     carves, allocating nothing;
  8. run the publication CAS, commit (install the context's slot table, its ring slots among
     them), and only then, after L1 is released, drop the slots the new plan superseded.

  In this path nothing held before the transaction is released before its publish, so a
  rollback never has to re-acquire room (r4 I5, I6). The tenant-only path's pre-L1 release is
  the one exception, and it is zhcn's step (i). A two-phase guard declared before L1 rolls
  back on any refusal: its L1 phase clears the pending ranges and discards the registry
  insert, and its second phase drops this call's new handles with no lock held (§2.4.2).
- **Error path:** a device-planned KV layer that does not land in its region, and a claim that
  exceeds its slot or names no slot, are plan violations. They are logged at ERROR, return an
  error status that reaches the graph (never a skipped op), and abort under
  `GGML_SYCL_STRICT_LEASES=1`, the switch for ownership, lifetime and plan defects (rulings
  §G1, §G1b). There is no fallback
  of any kind: a context-side miss never reaches a raw allocation, and never takes unplanned
  room (§2.4.3).
- **Holes:** weight-side holes and buried optional tenants are usable region extents
  (regions are multi-extent). Only holes smaller than one slot are a limit (§2.9).
- **The load plans its first context's room (rulings §M32 C-1, §M38 C-1).** The pack charges
  each expert it admits to a device its weight bytes plus its share of the ring's rows, sized by
  the experts resident on that device (the batched executor refuses a non-local expert), and
  the load reserves the rest of the first context's head slots as a `FIRST_CONTEXT` pending
  range in the shared zone, which the pack cannot fill and which the model's first context on
  the device draws from (§2.4.2 (b), step 3). **The reservation has one source:** it is the
  head-slot set that the context transaction's fit places, from the same function
  (`context_demand_records`, §2.4.3), evaluated at the load envelope's shape, never a list
  kept beside it. The envelope carries the caller's `n_ctx` from llama.cpp-fkpg (a), so L4 does
  not land before fkpg (a) (§4). RUNTIME has no floor on an arena device: its capacity is the
  planned sum of its named consumers (rulings §M32 I-2, §M38 C-2).
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
  L4 and L6 land as one commit (rulings §M32 I-6; §4). Until beni converts a site, that site
  keeps today's placement and is not context-side (the transition rule, §2.4.3).

## 0.1 Acceptance conditions: the four principles

Every step of this design, and its reviews, are held to these. Each row names the mechanism
that enforces the principle and the check that shows it.

| | principle | enforced by | shown by |
|---|---|---|---|
| **P1** | The unified cache is the only allocator. There is no out-of-arena fallback for planned KV or for a planned context-side cohort. | KV regions and reserved slots are carved from the arena TLSF (§2.4). The tiered device branch issues no `unified_alloc` (§2.6). A region reservation never falls through to `unified_cache_malloc_device_tracked` (§2.8). A context-side claim allocates nothing: it is a slice of a reserved slot, and a miss returns an error status with no fallback (§2.4.3). | H7a/b/p; C1 counts **zero** KV-role EXT-ALLOC lines, and total EXT-ALLOC bytes no higher than the base run (§3.3). |
| **P2** | `mem_handle` / `alloc_owner` are the only ownership surfaces. | Each region **extent** and each **reserved slot** is an owner-first `CACHE_SUBALLOCATION` with its own `mem_handle`, held by its owner: the context's registry entry, the ring's rows included (rulings §M32 I-1). The registry, each KV buffer, and each layer's view hold the extent handles they touch. A KV slot and a tenant's claim are `slice()`s, not allocations, so a reserved slot outlives its last claim by refcount and there is no orphaned state (r4 I3). Retained runs have no owner and are never handed out as ownership (§2.3.2). Final handle drops never happen under a listed lock (§2.4.2, §2.10). | G1 checks `arena_owns` and the handle refcounts. H7f checks that no raw pointer is stored as region state. H9 and H7t check where the drops run. |
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
  KV region carve sets KV_REGION; the ring's rows and `backend-buffer-kv-zone` set CONTEXT.
  CONTEXT and TRANSIENT are both `alloc_role::COMPUTE`, so the role alone cannot separate them.
- A context-side call site does not allocate at all once its producer exists: it **claims**
  a reserved slot by `(owner, cohort, slot index)`, where the owner is its ContextId (read from
  the backend context's `execution_context_id`), the ring's rows included (§2.3.2, r4 I1). The
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
| `OPTIONAL` | optional layout copies (jehw's `optional_layout`). Two kinds (rulings §M18.3a): an unplanned copy, outside every live model's admitted plan, which an arena device no longer stages (§2.4.2 (b), "The range bytes"); and a planned **OPTIONAL** copy, a duplicate layout whose primary layout is resident on the same device. A planned **PRIMARY** copy (every other planned copy) is `WEIGHT`, never optional, never an eviction or yield candidate, and freed only at unload. A planned OPTIONAL copy is never an eviction candidate either; it is yieldable only to a context's KV admission, through the yield path (§2.4.2 step 5) | unplanned: weight frontier; planned OPTIONAL: inside its model's `WEIGHT` range, where the fit sees it as a buried tenant (§2.9) | unplanned: `allocate_gap_front`, tag `TAG_OPTIONAL`; planned OPTIONAL: `allocate_within({LOAD, txn}, WEIGHT, ...)`, tag `TAG_OPTIONAL` |
| `KV_REGION` | the per-(ContextId, device) KV region, including each layer's persistent packed-K sidecar as a companion slot when the sidecar is enabled (§2.4.1) | context | the extents chosen by `kv_region_fit` (§2.4) |
| `CONTEXT` | the ring's activation and output slots, per (context, device) (§2.7; rulings §M32 I-1); zhcn's compute chunks and fattn slot; the recurrent state (§2.4.4, r4 I9); the context-lifetime cohorts beni routes here (the oneDNN activation half, `graph_input_stage`, the oneDNN Graph scratch) | context | a reserved slot, claimed by index (§2.3.2, §2.4.3) |
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
| u1bb ring contiguity, u1bb `ggml-sycl.cpp:17625-17626` (`zone_largest_free(KV)`) | the TLSF-wide largest free block, weight-side holes included | deleted: the ring's rows are head slots of the context's fit (§2.4.3, lead ruling 2; rulings §M32 I-1) |
| (r3 I8) u1bb ring compute reserve, `k_pp_moe_ring_compute_reserve_bytes_per_row` = 1 MiB/row (u1bb `:17462-17473`, fed at `:17629`) | an estimate of the compute buffer, by the constant's own comment *"An estimate, not a plan"* | **deleted by zhcn** (lead ruling 3): the compute buffer is zhcn's demand record, a head slot in the same fit. H7p/H7r check that no context-side admission keeps a per-row constant. |
| (r3) u1bb ring budget room, `budget_room_bytes = ggml_sycl_device_vram_budget_room(...)` (u1bb `:18362`, from `32136b16e`) | the plan's VRAM budget room | **deleted for in-arena head slots (r4 I7; lead ruling).** Revision 5 kept it as "a different fact", but it decided the same question the fit decides, "does this head slot fit", from a second source with no demotion lever: a head slot that passed the fit's geometry could fail the budget room where demoting one KV layer satisfies both, and u1bb ran it inside the ring admit, after the yield. Under an arena the budget authority already fixed the arena's size at load, so the geometry is the budget's physical form. The fit is the one source; nothing re-checks after the yield. Every carved head slot and KV extent is still **charged** to `vram_bytes`/`per_device_vram[dev]`, at exactly one site, the commit (§2.4.2 step 6), for accounting; no admission reads that charge for an in-arena slot. No-arena devices keep the check. |
| `-c` hint, `ggml_sycl_largest_fitting_n_ctx_live` | bytes | `kv_region_fit` |
| the demotion WARN's "free for KV" | bytes | the region capacity the fit computed |
| `largest_free_block()` anywhere in a fit decision | approximate (the head of the highest SL list) | never; fits read `frontier_walk` / `gap_below`, which are exact (L1, `9e0a708dc`) |
| (r2) MMID budget demotion: `ggml_sycl_try_demote_runtime_kv` on `BUDGET_EXCEEDED` / `GROWTH_BUDGET_EXCEEDED`, master `:17919-17960`, and its `-c` hint via `ggml_sycl_largest_fitting_n_ctx_live` | byte budget, after the re-fit, so it can demote layers the region already holds slots for | it keeps its constraint, which is a different fact (the device's total VRAM budget, including RUNTIME demand), but it runs **before the carve**. The commit re-fit starts from its residency and can only demote further (§2.4.2 step 3, before the yield). Its `-c` hint reads `kv_region_fit`. |
| (r2) `rebuild_runtime_per_device_vram` (`:17868`) and `moe_mmid_reaccount_replacement` (`:17881`) | byte accounting refusals | arena devices: their KV term is the fit's region bytes (Σ slot sizes), the same number the carve takes. Their refusals roll back through the transaction guard (§2.4.2). |
| (r2) u1bb ring admission: `kv_capacity_bytes = ggml_sycl_kv_capacity_live(...)` and `kv_bytes = ggml_sycl_device_kv_bytes_with_slack(...)` (u1bb `:18356-18357`; the functions at `:16958` and `:17504`); u1bb's re-fit capacity (`:18031`) | live bytes, counting the live ring's KV-zone bytes as free | the ring's rows are the context's demand record, whose slots are head slots of `kv_region_fit` (lead ruling 2; rulings §M32 I-1), installed with the context's slot table (§2.4.2 step 8 (c)). Both u1bb reads are deleted for arena devices, and the re-fit reads `kv_region_fit`. |
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
  on master the transaction returns `busy` if the published plan changed meanwhile
  (`:17911-17915`). That is master's behaviour; under L0 the same check is
  `[CONTEXT-PLAN-BUG]` (§2.4.2; r8 m-11).
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
  - CONTEXT scope: the context's registry entry (§2.5), for the context's life, its MMID device
    pools and its ring rows included (§2.4.2 step 7, §2.7; rulings §M9 I-3, §M32 I-1). 7.14e
    and earlier had a second scope, the device's ring record; it is withdrawn with the device
    ring (rulings §M32 I-1).

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
  context-side room when not under STRICT (§G1). zhcn's rule, "no fallback in scope", is the
  agreed one (and the lead's ruling, 23mk's TRANSIENT cohorts included), so that path and its
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
`allocate_excluding`, ladder or not (r4 m1)**: WEIGHT on a shared TLSF, and any allocation on a
tail TLSF (RUNTIME, SCRATCH or ONEDNN) that carries a load's pending ranges (§2.4.2 (b)). No
transaction records a range on a tail TLSF: a context's ranges are all in its `REGION` headroom,
the shared zone (§2.4.5; rulings §M32 I-2).
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

**The pending-range primitive, its owners and its terms (rulings §Z5 IMP-6, §Z43.3, §M9 I-4,
§M10, §M11 I-A; 23mk rev 4.4 §8.1).** There is **one** pending-range primitive, shared with
23mk, and this design's names are the canonical ones: `pending_ranges(c, d)`,
`allocate_excluding` and `allocate_at`. 23mk retires its `zone_hold`, carve-from-hold and
drop-hold names and specifies four additions as extensions of this primitive, adopted here as
written:
- **A1, the owner and the term.** Every pending range carries `pending_owner{kind, uint64_t
  id}` **and a term tag**, `pending_term` (rulings §M11 I-A): what the range holds room for.
  **This is the one definition of the shared term list (rulings §M11b, §M11b-amend)**, used
  by moua, zhcn, 23mk and 1oxa; no other design defines a term:
  - `WEIGHT`: a load's planned device weights (this design's, (b));
  - `ARENA`: reserved for a compute-arena range; no arena device records one ((b) "The
    compute arena");
  - `SCRATCH`: 23mk's load hold. 23mk records it and `MODEL_TERM` at the **early** stage and
    only checks them at the late stage, never re-recording (a larger late demand is the one late
    refusal both designs print, `[LOAD-PLAN] the late inventory changes the zones admitted at
    the early stage: term %s in zone %s on device %d, early %zu B, late %zu B (refused)`, with
    the term naming the LOAD hold and the zone SCRATCH; rulings §Z13.1, 23mk r8);
  - `MODEL_TERM`: 23mk's model-lifetime term;
  - `DEVICE_TERM`: **reserved, with no producer** (rulings §Z10.1, §M13). 23mk's row 118 is a
    REF under an arena, so 23mk records no DMA staging, and no other design records the term. A4
    step 2 therefore has no current input;
  - `REGION`: **a context's extents and head slots, from either design**: this design's (0) and
    step-5 ranges and zhcn's (0), step-5 and first-publish ranges. They go through one
    transaction and one fit, so they are one fact and one term; a second term would split
    `own_ranges`, and a `REGION`-only fit would then treat zhcn's ranges as allocated;
  - `ONEDNN_PP_A` and `SET_ROWS_STAGE`: 23mk's two **transient** `{CONTEXT, id}` terms (23mk rev
    4.5; rulings §M13), A's grow hold and the row-113 stage slot's grow hold. 23mk's slot guard
    and teardown clear them with `ONEDNN_PP_A | SET_ROWS_STAGE`, never with `PENDING_TERM_ALL`;
  - `ONEDNN_GRAPH_SCRATCH`: 23mk's (rev 4.8, rulings §M26a I-4; the 7.14f queue's 23mk (a)): a
    context's Graph-scratch range. This design's fit places it as a head slot like every C
    term, but step 5 records it under its own term, and step 6 **does not carve it** (rulings
    §V17 I-1: the design that owns a C term says whether it is carved; every head slot of this
    design, MMID's included, is carved at step 6, and this is the one term left pending): 23mk's
    consumer draws inside it many times in one hold, through `allocate_within({CONTEXT, id},
    ONEDNN_GRAPH_SCRATCH, size, align, tag, consume = false)` (A2), and 23mk's teardown clear
    names it. It lies in the shared zone like every range a transaction records (rulings §M32
    I-2; 23mk 4.8a puts it on the RUNTIME TLSF, relayed, §6.22). Its commit line is §M30's
    `[CONTEXT-PLAN] graph scratch range: ctx=%u dev=%d term=ONEDNN_GRAPH_SCRATCH backing=%s
    offset=%zu bytes=%zu`;
  - `FIRST_CONTEXT`: this design's (rulings §M32 C-1, §M38 C-1). The room a load reserves on a
    device for its model's first context's head-slot set (§2.4.2 (b), step 3), recorded under
    `{LOAD, txn}` with the load's `WEIGHT` ranges and retagged to `{MODEL, id}` at the commit
    (A4). **Its handover:** the model's first context transaction on that device (the first to
    reach step 2 while the range exists; transactions are serial under L0) counts it as free
    room in step 2's fit, and step 5, in the same group-mutex section that records the
    transaction's placements, clears it with `clear_pending_locked(tlsf, {MODEL, id},
    FIRST_CONTEXT)`. On any exit that does not publish, the guard's first phase re-records the
    saved ranges in the section that clears the transaction's own (§2.4.2 "The transaction
    guard"), so a refused first context leaves the reservation whole for the next. The
    model's unload clears it if no context took it (A4 names it beside `WEIGHT`);
  - `VM_TAIL_SURPLUS`: 1oxa's (rev 11 `f6d3015`; rulings §V13 I-4, §V14 I-E, §V17 m-10). At the
    mark of a rolled-back load **that leaves a live model** on a VM device, under the tail's
    group mutex, 1oxa records the tail's **surplus
    objects only**, `[round_up(max(highest live block end, largest live or admitted planned tail
    size), object boundary), backed end)`, so a live model's planned but uncarved room stays
    outside the fence. The owner is the **device-scoped `{DEVICE, d}`**, never the dead `{LOAD,
    txn}`, whose rollback has already cleared its own ranges, and the write is
    `record_pending_locked` (below), since the mark holds the group mutex. No dispatch carve
    draws a pending range, which is the fence. It is cleared at exactly four points, each
    under the tail's group mutex: an admit whose plan covers part of it, the short admit that
    takes the surplus (it clears the range and trims the extent), the last model's unload, and
    a rollback that leaves no live model (1oxa rev 13 `47206cb` L59-61; §V17 m-10).
    1oxa rev 11 recorded the whole trailing extent under `{LOAD, txn}`; §V14 I-E replaces that.
    **Four rules (r15 m-3; §V17 m-10 adopts (a) and withdraws (d)):** (a) the record
    **replaces** per `(term, device, TLSF)`, like a `{LOAD, txn}` record (the replace rule
    below), so a second rolled-back load's fence, which 1oxa computes from the same live state,
    supersedes the first and the two never overlap; (b) "the covered prefix becomes that plan's
    room" is a partial trim made of the existing primitives in one group-mutex section:
    `clear_pending_locked(tlsf, {DEVICE, d}, VM_TAIL_SURPLUS)`, then `record_pending_locked` of
    the uncovered remainder under the same owner and term, then the admit's own record over the
    prefix; no primitive trims a range in place; (c) the owner rules below cover `{DEVICE, d}`;
    (d) is **withdrawn (rulings §V17 m-10, 7.14g).** 7.14f recorded a failed **first** load's
    fence under `{DEVICE, d}` directly. A rollback that leaves **no** live model records no
    fence. It releases, that is trims, the surplus instead: with no live model there is no
    context, so nothing can hold a block there, and a `{DEVICE, d}` fence would strand those
    pages until some later load. Rules (a) to (c) stand, and 1oxa re-keys its "reads 2" RED to
    count fenced bytes rather than records (§V17 m-10).

  Two rules hold for every design (rulings §M11b):
  - `PENDING_TERM_ALL` is legal only at a load's commit and rollback, under `{LOAD, txn}`;
  - under a `{CONTEXT, id}` owner, every clear, fit or retag names only its own design's terms.
    An unfiltered or `PENDING_TERM_ALL` filter under `{CONTEXT, id}` is a defect. The same holds
    under `{MODEL, id}`, where the first rule already forbids `PENDING_TERM_ALL`: this design's
    unload clear names `WEIGHT | FIRST_CONTEXT`, and 23mk's names `MODEL_TERM` (A4). **And under
    `{DEVICE, d}` (r15 m-3 (c)),** which two designs share: 1oxa's clears and records name only
    `VM_TAIL_SURPLUS`, and 23mk's only `DEVICE_TERM`; `PENDING_TERM_ALL` there is a defect.

  **Every range a transaction records is `{CONTEXT, id}` (§M12 m-4)**, under `REGION`, the
  ring's rows and the MMID device pool included, except 23mk's Graph scratch, under
  `ONEDNN_GRAPH_SCRATCH` (above). **All of them lie in the shared zone (rulings §M32 I-2):**
  "the context's `REGION` headroom" is the free room of the shared zone's TLSFs, the
  allocator group of `vram_zone_id::KV`, to which `zone_alloc(WEIGHT)` delegates (§1.1),
  every TLSF of that group on a multi-chunk device, and nothing else. No transaction places
  into RUNTIME, ONEDNN or SCRATCH, and none reads their free bytes (§2.4.5). 7.14e placed the
  ring's rows in `REGION` but the MMID pool "on the RUNTIME TLSF", and wrote of a "RUNTIME
  half"; that was three rules for three C terms (r15 I-2) and is withdrawn.

  The owner kinds:
  - `LOAD`, id = `lifecycle::LoadTxnId::value`, minted by the Registry from `next_load_id_`
    (`model-lifecycle.hpp:425` at `3d9414c8c`), monotonic and never reused. It lives from the
    **early** `stage_inventory_plan` to `load_end`'s commit or any rollback (§2.4.2 (b);
    rulings §X7 I-3);
  - `MODEL`, id = `lifecycle::ModelId::value`, from the commit (A4) to the unload: 23mk's
    `MODEL_TERM` and this design's remaining `WEIGHT` room (A4 step 3);
  - `CONTEXT`, id = the backend context's `execution_context_id` (`common.hpp:5667`, set at
    `ggml-sycl.cpp:14841` and `:14873`), from the context's transaction to its destruction. This
    design's transaction ranges ((0) and step 5) are `CONTEXT` ranges with term `REGION`, and
    the commit re-fit's `own_ranges` are this context's ranges **of term `REGION` only**. 23mk's
    two transient holds under the same `{CONTEXT, id}`, `ONEDNN_PP_A` and `SET_ROWS_STAGE`, are
    other terms, so the re-fit treats them as allocated and never carves into them (rulings §Z8
    I-2). Since rev 4.5 23mk has no persistent `{CONTEXT, id}` hold: its row 73 is a `{MODEL,
    id}` hold and its row 113 a carved slot;
  - `DEVICE`, id = the device index: 23mk's `DEVICE_TERM` holds, from the first load (by
    retag) to the last unload, and 1oxa's `VM_TAIL_SURPLUS` ranges, from the mark of a
    rolled-back load that leaves a live model, which records them directly (above; rulings
    §V14 I-E, §V17 m-10).

  The identity travels **in the request**, in a new `alloc_constraints::pending_owner` field,
  never through a thread-local read inside the allocator. Its default, `kind = NONE`, honours
  every range through `allocate_excluding`. The caller that makes a planned draw fills it: the
  loading thread from `Registry::bound_candidate()`, `load_end` from its ticket's
  `token.load` (it can run on another thread, rulings §M9 F2), and a dispatch or transaction
  site from its context's `execution_context_id`. An allocation whose caller fills nothing is
  not anyone's own, whatever thread it runs on. `pending_ranges(c, d)` returns every range with
  its owner and term, and the filtered form `pending_ranges(c, d, owner, term_filter)` (23mk's
  addition, adopted; §M12 m-2) returns only `owner`'s ranges whose term is in `term_filter`.
  A fit's `own_ranges` come from the filtered form, never from filtering the full list by
  hand, so the primitive enforces the filter; this design's are `pending_ranges(c, d,
  {CONTEXT, id}, REGION)`. The fit treats every other range as allocated, a same-owner range
  of another term included.

  **Every operation that selects ranges by owner also takes a term filter (rulings §Z8 I-2,
  §M12 I-4; design-23mk-r6 I-2).** The operations, whose signatures both designs
  state identically:
  - `record_pending(owner, term, offset, size)` (23mk's name, adopted; rulings §M13): records
    one range at `offset`, on the TLSF that holds it, under that TLSF's group mutex. **Replace
    or append is decided by the owner's kind, in the primitive (r14 m-3; rulings §M25):** for a
    `{LOAD, txn}` owner the key is `(txn, term, device, TLSF)`, and a second record for the same
    key replaces the first, so the early stage's recording is idempotent (§2.4.2 (b); rulings
    §M18.4), and so does a `{DEVICE, d}` / `VM_TAIL_SURPLUS` record (r15 m-3 (a)); for every
    other owner a record **appends** a range beside that owner's ranges of
    the same term on the same TLSF, never replacing one. A transaction records one range per
    placement at step 5, and the yield path's retag of an OPTIONAL copy's extent (§2.4.2 step 5)
    lands on a TLSF that may already carry this context's step-5 `REGION` ranges, beside them.
    It is the only write, with its locked variant below; `pending_ranges` only reads;
  - `record_pending_locked(tlsf, owner, term, offset, size)`, the same record on one TLSF for a
    caller that already holds that TLSF's group mutex, the twin of `clear_pending_locked`
    (1oxa's surplus fence at the rollback's mark, rulings §V14 I-E);
  - `retag_pending(owner, term_filter, new_owner)` (A4);
  - `clear_pending(owner, term_filter)` (A2), under each TLSF's group mutex in turn;
  - `clear_pending_locked(tlsf, owner, term_filter)` (A2), the same clear on one TLSF, for a
    caller that already holds that TLSF's group mutex;
  - `pending_ranges(c, d, owner, term_filter)` (above), the ranges a fit may place into;
  - the query `pending_bytes(owner, term_filter)`: the free bytes of `owner`'s ranges whose
    term is in `term_filter`, a scalar for accounting; a fit places by the ranges, not by
    this number;
  - the device-wide query `pending_bytes_excluding(device, zone, except_owner, except_term)`
    (rulings §M13, §Z13.4; 23mk r7 I-A): the free bytes of **every** pending range on `device`
    in `zone` (or in every zone), across all owners, except the ranges whose owner is
    `except_owner` **and** whose term is `except_term` (one term, not a mask). **Its consumer is
    23mk's interim ring before moua L4 (below); from L4 no fit reads it.** 7.12a to 7.14c named
    23mk's A fit (rulings §M14 I-3), which read RUNTIME's free bytes less this query over
    `({CONTEXT, id}, ONEDNN_PP_A)`. Rulings §M27 (1) withdraws that fit: A is a head slot of the
    context's fit, placed by the ranges in the context's `REGION` headroom, and reads no RUNTIME
    scalar (§2.4.2 step 5, and the tenant-only path's (0) alike). 7.12a
    named the RUNTIME ring's replan (`ggml-sycl.cpp:17603`, `:17618`), which on an arena device
    this design replaces with the fit (the ring is a head slot); until moua L4 lands, 23mk's
    interim ring uses the same query. §Z13.4 had it exclude `({CONTEXT, id}, REGION)`, which
    names nothing before L4, since only L4 records `REGION` ranges. **Before L4 the ring owns no
    range (rulings §Z15):** the interim ring passes `except_owner = pending_owner{}` (kind
    `NONE`), which matches no range, so every range counts as taken (23mk 4.7 `cfb99d924`, accepted
    over the §Z13.4 tuple). No caller sums other owners' ranges by hand. Like `pending_bytes`, it
    is accounting; a fit places by the ranges.

  `term_filter` is a `pending_term_mask`; `PENDING_TERM_ALL` names every term, and only a load's
  commit and rollback use it, since a load owns every term it records. Why the filter is needed
  on the clear and the fit too: 23mk clears `{CONTEXT, id}` by RAII on every pre-carve exit (a
  busy ring during the climb is one), and an unfiltered clear there would wipe the same
  context's `REGION` ranges, as an unfiltered clear of this design's would wipe 23mk's transient
  holds; and a re-fit counting every `{CONTEXT, id}` range as its own could carve into 23mk's
  holds. So every clear this design makes of a context's ranges passes `REGION`, and every fit
  counts only `REGION`.
- **A2, drawing inside one's own ranges.** `allocate_within(owner, term, size, align, tag,
  consume)` is a first fit over the free parts of the owner's ranges **of that term**, carved
  with `allocate_at` semantics, so remainders stay whole and coalesced (r5 m-c). The term keeps
  a weight draw out of 23mk's `SCRATCH` hold of the same load, and the reverse:
  - `consume = false` (temporaries) leaves the range record untrimmed, so a freed block returns
    to the TLSF free lists but stays inside the range, excluded from every other allocator and
    reusable by its owner for the range's whole life;
  - `consume = true` (weights: the SYCL<n> weight buffers and the preload's staging; a lazy
    term materializing) trims the carved part from the record: this design's "cleared as
    consumed";
  - a free is the ordinary handle release; a range record is not a TLSF block, so coalescing
    is unaffected;
  - **the keyed form, for every `WEIGHT` draw (rulings §V13 I-5; 1oxa rev 11 relay):**
    `allocate_within(owner, WEIGHT, key, size, align, tag, consume = true)`, where `key =
    {TLSF, offset, size}` is the placement the load's replay recorded for the item (§2.4.2
    (b), "The range bytes"), one tuple wherever a key is named (r15 m-4 (b)); a draw whose size
    differs from its key's is `[ZONE-PLAN-BUG]`. **The items (r15 m-4 (a))** are every unit
    that draws on its own: each SYCL<n> buffer of ggml-alloc's split (the eager draws), and
    each tensor or layout copy that the preload or a lazy materialization draws individually
    (a WOQ copy, a host-extra tensor). The replay places each in plan order and records one key
    per item. It carves exactly `[offset, offset + size)` on that TLSF, inside
    the owner's `WEIGHT` range, with `allocate_at` semantics. There is no first fit at the draw,
    so the order the draws arrive in (the eager buffers, the preload, lazy materializations
    after the commit, in any order) cannot move an item to another TLSF or offset. Only an
    occupied keyed range is `[ZONE-PLAN-BUG]` (23mk's WEIGHT-zone miss row), and the draw never
    falls through to another offset. It serves a USM arena and a VM device alike: the B70's
    two-TLSF USM arena has the shape of 1oxa's bins. The first-fit form stays for `consume =
    false` temporaries and the other terms;
  - a miss inside the owner's own ranges is that owner's plan bug and never falls through to
    `allocate_excluding` outside them: `[KV-PLAN-BUG]` for a context's KV carve (§2.8), 23mk's
    `[ZONE-PLAN-BUG]` for its terms and for a load's weight draw (its WEIGHT-zone miss row);
  - `clear_pending(owner, term_filter)` is idempotent and clears that owner's ranges whose term
    is in `term_filter`, on every TLSF of the device, taking each TLSF's group mutex. A caller
    already inside one TLSF's group-mutex section (step 6's carve, §2.10 step 4) calls
    `clear_pending_locked(tlsf, owner, term_filter)` instead: the device-wide form would
    self-deadlock on the non-recursive mutex, and a hand-rolled per-TLSF clear would lose the
    filter and wipe 23mk's same-context holds on that TLSF (r11 I-4). No site clears pending
    ranges any other way. Ranges of the owner's other terms stay
    (rulings §Z8 I-2). This design's context clears pass `REGION`; a load's pass
    `PENDING_TERM_ALL`.

  The commit's planned carves keep `allocate_at` at the offsets the re-fit chose (fit ==
  carve, above), which is `allocate_within` with the offset fixed, and trim the same way.
- **A3, `replace_within(owner, term, mem_handle && old, offset, size) -> {status, new_block,
  remainder}`**, 23mk's `zone_replace` merge, with 23mk rev 4.5's term and status (rulings
  §M13), as amended by rulings §M16, §M16a and §M16b. The text is shared with 23mk, which
  mirrors it. **Scope: device TLSF ranges only (§M16b).** A host block is refused by
  precondition, with status `HOST_TIER`; host re-draws stay on 1oxa's host-zone path, as a draw
  and a release at the lease's end (relayed to 1oxa). The CpuExpertPool's `weight_lease` moves
  into the worker lambda (`cpu-dispatch.hpp:112-127`), so it is released with the work, with no
  event and no store entry. **No overlap (§M16 item 4):** the new layout is never sourced from
  `old`'s own bytes; a path that builds it from the old device copy stages from the host or from
  planned scratch, and an always-compiled witness checks that the source range does not overlap
  `old`'s block. In one section (the group mutex, then the arena authority's registration, then
  the retained-handle store's mutex; the §L6 order):
  1. **classify `old`'s other references when `old.owner_use_count() > 1` (§M16a).** A3 scans
     the retained-handle store (`mem-handle.cpp:48-64`) for `old`'s control identity:
     - **E, event-retained**: in the deferred-drain queue (`retain_handles_until_event`) or the
       drain worker's `in_hand` record (below). A3 does its bookkeeping now, returns those
       records' events as `after`, and marks `old`'s control **superseded**. The caller's
       staging of the new copy carries `sycl::depends_on(after)`, so same-queue and cross-queue
       consumers follow one rule and nothing waits on the host;
     - **G, recording-held**: in `graph_unwaitable`, in a registered recording sink or in a
       `releasing` snapshot (below). A3 returns `GRAPH_HELD` with `old` unconsumed; the old
       layout keeps serving, and the caller retires the recording (the drop path of §Z14.1, uwlx
       §R) and retries. zhcn §Z12's refresh does not apply, since a re-layout changes the kernel
       the recording baked;
     - **any other reference**: the residual, `use_count - 1 - found`, clamped at 0 (below). A
       non-zero residual while a `terminal_retention_ticket` is still publishing (`publishers >
       0`) returns `BUSY`, which is retryable. With `publishers == 0` it returns `SHARED`, a
       named refusal: a device `WEIGHT` block has no CPU user, so the reference is a leak. `old`
       is unconsumed in both.

     **Step 1 is one callable (rulings §V15; 1oxa rev 12 m-13):** `classify_references(const
     mem_handle &) -> {status, after}`, whose status is E (with `after`), `GRAPH_HELD`, `BUSY`
     or `SHARED`, by the rules above. It reads references, never the TLSF, so it applies to a
     slice's handle unchanged. `replace_within` calls it inside its section, and 1oxa's
     OPTIONAL re-stage calls it alone on a slice's handle, when it takes the store mutex
     itself, never under a group mutex it did not take first (the §L6 order). It classifies and
     returns; marking `old`'s control superseded stays `replace_within`'s bookkeeping, on an E
     result. A source check finds no second classification of the store's references;
  2. retire `old`'s registration and control, mark its block free in TLSF metadata, carve the
     new range with `allocate_at` inside `old`'s block plus the owner's ranges of `term`, and
     trim the consumed range parts. A superseded control stays alive through its E references
     and is released later by the ordinary destructor, which frees nothing for it (P2);
  3. **the rest of `old`'s block (§M16 items 1, 2).** An empty rest returns a null remainder
     handle and carves nothing. A `WEIGHT` re-draw uses **remainder mode**: in the same section
     both fragments, before and after the new block, are marked free and re-recorded as `{owner,
     WEIGHT}` pending, untrimmed, so every other allocator excludes them and only the same owner
     draws them again; no remainder handle is returned. Another term's rest is one block, owned
     by the returned remainder handle.

  A failure at a carve restores `old`'s block and registration and returns a failure status,
  with `old` unconsumed. No other allocator sees `old`'s block free. The callers are the re-draw
  paths of rulings §V11.2 (1oxa's replace and reclaim paths, which hand the old handle to A3
  instead of drawing first) and 23mk's A overlap form (term `ONEDNN_PP_A`, only at count == 1);
  moua adds none.

  **The store's windows and lock order (§M16b).** The store mutex is file-local
  (`mem-handle.cpp:53-64`) and is taken after the group mutex; no site takes the two the other
  way. The census is `cat ggml/src/ggml-sycl/mem-handle.cpp | grep -n
  "state.mutex\|g_retained_handles_state"`: ten sites (`:121`, `:131`, `:154`, `:172`, `:1899`,
  `:2032`, `:2048`, `:2056`, `:2065`, `:2092`), none of which destroys a handle under the lock.
  **Hazard:** `drain_retained_handles(true)` (`:2048`) is never called while holding a group
  mutex. Two windows let a reference leave every container before its handle is destroyed, and
  each keeps an identity snapshot until the destroy, which is the general rule for any store
  container that destroys handles outside the lock:
  - the drain worker (`:126-177`). (1) Under the lock: pop the record, set `in_hand = {ids,
    event}` (the control identities copied from `record.handles`) and `++active`. (2) Unlock and
    `record.event.wait_and_throw()`; the graph catch moves the handles to `graph_unwaitable` and
    clears `in_hand` in the same lock section. (3) Outside the lock, `record.handles.clear()`,
    the destroy; `in_hand` still names the ids, so A3 classifies them E with a completed event.
    (4) Under the lock: `in_hand.clear()`, `--active`, notify;
  - `release_graph_retained_handles` (`:2061-2068`) and a recording sink's detach: a `releasing`
    identity snapshot, taken under the lock before the out-of-lock destroy and cleared under the
    lock after it. A3 treats it as G. The recording sinks (`:108-117`, `:2134`) are thread-local
    today and insert without the lock; they are registered under the store mutex on attach,
    detach and insert, and A3 scans them as G.

  A3 reads only identity lists and events, under the lock. **The clamp is a documented
  limitation (§M16b).** The clamp can hide a leaked reference: if k snapshotted references are
  destroyed between their release and A3's read while L references have leaked, residual = L - k
  can reach 0 and A3 proceeds as E. This happens only inside that release window and only when a
  leaked-reference defect already exists. A3 is not a leak detector and does not claim to be
  one. The defect's detectors are unchanged: the ownerless-lease report, the witness balance
  (created == freed + superseded_released, live == 0 at teardown), and the zero-SHARED census
  arm. An exact count was rejected: a consistent cut of `use_count` and the snapshots needs
  either a destroy under the store lock, which is forbidden and inverts the lock order, or a
  decrement before the destroy, which brings back a false `SHARED`. **Witness counters**
  (`GGML_SYCL_PRIVATE_TESTING`): superseded controls created, freed and `superseded_released`,
  balanced at teardown as above. The ownerless report classifies entries, and a superseded entry
  keeps its model attribution.
- **A4, `size_t retag_pending(pending_owner owner, pending_term_mask term_filter, pending_owner
  new_owner)`** under the group mutex (rulings §M11 I-A). It moves **only** `owner`'s ranges
  whose term is in `term_filter` to `new_owner`, and returns how many it moved. 7.10 took 23mk's
  `retag_pending(from, to)`, which moved every range of `from`, 23mk's `SCRATCH` hold included:
  the clear that followed found nothing, and the hold stayed allocated for B's whole life (r10
  I-A, P4). **At `load_end`'s commit, once it is irrevocable** (after `finalize_end` returns
  committed and not `cleanup_required`, under `load_end`'s L0 hold; rulings §M12 I-1, §2.4.2 (b)
  "Lifetime"), in this order:
  1. 23mk's `retag_pending({LOAD, txn}, MODEL_TERM, {MODEL, id})`;
  2. `retag_pending({LOAD, txn}, DEVICE_TERM, {DEVICE, dev})`, which has **no current input**:
     `DEVICE_TERM` is reserved and nothing records it (rulings §Z10.1, §M13), so the step moves
     nothing and returns 0. It stays in the order so that a producer, if one is ever ruled in,
     has its retag;
  3. `retag_pending({LOAD, txn}, WEIGHT, {MODEL, id})` (1oxa r8 I-D; rulings §X9 I-D, §M13): B's
     `WEIGHT` room still undrawn at the commit passes to the model, which is the room of the
     host-extra tensors materialized lazily after the commit plus the rounding slack. A lazy
     materialization draws it with `allocate_within({MODEL, id}, WEIGHT, ..., consume = true)`,
     its owner filled from the owning model's `ModelId`. 7.12 dropped that room at the clear, so
     the undrawn `WEIGHT` never reached the model, and every lazy materialization missed and was
     refused. The same step moves the load's `FIRST_CONTEXT` ranges, `retag_pending({LOAD,
     txn}, FIRST_CONTEXT, {MODEL, id})` (rulings §M32 C-1), so the model's first context finds
     them under its model (A1);
  4. `clear_pending({LOAD, txn}, PENDING_TERM_ALL)`, which drops everything the load still
     holds: 23mk's `SCRATCH` hold. After steps 1-3 only that term remains under `{LOAD, txn}`,
     so the filter could name it; `PENDING_TERM_ALL` also covers a term a later design adds.

  A rollback runs only step 4, through `ggml_sycl_load_pending_rollback_noexcept(txn)` (§2.4.2
  (b) "Lifetime"), which every non-commit exit reaches; since the retags run only after an
  irrevocable commit, no rollback ever follows a retag. So after B's commit nothing of B's is
  `{LOAD, txn}`, and a later load C records and draws its own `SCRATCH` hold against a TLSF that
  holds only B's `{MODEL, B}` ranges (its `MODEL_TERM` hold and its remaining `WEIGHT` room).
  **At B's unload the `{MODEL, B}` ranges go:** `ggml_sycl_teardown_owner_effects`
  (`ggml-sycl.cpp:12585`, called from `ggml_backend_sycl_model_unloaded_token` at `:12313`)
  calls `clear_pending({MODEL, owner.model}, WEIGHT | FIRST_CONTEXT)` beside 23mk's `MODEL_TERM`
  clear, each naming only its own term (A1's rules), so no range outlives its model. **The clear
  is teardown's first statement (r12 m-12)**, ahead of the publication block, the MoE discovery
  release whose failure returns `false` (`:12615-12622`),
  `ggml_sycl_release_model_slot_resources` (`:12656`) and 1oxa's chunk release, since no range
  may outlive its chunk (the rollback's T13 rule). It is idempotent, so a teardown that returns
  `false` and is retried by the reaper runs it again harmlessly. **A planned weight is never
  evicted, so no free re-records (rulings §M18.3 as amended by §M18.3a, which withdraw r12 m-11
  and the free-site re-record of §V11.2).** A planned device weight or PRIMARY copy of a live
  model is never an eviction or yield candidate and is freed only at the unload. A planned
  OPTIONAL copy (a duplicate layout whose primary is resident on the same device) is never an
  eviction candidate; it is yieldable only to a context's KV admission, through the yield path
  (§2.4.2 step 5), which hands its extent to that context's `REGION` room before it releases the
  handle, so bytes freed while that range stands land inside it for that context's KV draw. A
  last reference that drops only after step 6's carve has cleared the range (a frees-stayed-
  queued pick, or an E reference the retained-handle store holds past the barrier, §M16a)
  returns its bytes to the general TLSF: the KV that needed them demotes at step 6's re-fit, and
  the key stays yielded (r14 m-4). The destructor stays the sole release point with no reason
  logic (P2): it never learns why it frees and never writes a range, and the unload frees with
  no re-record. A re-draw of a planned `(tensor, layout)` still goes through A3's
  `replace_within` (§V11.2's draw half), which reuses its own extent; a yielded copy is never
  re-drawn. The lead relays this to 1oxa. The exact signatures go to impl-23mk, so both designs
  name one primitive (§6.13).

Whichever design lands first implements them in the one primitive (L4 here); 23mk's §4 depends
on them. H1 adds `allocate_within` (both `consume` forms, reuse after a free inside an
unconsumed range, a miss reported and not fallen through, and a `WEIGHT` draw that never lands
in a `SCRATCH` range of the same owner), `clear_pending` idempotence, and `replace_within`'s
arms below, each with `check_invariants()` after. These arms carry the term filter (rulings §Z8
I-2, §M11b, §M12, §M13):
- **retag, then clear.** Load B records a `WEIGHT` range it draws partly, a second `WEIGHT`
  range it never draws (the **undrawn** range), a `SCRATCH` range and a `MODEL_TERM` range.
  After the commit's four steps the `SCRATCH` range is gone (the TLSF's free bytes include it);
  the undrawn range, the drawn range's remainder and the `MODEL_TERM` range survive whole as
  `{MODEL, B}` (rulings §X9 I-D); and nothing is left under `{LOAD, B}`. A lazy draw with owner
  `{MODEL, B}` then lands inside the undrawn range, and B's unload clear, `clear_pending({MODEL,
  B}, WEIGHT)`, drops the `WEIGHT` ranges and leaves the `MODEL_TERM` range whole for 23mk's
  clear. Two witnesses make the arm able to fail: 7.10's move-every-range retag, under which the
  `SCRATCH` range moves to `{MODEL, B}` and survives the clear; and 7.12's commit without the
  `WEIGHT` retag, under which the undrawn range is dropped and the lazy draw misses with
  `[ZONE-PLAN-BUG]`;
- **a filtered context clear.** `{CONTEXT, c}` holds a `REGION` range and a 23mk hold of
  another term; `clear_pending({CONTEXT, c}, REGION)` drops the first and leaves the second
  whole. Witness: an unfiltered clear, which drops both;
- **the fit places only in its terms, by geometry (rulings §M12 m-2).** The arm is built on
  placement, not on the scalar `pending_bytes`: on one TLSF, 23mk's hold under `{CONTEXT, c}`
  sits at the lower offset and this context's `REGION` range above it, and the `REGION` range
  is smaller than the fit's demand. The fit's `own_ranges` are `pending_ranges(c, d, {CONTEXT,
  c}, REGION)`, so it places what fits inside the `REGION` range and demotes the rest, and
  nothing lands in the hold. Witness: a fit whose `own_ranges` are every range of the owner;
  under first fit it places the demand's head inside 23mk's hold, which precedes the `REGION`
  range, and the arm reports the overlap. Without that geometry (a `REGION` range as large as
  the demand, or the hold placed after it) the witness would also place nothing in the hold,
  and the arm would pass vacuously;
- **the carve's locked clear keeps other terms (rulings §M12 I-4).** Under one TLSF's group
  mutex, a commit carve runs `clear_pending_locked(tlsf, {CONTEXT, c}, REGION)` on a TLSF
  that also carries 23mk's hold under `{CONTEXT, c}`; afterwards the hold is whole, and no
  lock is taken twice. Witnesses: an unfiltered per-TLSF clear (the hold is gone), and the
  device-wide `clear_pending` called inside the section (the model detects the self-deadlock
  with a try-lock and reports it);
- **idempotent early recording (§M12 m-9).** `record_pending({LOAD, B}, WEIGHT, ...)` twice for
  the same device replaces the range; the TLSF holds it once. Witness: an appending record,
  which holds it twice;
- **a context owner's records append (r14 m-3; rulings §M25).** On one TLSF, a step-5
  placement records `{CONTEXT, c}` / `REGION` at one offset, and the yield path's retag records
  `{CONTEXT, c}` / `REGION` over an OPTIONAL copy's extent at another. Both ranges survive, and
  `pending_ranges(c, d, {CONTEXT, c}, REGION)` returns both. Witness: the load's key-replace
  applied to context owners, under which the retag drops the step-5 KV range and the arm
  reports the missing range;
- **the device-wide query excludes one `(owner, term)` (rulings §M13; 23mk r7 I-A).** On one
  device, `{CONTEXT, c}` / `REGION`, `{CONTEXT, c}` / `ONEDNN_PP_A`, `{LOAD, B}` / `SCRATCH` and
  `{MODEL, M}` / `MODEL_TERM` each hold a range in RUNTIME, and `{LOAD, B}` / `WEIGHT` holds one
  in another zone. `pending_bytes_excluding(dev, RUNTIME, {CONTEXT, c}, ONEDNN_PP_A)` returns
  the free bytes of the other three RUNTIME ranges, and the every-zone form adds the `WEIGHT`
  range. Witnesses: an exclusion by owner alone, which also drops `{CONTEXT, c}` / `REGION`, and
  a sum of the caller's own owner's `pending_bytes`, which misses the other owners; each returns
  a smaller sum;
- **A is a head slot, and the MMID bytes have one source (rulings §M14 I-3, §M27 (1)).** One
  context: A plus an MMID pool of X bytes, both head slots of the fit, with `REGION` headroom
  for exactly both, on a shape where step 4 sized RUNTIME to exactly the load's terms. The fit
  places both, and the context is admitted. REDs: 7.14c's interim A fit, which reads RUNTIME's
  free bytes less `pending_bytes_excluding(dev, RUNTIME, {CONTEXT, id}, ONEDNN_PP_A)`, finds
  no room and refuses a context whose plan fits; and the double count, A's placement also
  subtracting `mmid_runtime_pending_bytes` = X, refuses the same context;
- **`replace_within` (rulings §M16, §M16a, §M16b).** Each case, with its RED:
  - a new range never lands in the owner's ranges of another term, and a failed carve restores
    `old` whole;
  - an empty rest returns a null remainder handle and carves nothing. RED: a zero-size remainder
    block;
  - remainder mode: a `WEIGHT` re-draw inside a larger `old` leaves both fragments recorded
    `{owner, WEIGHT}`, untrimmed; another owner's allocation of their size misses them, and the
    owner's `allocate_within` lands in one. RED: the fragments returned to the free lists, where
    the other owner's allocation takes one;
  - E: `old` also held by a deferred-drain record with an incomplete event, and, separately, by
    the drain worker's `in_hand` record between its pop and its destroy (a
    `GGML_SYCL_PRIVATE_TESTING` handshake parks the worker there). A3 succeeds, returns `after`
    naming the event and marks the control superseded; the witness balance holds after the
    drain. RED: no `after` returned (a probe records the staging submitted without the
    dependency); and, without `in_hand`, the parked case returns `SHARED`;
  - G: `old` in `graph_unwaitable`, in a registered sink, or in a `releasing` snapshot parked
    between its unlock and its destroy: `GRAPH_HELD`, `old` unconsumed. RED: without the
    snapshot, the parked case returns `SHARED`;
  - other: an extra plain copy of `old` returns `SHARED` with `publishers == 0` and `BUSY` while
    a ticket publishes. RED: `SHARED` returned while a ticket publishes;
  - a host block returns `HOST_TIER`; a source range overlapping `old`'s block fires the overlap
    witness by its message;
  - a census arm, lead-run with the GPU arms: zero `SHARED` over a decode. The cross-queue
    consumer-ordering RED (the staging without `depends_on(after)` read stale) is lead-only;

#### 2.3.4 Reset, settle, and the dead KV reclaim

- **zone_settle / TLSF reset (r1 M2).** Any path that `reset()`s a shared TLSF clears that
  TLSF's `context_side` in the same critical section. Retained runs are free space, so
  dropping them is correct. The settle's existing precondition (no live registered
  allocations) already excludes live regions and reserved slots.
  - Reserved slots are registered allocations held by their owners (§2.3.2), so the
    precondition already sees every one of them, vacant or claimed, including the ring's rows.
    Revision 5's unregistered vacant slots needed a special rule; held handles do not.
  - **L4 adds two refusals (r4 m10):** a settle or reset is refused while any pending range
    exists on the TLSF (a transaction is between its step 5 and its commit or rollback), since
    the ranges would otherwise survive a reset that invalidates their offsets. The ring's
    rows, CONTEXT slots, and each model's `moe_onednn` weight slot, a load draw, are covered by
    the registered-allocation rule above.
  - **The arena rebuild is refused the same way (rulings §M12 C-1; §Z8 I-3).**
    `ensure_planned_arena_zones`' rebuild (`arena_destroy()`, then `arena_reserve()`;
    `unified-cache.cpp:4553-4625`) counts any pending range on the device as live and refuses by
    name, as it already does for zone bytes, chunk leases and live scratch. Its failure is the
    stage's named refusal, never the `GGML_ABORT` at `ggml-sycl.cpp:16215` (§2.4.2 (b) "The
    zones come from the demands, before the pack"). So the refused operations are: a settle, a
    TLSF reset, `arena_destroy()` and the arena rebuild.
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

The B70 exercises this path (r13 I-E): its arena is two chunks with a per-chunk cap of
29472 MB, so its 29033.3 MB `WEIGHT` zone spans two TLSFs, ~28192 and ~841 MB, and a load's
`WEIGHT` ranges are keyed per TLSF (§2.4.2 (b), rulings §M18.4). The B50 at PCT≤100 is
single-chunk (§1.1).

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
  (§2.3.2, `size ≤ cap[index]`), so the shrink of a context's slot, its ring rows included, is
  never a carve and never demotes anything. A slot the new plan does not reuse (a growth) stays
  allocated in the fit's view,
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
  (§2.3.1). At the commit re-fit, this call's own ranges (term `REGION`) are passed separately
  as `own_ranges`,
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
- `forced_host`: layers an earlier fit of the same transaction already demoted (step 6's
  re-fit inherits step 2's residency). Revision 7.6 also fed it step 3's MMID budget demotion,
  which an arena device no longer has (§2.4.2 step 3; rulings §M8 I-5(c)). The fit may demote
  further but never promotes them;
- **the head slots (r3 C1, I5; lead ruling 2; r4 I4, I9).** The indexed slots of every demand
  record this transaction (re)plans: the context's CONTEXT-scope records (zhcn's tenants,
  beni's and jzvq's cohorts, the recurrent state), **the context's ring rows, one slot per
  (context, device) at its own `n_ubatch` (§2.7; rulings §M32 I-1)**, and its oneDNN
  scratchpad buffer per queue. 7.14e and earlier sized a device's ring record as the max over
  the live contributors; that is withdrawn with the device ring. Each head slot names its
  cohort and index; its room is the context's `REGION` headroom, the shared zone's TLSFs and
  no other (§2.3.3; rulings §M32 I-2). Reuse in place follows the `reservations` rule above.
  **The first context's room (rulings §M32 C-1):** on a device where the context's model still
  holds its `FIRST_CONTEXT` ranges (§2.3.3 A1), the fit counts them as free room for this
  context, since the load reserved them for it, and step 5 hands them over. The MMID device pool
  is a CONTEXT-scope head slot, placed only where the route is reachable and this context's pool
  on that device is missing or smaller than the candidate's workspace (§2.4.2 step 7; rulings
  §M9 I-2, I-3). Every C term of §2.4.5 is a head slot the same way: 23mk's `onednn_pp_a`
  (rulings §M27 (1)), `set_rows_stage` and `onednn_graph_scratch`, and this design's
  `moe_control` slot (rulings §M27 (2a)) and MMID device pool (rulings §M25 I-4);
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
  - `ring-growth` when the layer would have stayed on the device with this context's
    superseded ring slots counted free (r5 I-B; lead ruling). The old and new rows coexist
    until the publish, so
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
  naming the tenant. Otherwise KV demotes until both fit. **A head slot never takes tier 4
  (rulings §M18.3a, §V13 I-8; 1oxa rev 11 relay):** an OPTIONAL copy yields only to KV
  admission, so a head slot, the C-term slots among them, is placed in tiers 1-3 alone, and
  no yield is made for it. The tenants-alone condition is therefore that the head slots do
  not fit in tiers 1-3 with every KV layer on the host. Each slot goes into the cheapest tier
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
    a head slot that does not fit in tiers 1-3 with every KV layer on the host refuses the
    transaction; no OPTIONAL copy is yielded for it).

  Worked prediction (zhcn GA, B50 GPT-OSS `-c 65536 -ub 1024`; r5 m-g: one number, from one
  function, cited by both designs; corrected by design-zhcn-r4, §6.8; re-derived in 7.14f under
  rulings §M32 I-1 and I-2). **The fit reads actual allocations**, because its geometry is the
  live TLSF (lead ruling). The breakdown, from `kv_layer_bytes_for_kind` at `n_ubatch` = 1024
  (f16 K and V, 8 KV heads × head dim 64, so 2 KiB per cell per layer, K and V together):
  - room for KV and head slots, after the actual weights (11510.9 MiB): **2204.6 MiB**. That is
    7.14e's 1827.1 MiB, measured with RUNTIME at its 512 MiB floor, plus the 377.5 MiB the
    floor held idle: GPT-OSS 20B's RUNTIME D terms are `moe_onednn`, 32 × 4406528 =
    141008896 B (134.5 MiB), and `moe_ptr_table`'s k × 256 B (under 0.1 MiB), and RUNTIME has
    no floor on an arena device (§2.4.2 (b), the end states). The model's `FIRST_CONTEXT`
    reservation is part of this room, since GA's context is the model's first and the fit
    counts the reservation as its own;
  - full-attention KV: 12 layers × 65536 cells × 2 KiB = 12 × 128.0 = 1536.0 MiB;
  - SWA KV: 12 layers × 1280 cells × 2 KiB = 12 × 2.5 = 30.0 MiB, where 1280 =
    PAD(n_swa + n_ubatch, 256) = PAD(128 + 1024, 256);
  - room after full KV: 2204.6 − 1566.0 = 638.6 MiB;
  - head slots: **1348.0 MiB**, zhcn's 808.0 compute plus the ring's rows at `-ub 1024` and
    depth 1, both in `REGION` (rulings §M28 (1), §M32 I-1): activation 1024 · 32 · 2880 · 2 =
    188743680 B (180.0 MiB) and output 1024 · 32 · 2880 · 4 = 377487360 B (360.0 MiB), with
    `local(t)` = 32, since the B50 holds every expert of GPT-OSS 20B;
  - `free_after_full_kv` = 638.6 − 1348.0 = **−709.4 MiB**.

  Withdrawn: −726.9 (7.14e and the revisions before it back to design-zhcn-r4's correction, and
  zhcn 5.12's G2/GA), which counted only the
  ring's 180.0 activation slot as a head slot, with the 360.0 output slot in RUNTIME, while
  7.14e's own rule put both rows in `REGION` (r15 I-2: under that rule and the idle floor the
  figure is −1086.9 and demotes 9 layers); −714.9 (revisions 7 and 7.1, and zhcn rev 4.1),
  which took the SWA term at `-ub 512`'s PAD(128 + 512, 256) = 768 cells (18.0 MiB); and
  revision 6's −728.4, which no breakdown reproduces. Full-attention slots are 128 MiB; 5 × 128
  = 640 is short by 69.4 MiB and 6 × 128 = 768 covers the deficit with 58.6 MiB to spare
  (⌈709.4 / 128⌉ = 6), so the fit demotes **exactly the 6 highest-indexed full-attention layers
  and no SWA layer**. **The context's other head slots (r16 m-3), on GA's shape:**
  `moe_control`, 4 × 1024 × 8 + 256 = 33024 B; 23mk's A, 1024 × 4096 × 2 = 8388608 B
  (`max_K` 4096, `attn_output`); `set_rows_stage`, 0 on one device; the Graph scratch, 0,
  since the oneDNN SDPA route rejects GPT-OSS's sinks (`fattn-onednn.cpp:115-116`); the oneDNN
  scratchpad, 0 (rulings §M31a); the MMID pool, 0, its route unreachable in an ordinary build;
  the recurrent state, 0, since GPT-OSS has no recurrent layer; jzvq's MXFP4 MoE TG caches,
  223560 B on master's `-c 4096` baseline (3240 + 184320 + 23040 + 12960 B, llama.cpp-jzvq
  comment c-jv0r), which jzvq's function re-sizes at GA's shape; and jzvq's fattn workspaces,
  from jzvq's function, not yet in hand. The known ones sum to 8645192 B (8.2 MiB), so the count
  stays 6 while jzvq's fattn workspaces, plus any growth of its TG caches at this shape, stay
  under the remaining 50.4 MiB of the spare. RUNTIME's transitional terms are 0 on this shape
  (rulings §M38 C-2), so the room is unchanged. Head-slot alignment rounding adds
  under 512 B per slot (sizes are 512 B multiples), far inside both margins (rulings §GA).
  **These figures are hand arithmetic, and the score does not use them (r6 I-7):** GA's
  pre-registered numbers are the output of H2's run of `kv_region_fit` on this geometry, with
  every head slot, recorded before the lead's run. It demotes 7 only if the other C slots
  exceed the spare, or if weight-side holes smaller than one 128 MiB slot fragment the free
  room, in which case the §2.9 sub-slot WARN names them. **The 1348.0 assumes a ring depth of
  1 and no record-mode per-op index sets (r6 m-6):** if L4 finds the PP MoE oneDNN path
  reached while recording, that is `[ZONE-PLAN-BUG]` and the path claims no slot (rulings §M37
  Q6, §2.7), so the depth stays 1; GA is re-scored with the index-set bytes G1 prints. zhcn's
  G2 and GA carry 270.0 and 540.0 for the rows (at `-ub 512` and `-ub 1024`) and −709.4
  (relayed, §6.22). **zhcn 5.14's branches (§6.23):** −709.4 is neither zhcn's (R), 7.14e's
  −1086.9, nor its (F), 5.13's −726.9. It is (R) plus the 377.5 MiB that RUNTIME's floor held
  idle, which is how every one of zhcn's B50 GPT-OSS 20B rows moves under 7.14f. **After the
  ONEDNN floor goes (rulings §M37 Q3)** the room grows by a further 244842496 B (233.5 MiB;
  GPT-OSS 20B's W is 23592960 B, like 120B's) to 2438.1 MiB, and the ring-plus-compute
  deficit shrinks to −475.9 MiB. The same state moves the Graph scratch into `REGION` as a
  head slot, so the layer count there is H2's run at that state, pre-registered for that
  commit, not hand arithmetic.
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
  RAII token type, `ggml_sycl_replan_token`, backed by a thread-local held state that one
  accessor reads, **`ggml_sycl_replan_token_held()`** (rulings §Z43.4, §M77, §M9a). It is the
  only way anything outside the token asks whether this thread holds L0: the pool phase gates,
  the `:33769` witness check, 23mk's §Z42.4 witness and 1oxa's `vm_create_map_ror`
  assertion all call exactly that name, and no second flag exists:
  - an acquire on a thread that already holds L0 is a **nested hold** and does not lock;
  - an acquire on another thread blocks;
  - the outermost token unlocks.

  **The token's kind (rulings §M9a).** Every acquire names a kind, and the held state records
  the **outermost** token's kind beside its depth; that is one state, not a second flag:
  - `TRANSACTION`: a context's own planned transaction. The wrapper
    (`ggml_backend_sycl_set_runtime_context_for_model`), the constructor's first publish
    included, and llama's `ggml_backend_sycl_replan_scope` on the growth path;
  - `LOAD`: the load entries, `load_begin`, `stage_inventory_plan` and `load_end` (the weight
    preload runs inside `load_end`);
  - `LIFECYCLE`: every other entry of the list below: the probe, the FA recheck, activate,
    unload and the quarantine reaper, `can_unload`, shutdown, reactivation and the teardown
    release proc. None of them grows a pool for planned work. §M9a names the two kinds that
    decide the gate; the third only keeps the other holders from being labelled as either.

  The accessor is `bool ggml_sycl_replan_token_held(ggml_sycl_replan_kind kind =
  GGML_SYCL_REPLAN_KIND_ANY)`: true when this thread holds L0 and, unless `kind` is `ANY`, the
  outermost token has that kind. The pool phase gates ask for `TRANSACTION`; the preload's
  `:33769` assertion asks for `LOAD`; 23mk's witness and 1oxa's assertion ask for `ANY`. A
  nested acquire never changes the outermost kind. A nested `TRANSACTION` under a `LOAD` or
  `LIFECYCLE` token, or a nested `LOAD` under a `TRANSACTION`, fails a witness check (below):
  loads never nest inside a transaction and a transaction never triggers a load (below), so
  either is a code defect. A nested `LIFECYCLE` under any kind is legal (the reaper under
  `load_begin`, a backend free under the stage), except the teardown release proc's, which is
  outermost-only (§2.4.2 "The deadlock rule").

  **Witness checks are always compiled (rulings §Z9 I-3, relayed to this design).** The tests
  build Release with `-DNDEBUG`, so an `assert` or a debug-only check compiles out, and a test
  that relies on it passes on the mutant it exists to catch. So no witness in this design is a
  debug assertion. Each is `GGML_SYCL_WITNESS(cond, message)`, compiled in every build and
  independent of `NDEBUG`. It is evaluated in a `GGML_SYCL_PRIVATE_TESTING` build, where it is
  on unless the runtime switch `GGML_SYCL_WITNESS_CHECKS=0` turns it off for an arm that must
  reach the unchecked path, and in any other build when `GGML_SYCL_WITNESS_CHECKS=1` is set. A
  failure aborts with its own message, so a death arm scores by message, never by "it aborted".
  There is no separate `-UNDEBUG` build. **A disabled witness costs one load and one branch (r12
  m-10, r13 m-g):** the switch is a namespace-scope `const bool`, initialized once when the
  library loads, and the macro tests it before it evaluates `cond`, so a disabled check costs
  one plain load and one predictable branch, with no `getenv` and no `thread_local` read. 7.13
  used a function-local static, which is not one branch: every call first checks the static's
  guard variable, an acquire load and a branch, and the first call takes the `__cxa_guard` path.
  A test that sets `GGML_SYCL_WITNESS_CHECKS` sets it in the child's environment before the
  library loads, which every arm does. Graph compute's check, which runs on every decode, costs
  that load and branch in a production build. The checks and their messages:
  - an illegal nesting (above): `[REPLAN-TOKEN] illegal nesting: <inner> under <outer>`;
  - the release proc entered holding L0 (§2.4.2 "The deadlock rule"): `[REPLAN-TOKEN] release
    proc entered with L0 held`;
  - the preload without a `LOAD` token (`:33769`): `[REPLAN-TOKEN] preload without a LOAD
    token`;
  - the two checks of the shared invariant (§2.4.2 "The scope closes"): `[REPLAN-TOKEN]
    TRANSACTION token held at alloc_buffer entry` and `[REPLAN-TOKEN] TRANSACTION token held
    at alloc_buffer host fallback`;
  - graph compute under any token: `[REPLAN-TOKEN] token held in graph compute`;
  - the ring bind finding a retained owner (§2.7): `[RING-RETAIN] bind found a retained
    owner`.
  zhcn's witnesses use the same macro and switch.

  **Where it is defined (r9 addendum m-14).** `pinned-pool.cpp` sits below `ggml-sycl.cpp` in
  the layering and must not depend on a symbol defined there, and 23mk's witness calls the same
  name, so the definition site is shared across designs. The enum, the accessor and the token
  type are declared in `ggml/src/ggml-sycl/unified-cache.hpp`, beside `offload_stats_phase()`
  (`:6275` at `3d9414c8c`), whose declaration the gates already reach: `pinned-pool.cpp`
  includes `common.hpp` (`:10`), which includes `unified-cache.hpp` (`common.hpp:32`), and calls
  `offload_stats_phase()` in both gates (`:631`, `:875`). They are defined in
  `unified-cache.cpp` beside that function's definition (`:2564`), together with
  `g_replan_txn_mutex` and the `thread_local` depth and outermost kind. The token's constructor
  and destructor are the only writers of that state; there is no setter and no hook, so nothing
  can mark a thread as holding L0 without locking it. `ggml-sycl.cpp` constructs tokens and
  never touches the state directly. moua defines it (§M9a).

  There is one mutex, so there is no device order. zhcn's gate 31 function-local token is this
  same type.
- **Where it is taken (rulings §L0R, §M8 I-1): at the top of every public entry point that can
  publish the plan or prepare a live update,** before the module admission guard and before any
  ticket. The census is by reachability, at master `3d9414c8c`: every exported function (a
  `GGML_BACKEND_API` declaration or a registry proc address) from which a call of
  `ggml_sycl_publish_plan_locked`, `ggml_sycl_publish_prepared_plan_locked`,
  `lifecycle_replace_placement_plan`, `Registry::prepare_live_update` or
  `Registry::acquire_live_update` (which wraps `prepare_live_update`, `model-lifecycle.cpp:788`)
  is reachable. 7.6 listed the entries it knew of; the reachability census found three more
  exported publishers and one exported test hook (r8 I-1). The entries that remain:
  - `ggml_backend_sycl_set_runtime_context_for_model` (`:18840`), the wrapper around the
    transaction: its live-update ticket (`:18867`) and its publish (`:18955`) are under L0, and
    so is the inner transaction body (`:17852-18648`);
  - `ggml_backend_sycl_set_runtime_context` (`:18701`; `ggml-sycl.h:423`) **stops being exported
    (rulings §M9 I-1).** It is a one-line pass-through to
    `ggml_sycl_run_runtime_context_transaction` (`:18708`) with no model identity, so as an
    entry it could only hand the body the global snapshot, and an embedder calling it for A's
    backend while B published last would re-plan A against B's plan. Its callers in the whole
    tree are the wrapper (`:18964`) and the deleted `set_runtime_n_ctx` (`:19099`); every other
    hit is a comment or a source gate that quotes its text (the census method is §6.11's). It is folded into the
    wrapper: the wrapper calls the transaction body directly with the per-model snapshot it
    selected (`lifecycle_select_placement_plan(model)`, published at `:18955`) as the body's
    `current` parameter, and the four `thread_local` side channels it set for the inner call
    (`g_runtime_expected_model`, `_set`, `g_runtime_external_lease`,
    `g_runtime_update_succeeded`, `:2648-2651`) become the body's parameters and its return.
    There is no global-snapshot entry left. The gates that anchor on its text
    (`test-sycl-nonfa-attn-scratch-guard-source.py:116-121`, `:163`, `:280`, `:334`, `:439`,
    `:466-473`, `:519`; `test-sycl-compute-buffer-fallback-source.py:126`, `:142`, `:532-535`,
    `:587-602`; `test-sycl-ubatch-ring-replan-source.py:144`) are re-anchored on the wrapper in
    L6, and the header comment at `ggml-sycl.h:405-420` is rewritten. **The probe's side
    channel goes with them (rulings §M11 m-1).** The probe wrapper arms the same two
    `thread_local`s before it calls the body (`probe_expected_model_guard`,
    `:18794-18805`: `g_runtime_expected_model = model; g_runtime_expected_model_set = true`,
    cleared by the guard's destructor). With the side channels turned into parameters, the
    probe passes its `model` as the body's expected-model parameter, and the guard and both
    assignments are deleted;
  - the probe wrapper, `ggml_backend_sycl_probe_runtime_context_for_model` (`:18728`), and the
    FA recheck, `ggml_backend_sycl_recheck_runtime_context_flash_attn` (`:19024`);
  - `ggml_backend_sycl_activate_model_plan` (`:15228`, publish at `:15259`);
  - model unload, `ggml_backend_sycl_model_unloaded_token` (`:12283`), with its failure and
    exception republishes (`:12308`, `:12329` → `ggml_sycl_publish_restored_plan` `:12547`) and
    its latest-live republish (`:12599`). `ggml_backend_sycl_model_quarantine_token`
    (`:12337`) only enqueues the token and defers the quarantine; it publishes nothing and
    takes no token (r9 m-5). The reaper, `ggml_sycl_quarantine_reap` (`:12365`), retries
    `unloaded_token`, whose own token makes each retry an L0 hold. It runs from three places:
    `load_begin` (`:12739`), shutdown's drain, and **every `ggml_backend_sycl_free`**
    (`ggml_sycl_quarantine_drain_shutdown`, `:84108` → `:12386`), including the per-device
    temporary backends `stage_inventory_plan` frees through its `backend_guard`. Under
    `load_begin`'s or the stage's hold those are nested holds; a free on a thread holding no
    token takes L0 in `unloaded_token`. None of them runs under an L1-L5 lock;
  - the model-load entries (below);
  - **module shutdown and reactivation (rulings §M76.5).** `ggml_backend_sycl_shutdown`
    (`:109666`; `ggml-sycl.h:71`) drains the quarantine, reaps it and publishes the restored or
    torn-down plan, so it is an L0 holder like unload; module reactivation takes L0 too. So no
    publishing entry can overlap either, and a publishing entry that finds the module not ACTIVE
    while it holds L0 is a caller lifecycle violation: `[CONTEXT-PLAN-BUG]`, with no retry and
    no `BUSY` (below). **`ggml_backend_sycl_can_unload` (`:109358`) holds L0, taken by a
    TRY-lock (zhcn 5.5 row 32, r6 m-6; rulings §M11 I-F).** It is what closes module admission:
    it moves `ACTIVE` to `RETRY_CLOSED`, reserves the Registry's shutdown, and waits up to 5 s
    for the module's in-flight mutations to drain (`wait_for`, `:109390-109392`), reopening on a
    timeout. It is distinct from shutdown (`:109666`). Without L0, a publishing entry that
    already holds L0 could see admission close under it, and the `[CONTEXT-PLAN-BUG]` below
    would fire on a legal `can_unload`. **It never blocks on L0:** it takes a `LIFECYCLE` token
    with `try_lock`, and when another thread holds L0 it returns `false` at once, which
    `ggml_backend_unload_checked` reports as `BUSY` (`ggml-backend-reg.cpp:745-749`); a nested
    acquire on a thread that already holds L0 succeeds as usual. 7.10 took L0 with a blocking
    acquire, which hangs `test-sycl-lifecycle-runtime-wrapper.cpp:1476-1492` (r10 I-F): the test
    holds a live-update lease with no L0 (`test_hold_live_update`), starts the async
    `unloaded_token`, which takes L0 and waits for that lease, then calls
    `ggml_backend_unload_checked` on the lease-holding thread, which reaches `can_unload`; a
    blocking acquire there waits for the unload's L0, the unload waits for the lease, and
    `test_release_live_update`, which comes after the call returns, is never reached. With the
    try-lock the call returns `BUSY`, the test releases the lease, and the unload completes, so
    the test is unchanged. Once it holds L0, its drain wait cannot deadlock: every publishing
    entry takes L0 before the module guard that counts it as a mutation, so the mutations it
    waits for are ones that never need L0. `ggml_backend_sycl_cancel_unload` only reopens
    admission, which cannot turn an entry's check into a false bug, and takes no token. The
    token type gains that one `try_lock` form, used only here. H9 runs the would-be hang (§3.1).
    Shutdown joins the TP worker, the CPU worker, the prestage thread and the watchdog while it
    holds L0 (r9 m-5); none of them takes L0, since their work is graph compute and dispatch,
    which never take it, and H7ai walks each thread's entry function as a root to prove it;
  - **not** the exported test hook `ggml_backend_sycl_test_hold_live_update` (`:12772`, which
    calls `acquire_live_update` at `:12779`; proc address `:110029`). 7.7 put a token at its top
    "so that a test holding a lease models a real L0 holder". It does not (r9 m-4): the hook
    stores the guard in the static `g_test_live_update_guard` and returns holding it, so an
    RAII token at its top is released at that return and L0 is not held across the lease.
    Holding L0 across the return would need a second static and would change
    `test-sycl-lifecycle-runtime-wrapper.cpp:1476-1492`: there the async `unloaded_token` stays
    pending until `test_release_live_update` and then returns OK, and
    `ggml_backend_unload_checked` on the holding thread returns `BUSY`; with L0 held, the
    release hook would also have to release L0, and that unload would become a nested hold. So
    the hook keeps master's form, a lease with no L0, the test is unchanged, and it joins H7ai's
    classified test-only class. The H9 arms that need a parked L0 holder park inside a real
    entry through a test park point instead;
  - the tenant-only path's llama-held scope, `ggml_backend_sycl_replan_scope`, which llama opens
    before (s) on the growth path and closes before ALLOC's `graph_reserve` inside
    `sched_reserve_impl` (§2.4.2 "The coverage query"; rulings §M9a). The probe and (ii) that
    run inside it are nested holds;
  - the teardown release proc, and every other ring mutator outside graph compute.
- **Orphaned publishers are deleted (rulings §M8 I-1; docs/plans/2026-04-22 A5).** A caller
  census over the whole tree at `3d9414c8c` (the command is in §6.11) finds no caller for three
  exported publishers:
  - `ggml_backend_sycl_set_runtime_n_ctx` (`ggml-sycl.h:596`, body `:19086-19100`), which calls
    `ggml_backend_sycl_set_runtime_context`. Plan A5 already lists it as an orphan;
  - `ggml_backend_sycl_set_model_loading` (`:13273`; `ggml-sycl.h:1430`), the deprecated bool
    load boundary, which calls `load_begin`, `load_enter_nested` and `load_end`. llama uses the
    explicit hooks, and the lifecycle source contract already asserts that llama does not call
    it (`tests/test-sycl-lifecycle-source-contract.py:445`);
  - `ggml_backend_sycl_set_tensor_inventory` (`:16745`; `ggml-sycl.h:349`), the pre-091afc
    direct setter, which publishes through `ggml_sycl_set_tensor_inventory_impl` (`:16491`)
    with no load transaction. `tests/test-sycl-tiered-verdict-contract.py:190-192` already
    forbids tests to call it.

  All three are deleted in L6, declaration and body. The gates that name them change in the
  same commit: `test-sycl-nonfa-attn-scratch-guard-source.py` uses the `set_runtime_n_ctx`
  definition as the end anchor of the recheck's body (`:294`, `:344`, `:481`) and is
  re-anchored, and `test-sycl-host-zone-config-source-contract.py`'s `setter_fn` window
  (`:111-113`) is removed. The lead approved the deletion on the whole-tree census (§6.11).

  Two more come off the publisher list (r9 m-2, m-3):
  - `ggml_backend_sycl_compute_placement_plan_early` (`:16473`; exported, `ggml-sycl.h:357`)
    runs populate and the plan computation and **publishes nothing**; 7.7 called it a publisher
    and an L0 entry, which was wrong. **Its public `void` entry stays, with its one declaration
    at `ggml-sycl.h:357` (23mk's form; rulings §M14 m-13):** the body moves into a static `bool`
    impl, which the stage (`:16824`) calls and whose `false` it maps to its refusal, and the
    public entry logs and returns. 7.9 made the entry itself `static`, which conflicts with its
    `GGML_BACKEND_API` declaration;
  - `ggml_sycl_reset_model_load_scratch_state` publishes `nullptr` (`:14773`) when
    `!preserve_placement_authority`. The forward declaration defaults that parameter to false
    (`:12404`), and every caller (`:12422`, `:12721`) passes true, so the branch is dead at
    runtime, but a static reachability walk reaches a publish from `load_enter_nested`
    (`:12875`), which takes no token, and a future no-argument call would clear the publication
    with no token. The branch and the default argument are deleted.
- **The load's span (m-9; rulings §M76.2, §M8 I-3; r8 m-6).** A load takes L0 at the top of
  each public backend load entry, not across the whole load:
  - `ggml_backend_sycl_model_load_begin` (`:12728`);
  - `ggml_backend_sycl_stage_inventory_plan` (`:16775`), the inventory entry llama calls
    (`llama-model.cpp`). Its early form (`early` true; the apply at `llama-model.cpp:657`,
    inside `llama_model_sycl_compute_early_plan`, whose call moves from `:2106` to after `:2211`
    under rulings §M18.2, before `create_tensor` picks each tensor's buffer type from the plan;
    r13 m-a) reaches `ggml_backend_sycl_compute_placement_plan_early` (`:16473`; the stage calls
    its static impl, above) at `:16824`, which computes and publishes nothing; it is **the stage
    that admits B's room and records the load pending ranges** ((b) below; rulings §X7 I-3). Its
    late form, llama's `llama_model_sycl_set_late_inventory` (`llama-model.cpp:2408-2410`,
    reaching `:709`), reaches `ggml_sycl_set_tensor_inventory_impl` (`:16491`, static) at
    `:16826`, consumes the admitted plan by its identity, and provisions the host zones before
    the loader allocates weight buffers;
  - `ggml_backend_sycl_model_load_end` (`:13096`, publish at `:13205`). The weight preload,
    `ggml_sycl_preload_model_weights` (`:33515`, static), runs inside it (`:13124` → `:12476`,
    and `:13267` → `:12933`), so its republish (`:33769`) is under `load_end`'s hold; it is not
    a public entry.

  `ggml_backend_sycl_model_load_enter_nested` (`:12875`) publishes nothing and prepares no live
  update, so it takes no token. llama's loader code and the application's `progress_callback`
  therefore never run under L0, and the deadlock rule below needs no clause for callbacks into
  application code. Loads never nest inside a transaction, and a transaction never triggers a
  load. **Concurrent loads stay refused, not serialized:** `load_begin` returns `LOAD_BUSY`
  while another load is active or a model is `DRAINING_UPDATES` (`model-lifecycle.cpp:181-188`),
  and llama throws on it (`llama-model.cpp:161-162`). That is master's load-admission rule, a
  refusal of a second concurrent load by name, and this design does not change it. The same
  holds for a **closed module** (r9 m-6): `load_begin` answers `LOAD_BUSY` when its module
  guard fails (`:12729-12732`), which is the answer the Registry's own shutdown gate gives
  (`shutdown_reserved_ || shutdown_completed_`, `model-lifecycle.cpp:181-183`). It is not the
  `BUSY` that §M76.5 removes: that one was a retryable answer from a publishing entry holding
  L0, which a caller could loop on. `LOAD_BUSY` is a named load refusal that llama throws on
  and never retries, so it stays.
- **What an interleaved transaction observes between two load entries (rulings §M76.2, §M8
  I-3).** Between B's entries, a transaction for model A can run (a new context in router mode,
  or a growth republish). On master it would observe three things of B's half-finished load, and
  each is closed here:
  - **(a) Process-global load state.** `load_begin` → `ggml_sycl_model_loading_effects(true,
    true)` (`:12749`) → `ggml_sycl_reset_model_load_scratch_state` (`:12422`) clears
    `g_tensor_inventory_detail`, the `g_moe_*` totals, `g_placement_kv_info`,
    `g_model_n_layer` and `g_moe_expert_vram_reserve` (`:14749-14773`), and
    `populate_inventory_globals` (`:15891`) rewrites them with B's values at B's stage. A's
    transaction reads `g_tensor_inventory_detail` for its MMID re-plan and its KV demotion
    (`:18245`, `:18265`), so between B's `load_begin` and B's stage it would plan A's MMID
    workspaces from an empty inventory, and after B's stage from B's tensors. **Fixed by
    §M76a (below): the load state is per model.** B's entries write B's bound candidate's
    inventory record, and A's transaction reads A's own through its plan snapshot. Nothing of
    B's is process-global. `g_sycl_in_model_load` (`:10687`), set at `load_begin` (`:12414`)
    and cleared at `load_end` (`:12449`, and `:12753` on `load_begin`'s unwind), is process-wide
    today: while it is set, every SYCL_Host buffer takes role `WEIGHT`
    (`:42629`, `:42644-42645`), and the device caps report `async` false (`:107796`). So A's
    output buffer and LoRA tensors, allocated outside claim scopes (§2.4.3), would be
    classified as weights. **It becomes state of the load, keyed by the load's identity, not of
    a thread (rulings §M9 F2).** 7.7 made it `thread_local`, which is right only if
    `load_begin` and `load_end` run on one thread, and the candidate API lets `load_end` run on
    another: `load_end` re-binds the candidate on its own thread
    (`ggml_sycl_load_candidate_end_scope`, `:13110`, with `binding_required`). With a
    `thread_local` flag, T1's flag would stay set after T2's `load_end`, so T1's later SYCL_Host
    buffers would stay `WEIGHT` and its caps `async` false, while T2's preload ran with `async`
    true, the layout hazard the caps comment warns about. So every reader asks **whether the
    calling thread is bound to an active load**: its bound candidate
    (`ggml_sycl_bound_load_candidate`, `:2681`), which `load_begin` binds and `load_end`'s scope
    re-binds on whichever thread runs it, names a load the Registry still holds as active. A
    thread whose candidate names an ended load reads false. **The answer's window matches
    master's (rulings §M11 m-6).** The Registry's load record carries an `in_load` bit, set
    where master sets the flag (`load_begin` → `:12414`) and cleared where master clears it:
    `load_end`'s exit effects (`:12449`), before the compute arena and the preload, and
    `load_begin`'s unwind (`:12753`). The predicate is "bound to an active load whose `in_load`
    bit is set". So during the compute-arena reserve and the preload, on `load_end`'s thread,
    it answers **not in load**, as master's flag does. Neither reader below is reached on that
    path: the preload allocates through `unified_alloc` directly (`:34169-34177`), not through
    the SYCL_Host buft, and nothing there reads the caps; the window is pinned so the answer
    is defined, not because anything observes it. **`load_enter_nested`'s write (rulings §M11
    m-7).** `ggml_backend_sycl_model_load_enter_nested` (`:12875`) calls
    `ggml_sycl_model_loading_effects(true, false)` (`:12894`), which stores the flag true
    (`:12414`) and the phase LOAD (`:12415`). Under the binding model the flag store is a
    no-op and is deleted: the nested entry binds the same load's candidate (`:12886`), whose
    `in_load` bit `load_begin` already set, and a nested exit clears nothing (`outer` false).
    The phase store stays; it is process-global and llama.cpp-dhpw's. The readers:
    - the SYCL_Host buft's role (`:42629`, `:42644-42645`): `WEIGHT` only for an allocation on a
      thread bound to an active load (or under evictable weights, as today);
    - the device caps' `async` (`:107796`), read by llama's loader on the loading thread
      (`llama-model-loader.cpp:1947`), which is bound, so it keeps its answer;
    - the same caps, read at context construction by `llama-context.cpp:997` for
      pipeline-parallel eligibility, on the context's thread, which is bound to no load. It
      reads `async` true whatever another model is loading; on master it read false while any
      model loaded.

    `offload_stats_set_phase(LOAD)` and `(UNKNOWN)`, set in the same function (`:12415`,
    `:12450`, `:12754`), stay process-global. They feed dispatch heuristics (`:81230`, `:86377`,
    `:86391`) and the pool phase gates; making the phase per context is **llama.cpp-dhpw**
    (§M77), not this design;
  - **(b) Capacity.** B's plan is computed at B's stage entry, and B's weights are staged only
    in `load_end`'s preload. An A transaction in between could carve its regions and head slots
    into room B's plan counted on, and B's preload would then miss and spill: admit-then-spill
    against the plan B publishes at `:13205` (P4). **B's plan holds its weight room as pending
    ranges (§M8 I-3(b); rulings §M9 I-4, §X7 I-3, §M11 I-A to I-C),** by the mechanism of
    (0)'s pending ranges and 23mk's fit-and-hold (rulings §Z42.1), in the **one**
    pending-range primitive (rulings §Z5 IMP-6; §2.3.3). All of it is on arena devices; a
    non-arena device has no zone TLSF and no pending range, and keeps master's placement
    (§2.11):
    - **The create set: the early stage plans exactly what `create_tensor` creates (rulings
      §M18.1, §M18.2; r13 C-1, I-B).** 7.13's early inventory was `ml.weights_map`
      (`llama-model.cpp:638-640`), every weight in the file once. `create_tensor` does not
      create that set. A `TENSOR_DUPLICATED` request makes a new tensor whenever its buffer
      type's context does not already hold the name (`llama-model-loader.cpp:1741-1747`). The
      tied output head is `token_embd` requested again as `LLM_TENSOR_OUTPUT` (`:1270-1272`): it
      takes the output layer's list while `token_embd` stays in the input's. A per-layer
      `rope_freqs` (or `rope_long`, `rope_short`) requested with `TENSOR_DUPLICATED` for `i !=
      0` is one tensor per device its layers span. Of the 145 `TENSOR_DUPLICATED` uses in
      `src/models/*.cpp`, 88 are tied embeddings and most of the rest are those rope factors.
      The late inventory walks `ctx_map` (`llama-model.cpp:680-688`), so it sees the duplicates;
      the early one did not. gemma4 on the B70 shows it (r13 C-1; merge gates `g4-np1-b0`,
      `g4-np4-b0`, `server-g4-default-parallel-b0`): the early plan saw "720 weights, 7.62 GiB"
      and PLACE-4 7798.3 MB, the late stage "Collected 721 tensors, 8.28 GiB" and 8478.3 MB,
      NORM/EMBED 215 to 216 on the device, +680 MB. B's `WEIGHT` range was ~680 MB short, the
      tied head's buffer draw missed with `[ZONE-PLAN-BUG]`, and all three gates failed to load.
      So:
      1. **The early call moves after the buffer-type lists (§M18.2: the reorder, not a copied
         predicate).** `llama_model_sycl_compute_early_plan` is called at
         `llama-model.cpp:2106`, before `cpu_buft_list` (`:2140`) and `dev_layer` / `dev_output`
         (`:2186-2211`) exist. It moves to just after `:2211`, before the create block
         (`load_arch_tensors`, `:2230`). The loading guard stays at `:2104`.
      2. **The early inventory comes from a record pass of the create block itself.** The create
         block (`load_arch_tensors` and the generic scale and input-scale passes after it,
         `:2229-2385`) moves unchanged into one member,
         `llama_model_base::create_weight_tensors(ml)`, which the real load calls. The early
         stage calls it once more, on a probe of the same architecture
         (`llama_model_create(arch, params)`, `:1016`) with `hparams` and the buffer-type lists
         copied from the model, while the loader is in **record mode**. In record mode
         `llama_model_loader::create_tensor` (`:1220`) runs its checks as today (a missing
         required tensor throws the same way), resolves the request's site with
         `resolve_create_site` (item 3), appends `{name, flags, site}` to the loader's create
         record and returns the loader's own meta tensor. It takes no context, creates nothing,
         registers nothing and leaves `n_created` and `size_data` alone. The probe is destroyed
         after the pass, and the meta pointers it stored go with it. Upstream's
         `llama_params_fit` makes the same kind of pass with `no_alloc` (the fork disables it
         under SYCL); a record pass builds no context at all. The record is in `create_tensor`'s
         order, which is each context's tensor order.
      3. **One function resolves the site, and `create_tensor` calls it (§M18.2's "same
         function, not a copy").** The prefix of `buft_for_tensor`
         (`llama-model-loader.cpp:1258-1394`) moves into the member
         `llama_model_loader::resolve_create_site(tn, flags, t_meta)`. That prefix remaps a
         duplicated `TOKEN_EMBD` to `OUTPUT`, looks up the tensor info, applies the unused and
         `TENSOR_SKIP` rule, picks the layer's list and matches the `-ot` overrides. The lambda
         calls the member and keeps only its plan-consulting suffix (ir18's host layout, from
         `:1396`), and the record pass calls the same member. A source-contract gate checks that
         the override match (`std::regex_search` over `tensor_buft_overrides`) appears exactly
         once in `llama-model-loader.cpp`, inside that member.
      4. **User-forced placement is a fixed input to the pack (§M18.2; P3).** The site carries
         `forced`: `OFF_SYCL` when an override names a non-SYCL buffer type or the layer's list
         does not start on a SYCL device (`--cpu-moe` and `-ncmoe` are internal `-ot` overrides;
         a partial `-ngl` gives a layer the CPU list, which starts on SYCL_Host unless
         `--no-host`, `make_cpu_buft_list` at `:1682`); `DEVICE`, with the device, when an
         override names a SYCL<n> type; `NONE` otherwise. It crosses the dlopen boundary in
         `ggml_sycl_tensor_info` (`ggml-sycl.h:220`), in the four bytes of padding after `type`:
         `uint8_t site`, `uint8_t forced`, `uint8_t forced_device`, `uint8_t reserved`. The
         struct's size and array stride do not change, and all-zero means "unknown site, not
         forced", which is what an older producer sends. The pack places the forced tensors
         first, `OFF_SYCL` on the host and `DEVICE` on its device, and never moves one. A
         forced-off tensor contributes 0 to every placement-dependent term (below), so
         `--no-host --cpu-moe` charges no `moe_onednn` slot, and no ~1.6 GB of RUNTIME is held
         for CPU-executed experts (r13 I-B (1)).
      5. **The create set.** The pack places each record entry. A `TENSOR_DUPLICATED` name has
         one entry per distinct site, and the pack charges a name once per target device. That
         is `create_tensor`'s rule (one tensor per name per buffer-type context), because a
         device's weights take its one SYCL<n> buffer type. The placed entries are the **create
         set**, and it is the early inventory: every entry the pack puts on a device is
         admitted, and nothing else is. gemma4's tied head is then an entry at site `OUTPUT` on
         the B70, charged its 680 MB at the early stage. The planner's name index gains the site
         (`planned_target_device(name, site)`, the ir18 lookup), since `token_embd.weight` names
         two entries.

      **There is no admit-by-name for a late-only tensor (rulings §M18.1).** The late stage
      checks the late inventory against the create set (below). A SYCL-backed tensor with no
      create-set entry at its placement is refused by name, as a witness that the two walks
      diverged: `[LOAD-PLAN] tensor %s (site %s) exists in buffer type %s on device %d but has
      no admitted entry (refused)`. It fires before any buffer is allocated, so the tied head's
      draw can no longer miss;
    - **The zones come from the demands, in five steps (rulings §M14 C-1, §M17, §M17a, §M18; r13
      I-A, I-F).** The planner packs weights into the **live** zone capacities:
      `compute_placement_plan` reads `zone_capacity` of SCRATCH, ONEDNN and RUNTIME and packs
      into `unified_cache_kv_weight_capacity` ("Planned weights must fit the actual allocator
      zone", `unified-cache.cpp:28085-28096`), and the single-device caller does the same
      (`ggml-sycl.cpp:16336-16357`, "Plan budget tied to arena" at `:16349`). So a zone must
      hold its demand before the pack reads it. 7.12's "plan first, then ensure" packed against
      the pre-ensure zones (r12 C-1): on GPT-OSS 120B on the B50 the early pack would read a
      13338 MB weight zone, the ensure would then grow RUNTIME from 512 to 1617.9 MB and leave a
      12232.1 MB weight zone, and the early placement would refuse a load master completes
      (`merge-gates/gptoss120b-b1.log:57`, `:184`, `:191`, `:261`; `glkg-qwen35b-a3b-b1.log:73`,
      `:210`, `:217`, `:291` show the same growth, 512 to 1672 MB). 7.13 fixed the order but
      computed every demand before the pack, as if none depended on placement. Four do, and a
      fifth has a placement-dependent device set (r13 I-F), and an upper bound over the eligible
      weights is not the demand (rulings §M17; P4). Master charges the large term, the
      `moe_onednn` slots, for the whole inventory before the plan (`populate_inventory_globals`
      calls `unified_cache_set_planned_pp_moe_onednn_scratch` at `ggml-sycl.cpp:15967`, ahead of
      `compute_and_store_plan_for_inventory` in `compute_placement_plan_early`, `:16484-16488`),
      and records `moe_control` and the non-FA shape inside the plan
      (`populate_host_zone_sizing`, `unified-cache.cpp:27082`, `:27430`, `:27617`), which is why
      its late stage rebuilds again (`gptoss120b-b1.log:1544-1545`, 1617.9 to 1618.0 MB). Every
      term is a value of one closed enum, `ggml_sycl_zone_term`, shared with 23mk (§2.4.5).
      **The order, in `compute_and_store_plan_for_inventory` (one edit to one function, shared
      with 23mk):**
      1. **compute the placement-independent terms** (class P, §2.4.5): 23mk's
         `mmq_work_counter` (SCRATCH), whose bytes are fixed, not a function of where a weight
         or a layer goes. `ring` is not here from 7.14e: its bytes are the ring's activation and
         output slots, which scale with the context's `n_ubatch`, so it is class C (rulings §M28
         (1); `moe_onednn` in step 3, below, is the ring's weight slot). **The compute arena is
         not a term (rulings §M25 I-1):** `reserve_compute_arena` points it at the whole SCRATCH
         zone (`compute_arena_size_` = the zone's capacity, `unified-cache.cpp:20722-20728` at
         `3d9414c8c`), so it is SCRATCH's floor, capacity rather than demand (step 4's rule).
         7.14 listed it as a SCRATCH P term, which read additively would have grown SCRATCH past
         the pinned 536870912 B (r14 I-1). 7.14 also computed `nonfa_shape` and
         `onednn_graph_scratch` here, as placement-independent terms with a placement-dependent
         device set. Neither is class P
         (r13 I-F (5); rulings §Z15, §M19): `nonfa_shape` is charged by the pack in step 3, per
         device that hosts attention layers, and `onednn_graph_scratch` is a context-lifetime
         (C) term of 23mk's. **C terms leave the load stage entirely (rulings §M21.3):** the
         graph-scratch term is charged at the context transaction, from that context's `REGION`
         headroom like KV. Its value function is 23mk's, which calls 1oxa's §V11.3 D512 tile
         screen as a helper; 1oxa charges nothing for it (rulings §M26 I-4). The load stage
         evaluates it as 0 (§2.4.5, "The C rule"), and from the Graph-scratch commit on the
         load-stage ONEDNN ensure reads the no-floor W getter,
         `unified_cache_get_planned_onednn_pp_w_bytes` (below; rulings §M26a I-4, §M33 I-G);
      2. **ensure them**, through step 4's sized ensure over the P terms and the floors (never
         through a `planned_*` getter; rulings §M25 I-2), on every device `dev_layer` gives a
         layer, which the reordered early call (above) makes an input, at the early stage only,
         before the stage's device loop packs anything. No iteration reads a secondary device's
         zones before they are ensured (r13 m-d);
      3. **pack once**, the forced tensors first (above), and **charge each placement-dependent
         term per device as the pack places each weight**, against the capacity that remains:
         - `nonfa_shape` (SCRATCH), once per device at the first attention layer's weight the
           pack places on it. Its bytes are the context shape per attention class
           (`planner_n_head_ctx_max`, `planner_n_head_swa_max`, `planner_n_swa`,
           `planner_n_head_all_max`, `n_ubatch`, `n_ctx`; `unified-cache.cpp:27615-27618`), and
           a device that hosts no attention layer is charged 0. Master publishes the shape for
           `plan.device_id` alone (`:27616`), which under-reserves a secondary device that runs
           attention (r13 I-F (5)); under `--no-host` with the experts on the host, the device
           still runs attention and is still charged;
         - `moe_onednn` (RUNTIME), once per device at the first expert that device executes
           through oneDNN PP (master charges it for the whole inventory, unconditionally,
           `ggml-sycl.cpp:15961-15967`). **It is the ring's weight slot alone (rulings §M28
           (1)):** the ring's activation and output slots are the C term `ring`, one slot per
           context, placed at that context's transaction at its own `n_ubatch` (§2.7), because
           the PP MoE dispatch does not chunk a micro-batch (the end states, below); the pack
           charges the first context's share of them to the device's reservation (two bullets
           down). **Its value counts the experts resident on the device (r14 m-7; rulings §M17,
           §M32 C-1 (b)):** the batched executor refuses a batch whose active experts are not
           all on the device (`ggml-sycl.cpp:78554-78557`), and the WOQ arm sizes its weight
           bytes as the per-expert slot times the op's active count (`:78703-78709`), so the
           slot never holds more than the op's tensor's resident experts. The value is the
           maximum, over the expert tensors the pack places on the device and that device
           executes through oneDNN PP, of that tensor's per-expert slot size stepped by the
           sizing code (`src/llama-model.cpp:441-447`) times `local(t)`, the number of that
           tensor's experts the pack has placed on the device: a running maximum the pack raises
           as it places each expert. Master multiplies the inventory-wide per-expert maximum
           (`:430-452`) by `n_expert` (`:465-467`), which is never less. Static layer-first
           packing places whole layers (`unified-cache.cpp:24300-24320`), so on both merge-gate
           models the densest tensor has `local(t)` = `n_expert` and the two agree; H7ap asserts
           that, with a non-uniform fixture and a partial layer as the RED. **It has its own
           load-time setter (rulings §M32 I-3), and its store is keyed by (model, device)
           (rulings §M38 I-2):**
           `unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes(model, dev, bytes)` writes
           the entry for that model on that device, and the getter and the executor read the
           dispatching context's model's entry. The source-contract gate maps the setter to
           `MOE_ONEDNN`. It is not one of the device-global ring setters, which no arena path
           reaches (H7z (aj), §2.7), and it is the term's one load-time publication; the
           `unified_cache_set_planned_pp_moe_onednn_scratch` call at `:15961-15967` is deleted
           on arena devices. 7.14f and 7.14g gave the setter a `(dev, bytes)` signature over a
           per-device store, so a second model's load overwrote the first model's entry, the
           last writer winning, for a fact that is per (model, device) (r16 I-2). The slot
           itself is one per (model, device), planned and drawn inside that model's own load
           before `finalize_end` under `{MODEL, id}`, and released at the model's unload (§2.7).
           **Where a later load's D terms go (rulings §M38 I-2):** the load that lays out the
           arena on a device sizes RUNTIME to its own D terms (step 4). A later load on that
           device cannot grow RUNTIME, because a rebuild that meets live bytes refuses (below).
           So its pack charges each of its D terms against RUNTIME's free room where the term
           fits there, and otherwise against the shared zone's capacity, beside its `WEIGHT`
           ranges, and records the term's range under `{LOAD, txn}` either way, retagged
           `{MODEL, id}` at the commit (§2.3.3 A4) and drawn inside the load. Another model's
           entries and ranges are untouched. So RUNTIME plus the later loads' shared-zone D
           ranges hold exactly the sum, over the live models, of their D terms, and H9's
           second-load arm scores it with last-writer-wins as its RED (§3.1);
         - **the ring's rows for the model's first context (rulings §M32 C-1 (a), (b)).** As the
           pack admits an expert of tensor t to a device that executes t through oneDNN PP,
           `local(t)` rises by one. The ring's activation slot is align256(align64(n₀) · max
           over t of (min(`local(t)`, n₀ · `n_expert_used`) · K_t) · 2) and its output slot the
           same with N_t · 4, where n₀ = 512 is the auto-ubatch ladder's bottom rung (§2.7's
           value function at n₀). Whenever an admission raises either maximum, the pack charges
           the increase to the device's reservation (below) together with the expert's weight
           bytes, and admits the expert only if both fit the capacity that remains. The increase
           per expert is a constant the pack already knows, so this is one pass with no fixed
           point. On the merge gates the totals are **1132462080 B** on GPT-OSS 120B (377487360
           + 754974720) and **1610612736 B** on Qwen3.5-35B-A3B (536870912 + 1073741824),
           reached with the first whole layer each device packs;
         - **the rest of the model's first context's head slots (rulings §M32 C-1 (a), §M38
           C-1).** At the first placement that makes a device need them, the pack charges the
           device's reservation **the head-slot set of the model's first context on that device,
           from the one function the context transaction's fit uses**:
           `context_demand_records(model, dev, shape)`, the producer set that every context
           transaction's step 2 reconciles (§2.4.3), evaluated at the load envelope's shape,
           which is its `n_ctx`, n₀ for `n_ubatch`, and the flash-attention rule, KV types and
           `n_seq_max` of `llama_context_default_params()` (rulings §Z20 Q1). The reservation is
           the byte sum of that set's slots on the device, less the ring rows, which the set
           holds at the same `local(t)` and which the pack charges as it admits experts (above).
           **The load names no term.** 7.14f and 7.14g kept a hand list here (`mmid_workspace`,
           `moe_control`, `onednn_pp_a`, `set_rows_stage`, `onednn_graph_scratch`,
           `onednn_scratchpad` and zhcn's compute slot) that omitted the recurrent state and
           jzvq's fattn workspaces and TG caches, so the Qwen gate's first context was about
           190.8 MiB short and refused (r16 C-1). A term that a producer adds now reaches the
           reservation with no edit here, and H7ap's first-context arm carries a mutation RED
           for a second enumeration. **Its `n_ctx` is the caller's (rulings §M38 C-1):**
           llama.cpp-fkpg (a) makes the envelope carry the caller's intended context, and the
           planners size from it. Both merge gates run `-c 4096`, where the Graph scratch is
           201326592 B on the Qwen gate against 64 MiB at the planner's fallback of 512, so a
           reservation sized at 512 would refuse the gates' first contexts, which the owner rule
           "never shrink context, place KV" forbids. So L4, which lands the reservation, depends
           on fkpg (a) and does not land before it (§4), and §M37 Q2's interim (size at 512,
           refuse the excess by name) does not apply to `FIRST_CONTEXT`. It stands for every
           other case: a context that needs more than it holds places the excess in its `REGION`
           headroom like any head slot, demoting KV, or is refused at its transaction naming the
           term where that room does not exist; it is never silently short. Each producer's
           value is its owner's function, never a copy: `onednn_scratchpad` from the load-time
           table ("The ONEDNN zone's scratchpad" below; 0 at `c69d5774d`, rulings §M31a),
           `mmid_workspace` only where the route is reachable (step 7's predicate), and zhcn's
           compute slot from its measure pass at the envelope (below). H7ap's first-context arm
           pre-registers each term's bytes at the gates' `-c 4096 -ub 512`. **The reservation is
           not a zone term:** no zone grows for it, the dry run does not see it (the C rule,
           §2.4.5), and its bytes are a `FIRST_CONTEXT` pending range in the device's shared
           zone, recorded with the load's `WEIGHT` ranges (the recording, below). The pack's
           capacity on a device is the `WEIGHT` zone less the device's reservation, so the pack
           cannot spend the bytes the first context needs, and the model's first context on the
           device takes them over at its transaction (§2.4.2 step 5). **zhcn's compute slot at
           load (rulings §M37 Q1, §Z20):** zhcn runs its measure pass once at the load envelope,
           at n₀ and the envelope's `n_ctx`, through a transient measure-only context that owns
           no buffer, makes no unified-cache allocation, registers no live context of the model,
           freezes nothing and is destroyed inside the load before the commit, and that value
           sizes the reservation's compute slot. Until zhcn lands it, H7ap's first-context arm
           injects master's context-time value at the gate shape as a fixture constant, and the
           arm's output labels it as one, never as a measured load value (§3.1);
         - not `onednn_scratchpad`: **it is class C (rulings §M29a, amending §M29 and §M19),**
           the oneDNN primitives' own user scratchpad, consumer (b), charged at the context
           transaction as a head slot in the context's `REGION` headroom, and the load stage
           charges it only in the first context's reservation (above). Its value function stays
           moua's; §2.4.2 (b), "The ONEDNN zone's scratchpad", states it. Master's value,
           `onednn_reorder + onednn_eligible` (`ggml-sycl.cpp:15947`; the plan-side twin at
           `unified-cache.cpp:27565`), is withdrawn (rulings §M29): the comment above it says it
           sizes the `reserve_onednn_scratch` pair (`unified-cache.cpp:17514`, whose one caller
           is `acquire_onednn_pp_scratch`, `ggml-sycl.cpp:1483`, reserving at `:1496`), which is
           23mk's W (`onednn_pp_w`, class D in ONEDNN) and A (`onednn_pp_a`, class C in
           `REGION`), so master counts W twice. **The site is not deleted (rulings §M32 I-4,
           §M33 I-G):** 23mk's §4 commit re-points it to `onednn_pp_w`'s value function and
           renames the store for W (`unified_cache_set_planned_onednn_pp_w_bytes`; 23mk §4.3 at
           `e4f08213a`), and moua adopts the names. 7.14e said the value was deleted from both
           sites, which left the W getter 0 and one fact with two sources (r15 I-4);
         - `pp_pipeline` (RUNTIME), **one term, whose value function is 23mk's (rulings §Z15;
           23mk 4.7b `e81dc2327`)**: the allocation site's own bytes, `pp_pipeline_weight_bytes`
           (the allocation at `ggml-sycl.cpp:92998-93031` at `3d9414c8c`: the maximum
           `weight_bytes` over the schedule, times every buffer `b` the site sizes, and 0 when
           `pp_pipeline_env_enabled()` is off). The pack charges it by calling that function.
           Master's inventory field `pp_pipeline_scratch_bytes` and `plan.pp_pipeline_scratch_bytes`
           (`:15955-15958`) become that function's output, so the bytes have one source;
         - not `moe_control`: it is class C (rulings §M27 (2a)), one slot per context and device
           in the context's `REGION` headroom, charged at the context transaction, and the load
           stage charges it only in the first context's reservation (below). 7.14a to 7.14c
           charged its non-table part here, per (model, device), at the plan's `n_ubatch`;
         - not `mmid_workspace`: it is class C (rulings §M25 I-4; §M9 I-3, the MMID pools are
           context scope and queue-bound), placed once, at the context transaction, as that
           context's `REGION` head slot (§2.4.2 step 5), and the load stage charges it only in
           the first context's reservation (§2.4.5, the C rule). 7.14 listed it here as a
           RUNTIME D term, which would have grown RUNTIME at load by the MMID bytes (about 111.7
           MB on GPT-OSS 120B, the 12343.8 − 12232.1 MB of r13 m-b) and placed them a second
           time as the context's range (r14 I-4);
         - a **transitional RUNTIME term for each unconverted RUNTIME draw (rulings §M38 C-2)**,
           sized by that draw's own size function under its dispatch's gate, until the commit
           that converts the draw deletes the term. At L4+L6 the only unconverted ones are
           beni's rows 13 and 14 (the end states' census below), whose term is
           `XMX_MOE_BUFFERS`: the `xmx_moe_buffers_t` byte functions (`common.hpp:6804-6834` at
           `c69d5774d`) at the envelope's shape, and 0 unless both `GGML_SYCL_XMX_MOE_SORTED`
           and `GGML_SYCL_XMX_MOE_PREALLOC` are set (`ggml-sycl.cpp:67008-67018`). Its setter
           maps to that enum value like any other, and a larger request at a later context takes
           the site's existing declared fallback with a WARN that names the term;
         - and 23mk's placement-dependent terms (§2.4.5), `moe_ptr_table` (the **only** term
           for the MoE pointer tables) among them.
         A forced-off tensor contributes 0 to each. There is no re-pack: the pack charges what
         it places;
      4. **one sized, grow-only ensure from the pack's charge ledger (rulings §M25 I-1, I-2)**,
         per device, before any `record_pending`. The pack keeps a ledger, per device and per
         term, of the step-1 terms and of what it charged in step 3. Step 4's entry is a sized
         ensure whose per-zone inputs come from that ledger, `ensure_arena_zones_sized(dev,
         sizes)`, shared with 23mk; **it never reads a `planned_*` getter.** Each zone is sized
         by the one composition rule, 23mk's (23mk §6.8): **`ensured_Z = max(floor_Z, demand_Z
         + charged_Z)`**, where `demand_Z` is the zone's P terms, `charged_Z` its D terms from
         the ledger, and `floor_Z` its capacity floor. **A floor is capacity, not a term.** The
         floors are master's defaults: SCRATCH 512 MiB (`GGML_SYCL_COMPUTE_ARENA_MB`, the
         compute arena's span, `unified-cache.cpp:4353-4357`) and ONEDNN 256 MiB (`:4459`);
         master's ensure already composes each as a maximum (`:4353-4510`). **RUNTIME has no
         floor on an arena device (rulings §M32 I-2, §M38 C-2, I-3):** `ensured_RUNTIME =
         charged_RUNTIME`, the planned sum of RUNTIME's named consumers, which are its D terms
         and, until each converts, the transitional terms of the draws not yet converted (step
         3; the end states' census, below). The L4+L6 commit deletes master's 512 MiB default
         and its `GGML_SYCL_RUNTIME_ARENA_MB` read (`unified-cache.cpp:4505-4508` at
         `c69d5774d`) from `ensure_planned_arena_zones`, which is the dry run's input (step 5),
         so the dry run and step 4 size RUNTIME from the same terms. 7.14f and 7.14g cited the
         default at `:4486-4489` and kept it "for a device with no arena", but RUNTIME exists
         only with an arena, and a default left in the dry run's input makes the witness fire
         on every correct tree whose RUNTIME is under 512 MiB (r16 I-3, m-1). **So the arena's
         pre-plan layout puts RUNTIME at 0 on an arena device**, and step 4 grows it to the
         charged terms: the
         ensure stays grow-only, because the floor is never laid out, not shrunk away. WEIGHT
         shrinks by the same bytes. **The ONEDNN and SCRATCH floors go the same way (rulings
         §M37 Q3),** each in the commit that moves its zone's last unplanned consumer (the end
         states, below), and from that commit the pre-plan layout puts that zone at 0 too. The
         pack charged
         each term against the capacity that remained, so this ensure cannot fail on budget, and
         the rebuild refusal below is reachable only at the early stage;
      5. **two witnesses, always compiled, before any `record_pending`** (`GGML_SYCL_WITNESS`):
         - `[ZONE-PLAN-BUG] the plan packed against zone capacities that differ from the ensured
           arena`: per device in `plan.devices` and per zone, **`ensured_Z == max(floor_Z,
           demand_Z + charged_Z)`** from the ledger (RUNTIME with no floor on an arena device),
           the packed weights plus the charged terms are at most the ensured zones, and
           `plan.weight_vram_bytes` plus the device's first-context reservation is at most the
           `WEIGHT` zone (rulings §M25 I-1, §M32 C-1). It checks step 4 against the ledger. It
           names
           `weight_vram_bytes`, not `vram_bytes`, which at master includes the MMID workspace
           charges: 12343.8 MB against master's 12232.1 MB weight zone on the canonical gate
           would be a false fire (r13 m-b);
         - `[ZONE-PLAN-BUG] an ensure after the pack would still grow zone %s on device %d by
           %zu B`: per device in `plan.devices`, a **dry run** of `ensure_planned_arena_zones`,
           which sizes each zone from the `planned_*` globals the plan published, is compared
           with what step 4 ensured from the ledger, and must grow nothing (r13 I-A; rulings
           §M25 I-2). **It is the only reader of the getters at the load stage.** Step 4 reads
           the ledger and this witness reads the globals, what the zones' consumers size from,
           so the two sides have independent sources, and a term that step 1 or step 3 forgot,
           and that the plan still publishes, shows here as a growth. Were step 4 the unsized
           `ensure_planned_arena_zones` itself, the dry run would repeat its computation and
           could never fire (r14 I-2). r13 named the ONEDNN floor from the graph-scratch shape
           (`unified-cache.cpp:27581`); since §Z15 that term is class C, which the load-stage
           dry run evaluates as 0. From the Graph-scratch commit on (below; rulings §M26a I-4),
           the load-stage ONEDNN sizing (`ensure_planned_arena_zones`, `:4464`, and the plan's
           twin at `:27658`) reads the no-floor getter, which carries W:
           `unified_cache_get_planned_onednn_pp_w_bytes` (today
           `unified_cache_get_planned_onednn_scratchpad_bytes_stored`, `:2141`; renamed by
           23mk's §4 commit; rulings §M32 I-4, §M33 I-G), and the with-floor one (`:2148-2160`)
           is deleted, its floor branch gone with the term to 23mk's context transaction
           (rulings §M21.3). Until that commit both sides keep master's ONEDNN sizing (below),
           so the witness compares like with like. **This dry run is the one definition both
           designs use (rulings §M26a m13):** the pack's charge ledger compared with the
           published getters; 23mk mirrors it. **The RUNTIME getter it reads
           (`unified-cache.cpp:1566-1575`) sums the D setters' stores and the transitional
           terms' stores** on an arena device (r15 I-3, m-8; rulings §M38 C-2): `moe_onednn`'s
           own setter (`unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes`, the entries
           of the load that laid out the arena), `pp_pipeline`'s, 23mk's
           `unified_cache_set_planned_moe_ptr_table_bytes(dev, k × stride)` (23mk 4.8a L50,
           L5489), and `XMX_MOE_BUFFERS`'s until beni converts rows 13 and 14. At master it sums
           `pp_pipeline`, the ring less its KV-zone part and `moe_control` (`:1569-1574`); the
           L4+L6 commit rewrites it to those stores, `moe_control` leaving as class C and the
           KV-zone split with the device ring (rulings §M32 I-1). The load-stage case is a
           forgotten `moe_onednn` (H7ap); every C term, `ring` and `onednn_scratchpad` among
           them, is published as 0 at the load, and none has a setter. A C term's own check runs
           in the context transaction (23mk's). The enum is kept closed by a source-contract
           gate: every `unified_cache_set_planned_*` setter maps to one enum value, and an
           unmapped setter fails the gate.

      **`moe_control` is a C term: one slot per context and device, in `REGION` (rulings §M27
      (2), (2a); §M26 I-3).** Every byte that scales with `n_ubatch` is context-scoped, so the
      term is never sized from the plan's `n_ubatch`, and the load stage charges it nothing.
      Its objects today: the planned pool, which nothing reads at `3d9414c8c`
      (`moe-control-plan.hpp:225-246`, "LIVE BUT UNCALLED BY DESIGN"), and the same four parts
      as a live, unplanned block, MOE-PREALLOC, 23mk's row 134, which
      `moe_preallocate_inference_buffers` allocates at the first MoE graph
      (`unified-cache.cpp:22176`, the lambda; `:22203`, the tables). Row 73
      (`ggml_sycl_ensure_moe_ptr_table`, `ggml-sycl.cpp:56499`) slices its table part
      (`:56538-56540`), the ids staging is read at `:57592` and `:60689`, and the compact
      pointers and the missing flag at `mmvq.cpp:16602` and `:16636`. At master those are raw
      device allocations outside the arena, never freed (§M20.3, .4). 7.14 cited the
      `moe_transient_ptr_table` cohort (`ggml-sycl.cpp:7645`) as the tables' home; that cohort
      is the host-pinned source of the H2D copy, whose destination is row 73's table, the
      row-134 slice. One owner per byte (rulings §M26 I-3):
      - **the tables are `moe_ptr_table`'s alone (rulings §Z15),** and row 134's MODEL block is
        k × stride only, one per (model, device), drawn inside the load before `finalize_end`
        (rulings §M26a I-1). 23mk's design converts row 134, and this design cites it.
        `expert_ptrs` is not a RUNTIME term: its only reader is `moe_vram_runtime_bytes =
        max(expert_ptrs, control_pool)` (`:27520`), which feeds `host_scratch_total`
        (`:27726-27730`), the llama.cpp-s4ip defect, which this design only cites;
      - **the ids staging is row 75's** (beni; n_expert_used × n_ubatch × 4 B), and
        `moe_control`'s value function excludes it;
      - **`moe_control` is the compact pointer list and its missing flag**, one slot per
        (context, device) on each device that executes a GPU expert of the context's model.
        The list is n_expert_used × n_ubatch × 8 B (`total_batches × sizeof(void *)`,
        `mmvq.cpp:16589`, where `total_batches = n_ids × num_tokens`, `:15938`, `:16169`,
        `:17098`) and the flag 4 B (`:16649`), at the context's own `n_ubatch`. The slot is
        charged at the context transaction beside `onednn_pp_a` and `set_rows_stage`, placed
        as a head slot of the context's fit in its `REGION` headroom (§2.4.5, the C rule), and
        contributes nothing at load;
      - **its value function reads the ungated layout (rulings §M26a I-1):** the compact and
        missing arithmetic of `moe_build_context_control_layout` at the context's `n_ubatch`
        and alignment 256 (`moe-control-plan.cpp:478-497`), `total_bytes − compact_offset`,
        that is align256(compact_bytes) + align256(4). It never calls
        `moe_control_requirement_from_layout`, whose `required` gate returns a valid zero while
        `GGML_SYCL_MOE_CONTROL_CONSUMER` is 0 (`:503-507`) and whose answer otherwise is the
        whole pool, tables and ids included (`:508-518`). At `-ub 512` the slot is 4 × 512 × 8
        + 256 = **16640 B** on GPT-OSS 120B (`n_expert_used` 4) and 8 × 512 × 8 + 256 =
        **33024 B** on Qwen3.5-35B-A3B (`n_expert_used` 8); at `-ub 2048` it is **65792 B**
        and **131328 B**. The layout refuses an `n_ubatch` above `MOE_GPU_UBATCH_MAX` = 512
        (`exceeds_gpu_ubatch`, `:390-393`; `moe-control-plan.hpp:144`), but no dispatch site
        reads that flag: the auto ladder caps a MoE context at 512
        (`src/llama-context.cpp:1407-1429`), an explicit `-ub` is not capped, and the dispatch
        sizes the list from the batch. So the value function computes the two parts for the
        context's `n_ubatch` without that refusal (confirmed, rulings §M28 (2));
      - **the handle moves off the weight extra onto the backend context,** where
        `ctx.moe_ids_cache` already is (`ggml-sycl.cpp:19439`): the per-(tensor, device)
        `moe_expert_ptrs_compact_handle` and `moe_expert_ptrs_missing_handle` fields
        (`mmvq.cpp:16591-16649`) become one per-(context, device) pair, slices of the context's
        carved `REGION` block, owned under `{CONTEXT, id}` (P2). Every MoE op of the context on
        that device reuses them in its queue's order;
      - **one term for the list and flag (rulings §M25 I-5).** The allocations at
        `mmvq.cpp:16614` and `:16646` are fallbacks for the same two objects, taken only when
        row 134's block does not cover the request (`moe_get_compact_ptrs` returns null,
        `:16602`; the flag's twin at `:16636`). The commit that lands the slot deletes both
        fallbacks and both prealloc lookups, and shrinks row 134's block to k × stride in the
        same change (with row 75 taking the ids, rulings §M26 I-3). **The three are one landing
        set (rulings §M28 (3)):** row 134's shrink to k × stride, the `moe_control` slot (from
        moua L4) and row 75 (beni). The shrink cannot land before the other two, and 23mk
        mirrors the order; until the set lands, the block's non-table parts stay unconverted
        master bytes on H7p's list (§2.4.3's transition rule). A context transaction that cannot
        place the slot even with every KV layer on the host refuses that context **by name**,
        before any decode, never mid-inference (rulings §M27 (2)). 23mk's `moe_compact_storage`
        term stays deleted, so the list and flag are charged once;
      - **withdrawn (rulings §M27 (2), (2a)):** 7.14a to 7.14c's step-3 charge of the non-table
        part per (model, device), its routing-commit value change to `total_bytes −
        ids_offset` (r14 m-6), the plan's `n_ubatch` of 512 it was sized at, and its 24832 B
        and 49408 B. The load-stage publish at `unified-cache.cpp:27428`, gated by
        `GGML_SYCL_MOE_CONTROL_CONSUMER` to a valid zero, stays a zero in every commit and is
        not a charge.

      **The end states, in exact bytes (rulings §M20.1).** The display flips with the table
      count, so H3 and H7ap pre-register bytes, never one-decimal MiB. On the B50 merge-gate
      shape (arena 14618 MiB = 15328083968 B; master's pre-plan split `13338.0` = 14618 − 512 −
      512 − 256 agrees, and on an arena device from L4+L6 it is **14522777600 B** (13850.0 MiB
      = 14618 − 512 − 256), RUNTIME laid out at 0 (step 4); SCRATCH 536870912 B, ONEDNN
      268435456 B), stepped from
      `src/llama-model.cpp:426-510` at `3d9414c8c`:
      - **GPT-OSS 120B** (128 experts, every expert 2880 × 2880 MXFP4, WOQ arm): weight slot
        128 × (align256(2880 · 1440) + align256(90 · 2880)) = 128 × (4147200 + 259328) =
        **564035584 B** (537.906 MiB), which is `moe_onednn` (rulings §M28 (1)) at `local(t)` =
        128, the whole layers the pack places (rulings §M32 C-1 (b)). After step 4, RUNTIME =
        564035584 + k × 1024 B, which has no floor on an arena device (and is above master's
        536870912 B floor anyway), and the weight zone = **13958742016 − k × 1024 B** (13312.094
        MiB before `moe_ptr_table`, the k-independent part; r14 m-1), where k is the device's
        `moe_ptr_table` table count from the plan (23mk's term), asserted from the plan, never a
        pinned literal. **k from the term's first commit (rulings §M23 (2)):** the row-134
        conversion lands the charge and deletes row 73's own-allocation fallback in the same
        commit, so the (k + 1) state of §M20.3 never exists. **The pack's capacity is that zone
        less the device's first-context reservation** (step 3): 1132462080 B of ring rows, the
        same bytes master's RUNTIME held for the rows at 512 (1696497664 − 564035584), plus the
        other C head slots;
      - **Qwen3.5-35B-A3B** (256 experts, 512 × 2048 and 2048 × 512): weight slot 256 ×
        (524288 + 32768) = **142606336 B** (136.0 MiB), which is `moe_onednn` at `local(t)` =
        256. RUNTIME = **142606336 + k × 2048 B**, with no floor (rulings §M32 I-2), and the
        weight zone = **14380171264 − k × 2048 B** (13714.0 MiB before `moe_ptr_table`). 7.14e
        held RUNTIME at master's 536870912 B floor, so 394264576 − k × 2048 B of it (about 376
        MiB) was room no planned consumer drew, and its weight zone was 13985906688 B; that
        state is withdrawn. The pack's capacity is the zone less the reservation: 1610612736 B
        of ring rows plus the other C head slots;
      - **RUNTIME's capacity is the planned sum of its named consumers (rulings §M32 I-2,
        §M38 C-2).** 7.14f said two consumers drew master's floor with no term, the
        compute-buffer chain and the MMID pools, and concluded that nothing planned drew it
        from L4+L6 on. That census was incomplete (r16 C-2): after the floor goes, an
        unconverted draw that finds RUNTIME full either takes its declared fallback (23mk
        core's `forbid_vram_zone_spill`), a regression no arm pre-registered, or falls through
        `unified_alloc`'s zone miss to a raw device allocation, the `[EXT-ALLOC]` line
        (`unified-cache.cpp:15231-15320`), which is P1's violation. **The census:** every draw
        that names RUNTIME at `c69d5774d`, by an anchored search of the tracked tree for
        `vram_zone_id::RUNTIME` and every `prefer_vram_zone` assignment through a variable
        (tests excluded), with 23mk 4.9's rows (census at `e4f08213a`, master `2c4f5e45d`), its
        owner, and where it lands relative to L4+L6:
        | site at `c69d5774d` | 23mk row | what it is | owner | its RUNTIME term after L4+L6 |
        |---|---|---|---|---|
        | `ggml-sycl.cpp:93014` | 92 | the PP pipeline buffers | core | `pp_pipeline`, a D term (step 3); 0 with the pipeline off |
        | `unified-cache.cpp:18246` | 125 | the ring's weight slot | moua L4+L6 | `moe_onednn`, a D term |
        | `unified-cache.cpp:18203-18216` | 125 | the ring's activation and output rows | moua L4+L6 | none: per-context `REGION` head slots, converted by L4+L6 itself |
        | `unified-cache.cpp:16222` | 122 | the MMID device pools | moua L4+L6 | none: `mmid_workspace`, a `REGION` head slot, converted by L4+L6 |
        | `ggml-sycl.cpp:37657` | 49 | the compute buffer's RUNTIME leg | zhcn, before L4 | none: zhcn's compute head slot |
        | `fattn-onednn.cpp:457`, `:537` | 25, 26 | the oneDNN FA Q and K/V materializations (`:537` is role KV) | zhcn's carve, before L4+L6 (rulings §M38 C-2) | none |
        | `fattn.cpp:755`, `:1808` | 28, 30 | the fattn device workspaces and the XMX v2 split workspace | jzvq, closed before L4 | none: jzvq's per-context demands |
        | `unified-cache.cpp:18472` | 126 | the MoE reorder temporary | 23mk, before L4+L6 (rulings §M38 C-2) | none: 23mk's term |
        | `common.hpp:6855`, `:6887` | 13, 14 | the XMX MoE graph buffers (`xmx_moe_buffers_t`) | beni, after moua L4-L7 | `XMX_MOE_BUFFERS`, transitional (step 3) |
        | `dense-scheduler.cpp:29` | 20 | `dense_scheduler_alloc_slot`, whose class has no users | deleted in L4+L6 | none |

        **No site lacks a size function,** so no defect ticket is needed today; a site found
        later with none is named as a defect with a ticket, per the ruling. The census's own
        gate (H7p) fails on a RUNTIME draw that is neither a term's, on this list, nor
        converted, and the list only shrinks. The landing dependencies are stated in §4: zhcn's
        carve of `fattn-onednn.cpp:449`/`:529` (master `2c4f5e45d` numbering) and 23mk's reorder
        temporary land before L4+L6, and jzvq before L4. **On both merge gates every
        transitional term is 0** (neither opt-in is set), so the figures in these end states
        hold as written. Three RUNTIME **readers** are not draws and are named so the census is
        not mistaken for them: `ggml-sycl.cpp:16346` and `:16526` sum the zone capacities into
        an overhead, `:17603` reads RUNTIME's free room for the ring's re-plan, which L4+L6
        deletes (§2.7), and `:33997` sets the weight preload's I8 expert guard
        (`ggml_sycl_preload_model_weights`) to `max(256 MiB, RUNTIME capacity)`; each keeps
        reading the planned capacity. **The arm is C9 (§3.3):** zero `[EXT-ALLOC]` lines of any
        role on both merge gates. The same P4 test removes the SCRATCH and ONEDNN floors too,
        each in the commit that moves its zone's last unplanned consumer (rulings §M37 Q3;
        below);
      - **`ring` is each context's, not the load's (rulings §M28 (1), §M32 I-1).** Its
        activation and output slots are one head slot per (context, device), placed at the
        context's transaction, at that context's own `n_ubatch`, in its `REGION` headroom
        (§2.7): align256(align64(`n_ubatch`) · max over t of (min(`local(t)`, `n_ubatch` ·
        `n_expert_used`) · K_t) · 2) and align256(align64(`n_ubatch`) · max over t of
        (min(`local(t)`, `n_ubatch` · `n_expert_used`) · N_t) · 4), times the context's ring
        depth (1). At `-ub 512` that is 377487360 + 754974720 = **1132462080 B** on GPT-OSS 120B
        and 536870912 + 1073741824 = **1610612736 B** on Qwen; at `-ub 2048`, 4529848320 B and
        6442450944 B. Two contexts on one device hold two slots (2264924160 B for two `-ub 512`
        contexts on GPT-OSS 120B), each dying with its context. 7.14b to 7.14d charged the rows
        to load-time RUNTIME at 512 (1696497664 B and 1753219072 B, weight zones 12826279936 − k
        × 1024 B and 12769558528 − k × 2048 B); 7.14e freed them at load and let the pack refill
        the bytes with experts, which left the first context no room for them (r15 C-1). Both
        are withdrawn: the pack charges the first context's rows to the device's reservation
        (step 3), which that context takes over at its transaction (§2.4.2 step 5). The load
        publishes the rows as 0 (the C rule, §2.4.5); `moe_onednn`'s own setter carries the
        weight slot (step 3; rulings §M32 I-3);
      - **`moe_control` adds nothing to either, in any commit** (rulings §M27 (2a)): it is class
        C, placed in each context's `REGION` headroom (its first context's slot in the
        reservation). The figures above are re-derived without the 24832 B / 49408 B that 7.14c
        added at the routing commit, and are unchanged, since they never carried them: RUNTIME
        holds only `moe_onednn` (the ring's weight slot) and the k tables, and row 134's block
        is those k tables.
      - **No MMID bytes are in these figures (rulings §M25 I-4).** `mmid_workspace` is class
        C and placed as the context's `REGION` head slot, so RUNTIME holds no MMID bytes, and
        master's `plan.vram_bytes` excess over the weight zone (12343.8 − 12232.1 MB on
        GPT-OSS 120B at master, r13 m-b) is not a load-stage zone figure. The re-derivation adds
        no MMID byte to the figures above. **MMID bytes leave the load-time
        `plan.vram_bytes`, as intended (rulings §M27 (3)).** `account_moe_mmid_workspaces` no
        longer charges the load in `compute_placement_plan` (`unified-cache.cpp:28371-28375`)
        or `compute_multi_device_plan` (`:30394-30397`); `plan_moe_mmid_workspaces` still
        sizes the pools. Where the route is reachable, the first context's pool is in the
        device's reservation (step 3); a later context places its own as a `REGION` head slot
        (step 7), demoting KV layers to host tiers to make room (the context is never shrunk;
        its KV is placed instead), and a context for which even every KV layer on the host
        leaves no room is refused by name at its transaction, never mid-inference.
      - **Why `ring` is C: the PP MoE dispatch does not chunk (rulings §M28 (1), from code at
        `c69d5774d`).** The batched executor, `try_pp_mxfp4_soa_onednn_f16_batched`
        (`ggml-sycl.cpp:78424`), takes each expert's rows from the whole op's row maps
        (`:78536-78569`, `rows = row_end − row_begin`, at most the micro-batch's tokens), pads
        the busiest expert's rows to 64 (`:78596`), and sizes activation and output from that
        (`:78703-78709`). No loop splits a micro-batch at 512 or at any size, so the rows must
        hold the context's whole `n_ubatch`. Master already knows this: the load plans the ring
        at 512 (`src/llama-model.cpp:481-500`) and publishes per-row bytes so that each
        runtime-context transaction re-plans it at that context's `n_ubatch`
        (`ggml_sycl_replan_pp_moe_onednn_ring`, `ggml-sycl.cpp:17512`, called with
        `next_kv_info.n_ubatch` at `:18512`, llama.cpp-ibj0), placing an ubatch-scaled slot in
        the shared KV zone when RUNTIME cannot hold it. So the ring is `n_ubatch`-scaled today,
        and a fixed 512 ring is not what master runs. **There is no overflow:** the executor
        admits its required shape against the planned one (`pp_moe_onednn_admit_scratch`,
        `:78783`) and re-checks the reserved slot sizes (`:78832-78835`) before any write; a
        shape the ring does not cover is refused, with one WARN per process, "falling back to
        the serialized MoE route" (`:78794`), and for an `XMX_TILED`-claimed op the refusal
        throws `ggml_sycl_fallback_error` (`:78449-78479`, llama.cpp-71hx). Two refusals are
        reachable on master, and this design removes both structurally (§6.21 notes the
        ticket). **(a) The last writer:** the ring is one per device and re-planned by the last
        context's transaction, so a context whose `n_ubatch` exceeds the last one's loses its
        batched path or throws; here each context's rows are its own slot at its own
        `n_ubatch`, the executor reads the context's slot (§2.7), and no other context's
        transaction writes it. **(b) The 64-row pad:** master sizes the rows at `n_ubatch`
        while the executor pads each active expert's group to 64
        (`unified_cache_pp_moe_onednn_slots_for_ubatch`, `unified-cache.cpp:2292-2320`), so an
        `n_ubatch` that is not a multiple of 64 can exceed the plan by up to 63 rows **per
        active expert's slot**, that is up to 63 × `n_expert` rows against the plan's
        `n_ubatch` × `n_expert` (r15 m-2): at `-ub 500` with one expert holding all 500 rows
        and 128 active, 512 · 128 · 2880 · 2 = 377487360 B against align256(500 · 128 · 2880 ·
        2) = 368640000 B. `ring`'s value function sizes at align64(`n_ubatch`), equal to
        master's at every multiple of 64. **The MoE auto ladder tops out at 512 (r15 m-1):**
        llama caps a MoE model's auto ladder at `MOE_GPU_UBATCH_MAX` = 512
        (`src/llama-context.cpp:1423-1446`, "MoE GPU routing ceiling"), so a MoE context on the
        auto ladder has one rung, 512 (1132462080 B on GPT-OSS 120B); only an explicit `-ub`
        reaches 2048 or 4096, and the context's own `n_ubatch` sizes it then;
      - **SCRATCH sits at its floor** by the composition rule (step 4): on GPT-OSS 120B the
        SCRATCH terms (`nonfa_shape` at the load shape, 64 · 512 · 512 · 2 · 3 = 100663296 B;
        `mmq_work_counter`; 23mk's `load_reorder_temp` and `mxfp4_direct_f16_w`) sum under
        536870912 B, so it does not grow (23mk's H3 steps the same sums). **ONEDNN sits at its
        floor:** its only D term is 23mk's `onednn_pp_w`, 23592960 B on GPT-OSS 120B and under
        the floor on Qwen (rulings §M29a: `max(floor, onednn_pp_w + any other D term)`, and
        this design lists no other). 7.14d added master's 34.5 MB reorder + eligible value,
        which §M29 withdraws, and `onednn_scratchpad` is class C (§M29a). **Both floors go
        (rulings §M37 Q3, 7.14g).** The P4 test that removed RUNTIME's applies: a floor
        survives only where a named plan-time consumer demands it, and on an arena device each
        zone is the planned sum of its consumers' terms, with no ruled constant. Each floor
        goes in the commit that moves its zone's last unplanned consumer, so no draw that still
        needs it loses it; until then the figures above hold:
        - **ONEDNN.** At master four consumers draw the zone: W (`unified-cache.cpp:17835`,
          `:17997`), A (`:17836`, `:18017`), the Graph scratch (`:11650`) and the primitives'
          scratchpad (`common.hpp:5890`). A is a `REGION` head slot from moua L4 (rulings §M27
          (1)), the scratchpad from L4+L6 (§M29a), and the Graph scratch from 23mk's
          Graph-scratch commit (§M26a I-4). After the later of L4+L6 and that commit, W is the
          zone's only consumer and `ensured_ONEDNN = onednn_pp_w`; the floor and the
          `max(268435456, …)` form go in that commit. **Neither order is assumed (confirmed by
          the lead, 7.14h):** each of the two commits' arms checks whether the other has
          landed. The earlier one keeps the zone's capacity for the consumers that still draw
          it, as named planned terms each sized by its own function, not as the 268435456 B
          constant: the Graph scratch's `onednn_graph_scratch_zone_floor_bytes` at the
          envelope's shape when L4+L6 is first, and `onednn_pp_a` and the scratchpad's value
          function when 23mk's commit is first. The later one deletes the last of them, and its
          RED is a tree that still lays out the floor's bytes, which the exact ONEDNN bytes
          catch. On GPT-OSS 120B ONEDNN is then
          **23592960 B**, and the weight zone gains **244842496 B** (233.5 MiB) to become
          **14203584512 − k × 1024 B** (13545.6 MiB before `moe_ptr_table`). On the Qwen gate W
          is **33554432 B** (32.0 MiB, `blk.*.attn_qkv` Q8_0 2048 × 8192, by 23mk's
          `gguf_w2.py` rule over the header of the gate's model file,
          `Qwen3.6-35B-A3B-UD-Q5_K_S.gguf`, `glkg-qwen35b-a3b-b1.log:7`), so the weight zone
          gains **234881024 B** (224.0 MiB) to become **14615052288 − k × 2048 B** (13938.0
          MiB). From that commit the pre-plan layout puts ONEDNN at 0 as well, the split is
          **14791213056 B** (14106.0 MiB), and step 4 grows ONEDNN to W;
        - **SCRATCH.** Its floor is load-bearing today, and the consumers that need it are named
          here as defects: every SCRATCH draw in 23mk 4.9's allocation census that no enum term
          covers. The core rows are covered (`load_reorder_temp`: rows 18, 43 and 48;
          `onednn_pp_pool` and `lm_head_f16`: row 61's model-shaped part; `mmq_work_counter`:
          row 100). The others draw the floor with no term: beni's context-shaped rows 17, 21,
          31, 33, 34, 61 (the pool's peak), 81 and 90; zhcn's row 51 (the compute-buffer chain's
          SCRATCH leg); pqmm's rows 59 and 60; and 6lfq's row 9 (census at `e4f08213a`, master
          `2c4f5e45d`). Each becomes its owner's term or head slot. The two named exemptions,
          rows 101 and 138, also draw SCRATCH with no term and need one, or another source,
          first. Rows 102, 111 and 127 have no production caller and are deleted, and rows 129
          and 137 are unreachable under an arena. The floor goes in the commit that converts the
          last of them, and `ggml_sycl_compute_arena_bytes()` then returns the planned SCRATCH
          sum instead of `GGML_SYCL_COMPUTE_ARENA_MB`, so the compute arena spans exactly the
          terms. The freed bytes on GPT-OSS 120B are 536870912 B less the SCRATCH terms:
          `nonfa_shape` (100663296 B at the load shape), `mxfp4_direct_f16_w` (16588800 B),
          `mmq_work_counter` (4 B) and 23mk's `load_reorder_temp`, `woq_packed`,
          `onednn_pp_pool` and `lm_head_f16`. On 23mk's H3 fixture, whose load temporary is 64
          MiB and whose other 23mk terms are 0, that is **352509948 B** (336.2 MiB). The arm
          reads the live value from the ledger, never this fixture figure;
      The displayed values are 537.9 / 13312.1 on GPT-OSS 120B (k-independent part) and 136.0 /
      13714.0 on Qwen (§M20.1; RUNTIME with no floor, rulings §M32 I-2); master's early 1617.9 /
      12232.1 and 1672.0 / 12178.0 carry the ring's 512 rows, and its late-stage 1618.0 /
      12232.0 and 1672.3 / 12177.7 carry the whole control pool as well, and both are rejected.
      **The 120B weight slot is the code's 564035584 B (rulings §M22.1),** not the tensor's
      MXFP4 size 564019200 B (2880 · 2880 · 128 · 17 / 32) that §M20 first quoted: the sizing
      code rounds each expert's scale block to 256 B (`:445-446`; 259200 → 259328), 16384 B more
      in all, and the plan pre-registers what the code allocates (P4). The two print alike to
      one decimal, which is why the logs could not separate them.
      **The load-stage ONEDNN zone, with the C term gone (rulings §M21.3, §M22.2, §M25 I-1),
      from the Graph-scratch commit on (below):**
      by step 4's rule the zone is **`max(268435456, onednn_pp_w)`**, over every ONEDNN D term
      (r14 m-12; rulings §M29a), and **`onednn_pp_w` alone** once its floor goes (rulings §M37
      Q3, above). `onednn_scratchpad`, consumer (b) alone, the primitives' own
      scratchpad, is class C and placed per context, and `onednn_pp_w` is 23mk's W; master's
      reorder + eligible value (34.5 MB on GPT-OSS 120B, `gptoss120b-b1.log:202`; 49.0 MB on
      Qwen, `glkg-qwen35b-a3b-b1.log:228`) is not an input to either. Master's with-floor sum
      (34.5 + 96.0 MB; 49.0 + 64.0 MB, `glkg-qwen35b-a3b-b1.log:229`) is under 268435456 B
      ("oneDNN 256.0->256.0", `gptoss120b-b1.log:184`, `glkg-qwen35b-a3b-b1.log:210`), so on the
      merge-gate shapes removing the floor frees **0 B**. The weight zone absorbs the floor's
      bytes only on a shape whose with-floor sum exceeded the minimum; H7ap's C-rule arm uses
      such a fixture (the minimum precondition, §M22.2).
      **The ONEDNN zone's scratchpad: what the code and the oneDNN API say, the value
      function, and the probe (rulings §M29, §M29a).** Though it is named for the zone it drew
      from at master, `onednn_scratchpad` is a C term from 7.14e, and nothing of it is in the
      ONEDNN zone.
      - **The size is per descriptor, and the backend's descriptors carry M.** The oneDNN API
        gives the size only as a query on one primitive descriptor
        (`primitive_desc_base::scratchpad_desc()`, `dnnl.hpp:4989` in the installed oneDNN
        3.11.4; `query::memory_consumption_s64`, `:689-692`), and promises no invariance in
        any dimension; the installed tree ships no guide text beyond that. Every backend
        matmul is built with static dimensions and cached by a key that includes them
        (`DnnlPrimitiveKey` `m`, `n`, `k`, `gemm.hpp:348-364` and `:964-982`), so each
        micro-batch row count M is a separate descriptor with its own size. The docs therefore
        do not establish M-independence, and a descriptor probe must.
      - **The consumer is per queue, grow-only, and keeps what it replaces.**
        `get_scratchpad_mem` keeps one pool per queue (`scratchpad_map[q]`); a request larger
        than the current buffer allocates a new one and keeps the old "for the context lifetime
        so in-flight oneDNN work can finish" (`common.hpp:5954-5968`). Its peak is the sum of
        the ascending sizes a queue met, not their maximum. The only pre-size is the TG graph
        recording's `pre_allocate_scratchpad(get_max_scratchpad_size())`
        (`ggml-sycl.cpp:107167-107171`), the maximum over the descriptors created so far, a
        measured bound, and a process-global one (P4). **One buffer per (context, queue), at the
        planned maximum (rulings §M29a):** it is allocated once, at the context transaction, as
        a head slot in the context's `REGION` headroom (the first context's in the load's
        reservation, step 3), and never grown. A request above it is the named
        `[ZONE-PLAN-BUG]`: a STRICT (§G1) abort, otherwise the existing non-oneDNN decline with
        a WARN. That line is the plan == reality witness. Host RED: two growth steps on one
        queue, which today's retention serves by keeping both buffers, against the planned
        maximum, under which the pool holds one buffer and the second step is served inside it.
        **The pre-record call sizes nothing (rulings §V16 I-A; 1oxa rev 13 `47206cb` L3297-3306,
        adopted by name):** the TG record path's
        `pre_allocate_scratchpad(get_max_scratchpad_size())` becomes
        `assert_scratchpad_planned(stream)`, which has no size argument, reads no global and
        allocates nothing. It asserts that the (context, queue) buffer from the transaction
        exists at its planned size; for a planned 0, which has no buffer, it checks only that
        the planned value is 0. `get_scratchpad_mem`'s growth and `!valid()` branches are
        deleted, and its `get_size() == 0` early return stays (below). A missing buffer is
        `[ZONE-PLAN-BUG] oneDNN primitive scratchpad buffer missing for queue Q in context C on
        device D`, never a second draw into an occupied range. **A ladder climb re-draws the
        buffer as a C-term re-place (1oxa rev 13, adopted):** the climb's transaction places the
        larger buffer as a new head slot beside the old one, step 6's carve draws the new buffer
        from it, and the publish swaps `scratchpad_map[q]` to it. The old buffer's handle drops
        at the publish through the ordinary event-gated release (§2.4.2 step 8 (d)), so it is
        never freed under queued work and nothing waits on it; graphs that captured its pointer
        are already invalidated at the publish. Two host arms with REDs (§3.1 H7ap): two
        contexts at different `n_ubatch` each hold their own buffer at their own planned size,
        where the RED is master's global pre-size, which sizes the smaller context's recording
        from the larger one's descriptors; and a climb re-places and drops the old buffer only
        after its event, where the RED is an in-place grow that frees the old buffer at once;
      - **the value function (rulings §M29a, §M31, §M32 I-5):** the maximum of `get_size()`
        over the `primitive_desc` objects of **every user-scratchpad family the context's
        dispatch can reach, each under the same gate as its dispatch**, for the shapes the
        context runs on the device, over the M set the dispatch can reach up to its
        `n_ubatch`. The fork's descriptors carry a static M, not `DNNL_RUNTIME_DIM_VAL`
        (above), so one descriptor per weight shape is not enough: the M set is the dense
        path's oneDNN gate minimum up to `n_ubatch`, and the MoE groups' multiples of 64 up to
        align64(`n_ubatch`), unless the bound is proved monotone in M, which the probe's
        evidence does not license. The families are every live `ctx.get_scratchpad_mem` call
        site at `c69d5774d`, **eleven** of the thirteen, each with its gate (rulings §M38; two
        are dead code, below):
        - `gemm.hpp:405`, `:443`, F1, dense PP through `gemm` / `row_gemm`, default (and the
          f32-source router variant R, and `OUT_PROD`, `outprod.cpp:78-83`, which reach the
          same function with their own shapes);
        - `gemm.hpp:911`, the Q4_0 WOQ (`woq_gemm_q4_0` and `_packed`, `:487-523`), reached by
          default on a Q4_0 dense weight's oneDNN PP arm;
        - `gemm.hpp:680`, the Q8_0 WOQ (`woq_gemm_q8_0`, `:646`);
        - `gemm.hpp:1042`, `:1087`, F2, the MoE f16 batched arm, only under
          `GGML_SYCL_MOE_PP_WOQ=0`;
        - `gemm.hpp:1387`, the 3-D WOQ, only under `GGML_SYCL_MOE_PP_WOQ_3D`;
        - `gemm.hpp:1512`, F3, the MoE WOQ 2-D arm, default;
        - `dnnl-ops.hpp:184`, eltwise E (`DnnlEltwiseWrapper`), the **default** SiLU path for
          contiguous F32 with at least 4096 elements (`element_wise.cpp:677-697`, no env gate),
          and `:720`, `:754`;
        - `dnnl-ops.hpp:80`, softmax S, only under `GGML_SYCL_ONEDNN_SOFTMAX=1`
          (`softmax.cpp:393-404`);
        - `dnnl-ops.hpp:338`, binary B (`DnnlBinaryWrapper::binary_broadcast_row`), only under
          `GGML_SYCL_ONEDNN_MUL` (`binbcast.cpp:1159-1180`).

        **Two sites are dead and are deleted (rulings §M36 I-3, §M38):**
        `DnnlBinaryWrapper::binary` (`dnnl-ops.hpp:252-296`, the `:282` call) and
        `DnnlReductionWrapper::reduce_last_dim` (`:398-444`, the `:431` call) have no caller,
        and 23mk deletes both. The value function covers the eleven live sites. 7.14f counted
        thirteen, with the reduction as a caller-less family and `:282` inside binary B.

        A source-contract gate maps every `get_scratchpad_mem` call site to one family and fails
        on an unmapped one, the same closure the zone-term enum has, so a new family cannot
        enter unenumerated. The map records the two deleted wrappers as deleted, so a call site
        that comes back in either one fails the gate until it is given a family. **It is
        enumerated once, at load (rulings §M31):** the early stage creates the descriptors per
        (family, weight shape, M set up to the largest rung the plan knows) into the plan's
        table, outside L1, and a context transaction reads the table at its `n_ubatch`; it never
        creates descriptors per context. A context whose explicit `-ub` exceeds the table's
        largest M extends the table before its transaction takes L1, where zhcn's measure pass
        runs. The probe's timings bound the cost: about 15-30 µs per cached descriptor (median),
        and up to 1.7 s for a first create, which builds a kernel (r15 m-7). **The queue set, by
        name (rulings §M32 I-5):** the buffer is per (context, queue), and the queues are the
        ones the oneDNN sites pass: `ctx.stream(dev, 0)` at `ggml-sycl.cpp:64330`, `:64465`,
        `:65718` and `:79838`; the op's stream in `ggml_sycl_op_mul_mat_sycl` (`:45523`,
        `:45585`); the batched mul_mat's `queue` (`:50933-50972`); and the caller's stream at
        the dnnl-ops and `OUT_PROD` sites. L4's census resolves each to its (device, index) and
        asserts it is in the context's enumerated set; a claim on a queue outside the set is
        `[ZONE-PLAN-BUG]`. **The set is a per-context fact, frozen at the context's transaction
        (rulings §M35; llama.cpp-0j5w):** today `ggml_sycl_execution_queue_for_device` returns
        the TP queue whenever any model on the device has TP on, and `stream()` re-reads it per
        call (`common.hpp:5647-5655`, `:5808-5823`), so another model's TP enable can move a
        live context onto a queue outside its set. Until 0j5w lands that is a pre-existing
        defect shared by every per-context resource, not one this design introduces;
      - **The probe (lead-run; descriptor-only), and its results (rulings §M31, §M31a).** A
        standalone program creates `primitive_desc` objects, never primitives or memory, and
        prints `scratchpad_desc().get_size()` per (card, family, K, N, batch, M), with the
        creation time. Its families mirror the backend's call sites descriptor for descriptor:
        F1, dense PP (`row_gemm` → `gemm`, `gemm.hpp:468-483`, descriptors `:367-383`, called
        with the tokens as oneDNN's `n` at `ggml-sycl.cpp:65718-65724`), every M from 1 to
        4096; F2, the MoE f16 arm (`gemm_batch_strided`, `gemm.hpp:996-1028`,
        `ggml-sycl.cpp:79113`), M in steps of 64 and batch in {1, 2, 4, …, 256}; F3, the
        default MoE WOQ arm (`woq_gemm_batch_mxfp4`'s 2-D form, `gemm.hpp:1428-1461`,
        attributes `:719-727`), whose per-expert descriptor has no batch, M in steps of 64; and,
        in the gap probe (§M31a), at every M from 1 to 4096: the f32 router R (32 × 2880,
        128 × 2880 and 256 × 2048), eltwise E (swish, gelu_tanh and gelu_erf at W = 14336, 2880
        and 512, only where the tensor has at least 4096 elements, as the dispatch requires),
        softmax S (32 or 64 rows × M, 4096 features) and binary B (broadcast-row MUL at F =
        4096 and 2880). **Every row is 0 on both cards:** 3 × 17152 matmul rows (§M31) and
        65512 gap rows per card (§M31a), with no error row, rc = 0 on both cards and `Shmem`
        flat at 1.86 GB. **Positive controls:** an instrument anchor, a combined
        `reduction_sum` returning 16512 B and 32896 B on the same query and library, so a zero
        is a reading and not a blind instrument; and a descriptor match, a GPT-OSS B50 pp512
        run under **`ONEDNN_VERBOSE=profile_create`** (7.14e wrote `ONEDNN_VERBOSE=create`,
        which is not a valid option and prints nothing, so an empty log there was never "no
        primitives"), whose F3 md strings and attributes equal the probe's and whose F1 ones
        match up to a layout-equivalent tag. E is backed by descriptor equivalence only: no
        gate run in hand reaches oneDNN SiLU. **What the probe does not cover (r15 m-6; for
        the lead, §6.22):** F2's batch runs over powers of two, while a group's size is any
        integer (`ggml-sycl.cpp:78603-78620`), so the rows say nothing between them, and F2 is
        reachable only under `GGML_SYCL_MOE_PP_WOQ=0`; and the Q4_0 WOQ (default-reachable),
        Q8_0 WOQ and 3-D WOQ families have no rows. The value function covers them anyway,
        because it is computed, never assumed. **Until a lead-run extension probes them
        (rulings §M37 Q4: Q4_0 WOQ, Q8_0 WOQ, 3-D WOQ and non-power-of-two group sizes,
        descriptor-only by §M31a's method), those families are marked "unprobed" in the source
        gate's family map,** and nothing in this design reads "0" for them from the probe;
      - **What the probe decided.** Not the class, which §M29a fixes at C, and not the value,
        which stays computed. On today's descriptors every probed family's size is 0 on the
        B50 and the B70, so the term's value is 0 and its slot is empty on the merge gates, and
        H7ap's scratchpad arms are scored on the computed value. Re-probe after any oneDNN
        upgrade or new descriptor family (rulings §M31a). The two dead wrappers are deleted
        (above), so they add nothing;
      - **A planned 0 is tolerated by every caller (1oxa rev 13 relay; read at `c69d5774d`).**
        At the value 0 the context has no buffer for that queue, and each consumer already
        handles a zero-size scratchpad. `get_scratchpad_mem` returns an empty `dnnl::memory`
        before it touches `scratchpad_map` when `get_size() == 0` (`common.hpp:5951-5953`). The
        gemm sites call it only under `scratchpad_size > 0` (`gemm.hpp:404`, `:442`, `:679`,
        `:910`). Each dnnl-ops wrapper throws only when the memory is null **and** `get_size() >
        0` (`dnnl-ops.hpp:81`, `:185`, `:339`, the live wrappers), and otherwise passes the
        empty memory. So an empty slot is a legal plan, not a decline, and the healthy path
        needs no non-zero fixture. The early return is kept when the growth branches are deleted
        (above), and H7ap's scratchpad source gate asserts it survives.
      **The Graph-scratch commit (rulings §M26a I-4; §M26 I-4).** One commit, 23mk's, landing
      in beni (23mk §4.8), carries the whole Graph-scratch move. This design cites it and lands
      no part of it in C-1:
      - 23mk's `REGION` charge at the context transaction, under 23mk's value function;
      - the draw itself, moved out of the ONEDNN zone into the context's `REGION` range
        (`onednn_graph_scratch_alloc`, `unified-cache.cpp:11655`, draws from the ONEDNN zone at
        master), with the direct overflow deleted;
      - the load getter switch, `:4464` and `:27658` from the with-floor getter to the W getter,
        `unified_cache_get_planned_onednn_pp_w_bytes` (today `:2141`'s `_stored` getter), and
        the with-floor getter itself deleted (`:2148-2160`; no dead alias);
      - **the 0oxf clamp deleted (r14 m-13).** Master bounds the ONEDNN zone by `max(available /
        4, W)` (`:4476-4500`, where master reads the getter 23mk renames); its comment says it
        exists to bound the Graph-scratch floor. Without the floor, `max(256 MiB, W)` is at most
        that cap whenever `available / 4 ≥ 256 MiB`, so it cannot bind there, and the sized
        ensure takes the ledger's bytes, which the pack already charged against the capacity
        that remained, so a clamp below them would break plan == reality (P4). The dry run's
        `ensure_planned_arena_zones` loses it in the same commit, so the two sides size alike;
      - **the test updates, with the gates and comments that name the with-floor getter's
        load-stage role (r14 m-14):** `tests/test-sycl-onednn-graph-allocator-source.py`'s
        `WITH_FLOOR_GETTER_BODY_CODE` extraction (`:475-481` at `c69d5774d`) and its "graph
        scratch zone floor is additive" check (`:840-846`) re-anchor on 23mk's
        context-transaction sizing, and a new check asserts that `ensure_planned_arena_zones`
        and the plan's twin read the W getter; the "WITH-FLOOR getter, deliberately"
        comments at `unified-cache.cpp:4461-4463` and `:27654-27656` and the two-getter note
        at `unified-cache.hpp:1644-1650` are rewritten to say the load stage reads the W
        getter; and the fixture assertion at
        `ggml/src/ggml-sycl/tests/test-unified-runtime-alloc.cpp:996`, "inventory would exceed
        the conservative ONEDNN tail bound", reads the W getter, since it bounds the
        load-stage tail.
      So neither landing order leaves a gap (the floor gone, the charge not yet landed) or a
      double charge. **Until that commit, C-1 keeps master's ONEDNN sizing on both sides of the
      dry run.** The load stage reads the with-floor getter at `:4464` and `:27658`, under the
      clamp, and step 4 sizes ONEDNN as master does, over the ledger: `min(max(268435456,
      charged_ONEDNN + G), max(available / 4, charged_ONEDNN))`, where G is the Graph-scratch
      floor over the published shape (`onednn_graph_scratch_zone_floor_bytes_swa`,
      `:2148-2160`; 0 with the allocator off). G is capacity for the interim, like a floor
      (§2.4.5, "A floor is not a term"), never a ledger entry, and witness 1's ONEDNN check
      reads that sizing. The dry run computes the same expression from the getters, so it
      grows nothing exactly when the published W bytes are at most the ledger's ONEDNN
      terms, which is what it tests, and it does not false-fire where the with-floor sum
      exceeds the floor. On both merge-gate shapes the interim and the final sizing are both
      268435456 B. H7ap's C-rule arm is an arm of the Graph-scratch commit.
      So the early stage packs against the zones it admits, and B's ranges are recorded inside
      them. **The shared rule (rulings §Z8 I-3, §M13a): the sentence is this design's, the
      message is 23mk's, and each design mirrors both byte for byte.** The sentence: "an arena
      rebuild that meets live bytes or any pending range refuses by name; it never destroys
      them, and the `GGML_ABORT` at `ggml-sycl.cpp:16215` becomes that named refusal."
      `ensure_planned_arena_zones` counts any pending range on the device as live, beside zone
      bytes, chunk leases and live scratch, and `compute_and_store_plan_for_inventory` returns
      `bool`. On the refusal it logs the message, `[SYCL-PLAN] model load refused: arena zones
      on device %d cannot be rebuilt while live allocations or pending ranges remain (zone bytes
      %.1f MB, chunk leases %zu, scratch %.1f MB, pending ranges %zu)`, stores no plan and
      returns false, and the stage passes that up as its refusal
      (`GGML_SYCL_LIFECYCLE_EFFECT_FAILED`, through the stage's rollback guard), never the
      abort. **Its arguments (r12 m-7; relayed to 23mk, whose string it is):** at
      `unified-cache.cpp:4572-4588`, `%d` is the device; the zone bytes are `(double)
      live_zone_bytes / MiB`; the chunk leases are `(size_t) chunk_leases`, since the count is a
      `uint32_t` on master; the scratch is `(double) live_scratch_bytes / MiB`, a new `size_t`
      summed beside master's `has_live_scratch` bool from each member's recorded size
      (`compute_arena_used()`, the scratch pool, the two oneDNN scratches, the reorder temp
      buffer, each persistent scratch and each PP MoE oneDNN slot); and the pending ranges are
      the `size_t` count of pending ranges on the device's TLSFs (23mk §6.8, unchanged since
      `437073a29`). The late stage never calls the ensure, so this refusal is reachable only at
      the early stage;
    - **The admitting stage is the early stage (rulings §X7 I-3).** llama calls
      `stage_inventory_plan` twice. The early call is `llama_model_sycl_compute_early_plan`'s
      apply (`llama-model.cpp:657`); the function is called at `:2106` today and after `:2211`
      under §M18.2, and `create_tensor` then picks each tensor's buffer type from the early
      plan. The late call is at `:709`, through `llama_model_sycl_set_late_inventory`
      (`:2408-2410`). 7.10 recorded at the late stage, so a transaction between the two could
      change the room the late plan saw after the buffer types were already chosen: one fact,
      two sources. So B's room is **admitted at the early stage**. On the
      `compute_placement_plan_early` branch (`:16824`), under that entry's L0 hold, after the
      plan and the zones it sizes, it places on each device's weight side the bytes B's plan
      puts there and records them as load pending ranges with `record_pending({LOAD, B's
      LoadTxnId}, WEIGHT, offset, size)`. It stores the admitted plan, with its identity, its
      create set and its zone terms, in B's inventory record (§M76a). Placement that does not
      fit is the early stage's refusal, by name, before any buffer type is chosen. **A range per
      `(txn, term, device, TLSF)` (rulings §M18.4; r13 I-E); 7.13's one-extent limitation (r12
      m-5) and its refusal are withdrawn.** 7.13 held B's `WEIGHT` room on a device as one
      contiguous extent and said only another live model could fragment it. That was false on
      shipped hardware with one model. The B70's arena is two chunks with a per-chunk cap of
      29472 MB, so its 29033.3 MB `WEIGHT` zone spans two TLSFs, ~28192 and ~841 MB, and a plan
      with more than ~28.2 GB of device weights (GPT-OSS 120B on `level_zero:0` or
      `level_zero:0,1`) needed an extent larger than any TLSF and was refused on first load.
      Under 1oxa, whose weight chunks are at most 4 GiB each (1oxa `93fc6e6` :341), every model
      with more than 4 GiB of device weights would have been. A key now names the TLSF, and a
      device's room is one range on each weight TLSF it uses; the unit the stage places into a
      TLSF is the SYCL<n> buffer, since a buffer never spans chunks ("The range bytes", below);
    - **The recording runs once, after the device loop, from the admitted plan (r12 m-4; §M12
      m-9).** The stage loop calls `compute_placement_plan_early` once per device
      (`:16816-16824`). A multi-device plan covers every device each time (the multi-device
      branch, from `:16219`), but without `moe_multi_gpu_requested` each iteration plans its own
      device (the single-device branch, `:16336-16357`) and re-stages the candidate (`:16373`),
      so a range recorded inside the loop could come from a plan that is not the admitted one,
      and would then be retagged to `{MODEL, id}` and held for the model's life with no
      consumer. So the loop records nothing. After it, under the same L0 hold, the stage records
      B's ranges once, from the one admitted plan (the candidate the loop staged), for each
      device in that plan; a device absent from it records nothing. Reading another device's
      zones inside the loop is safe because step 2 ensured every device before the loop and
      nothing is recorded until after it (r13 m-d); the witnesses run per device over
      `plan.devices`. Each `(txn, term, device, TLSF)` key keeps replace semantics, so a
      re-staged candidate cannot hold its room twice. 23mk's records use the same rule;
    - **The range bytes are the draws' own bytes, each from one function (rulings §M12 I-2, §M14
      I-1; r13 I-C, I-E).** The inventory carries `ggml_nbytes` (`llama-model.cpp:604-611`), but
      a SYCL<n> draw is larger: `ggml_backend_sycl_buffer_type_get_alloc_size` adds a row of
      padding to every quantized tensor with `ne0 % MATRIX_ROW_PADDING != 0`
      (`ggml-sycl.cpp:38065-38075`; `MATRIX_ROW_PADDING` = 512, `presets.hpp:23`), ggml-alloc
      pads each tensor to the buffer's alignment (`GGML_SYCL_BUFFER_BASE_ALIGNMENT`, `:37946`)
      and splits the tensors into buffers at `get_max_size` (`:37951`), and each buffer is one
      `allocate_within` draw rounded to the TLSF's granularity. On GPT-OSS, `n_embd` = 2880 and
      2880 % 512 = 320, so every such quantized weight's draw exceeds its `nbytes`, and a range
      sized in `nbytes` misses on a correct load. So the padding moves into one function,
      `ggml_sycl_weight_alloc_bytes(type, ne)`, which `get_alloc_size` calls, and the plan
      charges each device tensor `placement_vram_charge_bytes(ggml_sycl_weight_alloc_bytes(t),
      GGML_SYCL_BUFFER_BASE_ALIGNMENT)` (`unified-cache.hpp:374` rounds), the call the range
      sums make too, so the plan's device bytes and the ranges are one number. **The buffers are
      replayed, not bounded.** 7.13 bounded the buffer count because the early inventory's order
      was not the order ggml-alloc walks. The create set's order is that order, since the record
      follows `create_tensor`'s calls and each context holds its tensors in call order. So the
      split loop of `ggml_backend_alloc_ctx_tensors_from_buft_impl` moves into one helper that
      takes the tensor sizes in order and returns the buffer sizes; ggml-alloc calls it, and the
      stage calls it over each device's create-set entries. The stage then places the buffers in
      that order, first fit over the device's weight TLSFs, and records each item's placement
      (the items A2 lists: each buffer, and each tensor or copy that draws on its own),
      `{TLSF, offset, size}`, as its key; every draw of that item is the keyed draw at that key
      (§2.3.3 A2; rulings §V13 I-5), so no draw re-decides the TLSF. A TLSF's range is the sum
      of the items placed on it, each
      rounded to its granule. The pack's device-fit test runs the same replay (O(tensors) per
      placement), so the pack never admits a byte the TLSFs cannot hold, and it stays one pass.
      **On a VM device (1oxa rev 10a, `fc1d356`; cited, not re-specified here)** no chunk
      exists when the pack runs, so the replay's first fit walks 1oxa's 4 GiB bins instead of
      existing TLSFs: each bin index is the `(txn, term, device, TLSF)` key, each bin's sum
      becomes its chunk's size, and 1oxa runs no first fit of its own. OPTIONAL copies are left
      out of that replay; 1oxa admits their group pages in the early stage's L1 section,
      beside the chunks. **The class and the group page charge are this design's (1oxa rev 11
      m-2, relayed; rulings §M18.3a):** the plan classes each copy (below), and on a VM device
      the pack charges an OPTIONAL copy its (model, layer, device) group's page-rounded
      increment, `ceil(new group sum / page) − ceil(old group sum / page)`, from the one room
      number 1oxa's fit reads, so the pack and the early stage's admit charge one figure.
      Then:
      - the host-extra tensors staged to that device, by the preload or lazily after the commit,
        add the byte function their staging allocation calls for the tensor's materialized
        layout, placed after the buffers in the order the preload walks them, which is the
        admitted plan's entry order;
      - **every extra copy the plan puts on the device (rulings §M14 I-1)**: the MoE PP
        `alternate_layouts` and the dense oneDNN WOQ `extra_layouts`
        (`add_dense_woq_alternates`, `unified-cache.cpp:27831`, called at `:28369` and charged
        against the plan's `remaining`). **Their bytes have one source, the planner's layout
        (r13 I-C).** The plan sizes a copy with `planner_layout_bytes_for_entry` / `_for_dims`,
        and today the staging sizes it again, in `configure_expert_preload`
        (`ggml-sycl.cpp:34080`) after `ggml_sycl_adjust_layout_for_tensor`
        (`ggml_sycl_layout_bytes_for_dims`, XMX-tiled and padded), so the two can disagree. The
        adjustment moves to planning time: the plan records each copy's adjusted layout and
        `dst_size`, and `configure_expert_preload` consumes them and derives neither. A copy
        whose adjusted layout collapses to the primary's or to AOS is deleted from the plan, so
        it is never admitted and never skipped at staging. The dense WOQ staging
        (`ggml-sycl.cpp:34905-34975`) stages only a copy the plan lists, at its `dst_size`, from
        its range. Its headroom test goes, and its `dense_woq_staged_unplanned` branch
        (`:34941`) becomes a refusal witness, `[ZONE-PLAN-BUG] staging a layout copy the plan
        does not list: %s layout %s on device %d (refused)`, with the copy not staged. An
        always-compiled witness at each staging call checks that the staged bytes equal the
        admitted `dst_size`. A copy whose `target_device` is not its entry's device
        (`ggml-sycl.cpp:25099`, `:25790`, `:29111`, `:29218`) is in **its target device's**
        range. 7.13 cited `:33586-33595` for the dense WOQ staging; that is the counters'
        declaration (r13 m-a).

      PLACE-4 already counts the copies as device bytes (`:16390-16410`), and 1oxa charges them
      (1oxa `b0a7ffb` :1036-1037), so the plan's copy list is the one source for all three.
      Their draws are `allocate_within({LOAD, txn}, WEIGHT, ..., consume = true)` draws, under
      `{MODEL, id}` after the commit. Every copy the plan counts on a device is admitted.
      **The plan classes each copy PRIMARY or OPTIONAL (rulings §M18.3a, amending §M18.3; r13
      I-D).** A copy is **OPTIONAL** when it is a duplicate layout whose primary layout the same
      plan makes resident on the copy's target device: the dense oneDNN WOQ `extra_layouts` of a
      device-resident primary, and an MoE PP alternate whose expert primary is on the same
      device. Every other copy is **PRIMARY**, a cross-device alternate among them, since its
      primary is not on its target device. The class is computed once, at planning time, beside
      the adjusted layout and `dst_size`, and `configure_expert_preload` consumes it like them.
      - A PRIMARY copy is `WEIGHT`: the preload does not call `mark_optional_layout` on it
        (`:34939`), it is never an eviction or yield candidate, and it is freed only at the
        model's unload (§2.1, §2.3.3 A4).
      - An OPTIONAL copy is drawn the same way, inside the same range, and the preload marks it
        optional (`mark_optional_layout`, tag `TAG_OPTIONAL`), so the fit sees it as a buried
        tenant (§2.9) and jehw's predicate classifies it. It is never an eviction candidate.
        It is yieldable only to a context's KV admission, through the yield path (§2.4.2 step
        5), which hands its extent to that context before it releases the handle. The
        a1_long gate's six oneDNN WOQ copies are OPTIONAL.
      The draws are then at most the ranges. What is undrawn at the commit, the lazy tensors' room, passes to
      `{MODEL, id}` with the `WEIGHT` retag (§2.3.3 A4 step 3; rulings §X9 I-D), where the lazy
      materializations draw it and the model's unload clears it; with no lazy tensor nothing is
      left, since the replay includes each buffer's rounding. H7ap asserts both, on a
      padded-`ne0` fixture;
    - **The late stage consumes the admitted plan by its identity and compares; it never ensures
      and never packs (rulings §M14 C-1, §Z13.1, §M18.1; r12 m-6; r13 m-e).** The
      `ggml_sycl_set_tensor_inventory_impl` branch (`:16826`) no longer re-packs placement
      against a live budget (it reads `compute_vram_budget_for_plan`'s free memory today). It
      derives only what the late inventory adds (layer streaming, which needs
      `g_sycl_host_weight_extras` from `create_tensor`, and the host-zone provisioning it
      already runs "synchronous before the loader allocates weight buffers") against the
      admitted placement. Every budget input it reads is the admitted one stored in B's
      inventory record, never `ggml_sycl_device_budget_authority`'s live free memory, which
      moves between the stages on a multi-device host (r13 m-e). **The inventory first.** The
      late inventory is every SYCL-backed tensor in `ctx_map` (`llama-model.cpp:680-688`), each
      with its site. A SYCL<n> tensor on device d must be a create-set entry admitted to d, and
      a SYCL_Host tensor an entry the plan put on the host. A tensor with no such entry is the
      §M18.1 refusal (above). An entry admitted to a device with no tensor there is refused by
      name (`[LOAD-PLAN] the late inventory changes the placement admitted at the early stage`,
      naming the first tensor and both placements). An entry placed on the host is consistent
      whether or not its tensor is SYCL-backed: in default mode `-ot <pat>=CPU` resolves to
      SYCL_Host (the lufn downgrade), and under `--no-host` to CPU. Both walks run
      `create_weight_tensors` and `resolve_create_site`, so either refusal is a witness that
      they diverged, not a case the design expects. The late stage therefore has nothing to
      re-pack. §M17's "re-packs only if the inventory changed" has no remaining trigger, since
      every change is a refusal, and the default-mode re-pack that could have moved a forced
      tensor onto a device (r13 I-B (2)) is gone with it (P3). **Then the terms.** It recomputes
      each zone term, the placement-dependent ones from the admitted placement with step 3's
      charging function, and compares each with its admitted value, per device, **before and
      instead of any ensure**, so the arena rebuild and its `[SYCL-PLAN]` refusal cannot run at
      the late stage. **Equality is the invariant, and the rule is one for moua and 23mk
      (rulings §Z13.1):** a term **larger** late than admitted refuses the load by name with
      this design's string, which 23mk mirrors byte for byte: `[LOAD-PLAN] the late inventory
      changes the zones admitted at the early stage: term %s in zone %s on device %d, early %zu
      B, late %zu B (refused)` (the term slot names which of a zone's terms grew, since one zone
      holds several terms). A term **smaller** late is admitted, never refused, with 23mk's
      WARN, mirrored byte for byte, `[ZONE-PLAN-BUG] the late inventory shrinks term %s on
      device %d: early %zu B, late %zu B (admitted; the early reservation stands)`, once per
      `(load, device, term)`, and 23mk's witness counter `late_term_shrink_admitted` +1 (a
      `GGML_SYCL_PRIVATE_TESTING` dump); the zones and ranges stay as admitted. A forced-off
      tensor charges 0 at both stages, so `--no-host --cpu-moe` logs no shrink WARN and the
      counter stays a signal (r13 I-B (1)). Neither refusal is a silent re-plan. An interleaved
      transaction plans around B's ranges (below), so it cannot cause either.
    - **"B's own" is defined by identity (rulings §M9 I-4).** An allocation is B's own when
      its request carries the owner `{LOAD, B's LoadTxnId}` (§2.3.3 A1, 23mk's addition to the
      primitive), which the caller fills for B's planned draws only: on the loading thread from
      `Registry::bound_candidate()`, and in `load_end` from its ticket's `token.load`, since
      `load_end` can run on another thread (§M9 F2). The allocator reads no thread state. An
      own draw goes through `allocate_within(owner, WEIGHT, ...)` (A2) and never lands outside
      B's `WEIGHT` ranges; every other allocation honours them through `allocate_excluding`
      (§2.3.3); and every transaction's snapshot sees them as allocated, so A's fit plans
      around them.
    - **The loader's weight buffers draw from B's ranges FIRST (rulings §M11 I-C).** Between
      the late stage and `load_end`, llama allocates the SYCL<n> weight buffers on the loading
      thread (`ggml_backend_alloc_ctx_tensors_from_buft`, `llama-model.cpp:2634`). They reach
      `ggml_backend_sycl_buffer_type_alloc_buffer` (`:37570`) with role `WEIGHT`
      (`:37615-37617`: not a compute buft, not a KV buft). On master, with the arena active,
      that function tries the RUNTIME zone first (as role `COMPUTE`, cohort
      `backend-buffer-runtime-zone`, `:37642-37673`), then the KV zone and the SCRATCH zone
      (`:37675-37740`), then the legacy path. Under a bound load that chain is not reached
      for such a buffer: when the arena is active, the buffer's role is `WEIGHT`, its memory
      type is device, and the calling thread is bound to an active load, the function's first
      step is `allocate_within({LOAD, txn}, WEIGHT, size, GGML_SYCL_BUFFER_BASE_ALIGNMENT,
      ..., consume = true)`, with the owner from the bound candidate. **That is a stated
      placement change:** these buffers leave RUNTIME and the KV-zone fallback and land in the
      weight-side room B's plan counted them in. A failed draw is a named refusal, logged as
      23mk's WEIGHT-zone miss row (`[ZONE-PLAN-BUG]`, naming the buffer, its size and B's
      remaining `WEIGHT` room on the device), and the buffer allocation returns null, so the
      load fails at `llama-model.cpp:2637` by name. It never falls through to the RUNTIME-first
      chain, which would change placement a second time. Every other SYCL<n> buffer, the
      scheduler's compute buffers included, is allocated outside a bound load and keeps the
      chain. 7.10 said the buffers "draw from B's ranges" without saying where in that chain,
      which left three readings open (r10 I-C).
    - **Each planned `(tensor, layout)` is drawn once (rulings §M11 I-C, §M14 I-1).** A device
      weight tensor's **primary** is in exactly one of two sets: the tensors `create_tensor`
      gave a SYCL<n> buffer type, which the SYCL<n> weight buffers back, and the host-extra
      tensors (`g_sycl_host_weight_extras`), which the preload stages to the device. A tensor
      has one buffer, so the sets do not overlap. A planned extra copy (above) is a second draw
      of the same tensor in another layout, and it too is drawn once. So the ranges hold the
      plan's per-device `(tensor, layout)` count once. The preload's early return
      (`:33526-33528`) matters only when weights are not evictable; in the default evictable
      mode (`ggml_backend_sycl_weights_evictable`, `:13591`) the preload runs and iterates the
      host-extra set. 7.10 attributed the disjointness to that early return, which is the wrong
      invariant. H7ap asserts, in both weight modes, that the draws cover each planned `(tensor,
      layout)` once, total at most the ranges and leave at most the rounding bound (above), and
      that no `(tensor, layout)` is drawn twice.
    - **The compute arena: no range on an arena device (rulings §M11 I-B).** `load_end`'s
      load-exit effects reserve a compute arena per managed device (512 MB by default,
      `GGML_SYCL_COMPUTE_ARENA_MB`; `:12455-12474`) before the preload, and abort on failure
      (`GGML_ABORT`, `:12470`). With the arena active, `reserve_compute_arena` allocates
      nothing: it points the compute arena at the SCRATCH zone's base when
      `zone_capacity(SCRATCH) >= arena_bytes` ("Budget already charged",
      `unified-cache.cpp:20707-20740`), and refuses otherwise. So on an arena device **the
      SCRATCH zone is the hold**, and 7.10's "range of B's own" was a phantom that no draw would
      ever consume (r10 I-B). The early stage checks `zone_capacity(SCRATCH) >= arena_bytes`
      arithmetically, records no range, and refuses the load by name when it fails
      (`[COMPUTE-ARENA] the model-load compute scratch (%zu MB) exceeds the SCRATCH zone (%zu
      MB)`). On a non-arena device the reserve is the real `COMPUTE` allocation it is on master
      (`:20757-20790`), drawn at `load_end` from no range, since non-arena devices have none.
      **One accessor, and a channel for the refusal (§M12 m-10).** `arena_bytes` is read today
      at `ggml-sycl.cpp:12455-12462` and `unified-cache.cpp:4353-4357`; both, and the early
      check, call one accessor, `ggml_sycl_compute_arena_bytes()`, which is `floor_SCRATCH` in
      step 4's composition rule and not a zone term (rulings §M25 I-1). At `load_end` the
      reserve's
      `GGML_ABORT` (`:12470`) becomes a named refusal, and `ggml_sycl_model_loading_effects`,
      which is `void` today (`:12411`), returns `bool`. The reserve runs before the preload
      inside it, so a refusal returns before the preload runs. On a `false` return `load_end`
      calls `ggml_sycl_abort_owner_effects_noexcept(ticket.token,
      "load_end/exit-effects-refused")` and then `finalize_end(ticket, false)`, so the rollback
      clear below runs. It does **not** reuse the not-committed branch (r12 m-1):
      `ticket.commit` is true on this path, and that branch's `finalize_end(ticket, cleanup_ok)`
      computes `commit = ticket.commit && effects_ok` (`model-lifecycle.cpp:660`), so with
      `cleanup_ok` true the model would go LIVE with no publication and no preload.
      `finalize_end(ticket, false)` marks it `QUARANTINED` with `EFFECT_FAILED` (`:697`), and
      the reaper re-runs the teardown. On an arena device the early check makes that refusal
      unreachable unless the zones changed, which C-1 forbids; on a non-arena device it is the
      real allocation's failure.
    - **Lifetime: the commit steps run only once the commit is irrevocable (rulings §M12 I-1).**
      Each range is trimmed as it is consumed. The retags and the clear (§2.3.3 A4) run in
      `load_end` **after `registry->finalize_end` returns `committed` and not
      `cleanup_required`** (`:13206-13219`), still under `load_end`'s L0 hold, **as the first
      statements of `if (result.committed)` (`:13219`), and `noexcept` (r12 m-2)**: a retag
      changes the owner field of existing records in place under the group mutex and allocates
      nothing, and a failure to take the mutex aborts with a named message, as the guard's does.
      Placed after `moe_discovery_activate_retrying` or the export, or able to throw, it would
      let an exception reach the `:13253` rollback, whose `PENDING_TERM_ALL` clear would drop
      the un-retagged ranges of a model already LIVE. Placed any earlier, a failed validation or
      a cleanup-required commit would leave B's `MODEL_TERM` and `WEIGHT` ranges under `{MODEL,
      id}` for a model that never becomes LIVE and never unloads (r11 I-1). **The rollback clear
      is one function, `ggml_sycl_load_pending_rollback_noexcept(lifecycle::LoadTxnId txn)`**,
      this design's name, which 23mk mirrors (rulings §M13a): `clear_pending({LOAD, txn},
      PENDING_TERM_ALL)` on every device, then 23mk's `onednn_w_rollback_pending(txn)`,
      idempotent and non-throwing. It passes the `LoadTxnId` on, with no model argument and no
      registry lookup (rulings §M15, which keeps 23mk's form for §M14 m-3): 23mk records the
      transaction on W's contribution (`onednn_w_contrib_` holds `{bytes, PENDING | COMMITTED,
      txn}`), and `onednn_w_rollback_pending(txn)` erases the `PENDING` entry whose txn matches.
      A lookup through the Registry's `txns_` record could miss at the validate-failed exit,
      where that record may already be finalized; this form needs no registry state there. So
      W's rollback has two idempotent call sites, the hook and the validate-failed exit.
      `ggml_sycl_abort_owner_effects_noexcept` (`:12675`) calls it with `owner.load`, beside its
      placement-plan abort, **before or together with 1oxa's T13 chunk destroy** (rulings §X9
      I-D, §M13): a range's offsets name a place in a chunk, so no range may outlive its chunk.
      Every non-commit exit reaches it (23mk's design states the same):
      - not committed (`:13124-13135`, `"load_end/not-committed"`);
      - the exit-effects refusal above, through its own abort call and `finalize_end(ticket,
        false)` (r12 m-1), not through the not-committed branch;
      - validation failed (`:13137-13148`). That exit calls the hook only when
        `pending.cleanup_required`; since a recorded range is an owner effect, the exit also
        calls `ggml_sycl_load_pending_rollback_noexcept(txn)` **unconditionally** (23mk mirrors
        it), so no branch of it leaves a range; where the hook ran too, the second call finds
        nothing;
      - committed, then cleanup required (`:13207-13218`, `"load_end/commit-then-cleanup-
        required"`), where no retag has run, since the retags follow this branch;
      - an exception (`:13240-13268`, `"load_end/exception-rollback"`), and the binding-failure
        recovery (`ggml_sycl_finalize_binding_failure_abort`, `:12928-12935`);
      - `load_begin`'s unwind (`:12759`) and the stage's rollback guard (`:16805`, which also
        carries C-1's zone refusal).
      A draw that misses inside B's own ranges is a plan-versus-materialized-bytes mismatch in
      B's own plan, 23mk's WEIGHT-zone miss row (`[ZONE-PLAN-BUG]`); no other transaction can
      cause it any more;
  - **(c) Identity.** The load's own identity is its bound candidate
    (`ggml_sycl_bound_load_candidate`, `:2681`; thread-local, `model-lifecycle.hpp:305`,
    `model-lifecycle.cpp:318-333`), never the published plan, and its publish at `load_end` is
    under L0.

  H9 carries the interleaving: A's transaction between B's early stage and B's `load_end`,
  including between the two stages (§3.1).
- **Per-model plan state (rulings §M76a; the inventory globals).** The transaction and the
  probe read the process-global inventory at `:18245`, `:18265` and `:18666-18675` (the
  `g_placement_kv_info` log), and the preload reads `g_model_n_layer` at `:35535-35568`; every
  publish writes `g_model_n_layer` and `g_placement_kv_info` (`:15347-15348`). That is one fact
  with two sources. So:
  - each model's plan snapshot carries a `shared_ptr<const>` to that model's own inventory
    record: the inventory detail and index, the `g_tensor_inventory_*` sizes, the MoE totals,
    `n_layer` and `kv_info`;
  - the transaction and the probe read it through `lifecycle_select_placement_plan(model)`;
  - the preload reads `n_layer` from its bound candidate, and so do its other two process-global
    reads, `g_moe_expert_total_bytes` and `g_moe_n_experts_total` (`:35535-35568`), which 7.7
    left global beside `n_layer`, one fact with two sources (r9 m-11);
  - the log reads the snapshot's `kv_info`;
  - the publish writes no device-global: `g_model_n_layer` and `g_placement_kv_info` lose their
    writers at `:15347-15348` and their readers above. They are not simply deleted (r9 m-11):
    populate (`:16017-16064`) and the plan computation (`:16208-16367`) use them as load
    scratch, so those uses move into B's inventory record, written by B's entries under B's
    candidate, and the two globals are deleted once no use is left.

  Gate H7ao. The dispatch side, `get_cached_tensor_ptr` (`:19236`, reading
  `g_tensor_inventory_index` at `:19246-19247`), is a separate ticket, **llama.cpp-fsgi** (P1).
  It reuses this snapshot mechanism and lands after moua; H7ao is scoped by path set so that
  fsgi extends it to dispatch.
- **The allowlist: decode's identity-preserving republishes (rulings §L0R, §ZR5 I-3; lead
  ruling on zhcn 5.4 row 30).** Four sites on master republish the current snapshot from
  inside graph compute and do **not** take L0: `:5887` and `:5908` (the MoE secondary-queue
  setup, `ggml_sycl_ensure_moe_secondary_queues_for_plan`, `:5852`), `:60125` (the lazy MoE
  layout materialization, `ggml_sycl_materialize_moe_tensor_planned_layout`, `:60062`) and
  `:74882` (`ggml_sycl_mul_mat_id`). Each republishes into one cache whose plan owner is empty.
  On master they call `ggml_sycl_republish_current_plan()` (`:2996-2999`), which re-runs the
  full publish: it writes `g_placement_publication` and every cache's snapshot, and the
  emptiness check runs outside the mutex (`:5885-5886`).

  **`:60125` is in dead code and is deleted.** `ggml_sycl_materialize_moe_tensor_planned_layout`
  is `static` and has no caller at `3d9414c8c`: its name occurs only at its forward declaration
  (`:29455`) and its definition (`:60062`). So it goes with its declaration, by the orphan rule
  above, and three live sites remain. (Its body's call of `ensure_moe_secondary_queues_for_plan`
  at `:60115` goes with it.) The deletion also removes the function-local `static
  std::atomic<int> planned_layout_log`, which is row 648 of
  `docs/backend/sycl-static-storage-inventory.csv` (rulings §M10, §M11 m-2). The same commit
  regenerates the CSV with `scripts/audit-sycl-static-storage.py` (it writes `--output`, the CSV
  by default), so the row goes. **The witness is `scripts/audit-sycl-static-storage.py
  --check`** (rulings §M12 m-7): rc 0 means the inventory is current, 1 stale, 2 a fail-closed
  rejection; read its own status, never through a pipe.
  `tests/test-sycl-static-storage-audit.py` is a unittest, which rejects `--check` with rc 2, so
  it is not the witness; its ctest, `test-sycl-static-storage-audit`, exits 77 without
  tree-sitter, and a 77 is a skip, not a pass. `--check` does not test that the function is
  gone, so H7ai adds an explicit absence assertion (no definition, declaration or call of
  `ggml_sycl_materialize_moe_tensor_planned_layout` in the tree). Row 648 already cites
  `ggml-sycl.cpp:56194` while the function sits at `:60062` on master, so the row is stale-lined
  today and `--check` fails on master. Both belong to llama.cpp-ldvb, which owns the CSV
  (rulings §M13). **So the regeneration clause is armed only when `--check` passes on this
  design's base (r12 m-15)**, that is, after ldvb's fix lands, as zhcn arms its own (zhcn
  `8a58ad4`); on a base where `--check` fails, this design leaves the CSV to ldvb, rewrites none
  of its rows, and the witness is the absence assertion alone.

  The three live sites instead call
  **`ggml_sycl_republish_current_plan_into_empty(cache, owner)`**, never the full helper (zhcn
  5.5 row 35 uses the same signature). `owner` is the `ModelToken` of the model the calling
  backend context is bound to, its execution root, read by
  `ggml_sycl_execution_current_owner(ctx, owner)` (`:15362-15377`; the `execution_root_*` fields
  are set at `:18943-18946`). `:74882` reads it from its own `ctx`.
  `ensure_moe_secondary_queues_for_plan` (`:5852`) has no `ctx`, so the token is threaded in as
  a parameter from its callers: `:6461`, `:28094` and `:28151`, in helpers that take it from
  their own callers in turn, and `:76240`, `:76621`, `:79907` and `:79944` in
  `ggml_sycl_mul_mat_id`, which reads it from its `ctx`. A context that is not bound to a model
  (no execution root, an unplanned run) skips the call, and the process-global
  `ggml_sycl_has_global_plan()` check at these sites goes away. In **one**
  `g_tensor_inventory_mutex` section `into_empty`:
  1. selects **the publication of the model that owns this cache**, from that model's own
     snapshot (§M76a): `lifecycle_select_placement_plan(owner)`, as `:15253` and `:18889` do.
     Never the latest publication of another loaded model (rulings §Z6 I-C; zhcn carries the
     same clause). A null selection for a bound owner is `[CONTEXT-PLAN-BUG]`. 7.7a read
     `g_placement_publication`, which names whichever model published last, so an emptied
     cache of model A would receive B's plan when B had published last;
  2. re-checks that this cache's plan owner is still empty, and returns as a no-op otherwise;
  3. computes this cache's participation the way `ggml_sycl_prepare_plan_publication_locked`
     (`:15314`) does;
  4. installs that snapshot, or `nullptr` for a non-participant, into **this cache only**;
  5. compares the installed snapshot's identity (version, `model_id`, `load_txn_id`) with that
     model's current publication, in the same section.

  It writes no device-global: not `g_model_n_layer`, not `g_placement_kv_info`, not
  `g_placement_publication`, and no other cache's snapshot. Every L0 committer publishes under
  the same mutex (the transaction's CAS at `:18646` runs under L1), so a concurrent commit lies
  wholly before or wholly after the section. A mismatch is therefore a code defect,
  `[CONTEXT-PLAN-BUG]` with a STRICT (§G1) abort, never a race. It relies on §M76a retiring the
  readers of the two inventory globals; the dispatch reader `get_cached_tensor_ptr` (`:19236`)
  is llama.cpp-fsgi. The full helper has one caller left, `:33769` in the weight preload, which
  is a load entry under `load_end`'s L0 hold with a witness check that the thread holds a
  `LOAD` token (`ggml_sycl_replan_token_held(LOAD)`); it is not an allowlist site. The full
  helper is the second source for one fact that §M76a removes, so this form is not an
  optimization. A gate (H7ai) proves it: the three live sites call
  only the narrow form, and it contains no write to the three globals or to another cache's
  snapshot. H9 runs it against a concurrent L0 committer.
- **Who never takes it (rulings §L0R; r7 m-7).** "Decode never takes L0" means **graph compute
  and dispatch never take it**. `llama_decode` can reach a re-plan legitimately through
  `sched_reserve` (the resync at `llama-context.cpp:1571`, and the FA recheck through
  `resolve_fused_ops` at `:1298`). That is a re-plan, and it takes L0 like any other. The
  accepted cost is that re-plans, loads and unloads on different devices and models serialize;
  they are rare (rulings §E.2).
- **The probe and the FA recheck validate against their own model's plan (rulings §L0R, §M8
  I-2).** On master both compare the model token with the one global snapshot (`:18761-18767`,
  `:19041-19056`). With two models loaded, B's load or re-plan publishes B's plan between two of
  A's L0 holds, and A's probe then answers STALE_IDENTITY ("not the published model"), which
  llama turns into a refusal or a throw (`llama-context.cpp:1531`, `:1195-1200`). L0 serializes;
  it does not re-bind. So:
  - **the entry resolves the plan once,** `lifecycle_select_placement_plan(model)`, the plan of
    the model it is called for, and validates against that;
  - **the transaction body reads the snapshot it is passed, and nothing else.** On master the
    shared body reads `current = ggml_sycl_global_plan_snapshot()` (`:17852`) and checks it
    against the expected model (`:17857-17867`), and the probe never publishes A's plan (its
    comment, `:18717-18720`). So after B publishes, the entry's per-model check would pass and
    the body would find B, which 7.6 had made a `[CONTEXT-PLAN-BUG]`; without the check it would
    evaluate A's candidate against B's plan (P4). The body therefore takes `current` as a
    parameter. There is no global snapshot read at `:17852`; the expected-model check applies to
    the passed snapshot; and both relock checks (`:17886` after taking L1, `:18122` after the
    yield) compare the model's current per-model plan with it, not the global publication;
  - **on the full path** the wrapper publishes the model's plan under the same L0 hold before
    the body runs (`:18955`), so the per-model and global answers coincide there, and the CAS
    (`:18646`) expects that pointer. In probe mode nothing is published and there is no CAS;
  - **the FA recheck's under-lock re-confirmation** (`:19054-19056`) compares the per-model
    plan with the one the entry resolved, not the global snapshot.

  H9 asserts that the probe evaluated A's plan: A and B have different residencies, and the
  probe's answer is A's. The FA recheck has a second defect of the same family: its headroom
  predicate reads **live** device free memory (`ggml_backend_sycl_get_device_memory()`, the
  pvjr comment at `:19058-19075`), a reading taken after this process's own destroys, which
  rulings §FM forbids for any sizing or gate (rulings §M76.4). On an arena device it is a second
  source of "fits" (P4; §2.2), so it reads the fit's ledger (the registry's residency answer)
  instead. For a `tenants_planned` context the recheck is subsumed (r8 m-4): an FA flip changes
  the tenant section, so it goes through MEASURE and the coverage query like any other tenant
  change, and a recheck that also charged the KQ chunk would count it twice, as step 4's non-FA
  check would (zhcn §3.8 row 21).
- **What changes.** Under L0 no other re-plan, release proc, load or unload runs concurrently,
  so every return below can fire only if a mutator skipped L0. Each is `[CONTEXT-PLAN-BUG]`
  (an abort under `GGML_SYCL_STRICT_LEASES=1`, §G1), never `busy` and never a retryable refusal:
  - jehw's five in the transaction body, at master `3d9414c8c`: `:17867` "stale identity",
    `:17881` the live-update lease, `:17886` "plan changed while acquiring the transaction
    lock", `:18122` the yield's relock, and `:18646` the lost publication CAS (a `refuse`, which
    is PLAN_REJECTED and fatal on the server);
  - the wrapper's own `BUSY` returns: module admission (`:18848-18849`), the ticket
    (`:18867-18870`), an open invocation or graph (`:18925-18929`), and `attach_root`
    (`:18934-18935`);
  - **the other L0 entries' `BUSY` returns (r8 m-5; rulings §M76.4, §M76.5):** the probe's
    module guard (`:18744-18747`) and its mapping of a transaction `busy` to `BUSY`
    (`:18817-18818`), the FA recheck's module guard (`:19036-19039`), unload's (`:12285`),
    activate's module guard (`:15230`) and live-update ticket (`:15238` onward), and the module
    guards of `stage_inventory_plan` (`:16779`) and `load_end` (`:13100`). A module guard that
    fails under L0 is the §M76.5 case: shutdown and reactivation hold L0, so the module can be
    found non-ACTIVE by another L0 holder only if a caller used it outside its lifecycle. A
    ticket or a transaction `busy` can fire only if a mutator skipped L0.

  `load_begin`'s `LOAD_BUSY` is not in this list. It refuses a concurrent load, or a load into a
  closed module (`:12729-12732`), by name, as on master, and this design keeps it (above).

  The ring's own `busy` at `:18515` ("scratch ring claimed by an in-flight dispatch") is gone on
  arena devices (r7 m-8): the full path no longer releases the ring, and on the tenant-only path
  a claimed slot after (s) is step 7's occupancy `[CONTEXT-PLAN-BUG]` (rulings §B step 7). On a
  device with no arena, u1bb's ring re-plan and that return stay as they are, outside this
  design. llama's `BUSY` sleep-backoff loops (`llama-context.cpp:1154-1158`, `:1522-1526`) are
  deleted (rulings §L0R); the deletion is in zhcn's scope, and this design depends on it.
- **No ring invariants are left to check.** RELEASING, `ring_plan_gen`, the step-2 copies and
  their pins (rulings §RING; r7 m-3), and the two ring returns that fired on them, guarded the
  device ring, which several contexts' transactions and release procs wrote. The ring's rows
  are now each context's own slots (rulings §M32 I-1), written only by that context's
  transactions and release proc, so all of it went with the device ring, and (e) checks the
  rows' handles with `use_count() == 1` like every other tenant slot.
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
    never reached on a thread that already holds L0. Its `LIFECYCLE` token is
    **outermost-only**: a witness check aborts if `ggml_sycl_replan_token_held()` is already
    true when it enters. That is
    distinct from the transaction entries' legal nesting, and H9 has an arm for each.
- **Gates.** H7ai is reachability-based: from every exported function, no call of the four
  publish and live-update callees or of `acquire_live_update` is reachable except under a token
  taken at the top of that entry, through one of the three allowlisted identity-preserving
  republishes, or from a classified test-only hook (below); graph compute and dispatch take no
  token; the token is taken before L1 and never under L1-L5. H7ao gates §M76a's per-model state.
  H9 runs:
  - two contexts re-planning concurrently, on one device and on two devices, a re-plan racing a
    load on the other device, and a re-plan racing an unload;
  - A in a transaction, then B loads, then A's probe and FA recheck, with zero STALE_IDENTITY,
    and with A's residency, not B's, in the probe's answer;
  - A's transaction between B's early stage and B's `load_end`, and between B's two stages;
  - every exported publisher §2.4.2 lists, against a parked holder, and the allowlisted
    republish against a concurrent L0 committer;
  - a covered read against a parked L0 holder that runs an unload, a quarantine restore and a
    load (rulings §M76.1);
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
- this call's pending ranges, and the `FIRST_CONTEXT` ranges step 5 handed over from the
  context's model, saved so that a rollback can re-record them (§2.3.3 A1; rulings §M32 C-1);
- the pending registry insert, and this call's step-7 MMID registry entry, non-accepting until
  the CAS (§2.4.2 step 7; rulings §M12 I-3);
- after a commit, **the superseded handles** (step 8; r5 m-a): device slots, the context's
  old ring rows among them, and any mirror handles 423j's retire withdrew (§2.9), all dropped
  after L1 is released. Host slots are never superseded within the plan (§2.4.3).

7.14e's guard also held copies of the device ring's slot handles with `pinned[slot]` counts, a
tentative ring contribution and a `RELEASING` mark; all four went with the device ring
(rulings §M32 I-1). It owns nothing that existed before the call. The context's current slots,
its ring rows included, stay with their owner until step 8. So on any return other than a
committed publish (a `refuse()`, a `[CONTEXT-PLAN-BUG]` return, or an exception, from any step;
there is no `busy()` under L0), the destructor
rolls back in two phases, and never has to re-acquire room:
1. **Metadata, with one leaf-class lock at a time.** Clear this call's pending ranges with
   `clear_pending({CONTEXT, id}, REGION)` on each device it recorded on, which takes each TLSF's
   group mutex itself, alone (L5; §2.3.3). This is the guard's one clear form (r12 m-8);
   `clear_pending_locked` is only for a caller already inside a group-mutex section, which phase
   1 is not, with one exception: on a TLSF where step 5 handed over the model's `FIRST_CONTEXT`
   ranges, the phase takes that TLSF's group mutex once and, inside it, runs
   `clear_pending_locked(tlsf, {CONTEXT, id}, REGION)` and then `record_pending_locked` of each
   saved range under `{MODEL, id}` / `FIRST_CONTEXT` (rulings §M32 C-1), so no allocation sees
   the reservation's room free between the two. Retire this call's step-7 MMID entry, if any.
   Discard the pending registry insert (local state). This phase needs no L1; 7.14e's ring-lock
   section here went with the device ring (rulings §M32 I-1).
2. **With no lock held.** Drop this call's new handles, the unused pre-minted controls, and on
   a commit the superseded handles, so each `zone_free` runs lock-clean. Phase 2 runs on every
   exit, the commit included; phase 1 runs on every exit that did not publish.

Every old slot is untouched, so after a refusal at the MMID step or the CAS the context's
slots, its ring rows included, are exactly where they were, and a claim in flight on them is
unaffected (r4 I5; H9).

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
     different KV shape`, aborting under `GGML_SYCL_STRICT_LEASES=1` (§G1). The key is frozen at
     the first publish, and every input to it is fixed for a context's life.
   - This is today's `admitted_kv` rule, *"a same-shape republish by an admitted context keeps
     the published residency"* (master `ggml-sycl.cpp:17738-17747`), made physical.
   - Step 1 takes only `kv_region_mutex_`, so on the full path it runs before L1, and the
     host-tier allocation that follows it ("before step 2") is still before L1 (r5 I-I(3)).
   - It makes the split case safe. Every backend's run of the transaction re-fits every device
     (*"each re-fits, because admission is per backend"*). The first run to reach `(c, d)`
     reserves and publishes the entry, and every later one matches it.
2. **Plan.**
   - **Reconcile the demand records (§2.4.3).** This context's CONTEXT-scope records come from
     the descriptor, its ring rows among them, at the reservation's `n_ubatch` (§2.7; rulings
     §M32 I-1). There is no device record to reconcile. 7.14e's ring-lock section, which copied
     the device ring's slots, pinned them and read its generation and `RELEASING` mark before
     the snapshot, went with the device ring: the context's ring rows are held by its own
     registry entry, which only its own transactions and its release proc write, all under L0.
   - Snapshot the geometry (§2.3.1: cache locks, then the group mutex, then release), over the
     shared zone's TLSFs only (the `REGION` headroom, §2.3.3; rulings §M32 I-2), with the
     model's `FIRST_CONTEXT` ranges on the device counted free when this is the model's first
     context there (§2.4.1), and run `kv_region_fit`. That yields the residency, the yield
     prefix, the extents, the head-slot placements, the superseded slots and
     `free_after_full_kv`.
   - If a head slot cannot be placed even with every KV layer on the host, the transaction
     refuses, naming the tenant; no OPTIONAL copy is yielded for a head slot (rulings §V13
     I-8).
3. **The byte-accounting steps, in their existing order, before any yield (r2 N-I4; r3 m4;
   rulings §M8 I-5(c)).** These are `rebuild_runtime_per_device_vram`,
   `moe_mmid_reaccount_replacement`, and the MMID re-plan
   (`replan_moe_mmid_workspaces_for_runtime`, master `3d9414c8c` `:18245`) with its
   `BUDGET_EXCEEDED` fallback through `ggml_sycl_try_demote_runtime_kv` (`:18255-18266`).
   - They run on step 2's residency. Under an arena, their KV term is the fit's region bytes,
     and the in-arena head-slot bytes are charged at the commit for accounting only (§2.3.2
     "Charging"; r5 m-b).
   - **The MMID re-plan sizes; it does not admit (rulings §M8 I-5(c)).** Revision 7.6 kept the
     re-plan's RUNTIME-growth budget and its demotion as "a different fact", while step 7 made
     the MMID device pool a head slot of step 2's fit: two sources for one fact. And on the
     tenant-only path (0) ran only the fit while (ii) ran steps 2-4, so a candidate whose MMID
     demand grows (a larger `n_ubatch`, or `n_expert_used` under the deprecated
     `llama_set_warmup`, rulings §D20.1) passed (0) and was refused by the budget in (ii), after
     (f) had released: a predictable refusal after the release. On an arena device the re-plan
     now only computes the workspace sizes, pure over the model's own inventory record
     (§M76a), and the fit is the one source of whether they fit: the device pool is a head slot
     (step 7), and a pool that does not fit demotes KV like any head slot, or refuses naming the
     tenant when even all-host does not fit. The `BUDGET_EXCEEDED` demotion is not reached on
     an arena device, and nothing in step 3 demotes there. **(0) and (ii) run the same
     arithmetic:** the MMID sizing, then the fit, with the same inputs. A device with no arena
     keeps master's step 3.
4. **Every predictable refusal, before any yield.** After this step, only a runtime shortfall
   or a lost race can change the outcome.
   - The non-FA scratch check (jehw `:18187-18246`) and the publication-ID check (jehw `:18351`)
     move here. The non-FA check, with its shape recording and its SCRATCH raise, runs only for
     a context whose tenants are not planned (`!tenants_planned`, a flag only zhcn's MEASURE
     sets). Under a plan the KQ chunk and the `context-nonfa-stage` slots are already head slots
     of step 2's fit, so the check would count them twice (zhcn §3.8 row 21, its gate 32).
     **Its headroom reads the ledger, not the driver (rulings §FM; r8 m-4).** The check,
     `ggml_sycl_check_nonfa_attn_scratch` (master `3d9414c8c` `:17164`), reads live device free
     memory (`ggml_backend_sycl_get_device_memory`, `:17208`). In (ii) that read would land just
     after (f)'s frees, in the stale-low window rulings §FM describes. On an arena device a
     `!tenants_planned` context remains possible only where zhcn's MEASURE did not run for it,
     and there the check reads the registry's residency answer (the fit's ledger), as the FA
     recheck does. A device with no arena keeps master's read.
   - The ring has no separate "does not fit" refusal (u1bb `:18388`) and no budget-room check
     (u1bb `:18362`, deleted, r4 I7): its slots are head slots of step 2's fit.
   - Nothing after the yield can fail for a runtime reason (rulings §M7 I-5, §E.2). The MMID
     workspace is planned and carved at step 6, so step 7 allocates nothing, and under L0 the
     CAS cannot lose. What is left is a shortfall, which step 6's re-fit demotes, and
     `[CONTEXT-PLAN-BUG]`.
   - **Probe mode ends here.** A probe runs steps 1-4 with no side effects: no pending range, no
     yield and no carve (rulings §M7 I-3). On a matched key it runs the tenant-only path's
     arithmetic only (the MMID sizing, then the fit; step 3), with the context's own slots,
     its ring rows included, counted free by arithmetic (zhcn's step (0); §2.4.2 "(0)
     Probe").
   - **Probes see no other re-plan's pending ranges, and do see a load's (r4 m13; rulings
     §E.2; r9 m-11).** A probe holds L0, so no other re-plan transaction can be mid-yield while
     it runs, and the spurious refusal that r4 m13 accepted is unreachable. A load's pending
     ranges are different: they outlive L0 across B's load span, from B's early stage to B's
     `load_end` ((b) above), and a probe **does** see them as allocated, correctly, since they
     hold room B's plan counted on.
5. **Record the pending ranges, then yield (addendum (b); r3 C2(a); r4 I8).**
   - **Before the yield, and before `lock.unlock()`**, record **every** placement the fit made
     (all KV extents and all new head-slot placements, not only the yielded run) with
     `record_pending({CONTEXT, id}, REGION, offset, size)`, each on its own TLSF under that
     TLSF's group mutex (rulings §M13; the query `pending_ranges(c, d)` only reads). The one
     exception is the extent of a planned OPTIONAL copy the fit picked: the yield path records it
     as its retag, in the same section as the handle's release (below; rulings §M18.3a). Two
     more are named: 23mk's Graph scratch slot, recorded under `{CONTEXT, id}` /
     `ONEDNN_GRAPH_SCRATCH` and never carved (§2.3.3 A1; the 7.14f queue's 23mk (a)); and **the
     first context's handover (rulings §M32 C-1):** on a device where the fit counted the
     model's `FIRST_CONTEXT` ranges free, each TLSF's section records this call's placements
     with `record_pending_locked` and clears the model's ranges with
     `clear_pending_locked(tlsf, {MODEL, id}, FIRST_CONTEXT)`, one section, so the room passes
     from the reservation to this call with no instant at which another allocation sees it
     free. The guard saves the cleared ranges and re-records them on any exit that does not
     publish ("The transaction guard", above).
   - Until step 6 carves them, or the guard's first phase clears them, every other placement
     treats them as occupied: `zone_alloc(WEIGHT)` through `allocate_excluding` (§2.3.3), and
     every other transaction's snapshot (§2.3.1). This holds whether or not a yield happens:
     jehw unlocks only if something was retired (jehw `:17899-17903`), but weights never take
     L1, so the ranges are needed with L1 held too.
   - A WEIGHT request that fits **only** inside a pending range gets the zone miss it would get
     a moment later, once the range is carved. That is the post-commit state, so this adds no
     new behaviour class.
   - **23mk's A is a head slot of step 2's fit, not a step of its own here (rulings §M27 (1);
     §M25 I-6).** `onednn_pp_a` is placed by `kv_region_fit` with the other head slots, in the
     context's `REGION` headroom, and step 5 records its placement with every other one. Its
     room is `REGION` headroom less every other owner's pending ranges, which the fit treats as
     allocated (§2.3.3), and it never reads RUNTIME's free bytes. A is mandatory like every
     head slot: if it does not fit with every KV layer on the host, step 2 refuses the
     transaction naming A, a pre-yield refusal (step 4's rule). This holds from moua L4, where
     the fit exists, and not from beni; 23mk switches A's fit-and-hold wording in the same
     fold. On an arena device the MMID pool's bytes have **one** source: this call's `REGION`
     range here, and its carved block after step 6. `runtime_pending` and 23mk's
     `mmid_runtime_pending_bytes` are zero there, and nothing subtracts them a second time.
     7.14c's interim form, A's fit in RUNTIME's free bytes less
     `pending_bytes_excluding(dev, RUNTIME, {CONTEXT, id}, ONEDNN_PP_A)` between the recording
     and the yield, is withdrawn: where the RUNTIME demand exceeds its 512 MiB floor, as on
     both MoE merge-gate shapes, step 4 sizes RUNTIME to exactly the load's terms, and that fit
     found no room (§6.19's question, answered by §M27 (1)).
   - **Step 5 has two L1-released windows, in this order (23mk r8 m14; 23mk `c3942d236`
     L1351-1353: "A's reap window (L1 released) sits before the yield's"):
     1. **23mk's A reap window**, before the yield. 23mk releases L1 for A's reap and relocks.
        After that relock the plan identity is checked under L0 (a change is
        `[CONTEXT-PLAN-BUG]`); this call's `REGION` ranges, A's placement among them, are still
        recorded, pending and excluded, since the reap touches none of them; and step 6's
        re-fit re-places A with the other head slots (rulings §M27 (1)), so A has no separate
        recompute;
     2. **the yield's window** (below). After that relock, step 6 re-snapshots and re-fits
        inside this call's own `REGION` ranges, with the same identity check.
   - Then the yield, as jehw HEAD runs it (jehw `:17873-17915`), with one change: the picks are
     the fit's strict prefix plus the buried tenants it chose, passed as an explicit pick list
     through **llama.cpp-uwlx** (owner impl-jehw, after jehw lands; proposed by impl-1oxa):
     `yield_optional_layouts_begin` takes the picks, one group per pick for moua, identified by
     key plus entry generation (no raw pointer, no caller-supplied size), re-checks each under
     L1 with the one predicate, and retires all-or-none per group. A skipped group is a
     shortfall, which step 6's re-fit demotes. On arena devices `select_optional_layout_yield`
     and `kv_zone_snapshot` stop being the pick source, and jehw's `kv_layers_allocatable` hold
     stays until L4's fit is exact. **Planned copies (rulings §M18.3a, amending §M18.3).** A
     PRIMARY planned copy is never a pick. A planned OPTIONAL copy (§2.4.2 (b), "The range
     bytes") is a pick like any buried tenant (§2.9, the cost-ordered ladder), and the yield path
     handles it in one critical section, under the TLSF's group mutex, per pick. **The
     section's lock order, stated once (rulings §M25; §L6, §M16b; r14 m-5 (b)):** L1 (the
     transaction body holds it) → the cache lock that guards the copy's entry, which jehw's
     retire already takes under L1 → the TLSF's group mutex, and then the TLSF operations inside
     the group mutex. The retained-handle store's mutex is never taken inside it, and no handle
     is destroyed inside it:
     1. **retag (rulings §M21.1)**: its block's extent passes from `{MODEL, m}` / `WEIGHT` to this call's
        `{CONTEXT, id}` / `REGION` pending room. The copy's draw consumed its part of the
        model's range (`consume = true`), so there is no `{MODEL, m}` range record over the
        block to move, and the retag is `record_pending({CONTEXT, id}, REGION, off, size)` over
        exactly the block's extent, made under the group mutex already held. Step 5's recording
        above leaves a planned-copy pick's extent to this retag, so the handover is atomic with
        the release. The record **appends** beside this context's step-5 `REGION` ranges on the
        same TLSF, never replacing one (§2.3.3, the append rule for every owner but a load's;
        rulings §M25; r14 m-3);
     2. **mark the key yielded on the copy's cache entry (rulings §M25; r14 m-5 (a))**: the
        entry's `yielded` flag, an atomic written only inside this section, under the group
        mutex. The admitted plan is a `shared_ptr<const>` record (§M76a) and is not written.
        The flag is the one source for "is the copy there": dispatch keys off the retire (the
        cache entry) together with the optional-layout epoch bump below and selects the kernel
        from the resident primary layout, "the loaded layout is the answer"; the staging
        witnesses and the lazy path read the same flag and treat the copy as absent, never as
        missing;
     3. **release the handle (rulings §M21.2)**: the plan's owning handle leaves the cache entry and joins the
        transaction's drop list, and it is destroyed in the yield's finish with neither L1 nor
        any group mutex held (L0 stays held; r14 m-5 (c)), like every retired pick (a destroy
        under the group mutex would self-deadlock, since the free takes it). The destructor
        stays reason-free (P2). Bytes freed while the retagged range stands land inside this
        context's `REGION` range, so they reach this context's KV draw at step 6. **A late last
        drop (r14 m-4):** a reference that outlives step 6's carve, which clears the range
        (`clear_pending_locked`), such as a pick whose frees stayed queued or an E reference
        the retained-handle store holds past the barrier (§M16a), returns its bytes to the
        general TLSF when it drops. Step 6's re-fit has already seen that part of the range as
        not free, so the KV that needed it demoted, and the key stays yielded.
     A pick the retire's predicate re-check skips gets none of the three, and step 6's re-fit
     sees the shortfall and demotes. **On a VM device the section is 1oxa's (rulings §V12):**
     one L1 hold (the begin) retags the group's pages to the context's guard and marks each
     retired key yielded; the handle release follows jehw's reader barrier (the finish),
     reason-free, the same begin/finish shape. The pages are the context's from the begin
     onward, and a draw of a yielded key between begin and finish is 1oxa's named `[VM-WT]
     unplanned layout` refusal. Unload frees a live copy with no re-record, and a yielded
     copy is never re-drawn. So:
     - begin under L1: retire the picks, and submit the reader barrier;
     - if anything was retired, bump the optional-layout epoch
       (`ggml_sycl_optional_layouts_retired()`, jehw `:17610-17612`) and `lock.unlock()`;
     - finish with L1 released: wait on the barrier, drop the withdrawn mirrors and the freed
       rows;
     - `lock.lock()`. A changed plan here (jehw's relock, master `3d9414c8c` `:18122`) is
       `[CONTEXT-PLAN-BUG]` under L0 (rulings §E.2), never `busy`; the guard's first phase then
       clears this call's pending ranges, `clear_pending({CONTEXT, id}, REGION)`.
   - **A yielded copy is never lazily re-staged.** Dispatch reads the primary's materialized
     layout (P3, "the loaded layout is the answer"), and H7h gates it.
6. **Commit the carve (r2 N-I7; r4 I8, m3).**
   - **Re-fit only inside this call's own pending ranges.** Re-snapshot, and re-run
     `kv_region_fit` with `forced_host` and `own_ranges` = this call's `REGION` ranges
     (never 23mk's holds of the same context, rulings §Z8 I-2), as the only room
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
     controls (§2.10), and clear this call's ranges on that TLSF with
     `clear_pending_locked(tlsf, {CONTEXT, id}, REGION)` (rulings §M12 I-4), inside the same
     section. The handles go into the guard. The Graph-scratch range is not carved: the section
     leaves it recorded under its own term for 23mk's draws (§2.3.3 A1).
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
7. **The MMID workspace, from planned room only (rulings §M7 I-5(b), §M8 I-5, §M9 I-2, I-3).**
   On master (`3d9414c8c`) `ggml_sycl_materialize_published_mmid_workspaces` (`:13024`) has
   **three** call sites, and 7.7 missed the first (r9 I-2):
   - **`load_end` (`:13181`)**, gated by `k_moe_mmid_route_reachable_compiled`. Its comment
     keeps it because "a later load that runs while a context IS live still materializes here"
     (`:13163-13166`). Materialization binds the latest backend context on the device
     (`ggml_sycl_get_backend_context_for_device` returns `contexts.back()`, `:25980-25985`),
     whatever model it belongs to, so model B's pools, loaded while model A has a context, are
     bound to A's queue, outside every transaction;
   - **step 7 (`:18612-18620`)**, which allocates the new plan's pools at runtime, after the
     yield: a runtime refusal that another context's RUNTIME or host allocation can cause,
     mapped to PLAN_REJECTED. At a context's first publish the MMID plan is stable
     (`ggml_sycl_same_mmid_workspace_plan`, `:12973`; checked at `:18595`), so it is skipped
     (`:18613`);
   - **the wrapper, after the inner transaction (`:18978-18990`)**, for the pre-transaction
     bound snapshot, after the fit and the carve. A failure there is a WARN tripwire, and the
     route falls through to the generic MoE path (the "silent performance cliff" its comment
     names, `:18981-18986`).

   Instead:
   - **The pools are CONTEXT scope (rulings §M9 I-3, amending §M8 I-5(b)).** They are
     queue-bound: the registry binds each pool to one backend queue cookie
     (`moe-mmid-workspace.cpp:1001-1014`), admission refuses any other cookie (`:1114`), and the
     registry keys them by model token, plan identity (the plan version) and queue cookie
     (`unified-cache.cpp:16250-16275`). 7.7 made them MODEL scope and said "a second context of
     the same model that auto-ladders to the top rung is covered". It is not: that context's
     own queue fails the cookie check, and its route falls silently to the generic MoE path. So
     each context's pools are its own: a device pool per device the route reaches and one host
     carve, held in **that context's** registry entry, bound to that context's queue, and
     released by their handles at that context's destruction (the release proc). The registry
     re-key that makes this true is designed below ("The registry, re-keyed by context"). MODEL
     scope remains for model-lifetime terms (§2.4.3 "Scopes"), not for MMID pools;
   - **only a reachable route is planned (rulings §M9 I-2; P4).** The fit takes master's
     predicate, `ggml_sycl_moe_mmid_route_reachable(ctx)` (`:11541`; its compile-time half is
     `k_moe_mmid_route_reachable_compiled`, `:11537`, and `k_moe_mmid_direct_route_validated`
     is false, `:11526`), the one master uses at `:18485-18504`, `:18613` and `:18978`. In an
     ordinary build it is false, so no pool slot is planned and no VRAM is carved for a route
     that cannot execute. Where it is true, the fit plans this context's device pool whenever
     the context has none on that device or has a smaller one than the candidate's workspace,
     as a head slot in the context's `REGION` headroom, the shared zone like every head slot
     (§2.3.3; rulings §M32 I-2; 7.14e placed it on the RUNTIME TLSF), and step 6 carves it with
     the other head slots (rulings §V17 I-1);
   - **both are sized at the context's own `n_ubatch` (rulings §V16a; §M26 I-3, §M27):** the
     workspace at the candidate's `n_ubatch` and the model's `n_expert_used`, so the term is on
     the changed-term list of a ladder climb. 7.14e and earlier sized them at the largest rung
     up to `MOE_GPU_UBATCH_MAX` (rulings §M8 I-5(d)), which held a rung the context was not on;
     that is withdrawn (§V16a: an idle rung is a P4 over-plan). A larger need (a ladder climb,
     an explicit `-ub`, or `n_expert_used := n_expert` under the deprecated `llama_set_warmup`)
     is a **planned growth within that context only**: the fit places a larger device pool as a
     head slot, a larger host carve is allocated (below), step 7 materializes into both, and the
     old pool and carve are superseded handles of that same context, dropped after the publish
     (step 8 (d)) once their last event completes. No context's transaction touches another
     context's pools. It is never a refusal while the room exists; when it does not, it is the
     fit's ordinary refusal naming the tenant, at step 2 or (0), before anything is released;
   - **the host carve** is one owner-first reservation made through `unified_allocate_owner`
     (§2.4.3's request with a new cohort, `moe-mmid-host`; contiguous, rulings §ZR5 I-2), held
     in the context's entry beside its host-slot reservations. The pool phase gates skip it
     because the thread holds a `TRANSACTION` token (rulings §M77, §M9a);
   - **where the carve and its growth sit (r9 m-9).** On the full path the host carve, first
     or grown, is allocated before L1 with the context's other host reservations (§2.4.3). On
     the tenant-only path the fit that decides a growth is (0)'s: the grown device pool is one
     of (0)'s placements, held as a pending range and carved at (ii) inside it, and the grown
     host carve is allocated directly after (0), before (i) releases anything and outside L1. A
     failure there is a candidate refusal before anything is released, never a refusal at
     (ii) after (i) has released, the class §M7 closed;
   - **step 7 materializes into those carves and allocates nothing.** An allocation there, or a
     carve too small, is `[CONTEXT-PLAN-BUG]`, never a refusal. The registry's allocator
     callback (`unified-cache.cpp:16201-16233`), which today calls `unified_allocate` for each
     pool, instead hands out slices of the context's carves: the device pool is a slice of
     step 6's head-slot carve, the host pool a slice of the context's held host carve, each a
     `mem_handle` slice whose `shared_ptr` becomes the blob's `owner`;
   - **step 7 takes the owning context explicitly (rulings §M11 I-D).** On master the
     materializer binds each pool's queue through `ggml_sycl_get_backend_context_for_device`
     (`:13054`), which returns `contexts.back()` (`:25980-25985`), and step 7 calls that same
     materializer. So deleting `load_end`'s site does not remove the binding to the latest
     context, as 7.10 claimed (r10 I-D): if C1 re-plans after C2 exists, step 7 in C1's
     transaction binds C1's pools to C2's queue. The materializer becomes
     `ggml_sycl_materialize_context_mmid_workspaces(ggml_backend_sycl_context & ctx, token,
     snapshot, reason)`, called by step 7 with the transaction's own `ctx`, and it never
     resolves a context itself. For a route whose pools span several devices, the pool on
     `owner_device` binds the queue of **the same llama context's** backend on that device,
     found through **the binding authority, `g_execution_backend_bindings`** (`:11780`; its
     records carry `context_id`, `participant_id`, `covered_devices` and `draining`, written
     at `:14826-14866`), never by device alone (§M12 m-13). Under
     `g_execution_backend_binding_mutex` the lookup reads `ctx`'s own record's `context_id`,
     then takes the backend whose record has that `context_id`, is not draining and covers
     `owner_device`. It does not read `ctx->execution_context_id` through
     `g_backend_context_by_device` (`:25977-25985`): that field is a copy written under
     `execution_state_mutex` (`:14841`, `:14873`) and zeroed at `:11877`, a second source of
     the same fact. The binding mutex is a leaf taken under L0 and L1 only, with no lock taken
     inside it. No such backend is `[CONTEXT-PLAN-BUG]`, since the fit planned a pool the
     context cannot bind. `ggml_sycl_get_backend_context_for_device` loses this caller;
   - **step 7's registry entry is the guard's, and non-accepting, until the CAS (rulings §M12
     I-3).** On master an inserted entry is accepting when `!exact_plan || !has_active`
     (`moe-mmid-workspace.cpp:1059-1071`), so a context's first entry is accepting the moment
     step 7 inserts it, and the retires at `:18633-18637` and `:18648-18651` run only under
     `!stable_mmid`, which is false at a first publish (`:18595`). So a first publish that
     fails after step 7 (step 8 (a)'s refusal, an exception from
     `ggml_sycl_prepare_plan_publication_locked` at `:18624`, or a lost CAS) would leave an
     accepting entry whose pools are slices of a carve the guard has dropped:
     `needs_materialize` then answers false on the retry, the fit plans no pool, and the pools
     live outside every plan until the context is destroyed (r11 I-3, P4). Instead, step 7
     always inserts its entry **non-accepting**, first publishes included, and records it in
     the guard (`this call materialized entry E`). The CAS commit flips E to accepting, in the
     registry's `replace_plan` step, and retires the entry E supersedes. The guard's first
     phase retires E on every other exit: step 8 (a)'s refusal, an exception, and the lost
     CAS. Retiring E drops the registry's slice references, so the carve's block is freed when
     the guard drops its carve handles. The two `!stable_mmid` conditions at `:18633` and
     `:18648` become "this call materialized";
   - **the trigger is "this context has no pools, or needs larger ones" (rulings §M11 I-E(1)).**
     Master's step 7 runs under `if (!stable_mmid && mmid_route_reachable && !materialize(...))`
     (`:18613`), where `stable_mmid` compares the model plan's MMID workspaces before and after
     (`ggml_sycl_same_mmid_workspace_plan`, `:18595`). At a first publish that is "stable", so
     master skips it and relies on `load_end` having materialized. With that site deleted, the
     first publish of a route-testing build would materialize nothing, and the post-transaction
     check below would report `[CONTEXT-PLAN-BUG]` on every first publish. The trigger becomes
     `mmid_route_reachable && ggml_sycl_context_mmid_needs_materialize(ctx, next->plan)`: true
     when the registry holds no accepting entry for `(model token, ctx.execution_context_id,
     submit device)`, or when the candidate's workspace exceeds the entry's recorded geometry
     on any device the route reaches (the in-context growth above). `stable_mmid` keeps its
     other use, the version reuse at `:18596`, unchanged;
   - **the registry, re-keyed by context (rulings §M9 I-3, §M11 I-E(2)).** Today an entry
     (`registry_context`, `moe-mmid-workspace.cpp:593-604`) is keyed by the model token, the
     submit device and the plan (its version and exact snapshot pointer). Each entry gains
     `context_id` (the backend context's `execution_context_id`), and the key becomes `(model
     token, context_id, submit device)`. The plan leaves the key: an entry records the
     geometry of the pools it holds, and admission already checks the request against the
     pool's own geometry (`out.lease.admitted_geometry_ = pool->geometry`, `:1134`). Every
     site, with its new argument:
     - **materialize** (`:976-1075`, through `unified_cache_materialize_moe_mmid_workspaces`,
       `unified-cache.cpp:16250-16262`) takes `context_id`. Its dedup loop (`:1061-1073`)
       compares entries with the same `(token, context_id, submit device)` only. An existing
       accepting entry with the same geometry answers `ALREADY_PUBLISHED`; otherwise the
       candidate is inserted as this context's non-accepting entry E (I-3 above), whether or
       not an accepting entry exists, and its CAS makes it accepting. So a second context of
       the same model, even with the same plan version, gets its own entry, accepting once its
       own CAS commits, not the "prepared replacement" of the first context's entry that
       master gives it (r10 I-E(2));
     - **the swap at the CAS**: `lifecycle_replace_placement_plan` calls the registry's
       `replace_plan` (`unified-cache.cpp:316-322`) with the model's expected and replacement
       snapshots. It takes the committing context's `context_id` too and touches only that
       context's entries: its prepared replacement becomes accepting and its old entry is
       retired (into `retired_contexts` while a lease is outstanding, as today). Other
       contexts' entries are untouched by another context's publish, since the plan is no
       longer their key. The `stable_workspace` in-place branch goes: a context whose geometry
       did not grow has no prepared entry and nothing to swap;
     - **acquire** (`:1081-1154`) and **the cookie check** (`:1114`) take `context_id` in
       place of `plan_identity`, and match the accepting entry of `(token, context_id, submit
       device)`, then the pool on `owner_device`. The cookie check is unchanged: the pool's
       cookie is the owning context's queue cookie, so C1's dispatch matches C1's pools and
       C2's matches C2's;
     - **exact_queue** (`:1484-1498`, the dispatch at `ggml-sycl.cpp:75071`) takes
       `ctx.execution_context_id` in place of the plan snapshot pointer; **admit** (`:1337`,
       the dispatch at `:75198`) checks that the authority's entry is still the accepting one
       for its `(token, context_id, submit device)`, where it compares the plan identity today;
     - **retire** (`:1652-1690`) takes `(token, context_id)`, with `context_id == 0` meaning
       every context of the token. The call sites:
       - unload (`:12321`) retires every entry of the token. Contexts are gone by then (a live
         context pins its model), so an entry still present is one whose context did not
         retire it at destruction: `[CONTEXT-PLAN-BUG]` naming the ownerless entry, the
         CLAUDE.md "owner is gone" leak class, and it is retired;
       - `load_end`'s cleanup and failure paths (`:13209`, `:13245`) are deleted: with
         `load_end`'s materialization gone, no entry of a loading token can exist, and H7am
         asserts none does at `load_end`;
       - the lost CAS (`:18635`), like every other non-publish exit after step 7, retires
         **this call's entry E** through the guard (above), not every entry at the new
         version;
       - the published CAS (`:18650`) retires **the entry E superseded**, not every entry at
         the old version;
       - **context destruction** (new): the release proc, under its `LIFECYCLE` token, first
         runs `recover_quarantined(token, context_id, wait = false)` for this context, then
         retires the entries of `(token, context_id)` (below);
     - **recover** (`recover_quarantined`, `:1572-1650`) takes `context_id` in place of
       `plan_identity`: the dispatch's recovery (`:75299`, in `ggml_sycl_mul_mat_id`) passes
       its `ctx.execution_context_id`, and `ggml_sycl_recover_exact_mmid_after_drain`
       (`:84283`, the call at `:84289`) passes its `ctx`'s;
   - **release at context destruction goes through the handle (rulings §M11 I-E(3); P2).** A
     pool's `device_pool.owner` and `host_pool.owner` are `shared_ptr<mem_handle>`s that the
     slices (`retained_subrange`, `:654-664`), the authorities and every outstanding lease share
     (`unified-cache.cpp:16229-16235`). So the memory is released when the last of them drops,
     by the `mem_handle` destructor, never by the retire itself. The release proc's order is:
     the context's last event has completed (rulings §Z6x 3), so its leases are terminal;
     `recover_quarantined(wait = false)` clears any quarantined slot; the retire removes the
     registry's references; then the region entry drops the carve handles the pools are slices
     of. **The recover does not wait (§M12 m-6):** with `wait = true`, `recover_quarantined`
     runs `wait_and_confirm` under the registry's `state_->mutex` and every pool mutex
     (`moe-mmid-workspace.cpp:1575-1618`), which would block every other context's MMID acquire
     on that registry for the wait. After the last-event drain every proof is already complete,
     so `wait = false` recovers every slot; a bundle that is not ready is `[CONTEXT-PLAN-BUG]`
     (the drain missed a submission), naming the slot. The retire is conditional in one way:
     `retire` skips an entry with a quarantined slot (`:1672-1676`), so such an entry, the bug
     just reported, stays in the registry, and the unload's retire-all (`:12321`) reports it
     again as ownerless. A reference still alive after that (a lease or slice someone kept) is a
     holder the release proc's holder check reports as `[CONTEXT-PLAN-BUG]` naming the MMID
     registry, never something it frees. Without the context-destroy retire, a destroyed
     context's pools would stay in the registry with no owner (r10 I-E(3));
   - **the other two sites are deleted.** `load_end` materializes nothing (`:13181-13191` go).
     The wrapper's post-transaction materialization (`:18978-18990`) goes too. After the inner
     transaction, where the route is reachable, the registry must answer `ALREADY_PUBLISHED`
     for this context on every device the route reaches; anything else is
     `[CONTEXT-PLAN-BUG]`. H7am enumerates the call sites and passes only when step 7 is the
     one left. **The source gates that name the deleted or changed sites are re-anchored in
     the same change (r10 m-3):** `tests/test-sycl-mmid-deferral-contract.py` (ctest
     `sycl-mmid-deferral-contract`, `ggml/src/ggml-sycl/CMakeLists.txt:5700`; its
     `LOAD_END_CALL` and `TRIPWIRE_SITE` `"context-bind"` name the two deleted sites),
     `tests/test-sycl-pp-moe-ring-kv-zone-source.py:199-207` (the `!stable_mmid` trigger), and
     `tests/test-sycl-moe-resolved-batch-source.py:476-488` (the materializer's
     `get_backend_context_for_device` binding).

   Any refusal still rolls back through the guard, which leaves the old ring, the old slots and
   the context's old pools untouched.
8. **Publish, then release what the new plan superseded (r4 I5; r5 I-A, m-a).**
   - **(a) No ring check (rulings §M32 I-1).** 7.14e checked the device ring's generation and
     `RELEASING` mark here, before the CAS, and recorded this context's tentative
     contribution. The ring's rows are now this context's own slots, placed by step 2's fit
     and carved at step 6 like every head slot, so there is nothing another transaction could
     have changed, and the step is gone.
   - **(b) The publication CAS** (jehw `:18386`), as today. A lost CAS rolls back through the
     guard.
   - **(c) Commit, which has no refusing step** (nothing in it returns `busy` or a plan error):
     - under `kv_region_mutex_`: insert the new entry, holding the extent handles, this
       context's slot handles (its tenant slots and its ring rows, one published slot table)
       and its host-slot reservations (a move of the pre-L1 handles), and the tenant key; on
       the tenant-only path, install the new table in the existing entry, with the device
       slots (c) kept and (ii) carved, the unchanged host-slot reservations (rulings §D15) and
       the new key; unlock. The executor finds the context's ring rows through that table, by
       the backend context (§2.7), and reads no device global. They were reused in place or
       carved at step 6, so the install allocates nothing;
     - still under L1, if the fit reported a KV demotion that releasing lease-vetoed copies
       would have covered: log the WARN naming those copies, their bytes and the demoted
       layers, and, once llama.cpp-423j has landed, issue its retire request for them, once
       (§2.9). This is the only site, and it runs once per committed transaction.
   - **(d) The superseded handles drop after L1 is released**, in the guard's second phase
     (r5 m-a). Revision 6's first draft dropped them "with no lock held" while the transaction
     still held L1, which contradicted P2 and H7t. A claim still retained by an in-flight event
     keeps its block until the event completes (P2); a superseded ring slot's last-generation
     claim retention goes to `retain_handles_until_event` first (§2.7).

**The tenant-only path (a matched key; zhcn's protocol, §3.1 of its design).** A
same-key republish, which is how the auto-ubatch ladder, the setters, encode and a republish
after `memory_update` arrive, re-plans only this context's head slots, its ring rows among
them. It never re-fits KV and never yields.
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
  tier; r7 m-11, m-1; rulings §M76.1, §M8 I-4).** A candidate is **covered** when all of these
  hold:
  - every candidate tenant slot fits the published slot at the same `(cohort, index, device)`
    (`slot_bytes ≤ cap`), host slots included;
  - its ring rows (§2.7) fit the context's published ring slots (`rows ≤ cap`). The rows are
    not part of the tenant key (the digest covers the tenant section only, §2.4.4), so a
    candidate whose tenant slots are all covered but whose rows grow (a larger `n_ubatch`) is
    **not** covered: it takes the growth path below (r7 m-11);
  - **the context's table is published and backed (rulings §M8 I-4):** the published tenant
    key is set, and the published slot table, ring rows included, is installed. After a
    refused (ii) the key was cleared at (c) and the table taken ("After a refused (ii)",
    below). Revision 7.6's rule read only caps and the ring contribution, so zhcn's ladder
    revert to the previous `n_ubatch` could answer COVERED there, carve nothing, and leave the
    next PP MoE claim to find no rows: a `[CONTEXT-PLAN-BUG]` and −2 on the server. A cleared
    key or a taken table now answers GROWTH, and the revert re-carves.

  A covered candidate takes **no L0 and makes no registry change** (zhcn §3.1 step 3; r7 m-1):
  no step (i), no fit, no allocation and no release. The published table, its caps, the
  published tenant key, any published index the candidate no longer uses, and ring rows larger
  than the candidate needs all stay as they are, as held planned room. Claims keep reading the
  same table, and a claim's
  `slot_bytes ≤ cap` is what makes the smaller need safe. Because the key is not re-recorded, a
  repeat of the same candidate takes the covered check again. zhcn rev 5 states the rule for its
  tenants; this design applies it to the host tier too (r6 m-10), so a covered republish neither
  allocates host bytes nor refuses. Only a candidate with a slot that outgrows its published
  cap, a new index, growing ring rows, a cleared key or a taken table takes the path
  below. **EQUAL and COVERED need no execution-binding refresh (r8 m-9):** they skip the
  wrapper's publish (`:18955`) and its `attach_root` (`:18933-18955`), and correctly, because
  neither changes the model, the plan identity, the backend set or any slot, which is all the
  binding records.

  **Why COVERED is safe without L0 (rulings §M76.1 condition (a)).** Everything a COVERED
  answer relies on is held by an owner, not read as free room, and nothing a concurrent L0
  holder does can shrink it while this context lives (zhcn 5.4 row 26 carries the same
  statement):
  - this context's CONTEXT slots, device and host (§2.4.3's reservations), are held by its own
    registry entry. That entry's only writers are this context's growth transaction, on its
    owner thread (the thread asking the query), and its teardown release proc, keyed by its
    never-reused ContextId. Other contexts' transactions write their own entries; a load, an
    unload, an activation and a quarantine restore write the publication and their own
    entries; and no L0 holder may free a live handle (the `mem_handle` rule; weight reclaim
    touches weights only);
  - the ring rows are this context's own slots in the same entry (rulings §M32 I-1), so the
    rule above covers them: no other context's transaction, no release proc but this
    context's, and no load touches them (llama.cpp-r7fz). Growing rows are GROWTH, never
    COVERED (r7 m-11);
  - the MMID workspaces are held by this context's own registry entry (CONTEXT scope; rulings
    §M9 I-3), bound to its own queue, and released only by this context's own growth
    transaction (a supersession within the context) or its release proc.

  So the answer is a function of state that only this context's own thread mutates. A stale read
  cannot become unplanned use, and the caller need not re-check under a transaction. H9 runs a
  covered read against a parked L0 holder that unloads another model, restores a quarantined
  one and loads a third: the slot handles, the ring rows and the host
  reservations are unchanged, and the answer is the same. zhcn's H9 has the two arms on its
  side: another model unloads between COVERED and ALLOC, and another context commits a
  growth; its RED is zhcn 5.3's key-record write.
- **The coverage query, and L0 on the tenant-only path (rulings §E.2, §L0R, §M76.1; zhcn rev
  5.1; r7 I-2, m-1; r8 m-1).** llama asks first, through a read-only backend query,
  `ggml_backend_sycl_tenant_coverage(ctx, candidate)`, which answers EQUAL, COVERED or GROWTH.
  It reads in one section, under the `kv_region_mutex_` leaf: the context's own entry, that is
  the tenant key, the caps (the ring rows' among them) and whether the table is installed.
  7.14e read a copy of the ring contribution here and then, in a second section under the ring
  record's lock, whether the device ring held slots; the entry now answers both. It
  neither publishes nor prepares a live update, so it is not an L0 entry point (§2.4.2 "The
  re-plan transaction mutex"), and EQUAL and COVERED end there with no L0. On GROWTH, llama
  opens `ggml_backend_sycl_replan_scope`, a `TRANSACTION` token that takes L0 **before (s)**,
  and holds it to the guard's second phase. The backend transaction wrapper it then calls takes
  the same token at its top as a nested hold (rulings §L0R). No path writes the entry without
  L0: the covered path does not write it at all. zhcn 5.4 decides coverage through this
  same read-only query and names 5.3's key-record write as its RED (row 26), so the r8 m-1
  difference is closed (§6.11; r9 addendum m-12).
- **The scope closes before ALLOC's `graph_reserve` (rulings §M9a, §M11 m-5; r9 addendum
  I-5(b); zhcn 5.7).** The growth runs inside `sched_reserve_impl` (zhcn f6218f3): MEASURE,
  the coverage growth, then the replan scope around steps 3-6, which llama closes once the
  guard's second phase has run; only then does `sched_reserve_impl` reach gallocr's ALLOC,
  whose `graph_reserve` rebuilds what (b) destroyed. That ALLOC (`graph_reserve`)
  allocates the SYCL compute buffers, and its failure path is a host-pinned retry
  (`ggml_backend_sycl_buffer_type_alloc_buffer`; `llama-context.cpp:1587-1589` at
  `3d9414c8c`). That growth is unplanned inference-time work on the decoding thread, where the
  phase is genuinely TG. So it must never run under a `TRANSACTION` token, whose holder the
  pool phase gates exempt (§2.4.3). The same holds for the wrapper on the constructor and
  ladder paths: its token is an RAII local, released when it returns, before llama's
  `sched_reserve`. **The shared invariant, worded identically in zhcn's design:** "no
  TRANSACTION token is held at gallocr ALLOC or at alloc_buffer's host fallback". Two
  always-compiled witness checks make it structural (§2.4.2 "The token's kind"), and a third
  keeps graph compute out of every scope:
  - `ggml_backend_sycl_buffer_type_alloc_buffer`, at its entry, for any buffer allocated
    outside a bound load (the buffers gallocr's ALLOC requests), checks
    `!ggml_sycl_replan_token_held(TRANSACTION)`: that is where gallocr's ALLOC reaches this
    backend;
  - its host-pinned fallback checks `!ggml_sycl_replan_token_held(TRANSACTION)`;
  - graph compute (`ggml_backend_sycl_graph_compute`) checks `!ggml_sycl_replan_token_held()`
    (any kind): graph compute never takes L0 and never runs inside a scope.

  zhcn's in-`sched_reserve` fixpoint scope, also `TRANSACTION`, closes before the same ALLOC
  (rulings §Z7 I-3). The FA recheck that `sched_reserve` reaches through `resolve_fused_ops`
  (`:1298`) takes its own `LIFECYCLE` token as an outermost hold.
- **The guard is declared before (s)**, so steps (s), (0), (i) and (ii) all run inside its
  lifetime (r5 I-A(d)).
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

  zhcn 5.2 and 5.3 carry every one of these, the cache's `get_bcs_queue()`, the TP worker's
  queue and the PP pipeline copy queue included (zhcn :49, :581-584). Revision 7.6 said zhcn's
  list lacked those three; that was stale (r8 m-2). **The invariant (rulings §Z52 "Order"):**
  MEASURE and the coverage query are read-only and run before (s); (s) runs only on the growth
  path, after L0; and it completes before (c)'s occupancy check and move-out and before any reap
  or release.

  The list is zhcn's gate 30 (the queue census). The between-graph scatter lists hold compute
  slices, and none needs a queue wait of its own, because one flush drains all four (zhcn C2t;
  zhcn 5.4 row 22): `ggml_sycl_cpu_tg_flush_pending()` (`:23911-23923`) covers
  `g_pending_scatter` (`:21473`), `g_pending_cpu_pipeline` (`:23216`),
  `g_pending_secondary_scatter` (`:23394`) and `g_pipeline_scatter` (`:23426`), not only the
  secondary scatter. llama's `synchronize()` reaches it through `ggml_backend_sycl_synchronize`
  (`:84304`) before (s)'s waits, and it also runs at every `graph_compute` exit that is not
  recording (`:94869-94871`). A recording is exempt from the exit flush: its scatter state goes
  into the recording sink, or the record is refused. A queue added later must join the list. If
  one is missed, (d)'s backstop makes the miss a loud `[CONTEXT-PLAN-BUG]`, never a free under
  queued work. **Nothing between (s) and (d) publishes a retention whose event postdates (s) (r6
  m-2):** (a)'s own-context clear submits no device work (its per-context input-staging reset,
  `graph_input_staging_clear`, ignores its queue: `common.hpp:6457-6460` at `3d9414c8c`); A4's
  gallocr free takes the queue's last event and submits no barrier (§2.3.2); and the record-mode
  vacate's marker is a claim-state event that retains nothing (§2.3.2).
- **(0) Probe, then hold its placements (rulings §M7 I-3(b), I-5(a)).** zhcn's measure pass has
  already sized the candidate's tenants. A fit places the candidate's head slots on the live
  geometry, with the context's current slots, its ring rows included, counted free **by
  arithmetic only** (§2.7). Their extents are copied out under the `kv_region_mutex_` leaf,
  which is then released; (0) takes no copy of a handle ("no side effects", step 4, "Probe
  mode"). So a growth that fits only in the old rows' room passes (0), and nothing (0) did
  keeps the old rows allocated into (ii). **The probe checks the host tier by
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
- **Host slots on this path (rulings §D15, §ZR5 I-1, I-2; r6 m-10; r7 I-6).** The context's host
  slots are its held per-index reservations, each at that index's maximum over every rung,
  allocated at the first publish (§2.4.3), so every candidate the plan admits is covered on the
  host tier: its host slots are
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
    - clear the tenant key (the coverage query's input); unlock.

    The context's ring rows are in the table, so a growing ring slot moves into the batch like
    any growing device slot; its last-generation claim retention moves separately ("The ring
    slot's retention", below). The reap's owner list (d) is the batch's device handles, never a
    host reservation, which nobody else can reach once the table is taken; they are destroyed
    at the end of (d). (e) checks only the moved-out handles. **This is
    the agreed text (zhcn 5.4 §3.1 step 5(c)):** zhcn 5.3 checked and opened the table at (e);
    5.4 checks and opens it at (c), in the words above, so the r8 m-2 difference is closed.
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
    slice between graphs. Five more do, at master `3d9414c8c`:
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
      the ids tensor's compute slice (`common.hpp:5713-5736`). They are cleared only by
      `ggml_sycl_moe_ids_cache_new_graph` (`:19979`, whose clears are `:19985` and `:19998`),
      which runs at graph start (`:92321`, in `ggml_backend_sycl_graph_compute_impl`) and at
      model teardown (`ggml_sycl_release_model_slot_resources`, `:12199`), and the admission map
      also by the weight preload's `MID_LOAD_REPLAN` clear (`:33767`). So between graphs they
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

    - **`g_moe_down_shadow`** (`:21153`, r8 m-3), a `thread_local` map whose key
      (`moe_down_shadow_key`, `:21115-21123`) holds an owning `mem_handle` of the source tensor
      but compares it only by `handle_identity` and `stable_identity_equal` (`:21130-21131`),
      and is cleared only at graph start (`:21313`). It is debug-gated
      (`GGML_SYCL_MOE_DOWN_SHADOW`, `:21169-21175`), so it is reachable only under that
      variable, but it is a holder there, and it takes the same key-type fix: the key's handle
      becomes a `mem_handle_identity`;
    - **the four C2t scatter lists** (zhcn 5.4 row 27): `g_pending_scatter` (`:21473`),
      `g_pending_cpu_pipeline` (`:23216`), `g_pending_secondary_scatter` (`:23394`) and
      `g_pipeline_scatter` (`:23426`), all `thread_local`. Each scatter entry's `dst_handle`
      owns the `MUL_MAT_ID` dst compute slice (`:21417`, `:23174`, `:23357`; the pipeline slot
      wraps a `pending_secondary_scatter`, `:23419`). Their fix is the drain, not a key type:
      `ggml_sycl_cpu_tg_flush_pending()` empties all four at every non-recording
      `graph_compute` exit and in `synchronize`, and a recording keeps them in its sink or is
      refused ((s) above).

    Each is either a non-owning identity key or holds no owning handle of a tenant slice past
    `graph_compute` exit. H7ag names all nine, each with a mutation witness. Not holders (zhcn
    5.4 row 27): `g_moe_down_sum_shadow_entries` (its own allocation), `g_mul_mat_exc_ctx` (raw
    pointers), `local_extra` (scoped), `retained_ids` (a host copy) and the `CpuExpertPool`
    tasks (a `weight_lease` only).
  - **(d) The reap, with no L1-L5 lock held (lead ruling "B"; r6 I-1, I-2, m-3, m-5; zhcn rev
    5).** zhcn's mem-handle call, shared with llama.cpp-uwlx's yield (jehw), is
    `retained_reap_result release_retained_referencing(const retained_reap_request &)`. The
    request is `{const mem_handle * owners; size_t n_owners; retained_reap_precondition pre;
    const char * reason; bool * owner_pending}` (rulings §R). Its owners are the batch's device
    handles (the old slots the candidate does not reuse, a growing ring slot among them); a
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
    abort under `GGML_SYCL_STRICT_LEASES=1` (§G1). The BUG line carries the count of records the
    backstop waited, and a `GGML_SYCL_PRIVATE_TESTING` counter accumulates it; §R's five-field
    result is unchanged (rulings §D16; r7 m-6). It is not re-queued and reported pending,
    because that would turn a missed queue into a silent pending. A `graph_unwaitable` match
    carries no event; it is dropped on the precondition that (s) and (a) destroyed every
    executable graph that could use it (zhcn rev 5). Retained handles are otherwise released
    only by the background worker (`mem-handle.cpp:2039-2052`), which is why the step exists.

    **The ring slot's moved-out retention is dropped here too (rulings §M7 I-3(a)).** The
    last-generation `retained_owners[slot]` moved out with a growing ring slot (below) are
    slices of the old slot and share its control (`mem-handle.hpp:70-72`, `:84`), so each would
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
    - the ring slots among them the same, `use_count() == 1`. 7.14e bounded an old device-ring
      slot by `1 + P[slot]`, the guard pins snapshotted at the ring's move-out (r7 m-3); the
      pins went with the device ring (rulings §M32 I-1), so the rows take the tenants' check.

    Anything else is `[CONTEXT-PLAN-BUG]`, aborting under `GGML_SYCL_STRICT_LEASES=1` (§G1). The
    BUG line is followed by a holder scan over the census containers (zhcn rev 5 row 13), so it
    names who holds the reference. Without STRICT (§G1), the call logs, drops the batch and
    continues (zhcn rev 5 row 17): the extra holder keeps its block allocated through its own
    reference, and (ii)'s live-TLSF fit counts that block as allocated, so the worst case is a
    refusal, never an overlap. After (d) it means a reference held **outside the retained
    store**, which the protocol does not allow: a tenant slice may be retained only by a claim,
    by the retained store, by a per-context graph container that (a) clears (rulings §B, §B.2;
    H7ag; r6 I-3). Two kinds of holder are design errors to fix in their producers, never to
    exempt:
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
  - **The ring rows (rulings §M32 I-1, §M7 I-3; r6 I-4).** The context's ring rows are
    CONTEXT slots in its own slot table, so rows that must grow move out at (c) with the other
    growing slots. Separately, (c) moves out each row's slot-state retention of its last
    generation (`retained_owners[slot]` with `done_events[slot]`; master `3d9414c8c`
    `ggml-sycl.cpp:1526`, set at the record, `:1722`, re-keyed to the context's slot by §2.7).
    That retention is not in the retained store, so the reap cannot see it, and without the
    move it keeps every row PP MoE has used allocated until that index is next claimed. (d)
    drops it through the backstop before (e); it never sits in the batch across (e). A claimed
    row is `[CONTEXT-PLAN-BUG]`, as for any tenant.

    **The slot table is the rows' only storage (r8 m-12).** Step 8 (c)'s install writes the
    rows into the context's slot table, and the executor claims a row by taking a slice of the
    table's handle at claim time, which travels in `retained_owners` (§2.7). No cache-side copy
    of a row handle is kept beside the table. So the move-out takes every holder, and (e)'s
    `use_count() == 1` is exact. H4's ring arm asserts it on every moved-out row, with a
    witness that installs a second, cache-side holder. (0) made the growth possible without
    any of this: it counted the old rows free by arithmetic and took no copy (rulings §M7
    I-3(b)).
    - Revision 7.1 exempted the old ring slots from (e) outright. That hid the slot-state
      retention: the release freed nothing, (ii) refused the growth without demoting and
      without saying why, and a stray holder of an old ring slot went unreported (r6 I-4).
    - Rows that do not grow are reused in place (§2.7), so nothing is released. No other
      context's rows are touched. 7.14e's device ring record, with its sole-contributor
      release, RELEASING mark, pins and other-contributor case, went with the device ring
      (rulings §M32 I-1).
  - **After a refused (ii) (r6 m-9).** If its rows grew, the context holds no ring rows (they
    left at (c)), and it holds only the device tenant slots kept for reuse (the others left at
    (e)). Its next PP MoE dispatch finds no row in its table: that claim is
    `[CONTEXT-PLAN-BUG]` with an error status, never a silent skip, and so is any tenant claim.
    zhcn's ladder revert re-carves both from empty. The tenant key was cleared at (c), so the
    revert does not match the equal-key no-op, which would return OK with nothing carved, and
    the coverage query answers GROWTH for it (the key is cleared; rulings §M8 I-4), so it does
    not take the covered path either. It takes this path, and its (ii) carves the previous
    candidate's slots, the ring rows among them. If the revert is refused too, the decode fails
    with that refusal, naming the tenant bytes.
- **(ii) Under L1.** Steps 2-4 run for the head slots only, with the region fixed. The fit is
  restricted to this call's own pending ranges, recorded at (0), exactly as step 6's re-fit is
  (`own_ranges`): nothing else can have entered them, because every allocation on those TLSFs
  honours them and no other re-plan runs under L0, and (f) freed what they overlapped. It reads
  the **live** TLSF, so a block whose release has not completed counts as allocated by
  construction (the belt; zhcn's row 4a): the worst case is a refusal, never an overlap. On a
  correct run (ii) therefore places exactly what (0) placed, step 6 carves it, and steps 7-8
  run; step 5 records nothing new and nothing yields. If (ii) does not fit, some old block is
  still allocated, which means a holder that (e) already reported as `[CONTEXT-PLAN-BUG]` (an
  abort under STRICT, §G1). Without STRICT (§G1) the candidate is then refused with the
  tenants-alone message, **with no demotion**, and the ladder moves on; a setter or encode
  surfaces the refusal as a decode error naming the tenant bytes. It is not a runtime race: no
  allocation that skips L0 can take the room (rulings §M7 I-5; r7 I-5), and zhcn's ladder revert
  republishes the previous candidate.
- **23mk's A step on this path, after L4 (rulings §Z15; 23mk 4.7 `cfb99d924`, adopted).** The
  auto-ubatch ladder, where A grows, arrives as a same-key republish and takes this path, which
  has no step 5 and no yield, so the full path's slot (between step 5's recording and the yield)
  does not exist here. A's three parts sit as follows:
  - **A's move-out and reap run after (s)'s synchronize**, with no L1 held, so the reap runs in
    COMPLETE mode;
  - **A's fit-and-hold is its head slot in (0)'s fit, before any release the path performs**,
    that is, before (i): (0)'s fit places A in the context's `REGION` headroom with the other
    head slots and records it as one of (0)'s `{CONTEXT, id}` / `REGION` ranges, and never
    reads RUNTIME's free bytes (rulings §M27 (1)), so a predictable A refusal is a (0) refusal
    and nothing has been released when it is made;
  - **A's carve runs after the publish.**
  23mk mirrors this bullet; the tenant-only path's other steps are unchanged.
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
it runs:
1. under `kv_region_mutex_`: move every `(c, *)` entry out into a local batch; unlock. The
   entries carry `c`'s ring rows, which are CONTEXT slots in its table (§2.7);
2. under `c`'s slot-state lock (L5): move out each ring row's slot-state retention of its last
   generation (`retained_owners[slot]` with `done_events[slot]`; r6 I-4); unlock;
3. with no lock held: hand each retention to `retain_handles_until_event(done_events[slot])`
   (§2.7, r6 I-6), so each old row is freed after its last event rather than held until a
   later claim reuses the index; then drop the batch. The last `mem_handle` reference (the
   registry's, the KV buffers', a transaction guard's copy, or a claim still retained by an
   event, whichever goes last) frees each extent and slot through `zone_free`, which runs the
   §2.3.2 free rule.

The proc takes no L1. 7.14e took it, with the ring record's lock, the contribution removal,
the RELEASING mark and the generation bumps, for the device ring record, and all of that went
with the device ring (rulings §M32 I-1): no other context reads `c`'s rows. The proc is
`noexcept`, and a lock failure aborts, as for the guard (r4 m14). Dropping the registry's
references is safe whatever
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
- **llama.cpp-jzvq:** the fattn thread-local device workspaces (`fattn.cpp:747` and its callers,
  the `:1800` XMX split workspace) and the **MXFP4 MoE token-generation caches**
  (`g_mxfp4_moe_tg_reuse`, `mmvq.cpp:1744`, and `g_mxfp4_stored_gemm_ksplit_scratch`,
  `mxfp4-stored-gemm.cpp:242`; ticket comment c-jv0r). The latter are on the **default GPT-OSS
  decode path** (four raw `role=6` EXT-ALLOC lines on the B50 master baseline), so without jzvq
  they would be unplanned claims on every GPT-OSS decode, a STRICT (§G1) abort. jzvq moves them
  to per-(ContextId, device) ownership with an exact demand function, and **closes before moua
  L4** (r4 I10(c)).
- **llama.cpp-23mk core** (lands after jehw): zones and `forbid_vram_zone_spill` on every
  out-of-arena row, the trace, the dead-code deletions (`scratch_pool`,
  `unified_cache_allocate_moe_q8_1_graph_scratch`), the oneDNN weights scratch reserved once at
  its max in the ONEDNN tail. The opt-in persistent-TG buffers (`unified-kernel.cpp:4890`) are
  **refused under an arena**, with one WARN (lead ruling on 23mk's Q5), so they need no record.
- **The persistent packed-K sidecar** is not a record: it is each layer's companion slot in the
  KV region (§2.4.1), sized by 23mk's function over `kv_layer_cells` and `n_stream`.
- **moua:** the recurrent state, one CONTEXT-scope slot per device RS buffer (§2.4.4, r4 I9).
- **llama.cpp-u1bb** (in master): the ring rows, CONTEXT scope, one activation row and one
  output row per context at that context's reservation `n_ubatch`, sized by the local-count
  value function (§2.7; rulings §M32 I-1, C-1(b)); moua records them as head slots, not zhcn.
  u1bb's weight slot is not context-side: it is `moe_onednn`, a D term (§2.4.5).
- A cohort routed into a tail zone (ONEDNN/RUNTIME/SCRATCH are separate TLSFs, laid out at
  load) is outside this interface. 7.14e excepted the ring's RUNTIME half; there is none
  (rulings §M32 I-1).

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
  on the pool's phase gate).** The one host allocation happens at the **first** publish, at
  context creation, and never at a republish. `grow_zone` and `grow_into` WARN, and at
  `GGML_SYCL_HOST_ALLOC_PHASE_GATE` ≥ 2 assert, when the pool grows while
  `offload_stats_phase()` reads PP or TG (`pinned-pool.cpp:613-640`, `:865-882` at
  `3d9414c8c`). **That phase does not describe a context, so this design keys the gate on
  provenance (checked for 7.7a at the lead's request):**
  - the phase is one process-global atomic, `g_offload_phase` (`unified-cache.cpp:2545-2547`),
    set to PP or TG at every graph-compute entry (`ggml-sycl.cpp:105145`) and reset only by the
    load hooks (`:12450`, `:12754`, UNKNOWN; `:12415`, LOAD) and around the warmup pass
    (`:106880-106882`), **never at graph exit**. Once any context in the process has computed a
    graph, every later host growth reads as inference-time growth;
  - so the gate fires on planned work: the first publish of any context created after another
    has decoded (`llama-bench` builds a context per test; a server can build several), and the
    MMID host carve's planned growth at a republish past `MOE_GPU_UBATCH_MAX` (§2.4.2 step 7).
    At the default gate that is a false WARN; at ≥ 2 it asserts on work the plan sized;
  - **the rule (rulings §M77, narrowed by §M9a):** both gates skip when
    `ggml_sycl_replan_token_held(TRANSACTION)` is true on the calling thread, and fire as today
    otherwise. Only a context's own planned transaction holds a `TRANSACTION` token (§2.4.2
    "The token's kind"): the first publish's reservations, the constructor's included, taken
    before L1 inside the full transaction, and the MMID carve's growth, under L0. zhcn's
    second-context carve (rulings §Z6a) runs under the same kind. Everything else stays gated:
    an expert-cache host fill, host weight staging, any allocation on a decoding thread, and any
    growth under a `LOAD` or `LIFECYCLE` token. The state is thread-scoped and the gate runs on
    the allocating thread, so a decode on another thread gets no exemption. The pool calls the
    accessor directly: it is declared in `unified-cache.hpp`, which `pinned-pool.cpp` already
    reaches (§2.4.2). Resetting `g_offload_phase` around the transaction instead is rejected:
    the phase is process-global, so the reset would also exempt a concurrent decode's fill;
  - **a load entry is NOT exempt (rulings §M9a; r9 addendum I-5(a)).** 7.7a exempted any token
    holder, and 7.9 said so for the load entries (r9 F3). That was wrong: `load_end` holds the
    token while the preload does exactly the growth §M77 keeps gated, the expert host fills
    through the pinned pool (`ggml-sycl.cpp:34169-34177`: role `WEIGHT`, `must_host_pinned`,
    `use_pinned_pool`), the host-zone configuration (`:33888`,
    `ggml_sycl_configure_host_zones_for_plan`) and the VRAM-pressure fallbacks (`:33530`). A
    held-any exemption would silence an admit-then-spill of B's load. So load-time growth
    stays gated. Its false positive while another model is in TG (the sticky phase reads TG
    during B's load) is **llama.cpp-dhpw**'s to fix; broadening the exemption is not the fix;
  - **`sched_reserve`'s host fallback is not exempt either.** llama's replan scope closes
    before ALLOC's `graph_reserve` (§2.4.2 "The scope closes"), so the
    host-pinned retry of `graph_reserve` (`llama-context.cpp:1587-1589`) runs with no
    `TRANSACTION` token and fires the gate as unplanned inference-time growth, with a witness
    check that it is so (§2.4.2 "The token's kind");
  - **each WARN names its site (zhcn 5.5 row 37).** Both gates' WARNs gain a `site=%s` field,
    fed from the owner request's cohort (`context-compute-host` for the §2.4.3 carve,
    `moe-mmid-host` for the MMID carve) or, for a request with no cohort, its role and tag.
    That is what makes a load spill's WARN attributable to B's load rather than to a context,
    and what zhcn's GP2 arm needs (rulings §Z6a);
  - **the root defect is llama.cpp-dhpw** (P2): `g_offload_phase` is process-wide and never
    reset at graph exit, so allocation phase gates misclassify other contexts' work. Making it
    per context or per thread is that ticket, which lands after this design's L0 token; this
    design only keys the two pool gates on a transaction's provenance.

  The reservations themselves:
  - **Held carves, one per host slot index (rulings §D15, §ZR5 I-2).** The first publish
    allocates one owner-first reservation per host `(cohort, index)`, through the request above,
    and the registry entry holds them together, for the context's life, as the context's **host
    reservation** (its HOLD). "Reserves" means these carves hold the bytes: a flag such as
    `forbid_host_zone_growth` reserves nothing, and revision 7.5's "records that capacity"
    reserved nothing either (r7 I-6). Revision 7.6 took one carve and sub-carved the slots from
    it as offset views; §ZR5 I-2 replaces that, because the slots of one context need not be
    adjacent and an offset sub-carve across a pool region is exactly what the ruling forbids.
  - **Each is contiguous, and the path exists (llama.cpp-nsl3; zhcn 5.4 row 19).** Host slot `i`
    **is** reservation `i`: no offset views, no slices across reservations, no summed envelope.
    The owner-first request (`unified_allocate_owner`, `unified-cache.cpp:16141` at `3d9414c8c`)
    reaches one contiguous allocation with no new primitive: `select_zone` (`:15361-15372`) maps
    `HOST_COMPUTE` to the WEIGHT zone, and `try_zone_alloc_contiguous` (`:15394-15437`) calls
    `host_zone_alloc` → `pinned_chunk_pool::zone_alloc` (`pinned-pool.cpp:500`, one chunk). On a
    miss it grows the zone through `host_zone_grow` (`unified-cache.cpp:20399`) → `grow_zone`
    (`pinned-pool.cpp:613`) by one chunk of `chunk_footprint_for` = `max(chunk_size_,
    align_up(need))` (`:798-803`), so a slot larger than a zone chunk gets a chunk of its own.
    Before the zones are configured, the same request takes `host_pool_alloc`
    (`unified-cache.cpp:20169-20174`) → `allocate_runtime` (`pinned-pool.cpp:209-212`) →
    `allocate_from_chunks` (`:805-850`), with the same footprint. `allocate_segmented`,
    `zone_alloc_segmented` and `host_zone_alloc_segmented` (`unified-cache.cpp:20486-20502`) are
    never on this path. The limits are the pool budget and Level Zero's per-allocation limit
    (about 11 GB); a failure there fails context creation. Both pool phase gates stay silent on
    this growth because the thread holds a `TRANSACTION` token (the provenance rule above), not
    because of the phase, which can read PP or TG here. Revision 7.7 routed a large slot to
    `allocate_runtime` and called routing it there an L4 gap; that was wrong, and the gap does
    not exist.
  - **Each slot's size is the plan's maximum for that index (rulings §ZR5 I-1; r8 m-10):**
    `R_h[i]`, the largest `slot_bytes` at that `(cohort, index)` over **every** rung in the
    ladder's rung set, not the top rung's demand. The two are equal only if host demand is
    monotone in `n_ubatch`, which nothing guarantees. The rung set is zhcn 5.4's
    `llama_auto_ubatch_rung_set(ladder, n_ladder, fallback_ubatch, cap, cached_ubatch, out)` in
    `src/llama-auto-ubatch.h`, new in zhcn's scope: the fallback, every ladder rung between the
    fallback and the cap, and a valid cached value. It is computed before `create_memory`; the
    fixpoint (zhcn's R*) measures each rung host-side before the first publish; and the ladder
    iterates only that set, so a rung is a candidate only if it was measured (zhcn 5.4 row 19).
    So every candidate the plan admits is covered on the host tier, a republish never allocates
    host bytes, and no host slot is ever released and re-allocated within the plan.
  - **A request beyond it is refused by arithmetic, against that slot's reservation**, never
    against live free room, before anything is released, never a growth under the gate and
    never a transient. Only a setter can reach it: the adapters, the sampler and
    `llama_set_warmup` (rulings §D20.1), which are not ladder candidates. There it is a
    **candidate refusal** with the tenants-alone message at rulings §B step 4 ((0) on the
    tenant-only path; zhcn's E9), deterministic in the plan. At a ladder rung, a live host need
    above `R_h[i]` contradicts the measured plan, so it is `[CONTEXT-PLAN-BUG]`, never a
    refusal.
  - Expert-cache host fills and host weight staging take no L0 and allocate from the same pinned
    pool (zhcn T7). Between publishes they can take any free host room, but not the
    reservation, which is allocated. They are themselves planned capacity (P4; 1oxa r3 I5); host
    growth the plan does not account for is a plan bug, not a race to tolerate.

  §3.1 H4 carries the cases, including the RED where a concurrent host fill takes the room.

**The record (r3 I2, I7; r4 I1, I4, m12).**
```
enum class demand_scope : uint8_t { MODEL, CONTEXT };  // 7.14e DEVICE withdrawn (§M32 I-1)

struct context_side_demand {
    int                  device;
    shared_zone_lifetime lifetime;  // CONTEXT or TRANSIENT (WEIGHT_SIDE_TRANSIENT under §2.1's lever)
    demand_scope         scope;     // whose lifetime the reservation follows
    uint64_t             owner;     // CONTEXT: the ContextId; MODEL: the ModelId
    const char *         cohort;    // the cohort_id its claims carry
    std::vector<size_t>  slots;     // slots[i] = cap of slot index i, one per allocation that can be live at once
};
```
- **Indexed slots, not a byte peak (r3 I7; r4 I1).** The producer lists, by index, the
  allocations that can be live at once, and each claim names its index (§2.3.2).
- **Scopes (r4 I4; rulings §M8 I-5(b), §M32 I-1).** There are two, each with one owner and
  one release site:
  - **MODEL**, owned by the model token: the model's weights (the cache's registered entries,
    under the model's leases) and its model-lifetime RUNTIME terms, held as pending ranges from
    the transaction until they materialize or the model is destroyed (rulings §Z5 IMP-5).
    Released at the model's unload. The MMID workspaces are **not** MODEL scope (rulings §M9
    I-3): they are queue-bound, so they are CONTEXT scope (below). 7.7 had them here;
  - **CONTEXT** (a tenant), owned by the context's registry entry (§2.5): its KV extents, its
    tenant slots, its ring rows (§2.7), its host-slot reservations and, where the route is
    reachable, its MMID device pools and host carve (§2.4.2 step 7). Released by the release
    proc (§2.4.2
    "Teardown"). **Every per-op cohort is CONTEXT scope** (beni's ticket already says
    "scope/owner per context"). Revision 5 let the per-op scratch be DEVICE scope, sized from
    one transaction's plan, which could not see other owners: model 2's context could shrink
    slots model 1's running contexts still used, two concurrently executing contexts would share
    one slot, and a dropped slot that was claimed had no release. Per context, each context's
    slots are its own, and each context executing concurrently (C5; the overlapping host
    submission of CLAUDE.md §5) claims its own.
  - 7.14e had a third, `DEVICE` scope, whose only member was the u1bb ring, owned by the
    device's ring record. It is withdrawn with the device ring (rulings §M32 I-1): the rows are
    CONTEXT slots and the weight slot is a MODEL-lifetime D term (§2.7), so no record is
    shared between contexts.
- **Zone.** There is no per-record zone (rulings §T, §Z42.3): the cohort table is the only
  source of a cohort's zone and tier, and §2.3.5's routing places each slot on the TLSF its
  cohort names. Revision 7.6 said "every record carries its zone (23mk, agreed)"; that is
  withdrawn, and the record above has no zone field.

**Reconciliation, per transaction.** A full transaction plans this context's CONTEXT records,
its ring rows among them. A slot is
reused in place when its owner already holds the same `(cohort, index)` with a capacity at
least the new need (r5 I-B; r6 m-1);
every other slot is carved new, and the old one it supersedes is released at the publish
(§2.4.2 step 8). Claimed slots are never moved.

**One function produces a context's records (rulings §M38 C-1).** `context_demand_records(model,
dev, shape)` returns the CONTEXT-scope records of a context of `model` on `dev`, where `shape`
is `n_ctx`, `n_ubatch`, `n_seq_max`, the flash-attention choice and the KV types. It calls each
producer once: zhcn's measure pass with beni's and jzvq's visitors, jzvq's demand functions,
23mk's `onednn_pp_a`, `set_rows_stage` and Graph-scratch functions, the ring rows' and
`moe_control`'s value functions, the recurrent state's size function, the oneDNN scratchpad's
load-time table and, where the route is reachable, the MMID workspace. A context
transaction's step 2 reconciles its output at the context's own shape, and the load's
`FIRST_CONTEXT` reservation sums it at the envelope's shape (§2.4.2 (b) step 3). No other code
enumerates the C terms: H7ap's first-context arm has a source gate that fails on a cohort or
term named at the reservation site.

**Timing (r3 I3; r4 I10(d)).** Every record exists before the transaction that consumes it.
zhcn's measure pass runs before each publish, for each ladder candidate, and computes its own,
beni's and (where they are graph-shaped) jzvq's demands with its visitors. The ring rows' and
the recurrent state's are pure over the plan, the model, `n_ubatch` and the descriptor. There is
no re-fit after `sched_reserve` (§2.4.2).

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
  - abort under `GGML_SYCL_STRICT_LEASES=1` (§G1).

  Revision 5 let a non-STRICT (§G1) violation take unreserved context-side room through
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
  never classified as a plan violation, and STRICT (§G1) does not abort on it;
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
(Qwen3.8-Flash-Next under STRICT (§G1)) forbids. This is that design:
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
  `:1936`). The FA recheck sends no descriptor at all: it takes the backend, the model token and
  the FA flag (`ggml-sycl.cpp:19024-19026`), and on a `tenants_planned` context an FA flip
  reaches the backend as a tenant change instead (§2.4.2; r8 m-7). The tenant section is
  re-measured per ladder candidate by zhcn.
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

#### 2.4.5 The zone terms: one closed enum (rulings §M18 I-A, §Z15, §M19; r13 I-A, m-h)

A **zone term** is one named demand inside a zone's size. It is not a `pending_term` (§2.3.3
A1), which names what a pending range holds room for; the two lists are separate. The zone terms
are the values of one enum, `ggml_sycl_zone_term`, shared by this design and 23mk's, spelled
`GGML_SYCL_ZONE_TERM_<NAME>`; the `%s` term slot of both late-stage strings prints the lowercase
`<name>`, with no `_bytes`. Each term has one zone, one owning design, one value function and one
class:
- **P**, placement-independent, computed in step 1 (§2.4.2 (b));
- **D**, placement-dependent, charged by the pack in step 3 as it places each weight;
- **C**, context-lifetime. **The C rule:** a C term is in the enum so the vocabulary is closed.
  The load-stage dry run (step 5's second witness) evaluates it as 0, no zone is sized for it,
  and its check runs in the context transaction, which is its owner's. The load's only charge
  for a C term is the model's first context's reservation (§2.4.2 (b) step 3; rulings §M32 C-1
  (a)), which is not a zone term. **Its room is
  that context's `REGION` headroom, like KV (rulings §M21.3, §M25 I-6):** the context
  transaction places it as a head slot of the context's fit, inside the ranges step 5 records.
  No C term waits for a zone to grow after load, since no zone may grow then (a rebuild that
  meets live bytes refuses, §2.4.2 (b)).

**A floor is not a term (rulings §M25 I-1).** Each zone is `max(floor_Z, demand_Z +
charged_Z)` (§2.4.2 (b) step 4). The floors, SCRATCH 512 MiB (the compute arena, which spans
the zone) and ONEDNN 256 MiB, are capacity and have no enum value; 7.14's `compute_arena` P row
is withdrawn. RUNTIME has no floor on an arena device (rulings §M32 I-2; the end states), so
`ensured_RUNTIME = charged_RUNTIME` there. ONEDNN and SCRATCH lose theirs the same way, each
in the commit that moves its zone's last unplanned consumer (rulings §M37 Q3; the end
states), after which `ensured_Z = demand_Z + charged_Z` for them too.

**Agreed with impl-23mk** (23mk rev 4.7b `e81dc2327`, which agreed the enum with three
amendments: the C class, `onednn_pp_pool` on SCRATCH, and `pp_pipeline` as one term with 23mk's
value function; and 4.7c `175dcd51b`, whose §6.8 term table, L2986-3011, confirms these names,
`load_reorder_temp` included). The owners of the two oneDNN scratches are the lead's (rulings
§M19). One term, one owner, one value function:

| term (`%s`) | zone | class | owner | value function / source |
|---|---|---|---|---|
| `ring` | the context's `REGION` headroom, as its own CONTEXT head slots, one activation and one output row per (context, device) (§2.7; rulings §M32 I-1); not a load-stage zone | C (rulings §M28 (1); P before 7.14e) | moua | the rows at the context's own `n_ubatch`, sized by the experts resident on the device (rulings §M32 C-1 (b)): align256(align64(`n_ubatch`) × max over t of (min(`local(t)`, `n_ubatch` × `n_expert_used`) × K_t) × 2) and the same with N_t × 4, times the depth (1); t runs over the expert tensors the device executes through oneDNN PP, `local(t)` is t's resident expert count, K_t and N_t per `src/llama-model.cpp:490-493`. The PP MoE dispatch does not chunk (§2.4.2 (b)). No device-wide record: 7.14e's max over contributions is withdrawn |
| `nonfa_shape` | SCRATCH | D: per device that hosts attention layers (r13 I-F (5)) | moua | `unified-cache.cpp:27615-27618`, sized at `:4401` |
| `onednn_scratchpad` | the context's `REGION` headroom, one buffer per (context, queue) over the context's enumerated queue set, at the context transaction; not a load-stage zone | C (rulings §M29a) | moua (rulings §M19, §M29, §M29a, §M32 I-5) | consumer (b) alone, the oneDNN primitives' own user scratchpad (`common.hpp:5878`, `:5945-5980`): the maximum of `scratchpad_desc().get_size()` over the `primitive_desc` objects of **every user-scratchpad family the context's dispatch can reach, each under its dispatch's gate** (the eleven live `get_scratchpad_mem` sites, matmul, WOQ, eltwise, softmax and binary; the reduction and `DnnlBinaryWrapper::binary` are dead and deleted, rulings §M36 I-3, §M38; §2.4.2 (b), "The ONEDNN zone's scratchpad"), for the shapes the context runs, over the M set up to its `n_ubatch`, read from the load's descriptor table (rulings §M31), never a measured bound; allocated once and never grown, a request above it the named `[ZONE-PLAN-BUG]`. It has **no load-stage setter**. Master's `onednn_reorder + onednn_eligible` (`ggml-sycl.cpp:15947`, `unified-cache.cpp:27565`) is withdrawn: it sized 23mk's W and A pair, and 23mk re-points that site to W (`onednn_pp_w`, below) |
| `moe_onednn` | RUNTIME | D | moua | the ring's weight slot alone (rulings §M28 (1)), one per (model, device), drawn at load under `{MODEL, id}`: the maximum over the device's oneDNN-PP expert tensors t of t's per-expert slot (`src/llama-model.cpp:441-447`) times `local(t)`, t's experts resident on the device (rulings §M32 C-1 (b)); master multiplies the inventory maximum by `n_expert` (`:465-467`). Published by its own setter, `unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes(model, dev, bytes)`, into a store keyed by (model, device) (rulings §M32 I-3, §M38 I-2); a later load on a laid-out arena places it in RUNTIME's free room or the shared zone (§2.4.2 (b) step 3); master's `unified_cache_set_planned_pp_moe_onednn_scratch` call with the 512-row slots (`ggml-sycl.cpp:15961-15967`) is deleted on arena devices |
| `moe_control` | the context's `REGION` headroom: one slot per (context, device) on each device that executes a GPU expert of the context's model, at the context transaction (rulings §M27 (2a)) | C | moua | the ungated layout's compact list and missing flag at the context's own `n_ubatch` (`moe-control-plan.cpp:478-497`; `mmvq.cpp:16589`, `:16649`): `total_bytes − compact_offset`, 16640 B / 33024 B at `-ub 512` on GPT-OSS 120B / Qwen3.5-35B-A3B; never `moe_control_requirement_from_layout`. The ids staging is row 75's (beni; rulings §M26 I-3), the pointer tables `moe_ptr_table`'s (rulings §Z15); the load charges nothing. 7.14a to 7.14c's RUNTIME D charge (24832 B / 49408 B from the routing commit, at r14 m-6's plan `n_ubatch` of 512) is withdrawn |
| `xmx_moe_buffers` | RUNTIME | D, transitional (rulings §M38 C-2) | beni, whose conversion of rows 13 and 14 deletes it | the `xmx_moe_buffers_t` byte functions (`common.hpp:6804-6834` at `c69d5774d`) at the envelope's shape, under the dispatch's gate: 0 unless `GGML_SYCL_XMX_MOE_SORTED` and `GGML_SYCL_XMX_MOE_PREALLOC` are both set (`ggml-sycl.cpp:67008-67018`) |
| `pp_pipeline` | RUNTIME | D | 23mk (rulings §Z15) | 23mk's `pp_pipeline_weight_bytes`, the allocation site's own bytes (`ggml-sycl.cpp:92998-93031` at `3d9414c8c`); moua's pack calls it, and master's `:15955-15958` fields become its output |
| `moe_ptr_table` | RUNTIME | D | 23mk (row 73) | 23mk's `moe_ptr_table_bytes`; the **only** term for the MoE pointer tables, whichever path holds them (rulings §Z15): k tables at the 256-aligned stride (1024 B on GPT-OSS 120B, 2048 B on Qwen), k from the term's first commit, which also deletes row 73's fallback (rulings §M23 (2)). No `expert_ptrs` term exists; `expert_ptrs` feeds only the host sum (s4ip) |
| `onednn_pp_w` | ONEDNN | D | 23mk (W) | `onednn_pp_w_bytes`; `unified-cache.cpp:17316`. The ONEDNN D store survives under 23mk's five I-G names: `g_tensor_inventory_onednn_pp_w_bytes`, the setter `unified_cache_set_planned_onednn_pp_w_bytes`, the store `g_planned_onednn_pp_w_bytes[dev]`, the getter `unified_cache_get_planned_onednn_pp_w_bytes` and `plan.onednn_pp_w_bytes` (today `onednn_scratchpad_bytes` in each, the getter with a `_stored` suffix; 23mk §4.3 at `e4f08213a` L1137-1165; rulings §M32 I-4, §M33 I-G). 23mk re-points master's `:15947` site to this value function. H7ap's C-rule fixture asserts the getter returns > 0 |
| `onednn_pp_pool` | SCRATCH | D | 23mk (POOL) | `onednn_pp_pool_w_bytes`; pool `:43324` via `:45640-45651`, `:65339-65355` |
| `load_reorder_temp` | SCRATCH | D, needs the layout | 23mk (the LOAD term) | `load_reorder_temp_bytes` |
| `woq_packed` | SCRATCH | D, reads the chosen layout | 23mk | `woq_packed_bytes`; `gemm.hpp:860` |
| `mxfp4_direct_f16_w` | SCRATCH | D | 23mk | `mxfp4_direct_f16_w_bytes`; `ggml-sycl.cpp:64056/64231` |
| `lm_head_f16` | SCRATCH | D | 23mk | `lm_head_f16_bytes`; pool `:43324` via `:45640-45651` |
| `mmq_work_counter` | SCRATCH | P (fixed) | 23mk | `mmq_work_counter_bytes`; `mmq.cpp:271` |
| `bf16_materialize` | WEIGHT | D | 23mk | `bf16_materialize_bytes`; `ggml-sycl.cpp:14332` |
| `fp16_slab` | WEIGHT | D | 23mk | `fp16_slab_bytes`; `ggml-sycl.cpp:1342`. Each copy's class, PRIMARY or OPTIONAL, comes from the plan (rulings §M18.3a); OPTIONAL is a copy class, not a zone (r14 m-9) |
| `onednn_graph_scratch` | the context's `REGION` headroom (rulings §M21.3); not a load-stage zone | C | 23mk (rulings §Z15, §M19) | 23mk's value function, which calls 1oxa's §V11.3 D512 tile screen as a helper; 1oxa charges nothing for it (rulings §M26 I-4). Charged at the context transaction from that context's `REGION` headroom, like KV. Master publishes its shape at `unified-cache.cpp:27580-27583`, adds its floor to the ONEDNN zone at `:2148-2160` and draws it from that zone (`onednn_graph_scratch_alloc`, `:11655`); the Graph-scratch commit (§2.4.2 (b); rulings §M26a I-4) moves the draw into the context's `REGION` range and takes the floor out of the load stage |
| `set_rows_stage` | the context's `REGION` headroom on the owner device, at the context transaction (rulings §M25 I-6) | C | 23mk | `set_rows_stage_bytes`; `set_rows.cpp:465` |
| `onednn_pp_a` | the context's `REGION` headroom, at the context transaction (rulings §M25 I-6), as a head slot of the context's fit from moua L4, not from beni (rulings §M27 (1)) | C | 23mk (A; 23mk's addition, for the same closure as `set_rows_stage`) | 23mk §4.5, `onednn_pp_a_bytes` |
| `mmid_workspace` | the context's `REGION` headroom: the MMID device pool is that context's `REGION` head slot, placed once, at step 5 (rulings §M25 I-4; §M9 I-3) | C | moua | `plan_moe_mmid_workspaces` (`unified-cache.cpp:26806` at `3d9414c8c`) and `account_moe_mmid_workspaces` (`:26910`); the context's rung through `replan_moe_mmid_workspaces_for_runtime` (`:26946`). 7.14 classed it D in RUNTIME and cited `:15865`, which is the `allocation_owner_test_*` hooks (r14 I-4) |

The context pending terms stay `ONEDNN_PP_A` and `SET_ROWS_STAGE` (§2.3.3 A1); the C rows name
the same demands in the zone-term vocabulary, and neither list replaces the other. A's
placement is a `{CONTEXT, id}` / `REGION` range of the fit (rulings §M27 (1)); what 23mk's
`ONEDNN_PP_A` term still holds (its A3 overlap form) is 23mk's.

**No `moe_compact_storage` term (rulings §M25 I-5).** 7.14a listed 23mk's
`moe_compact_storage` (RUNTIME, D, `mmvq.cpp:16614`, `:16646`) beside `moe_control`. Those two
sites are the fallback allocations of the compact pointer list and the missing flag, which row
134's block holds at master, so the same bytes had two terms. The commit that lands
`moe_control`'s slot deletes the fallbacks, and a context that cannot place the slot is
refused by name at its transaction (§2.4.2 (b); rulings §M27 (2)).

The enum is closed three ways. The post-pack dry-run witness (§2.4.2 (b), step 5) compares, per
device in `plan.devices`, what the published getters would size with what step 4 sized from the
ledger, over every value, a C value as 0 on both sides. The late check compares by enum value,
per device, over the P and D values. And a source-contract gate maps every
`unified_cache_set_planned_*` setter to exactly one value, so a new demand that publishes a
planned global without a term fails the build's gate, not a load. The gate's map carries
`unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes` → `MOE_ONEDNN` (rulings §M32 I-3),
`unified_cache_set_planned_onednn_pp_w_bytes` → `ONEDNN_PP_W` (23mk's rename; rulings §M32 I-4,
§M33 I-G) and 23mk's `unified_cache_set_planned_moe_ptr_table_bytes` → `MOE_PTR_TABLE`, and each
transitional term's setter to its value (`XMX_MOE_BUFFERS`, rulings §M38 C-2); no setter maps to
a C term, and the four device-global ring setters are outside the map because no arena path
calls them (H7z (aj); rulings §M38 I-1). A second gate of the same shape maps every
`get_scratchpad_mem` call site to one `onednn_scratchpad` family (rulings §M32 I-5), so a new
oneDNN family cannot enter unenumerated.

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

### 2.7 u1bb ring, MMID pools, RUNTIME, compute overflow (r1 M9 corrected; r3 I5, I8; r4 I4, I6, I7; rulings §M32 I-1)

**7.14f replaces the device ring (rulings §M32 I-1).** Through 7.14e this section described one
ring per device: a ring record under the L5 slot-state lock, sized as the component-wise max
over the live contexts' contributions, with a RUNTIME half and a KV-zone half, a
`ring_plan_gen`, a RELEASING mark, guard pins and a sole-contributor release. All of that is
withdrawn. The ring is split by lifetime into two objects with one owner each:

- **The rows are per-(context, device) CONTEXT slots.** A context's activation row and output
  row on a device are head slots of its own fit in its `REGION` headroom (the shared zone's
  TLSFs, allocator group `vram_zone_id::KV`; §2.4.1), sized at its own `n_ubatch` by the `ring`
  value function (§2.4.5), carved at step 6 and installed by step 8 (c) into **the context's
  published slot table**, which is their only storage (§2.4.2 "The ring rows"; r8 m-12). The
  depth is 1. They die with the context: its release proc drops them (§2.4.2 "Teardown").
  Another context's transaction never reads, pins, sizes or releases them, so no record is
  shared, no generation is needed, and no transaction can publish another's rows absent. The
  model's first context on a device takes over the rows the load reserved for it (§2.4.2
  (b) step 3, step 5; rulings §M32 C-1 (a)).
- **The weight slot is a per-(model, device) D term.** `moe_onednn` (§2.4.5) is one slot per
  model and device, sized by the resident-expert count, published by its own setter
  (`unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes`; rulings §M32 I-3), drawn from
  RUNTIME inside the load before `finalize_end` under `{MODEL, id}`, and released at the
  model's unload. No context transaction sizes, places or releases it.
- **The executor is re-keyed (rulings §M32 I-1).** Today the batched executor reads the
  planned shape per device (`ggml-sycl.cpp:78760-78767`, the four
  `unified_cache_get_planned_pp_moe_onednn_*` getters keyed by `ctx.device`), reserves the
  device ring lazily (`:78817`) and claims a slot of it (`:78823`). On an arena device it reads
  the activation and output bytes from the dispatching context's slot table and the weight
  bytes from its model's weight slot, admits with the same pure `pp_moe_onednn_admit_scratch`
  over that shape, and claims the context's rows and the model's weight slot. The lazy
  `reserve_pp_moe_onednn_scratch` becomes a presence check: a missing row or weight slot is
  `[CONTEXT-PLAN-BUG]` with an error status (§2.4.3), never a reserve at dispatch. The
  refusal WARN and its `reject_batched` fallback stay for a shape above the plan.
- **The device-global ring setters are unreachable on the arena path (rulings §M38 I-1).**
  `unified_cache_set_planned_pp_moe_onednn_scratch` (`unified-cache.cpp:2164` at `c69d5774d`,
  keyed by `device_id` with the weight, activation and output bytes and the depth),
  `_kv_zone_slots` (`:2217`), `_row_bytes` (`:2253`) and `_n_ubatch` (`:2278`) write
  device-keyed globals. The install writes the context's slot-table entry directly, so on an
  arena device no path reaches any of the four: not the load, not the commit's install, not
  the release proc, and not the executor. 7.14f and 7.14g kept them reachable "from the
  release proc and the commit's install, which write the context's slot-table entry", which
  contradicted this section's deletion of the `*_in_kv_zone` flags, the executor's reading no
  device global, and the setter gate's "no setter maps to a C term" (r16 I-1). They and
  `release_pp_moe_onednn_scratch_ring` stay only for a device with no arena, outside this
  design, and the L4+L6 commit deletes their arena-path calls. The weight-slot setter is not
  one of them: it is the load's one publication of a D term, keyed by (model, device).
- **A model load never releases or re-sizes another model's ring (llama.cpp-r7fz; rulings §M7
  I-4).** On master, `populate_inventory_globals` (`3d9414c8c` `ggml-sycl.cpp:15891`, reached
  from `ggml_backend_sycl_compute_placement_plan_early` at model load) overwrites the device's
  planned ring sizes and depth (`unified_cache_set_planned_pp_moe_onednn_scratch`,
  `:15967-15970`), releases the physical ring whenever it holds KV-zone bytes
  (`release_pp_moe_onednn_scratch_ring()`, `:15983-16000`; its own comment says this "also
  releases a ring that another still-loaded model's live context admitted"), clears the
  KV-zone slot flags (`:16002`), and overwrites the per-row bytes and the planned `n_ubatch`
  (`:16013-16015`). With model A's context live on device 0, B's load would release A's rows,
  and A's next PP MoE claim would find none: `[CONTEXT-PLAN-BUG]` with an error status, a −2
  that is fatal on the server. On arena devices a load writes only its own weight slot's
  setter; it releases nothing and writes no row size, depth, split flag, per-row bytes or
  `n_ubatch`. Since the rows are in each context's table, there is no device record left for
  a load to overwrite. **The load-time reserve goes too:** master's
  `ggml_sycl_configure_host_zones_for_plan` reserves the whole device ring at the load's
  sizes (`ggml-sycl.cpp:5657-5680` at `c69d5774d`, `reserve_pp_moe_onednn_scratch`); on an
  arena device the load draws only the weight slot. **Test (H9):** "load B while A's context
  holds its rows" (§3.1). A device with no arena keeps master's behaviour, outside this
  design.
- **The host runtime pre-size is re-derived from the terms (rulings §M34 (4)).** The same
  function, `ggml_sycl_configure_host_zones_for_plan` (a load-time function, not a context
  init), pre-sizes the host arena's runtime pool with `plan.onednn_pp_w_bytes` (today
  `plan.onednn_scratchpad_bytes`) plus `plan.dma_staging_pool_bytes +
  plan.pp_pipeline_scratch_bytes + plan.pp_moe_onednn_scratch_bytes` (`ggml-sycl.cpp:5647-5652`,
  `pre_allocate_runtime_chunks`, which reaches `host_arena_`, `unified-cache.cpp:20479-20484`).
  Three of the four are device terms, each already placed on the device: `onednn_pp_w` (the
  first field, renamed by 23mk) in ONEDNN, `pp_pipeline` in RUNTIME, and the ring in the
  context's `REGION` and the model's RUNTIME weight slot. Adding them to a host pool charges the
  same bytes twice, as idle host-pinned room (P4). On an arena device the pre-size is the
  host-tier terms only, which at `c69d5774d` is `dma_staging_pool_bytes`. L4's census is the
  check: an allocation of W, `pp_pipeline` or ring bytes from the host runtime chunks on an
  arena device is a host-tier consumer nobody named, and becomes a term of its own rather than a
  share of a composite. 23mk's rename carries the reader unchanged (23mk §4.3), so the
  re-derivation is this design's alone.
- **The KV-zone split and its size function are gone.** u1bb split the ring between RUNTIME
  and the shared KV zone from live `zone_available(RUNTIME)` (u1bb `:17596-17620`, `:17603`),
  and `unified_cache_get_planned_pp_moe_onednn_kv_zone_bytes` returned the KV-zone half times
  the depth. With the rows in `REGION` and the weight slot in RUNTIME by class, there is no
  split to decide and no half to report; the function and the `*_in_kv_zone` flags are
  deleted on arena devices. zhcn's G2/GA figures read the context's rows from the plan line
  (270.0 / 540.0; §2.4.1, GA), not from that function.
- **Growth.** A context's rows grow when its ladder climbs: GROWTH, never COVERED (§2.4.2
  tenant-only path), with the new rows carved beside the old and the old released after the
  publish, event-gated (step 8 (d)), or, on the tenant-only path, moved out at (c). The
  overlap is the context's own and is priced: its fit places KV around both, and a layer it
  demotes is labelled `ring-growth` with the old and new row bytes (§2.4.1). A shrink reuses the
  rows in place. No other context is affected.
- **Claims.** A context claims its own rows and its model's weight slot, with event-chained
  reuse (§2.3.2). On master the state is occupancy-until-completion:
  `pp_moe_onednn_claim_scratch_slot` round-robins, a slot stays `busy` until its recorded
  `done_event` completes, and when every slot is busy the claim host-waits
  (`wait_event.wait_and_throw()`, master `2c4f5e45d` `ggml-sycl.cpp:1648`; the drain's at
  `:1772` is teardown). Here a slot is *claimed* from the claim to
  `pp_moe_onednn_record_scratch_slot_event`, which is the submission; the recorded
  `done_event` becomes the slot's release event; the next claim returns that event for
  `depends_on` instead of waiting; `retained_owners` stays the slot's lifetime retention. The
  slot state (`g_pp_moe_onednn_scratch_slot_state`, today per device) is keyed per (context,
  device) for the rows and per (model, device) for the weight slot. **Two contexts of one
  model on one device share its weight slot** and serialise on its event order, with no host
  wait; contexts of different models share nothing (same-device concurrent inference stays
  unsupported, §2.10 §5.3; this only keeps the overlap that exists today safe).
  - **Two generations on one slot (r5 I-K; lead ruling).** Master's per-slot state holds one
    generation: `pp_moe_onednn_record_scratch_slot_event` overwrites `retained_owners[slot]`
    (`2c4f5e45d` `ggml-sycl.cpp:1722`), and the claim swaps the owners out and calls
    `release_pp_moe_onednn_scratch_slot(slot, generation)` only after the previous
    `done_event` completed (`:1625`, `:1635`), which the host wait at `:1648` guaranteed.
    Without the wait, a re-claim of a slot whose previous work is still queued must not drop
    that retention. So, at the re-claim, the previous generation's `retained_owners` are moved
    out, and after the slot-state mutex is unlocked they go to
    `retain_handles_until_event(previous done_event)`, before the new generation's record;
    they drop only after that event. The cache-side per-generation refcount
    (`claim_pp_moe_onednn_scratch_slot` / `release_pp_moe_onednn_scratch_slot`, `2c4f5e45d`
    `unified-cache.cpp:18037-18096`) is **deleted for arena devices**, because the reserved-slot
    handles make it redundant: each claim takes a slice of the slot's handle, the slice
    travels in `retained_owners`, and a row superseded by a growth keeps its block until the
    last such slice drops after its event. H7z's mutation witness adds the overwrite restored;
    H9 carries the supersession case.
  - **Every other removal of the retention hands off too (rulings §RING; r6 I-6).** With the
    refcount deleted, `retained_owners` is the only lifetime holder of a superseded slot.
    Master drops it with no event, under the L5 slot-state mutex, at five more sites
    (`ggml-sycl.cpp`, the same lines at `11faace69` and `3d9414c8c`):
    - `pp_moe_onednn_reset_slot_state_locked` (`:1551-1559`, `retained_owners.assign(ring_depth,
      {})`), reached from the claim, bind, record and release paths whenever the depth changes
      (`:1602-1604`, `:1693-1695`, `:1712-1714`, `:1733-1735`);
    - the generation-0 branch of the claim (`:1617-1621`);
    - release-unused and rollback (`:1585-1588`, `:1738-1742`);
    - **the bind (r7 m-5):** `pp_moe_onednn_bind_scratch_slot_generation` clears
      `retained_owners[slot]` (`:1698`). It is benign only while the claim before it has
      already moved the previous generation out (above), so the slot is empty at the bind. It
      gets the same hand-off, and a witness check that the vector it moves out is empty; the
      witness plants a retention at the bind and requires it to be handed off, not dropped.

    Each of these would be a free under queued work, and each is a `mem_handle` destruction
    under a listed lock (H7t). Each site now moves the vector out and, after unlocking the
    slot-state mutex, hands it to `retain_handles_until_event(done_events[slot])`. A plain clear
    remains only where the generation is 0 and nothing was recorded. H7z covers all six sites.
    The teardown is the seventh removal, with the same hand-off (§2.4.2 "Teardown").
  - **Record mode (r5 I-C).** A claim made while recording follows §2.3.2's record-mode rule:
    the recorded graph holds the slot for its life, and the slot vacates at the graph's
    destruction on an eager event. The rows are the context's own, so a recorded holder takes
    nothing from another context. The weight slot is shared by the model's contexts, so a
    recorded holder would take it from every other context of that model for the graph's life.
    **So the path never retains the shared weight slot across a recording (rulings §M37 Q6).**
    Reaching the PP MoE oneDNN path while `ggml_sycl_graph_recording_active()` is an
    always-compiled witness, `[ZONE-PLAN-BUG] PP MoE oneDNN path reached while recording:
    context C device D`: a STRICT (§G1) abort, otherwise the executor's existing refusal on the
    same device (the serialized MoE route, or for an `XMX_TILED`-claimed op the
    `ggml_sycl_fallback_error` graph failure, "Why `ring` is C" in §2.4.2 (b)), with no
    weight-slot claim. The weight slot's depth therefore never counts a recorded holder, and the
    rows' depth stays 1. H7 (ar) gates it. If the witness ever fires on a real run, the case
    returns to the lead.
- **What the head slot fixes (r3 I5).** Revision 4 carried u1bb's order: KV took the gap, and
  the ring was admitted afterwards against whatever was left, refusing the context when nothing
  was left (*"PP MoE oneDNN scratch ring does not fit"*, u1bb `:18388`), where demoting one KV
  layer would have fit both. H2 covers it.
- **The ring-held yield limitation, closed by L4+ (lead ruling, from jehw's merge).** On the
  jehw-merge order, the yield models the KV zone while the ring's KV-zone slots are still held
  (the ring is released and re-admitted after KV, inside `ggml_sycl_replan_pp_moe_onednn_ring`).
  So on a device holding both WOQ copies and ring slots, jehw's fit places fewer KV layers than
  the room allows: an error toward extra demotion, never an out-of-arena allocation, and only
  on a MoE model with Q4_0 dense weights (which have copies). The rows are head slots of the
  context's own fit, reused in place, so the limitation is gone. H2 carries the case with a
  RED on the jehw-merge order (§3.1).
- **The budget-room check is deleted for arena devices (r4 I7).** The rows' growth is no
  longer checked against `budget_room_bytes` (u1bb `:18362`): the fit is the single source of
  "a head slot fits" (§2.2).
- **The ladder.** Only a candidate's growth competes for leftover room: the tenant-only path
  places the larger rows and compute slots without re-fitting KV, and a candidate whose growth
  does not fit is refused, so the ladder moves on (§2.4.2).
- **The per-row estimate is deleted (r3 I8).** u1bb's admission subtracted
  `k_pp_moe_ring_compute_reserve_bytes_per_row` = 1 MiB/row (u1bb `:17462-17473`, fed at
  `:17629`) as a stand-in for the compute buffer. The compute buffer is zhcn's record in the
  same fit, so zhcn deletes the constant (lead ruling 3), and H7p/H7r check that no
  context-side admission keeps a per-row constant.
- **Retained runs.** A superseded slot released at a publish may leave a RETAINED run when it is
  interior. H4b keeps revision 4's bound (at most one per ladder candidate) and adds that
  fit == carve after the settle.
- **MMID pools are in `REGION` (rulings §M25 I-4, §M9 I-3, §M32 I-2).** Master draws them with
  `prefer_vram_zone = RUNTIME` (`unified-cache.cpp:16222`). Here, where the route is reachable,
  the context's device pool is a CONTEXT-scope head slot of its fit in its `REGION` headroom,
  sized at the context's own `n_ubatch` (rulings §V16a), carved at step 6, whenever the context
  has none on that device or a smaller one than the candidate's workspace; the host pool is the
  context's held MMID host carve. So step 7 allocates nothing, and the first materialization
  is inside the owning context's transaction, never at `load_end` and never after the
  transaction (§2.4.2 step 7; rulings §M7 I-5(b), §M8 I-5, §M9 I-2, I-3).
- **RUNTIME holds D terms only.** On an arena device RUNTIME is `moe_onednn`, `pp_pipeline`
  and `moe_ptr_table`, with no floor (rulings §M32 I-2; §2.4.2 (b), the end states), and every
  draw from it is one of those terms' own (H7p's census). No context-lifetime slot lands there.
- **ONEDNN and SCRATCH tail zones:** no change beyond §2.4.5; their floors go under rulings
  §M37 Q3, each in the commit that moves its zone's last unplanned consumer (the end states).

### 2.8 The error path (decision (a), scoped per r1 I3)

Refuse is the default; `GGML_SYCL_STRICT_LEASES=1` aborts instead (rulings §G1). It is the
switch that every ownership, lifetime and plan defect family folds into (rulings §G1b): lease
ownership and every `[*-PLAN-BUG]` family, in this design and in zhcn's, beni's, jzvq's,
1oxa's, 9g22's, th32's, uwlx's and ds41's. Master carries other switches for defects of the same
family; each is retired into this one by its own lane (rulings §G1b), and this design reads none
of them and adds none. §G1 unifies the switch, not the
tag, so each family keeps its own (`[KV-PLAN-BUG]`, `[CONTEXT-PLAN-BUG]`, `[ZONE-PLAN-BUG]`).
**The switch is read in one place (rulings §G1a).** Master's reader is TU-static in
`unified-cache.cpp` (:12962-12968, process-cached). **llama.cpp-uwlx exports it (rulings
§G1b):** uwlx is the first lane with strict code outside that TU, so it declares
`ggml_sycl_strict_enabled()` in `unified-cache.hpp` and defines it in `unified-cache.cpp` around
the TU-static reader, and every call site in this design is that accessor. uwlx's commit carries
§G1a's three doc edits: it adds the `GGML_SYCL_STRICT_LEASES` row to
`docs/backend/sycl-env-vars.md`, turns that file's :590 ("only on weight:leaked_lease") into the
family list, and gives the canonical contract's :178-179 ("entries_leaked — the sole counter")
the per-family tags. **The fallback:** if moua's L4+L6 commit lands before uwlx, it carries the
export and the three edits (§4), and uwlx converts to the accessor before it merges. No
design adds a second `GGML_SYCL_STRICT_*` switch or a second `getenv` of this one, and H7 (e)
gates both. `GGML_SYCL_HANDLE_STRICT` only reports and never aborts, so it is outside §G1.
**Withdrawn (7.14g):** up to 7.14f this section named a second variable, `GGML_SYCL_STRICT_PLAN`
(lead ruling 4, D3), and forbade an alias, `GGML_SYCL_STRICT_KV_PLAN`. §G1 retires both;
uwlx's branch reads the first and converts it before it merges (rulings §G1b). Outside this
paragraph only the first remains, in H7 (e)'s
mutation witness; both are kept here so a search for them finds the withdrawal. Every site below is a
**terminal** refusal: the fit's demotion order is a planned cascade, not a series of misses, so
nothing in it is counted or aborted as a plan bug (rulings §STRICT). The KV rule applies
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
admitted it"* on every dispatch, which is false, and STRICT (§G1) would abort inference.

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
multi-extent (r1 I4, §2.4.1), both go away. **The tenants here are the unplanned optional
copies and the planned OPTIONAL copies (rulings §M18.3a, which amends §M18.3 and restores this
ladder for them).** A planned OPTIONAL copy sits inside its model's `WEIGHT` range, so the fit
sees it as a buried tenant, and KV still wins over it: the a1_long gate's "KV admission released
6 optional oneDNN WOQ layout copies (220.5 MB) ... for n_ctx=2048's KV" stays pre-registered,
unchanged unless C8's premise (ii) says otherwise (§3.3). A planned PRIMARY copy is `WEIGHT`
and is never a tenant, buried or not. The yield of a planned OPTIONAL copy goes through step
5's yield path, which retags its extent to the context before it releases the handle (§2.4.2
step 5), never through the destructor:
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
    registry entry's (the context's ring rows among them; the model's for its weight slot),
    the KV buffers', the layer views', a claim's, or a
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
    4. Clear this call's pending ranges on the TLSF with `clear_pending_locked(tlsf, {CONTEXT,
       id}, REGION)`, which leaves other terms' ranges of the same context whole (§2.3.3);
       release the group mutex.

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
  - The commit's install of the ring rows (step 8 (c)) allocates nothing: it installs slot
    handles (§2.7). No `mem_handle` is destroyed under the L5 slot-state mutex: every removal
    of a slot's retention hands it off after the unlock (§2.7, r6 I-6). 7.14e's guard pin
    decrement went with the device ring (rulings §M32 I-1).
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
  disposition (error, never raw) is the only one possible there. **The VM zone capacities a
  context's fit reads (cap0) are frozen per context (rulings §V15a and its follow-up, which
  amend §V15's "per transaction"):** one snapshot, taken at the context's first publishing
  transaction and held by the context for its life; every later MEASURE and ALLOC of that
  context reads that copy, another model's load never changes it, and the context's own
  GROWTH does not re-read it. zhcn and 1oxa own the holder; on USM this design's fit reads the
  live TLSF geometry as before.
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

**A VOID arm is a failed arm (rulings §M38 I-5).** Throughout §3, an arm is VOID when its
precondition does not hold, its engagement witness did not fire, or its instrument could not
have seen the property, and VOID is scored as FAIL, never as a pass or a skip. "Void" below
means that.

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
  - **The ring rows are a head slot (r3 I5).** A zone where KV plus the context's ring rows
    does not fit, but KV minus one layer plus the rows does: the fit demotes one KV layer and
    places the rows.
    RED: revision 4's order (KV first, the ring admitted after) refuses the context. A zone
    where the ring alone does not fit with every KV layer on the host refuses, naming the ring.
  - **Mandatory head slots (r3 C1).** A head slot is never dropped to fit KV; KV demotes first.
  - **The budget is not a second source (r4 I7).** A head slot that fits the geometry but not
    revision 5's budget room is placed, with no demotion beyond what the geometry needs. RED:
    revision 5's order (the fit, then the ring's budget-room check) refuses it.
  - **The ring-held yield (lead ruling, §2.7).** A Q4_0-dense MoE fixture: optional copies and
    the context's ring rows on one TLSF. The number of device KV layers equals what the room
    allows with the rows counted once, as its own slots reused in place. RED on the jehw-merge
    order (the ring's slots held while the yield models the KV zone, then re-admitted): it
    places fewer layers.
  - **The rows are per context, in `REGION` (rulings §M32 I-1, I-2).** Two contexts of one
    model on one device, at `-ub 512` and `-ub 1024`: each context's fit places its own rows,
    at its own `n_ubatch` and its own resident-expert count, as head slots in `REGION` (the
    shared zone's TLSFs), and neither fit reads a RUNTIME free byte (a counting stub over
    `zone_available(RUNTIME)` and the RUNTIME TLSF's `available()` reads 0 calls; zhcn 5.13's
    arm (1h) makes the same assertion for A, §4). The rows' extents are disjoint and lie in
    `{CONTEXT, id} ∩ REGION` for their own id. Closing the `-ub 1024` context releases its rows
    and leaves the other's untouched. RED: 7.14e's device ring, which sizes one record at the
    max of the two contributions and keeps it after the larger context closes, and r4 I6's
    RUNTIME half, which a RUNTIME allocation in the yield window could take.
  - **Demotion order and `free_after_full_kv` (zhcn GA; re-derived in 7.14f, rulings §M32
    I-2).** A GPT-OSS-shaped fixture at zhcn's GA deficit, 709.4 MiB on the actual weights
    (11510.9 MiB) with 2204.6 MiB of room and 1348.0 MiB of head slots (the breakdown is in
    §2.4.1, and the fixture builds each term of it, RUNTIME at its D terms with no floor),
    against 128 MiB full-attention slots: 5 × 128 = 640 is short and 6 × 128 = 768 covers it,
    with margins of 69.4 and 58.6 MiB, so head-slot rounding (under 512 B per slot) cannot flip
    it (§2.4.1). 7.14e's fixture, 726.9 MiB with the output row in RUNTIME, is withdrawn.
    Exactly the 6 highest-indexed full-attention layers demote, K and V together, no SWA layer,
    `free_after_full_kv` reads the deficit, and each demoted layer's cause is `head_slot`. A
    variant with sub-slot weight holes demotes 7 and fires the sub-slot WARN.
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
      cached tables, so (e) aborts under STRICT (§G1) on this healthy run (r6 I-3). Further
      arms:
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
        `[CONTEXT-PLAN-BUG]` (a WARN, and an abort under STRICT (§G1), run in a subprocess); the
        BUG line carries the count 1 and the PRIVATE_TESTING backstop counter reads 1. On every
        other GREEN arm of H4 the counter reads 0 (rulings §D16; r7 m-6). RED: a COMPLETE reap
        that drops without the backstop, which frees under the queued kernel;
      - **the ring rows' bound (r6 I-4; rulings §M7 I-3, §M32 I-1).** The context's
        activation row was used once by two claims, so its slot-state retention of the last
        generation holds two slices that share the row's control, and a row growth fits only in
        the old row's room.
        - **(0) on this case:** (0) passes, counting the old row free by arithmetic, and holds
          no copy of it. RED: a (0) that copies the context's rows, whose copy keeps
          `use_count` at 2 into (ii), which refuses.
        - **The move-out and (d):** (c) moves the growing row out with the other growing slots
          and its retention separately; (d) backstops its `done_events` and drops it before
          (e); (e) sees `use_count() == 1`; (f) frees the block; and (ii) places the grown row.
          REDs: `f2e5606bc`, whose (ii) refuses with no demotion and no reason; and 7.5's order,
          which put the retention into the batch, so (e) sees `1 + 2` and reports
          `[CONTEXT-PLAN-BUG]` on this healthy run.
        - **One storage (r8 m-12):** every moved-out row has `use_count() == 1` at (e) on this
          healthy run. Witness: an install that keeps a second, cache-side copy of each row
          handle beside the slot table, on which (e) reads 2 and reports `[CONTEXT-PLAN-BUG]`.
        - 7.14e's nonzero-P arm (r7 m-3) is withdrawn with the guard pins (rulings §M32 I-1): a
          stray copy (`use_count` 2) is the one case, and it reports `[CONTEXT-PLAN-BUG]`;
      - **the freed room is held (rulings §M7 I-5(a)).** A growth whose grown slot lands in its
        old slot's room: between (f) and (ii) a modelled expert-cache fill and a modelled lazy
        MoE layout materialization, each through `allocate_excluding` and taking no L0, try to
        allocate exactly that room. Both are placed elsewhere or miss, and (ii) places exactly
        what (0) placed. RED: 7.5's order, which recorded no pending range before (f), so the
        fill takes the room and (ii) refuses a within-plan candidate;
    - **The covered path (r5 I-B; zhcn rev 5; r6 m-10; r7 m-11, m-1).** A candidate whose every
      slot fits its published cap, host slots and ring rows included, gets COVERED from the
      coverage query: no L0 (the modelled L0 records no acquire), no step (i), no fit, no
      allocation, no release, no registry write, and the published tenant key unchanged. A
      candidate whose tenant slots are covered but whose rows grow gets GROWTH and takes L0 and
      step (i). A candidate with one slot over its cap takes step (i); its host slots are
      reused and nothing is allocated on the host tier. A context whose tenant key is cleared,
      or whose slot table was taken, gets GROWTH for a candidate its caps cover (rulings §M8
      I-4). The query takes only the `kv_region_mutex_` leaf (an instrumented-lock check).
      REDs: `b30321a6f`, which re-allocates every host slot on any key change; 7.5's covered
      rule, which sends the row-growth candidate down the covered path, so its next PP MoE
      claim finds rows too small; and 7.6's rule, which answers COVERED for the cleared-key,
      taken-table state (the next arm).
    - **After a refused (ii) (r6 m-9).** A row growth that (ii) refuses leaves the context
      without rows and with only the tenant slots kept for reuse. The next PP MoE claim reports
      `[CONTEXT-PLAN-BUG]` with an error status and the op is not skipped; the ladder revert,
      whose key no longer matches (it was cleared at (c)), re-carves the previous candidate's
      slots, the rows among them. The positive control keeps the tenant key through (c), so the
      revert hits the equal-key no-op and returns OK with nothing carved. **The rows-only
      variant (rulings §M8 I-4):** a candidate grows only the rows, so every tenant slot is
      reused in place and moves into the registry entry at (c); the rows move out, (f) frees
      them, and (ii) is refused (forced by a test hook). The revert to the previous `n_ubatch`
      is covered by every held slot, but the coverage query answers GROWTH, because the key is
      cleared; the revert takes the growth path, re-carves the rows, and the next PP MoE claim
      succeeds. RED: 7.6's covered rule, which answers COVERED, carves nothing, and leaves the
      claim to report `[CONTEXT-PLAN-BUG]`.
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
    - **The host reservation is held (r5 queue R7b; rulings §D15, §ZR5 I-1, I-2; r7 I-6; r8
      m-10).** The first publish allocates one contiguous owner-first reservation per host slot
      index, with growth allowed, each sized at that index's maximum over every rung. The
      fixture is not monotone: index 0 is largest at rung 256 and index 1 at rung 512, so each
      slot's size comes from a different rung, and the top rung alone under-sizes index 0. One
      slot is larger than a zone chunk: `R_h[0]` = the chunk size + 1 MiB, which must lie inside
      one grown chunk (one segment), and the segmented form is detected. REDs: 7.6's single
      carve with offset sub-carves (a slot spanning a pool-region boundary), sizing from the top
      rung's demand, under which the rung-256 republish is refused, and a large slot served
      segmented. Then, after a modelled decode:
      - **a concurrent host fill fails to take the room:** a modelled expert-cache host fill,
        taking no L0, asks the pinned pool for every free byte; it gets the room outside the
        reservation and not one byte of the reservation. A republish then **grows** slot `i`'s
        need from its rung-512 value to the top rung's (still at most `R_h[i]`); the arm asserts
        that growth before it scores, since a republish whose need does not grow passes under
        the flag-only form too, and the arm would be vacuous (zhcn 5.4 row 19). The republish
        succeeds, allocates nothing on the host tier, and an instrumented `grow_zone` counter
        stays at 0. RED: revision 7.5's recorded-only reservation (host slots at the
        first candidate's size, later republishes allocating with `forbid_host_zone_growth =
        true`), where the fill takes the room and the top-rung republish refuses on live free
        room;
      - a republish whose host need exceeds the reservation is refused at (0) as a candidate
        refusal naming the tenants, by arithmetic, whatever the pool's live free room (the arm
        runs it with the pool both full and empty, and the answer is the same); `grow_zone`
        stays at 0. RED: the first draft's host path, which let every republish grow the pool;
      - **the phase gate keys on a transaction's provenance (7.7a; narrowed by rulings §M9a):**
        at `GGML_SYCL_HOST_ALLOC_PHASE_GATE=2`, after a modelled graph has set the phase to TG,
        a second context's first publish grows the pool for its reservations, and a republish
        past `MOE_GPU_UBATCH_MAX` grows the MMID host carve, both under a `TRANSACTION` token;
        neither asserts. A modelled expert-cache fill on another thread, with no token, still
        asserts. REDs: today's phase-only gate, which asserts on the second context's first
        publish, and a reset of `g_offload_phase` around the transaction, under which the
        concurrent fill no longer asserts;
      - **negative (a), a load spill is not exempt (r9 addendum I-5(a); rulings §M11 m-4):**
        context A is in TG (the modelled phase reads TG). The pinned pool is filled to its
        current capacity first, so any host allocation must grow it; the arm asserts the pool
        grew (a growth counter moves by at least one), since without a growth the WARN cannot
        fire and the arm would pass vacuously. Model B then loads, and B's preload misses VRAM
        and places a host-tier expert through the pinned pool, under `load_end`'s `LOAD`
        token, at `GGML_SYCL_HOST_ALLOC_PHASE_GATE=1`. The WARN fires, exactly once per
        growth, and its `site=` field names B's load weight fill (role `WEIGHT`), not a context
        carve; no abort is expected at gate 1. RED: 7.7a's held-any exemption, under which the
        WARN is silent;
      - **negative (b), gallocr's ALLOC is not exempt (I-5(b); §M11 m-4, m-5; §M12 m-11;
        §Z9a):** the vehicle is zhcn's H4h, cited at zhcn 5.11 (`2c511f6`; rulings §M24 (4))
        and adopted as is: `sched_reserve_impl`'s own growth path, MEASURE → coverage GROWTH → scope → steps 3-6 →
        scope closed → ALLOC, not the resync, which zhcn deletes (zhcn :293). **The arm reaches
        it by one route, zhcn's: a growing setter's next decode** (`sched_reserve_nothrow`,
        `llama-context.cpp:3111`), after two 1-token decodes, where the arm asserts that GROWTH
        was answered, or the run is void. 7.14 also named a ladder rung. zhcn scores only the
        setter, because the ladder-rung route runs inside the constructor, where the phase is
        still `UNKNOWN` (load exit resets it at `ggml-sycl.cpp:12450`, and the warmup pass again
        at `:106882`), so the correct tree would not die; moua matches it (zhcn r11 m-9 (i)).
        `memory_update` is not a route: it calls `graph_reserve` directly
        (`llama-context.cpp:2211`) and never opens a scope, so its post-update reserve is ALLOC
        only. The arm follows §Z9a's six points, through **zhcn's named
        `GGML_SYCL_PRIVATE_TESTING` hooks, at zhcn head `2c511f6`** (zhcn rev 5.11, H4h; the
        names are `8a58ad4`'s, unchanged through 5.10 `c75ce4d` and 5.11; this design defines
        no hooks of its own), **on zhcn's
        vehicle target, `test-sycl-growth-fallback-vehicle`** (rulings §Z12 I-3): its link line
        is exactly `llama-private-test-objects` + `ggml-sycl-private-fixtures`, so there is one
        backend copy, and under `GGML_BACKEND_DL=ON` it is a disabled placeholder that exits 77,
        which is a skip, never a pass. **It is lead-run (rulings §Z14.2):** on `level_zero:1`,
        registered as zhcn 5.11 registers H4h (`2c511f6`): the two labels `cache;mem-handle`
        (`-L 'cache|mem-handle'` is the selection regex, not the label), `FIXTURES_REQUIRED
        test-download-model` (it loads stories15M-q4_0), `SKIP_RETURN_CODE 77`, `RUN_SERIAL
        TRUE`, `TIMEOUT 300`, and the selector pinned by the registration's `ENVIRONMENT`
        property. The lead runs its death arms under `ulimit -c 0`, since a core-dumping process
        holds VRAM past its exit message. No CPU-device admission is added to ggml-sycl. It
        cannot run over a CPU device: ggml-sycl's devices are GPU-only
        (`ggml-sycl.cpp:25059-25072`), so a CPU-device run finds no device and its one-device
        control would exit 77 forever (zhcn r10 I-3 (b)). The exit-77 path under
        `GGML_BACKEND_DL=ON` stays:
        1. the fallback is reached through zhcn's forcing hook,
           `ggml_sycl_test_host_fallback_scope` (thread-local RAII). Its safe-max override,
           `ggml_sycl_test_override_safe_max_alloc_size(1 MiB)`, makes the real `alloc_buffer`
           oversize branch set `must_host_pinned`, and its claim bypass skips the planned claim
           branch, so the request runs the production fallback;
        2. zhcn's filler, `ggml_sycl_test_fill_host_zone(HOST_COMPUTE)`, takes the zone's free
           room, so the fallback must **grow** the pinned pool (`grow_zone` / `grow_into`).
           zhcn's grow counter must read ≥ 1 after ALLOC and its fallback-entry counter ≥ 1; a
           zero on either voids the run;
        3. the arm runs in its own subprocess with `GGML_SYCL_HOST_ALLOC_PHASE_GATE=2` set
           before the first growth;
        4. it is scored **by message**, never by "it aborted" (§M12 m-11): the GREEN tree dies
           with the phase gate's message, anchored on the regex `host pool (zone growth|chunk
           allocation) during inference`, with `site=` naming the compute buffer;
        5. zhcn's `GGML_SYCL_PRIVATE_TESTING` probe prints, to stderr at the fallback's entry
           and before the grow, exactly the literal zhcn owns (zhcn `2c511f6`, H4h):
           `[H4h] held(TRANSACTION)=<0|1> phase=<name>`. It reads 0 on the correct tree and 1 on
           the mutant, and the `phase=` field is what item 6's void rule reads. This design
           quotes no other form of the line (zhcn r11 m-10);
        6. the setup is a real `llama_context` on the SYCL backend on `level_zero:1` (§Z14.2),
           with a mock plan whose caps lie below the candidate. **The route is zhcn's, the
           growing setter alone (zhcn `2c511f6`, H4h; rulings §M24 (4)):** after construction
           the child decodes two 1-token batches of the
           same shape (the first is the warmup pass, the second sets TG at `graph_compute`
           entry, `ggml-sycl.cpp:105145`); immediately before enabling the growing setter
           (embeddings on, or a backend-sampler chain added) it **forces the phase** with one
           parameterised call, `ggml_sycl::offload_stats_set_phase(forced_phase)`, whose value
           is `offload_phase::TG` on every scored arm (zhcn 5.12 `d70c2d2`; the value, not a
           literal gate 34 pins); and the setter's next decode
           reaches the fallback in `sched_reserve`, before its own `graph_compute`. The forcing
           precedes every arm's ALLOC, the mutant's included. zhcn's coverage-query
           last-answer accessor must read GROWTH, or the run is void. **The phase is PP or TG at
           the fallback (r12 m-9):** the phase gate fires only then
           (`pinned-pool.cpp:630-632`), and the probe's `phase=` field reports it; a phase other
           than PP or TG voids the run.
        Controls: positive, the correct tree, which dies with the phase message and probe 0;
        mutant, a scope left open across ALLOC, run with `GGML_SYCL_WITNESS_CHECKS=0` so the
        path reaches the fallback: the gate exempts it, nothing dies, the probe reads 1, and the
        arm reports FAIL; void, a run where the coverage accessor did not read GROWTH, either
        counter read 0 or the phase was not PP or TG, reported as void, never as a pass; and
        zhcn 5.11's VOID positive control (zhcn r11 m-13), a child that forces the phase to
        `UNKNOWN` before ALLOC, which the scorer must report as void. A
        companion arm runs the same mutant with `GGML_SYCL_WITNESS_CHECKS=1` set explicitly and
        expects the alloc-entry check's own message (`[REPLAN-TOKEN] TRANSACTION token held at
        alloc_buffer entry`), not the phase message, so the two deaths are told apart by
        message;
      - **the tree each runs on:** both arms run on the landing tree, moua L4-L6 over zhcn
        (the landing order in §4), where the resync is gone and zhcn's in-`sched_reserve`
        scope exists; arm (b) also runs zhcn's scope as the open-scope RED. Neither depends on
        a code state the other's landing removes. The test build is Release with `-DNDEBUG`,
        so every check the arms rely on is an always-compiled witness (§2.4.2 "The token's
        kind");
      - **the kinds nest as ruled:** a nested `TRANSACTION` under a `LOAD` token and a nested
        `LOAD` under a `TRANSACTION` token each fail the nesting witness check with its
        message (`[REPLAN-TOKEN] illegal nesting`); a nested `LIFECYCLE`
        under either is legal and leaves the outermost kind unchanged (the gate still exempts
        inside a transaction whose reaper runs nested, and still fires inside a load).
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
  - **No yield for a head slot (rulings §M18.3a, §V13 I-8):** a context whose head slots fit
    only in the run a lease-free buried OPTIONAL copy would free, with every KV layer on the
    host. The transaction refuses with the tenants-alone message naming the head slot, and no
    copy is marked yielded or released. RED: a fit that admits head slots to tier 4, which
    yields the copy and admits the context.
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
    site, and each aborts under `GGML_SYCL_STRICT_LEASES=1` through `ggml_sycl_strict_enabled()`
    (rulings §G1, §G1a). Across the tree, the gate forbids any `GGML_SYCL_STRICT_[A-Z_]*` name
    other than `GGML_SYCL_STRICT_LEASES`, and any `getenv` of the switch outside its one reader.
    It also forbids the suffix form, a `GGML_SYCL_[A-Z_]*_STRICT` switch name, except the one
    exception that zhcn's gate 22 allowlists with its ticket until that switch's own lane
    retires it (rulings §G1b). This is gate 22's rule (r4 I3; §G1 follow-up, §G1b), and the gate
    reuses gate 22's allowlist rather than keeping a second one. It skips historical documents.
    `GGML_SYCL_HANDLE_STRICT` only reports and never aborts, and is outside §G1. The mutation
    witnesses are a planted `getenv("GGML_SYCL_STRICT_PLAN")` and a planted second
    `getenv("GGML_SYCL_STRICT_LEASES")` in `ggml-sycl.cpp`, and each must fail the gate;
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
    `[CONTEXT-PLAN-BUG]` line and the STRICT (§G1) abort follow its release (r5 m-k);
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
    mode, `g_moe_ids_d2h_cache`, `g_moe_prompt_admission_cache`, `g_data_ptr_cache` and, under
    `GGML_SYCL_MOE_DOWN_SHADOW`, `g_moe_down_shadow`, and the four C2t scatter lists
    (`g_pending_scatter`, `g_pending_cpu_pipeline`, `g_pending_secondary_scatter`,
    `g_pipeline_scatter`) (§2.4.2 "The holder census"; r8 m-3; zhcn 5.4 row 27). The
    identity type's no-recycling test and the `runtime_tensor_extras` publisher census are
    zhcn's gate 27 (rulings §B.2). Mutation witnesses: a beni-style owning park in a per-op
    cache, an owning `cached_src_handle` restored, a `ggml_backend_sycl_release_buffer_refs`
    step reintroduced, a `host_task` capture, and one per named container with an owning entry
    restored: an owning `graph_input_staging` handle with nothing recorded, an owning
    `moe_ids_cache_key::handle` in each of the two maps, an owning `g_data_ptr_cache` value with
    zhcn's exit clear present (the key fix alone must catch it), an owning
    `moe_down_shadow_key::handle`, and, for each of the four C2t scatter lists, an entry left
    undrained past a non-recording `graph_compute` exit;
  - (ah) **step 2 takes no ring lock (rulings §M32 I-1; r6 I-5 withdrawn).** 7.14e gated that
    step 2's ring-lock section (the copy of `ring_plan_gen` and of the device ring's slots, with
    the `pinned[slot]` increments) preceded the snapshot. With the device ring gone, the gate
    asserts that no path from step 2 takes the L5 slot-state mutex and that the fit reads the
    context's rows from its own published table. Mutation witness: a ring-lock section
    restored in step 2;
  - (ai) **every publish and live-update preparation is under L0, allowlisted or test-only
    (rulings §E.1, §E.2, §L0R, §M8 I-1; r7 I-1, I-2, m-7).** The census is by **reachability**,
    not by operation name or by a list of entries: from every exported function (a
    `GGML_BACKEND_API` declaration or a registry proc address), the gate walks the static call
    graph to every call of `ggml_sycl_publish_plan_locked`,
    `ggml_sycl_publish_prepared_plan_locked`, `lifecycle_replace_placement_plan`,
    `Registry::prepare_live_update` and `Registry::acquire_live_update`. Each path must pass a
    `ggml_sycl_replan_token` taken at the top of its exported entry, or end in one of the three
    allowlisted identity-preserving republishes (`:5887`, `:5908`, `:74882`, each a call of
    `ggml_sycl_republish_current_plan_into_empty(cache, owner)`; `:60125`'s function is deleted
    as dead code, and the gate asserts it is gone), whose one section re-checks
    emptiness, installs into that cache only, compares identity, and writes none of
    `g_model_n_layer`, `g_placement_kv_info`, `g_placement_publication` or another cache's
    snapshot; a witness restores the full `ggml_sycl_republish_current_plan()` at one site), or
    start at a **classified test-only hook**. The test-only class is the exported `test_*` hooks
    that call `ggml_sycl_publish_test_plan` (`:9947`, `:10028`, `:10113`, `:10274`, `:19650`,
    `:19768`, `:19884` → the publish at `:2993`), `test_set_kv_placement_plan` and
    `test_clear_kv_placement_plan` (`:16156`, `:16172`),
    `test_plan_publication_prepare_failure_is_caught` (`:16139`), and `test_hold_live_update`
    (`:12772`, a lease with no L0; r9 m-4), listed by name, each with no caller outside test
    sources (the gate checks every non-test tree the §6.11 census names: `src/`, `common/`,
    `tools/`, `examples/`, `include/` and `ggml/` outside its test directories). **The walk also
    starts at every worker thread's entry function** (the TP worker, the CPU worker, the
    prestage thread and the watchdog, which shutdown joins under L0; r9 m-5), and from those
    roots no token acquire and no publish or live-update callee may be reachable. Beside it: the
    token is the only way to take `g_replan_txn_mutex`; it is taken before L1 and never under an
    L1-L5 lock; graph compute and dispatch never take it (a `sched_reserve` re-plan does,
    legitimately); the release proc's token is the outermost-only kind; no wrapper or
    transaction return maps to `BUSY`; llama has no busy sleep loop; the probe, the FA recheck
    and the transaction body read the per-model snapshot, and the body has no
    `ggml_sycl_global_plan_snapshot()` read; **no exported entry reaches the transaction body
    except the wrapper, and the wrapper passes the snapshot it selected per model** (rulings §M9
    I-1); `ggml_backend_sycl_set_runtime_context` is not exported; the three orphaned publishers
    are gone, `compute_placement_plan_early`'s body is a static impl behind its public entry,
    and the dead `nullptr` publish at `:14773` is gone (r9 m-2, m-3); and no per-device re-plan
    mutex exists. Mutation witnesses:
    **an unlisted exported entry that reaches the CAS** (a new exported function calling the
    transaction body, which the reachability walk must find although no list names it), a
    publisher call site without a token, `ggml_backend_sycl_set_runtime_context` restored as an
    exported entry that passes the global snapshot (rulings §M9 I-1), the `:14773` branch
    restored, a test-only hook called from `src/`, a fifth identity-preserving republish that
    passes a new snapshot, the republish's identity comparison moved outside its section, the
    mutex taken under L1, a graph-compute path taking it, a restored wrapper `BUSY` return, a
    restored llama sleep loop, the body's `:17852` global read restored, a worker thread's entry
    reaching a token acquire, and a per-device mutex array. Two model arms back it (rulings §M9
    I-1, §Z6 I-C):
    - **A re-plans while B published last.** Model B's context publishes last, so the
      publication names B; then A's context re-plans through the wrapper. The transaction body
      evaluates A's plan (A's residency in its answer), takes A's live-update ticket, and
      publishes A's plan. RED: 7.7's exported entry with no model identity, driven for A's
      backend, which hands the body B's plan;
    - **`into_empty` installs the owner's plan.** Model A's cache is emptied while model B is
      the latest publisher; after `ggml_sycl_republish_current_plan_into_empty(A's cache, A)`,
      A's cache holds A's plan. RED: an `into_empty` that installs the current publication
      (7.7a's step 1, "reads the current publication"), which leaves B's plan in A's cache. A
      second arm drives the `:5887` site from a `ggml_sycl_mul_mat_id` dispatch of A's context,
      so the token threaded through `ensure_moe_secondary_queues_for_plan` is checked to be A's;
      a context with no execution root makes no call;
  - (aj) **no arena path reaches the device-global ring setters (llama.cpp-r7fz; rulings §M7
    I-4, §M38 I-1).** On the arena path, `release_pp_moe_onednn_scratch_ring` and the four
    device-keyed setters (`unified_cache_set_planned_pp_moe_onednn_scratch`, `_kv_zone_slots`,
    `_row_bytes`, `_n_ubatch`; `unified-cache.cpp:2164`, `:2217`, `:2253`, `:2278`) have no
    caller: the gate walks the call graph from the load's stages, the context transaction (its
    install included), the release proc and the batched executor, and fails on any path that
    reaches one of the five. The context's rows are written to its slot-table entry directly
    (§2.7). The weight-slot setter,
    `unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes(model, dev, bytes)`, is not on
    the list; it is reachable only from the load's step 3 (rulings §M32 I-3). Mutation
    witnesses: the `populate_inventory_globals` release restored; a call to
    `unified_cache_set_planned_pp_moe_onednn_row_bytes` planted in the commit's install (the
    r16 I-1 case, which 7.14g's text allowed); and the weight-slot setter called from a context
    transaction. Each must fail the gate;
  - (ak) **(0) takes no copies of the rows, and (d) drops the rows' retentions before (e)
    (rulings §M7 I-3).** Mutation witnesses: (0) copying the context's rows, and the move-out
    putting `retained_owners` into the batch;
  - (al) **the freed room is pending before (f) (rulings §M7 I-5(a)).** (0)'s placements are
    recorded as pending ranges before any release, and (ii) re-fits only inside them. Mutation
    witness: the recording moved after (f);
  - (am) **the MMID pools are materialized only inside the owning context's carve (rulings §M7
    I-5(b), §M8 I-5(a), §M9 I-2, I-3, §M11 I-D, I-E, §M12 I-3, m-6, m-13).** The gate
    enumerates every call site of
    `ggml_sycl_materialize_context_mmid_workspaces`, of the deleted
    `ggml_sycl_materialize_published_mmid_workspaces` and of
    `unified_cache_materialize_moe_mmid_workspaces`, and fails on any call site it does not
    classify, so it cannot pass by not looking at one. On the target tree the one site is step
    7, which receives its `ggml_backend_sycl_context &` as a parameter, takes its device pool
    from step 6's carve and its host pool from the context's held carve, and reaches no
    `unified_alloc`, `unified_allocate_owner` or pool growth; the wrapper, after the inner
    transaction, only checks that the registry answers `ALREADY_PUBLISHED` for this context;
    and the fit plans a pool slot only under `ggml_sycl_moe_mmid_route_reachable(ctx)`. GREEN
    arms:
    - an ordinary build, where the route is unreachable, plans no MMID slot and carves nothing
      for it;
    - in a route-testing build, two contexts of one model (one plan identity) each hold their
      own accepting registry entry, keyed by (token, context, device), with pools bound to
      their own queues, and each admits at its top rung with no cookie mismatch;
    - **C1 re-plans after C2 exists (I-D):** C1 is created, then C2, then C1 runs a growth
      republish; C1's pools are rebuilt on C1's queue (its `execution_context_id`), and C2's
      entry and pools are untouched;
    - **the first publish is not a bug (I-E(1)):** in a route-testing build, the constructor's
      first publish of a fresh context reports `PUBLISHED`, with zero `[CONTEXT-PLAN-BUG]`
      lines, and an unchanged republish with the same geometry reports `ALREADY_PUBLISHED`
      and rebuilds nothing;
    - **context destruction (I-E(2); §M12 m-6):** destroying C1 while a C1 dispatch holds a
      pool lease retires C1's entry after `recover` with `wait = false`, and the pool memory is
      released only by the last handle's destructor, after the lease drops (the blob's owner is
      shared with its slices); C2 still admits, and a C2 acquire issued during C1's destruction
      never waits on C1's recovery. A bundle modelled as not ready at destruction is reported
      as `[CONTEXT-PLAN-BUG]` naming the slot, and its entry is left for the unload's report;
    - **a failed first publish leaves no entry (§M12 I-3):** on C1's first publish, a failpoint
      at step 8 (a), one at the CAS (a lost CAS), and an exception from
      `prepare_plan_publication_locked` after step 7: afterwards the registry holds no entry
      for `(token, C1)`, the carve block is freed, and C1's retry plans and materializes its
      pools. RED: an entry accepting at insert (master's `!exact_plan || !has_active`), under
      which the retry's `needs_materialize` answers false and the pools outlive the plan;
    - **the binding authority (§M12 m-13):** with C2's backend unbinding (its binding
      `draining`, its `execution_context_id` copy zeroed at `:11877`), C1's multi-device
      lookup still binds C1's own backend on `owner_device`, read from
      `g_execution_backend_bindings`;
    - **load_end** leaves no MMID entry for the loading token.
    Mutation witnesses: master's allocation restored at `:18614`; master's `load_end`
    materialization restored at `:13181`, and the wrapper's at `:18980`, each of which the
    enumeration must report as unclassified; a fit that plans the slot without the predicate;
    the `get_backend_context_for_device` binding restored (C1's re-plan binds C2's queue,
    since `contexts.back()` is C2, and C1's next admit fails the exact-queue check); the
    `!stable_mmid` trigger restored (the first publish reports `[CONTEXT-PLAN-BUG]`); a
    registry keyed by plan identity (the second context's materialization dedups onto the
    first's entry, whose cookie fails its admission at `moe-mmid-workspace.cpp:1114`); a
    context destroy with no retire (the entry and its pools outlive the context); the recover
    at destruction with `wait = true` (the C2 acquire blocks on C1's registry lock); and a
    lookup through `g_backend_context_by_device` and the field copy (it returns no backend, or
    C2's, during the unbind);
  - (an) **the host tier allocates only at the first publish, one held carve per slot index
    (rulings §D15, §ZR5 I-1, I-2; r7 I-6).** Each host slot is its own contiguous owner-first
    reservation, sized at that index's maximum over every rung, and no tenant-only path reaches
    a host allocation; no host-slot path calls `allocate_segmented` or `zone_alloc_segmented`.
    The path is the one §2.4.3 cites (`try_zone_alloc_contiguous`, then `grow_zone` by one chunk
    sized to the request). Both pool phase gates read
    `ggml_sycl_replan_token_held(TRANSACTION)`, the one accessor, declared in
    `unified-cache.hpp`; no other definition of a held flag exists in the tree; and nothing in
    the transaction writes `g_offload_phase`. Mutation witnesses: 7.5's per-republish
    allocation with `forbid_host_zone_growth = true`, 7.6's single carve with offset views, a
    slot routed to `zone_alloc_segmented`, a gate that ignores the token, and a gate that asks
    for any kind (7.7a's form, which exempts a load);
  - (ao) **per-model plan state (rulings §M76a).** On every L0-entry path (the transaction
    body, the probe, the FA recheck, activate, unload and the load entries), no read of
    `g_tensor_inventory_*`, `g_placement_kv_info` or `g_model_n_layer` exists outside the load
    entry that writes the same model's record (populate followed by the plan computation, in one
    hold); the transaction and the probe read the inventory through
    `lifecycle_select_placement_plan(model)`; the preload reads `n_layer` from its bound
    candidate; and no publish writes a device-global. The gate takes its path set as a list, so
    llama.cpp-fsgi adds the dispatch paths (`get_cached_tensor_ptr`, `:19236`) to it without
    changing the check. Mutation witnesses: the `:18245` read of `g_tensor_inventory_detail`
    restored in the MMID re-plan, the `:15347` write restored in the publish, and the preload's
    `g_model_n_layer` read restored;
  - (ap) **a load's state is its own (rulings §M8 I-3, §M9 I-4, F2, §M11 I-A..I-C; §X7 I-3; §M12
    C-1, I-1, I-2, m-3, m-8; §X9 I-D, §M13).** The load flag is keyed by the load's identity:
    the host-buft role and the caps' `async` ask whether the calling thread is bound to an
    active load. The **early** stage (`llama-model.cpp:657`) admits B's plan and records its
    device weight bytes as load pending ranges, `{LOAD, txn}` with term `WEIGHT`, before it
    returns; the late stage (`:709`) consumes that plan by identity. The loader's SYCL<n> weight
    buffers draw first from those ranges, and `load_end` retags `MODEL_TERM` and `WEIGHT` to
    `{MODEL, id}`, then clears the rest. Arms:
    - `load_begin` on thread T1 and `load_end` on T2 (the candidate API's re-bind): during
      `load_end`'s preload the predicate answers **not in load**, as on master (the Registry's
      `in_load` bit is cleared before the preload, §2.4.2 F2), and after `load_end` T1's
      SYCL_Host allocations (weights not evictable) are `STAGING` and its caps report `async`
      true; a context constructed on a third thread during B's load reads `async` true
      (`llama-context.cpp:997`);
    - **a transaction between the stages (§X7 I-3):** A's transaction runs between B's `:657`
      and `:709` and fills every free byte it can; B's late stage then finds its admitted plan
      by identity, with zero `TERMINAL` misses and no device plan placed on the host. The arm
      records B's `WEIGHT` ranges and, since 23mk records its `SCRATCH` and `MODEL_TERM` holds
      at the early stage (23mk rev 4.5; §M12 m-12), those too, and expects no `TERMINAL` miss in
      23mk's holds either. No `DEVICE_TERM` range exists at any point (rulings §Z10.1);
    - **a late placement change is refused:** a modelled late inventory that moves one tensor to
      another device or tier gets the named refusal ("the late inventory changes the placement
      admitted at the early stage"), and nothing is re-planned;
    - **the create set, duplicates included (rulings §M18.1; r13 C-1):** a tied-embedding
      fixture with gemma4's shape (no `output.weight`; `token_embd` requested again with
      `TENSOR_DUPLICATED`), the input layer on the host and the output layer on the device, run
      through the record pass on the CPU (llama's own test-model generator, as
      `test-llama-archs` builds its models; no load, no GPU). The record holds the tied head as
      a second entry at site `OUTPUT`; the early plan admits it on the device with its bytes,
      B's `WEIGHT` range includes them, and the late check finds every SYCL-backed tensor at an
      admitted entry, with no refusal. The same fixture split over two devices shows one
      `rope_freqs` entry per device its layers span. RED: 7.13's `ml.weights_map` inventory. Its
      one outcome is the §M18.1 refusal, `[LOAD-PLAN] tensor token_embd.weight (site OUTPUT)
      exists in buffer type SYCL0 on device 0 but has no admitted entry (refused)`; that refusal
      is a load refusal compiled and enabled in every build, so the outcome is the same whatever
      `GGML_SYCL_WITNESS_CHECKS` says (r13 m-c). The g4 logs (720 early, 721 late, +680 MB) are
      the evidence that the defect is real, not a control. The lead's confirmation is the three
      gemma4 merge gates on the B70 (`g4-np1`, `g4-np4`, the server's default-parallel), which
      must load;
    - **user placement is a fixed input (§M12 m-8; rulings §M14 I-2, §M18.2; r13 I-B):** a model
      with a host-tiered layer whose tensors are `CPU_REPACK` and absent from the late inventory
      (`llama-model.cpp:680-684`) loads with no refusal; so do the user-placement fixtures `-ot
      <pat>=CPU`, `-ncmoe` and a partial `-ngl`, each once under `--no-host` (pinned, since the
      partial `-ngl` case is forced only there) and once in default mode (where the forced
      tensors are SYCL_Host tensors in the late inventory). In each, B's `WEIGHT` range excludes
      the forced tensors' bytes, the late stage packs nothing and moves no forced tensor, and
      under `--no-host --cpu-moe` on a MoE fixture the charged `moe_onednn` is 0 on every
      device, no shrink WARN is logged and `late_term_shrink_admitted` stays 0. The same absence
      for a tensor the admitted plan put on a device is the named refusal. The source-contract
      gate finds the override match once, in `resolve_create_site`. Witnesses: a comparison that
      reads an absent tensor as a tier change (it refuses the host-tiered model); 7.12a's early
      inventory without the forced flag (it refuses the user-placement fixtures); 7.13's
      inventory-wide `moe_onednn` (early 564035584 B, the weight slot, late 0: one shrink
      WARN, the counter at 1; 1617.9 MB before 7.14e moved the ring's rows to `ring`);
      and a late re-pack in default mode, which puts a forced expert on the device and gets the
      placement refusal;
    - **the zones come from the demands, in five steps (rulings §M14 C-1, §M17, §M17a, §M20; r13
      I-F, m-b, m-c):** positive cases from the merge-gate logs, with the terms re-derived by
      stepping the sizing code and **scored in exact bytes, never one-decimal MiB** (§M20.1; the
      byte derivations are §2.4.2 (b)'s "end states"). GPT-OSS 120B on the B50
      (`merge-gates/gptoss120b-b1.log:57`, `:184`, `:191`, `:204`, `:261`): step 1 ensures the
      placement-independent terms; the pack charges `moe_onednn` at the first oneDNN-PP expert
      on the device, the weight slot 564035584 B, which equals the inventory-wide field on this
      uniform shape (r14 m-7), `moe_ptr_table` k × 1024 B with k read from the plan (rulings
      §M23 (2)), and `ring`, `moe_control` and `mmid_workspace` no zone term (class C; their
      first-context values go to the reservation, below); step 4, sized from the ledger, grows
      RUNTIME from 0 (no floor on an arena device, rulings §M32 I-2) to 564035584 + k × 1024 B
      and leaves SCRATCH at 536870912 B and ONEDNN at 268435456 B, their floors (rulings §M25
      I-1; ONEDNN's one D term is 23mk's `onednn_pp_w`); the weight zone is 13958742016 − k ×
      1024 B; `plan.weight_vram_bytes` plus the device's `FIRST_CONTEXT` reservation is at most
      it; B's ranges fit; and the late stage rebuilds nothing and refuses nothing.
      Qwen3.5-35B-A3B on the B50 (`glkg-qwen35b-a3b-b1.log:73`, `:210`, `:217`, `:291`):
      `moe_onednn` 142606336 B, RUNTIME 142606336 + k × 2048 B, weight zone 14380171264 − k ×
      2048 B, with the same result. 7.14e's Qwen figures (RUNTIME at its floor, weight zone
      13985906688 B) are this arm's RED for the floor's removal: a tree that still lays
      RUNTIME's floor out ensures 394264576 − k × 2048 B of RUNTIME that no term charged, and
      the exact RUNTIME bytes catch it. A scorer that compares the one-decimal display is the
      arm's own RED on 120B: the display moves with k (§M20.1), so a correct tree need not read
      13312.1 there. A context on either model at `-ub 512` then places the ring's rows as head
      slots of its fit, 1132462080 B / 1610612736 B (§2.7), taking over the reservation, and no
      zone grows (the first-context arm, below, scores it). The witness names
      `weight_vram_bytes`; 7.14d's RED that read `vram_bytes` (r13 m-b) is withdrawn, since its
      false fire needed master's MMID charge in `vram_bytes` (12343.8 MB) against a 12232.1 MB
      zone, and §M27 (3) removes that charge while §M28 (1) moves the zone to 13312.1 MB, which
      7.14f keeps on 120B (RUNTIME is above the removed floor there).
      **Which check
      catches which mutant (rulings §M25 I-2):** the additive composition (a floor added as a
      term, 7.14's `compute_arena` read additively) is caught by the packed-equals-ensured
      witness, which recomputes `max(floor_Z, demand_Z + charged_Z)` from the ledger and names
      SCRATCH, and by the exact SCRATCH bytes; `mmid_workspace` charged as a RUNTIME D term (r14
      I-4) is caught by the exact RUNTIME bytes alone, since the getters' RUNTIME requirement
      (`unified-cache.cpp:1566-1575`) carries no MMID and a grow-only dry run cannot see an
      over-charge; so is `ring` charged at load as 7.14b to 7.14d did (RUNTIME 1696497664 + k ×
      1024 B on 120B, 1753219072 B on Qwen); the forgotten term, the C-rule mutant and the
      two-device RED below are caught by the dry run. RED, scored as one outcome per switch, on
      **both merge-gate models**: pack before ensure (7.12a's order), which packs at the
      pre-plan 14522777600 B zone (RUNTIME laid out at 0), 564035584 + k × 1024 B more than the
      ensured one on 120B and 142606336 + k × 2048 B on Qwen; both models fill the zone, so the
      packed weights exceed the ensured zone by those bytes. 7.14e voided it on Qwen, where
      RUNTIME sat at its floor in both orders; with the floor gone it is live there. With
      `GGML_SYCL_WITNESS_CHECKS=1` its one outcome is the packed-equals-ensured witness's
      message, since the witness runs before any recording; with `=0` it is the early stage's
      placement refusal. Master is a real RED for "no late rebuild": its late stage rebuilds
      RUNTIME from 1617.9 to 1618.0 MB (`gptoss120b-b1.log:1544-1545`);
    - **the floors' removal (rulings §M37 Q3; pre-registered before the lead's run):** an arm
      of the commit that removes each floor, on the same two fixtures. **ONEDNN**, from the
      later of L4+L6 and the Graph-scratch commit: the pre-plan split is 14791213056 B (ONEDNN
      and RUNTIME laid out at 0), and after step 4 ONEDNN is exactly W, 23592960 B on
      GPT-OSS 120B and 33554432 B on Qwen, with weight zones **14203584512 − k × 1024 B** and
      **14615052288 − k × 2048 B**. RED: a tree that keeps the floor ensures 244842496 B /
      234881024 B of ONEDNN that no term charged, and the exact ONEDNN bytes catch it; a
      scorer that reads 256.0 MB there is the arm's own RED. **The arm reads the order first
      (7.14h):** it asserts whether the other commit is in the tree and scores that state.
      In the earlier commit ONEDNN is W plus the named terms of the consumers still drawing it,
      never 268435456 B: with L4+L6 first, W plus the Graph scratch at the envelope's `-c
      4096`, 23592960 + 0 B on GPT-OSS 120B (sinks) and 33554432 + 201326592 B on Qwen; with
      23mk's commit first, W plus the terms 23mk's arm names. Its RED is the constant kept.
      **SCRATCH**, from the commit
      that converts the last untermed census row (the end states): after step 4 SCRATCH is
      exactly the ledger's SCRATCH terms, and `ggml_sycl_compute_arena_bytes()` returns that
      sum. Its precondition, asserted first or the arm is VOID: the census gate finds no
      SCRATCH draw without a term. RED: the floor kept, which ensures 536870912 B less the
      terms that no term charged. Both arms also run the dry-run witness, which must grow
      nothing;
    - **the model's first context places its C head slots (rulings §M32 C-1; pre-registered
      before the lead's run):** vehicles `gptoss120b-b1` and `glkg-qwen35b-a3b-b1`, the B50
      merge-gate shapes. **Scored by the pure-fit run**, not by a load: H2's `kv_region_fit`
      on the geometry the load recorded (its `WEIGHT` ranges and its `FIRST_CONTEXT`
      reservation, from the host replay of the pack), for the model's first context at the
      gates' shape, `-c 4096 -ub 512` (`merge-gates/run-merge-gates.sh:65`). Its precondition
      is fkpg (a): the envelope carries 4096, and a run whose load plan prints `n_ctx=512` is
      VOID, which fails (rulings §M38 I-5). **Pre-registered per term (rulings §M38 C-1), in
      bytes, `gptoss120b-b1` / `glkg-qwen35b-a3b-b1`:**
      - the ring rows: 1132462080 / 1610612736 (§2.7 at `-ub 512`; independent of `n_ctx`);
      - zhcn's compute slot: 423624704 / 516947968, master's SYCL0 compute buffers at this shape
        (404.00 MiB, `gptoss120b-b1.log:2171`; 493.00 MiB, `glkg-qwen35b-a3b-b1.log:2889`),
        labelled fixture constants until zhcn's load-time measure lands (below);
      - the oneDNN Graph scratch: 0 / 201326592. GPT-OSS's attention has sinks, which the oneDNN
        SDPA route rejects (`fattn-onednn.cpp:115-116`), so its term is 0 under its dispatch's
        gate; on Qwen it is 1.5 · 16 · 512 · 4096 · 4 (`n_head_ctx_max` 16,
        `glkg-qwen35b-a3b-b1.log:229`, where at the planner's 512 it is 24 MiB, clamped to the
        64 MiB minimum the log prints);
      - the recurrent state: 0 / 65863680, 30 recurrent layers × 4 B × (3 · (4096 + 2 · 16 ·
        128) + 128 · 4096) at `n_seq_max` 1 (`glkg-qwen35b-a3b-b1.log:154-158`; printed as
        62.81 MiB at `:2858`);
      - `onednn_pp_a`: 4194304 / 4194304, 23mk's 512 × `max_K` × 2 with `max_K` 4096 on both
        (GPT-OSS's `attn_output` Q8_0 4096 × 2880; on Qwen a 4096-wide plane-W tensor; 23mk's
        `gguf_w2.py` rule over each gate's model file);
      - `moe_control`: 16640 / 33024;
      - `set_rows_stage`, `onednn_scratchpad` and `mmid_workspace`: 0 each (one device; §M31a;
        the MMID route is unreachable in an ordinary build, step 7);
      - jzvq's fattn workspaces and MXFP4 MoE TG caches, from jzvq's exact demand functions
        (llama.cpp-jzvq closes before L4, §2.4.3). The TG caches are 0 on Qwen, which has no
        MXFP4 experts. On GPT-OSS 20B at `-c 4096` master's baseline draws 3240, 184320, 23040
        and 12960 B (llama.cpp-jzvq comment c-jv0r). The arm takes jzvq's values at this shape
        from jzvq's host test before the lead's run, and without them it is VOID, which fails.

      The known terms sum to 1560297728 B (1488.0 MiB) / 2398978304 B (2287.8 MiB), plus jzvq's.
      GREEN: the load's reservation equals the first context's head-slot set term by term, and
      the fit places every head slot in the reservation plus the free `REGION` room, demotes KV
      only, and does not refuse. REDs: the 7.14e pack, whose capacity is the whole `WEIGHT` zone
      with no reservation; its weights fill the zone, and the fit refuses the context, naming
      the ring head slot (r15 C-1). **A second enumeration (rulings §M38 C-1):** the load
      reserving from 7.14g's hand list while the fixture's producers include the recurrent
      state; the per-term comparison finds that term's bytes missing from the reservation, and
      the source gate fails too, since the reservation site names no cohort or term. **The 512
      fallback:** a reservation evaluated at `n_ctx` 512, which on Qwen is short by 201326592 −
      67108864 B of Graph scratch and is caught by that term. Handover and rollback, on the same
      fixture:
      - the context's step 5 records its placements and clears the `FIRST_CONTEXT` range in one
        group-mutex section; a failpoint just before and just after that section finds the
        reservation whole or fully taken over, never both nor neither;
      - a refused first context (a failpoint at step 6) leaves the reservation re-recorded by
        guard phase 1, in the same section that clears its `REGION` ranges, so the next context
        of the model finds it; RED: a guard that clears `REGION` and leaves the reservation
        cleared, after which the retry is refused naming the ring;
      - a second context of the model finds no reservation and fits in the free room;
      - the model's unload clears `WEIGHT | FIRST_CONTEXT`, and no `FIRST_CONTEXT` range
        survives it.

      **zhcn's compute slot at load (rulings §M37 Q1, §Z20).** zhcn runs its measure pass once
      at the load envelope, at n₀ and the envelope's `n_ctx`, and that value sizes the
      reservation's compute slot. Until that lands, the arm injects the figures above as fixture
      constants,
      labelled as a fixture constant in the arm's output, and the plan line says whether the
      reservation carried a measured compute slot. Without one, the reservation is short by that
      slot, and where the pack filled the zone (both merge gates) no KV demotion frees room for
      it, so the fit refuses the context naming zhcn's compute slot: that outcome is the open
      item, scored as FAIL with that name, never as GREEN. The lead's confirmation is the two
      gates reaching context creation with rc=0 and the plan line printing the rows inside the
      reservation;
    - **the enum is closed, and a forgotten term is caught (rulings §M18 I-A; r13 I-A):** on the
      correct tree the post-pack dry run of `ensure_planned_arena_zones` is a no-op on every
      device in `plan.devices`. **A dropped term must lift its zone above the zone's minimum on
      the fixture** (the ensure takes `max(floor, demand + charged)`), or dropping it grows
      nothing and the mutant passes silently. From 7.14e the mutant drops **`moe_onednn`**. With
      RUNTIME's floor gone on an arena device (rulings §M32 I-2), its minimum is 0, so both
      merge-gate shapes are fixtures: the weight slot lifts RUNTIME from 0 to 564035584 + k ×
      1024 B on 120B and 142606336 + k × 2048 B on Qwen (the end states), asserted before
      scoring, or the arm is void. **The dry run's input carries no RUNTIME default (rulings
      §M38 I-3), with a Mistral positive control:** Mistral 7B Q4_0 on the B50 shape, whose
      RUNTIME is 0 on both sides (no MoE, the pipeline off, every transitional term 0), grows
      nothing; on Qwen the dry run's RUNTIME equals step 4's, 142606336 + k × 2048 B. RED:
      master's 512 MiB default left in `ensure_planned_arena_zones`
      (`unified-cache.cpp:4505-4508`), under which the dry run reports RUNTIME growing by
      536870912 B on Mistral and by 536870912 − 142606336 − k × 2048 B on Qwen, both correct
      trees. **Mutant, against the term's own setter (rulings §M32 I-3):**
      step 3 drops `moe_onednn` from the ledger while the load still publishes it through
      `unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes`, which the RUNTIME getter sums
      (§2.4.2 (b) step 5). 7.14e named master's
      `unified_cache_set_planned_pp_moe_onednn_scratch` call, which H7z keeps off the load path,
      so under H7z no load-time getter carried the slot and the mutant could not fire (r15 I-3).
      The ledger lacks the term, so step 4 sizes RUNTIME at the table bytes and the
      packed-equals-ensured witness passes, since it checks step 4 against the same ledger; the
      dry run sizes RUNTIME from the published getter, which still carries the weight slot, and
      its witness fires, with `GGML_SYCL_WITNESS_CHECKS=1`, as its one outcome, naming RUNTIME
      and 564035584 B on 120B, 142606336 B on Qwen (the whole slot, since no floor absorbs part
      of it; 7.14e's 27164672 + k × 1024 B was the part above the floor). The load publishes the
      ring's activation and output slots as 0, the C rule (§2.4.5), so the correct tree's dry
      run is a no-op. 7.14d's mutant dropped `onednn_scratchpad`, which is class C from §M29a
      and 0 at the load stage, so it moved again. **The C rule's fixture** (below) carries an
      eligible weight whose W alone is at least 256 MiB (N × K × 2 ≥ 268435456 B, for example K
      = 8192 and N = 18432), so 23mk's `onednn_pp_w` lifts ONEDNN above its minimum; asserted
      before scoring, or the arm is void. The same precondition binds every dry-run RED below,
      SCRATCH's minimum being 512 MiB. 7.14's mutant dropped `onednn_graph_scratch`, which is
      class C since §Z15 and is 0 at the load stage, so it moved; 23mk keeps its own
      `moe_ptr_table` mutant, and both stand. **The C rule** (§2.4.5), an arm of the
      Graph-scratch commit (rulings §M26a I-4; before it the load stage reads the with-floor
      getter by design, and the mutant would be the tree): the mutant restores the with-floor
      getter (`:2148-2160` at master) and points the load-stage reads at
      `unified-cache.cpp:4464` and `:27658` at it (r14 m-15 (a)). Step 4 is sized from the
      ledger, which charges no C term, so only the dry run reads the restored getter. That
      getter is the W getter's value plus the Graph-scratch floor G (`:2152-2160`, added only
      under `onednn_graph_allocator_enabled()`), and **that value is W** from 23mk's re-point
      (rulings §M32 I-4), so the dry run sizes ONEDNN at W + G against the ledger's `max(256
      MiB, W)` = W, and the witness fires exactly when G > 0. **Two preconditions, asserted
      before scoring, or the arm is VOID (r15 I-4):**
      `unified_cache_get_planned_onednn_pp_w_bytes(dev) > 0` (the fixture's W ≥ 256 MiB, above),
      and G > 0, with the Graph allocator on and the fixture's attention shape published.
      7.14e's arm deleted W's source, so the W getter read 0 and the mutant fired only if G
      exceeded W, which the fixture did not assert. With both held it fires the same witness on
      the same fixture, naming ONEDNN and G's bytes. (Citing the getter's body alone would
      mutate code the load path no longer executes, and the arm would be void.) On the correct
      tree the same fixture's load-stage ONEDNN zone is exactly `max(268435456, onednn_pp_w)` B
      (rulings §M29a), and its weight zone is larger than master's by exactly master's ONEDNN
      zone minus that, where **master's zone is read from master's own ensure on the same
      fixture**, clamp included (`min(with_floor, max(available / 4, stored))`,
      `unified-cache.cpp:4476-4500`), never the unclamped with-floor sum (r14 m-15 (b)); the arm
      also asserts master logged no clamp WARN (`[VRAM-ARENA] planned ONEDNN zone ... clamping`)
      on the fixture, or it uses the clamped figure. Both are scored in bytes (rulings §M21.3);
      on the merge-gate shapes that difference is 0 B (§2.4.2 (b), "The end states"). A context
      on the same fixture then places the graph scratch once, as a head slot inside its `REGION`
      ranges, and no zone grows after the load (rulings §M25 I-6; 23mk's transaction does the
      charge, and this arm asserts only the room). **The placement witness is 23mk's commit line
      (rulings §M30, §V16a):** `[CONTEXT-PLAN] graph scratch range: ctx=%u dev=%d
      term=ONEDNN_GRAPH_SCRATCH backing=%s offset=%zu bytes=%zu`, printed by 23mk at its
      context-transaction commit, whose offset lies inside the context's `REGION` ranges and
      whose bytes equal 23mk's value function; a missing line makes the arm VOID, never a pass.
      The range is recorded at step 5 under its own `ONEDNN_GRAPH_SCRATCH` pending term, not
      carved at step 6 (§2.4.2). 7.14 pre-registered this mutant with 96 MB on GPT-OSS's shape,
      where it cannot fire, for the reason above. The source-contract gate fails on a
      `unified_cache_set_planned_*` setter with no enum value;
    - **a secondary device is charged its attention shape (r13 I-F (5), m-d; rulings §Z15):** a
      two-device plan whose `dev_layer` puts attention layers on device 1. Step 2 ensures device
      1 before the loop packs, the pack charges `nonfa_shape` to device 1 at the first attention
      layer it places there, neither device is charged `onednn_graph_scratch` at the load stage
      (class C, 23mk's context transaction charges it), and the witnesses pass on both devices.
      **RED (rulings §M25 I-3; the minimum precondition above, the device-1 `nonfa_shape`
      demand exceeding the 512 MiB SCRATCH floor on the fixture):** the pack charges
      `nonfa_shape` for `plan.device_id` only, master's rule, while the plan publishes the
      shape per device. Device 1's ledger lacks the shape, step 4 leaves its SCRATCH at the
      floor, and the dry run, reading device 1's published shape, fires on device 1 naming
      SCRATCH. 7.14a's RED, a `plan.device_id`-only *publication*, is withdrawn: with a correct
      charge it leaves the dry run wanting less than was ensured, and a grow-only dry run
      cannot see an under-publication (r14 I-3);
    - **`--no-host` with the experts on the host (r13 I-F; rulings §M18.2):** a MoE fixture on
      one device under `--no-host --cpu-moe`. The pack charges `moe_onednn` 0 and
      `onednn_scratchpad` nothing (class C, rulings §M29a), and still charges
      `nonfa_shape`, since the device runs every attention layer; the dry run is a no-op, the
      late stage logs no shrink WARN, and `late_term_shrink_admitted` stays 0. REDs: 7.13's
      inventory-wide `moe_onednn` (one shrink WARN, the counter at 1), and a `nonfa_shape`
      charged at the first expert instead of the first attention layer, under which the device
      is charged 0 and the dry run fires, naming SCRATCH (the minimum precondition above);
    - **`moe_onednn` is the device's own maximum (r14 m-7):** a two-device fixture with two
      expert shapes, where device 1 holds only experts of the smaller shape. Device 1's charge
      is its own maximum, stepped per tensor from `src/llama-model.cpp:441-447` and scored in
      bytes, less than the inventory-wide field. RED: the inventory-wide field, under which
      device 1 is over-charged by the difference, caught by the exact bytes (a grow-only dry
      run cannot see an over-charge);
    - **`mmid_workspace` is class C (rulings §M25 I-4):** on a MoE fixture the load stage
      charges it nothing, and a context's transaction places the MMID device pool once, as its
      `REGION` head slot; RUNTIME holds no MMID bytes. RED: `mmid_workspace` as a RUNTIME D
      charge, under which RUNTIME exceeds its exact bytes by the pool's bytes and the pool is
      placed a second time in the context's range; the exact bytes catch it (above);
    - **`moe_control` is a C slot, and non-table only (rulings §M27 (2a), §M26a I-1, §Z15):**
      on GPT-OSS 120B's and Qwen's shapes the load's ledger and the RUNTIME requirement
      (`unified-cache.cpp:1567-1575`) carry no `moe_control` bytes and no table bytes but
      `moe_ptr_table`'s, and row 134's block is k × stride. `expert_ptrs` is in no RUNTIME
      term; its host-sum reader is s4ip's and is not asserted here. A context at `-ub 512`
      places one `moe_control` slot on each device that executes a GPU expert, 16640 B and
      33024 B, as a head slot inside its `REGION` ranges; a context at `-ub 2048` places
      65792 B and 131328 B; two contexts of one model each place their own; a device with no
      GPU-executed expert places none. The sizes are stepped from the ungated layout (§2.4.2
      (b)), never from `moe_control_requirement_from_layout`. REDs, each of which fires on its
      mutant and not on the correct tree (rulings §M26a I-1):
      - **the slot charged 0 while the list is still taken from it** (the "charge on at 0" RED,
        restated for a C slot): the value function returns the gated requirement, `{}` while
        `GGML_SYCL_MOE_CONTROL_CONSUMER` is 0, so the fit places no slot, while the dispatch
        still takes the list from the context's slot. The lookup finds no slot and refuses with
        the named `[CONTEXT-PLAN-BUG]` miss at the first MoE op. The correct tree, on the same
        fixture, places the slot and its first MoE op takes the list inside the slot's carved
        range; the arm asserts that first and scores the mutant's refusal only then. A mutant
        under which both trees refuse discriminates nothing, and the arm is then void;
      - **the slot sized at the plan's 512 on a `-ub 2048` context** (7.14c's m-6 sizing):
        16640 B against the correct 65792 B on GPT-OSS, caught by the exact slot bytes;
      - **the list and flag left in row 134's block** (the block sized k × stride plus the
        non-table parts, 7.14c's 24832 B / 49408 B): RUNTIME exceeds its exact bytes, caught
        by the exact bytes;
      - **the ids staging charged in the slot** (rulings §M26 I-3): the slot exceeds its exact
        bytes by align256(n_expert_used × n_ubatch × 4), 8192 B / 16384 B at `-ub 512`, which
        row 75's term also counts; caught by the exact slot bytes;
      - **a context whose slot cannot be placed:** a fixture whose `REGION` headroom, with
        every KV layer on the host, is below the slot plus the context's other head slots. The
        correct tree refuses the context by name at its transaction, before any decode. RED:
        the fallback at `mmvq.cpp:16614` kept, which admits the context and allocates the list
        unplanned at the first MoE op, caught by the unplanned-claim STRICT (§G1) abort;
    - **`ring` is C, per context, at each context's own `n_ubatch` (rulings §M28 (1), §M32
      I-1, C-1 (b)).** Split in two (r15 m-5 (b)), because the executor needs a GPU:
      - **host arm** (this suite): on GPT-OSS 120B's and Qwen's shapes the load's ledger charges
        only the weight slot to RUNTIME (the end states' bytes). A context at `-ub 512` places
        its activation and output rows, 377487360 B + 754974720 B and 536870912 B +
        1073741824 B, as its own CONTEXT head slots inside its `REGION` ranges, never in
        RUNTIME; a context at `-ub 2048` places 4529848320 B and 6442450944 B in all, or is
        refused by name at its transaction. The admission is scored over the pure
        `pp_moe_onednn_admit_scratch` with the executor's own shape formula (the required
        weight, activation and output bytes of `ggml-sycl.cpp:78703-78782`, lifted into a pure
        function the executor also calls) against the planned shape the context's slot table
        and its model's weight slot carry: a `-ub 500` context's rows are align256(512 ·
        per-row), and a 500-token micro-batch with every expert active and one expert holding
        all 500 rows is admitted. Two contexts on one device at `-ub 512` and `-ub 1024` each
        hold their own rows, 1132462080 B and 2264924160 B on 120B, and each one's
        micro-batches are admitted against its own rows, whichever transacted last; closing
        the `-ub 1024` context releases its rows and leaves the other's. **The resident count
        (rulings §M32 C-1 (b)):** a non-uniform fixture with two expert shapes and a partial
        layer, where the device holds 64 of a tensor's 128 experts: the weight slot and the
        rows are stepped from `local(t)`, and the executor's required weight bytes for the
        largest admissible batch (every resident expert active) equal the slot. REDs: the rows
        sized at `n_ubatch` (master's `unified_cache_pp_moe_onednn_slots_for_ubatch`), under
        which the `-ub 500` micro-batch is refused at admission, caught by the refusal's
        reason; the planned shape read per device (master's executor, before the re-key),
        under which the `-ub 1024` micro-batch after the `-ub 512` context's transaction is
        refused the same way; the value at `n_expert` on the partial-layer fixture, over by
        the non-resident experts' bytes, caught by the exact bytes; and the rows charged at
        load (7.14b to 7.14d), caught by the exact RUNTIME bytes;
      - **GPU arm, lead-run (§3.3), with an engagement witness (r15 m-5 (a); rulings §M38
        I-5):** the refusal WARN's absence is not evidence, since a run that never reaches the
        executor also prints none and the INFO reject log is dropped at default verbosity. The
        executor counts admitted and refused batched dispatches per (context, device), in every
        build, and prints one line per context at teardown at **WARN**, which survives default
        verbosity: `[PP-MOE-RING] ctx=%u dev=%d batched_admitted=%zu refused=%zu`. 7.14f made it
        a `GGML_SYCL_PRIVATE_TESTING` counter, which is compiled only into test targets and
        never into the `ggml-sycl` library that `llama-completion` loads, so the arm was VOID on
        every run, a correct one included (r16 I-5). The arm requires `batched_admitted` ≥ 1 per
        context. Two contexts of GPT-OSS 20B on the B50 at `-ub 512` and `-ub 1024`, pinned to
        `level_zero:1`, each run one prompt; each line prints `batched_admitted` ≥ 1 and
        `refused=0`, each plan line prints that context's own row bytes, and the refusal WARN
        count is 0. **Positive control:** the `-ub 512` context alone must print ≥ 1, or the
        arm is VOID. **Negative control:** a Mistral 7B context prints the line with
        `batched_admitted=0`, so the line's presence is not the engagement;
    - **`onednn_scratchpad` is one buffer per (context, queue), at its planned maximum (rulings
      §M29a, §M31, §M32 I-5, §V16 I-A).** Host arms over the pool with a stub descriptor table
      (the sizes are injected, since a `primitive_desc` needs the device's engine):
      - **one buffer, never grown:** two growth steps on one queue, both under the planned
        maximum, are served inside the one buffer. RED: today's retention, which keeps both
        buffers;
      - **a request above the plan:** the named `[ZONE-PLAN-BUG]`, a STRICT (§G1) abort,
        otherwise the non-oneDNN decline with a WARN and the op still runs. RED: a silent grow;
      - **two contexts at different `n_ubatch` (§V16 I-A):** each holds its own buffer at its
        own planned size, and `assert_scratchpad_planned(stream)`, which replaces
        `pre_allocate_scratchpad` and has no size argument, only asserts the buffer exists at
        that size (a planned 0: that the value is 0). RED: master's global pre-size from
        `get_max_scratchpad_size()`, which sizes the smaller context's recording from the
        larger one's descriptors;
      - **a climb (§V16 I-A):** the larger buffer is placed as a new head slot at the climb's
        transaction and swapped into `scratchpad_map[q]` at the publish, and the old one drops
        at the publish, released only after its last event. RED:
        an in-place grow that frees the old buffer at once, caught by a queued-work stub whose
        event is incomplete at the free;
      - **the queue set:** a claim on a queue outside the context's enumerated set is
        `[ZONE-PLAN-BUG]`; the set is the context's, frozen at its transaction (rulings §M35);
      - **the source gate:** every `get_scratchpad_mem` call site maps to one family, eleven
        live sites at `c69d5774d` once 23mk deletes the two dead wrappers (rulings §M36 I-3,
        §M38). REDs: an unmapped call site added, and a call site restored in
        `DnnlBinaryWrapper::binary` or `DnnlReductionWrapper::reduce_last_dim` with no family,
        each of which fails the gate.

      Lead-run (§3.3): the load's table on the B50 and the B70, over every family under its
      gate, equals the probe's rows (0 at `c69d5774d`, rulings §M31a), with the probe's
      `reduction_sum` anchor as the positive control that the query reads non-zero sizes;
    - **a planned copy has one byte source and one class (rulings §M18.3a, §M18 I-C; r13 I-C,
      I-D):** a WOQ and MoE-alternate fixture with both classes: OPTIONAL dense WOQ copies of
      device-resident primaries, and a PRIMARY cross-device alternate. Each staged copy's bytes
      equal its admitted `dst_size` (the staging witness); a copy whose adjusted layout
      collapses to the primary's is absent from the plan and from the ranges; the cross-device
      alternate is PRIMARY and draws from its target device's range; a KV transaction that
      needs room never picks a PRIMARY copy; eviction never selects any planned copy, of either
      class; and a staging of a copy the plan does not list is refused with `[ZONE-PLAN-BUG]
      staging a layout copy the plan does not list`, the copy not staged. REDs:
      `mark_optional_layout` on a PRIMARY copy, which the yield then takes; the staging
      re-deriving `dst_size` after the adjustment, which trips the staging witness on the
      XMX-tiled fixture; and the unplanned branch counted and staged, as
      `dense_woq_staged_unplanned` is today;
    - **a yielded OPTIONAL copy's extent is drawn by that context's KV (rulings §M18.3a):** the
      same fixture, then a context whose KV needs exactly the OPTIONAL copies' room beyond the
      context side and the weight holes. The transaction picks the OPTIONAL copies, and the
      yield path, per pick, records the block's extent as `{CONTEXT, c}` / `REGION`, marks the
      key yielded in the plan and moves the handle to the drop list, all under one group-mutex
      section. After the finish the freed bytes lie inside `pending_ranges(c, d, {CONTEXT, c},
      REGION)`, step 6 carves KV extents in them, no layer is demoted, dispatch for the yielded
      key reads the primary's layout (H7h), and the model's unload frees nothing of the extent
      and records no range. A PRIMARY copy in the same TLSF is never picked, and the KV the
      OPTIONAL copies cannot cover demotes. RED: the yield path skipping the retag. The freed
      blocks return to the general TLSF free lists outside every range, a modelled unowned
      allocation between the finish and step 6 takes one, the re-fit sees the shortfall, and
      KV demotes; the arm reports the demoted layers and the extent outside `own_ranges`. A
      second RED releases the handle before the retag, in a separate section, and the same
      unowned allocation lands in the gap. The lead-run a1_long gate (B50, PCT 60, `-c 2048`)
      pre-registers, unchanged unless C8's premise (ii) says otherwise (§3.3), "KV admission
      released 6 optional oneDNN WOQ layout copies
      (220.5 MB) ... for n_ctx=2048's KV";
    - **a late term change follows §Z13.1, and a rebuild never meets a range (§M12 C-1; §Z8 I-3;
      rulings §Z13.1; r12 m-6):** a modelled late inventory whose `moe_onednn` term exceeds the
      admitted one gets exactly one `[LOAD-PLAN] the late inventory changes the zones admitted
      at the early stage: term %s in zone %s on device %d, early %zu B, late %zu B (refused)`
      line naming `moe_onednn`, RUNTIME and both sizes; no ensure is called and no `[SYCL-PLAN]
      model load refused` line appears. One whose term is below the admitted one is admitted,
      with exactly one `[ZONE-PLAN-BUG] the late inventory shrinks term %s on device %d: early
      %zu B, late %zu B (admitted; the early reservation stands)` line,
      `late_term_shrink_admitted` at 1 and the zones unchanged; an equal term logs nothing.
      Separately, `ensure_planned_arena_zones` called with a pending range on the device refuses
      with 23mk's `[SYCL-PLAN] model load refused` message (rulings §M13a) and leaves the range
      whole, with zero aborts (the `:16215` abort is not reached). The same arm runs 23mk's
      sequence: a smaller load, its unload, then a larger load, with zero aborts. Witnesses: the
      live check without pending ranges (the rebuild destroys the range), the `:16215` abort
      restored, a late stage that calls the ensure (the refusal comes out as `[SYCL-PLAN]`), and
      a late stage that refuses a smaller term;
    - **the commit is irrevocable before the retag (§M12 I-1):** a failpoint at `validate_end`
      and one at `cleanup_required` after `finalize_end`, each after the preload. After each,
      nothing is left under `{LOAD, txn}`, there is no `{MODEL, B}` range, and the `{DEVICE}`
      range count is unchanged. The exit-effects refusal (m-10) and an exception take the same
      path; after the exit-effects refusal the model is not LIVE and its code is `EFFECT_FAILED`
      (r12 m-1), and a mutant that reuses the not-committed branch with `cleanup_ok` true,
      making the model LIVE, is that case's RED. A failpoint that throws inside `if
      (result.committed)` after the retags leaves B's `{MODEL, B}` ranges whole (r12 m-2). RED:
      the retags placed before `validate_end` (the `{MODEL, B}` range survives a model that
      never became LIVE);
    - **the range bytes are the draws' bytes (§M12 I-2):** a quantized fixture with `ne0` = 2880
      (GPT-OSS's `n_embd`; 2880 % 512 = 320, so each row carries padding) and a buffer split at
      `get_max_size`: every draw lands inside B's range, and with no lazy tensor the draws total
      exactly the range, since the replay includes each buffer's rounding (r13 I-E). Positive
      control: the same fixture with the range sized in `ggml_nbytes`, where the last draw
      misses with `[ZONE-PLAN-BUG]`;
    - **the SYCL<n> buffer lands in B's range (I-C):** with a load bound, each SYCL<n> weight
      buffer's offset lies inside a `{LOAD, txn}` `WEIGHT` range, and the RUNTIME zone's used
      bytes do not move; a buffer larger than B's remaining range gets the named
      `[ZONE-PLAN-BUG]` refusal and a null buffer, never a RUNTIME allocation;
    - **each `(tensor, layout)` once (rulings §M14 I-1):** in both weight modes the draws on B's
      ranges cover the plan's per-device `(tensor, layout)` pairs once each, total at most the
      ranges, and leave at most the rounding bound; the SYCL<n> buffer tensors and
      `g_sycl_host_weight_extras` are disjoint sets. A WOQ and MoE-alternate fixture
      (Qwen1.5-MoE Q4_0's shape, 169 WOQ copies) with a transaction interleaved between the
      stages gives zero `dense_woq_planned_declined`, zero declined alternates and zero
      `[ZONE-PLAN-BUG]`. RED: a range of the primaries alone, under which the copies miss or the
      interleaved transaction takes their room;
    - **one recording, from the admitted plan (r12 m-4):** on a two-device host without
      `moe_multi_gpu_requested`, B's ranges are recorded once, after the stage's loop, and only
      for the devices in the admitted plan. RED: recording inside the loop, which leaves device
      0 a range from a plan that was not admitted, retagged to `{MODEL, B}` at the commit and
      never drawn;
    - **a range per TLSF (rulings §M18.4; r13 I-E):** the B70's geometry, two TLSFs of ~28192
      and ~841 MB under a 29472 MB per-chunk cap, and a plan with 28.9 GB of device weights
      (GPT-OSS 120B's shape on `level_zero:0`). It is admitted on first load, alone, with a
      `WEIGHT` range on each TLSF; each SYCL<n> buffer of the replayed split lies inside one
      TLSF's range; the draws land in the ranges in the replayed order; and nothing is left
      without a lazy tensor. A second fixture with 1oxa's geometry, weight chunks of at most 4
      GiB, and a model with more than 4 GiB of device weights is admitted the same way. RED:
      7.13's one extent per `(txn, term, device)`, refused in both. The lead's confirmation is
      GPT-OSS 120B on `level_zero:0`. **The draws are keyed (rulings §V13 I-5):** on the
      two-TLSF fixture, the lazy draws arrive in reverse replay order and each lands at its
      recorded `{TLSF, offset, size}`. REDs: a first-fit draw, which moves an item to the other
      TLSF and misses a later one; and a planted occupant at one key, which is the only
      `[ZONE-PLAN-BUG]`;
    - **the compute arena has no range (I-B):** on an arena device B's pending ranges carry no
      arena term, and a modelled `SCRATCH` capacity below `arena_bytes` gets the stage's
      named refusal, with no `GGML_ABORT` reached; on a non-arena device the arena is the real
      `COMPUTE` allocation, outside every range;
    - **one primitive, term-filtered (I-A; §M12 m-3; §X9 I-D):** B carries `WEIGHT`, `SCRATCH`
      and `MODEL_TERM` ranges, with part of its `WEIGHT` range undrawn. B's commit retags
      `MODEL_TERM` and `WEIGHT` to `{MODEL, id}` and then clears `{LOAD, B}`, which drops the
      `SCRATCH` hold. A **later load C** (owner `{LOAD, C}`; C is a load, since `SCRATCH` is a
      load term) then records its `SCRATCH` hold. The TLSF is sized so that C's hold fits only
      if B's `SCRATCH` hold was dropped, with B's undrawn `WEIGHT` room kept: the hold is larger
      than the free room left if B's hold is kept, and no larger than the free room once it is
      dropped. So C's hold is admitted at its full size on the correct tree, and refused under
      the unfiltered retag, which keeps B's `SCRATCH` hold under `{MODEL, B}`. A lazy
      materialization of B's after the commit, owner `{MODEL, B}`, then lands inside B's undrawn
      `WEIGHT` room; under 7.12's commit, which dropped that room, it misses with
      `[ZONE-PLAN-BUG]`.
    Mutation witnesses: the flag made a process-wide atomic again; 7.7's `thread_local` flag,
    under which the T1/T2 arm leaves T1 classifying `WEIGHT`; the ranges recorded at the late
    stage (the interleaved transaction takes B's room, and B's late plan misses as `TERMINAL` or
    places a device tensor on the host); a late stage that re-plans silently; master's
    RUNTIME-first chain for a bound load (the buffer lands in RUNTIME); a fall-through after a
    failed draw (the refusal arm allocates); a compute-arena range (the arena bytes are counted
    twice against `SCRATCH`, and the `SCRATCH` cap arm admits a load it must refuse); an
    unfiltered retag that moves every term (C's hold is refused, since B's `SCRATCH` hold
    survives under `{MODEL, id}`); the commit without the `WEIGHT` retag (the lazy draw misses);
    the pack before the ensure (both merge-gate shapes refuse at the early stage, and the
    packed-equals-ensured witness fires); and the `in_load` bit left set through the preload
    (the T2 arm reads "in load");
  - (aq) **one fit computation (rulings §M8 I-5(c)).** On the arena path the MMID re-plan's
    `BUDGET_EXCEEDED` demotion is unreachable, and (0) and (ii) call the same sizing-then-fit
    function. Mutation witnesses: the demotion restored on the arena path, and a (0) that skips
    the MMID sizing;
  - (ar) **no recording retains the shared weight slot (rulings §M37 Q6).** The PP MoE oneDNN
    batched entry checks `ggml_sycl_graph_recording_active()` before it claims the weight slot,
    and the `[ZONE-PLAN-BUG] PP MoE oneDNN path reached while recording` line exists at that
    check. A host arm drives the admission with the recording flag set and expects that line
    and no weight-slot claim. Mutation witness: the check removed, under which the arm sees the
    claim and no line.
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
  - **Each context's own ring rows (rulings §M32 I-1).** Context A reserves with `-ub 1024`,
    B with 512, on one device: each holds its own rows at its own size, and each one's claims
    are in plan. A closes: its rows are released after their events, and B's are untouched; a
    later transaction on the device carves nothing for A's. B closes the same way. RED: 7.14e's
    device ring, which held A's size after A closed, as planned room nobody used, until B left
    (and, before it, revision 5's "last re-plan wins", which shrank the ring under A).
  - **Two models' ring shapes (r8 m-8, restated).** On one device, model A's context and model
    B's context each hold rows sized by their own model's value function, and each model holds
    its own weight slot; a claim of each is admitted against its own context's rows and its
    own model's weight slot. RED: the executor's per-device read of master
    (`ggml-sycl.cpp:78760-78767`), under which the second commit's shape is read by the other
    model's claim, and one of them is refused. 7.14e's component-wise max over contributions
    is withdrawn with the device record.
- **H9 transaction guard (r2 N-I3; r3 I4, I6; r4 I5, I6, m9, m14; SYCL-free, in
  `kv-region-registry.hpp`).** A host model of the §2.4.2 steps, with a failpoint at every
  refusing step (the fit's head-slot refusal, the accounting step, the non-FA check, the
  yield's relock (a `[CONTEXT-PLAN-BUG]` under L0, forced by the skip-L0 hook below), after the
  carve of device 0 of 2, the MMID step, and the CAS). At each failpoint it asserts:
  - the registry and the context's published slot table are unchanged;
  - no yield ran for a failpoint before step 5;
  - this call's pending ranges are cleared under the instrumented group mutex, with the
    instrumented L1 **not** held (the guard's first phase takes no L1), and a first context's
    `FIRST_CONTEXT` range is re-recorded in the same section;
  - every handle this call carved, and every unused pre-minted control, is dropped, and dropped
    only after the instrumented L1, registry, group and binding locks are released;
  - **at the MMID and CAS failpoints, the context's ring rows are in their original slots**
    (the same handles, offsets and claim state as before the call), and a claim taken before
    the call is still valid (r4 I5). RED: revision 5's guard, whose ring slots were released at
    its step 7.

  **Serialized, with no `busy` (rulings §E.1, §E.2).** The model's process-global re-plan mutex
  (L0) is instrumented, and the model checks the lock order L0 before L1 on every path.
  - **Two contexts re-plan concurrently, on one device and on two devices**, in every pairing of
    a tenant-only republish, a tenant-only ring-row growth, a full-context publish and a
    teardown release proc, plus a re-plan on one device racing a model load on the other. The
    model's published plan is one global atomic, as jehw's is. Every call completes with **zero
    `busy` returns and zero lost CASes**, and each call observes the other's publish or refusal
    whole. RED: revision 7.2, which had no L0 and returned `busy` from step 2 and step 8 (a);
    and a per-device L0 (revision 7.3's first form), under which the two-device pairings trip
    the plan-identity checks and lose the CAS.
  - **Teardown's path.** The release proc runs from the guard member's destructor while another
    thread holds L0: it waits, then runs. A same-thread re-entry (a release proc
    reached with this thread already holding L0) trips the outermost-only token's witness
    check and aborts with its message, instead of hanging; the arm runs in a
    `GGML_SYCL_PRIVATE_TESTING` build, so a `-DNDEBUG` test build still checks it.
  - **Legitimate nesting (rulings §L0R; r7 I-2), positive.** llama's `replan_scope` takes the
    token; inside it the growth path's (0) probe, the transaction wrapper and (ii) each acquire
    it again. The nested acquires do not lock (the instrumented mutex records one lock and one
    unlock), the outermost token unlocks, and another thread's acquire blocks until then. RED: a
    plain `std::mutex` in place of the token, which self-deadlocks (the model detects it with a
    try-lock and reports it).
  - **A in a transaction, then B loads, then A's probe and FA recheck (rulings §L0R; r7 I-1).**
    Two modelled models. A opens a transaction and publishes; B's load entries run and publish
    B's plan; A then runs its probe and its FA recheck. Both validate against A's own plan
    (`lifecycle_select_placement_plan(A)`) and answer OK, with zero STALE_IDENTITY. A and B
    have different residencies, and the probe's answer is A's residency, so the arm proves the
    body evaluated A's plan and not only that the entry accepted A's token (rulings §M8 I-2).
    REDs: master's probe and recheck, which compare A's token with the one global snapshot and
    answer STALE_IDENTITY; and 7.6's form, whose entry passes per model while the body reads the
    global snapshot at `:17852`, finds B, and reports `[CONTEXT-PLAN-BUG]`.
  - **A's transaction between B's early stage and B's `load_end` (rulings §M8 I-3, §X7 I-3).**
    Model B's `load_begin` and early stage run; then A's context creation and a growth
    republish run their transactions, once between B's two stages and once between the late
    stage and `load_end`; then B's `load_end` runs the preload. A's MMID workspaces are planned
    from A's own inventory (the same as with no B), A's output buffer is not classified as a
    weight, and A's fit plans around B's load pending ranges. Every B weight then lands inside
    B's own ranges, where B's plan put it, with zero spills, and B's publish succeeds. REDs:
    master's process-global inventory (A's MMID re-plan reads B's tensors, or an empty
    inventory between B's `load_begin` and stage), the process-wide `g_sycl_in_model_load` (A's
    output buffer takes role `WEIGHT`), a stage that records no load ranges (A carves into
    B's planned room and B's preload spills), ranges recorded only at the late stage (the
    first interleave takes B's room), and ranges that do not cover B's SYCL<n> weight buffers
    (B's buffer allocation is refused and the load fails at `llama-model.cpp:2637`).
  - **`can_unload` against a pending unload (rulings §M11 I-F, §M12 m-5).** The arm replays
    `test-sycl-lifecycle-runtime-wrapper.cpp:1476-1492`: thread T holds a live-update lease
    with no L0 (`test_hold_live_update`) and starts the async `unloaded_token`, which takes L0
    and waits for the lease. The test's only sequencing today is a 20 ms `wait_for`
    (`:1481-1485`); if the unload has not taken L0 by then, a blocking acquire succeeds,
    `reserve_shutdown` answers `BUSY`, and the RED passes on a loaded host. So the arm adds a
    `GGML_SYCL_PRIVATE_TESTING` handshake, `ggml_backend_sycl_test_wait_unload_parked()`,
    which blocks until the unload thread holds L0 and is parked in its lease wait, and a
    positive control at the moment of the call: a probe `try_lock` of L0 from T fails. Then T
    calls `ggml_backend_unload_checked`, which reaches `can_unload`. Expected: the call
    returns `BUSY` at once, T releases the lease, and the unload completes. A 5 s watchdog
    fails the arm instead of hanging it. RED: 7.10's blocking acquire, under which the call
    waits for the unload's L0 and the watchdog fires; with the handshake, the RED cannot pass
    by timing. The wrapper test itself keeps its form; the handshake is used by this arm. A
    second arm, with L0 free, takes the `try_lock` and closes admission as before.
  - **Load B while A's context holds its rows (llama.cpp-r7fz; rulings §M7 I-4, §M32 I-1,
    §M38 I-2).** Model A's context holds claimed-then-vacated ring rows on device 0; model B
    loads on device 0. After B's load, A's rows (handles, sizes, depth) and A's model's weight
    slot are unchanged, B's load drew only B's weight slot, and A's next PP MoE claim succeeds.
    **The mechanism (§2.4.2 (b) step 3):** the weight-slot store is keyed by (model, device), so
    B's setter writes B's entry and A's executor reads A's; A's load laid out RUNTIME at A's D
    terms, so B's pack places B's slot in RUNTIME's free room where it fits and otherwise as a
    `{LOAD, txn}` range in the shared zone, retagged `{MODEL, B}` at the commit and drawn inside
    B's load. **Pre-registered, on two fixtures:** (1) A = GPT-OSS 20B (weight slot 32 ×
    4406528 = 141008896 B) and B = a GPT-OSS-20B-shaped fixture with one partial expert layer
    on the device, whose slot is smaller, so `get(A, 0)` = 141008896 B and `get(B, 0)` is B's
    value, both after B's load; B's slot is a shared-zone range, since RUNTIME has no free
    room, and RUNTIME's capacity is unchanged; (2) A = Mistral 7B (no MoE: RUNTIME laid out at
    0) and B = GPT-OSS 20B: B's 141008896 B slot and its k × 256 B tables are `{MODEL, B}`
    ranges in the shared zone, and the load refuses nothing. REDs: master's
    `populate_inventory_globals`, which releases the ring, so A's claim finds none and reports
    `[CONTEXT-PLAN-BUG]`; and **last-writer-wins**, 7.14g's `(dev, bytes)` store, under which
    `get(0)` after B's load returns B's value on fixture (1), so A's next claim admits against
    B's slot size and the exact bytes catch it.
  - **Every publisher under L0 (rulings §L0R, §M8 I-1, §M9 F5).** The model runs, against a
    parked L0 holder, each exported entry §2.4.2 lists: the wrapper, the probe, the FA recheck,
    activate, unload with its failure republish, the quarantine reap through
    `ggml_backend_sycl_free`, shutdown, `load_begin`, `stage_inventory_plan` and `load_end`.
    Each blocks until L0 is released. (`set_runtime_context` is no longer exported,
    `compute_placement_plan_early` publishes nothing (its body is a static impl), and
    `test_hold_live_update` is test-only with no L0.) RED, with its precondition stated (r9
    m-10): **model B publishes last**, so the publication names B; then 7.6's list, under which
    an exported `set_runtime_context` called on a second thread for A's backend publishes inside
    A's yield window, and A's relock reports `[CONTEXT-PLAN-BUG]`. Without that precondition the
    RED is void: if the publication names A, the second thread's `acquire_live_update`
    (`:17879`) meets A's outstanding ticket, returns `BUSY`, and never publishes.
  - **The allowlisted republish against a concurrent L0 committer (rulings §ZR5 I-3; zhcn 5.4
    row 30).** One thread loops `ggml_sycl_republish_current_plan_into_empty(cache, owner)` on
    an emptied cache while another commits legal L0 transactions that change the publication.
    Expected: 0 BUG lines, 0 STRICT (§G1) aborts, and 0 writes to `g_model_n_layer`,
    `g_placement_kv_info`, `g_placement_publication` or another cache's snapshot (write
    counters). A modelled republish that changes the identity reports `[CONTEXT-PLAN-BUG]`.
    RED: today's form, the full republish with the emptiness check outside the mutex and the
    identity read before and after, which reports a false BUG when a commit lands between the
    reads, and which writes the globals.
  - **A covered read against a parked L0 holder (rulings §M76.1).** Context A's candidate is
    covered. Another thread holds L0 and runs, in turn, an unload of model B, a quarantine
    restore of model C and a load of model D, parking inside each. A's coverage query runs at
    each park point, without blocking, and answers COVERED each time; A's slot handles, its host
    reservations, its ring rows, and A's MMID workspaces are unchanged
    across the whole run, and A's next claims are in plan.
  - **Shutdown takes L0 (rulings §M76.5).** Shutdown waits for a parked L0 holder, then runs. A
    publishing entry that finds the module non-ACTIVE while it holds L0 (forced by a hook that
    closes the module without shutdown's token) reports `[CONTEXT-PLAN-BUG]`, returns no
    `BUSY`, and does not retry.

  **No other context's ring (rulings §M32 I-1; replaces r4 I6, r5 I-A, r6 I-5's arms).** 7.14e
  carried four skip-L0 routes to an absent device ring (RELEASING observed at step 2, a
  teardown between step 2 and step 8, RELEASING raised after step 2, and a release proc after
  the ring-lock section) and two RELEASING arms (a (ii) refusal after a sole-contributor (i),
  and RELEASING's owner). With the rows per context, no transaction reads, pins or releases
  another's rows, so they are replaced by one arm. A test hook lets A's release proc **skip**
  L0 and run at each point of B's transaction: between step 2 and the snapshot, between the
  snapshot and the fit, in the yield window, and just before the CAS. At each, B's fit input,
  B's rows and B's publish are unchanged, B publishes its own rows, and A's rows are freed only
  after their events. RED: a fit that reads a device-keyed row record, 7.14e's, where A's
  teardown changes B's input and B's step 8 finds a generation it did not copy.

  **Ring growth and shrink (r5 I-B):** a growth fixture whose overlap demotes exactly the layers
  the context's old rows' bytes cover, each labelled `ring-growth` with the old and new bytes
  (RED: the first draft's `capacity` label); a shrink fixture (the context's ladder steps down)
  that carves nothing, demotes nothing and refuses nothing (RED: the same-size reuse rule,
  which carved smaller rows beside the old ones).

  **Superseded drops after L1 (r5 m-a):** at a committed publish, the superseded handles are
  dropped with the instrumented L1 released, including a retire's withdrawn mirrors (r5 I-J).
  RED: the first draft's drop inside step 8 under L1.

  **After a refused (ii) (r6 m-9):** a tenant-only row growth whose (ii) is refused leaves the
  context with no rows and its tenant key cleared; its next PP MoE claim is
  `[CONTEXT-PLAN-BUG]` with an error status, never a skipped op, and zhcn's ladder revert
  answers GROWTH and re-carves the previous candidate's slots, the rows among them. RED: a
  coverage query that answers COVERED from the caps alone, which carves nothing.

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
  successful close, the proc empties `(c, *)` under `kv_region_mutex_`, moves out each ring
  row's last-generation retention under `c`'s slot-state lock, hands the retentions to
  `retain_handles_until_event` and drops everything with no instrumented lock held; it takes no
  L1 (§2.4.2 "Teardown").
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
- checks STRICT (§G1) in a subprocess: exit 134, and the message is present;
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
  is taken, and the external-bytes counter does not grow. `GGML_SYCL_STRICT_LEASES=1` (§G1)
  aborts in a subprocess;
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
  `[CONTEXT-PLAN-BUG]` = 0 under `GGML_SYCL_STRICT_LEASES=1` (§G1); destroying the graph moves
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
    it. **This arm also scores `[CONTEXT-PLAN-BUG]` = 0 under `GGML_SYCL_STRICT_LEASES=1`
    (§G1)**: it is the pipelined decode in which revision 5's vacate-at-completion would have
    reported healthy reuse as over plan. The default (replay) arm is scored the same way, for
    the record-mode rule (r5 I-C).
  - The expected delta is 0 on every arm.
  - If it is not zero, look at the TRANSIENT reclassification: scratch addresses move from
    the weight side to the context side.
  - There is no fattn clause: the sidecar is opt-in only (§2.8, r2 m4).
- **C7 zhcn's shared gates, scored against this design's plan fields.** zhcn's GA (B50 GPT-OSS
  `-c 65536 -ub 1024`: exactly the 6 highest-indexed full-attention layers demote, scored
  against `free_after_full_kv` and the demotion causes, §2.4.1) and GH (B50 Qwen3.8-Flash-Next
  under `GGML_SYCL_STRICT_LEASES=1` (§G1): no unplanned per-context SYCL buffer, the recurrent
  state claimed from its planned slot, §2.4.4). They are listed in zhcn's §5.3 and run once
  each, after L6.
- **C8 a1_long, the OPTIONAL yield (rulings §M18.3a).** jehw's gate form, once, on the B50:
  ```
  GGML_SYCL_VRAM_BUDGET_PCT=60 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
    ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf -c 2048 \
    -p '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,' -n 15 --seed 42 --temp 0 \
    > c8.out 2> c8.err
  cat c8.err | grep -c 'KV admission released 6 optional oneDNN WOQ layout copies (220.5 MB)'  # 1
  cat c8.err | grep -c 'KV-PLAN-BUG'                                                          # 0
  ```
  Pre-registered unchanged from the jehw-era run: exactly one release line naming the six
  copies and 220.5 MB, rc=0 and zero aborts. The copies are planned OPTIONAL copies, so the
  line now comes from the yield path of §2.4.2 step 5. **That rests on two premises (r14
  m-8):**
  - **(i) every jehw-era copy was plan-listed. Checked.** The lead's a1_long run on uwlx
    `a4da787ae` (2026-09-26, `vuwlx-a4da/a1_long.err`, jehw's gate form) logs
    `[PLACE-4-WOQ] dense oneDNN WOQ alternates: device=0 eligible=225 added=182
    skipped_capacity=43` and no `[S1-PRELOAD] dense oneDNN WOQ copies disagree with the plan`
    line, which `ggml-sycl.cpp:35510-35515` (`3d9414c8c`) prints at WARN whenever a copy is
    staged but not planned (`dense_woq_staged_unplanned`, `:34941`). The same log shows WARN
    lines surviving default verbosity, so the absence is read at a level that prints; a run
    that printed it would void the premise. So the six released copies were plan-listed;
  - **(ii) moua's fit picks the same six. Not checkable before L4.** jehw picked a frontier
    prefix; moua picks buried tenants by §2.9's cost-ordered ladder inside the model's
    `WEIGHT` range. So before the lead run, a host-only dry run of `kv_region_fit` over the
    a1_long fixture's admitted plan (its copy list and ranges, the H7ap yielded-extent arm's
    vehicle) pre-registers the count and the MB it picks, and the C8 grep uses those. "6 ...
    (220.5 MB)" stands only if that dry run picks jehw's set; §M18.3a keeps the line's form
    either way. A run that logs no release line and
  demotes KV layers instead is §M18.3's regression, which §M18.3a withdraws, and fails C8.
- **C9 zero `[EXT-ALLOC]` lines of any role on both merge gates (rulings §M38 C-2; P1).** The
  two merge-gate commands of `merge-gates/run-merge-gates.sh` (`gptoss120b-b1` and
  `glkg-qwen35b-a3b-b1`, `-c 4096 -ub 512`, `level_zero:1`), each once, with
  `GGML_SYCL_EXT_ALLOC_TRACE=1` added. The line prints only under that variable
  (`unified-cache.cpp:3960-3968`, `:15309`), and the gate logs in hand were taken without it and
  hold none, so a zero from them would be vacuous:
  ```
  cat <gate>.err | grep -c 'EXT-ALLOC'      # 0, every role, on both gates
  ```
  - **Positive control, before the scored runs:** the two commands on `c69d5774d` with
    `GGML_SYCL_RUNTIME_ARENA_MB=0`, which takes RUNTIME's floor away from the unconverted draws
    of §2.4.2 (b)'s census (the precedent is the 423624704 B `backend-buffer-runtime-zone` line
    that 23mk's census records going raw on a GPT-OSS B50 run, its CR-2 control). At least one
    `[EXT-ALLOC]` line must print, or C9 is VOID. That shows the trace is armed and that a draw
    left without room reaches the line.
  - **Baseline:** the two gate commands on `c69d5774d` with the trace on. Every cohort they
    print is either converted by a commit that lands before L4+L6 (§4), a term's draw, or a
    named defect with a ticket, recorded on the ticket before L4+L6 lands. A cohort left out
    of all three fails C9's precondition, and L4+L6 does not land over it.
  - **Scored at the L4+L6 tree:** 0 lines of any role on both gates, rc=0, and the gates'
    own checks green. C1 keeps its KV-role count, which this generalises to every role.

## 4. Decomposition, effort, landing order

| id | work | files | effort | depends on | lands |
|----|------|-------|--------|------------|-------|
| L1 | TLSF placement primitives, tags, frontier walk; H1 | `tlsf-allocator.hpp`, `shared-zone-tags.hpp`, `tests/test-tlsf-allocator.cpp`, CMake | high | none | **done, approved:** `eab1ebeb6`, `9e0a708dc`, `97315421b`, `456650c01` |
| L3 | pure `kv_region_fit` (multi-extent, self extents, `forced_host`, `own_ranges`-restricted commit re-fit, strict prefix, carve mirroring, indexed head slots placed first with reuse in place, the context's ring rows as its own head slots (7.14f; rulings §M32 I-1), the recurrent slot, sidecar companion slots, the cost-ordered pack with the two-way optional classification, pending ranges as allocated, the stated demotion order, `free_after_full_kv` and demotion causes), the `context_side_demand` record (MODEL/CONTEXT scopes, indexed slots) and its reconciliation, `kv_layer_cells` + `kv_layer_tensor_bytes` (the one byte function) and the RS-buffer size function, `kv-region-registry.hpp` (registry with tenant slots, scope, residency answer with the no-region fallback, the two-phase guard model, the release proc model); H2, H3, H6, H8, H9 | `kv-runtime-demotion.{hpp,cpp}`, `kv-region-registry.hpp`, `unified-cache.hpp` (`kv_layer_bytes_for_kind` delegates), their tests | xhigh | L1, jehw on master, the zhcn protocol (agreed, §6.6) | after jehw |
| L4 | `context_side` (explicit `lifetime` field threaded into `zone_alloc`, LIVE/RETAINED states and the slot tag, the reserved-slot carve as owner-first handles, `claim_slot`/`release_claim` with event-chained reuse and per-slot claim state, atomic retained-run carve, settle refusals for pending ranges, the plan-violation ERROR with an error status, pending ranges on the shared zone's TLSFs (the context's head slots and KV, `WEIGHT`, `FIRST_CONTEXT`, and a later load's D terms that RUNTIME cannot hold) and on RUNTIME for the D terms of the load that laid it out (rulings §M38 m-2, I-2; §2.3.3 A1), `live_bytes()` beside an unchanged `zone_available`), `allocate_excluding` with range exclusion (used whenever a pending range exists), `allocate_at`, and a whole-TLSF block census (L1 follow-ups), retained-run registration hygiene and the `live_bytes` readers, the per-extent and per-slot owner-first carve inside `zone_alloc`'s locked section with controls pre-minted before L1 (`unified_allocate_owner` split into mint and bind halves), leaf `kv_region_mutex_` added to contract §12.5 (L3), locked geometry snapshot (cache locks then group mutex, with jehw's predicate), `reserve_kv_region`, strict-prefix `yield_optional_prefix`, `backend-buffer-kv-zone` passing its buffer's role, removal of the dead `KV_AUTO` reclaim and of the `arena_reserve` KV reclaim, N-chunk routing, and **1oxa's `GGML_SYCL_PRIVATE_TESTING` dump of `shared_zone_geometry` plus `kv_region_request` at each fit** (step 2 and the tenant-only path; lead-approved, for 1oxa's VM branch to test against); H4, H4b, H5 | `unified-cache.{hpp,cpp}`, `tlsf-allocator.hpp` (two primitives), `ggml-sycl.cpp` (the kv-zone fallback's role) | xhigh | L1, L3, **zhcn landed, beni's producers landed, llama.cpp-jzvq closed** (lead ruling, r4 I10), **llama.cpp-uwlx landed** (the pick-list yield, §2.4.2 step 5; r5 m-g), **llama.cpp-fkpg (a) landed** (the envelope's `n_ctx`, which sizes `FIRST_CONTEXT`; rulings §M38 C-1), **zhcn's `fattn-onednn` carve and 23mk's reorder temporary landed** (rulings §M38 C-2) | after zhcn, beni producers, jzvq, uwlx, fkpg (a) |
| L5 | the optional pass after all S1 staging (dense + expert/DPAS); `zone_alloc_optional` | `ggml-sycl.cpp` S1 block | medium | L4 | with L4/L6 |
| L6 | llama side: `llama_kv_layer_shapes` and `llama_rs_layer_shapes` factored out and stored at the first publish, the `ggml_sycl_runtime_context_desc` descriptor (KV-shape with sidecar and `n_stream`, recurrent section; zhcn's tenant section filled by zhcn) and its publish entry point, the scope procs with an RAII guard, the one `ggml_backend_sycl_kv_region_release` call site in the `sycl_plan_guard` member's destructor (zhcn M1, with the ContextId captured at `create_exec`), and `llama_recurrent_sycl_kv_buft` returning the recurrent-state buft. Backend side: the transaction steps of §2.4.2 (two-phase guard without L1, idempotent key without `n_ubatch`, the tenant-only path with zhcn's step (i) (the own-context graph clear behind its recorded-graph gate, the slot-table take at (c), the reap call with its backstop, the use-count bound; r6), reuse in place for host slots, the ring rows in the context's slot table with the executor re-keyed to them (7.14f; rulings §M32 I-1), the slot-state retention moved out at a row growth and at teardown and handed off after the unlock at every removal site (r6 I-4, I-6; r7 m-5), the model load's ring release and ring writes removed on arena devices (llama.cpp-r7fz), plan / accounting / predictable refusals / pending ranges / yield / restricted re-fit and carve / MMID / CAS / commit installing the slot table / superseded drops after L1), the registry release proc, `g_execution_backend_binding_mutex` census entry, the residency hook answering from the registry, the tiered claim with `set_owner(mem_handle)` slice views and the KV-only size check, the sidecar companion claim, the recurrent-state buft, the per-extent clear with event-held slices, BLOCK_EXEC_CANDIDATE_KV ignored under an arena (the VMEM_KV refusal is 23mk core's, §2.6), the second sources deleted (§2.2, the budget-room check included), both ERROR sites and their §G1 abort through `ggml_sycl_strict_enabled()` (uwlx exports the accessor and makes §G1a's three doc edits, rulings §G1b; only if this commit lands before uwlx does it carry both, §2.8), the dark B50 lever, `GGML_SYCL_KV_REGION_TRACE`; H7 with the unconverted-site list, the CPU-buft llama shape tests, G1. **Absorbs revision 1's L2.** | `ggml-sycl.cpp`, `ggml-sycl.h`, `unified-cache.cpp`, `fattn.cpp`, `common.hpp`, `src/llama-context.{h,cpp}`, `src/llama-model.cpp`, `src/llama-kv-cache.{h,cpp}`, `src/llama-memory-recurrent.cpp`, tests | xhigh | L3, L4, L5 | before beni's conversions |
| L7 | docs: memory-design section, contract §3/§5.2/§5.3/§12.5 (the binding-lock chain, the L5 group → `g_runtime_alloc_mutex` order, the step-6 carve exception), arena comment, limits (§2.9), lock order, the tenant protocol and the descriptor's layout rules, the owner-visible weight-hole line | docs, `unified-cache.hpp` comment | medium | L6 | with L6 |

**Revision 7.7's additions to the rows (r8; rulings §M8, §M76a, §ZR5).**
- **L4:** per-model plan state, with the inventory record carried by each model's snapshot
  (§M76a; H7ao); the loading thread's load flag and the load pending ranges from
  `stage_inventory_plan` to `load_end` (§M8 I-3; H7ap); the context's MMID pools (CONTEXT
  scope, route-gated, the `load_end` materialization deleted; rulings §M9 I-2, I-3), their first
  materialization inside the transaction and the planned growth of its carve (§M8 I-5; H7am);
  one fit computation with no MMID budget on arena devices (H7aq); the host tier's contiguous
  per-index reservations, through the existing nsl3 path (§ZR5 I-2; H7an); the pool phase
  gates' exemption for a thread holding a `TRANSACTION` token, through the one accessor with
  the token's kind, declared in `unified-cache.hpp` (7.7a, §M9a; H4, H7an); the
  coverage query's backed-table rule and the entry's contribution copy (§M8 I-4); and the ring
  as per-kind contributions with a component-wise max (r8 m-8; H8).
- **L6:** the transaction body taking the per-model snapshot (§M8 I-2); the token at the top of
  `shutdown`, `stage_inventory_plan` and the other entries of §2.4.2's list (revision 7.9 took
  `set_runtime_context` out of the exported set, moved `compute_placement_plan_early`'s body
  into a static impl (7.13),
  and left `test_hold_live_update` test-only with no L0; rulings §M9 I-1, m-2, m-4); the
  deletion of the three orphaned publishers and the re-anchoring of
  the gates that name them (§M8 I-1; H7ai); the three live decode sites switched to
  `ggml_sycl_republish_current_plan_into_empty(cache, owner)`, with the owner token threaded
  through `ensure_moe_secondary_queues_for_plan`, and the dead
  `materialize_moe_tensor_planned_layout` deleted (§ZR5 I-3, §Z6 I-C; zhcn 5.4 row 30, 5.5 row
  35); and the
  deletion of the wrapper's post-transaction MMID materialization.

**Revision 7.11's additions to the rows (r10; rulings §M11, §M10, §X7 I-3).**
- **L4:** the pending range's term tag and the term-filtered `retag_pending(owner,
  term_filter, new_owner)`, the one primitive shared with 23mk (I-A; H1, H7ap); the stage's
  `SCRATCH`-capacity check against `arena_bytes` on arena devices, with the named refusal
  replacing `GGML_ABORT` (I-B); the MMID registry re-keyed by (token, context, device) through
  dedup, the prepared replacement, `acquire`, the cookie, `admit`, `retire` and `recover`
  (I-E(2); H7am); and the `in_load` window matching master's (m-6; H7ap).
- **L6:** the load ranges recorded at the early stage and consumed by plan identity at the
  late stage, with its named refusal (§X7 I-3; H7ap, H9); a bound load's SYCL<n> weight
  buffers drawn first from B's `WEIGHT` range, with the named refusal and no fall-through (I-C;
  H7ap); step 7 taking its `ggml_backend_sycl_context &` explicitly and binding by the
  context's `execution_context_id` (I-D; H7am); the trigger redefined, the retire sites
  `:12321`, `:18635`, `:18650` changed and `:13209`, `:13245` deleted, and a retire at context
  destruction (I-E; H7am); `can_unload`'s `try_lock` (I-F; H9); the probe side channel at
  `:18794-18805` deleted (m-1); `load_enter_nested`'s flag write at `:12894` deleted (m-7);
  the alloc-entry and host-fallback assertions of the shared invariant (m-5; H4); the three
  MMID source gates re-anchored (m-3); and, with the dead function's deletion, row 648 of
  `docs/backend/sycl-static-storage-inventory.csv` regenerated so
  `scripts/audit-sycl-static-storage.py --check` returns 0, plus an absence assertion for the
  deleted function (§M10, m-2; §M12 m-7).

**Revision 7.12's additions to the rows (r11; rulings §M12, §M11b, §Z9 I-3, §Z9a).**
- **L4:** the shared term list with `REGION` covering both designs' context extents, and the
  two rules (§M11b); `clear_pending_locked(tlsf, owner, term_filter)` and the filtered
  `pending_ranges(c, d, owner, term_filter)` in the one primitive (I-4, m-2; H1);
  idempotent `(txn, term, device)` recording (m-9; H1); `ensure_planned_arena_zones`
  counting pending ranges as live and refusing by name, with `arena_destroy` and the rebuild
  on §2.3.4's refusal list (C-1; H7ap); the MMID registry's non-accepting step-7 entry,
  flipped at the CAS (I-3; H7am); `recover(wait = false)` at context destruction (m-6).
- **L6:** `compute_and_store_plan_for_inventory` computing the plan before it ensures the zones
  (7.12's order, which 7.13 reverses, below), in both stages, the late zone comparison and the
  `:16215` abort made a named refusal (C-1; H7ap); the retags only after an irrevocable commit,
  and the rollback clear in `ggml_sycl_abort_owner_effects_noexcept` (I-1; H7ap);
  `ggml_sycl_weight_alloc_bytes` shared by `get_alloc_size` and the stage, and the range bound
  (I-2; H7ap); the late comparison over the late inventory's subset (m-8); one
  `ggml_sycl_compute_arena_bytes()` accessor and `ggml_sycl_model_loading_effects` returning
  `bool` (m-10); the step-7 binding through `g_execution_backend_bindings` (m-13; H7am); every
  witness check as `GGML_SYCL_WITNESS` (§Z9 I-3); the `can_unload` arm's parked-unload handshake
  (m-5; H9); H4 (b) on zhcn's H4h hooks (§Z9a, m-11); and `scripts/audit-sycl-static-storage.py
  --check` with an absence assertion as the §M10 witness (m-7).

**Revision 7.12a's additions to the rows (rulings §M13, §M13a).**
- **L4:** the terms `ONEDNN_PP_A` and `SET_ROWS_STAGE`, and `DEVICE_TERM` reserved with no
  producer (§2.3.3 A1); `record_pending(owner, term, offset, size)`; `replace_within` with its
  term and status (A3; H1); and the device-wide `pending_bytes_excluding(device, zone,
  except_owner, except_term)` (H1).
- **L6:** A4 step 3's `WEIGHT` retag to `{MODEL, id}` and the unload's `{MODEL, id}` / `WEIGHT`
  clear (1oxa r8 I-D; H1, H7ap); `ggml_sycl_load_pending_rollback_noexcept(txn)`, called by
  `ggml_sycl_abort_owner_effects_noexcept` before or with 1oxa's T13 chunk destroy and
  unconditionally by the validate-failed exit; 23mk's `[SYCL-PLAN] model load refused` message,
  with `compute_and_store_plan_for_inventory` returning `bool` (C-1; H7ap); and H4 (b) on zhcn
  `8a58ad4`'s hooks.

**Revision 7.13's and 7.13a's additions to the rows (r12; rulings §M14, §Z13.1, §M15, §Z14.2).**
- **L4:** `ensure_planned_arena_zones`' named refusal arguments (m-7); the guard's one clear
  form (m-8); `retag_pending` as `noexcept` (m-2); and the eviction re-record of a planned
  weight's block (m-11; withdrawn in 7.14 by rulings §M18.3).
- **L6:** the demand-ensure-pack order with the packed-equals-ensured witness, and the late
  stage's per-term comparison under §Z13.1 with its two strings and counter (C-1; H7ap); the
  extra copies in B's `WEIGHT` range (I-1; H7ap); `weight_forced_off_sycl`, shared by
  `create_tensor` and the early inventory (I-2; H7ap; replaced in 7.14 by
  `resolve_create_site`); one MMID RUNTIME source on arena devices, with 23mk's A step between
  step 5's recording and the yield (I-3; H1); the exit-effects refusal through
  `finalize_end(ticket, false)` (m-1); the retags first in `if (result.committed)` (m-2);
  `onednn_w_rollback_pending(txn)` called with the transaction alone (m-3; rulings §M15); the
  recording after the device loop (m-4); the one-extent refusal (m-5; withdrawn in 7.14 by
  rulings §M18.4); the unload clear first in teardown (m-12); `compute_placement_plan_early`'s
  static `bool` impl behind the public entry (m-13); the audit clause armed only on a passing
  base (m-15); and H4 (b) on zhcn's H4h vehicle (cited at `2c511f6` since 7.14c) with its
  phase precondition (m-9), lead-run on `level_zero:1` (§Z14.2).

**Revision 7.14's additions to the rows (r13; rulings §M16, §M16a, §M16b, §M17, §M17a, §M18).**
- **L4:** A3 as §M16, §M16a and §M16b give it: device ranges only, with `HOST_TIER`; the E, G
  and other classification with `BUSY` and `SHARED`; the superseded control and its witness
  counters; remainder mode for a `WEIGHT` re-draw and the null empty remainder; the overlap
  witness; the store's `in_hand` and `releasing` snapshots and the registered, locked recording
  sinks (`mem-handle.cpp`); and the pending-range key with the TLSF (§M18.4) (H1).
- **L6:** llama side: the early call after the buffer-type lists, `create_weight_tensors`, the
  record pass, `resolve_create_site`, the site and forced bytes in `ggml_sycl_tensor_info`, and
  the site in `planned_target_device` (C-1, I-B; H7ap). Backend side: the five-step order over
  the closed `ggml_sycl_zone_term` enum, the pack-time charges, the grow-only ensure, both
  witnesses and the setter gate (I-A, I-F; H7ap); `moe_control` gated to a valid zero (§M17a;
  H7ap); the §M18.1 refusal and a late stage that never packs (C-1; H7ap); per-TLSF ranges from
  the replayed buffer split, whose helper ggml-alloc shares (`ggml/src/ggml-alloc.c`; I-E;
  H7ap); one byte source for the copies, collapsed copies deleted from the plan, the
  unplanned-staging refusal and the staging witness, and planned copies as `WEIGHT` (I-C, I-D;
  H7ap); the namespace-scope witness switch (m-g); and H4 (b) registered as zhcn `c75ce4d`
  registers H4h (m-f).

**Revision 7.14a's additions to the rows (rulings §M18.3a, §Z15, §M19).**
- **L4:** the yield path's section for a planned OPTIONAL copy: the retag of its block's extent
  to `{CONTEXT, id}` / `REGION`, the plan's `yielded` mark and the handle moved to the drop list,
  under one group-mutex hold (H7ap).
- **L6:** the planning-time PRIMARY/OPTIONAL class on each copy and `mark_optional_layout` for
  OPTIONAL copies only; step 5's recording that leaves a planned-copy pick to the retag;
  `nonfa_shape` charged by the pack per attention device; `onednn_scratchpad`'s value function
  as moua's and charged in step 3 only; `pp_pipeline` charged by calling 23mk's
  `pp_pipeline_weight_bytes`; the load-stage getter without the graph-scratch floor (moved to
  the Graph-scratch commit by rulings §M26a I-4; 7.14d below), and the load-stage dry run
  evaluating every C term as 0 (H7ap); C8 (§3.3).

**Revision 7.14c's additions to the rows (rulings §M23, §M24, §M25).**
- **L4:** `record_pending`'s append rule for every owner but a load's, with its H1 arm; the
  yielded flag on the copy's cache entry, written under the group mutex, and the section's
  lock order.
- **L6:** the pack's charge ledger and the sized ensure `ensure_arena_zones_sized(dev, sizes)`
  (shared with 23mk) with the one composition rule; witness 1 as `ensured == max(floor,
  demand + charged)`; the dry run as the only load-stage reader of the getters;
  `mmid_workspace` charged nothing at the load stage (class C); `moe_onednn`'s per-device
  maximum; the C8 fit dry run that pre-registers the a1_long count (§3.3).

**Revision 7.14d's changes to the rows (rulings §M26, §M26a, §M27).**
- **L4:** `onednn_pp_a` and `moe_control`'s slot as head slots of the fit in `REGION`
  headroom, with their H1 and H7ap arms (rulings §M27 (1), (2a)); `moe_control`'s handle on
  the backend context and the deletion of the compact list's and flag's fallbacks, in the
  commit that lands the slot, which also shrinks row 134's block to k × stride.
- **L6:** the load stage charges `moe_control` nothing (7.14a to 7.14c's step-3 charge is
  withdrawn, and "`moe_control` gated to a valid zero" is no longer a charge);
  `account_moe_mmid_workspaces` leaves the load's `plan.vram_bytes` (rulings §M27 (3)); master's
  ONEDNN sizing on both sides of the dry run until the Graph-scratch commit.
- **Not moua's:** the load getter switch, the with-floor getter's and the clamp's deletion,
  the direct overflow's deletion, the draw's move and the m-14 test and comment updates are
  the Graph-scratch commit's (23mk, in beni; rulings §M26a I-4). 7.14b and 7.14c placed the
  switch in L6 and C-1.

**Revision 7.14e's changes to the rows (rulings §M28, §M29; 1oxa rev 11).**
- **L4:** `ring`'s activation and output slots placed as head slots in the context's `REGION`
  headroom only, at align64(`n_ubatch`) rows (u1bb's RUNTIME branch withdrawn); the head-slot
  tier rule (tiers 1-3, no yield for a head slot) with its H5 arm; the `VM_TAIL_SURPLUS`
  term; the keyed `allocate_within` form. The row-134 shrink, `moe_control`'s slot and row 75
  are one landing set: the shrink cannot land before the other two (rulings §M28 (3)).
- **L6:** `moe_onednn` charged as the ring's weight slot alone, so the load's RUNTIME holds
  no ring rows; the load replay records each item's `(TLSF, offset)` key; the VM group page
  charge for OPTIONAL copies; `onednn_scratchpad` and `ring` published as 0 at the load
  (class C); master's reorder + eligible value deleted from both sites (`ggml-sycl.cpp:15947`,
  `unified-cache.cpp:27565`).
- **L4, for `onednn_scratchpad` (rulings §M29a):** one buffer per (context, queue) as a head
  slot of the context's fit, at the maximum `get_size()` over the enumerated
  `primitive_desc` objects, allocated once at the transaction and never grown; the grow path
  replaced by the named `[ZONE-PLAN-BUG]` line, with its host RED.

**Revision 7.14f's changes to the rows (rulings §M32, §M33 I-G, §M34 (4), §M35, §V15a, §V16
I-A, §V16a).**
- **L4 and L6 land as one commit (rulings §M32 I-6).** 7.14e's L4 placed the ring rows in
  `REGION` while master's load, until L6, still charged them to RUNTIME, so from L4 to L6 each
  context re-placed 1.08 GiB (GPT-OSS 120B) or 1.5 GiB (Qwen) that the load held idle, and
  with r15 C-1 the first context on either gate was refused from L4 on. The rows' `REGION`
  placement and the load's removal of them are now one commit, so no tree between them exists.
  The commit carries both rows' lists below and above; L5 stays with it as before.
- **L4+L6, the ring (rulings §M32 C-1, I-1, I-3):** the per-context rows as CONTEXT head
  slots in the context's slot table, sized by the resident-expert value function; the
  executor re-keyed to them (`ggml-sycl.cpp:78760-78767`, `:78817`, `:78823`), its lazy
  reserve made a presence check; the per-(model, device) weight slot drawn at load under
  `{MODEL, id}`, published by `unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes` and
  mapped to `MOE_ONEDNN` by the setter gate, and the `:15961-15967` call deleted on arena
  devices; the device ring record, its generation, RELEASING, pins and contributions deleted,
  with the KV-zone split and `unified_cache_get_planned_pp_moe_onednn_kv_zone_bytes`; the
  load-time reserve at `ggml-sycl.cpp:5657-5680` deleted on arena devices; the RUNTIME getter
  rewritten to the three D stores (§2.4.2 (b) step 5).
- **L4+L6, the first context (rulings §M32 C-1 (a), §M38 C-1):** the pack's per-expert
  ring-row increments and the rest of the first context's head-slot set, from
  `context_demand_records` at the envelope's shape, charged to the device reservation, with no
  term named at the reservation site; **it does not land before llama.cpp-fkpg (a)**;
  the `FIRST_CONTEXT` pending term recorded with the `WEIGHT` ranges, retagged at A4 step 3,
  taken over at the first context's step 5 in one group-mutex section, re-recorded by guard
  phase 1, and cleared at unload; H7ap's first-context arm.
- **L4+L6, the zones (rulings §M32 I-2; §M34 (4)):** RUNTIME with no floor on arena devices,
  laid out at 0 before the pack; the MMID pools in `REGION`; the host runtime pre-size at
  `ggml-sycl.cpp:5647-5652` re-derived to the host-tier terms; and the ONEDNN floor, removed
  by whichever of L4+L6 and 23mk's Graph-scratch commit lands later, with each commit's arm
  checking whether the other has landed and the earlier one keeping named planned terms in
  place of the constant (rulings §M37 Q3; the end states). The SCRATCH floor is not L4+L6's:
  it goes in the commit that converts the last untermed SCRATCH census row, whoever owns it
  (the end states).
- **L4+L6, the oneDNN scratchpad (rulings §M31, §M32 I-5, §V16 I-A, §M35):** the load's
  descriptor table over every family under its gate; the per-context queue set, frozen at the
  transaction; `assert_scratchpad_planned(stream)` in place of `pre_allocate_scratchpad`'s
  size argument and global read; the climb as a C-term re-place swapped into
  `scratchpad_map[q]` at the publish; the `get_scratchpad_mem` site gate.
- **L4 replaces zhcn's H2 (1)/(1b) interim arm with zhcn 5.13's arm (1h), in the same
  commit** (zhcn ships the interim arm because it lands before A becomes a head slot). (1h)
  checks three things: the head-slot extents are disjoint in `{CONTEXT, id} ∩ REGION`, A
  (`onednn_pp_a`) makes 0 RUNTIME free-byte queries, and `REGION` is whole after a CAS
  refusal. Its REDs are a `PENDING_TERM_ALL` guard, and the interim RUNTIME fit run on an
  exactly sized RUNTIME. H2's per-context rows arm (§3.1) makes the same RUNTIME-query
  assertion for the ring rows.
- **L4, the primitives (23mk 4.8 (a), (c); 1oxa rev 12):** the `ONEDNN_GRAPH_SCRATCH` and
  `FIRST_CONTEXT` pending terms; `VM_TAIL_SURPLUS`'s replace-per-(term, device, TLSF) rule,
  the partial trim, the `{DEVICE, d}` owner rules and a failed first load's direct recording
  (r15 m-3); the keyed draw's one `{TLSF, offset, size}` tuple over each SYCL<n> buffer and
  each individually drawn tensor or copy (r15 m-4); `classify_references` as one callable.
- **Not moua's:** the ONEDNN W setter's re-point and the five-name rename for W (23mk §4,
  rulings §M32 I-4, §M33 I-G; the `onednn_pp_w` row of §2.4.5); moua adopts the names, and
  7.14e's "deleted from both sites" is withdrawn.

**Revision 7.14h's changes to the rows (rulings §M38, §G1b).**
- **L4 waits for llama.cpp-fkpg (a) (rulings §M38 C-1).** The `FIRST_CONTEXT` reservation is
  sized at the envelope's `n_ctx`, and both merge gates run `-c 4096`, so the reservation
  cannot land until the envelope carries the caller's context.
- **L4+L6's prerequisites on RUNTIME (rulings §M38 C-2):** zhcn's carve of the oneDNN FA
  materializations and 23mk's reorder temporary land before it, jzvq before L4 as before, and
  beni's rows 13 and 14 keep the transitional `xmx_moe_buffers` term until beni converts them.
  L4+L6 deletes `dense-scheduler.cpp:29`'s caller-less draw and master's 512 MiB RUNTIME
  default at `unified-cache.cpp:4505-4508` (I-3), and adds C9, zero `[EXT-ALLOC]` lines on both
  merge gates (§3.3).
- **L4+L6, the ring's setters (rulings §M38 I-1):** the arena-path calls of the four
  device-global setters are deleted, and H7z (aj) gates their reachability.
- **L4+L6, the weight slot (rulings §M38 I-2):** the store is keyed by (model, device), and a
  later load's D terms are placed in RUNTIME's free room or the shared zone inside that load;
  H9's second-load arm.
- **L4+L6, the ring's engagement line (rulings §M38 I-5):** the per-context
  `[PP-MOE-RING]` WARN line, in every build.
- **The strict switch (rulings §G1b):** uwlx exports `ggml_sycl_strict_enabled()` and makes
  §G1a's three doc edits; L4+L6 carries them only if it lands before uwlx.

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
the owner-visible record. So: jehw → uwlx → 423j, and uwlx → moua L4. And fkpg (a) → moua
L4 (rulings §M38 C-1).

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
  - **u1bb (in master):** the ring's rows become per-(context, device) CONTEXT head slots in
    the context's slot table and its weight slot a per-(model, device) D term with its own
    setter (7.14f; rulings §M32 I-1, I-3; 7.14e's DEVICE-scope record with contributions is
    withdrawn); the executor reads and claims the context's rows;
    `release_pp_moe_onednn_scratch_ring` is no longer called by the full transaction; the
    `zone_largest_free(KV)`, `ggml_sycl_kv_capacity_live`, `ggml_sycl_device_kv_bytes_with_slack`
    and budget-room reads are deleted for arena devices; its "does not fit" refusal becomes the
    fit's head-slot refusal.
  - **zhcn (agreed, §6.6):** the measure pass and its walker (with beni's visitors), its tenant
    records and the descriptor's tenant section, the claim by live-object index, step (i)'s
    pre-L1 release, no fallback, charging at moua's commit, one STRICT switch with no alias (now
    `GGML_SYCL_STRICT_LEASES=1`, rulings §G1), the scheduler reset before a candidate publish,
    the gallocr return check, and the deletion of `k_pp_moe_ring_compute_reserve_bytes_per_row`.
    zhcn cites §2.3.2 and §2.4.4 as normative.
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

- **(a) Refuse by default, and abort under `GGML_SYCL_STRICT_LEASES=1` (§G1):** adopted, at
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
    the MXFP4 MoE TG caches; moua the recurrent state and u1bb's ring rows, which are
    per-context CONTEXT slots (7.14f; rulings §M32 I-1; 7.14e's one DEVICE-scope record is
    withdrawn). moua L3 consumes the rest.
  - Landing: jehw → moua L1-L3 → zhcn → beni producers → moua L4-L7 → beni conversions; 23mk
    core after jehw; jzvq before L4 (§4).
- **(f) The runtime context crosses the ABI once (r2 N-C1; lead ruling 4, D4).** One versioned,
  `struct_size`-headed descriptor, whose layout moua owns, with a KV-shape section and a
  recurrent section (moua) and a measured-tenant section (zhcn, carrying beni's and jzvq's
  demands). The old entry point is kept (§2.4.4).
- **(g) The STRICT switch: superseded by rulings §G1 (7.14g).** Lead ruling 4 (D3) named one
  plan variable for moua, zhcn, beni and jzvq, with no alias (agreed with zhcn, §6.6). §G1
  replaces it with `GGML_SYCL_STRICT_LEASES=1`, the switch every ownership, lifetime and plan
  defect family folds into (§G1b), read once through `ggml_sycl_strict_enabled()` (§G1a),
  which uwlx exports. The tags stay per family (§2.8).
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
- **(j) One tenant protocol (r4 I3; confirmed by impl-zhcn 2026-09-26, with A1-A4 and M1-M2).**
  Held handles, claim by index, event-chained reuse, zhcn's step (i), no fallback, one charge
  site, one strict switch for the family (§G1, §G1b). This document is the normative spec
  (§2.3.2); the record of who changed what is §6.6.
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
  = −709.4 MiB from 7.14f (rulings §M32 I-1, I-2), and 6 layers demote (§2.4.1, with the
  breakdown). The −714.9 agreed with zhcn rev 4.1 (`4bb0436`) carried the SWA term at `-ub
  512`; design-zhcn-r4 found it, and both designs printed −726.9 from 7.2 to 7.14e (planning
  −752.2, not re-derived since, because the score uses the live figure); revision 6's 728.4 is
  withdrawn. The pre-registration is scored by ⌈−printed / 128⌉ = 6, which neither correction
  changes. The state after the ONEDNN floor goes is H2's run at that state (§2.4.1).

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
impl-jehw; (2) the ring at the reservation's `n_ubatch` is a planned mandatory CONTEXT demand, a
head slot in the one fit; (3) zhcn deletes `k_pp_moe_ring_compute_reserve_bytes_per_row`, and
moua widens H7p to every context-side admission; (4) the zhcn contract is accepted, D3 named one
plan STRICT variable (superseded by rulings §G1: `GGML_SYCL_STRICT_LEASES=1`), D4 is one
versioned descriptor whose layout moua owns; (5) landing order jehw, then moua L1-L3, zhcn,
23mk, moua L4+. Also: 23mk owns the in-arena per-op TRANSIENT demand functions, and
gap-before-holes is approved.

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
| STRICT | one plan variable; a KV alias must not occur | one plan variable aliasing a KV one | one variable, no alias; **superseded by rulings §G1 (7.14g): `GGML_SYCL_STRICT_LEASES=1`** | **zhcn** |
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
  wrapper's other returns. Revision 7.6 left a shutdown race open here; rulings §M76.5 closes it
  in 7.7: shutdown and reactivation take L0 (§2.4.2).
- The load's L0 span is a choice (per entry, not load_begin..load_end). The alternative covers
  the whole load and needs a callback clause in the deadlock rule.
- The MMID host pool is held by the model's MMID entry, not a context's reservation, because
  the workspaces are keyed by the model token. (Superseded by rulings §M9 I-3: the pools are
  queue-bound, so they are CONTEXT scope.)

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

### 6.11 Design review r8 (design-moua-r8 on `874fe5490`), the lead's rulings §M8, and the r8 queue

design-moua-r8 found 0 Critical, 5 Important and 12 Minor. The lead ruled in rulings §M8, and
this round also carries the items queued since 7.6 (§M76's conditions, §M76a, §Z42.3, §ZR5).
Revision 7.7 answers every item; none is deferred inside moua. Item 0: `task/moua` merged
master `76c7f6548` at `db609bd15`, and every new line was checked with `git show 3d9414c8c:`.

| item | finding | disposition |
|------|---------|-------------|
| I-1 | two exported publishers are missing from the L0 census, and H7ai has no test-only class | **Changed (rulings §M8 I-1).** The census is now by reachability from every exported function (§2.4.2 "Where it is taken"). It found `set_runtime_context` (token at its top, nested under the wrapper), `shutdown` (§M76.5), `compute_placement_plan_early` and `test_hold_live_update`. Three exported publishers have no caller in the tree and are deleted in L6: `set_runtime_n_ctx` (plan A5), `set_model_loading` and `set_tensor_inventory`, with the gates that name them re-anchored. H7ai walks reachability and has a witness for an unlisted exported entry that reaches the CAS; the `test_*` callers of `publish_test_plan` are a named test-only class. H9's arm runs every listed entry. |
| I-2 | the probe's per-model fix stops at the entry; the body still reads the global snapshot | **Changed (rulings §M8 I-2).** The body takes `current` as a parameter, with no read at `:17852`; the expected-model check and both relocks (`:17886`, `:18122`) use the per-model plan, as does the recheck's re-confirmation (`:19054-19056`; the reviewer's `:19065-19067` is the pvjr comment). H9 gives A and B different residencies and asserts A's. |
| I-3 | per-entry load L0 lets a transaction observe B's half-finished load | **Changed (rulings §M76.2, §M8 I-3).** §2.4.2 names what an interleave observes. (a) The inventory state is per model (§M76a), and `g_sycl_in_model_load` is the loading thread's flag. (b) B's plan holds its device weight room as load pending ranges from `stage_inventory_plan` until the preload consumes them. (c) Identity is the bound candidate. H9 arm: A's transaction between B's stage and `load_end`; H7ap. |
| I-4 | COVERED is possible after a refused ring-only growth, with the ring empty | **Changed (rulings §M8 I-4).** A cleared tenant key, or an empty ring record for a contributor, answers GROWTH. The query reads the entry (with a copy of the ring contribution) under the leaf, then the ring record under its lock alone. H4 ring-only arm, RED on 7.6's rule; zhcn told. |
| I-5 | the MMID first materialization is outside the transaction; "no MODEL scope"; two fit sources; host carve sizing | **Changed (rulings §M8 I-5).** (a) The fit plans the model's device pool whenever it is unmaterialized or too small, step 7 materializes into the carves, and the wrapper's `:18978-18990` materialization is deleted; H7am enumerates call sites, with a RED arm reaching `:18980`. (b) Three scopes, MODEL, CONTEXT and RUNTIME, each with its owner (§2.4.3 "Scopes", §2.3.2, §2.4.1). (c) Step 3's MMID budget is removed on arena devices; (0) and (ii) run the same sizing then fit; H7aq. (d) The model's carves are sized at the largest rung any context may admit (up to `MOE_GPU_UBATCH_MAX`); beyond that is a planned growth of the model's carve in that context's transaction, under L0. The host carve is an owner-first `unified_allocate_owner` request. |
| m-1 | the coverage query's ring read and the zhcn API alignment | **Changed.** The two-section read is specified (I-4). The zhcn difference (its key recording, and coverage inside its publish) is tracked as open, below. |
| m-2 | two misstatements of zhcn 5.3 | **Fixed.** zhcn checks and opens the table at (e), not (c), and 7.7 says so; zhcn 5.2/5.3 already lists the three queues. |
| m-3 | census cites and one container | **Fixed.** `:19985`/`:19998` are inside `ggml_sycl_moe_ids_cache_new_graph` (`:19979`), called at graph start (`:92321`) and teardown (`:12199`); `:33767` is the preload's `MID_LOAD_REPLAN` clear. `g_moe_down_shadow` is named, with its key fix and an H7ag witness. |
| m-4 | the `!tenants_planned` non-FA check reads live free memory | **Changed (rulings §FM).** On an arena device it reads the registry, as the recheck does. For a `tenants_planned` context the FA recheck is subsumed by MEASURE and the coverage query. |
| m-5 | BUSY returns on the other L0 entries are unclassified | **Changed (rulings §M76.4, §M76.5).** Each is listed and classified as `[CONTEXT-PLAN-BUG]`; concurrent loads stay a named `LOAD_BUSY` refusal; §6.10's shutdown hedge is dropped. |
| m-6 | the load entry list | **Fixed.** The public entries are `load_begin`, `stage_inventory_plan` (with `compute_placement_plan_early`) and `load_end`; the preload is static inside `load_end`; `load_enter_nested` does not publish. |
| m-7 | §2.4.4 says the FA recheck sends the KV-shape copy | **Fixed:** it takes no descriptor. |
| m-8 | the ring max across two models | **Changed.** Contributions are per-kind records; the ring is their component-wise max, and the install writes that max to the device's planned setters. H8 arm. |
| m-9 | EQUAL and COVERED skip the binding refresh | **Stated** (§2.4.2 covered path): neither changes anything the binding records. |
| m-10 | host sizing: moua's per-index max versus zhcn's top candidate | **Changed (rulings §ZR5 I-1).** The max over every rung, per index; MEASURE measures every rung; zhcn aligns to moua. |
| m-11 | rulings list; present-tense busy in §2.3.1 | **Fixed.** |
| m-12 | a possible second ring holder | **Stated:** the ring record is the only storage; H4 asserts `use_count() == 1` with a second-holder witness. |

**The r8 queue.**

| item | disposition |
|------|-------------|
| §M76.1 condition (covered read vs a concurrent L0 holder) | §2.4.2 "Why COVERED is safe without L0" shows condition (a): everything COVERED depends on is owned, not free room. H9 arm with a parked holder running an unload, a quarantine restore and a load. |
| §M76.2 condition (what an interleave observes) | I-3 above. |
| §M76.5 (shutdown and reactivation take L0) | In the L0 list; a non-ACTIVE module under L0 is `[CONTEXT-PLAN-BUG]`; H9 arm. |
| §M76a (inventory globals) | §2.4.2 "Per-model plan state"; H7ao with the `:18245` witness, scoped by path set. |
| llama.cpp-fsgi pointer | One line in §2.4.2: `get_cached_tensor_ptr` (`:19236`) is fsgi's, reuses the snapshot, lands after moua. |
| §Z42.3 (no per-record zone) | The zone line and the record's zone field are deleted (§2.4.3 "Zone"). |
| §ZR5 I-1 (host sizing over every rung) | m-10 above. |
| §ZR5 I-2 (one owner-first reservation per slot index, contiguous) | §2.4.3's HOLD is per-index reservations through the existing nsl3 contiguous path (revision 7.7a; 7.7 wrongly routed a large slot to `allocate_runtime` as an L4 gap); H4 and H7an. |
| §ZR5 I-3 (identity compare and republish in one section) | §2.4.2 "The allowlist": the four sites call `ggml_sycl_republish_current_plan_into_empty(cache, owner)` (revision 7.7a, lead ruling; the `owner` argument since 7.10, §Z6 I-C; three sites once `:60125`'s dead function is deleted), which writes no device-global; H7ai and H9. |

**Revision 7.7a: the lead's allowlist ruling and zhcn 5.4's §3.8 rows (head `1547f42`).**

| row | request | disposition |
|-----|---------|-------------|
| 19 | host HOLD: one owner-first reservation per index, no offset views, `R_h` over the rung set | **Adopted** (§2.4.3). The contiguous path is nsl3's `try_zone_alloc_contiguous` → `grow_zone` by one request-sized chunk, cited at `3d9414c8c`; 7.7's `allocate_runtime` routing and its "L4 gap" are withdrawn. `llama_auto_ubatch_rung_set` is zhcn's new helper, not a `3d9414c8c` symbol. A need above `R_h[i]` is a setter-only refusal (zhcn's E9), and at a ladder rung a `[CONTEXT-PLAN-BUG]`. H4 asserts the slot's growth before scoring and adds the chunk + 1 MiB contiguity arm. |
| 22 | the C2t flush is `ggml_sycl_cpu_tg_flush_pending()` over four lists | **Adopted** (§2.4.2 (s)); cites checked at `3d9414c8c`. |
| 26 | the §M76.1 monotone statement for the covered path | **Adopted** (§2.4.2 "Why COVERED is safe without L0"): the entry's only writers, and the ring as a max over live contributions. zhcn's two H9 arms are pointed to. |
| 27 | H7ag gains the four scatter lists and `g_moe_down_shadow` | **Adopted** (§2.4.2 census, H7ag: nine named containers; the not-holders listed). One cite corrected: the shadow key's handle compare is `:21130-21131` (the `if (use_handle)` and its return), not `:21127-21129`. |
| 30 | the allowlist calls `ggml_sycl_republish_current_plan_into_empty(cache, owner)` (the `owner` argument since 7.10) | **Adopted** (lead ruling; §2.4.2 "The allowlist", H7ai, H9). The emptiness check outside the mutex is at `:5885-5886`. |

**The two cross-design items are closed (7.7a), and this is the agreed text.**
- **The table opens at (c).** zhcn 5.4 §3.1 step 5(c): under `kv_region_mutex_`, on each
  device, read each slot's atomic claimed flag only (a still-claimed current tenant slot is
  `[CONTEXT-PLAN-BUG]`, nothing taken); otherwise take every holder of the published table
  (move the entry's `shared_ptr` into the batch, reset each backend context's cached pointer);
  check `use_count() == 1`, else `[CONTEXT-PLAN-BUG]` naming the holder with nothing moved;
  only then open it (covered device slots and every host reservation into the registry entry,
  growing device slots and unused indexes into the batch; drop the emptied table); clear the
  tenant key; unlock. The reap's owners are the batch's device handles plus the ring's moved-out
  slots, never a host reservation, destroyed at the end of (d); (e) checks only those. §2.4.2
  (c) says the same.
- **The covered path's read (§M8 I-4).** zhcn adopts this design's wording as is: the coverage
  query reads the registry entry's copy of the ring contribution under the `kv_region_mutex_`
  leaf, releases it, then takes the ring record's lock alone, never nested; it answers GROWTH
  if the tenant key is cleared, or if the context still contributes and the ring record is
  empty, whatever the caps say. The §M76.1 statement holds with two sections, because the key
  and the contribution are written only by the context's owner-thread transaction and its
  release proc, and the ring never drops below a live contributor's contribution. zhcn carries
  this design's refused-(ii) revert arm in its H2/H9. It lands in zhcn's round after
  design-zhcn-r6, on the lead's hold, so it is agreed text not yet in zhcn's file.

zhcn 5.4 also uses the read-only query on the covered path and names 5.3's key-record write as
its RED (row 26).

**The census for the three deleted setters (§M8 I-1; lead approval, conditional on the whole
tree).** Command, run in the worktree against master `3d9414c8c`:

```
git grep -n -E 'set_runtime_n_ctx|set_model_loading|set_tensor_inventory' 3d9414c8c -- \
    src ggml tools examples tests common include ggml/include
```

It finds no call of `ggml_backend_sycl_set_runtime_n_ctx`, `ggml_backend_sycl_set_model_loading`
or `ggml_backend_sycl_set_tensor_inventory` outside their own definitions
(`ggml-sycl.cpp:19086`, `:13273`, `:16745`). `src/`, `tools/`, `examples/`, `common/` and
`include/` have no hit at all. The other hits are:
- the declarations and their comments, `ggml/include/ggml-sycl.h:349`, `:356`, `:378`, `:418`,
  `:596`, `:1430` and `:1434`, deleted with the definitions;
- comments only: `ggml-sycl.cpp:2632`, `:14733`, `:15889`, `:25223`, `:35529`,
  `layer-streaming.cpp:23`, `unified-cache.hpp:2432`,
  `ggml-sycl/tests/test-unified-runtime-alloc.cpp:705`, `tests/test-tiered-dispatch.cpp:95`,
  `tests/test-sycl-model-lifecycle-hooks.cpp:26`, `:39`, `:207` and `:283` (the last a
  diagnostic string), and `tests/test-sycl-tiered-verdict-contract.py:22` ("a direct
  set_tensor_inventory() call stages no plan"; r9 addendum m-13);
- `ggml_sycl_set_tensor_inventory_impl` (`:16491`), the **late** stage's plan computation,
  which **stays**. Its callers are the deleted wrapper (`:16748`) and the stage's late branch
  (`:16826`, the `else` of `if (early)` at `:16823`), so only the exported wrapper at `:16745`
  goes. It is not behind `compute_placement_plan_early` (r9 addendum m-13): that function
  (`:16473-16489`) calls `populate_inventory_globals`, `compute_vram_budget_for_plan` and
  `compute_and_store_plan_for_inventory` directly;
- source-contract tests: `test-sycl-lifecycle-source-contract.py:445` and
  `test-sycl-tiered-verdict-contract.py:192` assert the call is **absent** and stay true;
  `test-sycl-tiered-verdict-contract.py:237` is a text mutant and needs no symbol;
  `test-sycl-nonfa-attn-scratch-guard-source.py:294`, `:344` and `:481` and
  `test-sycl-host-zone-config-source-contract.py:12-14`, `:92`, `:109-113` and `:187` anchor on
  the deleted text, and are re-anchored in L6 as §2.4.2 records (the host-zone check's
  `setter_fn` window fails closed when empty, so it cannot pass vacuously).

The rest of the tree has hits only in `docs/` (history and plans), which is not re-anchored.

**Noted for the lead.**
- **The phase gate (checked for 7.7a).** 7.7 left open whether the MMID host carve's growth
  past `MOE_GPU_UBATCH_MAX` trips `GGML_SYCL_HOST_ALLOC_PHASE_GATE`. It does, and so would
  7.7a's row-19 claim that a first publish is "outside PP and TG": `g_offload_phase` is
  process-global and set at every graph-compute entry but never cleared at exit, so both
  growths read as PP or TG once any graph has run. The design now keys both gates on the
  replan token's held flag (§2.4.3; H4, H7an). This is a new mechanism in L4, and the lead
  may prefer another shape. (The lead accepted it as §M77; revision 7.10 narrows it to
  `TRANSACTION` tokens under §M9a, §6.12.)
- **`db609bd15` was not built as a whole, and it is not doc-only.** It merges master `76c7f6548`
  (61 files over `874fe5490`) into a branch whose own delta over master is this document plus
  L1: `tlsf-allocator.hpp`, `shared-zone-tags.hpp`, `tests/test-tlsf-allocator.cpp` and
  `tests/CMakeLists.txt`. The commits after it (7.7, 7.7a) are doc-only. The merge touches none
  of L1's sources, only `tests/CMakeLists.txt`. After it, `sycl-build.sh test-tlsf-allocator`
  reconfigured, found no work, and the host-only `test-tlsf-allocator` passed (25 PASSED lines,
  rc 0) at `85635feee`. Nothing else was built; no source file this design cites differs from
  `3d9414c8c`.

### 6.12 Revisions 7.9 and 7.10: design-moua-r9, its addendum, and the rulings (§M9, §M9a)

design-moua-r9 reviewed `85635feee` (7.7 and the commit labelled "7.8"), not 7.7a; the lead
extended the next review to `85635feee..HEAD`. Its findings marked "[7.7a addresses]" (the
census command, the (c) table text, the phase gate) are in 7.7a and are not repeated here.

| id | finding | disposition |
|----|---------|-------------|
| I-1 | the exported `set_runtime_context` has no model identity, so the global read comes back through the entry | **Changed (rulings §M9 I-1).** It stops being exported and is folded into the wrapper, which calls the body with the per-model snapshot it selected; its four `thread_local` side channels become parameters and the return. Whole-tree callers: the wrapper and the deleted `set_runtime_n_ctx` only; the source gates that quote it are re-anchored. H7ai: "A re-plans while B published last", and a witness restoring the exported entry. |
| I-2 | a third MMID site at `load_end` (`:13181`); the fit ignores route reachability; H7am cannot pass | **Changed (rulings §M9 I-2).** Three sites listed; `load_end`'s and the wrapper's are deleted; the fit plans a pool only under `ggml_sycl_moe_mmid_route_reachable(ctx)`. H7am gains GREEN arms (unreachable route plans nothing; two contexts each admit) and witnesses restoring `:13181` and `:18980`. |
| I-3 | MODEL scope contradicts the queue-bound registry | **Changed (rulings §M9 I-3, amending §M8 I-5(b)).** MMID pools are CONTEXT scope: held in the context's carve, bound to its queue, keyed by model token, ContextId and queue cookie, released at the context's destruction; supersession only within the context. The "a second context is covered" claim is deleted; MODEL scope keeps model-lifetime terms only; the covered-safety bullet, §2.4.3, §2.7, the owner list and §4 follow. |
| I-4 | "B's own" undefined; the loader's SYCL<n> weight buffers and the compute-arena reserve fall between | **Changed (rulings §M9 I-4).** Ownership is B's owning-load identity on each range (23mk's §Z5 IMP-6 addition, one primitive). The late stage (`:16826`) records the ranges; the SYCL<n> weight buffers (`llama-model.cpp:2634`) are B's own and draw from them; each byte is drawn once (the preload stages only under evictable weights, `:33526-33528`); the compute-arena reserve is a range of B's own and its `GGML_ABORT` (`:12470`) becomes a named refusal. H7ap and H9 arms. *Superseded in 7.11 (§6.13, I-B, I-C, §X7 I-3):* the early stage records; the compute arena has no range; the SYCL<n> buffers draw first; "drawn once" rests on the tensor sets. |
| F2 / m-1 | `thread_local` needs a same-thread precondition; readers unnamed; the offload phase | **Changed (rulings §M9 F2).** The load flag is keyed by the load's identity: readers ask whether the calling thread is bound to an active load, which survives `load_end` on another thread (`:13110`). Readers named, `llama-context.cpp:997` included; the phase stays process-global and is llama.cpp-dhpw. H7ap's T1/T2 arm. |
| F5 / m-10 | the H9 every-publisher RED needs "B published last" | **Fixed.** Stated in the arm, with why it is void otherwise. |
| m-2 | `compute_placement_plan_early` publishes nothing | **Fixed.** Off the publisher lists; made static (its one caller is the stage). |
| m-3 | the dead `publish_plan_locked(nullptr)` at `:14773` | **Fixed.** The branch and the default argument are deleted; H7ai witness. |
| m-4 | `test_hold_live_update` cannot model an L0 holder | **Changed.** It keeps master's form (a lease, no L0) in H7ai's test-only class; `test-sycl-lifecycle-runtime-wrapper.cpp:1476-1492` is unchanged; parked-holder arms park inside a real entry. |
| m-5 | `quarantine_token` misdescribed; the backend-free reap path; shutdown's joins | **Fixed.** `quarantine_token` only enqueues; the reaper runs from `load_begin`, shutdown and every `ggml_backend_sycl_free` (`:84108`), each retry an L0 hold in `unloaded_token`; H7ai walks the four worker threads' entries. |
| m-6 | `load_begin`'s closed-module `LOAD_BUSY` vs §M76.5 | **Rationale recorded.** It is the Registry shutdown gate's own answer, a named load refusal llama throws on, not the retryable `BUSY` §M76.5 removes. |
| m-7 | the phase-gate sentence | Fixed in 7.7a; 7.9 added a load-entry exemption (r9 F3), which 7.10 withdraws under §M9a (below), and the dhpw pointer. |
| m-8 | zhcn divergence | Fixed in 7.7a; zhcn has confirmed both items since. |
| m-9 | the MMID carve growth has no place in the sequence | **Fixed** (§2.4.2 step 7): the device growth is one of (0)'s placements, carved at (ii); the host growth is allocated directly after (0), before (i) releases anything, outside L1. |
| m-11 | three stale statements | **Fixed.** Probes see a load's pending ranges; `g_model_n_layer` and `g_placement_kv_info` become load scratch in the inventory record before deletion; the preload's `g_moe_*` total reads join §M76a's list. |

**The queue folded in with r9.**

| item | disposition |
|------|-------------|
| §Z6 I-C (`into_empty` installs the owning model's plan) | §2.4.2 "The allowlist" step 1: the owner is the calling context's execution root, selected per model; H7ai's two-model arm. |
| §Z43.4 / §M77 (one accessor) | `ggml_sycl_replan_token_held()`; the gates, `:33769`'s assertion, 23mk's witness and 1oxa's assertion call it; no second flag. 7.10 adds its kind argument and its definition site (below). |
| llama.cpp-dhpw | One pointer in §2.4.3's phase-gate rule, and one in the load-state text. |
| §Z5 IMP-6 (one pending-range primitive) | This design's names are canonical. The owning-load identity and sub-allocate-within-own-range are used in (b); **impl-23mk's exact spec for its three additions (sub-allocate and free, the owning-load identity, a `zone_replace` merge) had not arrived when 7.9 was committed**, so the primitive's text will take it as sent, in the next round. It arrived for 7.10 (below). |
| zhcn's cite | The shadow key's handle compare is `:21130-21131`. |

**Noted for the lead.**
- The load flag's new predicate costs a Registry lookup on every SYCL_Host allocation and caps
  query. Both are rare (buffer creation, device properties), so it is not a hot path.
- The weight-mode claim (a device weight byte is drawn by the SYCL<n> buffer or by the
  preload, never both) rests on the preload's early return at `:33526-33528`; H7ap measures it
  in both modes rather than assuming it.
- Nothing was built for 7.9; it is a document change only.

**Revision 7.10: design-moua-r9's addendum on 7.7a (rulings §M9a), 23mk rev 4.4 §8.1, zhcn
5.5.** The addendum reviewed 7.7a (`f8a628420`) and left 5 Important and 11 Minor open. I-1 to
I-4 and the 85635feee Minors are the ones 7.9 answered (table above); the new items are:

| id | finding | disposition |
|----|---------|-------------|
| I-5 (a) | the exemption covers `load_end`'s preload: expert host fills, host-zone configuration and the VRAM-pressure fallbacks (`:34169-34177`, `:33888`, `:33530`) | **Changed (rulings §M9a).** The token has a kind (`TRANSACTION`, `LOAD`, `LIFECYCLE`) recorded for the outermost hold, read through the same accessor with a kind argument; the gates skip only on `TRANSACTION`. Load-time growth stays gated; its false positive while another model is in TG is llama.cpp-dhpw's. 7.9's r9 F3 exemption is withdrawn. H4 negative arm (a) at gate 1: the WARN fires and its `site=` names B's load. |
| I-5 (b) | llama's replan scope has no stated end relative to `sched_reserve` | **Changed.** The scope closes after the guard's second phase and before `sched_reserve`; `alloc_buffer`'s host-pinned fallback asserts no `TRANSACTION` token, graph compute asserts no token at all. H4 negative arm (b): the resync's `sched_reserve` fallback asserts at gate 2. *Superseded in 7.11 (m-4):* the vehicle is `sched_reserve_impl`'s ALLOC, since zhcn deletes the resync. |
| m-12 | a stale "open (r8 m-1)" paragraph | **Fixed.** Restated as closed: zhcn 5.4 uses the read-only query (row 26). |
| m-13 | the census misdescribes `_impl` | **Fixed.** `_impl` is the late stage's computation (`:16826`); `compute_placement_plan_early` computes directly (`:16473-16489`); `tiered-verdict-contract.py:22` joins the comment hits. |
| m-14 | the accessor is unnamed and has no definition site | **Fixed.** Named, with its signature; declared in `unified-cache.hpp` beside `offload_stats_phase()`, which `pinned-pool.cpp` already reaches, and defined in `unified-cache.cpp` with the mutex, the `thread_local` state and the token type, which is the state's only writer. |

| item | disposition |
|------|-------------|
| 23mk A1 (owner identity: `LOAD`, `MODEL`, `CONTEXT`, `DEVICE`; carried in `alloc_constraints::pending_owner`) | **Adopted** (§2.3.3 "The pending-range primitive"). "B's own" in (b) is `{LOAD, B's LoadTxnId}`, filled by the caller, never read from a thread-local inside the allocator; this design's transaction ranges are `CONTEXT` ranges. |
| 23mk A2 (`allocate_within`, `consume`, `clear_pending`) | **Adopted.** The SYCL<n> weight buffers, the preload and the compute-arena reserve draw with `consume = true`; a miss is the owner's plan bug and never falls through. *7.11:* the compute arena draws nothing (I-B), and each call names a term (I-A). |
| 23mk A3 (`replace_within`) | **Adopted** into the primitive; this design has no caller. |
| 23mk A4 (`retag_pending`) | **Adopted**, with one ordering rule of this design's: at `load_end`'s commit the retag runs before this design's `clear_pending({LOAD, txn})`, so the clear cannot drop 23mk's terms. *Superseded in 7.11 (I-A):* the retag takes a term filter. |
| zhcn 5.5 row 35 (`into_empty(cache, owner)`) | **Adopted.** The owner is `ggml_sycl_execution_current_owner(ctx)`'s token, threaded through `ensure_moe_secondary_queues_for_plan` from its callers; an unbound context skips the call; `has_global_plan()` goes at these sites; a null selection is `[CONTEXT-PLAN-BUG]`. **Found while checking the callers:** `:60125`'s function, `materialize_moe_tensor_planned_layout`, is `static` with no caller at `3d9414c8c` (only `:29455` and `:60062` name it), so it is deleted as dead code and three live sites remain. |
| zhcn 5.5 row 32 (`can_unload` closes admission) | **Adopted.** `ggml_backend_sycl_can_unload` (`:109358`) takes a `LIFECYCLE` token; its drain wait under L0 cannot deadlock, since publishing entries take L0 before the module guard. *Superseded in 7.11 (I-F):* the token is taken by `try_lock`. |
| zhcn 5.5 row 37 (`site=` in both phase WARNs) | **Adopted**, fed from the request's cohort, or its role and tag. |
| zhcn 5.5 row 27 (the shadow-key cite) | Already `:21130-21131` in 7.9. |

**Noted for the lead (7.10).**
- The token has **three** kinds where §M9a names two. `TRANSACTION` versus `LOAD` is the
  ruling's discrimination and the gates use only `TRANSACTION`; `LIFECYCLE` exists so that the
  probe, the FA recheck, activate, unload, `can_unload`, shutdown, reactivation and the release
  proc are not labelled as a transaction or a load. It is still one state and one accessor. If
  the lead prefers two values, those holders take `LOAD` and nothing else changes.
- The dead `materialize_moe_tensor_planned_layout` deletion reaches zhcn's row 35 list
  (`:60125`, `:60115`); I am telling zhcn.
- Nothing was built for 7.10; it is a document change only.

### 6.13 Revision 7.11: design-moua-r10, the rulings (§M11, §M10), and §X7 I-3

design-moua-r10 reviewed `85635feee..9d4d826b3` and gave NOT READY: 0 Critical, 6 Important,
7 Minor. The lead ruled on every item in §M11 and queued design-1oxa-r7's I-3 (§X7) and §M10
into this round.

| id | finding | disposition |
|----|---------|-------------|
| I-A | 7.10's `retag_pending(from, to)` moves every term, so a later load's `SCRATCH` hold meets B's undrawn leftovers retagged as model terms | **Changed (rulings §M11 I-A).** Each pending range carries a term tag (`WEIGHT`, `SCRATCH`, `MODEL_TERM`, `DEVICE_TERM`, `ARENA`, `REGION`) beside its owner. `retag_pending(owner, term_filter, new_owner)` moves only the terms 23mk names (`MODEL_TERM` to `{MODEL, id}`, `DEVICE_TERM` to `{DEVICE, dev}`); `clear_pending(owner)` then drops the rest; a rollback runs only the clear. One primitive, the same in both designs (§2.3.3). *7.11a (§M11a):* the clear and the fit query take the term filter too (below). H7ap arm: after commit B's leftovers are gone and C's `SCRATCH` hold is admitted; witness: an unfiltered retag. |
| I-B | the compute-arena "own range" does not exist: on arena devices the arena is `SCRATCH` itself | **Changed (§M11 I-B).** On an arena device `reserve_compute_arena` points at `SCRATCH` and allocates nothing (`unified-cache.cpp:20707-20740`), so there is no range: the early stage checks `zone_capacity(SCRATCH) >= arena_bytes` and refuses by name, replacing the `GGML_ABORT` (`:12470`). On a non-arena device it is a real `COMPUTE` allocation (`:20757-20790`) and draws on no range. H7ap arm and witness. |
| I-C | where the SYCL<n> weight buffer draws is unstated, and master's chain is RUNTIME first | **Changed (§M11 I-C).** With a load bound, `alloc_buffer` for a SYCL<n> weight buffer calls `allocate_within({LOAD, txn}, WEIGHT, ..., consume = true)` first; a miss is `[ZONE-PLAN-BUG]` and a null buffer (the load fails at `llama-model.cpp:2637`), never the RUNTIME-first chain (`:37642-37673`). This is a stated placement change from master. "Drawn once" rests on the disjoint tensor sets (the SYCL<n> buffers' tensors and `g_sycl_host_weight_extras`), not on the preload's early return. H7ap arms. |
| I-D | deleting `load_end`'s site does not remove the `contexts.back()` binding: step 7 calls the same materializer | **Changed (§M11 I-D).** The materializer takes `ggml_backend_sycl_context &` from step 7 and binds by `(ctx.execution_context_id, owner_device)`; it never resolves a context itself. H7am arm: C1 re-plans after C2 exists and binds C1's queue; witness: the `get_backend_context_for_device` binding restored. |
| I-E | the `!stable_mmid` trigger skips the first publish; the plan-keyed registry cannot hold per-context entries; no retire at context destruction | **Changed (§M11 I-E).** (1) The trigger is `mmid_route_reachable && ggml_sycl_context_mmid_needs_materialize(ctx, next->plan)`, so the first publish materializes and reports `PUBLISHED`. (2) The registry key is `(token, context_id, submit device)`, designed through dedup and the prepared replacement, `replace_plan`, `acquire`, the cookie, `exact_queue`, `admit`, `recover` (`:75299`, `:84289`) and `retire` at every site (`:12321` retires all and reports an ownerless entry; `:13209`, `:13245` deleted; `:18635`, `:18650` scoped to the context; a new retire at context destruction). (3) Release is by the `mem_handle` destructor only, since `blob.owner` is shared with the slices and leases (P2). §2.4.2 step 7; H7am arms and witnesses. |
| I-F | `can_unload` under a blocking L0 acquire hangs `test-sycl-lifecycle-runtime-wrapper.cpp:1476-1492` | **Changed (§M11 I-F).** A `LIFECYCLE` token by `try_lock`; on contention it returns `false`, which `ggml_backend_unload_checked` reports as `BUSY` (`ggml-backend-reg.cpp:745-749`). The test is unchanged. H9 arm replays it under a watchdog; RED: the blocking acquire. |
| m-1 | the probe's `thread_local` side channel (`:18794-18805`) survives | **Fixed.** Deleted; the probe passes its model as a parameter. |
| m-2 | §M10's CSV row and the audit gate are unnamed | **Fixed.** The dead function's deletion regenerates `docs/backend/sycl-static-storage-inventory.csv` (row 648, `planned_layout_log`) with `scripts/audit-sycl-static-storage.py`, and `tests/test-sycl-static-storage-audit.py --check` must return 0 (§2.4.2 "The allowlist"; §4 L6). |
| m-3 | three MMID source gates quote deleted code | **Fixed.** `tests/test-sycl-mmid-deferral-contract.py` (ctest `sycl-mmid-deferral-contract`), `tests/test-sycl-pp-moe-ring-kv-zone-source.py:199-207` and `tests/test-sycl-moe-resolved-batch-source.py:476-488` join the re-anchor list (§2.4.2 step 7). |
| m-4 | H4 (a) can pass without a growth; H4 (b)'s vehicle is the resync zhcn deletes | **Fixed.** (a) fills the pinned pool to capacity first and asserts a growth happened; (b) uses `sched_reserve_impl`'s ALLOC with the host fallback forced; both run on the landing tree (§3.1 H4). |
| m-5 | the shared invariant is worded differently in the two designs | **Fixed.** Verbatim in both: "no TRANSACTION token is held at gallocr ALLOC or at alloc_buffer's host fallback"; `alloc_buffer`'s entry asserts it for buffers outside a bound load, and the fallback asserts it too. |
| m-6 | F2's preload window is unstated | **Fixed**, and the lead has since ruled to follow master (noted below): during `load_end`'s preload the predicate answers **not in load**, as master's flag does (cleared at `:12449` before the preload). |
| m-7 | `load_enter_nested`'s flag write is unnamed | **Fixed.** `:12894`'s `loading_effects(true, false)` store to the flag (`:12414`) is a no-op under the keyed predicate and is deleted; the phase store (`:12415`) stays (llama.cpp-dhpw). |

| item | disposition |
|------|-------------|
| §X7 I-3 (the load's room is admitted at the early stage) | **Adopted.** B's ranges are recorded at `llama-model.cpp:657`, through `compute_placement_plan_early` (`:16824`), under `{LOAD, txn}`; the late stage (`:709`, `_impl` at `:16826`) consumes the admitted plan by plan identity, and a placement change there is the named refusal "the late inventory changes the placement admitted at the early stage", never a silent re-plan. H7ap arm (a transaction between `:657` and `:709`: no `TERMINAL` miss, no host-placed device plan) and H9's interleave. |
| §M10 (three kinds; delete the dead function with its CSV row and gate) | **Adopted**, with m-2 above. |

**Noted for the lead (7.11).**
- **§M11 m-6 says "the flag answers 'in load' during the preload, matching master"; master's
  flag answers not in load there.** `load_end` clears the Registry's `in_load` bit at
  `:12449`, before the arena reserve and the preload, so on master every preload read is "not
  in load". 7.11 follows master. If the ruling means "in load", it is a one-line move of the
  clear after the preload, and the H7ap T2 arm's expectation flips with it. **Decided (lead,
  after 7.11):** follow master. The ruling's "in load" wording was wrong on the fact; the
  predicate answers not in load during the preload, and H7ap's T2 arm stands as written.
- The signatures sent to impl-23mk for one primitive: the `pending_term` enum;
  `allocate_within(owner, term, size, align, tag, consume)`; `size_t retag_pending(pending_owner
  owner, pending_term_mask term_filter, pending_owner new_owner)`; the commit order (retag
  `MODEL_TERM`, retag `DEVICE_TERM`, then `clear_pending({LOAD, txn})`); a rollback runs only
  the clear; `clear_pending` clears every term. *Superseded by 7.11a (below):* the clear takes
  a term filter, and the fit query `pending_bytes` joins them.
- m-5 leaves two peer differences for the lead to reconcile, not this design: zhcn's kind count
  and its `into_empty` arity. **Closed (lead, after 7.11):** zhcn 5.7 / f6218f3 uses the same
  three kinds (`held(TRANSACTION)` at the pool gates, `held(LOAD)` at `:33769`) and
  `into_empty(cache, owner)` (its m-14), so the two designs agree on both.
- Nothing was built for 7.11; it is a document change only.

**Revision 7.11a: §M11a, design-23mk-r6's H1 arm, zhcn 5.7 and f6218f3.** These arrived after
7.11 was committed.

| item | disposition |
|------|-------------|
| §M11a (design-23mk-r6 I-2; §Z8 I-2): the term filter on every by-owner operation | **Adopted** (§2.3.3 "The pending-range primitive"). The three signatures are stated exactly: `retag_pending(owner, term_filter, new_owner)`, `clear_pending(owner, term_filter)` and the fit query `pending_bytes(owner, term_filter)`. This design's context clears pass `REGION` (the guard's rollback at the relock, §2.4.2 step 5), its fits and the commit re-fit count only `REGION` ranges as `own_ranges` (step 6, (ii)), and a load's commit and rollback clear with `PENDING_TERM_ALL`. H1 arms: a filtered context clear leaves 23mk's hold of the same context, and a fit counting every owner range is the witness that carves into it. |
| design-23mk-r6: H1's retag-then-clear arm is vacuous without an undrawn range | **Fixed.** The arm records an undrawn `WEIGHT` range and a `MODEL_TERM` range; after the commit the undrawn range is gone and the `MODEL_TERM` range survives whole as `{MODEL, B}`. Under 7.10's move-every-range retag the undrawn range survives, so the arm can fail. A4's step 3 now names its filter, so "the clear drops the undrawn ranges" holds as written. |
| zhcn 5.7 (a): the scope closes before ALLOC's `graph_reserve`, not before `sched_reserve` | **Fixed** in §2.4.2's heading and text, the replan-scope bullet and §2.4.3's fallback bullet. The growth and its scope are inside `sched_reserve_impl`. |
| zhcn 5.7 (b): two rows still say `into_empty(cache)` | **Fixed.** Both rows (§6.10's §ZR5 I-3 row and §6.11's row 30) now read `(cache, owner)`. |
| zhcn f6218f3: an H4 (b) vehicle that survives zhcn | **Adopted.** H4 (b) uses zhcn's H4h (f6218f3 §3.1, §3.3), `sched_reserve_impl`'s own growth path with the compute buffer forced onto the host-pinned fallback. It runs from a ladder rung, a setter's next decode (`sched_reserve_nothrow`) and `memory_update`. |

**Noted for the lead (7.11a).**
- The fit query's ruled name, `pending_bytes`, is also the name of a field of the reap's result
  (rulings §R, §2.4.2 (i)(d)). They are different things, a query on the pending-range
  primitive and a result field, and never meet in one scope, so I kept the ruled name. If you
  would rather rename one, the query is the easier rename.
- I sent the three signatures to impl-23mk again, with the filter on the clear and the fit.
- The lead's two decisions on 7.11's notes (m-6: follow master; m-5: closed on zhcn's side)
  are recorded at those notes above.
- Nothing was built for 7.11a; it is a document change only.

### 6.14 Revision 7.12: design-moua-r11, the rulings (§M12), and the post-r11 queue

design-moua-r11 reviewed `9d4d826b3..c66848c26` (7.11, 7.11a and the lead's decisions) and gave
NOT READY: 1 Critical, 4 Important, 13 Minor. The lead ruled in §M12 and queued §M11b,
§M11b-amend, §Z9 I-3 and §Z9a into this round.

| id | finding | disposition |
|----|---------|-------------|
| C-1 | the late stage's arena rebuild destroys the `WEIGHT` ranges the early stage admitted | **Changed (rulings §M12 C-1).** `compute_and_store_plan_for_inventory` computes the plan first and then ensures the zones from it, in both stages (master's order at `ggml-sycl.cpp:16213-16216` is the defect), so the early stage sizes the zones before it records and the late stage rebuilds nothing. A late zone demand above the admitted one is the named refusal "the late inventory changes the zones admitted at the early stage". An arena rebuild that meets any pending range refuses by name, and the `:16215` abort becomes that refusal, in words shared with 23mk (§Z8 I-3). §2.3.4's refusal list names `arena_destroy` and the rebuild. H7ap arms, one on GPT-OSS 120B's `moe_control` shape (`merge-gates/gptoss120b-b1.log:1544`). |
| I-1 | the retags run before `validate_end`, so a failed validation or a cleanup-required commit keeps B's model and device terms | **Changed (§M12 I-1).** The retags and the clear run only after `finalize_end` returns committed and not `cleanup_required`. The rollback clear lives in `ggml_sycl_abort_owner_effects_noexcept`, and every non-commit exit is enumerated (§2.4.2 (b) "Lifetime"); the validate-failed exit that skips the hook calls the same idempotent clear. H7ap failpoint arms at `validate_end` and `cleanup_required`. |
| I-2 | the `WEIGHT` ranges are sized in `ggml_nbytes`, and the draws are larger | **Changed (§M12 I-2).** One byte function, `ggml_sycl_weight_alloc_bytes`, shared by `get_alloc_size` and the stage; ranges add ggml-alloc's alignment, a bounded TLSF rounding per buffer, and the preload's materialized sizes. H7ap's `ne0` = 2880 arm, with the `nbytes` sizing as its positive control. |
| I-3 | a step-7 registry entry survives a failed first publish, and it is already accepting | **Changed (§M12 I-3).** Step 7 inserts a guard-owned, non-accepting entry; the CAS flips it to accepting; the guard retires it on step 8 (a), an exception and a lost CAS; the `!stable_mmid` conditions become "this call materialized". H7am arm. |
| I-4 | the carve's clear has no term-filtered form it can call inside the group mutex | **Changed (§M12 I-4).** `clear_pending_locked(tlsf, owner, term_filter)` joins the one primitive; the carve and the guard's first phase use it with `REGION`. H1 arm. |
| m-1 | the enum does not match 23mk's | **Relayed (§M12 m-1/m-12).** §2.3.3 is the one list (§M11b-amend: `REGION` covers both designs' context extents; `CONTEXT_TENANT` withdrawn), with the two rules; 23mk renames `CONTEXT_TXN` to its named A terms, drops `ALL_TERMS` under `{CONTEXT, id}`, records `DEVICE_TERM` (superseded in 7.12a: `DEVICE_TERM` is reserved with no producer, rulings §Z10.1, §M13), and moves its `SCRATCH`/`MODEL_TERM` records to the early stage. |
| m-2 | own ranges filtered by hand; the fit arm cannot be built from a scalar | **Fixed.** The filtered `pending_ranges(c, d, owner, term_filter)` (23mk's) is adopted and is the only source of `own_ranges`; H1's fit arm is built on geometry, with the hold ahead of the `REGION` range and the demand larger than the range. |
| m-3 | the H7ap I-A arm's C is a load, and its sizing is unstated | **Fixed.** C is a later load; the TLSF is sized so C's hold fits only once B's leftover is dropped. *7.12a:* the dropped leftover is B's `SCRATCH` hold, since B's `WEIGHT` room now passes to `{MODEL, id}` (§X9 I-D). |
| m-4 | a transaction's ring and MMID ranges must be `REGION` | **Fixed** (§2.3.3): every range a transaction records is `{CONTEXT, id}` / `REGION`, the ring's RUNTIME half and the MMID device pool included. |
| m-5 | the I-F RED depends on a 20 ms wait | **Fixed.** A `GGML_SYCL_PRIVATE_TESTING` handshake parks the unload in its lease wait with L0 held, with a probe `try_lock` as positive control (H9). |
| m-6 | the recover at context destruction waits under the registry mutex | **Fixed.** `wait = false` after the last-event drain; a not-ready bundle is `[CONTEXT-PLAN-BUG]`; `retire` skipping a quarantined entry is stated. |
| m-7 | the audit witness command cannot pass | **Fixed.** `scripts/audit-sycl-static-storage.py --check` (rc 0/1/2), the ctest's 77 read as a skip, and an explicit absence assertion; row 648's stale anchor and the failing `--check` on master belong to llama.cpp-ldvb (rulings §M13). |
| m-8 | the late comparison is undefined on the late inventory's subset | **Fixed.** Only tensors in the late inventory are compared; an absent tensor is consistent iff the admitted plan put it on the host. H7ap arm. |
| m-9 | the early recording is not idempotent across the per-device loop | **Fixed.** Once per `(txn, term, device)`, with replace semantics; H1 arm. |
| m-10 | two `arena_bytes` parsers; the load_end refusal has no channel | **Fixed.** One accessor; `ggml_sycl_model_loading_effects` returns `bool`, and `false` takes the not-committed exit before the preload. |
| m-11 | H4 (b) cannot tell GREEN from RED by "it aborted" | **Fixed.** Scored by message: the phase-gate regex versus the alloc-entry witness's message, with the probe (§Z9a). |
| m-12 | 23mk records its holds at the late stage | **Relayed** to 23mk; the H7ap interleave arm extends to them once they move. *7.12a:* 23mk rev 4.5 records them at the early stage, and the arm records them. |
| m-13 | the binding lookup reads a field copy, one fact with two sources | **Fixed.** The lookup reads `g_execution_backend_bindings` under its leaf mutex. H7am arm. |

| item | disposition |
|------|-------------|
| §M11b, §M11b-amend (one term list; `REGION` for both designs; two rules) | **Adopted** (§2.3.3 A1). `CONTEXT_TENANT` does not appear. |
| §Z9 I-3 (always-compiled witnesses) | **Adopted.** Every witness is `GGML_SYCL_WITNESS(cond, message)`, compiled in every build, active in a `GGML_SYCL_PRIVATE_TESTING` build unless `GGML_SYCL_WITNESS_CHECKS=0`, and in any build under `GGML_SYCL_WITNESS_CHECKS=1`; each has its own message (§2.4.2 "The token's kind"). The converted checks: the nesting check, the outermost-only check, `:33769`, the two shared-invariant checks, graph compute's check, and the ring bind's empty-vector check. |
| §Z9a (H4 (b)'s limits) | **Adopted.** `memory_update` dropped; a growing setter with GROWTH asserted; the six points and three controls; zhcn's hooks cited from its next head, with a named placeholder until it lands. |

**Noted for the lead (7.12).**
- H4 (b) cites zhcn's H4h hooks by a placeholder, "zhcn next head, H4h hooks"; the citation
  gets the head once zhcn publishes it. *7.12a:* done, zhcn `8a58ad4`.
- The witness switch is named `GGML_SYCL_WITNESS_CHECKS` here; zhcn should use the same name
  and macro, which I am relaying.
- The shared C-1 rebuild sentence is in §2.4.2 (b), in quotes, for 23mk to mirror word for
  word.
- Nothing was built for 7.12; it is a document change only.

**Revision 7.12a: the lead's §M13 and §M13a, before the r12 review.** One commit on `e64dac7db`;
the r12 review covers `c66848c26` to this head.

| item | disposition |
|------|-------------|
| 23mk rev 4.5: `ONEDNN_PP_A`, `SET_ROWS_STAGE` | **Adopted** (§2.3.3 A1): 23mk's two transient `{CONTEXT, id}` terms, which its slot guard and teardown clear with `ONEDNN_PP_A \| SET_ROWS_STAGE`, never `PENDING_TERM_ALL`. |
| 23mk rev 4.5: A3's term and status | **Adopted** (§2.3.3 A3): `replace_within(owner, term, mem_handle && old, offset, size) -> {status, new_block, remainder}`; `SHARED` returns `old` unconsumed, and the new range lands only in the owner's ranges of `term`. H1 arm. |
| 23mk rev 4.5: `record_pending(owner, term, offset, size)` | **Adopted** as the one write (§2.3.3); the early stage, step 5 and H1's idempotence arm use it. 7.11a's step 5 recorded \"as `pending_ranges(c, d)`\", naming the read query for the write. |
| 23mk rev 4.5: the CONTEXT bullet; early-stage records | **Fixed.** 23mk has no persistent `{CONTEXT, id}` hold (row 73 is `{MODEL, id}`, row 113 a carved slot); it records `SCRATCH` and `MODEL_TERM` at the early stage and only checks them at the late one. H7ap's interleave arm records them, which closes m-12's relay. |
| `DEVICE_TERM` (§Z10.1; §M13) | **Interim accepted.** Reserved with no producer; A4 step 2 has no current input and moves nothing; H7ap expects no `DEVICE_TERM` range; the m-1 row's \"records `DEVICE_TERM`\" is superseded. |
| 1oxa r8 I-D (§X9 I-D; §M13) | **Changed.** A4 step 3 retags `WEIGHT` to `{MODEL, id}` at the commit, so the undrawn room reaches the model and lazy materializations draw it; B's unload clears `{MODEL, id}` / `WEIGHT` in `ggml_sycl_teardown_owner_effects`. `ggml_sycl_load_pending_rollback_noexcept(txn)` runs before or with 1oxa's T13 chunk destroy, and unconditionally at the validate-failed exit. H1's retag arm and H7ap's I-A arm are rebuilt on B's `SCRATCH` hold, with 7.12's commit as a witness. |
| 23mk r7 I-A (§Z11; §M13) | **Added.** `pending_bytes_excluding(device, zone, except_owner, except_term)`, the one device-wide query; the RUNTIME ring's replan subtracts every other hold through it. H1 arm. |
| C-1 text (§M13a) | **Changed.** The sentence stays this design's (§2.4.2 (b)); the message is 23mk's, `[SYCL-PLAN] model load refused: ...`, replacing 7.12's `[VRAM-ARENA]` line, and `compute_and_store_plan_for_inventory` returns `bool` and stores no plan on the refusal. H7ap's rebuild arm expects that message. |
| H4h hooks (§Z9a; §M13) | **Filled** from zhcn `8a58ad4`: `ggml_sycl_test_host_fallback_scope` (thread-local RAII: the 1 MiB safe-max override and the claim bypass), `ggml_sycl_test_fill_host_zone(HOST_COMPUTE)`, the grow and fallback-entry counters (a zero voids), the coverage last-answer accessor (must read GROWTH) and the `[H4h] fallback held(TRANSACTION)=N` probe (that literal is superseded: since 7.14a H4 (b) quotes only zhcn `c75ce4d`'s `[H4h] held(TRANSACTION)=<0|1> phase=<name>`). |
| row 648 and `--check` on master (§M13) | **Cited** to llama.cpp-ldvb in the dead-code paragraph and the m-7 row. |

**Noted for the lead (7.12a).**
- §M13's message and the rulings file's §M13a disagree on the load-refused message. The message
  said this design's 7.12 text, message included, was canonical; §M13a, which supersedes that
  clause and which this document follows (the file wins), gives the message to 23mk. So 7.12a
  carries 23mk's `[SYCL-PLAN] model load refused` string and drops `[VRAM-ARENA]`. The sentence
  and the rollback function's name are this design's under both.
- §M13 places the ring replan at `unified-cache.cpp:17603` / `:17618`. At `3d9414c8c` those
  lines of `unified-cache.cpp` are the deferred oneDNN zone release; the ring's `capacity_bytes
  = zone_available(RUNTIME)` and `runtime_pending` reads are `ggml-sycl.cpp:17603` and `:17618`,
  which this design cites.
- A lazy materialization after the commit fills the owner `{MODEL, id}` from its model; 1oxa's
  design names the draw site, and this design states only the owner and the term.
- Nothing was built for 7.12a; it is a document change only.

### 6.15 Revisions 7.13 and 7.13a: design-moua-r12, rulings §M14 and §M15, the post-r12 queue

design-moua-r12 reviewed `c66848c26..7f4808729` (7.12 and 7.12a) and gave NOT READY: 1 Critical,
3 Important, 15 Minor. The lead ruled in §M14, queued §Z13.1 and zhcn 5.9's vehicle into this
round, and then asked for a term slot in the late refusal.

| id | finding | disposition |
|----|---------|-------------|
| C-1 | \"plan first, then ensure\" packs against the pre-ensure zones; GPT-OSS 120B and Qwen3.5-35B-A3B are refused at the early stage | **Changed (rulings §M14 C-1).** One order in `compute_and_store_plan_for_inventory`: compute every non-weight zone demand from the inventory and context shape, ensure the zones, pack once, and witness packed == ensured; no iterative re-plan. The late stage recomputes the demands and compares them per term under §Z13.1, never ensures, and re-packs only if the inventory changed. H7ap's arm uses both merge-gate shapes as positive cases, with pack-before-ensure as its RED. One function, shared with 23mk. |
| I-1 | the `WEIGHT` range omits the plan's same-device extra copies | **Changed (§M14 I-1).** The range adds each MoE PP alternate and dense WOQ extra layout from the plan's copy list at its staging `dst_size`; the draws are `allocate_within` draws; each `(tensor, layout)` is drawn once. H7ap WOQ/alternate arm with an interleaved transaction. |
| I-2 | m-8's absence rule refuses loads with user placement flags | **Changed (§M14 I-2).** The early inventory carries `forced_off_sycl` from `llama_model_loader::weight_forced_off_sycl`, the function `create_tensor` also calls; forced tensors are host-placed with no range. H7ap arm with `-ot`, `-ncmoe` and a partial `-ngl`. |
| I-3 | `pending_bytes_excluding` names a removed consumer; the MMID RUNTIME bytes are counted twice | **Changed (§M14 I-3).** The consumer is 23mk's A fit, excluding `({CONTEXT, id}, ONEDNN_PP_A)`; A's step sits between step 5's recording and the yield; on arena devices the MMID bytes have one source, and `runtime_pending` / `mmid_runtime_pending_bytes` are zero there. H1 exact-fit arm. |
| m-1 | the exit-effects refusal's wording could commit a failed load | **Fixed.** `finalize_end(ticket, false)`: `QUARANTINED`, `EFFECT_FAILED`; H7ap asserts not LIVE. |
| m-2 | where the retags sit, and `noexcept` | **Fixed.** The first statements of `if (result.committed)` (`:13219`), `noexcept`; H7ap throw arm. |
| m-3 | the rollback's signature | **Fixed (rulings §M15).** The rollback passes the `LoadTxnId` to `onednn_w_rollback_pending(txn)`, which erases W's `PENDING` entry by the txn recorded on W's contribution; no model argument and no registry lookup, since `txns_` may already be finalized at the validate-failed exit. 7.13 added `Registry::token_for_txn` for this; §M15 withdrew it, and 7.13a removes it. W's rollback has two idempotent call sites. |
| m-4 | per-iteration recording on single-device plans | **Fixed.** Recorded once, after the loop, from the admitted plan; a device absent from it records nothing. H7ap arm. |
| m-5 | one contiguous range per key | **Declared limitation**, refused by name (`[LOAD-PLAN] no contiguous WEIGHT extent`), with an H7ap arm; a range set is the follow-up. |
| m-6 | the order of the two late zone refusals | **Fixed.** The late stage compares and never ensures, so only `[LOAD-PLAN]` is reachable there. |
| m-7 | the C-1 message's arguments | **Fixed and relayed** to 23mk, whose string it is: each value and its cast are named from 23mk §6.8 (`unified-cache.cpp:4572-4588`); the scratch is a new byte sum beside the bool. |
| m-8 | two forms of the guard's phase-1 clear | **Fixed.** `clear_pending({CONTEXT, id}, REGION)` per recorded device; `clear_pending_locked` is only for a caller already inside a group-mutex section. |
| m-9 | H4 (b)'s positive control can be void (phase `UNKNOWN`) | **Fixed.** PP or TG is a precondition, forced by zhcn and probed; any other phase voids the run; `GGML_SYCL_WITNESS_CHECKS=1` is explicit; one zhcn head (`4f8ff34`). |
| m-10 | witness cost on the hot path | **Fixed.** The switch is a static read once; a disabled witness is one branch. |
| m-11 | re-materialization after an eviction | **Fixed.** The free re-records an evicted planned weight's block as `{MODEL, id}` / `WEIGHT`; relayed to 1oxa. |
| m-12 | the unload clear's order | **Fixed.** Teardown's first statement; idempotent across the reaper's retry. |
| m-13 | `compute_placement_plan_early`'s shape | **Fixed.** 23mk's form: the public `void` entry at `ggml-sycl.h:357` stays, and the body is a static `bool` impl. |
| m-14 | citations | **Fixed.** §M11a is cited as §Z8 I-2; `:12411`. |
| m-15 | the audit witness depends on ldvb | **Fixed.** The regeneration is armed only when `--check` passes on the base. |

| item | disposition |
|------|-------------|
| §Z13.1 (the late-stage check) | **Adopted** (§2.4.2 (b)): larger late is refused by name, smaller late is admitted with 23mk's WARN and `late_term_shrink_admitted`. H7ap arm. |
| the late refusal's term slot (lead, after §M14) | **Changed.** The string now names the term, its zone, the device and both sizes: `[LOAD-PLAN] the late inventory changes the zones admitted at the early stage: term %s in zone %s on device %d, early %zu B, late %zu B (refused)`. It is this design's string; 23mk accepted it and mirrors it byte for byte once 7.13 lands (rulings §M15). The WARN is 23mk's `cc0e6e1d8` text, mirrored byte for byte. |
| §M15 (W's rollback; the late string) | **Adopted.** W's rollback is 23mk's txn form (the m-3 row); the widened string stands as written. |
| §Z14.2 (zhcn r10 I-3 (b)): who runs H4 (b) | **Adopted** in H4 (b) (7.13a): ggml-sycl's devices are GPU-only (`ggml-sycl.cpp:25059-25072`), so a CPU-device run would exit 77 forever; lead-run on `level_zero:1`, label `cache\|mem-handle`, selector pinned by the registration's `ENVIRONMENT`; no CPU-device admission in ggml-sycl. |
| zhcn 5.9 (`4f8ff34`): the H4h vehicle | **Adopted** in H4 (b): `test-sycl-growth-fallback-vehicle`, one backend copy, exit 77 under `GGML_BACKEND_DL`, `GGML_SYCL_WITNESS_CHECKS=1` explicit. |
| r12 seams | **Relayed**: to 23mk (C-1's order, the refusal string, m-3, m-7, I-3); 1oxa (I-1, m-11) and zhcn (m-9) through the lead. |

**Noted for the lead (7.13).**
- §Z13.4 has 23mk's ring subtract `pending_bytes_excluding` with `({CONTEXT, id}, REGION)`, and
  §M14 I-3 names A's fit as the consumer. Both hold, in different windows: the ring's before
  moua L4 lands (after it the ring is a head slot, and its replan is gone on arena devices), and
  A's from L4 until beni makes A a head slot.
- "Where A sits" is answered as between step 5's recording and the yield, so A's capacity
  refusal stays pre-yield. 23mk has to place its fit there.
- m-5 is declared, not fixed: a range set per key needs ggml-alloc's buffer split, which the
  early stage bounds and does not replay.
- m-11's re-record on eviction is new mechanism in the free path; 1oxa owns the site.
- 23mk accepted the widened refusal string (rev 4.6c, `4ffb34aa3`) and adopted A's placement;
  §M15 settles both. 23mk mirrors the string once 7.13 lands.
- Nothing was built for 7.13 or 7.13a; both are document changes only.

### 6.16 Revision 7.14: design-moua-r13, rulings §M16 to §M18, the post-r13 queue

design-moua-r13 reviewed the whole document at `8e1c5c094` (range `c66848c26..8e1c5c094`) and
gave NOT READY: 1 Critical, 6 Important, 10 Minor. Its I-F is the placement-dependence flag the
lead had already ruled on as §M17. The lead closed r12's C-1 order, §M15 and r12's I-3, ruled on
the rest in §M18, and asked for §M16, §M16a, §M16b, §M17, §M17a and 23mk r8's two items in the
same round. Peer pins for this round: 23mk `c3942d236` (rev 4.6d; 4.7 in progress), zhcn
`c75ce4d` (rev 5.10), 1oxa `93fc6e6` (rev 9; rev 10 in progress).

| id | finding | disposition |
|----|---------|-------------|
| C-1 | tied-embedding models make the late inventory a superset of the early one; gemma4 on the B70 is 680 MB short and three merge gates fail to load | **Changed (rulings §M18.1).** The early inventory is the create set: a record pass of the create block itself (`create_weight_tensors`) through `resolve_create_site`, the function `create_tensor` calls, with `TENSOR_DUPLICATED` entries per site and a name charged once per device. No admit-by-name; a SYCL-backed tensor with no admitted entry is the named refusal `[LOAD-PLAN] tensor %s (site %s) exists in buffer type %s on device %d but has no admitted entry (refused)`, as a witness. H7ap tied-embedding arm; the RED's one outcome is that refusal; the three gemma4 gates are the lead's confirmation. |
| I-A | the packed == ensured witness cannot see an omitted term | **Changed.** The terms are one closed enum, `ggml_sycl_zone_term` (§2.4.5, proposed to impl-23mk). A second witness dry-runs `ensure_planned_arena_zones` per device over the published `planned_*` globals and must be a no-op; a source-contract gate maps every setter to an enum value. H7ap mutant drops `onednn_graph_scratch`. |
| I-B | forced placement reaches placement but not the demands, and the early call precedes the buffer-type lists | **Changed (rulings §M18.2).** The early call moves after `:2211` (a reorder, not a copied predicate); the site's `forced` field (four padding bytes of `ggml_sycl_tensor_info`, stride unchanged) is a fixed input: the pack places forced tensors first and never moves one, and a forced-off tensor charges 0 to every placement-dependent term. The late stage never packs. H7ap arm under `--no-host` and in default mode. |
| I-C | copy bytes have two sources; cross-device copies are unassigned | **Changed.** The adjusted layout and `dst_size` are computed at planning time and `configure_expert_preload` consumes them; collapsed copies are deleted from the plan; `dense_woq_staged_unplanned` becomes a refusal witness; a staging witness checks staged == admitted bytes; a cross-device copy is in its target device's range. The `:33586` cite is corrected to `:34905-34975`. H7ap arm. |
| I-D | I-1 and m-11 conflict with the OPTIONAL class and with P2 | **Changed (rulings §M18.3; superseded for OPTIONAL copies by §M18.3a in 7.14a, §6.17: planned copies can be OPTIONAL, and those yield to KV admission; rulings §V13 m-2).** Planned copies are `WEIGHT`, never eviction or yield candidates, freed only at unload; the preload drops `mark_optional_layout` for them. The re-record on free is deleted (m-11 and §V11.2's free-site re-record withdrawn); re-draws keep `replace_within`. §2.1's OPTIONAL row and step 5's picks say so. H7ap arm. |
| I-E | one extent per key is refused on the B70 and under 1oxa, alone | **Changed (rulings §M18.4).** Ranges are keyed `(txn, term, device, TLSF)`; m-5's limitation and refusal are withdrawn. The buffers are replayed through ggml-alloc's own split helper and placed first fit over the device's TLSFs, and the pack's fit test runs the same replay. H7ap arms on the B70's geometry and on 1oxa's 4 GiB chunks. |
| I-F | step 1 computes placement-dependent terms as if they were not | **Changed (rulings §M17, §M17a).** Five steps: the placement-independent terms, their ensure, one pack that charges the placement-dependent terms per placed weight, one grow-only ensure of the charged sum, and the witnesses. The shapes' device set comes from `dev_layer`. `moe_control` is a valid zero until 0ywi; the end states are re-derived (RUNTIME 1617.9 / 1672.0 MB). H7ap two-device and `moe_control` arms. |
| m-a | citations | **Fixed.** `:16336-16357` (single device), `:16219` (multi-device), `:34905-34975` (dense WOQ staging), and the early call at `:2106` (`:657` is the apply inside it). |
| m-b | the witness names the wrong byte field | **Fixed.** `plan.weight_vram_bytes`; a `vram_bytes` mutant is H7ap's false-fire case. |
| m-c | the C-1 RED scores two outcomes | **Fixed.** One outcome per `GGML_SYCL_WITNESS_CHECKS` value; master cited as a real RED for "no late rebuild"; the logs are evidence, not a control. |
| m-d | multi-device iteration 0 reads unensured zones | **Fixed.** Step 2 ensures every device in the attention set before the loop packs; the witnesses run per device over `plan.devices`. |
| m-e | late re-pack budget inputs | **Fixed.** The late stage never packs, and every budget it reads is the admitted one, never the live `ggml_sycl_device_budget_authority`. |
| m-f | H4 (b)'s probe, label, fixture and zhcn cite | **Fixed.** The probe is `[H4h] held(TRANSACTION)=<0|1> phase=<name>`; the labels are `cache;mem-handle`, with the regex named as the selection; `FIXTURES_REQUIRED test-download-model`; `ulimit -c 0` for the death arms; cited as zhcn 5.10 (`c75ce4d`). |
| m-g | a function-local static is not one branch | **Fixed.** A namespace-scope `const bool` initialized at library load; the cost stated as one load and one branch, and the static's guard cost named. |
| m-h | two term vocabularies | **Changed:** one name per term in the enum table (§2.4.5), taken from 23mk's table where it has the term, and proposed to impl-23mk directly; its confirmation is pending, with `pp_pipeline` and `moe_ptr_table` flagged. |
| m-i | seams with 23mk | **Relayed.** 23mk deletes "that iteration stays moua's"; the interim ring's `({CONTEXT, id}, REGION)` exclusion names nothing before L4, so it is dropped (§2.3.3). |
| m-j | 7fb6afc2a's header | No action; fixed in `8e1c5c094`. |

| item | disposition |
|------|-------------|
| §M16 items 1, 2, 4 | **Adopted** (§2.3.3 A3): the null empty remainder, remainder mode for `WEIGHT` re-draws, and the overlap precondition with its witness. H1 arms. |
| §M16a, §M16b | **Adopted** (§2.3.3 A3): device ranges only (`HOST_TIER`); E, G and other, with `BUSY` while a ticket publishes and `SHARED` only at `publishers == 0`; the superseded control released by the destructor, with witness counters; the lock order and its ten-site census; the `in_hand` and `releasing` snapshots; the registered sinks; the clamp stated as a limitation. H1 arms; the census and cross-queue arms are lead-run. |
| §M17, §M17a | **Adopted** (§2.4.2 (b)): the five steps, the census of terms, and `moe_control` gated to a valid zero, zeroing only the control pool; the host-sum defect is llama.cpp-s4ip's and is only cited. |
| 23mk r8's items | **Adopted.** The SCRATCH bullet carries the one late string; step 5 lists both L1-released windows, A's reap window first (23mk `c3942d236` L1351-1353). |
| r13 seams | **Relayed**: to 23mk directly (C-1, I-A, m-h, m-i, the late stage that never packs); to 1oxa through the lead (C-1, I-D, I-E, A3's host scope). |

**Noted for the lead (7.14).**
- **§M18.3 changes what KV can take** (answered by the lead's §M18.3a, folded in 7.14a,
  §6.17). The a1_long gate (B50, PCT 60, `-c 2048`) logs "KV
  admission released 6 optional oneDNN WOQ layout copies (220.5 MB) ... for n_ctx=2048's KV".
  Those copies are planned, so under §M18.3 (since amended by §M18.3a: they are planned
  OPTIONAL copies and do yield) they are never yielded, and that KV demand demotes
  layers to the host tier instead. That is placement, not a shrink, and it is what the ruling
  says; but a gate that passed by yielding will now run slower, and its expected log line
  changes. It also reverses, for planned copies, the owner's rule that §2.9 cites from audit I5,
  that KV wins over optional layouts: a planned copy now holds its VRAM while KV demotes. §2.9's
  ladder keeps only unplanned copies, which an arena device no longer stages, so on arena
  devices the ladder and the step-5 yield have no members left.
- **§M17's "re-packs only if the inventory changed" has no trigger left.** Under §M18.1 every
  late change is a refusal, so the late stage never packs. 23mk mirrors 7.13's re-pack sentence;
  it is told.
- **The record pass runs the create block twice.** The probe copies `hparams` and the lists and
  is destroyed after the pass. A create block that built ggml operations on a returned tensor
  would do so in the loader's meta context; none does today (a census of every
  `load_arch_tensors` body in `src/models/*.cpp`, 2840 `create_tensor` lines, and of
  `:2229-2385` finds no ggml operation), and the §M18.1 refusal catches any divergence between
  the two walks by name.
- **The buffer replay moves a loop out of `ggml-alloc.c`,** an upstream file, into a helper both
  callers use. A rebase that reverts it leaves the stage with no split to replay, which fails to
  build rather than silently diverging.
- Nothing was built for 7.14; it is a document change only.

### 6.17 Revision 7.14a: §M18.3a, the relay that crossed 7.14, 23mk's table requests, §M19

The lead received 7.14 at `4dfa587e7` after its last relay had crossed it, and asked for the
relay (items 1-5), the amendment §M18.3a, 23mk's table-only requests (23mk 4.7b `e81dc2327`,
4.7c `175dcd51b`) and §M19 as 7.14a, before a fresh review. Peer pins: 23mk `175dcd51b` (rev
4.7c), zhcn `c75ce4d` (rev 5.10), 1oxa `93fc6e6` (rev 9).

| item | disposition |
|------|-------------|
| §M18.3a (amends §M18.3): planned copies are PRIMARY or OPTIONAL | **Adopted.** The plan classes each copy at planning time (§2.4.2 (b), "The range bytes"): OPTIONAL is a duplicate layout whose primary is resident on the same device, and every other copy, a cross-device alternate among them, is PRIMARY. PRIMARY copies are never yielded or evicted and are freed only at unload. OPTIONAL copies are marked optional, never evicted, and yieldable only to a context's KV admission, through the yield path (§2.4.2 step 5): one group-mutex section per pick retags the block's extent to `{CONTEXT, id}` / `REGION`, marks the key yielded in the plan, and moves the handle to the drop list. The destructor stays reason-free; unload does not re-record. §2.1's OPTIONAL row, A4's paragraph, step 5's picks and §2.9's ladder say so, and §2.9's ladder is restored for OPTIONAL copies. H7ap arm (a yielded copy's extent drawn by that context's KV) with the skipped-retag RED; C8 pre-registers a1_long's "released 6 optional oneDNN WOQ layout copies (220.5 MB)" unchanged. |
| §M18.3a: the late stage never packs | Already so in 7.14 (§2.4.2 (b)); no change. |
| relay 1, r13 I-F (5): the two shapes are published for `plan.device_id` only | **Changed (rulings §Z15, §M19).** Step 1 keeps only class P terms (`compute_arena`, `ring`, `mmq_work_counter`). `nonfa_shape` moves to step 3, charged per device at the first attention layer the pack places there. `onednn_graph_scratch` is 23mk's class C term, charged at the context transaction with 1oxa's §V11.3 D512 term; the load stage evaluates it as 0, and the load-stage getter stops adding its floor. H7ap's two-device arm now expects the pack's `nonfa_shape` charge on device 1, and a new `--no-host` host-experts arm charges `moe_onednn` 0 and still charges `nonfa_shape`. The drop-a-term mutant moves to `onednn_scratchpad`, and a C-rule mutant keeps the floor at the load stage. |
| relay 2, m-f residual: "like zhcn's H4h" at `4f8ff34` | **Fixed.** H4 (b) cites zhcn 5.10 `c75ce4d`; §4's row and the §6.14 probe literal are corrected. |
| relay 3, zhcn r11 H4 (b): the literal and the route | **Fixed.** Item 5 quotes only zhcn's `[H4h] held(TRANSACTION)=<0|1> phase=<name>` (zhcn r11 m-10). The route is zhcn's alone: two 1-token decodes, the phase forced to TG immediately before the growing setter, and the setter's next decode (zhcn r11 m-9 (i)); the ladder rung is dropped, since its phase is `UNKNOWN` inside the constructor. The forcing precedes every arm's ALLOC. |
| relay 4, §Z15: A's step on the tenant-only path | **Adopted** (§2.4.2, the tenant-only path): move-out and reap after (s), fit-and-hold inside (0) before any release, the carve after the publish. |
| relay 4, §Z15: the pre-L4 ring exclusion | **Adopted** (§2.3.3 A1): `except_owner = pending_owner{}`, which matches nothing, so every range counts as taken. |
| relay 4, §Z15: one term for the pointer tables | **Adopted** in the enum (§2.4.5): `moe_ptr_table` (23mk) is the only term, (k + 1) tables, k once row 73's fallback is unreachable (§M20.3; since §M23 (2), k from the term's first commit, §6.19); `moe_control` carries only its non-table parts; no `expert_ptrs` term. |
| relay 4, §Z15 and §M19: the two oneDNN scratches | **Adopted.** `onednn_scratchpad` is moua's, a D term charged in step 3 only, with moua's value function. `onednn_graph_scratch` is 23mk's, class C; moua lists it in the enum only. |
| relay 5 (a) and (b): the MOE-PREALLOC block and 1617.9 vs 1618.0 MB | **Ruled (§M20), second commit.** Row 134 is live and read by row 73, raw device allocations outside the arena and never freed at master (`unified-cache.cpp:22176`, `:22203`); 23mk converts it, and this design cites it. §M17a's "no reader" premise and its `moe_transient_ptr_table` cite are corrected. `moe_control` is a step-3 charge, 0 until the commit that first routes row 134's ids, compact and missing into RUNTIME, then 24832 B / 49408 B. `expert_ptrs` is in no RUNTIME term. 1617.9 / 12232.1 and 1672.0 / 12178.0 stand, and 1618.0 / 1672.3 are rejected. H7ap is pre-registered in exact bytes (§2.4.2 (b), "The end states"). |
| found while folding §M20: the ONEDNN mutants could not fire | **Fixed.** The ensure takes `max(floor, demand + charged)`, and on GPT-OSS's shape the ONEDNN demand (34.5 MB scratchpad + 96 MB floor) sits under the 256 MiB minimum, so 7.14's graph-scratch mutant and 7.14a's first `onednn_scratchpad` mutant would have passed silently there. Each dry-run RED now asserts that its term lifts its zone above the minimum on its fixture, or it is void. |
| §V12: the VM yield shape | **Cited** in step 5's yield path: one L1 begin that retags the group's pages and marks the keys yielded, the release after the reader barrier. |
| 1oxa rev 10a (`fc1d356`): VM bins and OPTIONAL exclusion | **Cited** in (b)'s replay text: on a VM device the replay's first fit walks 1oxa's 4 GiB bins (each bin a `(txn, term, device, TLSF)` key, its sum the chunk's size), and OPTIONAL copies are left out of the replay, their group pages admitted in the early stage's L1 section. |
| 23mk: add `onednn_pp_a` as a context row | **Adopted** (§2.4.5), with the C rule stated for every context row. |
| 23mk: `pp_pipeline` is one term with 23mk's value function | **Adopted** (§2.4.2 (b) step 3, §2.4.5). |
| 23mk: `moe_ptr_table` is the only pointer-table term | **Adopted** (§2.4.5). |
| 23mk: "this design adds none" in the shared A3 text | **Reworded:** "... and 23mk's A overlap form (term `ONEDNN_PP_A`, only at count == 1); moua adds none." 23mk mirrors it. |

**Noted for the lead (7.14a).**
- **The retag is a `record_pending`, not A4's `retag_pending`.** §M18.3a names `retag_pending`.
  A4 moves every range of an owner by term, but a drawn copy was consumed from `{MODEL, m}`'s
  range at its draw (`consume = true`), so no `{MODEL, m}` range record covers the block. The
  yield path therefore records the block's own extent as `{CONTEXT, id}` / `REGION` under the
  group mutex it already holds, and step 5's recording leaves a planned-copy pick's extent to
  that retag. The effect is the one ruled: the extent passes from the model to the context in
  the same section as the release, and the RED (the retag skipped) sends the freed bytes to the
  general TLSF. The alternative, drawing OPTIONAL copies with `consume = false` so a model range
  covers them, would make the skipped-retag RED return the bytes to the model's range rather
  than to the TLSF, and was not taken.
- **"Releases the handle" moves it to the drop list.** Destroying the handle inside the
  group-mutex section would self-deadlock when it is the last reference, since the free takes
  the same mutex. The handle leaves the cache entry in the section and is destroyed in the
  yield's finish with no lock held, like every retired pick; the retag has already made the
  extent the context's, so the bytes land in the context's range wherever the last reference
  drops.
- **Moving the graph-scratch floor out of the load stage does not move these figures.** The
  ONEDNN zone's demand on GPT-OSS's shape is the 34.5 MB scratchpad plus the 96 MB floor, both
  below the zone's 256 MiB minimum (`gptoss120b-b1.log:184`, "oneDNN 256.0->256.0"), so the zone
  is 268435456 B with or without the floor, and the weight zone is unchanged. On a shape whose
  ONEDNN demand exceeds the minimum, the load-stage zone shrinks by the floor, and 23mk's
  context transaction places it; that placement is 23mk's.
- **The 120B weight slot in §M20 (564019200 B) is the tensor's size, not the slot the code
  computes.** 564019200 = 2880 · 2880 · 128 · 17 / 32, the MXFP4 tensor. The sizing code
  (`src/llama-model.cpp:445-446` at `3d9414c8c`) rounds each expert's e8m0 scale block to 256 B,
  259200 → 259328, so the weight slot is 128 × 4406528 = 564035584 B, 16384 B more. Both print
  537.9 MB, which is why the logs cannot separate them. 7.14a pre-registers the stepped value
  (RUNTIME 1696497664 B, weight zone 12826279936 B before `moe_ptr_table`); §M22.1 ruled for
  it. Qwen's slot (142606336 B) has no rounding,
  so its figures are exact either way.
- **The arena's own bytes are assumed exact at 14618 MiB.** No log prints them in bytes; the
  pre-plan split (13338.0 = 14618 − 512 − 512 − 256) agrees to display precision. H7ap asserts
  the weight zone relative to the arena it reads (arena − SCRATCH − ONEDNN − RUNTIME), so an
  arena that is not a whole MiB moves only the absolute figure.
- Nothing was built for 7.14a; it is a document change only.

### 6.18 Revision 7.14b: rulings §M21 and §M22, and the §M20 message

The lead's §M20 message crossed 7.14a's first commit; 7.14a's second commit (`9f68c61bd`)
had already folded it, and 7.14b checks each item against it. §M21 and §M22 rule on the points
7.14a raised.

| item | disposition |
|------|-------------|
| §M21.1: the OPTIONAL yield records the block's own extent | **Already so** in 7.14a (§2.4.2 step 5); step 5's item 1 now cites the ruling. |
| §M21.2: the handle moves to the drop list, destroyed in the finish | **Already so** in 7.14a (§2.4.2 step 5), the same shape as §V12; item 3 now cites the ruling. |
| §M21.3: C terms leave the load stage entirely | **Changed.** Step 1, the dry-run witness text and the enum row say so: `onednn_graph_scratch` is charged at the context transaction from that context's `REGION` headroom, like KV; the load-stage ONEDNN sizing (`unified-cache.cpp:4464`, and the plan's twin at `:27658`) reads the stored, no-floor getter (`:2141`). The load-stage ONEDNN and weight-zone figures are re-derived in bytes from `ensure_planned_arena_zones` (§2.4.2 (b), "The end states"): the zone is `max(268435456, stored)` = 268435456 B on both merge-gate shapes, with and without the floor, so the weight zone absorbs 0 B there. §M21.3's parenthetical "−96 MB on GPT-OSS" does not survive the code path, since 34.5 + 96.0 MB is under the 256 MiB minimum; §M22.2 already accepts that. H7ap's C-rule arm scores the absorbed bytes on a fixture above the minimum. |
| §M22.1: the 120B weight slot is 564035584 B | **Adopted**; 7.14a's flag becomes the ruling's citation. |
| §M22.2: every dry-run RED asserts that its term lifts its zone above the minimum | **Already so** in 7.14a's second commit (H7ap); 23mk's and 1oxa's REDs are theirs. |
| §M20 figures: 1617.9 / 1672.0 stand, 1618.0 / 1672.3 rejected, exact bytes | **Already folded** (`9f68c61bd`), with the 120B slot as ruled in §M22.1. |
| §M20: `expert_ptrs` is not a RUNTIME term | **Already folded** (§2.4.2 (b), §2.4.5). |
| §M20: `moe_control` a step-3 charge, re-anchored to the row-134 routing commit (24832 B, 49408 B) | **Already folded.** |
| §M20: `moe_ptr_table` (k + 1) × 1024 B / × 2048 B, then k | **Already folded** (§2.4.5, H7ap); superseded by §M23 (2), k from the first commit (§6.19). |
| §M20: row 134 is live, outside the arena, never freed; 23mk converts it | **Already folded**, cited at `unified-cache.cpp:22176`, `:22203`. |
| §M20: remove the "held" note | **Already removed** in `9f68c61bd`. |
| the three mirror items for 23mk 4.7d | **Sent** to impl-23mk-2 directly: the tenant-only A bullet, the A3 rewording, the §M18.3a A4 paragraph. |

Nothing was built for 7.14b; it is a document change only.

### 6.19 Revision 7.14c: design-moua-r14, rulings §M25, §M23, §M24 and §V13 m-2

design-moua-r14 judged 7.14a (`9f68c61bd`) and failed it: 0 Critical, 6 Important, 11 Minor,
all in the zone-term layer that 7.14 and 7.14a introduced; it passed every exact-byte figure,
the §M18.3a yield section, the C-1 order, §M17a's re-anchoring and the peer citations. The
lead's rulings on it are §M25. §M23 (23mk 4.7d's byte mirror) and §M24 (23mk 4.7d's open items
and zhcn 5.11's relay) were queued behind the review, and §V13 m-2 comes from 1oxa's r10. Peer
pins: 23mk `026b69b86` (rev 4.7d, second commit), zhcn `2c511f6` (rev 5.11), 1oxa `fc1d356`
(rev 10a).

| id | finding | disposition |
|----|---------|-------------|
| I-1 | the composition rule is stated two ways; read additively, the pinned SCRATCH and weight-zone bytes are wrong and witness 1 false-fires | **Changed (rulings §M25 I-1).** One rule, 23mk's, in step 4, witness 1 and §2.4.5: `ensured_Z = max(floor_Z, demand_Z + charged_Z)`; a floor is capacity, not a term. `compute_arena` is withdrawn as a SCRATCH P term: the compute arena spans the zone (`unified-cache.cpp:20722-20728`), so `ggml_sycl_compute_arena_bytes()` is `floor_SCRATCH`. The floors (SCRATCH 512 MiB, ONEDNN 256 MiB, RUNTIME 512 MiB) are named with master's lines. Witness 1 is `ensured == max(floor, demand + charged)` with packed + charged ≤ ensured. The end states gain SCRATCH and ONEDNN at their floors, stepped. H7ap: the additive mutant, caught by witness 1 and the exact SCRATCH bytes. |
| I-2 | step 4's input is unstated, so the dry run can be tautological | **Changed (rulings §M25 I-2).** Step 4 is a sized ensure, `ensure_arena_zones_sized(dev, sizes)` (shared with 23mk), fed from the pack's charge ledger, and never reads a `planned_*` getter; the dry run is the only load-stage reader of the getters and compares them with the ledger. H7ap names, per mutant, the check that catches it: witness 1 (the additive rule, pack before ensure), the dry run (a forgotten term, the C-rule mutant, the two-device RED, `moe_control`'s charge left 0), and the exact bytes where a grow-only dry run cannot see an over-charge (`mmid_workspace` as D, the full control pool, a double-charged compact list, `moe_onednn`'s inventory-wide value). |
| I-3 | the two-device RED cannot fire | **Changed (rulings §M25 I-3).** The RED mutates the charge: the pack charges `nonfa_shape` for `plan.device_id` only while the plan publishes it per device, and the dry run fires on device 1 naming SCRATCH, on a fixture whose device-1 demand exceeds the SCRATCH floor. The publication RED is withdrawn. |
| I-4 | `mmid_workspace` is classed D in RUNTIME, against its context scope and the end states; wrong cite | **Changed (rulings §M25 I-4).** Class C: placed once, at the context transaction, as that context's `REGION` head slot (step 5 already places it there), and charged nothing at the load stage. Cited at `plan_moe_mmid_workspaces` (`unified-cache.cpp:26806`) and `account_moe_mmid_workspaces` (`:26910`); `:15865` was the `allocation_owner_test_*` hooks. §M20.1's end states re-derived without the MMID bytes: they never carried them, so every byte stands, with k for k + 1. H7ap arm, RED caught by the exact RUNTIME bytes. |
| I-5 | the compact pointer list and missing flag have two terms | **Changed (rulings §M25 I-5).** One term, `moe_control`. `mmvq.cpp:16614` and `:16646` are the fallback for the same objects; the row-134 conversion deletes them with row 73's fallback, a miss is the named `[MODEL-PLAN-BUG]` refusal, and 23mk's `moe_compact_storage` row goes in the same commit. The enum drops the row and says why. H7ap RED: both terms non-zero after the modelled routing commit, caught by the exact bytes. Relayed to 23mk by the lead. |
| I-6 | `onednn_graph_scratch` has no room after the load | **Changed (rulings §M21.3, §M25 I-6).** The C rule now names the room: every C term is placed at the context transaction in that context's `REGION` headroom, as a head slot of its fit, like KV, and no C term waits for a zone to grow. Every C row's zone column says so (`onednn_graph_scratch`, `set_rows_stage`, `onednn_pp_a`, `mmid_workspace`). H7ap's C-rule arm adds the context's placement of the graph scratch in its `REGION` ranges. |
| m-1 | "12232.094 MiB at k + 1 = 0" | **Fixed:** "before `moe_ptr_table`, the k-independent part". |
| m-2 | §M22.1 is ruled; header list; cite §M21.1/.2 at the yield path | **Fixed.** The §6.17 note's conditional sentence now reads "§M22.1 ruled for it"; §M21 to §M25 and §V13 are in the header's list; the yield items cite §M21.1 and §M21.2 (7.14b). The two mentions of 564019200 that explain its rejection stay, as the lead directs (§M24 (1)). |
| m-3 | `record_pending`'s replace-vs-append rule is implicit | **Fixed (rulings §M25).** The primitive states it by owner kind: a `{LOAD, txn}` record replaces per `(txn, term, device, TLSF)`; every other owner's record appends, so the yield's retag lands beside the step-5 `REGION` ranges. H1 arm: two `{CONTEXT, c}` / `REGION` ranges on one TLSF both survive; RED, key-replace for context owners. |
| m-4 | "never the general TLSF" holds only if the last drop precedes step 6's clear | **Fixed.** The yield path and A4 say: a last reference that outlives step 6's carve returns its bytes to the general TLSF; the KV demoted at step 6's re-fit, and the key stays yielded. The A4 sentence changes, so it goes to 23mk as a mirror item. |
| m-5 | the yielded flag's home, the section's lock sequence, "no lock held" | **Fixed (rulings §M25).** (a) The flag is an atomic on the copy's cache entry, written only inside the section under the group mutex; the admitted plan is not written; dispatch keys off the retire and the epoch bump, and the staging witnesses and the lazy path read the same flag. (b) The lock order, stated once: L1 → the cache lock that guards the entry → the TLSF's group mutex, then the TLSF operations; the store mutex is never taken inside. (c) The handle is destroyed with neither L1 nor any group mutex held; L0 stays held. |
| m-6 | `moe_control`'s value function after routing; its `n_ubatch` | **Fixed.** The routing commit changes `moe_control_requirement_from_layout` to the non-table part, `total_bytes − ids_offset` (`moe-control-plan.cpp:503-518`, `:388`, `:488-497`); the term and function are this design's, and the routing commit (23mk's conversion, or 0ywi's) lands them. The charge uses the plan's `n_ubatch`, the load inventory's 512 (`src/llama-model.cpp:407`), which also sizes `moe_onednn` and is row 134's `max_batch`. A larger context `n_ubatch` is put to the lead (below). |
| m-7 | `moe_onednn`'s value set | **Fixed.** Each slot is the device's own running maximum over the expert tensors it places and executes through oneDNN PP, as for `onednn_scratchpad`; equal to the inventory-wide field on both merge-gate shapes, which the five-step arm asserts; H7ap arm with a non-uniform two-device fixture, RED the inventory-wide field. |
| m-8 | C8 rests on two unverified premises | **Fixed.** (i) is checked against the lead's a1_long run on uwlx `a4da787ae`: `added=182` and no `[S1-PRELOAD] ... disagree with the plan` WARN, which prints whenever a copy is staged but not planned, so every copy there was plan-listed. (ii) cannot be checked before L4, so a host-only dry run of the fit over the a1_long fixture's admitted plan pre-registers the count and MB, and "6 ... (220.5 MB)" stands only if it picks jehw's set. |
| m-9 | `fp16_slab`'s zone "WEIGHT, OPTIONAL" | **Fixed:** WEIGHT, with each copy's class from the plan. 23mk's table carries the same; relayed. |
| m-10 | `moe_control` RED 2 may not fire | **Fixed.** The mutation is in step 3's charge alone while the publish at `unified-cache.cpp:27428` runs with `GGML_SYCL_MOE_CONTROL_CONSUMER` = 1; the arm asserts the published requirement non-zero before scoring, or it is void. |
| m-11 | seams | **Relayed** to 23mk directly: its §6.8 members list (L3127 at `026b69b86`) still names `compute_arena` and `mmid_workspace` as moua's terms of their old classes, and `weight_forced_off_sycl` (L851, L3072) is still cited where 7.14 named `resolve_create_site` (§M18.2). zhcn's superseded probe literal is gone from its H4h row at 5.11 (`2c511f6`). |
| §M23 (1), §M24 (2) | `moe_control` and row 134's block per (model, device) | **Adopted**, mirroring 23mk 4.7d's words: "charged once per (model, device) at the first GPU-executed expert placed there (rulings §M23 (1))"; row 134's block is one per (model, device), MODEL lifetime. H7ap cites 23mk's H3 two-model arm as the RED. |
| §M23 (2) | `moe_ptr_table` is k from its first commit | **Adopted** in the end states, §2.4.5 and H7ap; the (k + 1) rows of §6.17 and §6.18 are marked superseded. |
| §M24 (1) | 564035584 B; 564019200 withdrawn as a value | **Already so**; the two rejection notes stay. |
| §M24 (3) | the §6.11 streaming gate keys on `zone_backed()` | 23mk's §6.11 gate on 1oxa rev 10's predicate; this design has no streaming gate of its own to re-key, so no change here. |
| §M24 (4) | zhcn 5.11: H4 (b) names only the growing setter; one probe literal | **Already so** since 7.14a (§6.17 relay 3); H4 (b) now cites zhcn 5.11 (`2c511f6`) and inherits its VOID positive control (zhcn r11 m-13). |
| §V13 m-2 | planned copies can be OPTIONAL | **Already so** since 7.14a; §6.16's I-D row and its 7.14 note are marked superseded by §M18.3a. |

**r14's addendum on 7.14b (second commit).** Its two checks hold: the load-stage ONEDNN zone is
268435456 B on both merge-gate shapes, and the C-rule arm's fixture lifts ONEDNN above the
minimum. Its I-6 residual (the other C rows name no context-time room) is closed by the first
commit: every C row, `mmid_workspace` included, names the context's `REGION` headroom.

| id | finding | disposition |
|----|---------|-------------|
| m-12 | the ONEDNN formula leaves out `onednn_pp_w` | **Fixed.** The load-stage zone is `max(268435456, onednn_scratchpad + onednn_pp_w)`, over every ONEDNN term; 268435456 B on both shapes, with 23mk's H3 value for W. |
| m-13 | the 0oxf clamp can no longer bind at the load stage | **Deleted** with the floor, in `ensure_planned_arena_zones` too: its comment says it bounds the Graph-scratch floor, `max(256 MiB, stored)` stays under its cap whenever `available / 4 ≥ 256 MiB`, and a clamp below the ledger's charged bytes would break P4. |
| m-14 | the with-floor getter's gates and comments are not named | **Named for the same commit:** the allocator source gate's `WITH_FLOOR_GETTER_BODY_CODE` extraction (`:475-481`) and additive-floor check (`:840-846`) re-anchor on 23mk's context-time sizing, with a new check that the two load-stage reads use the stored getter; the comments at `unified-cache.cpp:4461-4463`, `:27654-27656` and `unified-cache.hpp:1644-1650` are rewritten; `test-unified-runtime-alloc.cpp:996` reads the stored getter. |
| m-15 | the C-rule mutant mutates code the load path no longer runs; the baseline ignores the clamp | **Fixed.** (a) The mutant reverts `unified-cache.cpp:4464` and `:27658` to the with-floor getter; with step 4 sized from the ledger (§M25 I-2), only the dry run reads it, and it fires. (b) The baseline is master's own ensure output on the fixture, clamp included, and the arm asserts master's clamp WARN absent or uses the clamped figure. |
| m-16 | 34.5 / 49.0 MB are master's inventory-wide values | **Fixed:** "at most", named as upper bounds of moua's per-device maximum. |

**Noted for the lead (7.14c).**
- **`onednn_pp_a`'s room.** §M25 I-6 names `REGION` headroom for every C row, `onednn_pp_a`
  included, but 23mk's A step still fits in RUNTIME's free bytes until beni makes A a head slot
  (§2.4.2 step 5, §M14 I-3). Where RUNTIME's demand exceeds its floor, as on both MoE
  merge-gate shapes, step 4 leaves RUNTIME no free room, so the interim fit has nothing to fit
  into. This design states the room as ruled and asks when 23mk's A fit becomes the head slot:
  in the same commit as §M25, or with beni.
- **A context whose `n_ubatch` exceeds the plan's 512.** Row 134's non-table parts scale with
  `n_ubatch`, the plan sizes them at the load inventory's 512, and once §M25 I-5 deletes the
  fallbacks, a context at `-ub 2048` meets the miss refusal mid-inference. The same bound sizes
  the `moe_onednn` slots. Recommendation: the context transaction refuses by name, before any
  decode, a context whose `n_ubatch` exceeds the admitted plan's MoE bound, so the miss never
  happens at runtime; the alternative, sizing the load for the largest `n_ubatch` any context
  may
  admit, charges every load for a shape few contexts use.
- **The load-stage MMID accounting moves.** Under §M25 I-4 the load stage charges no MMID bytes,
  so master's `account_moe_mmid_workspaces` at the load (`unified-cache.cpp:28371-28376`,
  `:30394-30397`) no longer adds them to `plan.vram_bytes`. A load that fits its weights but not
  its MMID pool now loads, and its first context places the pool in `REGION` headroom or
  demotes KV layers to make room, as KV does.
- **23mk's step 4 and dry-run text** still read "one grow-only ensure of exactly the charged
  sum"
  and "each recomputed over the packed placement" (4.7d §6.8). Both designs now say the sized
  ensure from the ledger and the dry run over the getters (§M25 I-2); the two mirror sentences
  go
  to 23mk with the A4 change.
- Nothing was built for 7.14c; it is a document change only.

### 6.20 Revision 7.14d: rulings §M27, §M26 and §M26a

The lead ruled 7.14c's three questions (§6.19, "Noted for the lead (7.14c)") in §M27, with a
same-day addendum, (2a), that makes `moe_control` class C. §M26 (23mk r9) and §M26a (23mk r9's
re-verdict) bind this design where they name its terms or its C-1 commit. One commit on top of
7.14c (`7b2cb5898`); design-moua-r15 starts on it. Peer pins: 23mk `026b69b86` (rev 4.7d; 4.7e
was in progress, with the Graph-scratch commit named and `moe_control`'s C wording not yet
written), zhcn `2c511f6` (rev 5.11), 1oxa `fc1d356` (rev 10a).

| id | ruling | disposition |
|----|--------|-------------|
| §M27 (1) | `onednn_pp_a` is the `REGION` head slot with §M25, not with beni; A's fit reads `REGION` headroom and pending, never free RUNTIME | **Changed.** Step 5: A is a head slot of step 2's fit, its room `REGION` headroom less every other owner's pending ranges, mandatory, and a pre-yield refusal naming A; 7.14c's RUNTIME fit is withdrawn. The reap window's recompute is step 6's re-fit. The tenant-only path's A bullet says (0)'s fit places A as a head slot and records it as a `REGION` range (mirror text sent to 23mk). The primitive's `pending_bytes_excluding` keeps one consumer, 23mk's pre-L4 interim ring. H1's A arm now carries 7.14c's RUNTIME fit as a RED. §2.4.1's head-slot list and the enum row name every C term. |
| §M27 (2) | every `n_ubatch`-scaled byte is class C; row 134's MODEL block holds only `n_ubatch`-independent bytes; the slice is per context, at its own `n_ubatch`, refused by name at its transaction; m-6's "plan's `n_ubatch`, 512" withdrawn | **Changed** (§2.4.2 (b), the `moe_control` section). m-6's value-function change and its 512 are withdrawn; the only figures that used them were 24832 B and 49408 B, withdrawn below. A context that cannot place its slot is refused by name before any decode. `moe_onednn`'s activation and output slots are also sized at 512 and are put to the lead (notes). |
| §M27 (2a) | option B: `moe_control` is one per-context, per-device `REGION` slot (compact list and missing flag) beside A and `set_rows_stage`; the handle moves to the backend context; row 134's block is k × stride; nothing at load | **Changed.** The enum row is class C. Step 3 lists "not `moe_control`". The value function reads the ungated layout's compact and missing parts, `total_bytes − compact_offset`: 16640 B / 33024 B at `-ub 512`, 65792 B / 131328 B at `-ub 2048` (GPT-OSS 120B / Qwen3.5-35B-A3B). The handles move from the weight extra to the backend context; the ids go to row 75; the slot's commit deletes the fallbacks and shrinks row 134's block. The end states keep their bytes, re-derived without 24832 B / 49408 B, which they never carried. H7ap's arm is rewritten with five REDs. 23mk had the same ruling and had not written it, so this text is proposed to 23mk as the shared wording, to be mirrored byte for byte. |
| §M27 (3) | MMID leaves the load-time `plan.vram_bytes`; the first context places its slice from `REGION`, demoting KV, or is refused by name | **Stated** in the end states: `account_moe_mmid_workspaces` leaves the load at `unified-cache.cpp:28371-28375` and `:30394-30397`, `plan_moe_mmid_workspaces` still sizes, a load that fits its weights loads, and the first context places the pool, demoting KV to host tiers, or is refused by name at its transaction. §4 L6 row. |
| §M26 I-3 | one owner per byte: ids staging row 75 (beni), list and flag `moe_control`, `n_ubatch`-scaled bytes class C | **Adopted** in the `moe_control` section, with an H7ap RED for the ids charged in the slot. |
| §M26 I-4 | `onednn_graph_scratch`'s value function is 23mk's, calling 1oxa's D512 tile screen as a helper; 1oxa charges nothing | **Fixed:** the stale "with 1oxa's §V11.3 D512 term" in step 1 and in the enum row. |
| §M26a I-1 | the "charge on at 0" RED restated for a C slot, firing on the mutant and not on the correct tree; sizes from the ungated layout | **Changed** (H7ap): the mutant places no slot and meets the named miss at its first MoE op; the arm first asserts that the correct tree's first MoE op takes the list inside the slot, and is void if both trees refuse. |
| §M26a I-4 | ONE commit carries the whole Graph-scratch move, cited by both designs | **Changed.** "The Graph-scratch commit" (23mk's, in beni) lists its parts: 23mk's `REGION` charge; the draw moved out of the ONEDNN zone (`onednn_graph_scratch_alloc`, `unified-cache.cpp:11655`) with the direct overflow deleted; the getter switch (`:4464`, `:27658` → `:2141`) with the with-floor getter deleted; the 0oxf clamp deleted; and the test and comment updates (m-14). C-1 lands none of it and keeps master's ONEDNN sizing on both sides of the dry run until then, so the witness cannot false-fire. H7ap's C-rule arm is an arm of that commit, and its mutant restores the deleted getter. §4's rows move the switch out of L6. No "in ONEDNN" place for the Graph scratch remains here except master's draw site, cited as master. |
| §M26a m13 | one dry-run definition, moua's | **Stated** at witness 2: the pack's charge ledger compared with the published getters; 23mk mirrors it. |
| §M26a m11 | the value function carries the 64 MiB minimum, the env override and the DNNL gate | 23mk's; this design cites 23mk's value function and does not restate it. |

**Noted for the lead (7.14d).**
- **`moe_onednn`'s activation and output slots.** They scale with `n_ubatch` (377487360 B +
  754974720 B on GPT-OSS 120B, 536870912 B + 1073741824 B on Qwen, all at 512), so §M27 (2)'s
  "any `n_ubatch`-scaled byte is class C" covers them as worded. But (2a) re-derives the load
  figures with them kept, and 512 is `MOE_GPU_UBATCH_MAX`, the ceiling of GPU MoE routing, not
  a context's choice. This revision keeps them as D at 512, the end states unchanged. Is that
  right, or are they per-context C slices like `moe_control`'s? The second option moves
  1132462080 B / 1610612736 B from load-time RUNTIME to each context's `REGION`, and 23mk's H3
  mirrors these figures.
- **`moe_control` above `MOE_GPU_UBATCH_MAX`.** The layout refuses an `n_ubatch` above 512
  (`exceeds_gpu_ubatch`), but no dispatch site reads that flag, and an explicit `-ub` is not
  capped (only the auto ladder is). As ruled, the value function sizes the slot at the
  context's own `n_ubatch` without that refusal (65792 B / 131328 B at `-ub 2048`). The
  alternative is a named refusal of a MoE context above 512 at its transaction, which the
  layout's own comment implies. Please confirm the first.
- **Landing coupling.** Row 134's block shrinks to k × stride only when the list and flag have
  their `moe_control` slot (from moua L4, where head slots exist) and the ids have row 75
  (beni). This revision puts the shrink in the slot's commit, with the block's non-table parts
  staying unconverted master bytes on H7p's list if 23mk's row-134 conversion lands earlier.
  That order is 23mk's and beni's to confirm.
- **The interim ONEDNN sizing** before the Graph-scratch commit treats the Graph-scratch floor
  G as capacity on both sides of the dry run, so C-1's witness cannot false-fire. It is sent
  to 23mk to mirror in its step-4 and dry-run text.
- Nothing was built for 7.14d; it is a document change only.

### 6.21 Revision 7.14e: rulings §M28, §M29, §M29a and §V14 I-E, 1oxa rev 11, zhcn 5.12

The lead ruled 7.14d's three questions (§6.20, "Noted for the lead (7.14d)") in §M28 and left
its (1) to one code fact, which this revision establishes from code at `c69d5774d`. §M29 rules
on this design's overlap finding (1oxa relay 5 below) and amends §M19; §M29a, on
design-1oxa-r11 I-A, amends §M29; §V14 I-E amends the `VM_TAIL_SURPLUS` relay. One commit on
top of 7.14d (`877c868e0`); design-moua-r15 reviews `44b4b9d66..` this commit, and no commit
follows it. Peer pins: 1oxa `f6d3015` (rev 11; §V14's fixes in progress), zhcn `d70c2d2` (rev
5.12), 23mk `026b69b86` (rev 4.7d; the mirror of this revision's end states is sent).

| id | ruling | disposition |
|----|--------|-------------|
| §M28 (1) | the ring's class follows one code fact: a dispatch that chunks at ≤ 512 keeps the ring D at 512; one that does not makes it C at the context's own `n_ubatch` | **Changed: class C.** The batched executor (`ggml-sycl.cpp:78424`) takes each expert's rows from the whole op's row maps (`:78536-78569`), pads to 64 (`:78596`) and sizes from that (`:78703-78709`); nothing splits a micro-batch. `ring` leaves step 1's P list (only `mmq_work_counter` remains) and is class C: its activation and output slots are head slots of the context's fit in `REGION`, at align64(`n_ubatch`) rows; u1bb's "in RUNTIME if it fits" branch is withdrawn (§2.7), and the load publishes the rows as 0. `moe_onednn` is the weight slot alone. End states: GPT-OSS 120B RUNTIME 564035584 + k × 1024 B, weight zone 13958742016 − k × 1024 B (537.9 / 13312.1); Qwen RUNTIME at its 536870912 B floor, weight zone 13985906688 B (512.0 / 13338.0). The ring at `-ub 512` is 1132462080 B / 1610612736 B per context. **There is no overflow** (the executor admits against the planned shape before any write); two refusals on master are named for a ticket (notes). H3's zones arm, the forced-tensor arm and the forgotten-term mutant are re-derived; the pack-before-ensure RED is scored on 120B only, void on Qwen; 7.14d's `vram_bytes` RED is withdrawn; H7ap gains a `ring` arm with three REDs. |
| §M28 (2) | `moe_control` at the context's own `n_ubatch`; no refusal above 512 on that account | **Stated** in the `moe_control` section ("confirmed"). |
| §M28 (3) | row 134's shrink, the `moe_control` C slot and row 75 are one landing set | **Stated** in the `moe_control` section and §4's L4 row: the shrink cannot land before the other two. |
| §M29 | the `reserve_onednn_scratch` pair is 23mk's (W D in ONEDNN, A C in `REGION`); `onednn_scratchpad` is consumer (b) alone; master's reorder + eligible value withdrawn; `get_size()` exact from descriptors, never measured; any M-dependent part is C | **Changed** (step 3, the enum row, "The ONEDNN zone's scratchpad"). Master's value (`ggml-sycl.cpp:15947`, `unified-cache.cpp:27565`) is withdrawn and deleted from both sites at L6. **M-dependence:** the oneDNN API gives the size only per descriptor and promises no invariance (`dnnl.hpp:4989`, `:689-692`), and the fork's descriptors carry a static M in their cache key (`gemm.hpp:348-364`, `:964-982`), so the docs do not settle it; a descriptor-only probe is specified for the lead to run (notes). §M29's zone formula is superseded by §M29a's. |
| §M29a | `onednn_scratchpad` is class C; one buffer per (context, queue) at the planned max, never grown; value = max `get_size()` over the context's descriptors, the M set enumerated unless monotonicity is proved; a request above the max is `[ZONE-PLAN-BUG]`; ONEDNN = max(floor, `onednn_pp_w` + other D terms) | **Changed.** Enum row class C; step 3 lists "not `onednn_scratchpad`"; the ONEDNN zone is `max(268435456, onednn_pp_w)`, since this design lists no other ONEDNN D term; the value function enumerates the reachable M set (the dense gate minimum to `n_ubatch`; multiples of 64 to align64(`n_ubatch`) for the MoE groups); the one-buffer rule, the named miss and the two-growth-step host RED; §4's L4 bullet. The probe decides the variation, the monotonicity and the enumeration's cost, not the class. H7ap: the ONEDNN mutant's fixture carries a W ≥ 256 MiB weight (K = 8192, N = 18432); the C-rule arm's zone reads `max(268435456, onednn_pp_w)`; the `--no-host` arm charges no `onednn_scratchpad`. |
| §V14 I-E | the surplus fence is scoped to the surplus objects, its owner device-scoped, cleared on a covering admit, the take and the last unload | **Changed** (§2.3.3 A1, A2): `VM_TAIL_SURPLUS` is owned by `{DEVICE, d}` and written through the new `record_pending_locked`, since the rollback's mark holds the group mutex; it clears at the three points. Relay 2's `{LOAD, txn}` owner is withdrawn and the LOAD owner's lifetime reverts. |
| §V14 I-D | the C-term set is every class-C member of the closed enum; no hand list | **Holds.** `ring` and `onednn_scratchpad` join the set by their enum rows alone. |
| 1oxa rev 11 relay 1 | keyed `WEIGHT` draws | **Changed** (§2.3.3 A2, §2.4.2): `allocate_within(owner, WEIGHT, key = {TLSF, offset}, ...)`; the pack's replay records each item's key and every draw of it is the keyed draw; an occupied keyed range is `[ZONE-PLAN-BUG]`; H7ap's keyed-draw arm. It covers §V14 m-12's moua items. |
| relay 2 | the `VM_TAIL_SURPLUS` pending term | **Changed**, as amended by §V14 I-E (above). |
| relay 3 (m-2) | the plan classes each copy; the pack charges a VM group's page-rounded increment | **Changed** (§2.4.2): `ceil(new group sum / page) − ceil(old group sum / page)`, from the one room number 1oxa's fit reads. |
| relay 4 | no yield for a head slot | **Changed** (§2.4.1): a head slot never takes tier 4, so it is placed in tiers 1-3 alone; the tenants-alone condition and step 2's refusal read tiers 1-3; §3.1's "No yield for a head slot" arm, with a tier-4 fit as its RED. |
| relay 5 (the overlap) | this design's finding, ruled in §M29 | Master's reorder + eligible value sizes the `reserve_onednn_scratch` pair, whose reorder half is W and whose eligible half stands in for A, so charging it beside 23mk's W and A counts W twice. |
| zhcn 5.12 (`d70c2d2`) | the forced phase is one parameterised call | **Fixed** (§3.1): `ggml_sycl::offload_stats_set_phase(forced_phase)`, quoted as its value, `offload_phase::TG` on every scored arm, not as a literal gate 34 pins. |
| self-found | `ring` (P, step 1) and `moe_onednn` (D, step 3) both named the one ring publish | **Fixed** with §M28 (1): `moe_onednn` is the weight slot, `ring` the rows, one owner per byte. |

**Noted for the lead (7.14e).**
- **Two refusals on master, for a ticket.** Neither is an overflow; each drops a micro-batch
  from the batched path (one WARN per process, "falling back to the serialized MoE route",
  `ggml-sycl.cpp:78794`) or, for an `XMX_TILED`-claimed op, throws
  `ggml_sycl_fallback_error` (`:78449-78479`). (a) The ring is per device and re-planned by
  the last transacting context (`ggml_sycl_replan_pp_moe_onednn_ring`, `:17512`, called at
  `:18512`), so a context whose `n_ubatch` exceeds the last transactor's is refused. (b) The
  rows are sized at `n_ubatch` (`unified-cache.cpp:2292-2320`) while the executor pads to 64
  (`:78596`), so an `n_ubatch` that is not a multiple of 64 can exceed the plan by up to 63
  rows per active expert's slot (up to 63 × `n_expert` rows; corrected in 7.14f, r15 m-2).
  `ring`'s per-context C slot at align64(`n_ubatch`) closes both; H7ap's `ring` arm
  carries them as REDs.
- **Cite correction.** `unified-cache.cpp:27338` and `:27391` are not the ring. `:27338`
  builds the `moe_control` layout at `MOE_GPU_UBATCH_MAX`, which §M28 (2) re-sizes per context.
  `:27391` sizes the sibling Q1_0/NVFP4 host recipe workspace at 512 × `n_expert_used`, and
  the dispatch throws above that (`ggml-sycl.cpp:75944-75950`). That workspace is not this
  design's term; it has the same `n_ubatch` question as the ring, for its owner. The ring is
  planned at `src/llama-model.cpp:481-500` and re-planned as in (a).
- **The probe (lead-run, descriptor-only, alone on the GPU).** Its source is in the session
  scratchpad at `m714e/onednn-scratchpad-m-probe.cpp` (not a committed artifact; "The ONEDNN
  zone's scratchpad" specifies it in full). After sourcing oneAPI: `icpx -fsycl -O2
  -DGGML_SYCL_F16 onednn-scratchpad-m-probe.cpp -ldnnl -o onednn-sp-probe`, then
  `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./onednn-sp-probe 4096 > sp-b50.csv` and the same with
  `level_zero:0` for the B70. The table is void unless the verbose positive control matches
  (7.14e wrote `ONEDNN_VERBOSE=create`, which prints nothing; the run used `profile_create`,
  rulings §M31).
- **The ring at the ladder's top rung (§V14 I-B).** 1oxa reserves VA for the auto-ubatch
  ladder's largest rung. With `ring` now a C term, that extent includes the ring's rows: at
  `-ub 4096`, 9059696640 B on GPT-OSS 120B and 12884901888 B on Qwen. Sent to 1oxa.
  **Corrected in 7.14f (r15 m-1):** a MoE model's auto ladder is capped at
  `MOE_GPU_UBATCH_MAX` = 512 (`src/llama-context.cpp:1423-1446`), so its top rung is 512
  (1132462080 B / 1610612736 B); only an explicit `-ub` reaches 4096, and 1oxa sizes that case
  from the context's fixed `n_ubatch`. The corrected premise is relayed (§6.22).
- **Mirrors.** 23mk: the end states (H3's bytes), `ring` and `onednn_scratchpad` class C,
  ONEDNN `max(floor, onednn_pp_w)`, and the landing set. 1oxa: `ring` joins the C-term set by
  its enum row, and `record_pending_locked` exists for the surplus fence.
- Nothing was built or run for 7.14e, the probe included; it is a document change only.

### 6.22 Revision 7.14f: design-moua-r15, rulings §M32, the 7.14f queue

design-moua-r15 reviewed `44b4b9d66..8fee92a67` and failed 7.14e (1 Critical, 6 Important, 9
Minor). The lead ruled its findings in §M32, and with them §M33 I-G, §V16 I-A, the probe
results (§M31, §M31a) and the queue held since 7.14e. §V15a, §V16a, §M34 (4) and §M35 were
ruled while this revision was open and are folded too. One commit on top of 7.14e
(`8fee92a67`). Nothing was built or run for it; it is a document change only.

| id | finding or ruling | disposition |
|----|-------------------|-------------|
| C-1 | the model's first context cannot place its ring: 7.14e freed the rows at load and the pack refilled the bytes with experts | **Changed, (a) and (b) both; (c) rejected (rulings §M32 C-1).** (b): the rows and the weight slot are sized by `local(t)`, the experts resident on the device (the executor refuses non-local ones, `ggml-sycl.cpp:78554-78557`). (a): the pack charges the device reservation each admitted expert's ring-row increment at n₀ = 512, the ladder's bottom rung, plus the first context's other mandatory C head slots at their first placement, so its capacity is the `WEIGHT` zone less the reservation; one pass, no fixed point. The reservation is the `FIRST_CONTEXT` pending term, recorded with the `WEIGHT` ranges, retagged to `{MODEL, id}`, taken over at the first context's step 5 in one group-mutex section, re-recorded by guard phase 1, cleared at unload (§2.3.3 A1, A4; §2.4.2 (b) step 3, step 5). H7ap's first-context arm on `gptoss120b-b1` and `glkg-qwen35b-a3b-b1`, scored by the pure-fit run, RED the 7.14e pack refused naming the ring head slot; conditional on zhcn's compute slot at load (notes). |
| I-1 | the ring was one device record over the max of contributions, with a stale RUNTIME half | **Changed (rulings §M32 I-1).** The rows are per-(context, device) CONTEXT slots in the context's published slot table; the executor's per-device read and claim (`ggml-sycl.cpp:78760-78767`, `:78817`, `:78823`) are re-keyed to them; the weight slot is a per-(model, device) D term. Every device-ring statement is deleted: the ring record, `ring_plan_gen`, RELEASING, guard pins, contributions, the sole-contributor release, the KV-zone split and its size function, and the RUNTIME half (§2.4.2 steps 2, 5, 8, the tenant-only path, Teardown; §2.4.3; §2.4.5; §2.7; §3.1 H2, H4, H8, H9, H7ah/aj/ak). |
| I-2 | `REGION` headroom named no TLSF set; GA and the RUNTIME idle floor | **Changed.** `REGION` headroom is the free room of the shared zone's TLSFs, allocator group `vram_zone_id::KV` (`WEIGHT` delegates to it), never RUNTIME, ONEDNN or SCRATCH (§2.4.1). RUNTIME has no floor on an arena device and is laid out at 0 before the pack, since master's two floor consumers, compute buffers and the MMID pools, are `REGION` head slots here (§2.4.2 (b) step 4, the end states). GA re-derived: room 2204.6 MiB, head slots 1348.0, −709.4, 6 layers, 58.6 spare (§2.4.1). Qwen's weight zone is 14380171264 − k × 2048 B; the ONEDNN and SCRATCH floors are ruled and kept (notes). |
| I-3 | the forgotten-term RED depended on a setter H7z keeps off the load | **Changed.** `moe_onednn` has its own setter, `unified_cache_set_planned_pp_moe_onednn_weight_slot_bytes`, mapped to `MOE_ONEDNN`; H7z stays as written; the RED is rewritten against the new setter and now binds on both models (564035584 B / 142606336 B, with no floor); the RUNTIME getter's three summands are stated (m-8). |
| I-4 | the C-rule RED and the ONEDNN end state leaned on a publication moua deleted and 23mk re-points | **Changed.** 23mk's re-point and rename are adopted (`unified_cache_set_planned_onednn_pp_w_bytes`, 23mk §4.3; §M33 I-G); "deleted from both sites" is withdrawn; the C-rule arm asserts W > 0 and G > 0, else VOID, with the §M30 line as its placement witness. |
| I-5 | the value function and the probe covered matmuls only | **Changed.** The value function spans all thirteen `get_scratchpad_mem` sites, each under its dispatch's gate, enumerated once at load (§M31); a source gate maps each site to a family; the queue set is named and frozen per context (§M35). §M31a's results (all 0) are recorded, with the families it did not probe (notes). |
| I-6 | L4 and L6 reserved the rows twice | **Changed:** L4 and L6 land as one commit (§4). |
| m-1 | the auto ladder's top rung | **Fixed** (§2.4.2 (b), §6.21 note): 512 for a MoE model. |
| m-2 | "up to 63 rows" | **Fixed:** per active expert's slot, up to 63 × `n_expert` rows. |
| m-3 | `VM_TAIL_SURPLUS` | **Fixed** (§2.3.3 A1): replace per (term, device, TLSF); the partial trim as `clear_locked` plus `record_locked` of the remainder in one section; owner rules extended to `{DEVICE, d}`; a failed first load records under `{DEVICE, d}` directly (that last rule, (d), is withdrawn in 7.14g by rulings §V17 m-10, §6.23). |
| m-4 | the keyed draw | **Fixed** (A2, §2.4.2): one key, `{TLSF, offset, size}`; the items are each SYCL<n> buffer and each individually drawn tensor or layout copy. |
| m-5 | the ring arm had no engagement witness and needed a GPU | **Fixed** (H7ap): a host arm over the pure `pp_moe_onednn_admit_scratch` with the executor's shape formula, and a lead-run GPU arm with a `pp_moe_batched_admitted` counter ≥ 1, else VOID. |
| m-6 | F2's powers-of-two batches | **Stated** as a gap (notes); the value function is computed, so it covers the gap anyway. |
| m-7 | the value function's cost | **Fixed:** enumerated at load, outside L1; 15-30 µs per cached descriptor, up to 1.7 s for a first create. |
| m-8 | the RUNTIME getter's summands | **Fixed** (§2.4.2 (b) step 5). |
| m-9 | the fold queue | **Fixed:** `classify_references` (A3), §M30's line (H7ap), §V15 as amended by §V15a (§2.11), the peer re-pins (below). |
| §M33 I-G | the ONEDNN store renamed for W | **Adopted** (23mk's). |
| §V16 I-A | `pre_allocate_scratchpad` sizes nothing; a climb is a C-term re-place | **Changed** (§2.4.2 (b), H7ap's two arms). |
| §V15a, §V16a | cap0 frozen per context; MMID at the context's own `n_ubatch` | **Changed** (§2.11; §2.4.2 step 7, whose "largest rung" sizing is withdrawn). |
| §M34 (4) | the host runtime pre-size composite | **Changed** (§2.7): re-derived to the host-tier terms. |
| §M35 | the execution queue is per context | **Cited** in the queue set; llama.cpp-0j5w is the precondition. |
| 23mk 4.8 (a), (c) | `ONEDNN_GRAPH_SCRATCH` term; not carved at step 6 | **Folded** (A1, step 5, step 6). |
| zhcn 5.13 (1h) | the L4 arm replacing zhcn's H2 (1)/(1b) | **Folded** into L4's contents (§4). |

**Noted for the lead (7.14f).**
- **zhcn's compute slot at load (open).** C-1 (a) reserves the first context's mandatory C
  head slots at load, and zhcn's compute slot is the largest of them (808.0 MiB in GA), but
  zhcn 5.13 measures it only at the context transaction. This design asks for zhcn's measure
  pass to run once at the load envelope; until then H7ap's first-context arm injects the
  context-time value, and a live load reserves short by that slot, which on the merge gates is
  a refusal naming zhcn's compute slot.
- **The load envelope's `n_ctx`.** The envelope carries 0, so the reservation is sized at the
  planner's 512. A first context with a larger `n_ctx` still fits its KV by demotion, but any C
  term that scales with `n_ctx` (the Graph scratch) is reserved at 512. Proposal: carry the
  context parameters the model was loaded for in the envelope.
- **ONEDNN and SCRATCH floors.** I-2's P4 test removes RUNTIME's floor. Applied to ONEDNN and
  SCRATCH it finds idle bytes too: 244842496 B (233.5 MiB) of ONEDNN on GPT-OSS 120B, and
  SCRATCH's floor less its terms. Both floors are ruled (§M25 I-1, §M29a) and are kept; is the
  test meant to reach them?
- **§M31a's gaps.** No probe rows for the Q4_0 WOQ (default-reachable on a Q4_0 dense
  weight's oneDNN PP arm), the Q8_0 WOQ, or the 3-D WOQ; F2's batches are powers of two while a
  group's size is any integer; F2 is reachable only under `GGML_SYCL_MOE_PP_WOQ=0`. The value
  function covers them, since it is computed; the probe's "all 0" does not.
- **The m-1 relay correction** is sent to 1oxa (below).
- **23mk's Graph scratch on the RUNTIME TLSF.** 23mk 4.9 still places the Graph-scratch range
  on "`dev`'s RUNTIME TLSF" (L2635, L2652, L4523, and the stage slot row at L5378). Under I-2,
  `REGION` is the shared zone's TLSFs, so those lines contradict this design; relayed.
- **The host runtime pre-size (§M34 (4))** keeps only `dma_staging_pool_bytes` on an arena
  device. If L4's census finds a host-tier consumer of W, `pp_pipeline` or ring bytes, it
  becomes a term of its own.
- **Record mode and the shared weight slot.** If L4 finds the PP MoE oneDNN path reached while
  recording, a recorded holder takes the weight slot from every other context of its model;
  that would need the slot's depth to count recorded holders, and it comes back to the lead.

**Peer pins.** The relays' line numbers are against 23mk `cf1b6da02` (4.8a), 1oxa `5781ede`
(rev 12) and zhcn `9a6d749` (5.13); the rename cites are against 23mk `e4f08213a` (4.9).
Later revisions exist (23mk 4.9 `e4f08213a`, 1oxa rev 13 `8789db6` / `47206cb`, zhcn 5.14
`ac1df3d`); each owner re-locates the relay lines there.

**Mirror relays.**
- **23mk** (at 4.9 `e4f08213a` unless marked): the Qwen end state is RUNTIME 142606336 + k ×
  2048 B and weight zone 14380171264 − k × 2048 B (L43-45, L117, L5696, L5744, L5747, L5763,
  L6441, L6780, L6803; at `cf1b6da02` the H3 GREENs L5410-5455, L6082-6085, L51, and row 125 at
  L1741-1770); the pre-plan split on an arena device is 14522777600 B (13850.0 MiB), so the M14
  message reads 13850 MB > 13312 MB on 120B (L3586, L5766; `cf1b6da02` L5469-5472, L5485), and
  the drop-a-term precondition is 564035584 B / 142606336 B, live on Qwen (L44); the Graph
  scratch lies in `REGION`, the shared zone, not the RUNTIME TLSF (above).
- **1oxa** (at `5781ede`): the "RUNTIME half" text at L526, L838, L2171, L2202, L2338 and L3788
  is gone; L923-925's "the same ranges sit on the device's RUNTIME TLSF" becomes the shared
  zone's TLSFs; L917-920's C-term list gains `ring` and `onednn_scratchpad`; L424-431's two
  items not at 7.14d (the keyed `allocate_within`, the OPTIONAL group charge) and the
  `onednn_scratchpad` row are in 7.14e/7.14f, so the cite moves to 7.14f; the §V14 I-B VA
  reservation's top rung is 512 for a MoE model (m-1).
- **zhcn** (at `9a6d749`): T4 at L308, L952 and L1475-1477 read the rows per context; G2/GA
  carry 270.0 and 540.0 for the rows and −709.4 for GA.

The disposition tables of §6.1 to §6.21 record what earlier revisions did; where one describes
the device ring, its RELEASING mark or its RUNTIME half, this section supersedes it.

### 6.23 Revision 7.14g: rulings §G1, §G1a, §V17, §M36 and §M37, and the peer revisions

No review ran between 7.14f and this revision. It folds the rulings made on 7.14f's range and on
its §6.22 questions: §G1 and §G1a (one STRICT switch, read in one place), §V17's items for
this design (m-10, I-1, I-2, I-3), §M37 (the answers to §6.22's notes) and §M36's relays. It
also folds three peer revisions: 23mk 4.9 (`fc463f6b3`, committed through `e4f08213a`), 1oxa
rev 13 (`8789db6`, `47206cb`) and zhcn 5.14 (`6db46c3`, through `5be1187`) and 5.15
(`edd507b`). One commit on top of 7.14f (`ee484eb5b`). Nothing was built or run on a GPU for
it. The two W values below were read from GGUF headers with 23mk's header parser.

| id | ruling or relay | disposition |
|----|-----------------|-------------|
| §G1 | the fork has one strict switch, `GGML_SYCL_STRICT_LEASES`; every `[*-PLAN-BUG]` abort uses it; no second switch | **Changed.** Every normative `GGML_SYCL_STRICT_PLAN` (20 occurrences on 18 lines) and every `GGML_SYCL_STRICT_KV_PLAN` (3, one line of its own) now reads `GGML_SYCL_STRICT_LEASES=1` with a §G1 cite, or is rewritten where a table or decision recorded the old D3 name ((g), §4's zhcn line, §6.5's summary, §6.6's contract row). Nineteen bare "STRICT" uses in §1-§5 cite §G1. The names survive only in §2.8's withdrawal note and in H7 (e)'s mutation witness, so a search finds the withdrawal. The disposition tables of §6.1 to §6.22 are history and are not rewritten; this row supersedes them. |
| §G1a | one reader, exported as `ggml_sycl_strict_enabled()` by the first lane that lands `[*-PLAN-BUG]` code outside `unified-cache.cpp`; no second `getenv`; tags per family; the landing lane's three doc edits | **Changed** (§2.8; H7 (e); §4's L6 row). Every call site in this design is the accessor, and the TU-static reader is cited only as master's location. H7 (e) forbids any `GGML_SYCL_STRICT_[A-Z_]*` other than `_LEASES` and any second `getenv` of the switch, which is zhcn's gate 22 rule. If moua's L4+L6 commit lands first, it exports the accessor and carries the `sycl-env-vars.md` row, the :590 correction and the contract's :178-179 correction. `GGML_SYCL_HANDLE_STRICT` only reports and is outside §G1. |
| §V17 m-10 | rule (d) withdrawn: a rollback that leaves no live model records no fence and trims the surplus; rule (a) adopted | **Changed** (§2.3.3 A1): (d) withdrawn; a fence is recorded only by a rolled-back load that leaves a live model; the rollback that leaves none is the fourth clear point; (a), (b) and (c) stand; the `DEVICE` owner kind's lifetime says the same. §6.22's m-3 row is annotated. |
| §V17 I-1 | the owning design says whether each C term is carved | **Cited** (§2.3.3 A1's `ONEDNN_GRAPH_SCRATCH`, step 7): every head slot of this design, MMID's included, is carved at step 6; 23mk's Graph scratch is the one term left pending. |
| §V17 I-2 | 1oxa's VA is 2·Σ_t max_r C_t(r), per-term A/B slots | **Noted** (below): no figure of this design moves. |
| §V17 I-3 | 1oxa adds a reserved ledger state for the C-1 mirror | 1oxa's; this design's `FIRST_CONTEXT` transitions (commit, handover, unload, rollback; §2.3.3 A1, A4) are the ones 1oxa's T14-T17 mirror. |
| 23mk I-G | the five ONEDNN W names | **Adopted everywhere** (§2.4.2 (b) steps 2 and 5, the Graph-scratch commit's parts, the C-rule arm, §2.4.5's `onednn_pp_w` row, §2.7's pre-size, §4). Master's names appear only as "today `…`". `onednn_scratchpad` has no load-stage setter (its enum row). |
| §M34 (4) | the host runtime pre-size | Already re-derived in 7.14f (§2.7). This revision names the function that holds it, `ggml_sycl_configure_host_zones_for_plan` (`ggml-sycl.cpp:5614-5680` at `c69d5774d`), a load-time function rather than a context init, and cites its first field by W's new name. **The result:** on an arena device the pre-size is the host-tier terms only, which at `c69d5774d` is `dma_staging_pool_bytes`. W, `pp_pipeline` and the ring's bytes leave it, since each is already placed on the device, and the load-time ring reserve at `:5657-5680` goes too. |
| §M31a | the gap probe's families, E's evidence, `reduce_last_dim`, the value computed | **Stated** (§2.4.2 (b), "The probe"): R's three routers, E's three eltwise algorithms at three widths, S, B, 65512 rows per card, rc = 0, `Shmem` flat; E backed by descriptor equivalence only; the reduction has no caller; the value stays computed and is 0 today. |
| 1oxa rev 13 relays | `assert_scratchpad_planned(stream)`; the climb swapped into `scratchpad_map[q]` at the publish; the covering admit as clear then record of the remainder; a planned 0 tolerated | **Adopted** (§2.4.2 (b); H7ap's scratchpad arms; §4). The covering-admit composition was already rule (b) of §2.3.3 A1, so no sub-range clear primitive exists, and 1oxa's text (rev 13 L59-61) agrees. The planned-0 tolerance is read at `c69d5774d`: `common.hpp:5951-5953`, the `gemm.hpp` guards at `:404`, `:442`, `:679` and `:910`, and `dnnl-ops.hpp`'s throws at `:81`, `:185`, `:283`, `:339` and `:432`, which fire only when `get_size() > 0`. |
| §V16a | MMID at the context's own `n_ubatch` | Already one statement since 7.14f (step 7). The remaining "largest rung" text is the scratchpad descriptor table's M set, which is correct: the table is enumerated at load for every rung and read at the context's own. |
| §M37 Q1 | zhcn measures its compute slot once at the load envelope, at the bottom rung; until then a labelled fixture constant | **Changed** (step 3; H7ap's first-context arm). |
| §M37 Q2 | `n_ctx` at load is llama.cpp-fkpg (a); until then 512, and a first context needing more is refused at its transaction naming the term | **Changed** (step 3): the excess is placed in the context's `REGION` headroom like any head slot, demoting KV, and the refusal naming the term is the case where that room does not exist; never silently short. |
| §M37 Q3 | the P4 test applies to the ONEDNN and SCRATCH floors | **Changed** (step 4; §2.4.5; the end states; H7ap's floors arm; §4's L4+L6 zones line). ONEDNN: its last unplanned consumers are A, the scratchpad and the Graph scratch; the floor goes in the later of L4+L6 and the Graph-scratch commit. **Freed: 244842496 B (233.5 MiB) on GPT-OSS 120B and 234881024 B (224.0 MiB) on the Qwen gate.** The weight zones become 14203584512 − k × 1024 B and 14615052288 − k × 2048 B, and the pre-plan split becomes 14791213056 B. SCRATCH: load-bearing today, and the untermed consumers are named as defects by their 23mk census rows. The floor goes in the commit that converts the last of them. The freed bytes are 536870912 B less the SCRATCH terms: **352509948 B (336.2 MiB) on 23mk's 120B H3 fixture**, and the live value comes from the ledger. |
| §M37 Q4 | the unprobed families stay computed, marked "unprobed" until a probe extension runs | **Changed** (§2.4.2 (b), "The probe"). |
| §M37 Q5 | relayed to 23mk (§M36) | Nothing for this design. |
| §M37 Q6 | the PP MoE oneDNN path while recording is `[ZONE-PLAN-BUG]`; it never retains the shared weight slot across a recording | **Changed** (§2.7 "Record mode"; §2.4.1's GA note; H7 (ar)). The depth question of §6.22's last note is closed. |
| zhcn 5.14 (1) | which branch zhcn keeps; re-derive GA and the H2 fixture | **Answered** (below; §2.4.1; §5 (n)). |
| zhcn 5.14 (2) | whether C-1's fix restores I8-G0's zero demotion | **Answered** (below). |

**zhcn 5.14's two questions.**
- **The branch.** zhcn keeps neither (R) nor (F). 7.14f is a third state: (R)'s rule, with
  both rows in `REGION`, but with RUNTIME's idle floor removed (rulings §M32 I-2). On the B50
  GPT-OSS 20B geometry RUNTIME drops from 512 MiB to its D terms, 134.5 MiB (`moe_onednn` 32 ×
  4406528 B, plus `moe_ptr_table`'s under 0.1 MiB). So every row gains **377.5 MiB** of room
  over (R): I8-G0 −190.9 → **+186.6** (0 demoted), I8-GE −400.9 → **−23.4** (1 demoted), G2
  −318.9 → **+58.6** (spare 58.6), and GA −1086.9 → **−709.4** (6 demoted). §2.4.1's GA and
  H2's fixture already read −709.4 from 7.14f, and this revision corrects §5 (n), which still
  read −726.9. This holds for every row that is the B50 GPT-OSS 20B geometry. A row on another
  geometry moves by that geometry's RUNTIME floor less its RUNTIME D terms. The score is the
  fit run, not this arithmetic. Once the ONEDNN floor also goes (§M37 Q3), each of these rows
  gains a further 233.5 MiB, and the Graph scratch becomes a `REGION` head slot in the same
  state, so zhcn pre-registers that state from H2's run at that commit.
- **I8-G0's zero demotion.** It is restored by I-2's floor removal, not by C-1's reservation.
  +186.6 MiB is room left after full KV and both rows. C-1's reservation does not bind on
  GPT-OSS 20B: the pack does not fill the zone (11510.9 MiB of weights against a 13715.5 MiB
  zone), so the reservation takes room the pack left free, and the fit counts it as the first
  context's own. So C-1 neither restores nor breaks I8-G0. This is conditional on I8-G0 being
  the B50 GPT-OSS 20B geometry, and the score is the fit run.

**§V17 I-2, 1oxa's per-term VA.** VA = 2·Σ_t max_r C_t(r). It equals 2 × C(top) only if every
term is monotone in the rung, and 1oxa computes the sum from the enum's values, so this design
need not prove monotonicity. For the record, this design's `ring` is linear in the rung
(1132462080 B at 512, 9059696640 B at 4096 on GPT-OSS 120B, ×8). The MoE auto ladder has one
rung, 512 (r15 m-1), and an explicit `-ub` fixes `n_ubatch`, so a MoE context's rung set is a
single rung and its max is its value. No figure of this design moves, and there is nothing to
relay back.

**Noted for the lead (7.14g).**
- **Which commit removes the ONEDNN floor** depends on the order of L4+L6 and 23mk's
  Graph-scratch commit in beni, which the landing plan (§4) does not fix. This design puts the
  removal in whichever lands second, and H7ap's floors arm is that commit's.
- **The SCRATCH floor's defects** belong to five owners: beni (8 rows), zhcn (1), pqmm (2), 6lfq
  (1), and the two exemptions. The floor goes only when the last of them converts. Until then
  it stays as capacity for named, untermed consumers, not as slack.
- **The Qwen gate's model file** is `Qwen3.6-35B-A3B-UD-Q5_K_S.gguf`
  (`glkg-qwen35b-a3b-b1.log:7`), while this document and 23mk name the shape "Qwen3.5-35B-A3B".
  The byte figures come from the file the gate loaded. 23mk's §4.2 W table does not list that
  file, so its W (33554432 B) is stated here and relayed.

**Peer pins.** 23mk at `e4f08213a` (4.9, committed; the working tree has uncommitted edits,
which this revision does not read), 1oxa at `47206cb` (rev 13), zhcn at `edd507b` (5.15).
§6.22's relays were pinned to older heads, and they are re-located here.

**Mirror relays.**
- **23mk** (`e4f08213a`): §6.22's relays stand at the same lines, since 23mk has not committed
  since (§M36 carries them). New in 7.14g:
  - under §M37 Q3, ONEDNN is `onednn_pp_w` alone once its floor goes, not `max(268435456,
    onednn_pp_w)` (L46, L1189, L1329, L2824-2827, L3524, L4525);
  - the H3 GREENs' "SCRATCH and ONEDNN are unchanged at their floors" (L5728) gain the
    ONEDNN-floor state: 23592960 B and weight zone 14203584512 − k × 1024 B on 120B,
    33554432 B and 14615052288 − k × 2048 B on the Qwen gate;
  - the SCRATCH floor's untermed census rows are the ones listed in the end states;
  - the Graph-scratch commit removes the ONEDNN floor if it lands after moua's L4+L6;
  - §G1a applies to 23mk's DECLINE class too (§M36 I-3 aborts it under the §G1 switch).
- **1oxa** (`47206cb`):
  - "as on USM, where the same ranges sit on the device's RUNTIME TLSF" (L980-982) still names
    RUNTIME; the ranges sit on the shared zone's TLSFs (§2.3.3; rulings §M32 I-2);
  - the §4 comparison row "pending ranges for head slots on existing chunks and on the RUNTIME
    TLSF" (L4056) is stale in the same way: no transaction records a range on RUNTIME;
  - the VA text (L995-1004) is 1oxa's to rewrite under §V17 I-2. Its `-ub 4096` figures remain
    correct for an explicit `-ub 4096`, since the ring is linear in the rung;
  - rule (d)'s withdrawal and the fourth clear point match 1oxa's m-3 (L59-61);
  - moua is at 7.14g from this commit, and 1oxa's pin at 7.14e (L467) moves when 1oxa next
    folds.
- **zhcn** (`edd507b`):
  - the branches (L39-42, L350, L1609-1616, L1645-1646, L1742, L2158-2161) collapse to 7.14f's
    state, with the integers above;
  - "the weight slot is a DEVICE-scope reserved slot in the RUNTIME TLSF" (L348, L1605, L1902)
    is stale. From 7.14f the weight slot is `moe_onednn`, one per (model, device), drawn at
    load under `{MODEL, id}`, and `demand_scope` has no DEVICE value (§2.4.3);
  - the compute slot at the load envelope is §M37 Q1's, and the lead has relayed it.

### 6.24 Revision 7.14h: design-moua-r16, rulings §M38 and §G1b

design-moua-r16 reviewed 7.14f (`8fee92a67..ee484eb5b`): 2 Critical, 5 Important, 3 Minor, all
ruled in §M38. This revision folds them, the note that two of the thirteen scratchpad sites are
dead, §G1b (the strict-switch correction), and §Z20 where it binds this design. 7.14g
(`3d782e264`) was drafted before the verdict arrived and carries none of it. One commit on top
of 7.14g. Nothing was built or run on a GPU for it; the code facts are read at `c69d5774d`.

| id | finding or ruling | disposition |
|----|-------------------|-------------|
| C-1 | the `FIRST_CONTEXT` reservation was a hand list sized at `n_ctx` 512; it omitted the recurrent state and jzvq's slots, so the Qwen gate's first context at `-c 4096` was about 190.8 MiB short and refused | **Changed.** The reservation is the byte sum of `context_demand_records` at the envelope's shape, the one function the context transaction's step 2 reconciles (§2.4.3, §2.4.2 (b) step 3); the load names no term. It is sized at the envelope's `n_ctx`, so **L4 depends on llama.cpp-fkpg (a)** (§4), and §M37 Q2's 512 interim no longer applies to `FIRST_CONTEXT`. H7ap's first-context arm pre-registers each term at `-c 4096 -ub 512` on both gates: known terms 1560297728 B / 2398978304 B plus jzvq's, with a second-enumeration mutation RED and a 512-fallback RED. |
| C-2 | the RUNTIME floor's removal rested on a two-consumer census | **Changed.** The end states carry the full census of RUNTIME draws at `c69d5774d` (an anchored search of the tracked tree, with 23mk's rows): each with its owner and landing point. zhcn's carve and 23mk's reorder temporary land before L4+L6; beni's rows 13 and 14 keep a transitional term, `xmx_moe_buffers`, sized by their own byte functions and 0 unless two opt-ins are set; `dense-scheduler.cpp:29` has no user and is deleted. No site lacks a size function. RUNTIME's capacity is the planned sum of its named consumers. New arm C9: zero `[EXT-ALLOC]` lines of any role on both gates, with the trace armed and a positive control (§3.3). |
| I-1 | the four device-global ring setters stayed reachable from the commit's install | **Changed.** No arena path reaches them (§2.7); H7z (aj) walks the call graph from the load, the transaction, the release proc and the executor, and its new witness plants `_row_bytes` in the install. |
| I-2 | the weight slot's store was per device, last writer wins | **Changed.** Keyed by (model, device) (§2.4.2 (b) step 3, §2.4.5). A later load on a laid-out arena places its D terms in RUNTIME's free room or the shared zone, inside its own load. H9's second-load arm pre-registers both fixtures, and last-writer-wins is its RED. |
| I-3 | the dry run kept master's 512 MiB RUNTIME default | **Changed.** L4+L6 deletes it from `ensure_planned_arena_zones` (`unified-cache.cpp:4505-4508`); a Mistral positive control grows nothing, RED 536870912 B (H7ap). |
| I-4 | §M37 not yet folded | **Already in 7.14g** (`3d782e264`, §6.23), which r16 did not see. 7.14h amends Q2 for `FIRST_CONTEXT` (C-1). |
| I-5 | the ring GPU arm's witness was compiled out in production | **Changed.** A per-context `[PP-MOE-RING]` line at WARN, in every build, with a positive and a negative control. §3 opens with the rule that a VOID arm fails. |
| m-1 | stale cite of the RUNTIME default | **Fixed:** `:4505-4508`. |
| m-2 | the L4 row's "pending ranges on shared and RUNTIME TLSFs" | **Fixed:** the row names which terms sit on the shared zone's TLSFs and which on RUNTIME. |
| m-3 | GA's hand arithmetic left out the other C slots | **Fixed:** every head slot on GA's shape is listed (§2.4.1); the known ones sum to 8.2 MiB, and the count stays 6 while jzvq's fattn workspaces stay under 50.4 MiB. |
| dead code | 2 of the 13 `get_scratchpad_mem` sites are dead (§M36 I-3) | **Changed.** The value function and the gate cover the eleven live sites; `binary` and `reduce_last_dim` are listed as deleted, and a call site restored in either fails the gate. `binary_broadcast_row` (`:338`) stays as family B. |
| §G1b | the "one switch" claims were false; uwlx exports the accessor | **Changed.** No text claims a single switch at master; §2.8 names the switch every ownership, lifetime and plan defect family folds into, lists uwlx and ds41, and cites no other switch by name. uwlx owns the export and the three doc edits, and L4+L6 carries them only if it lands first. H7 (e) adds the suffix form and reuses gate 22's allowlist. |
| §Z20 | zhcn's load-time measure through a transient measure-only context, at the planner's 512 until fkpg (a) | **Adopted** (step 3): since L4 waits for fkpg (a), the reservation's compute slot is measured at the envelope's `n_ctx`. |
| ONEDNN floor | the later of L4+L6 and 23mk's Graph-scratch commit removes it (lead, confirmed) | **Changed.** Neither order is assumed: each arm checks for the other, the earlier keeps named terms in place of the constant, and the later deletes the last with a RED on the floor's bytes (the end states, H7ap, §4). |
| §T9c | the host-buft sync guard is queue identity | **Not applicable:** this design does not cite that guard. |

**Notes for the lead.**
- **A later load's D terms (I-2).** r16's formula, "RUNTIME = Σ over live models' D terms + this
  load's", needs RUNTIME to grow on a live arena, which is the rebuild every design refuses. So
  RUNTIME holds the D terms of the load that laid it out, and a later load's D terms take
  RUNTIME's free room or the shared zone, as model-owned ranges drawn in that load. Once the
  first model unloads, RUNTIME's freed bytes stay laid out until the arena is re-laid; that idle
  room is bounded by the first model's D terms, and re-laying a quiescent arena is out of scope.
  23mk's `moe_ptr_table` `MODEL_TERM` meets the same case (relayed).
- **The reservation moves weights to the host.** On the Qwen gate the reservation is about
  2.29 GB, so the pack's capacity is about 11426 MiB (13714.0 less the reservation) against
  master's 12136.5 MB of device weights (`glkg-qwen35b-a3b-b1.log:2730`): fewer experts on the
  device, run by the CPU (placement decides the executor). That is the plan's answer, not a
  regression to tune away, but the perf arms should expect it.
- **jzvq's values are the one hole in the pre-registration.** The first-context arm needs
  jzvq's fattn workspaces and TG caches at both gates' shape from jzvq's host test before the
  lead's run, and it fails as VOID without them.
- **The compute-slot fixture constants** (423624704 B / 516947968 B) are master's measured
  compute buffers at the gate shape, until zhcn's load-time measure lands.
- **The recurrent state** is computed from the gate's hyperparameters (30 recurrent layers,
  `ssm_d_conv` 4, `ssm_d_inner` 4096, `ssm_d_state` 128, `ssm_n_group` 16) and equals r16's
  65863680 B.
- **`dense-scheduler.cpp:29`'s deletion** is assigned to L4+L6, since its class has no user
  (23mk row 20); 23mk may prefer to take it.
- **C9's baseline** is a lead run on `c69d5774d` with the trace armed, and its cohorts must be
  dispositioned before L4+L6 lands.
- **The `[PP-MOE-RING]` line** prints once per context in every run, including non-MoE ones.

**Relays.**
- **23mk** (`e4f08213a`): the reorder temporary (row 126) lands before L4+L6; a later load's
  `moe_ptr_table` range goes to RUNTIME's free room or the shared zone (I-2's rule); the ONEDNN
  floor order is checked by each commit's arm; the two dead wrappers' deletion is what the
  scratchpad gate expects.
- **zhcn** (`edd507b`): the `fattn-onednn` carve lands before L4+L6; the load-time compute
  measure runs at the envelope's `n_ctx` once fkpg (a) lands; H7 (e) reuses gate 22's
  allowlist.
- **jzvq:** pre-register the fattn workspaces and the MXFP4 MoE TG caches at `-c 4096 -ub 512`
  on both merge gates, and at GA's shape.
- **beni:** rows 13 and 14 carry `xmx_moe_buffers` until beni converts them; the conversion
  deletes the term.
- **uwlx:** the export and §G1a's three doc edits are uwlx's (rulings §G1b).
- **fkpg:** L4 depends on (a).
