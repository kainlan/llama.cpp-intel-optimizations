# JSON Extract

- **Rationale:** Merge of proposal-0 (by subsystem seam) and proposal-1 (by owner outcome). Proposal-1 is the spine: it covers all 396 ids, and its outcome framing states bars in the units the owner's 2026-09-01 efficiency ruling uses — 80% of hardware peak, bandwidth for decode (B50 224 GB/s, B70 608 GB/s) and INT8 XMX for prefill (B50 170 TOPS, B70 367 TOPS) — where proposal-0 mostly restates the perf-baseline floors. Proposal-0 contributes three things back. (1) Its separation of graph replay / submission overhead from reclamation: proposal-1's 'event-ordered-no-waits-no-resets' bundled ruling 3 (reclaim by refcount, kill zone_reset) with ruling 6 (order by events) and graph replay into one 20-member epic covering three orderable workstreams; the merge splits it into refcount-reclaim-no-zone-resets and event-ordered-graph-replay, taking the epic count to 18 with no epic above 35 members. (2) Its homes for tickets proposal-1 placed by topic rather than by the work: placement-ruling violations and -ngl deprecation to the planner, attribution tickets (pbb2, r9dn, 41sq) to decode-cost-attribution rather than to the optimisation epics they inform, skip/red-baseline/probe-control tickets to test integrity, and the fresh-boot validation run to the gates epic. (3) Its epic-ticket reuse for mubmt (allocator) and rg2ft (pointer escapes), which proposal-1 had crossed over. Proposal-0's separate attention epic was NOT kept: only about five of its eight members are genuinely fattn work, two are [ports] tickets and one is a correctness bug, so it could not carry an independent bar. All 396 keep ids are assigned exactly once; nothing is unplaceable.
## Epics

### One allocator: every SYCL byte enters through the unified cache

  - **Slug:** unified-cache-sole-allocator
  - **Existing epic id:** llama.cpp-mubmt
  - **Title:** One allocator: every SYCL byte enters through the unified cache
  - **Goal:** The unified cache is the only thing that calls a SYCL allocator. Close the remaining direct sycl::malloc_*/free sites, the 18-site pool-alloc backlog, the bare malloc_host cohorts (moe_transient_ptr_table, MMID h2d staging), the static staging caches, and the byte-contract and bound-check holes in the public allocation API, so that unified_alloc / unified_allocate / unified_allocate_owner are the only doors in. Also owns the cache's own entry-point defects: the lookup locking claim, the device-index bound check, READY-while-in-flight publish, the staging/mem_copy refusal channel, and the L0 staging-copy failure.
  - **Bar:** scripts/check-sycl-alloc-usage.sh and tests/test-sycl-owner-allocation-migration.py both green at HEAD with the allowlist reduced to the unified-cache implementation itself (today 18 pool-alloc sites remain in ggml-sycl.cpp/conv3d.cpp/cross_entropy_loss.cpp); a grep for sycl::malloc_device/malloc_host/sycl::free outside ggml/src/ggml-sycl/unified-cache*.cpp + pinned-pool.cpp returns zero; GGML_SYCL_ZERO_ALLOC_CHECK reports zero steady-state allocations across a Mistral tg128 and a GPT-OSS 20B tg128 run; llama.cpp-mubmt.13's 20B/120B/multi-GPU zero-allocation regression gate is registered and green; Mistral completion gate passes and B70 Mistral llama-bench stays within baseline (>=2495 pp512 / >=98 tg128, allowing the 10% B70 tg spread).
  - **Constraints:** Ruling 1 (sole allocator; no side caches, no raw TLSF outside unified-cache; allocation_control_class is fixed before coordinator admission and CACHE_BACKING is mintable only via the private cache_backing_token). Ruling 2 (an allocation surfaces only as a mem_handle or an alloc_owner; alloc_handle is a migration state, not a third surface). Ruling 10 (iGPU host-unified memory must never be budgeted as VRAM). Ruling 6 forbids adding a wait to make an allocation path safe. Ruling 11 (do not resurrect env-vars the source no longer reads).
  - **Ordering hint:** Start with the two umbrella tickets (llama.cpp-mubmt, llama.cpp-z2316, llama.cpp-32dg8) to fix scope, then the audit that enumerates the remaining sites (llama.cpp-9lkgd / llama.cpp-mubmt.12, llama.cpp-7phf). Land the API byte-contract and bound-check fixes (llama.cpp-l1g2, llama.cpp-enxh, llama.cpp-x12w, llama.cpp-lmhj, llama.cpp-gznx) BEFORE migrating call sites, since every migrated site consumes that API. Then the call-site migrations (llama.cpp-32dg8.6, llama.cpp-3h5gm.3, llama.cpp-mubmt.9, llama.cpp-doxi, llama.cpp-480a). llama.cpp-mubmt.13 (zero-alloc regression gate) closes the epic and must be registered before any default flips.
### Members

  - llama.cpp-32dg8
  - llama.cpp-32dg8.4
  - llama.cpp-32dg8.6
  - llama.cpp-z2316
  - llama.cpp-3h5gm.3
  - llama.cpp-3h5gm.7
  - llama.cpp-480a
  - llama.cpp-9lkgd
  - llama.cpp-aix
  - llama.cpp-c4kp
  - llama.cpp-mpin
  - llama.cpp-mubmt
  - llama.cpp-mubmt.12
  - llama.cpp-mubmt.13
  - llama.cpp-mubmt.9
  - llama.cpp-7phf
  - llama.cpp-doxi
  - llama.cpp-enxh
  - llama.cpp-gznx
  - llama.cpp-l1g2
  - llama.cpp-lmhj
  - llama.cpp-p0r7
  - llama.cpp-x12w
  - llama.cpp-4csx
  - llama.cpp-fjtw
  - llama.cpp-owkv
  - llama.cpp-vzjg

### Within epic duplicates

  - Item 1:
    - llama.cpp-c4kp
    - llama.cpp-mpin

  - Item 2:
    - llama.cpp-9lkgd
    - llama.cpp-mubmt.12

### Zero raw-pointer escapes — mem_handle is the only ownership token

  - **Slug:** handles-only-no-pointer-escapes
  - **Existing epic id:** llama.cpp-rg2ft
  - **Title:** Zero raw-pointer escapes — mem_handle is the only ownership token
  - **Goal:** No production path may store a raw device pointer as routing, ownership, cache-key or lifetime state; raw addresses exist only as transient views resolved for an immediate submission. Finish the data_device[] phase-out (phases 1-5), rekey pointer tables and dispatch caches onto the handle's stable identity hash, retire mem_handle::from_direct / from_chunk_ptr bridge minting on production paths, and close the lease-lifetime defects (stale expert pointers, cross-context lease frees, ownerless leases, tied-weight N-way redirects, multi-model lookup resolving only the published plan's model).
  - **Bar:** common.hpp no longer declares data_device[] / set_data_device() / data_device_ptr(); a census of mem_handle::from_direct/from_chunk_ptr in ggml-sycl.cpp/mmvq.cpp/unified-cache.cpp is zero or fully classified as ABI-transient with a test asserting the count; every pointer table and dispatch cache keys on the handle identity hash; a GGML_SYCL_STRICT_LEASES=1 run of the Mistral gate, the GPT-OSS gate (-c 4096) and a two-model load reports zero ownerless leased entries and zero aborts; the post-benchmark cleanup segfault (llama.cpp-npns9) and the test-sycl-moe-handle-resolution segfault (llama.cpp-r969f) are fixed, not skipped; ctest -R 'mem-handle|handle-resolution|pointer' green; no tg128 regression beyond the B70 10% spread.
  - **Constraints:** Ruling 2 (mem_handle is the sole ownership/lifetime token; pointer tables derive from handle identity; handles are retained for the lifetime of queued work or an executable graph). Ruling 3 (fix the missing release — never add a forced eviction/reap/zone reset to reclaim memory that still has a live handle; a live lease is a defect only when its owner is gone, and several models may be loaded at once). Ruling 1 (releases go back through the unified cache).
  - **Ordering hint:** llama.cpp-32dg8.15.1 (model- vs context-scoped ownership) and llama.cpp-32dg8.15.3 (in-flight lease lifetime through SYCL events) are the prerequisites — they define which lease is legitimate, and the reclaim/segfault tickets are unjudgeable without them. Then run the data_device phases strictly in order: llama.cpp-vxx0b (1) -> llama.cpp-rb988 (2) -> llama.cpp-q8ktt (3) -> llama.cpp-nhluo (4) -> llama.cpp-vb4q6 (5), with llama.cpp-oukgv as their umbrella. llama.cpp-mg359/llama.cpp-ialfa (duplicate pair — keep one) is the routing-state conversion that phase 3 depends on. The lifetime bugs (llama.cpp-xizj, llama.cpp-mhyw, llama.cpp-s83n, llama.cpp-goegc.1, llama.cpp-r969f, llama.cpp-npns9) can proceed in parallel once the ownership model is pinned.
### Members

  - llama.cpp-32dg8.15.1
  - llama.cpp-32dg8.15.3
  - llama.cpp-32dg8.7
  - llama.cpp-goegc.1
  - llama.cpp-mg359
  - llama.cpp-ialfa
  - llama.cpp-npns9
  - llama.cpp-rg2ft
  - llama.cpp-xizj
  - llama.cpp-fqev
  - llama.cpp-hakk
  - llama.cpp-mhyw
  - llama.cpp-nhluo
  - llama.cpp-orja
  - llama.cpp-oukgv
  - llama.cpp-q8ktt
  - llama.cpp-r969f
  - llama.cpp-rb988
  - llama.cpp-s83n
  - llama.cpp-vb4q6
  - llama.cpp-vxx0b
  - llama.cpp-x7z7
  - llama.cpp-41bs
  - llama.cpp-d5p1
  - llama.cpp-ivg0
  - llama.cpp-kz6w
  - llama.cpp-qq19
  - llama.cpp-upqb

### Within epic duplicates

  - Item 1:
    - llama.cpp-mg359
    - llama.cpp-ialfa

### Reclaim by refcount — finish eliminating zone reset and drain

  - **Slug:** refcount-reclaim-no-zone-resets
  - **Existing epic id:** NEW
  - **Title:** Reclaim by refcount — finish eliminating zone reset and drain
  - **Goal:** Finish the llama.cpp-iiff programme: remove the remaining reset/drain-based reclamation paths, give every transient zone epoch-refcounted ownership, and fix the reclaim predicates that either free too much (discarded drain timeouts, another live model's leases, -ot-forced DEVICE weights under weights_evictable) or re-arm state they should not (has_evictions_ set false while entries_preserved > 0). Eviction and demotion are driven by owner liveness, never by a global reset.
  - **Bar:** No caller of zone_reset/host_zone_reset/arena_reset remains on an inference or model-load path (grep-backed gate registered in tests/); drain_retained_handles()'s timeout is handled at every call site; the surviving-owner predicate exists once, not in two independent copies; a two-model residency test proves a second model's leases survive the first model's teardown; forced whole-layer host demotion at GGML_SYCL_VRAM_BUDGET_PCT=30 emits correct Mistral gate tokens on the B50; the WEIGHT-zone largest-free-block regression (llama.cpp-aog4, ~766 MB vs the pre-iiff HEAD) is explained or closed.
  - **Constraints:** Ruling 3 (zone resets are being ELIMINATED — Option-C epoch-refcounted transient zones is the shipped direction; a task that ADDS a reset/drain step is against design). Ruling 2 (no forced eviction/reap while a handle is live; a live lease is a defect only when its owner is gone, and MID_LOAD_REPLAN suppresses ownerless classification). Ruling 6 (a drain is not an acceptable substitute for an event dependency).
  - **Ordering hint:** llama.cpp-cadd (extract the single surviving-owner predicate) is the prerequisite for the reclaim-predicate fixes: llama.cpp-3ik3/llama.cpp-fy7n (duplicate pair — keep one), llama.cpp-nx1f, llama.cpp-1z58, llama.cpp-9m58 all edit the same decision. llama.cpp-cpsi (drain timeouts discarded at four call sites) and llama.cpp-2498 (non-arena-active fallback) must land before zone_reset itself is removed; llama.cpp-acsq and llama.cpp-ufqw are the deletion tail. llama.cpp-aog4 is a measurement to be taken after the reclaim path settles, not before.
### Members

  - llama.cpp-9m58
  - llama.cpp-nx1f
  - llama.cpp-1z58
  - llama.cpp-2498
  - llama.cpp-34hr
  - llama.cpp-3ik3
  - llama.cpp-acsq
  - llama.cpp-aog4
  - llama.cpp-cadd
  - llama.cpp-cpsi
  - llama.cpp-fy7n
  - llama.cpp-ufqw

### Within epic duplicates

  - Item 1:
    - llama.cpp-3ik3
    - llama.cpp-fy7n

### One VRAM ledger — arena zones are the only budget authority

  - **Slug:** budget-authority-one-ledger
  - **Existing epic id:** llama.cpp-792vn.5
  - **Title:** One VRAM ledger — arena zones are the only budget authority
  - **Goal:** Arena zones are the single source of truth for what is reserved and what is free: no shadow counters, no duplicate reserve literals, no untracked host-arena bytes, no TOCTOU window between a budget check and the allocation it authorises. This epic also fixes the sizing inputs — real compute-buffer demand queried from the ggml scheduler, FA on/off, oneDNN scratchpads sharing the ONEDNN zone — so the plan and the actual footprint agree, and makes per-device budgets independent and integrated-GPU-safe.
  - **Bar:** A plan-vs-actual audit at the end of a GPT-OSS 20B context (FA on and off) and a Mistral context shows zone reserved == sum of live allocations for WEIGHT/RUNTIME/KV/ONEDNN on both cards, with zero RUNTIME-zone overflows; arena used_ decrements on eviction; no allocation succeeds past the budget under a concurrent-admission stress canary; a GGML_SYCL_VRAM_BUDGET_PCT sweep {100,50,30} on B50 Mistral and GPT-OSS 20B loads without overcommit refusals and passes the correctness gates; with ONEAPI_DEVICE_SELECTOR unset (iGPU visible) host-unified memory is never budgeted as VRAM — peak Shmem stays below 30 GB on a model load.
  - **Constraints:** Ruling 10 (VRAM budget = min(total*pct, free_at_init); the Arrow Lake-S iGPU reports 231.7 GB of host-unified memory and must not be budgeted as VRAM; device selection stays with ONEAPI_DEVICE_SELECTOR and no code parses or refuses it). Ruling 1 (the cache owns the ledger). Ruling 7 (a budget shortfall is answered by placement, never by shrinking n_ctx). Ruling 3 (no reset-to-reclaim to make the numbers fit).
  - **Ordering hint:** llama.cpp-qq5p (budget contamination sources) and llama.cpp-8gz7y (query the scheduler for real compute-buffer sizes) are prerequisites — every other sizing fix is measured against them. llama.cpp-5ksb6 (arena sizing consumes weight_plan) then makes the ledger authoritative; llama.cpp-678i, llama.cpp-goegc.2/.3/.6/.8/.9 and llama.cpp-ulxr are the accounting defects that follow. llama.cpp-qm6gf (delete the orphan runtime-update path) should land before llama.cpp-kprrh's concurrency canary, or the canary tests a path that is being deleted. llama.cpp-u3usl is the GPT-OSS B50 FA-off symptom that should be re-measured last, as a check on the whole ledger.
### Members

  - llama.cpp-qq5p
  - llama.cpp-3h5gm.1
  - llama.cpp-5ksb6
  - llama.cpp-792vn.5
  - llama.cpp-8gz7y
  - llama.cpp-goegc.2
  - llama.cpp-goegc.3
  - llama.cpp-goegc.6
  - llama.cpp-kprrh
  - llama.cpp-qm6gf
  - llama.cpp-678i
  - llama.cpp-goegc.8
  - llama.cpp-goegc.9
  - llama.cpp-u3usl
  - llama.cpp-ulxr

### Within epic duplicates

  - _(empty)_

### The planner plans every byte — weights, KV, compute, scratch

  - **Slug:** planner-plans-every-byte
  - **Existing epic id:** llama.cpp-3h5gm
  - **Title:** The planner plans every byte — weights, KV, compute, scratch
  - **Goal:** One placement pass, run before anything is allocated, decides where every byte lives: weights by perf tier, KV, compute buffers, graph temporaries, oneDNN scratch. Nothing is placed by a runtime heuristic, a shadow weight-plan, an -ngl guess, or an orphan runtime-update path afterwards. This epic owns the planner's domain coverage, its priority policy and escape hatches, FA as a planned model property, and the plan-dump / plan-vs-actual diagnostics that make a placement claim checkable before a run.
  - **Bar:** GGML_SYCL_PLAN_DUMP=1 at context creation prints a complete plan whose domains sum to the auditor's measured allocation total within 1% for Mistral 7B, GPT-OSS 20B and GPT-OSS 120B at GGML_SYCL_VRAM_BUDGET_PCT in {100,50,30}, with no allocation performed outside a planned domain; the weight-plan shadow heuristics and FORCE_STREAMING are deleted and -ngl no longer decides placement; a multi-device plan is proved to see the scheduler's transfer edges; the Mistral completion gate and the GPT-OSS chat gate (-c 4096) are green at each budget, and -ngl 0 produces correct Mistral gate output.
  - **Constraints:** Ruling 4 (PLACEMENT DECIDES THE EXECUTOR — the planner places, the executor follows; no per-dispatch weight streaming and no per-token prefetch-to-VRAM). Ruling 5 (the plan names the layout, which is then materialized once, for the processor that executes it). Ruling 1 (planned destinations are unified-cache storage). Ruling 7 (KV is placed, never shrunk). Ruling 10 for the budget inputs the plan consumes.
  - **Ordering hint:** llama.cpp-32dg8.5 (planner covers all SYCL memory domains) is the structural prerequisite; llama.cpp-r5xv7 (GGML_SYCL_PLAN_DUMP) and llama.cpp-32dg8.15.16 (plan-vs-actual auditor) must land next because every remaining bar in this epic is stated in terms of their output. Then delete the competing deciders: llama.cpp-g9yex (shadow heuristics + FORCE_STREAMING), llama.cpp-6pm2 (-ngl), llama.cpp-2kmf and llama.cpp-np70. llama.cpp-p8ic3 (perf-tier priority policy) and its PLACE-* children (llama.cpp-xd14f, llama.cpp-jjlht, llama.cpp-vmhri) sit on top of a complete plan, not beside it. llama.cpp-3h5gm.2/.4 and llama.cpp-dm4.5 are the proof harnesses and land with the diagnostics.
### Members

  - llama.cpp-32dg8.15.16
  - llama.cpp-32dg8.5
  - llama.cpp-3h5gm
  - llama.cpp-3h5gm.2
  - llama.cpp-bkvc9
  - llama.cpp-dm4.5
  - llama.cpp-g9yex
  - llama.cpp-p8ic3
  - llama.cpp-r5xv7
  - llama.cpp-tzg5w
  - llama.cpp-xd14f
  - llama.cpp-2kmf
  - llama.cpp-3h5gm.4
  - llama.cpp-6pm2
  - llama.cpp-d3ls
  - llama.cpp-jjlht
  - llama.cpp-vmhri
  - llama.cpp-np70

### Within epic duplicates

  - _(empty)_

### Model-max context always initializes — tiered KV placement, never a refusal

  - **Slug:** never-shrink-context-tiered-kv
  - **Existing epic id:** llama.cpp-sk1xz
  - **Title:** Model-max context always initializes — tiered KV placement, never a refusal
  - **Goal:** Default n_ctx is the model maximum, and a KV cache that does not fit VRAM is placed into pinned host tiers rather than refused or silently shrunk. Finish mixed device/host KV slice execution and its gates, non-contiguous placement, the host KV arena's fragmentation and capacity-accounting defects, the pinned chunk pool's 2 GB contiguity limit, and then remove the -c workarounds and the misleading refusal paths that remain in llama-cli/llama-server.
  - **Bar:** llama-cli and llama-completion on GPT-OSS 20B with NO -c flag reach generation on B50 and B70 — today the default 131072 path refuses with '[SYCL-PLAN] runtime KV update rejected' — producing the GPT-OSS chat gate digit line with rc=0 and no segfault at cleanup; Mistral 7B at model-max context passes the completion gate; a mixed device/host KV run matches an all-device run token-for-token at -c 4096; no allocation is rejected for the 2 GB chunk limit at model-max context; tg128 on both cards stays within baseline spread; ctest -R 'kv-slice|tiered-kv|kv-tier' green.
  - **Constraints:** Ruling 7 (NEVER SHRINK CONTEXT — the fix is placement; never auto-shrink, never refuse init; explicit -c is the only way to get less). Ruling 1 (KV allocations are unified-cache allocations). Ruling 4 (host-tier KV means the ops touching it run where that data is). Ruling 2 (KV is handle-owned; no raw KV pointers in slice tables).
  - **Ordering hint:** llama.cpp-uize (the standing default-context refusal) is the epic's headline defect and its bar; llama.cpp-vlc46 (host arena KV fragmentation + TLSF capacity overcount) and llama.cpp-icgo (2 GB chunk limit) are its prerequisites — the refusal cannot be removed until a model-max KV can actually be filled. llama.cpp-kw0hv (mixed-placement execution + regression gates) then makes host-tier KV correct, with llama.cpp-fzukj and llama.cpp-si5wq as its tests. llama.cpp-9343 (LEAD-ONLY default-ctx gate + A/B) is the acceptance run, and llama.cpp-s6hx (docs + workaround removal, including the -c 4096 pin in CLAUDE.md) closes the epic last.
### Members

  - llama.cpp-kw0hv
  - llama.cpp-sk1xz
  - llama.cpp-9343
  - llama.cpp-uize
  - llama.cpp-vlc46
  - llama.cpp-fzukj
  - llama.cpp-icgo
  - llama.cpp-s6hx
  - llama.cpp-si5wq
  - llama.cpp-2o76
  - llama.cpp-rwy9

### Within epic duplicates

  - _(empty)_

### Ordered by events, replayed as a graph — no host waits in dispatch

  - **Slug:** event-ordered-graph-replay
  - **Existing epic id:** NEW
  - **Title:** Ordered by events, replayed as a graph — no host waits in dispatch
  - **Goal:** Ordering in the dispatch path is expressed as SYCL event dependencies, and SYCL graph record/replay is the steady-state path for both decode and prompt. Remove the nodes that are not graph-recordable (raw host staging in binbcast, memcpy nodes, raw thread_local recording-flag checks, ptr-table refusals that disable replay run-wide), and cut submission cost — init kernel spam, launch-argument batching, redundant stream waits, blocking event_complete on profiling queues, unbounded evict_and_flush queue_wait.
  - **Bar:** Zero wait()/wait_and_throw()/queue-drain calls remain in dispatch-path functions (teardown, error, debug and init excepted), verified by a source-contract gate with a positive control; a -v GPT-OSS 20B and a Mistral decode run report exec_graph>0 with replays on >=95% of decode steps and zero replay-disabled warnings; -ngl 0 no longer aborts; kernel-launch and init-submission counts per decode step drop measurably with no tg128 regression against docs/backend/sycl-perf-baselines.md.
  - **Constraints:** Ruling 6 (NO HOST WAITS — a task proposing a wait as the fix is against design; removing one in favour of depends_on / oneDNN sycl_interop deps is aligned; waits belong only in teardown/error/debug/init). Ruling 2 (graph-retained handles are leases — a per-context path may not release another context's, and handles must outlive the executable graph). Ruling 3 (do not add a periodic drain as a mechanism). Ruling 9 (both canonical gates green before any policy default flips).
  - **Ordering hint:** llama.cpp-saka (one ptr-table refusal disables replay run-wide) and llama.cpp-fvcx / llama.cpp-38af (binbcast's raw recording-flag check and non-recordable host staging, which abort -ngl 0) are the prerequisites: until they land, replay coverage cannot be measured, so llama.cpp-tott's exec_graph=0 evidence and every submission-overhead ticket are unmeasurable. llama.cpp-lw28z (graph-safe copy/fill) removes the memcpy nodes those fixes expose. Only then the overhead work — llama.cpp-7mala, llama.cpp-flq2n, llama.cpp-54sf0, llama.cpp-mubmt.10, llama.cpp-z8hr, llama.cpp-2o6q, llama.cpp-goegc.7 — and finally llama.cpp-v90xn.20 (handle-keyed graph cache) and llama.cpp-gxofa (is the periodic drain needed at all).
### Members

  - llama.cpp-v90xn.20
  - llama.cpp-38af
  - llama.cpp-7mala
  - llama.cpp-dm4.4
  - llama.cpp-fvcx
  - llama.cpp-gxofa
  - llama.cpp-lw28z
  - llama.cpp-mubmt.10
  - llama.cpp-saka
  - llama.cpp-z8hr
  - llama.cpp-2o6q
  - llama.cpp-54sf0
  - llama.cpp-flq2n
  - llama.cpp-goegc.7
  - llama.cpp-4j9a
  - llama.cpp-cobu
  - llama.cpp-tott

### Within epic duplicates

  - _(empty)_

### GPT-OSS decode at 80% of the memory-bandwidth roofline

  - **Slug:** gptoss-decode-bandwidth
  - **Existing epic id:** llama.cpp-30ak7
  - **Title:** GPT-OSS decode at 80% of the memory-bandwidth roofline
  - **Goal:** Single-stream token generation for GPT-OSS 20B (and the 120B path once it loads) must be limited by DRAM bandwidth, not by kernel class, launch overhead, or layout shifts. Every decode dispatch reads each needed weight byte once, from the layout the planner already materialized, over a graph that replays. This epic owns the MMVQ/XMX decode routers, the device-ID decode route lost in the b10630 merge, phase-owned XMX-tiled decode, grouped/batched decode formation, persistent-TG, and the PERF-EPIC B/E recovery tracks.
  - **Bar:** Instrument bytes-streamed-per-token, then require achieved bytes/token x tok/s >= 0.8 x 224 GB/s on B50 (level_zero:1) and >= 0.8 x 608 GB/s on B70 (level_zero:0) for GPT-OSS 20B MXFP4 tg128, measured by interleaved paired A/B. Hard no-regression floors while that is chased: GPT-OSS 20B tg128 >= 32 (B50) / >= 44 (B70), Mistral Q4_0 tg128 >= 47 / >= 108 (docs/backend/sycl-perf-baselines.md). Every default flip needs the Mistral completion gate and the GPT-OSS chat gate (-c 4096) green in the same build. No fixed tok/s anchor — the epic's legacy '~70 tok/s' and '>=50 tg' targets are retired.
  - **Constraints:** Ruling 5 (one layout per weight; decode consumers read the materialized layout, no dispatch-time shift). Ruling 6 (no host waits — ordering via SYCL events). Ruling 8 (fuse dequant into the matmul; ESIMD is not the lever for small-block dequant). Ruling 9 (correctness before throughput — GGML_SYCL_MOE_BLOCK_GRAPHLETS / XMX_MOE_PP / PP_PIPELINE stay opt-in until the GPT-OSS gate passes). Ruling 4 (device-resident experts execute on that GPU; no per-token prefetch to VRAM).
  - **Ordering hint:** llama.cpp-unpj (the deleted device-ID MoE decode route, which currently forces host-ID D2H) is the first prerequisite — every decode measurement below it is taken on a degraded route. llama.cpp-hmbk9 (persistent-TG core decode correctness) gates llama.cpp-nubvg and llama.cpp-m3j9q, which in turn gate llama.cpp-32dg8.16.2 / llama.cpp-0np6 (duplicate pair — the default-flip decision). The XMX-tiled phase work runs llama.cpp-30ak7.19.10 -> llama.cpp-5ctzf -> llama.cpp-30ak7.19.17, with llama.cpp-d4yko and llama.cpp-v90xn.31 (grouped/batched formation) after them. PERF-EPIC ordering is fixed by its own track: llama.cpp-zb27 -> llama.cpp-cv8w (flip gate) -> llama.cpp-rty8 -> llama.cpp-22sp (close). llama.cpp-d0bp is the standing regression that any claimed win must first explain.
### Members

  - llama.cpp-30ak7
  - llama.cpp-30ak7.19
  - llama.cpp-30ak7.19.10
  - llama.cpp-30ak7.19.14
  - llama.cpp-30ak7.19.17
  - llama.cpp-32dg8.16.2
  - llama.cpp-5ctzf
  - llama.cpp-d4yko
  - llama.cpp-hmbk9
  - llama.cpp-m3j9q
  - llama.cpp-nubvg
  - llama.cpp-v90xn.30
  - llama.cpp-0scz
  - llama.cpp-22sp
  - llama.cpp-cv8w
  - llama.cpp-d0bp
  - llama.cpp-e2kgu
  - llama.cpp-qz3yb
  - llama.cpp-rty8
  - llama.cpp-unpj
  - llama.cpp-v90xn.30.7
  - llama.cpp-v90xn.31
  - llama.cpp-0np6
  - llama.cpp-4wkt
  - llama.cpp-eju9
  - llama.cpp-ndzz1
  - llama.cpp-zb27
  - llama.cpp-zw4b
  - llama.cpp-y0it

### Within epic duplicates

  - Item 1:
    - llama.cpp-32dg8.16.2
    - llama.cpp-0np6

### Prefill at INT8 XMX peak — one GEMM-shaped route per role

  - **Slug:** prefill-xmx-int8-peak
  - **Existing epic id:** llama.cpp-xihy
  - **Title:** Prefill at INT8 XMX peak — one GEMM-shaped route per role
  - **Goal:** Prompt processing must run as real GEMMs on the XMX units, not as entry-batched GEMV. The gate/up MMID path still amplifies expert-weight traffic ~64x at pp512 (~413 GB of ~442 GB moved per pass against a ~9.7 GB ideal); the DOWN role already has a grouped-DPAS route to extend. This epic consolidates the XMX/ESIMD kernel families behind one capability-derived route per (role, layout), and owns the oneDNN PP path, the WOQ/packed-key plumbing, and the flash-attention front end that shares those units.
  - **Bar:** pp512 sustained INT8 utilization >= 80% of hardware peak, computed from the 2*M*N*K of the GEMMs actually issued: >= 136 TOPS on B50 (170 peak) and >= 294 TOPS on B70 (367 peak). Expert-weight bytes moved per pp512 pass for gate/up fall from ~413 GB to <= ~10 GB, proved by the [MOE-ROW-AGG] instrumentation. Floors that must never regress meanwhile: GPT-OSS 20B pp512 >= 894 (B50) / >= 1415 (B70), Mistral Q4_0 pp512 >= 1188 / >= 2495. GGML_SYCL_FA_ONEDNN stays ON (=0 costs ~39% pp) and the Mistral gate is green with it on; test-unified-kernel and the XMX-guarded tests pass positive-device correctness on both cards; no reachable abort from the fattn Q-type/config paths across supported CLI flag combinations.
  - **Constraints:** Ruling 5 (routes advertise only (type, layout) pairs whose kernels exist; no dispatch-time layout shift; AOS fallback is a stopgap). Ruling 8 (oneDNN SDPA is ON by default and correct for Mistral GQA — GGML_SYCL_FA_ONEDNN_ALLOW does not exist; small-block dequant stays on standard SYCL and the lever is fusing dequant into the matmul). Ruling 9 (GGML_SYCL_XMX_MOE_PP and GGML_SYCL_PP_PIPELINE stay opt-in until the GPT-OSS chat gate is green). Ruling 6 (event ordering, no per-op drains). Ruling 1 (packed layouts and oneDNN scratch allocate inside the unified cache).
  - **Ordering hint:** llama.cpp-twl6 / llama.cpp-613w (duplicate pair — keep one) is the epic's load-bearing item: the GEMM-shaped grouped PP route for gate/up. llama.cpp-sk67 extends the same route to DOWN and should follow it, not precede it. llama.cpp-1y3x (XMX-guarded tests fail positive-device correctness) must be fixed FIRST — until it is, no route change here has a trustworthy correctness signal. llama.cpp-xihy (kernel consolidation) and llama.cpp-98w0 (PP consumers use the route table) are the structural cleanup the two routes land into; llama.cpp-49pp is the final re-arm sweep and acceptance gate. The FA items (llama.cpp-ukwqh, llama.cpp-x1e6, llama.cpp-rtf1, llama.cpp-cldt, llama.cpp-v90xn.32) are an independent sub-track; llama.cpp-9klr and llama.cpp-qppk are hazards to resolve before any tiled/TM=16 default changes.
### Members

  - llama.cpp-49pp
  - llama.cpp-613w
  - llama.cpp-twl6
  - llama.cpp-xihy
  - llama.cpp-1y3x
  - llama.cpp-5ylh
  - llama.cpp-98w0
  - llama.cpp-sk67
  - llama.cpp-ukwqh
  - llama.cpp-9klr
  - llama.cpp-beth
  - llama.cpp-cldt
  - llama.cpp-e6c5
  - llama.cpp-qppk
  - llama.cpp-rtf1
  - llama.cpp-v90xn.32
  - llama.cpp-4sxu
  - llama.cpp-a7yg
  - llama.cpp-ajg9
  - llama.cpp-pdit
  - llama.cpp-x1e6
  - llama.cpp-xy26
  - llama.cpp-ur1

### Within epic duplicates

  - Item 1:
    - llama.cpp-twl6
    - llama.cpp-613w

### One layout per weight, and routes that only advertise kernels that exist

  - **Slug:** one-layout-honest-routes
  - **Existing epic id:** NEW
  - **Title:** One layout per weight, and routes that only advertise kernels that exist
  - **Goal:** Each weight is materialized exactly once, in the layout optimal for the processor that executes it; every consumer (PP grouped and TG small-batch, gate/up and down) reads that layout; and supports_op advertises only (type, layout) pairs a real kernel covers. This epic also carries the new-type recipe program (Q1/NVFP4 device decode, MMID f16/f32/bf16 and iq*, sliced-coalesced scale addressing) whose whole point is closing advertisement gaps rather than adding a second layout, plus the reconciliation between what the planner advertises and what the cache actually materialized.
  - **Bar:** A layout census at HEAD shows zero weights materialized in two layouts and zero dispatch-time layout shifts for GPT-OSS 20B on both cards; every route a planner advertises resolves to a compiled kernel, proved by a matrix test over (type, layout, role, ne12) that fails closed on an unadvertised pair; test-llama-archs reports no uncovered-type MUL_MAT_ID failures (today 234 via llama.cpp-yitq); the Q1/NVFP4 production route validates on the B70 (llama.cpp-tjk4) with the Mistral completion gate and the GPT-OSS chat gate green; test-sycl-layout-choice and the MMVQ layout-reconciliation gate are green and consistent with one-layout-per-weight.
  - **Constraints:** Ruling 5 (LAYOUT FOLLOWS RESIDENCY / one layout per weight / all consumers support it / no dispatch-time shifting / AOS fallback is a stopgap, not the design). Ruling 8 (small-block dequant stays on standard SYCL; fuse it into the matmul). Ruling 2 (layout identity travels on the handle, never on a raw pointer or an address key). Ruling 1 (layout conversion allocates inside the unified cache only).
  - **Ordering hint:** llama.cpp-yitq (supports_op claims MUL_MAT_ID unconditionally) is the prerequisite for the whole epic: it defines what 'advertised' means and produces the 234-failure baseline the bar is written against. llama.cpp-avhx (the planner advertises SOA from a probe that never checks what the cache materialized) and llama.cpp-8hz8 (planner-owned representation identity) close the advertise/materialize seam next; llama.cpp-34g5, llama.cpp-l8at and llama.cpp-71hx are the fallback waivers to remove once that seam holds. The Q1/NVFP4 sub-track has its own order: llama.cpp-omp4 (umbrella) -> llama.cpp-sgox -> llama.cpp-zqoe -> llama.cpp-5i7z / llama.cpp-7vpd -> llama.cpp-tjk4 (B70 validation). llama.cpp-h690 and llama.cpp-o54r (P6 router restoration) must land before the recipe gap tickets llama.cpp-0yi9 / llama.cpp-wh7o / llama.cpp-a6sy are judged.
### Members

  - llama.cpp-cyn.2
  - llama.cpp-h690
  - llama.cpp-o54r
  - llama.cpp-sgox
  - llama.cpp-tjk4
  - llama.cpp-zqoe
  - llama.cpp-0yi9
  - llama.cpp-5i7z
  - llama.cpp-7vpd
  - llama.cpp-8hz8
  - llama.cpp-a6sy
  - llama.cpp-n36
  - llama.cpp-omp4
  - llama.cpp-wh7o
  - llama.cpp-yitq
  - llama.cpp-71hx
  - llama.cpp-avhx
  - llama.cpp-f9fx
  - llama.cpp-kpv8
  - llama.cpp-l8at
  - llama.cpp-s2cw
  - llama.cpp-tqka
  - llama.cpp-19l2
  - llama.cpp-34g5
  - llama.cpp-95x2
  - llama.cpp-g4i8
  - llama.cpp-pwb2
  - llama.cpp-6sp4

### Within epic duplicates

  - _(empty)_

### Every op runs where its data lives — MoE residency and the CPU expert pool

  - **Slug:** experts-run-where-they-live
  - **Existing epic id:** llama.cpp-ic4f
  - **Title:** Every op runs where its data lives — MoE residency and the CPU expert pool
  - **Goal:** Placement decides the executor: device-resident experts run on that GPU, host-pinned experts run on the CPU through the live CpuExpertPool in a CPU-optimal (row-interleaved) layout, and nothing streams host weights into device scratch per dispatch or reads host memory from the GPU. CPU expert work overlaps the next layer's GPU attention through event dependencies. This epic also owns the MoE residency bookkeeping — owner-scoped prestage state, stale meta/groups registries, silent cache-key failures, pointer-table demotion — and the removal of the prefetch/prestage scaffolding that contradicts the ruling.
  - **Bar:** On B50 GPT-OSS 20B at GGML_SYCL_VRAM_BUDGET_PCT in {100,50,30} the GPT-OSS chat gate (-c 4096) and the Mistral completion gate stay green at every budget, with zero occurrences of weight streaming or per-token VRAM prestage in a -v run; forced whole-layer host demotion no longer emits garbage tokens; mixed device/host MoE tg128 beats the all-host CPU path and loses <= 20% against the all-device path at 100%; host-expert compute is shown hidden behind the next layer's GPU attention in a device-event timeline (not host chrono); GGML_SYCL_PIPELINE_CPU flips to default ON only after an interleaved paired A/B shows >= 10% tg128 gain at a budget that forces host-resident experts.
  - **Constraints:** Ruling 4 (PLACEMENT DECIDES THE EXECUTOR — GPU zero-copy reads of host memory and per-dispatch weight streaming are FORBIDDEN; measured CPU AOS 18-30 GB/s vs GPU zero-copy 11.3 GB/s). Ruling 6 (overlap is expressed as depends_on, not waits — llama.cpp-bwg8's 'explicit waits' formulation must be rewritten before it is worked). Ruling 5 (host-resident experts get the CPU-optimal layout, device-resident the device-optimal one, each materialized once). Ruling 2 (expert pointer tables key on handle identity).
  - **Ordering hint:** llama.cpp-azll (planned expert residency detection — MXFP4 materialization ignores the planned layout) and llama.cpp-po3nd.2.35 (planned host-resident GPT-OSS MoE execution correctness) are the prerequisites: the overlap and layout work below them is meaningless while residency detection is wrong. llama.cpp-onai (row-interleaved CPU layout) precedes llama.cpp-3oju9 / llama.cpp-cii8 (duplicate pair — the CPU/GPU overlap), which precede llama.cpp-xyqw and llama.cpp-wjth (the combined benchmarks). llama.cpp-bwg8 must be REWRITTEN to an event-based formulation before it is scheduled at all. The registry cleanups (llama.cpp-017kb.5, llama.cpp-017kb.6, llama.cpp-izkw, llama.cpp-swf4, llama.cpp-jzeo, llama.cpp-0ywi) are independent and can run in parallel.
### Members

  - llama.cpp-32dg8.15.17
  - llama.cpp-azll
  - llama.cpp-bwg8
  - llama.cpp-po3nd.2.35
  - llama.cpp-xyqw
  - llama.cpp-017kb
  - llama.cpp-017kb.8
  - llama.cpp-3oju9
  - llama.cpp-61hnu
  - llama.cpp-cii8
  - llama.cpp-goegc.5
  - llama.cpp-ic4f
  - llama.cpp-onai
  - llama.cpp-rw72x
  - llama.cpp-wjth
  - llama.cpp-017kb.5
  - llama.cpp-017kb.6
  - llama.cpp-0ywi
  - llama.cpp-4d0u
  - llama.cpp-4pwk
  - llama.cpp-60xr
  - llama.cpp-izkw
  - llama.cpp-jzeo
  - llama.cpp-swf4
  - llama.cpp-va1g
  - llama.cpp-61qk
  - llama.cpp-lurc

### Within epic duplicates

  - Item 1:
    - llama.cpp-3oju9
    - llama.cpp-cii8

### The second card earns its keep — dual-device PP and decode over a host bounce

  - **Slug:** second-card-earns-its-keep
  - **Existing epic id:** llama.cpp-po3nd.2
  - **Title:** The second card earns its keep — dual-device PP and decode over a host bounce
  - **Goal:** B70+B50 together must beat the best single card on GPT-OSS prompt processing and decode, with the planner owning contiguous layer-block placement under a fastest-dense policy and the executor pipelining across blocks. There is no PCIe P2P between the two cards, so every design here is a host-bounce design with an explicit replication/transfer cost model. This epic also owns per-device budget isolation, cross-device KV and activation routing, the event-chained secondary stage DAG, and the multi-GPU regression matrix that must pass before any default flips.
  - **Bar:** Same-build interleaved paired A/B across B70-only (level_zero:0), B50-only (level_zero:1) and level_zero:0,1: dual-device GPT-OSS 20B pp512 > max(1415, 894) and tg128 > max(44, 32) by a margin larger than run noise, with the Mistral single-device guard unregressed and both canonical gates green in the dual configuration; a committed dual-GPU section in docs/backend/sycl-perf-baselines.md holds the numbers; no code path assumes peer DMA; GGML_SYCL_MOE_MULTI_GPU stays opt-in until the above passes. 120B multi-GPU TG is re-baselined on B70+B50.
  - **Constraints:** No PCIe P2P between 0000:03:00.0 and 0000:07:00.0 — different CPU root ports, can_access_peer false both directions, and ext_oneapi_enable_peer_access returning OK is NOT a capability check (it fails later as UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY). All cross-device traffic host-bounces. Ruling 4 (placement decides which card executes), ruling 5 (one layout per weight per device), ruling 6 (cross-device ordering by events, not waits), ruling 10 (per-device budgets; device selection stays with ONEAPI_DEVICE_SELECTOR). Every B580-era figure here is obsolete-hw and must be re-derived on B70+B50.
  - **Ordering hint:** llama.cpp-po3nd.2.45 (contiguous layer-block placement with a fastest-dense policy) is the prerequisite for llama.cpp-po3nd.2.46 (pipeline PP and batched decode across blocks) — those two carry the epic. Before either, llama.cpp-po3nd.2.11 (profile current dual-device route ownership) and llama.cpp-fawh (independent per-device budget isolation) must land, or placement decisions are made against unknown ownership and a shared budget. llama.cpp-bci0 / llama.cpp-won9 (planned secondary expert placement) precede llama.cpp-po3nd.2.26 / llama.cpp-po3nd.2.20 (duplicate pair — dual-device decode executor) and llama.cpp-po3nd.2.36 (event-chained stage DAG). llama.cpp-po3nd.2.28 / llama.cpp-po3nd.2.7 (duplicate pair — the regression matrix) is the gate that everything else reports into. llama.cpp-vudft / llama.cpp-w2zf and llama.cpp-xzeu / llama.cpp-d8m7 are duplicate pairs whose B580 targets must be restated on B70+B50 before work starts.
### Members

  - llama.cpp-03k1
  - llama.cpp-bci0
  - llama.cpp-beas
  - llama.cpp-po3nd.2
  - llama.cpp-po3nd.2.11
  - llama.cpp-po3nd.2.20
  - llama.cpp-po3nd.2.22
  - llama.cpp-po3nd.2.24
  - llama.cpp-po3nd.2.25
  - llama.cpp-po3nd.2.26
  - llama.cpp-po3nd.2.28
  - llama.cpp-po3nd.2.3
  - llama.cpp-po3nd.2.32
  - llama.cpp-po3nd.2.33
  - llama.cpp-po3nd.2.36
  - llama.cpp-po3nd.2.37
  - llama.cpp-po3nd.2.42
  - llama.cpp-po3nd.2.44
  - llama.cpp-po3nd.2.45
  - llama.cpp-po3nd.2.46
  - llama.cpp-po3nd.2.47
  - llama.cpp-xzeu
  - llama.cpp-d8m7
  - llama.cpp-kkxtv
  - llama.cpp-kkxtv.7
  - llama.cpp-l80a3
  - llama.cpp-mk4l
  - llama.cpp-po3nd.2.27
  - llama.cpp-po3nd.2.7
  - llama.cpp-vudft
  - llama.cpp-w2zf
  - llama.cpp-won9
  - llama.cpp-fawh
  - llama.cpp-t00wi
  - llama.cpp-dt06

### Within epic duplicates

  - Item 1:
    - llama.cpp-vudft
    - llama.cpp-w2zf

  - Item 2:
    - llama.cpp-xzeu
    - llama.cpp-d8m7

  - Item 3:
    - llama.cpp-po3nd.2.26
    - llama.cpp-po3nd.2.20

  - Item 4:
    - llama.cpp-po3nd.2.28
    - llama.cpp-po3nd.2.7

### 120B loads fast and answers coherently

  - **Slug:** big-model-load-and-120b
  - **Existing epic id:** llama.cpp-m0yo
  - **Title:** 120B loads fast and answers coherently
  - **Goal:** GPT-OSS 120B must load in a bounded time by materializing weights straight into their planned unified-cache destinations — no persistent host double-copy, no reorder-then-copy-then-copy — and must then produce coherent, baseline-matching text end to end. This epic owns model-load materialization speed (including the -ngl 0 mmap/CPU_REPACK loss), the 120B correctness oracle, 120B PP/TG re-baselining on current hardware, and the cross-model regression suite at constrained budgets.
  - **Bar:** 120B load wall time is measured and improved against a recorded pre-change number on the B70 with the iGPU excluded, with Shmem staying flat under a pinned selector and no host double-copy visible in a -v load; a 120B completion produces coherent text matching a recorded token-for-token baseline at --seed 42 --temp 0; the Mistral 7B + GPT-OSS 20B + GPT-OSS 120B suite passes at GGML_SYCL_VRAM_BUDGET_PCT in {100,50,30}; 120B PP/TG are re-baselined on B70 and B70+B50 and recorded in docs/backend/sycl-perf-baselines.md. The 8.22 tok/s CPU-era floor, the ~1.03 tok/s B580 PP baseline and every other B580 figure in these tickets are retired.
  - **Constraints:** Ruling 4 (no weight streaming and no per-token expert prestage — llama.cpp-d3lj's closed dependency implemented exactly the forbidden demand_load pattern, so that gate must be re-verified before it is trusted). Ruling 1 (materialize into planned unified-cache storage). Ruling 5 (materialize once, in the executing processor's layout). Ruling 2 (load-time canonical handles). Obsolete-hw: the B580 is gone; jgc6q's and uwcn's numbers do not carry over.
  - **Ordering hint:** llama.cpp-frdkp / llama.cpp-mubmt.7 (duplicate pair — direct materialization into planned destinations) is the prerequisite for llama.cpp-817t (eliminate the persistent host double-copy) and llama.cpp-7r7x (load speed); measuring load time before those land produces a number that will not survive. llama.cpp-4gy4 (end-to-end 120B correctness oracle) must exist before any load-path change is accepted, otherwise a faster load has no coherence check. llama.cpp-t5mot (the budget-sweep regression suite) is the closing gate. llama.cpp-d3lj / llama.cpp-asgj.5 (duplicate pair) and llama.cpp-jgc6q.5 are verification tickets whose acceptance numbers must be restated on B70 before they are run.
### Members

  - llama.cpp-7r7x
  - llama.cpp-817t
  - llama.cpp-asgj.5
  - llama.cpp-d3lj
  - llama.cpp-frdkp
  - llama.cpp-jgc6q
  - llama.cpp-jgc6q.5
  - llama.cpp-m0yo
  - llama.cpp-uwcn
  - llama.cpp-4gy4
  - llama.cpp-mubmt.7
  - llama.cpp-t5mot
  - llama.cpp-qptd

### Within epic duplicates

  - Item 1:
    - llama.cpp-frdkp
    - llama.cpp-mubmt.7

  - Item 2:
    - llama.cpp-d3lj
    - llama.cpp-asgj.5

### The two canonical gates stay green, on every path

  - **Slug:** canonical-gates-stay-green
  - **Existing epic id:** NEW
  - **Title:** The two canonical gates stay green, on every path
  - **Goal:** The Mistral completion gate and the GPT-OSS chat gate are the fork's definition of correct, and they must pass on a fresh boot, through every dispatch path, with a stable numeric oracle behind them rather than an eyeballed digit line. This epic owns the outstanding correctness defects — DEVICE_LOST at the PP->TG transition, fresh-boot GPT-OSS garbage, Mistral K-quant corruption, the generic BLAS-fallback last-row corruption, the reachable fattn Q-type assert, the deepseek32 NMSE miss — and the validators that would have caught them (a logit oracle for the GPT-OSS fast paths, a startup canary, wider MMVQ-ID validation).
  - **Bar:** On a fresh boot, with ONEAPI_DEVICE_SELECTOR pinned to the discrete cards, both canonical gates pass on B70 and B50 in the same build: Mistral emits '1, 2, 3, 4, 5, 6, 7, 8, 9, 10' and GPT-OSS (-c 4096) emits its digit line with rc=0 and zero aborts, including across a PP->TG transition in one process; a logit-level oracle pins GPT-OSS fast-path logits against a CPU reference and is registered in ctest; deepseek32 B70 MoE NMSE <= 1e-04 both isolated and in-sweep; GGML_SYCL_MMVQ_ID_VALIDATE covers more than MXFP4; no reachable abort from supported CLI flag combinations.
  - **Constraints:** Ruling 9 (correctness before throughput; the two named gates are the gates; flags with known GPT-OSS failures stay opt-in). Ruling 8 (oneDNN SDPA on by default is correct for Mistral GQA — do not chase a phantom GGML_SYCL_FA_ONEDNN_ALLOW). Ruling 11 (do not restore env-vars that no longer exist). The gates check TOKENS, not the backend: confirm GGML_SYCL:BOOL=ON in build/CMakeCache.txt and libggml-sycl in ldd before trusting any green run, and treat a SKIP / exit-77 as 'not verified'. Single-digit tok/s on Mistral Q4_0 is a CPU fallback, not a regression.
  - **Ordering hint:** llama.cpp-30ak7.19.12 (the stable correctness/logit oracle) is the prerequisite for the rest: llama.cpp-sw2oj (fresh-boot garbage) and llama.cpp-aqzz3.2 cannot be closed convincingly against an eyeballed digit line, and llama.cpp-nhcww (startup canary) is the cheap early-fail that follows from it. llama.cpp-kgzj6 (DEVICE_LOST at the PP->TG transition) blocks any single-process PP+TG validation elsewhere and should be taken first among the bugs. llama.cpp-lhmy, llama.cpp-dqbc and llama.cpp-am2.21 are independent defects; llama.cpp-s0iy and llama.cpp-ur4j widen the validators once the oracle exists; llama.cpp-am2.12 and llama.cpp-g8m4 are the end-to-end runs that close the epic.
### Members

  - llama.cpp-g8m4
  - llama.cpp-kgzj6
  - llama.cpp-sw2oj
  - llama.cpp-30ak7.19.12
  - llama.cpp-am2.21
  - llama.cpp-aqzz3.2
  - llama.cpp-s0iy
  - llama.cpp-am2.12
  - llama.cpp-dqbc
  - llama.cpp-lhmy
  - llama.cpp-nhcww
  - llama.cpp-ur4j
  - llama.cpp-4jwx

### Within epic duplicates

  - _(empty)_

### Tests that cannot pass vacuously

  - **Slug:** tests-that-cannot-pass-vacuously
  - **Existing epic id:** NEW
  - **Title:** Tests that cannot pass vacuously
  - **Goal:** A green test must mean work happened. Today several do not: tests that PASS while skipping, source-contract gates that literal-match expressions which legitimately changed or are red at baseline, CHECK blocks that never execute, suites whose footer is absent from exactly the runs that hang, and orphan or never-compiled test sources. This epic makes skips visible (exit 77), makes RED baselines actually red, registers the unregistered gates, deletes the dead ones, and puts a fail-closed CTest policy around the GPU/TTM-memory family so a sweep cannot OOM the host.
  - **Bar:** A full sweep in the documented safe form (ctest --test-dir build --output-on-failure -j 1 -LE 'residency|mem-handle|cache' -E '^test-backend-ops$') is green at HEAD, plus the excluded family run serially; every skip exits 77 and no test reports PASS on a skip path (llama.cpp-9trn is the standing counter-example); all source-contract gates are green on master or converted to assertions that can fail for a real reason; the never-executed CHECK census in the graphlet-policy suite is mutation-proved; the G2/G3/G5b lifecycle gates and the M4/M5 mutation runner are registered in tests/CMakeLists.txt; no test file in tests/ is unregistered, never-compiled, or asserting a deleted contract; GPU-allocating tests carry labels so a label filter cannot fail open.
  - **Constraints:** A test that passes without doing its work is a defect, not coverage — every gate needs a positive control that can fire, and a probe must grep the string the code actually PRINTS (GGML_ABORT never reaches a log; the literal that survives is 'file.cpp:line:' or the message text). Host rules are non-negotiable: -j 1 on any sweep (-j is a memory multiplier, not a CPU throttle), -E '^test-backend-ops$', ONEAPI_DEVICE_SELECTOR pinned to level_zero:0,1 (unpinned runs take 195-227 GB of TTM shmem via the iGPU), never loop a model-loading binary, and all GPU runs are lead-only.
  - **Ordering hint:** llama.cpp-ezfm (fail-closed suite-wide CTest policy serialising the GPU/TTM tests) is the prerequisite for everything that requires running the suite — without it the sweep that proves the bar can take the host down. llama.cpp-9trn and llama.cpp-g290 ('77 means skip' has three homes) set the skip contract that llama.cpp-dy1r, llama.cpp-1lrh and llama.cpp-32dg8.20 then apply. The RED-at-baseline set (llama.cpp-1s31, llama.cpp-vec6, llama.cpp-vtu3, llama.cpp-stjn, llama.cpp-qqs2, llama.cpp-j8dy, llama.cpp-9lqc) must be resolved before any 'the suite is green' claim, and llama.cpp-9fgx (mutation-sweep the ~168 never-executed CHECKs) before that suite is cited as coverage. The deletions (llama.cpp-uhz2, llama.cpp-zxlp, llama.cpp-la7d, llama.cpp-eltp) are independent and cheap.
### Members

  - llama.cpp-24rt
  - llama.cpp-32dg8.20
  - llama.cpp-if0o
  - llama.cpp-xqksg
  - llama.cpp-09ep
  - llama.cpp-1lrh
  - llama.cpp-1s31
  - llama.cpp-3roy
  - llama.cpp-4t5v
  - llama.cpp-86xu
  - llama.cpp-9fgx
  - llama.cpp-9lqc
  - llama.cpp-9trn
  - llama.cpp-dy1r
  - llama.cpp-ezfm
  - llama.cpp-j8dy
  - llama.cpp-pwvf
  - llama.cpp-stjn
  - llama.cpp-vec6
  - llama.cpp-ybdd
  - llama.cpp-697m
  - llama.cpp-7tc7
  - llama.cpp-eltp
  - llama.cpp-g290
  - llama.cpp-gjkz
  - llama.cpp-mcdk
  - llama.cpp-mcv8
  - llama.cpp-qqs2
  - llama.cpp-s498
  - llama.cpp-tglk
  - llama.cpp-uhz2
  - llama.cpp-ui4s
  - llama.cpp-vtu3
  - llama.cpp-zxlp
  - llama.cpp-la7d

### Within epic duplicates

  - _(empty)_

### Upstream absorbed — every op SYCL claims is implemented and dispatched

  - **Slug:** op-coverage-and-upstream-ports
  - **Existing epic id:** llama.cpp-1yr6
  - **Title:** Upstream absorbed — every op SYCL claims is implemented and dispatched
  - **Goal:** The b10630 merge left a backlog of upstream SYCL commits unported and, worse, several ops that are implemented but never dispatched while docs/ops/SYCL.csv claims coverage. This epic lands the Phase-C port queue and closes the honesty gap between what supports_op advertises, what the dispatch switch reaches, and what the CSV says — including GGML_OP_TRI (whose SYCL implementation is missing outright), the seven implemented-but-undispatched ops from the lost merge hunks, and the bin_bcast / SET_ROWS / SSM_SCAN capability holes.
  - **Bar:** Every [ports] ticket is either landed with its upstream sha cited or closed as not-applicable with a stated reason; test-backend-ops (lead-run, single invocation, ONEAPI_DEVICE_SELECTOR pinned to level_zero:0,1) shows no SYCL op failures for the ops named in these tickets and no regression against the pre-port run; docs/ops/SYCL.csv is regenerated from the dispatch table rather than hand-maintained, with a gate that fails if an op is claimed but unreachable; supports_op returns false (CPU fallback) instead of producing wrong numbers for every uncovered type; test-llama-archs single-arch runs exit 0 for the architectures these ports unblock (rc, not the corruptible results table).
  - **Constraints:** Ruling 11 (verify a variable or knob still exists before restoring it — check docs/backend/sycl-env-vars.md and grep the source). Ruling 10 (the iGPU-classification port must not produce code that parses or refuses ONEAPI_DEVICE_SELECTOR). Rulings 1 and 2 for any port touching allocation — upstream code calling sycl::malloc_* directly must be adapted to unified_allocate / mem_handle BEFORE it lands, not after. Ruling 9 (a CPU fallback is preferable to a wrong number). test-backend-ops is lead-only, single invocation, selector-pinned (50-224 GB TTM shmem hazard).
  - **Ordering hint:** llama.cpp-8cou (seven more ops implemented but undispatched, plus the doc rows that claim them) is the prerequisite for the CSV work — it defines the honesty gap the bar measures; llama.cpp-tu4t (GGML_OP_TRI implementation missing, not just its dispatch) and llama.cpp-yitq's sibling coverage gaps (llama.cpp-9uic, llama.cpp-9vm3) follow it. The [ports] queue under llama.cpp-1yr6 is largely independent and parallelizable, but ports touching allocation (llama.cpp-5pgc, llama.cpp-jah9) need the ownership screen applied first, and llama.cpp-g7hq / llama.cpp-d6is (oneDNN SDPA and FLASH_ATTN_EXT) should be sequenced with the attention work in prefill-xmx-int8-peak to avoid two people in fattn at once.
### Members

  - llama.cpp-1yr6
  - llama.cpp-8cou
  - llama.cpp-8vwp
  - llama.cpp-hkuh
  - llama.cpp-xwnkf
  - llama.cpp-0w52
  - llama.cpp-1kcx
  - llama.cpp-3zzs
  - llama.cpp-596y
  - llama.cpp-5pgc
  - llama.cpp-922d
  - llama.cpp-9ppc
  - llama.cpp-ada5
  - llama.cpp-ag2d
  - llama.cpp-al2o
  - llama.cpp-d6is
  - llama.cpp-g7hq
  - llama.cpp-h2uz
  - llama.cpp-jah9
  - llama.cpp-kdcm
  - llama.cpp-pmzl
  - llama.cpp-qz0f
  - llama.cpp-tfap
  - llama.cpp-tjt9
  - llama.cpp-tp03
  - llama.cpp-tu4t
  - llama.cpp-v7lu
  - llama.cpp-vvci
  - llama.cpp-xoi2
  - llama.cpp-9uic
  - llama.cpp-9vm3
  - llama.cpp-j6qq

### Within epic duplicates

  - _(empty)_

### Every microsecond of a decode step is attributed

  - **Slug:** decode-cost-attribution
  - **Existing epic id:** llama.cpp-p8sq
  - **Title:** Every microsecond of a decode step is attributed
  - **Goal:** Before any further optimisation is chosen, a decode step and a pp512 pass must be fully accounted for: device-event time per kernel, host submission time, and the unrecorded remainder named rather than bucketed. The profiler already covers the decode MUL_MAT paths; this epic closes the gap attribution, produces the ranked most-expensive-region table and the ablation deltas that decide what is worth doing, and settles the standing claims (78-85% of Cluster B unrecorded; the B50 pp512 attribution questions). It owns the instruments and the baseline bookkeeping, not the optimisations.
  - **Bar:** A single ranked host+device cost table for GPT-OSS 20B tg128 and pp512 on B50 and B70 whose rows sum to >= 95% of measured wall time, with the unattributed remainder named (no anonymous bucket above 5%), produced from device-event timing rather than host chrono; region-ablation deltas correlate to that table; the B50 GPT-OSS pp512 question is answered against the real ~894 baseline rather than the retired >=1100 guardrail, and the Mistral pp512 floor-miss ticket carries a written verdict; docs/backend/sycl-perf-baselines.md rows are current for driver 26.31.
  - **Constraints:** Baselines come from docs/backend/sycl-perf-baselines.md, never from a remembered figure or a task's acceptance criteria (the >=1100 B50 pp512 and >=50 tg guardrails are stale and unreachable). Absolute numbers taken under this host's permanent ~60 ambient load are not baselines — verdicts come from interleaved paired A/Bs; do not defer a measurement waiting for a quiet host. OP_TIMING share is not recoverable time (drain-mode wall inflates 10-20x). Ruling 6 — a profiler must not introduce drains into the path it measures, and a profiling-enabled queue must not turn event_complete into a blocking wait. Efficiency targets are 80% of HARDWARE peak (bandwidth for decode, INT8 XMX for prefill).
  - **Ordering hint:** llama.cpp-p8sq's plan sets the order and its Track F/G children are strictly sequential: llama.cpp-mxfp4-tg-runtime-zy9z (ranked cost table) -> llama.cpp-mxfp4-tg-runtime-896h (lead-only region ablation) -> llama.cpp-mxfp4-tg-runtime-nqwo (correlate into the final report). llama.cpp-ejjq (falsify the 78-85%-unrecorded verdict) must be settled before the table is trusted, and llama.cpp-gnq8 (dedupe the event-duration helpers) before more call sites are instrumented. llama.cpp-pebz and llama.cpp-c1pb (duplicate pair) are the umbrella profiling runs; llama.cpp-z2h9, llama.cpp-41sq, llama.cpp-pbb2 and llama.cpp-r9dn are the specific attribution verdicts the table exists to produce.
### Members

  - llama.cpp-pebz
  - llama.cpp-c1pb
  - llama.cpp-mxfp4-tg-runtime-896h
  - llama.cpp-mxfp4-tg-runtime-nqwo
  - llama.cpp-mxfp4-tg-runtime-zy9z
  - llama.cpp-p8sq
  - llama.cpp-pbb2
  - llama.cpp-41sq
  - llama.cpp-ejjq
  - llama.cpp-r9dn
  - llama.cpp-z2h9
  - llama.cpp-7s30
  - llama.cpp-gnq8

### Within epic duplicates

  - _(empty)_

### A fast build, honest docs, working tooling

  - **Slug:** build-docs-and-dev-loop
  - **Existing epic id:** NEW
  - **Title:** A fast build, honest docs, working tooling
  - **Goal:** The developer loop is a first-order cost on this fork: every commit forces a full relink, a fresh worktree can spend ~50 minutes on one translation unit, and several docs and inventories describe code that no longer exists. Fix the relink triggers and CMake dependency mistakes (GGML_COMMIT as a compile definition, LLAMA_INSTALL_VERSION bumps, .git/index as a RERUN_CMAKE dep), port the upstream build-parallelism work (ocloc, AOT link jobs), cache kernel modules, delete dead sources, and correct the docs and tooling notes that have already produced wrong conclusions.
  - **Bar:** A no-op ninja after a commit does no relink and does not re-run CMake, completing in seconds (measured before/after and recorded in the ticket); a fresh incremental build after a one-line edit to ggml-sycl.cpp finishes within the documented ~10 min ccache-warm window, with ocloc/AOT link parallelism enabled and the full build time recorded before and after; every dead source named here is removed or given a CMake entry; the env-var catalog (docs/backend/sycl-env-vars.md), the allocation-site inventory and CLAUDE.md's pre-run memory-check instrument agree with a live grep at HEAD and with each other.
  - **Constraints:** Ruling 11 (env-var docs must match a grep of the source — a name that is no longer read is not a feature to restore; GGML_SYCL_FA_ONEDNN_ALLOW and GGML_SYCL_VISIBLE_DEVICES are the standing examples). Ruling 10 (docs must not prescribe parsing or overriding ONEAPI_DEVICE_SELECTOR). Build changes must not alter what the binary links: verify GGML_SYCL:BOOL=ON and libggml-sycl in ldd after any CMake change, since a reconfigure has silently reset GGML_SYCL to OFF here. Source oneAPI before any build; ccache is global but a NEW worktree path is close to a cold build; take BUILD.lock before the first edit in the shared checkout; with other agents live use clang-format-19 --dry-run -Werror on your own file only.
  - **Ordering hint:** llama.cpp-p0lc (GGML_COMMIT as a compile definition on all ggml targets) and llama.cpp-tw5e (LLAMA_INSTALL_VERSION bump per commit) are the two relink triggers and should land first — every other build-time measurement in this epic is contaminated by them; llama.cpp-i0cf (.git/index as a RERUN_CMAKE dep) is the third. Then the parallelism ports llama.cpp-achg and llama.cpp-17zg, then llama.cpp-l6eoj (cached kernel modules). The doc corrections (llama.cpp-iwln, llama.cpp-cyn.1, llama.cpp-017kb.9, llama.cpp-l2hx) are independent; llama.cpp-2rkc (isolate codescout daemon GPU work from B70 validation) should be done before any measurement campaign that needs the B70 quiet.
### Members

  - llama.cpp-cyn.1
  - llama.cpp-l2hx
  - llama.cpp-017kb.9
  - llama.cpp-17zg
  - llama.cpp-2rkc
  - llama.cpp-3658
  - llama.cpp-achg
  - llama.cpp-h9g6
  - llama.cpp-iwln
  - llama.cpp-oqv9
  - llama.cpp-p0lc
  - llama.cpp-tw5e
  - llama.cpp-bdav
  - llama.cpp-i0cf
  - llama.cpp-l6eoj
  - llama.cpp-srck
  - llama.cpp-wv6k
  - llama.cpp-1g5n
  - llama.cpp-hpan
  - llama.cpp-q9y8

### Within epic duplicates

  - _(empty)_

## Epic ticket dispositions

### llama.cpp-mubmt

  - **Id:** llama.cpp-mubmt
  - **Disposition:** reuse-as-epic
  - **Target slug:** unified-cache-sole-allocator
  - **Why:** Title 'Make unified_cache the sole owner of SYCL memory' matches the merged scope ~90%; its design-intent section already states the single-allocator contract and the no-inference-churn rule.

### llama.cpp-z2316

  - **Id:** llama.cpp-z2316
  - **Disposition:** fold-into
  - **Target slug:** unified-cache-sole-allocator
  - **Why:** Same goal as mubmt (one owner for VRAM + host-pinned) written a week earlier; fold its unique content (per-device layout section) into mubmt and close as duplicate.

### llama.cpp-32dg8

  - **Id:** llama.cpp-32dg8
  - **Disposition:** fold-into
  - **Target slug:** unified-cache-sole-allocator
  - **Why:** Umbrella for all three canonical primitives (planner + cache + handle); its children are already distributed across four merged epics, so it survives only as the allocator half. Keep its .4/.5/.6/.7/.15.x/.16.x children where they are assigned.

### llama.cpp-rg2ft

  - **Id:** llama.cpp-rg2ft
  - **Disposition:** reuse-as-epic
  - **Target slug:** handles-only-no-pointer-escapes
  - **Why:** 'Enforce zero-alloc inference — eliminate heap churn, raw pointers, and arena bypasses' matches the pointer-escape scope ~75%; its T2 (raw pointer caches bypass mem_handle) is exactly this epic's spine. T1/T3/T4 items move to the allocator, reclaim and planner epics.

### llama.cpp-24rt

  - **Id:** llama.cpp-24rt
  - **Disposition:** fold-into
  - **Target slug:** tests-that-cannot-pass-vacuously
  - **Why:** Session epic (2026-08-01 memory ownership + in_use_count + test integrity); the test-integrity half is this epic, the ownership half is already covered by mubmt/rg2ft.

### llama.cpp-3h5gm

  - **Id:** llama.cpp-3h5gm
  - **Disposition:** reuse-as-epic
  - **Target slug:** planner-plans-every-byte
  - **Why:** Title 'Unified Memory Placement — plan every byte (weights + KV + compute + scratch)' is a near-verbatim match; both proposals agreed.

### llama.cpp-p8ic3

  - **Id:** llama.cpp-p8ic3
  - **Disposition:** keep-as-member
  - **Target slug:** planner-plans-every-byte
  - **Why:** 'Priority-driven weight placement — perf-tier policy on top of placement plan' is a sub-programme of the planner epic, not a peer of it; both proposals agreed.

### llama.cpp-792vn.5

  - **Id:** llama.cpp-792vn.5
  - **Disposition:** reuse-as-epic
  - **Target slug:** budget-authority-one-ledger
  - **Why:** 'Eliminate redundant runtime budget counters — arena zones are the single source of truth' is the merged goal verbatim; both proposals agreed.

### llama.cpp-sk1xz

  - **Id:** llama.cpp-sk1xz
  - **Disposition:** reuse-as-epic
  - **Target slug:** never-shrink-context-tiered-kv
  - **Why:** 'Unified-cache-owned memory placement with KV/weight co-location' is the tiered-KV programme; both proposals agreed. Restate its bar around ruling 7 (no -c pin).

### llama.cpp-30ak7

  - **Id:** llama.cpp-30ak7
  - **Disposition:** reuse-as-epic
  - **Target slug:** gptoss-decode-bandwidth
  - **Why:** '[B50-GPTOSS] Double single-stream TG bandwidth utilization' matches the merged goal ~85%. RETITLE the bar: the '~70 tok/s' anchor is retired in favour of 80% of the 224/608 GB/s roofline.

### llama.cpp-v90xn.30

  - **Id:** llama.cpp-v90xn.30
  - **Disposition:** keep-as-member
  - **Target slug:** gptoss-decode-bandwidth
  - **Why:** '[B50-MOE-XMX] Direct VRAM-resident MXFP4 MoE layer executor' is one route inside the decode epic; both proposals agreed it is a member, not an epic.

### llama.cpp-xihy

  - **Id:** llama.cpp-xihy
  - **Disposition:** reuse-as-epic
  - **Target slug:** prefill-xmx-int8-peak
  - **Why:** 'Unified XMX Kernel: Consolidate All XMX Implementations' is the consolidation half of the merged epic; add the INT8-peak bar and the MMID amplification target to it.

### llama.cpp-ukwqh

  - **Id:** llama.cpp-ukwqh
  - **Disposition:** fold-into
  - **Target slug:** prefill-xmx-int8-peak
  - **Why:** The fattn-xmx/oneDNN-SDPA rewrite epic has only ~5 surviving tickets and no separate measurable bar on current hardware; SDPA is a GEMM on the same XMX units. Fold, keeping its 'select via queried capabilities, unified_cache-owned buffers' acceptance text.

### llama.cpp-ic4f

  - **Id:** llama.cpp-ic4f
  - **Disposition:** reuse-as-epic
  - **Target slug:** experts-run-where-they-live
  - **Why:** 'Optimize CPU offload staging path for TG performance' with the constraint 'without reintroducing weight streaming' matches the host-execution half; both proposals reused it. Widen the title to cover device-side residency.

### llama.cpp-017kb

  - **Id:** llama.cpp-017kb
  - **Disposition:** keep-as-member
  - **Target slug:** experts-run-where-they-live
  - **Why:** 'Unify MoE expert memory under unified-cache smart handles' is expert-memory ownership — a workstream inside this epic (and partly inside handles-only). Its B580+B50 acceptance clause is obsolete-hw and must be restated on B70+B50.

### llama.cpp-po3nd.2

  - **Id:** llama.cpp-po3nd.2
  - **Disposition:** reuse-as-epic
  - **Target slug:** second-card-earns-its-keep
  - **Why:** 'Make B580+B50 GPT-OSS 20B PP faster than best single GPU' is the merged goal with one substitution (B580 -> B70). It already carries 20+ children and the design invariants; reuse and re-baseline.

### llama.cpp-03k1

  - **Id:** llama.cpp-03k1
  - **Disposition:** keep-as-member
  - **Target slug:** second-card-earns-its-keep
  - **Why:** '[EPIC] Multi-GPU VRAM Budget Fix + Expert Distribution' overlaps po3nd.2 and budget-authority; keep it as the per-device-budget/expert-distribution member under po3nd.2 rather than a second multi-GPU epic.

### llama.cpp-kkxtv

  - **Id:** llama.cpp-kkxtv
  - **Disposition:** keep-as-member
  - **Target slug:** second-card-earns-its-keep
  - **Why:** Cross-device KV and activation routing is one workstream of the dual-device epic; both proposals agreed.

### llama.cpp-m0yo

  - **Id:** llama.cpp-m0yo
  - **Disposition:** reuse-as-epic
  - **Target slug:** big-model-load-and-120b
  - **Why:** '120B MoE end-to-end inference — planned residency + watchdog + performance' matches the merged scope ~80%; its B580+B50 clause is restated on B70(+B50).

### llama.cpp-uwcn

  - **Id:** llama.cpp-uwcn
  - **Disposition:** fold-into
  - **Target slug:** big-model-load-and-120b
  - **Why:** '[MASTER EPIC] 120B MoE performance — planned mixed residency + engine split' duplicates m0yo's goal; its 'Superseded direction' section (hot-expert VRAM caching, demand loading, Zipf pinning) is worth preserving verbatim as the record of what ruling 4 forbids.

### llama.cpp-jgc6q

  - **Id:** llama.cpp-jgc6q
  - **Disposition:** fold-into
  - **Target slug:** big-model-load-and-120b
  - **Why:** 'Recover 120B prompt-processing speed on SYCL B580' — the card is gone and its ~1.03 tok/s baseline is obsolete-hw; the surviving intent (120B PP recovery) is covered by the merged epic's re-baselining bar. Keep child jgc6q.5 as a member.

### llama.cpp-1yr6

  - **Id:** llama.cpp-1yr6
  - **Disposition:** reuse-as-epic
  - **Target slug:** op-coverage-and-upstream-ports
  - **Why:** 'EPIC: Phase C upstream-SYCL ports (post-merge)' already carries the 29 [ports] tickets as dependencies and the ownership-screen requirement; add the supports_op/CSV honesty bar to it.

### llama.cpp-p8sq

  - **Id:** llama.cpp-p8sq
  - **Disposition:** reuse-as-epic
  - **Target slug:** decode-cost-attribution
  - **Why:** '[EPIC][SYCL-PROFILING] Complete decode profiling coverage and gap attribution' is the merged goal verbatim, including the ranked-cost-table and ablation children.

## Unassigned

- _(empty)_