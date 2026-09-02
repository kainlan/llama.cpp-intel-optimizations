# Design brief for tracker triage (llama.cpp SYCL fork) — 2026-09-01

You are judging OPEN tracker tasks against the CURRENT state of the fork. Repo: /Apps/llama.cpp, branch master,
HEAD fed0b58e2 (2026-09-01). Upstream llama.cpp b10630 was merged on 2026-08-26 (merge commits around then).
Full task text for every open task is in ./by-id/<id>.txt (this directory). Do NOT run anything that loads a
model on a GPU; if a task needs GPU verification, say exactly what command and expected output, and the lead runs it.

## Hardware / environment truths (older tasks often assume otherwise)
- Cards NOW: Arc Pro B70 (level_zero:0, Battlemage G31, 256 CU, 32.6 GB, 608 GB/s) and Arc Pro B50 (level_zero:1,
  G21, 128 CU, 16 GB, 224 GB/s), plus an Arrow Lake-S iGPU that must always be excluded via ONEAPI_DEVICE_SELECTOR.
- The Arc B580 was REMOVED (replaced by the B70, 2026-07-24). Any task whose goal is B580-specific numbers or
  "B580+B50" topology is obsolete-hardware unless the intent transfers cleanly to B70+B50.
- No PCIe P2P between the two discrete cards (different root ports); all cross-device traffic host-bounces.
  Multi-GPU expert distribution is opt-in (GGML_SYCL_MOE_MULTI_GPU) and cannot rely on peer DMA.
- Driver: libze_intel_gpu 26.31 since 2026-08-18 (the patched 26.22 build is NOT loaded). oneAPI 2025.3.
- Perf baselines (docs/backend/sycl-perf-baselines.md): B70 Mistral Q4_0 ~2495 pp512 / ~108 tg128, GPT-OSS 20B
  ~1415/~44; B50 Mistral ~1188/~47, GPT-OSS ~894/~32. Any task gating on "B50 >=1100 pp512" or ">=50 tg" is stale.
- Efficiency goal (owner, 2026-09-01): 80% of HARDWARE theoretical peak — memory bandwidth for decode, INT8 XMX
  compute for prefill (B70 367 TOPS, B50 170 TOPS). Not "80% of Mistral's tok/s".

## Owner design rulings (a task that contradicts these is AGAINST-DESIGN and should be closed or rewritten)
1. The unified cache (ggml/src/ggml-sycl/unified-cache.cpp) is the SOLE allocator for all SYCL backend memory
   (device, host-pinned, staging, scratch, graph temporaries, KV, oneDNN scratch, weight layouts). No direct
   sycl::malloc_*/free, no side caches, no raw TLSF outside unified-cache. Entry points: unified_alloc /
   unified_allocate / unified_allocate_owner (owner-first). Contract: docs/design/sycl-canonical-memory-architecture.md.
2. mem_handle is the only ownership/lifetime token. Raw pointers are transient views for immediate submission; never
   stored as source of truth, never used as cache keys, never outliving the handle. Pointer tables / dispatch caches
   key on the handle's stable identity hash. Freed only when the last reference is released. No forced eviction,
   forced reap, or zone reset to reclaim memory with a live handle.
3. Zone resets are being ELIMINATED (epic llama.cpp-iiff, Option-C epoch-refcounted transient zones is shipped for
   several zones). Tasks that ADD reset/drain steps, or that build on zone_reset as a mechanism, are against design.
   Tasks that remove escapes (raw pointers surviving past handle ownership) are aligned.
4. PLACEMENT DECIDES THE EXECUTOR (2026-08-16): the planner places data where it fits/runs best; ops execute where
   the data already is. Device-resident -> that GPU. Host-pinned -> the CPU (CpuExpertPool path). FORBIDDEN: GPU
   zero-copy reads of host memory; weight STREAMING (copying host-resident weights to device scratch per dispatch);
   "prefetch experts to VRAM per token" style designs. On-device layout conversion of device-resident data is fine.
5. LAYOUT FOLLOWS RESIDENCY / ONE LAYOUT PER WEIGHT (2026-08-16/17): each weight is materialized once, in the layout
   optimal for the processor that executes it; ALL consumers (PP grouped and TG small-batch, gate/up AND down) must
   support that layout; never duplicate weights in two layouts, never shift layout at dispatch time. Routes only
   advertise (type, layout) pairs whose kernels exist. AOS fallback is a stopgap, not the design.
6. NO HOST WAITS (2026-08-20): no wait()/wait_and_throw()/drain-per-op in dispatch paths; ordering is expressed as
   SYCL event dependencies (depends_on, oneDNN sycl_interop::execute deps). Waits only in teardown/error/debug/init.
   A task proposing a wait as a fix is against design; a task removing a wait in favour of events is aligned.
7. NEVER SHRINK CONTEXT (2026-08-31): default n_ctx = model max. If KV does not fit VRAM the fix is PLACEMENT (tiered
   KV to pinned host), never auto-shrinking n_ctx, never refusing init. Explicit -c is the only way to get less.
8. Small-block dequant (Q4_0/Q8_0/Q4_K) belongs on standard SYCL, not ESIMD (measured 1.9x slower); the lever is
   fusing dequant into the matmul. oneDNN SDPA is ON by default and correct for Mistral GQA; GGML_SYCL_FA_ONEDNN=0
   costs ~39% pp. GGML_SYCL_FA_ONEDNN_ALLOW does not exist.
9. Correctness before throughput: the Mistral completion gate and GPT-OSS chat gate (CLAUDE.md) are the gates.
   Opt-in flags that have shown GPT-OSS correctness failures stay opt-in: GGML_SYCL_MOE_BLOCK_GRAPHLETS,
   GGML_SYCL_XMX_MOE_PP, GGML_SYCL_PP_PIPELINE.
10. Device selection belongs to oneAPI (ONEAPI_DEVICE_SELECTOR); no code parsing/refusing it. GGML_SYCL_VISIBLE_DEVICES
    does not work. VRAM budget = min(total*pct, free_at_init); iGPU host-unified memory must not be budgeted as VRAM.
11. Legacy env-vars that no longer exist or are unread are not "features to restore": check
    docs/backend/sycl-env-vars.md and grep the source before assuming a variable is live.

## Verdict vocabulary (use exactly these)
- done          : the work described has already landed (cite commit sha(s) and/or file:line proving it)
- superseded    : a later design/epic/ticket replaced the approach (cite the replacing id or doc section)
- obsolete-hw   : premised on hardware/driver/topology no longer present (B580, old driver, P2P assumptions)
- against-design: contradicts a ruling above (say which number) — close, or rewrite if a compliant core remains
- duplicate     : same work as another OPEN task (cite id); the lower-quality one closes
- unverifiable  : so vague / so stale (context gone, no repro, referenced code deleted) that it cannot be acted on
- keep          : still valid and needed; must be assignable to an epic theme
- needs-gpu     : keep-pending: the decision hinges on a single measurement/gate the lead must run (give the command)

Be concrete. A verdict of done/superseded/against-design MUST carry evidence a skeptic can check
(a commit sha you found with git log -S / git log --grep, a file:line, a doc section, a ruling number).
Uncertain -> keep (with your doubts), never a guessed closure.
