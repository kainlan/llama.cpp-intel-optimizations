# BCS eviction backpressure audit (WEDGE-T3)

Scope: every ggml-sycl BCS-submission callsite and adjacent free/release
callsite, classified by whether the pattern established in `llama.cpp-9ug6y`
(defer frees behind `submit_barrier_all()`, process only at drained sync
points) is required. Companion to T3 / epic `llama.cpp-rp4k4`.

## Race pattern (recap, from 9ug6y)

BCS CAT [18] faults when `sycl::free(ptr)` unmaps VRAM pages while BCS
(copy engine) is still writing nearby pages in the same L0 allocation
region. `sycl::free` does NOT block on in-flight DMA; BCS has no host-
side back-pressure once the memcpy is submitted. The fix class that
prevents the fault is any one of:

- **A: drain before free** — `queue_.wait()` on all relevant queues
  (CCS, DMA, BCS, compute) immediately before `sycl::free`. Cheap when
  a drain was imminent anyway.
- **B: defer free** — push `(ptr, size, submit_barrier_all event)` onto
  a deferred-free list. Process the list only at an explicit yield point
  after all queues have been drained (`process_deferred_frees()`).
- **C: reference the buffer** — take a shared ref before BCS submit,
  release after the BCS event signals.

9ug6y picked **B** everywhere that evicts mid-prestage, via
`unified_cache::enqueue_deferred_free`. The barrier events in
`enqueue_deferred_free` already cover ALL four queues (see
`submit_barrier_all()` at `unified-cache.cpp:4278-4315`), so any call that
routes through `enqueue_deferred_free` is automatically BCS-safe.

## Catalog

### Evict/free paths inside unified_cache (the source of 9ug6y)

| # | Site | Verdict | Note |
|---|------|---------|------|
| 1 | `unified-cache.cpp:3721-3900` `unified_cache::evict_one` | **Safe** (9ug6y) | Always routes through `enqueue_deferred_free` for non-pool, non-arena entries. Comment at 3882 explicitly names the BCS CAT risk; pool entries only update accounting. |
| 2 | `unified-cache.cpp:3649-3674` `evict()` | **Safe** | Lock-scoped; calls `process_deferred_frees()` at entry only when `!g_graph_compute_active` (i.e. NOT mid-inference) and delegates actual frees to `evict_one → enqueue_deferred_free`. |
| 3 | `unified-cache.cpp:3676-3707` `evict_and_flush()` | **Safe** | Phase 1 defers frees via `evict()`. Phase 2 waits on `queue_` (CCS) before Phase 3's `process_deferred_frees()`. The BCS guarantee comes from `submit_barrier_all` in each deferred entry, not from Phase 2's CCS-only wait. |
| 4 | `unified-cache.cpp:4317-4345` `enqueue_deferred_free(ptr,size)` | **Safe** (defines the pattern) | Skips pool-owned and arena-owned ptrs (can't be individually freed). Captures `submit_barrier_all()` event for the free. |
| 5 | `unified-cache.cpp:4347-4373` `enqueue_deferred_free(managed)` | **Safe** | Same as #4, for managed `alloc_handle` entries. |
| 6 | `unified-cache.cpp:4392-4495` `process_deferred_frees()` | **Safe (consumer)** | Walks the queue; only frees entries whose captured event has completed. Called by `process_deferred_frees_public()` wrapper from explicit yield points. |
| 7 | `unified-cache.cpp:1771`, `1996`, `2046`, `2488`, `3257`, `3898`, `3968`, `4747`, `4754` | **Safe** | All use `enqueue_deferred_free`. |
| 8 | `unified-cache.cpp:1340`, `1349`, `1361`, `1374`, `1383`, `1396`, `1409`, `1433`, `1454`, `1467`, `1479`, `1490` (destructor `~unified_cache`) | **Intentionally-not-fixed** | Destructor, runs after all queues are drained by the surrounding teardown. A direct `sycl::free` here is safe because no kernels or BCS copies are in flight at object destruction. Shutdown flag gates these. |
| 9 | `unified-cache.cpp:5463`, `5467`, `5492`, `5495`, `5816`, `7004`, `7015`, `7085`, `7095`, `7138`, `7262`, `7340`, `7504`, `7538`, `8016`, `8067`, `8235`, `9004`, `9329` | **Intentionally-not-fixed** | Pool/registry cleanup, scratch-buffer teardown, arena chunk release, all under shutdown or explicit drain guard. Inspection confirms no concurrent BCS at these points. |

### BCS-submitting sites (actual memcpy submissions to the BCS copy engine)

| # | Site | Submits what | Drain before free? | Verdict |
|---|------|--------------|--------------------|---------|
| 10 | `ggml-sycl.cpp:10383` `ggml_sycl_fill_reordered_gpu` H2D (BCS) + CCS reorder | H2D pinned→temp_vram on BCS, reorder kernel on CCS depends_on(dma_event) | `temp_vram` is arena-allocated (no individual free), or pushed to `ctx->temp_bufs` which is freed by caller after the full prestage drain. | **Safe** — arena ownership + post-drain free by `moe_prestage_popular_experts` at lines 1802-1808 (CCS + DMA + BCS wait + `process_deferred_frees_public`). |
| 11 | `unified-cache.cpp:2782, 2888` fill routines routed to `get_bcs_queue()` | H2D of expert weight content to newly-zone-allocated VRAM | Zone-allocated via `zone_alloc(WEIGHT, ...)` — arena-owned, no individual free. | **Safe** — arena-owned destinations. |
| 12 | `unified-cache.cpp:1242` BCS queue creation | — | — | **Safe** (construction). |
| 13 | `unified-cache.cpp:4301-4305` BCS barrier collection in `submit_barrier_all` | Empty-deps barrier | Used BY deferred-free protection, not a risk source. | **Safe** (is the fix). |
| 14 | `ggml-sycl.cpp:1694, 2060, 2185, 3920, 11654, 11824, 11969, 12118` BCS queue passed into fill ctx | — | Fill routed through `direct_stage_expert/direct_stage_weight` → arena zone alloc. | **Safe** — arena ownership. |
| 15 | `ggml-sycl.cpp:1801-1812, 1816-1820` prestage yield-loop drain | BCS wait + `process_deferred_frees_public()` | Explicit drain of CCS+DMA+BCS before the deferred free runs. | **Safe** (this is where 9ug6y plants the yield). |
| 16 | `ggml-sycl.cpp:10327, 11654, 11824, 11969, 12118` `h2d_queue.memcpy` / `cache->get_bcs_queue()` passed | Routed through the ctx above. | Arena-owned or deferred. | **Safe**. |
| 17 | `ggml-sycl.cpp:12152, 40667` `cache->get_bcs_queue().wait()` | Drains BCS | Explicit drain before subsequent deferred-free processing. | **Safe** (consumer of the fix). |
| 18 | `expert-prefetch.cpp:154-159` BCS fill submitted via `fill_reordered_host` | Async prestage in the prefetcher. | Arena/zone-owned dst. | **Safe** — same arena ownership model. |

### Scatter / pipeline paths (main CCS queue — NOT BCS, but same family)

The hot-path "scatter" and "pipeline" mechanisms in `ggml-sycl.cpp` dispatch
H2D `memcpy` on the **main compute queue (CCS)**, not on BCS. The race
pattern is not strictly BCS CAT [18] (which is BCS-specific address-
translation failure), but the same bug family — "free source buffer while
an async memcpy is still reading it."

| # | Site | Mechanism | Verdict |
|---|------|-----------|---------|
| 19 | `ggml-sycl.cpp:8002-8100` `flush_pending_cpu_scatter` | Host-pinned `out_pinned` → multiple device memcpys on `g_pending_scatter.stream` (CCS). Buffers moved to `prev_bufs` at end. Freed at TOP of next `flush_pending_cpu_scatter` via `flush_prev_scatter_bufs()` — which relies on the in-order CCS invariant ("the next flush runs after at least one kernel after the memcpys"). | **Safe by documented invariant** — the comment at 7966-7968 spells out the guarantee. No BCS involved. |
| 20 | `ggml-sycl.cpp:8313-8363` `flush_pending_cpu_pipeline` | Same pattern as #19 with `g_pending_cpu_pipeline.stream` (CCS) and `prev_bufs` deferred cleanup. | **Safe by documented invariant**. |
| 21 | `ggml-sycl.cpp:8519-8560` `flush_pending_secondary_scatter` | Cross-device: depends_on(e.last_event) is used to sync the D2H→H2D handoff. Source (`out_staging`) is a ring buffer slot. | **Safe** — cross-device `depends_on` respected; ring of 8 slots (MERGE_RING_SIZE at 29111) gives long recycle latency. |
| 22 | `ggml-sycl.cpp:2985-2991` `managed_host_pinned_buffer::reset()` | Calls `unified_free(handle)` on a host-pinned buffer. Called from `ensure()` on resize, and from `secondary_ring_buffers::~secondary_ring_buffers()` / reset paths. | **Safe by implicit invariant; no fix applied** — see analysis below. |

### Direct sycl::free outside unified-cache

| # | Site | Verdict |
|---|------|---------|
| 23 | `common.cpp:937-993`, `common.hpp:1765,1787,1964,3083,3106,3194,3201,3335-3380` | **Intentionally-not-fixed** — all are teardown of per-tensor / per-extra state at `ggml_tensor_extra_gpu` destruction or pool shutdown, after compute has drained. |
| 24 | `fattn.cpp:100,103,106,154,307,319,329,1278,1286,1327` flash-attn scratch frees | **Intentionally-not-fixed** — scratch is allocated/freed at per-op boundaries; queue is the alloc queue, so in-order semantics serialize any pending work. |
| 25 | `dense-scheduler.cpp:27,30,51,54` `vram_slot_[*]` frees | **Intentionally-not-fixed** — slot frees happen at scheduler teardown (not mid-compute). |
| 26 | `ccl-comm.cpp:374` staging free after `q.wait()` | **Safe** — preceded by explicit wait. |
| 27 | `device-pool.hpp:233` chunk free in pool shutdown | **Safe** — shutdown. Additional BCS drain added to `allocate()` / `grow_one_chunk()` (see `p2_bcs_drain_progress.md`). |
| 28 | `kv-offload.cpp:36` `block.cpu_ptr` release in KV-offload destructor | **Intentionally-not-fixed** — destructor path. |
| 29 | `layer-streaming.cpp:88` context stored for `sycl::free()` in `shutdown()` | **Intentionally-not-fixed** — shutdown. |
| 30 | `vram-pool.cpp:29,111` pool free in shutdown / slot release | **Intentionally-not-fixed** — pool-managed, shutdown path. |
| 31 | `cont-batching.hpp:421-434` sampler buffer frees | **Intentionally-not-fixed** — per-run teardown after sampling has completed. |
| 32 | `common.cpp:498, 782, 787` global-state frees | **Intentionally-not-fixed** — shutdown. |
| 33 | `unified-kernel.cpp:8771, 8826, 8974, 10837, 11135, 12452` persistent-TG scratch frees | **Safe by pattern** — freed after kernel completion (persistent TG path drains before any slot release). |
| 34 | `mmvq.cpp:3241` `extra->moe_expert_ptrs_compact_device` free | **Intentionally-not-fixed** — freed at extra teardown. |

### Deferred-free yield points (consumers of `process_deferred_frees_public`)

Every call site. These are the "safe sync points" the 9ug6y fix relies on:

| # | Site | Context |
|---|------|---------|
| 35 | `ggml-sycl.cpp:1804, 1818` | Prestage yield loop (+ final flush). |
| 36 | `ggml-sycl.cpp:11984, 12137, 12160` | S1-PRELOAD yield loop + final flush + trailing drain. |
| 37 | `ggml-sycl.cpp:39354` | Graph-boundary drain in `graph_compute_impl`. |

## Result: no demonstrable race found outside 9ug6y's scope

Audit conclusion: every BCS-submitting site in ggml-sycl writes to an
arena-owned / zone-owned destination (no individual free possible) or
to a deferred-freed destination routed through
`enqueue_deferred_free`, which already captures a 4-queue
`submit_barrier_all()` event that covers BCS.

The scatter / pipeline paths in `flush_pending_cpu_scatter`,
`flush_pending_cpu_pipeline`, and `flush_pending_secondary_scatter`
submit memcpys on the **main compute queue**, not BCS. They rely on
documented invariants (in-order CCS serialisation; 8-slot ring recycle)
to defer buffer reclaim safely to the next flush. These invariants
predate T3 and are still intact.

Every `sycl::free` call I examined outside unified-cache is in a
teardown / shutdown path — object destructor, pool shutdown, or a
per-op scratch free after the op's own drain. None races with an
in-flight BCS submission.

**No code fixes applied in this commit**; audit is the deliverable.

## Implicit-invariant sites worth a follow-up (not fixed here)

These sites are safe today, but the safety is **not** made explicit at
the callsite. A future P2 hardening pass could promote the invariant to
a structural guarantee (e.g. annotate the ring-slot struct with the
"scatter-was-flushed-before-resize" precondition; assert it in debug
builds). Not in scope for T3.

- `managed_host_pinned_buffer::ensure/reset` (site #22) — ring-slot
  resize assumes the last scatter for this slot has already been
  flushed. Holds because `flush_pending_secondary_scatter` happens
  per-MoE-layer and `MERGE_RING_SIZE==8` makes wrap-around take 8
  layers. An explicit event wait inside `reset()` would cost a drain on
  every resize; preferable to add a debug-only assert that
  `g_pending_secondary_scatter` is inactive at resize time.
- `flush_prev_scatter_bufs` / `flush_prev_cpu_pipeline_bufs` (sites
  #19-20) — documented "relies on in-order CCS kernel after memcpy"
  invariant. Could be promoted to a named predicate
  (e.g. `pending_scatter_dependency_satisfied()`) that asserts its own
  preconditions.

## Non-goals

- No new drain points beyond what 9ug6y already plants.
- No change to `process_deferred_frees()` semantics.
- No touching T2 watchdog timing (separate epic).
- Not promoting the implicit invariants to explicit checks (follow-up).

## References

- `llama.cpp-9ug6y` (closed) — original prestage fix, commits 7ae218e85 and 63e9d1b73.
- `memory/bug_bcs_cat_prestage.md` — race description.
- `memory/p2_bcs_drain_progress.md` — BCS drain in device-pool.
- `unified-cache.cpp:4278-4315` `submit_barrier_all()` — the 4-queue barrier used by every deferred free.
