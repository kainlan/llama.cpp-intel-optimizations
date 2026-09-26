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

## Placement decides the executor (owner ruling, 2026-08-16 — do not drift)

The division of labour between the planner and the dispatcher is fixed, and it
runs in exactly one direction:

> **The planning pass decides where data lives. Inference then executes each op
> where its data already is.** A `mem_handle` can point at a specific GPU's VRAM
> or at system host pinned memory; if the operand is in a device's VRAM, that
> device runs the op — if it is in host pinned memory, the **CPU** runs the op.

Placement is chosen for fit and for where execution will be most optimal — e.g.
**dense weights fill VRAM first, before MoE experts are placed there** — and
that decision is made once, in the planning pass. The dispatcher never
re-litigates it at op time. Two symmetric failure modes are therefore forbidden:

- **No GPU "zero-copy" reads of host memory.** Feeding a host-pinned pointer to
  a GPU kernel is slower than CPU dispatch (measured: CPU AOS 18–30 GB/s vs GPU
  zero-copy 11.3 GB/s) and breaks the tier abstraction.
- **No weight streaming.** Copying host-resident weights into device scratch
  per dispatch so the GPU can run the op is the same violation paid twice — a
  PCIe copy *plus* scratch pressure. The answer to "this expert is
  host-resident" is CPU expert dispatch (the `CpuExpertPool` machinery, live in
  `ggml_sycl_mul_mat_id`), overlapped with GPU work via `sycl::depends_on`
  (~9.7 µs cross-device latency). If a VRAM-starved configuration is too slow on
  CPU, the fix is placement (budget, eviction priority), never a streaming path.

Scope boundary, so the rule is not over-applied: staging that converts the
*format* of **device-resident** data (e.g. on-device dequant of MXFP4 into an
f16 scratch for oneDNN) is not a placement violation — the bytes never cross the
host/device boundary. The rule governs *residency*, not layout conversion. Even
so, such conversion scratch belongs to planned zones (see "Path-scoped zone
sizing"), not ad-hoc per-dispatch transients.

Conformance is auditable, not aspirational: `llama.cpp-v4jk` tracks the standing
audit that no dispatch route can receive a host-resident operand and stage it to
device (the routes themselves carry no residency branch — the diversion must be
proven structural, upstream, where `requires_host_staging` and the HOST tier are
modelled).

### Layout follows residency (owner ruling, 2026-08-16 — do not drift)

The companion rule to placement-decides-executor: **wherever a weight lives, it
is materialized in the optimal memory layout for the processor that executes on
it** — per device, not globally. Weights resident in B50 VRAM are arranged in
the optimal format for the B50's kernels; B70-resident weights in the optimal
format for the B70 (the two cards' capabilities differ — tile shapes, XMX
generation — so "optimal" is a per-device answer); host-pinned weights that the
CPU runs are arranged in the CPU's optimal format (AOS today, per the measured
CPU-AOS numbers above).

Three consequences, all enforceable:

- **The route layout follows the materialized layout, never the reverse.** A
  dispatch must consume what the cache actually materialized (or trigger the
  staging that re-materializes it) — it must never *advertise* a layout and then
  read differently-arranged bytes through it. The 2026-08-16 MMID NaN family
  (`llama.cpp-mn70`/`zoly`/`nkfc`) was exactly this drift: routes advertising
  SOA over AOS bytes with nothing reconciling them.
- **A layout may only be advertised if an executor kernel exists for that
  (type, layout) on that device** — the producer rule in `llama.cpp-nkfc`,
  anchored on the launcher roster.
- **"Fall back to AOS" is a correctness stopgap, not the design.** Where a
  type's optimal layout on a device has no matching indexed kernel yet (e.g.
  q6_K SOA/coalesced MMID today), the short-term fix is consistent AOS
  (materialization, advertisement, and kernel all agreeing); the design-conformant
  endgame is materializing the optimal layout *and* providing its kernel, the
  way MXFP4 already has layout-aware `_id` launchers. Track the gap; don't
  let the stopgap quietly become the architecture.

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

### oneDNN Graph allocations are a cache consumer too (llama.cpp-gwno)

The primitive-API scratchpad (`gemm.hpp`'s `scratchpad_mode::user` +
`get_scratchpad_mem()`) already routed through `unified_alloc()` into the
ONEDNN VRAM zone. Until llama.cpp-gwno, the **Graph API** did not: the
compiled SDPA partition's ~12 MiB scratch/intermediate was allocated by
oneDNN's own default SYCL allocator (`sycl::aligned_alloc_device`) on every
`dnnl::graph::sycl_interop::execute()` — a real `zeMemAllocDevice`/
`xe_vm_bind` round trip, invisible to the unified cache's own traces, firing
35x per ubatch on the B70 pp512 and blocking the dispatch thread ~2.7 ms each
time (S4 root cause, `llama.cpp-jmc5` comment `c-uxch`).

The fix builds the oneDNN engine (`ggml_backend_sycl_context::make_engine()`,
`common.hpp`) with a `dnnl::graph::allocator`
(`dnnl::graph::sycl_interop::make_allocator()` +
`make_engine_with_allocator()`). Since `dnnl::graph::engine` is literally
`dnnl::engine`, one engine object backs both the primitive API and the Graph
API, so this one change reaches both. The allocator's two C callbacks
(`ggml_sycl::onednn_graph_sycl_malloc/free`, `unified-cache.cpp`) have no
user-data slot to carry a `this` — the oneDNN C API doesn't offer one — so
they resolve the owning `unified_cache` themselves from the `sycl::device`
oneDNN hands back (`ggml_sycl_get_device_id_from_device()`) and forward to
`unified_cache::onednn_graph_scratch_alloc/free()`.

Those two methods serve requests from the **same ONEDNN zone** the
primitive-API pair already uses (`zone_alloc`/`zone_free` — host-side TLSF
suballocation, no device syscall) rather than a separate reservation.

The free path resolves to one of **three branches**, not two — getting this
down to two once already cost most of the win, and a second gap (TP) closed
it back to three:

1. **Zone-backed, TP not active — reclaim immediately, no completion-event
   check at all.** oneDNN hands the free callback a completion event, and the
   compiled partition's kernels may still be reading the buffer when the
   callback fires (it is called at submission, not at completion), so a first
   version of this allocator gated every `free()` (and every `alloc()`'s
   opportunistic drain) on `event.get_info<command_execution_status>()`. That
   call **blocks** rather than polls on a profiling-enabled queue (every
   backend compute queue is one; see `get_dma_queue()`'s comment in
   `unified-cache.hpp`) — and the event this allocator receives always comes
   from that same compute queue. The result was that every `free()` became a
   synchronous wait for that op's device completion, undoing most of the
   host-time win the allocator existed to deliver (measured: attention host
   stage stuck at 52 ms/ubatch instead of the expected drop). The fix removes
   the event check for this branch entirely and reclaims synchronously,
   justified by construction rather than by a host wait: the *only* consumer
   of a Graph-scratch buffer is a `dnnl::graph::sycl_interop::execute()` call
   built on `stream_dnnl(qptr)`, where `qptr == ctx.stream()` — the same
   in-order compute queue for every SDPA call, every layer, every ubatch. A
   later `zone_alloc()` handing the same block to a new `execute()` submits
   that execute's kernels on the identical queue, so the device enforces
   "finish reading the old contents before writing the new ones" via plain
   in-order submission order — the same guarantee an event wait would have
   bought, already free. Confirmed on hardware: attention host stage
   52 → 12.3 ms/ubatch, pp512 +15%.

2. **Zone-backed, TP active — defer via `enqueue_deferred_zone_free()`.** The
   branch above rests on one assumption: every consumer stays on
   `ctx.stream()`. `ggml_backend_sycl_context::stream()` (`common.hpp`)
   breaks that assumption under Tensor Parallelism — it returns the TP
   shared-context queue *ahead of* the unified-cache queue whenever TP is
   enabled for the device, so the queue `dnnl::graph::sycl_interop::execute()`
   actually submits on is not necessarily the queue this allocator's
   `malloc()` callback bound the engine to. Immediate reclaim would then be a
   real race, not a documented-but-safe one, so this branch fails closed
   instead: `ggml_sycl_get_tp_queue(device) != nullptr` routes to
   `enqueue_deferred_zone_free()` (the zone-suballocation analog of
   `retain_handles_until_event()` — that call takes a `mem_handle`, which a
   zone suballocation does not have; `enqueue_deferred_zone_free()` is the
   same primitive `reserve_onednn_scratch`'s own `defer_published_zone_release`
   already uses for exactly this "zone pointer, need an event-gated release"
   shape), gated on the real completion event (or a fresh barrier if none was
   supplied).

3. **DIRECT-fallback — always event-deferred, via
   `retain_handles_until_event()`.** If the zone has no room (an under-sized
   plan, or the arena not yet active), the allocator falls back to a real
   `mem_handle`-owned `unified_alloc()`. Unlike either zone branch above,
   immediate reuse here is *not* backed by the in-order-queue argument (a
   released DIRECT block goes back to the unified cache's general pool, not
   straight into this allocator's own reuse loop, so nothing pins its next
   consumer to the same queue as its last one) — correctness-preserving, but
   it reintroduces the round trip this change exists to avoid, so it is
   logged once, not per-call.

`unified_cache_get_planned_onednn_scratchpad_bytes()` adds a flat,
env-tunable floor (`GGML_SYCL_ONEDNN_GRAPH_ZONE_MB`) on top of the
primitive-API estimate at planning time, additive since both can be
resident at once — that with-floor getter is deliberately distinct from
`unified_cache_get_planned_onednn_scratchpad_bytes_stored()` (the bare
primitive-API-pair value with no floor added): `reserve_onednn_scratch`'s own
growth guard must compare against the stored form, or the floor silently
absorbs the pair's growth signal (every caller of either getter states in a
comment which one it wants). **Default 64 MiB** (superseded by
llama.cpp-0oxf: the default is now shape-derived, see the next section),
not the 512 MiB the allocator shipped with initially: that number was
sized for many buffers concurrently outstanding across a ubatch, which
was only ever true while branch 1 above deferred reclaim on the
completion event — once reclaim is immediate, at most ~1 buffer is
outstanding at a time in the common case (TP excepted, branch 2).
Measured high-water (`onednn_graph_scratch_high_water_bytes()`,
logged once at cache teardown, at `GGML_LOG_WARN` — `GGML_LOG_INFO` is
dropped at default verbosity in every tool, see CLAUDE.md's "llama-bench
traps" section, so nothing lower than WARN would reach a normal run's log at
all): 12.0 MB on a B70 gemma4 pp512 run, 3.0 MB on a two-model
`GGML_SYCL_STRICT_LEASES=1` run. The old 512 MiB default was not just
wasteful — it was a landing blocker: a second model's load-time zone-growth
request could exceed what the arena can grow into while the first model's
leases are still live, aborting VRAM-arena sizing on a run that passes on
master (which never needs this zone to grow at all). There is still no
cheap way to derive an exact per-model floor from this call site (that would
need the same structural (type, ne) classification zone-sizing.hpp already
does for the primitive-API pair, extended to a new consumer this change does
not attempt), so it stays a flat floor rather than a formula. Superseded by
llama.cpp-0oxf: the default is now shape-derived; see the next section.
`GGML_SYCL_ONEDNN_CACHE_ALLOCATOR=0` opts out of the whole allocator (back to
oneDNN's default, for A/B).

### The DIRECT path must be bounded, and the ONEDNN zone floor is now shape-derived (llama.cpp-0oxf)

The flat 64 MiB floor above was still too small for a real workload: at
n_kv up to 8192 (a long-prompt Mistral 7B Q4_0 `llama-bench` run), every
oneDNN SDPA `execute()` requests a ~144 MB Graph scratch — the zone's 256 MB
default holds ~152 MB of primitive-API pair, so the request never fits, and
**every single call** (32 layers × 16 ubatches × however many reps) takes
branch 3, the DIRECT fallback, not just an occasional under-sized one. That
turned a rare escape hatch into the steady-state path for that workload, and
exposed a bug branch 3's own reasoning had not accounted for: its release is
"always event-deferred, via `retain_handles_until_event()`" — onto the shared
background drain worker (`mem-handle.cpp`'s `retained_handle_drain_loop`,
a single detached thread that pops one retained-handle record, waits on its
event, and repeats). Under host CPU contention that worker starves. DIRECT
buffers pile up (each one's bytes stay counted as used by the unified cache
until the worker's wait completes and the `mem_handle` destructs), the next
`unified_alloc()` eventually fails, and the pre-0oxf code responded by
returning `nullptr` **silently** (`req.suppress_failure_log = true`) —
oneDNN then executed the compiled SDPA partition against an invalid scratch
pointer: a GPU page fault (`Engine memory CAT error class=ccs`), an engine
reset, then `UR_RESULT_ERROR_OUT_OF_RESOURCES` and a segfault. Reproduced and
bisected on a B50 under synthetic host load (20 busy loops): 3/3 faults on
the unmodified path, 0/3 with `GGML_SYCL_ONEDNN_GRAPH_ZONE_MB=256` raised
enough that the request fit the zone and branch 3 was never taken at all.

Two changes close this, and neither one alone would have been enough — a
bigger zone floor helps the common case but the DIRECT path still exists for
whatever a floor formula under-estimates, and a bounded DIRECT path with no
floor improvement would just abort sooner on exactly the workload this
ticket reproduced on:

- **The ONEDNN zone floor is now shape-derived**, not flat, and NOT
  context-alone — an earlier version of this fix anchored the floor to
  `n_ctx` in isolation, which the ticket's own follow-up measurement (comment
  c-xcop) showed was the wrong model: the same 144 MB request appeared at
  n_kv 2048, 4096 AND 8192 on Mistral 7B Q4_0's default ubatch, but a
  *smaller* request at a smaller ubatch and the *same* n_kv — meaning
  `n_ubatch` drives the size just as much as `n_ctx` does. Tracing
  `build_and_compile_sdpa()` (`fattn-onednn.cpp`) explains why: the
  intermediate f32 tensors oneDNN's fused SDPA partition must materialize
  carry `score_dims = {batch, H_q, ncols, ne11}` (or the 5-D GQA equivalent),
  where `ncols` is the per-call query-row count (`key.ncols = active_params.ne01`,
  i.e. the ubatch) and `ne11` is the per-call KV length
  (`key.ne11 = active_params.ne11`) — every distinct `(ncols, ne11, H_q, ...)`
  combination compiles its OWN partition via `sdpa_partition_cache`, and each
  can request its own scratch the first time it executes.
  `onednn_graph_scratch_zone_floor_bytes_swa()` now computes, per
  attention window class,
  `floor = max(64 MiB, 1.5 x max(n_head_ctx_max x n_ctx,
  n_head_swa_max x min(n_ctx, n_swa + n_ubatch)) x n_ubatch x
  sizeof(f32))`, where
  the `1.5x` factor covers ~5 SDPA scratch buffers measured concurrently
  in flight even on an idle host (the quantity the zone must actually
  hold is the PEAK OUTSTANDING size across whichever of those shapes are
  alive at once, not one request in isolation, and across whichever
  attention CLASS demands more, not their sum — the two classes' compiled
  partitions are never both outstanding for the same request).
  `n_head_ctx_max`/`n_head_swa_max` (max query-head count across layers
  eligible for the oneDNN SDPA route, split into non-SWA vs SWA) are
  threaded in from `hparams.n_head(il)`/`hparams.is_swa(il)` at
  `llama_model_sycl_populate_inventory()` (`src/llama-model.cpp`) through
  the `ggml_sycl_tensor_inventory.n_head_ctx_max`/`n_head_swa_max` ABI
  fields (llama.cpp-o3a0, appended at the end of the struct so every
  existing zero-init call site stays correct; these replace the earlier
  single `n_head_max` field), `placement_kv_info::n_head_ctx_max`/
  `n_head_swa_max`, and `placement_plan::planner_n_head_ctx_max`/
  `planner_n_head_swa_max`, mirroring how `n_ubatch`/`n_ctx` already flow
  through those same three layers. "Eligible" replicates
  `ggml_sycl_flash_attn_ext_onednn_plan()`'s D-based gate
  (`fattn-onednn.cpp`) since `llama-model.cpp` cannot include that
  SYCL-only source directly; the two copies are kept in sync by
  `test-sycl-onednn-graph-floor-eligibility-source.py`. Fed by
  `unified_cache_set_planned_onednn_graph_scratch_shape()` at the same
  "oneDNN scratchpad:" planning step (`populate_host_zone_sizing()`,
  `unified-cache.cpp`). Fit to five Mistral 7B Q4_0 (n_head_ctx_max=32,
  no SWA layers, so the ctx term equals the old flat formula exactly)
  measurements spanning two independent axes (48/192/768 MB at ubatch
  512 and n_ctx 512/2048/8192; 96/48 MB at n_ctx 2048 and ubatch
  256/128) — all five still match to the exact byte; a sixth gemma4
  point (n_head_swa_max=8) did not match the FLAT formula this replaced
  (predicted 192 MB, measured 24 MB), but was explained rather than
  anomalous: gemma4's oneDNN-served attention layers are sliding-window,
  so their real `ne11` is `min(n_ctx, n_swa + n_ubatch)`, not `n_ctx`
  and not `min(n_ctx, n_swa)` either — a ubatch of `n_ubatch` queries
  against an `n_swa`-key sliding window spans `n_swa + n_ubatch` keys in
  total (the first query looks `n_swa` keys back, the last one
  `n_ubatch` further along), and the oneDNN SDPA scratch scales with
  that whole span. llama.cpp-o3a0 closed the flat-formula gap first,
  using `min(n_ctx, n_swa)` (`n_swa` alone) — GPU-verified WRONG on the
  B50 (`GGML_SYCL_DEBUG=1` probe of gemma4 E4B's own SWA layers): at
  `n_ubatch=512` the measured Graph-scratch requests were
  24.00/12.00 MB (high-water 24.0 MB), and at `n_ubatch=256` they were
  9.00/6.00/3.00 MB (high-water 9.0 MB). Solving
  `1.5 x 8 x n_ubatch x K x 4 B` for `K` gives exactly
  `K = n_swa + n_ubatch` both times against gemma4's real `n_swa=512`
  (1024 = 512+512; 768 = 512+256) — which also resolves this section's
  own historical 24 MB measurement exactly
  (`1.5 x 8 x 512 x (512+512) x 4 B == 24 MB`). The formula now uses
  `min(n_ctx, n_swa + n_ubatch)` and takes the max across classes, so
  gemma4 (at its real `n_swa=512`, `n_ubatch=512`) computes 24 MiB raw,
  clamped to 64 MiB, matching the probe exactly — an 8x reduction in
  modeled demand vs. the flat formula's 192 MB (still 3x after the
  64 MiB clamp).
  ⚠️ **This gemma4 example describes the formula's behavior for the
  `n_ctx` it is actually given at PLANNING time, which is NOT today's
  real runtime context by default.** The floor is computed once, at
  MODEL LOAD (`populate_host_zone_sizing()`), from
  `plan.planner_n_ctx = kv_info.n_ctx`; at that point `kv_info.n_ctx` is
  `llama_model_sycl_populate_inventory()`'s conservative
  `inventory.n_ctx = inventory.n_ubatch` default (both 512) -- not
  `-c 8192` or whatever a caller eventually requests. The placement
  envelope carries its own `n_ctx` field, but it is set to 0 at load
  (`llama_model_sycl_make_placement_envelope()`) and its only reader
  anywhere is a diagnostic log line
  (`compute_placement_plan()`'s "[PLACEMENT] envelope..." print) --
  unlike `n_ubatch`, which the envelope DOES feed into
  `planner_n_ubatch` when set (`envelope->n_ubatch`, a separate field),
  nothing today threads a real `n_ctx` into `planner_n_ctx` at all.
  `ggml_backend_sycl_set_runtime_context()` later updates
  `planner_n_ctx` and KV/VRAM accounting for the real context, but does
  NOT call `unified_cache_set_planned_onednn_graph_scratch_shape()`
  again, so the ONEDNN Graph-scratch shape (and this floor) stays frozen
  at its load-time value regardless. So the SWA window term only
  actually narrows (`n_swa + n_ubatch < n_ctx`) once the Graph-scratch
  shape is re-planned against the real context, which no code path
  does today --
  the 24 MiB/192 MiB comparison above is the formula's behavior at
  whatever `n_ctx` it is given, not what a default run computes today.
  Re-planning the Graph-scratch shape on a runtime context change is
  tracked separately (llama.cpp-fkpg); the `[SYCL-PLAN]` floor log line
  below prints the `n_ctx` it actually used, which is what makes this
  gap visible in a real log -- provided the run captures `GGML_LOG_INFO`
  output at all, which is dropped at default verbosity in every tool
  (`-v` on `llama-bench`, or a raised verbosity threshold elsewhere).
  `GGML_SYCL_ONEDNN_GRAPH_ZONE_MB` still always
  overrides the formula, unchanged from before. The planned zone (pair +
  floor) is further clamped to 25% of the device's available budget —
  floored at the primitive-API pair's own bare requirement, since that pair
  has no DIRECT-path fallback of its own and clamping below its needs would
  starve the GEMM path rather than just the Graph-scratch floor — logging a
  `[VRAM-ARENA]` `GGML_LOG_WARN` naming the shortfall once if this triggers;
  the DIRECT path (next bullet) absorbs whatever the clamp removes. The
  per-request debug print in `onednn_graph_scratch_alloc()`
  (`GGML_SYCL_DEBUG`) is now gated on "differs from the last size printed"
  rather than a first-64 count, so a long run correlating shapes against
  sizes never goes silent partway through. **Deliberate deviation from the
  fix spec's literal "debug print of every request size" wording:**
  printing truly per-call would flood the log for a
  workload that repeats one shape thousands of times (every ubatch at a
  stable KV length, across `-r N` benchmark reps), while the shape space
  this print actually needs to surface is small — one entry per distinct
  compiled-partition shape, not per call — so "per distinct size" is a
  deliberate, documented substitution for "per request", not an oversight.
- ⚠️ **The zone floor above cannot actually reach the DIRECT path's steady
  state, and this is not a bug to fix in the floor** — it is a fact about
  when the arena is built. Verification found that at `-p 8192 -ub 512` on
  Mistral Q4_0, requests still took the DIRECT path (`did NOT fit the ONEDNN
  zone`) despite the shape-derived floor, because the arena (and its ONEDNN
  zone) is sized at MODEL LOAD, where `n_ctx` is still the conservative `512`
  default — the real runtime `n_ctx` (from `llama_context` creation) arrives
  much later, and nothing re-plans the zone at that point.
  `arena_reserve()`'s own body proves why not: its `if (arena_base_) { ...;
  return true; }` branch, taken on every call once the arena already exists,
  ignores the `scratch_bytes`/`onednn_bytes`/`runtime_bytes` arguments it is
  passed entirely — it only calls `zone_reclaim()` on the KV/RUNTIME zones so
  a new context's allocations can reuse existing physical capacity. There is
  no path in this design that grows an existing zone. (KV does not have this
  problem for an unrelated reason: it is not a small fixed-size zone sized by
  a formula the way ONEDNN/SCRATCH/RUNTIME are — it draws from whatever
  headroom the arena's construction already set aside, so a different
  context's real KV need is satisfied by fresh suballocation into that same
  pre-existing space.) The only alternative — destroying and rebuilding the
  whole arena once the real `n_ctx` is known — is refused by
  `ensure_planned_arena_zones()`'s own live-allocation check the moment any
  weight is resident, which by runtime context creation time it always is.
  So for any model whose real `n_ctx`/`n_ubatch` shape needs more than the
  load-time floor provisioned, EVERY oneDNN SDPA call of that shape takes the
  DIRECT path — not an occasional one — making the path below's efficiency,
  not just its safety, load-bearing.
- **The DIRECT path itself is now bounded, REUSES freed buffers, and fails
  loudly instead of silently.** Given the previous bullet, a background
  wait-for-a-generic-drain-worker design (this fix's first iteration) just
  means repeatedly paying a `zeMemAllocDevice` round trip for buffers of the
  identical size, every single call. `onednn_graph_scratch_free()`'s DIRECT
  branch instead PARKS a freed buffer in a size-bucketed reuse pool
  (`onednn_graph_scratch_reuse_pool_`: `size -> {mem_handle, release event}`
  entries) instead of handing it to the shared background drain worker via
  `retain_handles_until_event()`, and `onednn_graph_scratch_alloc()` checks
  that pool (for a completion-confirmed entry of the EXACT requested size --
  see the completion-flag bullet below for what "confirmed" means)
  before ever falling to a fresh `unified_alloc()`. This serves the common
  case for free: SDPA calls of a fixed `(n_head, ncols, ne11)` shape repeat
  across `-r N` benchmark reps and across decode steps at a stable KV length.
  A pooled-but-idle entry's bytes stay charged against
  `onednn_graph_scratch_direct_outstanding_bytes_` (a SEPARATE ledger from
  the zone/DIRECT maps used for the high-water stat above, because those
  clear the moment `onednn_graph_scratch_free()` is *called*, whether or not
  the memory is actually released at all — under this redesign it may now
  live in the pool indefinitely) — it is still real resident VRAM, just idle
  — so the pool cannot grow without limit alongside fresh allocations; it
  competes with them for the same `GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB`
  headroom (default: min(1 GiB, 25% of `available_budget()` snapshotted the
  last time this device's arena was successfully planned — a snapshot, not a
  live read, so the cap does not shrink out from under the allocator as the
  arena's own zones consume the budget it was planned against). A request
  that cannot be served from the pool and would exceed the cap evicts (a REAL
  release, destructing the owned `mem_handle`) completed pool entries — of
  any size, including the requested size's own bucket (a same-size entry that
  is complete but fails this request's alignment is exactly the kind of entry
  this sweep can evict — bucket iteration order is otherwise unspecified),
  stopping as soon as it fits — preserving as much of the pool as possible —
  and waits (bounded, dropping its own mutex so `onednn_graph_scratch_free()`
  — potentially called from a different thread — can keep parking newly-freed
  entries this wait might evict on its very next poll) for an in-flight entry
  to complete if none are immediately evictable. A single request larger
  than the whole cap by itself is a separate early-out: the eviction sweep
  above still runs (a real release of anything it can evict), but the
  bounded wait is skipped entirely — logged once — since no amount of
  waiting could ever make it fit, and the allocation is then attempted
  directly, same as after a timed-out wait. If the wait times out, the
  allocator proceeds anyway rather than refusing a possibly one-off spike
  pre-emptively; `unified_alloc()` below is still checked. If that allocation
  genuinely fails even after one drain-and-retry, the allocator now logs the
  sizes (request, outstanding, cap, prior waits, zone capacity/used/floor)
  and `GGML_ABORT`s — it no longer returns a null scratch pointer to oneDNN
  under any circumstance. `onednn_graph_scratch_direct_wait_count()` and
  `onednn_graph_scratch_pool_hit_count()` report how often a run actually had
  to wait, and how often it was served from the pool instead, respectively.
- **Pool completion checks go through a host-visible flag written by a
  device marker kernel, not a direct SYCL event query, and not a
  host_task either (llama.cpp-c6ah).** `event_complete()`'s bare
  `command_execution_status` query BLOCKS rather than polls on any
  profiling-enabled queue, and every backend stream — including the one
  oneDNN's free callback supplies `release_event` on — is
  profiling-enabled (see that function's own comment). Querying
  `release_event` directly from the reuse pool therefore WAITS for an
  in-flight SDPA kernel to finish instead of skipping it, defeating the
  "park and skip in-flight entries" design; this bare-query block remains
  the fallback for an entry whose flag could not be armed. The first fix
  attempt armed a `host_task`, on a new, lazily-created
  `get_event_watch_queue()`, `depends_on()`-ing `release_event`. That was
  measured (both discrete cards, 2026-09-09) to have a fatal flaw no GPU
  test caught: SUBMITTING a host_task whose `depends_on()` names an event
  from ANOTHER queue BLOCKS THE SUBMITTING THREAD until that event
  completes — so `onednn_graph_scratch_free()` itself stalled for the
  parked SDPA kernel's full duration on every single park, worse than the
  blocking-query bug this design exists to fix. The shipped design instead
  arms a DEVICE MARKER KERNEL on that same watch queue —
  `depends_on(release_event)` then a `single_task` writing a per-park
  generation value into a slot of a small, fixed-capacity host-USM slab —
  because submitting a device kernel the same way does NOT block its
  submitting thread (measured, both cards). The slab is owned via the
  sanctioned allocation path (`unified_cache_malloc_host_tracked()` +
  `unified_cache_adopt_raw_host_allocation()` with `cache_backing=true`) as
  the SECOND of exactly two allowlisted `CACHE_BACKING` mints — the first is
  the cache's own staging buffer; see
  `docs/design/sycl-canonical-memory-architecture.md` §3.1. `CACHE_BACKING`
  is required here, not merely reused as a convenient existing pattern: the
  slab must survive destructive teardown the same way the staging buffer
  does, because a marker kernel already in flight can hold a raw pointer
  into a specific slot at the moment shutdown runs, and `EXTERNAL_EXACT`
  carries no exemption from the pre-teardown census's live-allocation
  refusal. It is sized once and never reallocated
  — growing it later could leave an in-flight marker kernel's already-held
  raw pointer writing through a dangling host pointer; a free list plus the
  per-park generation tag make
  a slot safe to hand to a different pooled entry once its previous
  occupant is popped or evicted. `onednn_graph_scratch_clear_pool_locked()`
  (llama.cpp-c6ah) PERMANENTLY retires the slot of any entry it finds
  armed but still incomplete when it runs — never returning that slot to
  the free list — because the stale occupant's own marker kernel could
  still fire afterward and overwrite a new occupant's already-complete
  generation; the running total is exposed both in the DIRECT pool
  summary line's `retired_flag_slots=%zu` field and via the
  `onednn_graph_scratch_flag_slot_retired_count()` accessor, and a
  process that eventually checks out every slot this way degrades to the
  blocking `event_complete()` fallback with a once-only WARN, the same
  fallback path an individual arming failure already uses. Every reader
  of a pool entry's completion —
  `onednn_graph_scratch_entry_usable_locked()`,
  `onednn_graph_scratch_evict_pool_until_fits_locked()`, and
  `onednn_graph_scratch_clear_pool_locked()` — goes through
  `onednn_graph_scratch_pool_entry_release_complete()`, which prefers this
  flag and falls back to the bare `event_complete()` query only when no flag
  could be armed for that entry (slab allocation failed, every slot was
  checked out, the marker-kernel submit threw, `get_event_watch_queue()`
  returned `nullptr`, or the entry's own release event was null). A
  test-only hook,
  `ggml_sycl_test_onednn_graph_scratch_force_blocking_pool_check()`, makes
  the park site skip arming the flag so a GPU test can demonstrate that
  fallback directly. Confirmed on hardware, both discrete cards this fork
  validates against (`tests/test-sycl-event-status-blocking-probe.cpp`,
  isolated from the pool allocator entirely): the blocking behavior is
  specific to a DEVICE-KERNEL-produced `release_event` on a
  profiling-enabled queue (B50: query time 118 ms against a 122 ms kernel;
  B70: 110 ms against 114 ms) — a host_task-produced event, or a
  device-kernel event on a non-profiling queue, both measured a few ms or
  less either way. Every real pool release event in production is
  device-kernel-produced (oneDNN's own free callback), matching the case
  that actually blocks.
- **The pool is bounded per size, and reclaimed at every point that could
  otherwise leave it stale.** Nothing but the byte cap bounds how many
  buffers of ONE size the pool could hold, so `onednn_graph_scratch_free()`
  also caps each size bucket at `onednn_graph_scratch_pool_depth_per_size()`
  entries (default **8**, env `GGML_SYCL_ONEDNN_GRAPH_POOL_DEPTH_PER_SIZE`)
  — a workload that walks many distinct sizes (a pp8192 run touches ~16
  distinct ne11-derived shapes) cannot grow the pool's footprint without
  limit just because each individual size stays under the byte cap; a size
  whose bucket is already at the depth limit releases the overflow buffer
  for real via the shared event-gated drain path instead of parking it.
  Because the pool is a `unified_cache` member (survives across contexts and
  models), it is reclaimed (real release of every entry,
  `onednn_graph_scratch_reclaim_pool()`) at four production points: cache
  teardown (`shutdown_resources()`); the point `arena_reserve()` reclaims the
  KV/RUNTIME zones for a new context; `ggml_backend_sycl_set_runtime_context()`
  (`ggml-sycl.cpp`) on every successful runtime `n_ctx`/`n_ubatch` update —
  unlike the context-reclaim site above, this one does not go through
  `arena_reserve()` at all; and, llama.cpp-me60,
  `shutdown_unified_cache()`'s own pre-teardown pass, once per live cache
  unless SYCL is already shutting down or that cache's queue context is already
  invalid (llama.cpp-3lgu), BEFORE that function's pre-teardown census: a
  parked DIRECT buffer keeps its own `EXTERNAL_EXACT` allocation control alive
  until reclaimed, and that census refuses shutdown while any such control
  survives, so this pass runs ahead of it rather than relying on
  `shutdown_resources()`'s own, later reclaim. The runtime-context-update site
  reaches it through the free-function wrapper
  `unified_cache_reclaim_onednn_graph_scratch_pool()`; the context-reclaim and
  pre-census sites call the `onednn_graph_scratch_reclaim_pool()` member
  directly. Without the runtime-context-update site a pooled buffer sized for
  one context's shapes could sit on a 16 GB card holding up to the cap's worth
  of idle VRAM while the next model loads (`llama-bench` with several `-m`, a
  server switching models) or while the SAME model's context is resized to a
  different `n_ctx`. The four production sites do NOT all log in the same order
  relative to the clear: the context-reclaim, runtime-update, and pre-census
  sites share `reclaim_pool()`, which clears the pool and only then logs the
  summary; teardown instead logs the summary early, well before it actually
  clears the pool — see `docs/backend/sycl-env-vars.md`'s
  `GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB` row for the exact per-site ordering.
  Either way the line logged is `[UNIFIED-CACHE] oneDNN Graph scratch DIRECT
  pool summary (%s): hits=%zu misses=%zu evictions=%zu waits=%zu
  peak_pooled=%.1f MB retired_flag_slots=%zu (cumulative for this process, not
  just this reclaim)` (silent if the pool was never used), where `%s` is
  `"teardown"`, `"context reclaim"`, `"runtime context update"`, or `"module
  shutdown (pre-census)"`. Only the teardown call logs at
  `GGML_LOG_LEVEL_WARN`; the other three calls all log at
  `GGML_LOG_LEVEL_INFO`, which is dropped at default verbosity in every tool
  (see CLAUDE.md's "llama-bench traps" section) — so those three summaries are
  invisible in a normal run unless verbosity is raised. The summary call passes
  the enum `GGML_LOG_LEVEL_WARN` directly to `ggml_log_internal()` because its
  level is a runtime choice — WARN at teardown, INFO otherwise — which the
  level-baking `GGML_LOG_WARN`/`GGML_LOG_INFO` macros cannot express. The
  high-water line above it has a single fixed level and uses the
  `GGML_LOG_WARN` macro instead.

Two ALWAYS-compiled (not gated behind a `_TESTING` object-library variant —
see `ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail()`/
`_suppress_abort()`'s declarations in `unified-cache.hpp`) test hooks let
`tests/test-sycl-onednn-graph-scratch-direct.cpp` drive the abort decision
deterministically without needing genuine VRAM exhaustion and without
crashing the test process — a fork()-based death-check was considered and
rejected: forking a process that has already touched the SYCL/Level-Zero
runtime is a documented hang hazard on this fork's development host
(CLAUDE.md's SYCL Device Selection section). The pool-reuse and
eviction/wait properties need no such hook: they are exercised directly by
freeing and re-requesting real DIRECT allocations at controlled sizes.
**Deliberate, stated deviation:** the fix spec's
"genuine exhaustion aborts loudly" property is exercised through this
forced-fail hook, not real VRAM exhaustion — the test never actually drains
a card's VRAM. This is a sound proxy for the property under test (the
allocator's response to a failed `unified_alloc()` does not depend on WHY it
failed), not a weaker substitute standing in for a stronger test that was
skipped; but it should not be mistaken for direct evidence that a real OOM on
this specific path aborts loudly on real hardware, only that the code path
reached when `unified_alloc()` returns failure does. Since the two setters
are gated behind `GGML_SYCL_ONEDNN_GRAPH_TEST_HOOKS=1` (set by this test's own
ctest registration, and by the test binary itself via `setenv()` so a bare,
non-ctest invocation does not silently no-op the hooks and false-fail), a
production process cannot reach this simulated path at all.

### The one sanctioned exception: `ensure_cached_alloc()` (test-only)

`unified_cache::ensure_cached_alloc()` is the single allocation path that
**deliberately allocates outside the arena** — it gates on
`used_ + size > budget_`, evicts, and then calls `sycl::malloc_device` directly.
It has **zero production callers**; all 34 call sites are under `tests/`.

This does not violate the entry-point rule above: the implementation lives
*inside* `unified-cache.cpp` and hands back cache-owned memory, which is exactly
what the canonical contract requires. What it provides that nothing else does is
a **fail-closed, device-only, budget-gated, entry-creating** allocation with
`src_size` independent of `alloc_size` — the shape the cache fixtures need to
drive eviction refusal and allocation failure deterministically.

It carried `[[deprecated("use unified_alloc()")]]` until `llama.cpp-og9dt`. That
advice was wrong and the attribute has been removed: `unified_alloc()` creates no
cache entry (so `is_cached`/`evict`/`used()` cannot see the allocation),
`ensure_cached()` falls back to host memory rather than failing when eviction is
impossible (inverting every "refused under pressure" assertion, and host-resident
entries do not charge `used_`), and `allocate_slot()` prefers the arena with no
budget gate at all. The seam is retained by decision, not by inertia — full
adjudication in `llama.cpp-og9dt` comment `c-4lcs`.

**Do not add a production caller.** The zero-production-caller state is the
enforcement mechanism — a convention, not a constraint, so it is worth checking
rather than assuming:

```bash
grep -rnE 'ensure_cached_alloc\(' ggml/src src common tools examples \
     --include='*.cpp' --include='*.hpp' --exclude-dir=tests
```

Expect **exactly two** lines — the declaration in `unified-cache.hpp` and the
definition in `unified-cache.cpp`. A third line is the violation signature, and
it names the offending file. Verified both directions: two lines today, three
with a deliberately injected caller. `--exclude-dir=tests` drops both `tests/`
and `ggml/src/ggml-sycl/tests/`, which are legitimate callers. The pattern is
literal, so read a third line rather than trusting it — a production-side
*comment* that spells the name with parens would also appear.

Do **not** substitute a bare `grep -rn ensure_cached_alloc ggml/ src/ …`. That
returns 14 lines (the implementation's diagnostic strings, comments, the check's
own text, and a historical planning doc under
`ggml/src/ggml-sycl/docs/`), so it reads as a violation when nothing is wrong.

## Owner-first allocation: `alloc_owner` under the handle

`unified_allocate()` returns a `mem_handle` and is still the right front door for
most call sites. But it has an ordering property that turned out to matter: the
memory exists first, and something wraps it afterwards. If the wrap is skipped,
forgotten, or fails, there is a live allocation with no owner, and the only
record of it is a raw pointer somewhere.

**`unified_allocate_owner()` inverts that order.** It allocates the intrusive
`alloc_owner_control` *before* the physical allocation, so an allocation can
never exist without an owner:

```cpp
ggml_sycl::alloc_request req{ &stream, device, bytes, false, intent };
ggml_sycl::allocation_result allocation = ggml_sycl::unified_allocate_owner(req);
if (!allocation) { /* allocation.error is a typed allocation_error, not a bool */ }
ggml_sycl::mem_handle h =
    ggml_sycl::mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);
```

This is now the dominant shape in the backend, not a niche one: **48 call sites
at `0b7b49e07`** (25 in `ggml-sycl.cpp`, 23 across `common.cpp`, `common.hpp`,
`fattn.cpp`, `vram-pool.cpp`, `unified-cache.cpp` and the private fixtures).

What the owner adds over a bare handle:

- **`alloc_owner` is move-only**; the copyable form is `shared_alloc_owner`, and
  `mem_handle::from_owned_alloc()` moves the owner in via
  `std::move(owner).into_shared()`. Handle copies and `slice()` views retain
  *that exact intrusive control* rather than allocating a second `shared_ptr`
  control block — so a slice of a buffer is a real lease on the buffer's
  allocation, not a pointer alias with a hopeful comment.
- **Release is coordinator-mediated and can be refused.** Dropping the last
  reference calls into the device's `allocation_release_coordinator`, which
  returns a `release_attempt` (`RELEASED` / `RETRY_SCHEDULED` / …). A refused
  final release is queued through a `retry_next_` pointer embedded in the
  control, so the retry path allocates nothing — which is what lets it work
  under the memory pressure that caused the refusal.
- **Failure is typed.** `allocation_result::error` distinguishes
  `INVALID_REQUEST`, `CONTROL_ALLOCATION_FAILED`, `PHYSICAL_ALLOCATION_FAILED`,
  `METADATA_PUBLICATION_FAILED`, `RELEASE_RETAINED` — so a caller can tell "the
  request was malformed" from "the device is out of memory" without logging.

Each owner is also stamped with an `allocation_control_class` before the
coordinator admits it: `CACHE_BACKING`, `CACHE_SUBALLOCATION`, or
`EXTERNAL_EXACT`. That classification is the enforceable part of the contract and
is specified in
`docs/design/sycl-canonical-memory-architecture.md` §3.1 — including why
`CACHE_BACKING` cannot be requested through any public field, and the two
distinct mechanisms that can mint it.

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

### Optional layout copies yield to runtime KV (llama.cpp-jehw)

Weight reclaim has one authority, `weight_entry_reclaimable()`, and each
reclaim path names its purpose as a `weight_reclaim_mode`. Three modes reclaim
whole weights at model-load and teardown boundaries (`reset_model_weight_entries()`),
and there a live model's ownership (`owner_mask`) or a live buffer vetoes
reclaim even at `in_use_count == 0`. The fourth, `OPTIONAL_LAYOUT_YIELD`,
reclaims one physical layout of a tensor, not the tensor:

- **What it reclaims.** Only an entry marked `optional_layout`: a second
  physical layout staged beside the tensor's primary (today, a dense oneDNN
  WOQ copy). A primary is never reclaimable in this mode.
- **Why ownership does not veto it.** The tensor's owners, a live model or a
  live buffer, dispatch on the primary, which stays resident. Once the copy is
  gone, the route asks the cache and no longer offers the WOQ path, so the
  owner loses a faster PP route and keeps correctness. This holds for another
  loaded model's copies too.
- **What does veto it.** Any lease other than the cache's own direct-stage
  mirror, which the yield withdraws with the entry. A reader leases its copy
  through `unified_cache::acquire_layout_handle()` and keeps the lease until
  its work completes: `retain_handles_until_event()`, or the recording graph's
  sink for that graph's life. So a copy that is being read is not yieldable.
  The yield's queue barrier and the recorded-graph epoch drop are defence in
  depth behind that lease, not what makes the free correct.
- **Who calls it.** Only `yield_optional_layouts()`, at a context's KV
  admission. `reclaim_weight_entries()` refuses the mode, because it neither
  withdraws the mirror lease nor gates a free on readers.
- **Where it waits.** The KV admission transaction holds
  `g_tensor_inventory_mutex` (L1), and §12.5 of the canonical contract allows
  no wait and no destructor-running release under that lock. The yield is
  therefore split in two (`optional_layout_release`):
  - **Begin**, under the lock: pick the copies, retire them, and submit the
    barrier that gates their frees. It waits on nothing and drops no handle.
  - **Finish**, with the lock released: wait on the barrier, return the
    storage to its zone, and drop the withdrawn mirror handles.
  The transaction then takes the lock again. If a newer plan was published in
  between, the transaction reports busy, and the retired copies' room is there
  for the retry.
- **Beside the PP MoE oneDNN ring.** The KV fit reads
  `ggml_sycl_kv_capacity_live()`, which counts the ring's KV-zone slots as
  free: the ring is released and re-admitted after KV (llama.cpp-u1bb). The
  yield models the zone before that release, so it sees those slots as used.
  On a device holding both copies and ring slots, the fit therefore holds KV
  to fewer layers than the ring's room would take. The error is toward
  demotion, never toward a layer outside the arena. Only a MoE model whose
  dense weights are Q4_0 (the one WOQ type) has both.

## Where a weight's provenance comes from

A WEIGHT handle is keyed by `ggml_sycl_cache_id` — tensor identity, not pointer —
and that identity has to be *minted* by something. Today the only minting
occasion is a model load: `Registry::begin_outer()` issues a `ModelToken`, and
every weight identity row is filed under
`ggml_sycl_owner_name_key(owner, tensor_name)` so two loaded models cannot
collide on a shared tensor name. Everything downstream — expert cache keys,
residency reasons, ownership masks — reads that owner.

The consequence is easy to miss: **a tensor that never went through a model load
has no owner, so it cannot be admitted at all.** That is why ordinary backend
buffers (`test-backend-ops` cases, and anything else allocating through
`ggml_backend_sycl_buffer_type_alloc_buffer`) are refused rather than merely
slower.

### Buffer-scoped weight provenance

The fix adds a **second legitimate minting occasion — buffer allocation** — and
changes nothing else about the gates. Merged with the `llama.cpp-f8ws` lane
(2026-08-15); described here so the two mechanisms are not mistaken for one:

- **Minted at buffer allocation.** The buffer context already holds a unique
  per-allocation id (`managed_meta.id`, from the cache's own retention-identity
  counter). That id *seeds* a buffer owner; it cannot *be* the token, because
  `alloc_metadata` carries no model-identity field at all.
- **Namespaced by the top bit.** Buffer-scoped owner ids carry
  `1ull << 63`; the model `Registry` counts up from 1 and asserts the tag is
  clear, so the two id spaces are disjoint by enforcement rather than by
  assumption. Buffer owners take a sentinel slot outside the 32-slot model mask
  and so land in the *unattributed* ownership class — which is why the explicit
  drop below is load-bearing rather than belt-and-braces.
- **Consumed at `init_tensor`**, which already has the buffer context in scope:
  each tensor gets `extra->model_id` stamped and a row filed under
  `ggml_sycl_owner_name_key(buffer_owner, name)`, with a truthful synthetic
  parent identity (real byte offset within the buffer, real `ggml_nbytes`).
- **Dropped at buffer free via stored cache ids.** The buffer destructor **must
  not dereference tensors** — the `ggml_context` that owns them may already be
  gone (gallocr frees buffers first). So each tensor's `ggml_sycl_cache_id` is
  computed at registration time and stored on the buffer context as a plain
  vector; teardown drops by stored key and touches no tensor pointer. All the
  existing drop APIs are exact-key, so a stored key vector is required anyway.

No new reclamation mechanism is introduced: these are lookup-table erasures.
Memory release stays with `mem_handle` and the owner coordinator.

### The non-owning storage-handle route (merged)

There is a second, older answer for backend-buffer tensors that needs no cache
identity at all, and it is worth knowing because it is what the expert resolver
consults **first**.

`ggml_sycl_publish_backend_aos_expert_handles()` slices the buffer's own
`managed_handle` per expert — `ctx->managed_handle.slice(offset, expert_size)` —
validates every slice before publishing any of it (pointer matches the expected
address, on-device, correct layout, correct device, `has_stable_owner_identity()`
true), and stores the slices on the tensor's `extra`, keyed by
`moe_storage_handle_key(expert_id, layout)`. Publication is transactional: the
previous records are taken aside first and rolled back if any step throws, so a
partially installed prefix is not observable. It runs at the three points where
an AoS upload becomes complete — `set_tensor`, `buffer_cpy_tensor`, and the MMID
decode branch.

`ggml_sycl_resolve_moe_expert_route()` then tries
`ggml_sycl_try_moe_storage_handle_route()` for each candidate layout *before* it
forms any cache key; a hit returns `FOUND` and the unified cache is never
consulted. Two properties follow, and both are stronger than they look:

- **These slices are owning, but they are not cache entries.** Each is a real
  ref-counted lease on the buffer's allocation, so the bytes cannot go away
  underneath a kernel — and because the cache never learns of them, no eviction
  or free predicate can reach them either. "Non-owning" here means *the cache
  does not own them*, not that nobody does.
- **Layout discipline is structural.** The record key includes the layout, so a
  request for SOA, XMX or a packed layout **misses** rather than silently
  aliasing the AoS view.

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
- **Host-resident weights dispatch on CPU, not via GPU "zero-copy" — and not
  via per-dispatch staging copies either.** Feeding a host-pinned pointer to a
  GPU kernel is slower (measured 1.6–2.6×) *and* breaks the tier abstraction;
  streaming the weight into device scratch to dodge that is the same violation
  plus a copy. Let `resolve()` report residency and route accordingly. See
  "Placement decides the executor" above for the full rule and its scope.

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

The growth-path leak fix (above) moves the point-release earlier within this
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

## The VRAM budget authority (llama.cpp-o3h1)

`GGML_SYCL_VRAM_BUDGET_PCT` — how much of a device's VRAM the unified cache
is allowed to claim — used to be parsed and turned into a byte budget
independently at **six** different sites: the placement planner
(`compute_vram_budget_for_plan`, `ggml-sycl.cpp`), the unified cache's own
construction (`create_cache_for_device`, `unified-cache.cpp` — the one that
actually fed `arena_reserve()`'s physical reservation size), `arena_reserve()`
itself (a *second*, independent external-headroom computation layered on top
of whatever budget its caller handed it), the multi-GPU placement path
(`dev_budget`, with **no** headroom subtraction at all), and two
diagnostic/introspection sites that defaulted the percentage to 90 while
every allocation-governing site defaulted to 100. Nothing forced any of these
to agree, and two of them provably didn't: a clean-device GPT-OSS run at the
default `GGML_SYCL_VRAM_BUDGET_PCT=100` failed deterministically with
`UR_RESULT_ERROR_OUT_OF_RESOURCES` inside `FLASH_ATTN_EXT`/`MUL_MAT_ID`
dispatch, because the placement planner's promise (how much VRAM it told
itself it could fill) and the arena's actual physical reservation (after its
own, separately-computed external headroom) diverged by exactly the headroom
amount the arena withheld — traced to exact log-line arithmetic on both
sides. This is the same *disagreeing-independent-computations* disease as the
`llama.cpp-ytr7` stale-literal bug, recurring at a larger scale.

**The fix is structural, not a bigger constant.** `compute_vram_budget_authority()`
(`unified-cache.cpp`, declared in `unified-cache.hpp`) is now the **only**
function that parses `GGML_SYCL_VRAM_BUDGET_PCT`, derives the host-unified-
adjusted device total, caps to pre-probe free VRAM, and computes the external
headroom (via `unified_cache_max_external_headroom()`, itself a composition
of the arena's own device-proportional term and the oneDNN batched-pipeline-
aware floor from `vram-headroom.hpp`, conservatively assuming the pipeline
*is* planned since this authority runs before that flag is decided for the
current model). It returns a `vram_budget_authority` struct — `budget_pct`,
`base_mem`, `free_mem`, `external_headroom`, `budget_bytes` — and
`create_cache_for_device()` **publishes** that result onto the `unified_cache`
instance it constructs (`budget_pct()`, `external_headroom()`,
`authority_base_mem()` getters, alongside the pre-existing `base_budget()`).

Every other site now either calls `compute_vram_budget_authority()` directly
(only when no cache exists yet for the device in question — a defensive,
WARN-logged fallback, not the normal path) or reads the published values off
the live cache instance. Concretely: `arena_reserve()` no longer computes its
own external headroom at all — `budget_bytes` arrives already fully
headroom-adjusted by the authority, and a caller that already subtracted the
right amount must not have a second, independently-derived cap that can
disagree with it (that disagreement — `arena_reserve()`'s own
caller-reserved-headroom clamp collapsing toward a small value whenever
`budget_bytes` sat close to `device_total_vram` — is what actually produced
the error-40 failure above). `arena_reserve()` keeps only a passive sanity
floor (`alloc_size` may never exceed `device_total_vram` outright — a caller
bug, not a headroom policy question) and its pre-existing, unrelated
per-chunk-cap/N-chunk logic (a genuinely separate concern: the driver's
single-allocation hardware ceiling, not external headroom).

**The "only function that parses" claim above was false for a while and is
true again now.** `ggml_backend_sycl_device_get_memory` (`ggml-sycl.cpp`,
the public `ggml_backend_dev_t` device-memory-query vtable entry consumed
by `llama-model.cpp`/`arg.cpp`/`common.cpp` for params-fit) survived the
original consolidation as an uncounted **seventh** independent computation
— its own `getenv`+clamp+`max(256 MB, total/10)` headroom formula, gated
behind "only if the env var is explicitly set" (so at the common
unset-env-var default it reported the *raw*, unadjusted free VRAM to
params-fit, even though the cache was about to reserve headroom out of it
regardless). Found by the o3h1 spec review (`llama.cpp-o3h1` commit 6,
ticket comment c-53u5): measured disagreement with the authority on the
B70, 3265.6 MB vs 2048 MB headroom for the same budget percentage. Fixed
by folding it into the same reader *shape* `compute_vram_budget_for_plan`
already used — read `base_budget()` off the live cache; fall back to
`compute_vram_budget_authority()` only when no cache exists yet for the
device — and removing the "only if env var set" gate, since the authority
reserves headroom at any percentage, including the synthesized default.
This changes what params-fit sees at default settings — not a regression,
since the *cache* was always going to reserve that headroom; params-fit
had simply never been told about it before.

**Correction (spec re-review, same ticket comment c-53u5): the fallback
branch is silent, not "WARN-logged."** Commit 6's own commit message
described this fallback as WARN-logged by analogy with
`compute_vram_budget_for_plan`'s fallback branch, which does warn — but
the two branches are not the same case. `compute_vram_budget_for_plan`'s
fallback is reached only if the ordering invariant "caches are constructed
before placement-plan computation" is ever violated, which would be
anomalous and worth a WARN. `ggml_backend_sycl_device_get_memory`'s
fallback, by contrast, is the *expected, routine* path: device
enumeration and params-fit run before any model load creates a cache, so
hitting it is normal, not a violated invariant, and a WARN there would be
noise on every ordinary startup rather than a useful signal. The code is
correct as written (silent); this paragraph's earlier text and the commit
message for `299fd90c3` were wrong to describe it as warning — left
uncorrected in that commit's own message (git history is not rewritten),
fixed here.

This closes the single-authority half of the design the failure required.
The other half — making a still-insufficient budget a *handled* runtime
state instead of a crash, since no offline-chosen headroom constant can be
proven sufficient against an unqueried, driver/version/workload-dependent
kernel-submission requirement — lands in the same ticket's second commit:
investigation found the ggml-level evict/retry/tier-fallback machinery
already substantially exists (`unified_cache::ensure_cached()`'s weight-cache
eviction loop; `unified_alloc()`'s VRAM-pressure guard and zone-full
fallthrough for `RUNTIME`/`SCRATCH`/`KV` requests — the exact path
`FLASH_ATTN_EXT`/`MUL_MAT_ID` scratch buffers go through), correctly gated
off during `graph_compute_impl` (evicting VRAM a live kernel may still
reference is a `DEVICE_LOST` hazard, not a fix — see canonical contract §8.5).
The genuinely missing piece was narrower: a driver-internal kernel-submission
failure (§8.5) now triggers a per-op CPU-fallback recompute
(`ggml_sycl_cpu_fallback_graph`) instead of aborting, falling through to the
existing `ggml_sycl_fallback_error` → `GGML_STATUS_FAILED` clean-failure
contract if that also fails.

**That is not quite the whole terminal story, though — commit 4's
forced-pressure exercise (`6b5bfa8d6`, ticket comment c-s4na) found a third
outcome.** On this driver, a failed kernel submission can leave the *device
itself* hung (compute-runtime-internal behavior), and when it does, the CPU
fallback's own device→host staging copy queues behind that wedge and never
completes — so the "recompute via CPU" and "clean `GGML_STATUS_FAILED`"
paths above are both unreached. What actually terminates that case is the
pre-existing `[SYCL-WATCHDOG]` (`GGML_SYCL_OP_TIMEOUT_MS`), which performs
its designed bounded exit rather than leaving the process hung indefinitely
— the card was confirmed healthy afterward (no GT reset, no reboot), which
is the guarantee that matters. This is unreachable at honest budgets (the
authority above proves the condition doesn't arise by default); it is the
terminal floor only under deliberately hostile pressure. See canonical
contract §8.5 for the full three-outcome ladder — do not read this section
alone as implying transparent in-process recovery is always reached.

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

### A non-tensor consumer: the non-FA batched mul_mat scratch floor

Every consumer above sizes from a maximum over the *tensor inventory* —
`zone_scoped_maxima()`'s classifier groups tensors by `(type, ne)`. The
non-flash-attention batched mul_mat scratch demand
(`unified_cache_nonfa_attn_scratch_demand_bytes()`, llama.cpp-oyfl) is
**not** that kind of consumer: nothing it needs is a weight tensor. When
flash attention is off, each layer's attention runs through
`ggml_sycl_mul_mat_batched_sycl()` (`ggml-sycl.cpp`) twice — KQ =
`mul_mat(K, Q)`, then KQV = `mul_mat(V, softmax(KQ))` — and both calls
stage their non-f16 operand into an f16 buffer via
`scoped_unified_queue_temp`, whose shape comes from
`ggml_sycl_transient_device_intent()` (`common.hpp`): on the calling
thread's COMPUTE path — not recording a command graph, the common case
for this dispatch — that sets `prefer_vram_zone = SCRATCH`; while
recording, it takes the GRAPH_TMP shape instead and does not name SCRATCH
at all. This formula models the COMPUTE-path demand. The
KQV call's staged operand is `kq_soft_max`, shaped
`[n_kv, n_ubatch, n_head]` — at long context this dwarfs the KQ call's
staged Q operand — so the formula keeps only that term:

```
demand = max(16 MiB, n_head * n_ubatch * n_ctx * sizeof(f16) * 3)
```

**Rule for adding a consumer, extended:** when the demand scales with
*runtime shape* rather than any tensor's byte size, do not force it through
`zone_tensor_desc`/`zone_scoped_maxima` — follow the pattern this consumer's
sibling, `onednn_graph_scratch_zone_floor_bytes()`, already established: a
free function taking the shape directly (`n_head`, `n_ubatch`, `n_ctx`),
fed by a `unified_cache_set_planned_*_shape()`/`get_planned_*_shape()` pair
of atomics (one triplet per consumer — the non-FA one is **not** a reuse of
the oneDNN triplet, because it is unconditional while the oneDNN one is
gated behind `GGML_SYCL_DNNL`), called from `populate_host_zone_sizing()`
right where `plan.planner_n_head_all_max/n_ubatch/n_ctx` are already known --
`planner_n_head_all_max` (llama.cpp-rqak) is the max query-head count over ALL
attention layers, distinct from the oneDNN-eligible-only
`planner_n_head_ctx_max`/`planner_n_head_swa_max` pair above, because this
guard's non-FA path runs on every attention layer regardless of oneDNN
eligibility.

**The c=3 concurrency factor is MEASURED from the llama.cpp-oyfl repro
log, not carried over by analogy.** An earlier version of this formula
copied the oneDNN floor's c=1.5 by analogy, and it under-covered the
repro: the log (session scratchpad `oyfl/fa_off-2026-09-07.log`) shows the
SCRATCH zone at 460.0 MB of its 512 MB capacity immediately before the
failing request, whose own raw (no concurrency factor) demand at that
shape (n_head=32, n_ubatch=512, n_ctx=8192) is exactly 256 MiB — so the
real pressure was `460 + 256 = 716` MiB against a 256 MiB single-shot
request, a ratio of ~2.8x. c=3 clears that with a small margin. The
concurrency is not "one buffer plus headroom": `scoped_unified_queue_temp
::release()` (`ggml-sycl.cpp`) submits a marker event and keeps the handle
alive until it completes rather than freeing synchronously, and a
16-ubatch pp8192 run submits GPU work faster than each ubatch's release
event resolves, so several of the immediately preceding (smaller but
still substantial) ubatches' requests are typically still occupying the
zone when the largest one arrives. This is inferred from **one** repro's
zone-state snapshot, not fit to several independent hardware measurements
the way the oneDNN c=1.5 was (five captures, llama.cpp-0oxf comment
c-xcop).

**Two opposing pressures on this constant, both real.** Raise it only on
new hardware evidence (a real multi-point capture tracing SCRATCH_ZONE
occupancy across a whole pp8192 run) — never lower it without such
evidence, because lowering trades away the margin this check exists
to keep ahead of the abort it prevents. But raising it is not free
either: this is a headroom check, not a true worst-case model, so a
larger `c` also refuses **more** contexts that would actually have run
— trading false refusals for margin, on the same unvalidated single
snapshot. Neither direction is free; do not move this value without a
multi-point capture backing the move. `GGML_SYCL_NONFA_ATTN_SCRATCH_MB`
overrides the formula outright, so applying a future measurement needs
no code change.

**Where this can and cannot help, and why the automatic case is
llama.cpp-fkpg's scope, not this one's.** Like every zone above, the
SCRATCH zone is sized once, before the arena's single physical chunk is
allocated at model load — raising it later cannot resize that chunk.
`ensure_planned_arena_zones()` raises `scratch_zone` from this formula the
same way it raises `onednn_zone` and `runtime_zone` from theirs (bounded to
a quarter of the available budget, same reasoning as the oneDNN clamp), but
for the standard model-load flow this can never fire with the real shape:
`llama_model_sycl_make_placement_envelope()` (`llama-model.cpp`) hardcodes
`envelope.n_ctx = 0` unconditionally, so no caller's real `-c`/`-p` reaches
`plan.planner_n_ctx` before weights stream into the arena. Piping the real
context size back into pre-load sizing is llama.cpp-fkpg's scope — this
raise only takes effect automatically once that lands. `ggml_backend_sycl
_set_runtime_context()` also makes one opportunistic re-plan attempt with
the real runtime shape (`unified_cache_ensure_planned_arena_zones()`, the
same call `reserve_onednn_scratch()`'s own comment already documents as
succeeding only in "the rare case where the arena is still empty") —
cheap and harmless, but not expected to succeed once weights hold live
leases; its own log line only claims "raised" when the zone's capacity
actually grew, and otherwise reports only what was observed ("did not
raise the SCRATCH zone (... unchanged)") rather than naming a specific
cause this code never actually checked.

**An asymmetry between the two callers (llama.cpp-rqak), worth knowing
before touching either path.** An explicit `-fa 0` context goes through
the FULL transaction above and records its real runtime shape via
`unified_cache_set_planned_nonfa_attn_scratch_shape()` (restoring the
previous shape if the guard refuses); an AUTO context that resolves OFF
goes through `ggml_backend_sycl_recheck_runtime_context_flash_attn()`'s
narrow re-check instead, which deliberately records nothing (no replan, no
restore-on-refusal -- see its `allow_replan=false` call). So an
AUTO-resolved-OFF context is checked against whatever shape was recorded
earlier (at load time, or by a prior explicit `-fa 0` context), never its
own. This has no practical effect today, because the plan-time raise this
recorded shape feeds is already a no-op past model load for the reason
above -- weights hold live leases by the time any runtime-context call
happens, so there is nothing for a missed recording to have changed.

**The check is EMPIRICAL, not a modeled worst case, and has already been
revised once on new hardware evidence — read this before tightening or
loosening it.** A first draft of this predicate (llama.cpp-oyfl round 2)
reasoned from first principles that a SCRATCH-zone-capacity comparison
alone was insufficient: a zone overflow does not fail outright
(`unified_alloc()`, `unified-cache.cpp`, falls through to a raw device
allocation OUTSIDE the fixed arena when a preferred zone is full), and
that same outside-arena headroom is what the SYCL scheduler's own compute
buffer draws on when it regrows to the context's real `n_kv`
(`llama_context::graph_reserve()` reserves it at the KV cache's own
reserve-time `n_kv`, `llama_kv_cache::get_n_kv()`, sized from
currently-used cells — 0 at construction — not the real `n_ctx`; a
deliberate upstream design for graph-shape stability, not something to
change here). That draft modeled and summed every such term (compute
-buffer regrowth, SCRATCH overflow, oneDNN scratchpad) and compared the
sum against live free VRAM outside the arena, predicting the repro would
run once `GGML_SYCL_NONFA_ATTN_SCRATCH_MB=768` (the exact `c=3` demand)
eliminated the SCRATCH term (predicted available ~1334 MiB vs. ~1200 MiB
needed).

**Hardware measurement falsified that first model.** On the B50, the same
`-fa 0 -p 8192` repro with `GGML_SYCL_NONFA_ATTN_SCRATCH_MB=768` still
aborted (rc=134, staging-allocation failures, a VRAM-exhaustion line) — a
sweep across `768`/`1536` MiB zone sizes, `GGML_SYCL_VRAM_BUDGET_PCT=85`
(shrinking the arena to grow outside-arena headroom to ~4.0 GB), and an
explicit `GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB=3072` override all
still aborted, each time with tens of MB free rather than the modeled
headroom. The B70 (default ~2 GB outside its arena, and again with
headroom raised to ~4.0 GB) aborted identically. So the non-FA `-p 8192`
prefill consumed roughly 2–4 GB **outside the fixed arena on both discrete
cards**, far beyond anything the compute-buffer/SCRATCH/oneDNN model
above accounted for, and no zone size or headroom override recovered it —
filed as **llama.cpp-k1ev**, an unidentified outside-arena consumer
specific to the non-FA path. On the pre-o3a0 tree this was read as also
falsifying the model's original motivating concern (a card with more
outside-arena headroom, e.g. the B70, would wrongly refuse a shape that
actually runs): both cards aborted at the same shape there, so the
zone-only check that shipped (llama.cpp-oyfl round 3) was not observed to
be more conservative than reality on either card measured so far, and it
was adopted as the refusal: refuse when `nonfa_demand >
zone_capacity(SCRATCH)`, labeled **"scratch-limited"**.

**That zone-capacity predicate was itself falsified on 2026-09-09, on the
tree merging llama.cpp-o3a0 + oyfl + rqak (`f594574bf`) — task
llama.cpp-pvjr.** A bracketing sweep with the guard disabled
(`GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0`, so each shape runs or fails on its
own hardware merits, not the guard's opinion) on Mistral 7B Q4_0,
`-fa 0`, `n_ubatch=512`:

| card | `-p` | demand `d` | outcome |
|------|-----:|-----------:|---------|
| B50 (free 1627.3 MB) | 6144 | 576 MiB | ran clean |
| B50 | 7168 | 672 MiB | ran clean |
| B50 | 8192 | 768 MiB | **aborted** (VRAM exhausted, free=29 MB) |
| B70 (free 2045.2 MB) | 8192 | 768 MiB | ran clean |
| B70 | 11264 | 1056 MiB | ran clean |
| B70 | 12288 | 1152 MiB | 4 outside-arena staging failures, survived only on the scalar fallback |

`p6144`/`p7168` both exceed the 512 MiB SCRATCH zone yet ran clean on
BOTH cards, and the B70 additionally ran `p8192` clean — all three would
have been wrongly refused by the zone-capacity predicate, which keys off
a resource (this one SCRATCH zone) unrelated to what actually runs out
(outside-arena VRAM on the card with less of it). The B70's own
outside-arena growth above `p6144` is now explained by visible terms: the
scheduler's compute buffer (the f32 KQ tensor, which doubles there as it
scales with `n_ctx`) plus the batched-F16 `src1` staging buffer (up to
~370 MB at `p12288`) plus oneDNN scratch overflow (~126 MB) sum to
almost exactly its measured outside-arena headroom at `p12288`. The B50
still carries an extra, unattributed ~0.1–0.9 GB over the measured range
— llama.cpp-k1ev's consumer, now bounded far tighter than the pre-o3a0
"2–4 GB on both cards" figure, and apparently absent on the B70 within
this range.

**The refusal predicate is now: refuse iff `demand + reserve > free`**,
with `free` the device's LIVE free memory at guard time
(`ggml_backend_sycl_get_device_memory()`, not the arena's stored
`external_headroom()`, so a `GGML_SYCL_VRAM_BUDGET_PCT` override or
another tenant on the card — e.g. ComfyUI holding VRAM on the B70 —
changes the answer honestly instead of going stale) and `reserve` the
EMPIRICAL constant
`unified_cache_nonfa_attn_outside_arena_reserve_bytes()` = **928 MiB**.
That figure is bracketed from the sweep above: the B50 bracket (`p7168`
clean, `p8192` refused) gives `859 < R <= 955` MiB; the B70 bracket
(`p11264` clean, `p12288` degraded — treated as refuse, since a caller
that wants that context to run should not silently get a
perf-degraded one instead) gives `893 < R <= 989` MiB; the intersection
is `893 < R <= 955` MiB, and `R = 928 MiB` was chosen near its middle —
margins at the four measured bracket points are 27–69 MB, i.e. at the
sweep's own 1024-token resolution, not exact. `R` absorbs the oneDNN
overflow, the `src1` staging buffer, and the B50's residual k1ev
consumption, all **only over the measured range**; a shape well beyond it
is refused conservatively rather than extrapolated.

The demand term `d` itself is unchanged — still `n_head * n_ubatch *
n_ctx * sizeof(f16) * 3` (6 bytes/element) — even though the B70's own
visible consumers above scale closer to ~10 bytes/element (the doubling
f32 KQ compute buffer on top of the f16 staging term). A single `R` does
not fit both cards at that higher slope (at 10 B/elem the B70 would need
a fixed term ≤ 203 MB while the B50 needs > 289 MB — no single value
satisfies both), which is itself the k1ev signature: the B50 carries an
outside-arena consumer the B70 does not. Keeping `d` at 6 B/element and
letting `R` absorb the difference is what makes one constant work for
both cards across the measured range.

The zone comparison is dropped from the refusal entirely — it is exactly
what over-refused `p6144`/`p7168` above. The opportunistic SCRATCH-zone
re-plan (`unified_cache_ensure_planned_arena_zones()`) is unchanged; it
still runs for its own INFO logging, but only on the full transaction
(`allow_replan=true`) — never on the narrow re-check, and not at all when
`GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0`, since the explicit-0 skip returns
before either the re-plan or the fit/refuse decision. A refusal now
reports `needs` (= demand + reserve, broken out as `demand`/`reserve`),
`free`, and `over_by`, plus the largest-fitting `-c` at `capacity = free -
reserve` (`unified_cache_nonfa_attn_scratch_headroom_capacity_bytes()`,
labeled **"headroom-limited"**, replacing "scratch-limited" — it is
bounded by this device's own live outside-arena headroom, not a zone or a
whole-device guarantee). An explicit `GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0`
skips the guard entirely, before any comparison — `0 + reserve` compared
against `free` would otherwise refuse any card with less than 928 MiB
free, which is not what "disable" means. The refusal still names `-fa
1`/`auto` or a smaller `-c` as the only remediations with hardware support
— it deliberately does **not** suggest `GGML_SYCL_NONFA_ATTN_SCRATCH_MB`,
since that variable only replaces the demand term `d`, not the reserve
or the headroom comparison; treat it as an experimentation knob for
investigating k1ev, not a user-facing fix. Do not reintroduce a
zone-capacity-only or compute-buffer-regrowth-only predicate without
new hardware evidence that k1ev's consumer is understood and bounded
— the reasoning behind each retired predicate was sound in isolation
and still wrong in practice, which is the whole lesson of this subsection.

### A context-time consumer: the PP MoE oneDNN ring above the RUNTIME zone

The PP MoE oneDNN scratch ring is one weight slot plus two slots that scale with
`n_ubatch` (activation, output), times the ring depth. On GPT-OSS 20B it is
404.5 MB at `-ub 512` and 674.5 MB at `-ub 1024`. The RUNTIME zone is 512 MB.
So on the B50 `-ub 1024` was refused at the runtime-context update ("the
RUNTIME zone has 505.3 MB available; the largest -ub that fits is about 672",
`llama.cpp-ibj0`), even with ~1.6 GB of the shared KV/weight zone free.

**Why the RUNTIME zone is not grown at load (`llama.cpp-u1bb`).** Growing it at
model load looks natural and is wrong. The arena is one chunk with fixed zones,
`[shared KV+WEIGHT][ONEDNN][RUNTIME][SCRATCH]`, each with its own TLSF sized
when the arena is reserved. `ensure_planned_arena_zones()` resizes a zone only by
rebuilding an arena that holds nothing, and at context time the weights are
live. So a load-time growth G is subtracted from the shared zone for the
model's lifetime. The runtime-context transaction admits KV against that zone's
live free space (`unified_cache_kv_vram_available()`), and only then re-plans
the ring. Shrinking the ring then frees RUNTIME bytes, which KV cannot use. For
any G > 0 there is a `-c` whose KV fits in VRAM with the 512 MB zone and is
demoted to the host with the grown one. That is the micro-batch beating KV,
which the owner ruling forbids ("KV all-VRAM wins over a larger micro-batch").
The real `n_ctx` is not known at load (`llama.cpp-fkpg`), so no load-time amount
is safe.

**The rule: decide at context time, after KV.** The transaction admits KV first.
`ggml_sycl_replan_pp_moe_onednn_ring()` then places and admits the ring for the
runtime `n_ubatch` through `pp_moe_onednn_admit_ring()`
(`moe-scratch-admission.cpp`, a pure function):

- The weight slot(s) stay in the RUNTIME zone. If they do not fit, no `-ub` fits.
- Each ubatch-scaled slot kind stays in the RUNTIME zone if it still fits there,
  larger kind first (with two kinds that keeps the most bytes in RUNTIME).
  Otherwise it goes to the shared KV/weight zone.
- The RUNTIME zone is counted without what the transaction still places there
  after the ring: the MoE MMID workspace pools, when this update materializes
  them. It materializes them only for a route that can run them, the predicate
  `load_end` and the context-bind hook already use. In an ordinary build that
  route is closed, so this is 0.
- The KV-zone part may use only `headroom - reserve`. `headroom` is the device's
  KV capacity less the plan's device KV (with the allocator's per-layer slack).
  The capacity is `ggml_sycl_kv_capacity_live()`, the one number the KV re-fit
  and the all-VRAM `-c` hints also read: the live KV headroom
  (`unified_cache_kv_vram_available()`, read now, so after anything that yielded
  VRAM to KV), plus an admitted context's already-allocated KV, plus this
  device's ring slots in the KV zone. A refusal names the largest `-ub`
  (multiple of 32) that fits, as before.

B50 GPT-OSS figures, with 370.8 MB of RUNTIME left after the weight slot:

| `-ub` | activation | output | goes to the KV zone |
|------:|-----------:|-------:|--------------------:|
|   512 |      90 MB |  180 MB | nothing |
|  1024 |     180 MB |  360 MB | activation, 180 MB |
|  2048 |     360 MB |  720 MB | output, 720 MB |
|  4096 |     720 MB | 1440 MB | both, 2160 MB |

**KV wins, in both directions.** The ring never moves KV. It is admitted after KV
and refused rather than kept when it no longer fits. A later KV re-fit (a second
context, or a changed KV shape) counts this device's ring slots in the KV zone as
free, so the ring cannot be why KV is demoted. The re-fit then forces the ring to
be released and re-admitted after KV, even at an unchanged `n_ubatch`. The
all-VRAM `-c` hints count those slots as free too.

**Planned, charged, and inside the arena.** The planned placement is stored per
device next to the planned slot sizes. `reserve_pp_moe_onednn_scratch()` reads
it on every call, including the executor's per-dispatch re-reserve. It allocates
a flagged slot from the KV zone with `forbid_vram_zone_spill`, so a slot that
does not fit fails instead of spilling past the arena. The RUNTIME zone
requirement counts only the slots that live in RUNTIME. The transaction charges
the KV-zone bytes to the plan's `vram_bytes`, which the weight stager reserves
against. So the KV-zone part is admitted against the budget too: it must fit
what `vram_budget` leaves on the device once KV and the MMID pools are charged.
That room is read after the MMID budget check and before the charge, so the
published `vram_bytes` stays within `vram_budget`. A refused
re-plan restores the old placement with the old ring. A rollback of an accepted
one restores a ring that fit before, so it is not re-admitted against the
headroom. A rollback to an unchanged `n_ubatch` keeps a re-admitted placement:
it fits, but the published plan's charge is for the one before it until the next
transaction.

**What the ring holds is read from the ring.** The planned placement says where
the next reserve puts each slot. What the ring holds in the KV zone now comes
from its slots: each records the zone it was allocated from
(`unified_cache_get_pp_moe_onednn_kv_zone_bytes_held()`). The KV capacity, the
re-fit's forced re-admission and the `vram_bytes` charge all read that. The two
facts differ across a model load. The arena and the physical ring outlive a
model, and a load resets the planned placement to RUNTIME. So a load releases a
ring that still holds KV-zone bytes, because the reserve's "already sufficient"
path would otherwise keep it. A slot claimed by an in-flight dispatch refuses
that release, and it is never forced. The ring is then kept, KV still counts
what it holds, and the next re-fit re-admits it.

**Contiguity.** Each KV-zone slot is one allocation. So the whole KV-zone part
must also fit the zone's largest free block, read after the old ring is
released. That is an estimate, not a guarantee. The TLSF allocator's
`largest_free_block()` returns the head of its highest size class rather than a
scanned maximum. With two or more KV-zone slots (ring depth above 1, or both
kinds in the zone), a later allocation can take the exact-class fallback and
miss. The check covers the single-allocation case. For the rest, the reserve
fails, and the refusal says the allocator could not place the slots rather than
naming the size it just refused. An automatic `-ub` then keeps its last
accepted size; an explicit one fails context creation.

**The compute-buffer reserve is a known gap, not a solved term.** Neither the
compute buffers nor the flash-attention K/V conversion buffers are in the
placement plan (`llama.cpp-zhcn`). A compute buffer that misses the RUNTIME
zone does not go straight to the KV zone. Its RUNTIME request does not forbid a
spill, so it first falls through to raw device memory outside the arena, when
the physical-VRAM overcommit guard allows that. It reaches the KV zone ("Arena
RUNTIME zone full, runtime buffer ... allocated from KV zone") only when the
guard refuses. The guard decides which, not the plan (`llama.cpp-23mk`). When
the ring fills RUNTIME, GPT-OSS's compute buffer (about 404 MiB at `-ub 512`)
misses it and goes down that chain. `ggml-alloc` sizes
compute buffers from the graph at `graph_reserve`, which runs after the
transaction, and they are not in the plan's `vram_bytes`. So the transaction
cannot know their size. It holds back `k_pp_moe_ring_compute_reserve_bytes_per_row`
= 1 MiB per micro-batch row: the GPT-OSS figure (0.79 MiB per row) rounded up,
and scaled linearly with `n_ubatch` like the buffer. It keeps KV-zone room for
the case where the guard sends the buffer there. The reserve applies only when
part of the ring goes to the KV zone. The admission line names it at WARN:

```
[SYCL-PLAN] PP MoE oneDNN scratch ring for n_ubatch=1024 puts its activation slots (180.0 MB) in the shared
KV zone: <free> MB free there after KV, 1024.0 MB compute-buffer reserve (an estimate of 1.0 MB per
micro-batch row; compute buffers are not in the plan)
```

The honest fix is for the compute buffer to enter the plan. Until then the
reserve is a margin, not a measurement, and a model whose compute buffer grows
faster than 1 MiB per row can still find the KV zone short at `graph_reserve`.

Host tests: `ggml/src/ggml-sycl/tests/test-pp-moe-ring-admission.cpp` pins the
arithmetic. `tests/test-sycl-pp-moe-ring-kv-zone-source.py` pins the wiring, with
a mutation witness per check.

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
