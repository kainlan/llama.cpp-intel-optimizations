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
TLS-reset proof gate. `t5nq` is OPEN with merged
reviewed packed-K-sidecar code awaiting its live GPU failpoint/retry/teardown
gate; that gate has no foundation prerequisites and may close now. `otry`, not
`t5nq`, revalidates packed-K guarantees after foundations. “All
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
