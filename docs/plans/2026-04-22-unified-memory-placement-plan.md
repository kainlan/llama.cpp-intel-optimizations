# Unified Memory Placement Planning — Design Plan

**Date:** 2026-04-22 · **Branch:** `feature/sycl-coalescing` · **Epic target:** open

## Goal

Make the SYCL backend's `placement_plan` govern **every** byte allocated for
inference — weights, KV cache, MoE experts, inference infrastructure, **and**
the compute buffer / per-op scratch. Today the plan is a static weight-and-KV
blueprint that is computed before graph shape is known, and compute buffers /
activations are discovered post-hoc by ggml's scheduler. That asymmetry is the
root of multiple recent incidents (notably `llama.cpp-ioua6`, `llama.cpp-8gz7y`,
and the 16 GB attention-score allocation trap that motivated this plan).

Success means: **"if the plan says it fits, it fits."** One pre-flight pass
sees all memory demand, decides placement for everything, then allocation
proceeds deterministically in a known-safe order.

## Motivation

Three concrete incidents in the last 30 days have the same shape — plan-vs-reality
divergence for un-planned memory.

| Incident | Root cause | Bead |
|---|---|---|
| 20B @ `VRAM_BUDGET_PCT=30` fails with 16 GB alloc (after 14 s of pinned-pool churn) | Compute buffer is not in the plan; ggml asks for one contiguous buffer the arena cannot host | `llama.cpp-ioua6` |
| 120B compute buffer needs 16 GB but planner only reserves 1.5 GB scratch | Planner has no compute-buffer signal; sizes host scratch from `max_tensor_bytes` | `llama.cpp-8gz7y` |
| FA for 20B routed to CPU (dense-weight plan shadows FA supports-check) | Per-op "can we run here" logic consults dense-weight plan, but FA has no dense weights | `llama.cpp-15li2` (fixed 2026-04-22) |

Each incident was fixed by a targeted patch. None of them is a complete fix,
because the common ancestor — an incomplete plan — is still there.

`placement_plan` already has 10+ fields (`onednn_reorder_bytes`,
`moe_q8_workspace_bytes`, `expert_bias_bytes`, `dma_staging_pool_bytes`, …)
that track non-weight memory. This plan extends that pattern to the two
categories still outside the plan: **compute buffers** (activation/intermediate
memory between ops) and **per-op scratch** (dequant, Q8_1, …). Everything else
is already modeled.

## Current state (grounded in code, 2026-04-22)

### What the plan covers today

Per `ggml/src/ggml-sycl/unified-cache.hpp:189-310`:

```
placement_plan {
    // Per-weight decisions
    entries[]              // tensor name → {on_device, dev_id, bytes, expert_id}
    name_index_            // fast lookup

    // Per-layer decisions (three independent tracks)
    layer_device[layer]    // dense-weight placement (-1 = host)
    kv_device[layer]       // KV cache placement
    // MoE expert placement is in entries[] via expert_id >= 0

    // Fixed inference-buffer budgets (sized by static heuristics from weight stats)
    weight_vram_bytes / weight_host_bytes
    kv_vram_bytes    / kv_host_bytes
    onednn_reorder_bytes
    moe_q8_workspace_bytes
    expert_bias_bytes
    moe_routing_ids_bytes, moe_expert_ptrs_bytes
    dma_staging_pool_bytes
    onednn_scratchpad_bytes
    pp_pipeline_scratch_bytes
    cpu_quant_buffer_bytes
    graph_metadata_bytes
    tp_{ffn,attn,staging}_buffer_bytes
    max_tensor_bytes                  // used as sizing reference
}
```

### Arena zone layout (`unified-cache.hpp:48-66`)

```
[KV (bump→) ...free... (←bump WEIGHT)] [ONEDNN] [RUNTIME] [SCRATCH (bump→)]
```

- **KV + WEIGHT**: share a region; sized from the plan at model load
- **ONEDNN**: fixed 256 MB; fine
- **RUNTIME**: sized from the plan's MoE/DMA/PP fields at context creation
- **SCRATCH**: sized *empirically* via `reserve_scratch_pool()` / `reserve_compute_arena()` after `graph_reserve()` returns real demand

### Timeline today (half-migrated)

1. `compute_placement_plan()` is called at **model load** time from
   `ggml-sycl.cpp:6521`, using `inventory->n_ctx` (model's trained maximum
   — e.g., 131072 for GPT-OSS 20B). The user's `-c 4096` is not in scope.
2. `ggml_backend_sycl_set_runtime_n_ctx()` exists at `ggml-sycl.cpp:6562`
   to upgrade the plan from conservative-to-runtime n_ctx, but `grep -rn
   ... src/` confirms **nothing calls it**. It's orphan infrastructure.
3. `placement_plan::planner_n_ctx_is_runtime` flag tracks the upgrade —
   always false in practice.
4. VRAM arena is created from plan; zones sized from conservative n_ctx.
5. `graph_reserve()` builds the compute graph with real runtime n_ctx →
   scheduler computes actual compute buffer size.
6. `ggml_backend_sycl_notify_compute_buffer_sizes()` receives sizes **after**
   zones are fixed; arena reserves whatever space remains.
7. Compute buffer allocation races against available zone space.

Steps 5-7 are the un-planned region. When compute buffer > RUNTIME zone, the
code falls through to host-pinned with 2 GB chunk grows, then malloc_host
fails (case `ioua6`). When the compute buffer **fits**, it fits by luck of
the over-allocation at step 4.

### Open beads/plans already in this space

| ID | Title | Status | Role in target state |
|---|---|---|---|
| `llama.cpp-48330` | EPIC: Unified Cache & SYCL Memory Safety | open | Parent epic; absorb this plan as one of its deliverables |
| `llama.cpp-mubmt` | EPIC: unified_cache sole owner of SYCL memory | open | Residency-handle infrastructure this plan assumes |
| `llama.cpp-8gz7y` | Query ggml scheduler for actual compute buffer sizes in planner | open | **Superseded by A3a+A4** (mini-context + `graph_reserve(no_alloc=true)` replaces the scheduler-query approach). Close as superseded when A3a+A4 land. |
| `llama.cpp-w1rxh` | Set `max_buffer_size` on compute buffer type to arena zone capacity | closed | Landed 2026-04-22 (commit a60dc806a); structural cleanup that works under either design. Kept. |
| `llama.cpp-tyoc2` | Wire host compute buffer chunks through SCRATCH zone | open | Phase 2 — host-side compute buffer routing |
| `llama.cpp-01mcl` | Remove `must_device=true` for compute buffers | open | Phase 2 — enables host-pinned compute when VRAM can't host |
| `llama.cpp-ioua6` | Fail-fast when compute buffer exceeds all zones | closed (reverted, restore TBD) | Becomes a no-op under this plan; defense-in-depth only |
| `llama.cpp-6bdmc` | EPIC: Unified memory placement planner (sibling) | open | **Superseded by this epic.** Design doc `2026-04-22-unified-memory-planner-design.md` content absorbed below (see "Consolidated from 6bdmc" sections). Close when this epic's Track A lands. |

Existing related plans under `docs/plans/`:
- `2026-02-09-unified-cache-central-memory.md` — budget query API + expert staging
- `2026-02-09-unified-memory-accounting.md` — atomic runtime byte accounting
- `2026-02-09-unified-memory-system.md` — system-level allocation lifecycle
- `2026-02-10-fit-params-vs-unified-cache.md` — perf analysis (not-this-plan)
- `2026-04-22-unified-memory-planner-design.md` — **superseded sibling**; content lifted into this doc

This plan **consolidates** those efforts under a single target architecture
and sequences them.

### Known issues exposed by pre-flight canaries (bead `llama.cpp-m09zb`)

The Track D0 canaries exposed a P0 wedge in the SYCL backend. Initial
RCA (Task 6) pinned it to `staging_buffer_pool::acquire()` during
S1-PRELOAD. Subsequent D0.4 evidence broadens the scope: **the
underlying bug is an L0 DirectSubmission non-flush affecting any
`event.wait()` post-init** under oneAPI 2025.3 + the xe driver on
Arc B580. The same hang signature reproduces via a call path that does
not touch the staging pool, does not hold a mutex across the wait, and
does not involve the preload loop. The staging pool's mutex-hold pattern
is one amplifier, not the root cause.

**Symptom (all callsites)**: main thread spins in `sched_yield` inside
`L0::EventImp::queryStatus` via `sycl::event_impl::wait`. Only
`NEO::DirectSubmissionController::controlDirectSubmissionsState` lives
alongside, and it is in `pthread_cond_wait` — idle, not flushing the
batch. `strace` shows only the external SIGTERM at `timeout`; no abort,
no SIGSEGV, no assert. `EXIT=124`.

#### Example A — preload path (original m09zb RCA, Task 6)

D0.1 triggers this. Call chain (gdb, top frames):

```
#7  staging_buffer_pool::acquire          (common.hpp:1863)
#8  ggml_sycl_fill_reordered_host         (ggml-sycl.cpp:10764)
#9  unified_cache::direct_stage_weight    (unified-cache.cpp:1946)
#10 unified_cache_direct_stage_weight     (unified-cache.cpp:7649)
#11 ggml_sycl_preload_model_weights       (ggml-sycl.cpp:11707)
#12 llama_model::load_tensors
#13 llama_model_load_from_file_impl
```

**Amplifier analysis** (still accurate for this call path): two
independent back-pressure mechanisms collide during the 16-deep
sliding-window `s1_in_flight` at `ggml-sycl.cpp:11867`. Each
`fill_reordered_host` call performs `staging_pool.release(reorder_buf_raw,
copy_event)` at `ggml-sycl.cpp:10929`, installing the in-flight H2D
event on a slot. A subsequent `acquire()` best-fit picks an older slot
whose `copy_event` is still batched in L0 DirectSubmission (not yet on
the BCS hardware ring). `event.wait()` enters an L0 `sched_yield` spin
while holding the pool's `mutex_`. Reproducible in ~7 s on every cold
load of Mistral 7B Q4_0 with `n_gpu_layers=999`.

**Violated assumption**: `staging_buffer_pool::acquire()` assumes
`pending_event`s recorded via `release(ptr, evt)` are "close to
completion" by the time the slot is re-acquired, AND that holding the
pool mutex during `event.wait()` is safe. Neither holds under the
preload path's submission pattern. Notably, `drain_all()` in the same
class at `common.hpp:1991` already uses the correct drop-mutex pattern
(see its explicit comment).

#### Example B — direct tensor_set path (D0.4 finding)

D0.4 triggers this. The canary uses only `ggml-backend` APIs (no llama
API, no unified_cache, no staging_buffer_pool). Call chain (gdb, top
frames):

```
#6  sycl::_V1::detail::event_impl::wait                                   libsycl.so
#7  ggml_backend_sycl_buffer_set_tensor    (ggml/src/ggml-sycl/ggml-sycl.cpp:12685)
#8  main                                    (canary, ggml_backend_tensor_set call)
```

Same L0 `sched_yield` + idle controller signature. Wedges within 10 s of
the first `ggml_backend_tensor_set`. Distinguishing facts:

- No staging pool involved.
- No mutex held across `event.wait()`.
- No preload loop.
- Single synchronous `stream->memcpy(..., data, size).wait()` inside the
  SYCL backend.

**Implication**: the fix must address the underlying L0 DirectSubmission
non-flush, not just the staging-pool amplifier pattern. Candidate
approaches include (1) periodic `ext_oneapi_submit_barrier({})` nudges
on the relevant queues before user-facing `event.wait()`s, (2)
investigation of whether `flushCommands`/`executeCommandLists` invocation
from our queue submissions reliably wakes
`NEO::DirectSubmissionController`, and (3) oneAPI/xe-driver
version-matrix check (the combination here is known to change behavior
across compute-runtime revisions).

**Mitigation**: Track E below (task E1). Gates every canary in D0 and any
test that touches `llama_init_from_model` OR issues a user-facing
`event.wait()` against the SYCL backend on Arc B580 + oneAPI 2025.3.

### Canary results (interim, pre-E1)

| Canary | Status | Notes |
|---|---|---|
| D0.1 — skeleton determinism | INCONCLUSIVE (blocked by m09zb) | Binary builds; execution wedges in `llama_model_load_from_file` before any determinism measurement can complete. Resumes once E1 lands. |
| D0.2 — PP + TG graph union | PASS (commit `3ba255e`) | Mistral 7B: 13 op types. GPT-OSS 20B: 17 op types — MoE adds `ADD_ID, ARGSORT, MUL_MAT_ID, SOFT_MAX`. PP == TG on both models → A3a can size from a single shape (the double-reserve strategy is not needed for either of these models). |
| D0.3 — post-split CPY visibility | Not run (blocked by m09zb) | Requires multi-device context init, which hits the same wedge on first device's model load. |
| D0.4 — direct-weight-load mechanics | INCONCLUSIVE (commit `e594db1`) | Canary uses only `ggml-backend` APIs (no llama, no unified_cache, no staging pool). Builds clean. Runtime wedges on the first `ggml_backend_tensor_set` at `ggml_backend_sycl_buffer_set_tensor` (`ggml-sycl.cpp:12685`) with the same L0 hang signature as m09zb. **This proves m09zb is not staging-pool-specific** — it is a general L0 `event.wait()` issue (see "Example B" above). Resumes once E1 lands. |

When E1 lands, rerun all four canaries. Acceptance for closing D0 is the
outcome of each canary (PASS / FAIL / INCONCLUSIVE with rationale), not
"all PASS" — a FAIL informs which design variant (plan A vs plan B) we
pick.

## Target state

### Core principle

> The placement plan is the single source of truth for every byte consumed by
> inference. Allocation is a mechanical consequence of the plan, not an
> opportunistic request against whatever zone has space.

### Plan schema extensions

Add to `placement_plan`:

```cpp
struct placement_plan {
    // ... existing fields ...

    // Graph-shape-dependent demand (populated during the single plan call)
    struct compute_demand {
        size_t per_device_bytes[MAX_DEVICES];   // per-backend compute buffer
        size_t host_bytes;                      // cross-backend staging
        size_t per_op_scratch_peak;             // max pool demand across ops
        uint32_t graph_hash;                    // invalidation signal
    } compute;

    // Per-op demand map (optional, for operator-level placement)
    std::unordered_map<std::string, op_demand> ops;  // keyed by op name
};

struct op_demand {
    size_t scratch_bytes;                       // per-op temporary
    size_t output_bytes;                        // activation output live until consumed
    ggml_backend_dev_t preferred_device;        // plan's decision
    uint32_t consumed_at_node;                  // liveness range end
};
```

### Two-stage planning: weights once, compute per context

Weights are shared across contexts; compute buffers are per-context. The
plan reflects that:

- **Weight plan** — computed once per model, at `llama_model_load_from_file`
  time. Decides placement for every weight + KV cache capacity + MoE expert
  residency + fixed infrastructure (oneDNN scratchpad, DMA pool, etc.).
  Sized to hold the **maximum** context the user declared via
  `model_params.n_ctx`.
- **Compute plan** — computed per context, at `llama_init_from_model` time.
  Decides placement for the compute buffer, per-op scratch, and per-op
  routing. Fits within the zones the weight plan already reserved.

**Weights are written directly to their planned addresses as they come off
disk**; the compute plan allocates from zones the weight plan reserved; at
no point do weights move.

#### Motivating case: multi-context (e.g., `llama-server --parallel N`)

The weight/compute split exists specifically to serve **multiple
concurrent `llama_context` instances sharing one `llama_model`**. The
canonical example is `llama-server --parallel N`: N chat slots, each its
own `llama_context` with its own KV + compute buffer + per-op routing,
all reading from one shared set of weight arenas. Other cases (persistent
chat + ephemeral embedding on the same model; warmup context + hot
context) fall out of the same shape.

The `model_params.n_ctx` invariant follows: it is the **maximum context
any future context will ever request from this model**. The weight plan
reserves the RUNTIME zone for that max; each context's compute plan
allocates within it. A6 errors at context creation if `cparams.n_ctx >
model_params.n_ctx`. For `llama-server`, the server computes
`model_params.n_ctx = ctx_per_slot * n_parallel` (or similar) before
creating any slots.

**Ownership and lifecycle (binding):**
- `llama_model` owns `std::unique_ptr<weight_plan>`, immutable after load,
  freed in model free.
- `llama_context` owns `std::unique_ptr<compute_plan>`, freed in context
  dtor. Compute plan holds `const weight_plan *` (non-owning); model must
  outlive every context (enforce by ref count on the shared model or
  explicit assert in context dtor).
- Weight arenas are shared read-only across contexts; no synchronization
  needed on the weight path.

**Concurrent `llama_init_from_model`:** Shared `weight_plan` is read-only
(safe). Per-context `graph_reserve` touches ggml-alloc state that is
**not known to be re-entrant**. Track D adds a validation test that
exercises N parallel `llama_init_from_model` calls and verifies plan
determinism + no data races under TSAN. Until that test is green,
callers must serialize context creation (`llama-server` already does).

**Multi-model is NOT a goal.** Each `llama_model` stands alone; no shared
planner state between models. Two simultaneous models compete for VRAM
like two processes would — *correct* but not *optimized*. Speculative
decoding (draft + target) works as a free side-effect.

```
┌─────────────────────────────────────────────────────────┐
│ Caller (llama-cli, llama-completion, llama-server, ...):│
│   - Parse CLI: -c, -b, -ub, --flash-attn, etc.          │
│   - Populate llama_model_params with context shape      │
│     (treat model_params.n_ctx as "max context I'll      │
│      ever request from this model")                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ llama_model_load_from_file(path, model_params):         │
│   1. Open GGUF, build tensor metadata (no data copy)    │
│   2. Build skeleton graph using metadata tensors        │
│   3. graph_reserve(no_alloc=true) → real compute buffer │
│      demand for model_params.n_ctx (the MAX context)    │
│   4. compute_weight_plan(metadata, model_params, budget │
│                          + compute_demand_from_step_3)  │
│       - weights + KV capacity + MoE experts             │
│       - infrastructure buffers                          │
│       - RUNTIME zone capacity = max compute buffer need │
│   5. Reserve VRAM arena zones from weight plan          │
│   6. Stream weight DATA directly into planned addresses │
│      (mmap / DMA straight to the weight_plan.entries[i] │
│       location — no host bounce buffer)                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ llama_init_from_model(model, cparams):                  │
│   1. Validate cparams.n_ctx <= model_params.n_ctx       │
│      (error if user asks for bigger context than the    │
│       model was loaded for)                             │
│   2. compute_compute_plan(model.weight_plan, cparams)   │
│       - Build the real graph for cparams                │
│       - graph_reserve(no_alloc=true) → actual size      │
│       - Allocate compute buffer within RUNTIME zone     │
│         (spills to host-pinned if it doesn't fit)       │
│       - Assign every op a plan.ops[op_name] entry       │
│   3. graph_reserve(no_alloc=false) consumes the         │
│      pre-reserved compute memory                        │
└─────────────────────────────────────────────────────────┘
```

### Why model-params-level plumbing (not a pre-load hook)

The orphan `ggml_backend_sycl_set_runtime_n_ctx()` is today's half-baked
version of this idea (a backend-specific setter that callers could call
before load). It was never wired up because it's the wrong shape: it forces
every caller to know about SYCL-specific APIs and to call them in the right
order relative to `llama_model_load_from_file`. Making the plan inputs part
of `llama_model_params` is the natural API — the caller fills a struct, the
backends consume it, nothing to miscall.

This is a **fork-level API change**: we own this branch and don't merge back
upstream, so we can extend public structs freely. Existing callers in this
repo get updated in the same PR.

### Retirement of the orphan runtime-update path

- **Delete** `ggml_backend_sycl_set_runtime_n_ctx()` at `ggml-sycl.cpp:6562`
  and the declaration at `ggml-sycl.h:190`
- **Delete** `placement_plan::planner_n_ctx_is_runtime` flag
- **Delete** `update_placement_plan_runtime_n_ctx()` on `unified_cache`
- **Inline** the plan call at model load to use `model_params.n_ctx` directly
  instead of `inventory->n_ctx` (model's trained max)

### Direct placement mechanics

"Direct placement" means **no intermediate bounce buffer between GGUF
mmap and the final arena home** — not zero-copy mmap of GGUF pages into
device memory (impossible) or into SYCL's pinned pool (SYCL's allocator
owns those pages, can't substitute in external mmap pages).

Concretely:

- **Device-targeted weights**: the VRAM arena's KV/WEIGHT zone is one
  large allocation. `compute_weight_plan` assigns `(device_id, offset)`
  to each entry. At load time, the model loader reads bytes from the
  mmap'd GGUF and `ggml_backend_tensor_set`s them to
  `arena_base + offset`. One copy, destination is final location.

- **Host-pinned-targeted weights**: the host-pinned arena is a
  SYCL-allocated pinned pool. Planner assigns `(arena_id, offset)`.
  Loader copies bytes from mmap'd GGUF into `pinned_base + offset`.
  Same shape, different destination.

- **What goes away**: today's S1-PRELOAD path that staged into host
  pinned then streamed to device. Under direct placement, device-bound
  weights copy once (mmap → device) and host-bound weights copy once
  (mmap → pinned). No two-step stage.

### S1-PRELOAD folds into direct placement

S1-PRELOAD today is a two-step weight-materialization path: read from
GGUF into a host-pinned staging arena, then streamed to device. Under
this plan, weights go **directly** to their planned addresses when the
GGUF is read:

- Device-targeted weights: memory-mapped or DMA'd straight to the VRAM
  arena location computed by `compute_weight_plan`
- Host-targeted weights: mmap'd into the host-pinned arena at the
  planned offset

There is no staging arena. The preload concept collapses — tensors are
"preloaded" by virtue of being read into their final homes. Commit
history referencing S1-PRELOAD becomes a migration artifact; the name
is retired and the code paths merged into the regular load path in
Phase 2.

### Throwaway mini-context for `graph_reserve` at model load

`graph_reserve(no_alloc=true)` is ggml's authoritative sizing pass, but
today it's called from inside `llama_context`'s constructor — it needs a
backend scheduler and a compute graph. At model-load time we don't yet
have a real `llama_context`, so the planner constructs a **throwaway
mini-context** whose sole purpose is to host the skeleton graph + call
`graph_reserve`:

1. Open the GGUF header, build tensor metadata (shape only, no data yet)
2. Construct an ephemeral `llama_context_params`-like shape from
   `model_params.n_ctx` / `n_ubatch` / `flash_attn_type`
3. Spin up a minimal scheduler + call `llama_build_graph()` with the
   metadata tensors; resolve `flash_attn=auto` during this pass (see
   below)
4. `graph_reserve(no_alloc=true)` → authoritative per-backend buffer
   sizes for the max-context case
5. **Destroy the mini-context** — it has served its purpose. The weight
   plan carries forward the sizes it computed.
6. `compute_weight_plan` consumes the sizes + metadata, decides weight
   placements, sizes the RUNTIME zone for the max case
7. Reserve real VRAM + host-pinned arena zones from the plan
8. Load weight data from GGUF directly into planned arena offsets

The mini-context's lifecycle is strictly bounded to the planner call:
construct → skeleton graph → sizing pass → teardown. It never sees
real weight data, never dispatches kernels, never holds persistent
state.

At `llama_init_from_model` time, the same two-step pattern runs for the
per-context compute plan: build real graph with real cparams, run
`graph_reserve(no_alloc=true)`, allocate within the RUNTIME zone the
weight plan reserved. That's a normal non-throwaway context.

### Flash attention auto-detection runs inside the mini-context

`model_params.flash_attn_type = auto` is the default. Compute buffer size
depends critically on whether FA fires (16 GB attention score matrix vs
~400 MB without it). Today the auto-detection walks the graph in
`src/llama-context.cpp:570-608` to check whether the scheduler assigns
FA to the same backend as its layer — that logic must move earlier so
the plan sees the resolved decision.

The skeleton graph pass in the mini-context is the natural home: it
already builds the graph, already runs the scheduler, already has the
information needed. During that pass:

1. Walk `FLASH_ATTN_EXT` nodes
2. Check scheduler assignment (device match with layer?)
3. Resolve `model.effective_flash_attn = on | off` — stored on the
   model, read by the plan
4. Continue the sizing pass with FA resolved

By the time `compute_weight_plan` runs, `flash_attn` is a concrete
boolean. `cparams.flash_attn_type = auto` at context creation becomes a
consistency check against `model.effective_flash_attn` — no re-detection.

### Op-level placement consultation

Today `ggml_sycl_op_is_planned_on_host(op, device)` consults two tracks:
`layer_device` (dense weights) and `kv_device` (KV cache). After this plan:

```cpp
// plan.ops is a vector indexed by graph node index (op_id), populated
// when compute_compute_plan() walks the graph. Node count and ordering
// are fixed at context creation and match graph_reserve's view.
bool ggml_sycl_op_is_planned_on_host(const ggml_tensor * op, int device_handle) {
    int op_id = ggml_graph_node_index(op);  // graph-local index
    GGML_ASSERT_MSG(op_id >= 0 && op_id < (int)plan.ops.size(),
        "op has no plan entry — planner missed something: op_id=%d name=%s",
        op_id, op->name ? op->name : "(unnamed)");
    return plan.ops[op_id].preferred_device != device_handle;
}
```

**Keying: `op_id` (graph node index), not `op->name`.** Names are not
guaranteed unique (ggml reuses names across layers) or set (temp tensors
in intermediate ops are often unnamed). Node indices are concrete and
match what `graph_reserve` walks, so the plan.ops vector has 1:1
correspondence with graph nodes.

**Every op in the compute graph has a plan entry.** The per-op plan is
populated when `compute_compute_plan()` walks the graph at context
creation: each op gets assigned a `(preferred_device, scratch_bytes,
output_bytes)` tuple based on (a) its source tensors' placement from
the weight/KV plan, (b) the compute zone's remaining capacity, (c) the
op's own kernel preferences (e.g., FA prefers XMX device, ROPE is
cheap on either).

Missing plan entries assert — forces the planner to own every decision,
no silent fall-through to a default.

The old `ggml_sycl_layer_plan_applies_to_op` switch statement is deleted
entirely; placement is data-driven from `plan.ops`.

### Weight priority order (consolidated from 6bdmc)

When VRAM is tight, the planner fills greedily in priority order:

```
NORM_EMBED  > ATTENTION  > FFN  > MOE_DOWN  > MOE_UP  > MOE_GATE_PROJ
```

Rationale: norm/embed are tiny and touched every token; attention is
bandwidth-critical hot path; FFN is the bulk of dense compute; MoE
experts are routed per-token and the DOWN projection dominates
activation flow. Overflow spills to host-pinned arena in the same
priority order (hottest goes to VRAM, coldest spills first).

This matches the existing `placement_priority` enum in the weight
bin-packer; the plan formalizes the order as part of the spec.

### Allocation order invariant

Allocation proceeds in a fixed order that cannot fail when the plan is
consistent with real demand:

**At model load (weight plan):**
1. **Reserve** all VRAM zones from weight plan sizes — WEIGHT, KV, ONEDNN,
   RUNTIME (sized for `model_params.n_ctx` max), SCRATCH.
2. **Reserve** host-pinned arena zones for weight overflow + KV overflow +
   compute overflow.
3. **Fixed infrastructure buffers** (MoE tables, DMA pool, PP prefetch)
   allocated from their planned zones.
4. **Weight data written** directly into planned arena offsets:
   device-targeted entries via `ggml_backend_tensor_set(dst=arena_base +
   offset)`; host-targeted via `memcpy(pinned_base + offset, mmap_src)`.
   One copy per tensor; no intermediate bounce buffer.

**At context creation (compute plan):**
5. Compute buffer allocated within the RUNTIME zone reserved at step 1.
   If it doesn't fit (because this context's demand > the max-context
   reservation, which shouldn't happen under the `cparams.n_ctx <=
   model_params.n_ctx` validation), spill to host-pinned compute zone.
6. Per-op scratch from `plan.compute.per_op_scratch_peak` in SCRATCH zone.

**At context destroy:**
7. Compute buffer + per-op scratch allocations freed via the zone's TLSF
   free-list; the RUNTIME zone's capacity itself is **not** released.
   Subsequent `llama_context` creation re-allocates from the same RUNTIME
   zone — concurrent contexts share the pre-reserved capacity first-come-
   first-served (see D11).

### VRAM-insufficient policy

The point of the plan is to handle this correctly without shrinking the
user's context. Policy:

- Fill VRAM greedily in priority order (`NORM_EMBED > ATTENTION > FFN >
  MOE_DOWN > MOE_UP > MOE_GATE_PROJ`)
- Anything that doesn't fit VRAM goes to **host-pinned arena**
- KV cache tiers per-layer (hot layers on device, cold layers on host)
- Compute buffer spills to host-pinned if RUNTIME zone can't hold it
- **Never** silently shrink `n_ctx`, `n_batch`, `n_ubatch`, or any user
  parameter
- Only failure mode: **plan concludes it cannot fit even with full host
  overflow** → hard error at model-load time with a clear message
  ("requested n_ctx=131072 needs 42 GB across VRAM + host-pinned, available
  is 38 GB")

**Compute-buffer-on-host is loud.** If the compute buffer spills to
host-pinned, the GEMMs for that context run over PCIe at host-pinned
speeds — typically 10–50× slower than VRAM. This is correct but easy
to mistake for a mystery perf regression. Policy:

- **WARN at plan time** with projected perf impact, using the planner's
  own per-op compute estimate:
  > *compute buffer 4.2 GB spilled to host-pinned; projected TG impact
  > 71 → ~12 tok/s. Raise `GGML_SYCL_VRAM_BUDGET_PCT` (currently 30) or
  > lower `-c` to restore VRAM-resident compute.*
- **Default**: proceed with warning.
- **Opt-in strict mode**: `GGML_SYCL_FAIL_ON_COMPUTE_HOST=1` → fail at
  plan time instead of warning. This is **not** a fallback flag (doesn't
  disable the plan), just a policy knob for users who'd rather fail than
  silently slow.

The plan encodes the VRAM-greedy + host-overflow policy as its
fundamental contract; this is not a fallback, it is the plan.

## Design decisions

### D1 — Single plan struct, not separate plans per memory class

Keep one `placement_plan` **type**. A3 splits the plan into two **instances**
of that type: one long-lived weight plan on the process-wide `unified_cache`
(process-scoped — see D10), and one short-lived compute plan per
`llama_context`. D1 forbids new types; A3 creates new instances. No conflict.

Add fields; don't add types.

### D2 — Plan at load, weights placed directly

`compute_placement_plan` runs inside `llama_model_load_from_file` and its
output governs every allocation from that point forward — including where
the weights land as they're read from GGUF. There is never a phase where
weights exist in a temporary buffer awaiting relocation.

### D3 — Context params are model-params inputs on our fork

`llama_model_params` is extended with `n_ctx`, `n_ubatch`, `n_seq_max`,
`flash_attn_type` so the planner has every input it needs at model load.
This is a public-API change; we own this fork and don't merge upstream,
so it's acceptable. Every in-tree caller is updated in the same PR.

### D4 — Compute buffer is graph-queried at both stages

`graph_reserve(no_alloc=true)` is ggml's authoritative sizing pass, and
we run it twice: at model load (inside a throwaway mini-context with
`model_params.n_ctx` = max, to size the RUNTIME zone capacity) and at
each `llama_init_from_model` (normal context, `cparams.n_ctx` = this
context's actual size, to compute the per-context compute plan). Using
ggml's real sizing at both points eliminates drift. The mini-context
is strictly ephemeral — constructed, queried, destroyed inside the
planner call.

### D5 — Flash attention auto-detection moves to model load

FA auto-detection today runs in `llama_init_from_model`; under this
plan it runs inside the mini-context's skeleton graph pass so the
weight plan sees a resolved `effective_flash_attn` value. Context
creation inherits the model's resolved value; `cparams.flash_attn_type`
is validated for consistency, not re-detected.

### D6 — No mid-inference re-plans

Plans are computed once at model load (weight plan) and once per
context (compute plan). Inference never triggers a re-plan. If memory
pressure changes mid-run (eviction, paging) that's arena-level
eviction, not plan revision. Keeps the hot path free of planner state.

### D7 — No new env vars

`GGML_SYCL_VRAM_BUDGET_PCT` + existing dispatch knobs remain the user-facing
surface. Plan is internal.

### D8 — Hard migration, no fallback flag

Per user directive: the plan is a requirement, not an option. No
`GGML_SYCL_UNIFIED_PLAN=0` escape hatch. Regression suite (Track D3)
is the quality gate; if it catches a regression, the fix is the plan,
not a rollback.

### D9 — Incremental landing

Plan lands via the existing `w1rxh` (closed) → `tyoc2` → `01mcl` bead chain
plus the relocation work described in Track A. `8gz7y` is superseded by
A3a+A4 and closes when Track A lands. This document is the *why*; those
beads are the *how*. One PR per task, each individually deployable.

### D10 — One model per process

The plan assumes a single `llama_model` lives in a process at any time.
This is the realistic case for llama-cli, llama-completion, llama-server,
and every other in-tree caller. The weight plan is therefore
**process-scoped**, stored on the process-wide `unified_cache`, not on
the `llama_model` struct.

Speculative decoding (draft model + target model) would need two
weight arenas. It is deferred as future work; when it lands, it extends
the plan with a "model_id" tag on entries and a per-model arena partition.
The compute plan stays per-context either way.

### D11 — Compute plan + RUNTIME zone lifecycle

- RUNTIME zone capacity is set **once** at model load, sized for the
  `model_params.n_ctx` peak of a single context.
- Each `llama_context`'s compute plan allocates from the RUNTIME zone at
  context creation.
- On `llama_context_free`, the compute plan's allocations return to the
  RUNTIME zone via TLSF free. The plan struct itself is destroyed with
  the context.
- Concurrent contexts on the same device **share** the RUNTIME zone on a
  first-come-first-served basis. If context B's compute plan cannot fit
  because context A has already allocated most of RUNTIME, context B's
  creation fails with a clear error. This is by design; the alternative
  (reserve N× RUNTIME for N concurrent contexts) over-allocates for the
  common single-context case.

### D12 — `n_ctx` default when user doesn't pass `-c`

If the user doesn't pass `-c`, `llama_model_params.n_ctx` defaults to the
model's trained maximum (e.g., GPT-OSS 20B's 131072). The planner honors
that number; if the resulting plan is infeasible for the current VRAM +
host-pinned budget, load fails fast with a clear message listing the
shortfall and suggesting `-c <smaller>` as the remediation.

Rationale: respecting the model default is the principled behavior and
matches current llama.cpp semantics. Silently capping the context
surprises users who intentionally load a model for long-context work.
Failing fast with "requested 131072 context needs N GB; budget is M GB;
retry with -c K" puts the choice back in the user's hands.

### D13 — Multi-device split policy

Planner-autonomous by default:
- Each device's share of weights is proportional to
  `device.vram × device.compute_throughput`.
- MoE experts split by priority × layer stride (existing heuristic, now
  encoded in the planner rather than scattered).

Env vars override the autonomous split when set:
- `GGML_SYCL_VISIBLE_DEVICES` — restricts enumeration before the planner
  runs.
- `GGML_SYCL_SPLIT_RATIO` — explicit per-device ratio; replaces the
  auto-computed one.
- `GGML_SYCL_MULTI_GPU_MODE` — selects between LAYER / EXPERT / HYBRID
  parallelism modes.

### D14 — `GGML_SYCL_FORCE_STREAMING` is deprecated

Runtime weight-streaming contradicts "plan is authoritative." The mode is
removed by Track C3 alongside `layer_plan_applies_to_op`. Users on
memory-constrained hardware rely on host-pinned overflow (VRAM-greedy +
host overflow is the default).

Justification beyond design purity: streaming is **slower** than running
the cold layers on CPU for inference. CPU compute on host-resident
weights avoids the PCIe copy entirely; streaming pays the copy cost on
every layer access. Under the unified plan, host-overflow weights are
handled by the planner's dispatch decision (`plan.ops[op].preferred_device
= CPU`), which is strictly better than streaming.

### D15 — Mini-context owns its own backends

The throwaway mini-context constructs its own SYCL backends + scheduler
for the sizing pass, then destroys them on teardown. The real model's
backends are built separately afterward with the weight plan installed.
This avoids any contamination of the real model's backend state by the
sizing pass, at the cost of briefly constructing device contexts twice.

During the mini pass the unified_cache operates in "measure-only" mode
(see task **A3a-measure-only**): it tracks what would be allocated and
reserved but does not commit zones or take ownership of device memory.
The measure-only flag is cleared before the real model load begins.
Measure-only is a first-class prerequisite of A3a and lands as a
separate bead, not folded into the mini-context work.

### D16 — PP + TG graph union sizing

Because PP (`ubatch=max_ubatch`) and TG (`ubatch=1`) produce different
graphs with different op sets, the mini-context runs `graph_reserve` **twice**
— once at each shape — and the plan takes the **union** of ops and the
**max** of per-device compute demand. Every op produced by either shape
gets a `plan.ops` entry. This is the guarantee behind the "every op has
a plan entry" assertion in §"Op-level placement consultation".

## Task breakdown

### Track E — SYCL backend prerequisites (blocks every other track)

Exposed by Track D0 canaries. Every other track creates a
`llama_context` (or a mini-context, or multiple per-shape reserves) on a
GPU-offloaded model, or otherwise issues a user-facing `event.wait()`
against the SYCL backend — all of which currently wedge on the L0
DirectSubmission non-flush issue documented under "Known issues" above.
The bug manifests in multiple callsites (preload path and direct
`ggml_backend_tensor_set` path; see Examples A and B), so the fix must
address the underlying L0 interaction, not just the staging pool's
mutex-hold pattern. Land E1 before starting Tracks A/B/C.

| Task | Summary | Acceptance |
|---|---|---|
| E1 | Address the L0 DirectSubmission non-flush that wedges any post-init `event.wait()`. Primary direction: restructure `staging_buffer_pool` to delegate back-pressure to the SYCL in-order queue instead of per-slot `pending_event` + mutex wait. Callers own the event chain: `acquire()` returns a slot unconditionally; the next H2D submission gets the slot's last `copy_event` as a `depends_on` dep. Remove `has_pending_event` / `pending_event` from `slot`. `release(ptr, evt)` becomes `release(ptr)` + a caller-owned event map (or simply a caller-side `std::deque<event>` per queue). Since D0.4 shows the hang reproduces without the staging pool, the fix must ALSO address the broader L0 flush path (e.g., periodic `ext_oneapi_submit_barrier({})` nudge, or a compute-runtime version pin). ~200 LoC in the pool restructure plus smaller changes elsewhere. Aligned with D10 ("weight plan process-scoped, compute plan per-context"). Bead: `llama.cpp-m09zb`. | No mutex held across any `event.wait()`; **D0.1 canary** (stub from Task 0 currently; full impl when E1 lands) loads Mistral 7B cleanly in under 60 s; **D0.4 canary** (direct mmap → device `ggml_backend_tensor_set`) completes within 60 s with byte-identical readback; `ninja -C build` + full ctest green; no new env var; zero perf regression on `llama-bench Mistral7B-Q4_0 PP512/TG128` baseline (±3%) |

### Track A — Extend llama_model_params + plan schema (`llama.cpp-8gz7y` dependency)

| Task | Summary | Acceptance |
|---|---|---|
| A1 | Add `n_ctx`, `n_ubatch`, `n_seq_max`, `flash_attn_type` to `llama_model_params`; update every in-tree caller (`common/common.cpp`, `llama-cli`, `llama-completion`, `llama-server`, examples) to populate from their CLI args before `llama_model_load_from_file` | Build clean; all callers compile; model load sees the right `n_ctx` via the new fields |
| A2 | Extend `compute_placement_plan()` signature to consume `model_params` in addition to `kv_info`; stop using `inventory->n_ctx` | Plan decisions reflect user's `-c` value at load time; `GGML_SYCL_DEBUG=1` log confirms correct `n_ctx` |
| A3 | Split the struct: `weight_plan` (model-scope) + `compute_plan` (context-scope). Keep `placement_plan` as the shared backing type; add a lifecycle for the per-context part | Two `compute_plan` instances coexist for two contexts sharing a model; weight_plan is read-only after model load |
| A3a | Build the throwaway mini-context infrastructure: construct-from-`model_params`, build skeleton graph, run `graph_reserve(no_alloc=true)`, tear down | Mini-context creation/teardown leaks no memory (verified by stress test); `graph_reserve` returns authoritative sizes |
| A3b | Move FA auto-detection into the skeleton graph pass; store `effective_flash_attn` on the model | `cparams.flash_attn_type = auto` at context creation inherits the model's resolved value; `cparams.flash_attn_type = on` errors if `effective_flash_attn == off` |
| A4 | Arena zone sizing consumes `weight_plan` at reservation time; compute-plan allocations happen within the RUNTIME zone reserved in step A3 | Zones sized correctly the first time; no post-hoc `reserve_scratch_pool` / re-reserve path; model-load reserves RUNTIME for max context, compute plan per context allocates within it |
| A5 | Delete orphan runtime-update path: `ggml_backend_sycl_set_runtime_n_ctx()`, header declaration, `planner_n_ctx_is_runtime` flag, `update_placement_plan_runtime_n_ctx()` | `grep -rn` confirms all three symbols removed; builds clean |
| A6 | `llama_init_from_model` validates cparams consistency with weight_plan's assumptions; emit an error if `cparams.n_ctx > model_params.n_ctx` or `cparams.flash_attn_type` contradicts `effective_flash_attn` | Mismatched cparams surface as a clear error; matching cparams are a no-op |
| A7 | Weight loader writes directly to `arena_base + plan.entries[i].offset` (VRAM) or `pinned_base + plan.entries[i].offset` (host-pinned). Delete the S1-PRELOAD staging path | No intermediate bounce buffer in the load path; mmap → arena is one copy |

### Track B — Buffer type `get_max_size` (prerequisites to Track C)

| Task | Summary | Acceptance |
|---|---|---|
| B1 | `llama.cpp-w1rxh` — buffer type reports zone size, not L0 max alloc | `ggml_vbuffer` chunks compute buffer across zone limits |
| B2 | `llama.cpp-tyoc2` — host compute buffer chunks route through SCRATCH zone | Host-compute path available for CPU-backed experts/activations |
| B3 | `llama.cpp-01mcl` — drop `must_device=true` on compute buffer | Compute buffer can legally live host-pinned when plan directs |

### Track C — Op-level routing

| Task | Summary | Acceptance |
|---|---|---|
| C1 | Add per-source dependency check (`sources_needing_weights`, `sources_in_kv`) to op router | Replaces `layer_plan_applies_to_op` switch with data-driven dispatch |
| C2 | Populate `plan.ops` for **every** op in the post-scheduler-split graph (MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, synthetic CPY, etc.) with per-op preferred device | Per-op log line shows plan decision at first dispatch; CPY entries visible in plan dump; no change in Mistral output |
| C3 | Remove `layer_plan_applies_to_op`, `depends_on_planned_host_weight`, `g_tl_fattn_weight_stage` (FA per-op stage from llama.cpp-15li2), and `GGML_SYCL_FORCE_STREAMING` | grep for each symbol returns zero hits in src; behavior preserved via `plan.ops` |

### Track D — Validation, observability, + pre-flight canaries

| Task | Summary | Acceptance |
|---|---|---|
| D0.1 | **Canary (gates A3a)**: skeleton graph validity — build metadata-only graph for Mistral 7B + GPT-OSS 20B, call `graph_reserve(no_alloc=true)`, compare to a real-context reserve on same params | Sizes bit-identical → A3a approach validated. Divergent → switch to plan B (mmap-backed mini-context); document decision |
| D0.2 | **Canary (gates A3a)**: PP + TG graph union — reserve at `max_ubatch` vs `ubatch=1`, verify every op produced by either shape appears in the union walk | Union is complete; A3a's double-reserve strategy validated |
| D0.3 | **Canary (gates C2)**: post-split CPY visibility — force `GGML_SYCL_SPLIT_RATIO` multi-device, call `graph_reserve`, walk the post-split graph, confirm CPY nodes have stable deterministic names | CPY names stable across runs; `plan.ops` can key on them |
| D0.4 | **Canary (gates A7)**: direct-weight-load mechanics — prototype one tensor loaded from mmap bytes directly into `arena_base + offset` via `ggml_backend_tensor_set` | One-copy load verified on both Mistral 7B and GPT-OSS 20B; A7 proceeds on existing API |
| D1 | Plan dump diagnostic: `GGML_SYCL_PLAN_DUMP=1` prints full plan at context creation | Single-screen summary including compute.* fields; byte-deterministic across runs with same inputs |
| D2 | Plan-reality audit at inference end: compare plan vs actual allocation | Mismatch ≥ 5% logged as WARN, > 20% as ERROR |
| D3 | Regression suite: Mistral 7B Q4_0 + GPT-OSS 20B (FA on) + GPT-OSS 120B at `VRAM_BUDGET_PCT` ∈ {100, 50, 30} | All pass with plan dump showing zero mismatches; canonical outputs unchanged |

### Dependency graph

```
E1 ─┬─→ A1 → A2 → A3 → A3a → A3b → A4 → A5 → A6 → A7
    │                                             \
    │                                              → C1 → C2 → C3
    ├─→ B1 → B2 → B3 ____________________________/
    └─→ D0.1, D0.2, D0.3, D0.4

D1 can land after A3b (plan.compute populated + FA resolved)
D2 requires A7 + C2
D3 is epic-close gate
D0.1, D0.2, D0.3, D0.4 can re-run once E1 lands (they currently INCONCLUSIVE / blocked)
```

E1 gates every task that constructs a `llama_context` on a GPU-offloaded
model. That includes all of Track A (mini-context in A3a), the
compute-buffer tasks in Track B that exercise real contexts, and every D
canary except pure static-analysis work.

### Sequencing

- E1 is the hard prerequisite for **all** Track A/B/C/D work that creates a `llama_context` on a GPU-offloaded model. Land first.
- A1 is a hard prerequisite for everything else in Track A (API change first)
- Tracks A and B are independent after A1 (can go in parallel)
- Track C needs A7 + B3 (plan fully populated + buffer routing aware of it)
- Track D gates epic close

## Migration plan

No fallback flag. The plan is a hard migration: new code replaces old;
regression suite (D3) is the quality gate. If the regression suite fails
for a known case, the fix is the plan, not the escape hatch.

**Phase 0 — SYCL prerequisite (E1):**
Land the `staging_buffer_pool` restructure (bead `llama.cpp-m09zb`) so
that `llama_model_load_from_file` no longer wedges during S1-PRELOAD on
GPU-offloaded models. Gates every subsequent phase — none of Phases 1–5
can be validated (or even exercised by the D0 canaries) until E1 is in.

**Phase 1 — API + plumbing (A1 + A2 + B1):**
Extend `llama_model_params`; update every in-tree caller; change
`compute_placement_plan()` signature to consume `model_params`; land
buffer-type `get_max_size` chunking. No behavior change yet.

**Phase 2 — Weight plan is real (A3 + A3a + A3b + A4):**
Split struct into weight_plan (model-scope) + compute_plan (context-scope).
Build throwaway mini-context + skeleton graph for `graph_reserve(no_alloc=true)`
at model load. Move FA auto-detection into this pass. Arena zones sized
from authoritative graph_reserve output.

**Phase 3 — Direct placement + cleanup (A5 + A6 + A7):**
Weight loader writes directly to arena offsets; delete S1-PRELOAD staging.
`cparams` validation at `llama_init_from_model`. Orphan `set_runtime_n_ctx`
API deleted.

**Phase 4 — Buffer routing + op coverage (B2 + B3 + C1 + C2 + C3 + D1):**
Host-compute buffer path wired. Every op gets a `plan.ops` entry;
missing = assert. Delete `ggml_sycl_layer_plan_applies_to_op`. Plan
dump diagnostic available.

**Phase 5 — Audit + regression (D2 + D3):**
Plan-vs-reality audit at inference end. Regression suite across Mistral
7B / 20B (FA on) / 120B / various VRAM_BUDGET_PCT. Epic close when the
suite is green.

## Acceptance criteria (epic-level)

- [ ] `placement_plan` contains a populated `compute` field for every
  `llama_context` created on a SYCL backend
- [ ] `compute_placement_plan()` is called exactly once per model, at
  `llama_model_load_from_file`, and reads `n_ctx` / `n_ubatch` / etc. from
  the extended `llama_model_params`
- [ ] `ggml_backend_sycl_set_runtime_n_ctx()`, `planner_n_ctx_is_runtime`,
  and `update_placement_plan_runtime_n_ctx()` are deleted (plus the header
  declaration)
- [ ] Plan `n_ctx` equals the user-specified `-c` value (not the model's
  trained maximum) — verifiable via `GGML_SYCL_PLAN_DUMP=1`
- [ ] Weights are written directly into their planned zones as they're read
  from GGUF — no relocation copy exists in the load path
- [ ] `llama_init_from_model` contains no plan-mutating calls (only
  validation + `graph_reserve(no_alloc=false)`)
- [ ] No compute buffer allocation request ever fails with "zone too small"
  when the plan's aggregate budget can host it (plan-level errors surface at
  plan time, not allocation time)
- [ ] `GGML_SYCL_VRAM_BUDGET_PCT=30` + default context succeeds for Mistral 7B,
  GPT-OSS 20B, and (where the chain allows) GPT-OSS 120B without touching the
  `ioua6` fail-fast path
- [ ] Per-op routing produces the same kernel dispatch as today for Mistral 7B
  canonical and 20B canonical (byte-identical output)
- [ ] `GGML_SYCL_PLAN_DUMP=1` produces a plan summary readable in under 30
  seconds (not a 100+ page log dump)
- [ ] All six prerequisite beads (`48330`, `mubmt`, `8gz7y`, `w1rxh`, `tyoc2`,
  `01mcl`) close with this plan as the parent
- [ ] `ggml_sycl_layer_plan_applies_to_op()` is deleted (replaced by data-driven
  dispatch in Track C)

## Out of scope

- CUDA / Metal / Vulkan port of the unified plan (SYCL-only for now; other
  backends can adopt the pattern later)
- Runtime re-planning (D3 explicitly rejects this)
- Multi-context planning (one `placement_plan` per `llama_context`; contexts
  don't share plan state)
- NUMA-aware host placement (all host allocations go to the same pinned pool)
- Graph caching across contexts (liveness is per-context)
- Replacing ggml's scheduler (we consume it, not replace it)

## Risks

| Risk | Mitigation |
|---|---|
| Compute-buffer estimator drifts from ggml-alloc's actual demand | `llama_init_from_model`'s real `graph_reserve` will fail against pre-reserved zones; fail-plan loudly with the under-estimated term; regression suite (D3) catches known drifts |
| `llama_model_params` API change breaks external callers of this fork | We own this fork; update all in-tree callers in the same PR; external users of the fork must adopt |
| Per-op routing regresses an edge case | Regression surfaces in Track D3 canary suite → fix forward in same PR set. No `GGML_SYCL_UNIFIED_PLAN=0` escape hatch (see D8). |
| Multi-GPU tensor split needs per-device plans | Plan is already per-device via `per_device_bytes[]`; audit for gaps during Track A |
| `cparams.n_ctx > model_params.n_ctx` | A6 validates and errors at context creation; caller fix is to set `model_params.n_ctx` to the maximum they'll ever ask for |
| S1-PRELOAD path overlaps with direct placement | Delete S1-PRELOAD — its role (stage to host-pinned, then stream to device) is replaced by "mmap → arena is one copy." Done in Phase 3 (task A7). |
| Throwaway mini-context leaks resources | Explicit teardown in the planner call + stress test in Track D3; ASAN CI run validates zero leaks across 100 model loads |
| FA auto-detection timing regression for existing models | Track D3 regression suite includes FA auto-detection on Mistral (should resolve ON) and 20B (should resolve ON post-15li2); same result as current late-detection |
| Multi-context creates different compute plans simultaneously | Each `llama_context` owns its own compute plan; weight plan on the `llama_model` is shared. Tested in D3 regression. |
| No fallback means a regression blocks everybody | Mitigation is strict D3 regression suite coverage before epic-close; if regression hits in practice we fix the plan, not roll back. This is by design per user directive. |
| L0 DirectSubmission non-flush wedges `event.wait()` post-backend-init (bead `llama.cpp-m09zb`) | Fix via task E1 before starting Track A. Primary direction: restructure `staging_buffer_pool` to own caller-side event chains (~200 LoC in `ggml/src/ggml-sycl/common.hpp`). Secondary direction: address the underlying L0 flush path — see Known-issues Implication block for three candidate approaches. Reproduces in ~7 s on cold Mistral 7B load via the preload path AND via direct `ggml_backend_tensor_set` (D0.4 witness). |

## References

- `ggml/src/ggml-sycl/unified-cache.hpp:189-310` — current placement_plan struct
- `ggml/src/ggml-sycl/unified-cache.hpp:48-66` — arena zone layout
- `ggml/src/ggml-sycl/unified-cache.cpp:9811-9950` — populate_host_zone_sizing
- `src/llama-context.cpp:571-682` — scheduler integration timeline
- `docs/plans/2026-02-09-unified-cache-central-memory.md` — prior central memory design
- `docs/plans/2026-02-09-unified-memory-system.md` — prior lifecycle design
- `llama.cpp-ioua6` (closed) — the specific failure this plan prevents
- `llama.cpp-15li2` (closed 2026-04-22) — the FA routing example of the
  denylist problem this plan fixes
