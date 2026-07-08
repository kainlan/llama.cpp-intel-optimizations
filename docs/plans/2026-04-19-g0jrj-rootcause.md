# llama.cpp-g0jrj — VRAM-resident weight routed to cpu_mul_mat (GPT-OSS 20B SEGV)

**Bead**: `llama.cpp-g0jrj`
**Branch**: `feature/sycl-coalescing`
**Predecessors**: `mqxer`, `lj6p0`, `skgik`, `wmk39`, `a7l5w`, `vtf7f`. The
vtf7f root-cause (`2026-04-19-vtf7f-rootcause.md §9`) documents the decisive
probe observation that defines this bead: the weight
`blk.16.ffn_gate_inp.weight` is VRAM-resident (`on_device=1` in
`direct_weight_entries_`), yet `cpu_mul_mat` gets called on it.

**User directive**:
> "if the weight is on vram we shouldn't be dispatching to the cpu. That's
> the whole point of the smart pointers — at dispatch time we check which
> memory it resides in and dispatch to the appropriate device"

## 1. The bad decision — exact file:symbol

Two sites force CPU dispatch on MoE-routing-subgraph MUL_MATs without
consulting weight placement:

### Site A — `should_dispatch_to_cpu` (per-op dispatcher)

File: `ggml/src/ggml-sycl/ggml-sycl.cpp`
Symbol: `should_dispatch_to_cpu(ggml_backend_sycl_context &, const ggml_tensor *)`

The FIRST branch:

```cpp
if (ggml_sycl_planner_authoritative_residency_active(ctx.device) &&
    (ggml_sycl_op_is_moe_routing_subgraph(dst) ||
     ggml_sycl_op_is_host_gate_activation_chain(dst, ctx.device))) {
    g_last_dispatch_query  = dst;
    g_last_dispatch_result = true;
    return true;
}
```

`ggml_sycl_op_is_moe_routing_subgraph(dst)` returns true for any MUL_MAT /
ADD / ARGSORT / TOP_K / GET_ROWS / SOFT_MAX whose name tree contains
`"ffn_gate_inp"`, `"topk"`, `"router"`, etc. That is a STRING-PATTERN
heuristic expressing a *policy preference* ("MoE routing paths would
benefit from CPU locality under an authoritative placement plan").

For MUL_MAT, this bypasses the only real check of where the weight
actually lives — the `resolve_allocation(src0, device).on_device()` call
that the non-early branches perform later in the same function.

### Site B — `classify_cpu_layer_blocks` (graph-level pre-classifier)

File: `ggml/src/ggml-sycl/ggml-sycl.cpp`
Symbol: `classify_cpu_layer_blocks(ggml_backend_sycl_context &, const ggml_cgraph *, std::vector<int8_t> &)`

Phase 2:

```cpp
if (ggml_sycl_planner_authoritative_residency_active(ctx.device) &&
    (ggml_sycl_op_is_moe_routing_subgraph(node) ||
     ggml_sycl_op_is_host_gate_activation_chain(node, ctx.device))) {
    node_cpu_flags[i] = 1;
    continue;
}
```

Same structure. Writes the pre-classified flag that `should_dispatch_to_cpu`
later consumes via its fast path (`g_preclassified_cpu_flags[...]`).
Wrong decision here persists through the entire graph run.

## 2. Why the current heuristic picked CPU

For GPT-OSS 20B:

1. Loader registers a placement plan (`cache->has_placement_plan() ==
   true`). Layer 16's `ffn_gate_inp.weight` has
   `plan.layer_device[16] < 0` → **planned host**.
2. But the loader ALSO S1-PRELOADs `ffn_gate_inp.weight` into VRAM
   (`direct_weight_entries_` contains a DEVICE entry for it; the vtf7f
   probe confirmed `on_device=1`). This is the "both places" state.
3. The MoE-routing-subgraph heuristic sees the name match, sees the plan
   is active, and without any VRAM-residency check fires CPU dispatch.
4. `cpu_mul_mat` is called with `src0 = ffn_gate_inp.weight`. It looks
   for a host-accessible pointer, releases the mem_handle lease, and
   falls back to raw `tensor->data` — which points into a 2 GB pinned
   host arena chunk.
5. During DNNL SGEMM execution, that chunk gets munmap'd mid-inference
   (separate arena-refcount bug tracked as `dyhdl`), producing the
   vtf7f §9 SEGV signature.

The smart pointer (mem_handle → `unified_cache_entry` → `resolved_ptr.on_device`)
correctly reports that VRAM is the right target. The dispatcher is the one
that failed to listen.

## 3. Design alternatives

### A. Add a weight-placement gate INSIDE the MoE-routing-subgraph branch

Inside both sites, before returning "route to CPU" for a MUL_MAT, call
`ggml_sycl_resolve(dst->src[0], ctx.device)` and if `resolved.on_device`
is true, do NOT route to CPU — fall through to the rest of the dispatcher
logic (which will land on GPU for a VRAM-resident weight).

Pros:
- Minimal change, targeted at the actual bad branch.
- Preserves the MoE-routing-subgraph heuristic for the cases it was
  designed for (VRAM_BUDGET_PCT=30 where the weight is truly host-only).
- Uses the smart pointer exactly as the user directive specifies: "check
  which memory it resides in and dispatch to the appropriate device."
- No new global state or concept.

Cons:
- Only fixes MUL_MAT within the routing-subgraph branch. Activation-chain
  ops (ADD/MUL/NORM/RMS_NORM) still use the name-tree heuristic, but
  those ops don't read weights — they read activations, so weight
  placement doesn't apply to them.
- Still a "per-branch" gate rather than a unified policy.

### B. Move weight-placement awareness UP — pre-classification uses `resolve_allocation` as primary signal, heuristics as tiebreakers

Rewrite `classify_cpu_layer_blocks` so that Phase 1 ALREADY consults
`resolve_allocation(src0, device).on_device()` for every MUL_MAT, and
only applies the routing-subgraph pattern match as a tiebreaker when the
smart pointer says "I don't know" (resolution returns `.ptr == nullptr`
or `on_device` is ambiguous).

Pros:
- Cleaner architectural alignment with the user directive.
- One authoritative signal (smart pointer), with heuristics in their
  proper lower-priority role.

Cons:
- Invasive rewrite of a function that's the product of six layered
  incident fixes (each phase solves a specific shape of bug). Risk of
  regressing Mistral PP512 / Mistral canonical / GPT-OSS 120B without
  full re-validation of each phase's original constraint.
- Does not actually change the decision for the present bug (A already
  does that); it just relocates the logic.
- Higher test burden for a surface-equivalent fix.

### C. Replace the routing-subgraph MUL_MAT branch with pure placement lookup

Delete the `ggml_sycl_op_is_moe_routing_subgraph` check for MUL_MAT
entirely. Rely on the lazy per-layer classification further down in
`should_dispatch_to_cpu` (which already calls `resolve_allocation`) as
the sole signal.

Pros:
- Simplest. Heuristic deletion.

Cons:
- Loses the "authoritative plan" fast path for MUL_MATs whose layer_id
  extraction works fine and whose placement plan wants them host-routed.
  These would fall into Phase 2's per-layer cache, which is ALSO fed by
  `resolve_allocation` — so in principle equivalent, but this changes the
  order of classification (plan-then-routing vs routing-then-plan) and
  could expose ordering-sensitive bugs in multi-expert MoE.
- The routing-subgraph predicate still needs to fire for
  `host_gate_activation_chain` ops (ADD/MUL/NORM/RMS_NORM) where it IS
  the right signal, so deletion is partial only.

### Decision: A.

Reasons:
1. Exact match for the user directive ("check which memory it resides in").
2. Smallest, most surgical change — low regression risk.
3. Preserves all six phases of the pre-classifier's incremental fixes.
4. Uses the smart pointer infrastructure as the primary placement source,
   with the existing heuristic as the fallback it was always meant to be.

## 4. Why this doesn't regress CPU-offload modes

In `GGML_SYCL_VRAM_BUDGET_PCT=30` (or any mode where the plan legitimately
spills routing weights to host without a VRAM preload):

- `ggml_sycl_resolve(ffn_gate_inp.weight, device)` returns a pointer from
  the host-resident cache entry.
- The returned `resolved.on_device` is **false** (host tier).
- Our new gate "if on_device → skip CPU" does NOT fire.
- The existing MoE-routing-subgraph branch still returns `true`, and
  cpu_mul_mat runs exactly as before.

In `VRAM_BUDGET_PCT=100` / default GPT-OSS 20B:

- `ggml_sycl_resolve(ffn_gate_inp.weight, device)` returns the S1-PRELOAD
  VRAM pointer.
- `resolved.on_device` is **true**.
- Our gate fires: do NOT route to CPU.
- Dispatch flows down to the GPU MUL_MAT path, which handles this weight
  natively (SOA-layout MMVQ / DMMV depending on batch).

In `VRAM_BUDGET_PCT=30` on models where `ffn_gate_inp` is NOT a weight
the plan considers host-routed (plan says DEVICE), it was already going
to GPU regardless — no change.

## 5. Implementation plan

### Single commit: gate both sites on `resolve_allocation().on_device()`

Add a helper in `common.hpp`:

```cpp
// Returns true if `tensor` is a weight AND currently VRAM-resident on
// `device`. Queries the smart-pointer resolve path — does NOT allocate or
// trigger a cache load. Safe on graph build and dispatch paths.
inline bool weight_is_currently_device_resident(const ggml_tensor * tensor, int device) {
    if (!tensor || device < 0 || !ggml_sycl_tensor_is_weight(tensor)) {
        return false;
    }
    auto resolved = ggml_sycl_resolve(tensor, device);
    return resolved.ptr != nullptr && resolved.on_device;
}
```

Then in `should_dispatch_to_cpu`:

```cpp
if (ggml_sycl_planner_authoritative_residency_active(ctx.device) &&
    (ggml_sycl_op_is_moe_routing_subgraph(dst) || ggml_sycl_op_is_host_gate_activation_chain(dst, ctx.device))) {
    // User directive: the smart pointer is the source of truth for weight
    // placement. If the MUL_MAT weight is currently VRAM-resident (e.g.
    // S1-PRELOAD into device VRAM), dispatch to GPU regardless of the
    // routing-subgraph heuristic — that heuristic is a policy preference,
    // not a placement fact.
    if (dst->op != GGML_OP_MUL_MAT || !weight_is_currently_device_resident(dst->src[0], ctx.device)) {
        g_last_dispatch_query  = dst;
        g_last_dispatch_result = true;
        return true;
    }
    // Fall through: weight is on device, let the GPU path handle it.
}
```

Mirror in `classify_cpu_layer_blocks` Phase 2:

```cpp
if (ggml_sycl_planner_authoritative_residency_active(ctx.device) &&
    (ggml_sycl_op_is_moe_routing_subgraph(node) ||
     ggml_sycl_op_is_host_gate_activation_chain(node, ctx.device))) {
    if (node->op != GGML_OP_MUL_MAT || !weight_is_currently_device_resident(node->src[0], ctx.device)) {
        node_cpu_flags[i] = 1;
        continue;
    }
    // Fall through: let the main classification path decide (GPU).
}
```

Phase 3 also has a similar "keep_host_gate_chain" re-assertion check — it
needs to honor the same gate so the PP-reset Phase 3 doesn't re-assert
CPU on a VRAM-resident MUL_MAT:

```cpp
const bool keep_host_gate_chain = ggml_sycl_planner_authoritative_residency_active(ctx.device) &&
                                  node != nullptr &&
                                  (ggml_sycl_op_is_moe_routing_subgraph(node) ||
                                   ggml_sycl_op_is_host_gate_activation_chain(node, ctx.device)) &&
                                  !(node->op == GGML_OP_MUL_MAT &&
                                    weight_is_currently_device_resident(node->src[0], ctx.device));
```

Phase 3b re-application of CPU backpropagation is guarded by the SAME
`needs_cpu_inputs` predicate, which already uses routing-subgraph +
host-gate checks — update it too:

```cpp
const bool needs_cpu_inputs = (ggml_sycl_op_is_host_gate_activation_chain(node, ctx.device) ||
                               (ggml_sycl_op_is_moe_routing_subgraph(node) && node->op == GGML_OP_MUL_MAT)) &&
                              !(node->op == GGML_OP_MUL_MAT &&
                                weight_is_currently_device_resident(node->src[0], ctx.device));
```

### No changes to

- `unified-cache.hpp` / `unified-cache.cpp` (dyhdl's territory)
- `pinned-pool.cpp` (dyhdl's territory)
- `mem-handle.hpp` / `mem-handle.cpp` (vtf7f's refcount, already landed)
- `cpu-dispatch.cpp` (the downstream callee — fixing the caller, not the
  callee, per trap #1 in the bead)

### Files touched

1. `ggml/src/ggml-sycl/common.hpp` — add `weight_is_currently_device_resident`
   helper (read-only query using existing `ggml_sycl_resolve`).
2. `ggml/src/ggml-sycl/ggml-sycl.cpp` — gate four sites:
   - `should_dispatch_to_cpu` first branch
   - `classify_cpu_layer_blocks` Phase 2 branch
   - `classify_cpu_layer_blocks` Phase 3 `keep_host_gate_chain`
   - `classify_cpu_layer_blocks` Phase 3b `needs_cpu_inputs`

## 6. Gate expectations

| Gate | Pre-fix | Expected post-fix |
|------|---------|-------------------|
| 1. Mistral canonical 6..10 | PASS | PASS |
| 2. Mistral PP ≥ 1700 TG ≥ 81 | PASS (~1480 / ~81) | ≥ baseline |
| 3. 20B bench -p 512 -n 128 -r 3 | PASS | PASS |
| 4. 20B completion -n 128 | **FAIL (SEGV, vtf7f §9)** | **PASS** |
| 5. 120B -c 131072 bench | PASS | PASS |
| CPU offload 30% budget | PP ~269 TG ~14 | unchanged |

## 7. Traps addressed

- Not fixing cpu_mul_mat to "handle" a VRAM pointer — this fix is in the
  dispatcher, upstream of cpu_mul_mat. cpu_mul_mat never sees the tensor.
- Not adding a band-aid inside cpu_mul_mat — same.
- Not removing the MoE routing-chain heuristic — extending it to respect
  weight placement, keeping it for its legitimate host-only case.
- Not regressing budget modes — smart pointer returns `on_device=false`
  for truly host-only weights; gate doesn't fire; old CPU path runs.

## 8. Relation to parallel bead dyhdl

`dyhdl` is fixing the arena-level refcount so `host_arena` does not free
a chunk while a tensor points into it. That fix addresses the SEGV in
`cpu_mul_mat` when the CPU path is legitimately needed (budget modes).

`g0jrj` (this bead) ensures that when the weight is VRAM-resident, the
CPU path is not entered in the first place. Both fixes are complementary
and non-overlapping:

- `dyhdl` touches `unified-cache.cpp/.hpp`, `pinned-pool.cpp`
- `g0jrj` touches `ggml-sycl.cpp`, `common.hpp`

After both land, the 20B canary passes regardless of whether the weight
is VRAM-resident (g0jrj prevents bad dispatch) or host-only (dyhdl
prevents arena UAF during good dispatch).

## 9. Residual

- The MoE routing-chain heuristic is still a name-pattern match. It
  remains a policy signal, not a placement fact. If future models have
  different MoE layer naming, the heuristic may mis-classify — but with
  this fix, mis-classification for a VRAM-resident weight is harmless
  (gate corrects it at dispatch time).
- Phase 3b's producer-chain walk still uses name-based recursion. For
  a MUL_MAT whose weight is VRAM-resident, we no longer walk its
  producers to mark them CPU-bound. That's the desired behavior — those
  producers feed a GPU dispatch and should stay GPU-routed.
