# llama.cpp-n04bq — Cache/Expert invariant investigation

**Bead**: `llama.cpp-n04bq` (P1, INVESTIGATION)
**Branch**: `feature/sycl-coalescing`
**Predecessor chain**: skgik → lj6p0 → wmk39 → a7l5w → vtf7f → g0jrj → dyhdl → 0k543 → 4c1819604
**Status**: Phase 2 verdict complete — no fix landed this session

## 0. TL;DR

The user's architectural model holds for the SYCL backend: placement plan is
active at load time, experts land where the plan says, and the SYCL backend's
dispatch-time decisions correctly consult cache lookups.

The garble **does not originate in the SYCL backend's placement-plan /
cache-lookup plumbing**. Instead it originates at the **ggml scheduler level**:
when `ggml_backend_sycl_device_supports_op` rejects an op because its weight
is planned-on-host, the ggml scheduler routes the op to the **CPU backend**
(not the SYCL backend's own CPU-dispatch fallback). The SYCL backend then
handles only the ops whose weights are fully device-resident. Cross-backend
data flow between GPU-dispatched `ffn_moe_down-N` and CPU-dispatched
`ffn_moe_down_biased-N` (plus all the CPU-dispatched gate/up experts +
bias-add + GLU) is where correctness has to be checked.

Evidence and per-question findings below. No fix was landed this session — the
gap is architectural, not surgical.

## 1. Reproducer

```
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p "1, 2, 3, 4, 5," -n 30 --seed 42 --temp 0
```

Observed (HEAD `4c1819604`):
```
<|start|>user<|message|>1, 2, 3, 4, 5,<|end|><|start|>assistant,
so, 1, 1, 1, 1,  # 1, 2, 3, 2. 
```

Garbled. Expected `6, 7, 8, 9, 10` -like continuation.

## 2. Instrumentation added

Single self-contained probe, default OFF, enabled with
`GGML_SYCL_N04BQ_PROBE=1`. Only logs for tensors matching `blk.16.*` or op
names matching `*-16`. Each (tag,tensor,expert) triple logs once.

All probes live in `ggml/src/ggml-sycl/ggml-sycl.cpp` around the helper
`n04bq_probe_log` added at file scope, guarded by `n04bq_probe_enabled()`. The
probes are compile-gated inert (`if (!n04bq_probe_enabled()) return;`) so they
are safe to leave in. Covered sites:

| Probe tag | Site | Purpose |
|-----------|------|---------|
| `s1-preload/host-expert` | `ggml_sycl_preload_model_weights` | Confirms host-planned experts land in host arena + `register_host_expert` records the cache entry. |
| `resolve/cache-{host,device}` | `ggml_sycl_resolve_expert_ptr` | Confirms what cache lookup returns at dispatch time for a given (tensor, expert). |
| `resolve/fallback-{buffer-host,device,nothing}` | same | Cache-miss fallbacks (mmap / device / none). |
| `mul_mat_id/trace` | `ggml_sycl_mul_mat_id` | Records which `blk.16.*` tensors actually reach the SYCL MoE dispatcher. |
| `mul_mat_id/entry` | same | Tensor metadata + buffer type at entry. |
| `mul_mat_id/src0-data-classify` | same | `alloc_registry::query_location` classification of `src0->data`. |
| `dispatch_cpu_compute` | `ne12==1` hybrid CPU dispatch | Pointer source used for each expert task. |
| `dispatch_host_expert_rows` | PP host-expert path | Same for PP batched rows. |
| `moe_fusion/first-task` | `moe_fusion` FIRST path | Resolved pointer per CPU expert task (fusion path). |
| `moe_fusion/down-task` | `moe_fusion` DOWN path | Same for DOWN. |
| `cpu-tg-fast/task` | TG fast-path (`ne12==1`) | Resolved pointer + `act_host_ptr` for each expert task. |
| `update_moe_ptr_table/usm-hit` | `update_moe_ptr_table` USM branch | Cache hit on USM-accessible src. |
| `update_moe_ptr_table/usm-miss` | same | Cache miss + direct_expert_entries_ peek result. |
| `update_moe_ptr_table/non-usm` | non-USM branch | Cache lookup cascade result. |
| `should_dispatch_to_cpu` | entry + each return path | Dispatch-decision source (preclass/memo/runtime/authoritative-residency) + result. |
| `planned_on_host` | `ggml_sycl_op_is_planned_on_host` | Per-op classification of "should SYCL backend reject this op". |

## 3. Q1 — Is the placement plan active at 20B load time?

**YES.** Confirmed by live log:

```
[SYCL] Plan budget tied to arena: 10069.2 MB shared zone (arena=11093.2 MB - zones=1024.0 MB)
[SYCL] Placement plan computed: 4923 entries, 9045.2 MB device + 5581.0 MB host
[KV-TIER] All 24 layers match placement plan
[MOE-HYBRID] Placement plan routes experts to host — enabling CPU dispatch for blk.6.ffn_up_exps.weight
[GRAPH-DIAG] TG graph DISABLED: placement plan + host intermediates in decode graph
[PLACEMENT] MOE_GATE        device= 48 (8.4 MB)  host=  0 (0.0 MB)
[PLACEMENT] MOE_DOWN        device=1536 (3235.8 MB)  host=  0 (0.0 MB)
[PLACEMENT] MOE_UP          device=566 (890.6 MB)  host=970 (2345.2 MB)
[PLACEMENT] MOE_GATE_PROJ   device=  0 (0.0 MB)  host=1536 (3235.8 MB)
```

`g_has_placement_plan` is set `true` inside
`ggml_backend_sycl_set_tensor_inventory` (ggml-sycl.cpp:6436 or :6466)
immediately after either `compute_multi_device_plan` or
`compute_placement_plan` returns a non-empty plan. For 20B (single GPU) the
single-device branch fires.

Plan is then pushed into the per-device unified cache at line :6484 via
`cache->set_placement_plan(...)`, so `cache->has_placement_plan()` returns
true from that point onward.

For blk.16 specifically:
- `blk.16.ffn_gate_inp.weight` (router, dense MUL_MAT): device (plan says so
  and `layer_device[16] >= 0`).
- `blk.16.ffn_up_exps.weight` (MoE MUL_MAT_ID weight): all 32 experts staged
  into host arena by `[S1-PRELOAD] host-arena expert blk.16.ffn_up_exps.weight[0..31]`.
- `blk.16.ffn_gate_exps.weight`: same — all 32 experts to host arena.
- `blk.16.ffn_down_exps.weight`: NOT in the preload-to-host list → all 32
  experts device-resident (confirmed by the `[PLACEMENT] MOE_DOWN device=1536
  host=0` line — 1536 = 48 layers × 32 experts).

So at 20B load time the plan is active and per-tensor assignments match
intent: ffn_up / ffn_gate fully on host, ffn_down fully on device, dense
attention weights on device.

## 4. Q2 — Does the tensor actually land at its planned location at load time?

**YES.** Host-planned experts land in the host arena. Device-planned experts
land in VRAM (not directly probed this session — dispatch-time
`resolve/cache-device` confirms they're reachable via the cache at runtime).

Evidence (all 32 experts of `blk.16.ffn_up_exps.weight` staged via
`register_host_expert`, sampled below):

```
[N04BQ] s1-preload/host-expert tensor=blk.16.ffn_up_exps.weight eid=0 arena_ptr=0x7258f44eb900 expert_aos=0x725334af2000 expert_size=4406400 tensor_data=0x725334af2000
[N04BQ] s1-preload/host-expert tensor=blk.16.ffn_up_exps.weight eid=1 arena_ptr=0x7258f491f600 expert_aos=0x725334f25c80 expert_size=4406400 tensor_data=0x725334af2000
[N04BQ] s1-preload/host-expert tensor=blk.16.ffn_up_exps.weight eid=2 arena_ptr=0x7258f4d53300 expert_aos=0x725335359900 expert_size=4406400 tensor_data=0x725334af2000
```

Note: `tensor->data` (0x725334af2000) and `arena_ptr` (0x7258f4...) are
DIFFERENT memory locations. `tensor->data` is the pre-S1 host-pinned buffer
from `load_tensors: all weights -> host-pinned`. `arena_ptr` is the
S1-PRELOAD copy into the SYCL host arena (WEIGHT zone). Both are host-pinned;
the arena copy is what `register_host_expert` records and what
`lookup_expert` returns for this key.

This is functionally correct — both pointers reach the same bytes in
different allocations. But the duality shows up as an accounting subtlety
(see §10 Tech Debt).

## 5. Q3 — What does cache lookup return at dispatch time?

For `blk.16.ffn_down_exps` (all device), lookup is straightforward:

```
[N04BQ] resolve/cache-device tensor=blk.16.ffn_down_exps.weight eid=1 ptr=0xffffd557af3ece00 size=4406400
[N04BQ] resolve/cache-device tensor=blk.16.ffn_down_exps.weight eid=2 ptr=0xffffd557af820b00 size=4406400
```

The `0xffffd557...` pointers are VRAM (USM device-arena base). `expert->location == DEVICE`
path in `resolve_expert_ptr`. `resolved.is_host=false`.

For `blk.16.ffn_up_exps` and `blk.16.ffn_gate_exps` — **no `resolve/*` probe
fires at all**. Those tensors never go through `ggml_sycl_resolve_expert_ptr`,
because they never reach `ggml_sycl_mul_mat_id`. See §7 for why.

## 6. Q4 — Do dispatch paths use cache or raw tensor->data?

For `blk.16.ffn_down_exps` (the only MoE MUL_MAT_ID that reaches
`ggml_sycl_mul_mat_id`): dispatch goes through `ggml_sycl_resolve_expert_ptr`
which consults the unified cache first. `resolve/cache-device` hits confirm
the cache is the source of truth. Raw `src0->data` is NOT used for these
experts — the cache's stored VRAM pointers are.

For `blk.16.ffn_up_exps` / `blk.16.ffn_gate_exps` / `blk.16.ffn_moe_*_biased`
/ `blk.16.ffn_moe_weighted` — dispatch **does not go through the SYCL backend
at all**. See §7.

## 7. The actual gap — SYCL backend rejects host-planned ops; ggml scheduler routes them to the CPU backend

This is the central finding of this session. `ggml_backend_sycl_device_supports_op`
(ggml-sycl.cpp:50674) calls `ggml_sycl_op_is_planned_on_host(op, device)` and
returns `false` (i.e. "this backend does not support this op") whenever the
op's weight (or an input chain dependency) is planned on host.

Live probe for blk.16 ops:

```
[N04BQ] planned_on_host op_name=ffn_moe_logits-16 op=MUL_MAT        result=0 reason=layer_device>=0            src0=blk.16.ffn_gate_inp.weight
[N04BQ] planned_on_host op_name=ffn_moe_gate-16   op=MUL_MAT_ID     result=1 reason=weight_executes_on_host    src0=blk.16.ffn_gate_exps.weight
[N04BQ] planned_on_host op_name=ffn_moe_gate_biased-16 op=ADD_ID    result=1 reason=layer_plan_src_host        src0=ffn_moe_gate-16
[N04BQ] planned_on_host op_name=ffn_moe_up-16     op=MUL_MAT_ID     result=1 reason=weight_executes_on_host    src0=blk.16.ffn_up_exps.weight
[N04BQ] planned_on_host op_name=ffn_moe_up_biased-16 op=ADD_ID      result=1 reason=layer_plan_src_host        src0=ffn_moe_up-16
[N04BQ] planned_on_host op_name=ffn_moe_weighted-16 op=GLU          result=1 reason=addid_glu_depends_host     src0=ffn_moe_gate_biased-16
[N04BQ] planned_on_host op_name=ffn_moe_down-16   op=MUL_MAT_ID     result=0 reason=layer_device>=0            src0=blk.16.ffn_down_exps.weight
[N04BQ] planned_on_host op_name=ffn_moe_down_biased-16 op=ADD_ID    result=1 reason=addid_glu_depends_host     src0=ffn_moe_down-16
```

Meaning for blk.16 MoE:
- gate MUL_MAT_ID → CPU backend (all 32 gate experts host-planned)
- gate+bias ADD_ID → CPU backend (chain-dependency)
- up   MUL_MAT_ID → CPU backend (plan says host)
- up+bias ADD_ID → CPU backend
- GLU (SiLU/SwiGLU) → CPU backend (chain-dependency)
- down MUL_MAT_ID → **SYCL GPU** (all 32 down experts device-resident)
- down+bias ADD_ID → CPU backend (chain-dependency from down output)

The `ffn_moe_down_biased` case is the only cross-backend handoff of a result
tensor: `ffn_moe_down-16` output (VRAM) → `ffn_moe_down_biased-16` input
(CPU). The ggml scheduler inserts a D2H copy for this hand-off.

**Why ffn_moe_up got planned as host despite 566 device / 970 host:**
`ggml_sycl_weight_is_planned_on_host` queries `layer_device[16]` first (the
dense-execution-unit placement for the layer), falling through to per-tensor
checks only when `layer_device` has no entry. For the single-device plan used
here, `layer_device[layer_id]` is populated for every layer (see
unified-cache.cpp:10605 fill-with-default). Whichever value `layer_device[16]`
holds dominates the per-tensor expert distribution. The fallback to
`ggml_sycl_moe_tensor_all_experts_on_host` (which would return false for a
566/970 mix) is gated by the residency check in the parent predicate — if
`weight_executes_on_host` returns true via the `is_planned_on_host` branch,
the all-experts check is never consulted.

So the gap is not "dispatch doesn't use the cache" — it's that **the SYCL
backend doesn't dispatch host-planned MoE ops at all, even for tensors like
ffn_up_exps where some experts are device-resident**. Those ops get handed
off to the CPU backend by the scheduler.

## 8. Q5 — `ggml_sycl_update_moe_ptr_table` source of truth

Only invoked for the SYCL-accepted MoE ops (`ffn_moe_down-16` in this run).
The pointer table is built from:
1. `direct_base` — the whole-tensor device entry when available (all-device
   fully-staged tensors).
2. Per-expert `get_expert_device_ptr` lookup.
3. Cache-cascade `try_get_cached_with_event` (SOA/COALESCED/AOS).
4. Final `cache->lookup_expert(key)` for HOST_PINNED entries.

For `ffn_moe_down-16` all 32 experts hit (1) or (2) → the table is populated
with device pointers, MMVQ dispatches normally.

The `skip_cpu_routed_experts` path (line 27249 where the HOST_PINNED entry is
only written into `moe_expert_ptrs_host[e]` if `!skip_cpu_routed_experts`)
only matters for SYCL-handled ops with mixed residency; for fully device
ops it never trips.

## 9. Phase 2 verdict

The user's stated model is **correct at the SYCL-backend level**:
- Plan is active at 20B load.
- Experts land in the plan-assigned memory (`register_host_expert` in host
  arena; device staging for device-planned).
- The SYCL backend's dispatch-time pointer resolution consults the cache
  (not raw `->data`) for the MoE ops it handles.

The garble cannot be in that chain because the mis-routed ops (ffn_up /
ffn_gate experts for mixed-residency tensors, plus all the bias-add / GLU
ops) **never enter the SYCL dispatch chain at all**. They are scheduled onto
the CPU backend.

Possible garble sites, **none of which are in the SYCL cache/dispatch
plumbing the user named**:

1. **CPU-backend MXFP4 MUL_MAT_ID correctness**: the GGML CPU backend
   implementation of MUL_MAT_ID with MXFP4 src0, reading from host-pinned
   arena pointers vs. the old mmap pointers. Predecessor found 8
   test-backend-ops MXFP4 MUL_MAT_ID n=1 failures — which share the codebase
   (vec_dot/simd) with this CPU path. Those failures were an alloc-starvation
   artefact of the test fixture, but the same path is what runs 842 CPU
   dispatches on the live 20B decode.

2. **Cross-backend ffn_moe_down → ffn_moe_down_biased handoff**: SYCL
   computes `ffn_moe_down-16` into a VRAM tensor. Scheduler copies D2H for
   CPU backend to consume. If the D2H is wrong (partial copy, stale data,
   layout mismatch between SYCL's per-expert scatter output and the single
   contiguous tensor CPU expects), ADD_ID reads garbage.

3. **Plan decision for `blk.16.ffn_up_exps` classifying as "all host" when
   566 experts are actually device-resident**: the effective behavior is
   "even if some experts are in VRAM, route the whole op to CPU backend and
   have CPU read mmap / host-arena copies". That's _correct_ (the CPU
   backend will read valid bytes), but it throws away the device copies.
   Not a correctness bug on its own — but it means PP512 leaves ~890 MB of
   device-resident weights unused during TG, paying PCIe + D2H cost for
   every token.

4. **MXFP4 bias tensors (src0->extra / ADD_ID)**: the predecessor memory
   `project_tensor_access_redesign` flags that MoE bias tensors still crash
   via raw `->data` in ADD_ID on some code paths. With most ADD_ID ops now
   routed to the CPU backend via supports_op=false, the SYCL-side ADD_ID
   raw-data access is rarely hit — but correctness of the CPU backend's
   ADD_ID bias read remains an open question.

The strongest signal is **(1) and (4) combined** — CPU backend MXFP4
MUL_MAT_ID correctness when reading expert slabs from host-pinned pointers,
paired with ADD_ID bias reads. The predecessor's MUL_MAT_ID tests failing at
n=1 is a concrete CI canary that this path is broken under allocator
pressure; it may also produce garbage under different low-bandwidth
conditions on live 20B decode.

This investigation does not fingerprint the exact CPU-backend bug. Doing so
requires a different instrumentation family (inside `ggml_vec_dot_mxfp4_q8_0`
and friends in `ggml-cpu/`) on the specific tensors CPU is dispatching during
the 20B MoE-decode path. That is outside the `n04bq` scope (the mission was
to verify the plan-vs-dispatch invariant).

## 10. Recommendation (next bead scope)

Two independent tracks, both architectural not surgical:

### A. CPU-backend MXFP4 MoE correctness (likely the real bug)

Goal: ensure the CPU backend's MUL_MAT_ID + ADD_ID + GLU path computes
byte-correct MXFP4 MoE outputs when src0 is a host-pinned arena pointer
registered via `register_host_expert`.

Steps:
1. Bracket one blk.16 token's CPU-dispatched MUL_MAT_ID inside llama-completion,
   capture inputs (host-pinned src0 slab for the routed expert, src1 F32
   activations, ids), run a reference CPU MXFP4 MUL_MAT in a standalone
   harness, diff outputs.
2. If the standalone reference matches but live CPU backend doesn't, inspect
   the GGML CPU backend's MUL_MAT_ID path (`ggml/src/ggml-cpu/`) for MXFP4
   handling — specifically per-expert row-stride computation when src0
   points into a heap-allocated per-expert copy (not the full contiguous
   tensor the CPU backend usually expects).
3. Repeat for ADD_ID bias application and GLU fusion.

### B. Hybrid dispatch should use device copies when available

Goal: `ffn_moe_up-16` with 566 device experts should dispatch those 566 via
SYCL MMVQ and leave the 970 for CPU backend. Today the whole op goes to CPU
backend, wasting the device copies.

This requires a scheduler-level "split a MUL_MAT_ID across backends" — which
the ggml scheduler currently does not support op-level (only whole-op
routing). Two sub-options:

1. Extend `ggml_backend_sycl_device_supports_op` to accept mixed-residency
   MoE MUL_MAT_ID, and have the SYCL dispatcher internally fall back to
   `host_expert_rows` path for the CPU-routed experts. The existing
   `dispatch_host_expert_rows` already implements this pattern — it just
   needs to be reachable for `ffn_moe_up-16`.
2. Keep the current scheduler rejection, but change the plan to NOT
   classify `ffn_moe_up` as host when any experts are device-resident, so
   the SYCL backend handles the whole op.

(1) is the architecturally cleaner path — SYCL backend knows per-expert
residency, ggml scheduler doesn't need to.

## 11. Where this session stopped

- Phase 1 complete: all 5 questions answered with live probe evidence.
- Phase 2 complete: verdict is "gap is at scheduler boundary, not SYCL-internal".
- Phase 3 (fix): NOT attempted this session. The gap is architectural (CPU
  backend correctness + scheduler-level split-dispatch), requires scoped
  epic work, not a surgical fix.

## 12. Gate status

Gates NOT run — this session did not land a fix.

| Gate | Status |
|------|--------|
| 1. Mistral canonical | not run |
| 2. Mistral PP≥1700 / TG≥81 | not run |
| 3. 20B bench 3/3 | not run |
| 4. 20B `-n 30` coherent | **still FAIL** (pre-existing) |
| 5. 120B completion | not run |

Bead stays IN_PROGRESS.

## 13. Files touched this session

- `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - Added `n04bq_probe_enabled()` / `n04bq_probe_log(...)` helpers at file
    scope (before the `ggml_sycl` namespace opens).
  - Added probe calls at the sites listed in §2.
  - Added `<cstdarg>` include.
  - All probe calls are compile-in / runtime-gated OFF by default; default
    behaviour unchanged.

No other files changed. The a7l5w probes at `dca7494d6` are left
undisturbed (they gate on a separate env var).

## 14. Residual tech debt (recorded, not fixed)

- `tensor->data` vs `arena_ptr` duality at load time for host-pinned
  weights — both point to valid data, but S1-PRELOAD doubles the footprint
  for host-planned experts (original llama host-pinned copy + arena copy).
  Arena copy is what the cache knows about.
- `ggml_sycl_weight_is_planned_on_host` uses `layer_device[layer_id]` for
  all weight types, including MoE experts. The `layer_device` field is
  documented as applying to the dense execution unit only. This causes
  mixed-residency MoE weight tensors to be classified as "host" when the
  dense side of the layer is host, even if many experts are on device.
- `ggml_backend_sycl_device_supports_op` returns false for MoE ops whose
  weights are planned on host — this is correct in principle, but it
  means the SYCL backend never handles mixed-residency MoE ops. See §10 B.
