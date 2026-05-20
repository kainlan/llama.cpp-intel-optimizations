# MoE Layer Execution Architecture

Status: accepted for implementation
Date: 2026-05-18
Tracking: `llama.cpp-v90xn.30.8` through `llama.cpp-v90xn.30.18`

## Context

GPT-OSS20B decode on the Arc Pro B50 is now fully VRAM-resident for the
profiled all-GPU path, but token generation is still dominated by MoE expert
execution. The current hot path is organized around many `MUL_MAT_ID`
decisions, pointer-table materialization, graph skip state, and historical
fast/slow branches. That structure predates the unified cache and smart
`mem_handle` policy.

Recent B50 GPT-OSS20B profile facts:

- FA-on TG128 is about 26 tok/s.
- Warmed op timing spends about 62.7% in `MUL_MAT_ID`, about 18.7% in dense
  `MUL_MAT`, and about 2.2% in `FLASH_ATTN_EXT`.
- The MoE profile reports 24 layers, top-k 4, and 288 selected expert-role
  entries per token.
- The all-VRAM profile has no CPU experts, no host rows, no missing rows, no
  layout mismatch, and no weight-load or PCIe events.
- Single-stream row aggregation has `avg_rows_per_group=1.00`; batching proves
  throughput headroom by reaching roughly 69-72 tok/s at `--tg-batch 32`.
- Hardware telemetry during TG shows high clocks and power, so the limiter is
  kernel/dataflow efficiency rather than downclocking or host fallback.

Follow-up microbench facts from 2026-05-18:

- The diagnostic full-layer sequence `gate/up/GLU -> GLU Q8_1 SOA -> down`
  takes about 0.80-0.82 ms/layer on B50 for the GPT-OSS local expert shape
  `[2880, 2880, topk=4]`.
- The split in the same run is about 0.52 ms for gate/up/GLU and about
  0.27 ms for down, leaving only about 0.02 ms/layer for the GLU-to-Q8
  artifact and launch glue.
- Token-row variants of the current SOA layer sequence scale linearly and stay
  around 64-65 GB/s effective bandwidth; they do not unlock extra bandwidth.
- A oneDNN MXFP4 dense-equivalent probe for one selected role has a model-load
  reorder style cost over 100 ms and a reordered GEMM time around 0.285 ms for
  `M=11520,N=1,K=2880`, which is not faster than the current SOA down role.
- A first DPAS-ready selected down-role prototype now materializes an
  XMX-tiled MXFP4 layout before the measured hot path and dispatches a
  hardware-capability-selected joint-matrix kernel from Q8_1 SOA activations.
  It validates against the current SOA kernel, but it is not competitive for
  GPT-OSS shape: on B50, SOA down is about 0.270 ms/layer while XMX-tiled is
  about 0.734 ms at `tiles_n=1` and about 2.32 ms at the queried
  `tiles_n=4`. B580 shows the same shape: about 0.182 ms SOA versus about
  0.650 ms XMX-tiled `tiles_n=1`.
- The B50 K-sweep isolates the issue: XMX-tiled `tiles_n=1` beats SOA at
  `K=32` (about 14.5 us versus 60.8 us), ties around `K=256`, then loses at
  `K=1024` and `K=2880`. The blocker is therefore per-K-block MXFP4/Q8 scale
  extraction and accumulator materialization around DPAS, not the selected
  expert pointer table or model-load tiling itself.
- A raw-accumulate diagnostic that removes per-K-block scale application from
  the naive joint-matrix loop still loses badly at full GPT-OSS shape. On B50
  `M=2880,Nselected=4,K=2880`, SOA r4 is about 268 us, exact XMX-tiled tn1 is
  about 729 us, and raw XMX-tiled tn1 is about 568 us. On B580, SOA r4 is
  about 181 us, exact XMX-tiled tn1 is about 642 us, and raw XMX-tiled tn1 is
  about 499 us. This rules out the current joint-matrix mapping itself for
  single-token long-K selected-expert decode, not only the MXFP4/Q8 scale
  placement.

These measurements rule out wrapper-only fusion, current-SOA token-row batching,
oneDNN MXFP4 as-is, and the naive MXFP4 XMX-tiled joint-matrix family as the
single-stream TG breakthrough. Any future XMX executor needs a materially
different long-K design and must beat the current SOA selected-role baselines
before production promotion. The near-term implementation focus is therefore
the existing SOA/ESIMD-style selected-row executor, layer-level fusion, and
graph/event scheduling through planner-owned smart handles.

## Decision

MoE decode will move to one planner-owned layer execution model. Kernel variants
may differ, but ownership, routing, layout, residency, graph, and fallback
semantics must flow through the same layer plan.

The canonical layer DAG is:

```text
route / top-k ids
  -> activation artifact
  -> selected gate expert rows
  -> selected up expert rows
  -> activation/product artifact
  -> selected down expert rows
  -> accumulated MoE layer output
```

The unified cache planner owns every allocation and returns smart
`mem_handle`s. Model load or cache materialization chooses expert residency and
layout from `capability + model shape + budget + policy`. Decode dispatch
consumes a `moe_layer_decode_plan`, resolves smart handles for the active
device, and selects an executor from actual residency, layout, shape, and
capability.

Raw pointers are allowed only as transient kernel ABI payloads derived from
canonical handles immediately before launch or graph capture. They must not be
cached as ownership state.

## Invariants

1. Unified cache and smart `mem_handle`s own memory identity and lifetime.
2. Holding a handle is the lifetime pin; graph entries and async submissions
   retain handles rather than inventing graph-local raw pointer pinning.
3. Handles that refer to the same backing allocation must hash and compare the
   same once resolved.
4. Pointer tables are launch ABI data. Their source of truth is the selected
   expert handle group.
5. Default executor selection is planner/capability driven. Environment
   variables are diagnostic force, disable, or trace controls only.
6. Board names are diagnostic text only. B50/B580 differences must come from
   queried capabilities, placement budget, layout support, and measured policy
   thresholds.
7. A B50 all-VRAM GPT-OSS20B run must not route selected experts to CPU, stage
   weights from host to GPU for decode, or rely on user-set performance env
   vars.
8. When a model does not fit in VRAM, host-resident expert weights execute on a
   host/CPU-capable path. Decode must not stream planned host weights to GPU.
9. Graph capture must use explicit event dependencies; no queue-wide waits or
   host reads of recorded-only route/control data are allowed.

## Data Structures

`ggml/src/ggml-sycl/moe-layer-plan.hpp` is the ABI home for the layer plan:

- `moe_gate_up_pair`: topology discovered from the ggml graph.
- `moe_layer_decode_role_plan`: selected experts for one role, with canonical
  handles, route rows, and ready events.
- `moe_layer_decode_artifact_plan`: activation/control/intermediate tensor
  handle plus resolved metadata.
- `moe_layer_decode_plan`: one layer-level contract consumed by MMVQ, XMX,
  fused, batched, and mixed-residency executors.
- `moe_layer_grouped_route_view`: grouping view keyed by handle identity,
  layout, residency, device, role, and expert id.

Executor-specific raw pointer arrays are derived from these structures and have
no independent ownership.

## Fallback Taxonomy

Fallback or rejection reasons must be explicit and low-noise:

- topology mismatch
- unsupported tensor type or activation
- unsupported shape or top-k
- unsupported layout
- missing handle
- handle resolve failure
- layout mismatch
- host residency where GPU-only executor was requested
- mixed residency requiring split execution
- missing device capability
- graph-incompatible dynamic shape
- correctness guard
- diagnostic override

Counters such as "host routing" must mean actual host execution, not a failed
probe boundary.

## Layout Policy

MoE expert layout is a load/materialization decision. The policy input is:

```text
device capability + model shape + residency + budget + measured executor status
```

VRAM-resident experts choose the best validated GPU layout, such as SOA or an
XMX-tiled layout when that path is proven. Host-resident experts choose a
CPU/host-friendly layout. Decode must not convert expert weights per token.

Capability records should include SLM/local memory, subgroup sizes, max
workgroup size, XMX/joint_matrix support, relevant data type support, graph
support, allocation caps, and `unknown` for facts the runtime cannot query.
Memory bandwidth may be recorded as measured metadata, but it must not be
hardcoded into correctness policy.

## Executor Plan

Implementation proceeds in this order:

1. Define this architecture and make the layer plan a first-class header.
2. Replace persistent expert pointer/extras identity with selected expert
   handle groups. Keep pointer arrays as transient ABI payloads.
3. Make MoE layout a capability-driven planner decision at load/materialization
   time.
4. Route existing MMVQ gate/up/down dispatch through the graph-safe layer ABI
   without changing behavior.
5. Implement a direct MXFP4 selected-expert primitive for the selected layout.
6. Implement layer-level gate/up/activation/down fusion or the best measured
   partial fusion.
7. Integrate continuous-batch grouping through the same layer ABI.
8. Add mixed-residency split execution without GPU weight streaming.
9. Promote only after correctness, residency, graph, performance, and cleanup
   gates pass.

## Validation

Required gates before default promotion:

- Deterministic GPT-OSS short completion or token-hash comparison.
- B50 GPT-OSS PP512/TG128 with FA off and FA on.
- B50 MoE profile showing token time, MoE time, kernel time, gate/up, down,
  quant/artifact, and effective GB/s.
- B50 all-VRAM residency proof: zero host rows, zero missing rows, zero CPU
  experts, no H2D/D2H weight staging.
- Mixed-residency forced-budget proof: device experts run on GPU executors,
  host experts run on host/CPU-capable executors, and outputs merge correctly.
- Graph replay/direct-dispatch proof for alternating PP/TG and changing decode
  shapes.
- B580 Mistral Q4_0 FA off/on regression guard.

## Cleanup

After the new executor is promoted, obsolete sorted/scatter wrappers, duplicate
fast/slow dispatch routes, stale pointer caches, and env-only default paths
must be removed or explicitly demoted to diagnostic-only code.
