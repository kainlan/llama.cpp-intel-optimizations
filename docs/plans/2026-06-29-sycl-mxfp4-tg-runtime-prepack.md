# SYCL MXFP4 TG Runtime Route Plan: Selected-Expert Gate/Up Prepack

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to execute this plan with fresh implementers and fresh spec/quality reviewers. Lead owns B50/B580/model validation.

## Goal

Implement a default-off runtime candidate for B50 GPT-OSS MXFP4 token-generation that attacks the remaining gate/up+GLU bottleneck while preserving the current safe direct down-sum route.

Target outcome:

- preserve B50 `PP512 >= 1100 tok/s`;
- improve B50 `TG128` from the isolated baseline `37.05 tok/s` toward `>=45 tok/s`;
- keep all new runtime behavior explicit opt-in/default-off;
- never assume direct B50/B580 P2P;
- preserve unified-cache `mem_handle` ownership.

## Current Evidence

Evidence document: `activation/mxfp4-tg-runtime-baseline.md`.

Isolated runtime baseline commit: `b08e732cd` on `/Apps/llama.cpp-mxfp4-tg-runtime`.

Current safe direct route:

```bash
export GGML_SYCL_MOE_PHASE_MATERIALIZE=1
export GGML_SYCL_MOE_PHASE_BULK_XMX=1
export GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
```

Lead-owned B50 evidence:

```text
count gate: exact `1, 2, 3, 4, 5`, fatal-free
pp512: 1234.06 ± 9.51 tok/s
tg128:   37.05 ± 0.06 tok/s
profile: gateup_glu ~5.55-5.72 ms / 48 calls, down ~0.73 ms / 24 calls
```

Rejected or non-winning existing candidates:

- `GGML_SYCL_MOE_LAYER_EXECUTOR=1`: count/fatal-free but slower, `TG128 34.90`.
- `GGML_SYCL_MOE_FUSED_GLU_Q8=1`: count/fatal-free but not faster, `TG128 36.87`.
- `GGML_SYCL_MOE_DOWN_SUM_FUSION=1 + GGML_SYCL_MOE_DOWN_SUM_FROM_BIAS=1`: faster but count-incorrect (`1, 2, 3.`), rejected.

Dry-run suite ranking still points to fused-layer/prepack-style routes, but the current runtime layer executor is not the implementation to promote. The next route should be a narrower selected-expert gate/up prepack path that leaves the direct down-sum path in place.

## Non-Negotiable Rules

- Runtime behavior is default-off. Add only explicit opt-in env gates.
- If a route cannot prove handle/event/capacity/metadata correctness, it must fall back to the current safe route, not partially execute.
- All GPU, host-pinned, staging, scratch, graph-temp, and layout allocations must flow through unified-cache APIs and be retained by `mem_handle` for the lifetime of CPU use, SYCL events, command graphs, pointer tables, and queued kernels.
- Raw pointers are transient ABI views only. Do not store raw pointers as ownership state or cache keys.
- Do not add direct `sycl::malloc_device`, `sycl::malloc_host`, `sycl::free`, raw TLSF allocation, or side caches outside unified-cache implementation.
- No blocking D2H copies on the hot path.
- No route that can hard-fail the device (`UR_RESULT_ERROR_DEVICE_LOST`, Level Zero error 20) may remain reachable except as a rejected diagnostic in a commit that is then removed.
- Workers must not run B50/B580 model gates, `sycl-ls`, `/Storage/GenAI/models`, multi-GPU selectors, direct P2P checks, or non-dry-run GPU probes. Lead only.

## Route Shape

New opt-in env:

```text
GGML_SYCL_MOE_GATEUP_PREPACK=1
```

Optional diagnostics:

```text
GGML_SYCL_MOE_GATEUP_PREPACK_TRACE=1
GGML_SYCL_MOE_GATEUP_PREPACK_VALIDATE=1
```

The candidate route applies only to GPT-OSS-style MXFP4 TG gate/up pairs when all of these hold:

- decode/TG shape (`ne12 == 1`) and known MoE layer id;
- gate and up selected expert ids are available and deterministic for the current token/layer;
- gate/up tensors resolve through unified-cache-owned `mem_handle` leases on the submit device;
- selected experts are resident or materializable through the existing planner path;
- scratch capacity can be obtained through unified-cache and retained until the compute event completes;
- graph recording/replay metadata can retain handles, or route rejects while graph recording;
- direct down-sum route remains valid or route falls back before prepack execution.

Implementation concept:

1. Reuse existing selected-expert metadata and pointer-table construction near the current `packed-q8-m2` gate/up path.
2. Prepack only the active gate/up experts for one layer/token into a DPAS-friendly scratch layout. This is a new scratch artifact, not a persistent raw-pointer cache.
3. Launch a compact gate+up+GLU kernel that consumes the prepacked scratch and writes the normal GLU output expected by the existing direct down-sum route.
4. Publish profile evidence under a distinct route label, e.g. `gateup-prepack-dpas`, and record prepack/compute timings separately.
5. Reject/fallback on any malformed shape, missing handle, stale metadata, non-resident expert, event dependency gap, or capacity failure.

## Task Breakdown

### Task 1: Policy, diagnostics, and fail-closed tests

File scope:

- `ggml/src/ggml-sycl/ggml-sycl-test.hpp`
- `ggml/src/ggml-sycl/ggml-sycl.cpp`
- `tests/test-sycl-moe-gateup-prepack-policy.cpp`
- `ggml/src/ggml-sycl/CMakeLists.txt`

Acceptance:

- Add pure policy helpers for env/default-off, shape gating, metadata/handle/capacity reject reasons, and graph-recording behavior.
- Unit tests prove default-off reject, malformed shape reject, missing handle reject, capacity reject, graph-recording reject, and accepted case.
- No runtime dispatch change yet.
- Worker-safe tests only.

### Task 2: Unified-cache scratch descriptor and handle retention

File scope:

- `ggml/src/ggml-sycl/moe-layer-plan.hpp`
- `ggml/src/ggml-sycl/ggml-sycl.cpp`
- `tests/test-sycl-moe-gateup-prepack-scratch.cpp`

Acceptance:

- Add a descriptor for selected gate/up prepack scratch that owns `mem_handle` leases and explicit event dependencies.
- Scratch allocation uses existing unified-cache allocation APIs only.
- Tests prove handles are retained until the returned event/descriptor lifetime ends and raw pointers are not used as identity.
- No model execution.

### Task 3: Device prepack kernel skeleton and CPU reference test

File scope:

- `ggml/src/ggml-sycl/mmvq.hpp`
- `ggml/src/ggml-sycl/mmvq.cpp`
- `ggml/src/ggml-sycl/tests/test-xmx-moe-mxfp4.cpp`

Acceptance:

- Add a default-off internal prepack helper for selected gate/up MXFP4 rows into the scratch descriptor layout.
- Add a CPU/reference comparison test for layout packing on small synthetic tensors.
- Preserve existing `packed-q8-m2` route when env is off.
- No direct model/hardware gate by workers.

### Task 4: Gate+up+GLU prepack compute path, opt-in only

File scope:

- `ggml/src/ggml-sycl/mmvq.cpp`
- `ggml/src/ggml-sycl/mmvq.hpp`
- `tests/test-sycl-moe-gateup-prepack-policy.cpp`

Acceptance:

- Wire `GGML_SYCL_MOE_GATEUP_PREPACK=1` to attempt the prepack+compute path only after Task 1 policy accepts.
- Route label/profile evidence includes `gateup-prepack-dpas`.
- On any prepack or compute submission failure, route falls back before publishing partial artifacts.
- The path retains all source/scratch handles until queued work completes.
- Existing current safe route remains unchanged when env is off.

### Task 5: Lead-only B50 validation matrix update

File scope:

- `activation/mxfp4-tg-runtime-baseline.md`
- `activation/mxfp4-gateup-prepack-validation.md`

Acceptance:

- Document exact lead-only commands for count, PP/TG, profile, fatal-marker scans, route-label requirement, and fallback-absence requirement.
- Workers do not run these commands.
- Promotion checklist repeats: exact count, fatal.total 0, required route label, forbidden fallback absence, MXFP4 profile evidence, `PP512 >= 1100`, and `TG128 >= 45`. A lead may record a near-target result only as non-promoted follow-up evidence when `42.0 <= TG128 < 45.0`, route evidence is clean, and the profile shows gate/up+GLU `<= 4.2 ms/token`; that does not authorize default-on promotion.

### Task 6: Docs and final review handoff

File scope:

- `docs/backend/SYCL.md`
- this plan file if already tracked

Acceptance:

- Document `GGML_SYCL_MOE_GATEUP_PREPACK` as experimental/default-off and not promoted.
- Document known rejected routes from this phase: current layer executor, fused GLU Q8, from-bias down-sum fusion.
- Final integration review passes.

## Lead-Owned Validation Commands

Workers must not execute these.

Count:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:1
export GGML_SYCL_MOE_PHASE_MATERIALIZE=1
export GGML_SYCL_MOE_PHASE_BULK_XMX=1
export GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
export GGML_SYCL_MOE_GATEUP_PREPACK=1
./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -ngl 99 \
  -cnv -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0
```

Bench:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:1
export GGML_SYCL_MOE_PHASE_MATERIALIZE=1
export GGML_SYCL_MOE_PHASE_BULK_XMX=1
export GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
export GGML_SYCL_MOE_GATEUP_PREPACK=1
./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -ngl 99 -fa 1 -p 512 -n 128 -r 3
```

Profile:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:1
export GGML_SYCL_MOE_PHASE_MATERIALIZE=1
export GGML_SYCL_MOE_PHASE_BULK_XMX=1
export GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
export GGML_SYCL_MOE_GATEUP_PREPACK=1
export GGML_SYCL_MXFP4_TG_PROFILE=1
export GGML_SYCL_MOE_PROFILE=1
./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -ngl 99 -fa 1 -p 64 -n 32 -r 1
```

## Stop/Kill Criteria

- Count output does not start with exact `1, 2, 3, 4, 5` or `: 1, 2, 3, 4, 5`.
- Any fatal/error/device-lost marker appears.
- Required route label `gateup-prepack-dpas` is missing when env is enabled.
- Fallback path silently runs while claiming prepack route success.
- `PP512 < 1100`.
- `TG128 < 40.0 tok/s` after optimization, or `TG128 <= 37.05 tok/s` relative to the current direct baseline.
- `40.0 <= TG128 < 42.0 tok/s` after two implementation iterations; keep only as a documented negative probe, not an active route.
- Prepack+compute gate/up+GLU profile bucket remains `> 4.8 ms/token`, or saves `< 0.8 ms/token` versus the `5.6 ms/token` direct-route baseline. Promotion-quality evidence should show gate/up+GLU `<= 4.2 ms/token` unless TG throughput already reaches `>= 45 tok/s`.

## Initial Expected Outcome

This plan is not allowed to make the route default-on. A successful implementation produces opt-in evidence. Promotion/default changes require a later decision after B50 and then B580 validation.
