# Placement Envelope Validation Findings

**Date:** 2026-04-25  
**Epic:** `llama.cpp-3h5gm` Unified Memory Placement  
**Scope:** upfront validation for multi-user / agent-team serving semantics before Track A implementation.

## Question

Does the long-term plan make more sense as "pass each user's context into model load" or as "load one model with a placement envelope, then plan actual contexts/slots separately"?

## Finding

The correct long-term target is **one loaded model with a declared placement envelope**, plus per-context / per-slot compute planning inside that envelope.

The current llama.cpp architecture already points in this direction:

- `common_context_params_to_llama()` maps `params.n_parallel` to `cparams.n_seq_max`, so parallelism is currently expressed as multiple sequences/slots inside one context rather than one model load per user.
- Server CLI uses `-np/--parallel` as "number of server slots".
- `examples/batched` computes aggregate KV demand as prompt plus generated tokens across `n_parallel`, then creates one `llama_context` and one batch with multiple sequence IDs.
- KV-cache construction consumes `cparams.n_ctx_seq`, `cparams.n_seq_max`, `cparams.n_ubatch`, and `cparams.kv_unified`, which is exactly the shape a placement envelope needs to reserve aggregate capacity while allowing per-sequence use.

## Evidence

| Evidence | File / lines | Interpretation |
|---|---|---|
| Runtime planner hints already flow from common params into model params | `common/common.cpp:1696-1697` | Existing fork already started passing runtime capacity hints to the GPU planner. Rename/reframe these as placement-envelope fields, not per-user context. |
| Context params carry `n_seq_max = params.n_parallel` | `common/common.cpp:1719-1725` | Parallel sessions are represented as max sequences/slots in one context. |
| Server `--parallel` means slot count | `common/arg.cpp:1966-1977` | Server-facing semantics match envelope `max_active_slots`. |
| Batched example creates aggregate KV demand for `n_parallel` and one context | `examples/batched/batched.cpp:59-99` | Existing examples validate aggregate-envelope reasoning: KV must be sized for all active sequences. |
| KV cache construction consumes context sequence and ubatch capacity | `src/llama-model.cpp:9216-9226` | KV allocation already depends on per-context aggregate capacity, not just model metadata. |

## What This Validates

- A1/A2 should pass a **placement envelope** into model load, not a per-user context object.
- Actual `llama_context` / server slot creation should still build a separate compute plan for the requested shape.
- `n_ctx`, `n_seq_max`, `n_ubatch`, and FA policy are capacity inputs. For server/agent teams, `n_seq_max` / slot count and total KV tokens are as important as max per-slot context.
- Exceeding the envelope should fail clearly at context/slot creation with "reload with a larger envelope" rather than silently shrinking or first-come-first-served starving another slot.

## What Is Still Not Proven

This was code-inspection validation, not a runtime canary. The following still need empirical proof before the claims are treated as fully backed:

1. **D0.5 runtime canary:** load one model with an envelope, create/admit multiple slots with varied context sizes, verify plan dump shows aggregate KV/runtime capacity and admission rejects over-envelope requests.
2. **D0.6 concurrency canary:** repeated and concurrent context/slot creation/destruction under TSAN or stress harness does not race and does not leak arena allocations.
3. **D0.7 allocation-ownership audit:** enumerate remaining SYCL direct allocations / raw `tensor->data` bypasses and classify them as planned, exempt, or bugs under the unified-cache-owns-memory goal.
4. **D0.8 CPU-dispatch-vs-streaming benchmark:** verify CPU dispatch for host-resident weights beats GPU streaming over PCIe for dense and MoE cases before treating that as benchmark-backed.

## Plan Impact

The main design plan was updated on 2026-04-25 to use "placement envelope" terminology and to keep per-context/per-slot compute planning separate. The bead tasks should use this language so implementers do not accidentally bake one user's context into model load.

## Additional Upfront Validation Added 2026-04-25

## Runtime Probe Added 2026-04-25

Command shape:

```bash
source /opt/intel/oneapi/setvars.sh --force
env -u ONEAPI_DEVICE_SELECTOR \
  LD_LIBRARY_PATH=build/bin:$LD_LIBRARY_PATH \
  SYCL_DEVICE_FILTER=level_zero:gpu \
  timeout 180 ./build/bin/llama-batched \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 0 -np 4 -n 16 -p "Hello"
```

Findings:

- Device discovery is currently selector-sensitive. `ONEAPI_DEVICE_SELECTOR=level_zero:*` caused the dpct device manager to report no devices, while unsetting it and using `SYCL_DEVICE_FILTER=level_zero:gpu` exposed both local GPUs and initialized unified-cache arenas.
- Runtime context creation confirms the current context side already has envelope-like semantics: the run logged `n_seq_max = 4`, `n_ctx = 1024`, `n_ctx_seq = 256`, and a 128 MiB KV buffer for 4/4 sequences.
- Model-load placement still logged `n_ctx=32768 (conservative)` for KV sizing before context creation, despite the small runtime context. That means A1/A2 remain necessary: the plan must move from model metadata / conservative defaults to the caller-declared placement envelope at model load.
- The canary did not complete generation because the non-unified KV split failed for coupled sequences: `decode: failed to find a memory slot for batch of size 2`. This does not refute the envelope model; it means the runtime admission canary must use a supported slot/server path or a CLI that accepts unified KV for this scenario.

Status: **D0.5 is PARTIAL, not closed**. The architecture direction is backed by source inspection and runtime context logs, but model-load placement is not yet envelope-driven and over-envelope admission is not yet proven.

### D0.7 ownership audit is required before epic close

A quick scan confirms the broader unified-cache ownership goal is not yet proven by 3h5gm alone:

- Direct allocation wrappers / SYCL allocation references still appear across production SYCL files, including `ggml-sycl.cpp`, `unified-cache.cpp`, `common.hpp`, `unified-kernel.cpp`, `mmvq.cpp`, `fattn.cpp`, and others. Some are legitimate allocator wrappers or bootstrap paths, but the set is too large to treat as closed without classification.
- Raw `tensor->data` references remain in `ggml-sycl.cpp`, `common.hpp`, and a few support files. Some are resolver implementation or backend buffer-boundary code, but D0.7 must classify every inference hot-path reference before the epic claims "ggml functions don't bypass unified_cache / smart handles."
- Existing fork fields `n_ctx_hint` and `n_ubatch_hint` already exist in `llama_model_params` (`include/llama.h`) and are populated in `common/common.cpp`, which reinforces that A1 should evolve these into explicit placement-envelope inputs rather than introduce a parallel per-user context mechanism.
- Deeper 2026-04-25 scan found 183 non-comment `sycl::malloc_*` / `sycl::free` lines outside tests/docs, 94 even after excluding allocator internals/common wrappers, 228 non-comment `tensor->data` lines, and 20 `data_device[]` lines. This proves the current code is still a compatibility bridge, not the intended handle-only end state.
- Architecturally, segmented or relocatable host storage must not be exposed to shared ggml as a fake flat `get_base()+offset` pointer. The SYCL backend must either provide a real stable virtual/logical base or remap tensors into handles before op dispatch.

This scan does not close D0.7. It just proves the audit is necessary and should block epic close / D3 completeness.

Detailed audit note: [`d0.7-ownership-audit.md`](d0.7-ownership-audit.md).

### D0.5/D0.6 canary results are now reproducible

A buildable canary was added at `tests/test-placement-envelope-canary.cpp`.
It is intentionally not in default `ctest` because the current branch is
expected to fail the over-envelope admission assertion until A6 lands.

Results are documented in
[`d0.5-d0.6-placement-envelope-canary.md`](d0.5-d0.6-placement-envelope-canary.md):

- D0.5 context geometry PASS: repeated and concurrent context creation produced
  the expected `n_ctx`, `n_ctx_seq`, and `n_seq_max` for both split KV and
  unified KV.
- D0.5 admission FAIL: an over-envelope context (`envelope_ctx=1024`,
  requested `n_ctx=2048`) succeeded. A6 must add deterministic rejection.
- D0.6 lifecycle FAIL: repeated create/destroy grew host pinned/SCRATCH arena
  by 2 GiB chunks for ~30-36 MiB compute-buffer requests. This does not meet
  the "arena allocations return to unified_cache" requirement.

Implementation follow-up: `llama.cpp-3h5gm.1` tracks the SCRATCH/pinned
lifecycle fix required before D0.6 can pass. The intended fix is two-layer:
reserve/admit host compute capacity instead of additive growth, and split
context/slot-lifetime `HOST_COMPUTE` / `HOST_RUNTIME` from graph-reset
`HOST_SCRATCH`.

### D0.8 benchmark is required before deleting streaming paths

The plan's CPU-dispatch-over-GPU-streaming policy is architecturally plausible, but the first benchmark attempt did **not** produce backing data. Existing kernel-scale streaming benches were run with `SYCL_DEVICE_FILTER=level_zero:gpu`:

- `GGML_SYCL_MMVQ_BENCH=1 GGML_SYCL_MMVQ_BENCH_MODE=both ./build/bin/test-mmvq-q8-0-streaming-bench`
- `GGML_SYCL_MMVQ_BENCH=1 GGML_SYCL_MMVQ_BENCH_MODE=mmq ./build/bin/test-mmvq-q8-0-streaming-bench`

Both attempts failed with Level Zero `UR_RESULT_ERROR_DEVICE_LOST` during the GPU streaming/compute path. The MMQ run also showed the benchmark's forced 1 MiB unified-cache budget refusing arena reservation, then falling back to per-entry allocation before device loss.

Follow-up smaller sequential runs also failed: a 1024x1024 run failed with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on the streaming memcpy path, a 256x256
run failed with `UR_RESULT_ERROR_DEVICE_LOST`, and `test-sycl-cpu-dispatch`
segfaulted immediately (`exit=139`). The available local evidence therefore
does not prove either side of the policy comparison.

Status: **D0.8 is BLOCKED/UNPROVEN**. C3 should stay gated so `GGML_SYCL_FORCE_STREAMING` and related streaming paths are not deleted until either:

1. the benchmark path is fixed and dense + MoE results show CPU dispatch is faster/safer, or
2. D14 is revised to keep a streaming fallback for cases where CPU dispatch is not proven.

Detailed benchmark note: [`d0.8-benchmark-attempt.md`](d0.8-benchmark-attempt.md).

Implementation follow-up: `llama.cpp-3h5gm.2` tracks repair of the
CPU-dispatch and streaming benchmark proof harnesses.

### D0.9 remaining proof audit found additional gaps

The deeper proof pass found four additional issues that should stay explicit
before implementation starts:

- The canonical D0.1 skeleton determinism canary was stale when this audit
  started, but has now been replaced and rerun. It invokes
  `test-mini-context-prototype` and passes on Mistral 7B dense plus GPT-OSS 20B
  active SWA+MoE, closing follow-up `llama.cpp-2t09r`.
- State-space coverage is not just missing a local fixture. Mamba/Mamba2 and
  Plamo2 graph builders emit `ggml_ssm_scan`, while the SYCL backend has no
  `GGML_OP_SSM_SCAN` support/dispatch entry. The plan cannot claim
  state-space support until both fixture coverage and SYCL `SSM_SCAN`
  execution/routing are addressed.
- Multi-device CPY-node visibility is refuted by the 2026-04-26 synthetic
  scheduler proof. D0.3a exposes B580+B50 with `GGML_SYCL_SPLIT_RATIO=50,50`
  and records stable scheduler copy-edge tensors, but `0` `GGML_OP_CPY` nodes.
  C2 must plan scheduler transfer edges instead of synthetic CPY graph nodes.
- `GGML_SYCL_DIRECT_ALLOC_GUARD=2` catches a runtime
  `kv_tier:migrate` direct host allocation after KV-clear fallback, confirming
  that D0.7's direct-allocation concern has runtime evidence.

Detailed proof note: [`d0.9-remaining-proof-audit.md`](d0.9-remaining-proof-audit.md).
