# Planner Validation Summary — Phase A/B/C Outcomes

**Date:** 2026-04-24
**Branch:** `feature/sycl-coalescing`
**Plan:** [`docs/plans/2026-04-24-planner-validation-phases.md`](../../2026-04-24-planner-validation-phases.md)
**Main epic:** `llama.cpp-3h5gm` — [`docs/plans/2026-04-22-unified-memory-placement-plan.md`](../../2026-04-22-unified-memory-placement-plan.md)

This document aggregates the empirical validation work from Tasks 1-7
(planner-validation team, 2026-04-24) and maps each of the 8 uncertainty
items raised in the audit to its current `validated` / `partially
validated` / `refuted` / `still-blocked` status.

## Uncertainty audit — before/after table

Severity column from the original audit. Status reflects evidence
collected as of `833532266` (Task 4 polish) plus the `33e05fa36`
broader-envelope finding (Task 10) that was queued to resolve E1.

| # | Uncertainty                                                                                                       | Severity | Test       | Result                          | Status                              |
|---|-------------------------------------------------------------------------------------------------------------------|----------|------------|---------------------------------|-------------------------------------|
| 1 | C2's `plan.ops` keying on op_id is stable across repeated `graph_reserve`                                         | HIGH     | Task 1 / D0.3 | PASS (`607fca77e`)           | **validated**                       |
| 2 | A3a's "single-shape sizing suffices" generalizes beyond Mistral 7B + GPT-OSS 20B                                  | HIGH     | Task 2 / D0.2 | PASS (`a3e0f29b7`)           | **partially validated** — Mistral-dense + GPT-OSS-MoE families only; SWA + state-space architectures absent locally, untested |
| 3 | Plan doc reflects the 2026-04-22 deprecate-layer-streaming directive consistently                                 | MEDIUM   | Task 3       | PASS (`6f7968833`, polished `209303892`) | **validated** — all 7 stale references annotated or rewritten; CPU-dispatch is now the canonical answer |
| 4 | E1 / m09zb root cause identified, mitigation chosen                                                                | HIGH     | Task 4 / Task 10 | PARTIAL — minimal-repro PASS, broader-envelope FAIL (`3cffaae9f` initial RCA, `33e05fa36` broader-envelope finding) | **partially validated** — bare async-H2D + event.wait() refuted as the trigger pattern; trigger isolated to first H2D in `ggml_backend_sycl_buffer_set_tensor` after `ggml_backend_sycl_init`. 4 ranked hypotheses, 2 next-step probes outlined. Root cause not yet pinned. |
| 5 | A3a's mini-context + A3b's FA auto-detection produce sizes/decisions byte-identical to a real context              | HIGH     | Task 5       | NOT RUN — blocked by E1         | **still blocked** — re-run once E1 unblocks GPU model load |
| 6 | A7's direct mmap → device `ggml_backend_tensor_set` achieves byte-exact readback within 60 s                       | HIGH     | Task 6 / D0.4 re-run | NOT RUN — blocked by E1   | **still blocked** — D0.4 binary already exists; remains INCONCLUSIVE pending E1 |
| 7 | Weight priority order `NORM_EMBED > ATTENTION > FFN > MOE_DOWN > MOE_UP > MOE_GATE_PROJ` outperforms alternatives  | MEDIUM   | Task 7       | NOT RUN — blocked by E1 (model load wedge under `VRAM_BUDGET_PCT=30`) | **still blocked** |
| 8 | The plan doc is internally consistent, the canary-results section reflects current findings, and the bead graph is honest about which tracks are unblocked | LOW | Task 8 (this doc) | This document + main plan edits | **validated** by the act of writing this summary; main plan canary-results section already reflects items 1-3 |

Three items fully validated (1, 3, 8). One partially validated with
clear architecture-coverage caveat (2). One partially validated with
trigger isolated but root cause still narrowing (4). Three still
blocked on E1 (5, 6, 7).

## Per-task evidence trail

### Task 1 — D0.3 single-device op_id stability (HIGH, validated)
- **Commit**: `607fca77e` (canary), `33a1a2714` (rationale comment from spec-review).
- **Findings**: [`d0.3-cpy-visibility.md`](d0.3-cpy-visibility.md) +
  [`tests/data/planner-canaries/d0.3.json`](../../../../tests/data/planner-canaries/d0.3.json).
- **Result**: PASS. 5 repeated `graph_reserve` calls on Mistral 7B
  (CPU backend, `ONEAPI_DEVICE_SELECTOR=opencl:cpu`) produced 5 ×
  1031-node sequences, every (op_id, op_type, name) triple
  byte-identical across runs.
- **Design impact**: C2 (`llama.cpp-oib0o`) can proceed with op_id
  keying as specified. The original multi-device CPY-name scope is
  preserved as a `todo_multidevice` evidence key in the canary
  findings — still requires ≥2 SYCL devices safely usable on this
  host (B50 disabled per `feedback_disable_b50.md`) AND E1 to land
  before re-run.

### Task 2 — D0.2 multi-model generalization (HIGH, partially validated)
- **Commit**: `a3e0f29b7`.
- **Findings**: [`d0.2-generalization.md`](d0.2-generalization.md).
- **Result**: PASS across all 5 locally-runnable variants. Mistral 7B
  Q4_0 / Q2_K / Q3_K_M / Q4_K_M / Q8_0 all produce a 13-op set with
  PP == TG; GPT-OSS 20B baseline reproduces with the 17-op MoE-extended
  set. Quantization format does NOT perturb op enumeration (expected
  — dequant is internal to MUL_MAT, not a separate op).
- **Architecture-coverage caveat**: `/Storage/GenAI/models/` has only
  Mistral 7B v0.1 (dense, Llama-style) and GPT-OSS (MoE) families. No
  Llama 3.x, Qwen 2.x, Gemma, or state-space models locally. The
  "single-shape sizing suffices" claim is empirically supported on
  dense + MoE; sliding-window-attention and state-space architectures
  remain a documented blind spot.
- **Design impact**: A3a (`llama.cpp-dyeyy`) sizing direction is
  validated for the architecture surface tested; D16's "PP + TG graph
  union sizing" remains future-proofing. A pre-deployment re-run on a
  SWA + state-space model is still recommended before declaring the
  single-shape claim universally applicable.

### Task 3 — Deprecate-layer-streaming policy audit (MEDIUM, validated)
- **Commits**: `6f7968833` (initial sweep), `209303892` (quality-review
  polish — annotation labels + cross-refs).
- **Files**: `docs/plans/2026-04-22-unified-memory-placement-plan.md`
  (consistency edits only; no behavioral change to the plan).
- **Result**: 7 stale references to `host-pinned compute spillover`,
  `compute buffer spills`, `GGML_SYCL_FORCE_STREAMING`, etc. classified
  and annotated. One sentence in D14 removed as truly stale (it
  contradicted the directive-aligned next paragraph). The
  `§VRAM-insufficient policy` paragraph is preserved verbatim under a
  deprecation banner because rewriting it is beyond a consistency
  sweep — flagged as a follow-up.
- **Design impact**: plan doc is now internally consistent on the
  CPU-dispatch-not-host-pinned-spillover policy.

### Task 4 — E1 RCA minimal-repro investigation (HIGH, partially validated)
- **Commit**: `3cffaae9f` (initial RCA), `833532266` (quality-review
  polish — null-checks, table consistency, file cleanup).
- **Findings**: [`docs/plans/data/e1-rca/findings.md`](../e1-rca/findings.md).
- **Result**: bare async-H2D + `event.wait()` does NOT reproduce m09zb
  in isolation. Three mitigations tested:
  - `ext_oneapi_submit_barrier` (empty waitlist): WORSE — triggers
    GuC kernel-job timeout + GT-reset cascade. Filed as a separate
    Intel bug with `probe-barrier-bug.cpp` + `probe-barrier-deps.cpp`
    as the minimal repro pair.
  - `q.wait_and_throw()`: safe but unnecessary at the bare-repro
    level (events already complete by the time we wait).
  - `ZE_SERIALIZE`: no behavior change at this layer (1 = enqueue
    serialize, 2 = completion serialize).
- **Follow-on**: Task 9 applied the wait_and_throw fix to
  `staging_buffer_pool` (`43cd00782`); did NOT clear m09zb. Task 10
  (commit `33e05fa36`) ran a broader-envelope bisection on top of
  `ggml_backend_sycl_init` and **isolated the trigger to the FIRST
  H2D copy on a freshly-initialized backend stream** — specifically
  `(*stream).memcpy(...).wait()` at `ggml-sycl.cpp:~13076` inside
  `ggml_backend_sycl_buffer_set_tensor`. The wedge is downstream of
  the GLOBAL/STREAM_FENCE sync-mode dispatch and unrelated to the
  staging-pool acquire path that Task 9 patched. Four hypotheses
  ranked (probe-state, DPCT context, BCS queue, pinned chunk
  allocation); two concrete next-step probes outlined.
- **Design impact**: E1 acceptance criteria need to be **refined**
  (see "E1 acceptance refinement" section below). The Task 9 fix
  (`event.wait_and_throw`) is keep-but-not-fixing-m09zb. The actual
  fix targets one of: `ggml_backend_sycl_buffer_set_tensor` H2D path,
  the init-time `ggml_backend_probe_max_alloc_size` probe, or the
  DPCT helper's queue construction.

### Task 5 — Mini-context + FA prototype (HIGH, still blocked)
- **Status**: NOT RUN. Depends on Task 4 / E1 unblocking GPU model
  load.
- **Note**: per the plan doc, Task 5 requires constructing a
  throwaway mini-context with `n_gpu_layers=999`, calling
  `graph_reserve(no_alloc=true)`, and comparing per-backend buffer
  sizes + FA auto-detect decision against a real context. Until E1
  clears, GPU model load wedges before any of this can execute.
- **Re-run criteria**: post-E1, run on Mistral 7B + GPT-OSS 20B (per
  plan §Task 5 acceptance).

### Task 6 — D0.4 re-run post-E1 (HIGH, still blocked)
- **Status**: NOT RUN. The D0.4 canary binary
  (`tests/test-planner-canary-direct-load`) is already built and
  ready; it currently emits INCONCLUSIVE because runtime wedges in
  `ggml_backend_sycl_buffer_set_tensor` at `ggml-sycl.cpp:12685`.
  This is the SAME wedge site Task 10 isolated for E1.
- **Re-run criteria**: post-E1, simply re-invoke
  `ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 120 ./build/bin/test-planner-canary-direct-load`.
  Expect either PASS (byte-exact readback, real `tensor_set_us`) or
  FAIL with a specific mismatch.

### Task 7 — Weight priority order benchmark (MEDIUM, still blocked)
- **Status**: NOT RUN. The benchmark needs to load Mistral 7B at
  `GGML_SYCL_VRAM_BUDGET_PCT=30` (forces spill-to-CPU) and measure
  PP/TG via `llama-bench`. GPU model load wedges on m09zb before any
  benchmark can run.
- **Re-run criteria**: post-E1.

### Task 8 — This summary
- **Status**: this document. Main plan canary-results section
  already updated under Task 2's commit (`a3e0f29b7`) for D0.2 and
  D0.3 row updates (D0.3 row later split into D0.3a multi-device-CPY
  + D0.3b op-id-stability at `16306e2bd` per Task 2 polish).

## E1 acceptance refinement

Original E1 acceptance criterion (`docs/plans/2026-04-22-unified-memory-placement-plan.md` §Track E):
> No mutex held across any `event.wait()`; D0.1 canary loads Mistral
> 7B cleanly in under 60 s; D0.4 canary completes within 60 s with
> byte-identical readback; ninja + ctest green; no new env var; zero
> perf regression on llama-bench Mistral 7B Q4_0 PP512/TG128 (±3%).

Post-Task-4/9/10 evidence supports these refinements:

1. **The "no mutex held across `event.wait()`" sub-criterion is
   insufficient** — Task 9 showed event-level fix to `acquire()` is
   keep-but-not-fixing. The actual trigger is in
   `ggml_backend_sycl_buffer_set_tensor`'s first H2D, which is
   structurally upstream of any staging-pool reuse.
2. **Add a positive criterion**: bare-SYCL `minimal-repro` (Task 4
   artifact) must continue to PASS — it demonstrates the H2D +
   event.wait pattern is sound in isolation; any future change to
   the staging pool that breaks `minimal-repro` reveals a regression.
3. **Add a negative criterion**: `broader-envelope-repro` (Task 10
   artifact) at `STAGE=init-set` must complete in < 30 s. This is
   the smallest envelope that today reproduces m09zb; a fix that
   doesn't clear this stage isn't an E1 fix.
4. **Add a derived criterion**: a `set_tensor` H2D timing trace must
   show < 1 s wait at the first call after backend init, on a fresh
   process. Today that wait is unbounded.
5. **Companion bug split**: the empty-waitlist-`submit_barrier`
   bug discovered in Task 4 is independent of m09zb; do NOT roll its
   fix into E1's scope. File it as a separate Intel bug report (Task
   4 findings already drafted the report). E1's scope stays focused
   on the `set_tensor` H2D wedge.

Recommended E1 acceptance criteria text (proposed for plan-doc edit
in this commit):

> No mutex held across any `event.wait()`; **bare-SYCL minimal-repro
> + broader-envelope-repro both PASS** (the latter at `STAGE=init-set`
> within 30 s); D0.1 canary loads Mistral 7B cleanly in under 60 s;
> D0.4 canary completes within 60 s with byte-identical readback;
> first `ggml_backend_tensor_set` H2D after backend init waits < 1 s
> (measured trace); `ninja -C build` + full ctest green; no new env
> var; zero perf regression on `llama-bench Mistral 7B Q4_0
> PP512/TG128` baseline (±3%).

## Track A unlock decisions (post-validation)

| Track A item | Bead | Pre-validation status | Post-validation status |
|---|---|---|---|
| A3 (split into weight_plan + compute_plan) | (no dedicated bead — tracked under parent epic `llama.cpp-3h5gm`) | OPEN | OPEN — no validation gate; structural change only |
| A3a (mini-context infrastructure) | `llama.cpp-dyeyy` | OPEN, gated on D0.1 | **STILL BLOCKED on E1**. Sizing direction unblocked by D0.2 (Task 2); mini-context mechanics still need D0.1 (currently INCONCLUSIVE). |
| A3b (FA auto-detect in skeleton graph) | open | gated on Task 5 | **STILL BLOCKED on E1**. |
| A4 (arena zone sizing from weight_plan) | open | gated on A3, A3a | structurally OK to start once A3a is unblocked. |
| A7 (weight loader writes direct to arena) | `llama.cpp-wuozk` | OPEN, gated on D0.4 | **STILL BLOCKED on E1**. D0.4 INCONCLUSIVE; will re-run when E1 clears. |
| C2 (populate plan.ops for every graph op) | `llama.cpp-oib0o` | gated on D0.3 | **op-id keying validated** (Task 1). C2 can proceed with op_id keying as specified. Multi-device CPY-name stability remains a future TODO. |
| C3 (remove `GGML_SYCL_FORCE_STREAMING`) | open | not gated on a canary | OK to proceed; consistency sweep done (Task 3). |

## Open follow-ups (post-Task-8)

- **E1**: continue Task 11 (bisection probes — test H1 probe-state,
  H4 pinned-chunk-allocation hypotheses). E1 cannot land until one of
  the four hypotheses is confirmed and a fix shape derived from that
  confirmation.
- **Phase C re-run**: once E1 clears, run Tasks 5, 6, 7 in sequence.
  Each is well-scoped (~1-2 h each) once GPU model load works.
- **Task 2 architecture coverage**: re-run D0.2 generalization on a
  sliding-window-attention model (Gemma 2 / Mistral v0.3) and a
  state-space model (Mamba / RWKV) once GGUFs are present at
  `/Storage/GenAI/models/`. Until then, the "single-shape sizing
  generalizes universally" claim has a documented architecture-coverage
  caveat.
- **Task 3 follow-up**: rewrite the §VRAM-insufficient policy
  paragraph to describe the new CPU-dispatch behavior instead of
  preserving the deprecated host-pinned-spillover framing under a
  banner.
- **Task 4 companion bug**: file the `ext_oneapi_submit_barrier`
  empty-waitlist GT-reset bug with Intel using
  `probe-barrier-bug.cpp` + `probe-barrier-deps.cpp` + `findings.md`
  excerpts.

## Status snapshot

```
Phase A (Tasks 1, 2, 3):   3/3 complete + reviewed
Phase B (Task 4):          complete; root cause narrowed by Task 10 (33e05fa36)
Phase C (Tasks 5, 6, 7):   0/3 — blocked on E1
Phase D (Task 8):          this document
E1 (Tasks 9, 10, 11):      Task 9 applied, did NOT clear; Task 10 isolated trigger; Task 11 in progress
```

8 uncertainty items: 3 fully validated, 2 partially validated, 3
still blocked on E1, 0 refuted.

The plan is no longer "plausible design backed by D0.2 PASS" — it is
"design backed by D0.2 + D0.3 PASS, with E1's actual trigger isolated,
remaining items time-deferred behind E1's resolution." Track A's
sizing direction (A3a) and op-id-keying claim (C2) are concretely
unblocked on the design front; their mechanics remain gated on E1.
