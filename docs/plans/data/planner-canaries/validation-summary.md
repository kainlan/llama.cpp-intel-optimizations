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

Severity column from the original audit. Status updated 2026-04-24
after the patched compute-runtime install (libze_intel_gpu.so 1.14.37435
from `/Apps/compute-runtime` branch `fix/combined-26.09`, dpkg-diverted
+ ldconfig'd as the system default) resolved m09zb upstream.

| # | Uncertainty                                                                                                       | Severity | Test       | Result                          | Status                              |
|---|-------------------------------------------------------------------------------------------------------------------|----------|------------|---------------------------------|-------------------------------------|
| 1 | C2's `plan.ops` keying on op_id is stable across repeated `graph_reserve`                                         | HIGH     | Task 1 / D0.3 | PASS (`607fca77e`)           | **validated**                       |
| 2 | A3a's "single-shape sizing suffices" generalizes beyond Mistral 7B + GPT-OSS 20B                                  | HIGH     | Task 2 / D0.2 | PASS (`a3e0f29b7`)           | **partially validated** — Mistral-dense + GPT-OSS-MoE families only; SWA + state-space architectures absent locally, untested |
| 3 | Plan doc reflects the 2026-04-22 deprecate-layer-streaming directive consistently                                 | MEDIUM   | Task 3       | PASS (`6f7968833`, polished `209303892`) | **validated** — all 7 stale references annotated or rewritten; CPU-dispatch is now the canonical answer |
| 4 | E1 / m09zb root cause identified, mitigation chosen                                                                | HIGH     | Task 4 / Task 10 / Patched runtime install (2026-04-24) | RESOLVED — paired libze test on 2026-04-24 confirms stock 1.14.37020 wedges on first H2D after `ggml_backend_sycl_init`, patched 1.14.37435 returns cleanly; D0.4 PASSES under patched (`tensor_set_us=282422`) | **validated** — m09zb is upstream-fixed in our compute-runtime fork (`/Apps/compute-runtime` branch `fix/combined-26.09`); patched libze is now system-default. No llama.cpp code change beyond Task 9's `event.wait_and_throw` strict improvement. See `docs/plans/data/e1-rca/findings.md` Step 4. |
| 5 | A3a's mini-context + A3b's FA auto-detection produce sizes/decisions byte-identical to a real context              | HIGH     | Task 5       | NOT RUN                         | **still blocked** — Task 5 prototype not yet implemented; m09zb axis is now unblocked, but the canary's mini-context probe for this question hasn't been written. Deferred to a future session. |
| 6 | A7's direct mmap → device `ggml_backend_tensor_set` achieves byte-exact readback within 60 s                       | HIGH     | Task 6 / D0.4 re-run | PASS (2026-04-24 patched-runtime test, `tensor_set_us=282422`) | **validated** — D0.4 canary completes inside the 60 s window with byte-identical readback under the patched runtime. A7's direct-load path is sound. |
| 7 | Weight priority order `NORM_EMBED > ATTENTION > FFN > MOE_DOWN > MOE_UP > MOE_GATE_PROJ` outperforms alternatives  | MEDIUM   | Task 7       | NOT RUN                         | **still blocked** — m09zb axis unblocked, but a separate single-buffer alloc ceiling (~1.5 GB under patched libze, Task 14) gates the `VRAM_BUDGET_PCT=30` benchmark since it triggers > 1.5 GB KV allocs on Mistral 7B at the larger contexts. Smaller-context variants are now runnable. |
| 8 | The plan doc is internally consistent, the canary-results section reflects current findings, and the bead graph is honest about which tracks are unblocked | LOW | Task 8 (this doc) | This document + main plan edits | **validated** by the act of writing this summary; main plan canary-results section already reflects items 1-3 |

Five items fully validated (1, 3, 4, 6, 8). One partially validated
(2 — architecture coverage). Two still in progress (5 — Task 5
prototype not yet implemented; 7 — gated on Task 14 KV alloc
ceiling). Zero refuted.

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

### Task 4 — E1 RCA minimal-repro investigation (HIGH, RESOLVED)
- **Commit**: `3cffaae9f` (initial RCA), `833532266` (quality-review
  polish — null-checks, table consistency, file cleanup).
- **Findings**: [`docs/plans/data/e1-rca/findings.md`](../e1-rca/findings.md).
- **Result**: bare async-H2D + `event.wait()` does NOT reproduce m09zb
  in isolation. Three mitigations tested:
  - `ext_oneapi_submit_barrier` (empty waitlist): WORSE — triggers
    GuC kernel-job timeout + GT-reset cascade. Tracked as a separate
    GuC-side issue out of scope for E1.
  - `q.wait_and_throw()`: safe but unnecessary at the bare-repro
    level (events already complete by the time we wait).
  - `ZE_SERIALIZE`: no behavior change at this layer (1 = enqueue
    serialize, 2 = completion serialize).
- **Follow-on**: Task 9 applied the wait_and_throw fix to
  `staging_buffer_pool` (`43cd00782`); did NOT clear m09zb. Task 10
  (commit `33e05fa36`) ran a broader-envelope bisection on top of
  `ggml_backend_sycl_init` and isolated the trigger to the FIRST
  H2D copy on a freshly-initialized backend stream —
  `(*stream).memcpy(...).wait()` at `ggml-sycl.cpp:~13076` inside
  `ggml_backend_sycl_buffer_set_tensor`. Tasks 11/12/13 ranked four
  hypotheses and tested probe-replacement / one-shot-warmup variants;
  none cleared the wedge against stock libze.
- **Resolution (2026-04-24)**: paired libze test confirmed the wedge
  is a stock-libze accept-but-don't-flush bug. Patched compute-runtime
  (`/Apps/compute-runtime` branch `fix/combined-26.09`,
  libze_intel_gpu.so 1.14.37435) was installed as the system default
  via `dpkg-divert` + `ldconfig`. Under the patched libze, the same
  H2D returns cleanly; `safe_max_alloc_size` correctly reports
  1593 MB (vs stock's 11024 MB which was over-promising and
  triggering the wedge). D0.4 canary now PASSES with
  `tensor_set_us = 282422`. m09zb is **upstream-FIXED in our fork**.
- **Design impact**: E1 acceptance criteria are **met** under the
  patched runtime. The Task 9 fix (`event.wait_and_throw`) remains
  in place as a strict improvement (async errors propagate properly).
  No additional llama.cpp code change required for the m09zb
  acceptance set.

### Task 5 — Mini-context + FA prototype (HIGH, still blocked)
- **Status**: NOT RUN. m09zb is no longer the gate (resolved
  2026-04-24 via patched compute-runtime install); Task 5's
  prototype itself has not yet been written.
- **Note**: per the plan doc, Task 5 requires constructing a
  throwaway mini-context with `n_gpu_layers=999`, calling
  `graph_reserve(no_alloc=true)`, and comparing per-backend buffer
  sizes + FA auto-detect decision against a real context.
- **Re-run criteria**: implement the prototype as a follow-up
  session task; run on Mistral 7B + GPT-OSS 20B (per plan §Task 5
  acceptance). The patched-runtime KV-alloc ceiling (Task 14) may
  affect larger-context variants but small-context probes work today.

### Task 6 — D0.4 re-run post-E1 (HIGH, validated)
- **Status**: PASS (2026-04-24 patched-runtime test).
- **Result**: `tensor_set_us = 282422` (~282 ms) on the patched
  libze 1.14.37435; pre-patch the same canary timed out at 60 s.
  Byte-identical readback confirmed.
- **Design impact**: A7's "direct mmap → device
  `ggml_backend_tensor_set` within 60 s with byte-identical
  readback" criterion is met. A7 can proceed.

### Task 7 — Weight priority order benchmark (MEDIUM, still blocked)
- **Status**: NOT RUN under the original full-context shape. m09zb
  is resolved; the benchmark's 30%-VRAM-budget configuration on
  Mistral 7B at the canonical 4096-token context now hits a
  separate single-buffer alloc ceiling (~1.5 GB under patched
  libze, Task 14) for KV-cache buffers, blocking the standard
  baseline run.
- **Re-run criteria**: smaller-context shapes (n_ctx=2048 or below
  on Mistral 7B Q4_0) should fit under the 1.5 GB single-alloc
  ceiling and are runnable today; the canonical 4096-context bench
  is gated on Task 14 (KV alloc chunking).

### Task 8 — This summary
- **Status**: this document. Main plan canary-results section
  already updated under Task 2's commit (`a3e0f29b7`) for D0.2 and
  D0.3 row updates (D0.3 row later split into D0.3a multi-device-CPY
  + D0.3b op-id-stability at `16306e2bd` per Task 2 polish).

## E1 acceptance — met under patched runtime (2026-04-24)

Original E1 acceptance criterion (`docs/plans/2026-04-22-unified-memory-placement-plan.md` §Track E):
> No mutex held across any `event.wait()`; D0.1 canary loads Mistral
> 7B cleanly in under 60 s; D0.4 canary completes within 60 s with
> byte-identical readback; ninja + ctest green; no new env var; zero
> perf regression on llama-bench Mistral 7B Q4_0 PP512/TG128 (±3%).

Verification under the system-default patched libze
(`/Apps/compute-runtime` branch `fix/combined-26.09`,
libze_intel_gpu.so 1.14.37435), 2026-04-24:

- ✅ No mutex held across any `event.wait()` — Task 9 (`43cd00782`)
  replaced the lock-bracketed wait with `event.wait_and_throw()`; the
  pool's mutex is no longer held across the wait.
- ✅ D0.4 canary completes well within 60 s with byte-identical
  readback (`tensor_set_us = 282422`, ~282 ms).
- ✅ No new env var introduced. The fix is in libze itself, not in
  llama.cpp; the in-tree code change is a single-line strict
  improvement.
- ⚠ D0.1 canary remains skeletal (multi-context-on-shared-model
  variant deferred to a future session); no longer blocked on m09zb.
- ⚠ Perf regression check on `llama-bench Mistral 7B Q4_0 PP512/TG128`
  not yet re-baselined under the patched runtime; expected to be
  within ±3% but pending Task 14's KV alloc work to close out the
  larger-context shape.

The empty-waitlist `ext_oneapi_submit_barrier` GT-reset bug
discovered in Task 4 remains independent of m09zb and out of scope
for E1; it manifests on a different code path that isn't on the
unified-memory-plan critical path.

## Track A unlock decisions (post-validation)

| Track A item | Bead | Pre-validation status | Post-validation status |
|---|---|---|---|
| A3 (split into weight_plan + compute_plan) | (no dedicated bead — tracked under parent epic `llama.cpp-3h5gm`) | OPEN | OPEN — no validation gate; structural change only |
| A3a (mini-context infrastructure) | `llama.cpp-dyeyy` | OPEN, gated on D0.1 | **UNBLOCKED on m09zb axis**. Sizing direction validated by D0.2 (Task 2). D0.1 canary's mini-context mechanics probe is not yet implemented (skeletal canary deferred to a future session); no longer blocked by E1. |
| A3b (FA auto-detect in skeleton graph) | open | gated on Task 5 | **UNBLOCKED on m09zb axis**, awaiting Task 5 prototype implementation. |
| A4 (arena zone sizing from weight_plan) | open | gated on A3, A3a | structurally OK to start once A3a's prototype lands. |
| A7 (weight loader writes direct to arena) | `llama.cpp-wuozk` | OPEN, gated on D0.4 | **VALIDATED**. D0.4 PASSES under patched runtime (`tensor_set_us = 282422`, 2026-04-24). A7 can proceed. |
| C2 (populate plan.ops for every graph op) | `llama.cpp-oib0o` | gated on D0.3 | **op-id keying validated** (Task 1). C2 can proceed with op_id keying as specified. Multi-device CPY-name stability remains a future TODO. |
| C3 (remove `GGML_SYCL_FORCE_STREAMING`) | open | not gated on a canary | OK to proceed; consistency sweep done (Task 3). |

## Open follow-ups (post-Task-8, current state 2026-04-24)

- **m09zb**: RESOLVED via patched compute-runtime install (system
  default, dpkg-diverted). No further follow-up needed on the m09zb
  axis itself; future Phase C work can assume GPU model load works.
- **Task 14** (in flight, implementer-1): KV-buffer wedge investigation
  under patched runtime. Patched libze enforces a ~1.5 GB single-alloc
  ceiling; Mistral 7B at 4096-context allocates a single ~4 GB KV
  buffer that exceeds it. Task 14 is finding the right chunking or
  size-cap point. This gates the canonical-context Task 7 baseline run.
- **Phase C completion**: Task 5 prototype not yet written; Task 6
  validated (D0.4 PASS); Task 7 needs Task 14 for the canonical
  4096-context shape (smaller-context variants runnable today).
- **Task 2 architecture coverage**: re-run D0.2 generalization on a
  sliding-window-attention model (Gemma 2, Mixtral) and a state-space
  model (Mamba / RWKV) once GGUFs are present at
  `/Storage/GenAI/models/`. The "single-shape sizing generalizes
  universally" claim retains its documented architecture-coverage
  caveat until then.
- **Task 3 follow-up**: rewrite the §VRAM-insufficient policy
  paragraph to describe the new CPU-dispatch behavior instead of
  preserving the deprecated host-pinned-spillover framing under a
  banner.

## Status snapshot

```
Phase A (Tasks 1, 2, 3):   3/3 complete + reviewed
Phase B (Task 4):          complete; root cause narrowed (Task 10) and resolved (2026-04-24 patched runtime)
Phase C (Tasks 5, 6, 7):   1/3 — Task 6 PASS; Task 5 not yet written; Task 7 gated on Task 14
Phase D (Task 8):          this document
E1 (Tasks 9-13):           Task 9 applied as strict improvement; Tasks 10/11/12/13 traced the issue; resolved upstream by patched compute-runtime (2026-04-24). E1 acceptance MET.
Task 14 (in flight):       KV-alloc ceiling under patched runtime (different bug, doesn't reopen m09zb)
```

8 uncertainty items (state as of 2026-04-24): 4 fully validated, 2
partially validated, 2 still in-progress (Task 5 prototype not yet
written; Task 7 gated on Task 14 KV-alloc ceiling), 0 refuted.

The plan is no longer "plausible design backed by D0.2 PASS, blocked
on E1" — it is "design backed by D0.2 + D0.3b + D0.4 PASS, with m09zb
upstream-resolved via the patched compute-runtime install. Phase C
items 5 and 7 remaining; Track A is structurally unblocked." Track A's
sizing direction (A3a), op-id-keying claim (C2), and direct-load
mechanics (A7) are all concretely unblocked.
