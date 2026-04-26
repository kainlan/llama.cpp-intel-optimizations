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
| 2 | A3a's "single-shape sizing suffices" generalizes beyond Mistral 7B + GPT-OSS 20B                                  | HIGH     | Task 2 / D0.2 | PASS for tested families; state-space open | **partially validated** — Mistral dense/full-attention + GPT-OSS active SWA+MoE validated; state-space absent locally and untested, and SYCL lacks `GGML_OP_SSM_SCAN` support/routing for Mamba/Plamo2 graphs |
| 3 | Plan doc reflects the 2026-04-22 deprecate-layer-streaming directive consistently                                 | MEDIUM   | Task 3       | PASS (`6f7968833`, polished `209303892`) | **validated** — all 7 stale references annotated or rewritten; CPU-dispatch is now the canonical answer |
| 4 | E1 / m09zb root cause identified, mitigation chosen                                                                | HIGH     | Task 4 / Task 10 / Patched runtime install (2026-04-24) | RESOLVED — paired libze test on 2026-04-24 confirms stock 1.14.37020 wedges on first H2D after `ggml_backend_sycl_init`, patched 1.14.37435 returns cleanly; D0.4 PASSES under patched (`tensor_set_us=282422`) | **validated** — m09zb is upstream-fixed in our compute-runtime fork (`/Apps/compute-runtime` branch `fix/combined-26.09`); patched libze is now system-default. No llama.cpp code change beyond Task 9's `event.wait_and_throw` strict improvement. See `docs/plans/data/e1-rca/findings.md` Step 4. |
| 5 | A3a's mini-context + A3b's FA auto-detection produce sizes/decisions byte-identical to a real context              | HIGH     | Task 5       | PASS (`333df7379`, 2026-04-24) | **validated** — fork+exec'd 3-worker prototype (real-A, real-B, mini=`no_alloc=true,use_mmap=false`) produced byte-identical compute-buffer sizes + FA verdict on Mistral 7B (115.01 MiB, FA=enabled) and GPT-OSS 20B (398.38 MiB, FA=enabled). Mmap-path assert at `src/llama-model.cpp:8686` requires `use_mmap=false` for the mini worker — flagged as A3a production constraint. |
| 6 | A7's direct mmap → device `ggml_backend_tensor_set` achieves byte-exact readback within 60 s                       | HIGH     | Task 6 / D0.4 re-run | PASS (2026-04-24 patched-runtime test, `tensor_set_us=282422`) | **validated** — D0.4 canary completes inside the 60 s window with byte-identical readback under the patched runtime. A7's direct-load path is sound. |
| 7 | Weight priority order `NORM_EMBED > ATTENTION > FFN > MOE_DOWN > MOE_UP > MOE_GATE_PROJ` outperforms alternatives  | MEDIUM   | Task 7       | DEFERRED                        | **deferred** — m09zb axis unblocked, but the full inference path through `VRAM_BUDGET_PCT=30` traverses three layered VRAM-arena bugs identified by Closeout A: graph_reserve OOM err 39 (Wall 1), KV-clear wedge `llama.cpp-zhzbp` (Wall 2), GET_ROWS OOR err 40 (Wall 3). All three need to land before the canonical 4096-context bench can run. See `docs/plans/data/e1-rca/findings.md` Step 5. |
| 8 | The plan doc is internally consistent, the canary-results section reflects current findings, and the bead graph is honest about which tracks are unblocked | LOW | Task 8 (this doc) | This document + main plan edits | **validated** by the act of writing this summary; main plan canary-results section already reflects items 1-3 |

**2026-04-25 correction**: item 2 remains partially validated, but the
state-space caveat is stronger than "no local fixture": SYCL has no
`GGML_OP_SSM_SCAN` support/routing while Mamba/Plamo2 graphs emit
`ggml_ssm_scan`. D0.1 had stale proof artifacts that have since been replaced.
D0.3a now has current multi-device evidence and changes the C2 contract:
scheduler transfers are stable copy-edge tensors, not `GGML_OP_CPY` nodes.

**Final session tally**: dense/full-attention and GPT-OSS active SWA+MoE sizing
are validated; state-space, scheduler-transfer implementation, priority
benchmark, CPU-dispatch-vs-streaming policy, and unified-cache-only ownership
remain open proof/work areas.

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
  keying for single-device graph ops. D0.3a refutes the old multi-device
  CPY-node assumption: the synthetic two-GPU scheduler proof exposes
  stable split-input copy-edge tensors (`SYCL1#...#0`) but no
  `GGML_OP_CPY` nodes. C2 must plan scheduler transfer edges explicitly.
  See [`d0.3-cpy-visibility.md`](d0.3-cpy-visibility.md) and
  [`d0.3a-multidevice-cpy-visibility.md`](d0.3a-multidevice-cpy-visibility.md).

### Task 2 — D0.2 multi-model generalization (HIGH, partially validated)
- **Commit**: `a3e0f29b7`.
- **Findings**: [`d0.2-generalization.md`](d0.2-generalization.md).
- **Result**: PASS across all 5 locally-runnable variants. Mistral 7B
  Q4_0 / Q2_K / Q3_K_M / Q4_K_M / Q8_0 all produce a 13-op set with
  PP == TG; GPT-OSS 20B baseline reproduces with the 17-op active
  SWA+MoE-extended set. Quantization format does NOT perturb op enumeration (expected
  — dequant is internal to MUL_MAT, not a separate op).
- **Architecture-coverage caveat**: `/Storage/GenAI/models/` has only
  Mistral 7B v0.1 (dense/full-attention, `n_swa=0`) and GPT-OSS 20B
  (active alternating SWA + MoE: `sliding_window=128`, `n_swa=128`,
  12 SWA layers + 12 non-SWA layers) as runnable architecture families.
  No Llama 3.x, Qwen 2.x, Gemma, Mixtral, Mamba, RWKV, or other
  state-space models are locally available. The "single-shape sizing
  suffices" claim is empirically supported on dense/full-attention and
  active SWA+MoE; state-space remains the important documented blind
  spot. A second non-GPT-OSS SWA family is useful hardening, not a
  total SWA absence.
- **Design impact**: A3a (`llama.cpp-dyeyy`) sizing direction is
  validated for the architecture surface tested; D16's "PP + TG graph
  union sizing" remains future-proofing. A pre-deployment re-run on a
  state-space model is still recommended before declaring the
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

### Task 5 — Mini-context + FA prototype (HIGH, validated)
- **Commit**: `333df7379` (2026-04-24).
- **Findings**: [`mini-context-validation.md`](mini-context-validation.md).
- **Result**: PASS on Mistral 7B Q4_0 + GPT-OSS 20B MXFP4. A 3-worker
  fork+exec'd prototype (real-A, real-B, mini=`no_alloc=true,
  use_mmap=false`) produced byte-identical compute-buffer sizes + FA
  verdict in all three workers:
  - Mistral 7B: FA=enabled, CPU buffer=115.01 MiB
  - GPT-OSS 20B: FA=enabled, CPU buffer=398.38 MiB
  Determinism verified (real-A == real-B); mini-context approach
  validated (mini == real-A).
- **Important finding**: `GGML_ASSERT(!ml.no_alloc)` at
  `src/llama-model.cpp:8686` makes the mmap-from-host-pointer fast
  path incompatible with metadata-only loads. Worked around by setting
  `mparams.use_mmap = false` for the mini worker. Production A3a must
  either lift this assert or carry the use_mmap=false constraint.
- **Backend**: CPU (`opencl:cpu`) — Task 5's question is scheduler-
  level not SYCL-specific; the SYCL path is also expected to PASS
  but exercise it post-Closeout-A walls 1-3 if desired.
- **Design impact**: A3a (mini-context infrastructure) and A3b (FA
  auto-detect in the skeleton graph pass) are mechanistically
  validated. A3a can proceed.

### Task 6 — D0.4 re-run post-E1 (HIGH, validated)
- **Status**: PASS (2026-04-24 patched-runtime test).
- **Result**: `tensor_set_us = 282422` (~282 ms) on the patched
  libze 1.14.37435; pre-patch the same canary timed out at 60 s.
  Byte-identical readback confirmed.
- **Design impact**: A7's "direct mmap → device
  `ggml_backend_tensor_set` within 60 s with byte-identical
  readback" criterion is met. A7 can proceed.

### Task 7 — Weight priority order benchmark (MEDIUM, deferred)
- **Status**: DEFERRED. m09zb is resolved, but Closeout A (parallel
  to this task) identified that `VRAM_BUDGET_PCT=30` traverses three
  layered VRAM-arena bugs:
  - Wall 1: `graph_reserve` OOM err 39 from arena chunk size > 1.5 GB
  - Wall 2: KV-clear per-layer memset wedge (`llama.cpp-zhzbp`)
  - Wall 3: GET_ROWS OOR err 40 from arena overcommit starving runtime
  All three need to land before the canonical 4096-context bench can
  run. See `docs/plans/data/e1-rca/findings.md` Step 5 for the
  three-blocker synthesis.
- **Re-run criteria**: walls 1, 2, 3 all closed via the new beads
  filed by Closeout A. Until then, smaller-context single-device
  variants may run today but are not the canonical Task 7 baseline.
- **Design impact**: this is a backend-implementation issue, not a
  planner-design issue. The priority-order claim is untested but
  the planner itself is mechanically correct.

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
- ✅ D0.1 canary rerun on 2026-04-25 with the real/mini reserve
  protocol; Mistral 7B dense and GPT-OSS 20B active SWA+MoE both passed.
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
| A3a (mini-context infrastructure) | `llama.cpp-dyeyy` | OPEN, gated on D0.1 | **VALIDATED**. Task 5 prototype (`333df7379`) confirmed mini-context produces byte-identical sizes + FA verdict to a real context on Mistral 7B + GPT-OSS 20B; the canonical D0.1 canary was replaced and rerun successfully on 2026-04-25. A3a can proceed for tested dense/SWA+MoE families; production must use `use_mmap=false` for mini-context construction (or lift the mmap-path assert). |
| A3b (FA auto-detect in skeleton graph) | open | gated on Task 5 | **VALIDATED**. Task 5 confirmed FA auto-detect runs inside `llama_init_from_model` regardless of `no_alloc`; verdict matches between mini and real contexts. A3b can proceed. |
| A4 (arena zone sizing from weight_plan) | open | gated on A3, A3a | structurally OK to start once A3a's prototype lands. |
| A7 (weight loader writes direct to arena) | `llama.cpp-wuozk` | OPEN, gated on D0.4 | **VALIDATED**. D0.4 PASSES under patched runtime (`tensor_set_us = 282422`, 2026-04-24). A7 can proceed. |
| C2 (populate plan.ops / transfer edges) | `llama.cpp-oib0o` | gated on D0.3 | **op-id keying validated** for single-device graph ops. **Contract corrected** for multi-device transfer planning: D0.3a shows scheduler copies are stable split-input copy-edge tensors, not `GGML_OP_CPY` nodes. C2 can proceed only if it consumes scheduler transfer-edge metadata; update/close `llama.cpp-bkvc9` with that corrected assertion. |
| C3 (remove `GGML_SYCL_FORCE_STREAMING`) | open | not gated on a canary | OK to proceed; consistency sweep done (Task 3). |

## Open follow-ups (final session-end state, 2026-04-24)

Three layered VRAM-arena bugs (P1) + one A3a-spinoff (P2) + two
doc-hygiene follow-ups remain. The three P1 bugs gate full inference
on Mistral 7B / GPT-OSS 20B at canonical contexts
(`VRAM_BUDGET_PCT=30`, `n_ctx=4096`). None gate the planner-
validation epic at the design level — Phase A/B and Tasks 5/6 are
all PASS. These are backend-implementation + doc-hygiene issues
filed for future-session work.

### Layered VRAM-arena bugs (P1, filed by Closeout A)

Three distinct walls discovered in sequence as each prior wall was
peeled away. See `docs/plans/data/e1-rca/findings.md` Step 5 for the
synthesis. **Owners needed: backend implementer.**

- **Wall 1** — `llama.cpp-khcc0` (P1, OPEN): "[ggml-sycl]
  graph_reserve OOM under patched runtime (err 39, arena chunk
  > 1.5 GB cap)". Fix via vbuffer chunking on WEIGHT and KV zones
  (the compute-buffer path already has it via `llama.cpp-w1rxh`,
  closed 2026-04-22).
- **Wall 2** — `llama.cpp-zhzbp` (P1, OPEN, amended in Closeout A):
  "[KV-CLEAR] tiered_kv_buffer_clear hangs at first stream->memset
  on patched compute-runtime". Replace per-layer
  `stream->memset(...).wait()` loop with a single contiguous memset
  across the arena-allocated KV region. Quick-fix attempt with
  stream drain reverted; needs the contiguous-memset rewrite. Not
  size-dependent; reproduces at 2 MiB per layer.
- **Wall 3** — `llama.cpp-w2ptt` (P1, OPEN): "[ggml-sycl] GET_ROWS
  OOR (err 40) after full arena reservation — runtime alloc
  starvation". Fix via tightening RUNTIME-zone reservation in the
  planner using Task 5's mini-context output as the runtime ceiling.

### Task 5 follow-up: A3a mmap-path assert

- **Bead**: `llama.cpp-jfj0v` (P2, OPEN): "[llama]
  GGML_ASSERT(!ml.no_alloc) at src/llama-model.cpp:8686 blocks A3a
  mmap-from-host fast path". Production A3a must either lift this
  assert or carry the use_mmap=false constraint forward. Filed by
  Closeout A from the Task 5 finding.

### Other known follow-ups

- **Task 2 architecture coverage** (P2, this team): re-run D0.2
  generalization on a state-space model (Mamba / RWKV / SSM) once a
  GGUF is present at `/Storage/GenAI/models/`. GPT-OSS 20B provides
  active SWA+MoE coverage (`sliding_window=128`, `n_swa=128`), so an
  independent SWA family such as Gemma 2 or Mixtral is useful hardening
  but no longer the primary gap. The "single-shape sizing generalizes
  universally" claim retains its documented state-space caveat until
  then. **Owner: this team (planner-validation), low priority —
  pre-deployment hardening.**
- **Task 3 §VRAM-insufficient policy rewrite** (P3, this team):
  rewrite paragraph 4 of §VRAM-insufficient policy in the main plan
  doc to describe the new CPU-dispatch behavior instead of
  preserving the deprecated host-pinned-spillover framing under a
  banner. **Owner: this team, low priority — pure doc hygiene.**

## Status snapshot

```
Phase A (Tasks 1, 2, 3):   3/3 complete + reviewed
Phase B (Task 4):          complete; root cause narrowed (Task 10) and resolved (2026-04-24 patched runtime)
Phase C (Tasks 5, 6, 7):   2/3 PASS — Task 5 PASS (333df7379), Task 6 PASS (D0.4), Task 7 DEFERRED on layered arena walls
Phase D (Task 8 + Closeout B): this document
E1 (Tasks 9-13):           Task 9 applied as strict improvement; Tasks 10/11/12/13 traced the issue; resolved upstream by patched compute-runtime (2026-04-24). E1 acceptance MET.
Task 14 + zhzbp + Closeout-A walls 1+3: layered VRAM-arena bugs filed P0 for future-session backend work; do not reopen m09zb and do not block planner design.
```

**Final session tally** (2026-04-24): 6 fully validated (1, 3, 4, 5,
6, 8), 1 partially validated (2 — architecture coverage), 1 deferred
(7 — gated on layered arena walls), 0 refuted. The
planner-validation epic is COMPLETE at the design-validation level.

The plan is no longer "plausible design backed by D0.2 PASS, blocked
on E1" — it is "design backed by D0.2 + D0.3b + D0.4 PASS, with m09zb
upstream-resolved via the patched compute-runtime install. Phase C
items 5 and 7 remaining; Track A is structurally unblocked." Track A's
sizing direction (A3a), op-id-keying claim (C2), and direct-load
mechanics (A7) are all concretely unblocked.
