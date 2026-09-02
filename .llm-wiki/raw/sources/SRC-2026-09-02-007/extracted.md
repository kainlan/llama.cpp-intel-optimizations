# One layout per weight, and routes that only advertise kernels that exist

- **Slug:** one-layout-honest-routes
- **Title:** One layout per weight, and routes that only advertise kernels that exist
- **Epic ticket:** NEW
- **Priority:** 1
- **Description:** ## Goal

Every SYCL-backend weight is materialized exactly once, in the layout optimal for the
processor that executes it (ruling 5). Every consumer of that weight — PP grouped
dispatch and TG small-batch dispatch, gate/up and down projections — reads the same
materialized layout; nothing shifts layout at dispatch time and nothing keeps a second
copy in a second layout "just in case". `supports_op` and the MoE route selectors
advertise only `(type, layout)` pairs a compiled kernel actually covers, so an
unadvertised combination refuses cleanly (CPU fallback or an honest error) instead of
silently computing wrong numbers. This epic also carries the deferred Q1/NVFP4 device-decode
program (whose whole point is closing an advertisement gap for two new types, not adding a
second layout) and the reconciliation work between what the planner *advertises* and what
the unified cache actually *materialized* — the seam that produced this cycle's worst
correctness bugs (mn70/zoly/avhx).

## Why now

`test-backend-ops` currently reports 234 uncovered-type `MUL_MAT_ID` failures that are not
"missing coverage" in the harmless sense. Re-derived against HEAD `fed0b58e2` (the review's
correction): `ggml_backend_sycl_device_supports_op` (`ggml-sycl.cpp:102023`, NOT the
`[PERSISTENT-TG]` logging lambda at `:94060-94067` an earlier pass mis-cited) does not claim
`MUL_MAT_ID` unconditionally — its `MUL_MAT_ID`/`ADD_ID` branch (`:102120-102142`) already
type-gates via `ggml_sycl_mul_mat_type_supported(indexed_a_type)` with an explicit
Q1_0/NVFP4 carve-out, and commit `186348705` (2026-08-26) already closed the gap for BF16
and b10630's q2_0/tq2_0 through that same allowlist. The gap that remains is narrower and
structural, not a missing check: `ggml_sycl_mul_mat_type_supported` (`:101973-102000`) is
the **regular MUL_MAT** allowlist, shared verbatim by the MoE/indexed admission gate, so
F32/F16/IQ1_S..IQ4_XS return `true` there (they are real MUL_MAT types) even though no
MMID-specific executor covers them — the MMVQ capability tables (`moe-mmvq-tables.hpp`)
correctly default `false` for those same types. The admission gate is honest about MUL_MAT
coverage and dishonest about MUL_MAT_ID coverage, because the two share one allowlist. The
fix is a second, MMID-specific coverage list consulted only by the indexed-op branch — not
an added `src0->type` check, which already exists and is exactly the code doing the
over-admitting.
Separately, at least three independent producer/consumer sites have been found trusting an
*advertised* layout that disagrees with what storage actually holds (mn70's consumer guard,
avhx's producer `ggml_sycl_select_moe_expert_cache_layout`, 34g5's `allow_aos_fallback`
waiver in mmvq.cpp) — the same bug class recurring because there is still no single
enforced invariant that a route's `requested_layout` and a tensor's `actual_layout` must
agree before dispatch. Fixing the class, not the third instance of it, is why this is an
epic rather than three point fixes.

## Current state (file:line refs)

- `ggml_backend_sycl_device_supports_op` (`ggml/src/ggml-sycl/ggml-sycl.cpp:102023`) gates
  `MUL_MAT_ID`/`ADD_ID` (`:102120-102142`) through `ggml_sycl_mul_mat_type_supported`
  (`:101973-102000`) — the SAME allowlist used for the regular `MUL_MAT` path (`:102136`,
  `:102312`). That allowlist correctly admits F32/F16/IQ1_S..IQ4_XS as valid MUL_MAT types,
  which is why they also pass the MoE/indexed gate even though no MMID executor covers them
  — this is a shared-allowlist gap, not an unconditional `true`. BF16 and b10630's
  q2_0/tq2_0 are already refused (absent from the allowlist; closed by `186348705`,
  2026-08-26).
- `moe_mmvq_capability_supports_layout` / `moe_mmvq_batched_dispatch_supports_type`
  (`ggml/src/ggml-sycl/moe-mmvq-tables.hpp:37-132`) default `false` for F32/F16/BF16/IQ* —
  correctly, per the header's own warning (`:23-28`) that a false-positive capability means
  wrong numbers, not a missing table entry.
- Commit `186348705` (llama.cpp-phbr, 2026-08-26) partially closed the gate for BF16 and the
  new b10630 q2_0/tq2_0 types via `ggml_sycl_mul_mat_type_supported()`
  (`ggml-sycl.cpp:102120-102139`) — F32/F16/IQ2_XXS/other IQ* are still unclosed.
- `ggml_sycl_select_moe_expert_cache_layout` (`ggml-sycl.cpp:26492-26520`, moved from the
  ticket's stale `25437-25460`) still returns SOA for MXFP4/Q6_K gate/up under XMX_TILED with
  `n_tokens>1` from an `ggml_sycl_adjust_layout_for_tensor` call whose result can disagree
  with what the cache actually materialized (observed AOS in the same run) — this is the real
  producer avhx names, corrected from the ticket's original (wrong) citation.
  `allow_aos_fallback` in `mmvq.cpp:20742-20751` is a third, distinct instance: it detects the
  route/storage disagreement and *waives* the refusal without performing any reconciling
  staging.
- `llama.cpp-71hx`'s per-expert staging fallback silent-garbage hazard is fixed:
  `c7b805740` (perf-recovery track C, 2026-08-22) deleted the dequant path for the batched PP
  executor and the fail-closed contract is now in the code with explicit `llama.cpp-71hx`
  citations (`ggml-sycl.cpp:67586-67588`, `:74243`).
- The Q1/NVFP4 device-decode program (`llama.cpp-omp4` umbrella) is real, has extensive test
  coverage already written (`tests/test-sycl-moe-resolved-batch.cpp`,
  `test-sycl-moe-handle-resolution.cpp`, `test-sycl-supports-op-indexed-moe-source.py`), and
  is deferred-not-cancelled per the 2026-08-14 owner ruling — its capability gate stays
  fail-closed until the program is executed and certified on post-merge master (which landed,
  `1ebfa4e4a`).
- No layout-census tool exists in-repo (`search_text` for `layout_census`/`LAYOUT-CENSUS`
  returns zero hits) — the epic bar's "zero weights materialized in two layouts" claim has no
  current instrument to produce it.

## Design constraints

- **Ruling 5 (LAYOUT FOLLOWS RESIDENCY / one layout per weight)**: "each weight is
  materialized once, in the layout optimal for the processor that executes it; ALL consumers
  (PP grouped and TG small-batch, gate/up AND down) must support that layout; never duplicate
  weights in two layouts, never shift layout at dispatch time. Routes only advertise (type,
  layout) pairs whose kernels exist. AOS fallback is a stopgap, not the design." Every
  producer/consumer-mismatch ticket in this epic (avhx, 34g5, tqka, 8hz8, l8at) is this
  ruling's enforcement surface.
- **Ruling 8 (small-block dequant stays on standard SYCL)**: "Q4_0/Q8_0/Q4_K belongs on
  standard SYCL, not ESIMD (measured 1.9x slower); the lever is fusing dequant into the
  matmul." Bears on kpv8's fused-ESIMD MXFP4 kernels and any new dense-GEMM executor 0yi9
  designs for float MMID.
- **Ruling 2 (layout identity on the handle)**: `mem_handle` carries layout identity; never a
  raw pointer or an address key. 8hz8's planner-owned representation identity work and tqka's
  `requested_layout`/`actual_layout` contract both operate inside this constraint.
- **Ruling 1 (allocation flows through the unified cache only)**: any new materialization path
  (a dense-GEMM scratch buffer for float MMID, a COALESCED reorder buffer for f9fx/95x2) must
  allocate via `unified_alloc`/`unified_allocate`/`unified_allocate_owner`, never
  `sycl::malloc_*` directly.
- **Ruling 9 (correctness before throughput)**: the Mistral completion gate and GPT-OSS chat
  gate are the closing bar for every member that touches dispatch; a CPU-fallback change
  (yitq) needs its own perf check because CPU fallback is not neutral (the graph split around
  a correct op can itself be the defect).

## External / upstream status

See `web_findings` for full citations. Summary relevant to this epic's decisions:

- Upstream `ggml-org/llama.cpp` has shipped **generic** (non-MoE) SYCL reorder/MUL_MAT
  coverage for the full IQ family (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS — this fork's
  own `docs/backend/SYCL.md:71` already lists them) but no evidence was found of an `_id`
  (MUL_MAT_ID / MoE-indexed) variant for any IQ type upstream — confirming llama.cpp-wh7o's
  claim that the 9 bespoke iq kernels have no transcribable `_id` sibling anywhere, upstream
  included. This is a genuine new-kernel program, not a port.
- Upstream has extended the SYCL MoE `mul_mat_id` reorder path to Q6_K experts (release
  b9664), completing reordered MoE coverage for K-quant down-projections — orthogonal to this
  epic's float/iq gap but confirms the reorder-path pattern llama.cpp-95x2/f9fx are auditing
  is the same one upstream keeps extending.
- oneDNN 2026.x documents `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY` grouped-GEMM support for MoE
  token routing, and Intel's oneDNN 2026 release notes claim improved Xe2/Xe3 matmul
  performance including float16-with-low-precision-weight paths. This is directly relevant to
  llama.cpp-0yi9's stated need for a dense per-expert GEMM for F16/F32 MMID (BF16 is already
  cleanly refused, see 0yi9's addendum) — oneDNN's grouped GEMM is the shape 0yi9 should
  evaluate before hand-writing a new kernel family, matching how the existing oneDNN SDPA and
  batched-PP-WOQ paths in this fork already delegate to oneDNN rather than writing bespoke
  device code. Note the cost this presupposes: `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY` is a
  build-time experimental CMake flag on oneDNN itself, and this fork links the oneAPI-shipped
  oneDNN, not a fork-built one — "evaluate first" means standing up a custom oneDNN build (or
  confirming the shipped oneAPI release already has grouped GEMM compiled in) before the
  recommendation is actionable.
- An upstream ggml-org discussion (#22042) flags current NVFP4 support as having "risk of
  functional incorrectness due to unclear separation of concerns" — a caution directly
  relevant to sgox/zqoe/tjk4's Q1/NVFP4 device-decode work: the certification bar those
  tickets already carry (CPU-oracle comparison, no seam exports) is the right response to
  exactly this upstream-documented risk, not overkill.
- A vLLM issue (#41663) reports a GP fault / BCS engine reset on dual-B70 tensor-parallel
  Level Zero use — but the reported rig is a Z890 board WITH a PCIe switch and no XeLink,
  where standalone XCCL/SYCL collective sweeps PASS and the fault is specifically a vLLM
  TP=2 worker GP fault plus a BCS engine reset. This corroborates cross-device Level Zero
  fragility under multi-GPU tensor-parallel use in general, NOT this fork's specific
  no-PCIe-P2P root-port topology finding (different rig, different fault mechanism) — still
  relevant to llama.cpp-7vpd (B70→B50 secondary decode) as a reason to test the host-bounce
  path explicitly rather than assume it "just works," but not as independent confirmation of
  the topology conclusion itself.
- No evidence found of an upstream or Intel fix changing the P2P topology conclusion or the
  VRAM budget iGPU issue.
- **CORRECTION (verified 2026-09-01, supersedes the epic's earlier "fork-local types" claim)**:
  `GGML_TYPE_NVFP4 = 40` and `GGML_TYPE_Q1_0 = 41` ARE present in upstream ggml-org/llama.cpp
  master today (confirmed by direct read of upstream's `ggml/include/ggml.h`, same enum
  values as this fork's `ggml.h:443-444`) — these types are no longer fork-local. Whether
  upstream ships SYCL MUL_MAT/MUL_MAT_ID kernels for them, and whether upstream's byte layout
  matches this fork's, was NOT established (GitHub code search required authentication this
  session lacked). sgox must check this before further Q1/NVFP4 wiring proceeds on the old
  "definitely original work, nothing to port" assumption — see `web_findings` for detail.

## Work breakdown

Prerequisites first; the Q1/NVFP4 sub-track and the P6 router sub-track are separate chains
that both feed the epic bar but do not block each other.

1. **llama.cpp-yitq** — gate `MUL_MAT_ID` admission in `supports_op` on real executor type
   coverage (the one-line fix that turns 234 wrong-numbers failures into clean CPU
   fallbacks). Epic prerequisite; everything downstream is judged against its post-fix state.
2. **llama.cpp-avhx** — fix the producer (`ggml_sycl_select_moe_expert_cache_layout`) so the
   advertised route layout matches what the cache actually materialized, for the MXFP4/Q6_K
   PP gate/up case.
3. **llama.cpp-34g5** — remove (or properly reconcile via staging) the `allow_aos_fallback`
   waiver in `mmvq.cpp`, the third known instance of the advertise/materialize mismatch class.
4. **llama.cpp-tqka** — close the `requested_layout` vs `actual_layout` contract gaps (F-6/F-7/F-8)
   that made the mismatch class possible to introduce three times.
5. **llama.cpp-8hz8** — enforce planner-owned representation identity generally: reject an
   unplanned second layout for a tensor at cache-entry-creation time, not just react to it.
6. **llama.cpp-l8at** — adjudicate `test-sycl-layout-choice`'s stale SOA-primary expectation
   against the current one-layout-per-weight ruling; fix whichever side (test or planner) is
   wrong, then sweep the rest of that binary's never-executed cases.
7. **llama.cpp-pwb2** — read-only reachability proof: is the type switch in
   `ggml_sycl_mul_mat_id_vec_q` dead code behind an earlier gate, or a missing route? Feeds the
   real-coverage answer that 0yi9/wh7o need before sizing new kernel work.
8. **llama.cpp-0yi9** — build the dense-GEMM MMID recipe for F16/F32/BF16 (evaluate oneDNN
   grouped GEMM per the web finding above before hand-rolling XMX).
9. **llama.cpp-wh7o** — port the 9 bespoke iq* MMVQ kernels into `_id` (MoE-indexed) variants;
   confirmed genuine new-kernel work, not a port from upstream.
10. **llama.cpp-a6sy** — fix or accept the MMID `route_unavailable` refusal for the NVFP4/fp8
    aux-scale interaction on the llama-MoE archs fixture.
11. **llama.cpp-f9fx** — determine reachability of the device AoS→COALESCED reorder kernel and
    add a dispatch-driven MMVQ_COALESCED fixture (or delete the dead kernel).
12. **llama.cpp-95x2** — resolve the Q6_K "variable tile" design question: wire the live
    variable-tile CPU writer to a matching dispatch-side eligibility check, or delete the
    unreachable generality.
13. **llama.cpp-19l2** — guard the split-buffer (`-sm row`) Q8_0 coalesced fp32 dequant with
    `src0_full_tensor`, same defect class 95x2/f9fx are auditing.
14. **llama.cpp-kpv8** — deliberately exercise the two repaired `<0>`-template fused-ESIMD
    kernels (q8_0, mxfp4) with a positive control, per the standing acceptance criterion.
15. **llama.cpp-g4i8** — identify the six repeating `name_hash` tensors behind the
    `direct_stage_weight` rejection noise; demote or resolve.
16. **llama.cpp-s2cw** — verify attention SOA weights actually pin in VRAM from a host-pinned
    source (re-scope against the current canonical contract; this predates several rulings).
17. **llama.cpp-6sp4** — give the test-only layout override a fixture that materializes
    matching layout info first, so it can actually bind on the device path.
18. **llama.cpp-cyn.2** — re-verify against current Option T / XMX-tiled state whether the
    original OOM (SoA buffer not freed after tiled conversion) still reproduces; close or
    re-scope with a concrete repro.
19. **llama.cpp-n36** — re-verify whether the load-time tiled-conversion pipeline (task 1) is
    already served by the production `scoped_planned_materialization` path; close or narrow to
    the actual gap.
20. **llama.cpp-omp4** — Q1/NVFP4 device-decode program umbrella (unblocked post-merge).
21. **llama.cpp-sgox** — wire primary Q1_0/NVFP4 direct-AoS device decode.
22. **llama.cpp-zqoe** — Stage 2: direct AoS MMVQ-ID device kernels.
23. **llama.cpp-5i7z** / **llama.cpp-7vpd** — bounded prompt chunks / secondary (B70→B50)
    owner-queue decode, in parallel once zqoe lands.
24. **llama.cpp-tjk4** — post-reboot B70 certification of the private production route; the
    program's shipping gate.
25. **llama.cpp-h690** / **llama.cpp-o54r** — P6 router fast-path and fusion restoration over
    retained batches (independent chain, same epic, same `omp4` umbrella).
26. **NEW: layout-census tool** (see `new_tasks`) — the instrument the epic's own bar requires
    and that does not currently exist.

## Bar / closing gate

- **Correctness (mandatory, every member that touches dispatch or layout selection):**
  ```
  source /opt/intel/oneapi/setvars.sh --force
  ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
    -m /models/mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
  # expect: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

  ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
    -m /models/gpt-oss-20b-mxfp4.gguf -ngl 99 -c 4096 \
    -cnv -st --simple-io --no-display-prompt \
    --chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --reasoning-format none --reasoning-budget 0 \
    -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' -n 48 --seed 42 --temp 0
  # expect: 1, 2, 3, 4, 5
  ```
- **Coverage (yitq/0yi9/wh7o/a6sy):** `test-backend-ops` MUL_MAT_ID failures for previously
  uncovered types drop from wrong-numbers to either a passing dense-GEMM result (0yi9/wh7o
  once ported) or a clean `SKIPPED`/CPU-fallback status (yitq alone) — zero remaining
  `ERR 86-99`-class escapes. `test-llama-archs -a llama` reports no `MOE-PROMPT-REFUSAL`
  aborts (a6sy) and `rc=0`.
- **Layout census (new instrument, avhx/34g5/8hz8/tqka/l8at):** a census run over a full
  GPT-OSS 20B load + one PP + one TG step reports zero tensors with two live materialized
  layouts and zero `route_layout != src0_layout` disagreements in `[MOE-DISPATCH-LAYOUT]`
  output.
- **Perf non-regression (yitq's CPU-fallback change, any dense-GEMM addition):** gate against
  `docs/backend/sycl-perf-baselines.md`, not this file — B70 GPT-OSS ~1415 PP512/~44 TG128,
  B70 Mistral ~2495/~108, B50 GPT-OSS ~894/~32, B50 Mistral ~1188/~47 (interleaved paired A/B
  against pre-change HEAD; absolute numbers under this host's ambient load are not baselines).
- **Q1/NVFP4 track (sgox→zqoe→5i7z/7vpd→tjk4):** `tjk4`'s private production-route CTest suite
  green serially on B70 (lifecycle commit, CPU-oracle comparisons, candidate/admit/submit/
  terminal/recycle counters, no hangs), plus the Mistral and GPT-OSS gates above still green
  with the capability flag enabled.
- **P6 router track (h690/o54r):** the numeric primary/secondary/mixed/repeated-ID matrix
  each ticket specifies, plus a PP/TG perf gate against the same baselines table.

## Out of scope

- The specific mn70/zoly consumer-side commits that already landed (cited as current-state
  evidence above) — this epic covers what remains, not a re-litigation of closed work.
- Any B580-hardware-specific figure or topology (card removed 2026-07-24).
- General SYCL device-selection, VRAM-budget, or P2P-topology work — those are settled
  rulings (10, and the P2P entry in CLAUDE.md), referenced here only as constraints.
- `llama.cpp-jjm7`, `llama.cpp-wm4j`, `llama.cpp-0oad` and their children — owned by another
  session per this triage's exclude list.

## Dependencies on other epics

- Depends on the `mn70`/`zoly` consumer-side layout-mismatch fixes already landed (external to
  this epic's member list, cited as current-state evidence) — avhx and 34g5 build directly on
  top of that landed work.
- The Q1/NVFP4 sub-track (`omp4` and its children) depends on the merge of
  `feature/sycl-b70-capability` into master, which has landed (`1ebfa4e4a`) — no longer a
  blocker, only a prerequisite already satisfied.
- Shares the "advertise only what a kernel covers" invariant with any future epic that adds a
  new quantization type (the recipe program is the template for how a new type should be
  wired: capability table entry, dispatch table entry, both gates, in one change — per
  llama.cpp-wh7o's own note on this).

## Tasks

### llama.cpp-yitq

  - **Id:** llama.cpp-yitq
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Stop `supports_op` from admitting `MUL_MAT_ID`/`ADD_ID` for types no SYCL MMID executor covers, so uncovered types get a clean CPU fallback instead of a silently wrong GPU answer.

**Design**:
- Corrected mechanism (the earlier premise that admission is unconditional was wrong — see the epic's "Why now"): the `MUL_MAT_ID`/`ADD_ID` branch in `ggml_backend_sycl_device_supports_op` (`ggml-sycl.cpp:102023`, branch at `:102120-102142`) already checks `src0->type`, but against `ggml_sycl_mul_mat_type_supported()` (`:101973-102000`) — the shared regular-MUL_MAT allowlist, which is why F32/F16/IQ* pass. The fix is not "add a missing check", it is: build an MMID-specific coverage predicate (do not reuse the MUL_MAT allowlist for this branch) and gate on it instead. `moe_mmvq_batched_dispatch_supports_type()` (`moe-mmvq-tables.hpp:73`) is the right ingredient — it already has two call sites in `mmvq.cpp` (`:16292` diagnostic-reason string, `:20763` a real layout-refusal gate) — but it answers "does an executor exist for this type at all", one axis short of admission; combine it with (or replace it by) a query that also confirms at least one advertised layout, so admission and the MMVQ tables' own invariant (capability ⊆ union of executor tables) stay in the same shape.
- Do NOT widen `moe_mmvq_capability_supports_layout` or the dispatch tables — `test-sycl-moe-mmvq-tables.cpp` gates their current false-for-uncovered state on purpose.
- This will not clear Mechanism D (nvfp4/q1_0 n=1, deliberately rejected at `choose_moe_batch_executor`, `ggml-sycl.cpp:24137-24141`) — leave that path alone.
- Commit `186348705` (2026-08-26) already closed BF16 and the b10630 q2_0/tq2_0 types through `ggml_sycl_mul_mat_type_supported()` (`ggml-sycl.cpp:102120-102139`) — those are NOT part of this ticket's remaining scope. The remaining gap is specifically F32/F16/IQ1_S..IQ4_XS, which pass admission today only because they are legitimate MUL_MAT types sharing that allowlist; this ticket's new MMID-specific predicate must refuse them (cleanly, to CPU) until 0yi9/wh7o land real executors, then re-admit them from the new predicate, not the shared one.
- CPU fallback is not neutral (see CLAUDE.md): a graph-split around a previously-GPU op has its own cost, so this needs its own PP/TG perf check, not just a correctness check.

**Bar**: `test-backend-ops` MUL_MAT_ID cases for F32/F16/IQ1_S..IQ4_XS (the re-derived residual gap; BF16/q2_0/tq2_0 already closed by `186348705`) stop reporting `ERR 86-99`-class garbage and instead show a clean fallback (or a passing dense-GEMM result once 0yi9/wh7o land). Zero regression on Mistral/GPT-OSS PP/TG baselines from the added graph splits.

**Constraints**: Ruling 5 (advertise only real kernel coverage), ruling 9 (perf check required for any CPU-fallback change).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '^test-backend-ops$'` (manually, monitored, never in a subagent — see CLAUDE.md never-loop rule; the selector pin excludes the Arrow Lake-S iGPU, whose 231.7GB advertised VRAM is the documented 50-224GB TTM-shmem OOM hazard) for the type coverage; `llama-bench` Mistral/GPT-OSS PP512/TG128 paired A/B against pre-change HEAD for perf.

**Depends on**: none — this is the epic's prerequisite.

### llama.cpp-avhx

  - **Id:** llama.cpp-avhx
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-yitq

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Make the MoE layout producer advertise the layout the unified cache actually materialized, not an optimistic probe result, for MXFP4/Q6_K gate/up under XMX_TILED at n_tokens>1.

**Design**:
- Correct target (per the ticket's own self-correction, c-r154) is `ggml_sycl_select_moe_expert_cache_layout` (`ggml-sycl.cpp:26492-26520`, current HEAD line numbers — moved from the description's stale `25437-25460`), specifically the `n_tokens > 1` GATE/UP branches under MXFP4/Q6_K that return `ggml_sycl_adjust_layout_for_tensor(src0, GGML_LAYOUT_SOA, device)`.
- The upstream seam is `common.hpp:1101-1107`, which returns `layout=SOA, reason="xmx-tiled-not-validated-shared-soa"` while materialization simultaneously logs `layout=aos` with the identical reason string — trace which side adjusts and which keeps the unadjusted value; that divergence point is where to fix it.
- Two valid fix shapes: (a) have the selector consult actual materialization state instead of a probe, or (b) have the cache's materialization decision feed back into the route before it is advertised. Pick whichever keeps a single source of truth per ruling 5.
- This is the planner side; do not conflate with mn70's consumer-side refusal guard (`:57534-57565`) or 34g5's `allow_aos_fallback` waiver — three distinct sites in the same bug class.
- Judge success on planner-correctness alone (per c-q25c): does the selector's output match materialization, independent of which consumer currently asks. A consumer no longer reaching this path is not evidence of a fix.
- Needs its own perf check on the GPT-OSS PP path specifically — this selector is the one production path serving it (0/2304 experts host-resident, device-side dequant/staging).

**Bar**: For the previously-mismatching shapes, `[MOE-DISPATCH-LAYOUT]` reports `route_layout` equal to the actually-materialized layout; mn70's refusal guard does not fire on them. GPT-OSS PP512/TG128 unchanged against baseline.

**Constraints**: Ruling 5 (one materialized layout, advertised honestly).

**Acceptance test**: `GGML_SYCL_FA_DISPATCH_DEBUG` is the wrong knob — it is read only in `fattn.cpp`/`fattn-onednn.cpp` (flash-attention dispatch), not the MoE layout path. The line that actually matters, `[MOE-DISPATCH-LAYOUT] tensor=... route_layout=... src0_layout=...`, is emitted from `ggml-sycl.cpp:70815`, gated by `ggml_sycl_moe_route_log_enabled()` (`:5423`), which fires on `GGML_SYCL_MOE_ROUTE_LOG=1` (or `GGML_SYCL_DEBUG=1`). Reproduce the mn70/zoly signature with `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_MOE_ROUTE_LOG=1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 4 -r 1` on a GPT-OSS PP run; confirm zero `route_layout != src0_layout` lines. This line is emitted via raw `fprintf(stderr)`, not `GGML_LOG_INFO`, so it is NOT subject to `llama-bench`'s null-log-callback suppression — `-v` is not required to see it. Follow with the GPT-OSS chat gate and a PP512 baseline check.

**Depends on**: llama.cpp-yitq (advertise-vs-materialize seam is easier to reason about once the coarse admission gate is honest first), llama.cpp-mn70 (external, already landed — cited as current-state evidence).

### llama.cpp-34g5

  - **Id:** llama.cpp-34g5
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-avhx

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Close the third known instance of the advertise-vs-storage mismatch class: `mmvq.cpp`'s `allow_aos_fallback` waives the layout-disagreement refusal instead of refusing or reconciling.

**Design**:
- Confirmed current at `mmvq.cpp:20742-20751`: `allow_aos_fallback = forced_layout && layout == GGML_LAYOUT_AOS` waives `effective != layout`, logs at DEBUG (invisible at default verbosity — violates the guard-visibility convention this codebase otherwise follows), and proceeds without staging.
- Ruling 5 forbids shifting layout at dispatch time at all, so the only ruling-compliant fix shape is the refusal arm: an executor that finds a disagreeing advertised layout must refuse at dispatch (WARN, not DEBUG-and-silent), never reconcile via staging inline in the dispatch path. This branch is a violation-by-waiver, distinct from mn70's no-reconciliation-at-all case and avhx's no-kernel-availability-check case — it does not license adding a dispatch-time conversion here.
- Confirmed LATENT: zero `Using AoS fallback` / `mismatches effective` hits in available logs — establish a reachability case (find or construct one) before claiming the fix is exercised; an unexercised guard fix is a standing hazard per the kpv8 precedent.
- Fix shape: replace the silent waiver with a WARN-level refusal matching the guard-visibility rule (ruling 5 forbids a dispatch-time reconciling conversion inside `mmvq.cpp`, so that is not an available second option here). Read the surrounding `!mxfp4_moe_reorder_dispatch` block first; the waiver is confirmed NOT inside the legitimate MXFP4 reorder carve-out. If investigation turns up a genuinely legitimate AOS-is-fine case that would otherwise need staging, that staging belongs in the producer (avhx's `ggml_sycl_select_moe_expert_cache_layout`) at materialization time, not in this dispatch-time consumer — escalate rather than adding it here.
- Sequence after the mn70/zoly active fix series and after avhx's producer fix, per the ticket's own note — those may narrow or eliminate this branch's live callers before you touch it.

**Bar**: No production dispatch takes the `allow_aos_fallback` branch silently; the branch either refuses at WARN (the compliant outcome) or is removed because closing avhx eliminates its live callers. No dispatch-time reconciling staging is added to `mmvq.cpp`.

**Constraints**: Ruling 5 — never shift layout at dispatch time. A route that disagrees with storage at dispatch time must refuse (WARN-level, visible); it does not get to reconcile in place. Reconciliation, if genuinely needed, is producer-side (materialization-time) work, out of scope for this ticket.

**Acceptance test**: Construct or find the reaching case; confirm the new WARN-level refusal line fires under `GGML_SYCL_DEBUG=1` on that case (or confirm the branch has zero live callers once avhx lands, and remove it); Mistral/GPT-OSS gates unaffected.

**Depends on**: llama.cpp-avhx (closing the producer first likely shrinks or eliminates this waiver's live callers).

### llama.cpp-tqka

  - **Id:** llama.cpp-tqka
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-avhx

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Close three related hardening gaps (F-6/F-7/F-8) in the `requested_layout` vs `actual_layout` contract `ab38287d5` introduced — the contract that, once solid, prevents the class of bug avhx/34g5 fix instances of.

**Design**:
- F-6: `ggml_sycl_moe_validate_retained_role_bundle`'s call site (`ggml-sycl.cpp:67624-67626`, confirmed current) passes `batch.operands.front().actual_layout()` to `moe_batch_role_admissible` — self-referential, can never detect disagreement with what was *requested*. Fix per-site: the function has five `front().actual_layout()` uses and at least one (`:67640`-area capability query) is legitimately about actual layout — do NOT blanket-swap all five.
- F-7: `moe_retained_role_batch::requested_layout` (`moe-resolved-batch.hpp:654`) defaults to `GGML_LAYOUT_AOS` (enum 0), a valid-looking value, on three default-constructed aggregate members — exactly the failure shape ("reads AOS more often than it should") this contract exists to prevent. Add a NONE/COUNT sentinel to the enum, or a `requested_layout_valid` bool, or at minimum document the single-writer invariant.
- F-8: `moe_batch_role_admissible`'s precondition ("only immediately after the same batch's own build, no queue submission or yield in between", `moe-resolved-batch.hpp:558-574`) is unenforced; add a test pinning the documented divergence (stale lease → `local_view` returns `STALE_HANDLE`, `role_admissible` returns `true` anyway) so the contract is executable, not prose.
- `ggml-sycl.cpp` is codescout-index-blind — verify all line numbers with `cat | grep -n`, not the index.

**Bar**: F-6's five call sites are individually reviewed and only the genuinely-actual-layout ones keep `front().actual_layout()`; F-7's sentinel/valid-bool exists and is checked at read sites; F-8's divergence has a red-then-green regression test.

**Constraints**: Ruling 2 (layout identity travels on the handle — `requested_layout` must be a first-class, validity-checked field, not a defaultable enum that silently reads as a real value).

**Acceptance test**: New unit test in the MoE retained-batch test suite exercising F-8's stale-lease divergence; existing `test-sycl-moe-resolved-batch.cpp` suite green after the F-6 site-by-site fix.

**Depends on**: llama.cpp-avhx (shares the same root cause class; fixing the producer first reduces the number of live disagreement cases F-6 has to handle correctly).

### llama.cpp-8hz8

  - **Id:** llama.cpp-8hz8
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-yitq

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Prevent an unplanned second layout from ever overwriting a tensor's single planned representation — the proactive counterpart to the reactive `GGML_ABORT` checks that already exist.

**Design**:
- Current state (design_gaps): reactive `GGML_ABORT("unified_cache layout mismatch")` enforcement exists broadly (`unified-cache.cpp:4314-4318,8368-8372,11183-11187`) and `layout_supports_coalesced` gates dimension mismatches pre-selection (`common.hpp:4184`, `ggml-sycl.cpp:26245`) — but nothing validates a *layout request* against the placement *plan* before a cache entry is created.
- Add validation at cache-entry-creation time: compare the incoming layout request against the placement plan's recorded representation for that logical tensor id; reject (not abort) an unplanned second layout, with a log line naming logical tensor id, representation/layout id, device/host, and caller.
- Do NOT add a "planner-budgeted second layout" opt-in — ruling 5 forbids duplicating a weight in two layouts unconditionally, and a planner-authorized duplicate is still a duplicate. If a legitimate case for two live representations of the same logical tensor is ever found, that is a proposed change to ruling 5 itself and belongs in front of the owner, not encoded as a quiet exception inside this hardening ticket. This ticket's validation must reject every unplanned second layout with no opt-in path.
- `mem_handle` hash/equality is backing-allocation identity: two handles to the same allocation hash the same; different layout allocations for the same logical tensor are different handles — this identity model does not change, only the request-time validation gets added.
- Original rewritten (2026-08-02) acceptance ("no loader-local placement authority...") is generic architecture boilerplate, not falsifiable — replace it with the concrete reject-and-log behavior above before calling this done.

**Bar**: A synthetic test that requests a second, unplanned layout for an already-planned tensor is rejected with a logged mismatch (tensor id, layout ids, device/host, caller), not a silent overwrite and not an abort.

**Constraints**: Ruling 2 (mem_handle owns layout identity), ruling 1 (validation lives in the unified cache, the sole allocator).

**Acceptance test**: New unit test in the unified-cache test suite constructing the reject case; existing `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '^test-unified-cache-bugs$'` (serial only, peaks ~8.5GB RSS, selector pin excludes the iGPU OOM hazard) stays green.

**Depends on**: llama.cpp-yitq (the advertise-vs-materialize seam this closes is cleaner to validate once the coarse admission gate is honest).

### llama.cpp-l8at

  - **Id:** llama.cpp-l8at
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-avhx

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Adjudicate whether `test-sycl-layout-choice`'s SOA-primary expectation for gate/up is stale, or whether the planner genuinely mis-prioritizes for PP, under the 2026-08-17 one-layout-per-weight ruling — then fix whichever side is wrong.

**Design**:
- Confirmed still reproducing post-b10630 merge (`artifacts/merge-b10630/sweeps/filtered.log:580`): `run_single_device_moe_pp_complete_soa_layout_test` FAILS with `gate=4 up=4` (XMX_TILED) where the test expects SOA-primary.
- The case feeds `compute_multi_device_plan` a synthetic `device_budget` directly — confirmed unaffected by the o3h1 budget-authority change (`9c16dc878` touches only the production caller `compute_and_store_plan_for_inventory`), so this is a genuinely latent, pre-existing failure, not a regression from that commit.
- Do the `git log -L` adjudication the ticket specifies: trace the planner's layout-preference logic history vs. the test's expectation history to see which one moved and which one is stale.
- Under ruling 5 (one best combined PP+TG layout per weight, no dispatch-time shifts), XMX_TILED-as-primary may be correct current behavior for gate/up — if so, fix the test's expectation, not the planner.
- The binary short-circuits on first FAIL — sweep every case in `main()` after this one once it is fixed; they have never executed and may hide further latent failures.

**Bar**: `test-sycl-layout-choice` runs to completion (no short-circuit) with either the test's expectation corrected to match the ruling-5-compliant planner behavior, or the planner corrected if it is shown to genuinely mis-prioritize PP.

**Constraints**: Ruling 5 (one layout per weight — the adjudication itself is deciding what "optimal for PP+TG" means for gate/up under XMX_TILED).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '^test-sycl-layout-choice$' --output-on-failure` (selector pin excludes the Arrow Lake-S iGPU OOM hazard) green, all cases in `main()` executed (not short-circuited).

**Depends on**: llama.cpp-avhx (the producer this test exercises is the one avhx is fixing; resolve avhx first so the test isn't adjudicated against a producer mid-fix).

### llama.cpp-pwb2

  - **Id:** llama.cpp-pwb2
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - llama.cpp-yitq

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Read-only reachability proof — is the extended type switch in `ggml_sycl_mul_mat_id_vec_q` (covering Q1_0/NVFP4/Q4_1/Q4_K/Q5_K/Q6_K/Q5_0) dead code behind an earlier restrictive gate, or a missing route that should dispatch there?

**Design**:
- Confirmed current: the early "Check for supported types" gate (`mmvq.cpp:20801`) restricts to Q4_0/Q8_0/MXFP4 (`return false` else); the later dispatch switch (`mmvq.cpp:21187`) only implements those 3 cases; a THIRD switch further down (`~21420`) still carries the wider type list.
- Q6_K's actual execution path is confirmed to be a different function, `mmvq_moe_batched_dispatch` (`mmvq.cpp:~16118`), which carries its own roster-derived gate from commit `28df70626` — so Q6_K at least is not starved by this.
- Determine which of the two possibilities holds by reading both functions' *callers*, not by name: (a) the gx30 coverage wave wired instantiations into a function that can never dispatch them (dead branches, re-derive gx30's real per-type coverage claim against whichever function actually serves each type), or (b) the early gate is over-restrictive and a route is genuinely missing.
- This bears directly on 0yi9/wh7o's sizing: if (a), gx30's "8 types conclusively wired" claim needs re-derivation against the function that actually serves them before 0yi9/wh7o budget new kernel work assuming a clean baseline.
- Read-only diagnosis first — no fix without the reachability proof, per the ticket's own instruction.

**Bar**: A written determination, per type, of which function actually dispatches it in production, with caller-graph evidence (not name-matching) — feeding a corrected coverage table for 0yi9/wh7o to build on.

**Constraints**: None directly — this is a diagnostic ticket that de-risks 0yi9/wh7o's design.

**Acceptance test**: No code change required to close the diagnostic half; if (a) is confirmed, follow-up PR removes the dead switch cases and updates gx30's coverage record.

**Depends on**: llama.cpp-yitq (establishes the honest capability-table baseline this reachability proof should be read against).

### llama.cpp-0yi9

  - **Id:** llama.cpp-0yi9
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-yitq
  - llama.cpp-pwb2

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Build the missing MMID (MoE-indexed MUL_MAT) executor for F16/F32 (BF16 is already refused cleanly, see below) — a dense per-expert GEMM, since the MMVQ path (which quantizes activations to Q8_1) has no meaning for a float weight.

**Design**:
- Corrected scope: `moe-mmvq-tables.hpp:37-132` excludes F32/F16/BF16 from all three MMVQ tables (default false, correctly — no MMID executor covers them). But BF16 is stale as a target here: `ggml_sycl_mul_mat_type_supported()` (`ggml-sycl.cpp:101973-102000`) does NOT list `GGML_TYPE_BF16`, so a BF16 `MUL_MAT_ID` is already refused at admission by `186348705` (2026-08-26) — it produces a clean CPU fallback today, not garbage. This ticket's real remaining scope is F16/F32 only (both ARE in that allowlist, so they pass admission with no MMID executor behind them — the shared-allowlist gap `yitq` is fixing).
- **Evaluate oneDNN grouped GEMM first** (`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`, per this epic's web findings) before hand-writing a new dense-GEMM device kernel or generalizing the XMX tiled path off MXFP4 — this fork already delegates comparable dense-matmul work (SDPA, batched-PP-WOQ) to oneDNN rather than bespoke device code, and grouped GEMM is designed for exactly MoE's per-expert routing shape. Note the cost this presupposes: `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY` is a **build-time** experimental CMake flag on oneDNN itself, and this fork links the oneAPI-toolkit-shipped oneDNN, not a fork-built one — "evaluate first" means standing up a custom oneDNN build (or confirming the shipped oneAPI release already has grouped GEMM compiled in) before this recommendation is actionable, not just flipping a runtime switch.
- Establish reachability first: `test-backend-ops` constructs these cases; whether a shipped GGUF produces float MoE experts is not established. If the only consumer is the harness, size accordingly — this changes cost/benefit sharply versus the q4_K case gx30 closed.
- Denominator: the prior census's `F16 80, F32 73, BF16 3` split is stale for the reason above (BF16 is already refused, not garbage) — re-measure `test-backend-ops` MUL_MAT_ID failures against post-`186348705` HEAD before sizing this work, and expect the true F16/F32-only denominator to differ from the old F16+F32+BF16=156 total. MUL_MAT_ID_FUSION adds up to 126 more (float-heavy, exact split not measured) — re-measure that split too.
- Must allocate any new scratch/staging through the unified cache (ruling 1), and the new route must only advertise once a real kernel exists (ruling 5) — do not widen the capability table ahead of the kernel landing.

**Bar**: `test-backend-ops` MUL_MAT_ID F16/F32 cases pass numerically (CPU-oracle comparison) via the new dense-GEMM route; BF16 stays cleanly refused (no change needed, already correct); GPT-OSS/Mistral gates unaffected (these types are not on the production hot path per the reachability caveat).

**Constraints**: Ruling 5 (advertise only once the kernel exists), ruling 1 (unified-cache allocation), ruling 8 (small-block dequant stays off ESIMD — not directly applicable to float types but the same "fuse into the matmul" philosophy applies to any staging this adds).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '^test-backend-ops$'` (manual, monitored, never in a subagent; selector pin excludes the Arrow Lake-S iGPU OOM hazard) scoped to F16/F32 MUL_MAT_ID cases (re-measured denominator, not the stale F16/F32/BF16 one); `test-sycl-moe-mmvq-tables.cpp` and `test-sycl-moe-mmvq-consumer-coverage.py` updated in the same change per their own "move both gates together" requirement.

**Depends on**: llama.cpp-yitq (clean refusal baseline), llama.cpp-pwb2 (coverage-table reachability proof feeds accurate sizing).

### llama.cpp-wh7o

  - **Id:** llama.cpp-wh7o
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-yitq
  - llama.cpp-pwb2

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Port the 9 bespoke iq* MMVQ kernels into `_id` (MoE-indexed) variants — confirmed genuine new-kernel work with no shortcut, upstream included (see web findings).

**Design**:
- Confirmed unchanged at HEAD: none of the 9 iq types (`iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, `iq4_xs`) appear in `moe_mmvq_batched_dispatch_supports_type`/`_layout` or `moe_mmvq_capability_supports_layout` (all default false).
- Each type is a **bespoke** kernel (`mmvq.cpp`), not a generic-template instantiation: different arity (4 template params vs the generic's 5, no `vec_dot` parameter — decode is baked into the kernel body), adjusted `qi` divisors (`/2`, `/4`) that don't fit the generic `mul_mat_vec_q_id` lane partitioning, and grid-table lookups the generic AoS indexing has no provision for. Port each one individually, preserving its own decode path and grid-table access plus adding the `_id` per-expert pointer-table indexing (`expert_ptrs[expert_id]` selection, 2D ids/token/y offsets) that `mul_mat_vec_q_id` performs generically.
- **Preserve the `iq1_m` uses `QI1_S` (not `QI1_M`) quirk** — match the existing launcher exactly; do not "fix" it, same class as q4_1's `qk = QK4_0` that gx30 also preserved.
- Web finding: upstream ships generic (non-MoE) IQ reorder/MUL_MAT support (this fork's own `docs/backend/SYCL.md:71` already lists it) but no `_id` variant was found — nothing to port from upstream, this is original work.
- Covering any iq type must move the capability table, the dispatch table, AND both gates (`test-sycl-moe-mmvq-tables.cpp`, `test-sycl-moe-mmvq-consumer-coverage.py`) in one change — do not widen capability ahead of the kernel.
- Establish reachability first (does any shipped MoE GGUF use iq experts?) before budgeting full 9-kernel effort — same caveat as 0yi9.

**Bar**: `test-backend-ops` iq2_xxs (73 cases, the majority) plus the remaining 8 types (2 each, 24 cases) pass numerically. GPT-OSS/Mistral gates unaffected unless a production model actually uses iq experts.

**Constraints**: Ruling 5 (advertise only once each specific kernel lands — this is 9 separate advertisements, not one), ruling 8 (fuse dequant into the matmul, not ESIMD, for the small-block iq formats).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '^test-backend-ops$'` (selector pin excludes the Arrow Lake-S iGPU OOM hazard) scoped to each iq type's MUL_MAT_ID case, incrementally as each port lands; both gate tests updated per-type.

**Depends on**: llama.cpp-yitq, llama.cpp-pwb2. Independent of llama.cpp-0yi9 (different executor family — this is per-type bespoke kernels, not one shared dense-GEMM path) so the two can proceed in parallel.

### llama.cpp-a6sy

  - **Id:** llama.cpp-a6sy
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Determine why the llama-MoE archs fixture is refused at the MMID prompt route (`recipe-unavailable`) for NVFP4/fp8 aux-scale companion tensors, and fix the generator or the classifier.

**Design**:
- Confirmed still live at HEAD: `route_unavailable`/`MOE-PROMPT-REFUSAL` machinery present at `ggml-sycl.cpp:25722,69081,72082-72496`; no fix commit found since ticket creation (`git log --grep` empty).
- Layer 1 (context creation) is already fixed on the merge branch: `plan_moe_mmid_workspaces` now only lets `.weight` tensors drive workspace geometry, so a model carrying NVFP4/fp8 expert-scale companion tensors (`blk.N.ffn_*_exps.scale` / `.input_scale`, ne=2x1) no longer gets its whole context refused.
- Layer 2 (this ticket): with context creation fixed, decode is still refused: `[MOE-PROMPT-REFUSAL] tensor=blk.0.ffn_gate_exps.weight reason=route_unavailable recipe-unavailable`. Two hypotheses to distinguish: (a) the archs fixture generator emits NVFP4 scale companions for a non-NVFP4 synthetic model and their mere *presence* flips route classification into the fail-closed NVFP4 path (fixture bug or classification-by-presence bug), or (b) the fixture's expert weight type genuinely has no MoE prompt recipe, and the refusal is designed-correct — in which case fix the fixture generator to not emit that type, or mark the case as an expected-refusal row.
- Bisect window: the Aug-10 sweep was green, so the regression landed in the E-flip/o3h1 window — candidates are NVFP4 optional-tensor additions in `llama-model.cpp` or route-table/publication changes in that window.
- Also affects `test-recurrent-state-rollback-dsv4` (same failure mechanism, confirmed by lead 2026-08-27) — add it to this ticket's repro set once fixed.

**Bar**: `test-llama-archs -a llama` runs to completion with no `MOE-PROMPT-REFUSAL` abort on the MoE config; if (b) is the answer, the fixture instead produces an explicit expected-refusal row rather than aborting the sweep.

**Constraints**: Ruling 5 (a route decision should be made on the real type/layout, not tensor presence as a proxy for type).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a llama` (single run only, per the never-loop rule) reaches the MoE config without abort, rc=0.

**Depends on**: none directly, but should be judged after llama.cpp-h690/llama.cpp-o54r land per the epic's ordering hint (their P6 router changes touch the same MMID prompt admission path this ticket's refusal sits in).

### llama.cpp-71hx

  - **Id:** llama.cpp-71hx
  - **Action:** close
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Close evidence:** Fixed by c7b805740 (perf-recovery track C, 2026-08-22, llama.cpp-sr83): the batched PP oneDNN executor now consumes WOQ directly (dequant deleted), and the commit message states verbatim 'the remaining per-expert staging fallback cannot decode XMX_TILED weights and would silently produce garbage (llama.cpp-71hx)... SOA-claimed (down) declines keep the fail-closed refuse every dispatch'. Current-HEAD grep confirms explicit llama.cpp-71hx citations at ggml-sycl.cpp:67586-67588 and :74243 documenting the fail-closed contract is in place, not the silent-garbage behavior the ticket describes.

### llama.cpp-f9fx

  - **Id:** llama.cpp-f9fx
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Determine whether the szv8b-fixed device AoS→COALESCED reorder kernel is production-reachable, and add missing dispatch-driven MMVQ_COALESCED coverage.

**Design**:
- Gap 1: `reorder_q4_0_aos_to_coalesced_kernel` (`convert.cpp`) is reachable only via `convert_tensor_to_coalesced`'s `reorder_mode::NONE` branch (`mmvq.cpp:~3602`); S1-PRELOAD reorders on the HOST instead (`fill_reordered_host` → `reorder_q4_0_coalesced_cpu`), so the current test validates only the host producer, never touching the device kernel. Confirmed: no `test-mmvq-coalesced.cpp` or dispatch-driven fixture exists under `ggml/src/ggml-sycl/tests/` (search_text: 0 matches).
- First determine whether the `NONE` branch is reachable from a device-resident AoS weight in production at all. If not, the device kernel may be dead code — the right fix then is deletion, not a fixture, per the design_gaps note.
- Gap 2: nothing drives `MMVQ_COALESCED` through real dispatch — existing coverage launches kernels directly on isolated test buffers, the same test-misses-target gap class llama.cpp-95x2/llama.cpp-19l2 are independently finding in this coalesced-layout family.
- Related: llama.cpp-4wkt (the sliced-dispatch real fix) wants the same dispatch-driven sliced fixture — coordinate rather than duplicate fixture-building effort.

**Bar**: Either (a) a dispatch-driven fixture exists that reaches the device reorder kernel from a real device-resident AoS weight and a real MMVQ_COALESCED dispatch, both scored against CPU reference, or (b) both are confirmed dead and removed with a commit citing the reachability proof.

**Constraints**: Ruling 1 (any new fixture-driven materialization goes through the unified cache, not a standalone test buffer).

**Acceptance test**: New test under `ggml/src/ggml-sycl/tests/` exercising the device path end-to-end via real dispatch, or a removal PR citing the negative reachability proof.

**Depends on**: none — independently diagnosable; coordinate with llama.cpp-4wkt if that work is picked up concurrently.

### llama.cpp-kpv8

  - **Id:** llama.cpp-kpv8
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Deliberately exercise the two repaired `<0>`-template fused-ESIMD kernels (q8_0, mxfp4) with a positive control — the standing acceptance criterion the fix commit itself declared out of scope.

**Design**:
- Confirmed still open: `3044936fe`'s own commit message states "Validating the repaired arm is an explicit non-goal here and is tracked as an acceptance criterion on llama.cpp-kpv8"; the dead `hdb` capture at `fused-moe-esimd.hpp:1173` is still unread (cleanup item, lower priority than the validation).
- (a) `fused_moe_q8_0_kernel<0>` — reachable via a temporary guard bypass on a k=256 q8_0 case, scored against the CPU reference. The guard currently deliberately refuses this arm; bypass it only for the validation run, then confirm the guard is restored.
- (b) `fused_moe_mxfp4_kernel<0>` — harder: only reachable under graph recording (`launch_fused_moe_mxfp4_impl<0>` at `:1010`) or an induced Q8_1 scratch-allocation failure (`:1034`). Deliberately establish one of those two conditions for the validation run.
- Both validations need a positive control proving the SCALAR kernel ran, not its correct sibling (dp4a for mxfp4) — the `[MoE FUSED]` launch line naming the specific kernel, per the widened acceptance criterion (c-glwf, supersedes the narrower q8_0-only phrasing in c-8jiw).
- Rationale for why this matters: with the guard retained, no test in the ordinary suite reaches the repaired arm — an unexercised repair is a standing hazard (a generic fallback can be a zero-writer, per the memory entry of the same name), even though the ternary fix itself already landed.

**Bar**: Both repaired `<0>` arms produce numerically correct output (CPU-oracle comparison) under their respective forcing condition, each with a positive-control dispatch line proving the specific kernel variant executed.

**Constraints**: Ruling 8 (small-block dequant fused into the matmul is the intended design here — this validates that the fusion, once forced to the edge-case arm, is still correct).

**Acceptance test**: No existing ctest reaches either repaired `<0>` arm (confirmed: neither is on the ordinary dispatch path, per the design section above), so this is necessarily a manual, monitored, non-subagent run, never a background/looped one (GPU model-loading rule): `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DEBUG=1 GGML_SYCL_MMVQ_ID_VALIDATE=1 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q8_0.gguf -p 0 -n 4 -r 1 -v` as the base vehicle for the q8_0 `<0>` arm (guard bypass required to reach it, add the specific bypass flag/build variant used and record it here once chosen), scored against the CPU reference; for the mxfp4 `<0>` arm, force one of the two reaching conditions (graph recording, or an induced Q8_1 scratch-allocation failure) on a GPT-OSS MXFP4 MoE run and capture the `[MoE FUSED]` launch line as the positive control that the SCALAR kernel (not dp4a) executed. Sample `Shmem`/`MemAvailable` before and ~5s after per the GPU-testing safety rules. Not a permanent ctest addition unless the forcing mechanism is made a standing test hook.

**Depends on**: none — the ternary fix (3044936fe) already landed; this is pure validation.

### llama.cpp-95x2

  - **Id:** llama.cpp-95x2
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Resolve the Q6_K "variable tile" design question — wire the live variable-tile CPU writer to a matching dispatch-side eligibility check, or delete the now-confirmed-unreachable kernel generality.

**Design**:
- Corrected understanding (per the ticket's own self-correction, c-e8hu): there are THREE Q6_K coalesced writers, not one. `reorder_q6_k_aos_to_coalesced_sycl` (GPU) asserts `blocks_per_row % 32 == 0`; `reorder_q6_K_variable_tile` (CPU) has zero call sites outside docs; but `reorder_q6_k_coalesced_cpu` (`ggml-sycl.cpp:46172`) is the **live production writer**, called from the `ggml_backend_tensor_set` path and `:25875`, and it DOES emit correct variable tiles (32+16+8 etc).
- The actual blocker is dispatch-side, not the writer: `ggml_sycl_layout_supports_coalesced` (`common.hpp:4184`) returns `(blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS) == 0` for Q6_K — Q8_0 is explicitly exempted one line above with a padded-tail-tile treatment; Q6_K is not. So even though the writer can lay out 56 blocks as 32+16+8, the eligibility check refuses that tensor before the kernel's own (already-correct) variable-tile decomposition loop ever sees a non-32 tile.
- Two resolutions, owner decision needed: (a) give Q6_K the same padded-tail-tile exemption Q8_0 already has at `common.hpp:4184-4217` — the smallest change that makes the existing generality real, or (b) if non-multiple-of-32 Q6_K is not wanted, delete the kernel's decomposition loops, `tile_count`/`tile_size_at`, and the CPU writer's variable-tile path in favor of the fixed-32-block form.
- The ug4p RCA result (kernel computes the exact answer for the real production layout, not just a model of it) already validates option (a)'s correctness — it is not a new-kernel risk, purely an eligibility-gate change.

**Bar**: Either a shape test at e.g. 56 blocks (32+16+8) exercising `tile_size != 32` passes end-to-end through real dispatch, or the dead generality is removed and `blocks_per_row % 32 == 0` becomes an explicit, documented invariant.

**Constraints**: Ruling 8 (Q6_K stays on standard SYCL dequant, not ESIMD — this ticket doesn't change that, only the tile-eligibility gate).

**Acceptance test**: New `test-backend-ops`-style or dedicated shape test at a non-multiple-of-32 Q6_K block count if (a) is chosen; a removal PR plus updated kernel comments if (b) is chosen.

**Depends on**: none — owner decision needed on (a) vs (b) before implementation starts.

### llama.cpp-19l2

  - **Id:** llama.cpp-19l2
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Guard the split-buffer (`-sm row`) Q8_0 coalesced fp32 dequant call with `src0_full_tensor`, the same missing guard class 95x2/f9fx are auditing in the coalesced-layout family.

**Design**:
- Confirmed current at `ggml-sycl.cpp:~42145-42174` (line numbers drifted from the ticket's stale `41874-41878`): `src0_full_tensor` is computed at `:42145` but the `dkw0_q8_0_coalesced_ptr` branch (`dequantize_row_q8_0_coalesced_to_fp32_rowmajor`) does not check it before dispatching.
- Under `-sm row` split buffers, `row_low`/`row_high` (`:42608-42614` area) make `row_diff < ne[1]` — this reproduces the wrong-scale-plane-offset defect class inside the very fix that closed it for the non-split case, plus an additional row-shift-by-`row_low` error.
- Fix: guard the coalesced branch with `src0_full_tensor`; fall back to the AOS/flat `to_fp32_sycl` path when `row_diff != src0->ne[1]` (partial tensor).
- Not a regression — the old flat path was equally wrong there; this is a pre-existing gap surfaced by review, not something the coalesced work introduced.
- Needs a `-sm row` reproducer (2+ device split) to verify; `-sm row` is unused on this host today, hence low priority, but the fix is cheap and the guard pattern is already established (Q8_0 elsewhere in this family already carries `src0_full_tensor` checks).
- Feasibility caveat: `-sm row` splits a tensor's rows across devices, which needs the split-buffer machinery to move partial results between devices — on this host the two discrete cards have no PCIe P2P (CLAUDE.md, re-verified on the B70 2026-07-31), so whatever cross-device path `-sm row` uses here necessarily host-bounces. Establish that the `-sm row` reproducer actually completes (host-bounce works, or the mode is otherwise infeasible on this topology) before relying on it as this ticket's acceptance vehicle; if it does not run at all on this host, the reproducer needs to move to a review of the split-buffer code path by inspection instead.

**Bar**: A `-sm row` split-buffer Q8_0 dequant with `row_diff != ne[1]` falls back to AOS/flat and produces correct output; the coalesced fast path is untouched for the common (non-split) case.

**Constraints**: Ruling 5 (a layout-specific kernel must not silently run on a shape it doesn't cover — this is exactly that guard, ported to the split-buffer case).

**Acceptance test**: First confirm feasibility: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q8_0.gguf -sm row -p 'test' -n 4` completes at all (given no PCIe P2P, this host-bounces; confirm it does not hang or refuse). If it runs, compare split-buffer Q8_0 output against the non-split reference for the specific `row_diff != ne[1]` case; `test-backend-ops` Q8_0 cases unaffected on this single-effective-device (non-split) host. If `-sm row` proves infeasible on this topology, fall back to a code-inspection-based fix with a targeted unit test constructing the `row_diff != ne[1]` shape directly.

**Depends on**: none — independently fixable; low priority since `-sm row` is not exercised on this host.

### llama.cpp-g4i8

  - **Id:** llama.cpp-g4i8
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Identify the six repeating `name_hash` tensors behind the `direct_stage_weight` rejection noise (48+ ERROR-level lines per `test-llama-archs` run) and demote or resolve.

**Design**:
- Confirmed still reproducing: `git log --grep g4i8` empty, no commit addresses it. Six distinct `name_hash` values repeat per graph (not 48 distinct tensors) — `0x2c2c03c5845a4428`, `0x474cbb7ffa4be3cb`, `0x9436376037e599f5`, `0x2f51d65e2e10e7e6`, `0x5c5ab6125ec1b4e2`, `0xe55e3ca93cf2a5d1` — all `type=0` (F32), `ne=[1,2,1,1]` (two elements), all `graph_active=1`.
- Not the qwen3next MoE abort (a different defect, see llama.cpp-zozu) — this is confirmed non-fatal, both affected archs (`llama`, `qwen3moe`) pass with correct numerics.
- Step 1: resolve each `name_hash` to a tensor name — find where the hash is computed and either reverse it from the model's tensor list, or add a temporary debug print behind an existing debug flag.
- Step 2: determine whether the rejection is *correct*. `graph_active=1` suggests mid-graph materialization is being requested, which planned mode legitimately refuses — if so, the real question moves upstream: why does this tensor reach `direct_stage_weight` at all instead of being resolved during load planning (a placement/classification question, not a cache-rejection question).
- Step 3: if rejection is correct and harmless, demote the log level (ERROR → DEBUG/WARN with rate-limiting) rather than leaving dozens of red lines in a passing run — that noise actively obscured the qwen3next diagnosis once already.
- Never loop `test-llama-archs` (two global OOMs, 2026-07-30); one run is ~34s, check `Shmem`/`MemAvailable` first per CLAUDE.md.

**Bar**: The six tensors are named; either they are shown correct-and-demoted-to-non-ERROR, or the upstream classification gap that routes them to `direct_stage_weight` is fixed so they never reach the rejection path.

**Constraints**: Ruling 1 (this is a unified-cache rejection path; any fix stays inside unified-cache.cpp's authority over materialization decisions).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a llama` (single run, manual, monitored) shows either zero `direct_stage_weight` rejections for named/expected tensors, or the same rejections at a demoted log level with tensor names attached.

**Depends on**: none — independently diagnosable.

### llama.cpp-s2cw

  - **Id:** llama.cpp-s2cw
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Verify attention SOA weights actually get pinned in VRAM when sourced from host-pinned memory, re-scoped against the current (post-2026-08-16) canonical memory contract.

**Design**:
- Confirmed all three functions this ticket names still exist and are exercised by current tests: `ggml_sycl_get_weight_layout_ptr`, `ggml_sycl_is_host_resident_weight` (`mmvq.cpp:2233`), `graph_preload_weights` (`common.hpp:1857` area).
- Original ticket (2026-03-17) predates the placement-decides-executor (2026-08-16) and layout-follows-residency (2026-08-16/17) rulings — re-scope the acceptance criteria against `docs/design/sycl-canonical-memory-architecture.md` before touching the specific fix candidate below.
- Original potential fix noted: `graph_preload_weights` (`:22354`-area) checks `ggml_backend_buffer_is_host()`, which may miss SYCL host-pinned buffers specifically — may need an `ggml_sycl_is_host_resident_weight()` check added at that gate instead of (or alongside) the generic host check.
- Expected outcome (from the original plan doc, Task 2): ~630 MB of attention SOA pinned in VRAM after model load.
- Under ruling 4 (placement decides the executor), verify this doesn't conflict — a host-pinned attention weight legitimately executing on CPU is correct; the question here is specifically whether SOA *materialization* pins correctly, not whether it should always live in VRAM.

**Bar**: A model load with attention weights sourced host-pinned shows ~630 MB of attention SOA resident in VRAM (measured via the cache's own residency accounting, not an external tool), and `graph_preload_weights`'s host-buffer check does not silently skip SYCL host-pinned buffers.

**Constraints**: Ruling 4 (placement decides the executor — this is a placement-correctness verification, not a new placement policy), ruling 1 (pinning happens through unified-cache APIs).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DEBUG=1 ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3,' -n 4 --seed 42 --temp 0` (single manual run; model load alone is enough to observe residency, generation is incidental), grepping the `[UNIFIED-CACHE]` residency lines for attention SOA tensor count/size in VRAM, confirming it matches the ~630 MB expectation.

**Depends on**: none — independently verifiable, but should be read against the current canonical contract doc before any fix, since the ticket predates it.

### llama.cpp-6sp4

  - **Id:** llama.cpp-6sp4
  - **Action:** keep
  - **Priority:** 4
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Give the test-only layout override a fixture that materializes matching layout info first, so it can actually bind on the device path (currently refused for fresh weights).

**Design**:
- Confirmed current: `ggml_sycl_can_use_layout_for_kernel` still gates override admission at multiple sites (`ggml-sycl.cpp:32074,57604,57717,57740,58046,58101`, current HEAD) — every production consumer admits the test override only behind this check.
- The 0tkh run's pre-registered finding still holds: `dev/auto` and `dev/soa` both resolve to COALESCED with no override honored for a fresh weight, because a fresh weight lacks layout info and `ggml-sycl.cpp:53014`-area refuses a non-AOS override without matching layout info.
- Fix: before applying the `ggml-sycl-test.hpp` override in a test fixture, first materialize the weight with matching layout info via the cache materialization helpers (the same helpers production code uses), then apply the override. This makes `dev/coalesced` actually exercise the override path instead of silently measuring the default policy.
- Not a product bug — purely a test-infrastructure gap. Low priority unless a future kernel-selection test needs the override to bind for real.

**Bar**: A device-path test using the layout override with a pre-materialized fixture shows the override actually taking effect (dispatched kernel matches the overridden layout, not the default-policy layout).

**Constraints**: Ruling 1 (the fixture's pre-materialization step must go through the same unified-cache APIs production code uses, not a shortcut).

**Acceptance test**: New or modified test fixture in the layout-override test suite (run with `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` to exclude the iGPU OOM hazard, since this exercises the device path) demonstrating override binding on the device path; existing override tests for the non-device (or already-materialized) path unaffected.

**Depends on**: none — self-contained test-infrastructure fix, low priority.

### llama.cpp-cyn.2

  - **Id:** llama.cpp-cyn.2
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Re-verify against the current (substantially reworked) XMX/MXFP4 tiled area whether the original OOM (SoA buffer not freed after tiled conversion) still reproduces, before continuing to carry this as a P0.

**Design**:
- Parent epic `llama.cpp-cyn` closed 2026-01-27 as "superseded by llama.cpp-xihy"; the XMX/MXFP4 tiled area has since been substantially reworked (`5a623248b`, 2026-08-21, XMX_TILED-to-WOQ repack under the active Option T track; `c7b805740`, the same track's dequant-deletion commit that closed llama.cpp-71hx).
- The ticket's rewritten (2026-08-01) acceptance criteria is generic architecture-alignment boilerplate ("device/host decisions use authoritative placement and handle resolution... unresolved device-planned work fails explicitly") — not falsifiable as written.
- The REOPENED note ("XMX tiled path is NOT executing... ESIMD fallback is running") is from before the Option T rework; re-run the original repro against current HEAD first, since the whole tiled-conversion code path this bug lived in may no longer exist in the form the notes describe.
- If the OOM genuinely still reproduces: locate the current tiled-conversion path's SoA buffer lifetime, confirm it is a `mem_handle`-owned allocation (ruling 2), and check for a missing release rather than adding a forced-eviction workaround (ruling on "no forced eviction to reclaim memory with a live handle").
- If it does not reproduce (most likely, given the scale of the rework since January): close with the current-HEAD non-repro evidence and a note pointing to the superseding work (5a623248b / c7b805740 / Option T track).

**Bar**: Either a current-HEAD repro of the original OOM symptom (with a concrete `mem_handle` leak identified), or a documented non-repro closing this ticket.

**Constraints**: Ruling 2 (mem_handle is the sole ownership token — any real fix here is "find the missing release", never a forced eviction).

**Acceptance test**: The original ticket's repro is not recoverable from the current record, so re-derive it concretely rather than deferring the name: `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DEBUG=1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 4 -r 1 -v`, sampling `Shmem`/`MemAvailable` before load and ~5s after (per CLAUDE.md's post-run-settle guidance) — this is the MXFP4/XMX-tiled load path the ticket's symptom describes. If `Shmem` stays flat (no runaway growth) and load completes, treat as non-repro and close citing 5a623248b/c7b805740/Option T as the superseding rework; only escalate to a live-handle-leak investigation if `Shmem` does climb unexpectedly during this single run. Single run only — do not loop (never-loop rule).

**Depends on**: none — independently re-verifiable, should be done early given how stale (Jan 2026) the underlying code area assumption is.

### llama.cpp-n36

  - **Id:** llama.cpp-n36
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Re-verify whether the load-time tiled-conversion pipeline (originally "Task 1") is already served by the production `scoped_planned_materialization` path; close or narrow to the actual remaining gap.

**Design**:
- Confirmed: `unified-cache.hpp:3800`'s `scoped_planned_materialization` (query-driven planned materialization) already exists and is used in production weight-registration paths (`tests/test-sycl-moe-handle-resolution.cpp`, `test-sycl-reset-*` suite) — the general architecture this ticket asks for (query-driven planner chooses representations before materialization, unified_cache performs the canonical conversion, mem_handles retain layout/residency/readiness/ownership) is largely in place.
- The 2026-08-02 rewritten acceptance criteria ("load conversion has no loader-local placement authority or raw-pointer cache, executor consumption is through the emitted handle contract") is generic boilerplate, not a falsifiable test against a specific remaining gap.
- Original scope (per the "Historical context" section this ticket still carries) was specifically XMX-tiled MoE load-time conversion — narrow the re-verification to that specific case: does a fresh XMX-tiled MoE weight get its layout conversion planned and executed through `scoped_planned_materialization`, or does some loader-local path still bypass it?
- Depends on `llama.cpp-cyn` (closed, superseded) — the original dependency chain is stale; re-derive what (if anything) remains open in this specific area rather than trusting the dependency edge.

**Bar**: Either a concrete remaining gap is identified (a specific loader path that still bypasses `scoped_planned_materialization` for XMX-tiled MoE weights) with a file:line citation and a test proving it, or the ticket closes citing the existing production usage as satisfying the original scope.

**Constraints**: Ruling 1 (unified cache is the sole allocator — any remaining gap is by definition a path that doesn't yet go through it).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '^test-sycl-moe-xmx-tiled-materialization$' --output-on-failure` (confirmed registered in `build/ggml/src/ggml-sycl/CTestTestfile.cmake`) as the starting point — read what it already covers first; if it already exercises fresh XMX-tiled MoE weight conversion through `scoped_planned_materialization` end-to-end, that satisfies this ticket's re-verification directly. If it covers only a narrower slice, extend it (or add a sibling test) for the specific gap found, or close citing the existing coverage as satisfying the original scope if none is found.

**Depends on**: none directly (llama.cpp-cyn is closed); should be re-verified together with llama.cpp-cyn.2 since both are stale Jan-2026 tickets in the same code area.

### llama.cpp-omp4

  - **Id:** llama.cpp-omp4
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Umbrella/tracking ticket for the deferred Q1/NVFP4 device-decode program plus P6 router restoration — execute and certify on post-merge master now that the blocking merge has landed.

**On being an epic-shaped member inside this epic (triage 2026-09-01 note)**: `llama.cpp-omp4` is the pre-existing external tracker ticket the Q1/NVFP4 program is filed under, predating this design epic. It is retained here as a bookkeeping/aggregation node, not as a second nested epic with its own independent bar — its 7 children (sgox, zqoe, 5i7z, 7vpd, tjk4, h690, o54r) are already separately tracked as members of this same 28-member epic with their own bars, so omp4 adds no work of its own; closing it is definitionally closing all 7. Absorbing it entirely (deleting it as a member) was considered and rejected: it is the ticket the pre-existing external Q1/NVFP4 program history is attached to, and dropping the reference would orphan that history. Treat "no member carries action 'merge'" as expected, not a gap — this epic has no merge-shaped member; omp4's aggregation role is the closest thing to it and is explicitly not a merge.

**Design**:
- All 7 absorbed packages (sgox, zqoe, 7vpd, 5i7z, o54r, h690, tjk4) remain open — this epic's "done" state is entirely a function of its children closing, not independent work at this level.
- Merge landed (`1ebfa4e4a`), so per the epic's own text ("done means: the deferred decode program is executed and certified on post-merge master") this epic is now unblocked and its children are no longer deferred-and-inert.
- Sub-track order per this design epic's ordering hint: `omp4` (this) → `sgox` → `zqoe` → `5i7z`/`7vpd` (parallel) → `tjk4` (B70 validation, the shipping gate). `h690`/`o54r` (P6 router) form an independent chain under the same umbrella.
- The shipping capability stays fail-closed by owner ruling until `tjk4`'s certification passes — do not flip the flag based on any individual child's local test passing.

**Bar**: All 7 children closed with their own bars met; `tjk4`'s private production-route CTest suite green serially on B70; Mistral and GPT-OSS gates green with the capability enabled.

**Constraints**: The 2026-08-14 owner ruling itself: "capability stays fail-closed; landing = merge into master + push" — this epic's existence does not change that default.

**Acceptance test**: Aggregate of all 7 children's acceptance tests; no independent test at this level.

**Depends on**: none (merge prerequisite already satisfied); gates sgox/zqoe/5i7z/7vpd/tjk4/h690/o54r as their parent.

### llama.cpp-sgox

  - **Id:** llama.cpp-sgox
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-omp4

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Wire primary Q1_0/NVFP4 direct-AoS device decode, re-enabling decode-then-prompt execution only after canonical allocation handles and planner-owned workspace prerequisites land.

**Design**:
- Confirmed real: `GGML_TYPE_Q1_0`/`GGML_TYPE_NVFP4` are real ggml types (`ggml.h:437-489`); real SYCL scaffolding exists behind `GGML_SYCL_Q1_NVFP4_ROUTE_TESTING` (`ggml-sycl.cpp:81,10987` etc).
- Correctly blocked, not stuck for lack of spec (per design_gaps): depends on `llama.cpp-s0h5`, `llama.cpp-hbyo` (both closed) and `llama.cpp-tjk4` (open — the final certification gate, which is downstream in the actual execution order despite the raw dependency edge pointing at it; treat `tjk4` as validating this work, not blocking its start).
- Primary/secondary owner queues, immutable recipe once admitted, terminal retention of handles — this is the standard `mem_handle`-owned pattern this fork already uses elsewhere; no new ownership model needed, just applied to the two new types.
- No weight migration — Q1/NVFP4 weights stay in their materialized AoS layout; this ticket wires *decode* kernels to consume them, not a repack.
- Web finding: upstream ggml-org discussion #22042 flags NVFP4 support risk from "unclear separation of concerns" — keep the primary/secondary queue separation and immutable-recipe design explicit and tested, since that discipline is exactly what avoids the class of bug being flagged upstream.
- **New pre-work item (found during this triage's re-review, corrects an earlier epic claim)**: `GGML_TYPE_NVFP4 = 40` and `GGML_TYPE_Q1_0 = 41` are present in upstream ggml-org/llama.cpp master TODAY (verified live against upstream's `ggml.h`, same values as this fork's) — the earlier "these types are fork-local, nothing to reconcile against upstream" premise is WRONG and this ticket should not proceed on it unexamined. Before wiring further device-decode kernels: check whether upstream's SYCL backend already ships MUL_MAT/MUL_MAT_ID coverage for either type (this session's search suggested NVFP4 gained CUDA/SYCL/Vulkan paths upstream by April 2026 but could not confirm via authenticated code search) and whether upstream's block/scale byte layout for these types matches what this fork's decode kernels assume — a shared enum value does not guarantee a shared physical format. If upstream already has usable SYCL kernels, port/adapt rather than hand-write; if the formats disagree, document that explicitly since it changes what "compatible with upstream" means for this fork's GGUFs.

**Bar**: B70/B50 numeric tests pass (CPU-oracle comparison) for primary decode; no weight migration observed; terminal handle retention verified (no leaked leases at teardown).

**Constraints**: Ruling 1 (unified-cache allocation for any new workspace), ruling 2 (mem_handle owns the recipe/decode state).

**Acceptance test**: Whatever private ctest regex `tjk4` names for primary decode (per tjk4's own design_gaps, this needs to be pinned down explicitly) — coordinate with tjk4's addendum for the exact command.

**Depends on**: llama.cpp-omp4. Note: sgox's own ticket text lists tjk4 as a dependency, but per the epic's ordering_hint tjk4 is actually downstream validation — treat sgox as startable now, tjk4 as its certification gate.

### llama.cpp-zqoe

  - **Id:** llama.cpp-zqoe
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-sgox

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Stage 2 — wire the existing Q1_0/NVFP4 device vec-dot primitives into direct AoS MMVQ and MMVQ-ID dispatch for primary/secondary owning queues, then bounded prompt chunks.

**Design**:
- Confirmed real and actively landing: `df17991c6` "add Q1 and NVFP4 direct AoS MMVQ primitives" (2026-08-11), `ac744e4d0` "wire authoritative Q1 NVFP4 decode" (2026-08-12), `186348705` post-merge type-gate fix (2026-08-26); `tests/test-q1-nvfp4-*` suite exists.
- Consume immutable retained execution recipes/tables from sgox — no weight migration or FP16 materialization; this stays a direct-AoS-decode path throughout.
- Add CPU-oracle GPU tests for the secondary/mixed/repeated-ID matrix, ready-event retention, memory/transfer and performance gates, per the ticket's own acceptance list.
- Dependencies riding along (`llama.cpp-vpjy`, `llama.cpp-ihkf`) are already closed per omp4's rollup — no longer live blockers.

**Bar**: Primary and secondary MMVQ-ID dispatch for Q1_0/NVFP4 pass the CPU-oracle matrix (primary, secondary, mixed, repeated-ID); no weight migration observed; memory/transfer gate shows no unexpected host-device traffic.

**Constraints**: Ruling 4 (placement decides the executor — no weight streaming per dispatch; this is direct-AoS decode of already-placed weights, consistent with the ruling).

**Acceptance test**: `tests/test-q1-nvfp4-*` suite (name the exact ctest regex from the current test file); B70/B50 CPU-oracle comparison run.

**Depends on**: llama.cpp-sgox (consumes sgox's primary decode wiring and canonical handles).

### llama.cpp-5i7z

  - **Id:** llama.cpp-5i7z
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-zqoe

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Implement bounded Q1/NVFP4 prompt chunk execution from planner-owned workspace bundles, after primary decode closure.

**Design**:
- Confirmed real feature program with extensive existing test coverage (`test-sycl-moe-resolved-batch.cpp`, `test-sycl-moe-handle-resolution.cpp`, `test-sycl-supports-op-indexed-moe-source.py`) — not a hallucinated ticket, per design_gaps evidence.
- Execute prompt chunks from planner-owned workspace bundles on primary/secondary queues with occurrence descriptors; final terminal retention of all handles; no allocation or weight transfer mid-chunk.
- Ticket body itself is thin (one paragraph, no inline acceptance test or file pointers) — the real task shape lives with `tjk4`/`bwmz`'s definitions; pull the concrete bar from there rather than re-deriving from this ticket alone.
- Add hardware oracle/performance tests once the primary decode gate (zqoe) is stable.

**Bar**: Bounded prompt chunks execute correctly (CPU-oracle comparison) with no allocation/weight-transfer events observed mid-chunk; terminal handle retention verified.

**Constraints**: Ruling 1 (planner-owned workspace bundles are unified-cache allocations), ruling 4 (no per-dispatch weight streaming).

**Acceptance test**: Same private-route CTest family tjk4 certifies against — `ONEAPI_DEVICE_SELECTOR=level_zero:0 ctest --test-dir build -R 'q1-nvfp4' --output-on-failure` (currently `sycl-q1-nvfp4-adapter-device`, `sycl-q1-nvfp4-admitted-device`; re-confirm the current set with `ctest --test-dir build -N | grep q1-nvfp4` since sgox/zqoe land first and may add more). If bounded-prompt-chunk coverage is not yet in that family when this ticket starts, add it there rather than inventing a separate target.

**Depends on**: llama.cpp-zqoe (primary decode closure is this ticket's stated precondition).

### llama.cpp-7vpd

  - **Id:** llama.cpp-7vpd
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-zqoe

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Implement secondary Q1/NVFP4 owner-queue decode (B70→B50), after primary decode gate passes on B70.

**Design**:
- Same omp4 epic, same primary-decode-closure precondition as 5i7z; real feature program, not hallucinated (design_gaps evidence: same Q1_0/NVFP4 test suite coverage).
- Wire B70→B50 secondary decode using planner-owned descriptor/activation/output bounce slices; exact queue authority; no weight movement/rings; terminal/quarantine ownership retained.
- **Directly implicates the no-P2P B70/B50 topology** (CLAUDE.md, re-verified on B70 2026-07-31): there is no direct PCIe P2P between the two discrete cards — different CPU root ports, no shared PCIe switch, kernel-level refusal confirmed. The "bounce slices" language in the ticket's own description is already consistent with this — this must host-bounce, never attempt direct device-to-device copy.
- Web finding: an independent vLLM issue (#41663) reports a GP fault/BCS engine reset on dual-B70 tensor-parallel Level Zero use — corroborates treating cross-device paths here as fragile; test the host-bounce path explicitly rather than assuming it "just works" because it avoids P2P.
- `ext_oneapi_enable_peer_access()` returns OK on this hardware despite no real P2P (documented false-positive in CLAUDE.md) — do not use it as a capability check anywhere in this path; use `can_access_peer` if any capability check is needed at all, though the design should not need one since it's host-bouncing by design.

**Bar**: B70→B50 secondary decode produces correct output (CPU-oracle comparison) via host-bounce, with measured bandwidth/latency consistent with a host round-trip, not a (nonexistent) direct P2P path.

**Constraints**: The no-P2P topology fact from CLAUDE.md (not a numbered ruling, but a hard architectural constraint) — this whole ticket exists inside that constraint.

**Acceptance test**: A B70→B50 secondary-decode CPU-oracle test; explicit check that no direct peer-copy API is invoked (grep the implementation for `enable_peer_access`/direct device-to-device copy calls and confirm none appear on this path).

**Depends on**: llama.cpp-zqoe (primary decode gate must pass B70 first, per the ticket's own precondition).

### llama.cpp-tjk4

  - **Id:** llama.cpp-tjk4
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-5i7z
  - llama.cpp-7vpd

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Post-reboot B70 certification of the private Q1/NVFP4 production route and adapter CTests — the program's shipping gate.

**Design**:
- Still a live dependency of open epic omp4; the cache_backing_token P0 forgery blocker cited in the ticket's own comment history (c-vuv6) is fixed (`0b7b49e07`, unforgeable-by-compiler token, verified by tests referenced in that merge) — one of the historical blockers is closed, re-check the others (model-key aliasing, async prestage source lifetime, pointer-table fixture) against current HEAD before re-running certification.
- Description names no concrete ctest/binary — before restarting certification, (1) confirm pointer-table fixture status post-merge, (2) name the exact private-route ctest regex from current test files, (3) re-derive which of the three historical blockers (P0 model-key aliasing, async prestage source lifetime, pointer-table fixture red) still apply post-b10630-merge.
- Requirements per the ticket: lifecycle commit, Q1/NVFP4 CPU oracle, candidate/admit/submit/terminal/recycle counters, same-slot generation reuse, injected pre/post/async failure cleanup, no seam exports, no hangs.
- Run serially on B70, per the GPU-serialization rule in CLAUDE.md — this is model-loading/GPU work, so it belongs to the lead session, never a subagent.
- This is the shipping gate: only once this passes should the fail-closed capability flag be reconsidered, per the omp4 epic's own "done means" text.

**Bar**: Private production-route CTest suite green serially on B70: all counters correct, no hangs, no seam exports, CPU-oracle comparisons pass across the injected-failure matrix.

**Constraints**: The 2026-08-14 owner ruling — capability stays fail-closed until this certification passes; do not flip the flag preemptively.

**Acceptance test**: `build/bin` confirms two registered private-route CTests exist today: `sycl-q1-nvfp4-adapter-device` and `sycl-q1-nvfp4-admitted-device` (`ctest --test-dir build -N | grep q1-nvfp4` to re-confirm at run time — more may exist by the time sgox/zqoe/5i7z/7vpd land and should be swept in). Run `ONEAPI_DEVICE_SELECTOR=level_zero:0 ctest --test-dir build -R 'q1-nvfp4' --output-on-failure` serially on B70 by the lead session, with `Shmem`/`MemAvailable` sampled before/after per the GPU-testing safety rules, plus the Mistral and GPT-OSS chat gates.

**Depends on**: llama.cpp-5i7z, llama.cpp-7vpd (both feed the production route this certifies); functions as the whole sub-track's closing gate.

### llama.cpp-h690

  - **Id:** llama.cpp-h690
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-omp4

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Restore CPU-TG/pair, GPU pair, secondary pair, sorted XMX, and MMVQ decode fast paths over retained batches, after the llama.cpp-zx6x correctness base.

**Design**:
- Deps `llama.cpp-igi0` and `llama.cpp-o54r`'s own deps (`yej4`) are closed; parent epic omp4 is open and active, listing h690 as a live absorbed package.
- Consume the already-admitted `moe_resolved_batch` only — no independent ID resolution, raw pointer identity, `from_chunk_ptr`, or second admission. Pointer-table ABI payloads derive from exact operand handles; ready events and owner queues are preserved; same handles retained through terminal event/fused state.
- Run the numeric primary/secondary/mixed/repeated-ID matrix plus performance gates before closure, per the ticket's own acceptance criteria — well-specified already, per design_gaps.
- Missing piece per design_gaps: a concrete post-b10630-merge baseline to gate against (`docs/backend/sycl-perf-baselines.md`) and confirmation the retained-batch admission it depends on (igi0) still holds its contract post-merge — verify both before claiming the perf gate passed.

**Bar**: Numeric matrix (primary/secondary/mixed/repeated-ID) passes; PP/TG perf on Mistral and GPT-OSS matches or exceeds `docs/backend/sycl-perf-baselines.md` (B70 ~2495/~108 Mistral, ~1415/~44 GPT-OSS; B50 ~1188/~47 Mistral, ~894/~32 GPT-OSS) via interleaved paired A/B against pre-change HEAD.

**Constraints**: Ruling 2 (mem_handle-derived pointer-table payloads, no raw-pointer identity), ruling 9 (perf gate against the baselines doc, not a remembered figure).

**Acceptance test**: No single existing ctest is confirmed to already cover the full primary/secondary/mixed/repeated-ID matrix for this fast-path restoration; the closest current candidates to build on or extend are `test-sycl-moe-resolved-batch` and `test-sycl-moe-handle-resolution` (both registered in `build/ggml/src/ggml-sycl/CTestTestfile.cmake`) — confirm at implementation start whether either already carries this matrix or needs extending, and pin the actual ctest name(s) used before claiming this bar met. Run with `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`, plus `llama-bench` Mistral/GPT-OSS PP512/TG128 paired against pre-change HEAD.

**Depends on**: llama.cpp-omp4 (parent epic, now unblocked); llama.cpp-o54r per the epic's ordering hint (P6 router restoration should land together before recipe-gap tickets like 0yi9/wh7o/a6sy are judged).

### llama.cpp-o54r

  - **Id:** llama.cpp-o54r
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-omp4

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Restore prompt gate/up/down fusion with retained role batches, after llama.cpp-igi0.

**Design**:
- Both deps (`llama.cpp-igi0`, `llama.cpp-yej4`) closed — transactional fusion foundation and retained prompt batch admission both landed. Parent epic omp4 open, this was deferred here per the 2026-08-14 owner ruling ("hardening-first" merge).
- Admit occurrence-preserving gate/up/down batches from one immutable ID snapshot; each role owns independent handles/layouts/readiness. Pair/fusion executors and transient tables consume exact role batches; terminal event retains all role/table/intermediate handles. Skip publication is transactional after successful submission — no post-admission resolver/materialization/raw ownership.
- Well-specified acceptance already (host/primary/secondary/mixed/repeated-ID/failure/numeric/perf coverage) per design_gaps — needs a claim/lease before work resumes; lease was reset 2026-08-14, no active agent currently holds it.

**Bar**: Full coverage matrix (host/primary/secondary/mixed/repeated-ID/failure/numeric) passes; prompt-path PP performance matches or exceeds baselines (same table as h690's addendum).

**Constraints**: Ruling 2 (each role's handles/layouts/readiness independent — no shared mutable state across roles that would violate mem_handle ownership), ruling 6 (no host waits — the terminal-event retention design must stay event-chained, not drain-per-role).

**Acceptance test**: Same caveat as h690: no single existing ctest is confirmed to already cover the full host/primary/secondary/mixed/repeated-ID/failure matrix for prompt gate/up/down fusion; the closest current candidates are the ctest-registered `sycl-moe-fused-transaction` (binary `test-sycl-moe-fused-transaction`) and `test-sycl-moe-resolved-batch` (both registered in `build/ggml/src/ggml-sycl/CTestTestfile.cmake`) — confirm at implementation start whether either already carries this matrix or needs extending, and pin the actual ctest name(s) used before claiming this bar met. Run with `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`, plus a PP baseline check against `docs/backend/sycl-perf-baselines.md`.

**Depends on**: llama.cpp-omp4; per the epic's ordering hint, should land alongside llama.cpp-h690 before the recipe-gap tickets (0yi9/wh7o/a6sy) are judged.

## New tasks

### SYCL layout census tool: enumerate materialized layout per weight and flag route/storage disagreement

  - **Title:** SYCL layout census tool: enumerate materialized layout per weight and flag route/storage disagreement
  - **Priority:** 1
### Depends on

  - llama.cpp-yitq
  - llama.cpp-avhx

  - **Description:** ## Design addendum (triage 2026-09-01)

**Intent**: Build the instrument this epic's own closing bar requires and that does not currently exist — a census showing, for a full model load plus one PP and one TG step, how many logical weights are materialized in more than one layout and how many dispatch decisions disagreed with actual storage.

**Design**:
- No existing tool: `search_text` for `layout_census`/`LAYOUT-CENSUS`/`layout_census` across the whole repo returns zero matches — this needs building from scratch, not extending something.
- Add a debug-gated pass (behind an existing debug flag pattern, e.g. `GGML_SYCL_LAYOUT_CENSUS=1`) that walks the unified cache's live entries after a load + one PP + one TG step on GPT-OSS 20B, and for each logical tensor id reports: materialized layout(s) (flag if >1), and cross-references every `[MOE-DISPATCH-LAYOUT]` line emitted during that run for `route_layout != src0_layout`.
- Output a single summary line machine-parseable for CI/gate use: `[LAYOUT-CENSUS] tensors=N multi_layout=0 route_mismatches=0` (or nonzero, naming the offenders).
- This is the concrete instrument avhx/34g5/8hz8/tqka/l8at's fixes should be scored against, and the epic's "layout census at HEAD shows zero weights materialized in two layouts" bar has no other way to be checked today.
- Allocate nothing new outside the unified cache (ruling 1) — this walks existing cache state, it does not materialize anything itself.

**Bar**: The tool runs on a GPT-OSS 20B load + PP + TG cycle and reports `multi_layout=0 route_mismatches=0` on current HEAD (post avhx/yitq), or names the specific offenders if not yet zero.

**Constraints**: Ruling 5 (this is the direct instrument for that ruling), ruling 1 (read-only walk of unified-cache state, no new allocation).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_LAYOUT_CENSUS=1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 4 -r 1 -v` (or an equivalent llama-cli/completion run) followed by grepping the summary line.

**Depends on**: llama.cpp-yitq, llama.cpp-avhx (the tool is most useful, and its zero-mismatch bar is only achievable, after those land — but the tool itself can be built in parallel).

## Web findings

### Item 1

  - **Claim:** Upstream ggml-org/llama.cpp's SYCL backend already ships generic (non-MoE) reorder/MUL_MAT support for the full IQ family (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS) — this fork's own docs/backend/SYCL.md:71 already lists them — but no MUL_MAT_ID (_id/MoE-indexed) variant for any IQ type was found upstream, confirming llama.cpp-wh7o's claim that the 9 bespoke iq kernels have no transcribable _id sibling anywhere.
  - **Url:** https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
  - **Date:** 2026 (docs, ongoing)
  - **Impact:** confirms wh7o is genuine new-kernel work, not a port; do not resize it down expecting an upstream shortcut

### Item 2

  - **Claim:** Release b9664 adds SYCL support for reordered Q4_K/Q5_K/Q6_K MoE MUL_MAT_ID (PR #24452), extending the existing SYCL MoE mul_mat_id reorder path to Q6_K expert weights and completing reordered MoE coverage for mixed K-quant MoE models whose down-projection experts are Q6_K.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/24452
  - **Date:** 2026
  - **Impact:** orthogonal to this epic's float/iq gap (Q6_K already covered in this fork per gx30), but confirms the reorder-path extension pattern that f9fx/95x2 are auditing is the same one upstream keeps extending — worth mirroring upstream's eligibility-check shape if 95x2 gives Q6_K the padded-tail-tile treatment

### Item 3

  - **Claim:** oneDNN 2026.x documents ONEDNN_EXPERIMENTAL_GROUPED_MEMORY grouped-GEMM support intended for MoE token routing, and Intel's oneDNN 2026 release notes claim improved Xe2/Xe3 matmul performance including float16-with-low-precision-weight paths.
  - **Url:** https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html
  - **Date:** 2026
  - **Impact:** directly actionable for llama.cpp-0yi9 — evaluate oneDNN grouped GEMM before hand-writing a new dense per-expert GEMM kernel for F16/F32/BF16 MMID; this fork already delegates comparable work (SDPA, batched-PP-WOQ) to oneDNN rather than bespoke device code

### Item 4

  - **Claim:** An upstream ggml-org discussion (#22042) states current NVFP4 support carries risk of functional incorrectness due to unclear separation of concerns.
  - **Url:** https://github.com/ggml-org/llama.cpp/discussions/22042
  - **Date:** 2026
  - **Impact:** corroborates why sgox/zqoe/tjk4's certification bar (CPU-oracle comparison at every stage, no seam exports, fail-closed default) is proportionate rather than excessive for the Q1/NVFP4 device-decode program

### Item 5

  - **Claim:** A vLLM issue (#41663) reports a GP fault and BCS engine reset on dual Intel Arc Pro B70 tensor-parallel (XPU TP=2) Level Zero use.
  - **Url:** https://github.com/vllm-project/vllm/issues/41663
  - **Date:** 2026
  - **Impact:** independent (different project, same driver/hardware stack) corroboration of this fork's documented no-PCIe-P2P finding between the two discrete cards; reinforces that llama.cpp-7vpd's B70->B50 secondary decode must host-bounce and must not assume a shared Level Zero context

### Item 6

  - **Claim:** CORRECTION to this epic's earlier claim: `GGML_TYPE_NVFP4 = 40` and `GGML_TYPE_Q1_0 = 41` ARE present in upstream ggml-org/llama.cpp master (verified live 2026-09-01 against raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h — same enum values as this fork's ggml.h:443-444). NVFP4 merged upstream via a PR series March-April 2026 with CUDA, SYCL, and Vulkan paths reported (search-engine summary, not independently confirmed by file read); Q1_0's upstream kernel coverage was not established by search (GitHub code search requires authentication this session lacked, so 'no SYCL MUL_MAT_ID kernel for these types was found' is a statement about what the session could search, not a confirmed negative).
  - **Url:** https://github.com/ggml-org/llama.cpp/discussions/23853
  - **Date:** 2026
  - **Impact:** REVERSES the prior claim that these types are fork-original with nothing to reconcile against upstream. The type IDs are now shared with upstream, so before further sgox/zqoe/tjk4 work, someone with authenticated GitHub code search (or a local upstream checkout) must check whether upstream's SYCL backend has landed any Q1_0/NVFP4 MUL_MAT or MUL_MAT_ID kernels that could be ported instead of hand-written, and whether upstream's enum semantics (block layout, scale format) match this fork's usage byte-for-byte before assuming type-ID equality implies format equality. This is now an open question this epic surfaced, not one it answered — sgox is the ticket that should absorb it before wiring proceeds on the old 'definitely original work' assumption.