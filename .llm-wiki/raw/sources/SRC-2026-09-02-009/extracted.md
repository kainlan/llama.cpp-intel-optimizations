# GPT-OSS decode at 80% of the memory-bandwidth roofline

- **Slug:** gptoss-decode-bandwidth
- **Title:** GPT-OSS decode at 80% of the memory-bandwidth roofline
- **Epic ticket:** llama.cpp-30ak7
- **Priority:** 0
- **Description:** ## Goal
Single-stream token generation (batch=1 decode) for GPT-OSS 20B MXFP4 must be limited by
DRAM bandwidth, not by kernel-class mismatch, host round-trips, or per-token layout churn.
Every decode dispatch should read each needed weight byte once, from the layout the planner
already materialized, over a graph that replays instead of re-recording. This epic owns the
MMVQ/XMX decode routers, the device-ID MoE decode route that a 2026-08-11 commit deleted,
phase-owned XMX-tiled decode for gate/up/down, grouped/batched decode formation, persistent-TG
graph replay, and the PERF-EPIC B/E recovery tracks that gate the eventual default flip.

## Why now
d0bp's 2026-08-21 clean-protocol A/B proved there is no structural pp/tg tradeoff: the
79ae63559 baseline worktree hit BOTH best pp512 (908.9) and best tg128 (35.24) with one SOA
layout. The regression since then is a kernel-variant-selection defect (DPAS decode admission
falsified by policy plumbing), not a fundamental bandwidth ceiling — so there is real headroom
to recover before this epic even reaches its 80%-of-roofline bar. Separately, llama.cpp-unpj
found the 2026-08-11 commit abecb785d deleted ~6.3k lines of MoE layer-executor code including
the entire decode-time device-ID route (nullptr host-ids, graph-recordable dispatch); decode
today instead does a blocking host-ID D2H per MMID op per layer and is excluded from graph
replay. The owner ruled RESTORE-BEFORE-MERGE on 2026-08-16; RESTORE-T1 has since landed
(90166d9f9, 19e5bafc9, 190533a92) with the production guard
(`use_device_grouped_moe_decode`/`use_device_ids_for_pair_glu`, `ggml-sycl.cpp:70274-70284`)
live, so decode no longer forces a blocking host-ID D2H per MMID op unconditionally. T2-T6
(PP gate/up GEMM route, down 9-pass chunking, conformance discipline, mixed/secondary
investigation, final re-arm) remain open. Decode throughput numbers this epic measures are
still taken on a route missing T2-T6's restoration until those land.

## Current state (file:line refs)
- **Persistent-TG graph replay**: opt-in, default OFF (`ggml/src/ggml-sycl/dispatch.hpp:250`,
  `GGML_SYCL_PERSISTENT_TG=1` required). Core decode correctness is still wrong even outside
  the F16-Q attention path (Mistral FA-off persistent emits `#############` instead of digits) —
  llama.cpp-hmbk9 is unresolved. `should_use_persistent_tg()` unconditionally rejects any graph
  containing `MUL_MAT_ID` (llama.cpp-nubvg), so GPT-OSS never reaches persistent dispatch at all.
- **Graph replay gating** (non-persistent path): `ggml-sycl.cpp:100065` and `:100226` gate SYCL
  graph use on `ggml_sycl_graph_has_host_inputs(cgraph)` (`:3953`) — a graph with any host-side
  input (the current decode MMID host-ids copy included) is excluded from replay.
- **Device-ID MoE decode route**: deleted by `abecb785d` (2026-08-11, -6366 lines). RESTORE-T1
  (llama.cpp-haqk) has landed the decode-time device-ID wiring (`90166d9f9`, `19e5bafc9`,
  `190533a92`); the production guard `use_device_grouped_moe_decode`/`use_device_ids_for_pair_glu`
  (`ggml-sycl.cpp:70274-70284`) is the live gate. `moe_grouped_decode_candidate_env_enabled()`
  (`:26608`, callers `:55517`, `:56432`) still exists alongside it. Remaining restoration work
  (T2: PP gate/up GEMM route, T3: down 9-pass chunking, T4: conformance discipline, T5:
  mixed/secondary investigation, T6: final re-arm + full gate) is tracked under llama.cpp-unpj,
  launched 2026-08-16, still converging as of the 2026-08-26 comment.
- **XMX-tiled phase-owned decode** (llama.cpp-30ak7.19 family): `apply_single_moe_pp_safe_xmx_layouts`
  rewrites XMX_TILED gate/up to SOA for PP safety, which cost decode its faster TG layout — this
  is the open phase-ownership gap. Partial landing only: 32.91 tok/s FA-off (comment c-ub3y,
  `613df0d45`), below even the ticket's own ~37 tok/s creation-time baseline.
- **Grouped/batched decode formation**: `grouped_experts_device` (`mmvq.cpp:16563`,
  `:16730-16914`) is landed and live for kernel-level DPAS grouping; server-level auto batch
  formation across concurrent decode requests is unverified (llama.cpp-d4yko).
- **MXFP4 MoE layer executor** (llama.cpp-v90xn.30): infra landed — `moe-layer-plan.hpp`
  (`DIRECT_XMX`/`FUSED_LAYER`/`MIXED_RESIDENCY` kinds), `unified-cache.cpp
  build_moe_triplet_groups()` — but `ggml-sycl.cpp` production dispatch sites
  (`~L64556-64629`, `moe_route_kernel::MMVQ_COMPAT` at `:25261`/`:25905`) still hardcode the
  compat executor rather than routing through the landed infra.
- **Hybrid CPU/GPU expert dispatch**: live and residency-driven —
  `g_cpu_expert_pools`/`moe_expert_route_kind::HOST` (10 occurrences, e.g. `ggml-sycl.cpp:5339`,
  `:5982`, `:6281`, `:6374`, `:8395`) — this already implements what llama.cpp-0scz asked for.
- **Current B50/B70 GPT-OSS 20B MXFP4 FA-on baseline** (`docs/backend/sycl-perf-baselines.md:538`):
  B70 ≈1410 pp512 / ≈43 tg128, B50 ≈894 pp512 / ≈32 tg128. d0bp's root-cause comment
  (2026-08-21) predicts the in-flight DPAS-admission fix lands B50 at tg128 ~33-34, short of
  rty8's ≥34.5 bar without also recovering the ~1ms/token down-i8 residual.
- **Q6_K coalesced-slice guard** (llama.cpp-4wkt): `mmvq_assert_coalesced_slice_supported`
  (`mmvq.cpp:3853`) fail-closed-guards all 4 call sites (`:4554`, `:21898`, `:22001`, `:22030`);
  no Q6_K-pattern rewrite has landed.
- **XMX threshold** (llama.cpp-eju9): `sycl_env_settings[]` table (`ggml-sycl.cpp:23082`) is the
  sole source of the default (64); the intended 1024 bump was never measured against current
  baselines.

## Design constraints
- **Ruling 4 (placement decides the executor)**: device-resident experts execute on that GPU;
  host-resident experts run on the CpuExpertPool path. No per-token prefetch of experts into
  VRAM, no GPU zero-copy reads of host memory. The unpj restoration must re-route into the
  *existing* `g_cpu_expert_pools` call sites, not reconstruct cross-tier dispatch.
- **Ruling 5 (layout follows residency / one layout per weight)**: a restored or new decode
  route may consume an advertised layout differing from materialized storage only if it performs
  or provably triggers the reconciling staging (exemplars: `ggml-sycl.cpp:~65585`,
  `:~25334-25338`); never advertise a `(type,layout)` pair with no indexed kernel — anchor on the
  `moe-mmvq-tables.hpp` launcher roster, never a second hand-maintained predicate copy (this
  exact drift produced the 2026-08-16 MXFP4/Q6_K NaN family per llama.cpp-mn70/zoly/nkfc).
- **Ruling 6 (no host waits)**: the host-ID D2H the deleted device-ID route avoided, and any
  `wait()`/drain in a restored decode path, is against design; ordering must be SYCL event
  dependencies, not blocking copies in the hot decode loop.
- **Ruling 8**: small-block dequant fusion, not ESIMD, is the decode lever; oneDNN SDPA stays ON
  by default for attention and is not this epic's target (it owns MoE MUL_MAT_ID, not FA).
- **Ruling 9 (correctness before throughput)**: `GGML_SYCL_MOE_BLOCK_GRAPHLETS`,
  `GGML_SYCL_XMX_MOE_PP`/`_ALLOW_UNSAFE_PP`, `GGML_SYCL_PP_PIPELINE` stay opt-in — any default
  flip in this epic (persistent-TG, grouped decode, XMX-tiled decode) needs the Mistral and
  GPT-OSS gates green in the same build before promotion, per llama.cpp-cv8w's pre-registered
  flip-gate pattern.

## External / upstream status
- Upstream landed `#24676` (SYCL fix, 2026, commit prefix `b9674`): made the MoE-prefill
  async-memcpy source buffer persistent to fix a use-after-free beyond function scope. Different
  bug (prefill, not decode) but the same category as this epic's graph-safety concerns —
  confirms upstream is independently hardening MoE-path buffer lifetime for async SYCL copies.
  Does not obsolete or replace any member task; worth a diff-read before T1/T2 land in case the
  same buffer-lifetime pattern applies to the restored decode route.
  (https://github.com/ggml-org/llama.cpp, PR #24676, 2026)
- Upstream `#24452` added reordered Q4_K/Q5_K/Q6_K MoE MUL_MAT_ID support for the SYCL backend.
  Directly relevant to llama.cpp-4wkt (Q6_K coalesced-slice guard still fail-closed): read this
  PR's reorder pattern before hand-deriving one — it may shorten 4wkt from "derive from Q6_K's
  layout_base pattern" to "port an upstream reorder." Does not itself fix 4wkt (different code
  path, our fork's `mmvq_assert_coalesced_slice_supported` guard has no upstream analog).
  (https://github.com/ggml-org/llama.cpp/pull/24452, 2026)
- Intel compute-runtime 26.31.39395.13 (current driver, per CLAUDE.md) is confirmed an
  enablement/maintenance release — Crescent Island device-ID prep, Nova Lake Xe3P groundwork,
  Level Zero API bumped to 1.17 for listed platforms — with Battlemage (this host's B70/B50)
  retained at production status but no decode-bandwidth-relevant changelog entry found. The
  epic's driver pin is not stale relative to a materially faster release; no action item.
  (https://github.com/intel/compute-runtime/releases, 26.31.39395.13, 2026-08)
- am17an's "Optimizing Token Generation in llama.cpp's CUDA Backend" documents the CUDA
  backend's own strategy for the identical problem this epic solves: batch-1 decode is
  bandwidth-bound, kernel fusion reduces memory traffic, and CUDA graph capture isolates
  per-kernel launch overhead by replaying a captured sequence. This is architecturally the same
  bet as llama.cpp-hmbk9/nubvg (persistent-TG graph replay) and the unpj restoration
  (graph-recordable device-ID decode) — no code ports directly (different backend API) but it
  corroborates that fusion+replay is the right mechanism, not a detour.
  (https://am17an.bearblog.dev/new-post/, 2025-12-01)

## Work breakdown (prerequisites first)
1. **llama.cpp-unpj** — restore the deleted device-ID MoE decode route. T1 (decode wiring) has
   landed; T2 (PP gate/up GEMM route), T3 (down 9-pass chunking), T4 (conformance gate), T5
   (mixed/secondary investigation), and T6 (final re-arm + full gate) remain open. Decode
   measurements below are taken on a route still missing T2-T6's restoration until those land —
   this is the epic's true first prerequisite even though it is tracked as its own multi-task
   restoration effort, not a single member here.
2. **llama.cpp-d0bp** — standing decode regression vs the 79ae63559 baseline; root-caused to
   DPAS decode-kernel admission being falsified by policy plumbing. Any other member's claimed
   win must first explain this regression, per the taxonomy's own ordering hint.
3. **llama.cpp-hmbk9** — fix core persistent-TG decode correctness (blocks m3j9q, nubvg).
4. **llama.cpp-m3j9q** — F16-Q flash-attention support for persistent TG (blocked on hmbk9).
5. **llama.cpp-nubvg** — MoE MUL_MAT_ID support for persistent TG (blocked on hmbk9; needed
   before GPT-OSS can ever reach the persistent path at all).
6. **llama.cpp-32dg8.16.2** (absorbs llama.cpp-0np6) — promote persistent-TG to default policy,
   conditional on the correctness prerequisites above and a re-measured bar (not the stale
   B580/76-tok/s figures either duplicate cites).
7. **llama.cpp-30ak7.19** — phase-owned MoE layout policy so PP-safe SOA and TG-fast XMX_TILED
   coexist without a per-phase regression; parent of .10/.14/.17.
8. **llama.cpp-30ak7.19.10** — prove the planner-owned XMX-tiled gate/up packed-M2 decode path.
9. **llama.cpp-30ak7.19.14** — enable graph replay for all-device smart-handle MoE decode
   (depends on unpj T1 landing a graph-recordable decode route).
10. **llama.cpp-30ak7.19.17** — fix down-I8 residency after the XMX decode phase.
11. **llama.cpp-5ctzf** — integrate XMX-capable down projection with phase-owned decode; already
    has a profiler-backed negative result on its own (kept opt-in), revisit once .19.17 lands.
12. **llama.cpp-v90xn.30** — wire the landed MXFP4 MoE executor infra (triplet groups,
    DIRECT_XMX/FUSED_LAYER/MIXED_RESIDENCY) into the production dispatch sites that still
    hardcode MMVQ_COMPAT.
13. **llama.cpp-v90xn.30.7** — promote the winning bandwidth path with both canonical gates;
    separate the already-landed PP promotion (e3xj) from the still-open TG/decode promotion.
14. **llama.cpp-d4yko** — verify/land server-level decode-batch formation on top of the landed
    kernel-level `grouped_experts_device` grouping.
15. **llama.cpp-v90xn.31** — continuous batching + selected-expert row aggregation for GPT-OSS
    TG, on top of d4yko's batch formation.
16. **llama.cpp-e2kgu** — B50 TG MMVQ beyond the AOS workaround; rewrite bar against current
    baselines before resuming.
17. **llama.cpp-qz3yb** — dense MUL_MAT bucket optimization, blocked on v90xn.30 landing.
18. **llama.cpp-4wkt** — remove the Q6_K coalesced-slice fail-closed guard (implementation
    fully specified; needs GPU verification only).
19. **llama.cpp-eju9** — A/B the XMX threshold default (64 vs 1024) on current baselines.
20. **llama.cpp-zw4b** — reproduce `GGML_SYCL_MOE_GROUPED_DECODE=1` res=-3 failure at current
    HEAD (post b10630 merge; many MoE-route commits since the original repro).
21. **llama.cpp-y0it** — fix the DOWN-artifact reuse layout mismatch (P3, low-frequency dead-end
    path, not a correctness bug post-correction).
22. **llama.cpp-zb27** — PERF-EPIC E2: capability-derived route-default rule (pure function +
    matrix test); ready to implement, unblocked.
23. **llama.cpp-cv8w** — PERF-EPIC E3: wire the rule into the policy default (flip gate),
    blocked on zb27.
24. **llama.cpp-rty8** — PERF-EPIC B5: strip B1 instrumentation + decode validation (tg≥34.5),
    depends on the d0bp fix and zb27/cv8w's own internal ordering.
25. **llama.cpp-22sp** — PERF-EPIC E4: final end-to-end validation + epic close, blocked on
    cv8w and rty8.
26. **llama.cpp-ndzz1** — document TG batching as the primary throughput lever, once the above
    settles (doc-only, no code dependency, can run any time after v90xn.31/d4yko land something
    worth documenting).
27. **llama.cpp-30ak7** — epic umbrella; rewrite its acceptance criteria against the 80%-roofline
    bar (see below) instead of the stale fixed "~70 tok/s" anchor once the members above close.

## Bar / closing gate
Primary bar (owner ruling, DESIGN-BRIEF): achieved bytes/token × tok/s ≥ 0.8 × 224 GB/s on B50
(`level_zero:1`) and ≥ 0.8 × 608 GB/s on B70 (`level_zero:0`) for GPT-OSS 20B MXFP4 tg128,
measured by interleaved paired A/B with a bytes-streamed-per-token instrumentation pass (not
yet built — first deliverable of zb27/cv8w's route-default work).

Hard no-regression floors while that is chased (docs/backend/sycl-perf-baselines.md):
```
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1
# expect: pp512 >= 894, tg128 >= 32   (B50)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1
# expect: pp512 >= 1410, tg128 >= 44  (B70)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# expect: pp512 >= 1188, tg128 >= 47  (B50)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# expect: pp512 >= 2495, tg128 >= 108 (B70)
```
Every default flip (persistent-TG, grouped decode, phase-owned XMX layouts) additionally needs,
in the same build:
```
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf   -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0        # must emit 6, 7, 8, 9, 10
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli -m /models/gpt-oss-20b-mxfp4.gguf -ngl 99   -cnv -st --simple-io --no-display-prompt -c 4096   --chat-template-kwargs '{"reasoning_effort":"medium"}' --reasoning-format none --reasoning-budget 0   -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' -n 48 --seed 42 --temp 0
  # must emit exactly: 1, 2, 3, 4, 5
```
rty8's own pre-registered bar (tg≥34.5 B50) and cv8w's flip thresholds (B50 ≥863 pp512/≥34.5
tg128) are the PERF-EPIC track's intermediate checkpoints toward the 80%-roofline bar above, not
a substitute for it.

## Out of scope
- PP (prompt processing) throughput as an end in itself — this epic gates on PP not regressing,
  not on improving it; PP-focused work belongs to whichever epic owns qz3yb's dense-bucket and
  the XMX-tiled PP route (already default-on per v90xn.30.7's design_gaps).
- Multi-GPU / cross-device expert splitting (no PCIe P2P between B70 and B50; any such work is
  host-bounce by construction and is llama.cpp-unpj T5's investigation, not a throughput lever
  this epic can rely on).
- The 120B GPT-OSS model path is named in the epic title as aspirational ("once it loads") but
  no member task here targets it; do not scope new 120B-specific work into this epic without a
  new ticket.
- ESIMD dequant work (ruling 8 forecloses it as the lever; any resurrection attempt is
  against-design).

## Dependencies on other epics
- llama.cpp-jjm7, llama.cpp-wm4j, llama.cpp-0oad and their children are explicitly out of scope
  (owned by another session) — do not assign or touch, even if a member task here references
  them.
- llama.cpp-omp4 (post-merge Q1/NVFP4 device decode program) is adjacent scope-wise: unpj's
  T5 flagged that omp4's dependent ticket names ("primary decode," "secondary decode B70→B50")
  read suspiciously close to the deleted `try_mixed_moe_layer_pair`/`try_secondary_moe_layer_pair`
  shape, but omp4's guards are Q1/NVFP4-gated, not MXFP4-gated, so the two should stay separable
  pending explicit owner confirmation (unresolved as of the unpj restoration plan).

## Tasks

### llama.cpp-30ak7

  - **Id:** llama.cpp-30ak7
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-30ak7.19
  - llama.cpp-cv8w
  - llama.cpp-rty8
  - llama.cpp-22sp

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Re-anchor this umbrella epic's acceptance criteria on the owner's 80%-of-hardware-
roofline ruling instead of the fixed "~70 tok/s" figure this ticket still carries.

**Design**:
- Replace the "Acceptance Criteria" section's "~70 tok/s TG with FA on by default" with the
  bytes-streamed-per-token × tok/s ≥ 0.8 × 224 GB/s (B50) / 608 GB/s (B70) formulation from the
  epic doc's Bar section.
- Keep this ticket as the organizational roll-up only (per its own "Hierarchy role" note:
  "EPIC depends_on CHILD... reports child readiness without making a child blocked"); do not add
  new implementation scope directly to it.
- Its `depends_on` list already names llama.cpp-30ak7.19 and llama.cpp-d4yko among its 27
  children — leave that graph intact, just correct the acceptance text.
- Re-baseline the "Historical context" pp512/tg128 figures (1018/36.67 from 2026-05-21) to the
  current docs/backend/sycl-perf-baselines.md row (≈894 pp512/≈32 tg128 B50 FA-on) so future
  readers don't anchor on a stale number that predates the abecb785d regression and its restore.

**Bar**: umbrella closes only when 30ak7.19's family, cv8w's flip gate, and rty8/22sp's
PERF-EPIC close all report green against the current baselines doc, not this ticket's own
superseded numbers.

**Constraints**: Ruling on fixed-throughput anchors (DESIGN-BRIEF "efficiency goal" section) —
a fixed absolute tok/s target is explicitly named as the kind of drift that produced false
greens; this rewrite removes that anchor, it does not just update its value.

**Acceptance test**: none directly — this ticket is satisfied by its children's gates (see
Bar / closing gate section of the epic doc) all passing.

**Depends on**: llama.cpp-30ak7.19, llama.cpp-cv8w, llama.cpp-rty8, llama.cpp-22sp.

### llama.cpp-30ak7.19

  - **Id:** llama.cpp-30ak7.19
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-unpj

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Give PP and TG independently correct MoE layouts (PP-safe SOA, TG-fast XMX_TILED)
without the current phase-ownership regression where the PP-safe rewrite steals decode's faster
layout.

**Design**:
- Root cause is named in the ticket itself: `apply_single_moe_pp_safe_xmx_layouts` rewrites
  XMX_TILED gate/up entries to SOA for PP safety, and decode inherits that rewrite.
- Implement a planner-owned, phase-aware policy that rematerializes/releases the XMX_TILED vs SOA
  layout between phases through `mem_handle` — never a second resident copy of the same weight in
  two layouts (ruling 5 forbids dual materialization outright; the epic's own 613w lesson found a
  second resident copy cost pp512), never a raw-pointer cache, never a `GGML_SYCL_XMX_TILED_PP_
  PROOF`-style env requirement for default correctness.
- Layout choice must come from queried device capability + shape + planner budget
  (`ggml_sycl_moe_query_route_capability`), never a B50/B580 name branch (ruling 10 analog).
- This depends on llama.cpp-unpj's restoration landing first — decode measurements taken before
  T1/T6 land are on the degraded host-ids-D2H route and will misattribute unpj's own deficit to
  this ticket's phase-layout work.
- Sub-tasks .10 (gate/up XMX-tiled proof), .14 (graph replay), .17 (down-I8 residency) are this
  ticket's decomposition; land them in that order per the epic's ordering hint.

**Bar**: B50 GPT-OSS FA-on tg128 recovers toward the current baseline class (~32-34, not the
stale ~49-51 this ticket's history cites) with PP512 not collapsing below the PP-safe SOA class;
document any residual PP/TG tradeoff explicitly rather than silently regressing one phase.

**Constraints**: Ruling 5 (layout follows residency, one layout per weight — no per-phase-context
exemption; rematerialize/release the single resident copy through `mem_handle` between phases,
planner-owned, never dispatch-time-shifted per individual op). Ruling 2 (no raw pointer caches).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m
/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1 -r 3 -v`; route/profile logs show planner-
sourced layout selection, zero host/missing/layout-mismatch rows.

**Depends on**: llama.cpp-unpj.

### llama.cpp-30ak7.19.10

  - **Id:** llama.cpp-30ak7.19.10
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-30ak7.19
  - llama.cpp-unpj

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Prove the planner-owned XMX-tiled gate/up packed-M2 decode path actually dispatches
and measure its real contribution, not just its structural eligibility.

**Design**:
- Comment c-ub3y (`613df0d45`) already reached 32.91 tok/s FA-off — below even this ticket's own
  ~37 tok/s creation-time baseline, let alone the stale 49-51 target. Start from that landed
  partial state, not from scratch.
- Use a dispatch-path *counter*, not a structural/eligibility trace, to prove the packed-M2 path
  runs — llama.cpp-613w's lesson (cited in unpj's plan) is that a layout merely being ASSIGNED
  is not a route that RUNS; that attempt got 0 dispatches with resident i8 tensors and lost pp512
  to a second resident copy.
- Coordinate with 30ak7.19.14 (graph replay) — a device-ID-clean decode dispatch is a
  precondition for that ticket's graph-recordability goal, so land .10's kernel-selection fix
  before or alongside .14, not after.
- Replace the "B580 Mistral FA on/off" validation item with B70 (`level_zero:0`) — the intent
  transfers, the card does not exist anymore.

**Bar**: B50 GPT-OSS tg128 measured against the *current* baseline (~32-34), with a real
dispatch-count proof the packed-M2 XMX path engaged (not just was eligible); B70 Mistral gate
unregressed.

**Constraints**: Ruling 5 (kernel-existence checks device-agnostic on (type,layout); per-device
optimality in the selector only). Ruling 9 (no default flip without both canonical gates green).

**Acceptance test**: `GGML_SYCL_MXFP4_TG_PROFILE=1` device-event dispatch-variant census (per
d0bp's own successful use of this instrument) showing dpas/i8 kernel counts > 0, plus
`ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf
-p 512 -n 128 -fa 0 -r 3`.

**Depends on**: llama.cpp-30ak7.19, llama.cpp-unpj.

### llama.cpp-30ak7.19.14

  - **Id:** llama.cpp-30ak7.19.14
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-unpj
  - llama.cpp-30ak7.19.10

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Make all-device smart-handle MoE decode graph-recordable so it can replay instead of
re-recording every token.

**Design**:
- `ggml-sycl.cpp:100065` and `:100226` gate SYCL graph use on
  `ggml_sycl_graph_has_host_inputs(cgraph)` (`:3953`) — any host-side input excludes the graph
  from replay. The current decode MMID host-ids D2H (pre-unpj-restoration) is exactly such a
  host input, so MoE decode graphs cannot replay today regardless of this ticket's own work.
- This ticket is therefore gated on llama.cpp-unpj T1 landing the device-ID (nullptr host-ids)
  decode wiring first — a graph-recordable decode route is unpj's own stated acceptance
  criterion ("keeping decode graph-recordable"), so verify T1 closes `has_host_inputs=0` for the
  MoE decode graph before starting this ticket's own graph-replay wiring on top.
- Task's own 2026-07-10 comment shows `use_graph=0` again after a claimed 2026-05-29 fix
  (regressed) — get a fresh `GRAPH-DIAG` capture on current HEAD before assuming the earlier fix
  still holds.
- Once host-inputs are eliminated, confirm the replay path actually re-executes the captured
  command graph rather than falling back silently — instrument segment counts, not just a
  boolean flag.

**Bar**: B50/B70 GPT-OSS decode graph shows `has_host_inputs=0` and `use_graph=1` in
`GRAPH-DIAG` logging; tg128 improves measurably (launch-overhead reduction) with no correctness
regression on the GPT-OSS chat gate.

**Constraints**: Ruling 6 (no host waits — this IS the no-host-waits ruling applied to graph
capture: a host D2H inside a captured graph is exactly what the ruling forbids). Ruling 2 (mem_handle
lifetime must outlive the replayed graph's queued work).

**Acceptance test**: grep the string the code actually prints — the `[SYCL-SEG-MOE-POLICY]`
fprintf at `ggml-sycl.cpp:100227` (raw stderr, gated on `ggml_sycl_graph_diag_enabled()`,
not `GGML_SYCL_DEBUG`) — confirming `has_host_inputs=0 graphs_disabled=0
moe_graphs_disabled=0` on a GPT-OSS decode run. `GRAPH-DIAG` is a different
`GGML_SYCL_DEBUG` line (`ggml-sycl.cpp:~3442`) that does not carry these fields — do not
grep for it here.

**Depends on**: llama.cpp-unpj (T1 specifically), llama.cpp-30ak7.19.10.

### llama.cpp-30ak7.19.17

  - **Id:** llama.cpp-30ak7.19.17
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-30ak7.19.10

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Fix down-I8 residency loss after the XMX decode phase — the down projection's
selected-expert I8 artifact must survive phase transitions without being silently evicted or
rematerialized.

**Design**:
- Commits `5bfbdc1b7`/`e277c56bf`/`f61d990f7`/`3da68db87`/`925ea1e49` (2026-06-10/11) already
  added decode-side XMX_TILED rematerialization with a cached-q8 down bypass — read that diff
  first to determine whether it already subsumes this ticket's "selected-expert down-I8
  materialization" intent, or whether WEIGHT-zone fragmentation (the ticket's original concern)
  is still unaddressed on current HEAD.
- d0bp's 2026-08-21 root-cause comment names this exact residual: "down-i8 (~1 ms/token)
  deliberately deferred... may need the down decision revisited to hit >= 34.5" — this ticket is
  that deferred work, now on the critical path for rty8's bar.
- Coordinate with llama.cpp-5ctzf (integrate XMX-capable down with phase-owned decode) — 5ctzf
  already measured a negative result for a *standalone* down swap; this ticket's residency fix is
  a different axis (keeping the I8 artifact resident across phases, not choosing a different down
  kernel) and should be evaluated independently before assuming 5ctzf's conclusion applies here.
- Verify via unified-cache lease/eviction logging that the down-I8 `mem_handle` is not evicted
  between the gate/up XMX phase and the down consumption — this is a residency bug, not a kernel
  choice bug, if the artifact is correct but re-fetched from a colder tier.

**Bar**: B50 tg128 gains the ~1-1.2 ms/token d0bp attributes to this residual, landing at or
above rty8's 34.5 bar in combination with .10's gate/up fix.

**Constraints**: Ruling 2 (mem_handle lease correctness — do not force-evict a live down-I8
handle to "fix" apparent staleness; if it's stale, find the missing release). Ruling 5 (down
consumes the layout the planner materialized, no dispatch-time reconciliation invented ad hoc).

**Acceptance test**: `GGML_SYCL_MXFP4_TG_PROFILE=1` device-event census showing down-kernel
variant stays i8 across consecutive decode tokens (not falling back to a colder-tier reload);
`ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf
-p 512 -n 128 -fa 1 -r 3`.

**Depends on**: llama.cpp-30ak7.19.10.

### llama.cpp-32dg8.16.2

  - **Id:** llama.cpp-32dg8.16.2
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-hmbk9
  - llama.cpp-m3j9q
  - llama.cpp-nubvg

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Resolve the split-brain persistent-TG default — layout selection already treats
persistent-TG preference as default-ON while dispatch execution stays default-OFF — and promote
the fast path to default once correctness is proven.

**Design**:
- `dispatch.hpp:250` documents "Default: disabled. Set GGML_SYCL_PERSISTENT_TG=1 to enable" —
  confirmed still true at HEAD. This ticket absorbs llama.cpp-0np6 (identical intent, weaker
  spec — 0np6's own numeric bar, TG128≥76/PP512≥1200, is a stale absolute figure this ticket's
  own bar formulation already supersedes).
- Do NOT flip the default before llama.cpp-hmbk9 (core decode correctness — currently emits
  `#############` garbage on Mistral FA-off persistent) and llama.cpp-nubvg (MoE MUL_MAT_ID
  support — GPT-OSS never reaches persistent dispatch today) both land. Flipping the default with
  either open ships wrong tokens or silently degrades GPT-OSS to normal dispatch with the flag on.
- Re-baseline: this ticket's own B580 target (89.54 tok/s) is moot — default (no env var) TG128
  on B70 is already ~108 tok/s, well above that historical number, on hardware this card no
  longer represents. The comparison that matters is persistent-ON vs current-default-OFF on
  B70/B50 *today*, not vs a B580 figure.
- `should_use_persistent_tg()`'s existing safety guards (multi-device disabled unless
  `GGML_SYCL_PERSISTENT_SPLIT`, eviction/overflow handling) stay as-is — this ticket flips the
  *default*, it does not relax those guards.

**Bar**: persistent-TG default-ON only if it beats current-default TG128/PP512 on BOTH cards
(B70 ~108/~2495, B50 ~47/~1188 Mistral; B50 ~32/~894, B70 ~44/~1410 GPT-OSS) with both canonical
gates green; otherwise stays opt-in and this ticket closes won't-fix with the negative
measurement recorded.

**Constraints**: Ruling 9 (correctness before throughput — this is the textbook case: hmbk9's
open bug is a hard blocker on any default flip). Ruling 6 (no host waits in the promoted path).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=1
./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128` (repeat
`level_zero:1`), plus the Mistral completion gate, compared against default-off on the same
build.

**Depends on**: llama.cpp-hmbk9, llama.cpp-m3j9q, llama.cpp-nubvg.

### llama.cpp-0np6

  - **Id:** llama.cpp-0np6
  - **Action:** merge_into
  - **Merge into:** llama.cpp-32dg8.16.2
  - **Close evidence:** within_epic_duplicates pair in taxonomy/final.json; identical intent (flip persistent-TG default) with a strictly weaker/staler numeric bar (TG128>=76/PP512>=1200, pre-dating docs/backend/sycl-perf-baselines.md and the 80%-roofline ruling) than 32dg8.16.2, which already carries the split-brain diagnosis (dispatch.hpp:250 vs common.hpp layout preference) and the correct current-baseline comparison target.

### llama.cpp-5ctzf

  - **Id:** llama.cpp-5ctzf
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-30ak7.19.17

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Determine whether integrating an XMX-capable down projection into the phase-owned
decode executor (rather than as an isolated down-only swap) closes any further TG gap once
gate/up is correct.

**Design**:
- The ticket's own acceptance criteria already permit a "profiler-backed negative result" as a
  valid closure, and comment c-1mjn (2026-07-10) delivered exactly that: direct-down measured
  only +2.8% TG with lower PP (1203.85/33.13 → 1187.83/34.06), below the 5% promotion bar, so it
  was correctly kept opt-in and not included in `613df0d45`.
- Revisit only after llama.cpp-30ak7.19.17 (down-I8 residency fix) lands — that ticket targets a
  different axis (keeping the correct artifact resident across phases) than this one's kernel-
  choice question, and .19.17 landing may change the baseline this ticket should re-measure
  against.
- If re-measured and still <5% with lower PP, close as measured-no-win — do not keep re-running
  the same A/B expecting a different result absent an upstream change to what it's measuring
  against.
- `moe_layer_decode_plan` still lacks a down-bias artifact (`moe-layer-plan.hpp:549-558`) per
  comment c-f30r — if a future attempt wants planner-owned down-bias handle retention, that gap
  is the concrete missing piece, not a vague "integrate better."

**Bar**: >5% TG improvement over the phase-owned-decode-with-.19.17-fix baseline, with PP512 not
regressing, or a recorded negative result and won't-fix closure.

**Constraints**: Ruling 2 (mem_handle-owned down-bias artifact if built, no raw-pointer bias
cache). Ruling 5 (down layout chosen by queried capability/shape/budget, no B50 special case).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m
/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1 -r 3` before/after, plus GPT-OSS chat gate.

**Depends on**: llama.cpp-30ak7.19.17.

### llama.cpp-d4yko

  - **Id:** llama.cpp-d4yko
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Verify whether server-level continuous batching already coalesces concurrent decode
requests into grouped-DPAS batches ≥16, on top of the already-landed kernel-level grouping.
Note: `--tg-batch` exists nowhere in `tools/` or `common/arg.cpp` at HEAD (llama-bench's only
batch knob is `-b`/`--batch-size`, `tools/llama-bench/llama-bench.cpp:606`) — it survives
only in shell harnesses (`scripts/sycl-moe-regression-harness.sh:149`,
`scripts/sycl-moe-roofline-harness.sh:329`, `AGENTS.md:85`), so this ticket must not gate on
a CLI flag; drive concurrency via concurrent client connections instead.

**Design**:
- Kernel-level grouped-DPAS MoE dispatch is confirmed landed: `grouped_experts_device`
  (`mmvq.cpp:16563`, allocation at `:16730`, dispatch at `:16748-16914`), commits
  `4f634cbb0`/`a8251fe3f`/`7352bad66` (llama.cpp-sk67). What remains unverified is purely the
  server-level formation question, not the kernel.
- Instrument `llama-server` (or a harness driving it) with N concurrent decode streams and check
  whether the grouped-DPAS path's dispatch-count census shows batch sizes >1 without
  `--tg-batch` — if it already does, this ticket closes as verified-done with that evidence; if
  not, scope the explicit batch-formation logic as new work here.
- Check dependency llama.cpp-5crav's status (DPAS-native MXFP4 layout) — if closed, note that in
  the closure evidence; if open, this ticket may still need it for the grouped path's layout.
- This ticket has no useful throughput bar of its own — it feeds llama.cpp-v90xn.31's continuous-
  batching + row-aggregation work, which is where the actual multi-sequence tg-batch numbers
  should be re-measured (the old 26/46/73/141/210 tok/s FA-off figures cited elsewhere are stale
  and belong to v90xn.31's re-measurement, not this ticket).

**Bar**: a concrete finding — either "server-level formation already coalesces to N≥16" (cite the
census) or "formation logic is missing, scoped as X" — not a vague "investigate."

**Constraints**: Ruling 4 (grouped batches must still respect per-expert residency — grouping
concurrent requests must not become a backdoor for streaming host-resident experts to device
scratch per batch).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_MXFP4_TG_PROFILE=1
./build/bin/llama-server -m /models/gpt-oss-20b-mxfp4.gguf -ngl 99` (no `--tg-batch` — the
flag does not exist in the binary) driven by N=8 concurrent decode clients; grep the
dispatch-variant census for the grouped-batch size actually observed.

**Depends on**: none structurally (kernel prerequisite already landed).

### llama.cpp-hmbk9

  - **Id:** llama.cpp-hmbk9
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Fix persistent-TG decode correctness outside the F16-Q attention path before any
F16-Q or MoE persistent work is allowed to proceed — this is the hard blocker for m3j9q, nubvg,
and 32dg8.16.2's default-flip.

**Design**:
- Confirmed still open and still the right blocking priority: Mistral Q4_0 FA-off persistent
  (`-c 4096 GGML_SYCL_PERSISTENT_TG=1`) emits `#############` instead of `6, 7, 8, 9, 10`.
- The ticket's own notes narrow this correctly: no-FA forced persistent now falls back cleanly on
  F16 MUL_MAT and emits correct output (a real fix landed there) — the remaining defect is FA-on
  persistent with the mask/V-stride fixes already applied but still emitting garbage.
- Required next step per the ticket: build a trace comparison of normal-TG vs persistent-TG op
  hashes for the DECODE token specifically (not prompt-phase hashes — the ticket flags this
  distinction as unresolved). Start from no-FA to remove FLASH_ATTN_EXT from the search space,
  per the ticket's own required-work ordering.
- Audit every `fallback other` node in the persistent micro-graph — identify whether each is
  MATMUL/GLU/view-like-noop/unsupported-required, since the ticket notes this census was never
  finished.
- Add a correctness gate so `GGML_SYCL_PERSISTENT_TG=1` refuses (falls back) on any graph shape
  it cannot prove correct, rather than emitting wrong tokens silently — this is itself part of
  the required deliverable, not just the eventual fix.

**Bar**: `GGML_SYCL_PERSISTENT_TG=1 -fa 0 -c 4096` Mistral completion emits `6, 7, 8, 9, 10`;
default gated path continues to fall back and emit correct output. FA-on/F16-Q correctness
is llama.cpp-m3j9q's scope (there is currently no env gate for forcing F16-Q FA under
persistent — see m3j9q's addendum), not duplicated here.

**Constraints**: Ruling 9 (correctness before throughput — this ticket IS that ruling in its
purest form: no persistent-TG work proceeds past this). Ruling 6 (persistent execution must not
introduce host waits to work around the correctness gap).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=1
./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf -fa 0 -c 4096 -p '1, 2, 3, 4, 5,'
-n 15 --seed 42 --temp 0` → must print `6, 7, 8, 9, 10`.

**Depends on**: none — this is the root blocker.

### llama.cpp-m3j9q

  - **Id:** llama.cpp-m3j9q
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-hmbk9

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Support F16 Q in persistent-TG's flash-attention descriptors/kernels, currently
rejected with "FLASH_ATTN unsupported Q type=1 (expected F32)".

**Design**:
- Hard-blocked on llama.cpp-hmbk9 landing first — do not resume F16-Q work until hmbk9's core
  decode correctness (including the FA-on garbage-token defect) is fixed; this ticket's own
  notes already found forced F16-Q FA persistent produces a clean graph (no fallback, MMVQ=193,
  attn=32, copy=33) but still emits blank/space tokens — the same shape of defect hmbk9 is
  chasing, likely a shared root cause (Q/K/V or SET_ROWS ordering/content, per the ticket's own
  "Remaining requirement" note).
- Do not treat this as a separate investigation from hmbk9's first-divergence trace — run that
  trace on the F16-Q path specifically once hmbk9's no-FA/F32-Q trace methodology is proven, and
  reuse it rather than building a parallel one.
- Preserve F32-Q correctness throughout (default gated path already does — confirmed still
  falling back correctly and emitting `6, 7, 8, 9, 10`).
- Retarget validation hardware: ticket cites B580, replace with B70 (`level_zero:0`) — the intent
  (deterministic FA-on completion with persistent forced) transfers cleanly, the card does not
  exist.
- `GGML_SYCL_PERSISTENT_TG_F16_Q` does not exist anywhere in the repo (ruling 11) — the
  current code (`ggml-sycl.cpp:~96893`) unconditionally rejects any Q tensor with
  `type != GGML_TYPE_F32` inside persistent FLASH_ATTN, with no bypass. Part of this
  ticket's own deliverable is adding that opt-in gate (name it
  `GGML_SYCL_PERSISTENT_TG_F16_Q` if that name is kept, or state the name actually
  chosen) so F16-Q persistent FA can be forced for testing before being proven correct
  enough to remove the gate entirely; do not write an acceptance test against an env var
  that does not yet exist without first landing it.

**Bar**: B70 Mistral Q4_0 FA-on deterministic completion emits `6, 7, 8, 9, 10` with
persistent forced via the new opt-in gate this ticket adds (see Design) and no fallback
logged; FA-off bench with persistent disabled stays at expected PP/TG.

**Constraints**: Ruling 9 (blocked on hmbk9's correctness fix by design — do not ship F16-Q
persistent ahead of core decode correctness).

**Acceptance test**: once this ticket lands its own opt-in gate (the review found
`GGML_SYCL_PERSISTENT_TG_F16_Q` has zero occurrences in the repo today, so this exact
invocation is not runnable at HEAD): `ONEAPI_DEVICE_SELECTOR=level_zero:0
GGML_SYCL_PERSISTENT_TG=1 <the landed gate>=1 ./build/bin/llama-completion -m
/models/mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0` → must
print `6, 7, 8, 9, 10`.

**Depends on**: llama.cpp-hmbk9.

### llama.cpp-nubvg

  - **Id:** llama.cpp-nubvg
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-hmbk9

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Teach persistent-TG eligibility and planning to handle `GGML_OP_MUL_MAT_ID` so
GPT-OSS decode can reach the persistent/graph-replay path at all.

**Design**:
- Confirmed still open: `should_use_persistent_tg()` (`ggml-sycl.cpp:98977-98995`) unconditionally
  rejects any graph containing MUL_MAT_ID; only probe/descriptor-capture diagnostic scaffolding
  exists today, no actual persistent MoE dispatch path.
- Must honor placement-decides-the-executor (ruling 4): device-resident experts dispatch through
  a persistent-compatible MoE descriptor; host-resident experts route to the existing
  `g_cpu_expert_pools` CPU path (confirmed live, `ggml-sycl.cpp` — 10 occurrences) — build on
  that, do not reintroduce a competing host-dispatch mechanism.
- Use `mem_handle`/resolver ownership for expert weights across tokens — never cache a raw
  expert pointer that could go stale between persistent-graph replays (this is the general "no
  raw pointer as source of truth" rule applied to a specifically dangerous case: a captured graph
  that outlives a single dispatch).
- This ticket is a hard prerequisite for llama.cpp-30ak7.19.14's graph-replay-for-MoE-decode goal
  and for 32dg8.16.2's default flip — GPT-OSS cannot benefit from persistent-TG promotion while
  its dominant op (MUL_MAT_ID) always falls back.
- Blocked on hmbk9 landing first — building MoE persistent dispatch on top of a still-incorrect
  core decode path means any MoE-specific bug is unattributable.

**Bar**: B50 GPT-OSS default path (no env var) unregressed (~894 pp512/~32 tg128); with
`GGML_SYCL_PERSISTENT_TG=1 GGML_SYCL_PERSISTENT_TG_LOG_POLICY=1` either a safe persistent MoE
dispatch engages (measure tg128 delta) or an explicit unsupported-reason log precedes clean
fallback — never a silent wrong-answer path.

**Constraints**: Ruling 4 (placement decides executor — no per-token host-to-device expert
streaming to make persistent dispatch "work"). Ruling 2 (mem_handle-owned expert weight
lifetime across graph replays, no stale pointer tables).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_PERSISTENT_TG=1
GGML_SYCL_PERSISTENT_TG_LOG_POLICY=1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf
-p 0 -n 128 -fa 1 -r 2 -v` plus GPT-OSS chat gate on the same build.

**Depends on**: llama.cpp-hmbk9.

### llama.cpp-v90xn.30

  - **Id:** llama.cpp-v90xn.30
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-unpj

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Wire the already-landed MXFP4 MoE executor infrastructure (triplet grouping,
DIRECT_XMX/FUSED_LAYER/MIXED_RESIDENCY plan kinds) into the production dispatch sites, which
still hardcode the older MMVQ_COMPAT executor.

**Design**:
- Confirmed via grep: `moe-layer-plan.hpp` has the DIRECT_XMX/FUSED_LAYER/MIXED_RESIDENCY kinds
  and `unified-cache.cpp` has `build_moe_triplet_groups`, but `ggml-sycl.cpp` production sites
  (`~L64556-64629`) and the recipe defaults (`moe_route_kernel::MMVQ_COMPAT` at `:25261`,
  `:25905`) still hardcode the compat path.
- Rewrite this epic's own description against the landed infra instead of proposing it from
  scratch (its design_gaps note this explicitly). Point to child llama.cpp-hsx04 (referenced as
  "the concrete remaining TG-XMX work") as the actual work item rather than duplicating scope
  here — verify hsx04 is still open before treating it as the sole remaining child.
- This epic's own restoration is entangled with llama.cpp-unpj's — unpj deleted the
  device-ID decode wiring that a restored/wired executor would need to route through, so treat
  unpj as a hard prerequisite, not a parallel track.
- All B50/B580 numbers embedded in this ticket's history need updating to current baselines
  (B580 no longer exists as hardware).

**Bar**: production dispatch sites route through `moe-layer-plan.hpp`'s plan kinds for at least
one real (non-diagnostic) decode or PP case, verified by a dispatch-count census (not a
structural "is eligible" trace, per the 613w lesson cited elsewhere in this epic).

**Constraints**: Ruling 5 (kernel-existence checks device-agnostic, roster-anchored — wiring the
new executor must not create a second hand-maintained (type,layout) predicate list alongside the
existing MMVQ_COMPAT one).

**Acceptance test**: whatever hsx04 (if still open) specifies as its own acceptance test; absent
that, a `GGML_SYCL_MXFP4_TG_PROFILE=1` census showing DIRECT_XMX/FUSED_LAYER dispatch counts > 0
on `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf
-p 512 -n 128 -fa 1`.

**Depends on**: llama.cpp-unpj.

### llama.cpp-0scz

  - **Id:** llama.cpp-0scz
  - **Action:** close
  - **Close evidence:** Ticket body is empty (bare title only, created 2026-03-13, pre-dates the current MoE hybrid architecture). Current code already implements exactly the requested hybrid dispatch: moe_expert_route_kind::HOST (ggml-sycl.cpp:5339,5982,6281,6374,8395) and g_cpu_expert_pools (10 occurrences) drive residency-based CPU/GPU expert routing for MXFP4 — there is no remaining CPU-TG-fast-path-for-MXFP4 to skip; the hybrid dispatch this ticket asked for is the shipped design.

### llama.cpp-22sp

  - **Id:** llama.cpp-22sp
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-cv8w
  - llama.cpp-rty8

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: PERF-EPIC E4 — final end-to-end validation and epic close for the GPT-OSS perf
recovery track, once its own dependents land.

**Design**:
- Spec is already complete: `docs/plans/2026-08-21-gptoss-perf-recovery-epic.md` §"End-to-End
  Validation" — no rewrite needed, per the keep.json design_gaps.
- Deps `llama.cpp-lvzb` (`c4b0f903c`) and `llama.cpp-vsnr` (`06272685f`) are landed; this ticket
  is blocked purely on `llama.cpp-rty8` (B5) and `llama.cpp-cv8w` (E3 flip gate) finishing, per
  the tracker's own dependency state.
- When rty8 and cv8w both close, run the full end-to-end validation the plan doc specifies
  against current baselines (docs/backend/sycl-perf-baselines.md), not against any number
  embedded in the plan doc itself if it has since drifted — the plan doc's role is the procedure,
  the baselines doc is the numeric authority (per CLAUDE.md's standing rule).
- This ticket is the epic's own closing checkpoint — do not let it become a second acceptance
  bar independent from the 80%-roofline bar this epic (gptoss-decode-bandwidth) sets; treat
  the PERF-EPIC's own thresholds as the intermediate checkpoint the DESIGN-BRIEF's ordering hint
  already names them as.

**Bar**: full E2E validation per the plan doc, both canonical gates green, B50/B70 GPT-OSS and
Mistral pp512/tg128 at or above current baselines.

**Constraints**: Ruling 9 (both canonical gates must be green before this epic-closing validation
is considered complete).

**Acceptance test**: the procedure in `docs/plans/2026-08-21-gptoss-perf-recovery-epic.md`'s
End-to-End Validation section, run on current HEAD after rty8/cv8w land, pinned as:
```
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1   # B50 GPT-OSS
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1   # B70 GPT-OSS
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128      # B50 Mistral
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128      # B70 Mistral
```
against current `docs/backend/sycl-perf-baselines.md` rows, plus both canonical completion gates
(Mistral + GPT-OSS `-c 4096`, per CLAUDE.md's Verification Commands section) green in the same
build.

**Depends on**: llama.cpp-cv8w, llama.cpp-rty8.

### llama.cpp-cv8w

  - **Id:** llama.cpp-cv8w
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-zb27

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: PERF-EPIC E3 — wire the capability-derived route-default rule into the policy
default, with pre-registered flip thresholds.

**Design**:
- Dep `llama.cpp-vsnr` landed (`06272685f`, D3 B70 validation + Track E flip thresholds). Dep
  `llama.cpp-zb27` (E2, the rule this ticket wires in) is still open per the tracker — this
  ticket cannot start its flip until zb27 lands the pure function it consumes.
- No rewrite needed: thresholds are already pre-registered and owner-approved in the ticket text
  (B50 ≥863 pp512 / ≥34.5 tg128, B70 D3-derived) — per keep.json design_gaps, this is purely
  blocked, not underspecified.
- When zb27 lands, wire its `moe_pp_route_default_input`/`ggml_sycl_moe_pp_batched_default_from_
  caps` function as the actual default-selection logic (not a second hand-maintained default
  table) — this is the flip-gate pattern the epic doc's Ruling 9 note points to as the model for
  every other default flip in this epic (persistent-TG, phase-owned layouts).

**Bar**: default route selection is driven by zb27's capability-derived function; B50 ≥863
pp512/≥34.5 tg128, B70 at its D3-derived threshold, both canonical gates green in the same build.

**Constraints**: Ruling 9 (pre-registered thresholds before the flip, exactly as done here —
this ticket is the epic's own reference implementation of the "flip gate" pattern named in the
epic doc's Constraints and Bar sections).

**Acceptance test**: this ticket's own pre-registered E3 acceptance test (per the ticket text),
pinned as:
```
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1   # B50: expect pp512>=863, tg128>=34.5
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1   # B70: expect >= its D3-derived threshold (see llama.cpp-vsnr)
```
plus both canonical completion gates (Mistral + GPT-OSS `-c 4096`) green in the same build.

**Depends on**: llama.cpp-zb27.

### llama.cpp-d0bp

  - **Id:** llama.cpp-d0bp
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Explain and fix the B50 GPT-OSS decode regression vs the 79ae63559 baseline
(SOA decode 35.2→25.7 tg128, tiled default 31.6) — the epic's own "standing regression that any
claimed win must first explain," per the taxonomy's ordering hint.

**Design**:
- Root cause is already found and event-timed (lead comment c-kobw, 2026-08-21): kernel-VARIANT
  selection, not host overhead, not route resolution. `GGML_SYCL_MXFP4_TG_PROFILE=1` device
  events show HEAD-policy MoE kernels at 18.3ms/token (all plain SOA q8_1) vs baseline
  10.6ms/token (mixed soa/dpas/i8, with gate/up on the DPAS decode kernel). The chain of
  eliminated hypotheses (route-table resolution, ids D2H, admission bookkeeping, host waits) is
  documented in the comment — do not re-litigate those; the remaining work is purely the DPAS
  admission fix.
- Fix in flight per the same comment: restore DPAS decode-kernel admission under the batched-PP
  policy (impl-1tjn) — the policy plumbing's chokepoint demotion/predicate leak falsifies an
  admission condition the DPAS-on-SOA decode variant checks despite needing no tiled artifacts.
  Verify this landed at current HEAD (the ticket's own comment names commits
  `5100ae8bb`+`2ef7c65e3` as the restore per keep.json) before assuming it is still pending.
- Down-i8 (~1ms/token) is deliberately deferred to llama.cpp-30ak7.19.17 per this same comment —
  do not fold that work back into this ticket; it has its own home.
- With gate/up DPAS restored, this comment predicts policy tg ~33-34 — track whether that lands
  and whether it clears rty8's ≥34.5 bar on its own or needs .19.17's down-i8 fix too.

**Bar**: B50 GPT-OSS tg128 recovers to the 79ae63559 baseline class (~35.2) or documents the
specific residual gap and hands it to .19.17; docs/backend/sycl-perf-baselines.md already shows
tg128=33.68 post-fix per keep.json — confirm this figure is current and update the epic's
prerequisite chain accordingly.

**Constraints**: Ruling 9 (this regression must be explained before any other member claims a
win — per the epic's own ordering hint, this is load-bearing for interpreting every other
decode-throughput measurement in the epic).

**Acceptance test**: `GGML_SYCL_MXFP4_TG_PROFILE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1
./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 0 -n 128 -fa 1 -r 2 -v` — kernel-
variant census should show dpas/i8 counts matching the 79ae63559 baseline shape (soa=17 dpas=48
i8=7 class), not the all-soa HEAD-policy shape.

**Depends on**: none — this is itself a prerequisite for other members' throughput claims.

### llama.cpp-e2kgu

  - **Id:** llama.cpp-e2kgu
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Optimize Arc Pro B50 TG MMVQ beyond the current AOS workaround, rewritten against
current baselines instead of the ticket's stale B580/pre-architecture-overhaul figures.

**Design**:
- Ticket's historical baseline (tg32 5.54→8.56 tok/s) is from before the architecture overhaul
  and is not a useful comparison point any more — broad architecture work (unified cache,
  planner-owned placement, grouped-DPAS decode) already closed most of the gap that baseline
  represented.
- Rewrite acceptance against the current validated baseline: B50 Mistral tg128 ~47, B50 GPT-OSS
  tg128 ~32 (docs/backend/sycl-perf-baselines.md), with a concrete target derived from the
  80%-of-224GB/s bandwidth ceiling (the same formulation this epic's Bar section uses), not an
  arbitrary tok/s delta.
- Drop the "preserve B580 ~80 tok/s default" validation item entirely — that card is removed
  from this machine; if any intent from it transfers, restate it against B70 (`level_zero:0`).
- Scope this against whatever's left of the AOS-workaround gap after d0bp's DPAS-admission fix
  and 30ak7.19's phase-owned XMX layouts land — measure the residual after those, not before,
  since both directly touch MMVQ kernel selection on the same hot path this ticket targets.

**Bar**: measurable B50 tg128 improvement beyond the AOS-workaround baseline, expressed as a
fraction of the 224 GB/s roofline (per the epic doc's Bar section), with both canonical gates
green.

**Constraints**: Ruling 8 (the lever is fusing dequant into the matmul, not ESIMD — any AOS-
replacement kernel must not resurrect ESIMD small-block dequant, measured 1.9x slower).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m
/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -r 3` and the equivalent GPT-OSS run, against
current baselines, both canonical gates green.

**Depends on**: none formally, but should be re-scoped after llama.cpp-d0bp and llama.cpp-30ak7.19 land.

### llama.cpp-qz3yb

  - **Id:** llama.cpp-qz3yb
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-v90xn.30

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Optimize the dense MUL_MAT bucket (non-MoE ops in the GPT-OSS graph) after the MoE
executor work stabilizes.

**Design**:
- Explicitly blocked on `llama.cpp-v90xn.30` landing (MXFP4 MoE executor wiring still WIP) —
  keep this dependency; optimizing the dense bucket while the MoE executor's dispatch sites are
  still being rewired risks profiling against a moving baseline.
- The ticket's existing data (598.6ms bucket) predates current baselines (2026-05-18,
  pre-planner-rewrite) — refresh the dense-bucket profile against the current baseline (B50
  GPT-OSS ~893.77 pp512/~32.06 tg128) before scoping any specific optimization.
- Use the same `GGML_SYCL_MXFP4_TG_PROFILE=1`-style per-op device-event timing this epic's other
  members already use (d0bp, 30ak7.19.10) for consistency of measurement method across the epic.

**Bar**: a fresh dense-bucket time profile on current HEAD, with a specific optimization target
identified (not "optimize" in the abstract) — dense MUL_MAT is a smaller fraction of total decode
time than MoE, so the bar here is proportionate: any real improvement counts, no fixed percentage
target given the ticket's own thin spec.

**Constraints**: Ruling 5 (dense weight layout still follows residency — this ticket must not
introduce a second dense-weight materialization path parallel to what the planner already
produces).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_MXFP4_TG_PROFILE=1
./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1 -v` with dense-op
timing isolated from MoE-op timing in the resulting profile.

**Depends on**: llama.cpp-v90xn.30.

### llama.cpp-rty8

  - **Id:** llama.cpp-rty8
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-d0bp
  - llama.cpp-30ak7.19.17

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: PERF-EPIC B5 — strip the B1 instrumentation added earlier in the recovery track and
validate decode reaches tg≥34.5.

**Design**:
- Instrumentation is confirmed still present: `6f02a6681` (the B1 probe, whose own commit
  subject says "removed by B5/rty8") and the census commits (`4f5b77158`/`7f764465d`/
  `06f881473`) are all still in git history with no later removal commit found — this ticket's
  own strip-list is already precise (per keep.json design_gaps), it is mainly blocked on
  `98w0`/`pbb2` landing first so the strip doesn't remove instrumentation that B7's own
  attribution work still needs.
- The 34.5 bar is the same one d0bp's root-cause comment predicts landing ~33-34 without the
  down-i8 fix and higher with it — treat this ticket's validation as the point where d0bp
  (kernel-variant fix) and 30ak7.19.17 (down-I8 residency) combine to clear the bar, not as an
  independent throughput lever of its own.
- Do not strip any instrumentation still referenced by an open attribution ticket (B7) — verify
  with a grep for each probe's call sites before removing, not just its definition.

**Bar**: B50 GPT-OSS tg128 ≥ 34.5 with the B1 instrumentation stripped and no regression in
either canonical gate.

**Constraints**: Ruling 9 (this is an intermediate checkpoint toward the epic's real
80%-roofline bar, not a substitute for it — see epic doc Bar section).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m
/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1 -r 3` → tg128 ≥ 34.5, plus GPT-OSS chat gate.

**Depends on**: llama.cpp-d0bp, llama.cpp-30ak7.19.17 (both feed the tg128 number this ticket
gates on), and 98w0/pbb2 (external to this epic's member list) for the instrumentation-removal
ordering.

### llama.cpp-unpj

  - **Id:** llama.cpp-unpj
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Restore the device-ID MoE decode route (and the wider MXFP4 layer executor) that
`abecb785d` (2026-08-11, -6366 lines) deleted — the epic's true first prerequisite; every decode
throughput measurement elsewhere in this epic is taken on a degraded route until this lands.

**Design**:
- Unusually well-specified already — restoration plan (comment c-ys6x), Option B (targeted
  re-route onto live surviving machinery, not a blind revert of `abecb785d`) adopted, decomposed
  into T1 (decode device-ID wiring), T2 (PP gate/up GEMM route), T3 (down 9-pass chunking), T4
  (conformance discipline applied at every review, not a separate task), T5 (mixed/secondary
  investigation), T6 (final re-arm sweep + full gate). All filed as separate tickets
  (`llama.cpp-haqk`/T1, `llama.cpp-qvzq`/T3, `llama.cpp-twl6`/T2, `llama.cpp-mk4l`/T5,
  `llama.cpp-49pp`/T6) and launched 2026-08-16.
- Owner ruling: RESTORE-BEFORE-MERGE (comment c-aqy1, 2026-08-16) — this is pre-merge critical
  path, not deferred cleanup.
- Do NOT `git revert abecb785d` — it also added the live transactional retained-prompt-fusion
  model (moe_retained_terminal_bundle) that currently serves the PP fallback path correctly, and
  9+ days of unrelated fixes (mn70/zoly/nkfc layout fixes) have landed inside the same function
  since. Build on current code, don't revert past it.
- T1 re-routes onto the CALLERLESS `mmvq_moe_batched_dispatch_pair_mxfp4_soa` (mmvq.cpp:17274)
  and the LIVE `mxfp4_build_grouped_metadata_from_ids_sycl`/`mxfp4_xmx_tiled_grouped_direct_q8_
  sycl` kernels (mmvq.cpp) — these were never deleted, only their caller-side wiring in
  `ggml_sycl_mul_mat_id` was.
- T2/T3 extend the working grouped-DPAS pattern already proven at PP for down
  (`try_mxfp4_i8_grouped_down`, ggml-sycl.cpp:64748) to gate/up and fix its 9-pass chunking.
- Any host-resident branch routes through the EXISTING `g_cpu_expert_pools` call sites — do not
  reintroduce `try_mixed_moe_layer_pair`/`try_secondary_moe_layer_pair` verbatim (T5 investigates
  whether current planner-side residency assignment already makes them redundant).

**Bar**: RESTORE-T1 (llama.cpp-haqk) is already landed and re-armed its 14
`CHECK_SKIPPED("llama.cpp-unpj", ...)` sites (0 remaining today — do not re-litigate T1).
This ticket now closes on T2 (PP gate/up GEMM route), T3 (down 9-pass chunking), T4
(conformance discipline applied throughout), T5 (mixed/secondary investigation), and T6
(final re-arm + full gate) landing; ~9.7 GB expert-weight traffic per pp512 pass (not
~442 GB); GPT-OSS PP512/TG128 on both B50/B70 at/above sycl-perf-baselines.md; Mistral
unregressed; both canonical gates green.

**Constraints**: Ruling 4 (host-resident branch re-routes into existing CpuExpertPool, no
reconstruction). Ruling 5 (R1/R2 conformance — every restored dispatch consumes materialized
layout or provably reconciles it; every advertised (type,layout) has an indexed kernel, anchored
on the moe-mmvq-tables.hpp roster). Ruling 6 (no host waits — this IS the property being
restored: nullptr host-ids keeps decode graph-recordable). Ruling 2 (all new/restored allocation
via unified-cache entry points regardless of what the 2026-08-11-vintage deleted code did).

**Acceptance test**: confirm T1 remains live — `grep -n 'use_device_grouped_moe_decode\|use_
device_ids_for_pair_glu' ggml-sycl.cpp` around `:70274-70284` — then grep
`tests/test-sycl-moe-sequence-graphlet-policy.cpp` for any `CHECK_SKIPPED("llama.cpp-<T2..T6
sub-ticket id>"` sites and confirm each clears as its owning sub-ticket (haqk/qvzq/twl6/mk4l/
49pp) lands; plus the full GPT-OSS/Mistral gate matrix per the Bar above.

**Depends on**: llama.cpp-49pp (T6, tracked outside this epic's member list; the epic's
practical dependency is on T1-T6 collectively landing).

### llama.cpp-v90xn.30.7

  - **Id:** llama.cpp-v90xn.30.7
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-v90xn.30

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Promote the winning bandwidth path with both canonical gates, separating the
already-landed PP promotion from the still-open TG/decode promotion this epic actually needs.

**Design**:
- PP-side promotion already landed for a related but different scope:
  `planner_xmx_tiled_pp_route_active()` in `unified-cache.cpp` is default-ON since
  `llama.cpp-e3xj`/2026-08-17 — this is the grouped-DPAS XMX_TILED PP route for MXFP4 gate/up.
  Do not re-promote this; it's done.
- Rewrite this ticket's scope to explicitly split "PP promotion" (done, cite e3xj) from
  "TG/decode promotion" (not done — this is the epic's actual remaining ask), so future readers
  don't conflate the two halves.
- The TG/decode promotion half is downstream of llama.cpp-30ak7.19's phase-owned layout work and
  llama.cpp-v90xn.30's executor wiring — this ticket's job is the FINAL flip-gate step (both
  canonical gates + current baselines), not new dispatch-path implementation.
- Update the acceptance/gates list to current B70/B50 baselines — the ticket's original targets
  are undocumented/stale per keep.json design_gaps.

**Bar**: TG/decode path promoted to default with B50/B70 GPT-OSS and Mistral pp512/tg128 at or
above current sycl-perf-baselines.md rows, both canonical gates green in the same build.

**Constraints**: Ruling 9 (flip-gate pattern — pre-register the thresholds, as cv8w already does
for the PERF-EPIC track; reuse that pattern here rather than inventing a separate one).

**Acceptance test**: full llama-bench matrix, pinned as:
```
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1   # B50 GPT-OSS
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -fa 1   # B70 GPT-OSS
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128      # B50 Mistral
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128      # B70 Mistral
```
against current `docs/backend/sycl-perf-baselines.md` rows, plus both canonical completion gates
green in the same build.

**Depends on**: llama.cpp-v90xn.30.

### llama.cpp-v90xn.31

  - **Id:** llama.cpp-v90xn.31
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-d4yko
  - llama.cpp-v90xn.30

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Continuous batching + selected-expert row aggregation for GPT-OSS TG, building on
the landed kernel-level grouping and d4yko's server-formation verification.

**Design**:
- Scaffolding exists but is diagnostic-only: `ggml_sycl_moe_row_agg_log_descriptor`/
  `moe_layer_grouped_route_view` (`ggml-sycl.cpp:64369-64629`), gated by
  `ggml_sycl_moe_row_agg_debug_enabled()` — no actual multi-sequence dispatch logic yet.
- No server-side multi-sequence decode harness exists to validate row aggregation — build one
  (or extend the harness d4yko needs anyway for its own server-formation verification) rather
  than duplicating harness work across both tickets.
- Needs `llama.cpp-hsx04` (the TG-XMX executor referenced under v90xn.30) landed first — row
  aggregation across sequences only makes sense once the underlying per-sequence executor is
  itself correct and fast.
- Acceptance criteria reference old tg-batch numbers (26/46/73/141/210 tok/s FA-off) — these
  predate the current architecture; re-measure fresh on current B50/B70 rather than gating
  against them. Note `--tg-batch` is not a real CLI flag on any binary at HEAD (see
  llama.cpp-d4yko's addendum) — scale concurrency via concurrently issued decode requests,
  not a batch-size flag.

**Bar**: a working multi-sequence continuous-batching path with measured throughput scaling at
1/4/8/16 concurrently issued decode requests on current B50/B70, both canonical gates green at
concurrency=1 (single-stream correctness must not regress even though this ticket's focus is
the batched case).

**Constraints**: Ruling 4 (row aggregation across sequences must still respect per-expert
residency — aggregating rows from multiple sequences is not license to prefetch experts
speculatively).

**Acceptance test**: a new/extended multi-sequence llama-server harness run at 1/4/8/16
concurrently issued decode requests on `ONEAPI_DEVICE_SELECTOR=level_zero:1` GPT-OSS 20B
(no `--tg-batch` flag — it does not exist in the binary), reporting tok/s per concurrency
level, plus the GPT-OSS chat gate at concurrency=1.

**Depends on**: llama.cpp-d4yko, llama.cpp-v90xn.30 (via hsx04).

### llama.cpp-4wkt

  - **Id:** llama.cpp-4wkt
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Remove the fail-closed guard on the 3 sliced-coalesced MMVQ kernels
(`mmvq_assert_coalesced_slice_supported`) by applying the same layout_base/global_row addressing
pattern Q6_K already uses.

**Design**:
- Guard confirmed still present and still the only protection at all 4 call sites: definition
  `mmvq.cpp:3853-3863`; call sites `:4554` (mxfp4 coalesced direct), `:21898`, `:22001`, `:22030`.
- Fix shape is already fully specified by a prior groundwork comment (impl-szv8): apply Q6_K's
  `layout_base`/`global_row` pattern to the 3 coalesced kernels, remove the guard, add a source-
  contract gate in its place (a compile-time or structural check that the layout invariant holds,
  not just a runtime abort). Check `#24452` (upstream reordered Q4_K/Q5_K/Q6_K MoE MUL_MAT_ID
  SYCL PR, see epic doc's web findings) before hand-deriving — it may already contain a portable
  reorder pattern for the analogous case.
- This is implementation-complete on the design side; the only missing piece is GPU
  verification (per keep.json design_gaps).

**Bar**: guard removed at all 4 call sites, replaced by a static/structural contract check;
`test-backend-ops` (manual run only, never subagent) passes for the affected quant types with no
new fail-closed aborts; Mistral/GPT-OSS gates unregressed.

**Constraints**: Ruling 8 (this is standard-SYCL dequant addressing work, not ESIMD — stay
within that lane). Ruling 5 (the new addressing must still respect the materialized layout the
planner chose, not introduce a second addressing convention).

**Acceptance test**: manual `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops`
(never in a subagent/background task — memory-exhaustion hazard) plus the Mistral completion
gate; the lead runs this.

**Depends on**: none — ready to implement.

### llama.cpp-eju9

  - **Id:** llama.cpp-eju9
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Measure whether the XMX threshold's intended bump to 1024 (from the current default
64) actually wins on current hardware/baselines — the bump was specified but never took effect
and was never re-measured.

**Design**:
- `43d04b327` already fixed the record-correctness half: the initializer is now a deliberate
  fail-closed 0, and the `sycl_env_settings[]` table row (`ggml-sycl.cpp:23082`, current
  default 64) is the sole source of truth.
- Build precondition, not optional: `g_ggml_sycl_xmx_threshold` lives inside
  `#ifdef GGML_SYCL_XMX_GEMM` (`ggml/src/ggml-sycl/xmx-dispatch-gate.hpp:34`), and
  `ggml/src/ggml-sycl/CMakeLists.txt:54` defaults that CMake option OFF. On a standard build,
  `GGML_SYCL_XMX_THRESHOLD=1024` changes nothing — both A/B arms must be built with
  `-DGGML_SYCL_XMX_GEMM=ON -DGGML_SYCL_MMQ_XMX=ON` or the run is vacuous.
- Positive control required: before trusting a null result, confirm the value is actually read
  (e.g. log the resolved threshold at init, or verify a deliberately extreme value like
  `GGML_SYCL_XMX_THRESHOLD=999999` visibly changes dispatch) so an all-OFF build can't
  silently report "no difference" as if it were a real measurement.
- Once the build precondition is confirmed live, run the interleaved paired A/B (per the
  standing A/B-must-be-interleaved rule) with both threshold values on the same build, same
  host state.

**Bar**: if 1024 wins PP by a real margin beyond B50's ~1% noise floor / B70's ~10% TG noise
floor AND the completion gate stays correct, change the table row default from 64 to 1024 and
update `docs/backend/sycl-env-vars.md`; otherwise close as measured-no-win with the recorded
numbers.

**Constraints**: Ruling 9 (any default change needs both canonical gates green in the same
build before landing).

**Acceptance test**: rebuild with `-DGGML_SYCL_XMX_GEMM=ON -DGGML_SYCL_MMQ_XMX=ON` for both
arms; confirm the threshold is read (positive control, see Design); then
`ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m
/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -r 5` interleaved between threshold=64 and
threshold=1024 (env override), repeated on B50; plus Mistral completion gate at each threshold.

**Depends on**: none — ready to run; needs-gpu, lead executes.

### llama.cpp-ndzz1

  - **Id:** llama.cpp-ndzz1
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-v90xn.31

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Document concurrent-decode batching as a throughput lever distinct from this epic's
single-stream bar, since `docs/backend/SYCL.md` currently has no such section at all.

**Design**:
- Confirmed gap: `search_text` in `docs/backend/SYCL.md` returns 0 matches for TG-batching
  content.
- Per ruling 11, liveness must be checked before documenting: `--tg-batch` is NOT a real flag on
  any binary at HEAD — it exists nowhere in `tools/` or `common/arg.cpp` (llama-bench's only batch
  knob is `-b`/`--batch-size`, `tools/llama-bench/llama-bench.cpp:606`), and survives only in
  shell harnesses (`scripts/sycl-moe-regression-harness.sh:149`,
  `scripts/sycl-moe-roofline-harness.sh:329`, `AGENTS.md:85`). Do not document it as a live
  llama-server/llama-bench CLI flag; document the mechanism (server-level concurrent decode
  request batching, per llama.cpp-d4yko/v90xn.31) and, if a CLI knob exists once that work lands,
  name whatever flag actually ships, not `--tg-batch`.
- Doc-only task, no code dependency — but wait for `llama.cpp-v90xn.31` (continuous batching +
  row aggregation) to land or at least produce fresh numbers before writing the numeric examples,
  so the doc doesn't ship the same stale 26/46/73/141/210 tok/s figures v90xn.31 itself is asked
  to replace.
- Cover: what concurrent-decode batching does, when it helps (concurrent decode streams) vs when
  it doesn't (single-stream, which is this epic's own primary target and is NOT what batching
  improves), and a worked example table with current B50/B70 numbers once v90xn.31 supplies them.
- Cross-reference this epic's Bar section so a reader understands concurrent-decode batching is a
  *different* lever from the bandwidth-roofline bar (concurrent throughput vs single-stream
  latency), not a substitute for it.

**Bar**: `docs/backend/SYCL.md` gains a concurrent-decode-batching section with a worked example
using current measured numbers and the actual CLI mechanism (not a fabricated flag name),
cross-linked from `docs/backend/sycl-perf-baselines.md`.

**Constraints**: none code-facing — documentation accuracy is the only bar (must not misstate
single-stream vs batched throughput as interchangeable, and must not name a flag that doesn't
exist in the binaries).

**Acceptance test**: manual doc review confirming (a) the section exists, (b) it names only
CLI mechanisms verified live at HEAD (per ruling 11), (c) `grep -c 'tg-batch'
docs/backend/SYCL.md` is either 0 (if the section describes the mechanism without inventing a
flag name) or >0 only if v90xn.31 actually landed a flag by that name — verify before citing it.

**Depends on**: llama.cpp-v90xn.31 (for current numbers and the actual CLI mechanism — can draft
prose in parallel).

### llama.cpp-zb27

  - **Id:** llama.cpp-zb27
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: PERF-EPIC E2 — build the capability-derived route-default rule as a pure function
with a matrix test, replacing per-card special-casing.

**Design**:
- Dep `llama.cpp-t9x5` closed 2026-08-22 (E1 probe API landed, commits `3b36734d5`/`d563c19fa`/
  `12de843ba`) — this ticket is unblocked and unstarted ("released untouched at session pause,
  no edits made" per keep.json), ready to pick up.
- Function signature and test file are already specified in the plan doc:
  `moe_pp_route_default_input` + `ggml_sycl_moe_pp_batched_default_from_caps`, test file
  `tests/test-sycl-moe-route-default.cpp` — implement to that spec, don't re-derive it.
- The owner ruling against per-card tables is already embedded in the ticket text — the function
  must take queried capability facts (XMX support, tile shape, TOPS/bandwidth) as input, never a
  board-name switch (this is Ruling 10's device-selection-belongs-to-oneAPI principle applied to
  route defaults, not just device enumeration).
- This is the function llama.cpp-cv8w (E3) will wire into the actual policy default — keep the
  interface stable once cv8w starts consuming it.

**Bar**: `moe_pp_route_default_input`/`ggml_sycl_moe_pp_batched_default_from_caps` implemented
and unit-tested via `tests/test-sycl-moe-route-default.cpp` (pure function, no GPU needed for
this ticket's own test — a CPU-only matrix test over capability inputs).

**Constraints**: Ruling 10 analog (no board-name branches — capability-derived only).

**Acceptance test**: `ctest --test-dir build -R '^test-sycl-moe-route-default$' --output-on-failure`
(pure-Python/CPU-only gate style test per the plan doc, safe at any parallelism).

**Depends on**: none — ready to implement, unblocked.

### llama.cpp-zw4b

  - **Id:** llama.cpp-zw4b
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Confirm whether `GGML_SYCL_MOE_GROUPED_DECODE=1` still fails the first decode batch
(res=-3) on GPT-OSS 20B at current HEAD, or whether the many MoE-route commits since the
2026-08-17 original repro (including the unpj restoration work) have already fixed or obsoleted
it.

**Design**:
- `moe_grouped_decode_candidate_env_enabled()` (`ggml-sycl.cpp:26608`) confirms the env var is
  still live and read; no fix commit found since 2026-08-17; the original repro has not been
  re-run against current HEAD (post b10630 merge).
- This is a needs-gpu ticket: reproduce fresh, and trace which `llama_decode` call site actually
  returns -3 rather than assuming it's a kernel fault — the failure could equally be a planner
  rejection, an allocation failure, or a genuine kernel bug, and each has a different fix.
- Coordinate timing with llama.cpp-unpj: if T1's device-ID decode restoration changes the
  grouped-decode candidate path's behavior, re-run this repro AFTER T1 lands, not before, to
  avoid chasing a bug unpj's restoration already fixes as a side effect.

**Bar**: either a fresh repro confirming res=-3 still occurs with a specific call-site
attribution (file:line of the returning function), or confirmation it's fixed (cite the commit)
and this ticket closes.

**Constraints**: Ruling 9 (an env-gated candidate path failing outright is a correctness bug, not
a performance question — must be fixed or the env var must stay refused/gated before any decode
throughput work relies on it).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_MOE_GROUPED_DECODE=1
./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 0 -n 128 -fa 1 -r 2` — expect rc=0
and tg128 printed (control arm without the flag: rc=0, tg128~20-32); rc=1 with
`[MOE-DECODE-CANONICAL] ... res = -3` in stderr confirms still-broken.

**Depends on**: none directly, but should be re-run after llama.cpp-unpj T1 lands for a clean
attribution.

### llama.cpp-y0it

  - **Id:** llama.cpp-y0it
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent**: Fix the DOWN-artifact reuse layout mismatch where the writer hardcodes
`GGML_LAYOUT_SOA` while the reader's validity check requires an exact layout match — currently a
performance waste (dead-end fused attempt + arena churn), not the data-loss bug it was originally
filed as (already corrected per the ticket's own title).

**Design**:
- Confirmed live: `mxfp4_moe_tg_reuse_store` (`mmvq.cpp:2144`, actual line may drift — grep at
  time of work) hardcodes `GGML_LAYOUT_SOA` as `cache.layout` for the DOWN artifact writer, while
  `mxfp4_moe_tg_reuse_can_use` (`mmvq.cpp:2049`/`:2031`) requires an exact layout match to
  consider the cached artifact reusable — the mismatch means the reuse check always fails,
  costing a dead-end fused attempt and extra arena allocation/free churn per occurrence, but
  never wrong output (the exact-match gate correctly refuses the mismatched cache and re-does the
  work instead of returning stale data).
- Fix requires deciding whether to change what the writer tags (fix the writers,
  `mxfp4_moe_tg_store_down_q8_soa_artifact`/`mxfp4_moe_tg_publish_q8_soa`) or relax the reader's
  exact-match on the DOWN path specifically — both touch a hot path shared with gate/up reuse, so
  either change needs to verify it doesn't loosen gate/up's own correctness gate as a side effect.
- Low priority/low frequency by design (P3) — this is arena churn, not a throughput blocker on
  the epic's critical path; sequence it after the higher-priority decode-path work above lands,
  since a landed 30ak7.19.17 (down-I8 residency) may change what the "correct" layout tag even is.

**Bar**: reuse check succeeds for DOWN artifacts when the underlying data is genuinely
layout-compatible; measurable reduction in dead-end fused attempts / arena alloc-free churn per
decode token (via allocator event counting, not just code inspection).

**Constraints**: Ruling 5 (R2 — the fix must not create a second hand-maintained layout tag path
diverging from whatever 30ak7.19.17 settles the down artifact's canonical layout to be).

**Acceptance test**: allocator/arena event count for DOWN-artifact reuse attempts before/after,
on `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf
-p 512 -n 128 -fa 1`; GPT-OSS chat gate unregressed.

**Depends on**: sequence after llama.cpp-30ak7.19.17 (down-I8 residency) for a stable layout
target; not a hard blocker.

## New tasks

### Build bytes-streamed-per-token instrumentation for GPT-OSS MXFP4 decode

  - **Title:** Build bytes-streamed-per-token instrumentation for GPT-OSS MXFP4 decode
  - **Description:** ## Design addendum (triage 2026-09-01)

**Intent**: Build the missing device-event instrumentation the epic's primary bar depends on —
achieved bytes/token × tok/s measured against the hardware bandwidth roofline (0.8 × 224 GB/s
B50, 0.8 × 608 GB/s B70). `scripts/sycl-moe-roofline-harness.sh` already emits
`weight_gb_per_token`/`effective_weight_gbps`/`theoretical_bw_gbps`/`roofline_tg_tok_s`
(line 88) — reconcile with that rather than treating this as greenfield: either (a) extend
that harness's own accounting into a device-event pass if its figure is derived from static
model-size math rather than actual per-dispatch bytes read, or (b) if it already measures
real bytes read, this ticket narrows to wiring it into the zb27/cv8w flip-gate consumers
instead of building a parallel instrument. State which case applies before implementing.
Note the harness currently invokes the dead `--tg-batch` flag (see llama.cpp-d4yko's
addendum) — fix or drop that invocation as part of touching the harness.

**Design**:
- Extend the existing `GGML_SYCL_MXFP4_TG_PROFILE` device-event instrumentation (already proven
  reliable by d0bp's root-cause work) to also sum bytes read per dispatched kernel — expert
  weight bytes (from `grouped_experts_device`/`grouped_offsets_device` row counts × per-row
  bytes for the resolved (type,layout)), dense-layer weight bytes, and KV-cache bytes, per token.
- Report both the raw bytes/token figure and the derived (bytes/token × tok/s) as a fraction of
  the queried device's theoretical peak bandwidth (already available via `caps.global_mem_size`
  and driver-reported bandwidth, or the DESIGN-BRIEF's documented 224/608 GB/s figures as a
  fallback constant per device).
- This is a read-only instrumentation pass — it must not change dispatch behavior, only observe
  it, and must itself avoid host waits in the hot path (gate behind the existing env-var pattern,
  same as `GGML_SYCL_MXFP4_TG_PROFILE`).
- Land this BEFORE llama.cpp-zb27/cv8w's flip-gate work needs a bandwidth-fraction number to
  gate on, and before llama.cpp-22sp's final E2E validation, since both ultimately need to report
  against the 80%-roofline bar, not just tok/s.

**Bar**: a single env-gated instrumentation flag that prints, per `llama-bench` run, bytes/token
and the resulting fraction of theoretical peak bandwidth for GPT-OSS 20B MXFP4 tg128 on both
B50 and B70.

**Constraints**: Ruling 6 (instrumentation must not introduce a host wait in the default hot
path — env-gated, same pattern as existing profiling flags). Ruling 10 / llama.cpp-zb27's
no-board-name-branch rule (device bandwidth constants keyed on queried capability, not a
board-name switch or a captured device name string — even though the DESIGN-BRIEF's own
224/608 GB/s figures are card-specific, key them off `caps.global_mem_size` and other queried
capability facts, never a device name string compared against a literal).

**Acceptance test**: `ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_MXFP4_BW_PROFILE=1
./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 0 -n 128 -fa 1 -r 2 -v` prints a
bytes/token and bandwidth-fraction line; repeat on `level_zero:0` for B70.

**Depends on**: llama.cpp-unpj (a degraded route's bytes/token figure is meaningless as a
baseline — build this after or alongside T1, not before).
  - **Priority:** 1
### Depends on

  - llama.cpp-unpj

## Web findings

### Item 1

  - **Claim:** Upstream landed PR #24676: made the MoE-prefill async-memcpy source buffer persistent to fix a use-after-free beyond function scope in the SYCL backend.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/24676
  - **Date:** 2026
  - **Impact:** Neither obsoletes nor replaces any member task; confirms upstream is independently hardening MoE-path buffer lifetime for async SYCL copies (a prefill bug, not decode). Worth a diff-read before unpj's T1/T2 land in case the same buffer-lifetime pattern applies to the restored decode route.

### Item 2

  - **Claim:** Upstream PR #24452 added reordered Q4_K/Q5_K/Q6_K MoE MUL_MAT_ID support for the SYCL backend.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/24452
  - **Date:** 2026
  - **Impact:** Directly relevant to llama.cpp-4wkt (Q6_K coalesced-slice fail-closed guard still present, no Q6_K-pattern rewrite landed). Since the fork merged b10630 on 2026-08-26, this PR (merged 2026-06-16) may already be vendored — llama.cpp-4wkt must grep the tree for the reorder pattern before assuming it needs to be hand-derived, and only then read the PR — may shorten the implementation from 'derive a new pattern' to 'port an existing one', though the fork's own mmvq_assert_coalesced_slice_supported guard has no direct upstream analog.

### Item 3

  - **Claim:** Intel compute-runtime 26.31.39395.13 (the currently-loaded driver per CLAUDE.md) is an enablement/maintenance release: Crescent Island device-ID prep, Nova Lake Xe3P groundwork, Level Zero API bumped to 1.17 for listed platforms; Battlemage (B70/B50, this host's cards) retained at production status.
  - **Url:** https://github.com/intel/compute-runtime/releases
  - **Date:** 2026-08 (26.31.39395.13)
  - **Impact:** No decode-bandwidth-relevant changelog entry found for this driver version. The epic's driver pin is not stale relative to a materially faster release currently available — no action item for this epic.

### Item 4

  - **Claim:** The CUDA backend's own token-generation optimization strategy (am17an, 2026) documents that batch-1 decode is memory-bandwidth-bound, kernel fusion is the primary lever, and CUDA graph capture isolates per-kernel launch overhead by replaying a captured sequence once recorded.
  - **Url:** https://am17an.bearblog.dev/new-post/
  - **Date:** 2025-12-01
  - **Impact:** Architecturally the same bet as this epic's persistent-TG graph-replay work (hmbk9/nubvg) and the unpj device-ID decode restoration (graph-recordable dispatch). No code ports directly across backend APIs, but corroborates that fusion+replay is the correct mechanism for this class of problem, not a detour — supports keeping the epic's current work breakdown ordering (correctness-then-graph-replay-then-fusion) rather than reprioritizing.