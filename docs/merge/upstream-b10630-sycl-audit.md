# Upstream SYCL Audit — Consolidated Ledger (b10630)

Consolidates `docs/merge/upstream-b10630-sycl-audit-a.md` (commits 1–25 of 49) and
`docs/merge/upstream-b10630-sycl-audit-b.md` (commits 26–49 of 49) — the full set,
classified per the rubric in `docs/plans/2026-08-25-phase-c-upstream-merge.md`
Task 6/7 (N/A → superseded → port-candidate, in that order; diff evidence, not
commit titles).

## Provenance

**Part A** (commits 1–25) was enumerated via
`git rev-list --reverse 81ff7abe5..b10630 -- ggml/src/ggml-sycl | head -25`.
Merge-base: `81ff7abe5`. Upstream target: tag `b10630` (remote `ggml-org`). Fork
side: `master @ d37e4cedb269` ("fix(merge-guards): symmetric rc-check +
non-vacuity on both safety-net halves") — the commit `master` pointed to when
Part A's fork-state greps (source line numbers, symbol presence/absence,
allocation-pattern counts) were run. Pinned because `master` keeps moving under
a shared checkout; every Part A fork-state citation is scoped to this SHA unless
stated otherwise.

**Part B** (commits 26–49) was enumerated via
`git rev-list --reverse 81ff7abe5..b10630 -- ggml/src/ggml-sycl | tail -n +26`.
Merge-base: `81ff7abe5`. Upstream target: tag `b10630` (remote `ggml-org`). Fork
side: `master @ 24bebf19a4b5` ("docs(merge): audit ledger A -- reclassify entry
23 DSv4 ops as port-candidate post-merge") — the commit `master` pointed to
when Part B's `ggml-sycl.cpp`/`cpy.cpp`/`mmvq.cpp`/`set_rows.cpp`/
`dmmv-esimd.hpp` greps and cites were derived, i.e. the parent of Part B's own
first commit (`1f9901994820`). Pinned because `master` keeps moving under a
shared checkout; every Part B citation is scoped to this SHA unless stated
otherwise. Every key line cite this pin covers holds verbatim at the pinned SHA
and remains unchanged at every later tip Part B was amended from
(`2ef0681c4386`, `367f3a7e47fd`): `ggml-sycl.cpp:98706`
(`ggml_sycl_mul_mat_type_supported` definition), `:98745` (`ADD_ID`/
`MUL_MAT_ID` unconditional `return true`), `:98905` (the `MUL_MAT` allowlist
call site), `:98985` (the `CPY` same-type-contiguous fast path); `cpy.cpp:842`
(`ggml_cpy_f32_iq4_nl_sycl` definition) and its single-lane launch body at
`:861-863`; and the five over-launch same-type passthrough sites
(`ggml_cpy_q8_0_q8_0`, `ggml_cpy_q5_0_q5_0`, `ggml_cpy_q5_1_q5_1`,
`ggml_cpy_q4_0_q4_0`, `ggml_cpy_q4_1_q4_1`, each `num_blocks =
ceil_div(ne, SYCL_CPY_BLOCK_SIZE)`) now at `cpy.cpp:1176/1203/1230/1258/1285`.
No source file Part B cites has changed between the pinned SHA and HEAD — only
docs/scripts commits from parallel tracks have landed in between.

Several commits in Part B touch only trivially inside `ggml/src/ggml-sycl`
while the substantive change is a cross-backend/core-ggml feature (load-mode
auto, `ggml_rope_set_offset`). Per the Task 7 gotcha, a new file is classified
against the fork subsystem it would slot into; the mirror case — a core-scoped
commit whose SYCL touch is a one-line, meaningless-without-the-core-plumbing
hunk — is classified N/A as out of this SYCL-only audit's scope, with a
pointer to the core-ggml brief (`docs/merge/briefs/core-ggml.md`) that already
covers the substantive change.

## Ownership screen

The rubric includes an ownership screen (owner directive, mid-flight): every
port-candidate is checked against the fork's canonical memory-ownership
contract (CLAUDE.md "SYCL Memory Ownership";
`docs/design/sycl-canonical-memory-architecture.md`;
`docs/backend/sycl-memory-design.md`) — raw `sycl::malloc_*`/TLSF/side-cache/
pool/scratch allocation outside unified-cache code is never
port-candidate-as-written (re-expression through
`unified_allocate`/`unified_allocate_owner` → `mem_handle` is required
instead); raw device pointers as cache/identity keys get the same treatment;
forced eviction/reap/zone-reset-style reclamation is N/A by design. Each
port-candidate below names the ownership surface (or "none touched") its
landing zone would consume.

**49 commits: 27 port-candidate / 16 superseded / 6 n-a**

(Cross-checked independently against the Class line of every entry in both
source parts at consolidation time: Part A tallies 16 port-candidate / 9
superseded / 0 n-a = 25; Part B tallies 11 port-candidate / 7 superseded / 6
n-a = 24. Sum: 27/16/6 = 49. No discrepancy from the header counts already
recorded in the two source files.)

## Tracker

Epic: **llama.cpp-1yr6** — "EPIC: Phase C upstream-SYCL ports (post-merge)".
One ticket was filed per port-candidate entry below (27 total), plus two extra
cross-cutting tickets from the "Cross-cutting items" section (29 tickets
total, all blocking the epic). Special dispositions: entry A-23 (DSv4) is
BLOCKED-ON-MERGE (`llama.cpp-pmzl` depends on `llama.cpp-t9l9`, the campaign's
merge-landing task); entries B-10 and B-16 are CORE-GGML WAVE COMPANIONS that
land with the merge itself, not post-merge (`llama.cpp-8vwp` and
`llama.cpp-hkuh` depend on `llama.cpp-90ns`, the campaign's wave-3 task) —
filed here for tracking only. Ticket IDs are recorded next to each entry
below.

## Cross-cutting items

🔶 **CARRY TO TRACKER (general, not scoped to any single entry's files):** 18 live
`ggml_sycl_pool_alloc`/`ctx.pool()` call sites remain SYCL-backend-wide — a candidate unified-cache
migration backlog. Verified per file (`cat <file> | grep -cE
'ggml_sycl_pool_alloc|ctx\.pool\(\)'`, positive-controlled against
`fattn-onednn.cpp`'s already-migrated 0) and confirmed each hit is a real call
site, not the template's own definition (`common.hpp:3215-3259` is the
`ggml_sycl_pool_alloc` struct/ctors/dtor/deleted members — infrastructure, not
a site to migrate, and should be removed last, once no callers remain):
- `ggml-sycl.cpp` (13): lines 41129, 41257, 41310, 41323, 41342, 41343, 52068, 58718,
  58719, 58893, 58894, 59989, 59990
- `conv3d.cpp` (3): lines 84, 104, 105
- `cross_entropy_loss.cpp` (2): lines 124, 175

Two further mentions are comments, not call sites, and don't count toward the 18:
`mmvq.cpp:15998` (documents a graph-mode workaround) and `fattn-buffers.hpp:54` (documents
a replacement wrapper).

**Ticket: llama.cpp-7phf** ("[ports] pool-alloc migration backlog — 18 sites onto unified_allocate/mem_handle").

(a) Entries B-10 (`1692f9e50bb2`) and B-16 (`749f688fcaa4`) are core-ggml-wave
companions, not Phase C epic tickets — both must land with the merge's
core-ggml wave alongside their respective `ggml.h` contract changes
(`ggml_ssm_scan()`'s new `K` param, `ggml_rope_set_offset()`), not deferred to
post-merge Phase C. **Tickets:** `llama.cpp-8vwp` (B-10), `llama.cpp-hkuh`
(B-16) — both filed for tracking only, depend on `llama.cpp-90ns`.

(b) Entry B-6 (`154d57af3eec`) removes one site (`ggml-sycl.cpp:41323`) from
the cross-cutting `ggml_sycl_pool_alloc`/`ctx.pool()` migration backlog (18
live call sites SYCL-backend-wide) — after this port lands, that backlog's
`ggml-sycl.cpp` count drops from 13 to 12. **Ticket:** `llama.cpp-ag2d`.

(c) Filed alongside this backlog, from the wave-1 build-system ruling recorded
at `llama.cpp-oxro` comment c-y8ru: the `-fsycl-max-parallel-link-jobs`
AOT-block port (entry A-10, `8e8681e0e208`, deferred here because
`GGML_SYCL_DEVICE_ARCH` is unset in the canonical build and the upstream block
is inert on this fork today) gets its own precisely-scoped follow-up ticket —
port the parallel-link-jobs cache var onto the fork's `--offload-arch` form;
inert until `GGML_SYCL_DEVICE_ARCH` is set. **Ticket:** `llama.cpp-17zg`.

---

## Part A: commits 1–25 of 49

**25 commits: 16 port-candidate / 9 superseded / 0 n-a**

### 1. `e474bba7af8b` — sycl: add Q2_K to DMMV reorder path (#25064)
**Files:** ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Adds a GPU-side SOA reorder kernel + `reorder_qw_q2_k` for Q2_K, keyed off `ggml_tensor_extra_gpu::optimized_feature.reorder`. The fork's `ggml_sycl_supports_reorder_dmmv()` (`ggml-sycl.cpp`) returns `true` only for `GGML_TYPE_Q4_0` — Q2_K/Q3_K/Q4_K/Q5_K/Q6_K reorder-dmmv support that upstream still has was already dropped by the fork's own rewrite of the reorder/SOA-layout system (`layout_policy`, CPU-side `reorder_q4_0_cpu`, "One layout per weight" owner ruling). Adding upstream's Q2_K variant would extend a mechanism the fork deliberately narrowed.

### 2. `efb3036c1826` — sycl: add fused top-k MoE (#25217)
**Files:** docs/backend/SYCL.md, ggml/src/ggml-sycl/backend.hpp, ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/topk-moe.cpp (new), ggml/src/ggml-sycl/topk-moe.hpp (new)
**Class:** superseded
**Why:** Adds a generic `ggml_sycl_fuse()` graph-fusion hook called from `ggml_backend_sycl_graph_compute_impl`'s node loop, plus a specific ARGSORT+VIEW+GET_ROWS MoE-routing fusion. The fork has neither `topk-moe.{cpp,hpp}` nor any `ggml_sycl_fuse` call site — its MoE routing is a wholesale rewrite (`moe-route-table.hpp`, `moe-sort.hpp`, `moe-resolved-batch.hpp`, `moe-control-plan.cpp`, `moe-mmid-workspace.cpp`, etc.), and graph-node dispatch is already deeply customized for cache/placement bookkeeping. Re-adding upstream's fusion shortcut would bypass that bookkeeping.

### 3. `c1063ac9d75f` — sycl: set fattn_vec_nthreads to 256 for Battlemage (#25205)
**Files:** ggml/src/ggml-sycl/fattn-vec.hpp
**Class:** port-candidate
**Why:** Detects Battlemage/Lunar Lake (`gpu_arch::intel_gpu_bmg_g21/g31/lnl_m`) and runs the flash-attention vec kernel with a 256-thread work group instead of 128. The fork's `fattn-vec.hpp` has no per-arch dispatch at all — `nthreads` is the flat compile-time macro `FATTN_VEC_NTHREADS = 128` (`fattn-common.hpp:22`) — so this is a real, unapplied tuning fact for exactly this fork's two cards (B70 = bmg_g31, B50 = bmg_g21). Must land together with entry 22 (`eef5f3e3430a`), a required companion fix for a template-instantiation bug this change introduces.
**Landing zone:** `ggml/src/ggml-sycl/fattn-common.hpp` (replace the flat `FATTN_VEC_NTHREADS` macro with an arch-conditional helper) + the launch site in `ggml/src/ggml-sycl/fattn-vec.hpp` that currently hardcodes `nthreads` from the macro.
**Ownership surface:** none — pure launch-geometry constant (thread/warp count), no allocation of any kind.
**Ticket:** llama.cpp-ada5

### 4. `32b741c336de` — [SYCL] Flash Attention with XMX engine via oneDNN (#25222)
**Files:** ggml/src/ggml-sycl/backend.hpp, ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/fattn-onednn.cpp (new), ggml/src/ggml-sycl/fattn-onednn.hpp (new), ggml/src/ggml-sycl/fattn.cpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** This is the upstream origin of the oneDNN SDPA flash-attention path (`g_ggml_sycl_fa_onednn`, BMG gate, `device_count > 1` wait). The fork's `fattn-onednn.cpp` is 1189 lines (vs. upstream's much smaller original) and has already grown far beyond this commit — `MATERIALIZE_REQUIRED` layout planning for nc≠D GQA, an `sdpa_partition_cache`, mem_handle-backed USM scale buffers with `cohort_id` caching. The fork's version is a superset built on top of, and past, exactly what this commit introduces.

### 5. `0e148a573f0c` — sycl: Increase minimum buffer size for USM system allocations (#25525)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Raises `check_usm_system()`'s threshold from 1 GiB to 4 GiB before falling back to `aligned_malloc_host` for USM system (host-backed) allocation. Neither `check_usm_system`, `g_ggml_sycl_usm_system`, nor `aligned_malloc_host` exist anywhere in the fork — this whole size-threshold heuristic in `ggml_backend_sycl_buffer_type_alloc_buffer` was replaced by the unified cache's tiered placement (device VRAM → pinned host → mmap, budget-driven). The canonical memory contract also forbids raw `sycl::malloc_host`-style allocation outside the unified cache, which is what this hunk is patching.

### 6. `22b208b1cacb` — sycl: implement xielu op (#25550)
**Files:** ggml/src/ggml-sycl/element_wise.cpp, ggml/src/ggml-sycl/element_wise.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** Adds `GGML_UNARY_OP_XIELU` support via the existing `dispatch_ggml_sycl_op_unary` helper — the same dispatch pattern the fork's own `ggml_sycl_op_clamp`/`ggml_sycl_elu` already use. `GGML_UNARY_OP_XIELU` already exists in the fork's `ggml/include/ggml.h`, but there is no `xielu`/`XIELU` symbol anywhere in the fork's SYCL backend — any model using xIELU activation currently has no SYCL kernel for it. Small, self-contained, no touched subsystem is rewritten.
**Landing zone:** `ggml/src/ggml-sycl/element_wise.cpp`/`.hpp` (new `ggml_sycl_op_xielu`/`ggml_sycl_xielu`, mirroring `ggml_sycl_op_clamp`) + the `GGML_UNARY_OP_XIELU` case in `ggml_sycl_compute_forward`'s unary-op switch in `ggml-sycl.cpp`.
**Ownership surface:** none — the dispatch helper consumes already-resolved src/dst data pointers from existing tensor buffers; no new allocation, cache, or pool of any kind.
**Ticket:** llama.cpp-tp03

### 7. `ae9291e16b97` — sycl : support kernel type fp16 for conv2d_dw (#25653)
**Files:** ggml/src/ggml-sycl/conv2d-dw.cpp
**Class:** port-candidate
**Why:** Adds an F16 kernel-tensor code path to depthwise conv2d. The fork's `conv2d-dw.cpp` asserts `kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32` unconditionally — F16 conv2d_dw is simply unimplemented, not superseded by anything; this op sits outside every subsystem the fork rewrote (mul_mat/fattn/memory/graph/MoE).
**Landing zone:** `ggml/src/ggml-sycl/conv2d-dw.cpp` (the type-assert + kernel dispatch).
**Ownership surface:** none — operates in-place on the existing input/kernel/dst tensor buffers; no allocation.
**Ticket:** llama.cpp-1kcx

### 8. `d3fba0c79db8` — sycl : fix get_rows Q2_K, Q4_K, Q5_K (#25656)
**Files:** ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/getrows.cpp
**Class:** port-candidate
**Why:** Adds `dequantize_{q2_K,q4_K,q5_K}_f32` helpers and wires `get_rows` dispatch for those three types. The fork's `getrows.cpp` dispatch switches (~lines 2054, 2829) cover F16/F32/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K only — Q2_K, Q4_K and Q5_K are absent from `get_rows` entirely, and none of the three `_f32` dequantize helpers exist in `dequantize.hpp`. This is exactly the rubric's "dequant tables in dequantize.hpp... shared types" port-candidate category — a real coverage gap, not a rewritten path.
**Landing zone:** `ggml/src/ggml-sycl/dequantize.hpp` (add the three `_f32` functions) + `ggml/src/ggml-sycl/getrows.cpp` (add the three `case GGML_TYPE_*` arms to both dispatch switches).
**Ownership surface:** none — pure per-element dequantize math reading existing weight-buffer bytes; no allocation.
**Ticket:** llama.cpp-tjt9

### 9. `0bd0ec60998d` — sycl: fix row calculation when K_QUANTS_PER_ITERATION is 1 (#25690)
**Files:** ggml/src/ggml-sycl/dmmv.cpp
**Class:** port-candidate
**Why:** Fixes an off-by-one (`row > nrows` → `row >= nrows`) across the base (non-reorder) Q2_K/Q3_K/Q4_K/Q6_K DMMV kernels, plus a second-half fix that only applies to the *reorder* Q5_K kernel. Verified against the fork's current source: `dmmv.cpp:1981` (q2_k), `:2089` (q3_k), and `:2203` (q4_k) all still read `if (row > nrows)`, while q6_k (`:2475`) is already `>=` — three of the four sites genuinely need the same one-character fix. Upstream's diff also touches the non-reorder Q5_K kernel (`nrows` parameter, multi-row calculation, `>=` guard, KQPI-style launcher), but the fork's own non-reorder Q5_K kernel already distributes both `im` halves across different threads in the work-group (`:2366-2374`) and its launcher (`:2941-2952`) already grids exactly `nrows` groups with `local_range(1)==1`, making an `nrows` guard structurally inert there; upstream's separate im-loop restructure lands in the *reorder* Q5_K kernel, unreachable in this fork (`ggml_sycl_supports_reorder_dmmv()` returns true only for `GGML_TYPE_Q4_0`).
**Landing zone:** `ggml/src/ggml-sycl/dmmv.cpp`, the three `if (row > nrows)` guards in `dequantize_mul_mat_vec_q2_k` (`:1981`), `_q3_k` (`:2089`), and `_q4_k` (`:2203`) — change to `>=`. No change needed to Q5_K (non-reorder is correct and its launcher makes the guard inert; the reorder-side fix is unreachable).
**Ownership surface:** none — a one-character bounds-comparison fix over existing device buffers; no allocation.
**Ticket:** llama.cpp-0w52

### 10. `8e8681e0e208` — sycl(build): parallelize ocloc invocations (#25903)
**Files:** ggml/src/ggml-sycl/CMakeLists.txt
**Class:** port-candidate
**Why:** Adds `-fsycl-max-parallel-link-jobs=<nproc>` to the `target_link_options` guarded by `if (GGML_SYCL_DEVICE_ARCH)` / `spir64_gen`. In the fork, that exact `GGML_SYCL_DEVICE_ARCH` block (`CMakeLists.txt:836-838`) is scoped to the AMD HIP `--offload-arch` path, not Intel AOT — the fork's own Intel Battlemage AOT uses a separate `GGML_SYCL_BMG_AOT_ENABLED`/`intel_gpu_bmg_g21` block with no parallel-link-jobs flag. The upstream hunk's literal context doesn't exist in the fork, but the underlying idea (parallelize `ocloc` invocations during AOT link) is a real, low-risk build-time win worth re-applying to the fork's own AOT block — build time is an explicit repo pain point (CLAUDE.md: ~10 min with ccache / ~25 without).
**Landing zone:** `ggml/src/ggml-sycl/CMakeLists.txt`, the `GGML_SYCL_BMG_AOT_ENABLED` block's `target_link_options(ggml-sycl ...)` (~line 556-560).
**Ownership surface:** none — build-system/link-flag change, no runtime allocation.
**Ticket:** llama.cpp-achg (see also llama.cpp-17zg, the precisely-scoped follow-up filed from the wave-1 ruling — this port was deferred during wave 1 because `GGML_SYCL_DEVICE_ARCH` is unset in the canonical build).

### 11. `d6b61ac0d361` — sycl: fix use-after-return of the SDPA scale in the oneDNN flash-attention path (#25880)
**Files:** ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/fattn-onednn.cpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Fixes a real hazard (async memcpy of the SDPA scale from a stack local, racing the K/V staging kernels on long contexts) by writing the scale via a captured `single_task` into a per-(device,value) cached USM buffer. The fork's `fattn-onednn.cpp` already allocates the SDPA scale through a `cache_backing_token`-style host-pinned USM buffer keyed by `cohort_id = "onednn_sdpa_scale"` with explicit `throw`s if the buffer isn't host-pinned USM (lines ~838-866) — an independently-built, unified-cache-routed mechanism that structurally avoids the same host-stack-local hazard this commit patches. Re-applying upstream's narrower fix would regress the fork onto a raw `pool_alloc` outside the cache.

### 12. `11b068d06605` — sycl: contiguous fast path + 32-bit index math for unary elementwise ops (#25946)
**Files:** ggml/src/ggml-sycl/element_wise.cpp
**Class:** superseded
**Why:** Adds a contiguous fast path and 32-bit index math to upstream's raw-pointer unary elementwise kernels. The fork's `element_wise.cpp` already uses a different tensor-wrapper abstraction (`ggml_sycl::sycl_tensor`, `src0.raw()`, `.is_contiguous()`) with its own contiguous-fast-path branches at multiple call sites (e.g. lines 685, 714, 748) predating this commit. The upstream hunk's raw kernel signatures and index-math don't correspond to code that exists in the fork's rewritten file.

### 13. `155372596547` — sycl: fuse RMS_NORM + MUL (#26015)
**Files:** ggml/src/ggml-sycl/backend.hpp, ggml/src/ggml-sycl/fusion.cpp (new), ggml/src/ggml-sycl/fusion.hpp (new), ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/norm.cpp, ggml/src/ggml-sycl/norm.hpp
**Class:** superseded
**Why:** Builds a generic two-op graph-fusion detector (RMS_NORM followed by MUL) on top of the same `ggml_sycl_fuse()` mechanism entry 2 introduced — which the fork does not have. The fork's `fused-norm-gemm.hpp` (comment: "Fused RMSNorm + MUL (gamma) + GEMM for Intel Arc GPUs") already fuses RMSNorm+MUL into a *larger* single kernel that also folds in the following GEMM/quantize step — a strict superset of what this commit does, built independently of upstream's generic fusion framework.

### 14. `a2be61dc8794` — [SYCL] Support q2 mul_mat (#26231)
**Files:** ggml/src/ggml-sycl/convert.cpp, ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/vecdotq.hpp
**Class:** port-candidate
**Why:** Adds MMVQ dot-product and dequantize support for `GGML_TYPE_Q2_0`. `GGML_TYPE_Q2_0` (value 42) already exists in the fork's `ggml/include/ggml.h`, but there is no `Q2_0`/`q2_0` symbol anywhere in the fork's `vecdotq.hpp` or `mmvq.cpp` — mul_mat for this type is simply unimplemented. `vecdotq.hpp`/`convert.cpp`/`dequantize.hpp` are explicitly the rubric's shared-table port-candidate files, and mmvq.cpp additions here are purely additive (new `case` arms), not a rewrite of the fork's own MMVQ dispatch.
**Landing zone:** `ggml/src/ggml-sycl/{dequantize.hpp,convert.cpp,vecdotq.hpp,mmvq.cpp}`, paired with entry 16 (`d5d3e05bf8d2`, cpy-side Q2_0 support).
**Ownership surface:** none confirmed in the diff (`grep`'d for `malloc`/`pool_alloc` in `mmvq.cpp`/`convert.cpp` — no hits) — dot-product/dequant/convert code reading existing weight buffers; no allocation.
**Ticket:** llama.cpp-9ppc

### 15. `1c5b89ff6315` — sycl : support dev2dev memcpy by DEV2DEV_MEMCPY_FORWARD (#26234)
**Files:** ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Adds a third `ggml_sycl_dev2dev_memcpy_mode` (`DEV2DEV_MEMCPY_FORWARD`) alongside `SYCL`/`L0`, selected via `g_ggml_sycl_dev2dev_memcpy`. None of `DEV2DEV_MEMCPY_SYCL/L0/FORWARD`, the enum, or `g_ggml_sycl_dev2dev_memcpy` exist in the fork — its device-to-device copy (`dev2dev_memcpy`, `ggml-sycl.cpp:33460`) is a `mem_handle`-based function integrated with the unified cache and the fork's documented no-direct-P2P/host-bounce topology (no direct P2P between the B70 and B50 — re-verified 2026-07-31), not an env-var-selectable mode switch.

### 16. `d5d3e05bf8d2` — [SYCL] support the missed types in cpy (#26005)
**Files:** ggml/src/ggml-sycl/cpy.cpp, ggml/src/ggml-sycl/cpy.hpp
**Class:** port-candidate
**Why:** Adds `cpy_blck_f32_q2_0` (F32→Q2_0 quantize) and `cpy_blck_q2_0_f32` (Q2_0→F32 dequantize), plus their dispatch-table pairing (paired with entry 14's Q2_0 mul_mat support). The fork's `cpy.cpp` dispatch (grep of `case GGML_TYPE_*` pairs) covers F32/F16/BF16/I16/I32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL pairs but has no `Q2_0` arm at all — this is additive coverage in the same shared-conversion-table category, not a rewrite.
**Landing zone:** `ggml/src/ggml-sycl/cpy.hpp` (new `cpy_blck_f32_q2_0`) + `ggml/src/ggml-sycl/cpy.cpp` (new `cpy_blck_q2_0_f32` and the dispatch table), alongside entry 14. (Verified via `git show --stat d5d3e05bf8d2`: `cpy_blck_f32_q2_0` is the inline helper added to cpy.hpp; `cpy_blck_q2_0_f32` is the static helper added to cpy.cpp.)
**Ownership surface:** none confirmed in the diff (`grep`'d `cpy.cpp`/`cpy.hpp` for `malloc`/`pool_alloc` — no hits) — per-block quantize/copy math over existing src/dst buffers; no allocation.
**Ticket:** llama.cpp-922d

### 17. `9d9a6d29f6b9` — SYCL: add oneMKL GEMM flash attention for XMX-accelerated prompt processing (#25025)
**Files:** ggml/src/ggml-sycl/fattn-mkl.cpp (new), ggml/src/ggml-sycl/fattn.cpp, ggml/src/ggml-sycl/fattn.hpp
**Class:** superseded
**Why:** A large (690-line), independent third flash-attention path using oneMKL GEMM to XMX-accelerate long-context prefill, with its own env vars and gating envelope. `fattn-mkl.cpp` does not exist in the fork and is not included from `backend.hpp`. The fork already has two integrated prefill-acceleration paths serving the identical goal — XMX-accelerated prompt processing — via `fattn-xmx-f16-v2.hpp` and the oneDNN SDPA path (`fattn-onednn.cpp`, entry 4's lineage, extended by entry 19), the latter already measured at ~32% PP uplift with FA on, so importing a third GEMM-FA implementation would duplicate functionality the fork already ships and integrates with its own cache/materializer machinery rather than port cleanly.

### 18. `272700b36094` — sycl: fix classification of iGPUs (#26105)
**Files:** ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** Adds `l0_device_type_valid` (so a failed `zeDeviceGetProperties` query no longer silently misclassifies a device as integrated) and makes `ggml_backend_sycl_device_get_type()` return `GGML_BACKEND_DEVICE_TYPE_IGPU` for integrated GPUs instead of always `GGML_BACKEND_DEVICE_TYPE_GPU`. The fork's `ggml_backend_sycl_device_get_type()` (`ggml-sycl.cpp:98570-98573`) still unconditionally returns `GGML_BACKEND_DEVICE_TYPE_GPU`, and neither `l0_device_type_valid` nor `l0_discrete_gpu` exist in the fork's `sycl_device_info` struct. This is distinct from (and complements, not duplicates) the fork's own `host_unified_memory`/`ggml_sycl_device_is_host_unified()` VRAM-budget fix for llama.cpp-403s — that fix scales the *budget calculation*; this fix would let ggml-core's backend scheduler itself see the iGPU as `IGPU` rather than a generic `GPU`, which is a different, real gap.
**Landing zone:** `ggml/src/ggml-sycl/common.hpp` (`sycl_device_info` struct, add `l0_device_type_valid`) + `ggml/src/ggml-sycl/ggml-sycl.cpp` (`ggml_sycl_init()`'s L0 properties query and `ggml_backend_sycl_device_get_type()`, ~lines 167 and 98570).
**Ownership surface:** none — a boolean classification field and a backend-type enum return value; no allocation of any kind.
**Ticket:** llama.cpp-jah9

### 19. `66fa168a5617` — Extended SYCL oneDNN SDPA to non-FP16 KV caches (Q4_0–Q8_0 and FP32) (#25874)
**Files:** ggml/src/ggml-sycl/fattn-onednn.cpp, ggml/src/ggml-sycl/fattn.cpp
**Class:** port-candidate
**Why:** Extends the oneDNN SDPA gate to accept Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and F32 KV by dequantizing/converting K/V to dense F16 before the SDPA graph (prefill-only, K≥1024, Q≥32). The fork's `ggml_sycl_flash_attn_ext_onednn_supported()` (`fattn-onednn.cpp:277`) still hard-gates `params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16` — quantized-KV prefill (which the fork's own perf baselines explicitly benchmark, e.g. "q8_0 KV" runs) cannot reach the oneDNN path today and falls back to the slower native kernel. This is a genuine capability gap in a subsystem the fork already extended once (`MATERIALIZE_REQUIRED` K/V materialization for GQA) — reusing that same materializer machinery for a dequant-to-F16 conversion is a natural, moderate-risk extension rather than a duplicate.
⚠️ **Ownership screen:** upstream's own diff stages the converted K/V into `ggml_sycl_pool_alloc<sycl::half>` scratch (`ctx.pool()`) — that is upstream's code, not the fork's. Verified against the fork's actual current `fattn-onednn.cpp`: `grep -cE 'ggml_sycl_pool_alloc|ctx\.pool\(\)'` returns **0** there (vs. **11** in `git show b10630:.../fattn-onednn.cpp`, upstream's version) — the fork's own pre-existing K/V materialization scratch is already routed through the unified-cache-backed materializer (see the file's own comments, e.g. "ask the unified-cache-backed materializer for dense f16 K/V"), i.e. already on a sanctioned surface. **This re-expression requirement applies only to the NEW scratch this port itself would add** for dequantizing quantized/F32 KV to F16 — that new code must go through `unified_allocate`/`unified_allocate_owner()` → `mem_handle` (or an `alloc_owner` via `mem_handle::from_owned_alloc()`), not `ctx.pool()`, matching the standard the file's existing code already meets.
(A general legacy-`ctx.pool()` backlog exists SYCL-backend-wide but is unrelated to this file/port — see "Cross-cutting items" above.)
**Landing zone:** `ggml/src/ggml-sycl/fattn-onednn.cpp`, `ggml_sycl_flash_attn_ext_onednn_supported()` gate (line ~277) and the K/V staging call sites feeding the `MATERIALIZE_REQUIRED` plan.
**Ownership surface:** `mem_handle` lease over a unified-cache scratch allocation for the dequantized K/V dense-F16 buffers — NOT `ggml_sycl_pool_alloc`/`ctx.pool()` as upstream wrote it.
**Ticket:** llama.cpp-g7hq

### 20. `6c8dcaa7ae41` — sycl: parallelize the non-contiguous concat kernel (#25852)
**Files:** ggml/src/ggml-sycl/concat.cpp
**Class:** port-candidate
**Why:** Launch-geometry-only fix: the non-contiguous concat kernel launched a single-lane `(1,1,1)` work-group; this widens it to `(1,1,SYCL_CONCAT_BLOCK_SIZE)`. Measured upstream at +9.4% PP2048 on Arc Pro B70 — this fork's exact hardware. Confirmed the fork's `concat_T_sycl_non_cont` (`concat.cpp:130`) still launches `sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1))` — the bug is present, unmodified, and the fix is a narrow, low-risk launch-geometry change with no interaction with any rewritten subsystem.
**Landing zone:** `ggml/src/ggml-sycl/concat.cpp`, function `concat_T_sycl_non_cont`.
**Ownership surface:** none — launch-geometry change only (work-group shape), no allocation.
**Ticket:** llama.cpp-qz0f

### 21. `c074cb3f763d` — sycl : enhance OP set_rows to support all missed data types (#26515)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/set_rows.cpp
**Class:** port-candidate
**Why:** Adds `set_rows` support for `Q2_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_XS`. The fork's `set_rows.cpp` dispatch (~lines 935-976) covers only F32/F16/BF16/Q8_0/Q5_1/Q5_0/Q4_1/Q4_0/IQ4_NL — every type this commit adds is genuinely absent from the fork. Same shared-quant-table category as entries 8/14/16.
**Landing zone:** `ggml/src/ggml-sycl/set_rows.cpp` dispatch switch(es), reusing `dequantize.hpp`/`vecdotq.hpp` quantize helpers where already ported (entry 14) and adding the remaining IQ-type quantizers.
**Ownership surface:** none confirmed in the diff (`grep`'d `set_rows.cpp` for `malloc`/`pool_alloc` — no hits) — per-row quantize math writing into the existing destination tensor's buffer; no allocation.
**Ticket:** llama.cpp-596y

### 22. `eef5f3e3430a` — sycl : fix error Error OP FLASH_ATTN_EXT on arc770 (#26441)
**Files:** ggml/src/ggml-sycl/fattn-vec.hpp
**Class:** port-candidate
**Why:** Converts a runtime `if (D <= 256 && nthreads == 256)` into `if constexpr (D <= 256) { if (nthreads == 256) {...; return;} }` in `ggml_sycl_flash_attn_ext_vec_case_impl`, fixing a template-instantiation bug that entry 3 (`c1063ac9d75f`, the Battlemage-256-thread change) introduced: at `D > 256` the `nthreads_hw=256` branch was still being *instantiated* (just not executed), overflowing 64 KB work-group local memory. This has no independent value without entry 3 and does not apply to the fork's current flat-macro `fattn-vec.hpp` — it is a required companion fix that must land in the same change as entry 3's port.
**Landing zone:** Same as entry 3 — the arch-conditional launch dispatch in `ggml/src/ggml-sycl/fattn-vec.hpp`; use `if constexpr` from the start rather than porting entry 3's runtime `if` and immediately following with this fix.
**Ownership surface:** none — compile-time template-instantiation fix, no allocation.
**Ticket:** llama.cpp-d6is

### 23. `31558dbb7657` — sycl : Support DSv4 OPs: LIGHTNING_INDEXER,DSV4_HC_COMB,DSV4_HC_POST,DSV4_HC_PRE (#26568)
**Files:** ggml/src/ggml-sycl/dsv4-hc.cpp (new), ggml/src/ggml-sycl/dsv4-hc.hpp (new), ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/lightning-indexer.cpp (new), ggml/src/ggml-sycl/lightning-indexer.hpp (new)
**Class:** port-candidate (post-merge — blocked on this Phase C merge landing)
**Why:** Implements SYCL kernels for four DeepSeek-v4-specific ggml ops that are unreachable against pre-merge master (`GGML_OP_LIGHTNING_INDEXER`/`GGML_OP_DSV4_HC_*` don't exist in today's `ggml/include/ggml.h`), but the blocker is satisfied by this exact merge: confirmed both prerequisite commits are inside `81ff7abe5..b10630` (`git merge-base --is-ancestor`) — `00f5442cc4e8` adds `GGML_OP_LIGHTNING_INDEXER` (ggml.h/ggml.c/CPU ops, no SYCL) and `0dc74e332ede` adds `GGML_OP_DSV4_HC_COMB/PRE/POST` (ggml.h/ggml.c/CPU/CUDA, no SYCL) — so the enum, CPU reference, and graph-building support all land with the merge itself, not as a separate out-of-scope commit. The fork's dispatch has diverged from upstream's single `do_ggml_backend_sycl_device_supports_op` switch (the fork's equivalent is `ggml_backend_sycl_device_supports_op` at a different line, and there are additional `GGML_OP_*` switches for node classification and storage-readiness gating that don't exist upstream), so upstream's ~35-line switch addition is a reimplementation against the fork's dispatch shape, not a copy.
🚫 **Do NOT port** the `docs/ops/SYCL.csv` churn this commit carries (23,541 lines) — that file is an auto-generated ops-support matrix, not hand-maintained content.
**Landing zone:** new `ggml/src/ggml-sycl/{dsv4-hc,lightning-indexer}.{cpp,hpp}` (fork-native kernel implementations) + fork-native insertions into `ggml_sycl_compute_forward`'s op switch and `ggml_backend_sycl_device_supports_op`'s capability switch in `ggml-sycl.cpp`.
**Ownership surface:** none — clean against the screen's allocation patterns (verified with a positive control: the same `malloc`/`pool_alloc`/TLSF grep against this commit's SYCL diff returns zero hits, while the identical grep against entry 19's diff — known to allocate — returns 6, so the zero here isn't vacuous); ticket-writer must confirm the kernels need no scratch beyond work-group local memory.
**STATUS: BLOCKED-ON-MERGE.** **Ticket:** llama.cpp-pmzl (depends on llama.cpp-t9l9, the campaign's PC T24 landing task).

### 24. `6b5c2efb4e2f` — sycl: *glu flat path (#26354)
**Files:** ggml/src/ggml-sycl/element_wise.cpp, ggml/src/ggml-sycl/presets.hpp
**Class:** port-candidate
**Why:** Consolidates the fused-GLU kernels (geglu/reglu/swiglu/geglu_erf/geglu_quick) behind one shared launcher and adds a contiguous fast path when `o0 == n && o1 == n` (split-GLU operands), collapsing the de-interleave index math to identity. Measured +14% f16 / +4% f32 on `SWIGLU` on Arc Pro B70 — this fork's own hardware. The fork's `element_wise.cpp` still has the separate, non-consolidated `gated_op_fused_{geglu,reglu,swiglu,...}` kernels taking `(o0, o1)` with no contiguous-fast-path branch — the optimization is genuinely missing, not superseded (this is the same file as entry 12, but a different, additive change to the GLU-specific kernels rather than the generic unary-op fast path entry 12 already covers).
**Landing zone:** `ggml/src/ggml-sycl/element_wise.cpp`, the `gated_op_fused_*` kernel family and their `ggml_sycl_op_{geglu,reglu,swiglu}` callers.
**Ownership surface:** none — kernel-consolidation and an index-math fast path over existing src/dst buffers (`grep`'d for `malloc`/`pool_alloc` in the diff — no hits); no allocation.
**Ticket:** llama.cpp-vvci

### 25. `fc3f10b3895e` — sycl: fix UE4M3 parsing (#25608)
**Files:** ggml/src/ggml-sycl/common.hpp
**Class:** port-candidate
**Why:** Fixes NVFP4's unsigned UE4M3 scale-byte decode, which upstream's *old* code mishandled by reinterpreting it as signed E4M3. The fork's `ggml_sycl_ue4m3_to_fp32()` (`common.hpp:6797`) already uses the same corrected unsigned-decode formula as this commit's fix, but retains an extra `x == 0xff → 0.0f` special case left over from the old buggy code that upstream's fixed formula explicitly removed. This divergence is latent, not live: `ggml_fp32_to_ue4m3` saturates at 0x7E and never sets bit 7, so no real encoder can ever emit byte `0xFF` — the fork's own test documents exactly this (`ggml/src/ggml-sycl/tests/test-q1-nvfp4-adapter-device.cpp:271-276`: "0xFF is deliberately absent because ggml_sycl_ue4m3_to_fp32 maps it to 0.0f while ggml_ue4m3_to_fp32 does not -- a pre-existing divergence in common.hpp that no encoder can reach"). Narrow, low-risk reconciliation of a dead branch, not a live-data bug.
**Landing zone:** `ggml/src/ggml-sycl/common.hpp`, function `ggml_sycl_ue4m3_to_fp32` — drop the `x == 0xff` branch of the leading guard; `ggml/src/ggml-sycl/tests/test-q1-nvfp4-adapter-device.cpp` (its comment at lines 271-276 documents the divergence this drop resolves and goes stale once the branch is gone).
**Ownership surface:** none — a pure scalar decode function, no allocation.
**Ticket:** llama.cpp-h2uz

---

## Part B: commits 26–49 of 49

**24 commits: 11 port-candidate / 7 superseded / 6 n-a**

### 1. `f8e30266d23f` — sycl: coalesce the ssm_conv window loads (#26612)
**Files:** ggml/src/ggml-sycl/ssm_conv.cpp
**Class:** port-candidate
**Why:** Reorders `kernel_ssm_conv`'s flattened-index decomposition from channel-fastest to token-fastest, coalescing the per-thread `d_conv` window loads (measured 1.85–1.87x on `test-backend-ops perf -o SSM_CONV`, +1.8–2.2% end-to-end `llama-bench` pp2048 on a Mamba-family model). The fork's `ssm_conv.cpp` (`kernel_ssm_conv`) is byte-for-byte the pre-patch layout (`channel = idx % d_inner; token = (idx / d_inner) % n_t`) — confirmed live. SSM_CONV is not part of any subsystem the fork rewrote (mul_mat/fattn/memory/graph/mmvq/mmq); pure index-math reordering over the existing kernel.
**Landing zone:** `ggml/src/ggml-sycl/ssm_conv.cpp`, function `kernel_ssm_conv` (the `channel`/`token`/`seq` decomposition).
**Ownership surface:** none — index-math reordering only, same src/weights/dst pointers, no allocation.
**Ticket:** llama.cpp-3zzs

### 2. `153d324bcf86` — llama: add default load-mode auto, which avoids mmap on iGPUs (#26081)
**Files:** common/arg.cpp, common/common.h, ggml/include/ggml-backend.h, ggml/src/ggml-backend-meta.cpp, ggml/src/ggml-{blas,cann,cpu,cuda,et,hexagon,metal,opencl,openvino,rpc,sycl,virtgpu,vulkan,webgpu,zdnn,zendnn}/*, include/llama.h, src/llama-model-loader.cpp, src/llama-model.cpp, src/llama.cpp, tools/*
**Class:** n-a
**Why:** This is a cross-backend/core-ggml/llama feature (new `LLAMA_LOAD_MODE_AUTO`, the `--load-mode auto` CLI value, `ggml_backend_dev_props::mmap_support`, and the model-loader logic that picks mmap vs. host-pinned based on device unified-memory classification) — the SYCL touch is a single line (`/* .mmap_support = */ true,` in `ggml_backend_sycl_device_get_props`), meaningless without the enum, the CLI plumbing, and the loader decision logic none of which live under `ggml/src/ggml-sycl`. Confirmed the fork has **no** `LLAMA_LOAD_MODE_*` enum, no `--load-mode` flag, and no `mmap_support` field in `ggml_backend_dev_props` at all — this whole feature is absent, not narrowed or superseded. Per the Task 7 gotcha's mirror case: a core-scoped commit whose SYCL hunk is trivial and inseparable from the untouched core/llama plumbing is out of this SYCL-only audit's scope. The core-ggml conflict brief (`docs/merge/briefs/core-ggml.md`) already tracks `ggml_rope_set_offset` and related contract changes for this merge; `mmap_support`/load-mode auto belongs alongside it there, not in the Phase C SYCL-ports epic.
**Landing zone:** n/a (belongs to the core-ggml/llama merge wave, not a post-merge SYCL port).

### 3. `d415e65a5784` — sycl : enhance concat to support Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (#26800)
**Files:** ggml/src/ggml-sycl/concat.cpp
**Class:** port-candidate
**Why:** Adds `concat_impl_{q4_0,q4_1,q5_0,q5_1,q8_0}_sycl` block-quantized concat paths (mirroring the existing `concat_impl_sycl<T>` template shape) plus a small unrelated correctness cleanup in the pre-existing float path (drops two spurious `.wait()`s after device-to-device memcpys that already complete in-order on the same queue). Confirmed the fork's `ggml_sycl_op_concat` dispatch switch (`concat.cpp`) only covers F32/F16/BF16/I32/I16/I64/I8 — none of the five quant types this commit adds exist in the fork's concat path at all. Same shared-conversion-table additive-coverage category as Part A entries 8/14/16/21 (dequant/cpy/set_rows type-table gaps); no rewritten subsystem is touched.
**Landing zone:** `ggml/src/ggml-sycl/concat.cpp`, the `ggml_sycl_op_concat` dispatch switch (new `case GGML_TYPE_Q4_0/Q4_1/Q5_0/Q5_1/Q8_0` arms) plus the new `concat_impl_q*_sycl` functions.
**Ownership surface:** none — the quantized paths use `stream->memcpy`/kernel writes into the pre-existing `dst->data` buffer, same as the float path; no allocation.
**Ticket:** llama.cpp-tfap

### 4. `8efbf65dbd55` — sycl : Add DMMV ESIMD Q3_K kernel (#26251)
**Files:** docs/backend/SYCL.md, ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp (new), ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** n-a
**Why:** Squashes in DMMV ESIMD kernels for Q4_K, Q6_K and Q3_K (per the commit's own sub-history), defaulting them ON via a new `g_ggml_sycl_enable_esimd` flag (`GGML_SYCL_ENABLE_ESIMD`, default 1) that takes priority over the fork's existing reorder-MMVQ dispatch whenever the compiler is `__INTEL_LLVM_COMPILER`. **The strongest evidence against this is in the fork's own tree, not just CLAUDE.md's prose**: the fork already ships a disabled-by-default DMMV ESIMD kernel for exactly this call path — `ggml/src/ggml-sycl/dmmv-esimd.hpp` (406 lines, included from `dmmv.cpp`, gated by its own `GGML_SYCL_DMMV_ESIMD` env var, default off) — carrying its own in-file benchmark comment: *"EXPERIMENTAL - DISABLED BY DEFAULT... Benchmark results show ESIMD is 3.5x SLOWER than standard DMMV: Standard DMMV: 21.10 t/s, DMMV ESIMD: 6.06 t/s"*, with root causes documented (no subgroup-shuffle reduction on ESIMD, unaligned `block_q4_0.qs` loads, SIMD register pressure) — and a literal `// TODO: Add Q8_0, Q4_K, etc.` at `dmmv_esimd_supported()` (`dmmv-esimd.hpp:363`), i.e. the fork already anticipated and declined to extend this exact kernel to more K-quant types. This corroborates (rather than merely restates) CLAUDE.md's separate, general rule — *"Small-block dequant (Q4_0/Q8_0/**Q4_K**) belongs on standard SYCL, not ESIMD. ESIMD measured 1.9x SLOWER on Arc B580 + oneAPI 2025.3"* — with a second, independent, in-repo measurement on the same code path. The upstream commit's own three new types (Q3_K/Q4_K/Q6_K) are not directly measured by either source (both cite Q4_0 numbers) — the extension to Q3_K/Q4_K/Q6_K is an extrapolation from a Q4_0 measurement via the shared "block granularity too small to amortize LSC loads" mechanism, not a cited number for those specific types — but porting it as written would still default-enable exactly the direction the fork's own kernel measured and disabled, on the same hardware. Confirmed live: no `g_ggml_sycl_enable_esimd` and no `esimd.hpp` (upstream's new file name) exist in the fork; the fork's ESIMD-dequant surface is `convert-esimd.hpp`/`dmmv-esimd.hpp`/`mmq-esimd.hpp`/`xmx-esimd-*.hpp`, independently built, all opt-in. Residual value: the new Q3_K/Q4_K/Q6_K ESIMD kernels have reference value as instrumented code behind the fork's existing disabled gate (for a future retest on current driver/hardware), not as a default-ON port.
**Landing zone:** n/a — contradicts a settled, hardware-measured architecture rule that the fork has independently corroborated in-repo on the same DMMV code path.

### 5. `1ee1cd9bc65a` — sycl: fuse UNARY(silu|sigmoid|softplus) + MUL (#26411)
**Files:** ggml/src/ggml-sycl/element_wise.cpp, ggml/src/ggml-sycl/element_wise.hpp, ggml/src/ggml-sycl/fusion.cpp, ggml/src/ggml-sycl/fusion.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, tests/test-backend-ops.cpp
**Class:** superseded
**Why:** Extends `ggml_sycl_can_fuse()` (the generic two-op graph-fusion detector `ggml_backend_sycl_graph_compute_impl`'s node loop calls to skip ahead) with a `{GGML_OP_UNARY, GGML_OP_MUL}` pattern for SiLU/Sigmoid/Softplus, and adds the `ggml_sycl_op_unary_mul_fused` kernel. This is exactly the `ggml_sycl_fuse()` framework Part A entries 2 and 13 already classified superseded ("the fork has neither `topk-moe.{cpp,hpp}` nor any `ggml_sycl_fuse` call site... graph-node dispatch is already deeply customized for cache/placement bookkeeping"). Confirmed live: `fusion.cpp`/`fusion.hpp` do not exist in the fork, and neither does `ggml_sycl_can_fuse` or `g_ggml_sycl_enable_fusion`. Measured upstream gain is also marginal (+0.38–0.62%, "within run-to-run spread" per the commit's own message) — low value on top of a rejected mechanism.

### 6. `154d57af3eec` — sycl: remove separate fp32 type promotion in gemm non-oneDNN path (#26372)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** In the non-oneDNN fallback GEMM branch of `ggml_sycl_op_mul_mat_sycl`, replaces an f16-in/f16-out `dpct::gemm` call that writes into a scratch `ggml_sycl_pool_alloc<sycl::half> dst_f16` buffer (freed via `ctx.pool()`) followed by a `to_fp32_sycl` conversion kernel, with a single `dpct::gemm` call that lets the library promote directly to f32 output — eliminating both the scratch allocation and the extra conversion kernel. Confirmed live: the fork's `ggml_sycl_op_mul_mat_sycl` (`ggml-sycl.cpp:41080`, dpct::gemm call at :41327) still has this exact `ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols)` pattern verbatim (only `oneapi::mkl::transpose` vs. the fork's `oneapi::math::transpose` differs — a pre-existing oneAPI-version rename elsewhere in the fork, not a functional divergence), and this fallback path is confirmed live and commented on at two other call sites (`ggml-sycl.cpp:56966`, `:60637`). Unlike most entries this is a rare *ownership-screen-favorable* port: it **removes** a `ggml_sycl_pool_alloc` scratch allocation rather than adding one, moving this call site slightly closer to canonical-contract compliance. This removes one site from ledger A's cross-cutting `ggml_sycl_pool_alloc`/`ctx.pool()` backlog — `ggml-sycl.cpp:41323` is one of the 13 `ggml-sycl.cpp` sites Part A's "Cross-cutting items" census lists (verified: it is line 4 of that list, `41129, 41257, 41310, 41323, 41342, 41343, ...`); once this port lands the `ggml-sycl.cpp` count in that backlog drops from 13 to 12. The other two nearby sites remain, for different reasons: `:41310` (`src0_as_f16`) sits in the same `#elif GGML_SYCL_HAS_ONEAPI_MATH` block as `:41323` (`:41309-41336`) but is a separate, untouched allocation serving a different purpose — input promotion (converting `src0` to f16 before the GEMM), which this fix does not touch — versus `:41323`'s output staging (`dst_f16`), which this fix eliminates entirely; `:41342`/`:41343` (`src0_ddq_as_f32`/`src1_ddq_as_f32`) are genuinely in a different branch — the `else` at `:41341`, which is a *separate f32-precision GEMM arm* (fp32 input promotion at `:41348-41363`, then `DnnlGemmWrapper::row_gemm` at `:41371` or `oneapi::math::blas::column_major::gemm` at `:41379`), not a non-GEMM path: upstream's fix removes an f16-output-staging buffer specifically in the f16-GEMM arm this port targets, and does not touch the separate f32-GEMM arm's own f32-input-promotion buffers, which is why `:41342`/`:41343` stay in the backlog.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_sycl_op_mul_mat_sycl`, the non-oneDNN f16 GEMM branch (~line 41320–41335).
**Ownership surface:** none touched — net effect removes one `ggml_sycl_pool_alloc<sycl::half>` legacy pool allocation; no new allocation of any kind is introduced.
**Ticket:** llama.cpp-ag2d

### 7. `a97123e49796` — [SYCL] Support host pinned mem to improve SYCL Host-to-Device Memory Access (#26789)
**Files:** docs/backend/SYCL.md, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Replaces `ggml_backend_sycl_host_buffer_type_alloc_buffer`'s legacy `aligned_malloc_host` with a new `ggml_backend_sycl_host_malloc()` that calls raw `sycl::malloc_host()` directly on a bare device queue, freed via raw `sycl::free()`, gated by a new `g_ggml_sycl_enable_host_pinned_mem` env flag (default ON) — exactly the direct-allocation-outside-the-cache pattern the canonical memory contract forbids. Confirmed live: the fork's `ggml_backend_sycl_host_buffer_type_alloc_buffer` (`ggml-sycl.cpp:38474`) is already a complete, independently-built superset of this commit's goal — it routes through `ggml_sycl::unified_alloc()` with a full `alloc_request` (`alloc_role::WEIGHT`/`STAGING`, `runtime_category::HOST_COMPUTE`, `must_host_pinned = true`, zone-aware `use_pinned_pool`), wraps the result in a `mem_handle` via `detail::from_legacy_owned_alloc`, and `ggml_backend_sycl_host_buffer_type_get_max_size` (`:38392`) already reasons about per-zone largest-contiguous-free-block accounting that upstream's simple `get_max_mem_alloc_size()`/`SIZE_MAX` toggle (added one commit later, entry 17) doesn't have. Same disposition as Part A entry 5 (USM system allocation threshold, superseded by the same tiered unified-cache placement).

### 8. `3d93885352a0` — sycl: fuse the gated-delta-net state writeback cpy (#26643)
**Files:** ggml/src/ggml-sycl/gated_delta_net.cpp, ggml/src/ggml-sycl/gated_delta_net.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Adds `ggml_sycl_try_gdn_cache_fusion()`, a bespoke graph-pattern detector (gated behind `g_ggml_sycl_enable_fusion`, "ported from `ggml_cuda_try_gdn_cache_fusion`") wired into `ggml_backend_sycl_graph_compute_impl`'s node loop that lets the gated-delta-net kernel write its recurrent-state snapshots directly into a downstream KV-cache buffer, skipping a materialize-then-copy node. Although self-contained rather than routed through the generic `ggml_sycl_fuse()` helper (unlike entries 5/9/Part A 2/13), it still (a) depends on `g_ggml_sycl_enable_fusion`, which does not exist anywhere in the fork (confirmed: zero hits), and (b) inserts a new graph-level "detect pattern, write output elsewhere, skip N nodes" shortcut into the same `ggml_backend_sycl_graph_compute_impl` node loop the fork has extensively customized for cache/placement bookkeeping per node — exactly the risk Part A entry 2 flagged ("re-adding upstream's fusion shortcut would bypass that bookkeeping"). The underlying `ggml_sycl_op_gated_delta_net`/`gated_delta_net.cpp` op itself (with its pre-existing `K`/`keep_rs` rollback-slot support) is confirmed present and unmodified in the fork — only the fusion optimization on top of it is being declined.

### 9. `650913862270` — sycl: fuse mul_mat(gate) + mul_mat(up) + GLU for q4_K dense FFN (#26779)
**Files:** docs/backend/SYCL.md, ggml/src/ggml-sycl/element_wise.{cpp,hpp}, ggml/src/ggml-sycl/fusion.cpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.{cpp,hpp}, tests/test-backend-ops.cpp
**Class:** superseded
**Why:** Adds `ggml_sycl_mul_mat_glu_mmvq_fused()`, gated by `ggml_sycl_can_fuse(cgraph, node_idx, {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU}, {})` — the same rejected `ggml_sycl_fuse()` framework as entry 5 (confirmed absent from the fork).
⚠️ **Ownership screen:** the fused kernel additionally stages its quantized activation into `ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), ...)` — a raw legacy `ctx.pool()` scratch allocation, the direct-allocation pattern the canonical contract disallows outside unified-cache surfaces — an independent reason this would need re-expression through `unified_allocate`/`mem_handle` even if the fusion framework itself were adopted. Two independent reasons to decline as written.

### 10. `1692f9e50bb2` — ggml : recurrent state rollback for ggml_ssm_scan (#26623)
**Files:** ggml/include/ggml.h, ggml/src/ggml-{cpu,cuda,et,metal,vulkan,webgpu}/*, ggml/src/ggml-sycl/ssm_scan.cpp, ggml/src/ggml.c, src/llama-arch.cpp, src/llama-context.cpp, src/llama-model-loader.cpp, src/models/{mamba-base,plamo2}.cpp, tests/*
**Class:** port-candidate (core-ggml wave companion — lands with the merge, not deferred to Phase C)
**Why:** Core-ggml commit adding a trailing `int64_t K` parameter to `ggml_ssm_scan()` (Nemotron-style recurrent-state rollback, K rollback slots instead of always 1), implemented for CPU/CUDA/Metal/Vulkan/WebGPU/et and required, mechanically, for every backend once the signature lands — the fork's `ggml_sycl_op_ssm_scan` (`ssm_scan.cpp`) does not yet read `K` at all (confirmed: no `K`/`ggml_get_op_params_i32(dst, 0)` in the fork's file; matches the pre-patch signature verbatim). Unlike entry 2, this is not out-of-scope: the `ggml.h` signature change is a core-ggml-wave (Task 10) contract change that lands with the merge regardless of this audit's SYCL keep-ours decision, so the SYCL side must apply the matching `K`/snapshot-write hunk in `ssm_scan_f32_group`/`ssm_scan_f32_sycl`/`ggml_sycl_op_ssm_scan` at the same time — a mandatory correctness companion to wave 2, not a deferred nice-to-have.
**Landing zone:** `ggml/src/ggml-sycl/ssm_scan.cpp` (`ssm_scan_f32_group`, `ssm_scan_f32_sycl`, `ggml_sycl_op_ssm_scan`) — must land together with the core `ggml_ssm_scan()` signature change from the merge's core-ggml wave (see `docs/merge/briefs/core-ggml.md`), not deferred to the post-merge Phase C epic.
**Ownership surface:** none — writes the rollback snapshot into the pre-existing `dst` tensor buffer at a computed offset; no allocation.
**STATUS: CORE-GGML WAVE COMPANION.** **Ticket:** llama.cpp-8vwp (depends on llama.cpp-90ns, the campaign's PC T17 wave-3 task; filed for tracking only).

### 11. `37a215c9e909` — [SYCL] support OP OPT_STEP_ADAMW, OPT_STEP_SGD (#25268)
**Files:** docs/ops.md, docs/ops/SYCL.csv, examples/sycl/update-ops-doc.sh, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/opt-step.cpp (new), ggml/src/ggml-sycl/opt-step.hpp (new)
**Class:** port-candidate
**Why:** Adds `ggml_sycl_opt_step_adamw`/`ggml_sycl_opt_step_sgd` — training-time optimizer-step kernels (in-place parameter update from gradient + optimizer state) wired into `ggml_sycl_compute_forward` and `ggml_backend_sycl_device_supports_op`. Confirmed absent from the fork entirely (`opt-step.{cpp,hpp}` don't exist; no `OPT_STEP_ADAMW`/`OPT_STEP_SGD` case anywhere in `ggml-sycl.cpp`). This is a genuine capability gap (SYCL-backend fine-tuning/training support) untouched by any rewritten subsystem — small, self-contained, additive.
**Landing zone:** new `ggml/src/ggml-sycl/opt-step.{cpp,hpp}` + the `GGML_OP_OPT_STEP_ADAMW`/`GGML_OP_OPT_STEP_SGD` cases in `ggml_sycl_compute_forward`'s op switch and `ggml_backend_sycl_device_supports_op` (`ggml-sycl.cpp:98736`).
**Ownership surface:** none — in-place update of the existing parameter/gradient/optimizer-state tensor buffers; no allocation.
**Ticket:** llama.cpp-al2o

### 12. `f275595dd16f` — sycl: fix thread/block count in quantized cpy kernel launches (#27160)
**Files:** ggml/src/ggml-sycl/cpy.cpp
**Class:** port-candidate
**Why:** The commit fixes launch-size errors in two directions across `cpy.cpp`'s block-quantized launchers, both confirmed still present in the fork's file (checked function-by-function against a listing of every `ggml_cpy_*_sycl` in the fork, not inferred from the upstream diff alone):
  1. **Under-launch (single-lane-per-block).** The cross-type conversion launchers (`ggml_cpy_{f32_q8_0,q8_0_f32,f32_q4_0,q4_0_f32,f32_q4_1,q4_1_f32,f32_q5_0,q5_0_f32,f32_q5_1,q5_1_f32,f32_iq4_nl}_sycl`: 5 bidirectional type pairs (10 launchers) plus the one-directional `f32_iq4_nl` — 11 launchers total) still compute `num_blocks = ne` or `ne / QK` and launch `sycl::nd_range<3>(sycl::range<3>(1,1,num_blocks), sycl::range<3>(1,1,1))` — one single-thread work-group per quant block. (Note: upstream's diff also touches an `f32_q2_0`/`q2_0_f32` pair and several `f16→q4_0/q4_1/q5_0`/`mxfp4_f32` launchers that do **not exist** in the fork's `cpy.cpp` at all — the fork's type coverage here is narrower than upstream's, so those specific hunks are moot; `f32_iq4_nl` (`cpy.cpp:842`, single-lane launch body at `:861-863`) does exist in the fork and needs the same width fix as the other 10.) The fix widens every one of these 11 to `ceil_div(ne/QK, SYCL_CPY_BLOCK_SIZE)` work-groups of `SYCL_CPY_BLOCK_SIZE` threads, matching the file's own non-quantized f16/f32 cpy launchers. Measured 20.21 → 158.19 GB/s (7.8x) on the q4_0→f32 path on an Arc Pro B70 — this fork's own hardware.
  2. **Over-launch (unscaled by block size).** The 5 same-type passthrough copies that exist in the fork — `ggml_cpy_q8_0_q8_0` (`cpy.cpp:1176`), `ggml_cpy_q5_0_q5_0` (`:1203`), `ggml_cpy_q5_1_q5_1` (`:1230`), `ggml_cpy_q4_0_q4_0` (`:1258`), `ggml_cpy_q4_1_q4_1` (`:1285`) — already use the wide `SYCL_CPY_BLOCK_SIZE` work-group shape, but each computes `num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE)` — treating `ne` (the scalar element count) as if it were already the block count, rather than dividing by `QK*` first. Confirmed live: all 5 still have this exact pattern at these lines. The upstream fix corrects it to `ceil_div(ne / QK*, SYCL_CPY_BLOCK_SIZE)`. This over-launches the grid by a factor of `QK*` (up to 32x); it is not a memory-safety bug — the shared `cpy_q_q` kernel bounds-checks `i >= ne` internally (`i` is pre-scaled by `qk`) — but it wastes a large fraction of every launch on threads that immediately return. (Upstream's diff also touches ~15 further same-type pairs — q1_0/q2_0/mxfp4/nvfp4/every K-quant/every IQ-type — that don't exist in the fork's `cpy.cpp`; moot for the same reason as above.)
  Pure launch-geometry fix in both directions, same category as Part A entry 20 (concat) and this ledger's entry 1 (ssm_conv); no rewritten subsystem is touched.
**Landing zone:** `ggml/src/ggml-sycl/cpy.cpp` — the 11 single-lane cross-type launchers (`f32_q8_0`, `q8_0_f32`, `f32_q4_0`, `q4_0_f32`, `f32_q4_1`, `q4_1_f32`, `f32_q5_0`, `q5_0_f32`, `f32_q5_1`, `q5_1_f32`, `f32_iq4_nl` at `:842`/body `:861-863`) get the width fix; the 5 same-type passthrough launchers `ggml_cpy_q8_0_q8_0` (`:1176`), `ggml_cpy_q5_0_q5_0` (`:1203`), `ggml_cpy_q5_1_q5_1` (`:1230`), `ggml_cpy_q4_0_q4_0` (`:1258`), `ggml_cpy_q4_1_q4_1` (`:1285`) get the `ne`→`ne/QK*` block-count fix. Skip the upstream hunks for `q2_0`/`mxfp4`/K-quant/IQ-type variants — those functions don't exist in the fork.
**Ownership surface:** none — launch-geometry/grid-sizing change only, same src/dst buffers.
**Ticket:** llama.cpp-xoi2

### 13. `0882c7bc8907` — sycl: honor GGML_HINT_SRC0_IS_HADAMARD (#27298)
**Files:** ggml/src/ggml-sycl/fwht.cpp (new), ggml/src/ggml-sycl/fwht.hpp (new), ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** Adds a Fast Walsh-Hadamard Transform kernel (`ggml_sycl_op_fwht`, "port of `ggml-cuda/fwht.cu`") and a narrow early-return at the very top of `ggml_sycl_mul_mat()`: when `dst->op == GGML_OP_MUL_MAT` and `ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD`, run the FWHT kernel instead of the GEMM. Measured 3.3–6x speedup vs. GEMM on the affected shapes. Confirmed the ggml-core prerequisite already exists in the fork (`GGML_HINT_SRC0_IS_HADAMARD` and `ggml_mul_mat_set_hint()` are both present in `ggml/include/ggml.h`) — unlike Part A entry 23 (DSv4), there is no missing compile-time dependency. The new `fwht.cpp` allocates nothing (`grep`'d for `malloc`/`pool_alloc` — no hits), and the bail-out sits before all of the fork's customized layout/route dispatch in `ggml_sycl_mul_mat` (confirmed present at `ggml-sycl.cpp:58456`, with a fork-added `forced_layout` parameter) rather than competing with it.
**Landing zone:** new `ggml/src/ggml-sycl/fwht.{cpp,hpp}` + the early Hadamard-hint check at the top of `ggml_sycl_mul_mat()` in `ggml-sycl.cpp` (~line 58456), before the fork's layout/route dispatch begins.
**Ownership surface:** none — reads `src1`/writes `dst` directly; no allocation.
**Ticket:** llama.cpp-kdcm

### 14. `fe8156f78901` — ggml: add ggml_rope_set_offset (+ metal support) (#27120)
**Files:** ggml/include/ggml.h, ggml/src/ggml-{cann,cpu,cuda,et,hexagon,metal,opencl,openvino,sycl,vulkan,webgpu}/*, ggml/src/ggml.c, ggml/src/ggml-sycl/ggml-sycl.cpp, tests/test-backend-ops.cpp
**Class:** n-a
**Why:** Core-ggml commit introducing `ggml_rope_set_offset()` (a `n_offs` op-param for partial-range RoPE) with CPU/Metal/CUDA support; the SYCL-side hunk in this specific commit is a two-line placeholder guard in `ggml_backend_sycl_device_supports_op` (`case GGML_OP_ROPE/ROPE_BACK: return op_params[15] == 0;`, commented `// FIXME: support ggml_rope_set_offset`) — SYCL is explicitly *not* implementing the feature here, only refusing it. Confirmed this exact two-line guard is **deleted verbatim** by entry 16 (`749f688fcaa4`, two commits later, same file/region), which replaces it with the real `n_offs`-aware kernel support. Applying entry 14 and then entry 16 nets to exactly entry 16's diff alone; the intermediate placeholder guard has no independent value to port and would be immediately overwritten. Port entry 16 directly.
**Landing zone:** n/a — superseded in-sequence by entry 16; see that entry instead.

### 15. `6cc504a2e90d` — sycl: report zero devices instead of aborting when the host has none (#27291)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** Wraps the `dpct::dev_mgr::instance().device_count()` call in `ggml_sycl_init()` in a try/catch, falling back to `info.device_count = 0` (and logging) instead of letting a SYCL exception propagate and crash — needed so SYCL-agnostic tools like `llama-quantize` don't hard-crash on a host with the SYCL runtime installed but no usable device. Confirmed the fork's `ggml_sycl_init()` (`ggml-sycl.cpp:21901`) still calls `dpct::dev_mgr::instance().device_count()` completely unguarded (`const int raw_device_count = ...;`, no try/catch) — the exact pre-patch call site, just wrapped in the fork's own added `GGML_SYCL_VISIBLE_DEVICES`/`GGML_SYCL_DEVICE` device-map parsing that runs after it. Small, low-risk robustness fix, no interaction with any rewritten subsystem.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_sycl_init()` — wrap the `raw_device_count` acquisition in try/catch.
**Ownership surface:** none — device enumeration only, no allocation.
**Ticket:** llama.cpp-5pgc

### 16. `749f688fcaa4` — ggml: support ggml_rope_set_offset on opencl, sycl, wgpu, hexagon (#27345)
**Files:** ggml/src/ggml-hexagon/*, ggml/src/ggml-opencl/*, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/rope.cpp, ggml/src/ggml-webgpu/*
**Class:** port-candidate (core-ggml wave companion — lands with the merge, not deferred to Phase C)
**Why:** Implements real `n_offs` support in the SYCL `rope_norm`/`rope_neox`/`rope_multi` kernels (threading a new `n_offs` parameter through the per-thread bounds check and index math so only the `[n_offs, n_offs+n_dims)` channel range is rotated) and removes entry 14's placeholder guard entirely (the `case GGML_OP_ROPE/ROPE_BACK:` falls through to the general `return true;` group below it). Confirmed the fork's `rope.cpp` (`rope_norm`/`rope_neox`/`rope_multi`) is unmodified pre-patch (`if (i0 >= n_dims)`, no `n_offs` parameter anywhere). The core `ggml_rope_set_offset()` prerequisite this depends on is a genuine cross-cutting `ggml.h` change (see entry 14 and the core-ggml brief) that lands via the merge's core-ggml wave regardless of this audit; the SYCL kernel work here is real, self-contained, and must accompany that wave the same way entry 10 does.
**Landing zone:** `ggml/src/ggml-sycl/rope.cpp` (`rope_norm`, `rope_neox`, `rope_multi` — thread the `n_offs` parameter through) + `ggml_backend_sycl_device_supports_op` (`ggml-sycl.cpp:98736`, remove entry 14's placeholder guard). Land together with the core `ggml_rope_set_offset()` wave (`docs/merge/briefs/core-ggml.md`), not deferred to Phase C.
**Ownership surface:** none — reads/writes the existing `x`/`dst` tensor buffers with adjusted index math; no allocation.
**STATUS: CORE-GGML WAVE COMPANION.** **Ticket:** llama.cpp-hkuh (depends on llama.cpp-90ns, the campaign's PC T17 wave-3 task; filed for tracking only).

### 17. `9e96cf77ffd4` — sycl : fix load model with mlock issue (#27250)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Patches `ggml_backend_sycl_host_buffer_type_get_max_size()` to return `SIZE_MAX` when `g_ggml_sycl_enable_host_pinned_mem` is false (falling back to the legacy `aligned_malloc_host` path, whose allocations aren't bounded by `get_max_mem_alloc_size()`) — a fix to the exact mechanism entry 7 introduced. Since entry 7 is superseded (the fork's host-buffer allocator is already a completely different, unified-cache-routed implementation with no `g_ggml_sycl_enable_host_pinned_mem` variable at all — confirmed live, see entry 7), the code this commit patches does not exist in the fork. The fork's own `ggml_backend_sycl_host_buffer_type_get_max_size()` (`ggml-sycl.cpp:38392`) already does zone-aware largest-contiguous-free-block accounting, a different and more thorough mechanism than the toggle this commit adds.

### 18. `6602dd338941` — sycl: fix multiple warnings in compiling sycl backend (#26713)
**Files:** ggml/src/ggml-cpu/CMakeLists.txt, ggml/src/ggml-sycl/dpct/helper.hpp, ggml/src/ggml-sycl/element_wise.cpp, ggml/src/ggml-sycl/fattn-mkl.cpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/im2col.cpp, ggml/src/ggml-sycl/norm.cpp, ggml/src/ggml-sycl/set_rows.cpp
**Class:** port-candidate
**Why:** A grab-bag of compiler-warning fixes with uneven applicability, checked hunk-by-hunk against the fork: (a) `ggml-sycl.cpp`'s `%lu` → `%zu` format-string fixes for `size_t` args in the device/host-alloc error paths — **confirmed live and unfixed** at 5 call sites (`ggml-sycl.cpp:34103`, `:35949`, `:38634`, `:39505`, `:39298`); applies cleanly. The 5th, `:39298` (`GGML_LOG_WARN("%s: can't allocate %lu bytes (neither VRAM nor host-pinned)\n", __func__, rounded_size)` inside `ggml_sycl_pool_leg::alloc`, `rounded_size` a `size_t`), is the fork's reworded counterpart of the exact hunk upstream patches in that same function. `ggml/src/ggml-sycl/` is the merge's keep-ours scope, so this hunk is a real Phase C port item. (b) `ggml/src/ggml-cpu/CMakeLists.txt`'s `IntelLLVM` compiler-ID check widened to cover both C and CXX (`CMAKE_C_COMPILER_ID` as well as `CMAKE_CXX_COMPILER_ID`) plus a Windows `/clang:` flag-spelling fix. `ggml-cpu/` is **not** keep-ours — confirmed `git diff --stat 81ff7abe5 master -- ggml/src/ggml-cpu/CMakeLists.txt` is empty (the fork has made zero changes to this file since the merge-base), so it is not in the both-touched conflict set and will auto-merge to upstream's content with no resolution needed. Confirmed at tag `b10630` the file already carries the widened `CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"` form — so hunk (b) arrives automatically with the merge itself and needs **no Phase C ticket action**; only (a) is an actual port item. (c) The `element_wise.cpp` (`acc_f32` unused-`ne3`), `norm.cpp` (unused `nrows`/`nchannels`), and `set_rows.cpp` (`[[intel::reqd_sub_group_size]]` → `[[sycl::reqd_sub_group_size]]`) hunks all target code the fork has already rewritten past — none of the targeted lines/signatures exist in the fork's current versions of those files (confirmed via grep: no match for any of the three patterns). (d) `fattn-mkl.cpp` doesn't exist in the fork at all (Part A entry 17: the oneMKL GEMM flash-attention path was declined as duplicating the fork's own oneDNN SDPA path) — that hunk is moot. Net: a real but narrow port limited to (a); (b) needs no action (arrives with the merge, outside keep-ours scope); (c) and (d) should be dropped rather than mechanically reapplied.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp` — the 5 `%lu`→`%zu` format-string sites, each with a `size_t` argument: `:34103` (`ggml_backend_sycl_buffer_type_alloc_buffer`, `"...can't allocate %lu Bytes of memory on device\n"`), `:35949` (`ggml_backend_sycl_split_buffer_init_tensor`'s `snprintf`, "...on device"), `:38634` (`ggml_backend_sycl_host_compute_buffer_alloc`, "...of host memory for TP"), `:39505` (`ggml_sycl_pool_host::alloc`, "...on host"), and `:39298` (`ggml_sycl_pool_leg::alloc`, `GGML_LOG_WARN("...can't allocate %lu bytes (neither VRAM nor host-pinned)\n"`). Do not add `ggml/src/ggml-cpu/CMakeLists.txt` to this ticket — it is outside `ggml/src/ggml-sycl/` keep-ours scope and the fix already arrives with the merge itself. Skip the `element_wise.cpp`/`norm.cpp`/`set_rows.cpp`/`fattn-mkl.cpp` hunks — not applicable to the fork's current code.
**Ownership surface:** none — cosmetic/warning fixes only; the format-string sites do not change what or how much is allocated.
**Ticket:** llama.cpp-oqv9

### 19. `1cb3f5eb41d5` — sycl: Update gate logic for Alchemist GPUs regarding OneDNN features. (#26635)
**Files:** ggml/src/ggml-sycl/fattn-onednn.cpp
**Class:** n-a
**Why:** Loosens upstream's blanket `arch != bmg_g21 && arch != bmg_g31 → return false` oneDNN-SDPA gate (which restricted the fused-SDPA path to Battlemage only, citing an oneDNN bug — issue #5510 — that returns wrong results for some shapes, e.g. head_dim=64, on Alchemist/Arc-A `dg2_g10/g11/g12`) into a narrower per-shape gate: block only `K->ne[0] == 64` on Alchemist, allow everything else on every architecture. Confirmed the fork's `fattn-onednn.cpp` has **no architecture-based gate at all** — no `gpu_arch::`, `bmg_g21`/`bmg_g31`/`dg2_g1*`, and no `ggml_sycl_flash_attn_ext_onednn_supported` symbol anywhere in the file (consistent with Part A entry 4: the fork's oneDNN-SDPA path is an independently-rewritten superset, `MATERIALIZE_REQUIRED` planning + `sdpa_partition_cache`, past the point this whole gating mechanism lived at). This fork's hardware is exclusively Battlemage (B70 = `bmg_g31`, B50 = `bmg_g21`) plus an Arrow Lake-S iGPU (Xe-LPG, not Alchemist) — it never runs the `dg2_g10/g11/g12` architectures this commit's gate even discriminates on, so porting it would have zero observable effect even setting aside the missing gate mechanism.

### 20. `9e89a196b814` — sycl : Add Q5_K ESIMD kernel (#26376)
**Files:** ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** n-a
**Why:** Adds a fourth DMMV ESIMD kernel type (Q5_K) on top of entry 4's `esimd.hpp`/`g_ggml_sycl_enable_esimd` infrastructure, defaulted ON the same way. Same disposition as entry 4, with the same evidence — the fork's own `dmmv-esimd.hpp` (406 lines, `GGML_SYCL_DMMV_ESIMD` env var, default off) already carries an in-file measurement of 3.5x slower (21.10 vs. 6.06 t/s) for the same DMMV-ESIMD direction, plus a `// TODO: Add Q8_0, Q4_K, etc.` at its `dmmv_esimd_supported()` gate. **To be precise about what that measurement actually covers:** it (and the CLAUDE.md rule it corroborates, "1.9x SLOWER on Arc B580 + oneAPI 2025.3") was taken on **Q4_0**, not Q5_K. This entry's rejection of Q5_K ESIMD is an extrapolation from a Q4_0 measurement via the shared "block granularity too small to amortize LSC loads" mechanism (a mechanism, not a per-type guarantee) — not a cited Q5_K number. Confirmed `esimd.hpp` and `g_ggml_sycl_enable_esimd` are absent from the fork (see entry 4).

### 21. `ff14356e0caf` — sycl : add Q2_K reordered MMVQ and ESIMD kernels (#26336)
**Files:** ggml/src/ggml-sycl/convert.cpp, ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/quants.hpp, ggml/src/ggml-sycl/vecdotq.hpp
**Class:** superseded
**Why:** Two bundled changes for Q2_K: (a) a reordered-SOA MMVQ kernel (non-ESIMD), and (b) a DMMV ESIMD kernel on entry 4's infrastructure. (a) extends the same reorder-dispatch mechanism Part A entry 1 already classified superseded ("the fork's `ggml_sycl_supports_reorder_dmmv()` returns `true` only for `GGML_TYPE_Q4_0`... already dropped by the fork's own rewrite of the reorder/SOA-layout system... deliberately narrowed") — Q2_K reordered-MMVQ support is the same kind of extension to a mechanism the fork intentionally scoped down to Q4_0. (b) is the entry-4 ESIMD-dequant direction, already declined. Note: this exact commit was reverted upstream one commit later (entry 22) due to a bug and re-landed with a fix two commits after that (entry 23, "add gate params") — see that entry.

### 22. `7a0e42fd01fb` — Revert "sycl : add Q2_K reordered MMVQ and ESIMD kernels (#26336)" (#27486)
**Files:** ggml/src/ggml-sycl/convert.cpp, ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/quants.hpp, ggml/src/ggml-sycl/vecdotq.hpp
**Class:** n-a
**Why:** A pure `git revert` of entry 21, upstream's own response to a bug in that commit (fixed and re-landed as entry 23). It has no independent content to classify — it is byte-for-byte the inverse of entry 21's diff. Since entry 21 is itself superseded for the fork (reorder mechanism deliberately narrowed to Q4_0; ESIMD direction declined), this revert has no bearing on the fork either way — there is nothing here to port or decline that entries 21/23 don't already cover.

### 23. `86722900390a` — sycl : add Q2_K reordered MMVQ and ESIMD kernels (again) (#27490)
**Files:** ggml/src/ggml-sycl/convert.cpp, ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/quants.hpp, ggml/src/ggml-sycl/vecdotq.hpp
**Class:** superseded
**Why:** Re-lands entry 21 (undoing entry 22's revert) with one real fix on top ("add gate params") — diffed against entry 21, the only substantive delta is in `mul_mat_vec_q2_K_q8_1_sycl_switch_ncols`, which now threads a `vgate`/`glu_op` parameter pair through the reordered-MMVQ kernel launch to match the fused-GLU MMVQ ABI entry 9 (`650913862270`) introduced in between. That dependency is itself on a superseded mechanism (entry 9's `ggml_sycl_can_fuse`-based fusion, absent from the fork). Disposition is otherwise identical to entry 21: the reordered-MMVQ half extends a reorder mechanism the fork deliberately narrowed to Q4_0, and the ESIMD half is the declined small-block-dequant-on-ESIMD direction.

### 24. `814d84bc9da5` — sycl : mark tq2_0 as not supported (#27660)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/set_rows.cpp
**Class:** port-candidate
**Why:** Upstream's pre-patch `do_ggml_backend_sycl_device_supports_op` claimed TQ2_0 support in several op checks with no matching kernel arm; this commit adds explicit `TQ2_0 → return false` refusals plus more diagnostic `GGML_ABORT` messages. **Verifying each claim individually against the fork (`ggml_backend_sycl_device_supports_op`, `ggml-sycl.cpp:98736`) rather than name-grepping upstream's checks shows the fork's actual gap is narrower and in different places than a literal port would fix** — a lesson worth stating explicitly: against a fail-closed **allowlist**, the absence of a type name means it is *already* refused, so a plain "does `TQ2_0` appear near this check" probe fails open backwards (it flags sites that need nothing and misses the ones that do). Two of the three sites upstream patches are already safe in the fork:
  - `MUL_MAT`/`MUL_MAT_ID` end in `ggml_sycl_mul_mat_type_supported()` (`:98706`, called at `:98905`) — a positive allowlist of ~21 types that does not include `TQ2_0`, so `MUL_MAT` on `TQ2_0` is already refused. (`MUL_MAT_ID`/`ADD_ID` themselves `return true` unconditionally at `:98745`, *before* reaching that allowlist or any switch — see below, this is the real gap for those two ops.)
  - `SET_ROWS` (`:98932`) already has a positive dst-type allowlist (`:98953-98958`: F32/F16/BF16/fp8-e4m3/Q8_0/Q5_1/Q5_0/Q4_1/Q4_0/IQ4_NL) that excludes `TQ2_0` — already refused.

  The fork's real, confirmed-live gaps are two different sites than a literal port of this commit would touch:
  1. **`CPY`'s same-type contiguous passthrough** (`:98985`): `if (src0_type == src1_type && ggml_is_contiguous(...) && src0_type != GGML_TYPE_BF16) return true;` accepts a `TQ2_0`→`TQ2_0` contiguous copy unconditionally (`TQ2_0` isn't excluded the way `BF16` is) — but `cpy.cpp`'s dispatch (`ggml_sycl_cpy`, its type-pair `if`/`else if` chain ending at `cpy.cpp:1505-1507`) has no `TQ2_0`↔`TQ2_0` arm, so the op is accepted by `supports_op` and then hits the chain's `else { GGML_ABORT("fatal error"); }`.
  2. **`MUL_MAT_ID`/`ADD_ID` return `true` unconditionally** (`:98745`, before any type check): confirmed live — `if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID) { ...; return true; }` sits ahead of `ggml_sycl_mul_mat_type_supported()`, so upstream's in-`ggml_sycl_mul_mat_type_supported`-switch refusal (which this task's rev-list commit adds) is architecturally unreachable for those two ops in the fork's current structure — the refusal would need to land at this earlier unconditional-return site instead.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp`, `ggml_backend_sycl_device_supports_op` — the `GGML_OP_CPY` same-type-contiguous fast path (`:98985`) needs a `TQ2_0` exclusion alongside the existing `BF16` one, and the `GGML_OP_ADD_ID`/`GGML_OP_MUL_MAT_ID` unconditional-`true` block (`:98745`) needs a `TQ2_0` check before its `return true`. (The `MUL_MAT`/`SET_ROWS` allowlists need no change — already refuse `TQ2_0`.)

The `GGML_ABORT` diagnostic-message component is a mixed result, checked per file rather than assumed to carry: `set_rows.cpp` has exactly **one** `GGML_ABORT` (`:983`, `GGML_ABORT("Unsupported tensor type!");`) — this matches upstream's pre-patch text verbatim, so upstream's improved message (naming `src0`/`src1`/`dst` types) applies cleanly there. `mmvq.cpp` has **11** `GGML_ABORT` occurrences (`:3828`, `:21355`, `:21375`, `:21381`, `:21781`, `:21999`, `:22136`, `:22143`, `:22150`, `:23062`, `:23099`) but **none matches upstream's exact pre-patch text** (`default: GGML_ABORT("fatal error: unsupport data type=%s\n", ...)` inside `ggml_sycl_op_mul_mat_vec_q`'s per-type switch) — the fork's `ggml_sycl_op_mul_mat_vec_q` (`:22931`) has been rewritten to dispatch through the MMVQ-streaming mechanism rather than an inline per-`src0->type` switch, and contains no default-case type-name abort at all. The closest analogue is the `:22150` default in the delegated dispatch switch, which a port should improve: `ggml_sycl_op_mul_mat_vec_q` delegates to `ggml_sycl_mmvq_dispatch` (`:21787`), whose own `switch (src0->type)` (declared `:21816`) ends with `default: GGML_ABORT("fatal error");` at `:22150` — carrying no type name or any other diagnostic detail at all, strictly worse than upstream's pre-patch message. Naming this "not a mechanical port" is precise, not vague: it is a different function than the one upstream's diff touches, its per-type case bodies branch further on reorder mode/layout rather than upstream's flatter per-type dispatch, and the message to add has to be written fresh rather than copied — but the type discriminant (`src0->type`) and the improvement's intent (name the type in the abort) are the same. Applying the same "name the offending type" improvement there is a reasonable extension of this commit's intent, not a copy of its diff.
**Landing zone (GGML_ABORT component):** `ggml/src/ggml-sycl/set_rows.cpp:983` — improve to name `src0`/`src1`/`dst` types, matching upstream's fixed message (direct port). Optionally also `ggml/src/ggml-sycl/mmvq.cpp:22150` (`ggml_sycl_mmvq_dispatch`'s default case) — not a mechanical port, but the same diagnosability gap upstream's commit targets.
**Ownership surface:** none — supports-op refusal logic and diagnostic strings only; no allocation.
**Ticket:** llama.cpp-v7lu
