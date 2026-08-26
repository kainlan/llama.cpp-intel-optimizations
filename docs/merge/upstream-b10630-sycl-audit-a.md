# Upstream SYCL Audit — Part A (commits 1–25 of 49)

Enumerated via `git rev-list --reverse 81ff7abe5..b10630 -- ggml/src/ggml-sycl | head -25`.
Classified per the rubric in `docs/plans/2026-08-25-phase-c-upstream-merge.md` Task 6
(N/A → superseded → port-candidate, in that order; diff evidence, not commit titles).
Merge-base: `81ff7abe5`. Upstream target: tag `b10630` (remote `ggml-org`). Fork side:
`master @ d37e4cedb269` ("fix(merge-guards): symmetric rc-check + non-vacuity on both
safety-net halves") — the commit `master` pointed to when this fix round's fork-state
greps (source line numbers, symbol presence/absence, allocation-pattern counts) were run.
Pinned because `master` keeps moving under a shared checkout; every fork-state citation
below is scoped to this SHA unless stated otherwise.

The rubric includes an ownership screen (owner directive, mid-flight): every port-candidate
is checked against the fork's canonical memory-ownership contract (CLAUDE.md "SYCL Memory
Ownership"; `docs/design/sycl-canonical-memory-architecture.md`;
`docs/backend/sycl-memory-design.md`) — raw `sycl::malloc_*`/TLSF/side-cache/pool/scratch
allocation outside unified-cache code is never port-candidate-as-written (re-expression
through `unified_allocate`/`unified_allocate_owner` → `mem_handle` is required instead);
raw device pointers as cache/identity keys get the same treatment; forced eviction/reap/
zone-reset-style reclamation is N/A by design. Each port-candidate below names the
ownership surface (or "none touched") its landing zone would consume.

**25 commits: 16 port-candidate / 9 superseded / 0 n-a**

## Cross-cutting items

🔶 **CARRY TO TRACKER (general, not scoped to any single entry's files):** 18 live
`ggml_sycl_pool_alloc`/`ctx.pool()` call sites remain fork-wide — a candidate unified-cache
migration backlog for T8 to file as ONE ticket. Verified per file (`cat <file> | grep -cE
'ggml_sycl_pool_alloc|ctx\.pool\(\)'`, positive-controlled against `fattn-onednn.cpp`'s
already-migrated 0) and confirmed each hit is a real call site, not the template's own
definition (`common.hpp:3215-3259` is the `ggml_sycl_pool_alloc` struct/ctors/dtor/deleted
members — infrastructure, not a site to migrate, and should be removed last, once no
callers remain):
- `ggml-sycl.cpp` (13): lines 41129, 41257, 41310, 41323, 41342, 41343, 52068, 58718,
  58719, 58893, 58894, 59989, 59990
- `conv3d.cpp` (3): lines 84, 104, 105
- `cross_entropy_loss.cpp` (2): lines 124, 175

Two further mentions are comments, not call sites, and don't count toward the 18:
`mmvq.cpp:15998` (documents a graph-mode workaround) and `fattn-buffers.hpp:54` (documents
a replacement wrapper).

---

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

### 7. `ae9291e16b97` — sycl : support kernel type fp16 for conv2d_dw (#25653)
**Files:** ggml/src/ggml-sycl/conv2d-dw.cpp
**Class:** port-candidate
**Why:** Adds an F16 kernel-tensor code path to depthwise conv2d. The fork's `conv2d-dw.cpp` asserts `kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32` unconditionally — F16 conv2d_dw is simply unimplemented, not superseded by anything; this op sits outside every subsystem the fork rewrote (mul_mat/fattn/memory/graph/MoE).
**Landing zone:** `ggml/src/ggml-sycl/conv2d-dw.cpp` (the type-assert + kernel dispatch).
**Ownership surface:** none — operates in-place on the existing input/kernel/dst tensor buffers; no allocation.

### 8. `d3fba0c79db8` — sycl : fix get_rows Q2_K, Q4_K, Q5_K (#25656)
**Files:** ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/getrows.cpp
**Class:** port-candidate
**Why:** Adds `dequantize_{q2_K,q4_K,q5_K}_f32` helpers and wires `get_rows` dispatch for those three types. The fork's `getrows.cpp` dispatch switches (~lines 2054, 2829) cover F16/F32/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K only — Q2_K, Q4_K and Q5_K are absent from `get_rows` entirely, and none of the three `_f32` dequantize helpers exist in `dequantize.hpp`. This is exactly the rubric's "dequant tables in dequantize.hpp... shared types" port-candidate category — a real coverage gap, not a rewritten path.
**Landing zone:** `ggml/src/ggml-sycl/dequantize.hpp` (add the three `_f32` functions) + `ggml/src/ggml-sycl/getrows.cpp` (add the three `case GGML_TYPE_*` arms to both dispatch switches).
**Ownership surface:** none — pure per-element dequantize math reading existing weight-buffer bytes; no allocation.

### 9. `0bd0ec60998d` — sycl: fix row calculation when K_QUANTS_PER_ITERATION is 1 (#25690)
**Files:** ggml/src/ggml-sycl/dmmv.cpp
**Class:** port-candidate
**Why:** Fixes an off-by-one (`row > nrows` → `row >= nrows`) across the base (non-reorder) Q2_K/Q3_K/Q4_K/Q6_K DMMV kernels, plus a second-half fix that only applies to the *reorder* Q5_K kernel. Verified against the fork's current source: `dmmv.cpp:1981` (q2_k), `:2089` (q3_k), and `:2203` (q4_k) all still read `if (row > nrows)`, while q6_k (`:2475`) is already `>=` — three of the four sites genuinely need the same one-character fix. Upstream's diff also touches the non-reorder Q5_K kernel (`nrows` parameter, multi-row calculation, `>=` guard, KQPI-style launcher), but the fork's own non-reorder Q5_K kernel already distributes both `im` halves across different threads in the work-group (`:2366-2374`) and its launcher (`:2941-2952`) already grids exactly `nrows` groups with `local_range(1)==1`, making an `nrows` guard structurally inert there; upstream's separate im-loop restructure lands in the *reorder* Q5_K kernel, unreachable in this fork (`ggml_sycl_supports_reorder_dmmv()` returns true only for `GGML_TYPE_Q4_0`).
**Landing zone:** `ggml/src/ggml-sycl/dmmv.cpp`, the three `if (row > nrows)` guards in `dequantize_mul_mat_vec_q2_k` (`:1981`), `_q3_k` (`:2089`), and `_q4_k` (`:2203`) — change to `>=`. No change needed to Q5_K (non-reorder is correct and its launcher makes the guard inert; the reorder-side fix is unreachable).
**Ownership surface:** none — a one-character bounds-comparison fix over existing device buffers; no allocation.

### 10. `8e8681e0e208` — sycl(build): parallelize ocloc invocations (#25903)
**Files:** ggml/src/ggml-sycl/CMakeLists.txt
**Class:** port-candidate
**Why:** Adds `-fsycl-max-parallel-link-jobs=<nproc>` to the `target_link_options` guarded by `if (GGML_SYCL_DEVICE_ARCH)` / `spir64_gen`. In the fork, that exact `GGML_SYCL_DEVICE_ARCH` block (`CMakeLists.txt:836-838`) is scoped to the AMD HIP `--offload-arch` path, not Intel AOT — the fork's own Intel Battlemage AOT uses a separate `GGML_SYCL_BMG_AOT_ENABLED`/`intel_gpu_bmg_g21` block with no parallel-link-jobs flag. The upstream hunk's literal context doesn't exist in the fork, but the underlying idea (parallelize `ocloc` invocations during AOT link) is a real, low-risk build-time win worth re-applying to the fork's own AOT block — build time is an explicit repo pain point (CLAUDE.md: ~10 min with ccache / ~25 without).
**Landing zone:** `ggml/src/ggml-sycl/CMakeLists.txt`, the `GGML_SYCL_BMG_AOT_ENABLED` block's `target_link_options(ggml-sycl ...)` (~line 556-560).
**Ownership surface:** none — build-system/link-flag change, no runtime allocation.

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

### 15. `1c5b89ff6315` — sycl : support dev2dev memcpy by DEV2DEV_MEMCPY_FORWARD (#26234)
**Files:** ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Adds a third `ggml_sycl_dev2dev_memcpy_mode` (`DEV2DEV_MEMCPY_FORWARD`) alongside `SYCL`/`L0`, selected via `g_ggml_sycl_dev2dev_memcpy`. None of `DEV2DEV_MEMCPY_SYCL/L0/FORWARD`, the enum, or `g_ggml_sycl_dev2dev_memcpy` exist in the fork — its device-to-device copy (`dev2dev_memcpy`, `ggml-sycl.cpp:33460`) is a `mem_handle`-based function integrated with the unified cache and the fork's documented no-direct-P2P/host-bounce topology (no direct P2P between the B70 and B50 — re-verified 2026-07-31), not an env-var-selectable mode switch.

### 16. `d5d3e05bf8d2` — [SYCL] support the missed types in cpy (#26005)
**Files:** ggml/src/ggml-sycl/cpy.cpp, ggml/src/ggml-sycl/cpy.hpp
**Class:** port-candidate
**Why:** Adds `cpy_blck_f32_q2_0` (F32→Q2_0 quantize) and `cpy_blck_q2_0_f32` (Q2_0→F32 dequantize), plus their dispatch-table pairing (paired with entry 14's Q2_0 mul_mat support). The fork's `cpy.cpp` dispatch (grep of `case GGML_TYPE_*` pairs) covers F32/F16/BF16/I16/I32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL pairs but has no `Q2_0` arm at all — this is additive coverage in the same shared-conversion-table category, not a rewrite.
**Landing zone:** `ggml/src/ggml-sycl/cpy.hpp` (new `cpy_blck_f32_q2_0` and `cpy_blck_q2_0_f32`) + `ggml/src/ggml-sycl/cpy.cpp` dispatch table, alongside entry 14.
**Ownership surface:** none confirmed in the diff (`grep`'d `cpy.cpp`/`cpy.hpp` for `malloc`/`pool_alloc` — no hits) — per-block quantize/copy math over existing src/dst buffers; no allocation.

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

### 19. `66fa168a5617` — Extended SYCL oneDNN SDPA to non-FP16 KV caches (Q4_0–Q8_0 and FP32) (#25874)
**Files:** ggml/src/ggml-sycl/fattn-onednn.cpp, ggml/src/ggml-sycl/fattn.cpp
**Class:** port-candidate
**Why:** Extends the oneDNN SDPA gate to accept Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and F32 KV by dequantizing/converting K/V to dense F16 before the SDPA graph (prefill-only, K≥1024, Q≥32). The fork's `ggml_sycl_flash_attn_ext_onednn_supported()` (`fattn-onednn.cpp:277`) still hard-gates `params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16` — quantized-KV prefill (which the fork's own perf baselines explicitly benchmark, e.g. "q8_0 KV" runs) cannot reach the oneDNN path today and falls back to the slower native kernel. This is a genuine capability gap in a subsystem the fork already extended once (`MATERIALIZE_REQUIRED` K/V materialization for GQA) — reusing that same materializer machinery for a dequant-to-F16 conversion is a natural, moderate-risk extension rather than a duplicate.
⚠️ **Ownership screen:** upstream's own diff stages the converted K/V into `ggml_sycl_pool_alloc<sycl::half>` scratch (`ctx.pool()`) — that is upstream's code, not the fork's. Verified against the fork's actual current `fattn-onednn.cpp`: `grep -cE 'ggml_sycl_pool_alloc|ctx\.pool\(\)'` returns **0** there (vs. **11** in `git show b10630:.../fattn-onednn.cpp`, upstream's version) — the fork's own pre-existing K/V materialization scratch is already routed through the unified-cache-backed materializer (see the file's own comments, e.g. "ask the unified-cache-backed materializer for dense f16 K/V"), i.e. already on a sanctioned surface. **This re-expression requirement applies only to the NEW scratch this port itself would add** for dequantizing quantized/F32 KV to F16 — that new code must go through `unified_allocate`/`unified_allocate_owner()` → `mem_handle` (or an `alloc_owner` via `mem_handle::from_owned_alloc()`), not `ctx.pool()`, matching the standard the file's existing code already meets.
(A general legacy-`ctx.pool()` backlog exists fork-wide but is unrelated to this file/port — see "Cross-cutting items" above.)
**Landing zone:** `ggml/src/ggml-sycl/fattn-onednn.cpp`, `ggml_sycl_flash_attn_ext_onednn_supported()` gate (line ~277) and the K/V staging call sites feeding the `MATERIALIZE_REQUIRED` plan.
**Ownership surface:** `mem_handle` lease over a unified-cache scratch allocation for the dequantized K/V dense-F16 buffers — NOT `ggml_sycl_pool_alloc`/`ctx.pool()` as upstream wrote it.

### 20. `6c8dcaa7ae41` — sycl: parallelize the non-contiguous concat kernel (#25852)
**Files:** ggml/src/ggml-sycl/concat.cpp
**Class:** port-candidate
**Why:** Launch-geometry-only fix: the non-contiguous concat kernel launched a single-lane `(1,1,1)` work-group; this widens it to `(1,1,SYCL_CONCAT_BLOCK_SIZE)`. Measured upstream at +9.4% PP2048 on Arc Pro B70 — this fork's exact hardware. Confirmed the fork's `concat_T_sycl_non_cont` (`concat.cpp:130`) still launches `sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1))` — the bug is present, unmodified, and the fix is a narrow, low-risk launch-geometry change with no interaction with any rewritten subsystem.
**Landing zone:** `ggml/src/ggml-sycl/concat.cpp`, function `concat_T_sycl_non_cont`.
**Ownership surface:** none — launch-geometry change only (work-group shape), no allocation.

### 21. `c074cb3f763d` — sycl : enhance OP set_rows to support all missed data types (#26515)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/set_rows.cpp
**Class:** port-candidate
**Why:** Adds `set_rows` support for `Q2_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_XS`. The fork's `set_rows.cpp` dispatch (~lines 935-976) covers only F32/F16/BF16/Q8_0/Q5_1/Q5_0/Q4_1/Q4_0/IQ4_NL — every type this commit adds is genuinely absent from the fork. Same shared-quant-table category as entries 8/14/16.
**Landing zone:** `ggml/src/ggml-sycl/set_rows.cpp` dispatch switch(es), reusing `dequantize.hpp`/`vecdotq.hpp` quantize helpers where already ported (entry 14) and adding the remaining IQ-type quantizers.
**Ownership surface:** none confirmed in the diff (`grep`'d `set_rows.cpp` for `malloc`/`pool_alloc` — no hits) — per-row quantize math writing into the existing destination tensor's buffer; no allocation.

### 22. `eef5f3e3430a` — sycl : fix error Error OP FLASH_ATTN_EXT on arc770 (#26441)
**Files:** ggml/src/ggml-sycl/fattn-vec.hpp
**Class:** port-candidate
**Why:** Converts a runtime `if (D <= 256 && nthreads == 256)` into `if constexpr (D <= 256) { if (nthreads == 256) {...; return;} }` in `ggml_sycl_flash_attn_ext_vec_case_impl`, fixing a template-instantiation bug that entry 3 (`c1063ac9d75f`, the Battlemage-256-thread change) introduced: at `D > 256` the `nthreads_hw=256` branch was still being *instantiated* (just not executed), overflowing 64 KB work-group local memory. This has no independent value without entry 3 and does not apply to the fork's current flat-macro `fattn-vec.hpp` — it is a required companion fix that must land in the same change as entry 3's port.
**Landing zone:** Same as entry 3 — the arch-conditional launch dispatch in `ggml/src/ggml-sycl/fattn-vec.hpp`; use `if constexpr` from the start rather than porting entry 3's runtime `if` and immediately following with this fix.
**Ownership surface:** none — compile-time template-instantiation fix, no allocation.

### 23. `31558dbb7657` — sycl : Support DSv4 OPs: LIGHTNING_INDEXER,DSV4_HC_COMB,DSV4_HC_POST,DSV4_HC_PRE (#26568)
**Files:** ggml/src/ggml-sycl/dsv4-hc.cpp (new), ggml/src/ggml-sycl/dsv4-hc.hpp (new), ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/lightning-indexer.cpp (new), ggml/src/ggml-sycl/lightning-indexer.hpp (new)
**Class:** port-candidate (post-merge — blocked on this Phase C merge landing)
**Why:** Implements SYCL kernels for four DeepSeek-v4-specific ggml ops that are unreachable against pre-merge master (`GGML_OP_LIGHTNING_INDEXER`/`GGML_OP_DSV4_HC_*` don't exist in today's `ggml/include/ggml.h`), but the blocker is satisfied by this exact merge: confirmed both prerequisite commits are inside `81ff7abe5..b10630` (`git merge-base --is-ancestor`) — `00f5442cc4e8` adds `GGML_OP_LIGHTNING_INDEXER` (ggml.h/ggml.c/CPU ops, no SYCL) and `0dc74e332ede` adds `GGML_OP_DSV4_HC_COMB/PRE/POST` (ggml.h/ggml.c/CPU/CUDA, no SYCL) — so the enum, CPU reference, and graph-building support all land with the merge itself, not as a separate out-of-scope commit. The fork's dispatch has diverged from upstream's single `do_ggml_backend_sycl_device_supports_op` switch (the fork's equivalent is `ggml_backend_sycl_device_supports_op` at a different line, and there are additional `GGML_OP_*` switches for node classification and storage-readiness gating that don't exist upstream), so upstream's ~35-line switch addition is a reimplementation against the fork's dispatch shape, not a copy.
🚫 **Do NOT port** the `docs/ops/SYCL.csv` churn this commit carries (23,541 lines) — that file is an auto-generated ops-support matrix, not hand-maintained content.
**Landing zone:** new `ggml/src/ggml-sycl/{dsv4-hc,lightning-indexer}.{cpp,hpp}` (fork-native kernel implementations) + fork-native insertions into `ggml_sycl_compute_forward`'s op switch and `ggml_backend_sycl_device_supports_op`'s capability switch in `ggml-sycl.cpp`.
**Ownership surface:** none — clean against the screen's allocation patterns (verified with a positive control: the same `malloc`/`pool_alloc`/TLSF grep against this commit's SYCL diff returns zero hits, while the identical grep against entry 19's diff — known to allocate — returns 6, so the zero here isn't vacuous); ticket-writer must confirm the kernels need no scratch beyond work-group local memory.

### 24. `6b5c2efb4e2f` — sycl: *glu flat path (#26354)
**Files:** ggml/src/ggml-sycl/element_wise.cpp, ggml/src/ggml-sycl/presets.hpp
**Class:** port-candidate
**Why:** Consolidates the fused-GLU kernels (geglu/reglu/swiglu/geglu_erf/geglu_quick) behind one shared launcher and adds a contiguous fast path when `o0 == n && o1 == n` (split-GLU operands), collapsing the de-interleave index math to identity. Measured +14% f16 / +4% f32 on `SWIGLU` on Arc Pro B70 — this fork's own hardware. The fork's `element_wise.cpp` still has the separate, non-consolidated `gated_op_fused_{geglu,reglu,swiglu,...}` kernels taking `(o0, o1)` with no contiguous-fast-path branch — the optimization is genuinely missing, not superseded (this is the same file as entry 12, but a different, additive change to the GLU-specific kernels rather than the generic unary-op fast path entry 12 already covers).
**Landing zone:** `ggml/src/ggml-sycl/element_wise.cpp`, the `gated_op_fused_*` kernel family and their `ggml_sycl_op_{geglu,reglu,swiglu}` callers.
**Ownership surface:** none — kernel-consolidation and an index-math fast path over existing src/dst buffers (`grep`'d for `malloc`/`pool_alloc` in the diff — no hits); no allocation.

### 25. `fc3f10b3895e` — sycl: fix UE4M3 parsing (#25608)
**Files:** ggml/src/ggml-sycl/common.hpp
**Class:** port-candidate
**Why:** Fixes NVFP4's unsigned UE4M3 scale-byte decode, which upstream's *old* code mishandled by reinterpreting it as signed E4M3. The fork's `ggml_sycl_ue4m3_to_fp32()` (`common.hpp:6797`) already uses the same corrected unsigned-decode formula as this commit's fix, but retains an extra `x == 0xff → 0.0f` special case left over from the old buggy code that upstream's fixed formula explicitly removed. This divergence is latent, not live: `ggml_fp32_to_ue4m3` saturates at 0x7E and never sets bit 7, so no real encoder can ever emit byte `0xFF` — the fork's own test documents exactly this (`ggml/src/ggml-sycl/tests/test-q1-nvfp4-adapter-device.cpp:271-276`: "0xFF is deliberately absent because ggml_sycl_ue4m3_to_fp32 maps it to 0.0f while ggml_ue4m3_to_fp32 does not -- a pre-existing divergence in common.hpp that no encoder can reach"). Narrow, low-risk reconciliation of a dead branch, not a live-data bug.
**Landing zone:** `ggml/src/ggml-sycl/common.hpp`, function `ggml_sycl_ue4m3_to_fp32` — drop the `x == 0xff` branch of the leading guard; `ggml/src/ggml-sycl/tests/test-q1-nvfp4-adapter-device.cpp` (its comment at lines 271-276 documents the divergence this drop resolves and goes stale once the branch is gone).
**Ownership surface:** none — a pure scalar decode function, no allocation.
