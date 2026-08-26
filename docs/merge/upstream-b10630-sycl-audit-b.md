# Upstream SYCL Audit — Part B (commits 26–49 of 49)

Enumerated via `git rev-list --reverse 81ff7abe5..b10630 -- ggml/src/ggml-sycl | tail -n +26`.
Classified per the rubric in `docs/plans/2026-08-25-phase-c-upstream-merge.md` Task 6/7
(N/A → superseded → port-candidate, in that order; diff evidence, not commit titles).
The rubric includes an ownership screen (owner directive, mid-flight): every port-candidate
is checked against the fork's canonical memory-ownership contract (CLAUDE.md "SYCL Memory
Ownership"; `docs/design/sycl-canonical-memory-architecture.md`;
`docs/backend/sycl-memory-design.md`) — raw `sycl::malloc_*`/TLSF/side-cache/pool/scratch
allocation outside unified-cache code is never port-candidate-as-written (re-expression
through `unified_allocate`/`unified_allocate_owner` → `mem_handle` is required instead);
raw device pointers as cache/identity keys get the same treatment; forced eviction/reap/
zone-reset-style reclamation is N/A by design. Each port-candidate below names the
ownership surface (or "none touched") its landing zone would consume.

Several commits in this half touch only trivially inside `ggml/src/ggml-sycl` while the
substantive change is a cross-backend/core-ggml feature (load-mode auto, `ggml_rope_set_offset`).
Per the Task 7 gotcha, a new file is classified against the fork subsystem it would slot
into; the mirror case — a core-scoped commit whose SYCL touch is a one-line, meaningless-
without-the-core-plumbing hunk — is classified N/A as out of this SYCL-only audit's scope,
with a pointer to the core-ggml brief (`docs/merge/briefs/core-ggml.md`) that already covers
the substantive change.

**24 commits: 11 port-candidate / 6 superseded / 7 n-a**

---

### 1. `f8e30266d` — sycl: coalesce the ssm_conv window loads (#26612)
**Files:** ggml/src/ggml-sycl/ssm_conv.cpp
**Class:** port-candidate
**Why:** Reorders `kernel_ssm_conv`'s flattened-index decomposition from channel-fastest to token-fastest, coalescing the per-thread `d_conv` window loads (measured 1.85–1.87x on `test-backend-ops perf -o SSM_CONV`, +1.8–2.2% end-to-end `llama-bench` pp2048 on a Mamba-family model). The fork's `ssm_conv.cpp` (`kernel_ssm_conv`) is byte-for-byte the pre-patch layout (`channel = idx % d_inner; token = (idx / d_inner) % n_t`) — confirmed live. SSM_CONV is not part of any subsystem the fork rewrote (mul_mat/fattn/memory/graph/mmvq/mmq); pure index-math reordering over the existing kernel.
**Landing zone:** `ggml/src/ggml-sycl/ssm_conv.cpp`, function `kernel_ssm_conv` (the `channel`/`token`/`seq` decomposition).
**Ownership surface:** none — index-math reordering only, same src/weights/dst pointers, no allocation.

### 2. `153d324bc` — llama: add default load-mode auto, which avoids mmap on iGPUs (#26081)
**Files:** common/arg.cpp, common/common.h, ggml/include/ggml-backend.h, ggml/src/ggml-backend-meta.cpp, ggml/src/ggml-{blas,cann,cpu,cuda,et,hexagon,metal,opencl,openvino,rpc,sycl,virtgpu,vulkan,webgpu,zdnn,zendnn}/*, include/llama.h, src/llama-model-loader.cpp, src/llama-model.cpp, src/llama.cpp, tools/*
**Class:** n-a
**Why:** This is a cross-backend/core-ggml/llama feature (new `LLAMA_LOAD_MODE_AUTO`, the `--load-mode auto` CLI value, `ggml_backend_dev_props::mmap_support`, and the model-loader logic that picks mmap vs. host-pinned based on device unified-memory classification) — the SYCL touch is a single line (`/* .mmap_support = */ true,` in `ggml_backend_sycl_device_get_props`), meaningless without the enum, the CLI plumbing, and the loader decision logic none of which live under `ggml/src/ggml-sycl`. Confirmed the fork has **no** `LLAMA_LOAD_MODE_*` enum, no `--load-mode` flag, and no `mmap_support` field in `ggml_backend_dev_props` at all — this whole feature is absent, not narrowed or superseded. Per the Task 7 gotcha's mirror case: a core-scoped commit whose SYCL hunk is trivial and inseparable from the untouched core/llama plumbing is out of this SYCL-only audit's scope. The core-ggml conflict brief (`docs/merge/briefs/core-ggml.md`) already tracks `ggml_rope_set_offset` and related contract changes for this merge; `mmap_support`/load-mode auto belongs alongside it there, not in the Phase C SYCL-ports epic.
**Landing zone:** n/a (belongs to the core-ggml/llama merge wave, not a post-merge SYCL port).

### 3. `d415e65a5` — sycl : enhance concat to support Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (#26800)
**Files:** ggml/src/ggml-sycl/concat.cpp
**Class:** port-candidate
**Why:** Adds `concat_impl_{q4_0,q4_1,q5_0,q5_1,q8_0}_sycl` block-quantized concat paths (mirroring the existing `concat_impl_sycl<T>` template shape) plus a small unrelated correctness cleanup in the pre-existing float path (drops two spurious `.wait()`s after device-to-device memcpys that already complete in-order on the same queue). Confirmed the fork's `ggml_sycl_op_concat` dispatch switch (`concat.cpp`) only covers F32/F16/BF16/I32/I16/I64/I8 — none of the five quant types this commit adds exist in the fork's concat path at all. Same shared-conversion-table additive-coverage category as Part A entries 8/14/16/21 (dequant/cpy/set_rows type-table gaps); no rewritten subsystem is touched.
**Landing zone:** `ggml/src/ggml-sycl/concat.cpp`, the `ggml_sycl_op_concat` dispatch switch (new `case GGML_TYPE_Q4_0/Q4_1/Q5_0/Q5_1/Q8_0` arms) plus the new `concat_impl_q*_sycl` functions.
**Ownership surface:** none — the quantized paths use `stream->memcpy`/kernel writes into the pre-existing `dst->data` buffer, same as the float path; no allocation.

### 4. `8efbf65db` — sycl : Add DMMV ESIMD Q3_K kernel (#26251)
**Files:** docs/backend/SYCL.md, ggml/src/ggml-sycl/common.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp (new), ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** n-a
**Why:** Squashes in DMMV ESIMD kernels for Q4_K, Q6_K and Q3_K (per the commit's own sub-history), defaulting them ON via a new `g_ggml_sycl_enable_esimd` flag (`GGML_SYCL_ENABLE_ESIMD`, default 1) that takes priority over the fork's existing reorder-MMVQ dispatch whenever the compiler is `__INTEL_LLVM_COMPILER`. This is exactly the optimization direction CLAUDE.md documents as already measured and rejected for this fork's hardware: *"Small-block dequant (Q4_0/Q8_0/**Q4_K**) belongs on standard SYCL, not ESIMD. ESIMD measured 1.9x SLOWER on Arc B580 + oneAPI 2025.3 (block granularity too small to amortize LSC loads)"* — Q4_K is named explicitly, and the same block-granularity argument applies to Q3_K/Q6_K. Confirmed live: the fork's own opt-in retest hatch already exists (`convert.cpp`: `GGML_SYCL_ESIMD_DEQUANT` env var, `GGML_SYCL_ESIMD_DEQUANT_AVAILABLE` compile gate, default **OFF** — currently scoped to Q4_0 only), and there is no `dmmv-esimd`-style DMMV ESIMD kernel, no `g_ggml_sycl_enable_esimd`, and no `esimd.hpp` file matching this commit's shape anywhere in the fork (the fork's own `esimd.hpp`-adjacent files — `convert-esimd.hpp`, `dmmv-esimd.hpp`, `mmq-esimd.hpp`, `xmx-esimd-*.hpp` — are a different, independently-built ESIMD surface for MoE/XMX/Q4_0-dequant, all opt-in or narrowly scoped). Porting this commit as written (default-ON, three new K-quant types) would reintroduce the documented regression on this exact fork's cards.
**Landing zone:** n/a — contradicts a settled, hardware-measured architecture rule.

### 5. `1ee1cd9bc` — sycl: fuse UNARY(silu|sigmoid|softplus) + MUL (#26411)
**Files:** ggml/src/ggml-sycl/element_wise.cpp, ggml/src/ggml-sycl/element_wise.hpp, ggml/src/ggml-sycl/fusion.cpp, ggml/src/ggml-sycl/fusion.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, tests/test-backend-ops.cpp
**Class:** superseded
**Why:** Extends `ggml_sycl_can_fuse()` (the generic two-op graph-fusion detector `ggml_backend_sycl_graph_compute_impl`'s node loop calls to skip ahead) with a `{GGML_OP_UNARY, GGML_OP_MUL}` pattern for SiLU/Sigmoid/Softplus, and adds the `ggml_sycl_op_unary_mul_fused` kernel. This is exactly the `ggml_sycl_fuse()` framework Part A entries 2 and 13 already classified superseded ("the fork has neither `topk-moe.{cpp,hpp}` nor any `ggml_sycl_fuse` call site... graph-node dispatch is already deeply customized for cache/placement bookkeeping"). Confirmed live: `fusion.cpp`/`fusion.hpp` do not exist in the fork, and neither does `ggml_sycl_can_fuse` or `g_ggml_sycl_enable_fusion`. Measured upstream gain is also marginal (+0.38–0.62%, "within run-to-run spread" per the commit's own message) — low value on top of a rejected mechanism.

### 6. `154d57af3` — sycl: remove separate fp32 type promotion in gemm non-oneDNN path (#26372)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** In the non-oneDNN fallback GEMM branch of `ggml_sycl_op_mul_mat_sycl`, replaces an f16-in/f16-out `dpct::gemm` call that writes into a scratch `ggml_sycl_pool_alloc<sycl::half> dst_f16` buffer (freed via `ctx.pool()`) followed by a `to_fp32_sycl` conversion kernel, with a single `dpct::gemm` call that lets the library promote directly to f32 output — eliminating both the scratch allocation and the extra conversion kernel. Confirmed live: the fork's `ggml_sycl_op_mul_mat_sycl` (`ggml-sycl.cpp:41080`, dpct::gemm call at :41327) still has this exact `ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols)` pattern verbatim (only `oneapi::mkl::transpose` vs. the fork's `oneapi::math::transpose` differs — a pre-existing oneAPI-version rename elsewhere in the fork, not a functional divergence), and this fallback path is confirmed live and commented on at two other call sites (`ggml-sycl.cpp:56966`, `:60637`). Unlike most entries this is a rare *ownership-screen-favorable* port: it **removes** a `ggml_sycl_pool_alloc` scratch allocation rather than adding one, moving this call site slightly closer to canonical-contract compliance.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_sycl_op_mul_mat_sycl`, the non-oneDNN f16 GEMM branch (~line 41320–41335).
**Ownership surface:** none touched — net effect removes one `ggml_sycl_pool_alloc<sycl::half>` legacy pool allocation; no new allocation of any kind is introduced.

### 7. `a97123e49` — [SYCL] Support host pinned mem to improve SYCL Host-to-Device Memory Access (#26789)
**Files:** docs/backend/SYCL.md, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Replaces `ggml_backend_sycl_host_buffer_type_alloc_buffer`'s legacy `aligned_malloc_host` with a new `ggml_backend_sycl_host_malloc()` that calls raw `sycl::malloc_host()` directly on a bare device queue, freed via raw `sycl::free()`, gated by a new `g_ggml_sycl_enable_host_pinned_mem` env flag (default ON) — exactly the direct-allocation-outside-the-cache pattern the canonical memory contract forbids. Confirmed live: the fork's `ggml_backend_sycl_host_buffer_type_alloc_buffer` (`ggml-sycl.cpp:38474`) is already a complete, independently-built superset of this commit's goal — it routes through `ggml_sycl::unified_alloc()` with a full `alloc_request` (`alloc_role::WEIGHT`/`STAGING`, `runtime_category::HOST_COMPUTE`, `must_host_pinned = true`, zone-aware `use_pinned_pool`), wraps the result in a `mem_handle` via `detail::from_legacy_owned_alloc`, and `ggml_backend_sycl_host_buffer_type_get_max_size` (`:38392`) already reasons about per-zone largest-contiguous-free-block accounting that upstream's simple `get_max_mem_alloc_size()`/`SIZE_MAX` toggle (added one commit later, entry 17) doesn't have. Same disposition as Part A entry 5 (USM system allocation threshold, superseded by the same tiered unified-cache placement).

### 8. `3d9388535` — sycl: fuse the gated-delta-net state writeback cpy (#26643)
**Files:** ggml/src/ggml-sycl/gated_delta_net.cpp, ggml/src/ggml-sycl/gated_delta_net.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Adds `ggml_sycl_try_gdn_cache_fusion()`, a bespoke graph-pattern detector (gated behind `g_ggml_sycl_enable_fusion`, "ported from `ggml_cuda_try_gdn_cache_fusion`") wired into `ggml_backend_sycl_graph_compute_impl`'s node loop that lets the gated-delta-net kernel write its recurrent-state snapshots directly into a downstream KV-cache buffer, skipping a materialize-then-copy node. Although self-contained rather than routed through the generic `ggml_sycl_fuse()` helper (unlike entries 5/9/Part A 2/13), it still (a) depends on `g_ggml_sycl_enable_fusion`, which does not exist anywhere in the fork (confirmed: zero hits), and (b) inserts a new graph-level "detect pattern, write output elsewhere, skip N nodes" shortcut into the same `ggml_backend_sycl_graph_compute_impl` node loop the fork has extensively customized for cache/placement bookkeeping per node — exactly the risk Part A entry 2 flagged ("re-adding upstream's fusion shortcut would bypass that bookkeeping"). The underlying `ggml_sycl_op_gated_delta_net`/`gated_delta_net.cpp` op itself (with its pre-existing `K`/`keep_rs` rollback-slot support) is confirmed present and unmodified in the fork — only the fusion optimization on top of it is being declined.

### 9. `650913862` — sycl: fuse mul_mat(gate) + mul_mat(up) + GLU for q4_K dense FFN (#26779)
**Files:** docs/backend/SYCL.md, ggml/src/ggml-sycl/element_wise.{cpp,hpp}, ggml/src/ggml-sycl/fusion.cpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.{cpp,hpp}, tests/test-backend-ops.cpp
**Class:** superseded
**Why:** Adds `ggml_sycl_mul_mat_glu_mmvq_fused()`, gated by `ggml_sycl_can_fuse(cgraph, node_idx, {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU}, {})` — the same rejected `ggml_sycl_fuse()` framework as entry 5 (confirmed absent from the fork). **Ownership screen:** the fused kernel additionally stages its quantized activation into `ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), ...)` — a raw legacy `ctx.pool()` scratch allocation, the direct-allocation pattern the canonical contract disallows outside unified-cache surfaces — an independent reason this would need re-expression through `unified_allocate`/`mem_handle` even if the fusion framework itself were adopted. Two independent reasons to decline as written.

### 10. `1692f9e50` — ggml : recurrent state rollback for ggml_ssm_scan (#26623)
**Files:** ggml/include/ggml.h, ggml/src/ggml-{cpu,cuda,et,metal,vulkan,webgpu}/*, ggml/src/ggml-sycl/ssm_scan.cpp, ggml/src/ggml.c, src/llama-arch.cpp, src/llama-context.cpp, src/llama-model-loader.cpp, src/models/{mamba-base,plamo2}.cpp, tests/*
**Class:** port-candidate
**Why:** Core-ggml commit adding a trailing `int64_t K` parameter to `ggml_ssm_scan()` (Nemotron-style recurrent-state rollback, K rollback slots instead of always 1), implemented for CPU/CUDA/Metal/Vulkan/WebGPU/et and required, mechanically, for every backend once the signature lands — the fork's `ggml_sycl_op_ssm_scan` (`ssm_scan.cpp`) does not yet read `K` at all (confirmed: no `K`/`ggml_get_op_params_i32(dst, 0)` in the fork's file; matches the pre-patch signature verbatim). Unlike entry 2, this is not out-of-scope: the `ggml.h` signature change is a core-ggml-wave (Task 10) contract change that lands with the merge regardless of this audit's SYCL keep-ours decision, so the SYCL side must apply the matching `K`/snapshot-write hunk in `ssm_scan_f32_group`/`ssm_scan_f32_sycl`/`ggml_sycl_op_ssm_scan` at the same time — a mandatory correctness companion to wave 2, not a deferred nice-to-have.
**Landing zone:** `ggml/src/ggml-sycl/ssm_scan.cpp` (`ssm_scan_f32_group`, `ssm_scan_f32_sycl`, `ggml_sycl_op_ssm_scan`) — must land together with the core `ggml_ssm_scan()` signature change from the merge's core-ggml wave (see `docs/merge/briefs/core-ggml.md`), not deferred to the post-merge Phase C epic.
**Ownership surface:** none — writes the rollback snapshot into the pre-existing `dst` tensor buffer at a computed offset; no allocation.

### 11. `37a215c9e` — [SYCL] support OP OPT_STEP_ADAMW, OPT_STEP_SGD (#25268)
**Files:** docs/ops.md, docs/ops/SYCL.csv, examples/sycl/update-ops-doc.sh, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/opt-step.cpp (new), ggml/src/ggml-sycl/opt-step.hpp (new)
**Class:** port-candidate
**Why:** Adds `ggml_sycl_opt_step_adamw`/`ggml_sycl_opt_step_sgd` — training-time optimizer-step kernels (in-place parameter update from gradient + optimizer state) wired into `ggml_sycl_compute_forward` and `do_ggml_backend_sycl_device_supports_op`. Confirmed absent from the fork entirely (`opt-step.{cpp,hpp}` don't exist; no `OPT_STEP_ADAMW`/`OPT_STEP_SGD` case anywhere in `ggml-sycl.cpp`). This is a genuine capability gap (SYCL-backend fine-tuning/training support) untouched by any rewritten subsystem — small, self-contained, additive.
**Landing zone:** new `ggml/src/ggml-sycl/opt-step.{cpp,hpp}` + the `GGML_OP_OPT_STEP_ADAMW`/`GGML_OP_OPT_STEP_SGD` cases in `ggml_sycl_compute_forward`'s op switch and `do_ggml_backend_sycl_device_supports_op` in `ggml-sycl.cpp`.
**Ownership surface:** none — in-place update of the existing parameter/gradient/optimizer-state tensor buffers; no allocation.

### 12. `f275595dd` — sycl: fix thread/block count in quantized cpy kernel launches (#27160)
**Files:** ggml/src/ggml-sycl/cpy.cpp
**Class:** port-candidate
**Why:** The block-quantized `cpy` launchers (`ggml_cpy_{f32_q8_0,q8_0_f32,f32_q4_0,q4_0_f32,f32_q4_1,q4_1_f32,f32_q5_0,...}_sycl` and friends) launched one work-item per quant *block* (`num_blocks = ne / QK`, work-group size 1) — massively undersubscribing the device. This widens every launch to `ceil_div(ne/QK, SYCL_CPY_BLOCK_SIZE)` work-groups of `SYCL_CPY_BLOCK_SIZE` threads, matching the pattern already used by the file's own non-quantized f16/f32 cpy launchers. Measured 20.21 → 158.19 GB/s (7.8x) on the q4_0→f32 path on an Arc Pro B70 — this fork's own hardware. Confirmed live: the fork's `ggml_cpy_q4_0_f32_sycl` (and siblings) still has `const int num_blocks = ne;` with a `sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)` single-lane launch — the unfixed pre-patch pattern — while the file's f16/f32 (non-block-quantized) cpy launchers already use the wide `SYCL_CPY_BLOCK_SIZE` pattern this commit brings to the quantized side. Pure launch-geometry fix, same category as Part A entry 20 (concat) and this ledger's entry 1 (ssm_conv).
**Landing zone:** `ggml/src/ggml-sycl/cpy.cpp`, every block-quantized `ggml_cpy_*_sycl` launcher (`f32_q8_0`, `q8_0_f32`, `f32_q4_0`, `q4_0_f32`, `f32_q4_1`, `q4_1_f32`, `f32_q5_0`, `q5_0_f32`, and the equivalent q5_1/q2_0 pairs).
**Ownership surface:** none — launch-geometry change only (work-group shape), same src/dst buffers.

### 13. `0882c7bc8` — sycl: honor GGML_HINT_SRC0_IS_HADAMARD (#27298)
**Files:** ggml/src/ggml-sycl/fwht.cpp (new), ggml/src/ggml-sycl/fwht.hpp (new), ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** Adds a Fast Walsh-Hadamard Transform kernel (`ggml_sycl_op_fwht`, "port of `ggml-cuda/fwht.cu`") and a narrow early-return at the very top of `ggml_sycl_mul_mat()`: when `dst->op == GGML_OP_MUL_MAT` and `ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD`, run the FWHT kernel instead of the GEMM. Measured 3.3–6x speedup vs. GEMM on the affected shapes. Confirmed the ggml-core prerequisite already exists in the fork (`GGML_HINT_SRC0_IS_HADAMARD` and `ggml_mul_mat_set_hint()` are both present in `ggml/include/ggml.h`) — unlike Part A entry 23 (DSv4), there is no missing compile-time dependency. The new `fwht.cpp` allocates nothing (`grep`'d for `malloc`/`pool_alloc` — no hits), and the bail-out sits before all of the fork's customized layout/route dispatch in `ggml_sycl_mul_mat` (confirmed present at `ggml-sycl.cpp:58456`, with a fork-added `forced_layout` parameter) rather than competing with it.
**Landing zone:** new `ggml/src/ggml-sycl/fwht.{cpp,hpp}` + the early Hadamard-hint check at the top of `ggml_sycl_mul_mat()` in `ggml-sycl.cpp` (~line 58456), before the fork's layout/route dispatch begins.
**Ownership surface:** none — reads `src1`/writes `dst` directly; no allocation.

### 14. `fe8156f78` — ggml: add ggml_rope_set_offset (+ metal support) (#27120)
**Files:** ggml/include/ggml.h, ggml/src/ggml-{cann,cpu,cuda,et,hexagon,metal,opencl,openvino,sycl,vulkan,webgpu}/*, ggml/src/ggml.c, ggml/src/ggml-sycl/ggml-sycl.cpp, tests/test-backend-ops.cpp
**Class:** n-a
**Why:** Core-ggml commit introducing `ggml_rope_set_offset()` (a `n_offs` op-param for partial-range RoPE) with CPU/Metal/CUDA support; the SYCL-side hunk in this specific commit is a two-line placeholder guard in `do_ggml_backend_sycl_device_supports_op` (`case GGML_OP_ROPE/ROPE_BACK: return op_params[15] == 0;`, commented `// FIXME: support ggml_rope_set_offset`) — SYCL is explicitly *not* implementing the feature here, only refusing it. Confirmed this exact two-line guard is **deleted verbatim** by entry 16 (`749f688fc`, two commits later, same file/region), which replaces it with the real `n_offs`-aware kernel support. Applying entry 14 and then entry 16 nets to exactly entry 16's diff alone; the intermediate placeholder guard has no independent value to port and would be immediately overwritten. Port entry 16 directly.
**Landing zone:** n/a — superseded in-sequence by entry 16; see that entry instead.

### 15. `6cc504a2e` — sycl: report zero devices instead of aborting when the host has none (#27291)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** port-candidate
**Why:** Wraps the `dpct::dev_mgr::instance().device_count()` call in `ggml_sycl_init()` in a try/catch, falling back to `info.device_count = 0` (and logging) instead of letting a SYCL exception propagate and crash — needed so SYCL-agnostic tools like `llama-quantize` don't hard-crash on a host with the SYCL runtime installed but no usable device. Confirmed the fork's `ggml_sycl_init()` (`ggml-sycl.cpp:21901`) still calls `dpct::dev_mgr::instance().device_count()` completely unguarded (`const int raw_device_count = ...;`, no try/catch) — the exact pre-patch call site, just wrapped in the fork's own added `GGML_SYCL_VISIBLE_DEVICES`/`GGML_SYCL_DEVICE` device-map parsing that runs after it. Small, low-risk robustness fix, no interaction with any rewritten subsystem.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_sycl_init()` — wrap the `raw_device_count` acquisition in try/catch.
**Ownership surface:** none — device enumeration only, no allocation.

### 16. `749f688fc` — ggml: support ggml_rope_set_offset on opencl, sycl, wgpu, hexagon (#27345)
**Files:** ggml/src/ggml-hexagon/*, ggml/src/ggml-opencl/*, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/rope.cpp, ggml/src/ggml-webgpu/*
**Class:** port-candidate
**Why:** Implements real `n_offs` support in the SYCL `rope_norm`/`rope_neox`/`rope_multi` kernels (threading a new `n_offs` parameter through the per-thread bounds check and index math so only the `[n_offs, n_offs+n_dims)` channel range is rotated) and removes entry 14's placeholder guard entirely (the `case GGML_OP_ROPE/ROPE_BACK:` falls through to the general `return true;` group below it). Confirmed the fork's `rope.cpp` (`rope_norm`/`rope_neox`/`rope_multi`) is unmodified pre-patch (`if (i0 >= n_dims)`, no `n_offs` parameter anywhere). The core `ggml_rope_set_offset()` prerequisite this depends on is a genuine cross-cutting `ggml.h` change (see entry 14 and the core-ggml brief) that lands via the merge's core-ggml wave regardless of this audit; the SYCL kernel work here is real, self-contained, and must accompany that wave the same way entry 10 does.
**Landing zone:** `ggml/src/ggml-sycl/rope.cpp` (`rope_norm`, `rope_neox`, `rope_multi` — thread the `n_offs` parameter through) + `ggml-sycl.cpp`'s `do_ggml_backend_sycl_device_supports_op` (remove entry 14's placeholder guard). Land together with the core `ggml_rope_set_offset()` wave (`docs/merge/briefs/core-ggml.md`), not deferred to Phase C.
**Ownership surface:** none — reads/writes the existing `x`/`dst` tensor buffers with adjusted index math; no allocation.

### 17. `9e96cf77f` — sycl : fix load model with mlock issue (#27250)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** superseded
**Why:** Patches `ggml_backend_sycl_host_buffer_type_get_max_size()` to return `SIZE_MAX` when `g_ggml_sycl_enable_host_pinned_mem` is false (falling back to the legacy `aligned_malloc_host` path, whose allocations aren't bounded by `get_max_mem_alloc_size()`) — a fix to the exact mechanism entry 7 introduced. Since entry 7 is superseded (the fork's host-buffer allocator is already a completely different, unified-cache-routed implementation with no `g_ggml_sycl_enable_host_pinned_mem` variable at all — confirmed live, see entry 7), the code this commit patches does not exist in the fork. The fork's own `ggml_backend_sycl_host_buffer_type_get_max_size()` (`ggml-sycl.cpp:38392`) already does zone-aware largest-contiguous-free-block accounting, a different and more thorough mechanism than the toggle this commit adds.

### 18. `6602dd338` — sycl: fix multiple warnings in compiling sycl backend (#26713)
**Files:** ggml/src/ggml-cpu/CMakeLists.txt, ggml/src/ggml-sycl/dpct/helper.hpp, ggml/src/ggml-sycl/element_wise.cpp, ggml/src/ggml-sycl/fattn-mkl.cpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/im2col.cpp, ggml/src/ggml-sycl/norm.cpp, ggml/src/ggml-sycl/set_rows.cpp
**Class:** port-candidate
**Why:** A grab-bag of compiler-warning fixes with uneven applicability, checked hunk-by-hunk against the fork: (a) `ggml-sycl.cpp`'s `%lu` → `%zu` format-string fixes for `size_t` args in the device/host-alloc error paths — **confirmed live and unfixed** at 4+ call sites (`ggml-sycl.cpp:34103,35949,38634,39505`); applies cleanly. (b) `ggml/src/ggml-cpu/CMakeLists.txt`'s `IntelLLVM` compiler-ID check widened to cover both C and CXX (`CMAKE_C_COMPILER_ID` as well as `CMAKE_CXX_COMPILER_ID`) plus a Windows `/clang:` flag-spelling fix — **confirmed the fork still has the narrower CXX-only check** (`ggml-cpu/CMakeLists.txt:719`); applies cleanly. (c) The `element_wise.cpp` (`acc_f32` unused-`ne3`), `norm.cpp` (unused `nrows`/`nchannels`), and `set_rows.cpp` (`[[intel::reqd_sub_group_size]]` → `[[sycl::reqd_sub_group_size]]`) hunks all target code the fork has already rewritten past — none of the targeted lines/signatures exist in the fork's current versions of those files (confirmed via grep: no match for any of the three patterns). (d) `fattn-mkl.cpp` doesn't exist in the fork at all (Part A entry 17: the oneMKL GEMM flash-attention path was declined as duplicating the fork's own oneDNN SDPA path) — that hunk is moot. Net: a real but narrow port limited to (a) and (b); (c) and (d) should be dropped rather than mechanically reapplied.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (the `%lu`→`%zu` format-string sites in the device/host buffer-alloc error paths) + `ggml/src/ggml-cpu/CMakeLists.txt` (the `IntelLLVM` C/CXX compiler-ID check). Skip the `element_wise.cpp`/`norm.cpp`/`set_rows.cpp`/`fattn-mkl.cpp` hunks — not applicable to the fork's current code.
**Ownership surface:** none — cosmetic/warning fixes only; the format-string sites do not change what or how much is allocated.

### 19. `1cb3f5eb4` — sycl: Update gate logic for Alchemist GPUs regarding OneDNN features. (#26635)
**Files:** ggml/src/ggml-sycl/fattn-onednn.cpp
**Class:** n-a
**Why:** Loosens upstream's blanket `arch != bmg_g21 && arch != bmg_g31 → return false` oneDNN-SDPA gate (which restricted the fused-SDPA path to Battlemage only, citing an oneDNN bug — issue #5510 — that returns wrong results for some shapes, e.g. head_dim=64, on Alchemist/Arc-A `dg2_g10/g11/g12`) into a narrower per-shape gate: block only `K->ne[0] == 64` on Alchemist, allow everything else on every architecture. Confirmed the fork's `fattn-onednn.cpp` has **no architecture-based gate at all** — no `gpu_arch::`, `bmg_g21`/`bmg_g31`/`dg2_g1*`, and no `ggml_sycl_flash_attn_ext_onednn_supported` symbol anywhere in the file (consistent with Part A entry 4: the fork's oneDNN-SDPA path is an independently-rewritten superset, `MATERIALIZE_REQUIRED` planning + `sdpa_partition_cache`, past the point this whole gating mechanism lived at). This fork's hardware is exclusively Battlemage (B70 = `bmg_g31`, B50 = `bmg_g21`) plus an Arrow Lake-S iGPU (Xe-LPG, not Alchemist) — it never runs the `dg2_g10/g11/g12` architectures this commit's gate even discriminates on, so porting it would have zero observable effect even setting aside the missing gate mechanism.

### 20. `9e89a196b` — sycl : Add Q5_K ESIMD kernel (#26376)
**Files:** ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp
**Class:** n-a
**Why:** Adds a fourth DMMV ESIMD kernel type (Q5_K) on top of entry 4's `esimd.hpp`/`g_ggml_sycl_enable_esimd` infrastructure, defaulted ON the same way. Same disposition as entry 4: this is the exact small-block ESIMD dequant direction CLAUDE.md documents as measured 1.9x slower on this fork's Battlemage hardware, with an existing opt-in-only, default-OFF retest hatch (`GGML_SYCL_ESIMD_DEQUANT`) the fork chose instead. Confirmed `esimd.hpp` and `g_ggml_sycl_enable_esimd` are absent from the fork (see entry 4).

### 21. `ff14356e0` — sycl : add Q2_K reordered MMVQ and ESIMD kernels (#26336)
**Files:** ggml/src/ggml-sycl/convert.cpp, ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/quants.hpp, ggml/src/ggml-sycl/vecdotq.hpp
**Class:** superseded
**Why:** Two bundled changes for Q2_K: (a) a reordered-SOA MMVQ kernel (non-ESIMD), and (b) a DMMV ESIMD kernel on entry 4's infrastructure. (a) extends the same reorder-dispatch mechanism Part A entry 1 already classified superseded ("the fork's `ggml_sycl_supports_reorder_dmmv()` returns `true` only for `GGML_TYPE_Q4_0`... already dropped by the fork's own rewrite of the reorder/SOA-layout system... deliberately narrowed") — Q2_K reordered-MMVQ support is the same kind of extension to a mechanism the fork intentionally scoped down to Q4_0. (b) is the entry-4 ESIMD-dequant direction, already declined. Note: this exact commit was reverted upstream one commit later (entry 22) due to a bug and re-landed with a fix two commits after that (entry 23, "add gate params") — see that entry.

### 22. `7a0e42fd0` — Revert "sycl : add Q2_K reordered MMVQ and ESIMD kernels (#26336)" (#27486)
**Files:** ggml/src/ggml-sycl/convert.cpp, ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/quants.hpp, ggml/src/ggml-sycl/vecdotq.hpp
**Class:** n-a
**Why:** A pure `git revert` of entry 21, upstream's own response to a bug in that commit (fixed and re-landed as entry 23). It has no independent content to classify — it is byte-for-byte the inverse of entry 21's diff. Since entry 21 is itself superseded for the fork (reorder mechanism deliberately narrowed to Q4_0; ESIMD direction declined), this revert has no bearing on the fork either way — there is nothing here to port or decline that entries 21/23 don't already cover.

### 23. `867229003` — sycl : add Q2_K reordered MMVQ and ESIMD kernels (again) (#27490)
**Files:** ggml/src/ggml-sycl/convert.cpp, ggml/src/ggml-sycl/dequantize.hpp, ggml/src/ggml-sycl/dmmv.cpp, ggml/src/ggml-sycl/esimd.hpp, ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/quants.hpp, ggml/src/ggml-sycl/vecdotq.hpp
**Class:** superseded
**Why:** Re-lands entry 21 (undoing entry 22's revert) with one real fix on top ("add gate params") — diffed against entry 21, the only substantive delta is in `mul_mat_vec_q2_K_q8_1_sycl_switch_ncols`, which now threads a `vgate`/`glu_op` parameter pair through the reordered-MMVQ kernel launch to match the fused-GLU MMVQ ABI entry 9 (`650913862`) introduced in between. That dependency is itself on a superseded mechanism (entry 9's `ggml_sycl_can_fuse`-based fusion, absent from the fork). Disposition is otherwise identical to entry 21: the reordered-MMVQ half extends a reorder mechanism the fork deliberately narrowed to Q4_0, and the ESIMD half is the declined small-block-dequant-on-ESIMD direction.

### 24. `814d84bc9` — sycl : mark tq2_0 as not supported (#27660)
**Files:** ggml/src/ggml-sycl/ggml-sycl.cpp, ggml/src/ggml-sycl/mmvq.cpp, ggml/src/ggml-sycl/set_rows.cpp
**Class:** port-candidate
**Why:** `do_ggml_backend_sycl_device_supports_op` currently claims support for `GGML_TYPE_TQ2_0` in the `MUL_MAT`/`MUL_MAT_ID`, `SET_ROWS`, and `CPY` op-support checks even though `mmvq.cpp`/`set_rows.cpp` have no kernel arm for it — meaning the op is scheduled to SYCL and then hits a `GGML_ABORT` (or worse, silently mis-dispatches) instead of falling back to CPU. This commit adds explicit `TQ2_0 → return false` refusals at all three sites and improves the two `GGML_ABORT` messages to name the offending type(s), matching the "a refusal can be reported as wrong numbers" concern (memory: `a-refusal-can-be-reported-as-wrong-numbers`) in the safe direction — it adds a correct refusal where none existed, not a new refusal over a previously-working case. Confirmed live: `TQ2_0` does not appear anywhere in the fork's `ggml-sycl.cpp`, `mmvq.cpp`, or `set_rows.cpp` — the fork has the identical gap.
**Landing zone:** `ggml/src/ggml-sycl/ggml-sycl.cpp`, `do_ggml_backend_sycl_device_supports_op` (the `MUL_MAT`/`MUL_MAT_ID` op-support block, the `SET_ROWS` case, and the `CPY` type-pair check) + the `GGML_ABORT` message improvements in `mmvq.cpp`/`set_rows.cpp`.
**Ownership surface:** none — supports-op refusal logic and diagnostic strings only; no allocation.
