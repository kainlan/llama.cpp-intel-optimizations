# ggml-cpu Buffer-Accessor Cross-Backend Audit

**Date:** 2026-04-17 · **Task:** `llama.cpp-rs0oy` (read-only audit) · **Scope:** `ggml/src/ggml-cpu/**`
Driven by `llama.cpp-4oi3i` ADD_ID crash (Phase 6+). Parallel to `llama.cpp-tlcjr` MXFP4 parity (in flight).

## 1. Current cross-backend staging model

Tensor assignment + split boundary mechanics in `ggml/src/ggml-backend.cpp`:

| Step | Function | Key logic |
|------|----------|-----------|
| Assign backend to op | `ggml_backend_sched_backend_id_from_cur` (:904-963) | Pre-allocated output → owner's backend (`1.dst`); graph INPUT flag → last backend = CPU (`1.inp`); else inherit `WEIGHTS` src's backend (`1.wgt%d`). |
| Propagate assignment | `ggml_backend_sched_split_graph` expand passes (:1108-1222) | GPU assignments expand up+down; CPU is lowest-prio (the fallback). TP pass re-homes non-sharded ops to primary TP backend (:1183-1223). |
| Split-boundary copy gate | `ggml_backend_sched_split_graph` per-src loop (:1435-1468) | `if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id))` → allocate `input_cpy` via `ggml_dup_tensor_layout`, push to `split->inputs[]`, rewire `node->src[j] = tensor_id_copy(...)`. |
| Buffer-type support check | `ggml_backend_sched_buffer_supported` (:1016-1035) | Returns `ggml_backend_supports_buft(backends[backend_id], t->buffer->buft)`. |
| CPU backend's view of supported buffer types | `ggml_backend_cpu_device_supports_buft` at `ggml-cpu/ggml-cpu.cpp:462-465` | Accepts any `ggml_backend_buft_is_host(buft)` OR any registered extra buffer type. |
| Staged-copy execution | `ggml_backend_sched_compute_splits` (:1693-1964) | For each input of each split, runs `ggml_backend_tensor_copy(input, input_cpy)` OR (MoE fast-path) per-expert `tensor_set_async` slices. |
| MoE expert-weight skip | same function (:1441-1446) | `GGML_OP_MUL_MAT_ID && j==0 && is_host(buffer) && usage==WEIGHTS` → **no copy**; GPU handles via per-expert cache. (Comment at :1443 explicitly disables the analogous ADD_ID skip — "ADD_ID weights are small and not worth splitting".) |

**Where staging fails for ADD_ID src2 (ids):** The scheduler *does* insert `input_cpy` entries when `src_backend_id != cur_backend_id`. However, three paths let a non-CPU-resident tensor reach the CPU op without staging:

1. **Same-backend assignment.** When `hv_tensor_backend_ids[src]` already equals `cur_backend_id` (CPU), the `!=` guard at :1435 never fires. The assign passes (:1066-1222) can legitimately put src2 on CPU because the ids tensor is a view-chain off MUL_MAT_ID's routing output whose `view_src` buffer is sometimes CPU-assigned.
2. **`sched_buffer_supported` says yes.** If src lives in `ggml_backend_sycl_host_buffer_type()` (SYCL USM host, `.is_host = cpu_buffer.is_host` → true per `ggml-sycl.cpp:17837`), CPU's `supports_buft` returns true and the copy is suppressed. The CPU op then dereferences a SYCL USM-host pointer. This is correct *if* the pointer is CPU-addressable, but MXFP4/MoE staging produces device-only USM that is mis-classified under planner-inactive conditions (see MXFP4 audit §2 Gate 30029).
3. **Pre-allocated, not yet populated.** For compute-graph intermediates, `node->buffer` is a SYCL compute buffer (device USM). `supports_buft` fails, copy IS staged. This path works. The ADD_ID crash is NOT this path — it's path (1) or (2) for src2 that was produced by an upstream SYCL op whose output left a SYCL-device pointer in `->data`.

## 2. Enumeration of direct `->data` reads in ggml-cpu op functions

Grep summary: 427 occurrences of `->data` in `ggml/src/ggml-cpu/ops.cpp`, 47 in `ggml-cpu.c`, 14 in `unary-ops.cpp`/`binary-ops.cpp`. Classification focuses on the "indexing/routing" pattern that triggered the ADD_ID crash, plus representative float-data reads.

Legend: **src reads** = `src0->data` / `src1->data` / `src2->data` / `src3->data` reads. **idx-typed** = dereferenced as `int32_t *` / `int64_t *` (pointer to integer indices). **dst writes** = `dst->data` writes (lowest risk — dst is always scheduler-assigned and allocated in `cur_backend_id`). `op_params` reads are NOT listed — those are always CPU-resident metadata, zero cross-backend risk.

### 2a. HIGH-risk: integer-index dereferences (the ADD_ID pattern)

| Op | File:line | Src | Pattern | Already crashed? |
|---|---|---|---|---|
| ADD_ID | `ops.cpp:734` (diag), `:777` (hot) in `ggml_compute_forward_add_id_f32` | src2 | `*(int32_t *) ((char *) src2->data + …)` | **YES** — `GGML_ASSERT(i11 >= 0 && i11 < ne11)` fires (ref: `mxfp4-parity-t5-wedge-resolution.md` §3 Step 1). |
| GET_ROWS (quantized) | `ops.cpp:4661` in `ggml_compute_forward_get_rows_q` | src1 | identical `int32_t *` pattern on index tensor | Latent. MUL_MAT_ID routing calls GET_ROWS on expert outputs in some paths. |
| GET_ROWS (f16) | `ops.cpp:4702` in `ggml_compute_forward_get_rows_f16` | src1 | identical | Latent. |
| GET_ROWS (bf16) | `ops.cpp:4743` in `ggml_compute_forward_get_rows_bf16` | src1 | identical | Latent. |
| GET_ROWS (f32) | `ops.cpp:4784` in `ggml_compute_forward_get_rows_f32` | src1 | identical | Latent. |
| GET_ROWS_BACK (f16) | `ops.cpp:4967` in `ggml_compute_forward_get_rows_back_f32_f16` | src1 | `((int32_t *) src1->data)[i]` | Latent (training-only op). |
| GET_ROWS_BACK (f32) | `ops.cpp:5000` in `ggml_compute_forward_get_rows_back_f32` | src1 | identical | Latent (training-only op). |
| ROPE | `ops.cpp:5778` in `ggml_compute_forward_rope_flt` | src1 | `const int32_t * pos = (const int32_t *) src1->data;` | Latent but well-exercised on CPU offload; pos tensor is usually INPUT-flagged → scheduler guarantees staging, but relies on graph structure. |
| COUNT_EQUAL | `ops.cpp:1684-1685` in `ggml_compute_forward_count_equal_i32` | src0, src1 | `*((const int32_t *) (data0 + i00*nb00))`; same for data1 | Latent. Only used in test harnesses today. |
| SET (i32) | `ops.cpp:4555-4556` in `ggml_compute_forward_set_i32` | src1, dst | `int32_t *` cast for copy | Latent. |
| ARGMAX | `ops.cpp:1618` (`dst_`) in `ggml_compute_forward_argmax_f32` | dst | writes i32 to dst only | Low — dst always scheduler-owned. |

**Pattern commonality:** all HIGH items are *reads* from a tensor whose semantic type is `GGML_TYPE_I32` and therefore is usually a routing/indexing product of an upstream op (MUL_MAT_ID router, rope pos encode, MoE ids). Exactly the tensors most likely to be produced on GPU in SYCL MoE pipelines.

### 2b. MEDIUM-risk: float-data reads in core compute ops

All use `src*->data` / `dst->data` and are called across every inference step. Crash only if the src lives in true device-only USM without staging (which `sched_buffer_supported` generally prevents), but a layout/pointer-validity bug at the SYCL backend boundary (like MXFP4 Gate 30029) also reaches these sites.

| Op | Representative symbol | Notes |
|----|----|------|
| MUL_MAT | `ggml_compute_forward_mul_mat` (`ggml-cpu.c:1226-1804`, 60+ `->data` reads) | `src0`, `src1` dereferenced as `const char *` for `vec_dot` / `llamafile_sgemm`. Host-resident weights path OK; device-only USM crash possible if supports_buft mis-reports. |
| MUL_MAT_ID | `ggml_compute_forward_mul_mat_id` (same file) | src0=experts (host/arena), src1=activations, src2=ids (i32). Combines HIGH (ids) + MEDIUM (activations) risk in one op. |
| ADD / ADD1 / ACC | `ggml_compute_forward_add_q_f32`, `_add`, `_add1_*`, `_acc_*` (`ops.cpp:340-1220`) | Elementwise `float *` / quantized `char *` reads of src0, src1, dst. |
| SUM / SUM_ROWS / CUMSUM / MEAN | `ops.cpp:1260-1590` | src0 float reads. |
| DUP / CPY / CONT | `ops.cpp:40-700` (`ggml_compute_forward_dup_*`) | src + dst `char *` reads. The quantize/dequantize variants also call `ggml_get_type_traits(type)->to_float`. |
| REPEAT / REPEAT_BACK / CONCAT | `ops.cpp:1000-1188` | src `char *` reads. |
| NORM / RMS_NORM / L2_NORM / GROUP_NORM | `ops.cpp:3360-3854` | src0 float reads. |
| SCALE / SOFT_MAX / DIAG / DIAG_MASK / CLAMP | `ops.cpp:4200-5570` | src0 float reads, dst writes. |
| SILU / GELU / RELU / UNARY / GLU family | `ops.cpp:2000-3360` + `unary-ops.cpp` | Elementwise src/dst. |
| OUT_PROD | `ops.cpp:3900-4150` | src0, src1 float reads. |
| CPY / SET_ROWS | `ops.cpp:4400-4920` | dst + src reads; SET_ROWS also reads an i32 src1 (HIGH-risk overlap). |
| CONV / IM2COL / POOL / UPSCALE / PAD / ROLL | `ops.cpp:5850-6800` | src0 float reads, dst writes. |
| SSM_CONV / SSM_SCAN | `ops.cpp:7110-7400` | src0/1/2/3 float reads. |
| RWKV_WKV6 / WKV7 / GLA | `ops.cpp:7500-7950` | src0..src6 float reads. |
| FLASH_ATTN_EXT / BACK | `ops.cpp:6900-7100` | src0 (Q), src1 (K), src2 (V), src3 (mask) — MEDIUM-HIGH: mask/bias tensors can be GPU-produced in the SYCL fused path. |
| CROSS_ENTROPY_LOSS / OPT_STEP_* | `ops.cpp:8050-8400` | Training-only; low cross-backend risk today. |

### 2c. LOW-risk: `op_params` reads and inert constants

All `((int32_t *) dst->op_params)[…]` sites (`ops.cpp:1198-1202`, `:4430-4505`, `:5121`, `:5709-5721`, `:6098-6275`, `:6367-6370`, etc. — ~70 sites) read scheduler-managed metadata that lives in the tensor struct itself, never in a device buffer. Zero cross-backend risk. Excluded from refactor.

### 2d. Non-ops.cpp sites

| File | Symbol | Reads | Notes |
|---|---|---|---|
| `ggml-cpu.c:740-970` | `ggml_get_*` / `ggml_set_*` public helpers | `tensor->data` as typed scalar | Called by unit tests, never by the scheduler hot path. |
| `ggml-cpu.c:1226-1804` | `ggml_compute_forward_mul_mat` and wrappers | Hot path, MEDIUM-risk. |
| `binary-ops.cpp` | Elementwise dispatchers | 3 sites wrapping `src0/1/dst ->data`. MEDIUM-risk in-line with ops.cpp ADD/MUL. |
| `unary-ops.cpp` | Elementwise unary dispatchers | 4 sites. MEDIUM. |
| `amx/mmq.cpp`, `amx/amx.cpp`, `repack.cpp`, `kleidiai/kleidiai.cpp`, `spacemit/ime.cpp`, `quants.c` | Arch-specific GEMM/quant kernels | Only called after ggml-cpu has already chosen "weights on host" path; cross-backend risk inherited from ops.cpp callers. |

## 3. Risk classification summary

| Tier | Count (approx.) | Examples | Cross-backend risk |
|------|-----------------|----------|--------------------|
| HIGH | 11 direct sites across 7 ops | ADD_ID, GET_ROWS×4, GET_ROWS_BACK×2, ROPE pos, COUNT_EQUAL, SET(i32) | Already crashed OR reads integer index tensor routinely produced on GPU. Scheduler's `supports_buft`-based gate does not guarantee these will be staged — depends on where the i32-producing op ran. |
| MEDIUM | ~40 ops × 1-6 sites each (~150 sites) | MUL_MAT, MUL_MAT_ID, ADD, MUL, NORM, SCALE, SOFT_MAX, UNARY, GLU, ROPE (non-pos), FLASH_ATTN, COPY/CONT/DUP, SET_ROWS, CONV, IM2COL | Float data; depends on accurate `supports_buft`. MXFP4 parity epic §2 shows this is where SYCL's host/device mis-classification leaks in. |
| LOW | ~70 `op_params` + dst-only writes | Stride params, fill writes | Irrelevant — struct-resident or scheduler-owned. |

## 4. Minimal immediate fix scope (for the ADD_ID case)

The SYCL-side option Z (self-stage src2 before returning to CPU) is the correct pragmatic workaround and should land as planned. The ggml-cpu-side architectural complement has three candidate shapes, ordered by surface area:

| Option | Change | Surface area | Blast radius | Perf impact |
|--------|--------|--------------|--------------|-------------|
| (a) Per-op accessor wrapper | Replace `src2->data` reads in HIGH-risk ops (§2a, 11 sites) with `ggml_backend_tensor_get`-style staging helper that no-ops when `buf->iface.is_host(buft)` | ~30 lines across ops.cpp + 1 new helper in a new header | Local to ggml-cpu/ops.cpp; no scheduler or SYCL changes. | Zero for host-resident (no-op); one memcpy per CPU op when src2 is non-host. |
| (b) Scheduler-side conservative stage | In `ggml_backend_sched_split_graph` (`ggml-backend.cpp:1435-1468`), drop the `!ggml_backend_sched_buffer_supported` guard for ops in a known-indexing-sensitive set (ADD_ID, GET_ROWS, ROPE, SET_ROWS, COUNT_EQUAL) and always stage their i32 src. | ~15 lines in scheduler + a 1-op static allow-list | All backends benefit; CPU oneDNN path sees extra copy when src IS actually host-accessible. | One per-token memcpy per covered op where src was previously "zero-copy host". |
| (c) Backend-buffer accessor everywhere | Introduce `ggml_tensor_data_cpu(const ggml_tensor *)` that returns a CPU-readable pointer, staging via a per-thread scratch if needed. Replace all HIGH-risk `->data` reads. | ~200 lines touching all ops in §2a + new helper | Large; requires a thread-local scratch arena to avoid mallocs in hot path. | Measurable if scratch is ever warm; null-cost when the buffer type's `is_host` is true. |

**Recommendation for ggml-cpu-side fix:** **(a) — per-op accessor wrapper at the HIGH-risk 11 sites.** It fully resolves ADD_ID's class of bugs (index-tensor reads producing garbage/faults) while keeping blast radius local and avoiding scheduler policy churn. Leaves MEDIUM-risk float-data reads alone because they are well-covered by `sched_buffer_supported` in practice (path 3 in §1). If a future MEDIUM crash surfaces (e.g., SOFT_MAX bias or FLASH_ATTN mask from GPU), promote that op to HIGH and add it to the accessor wrapper — same pattern.

Option (b) would be strictly safer but introduces unnecessary per-token copies on correct, host-resident paths (CPU-backend Mistral 7B benchmarks), so it is worse than (a) at steady state.

Option (c) is the correct long-term architectural fix but is an epic, not a quick-win — see §5.

## 5. Proper refactor scope

A full "all ggml-cpu ops are backend-safe" refactor would introduce a thin accessor layer mirroring `ggml_backend_tensor_get_async` but synchronous and zero-copy for host buffers.

| Dimension | Estimate |
|-----------|----------|
| Sites to change | ~160 `src*->data` reads across 40+ op functions (§2a + §2b) |
| New API surface | 1-2 inline helpers: `cpu_tensor_data(const ggml_tensor *)` returning a guaranteed-CPU-readable `const char *`; optional `cpu_tensor_scratch_t` RAII for staged copies |
| Thread-local scratch | ~one 2 MB pinned arena per OMP worker to hold staged copies during an op (sized from max element-count × sizeof(float)); frees after the op |
| Perf concern | In the common case (host-resident), the helper is an `inline` pointer return with a dead-code `if (buf->iface.is_host)` check — zero cost. In the staged case, adds one memcpy per op per thread per step. For TG workloads (~200 ops/token) the worst-case overhead is bounded by (n_ops × n_threads × avg_tensor_bytes) and is almost always <1% because staged tensors are small (index, mask, bias). |
| Existing precedent | `ggml_backend_tensor_get` (for out-of-graph reads), `ggml_get_f32_1d` / `ggml_set_f32_1d` (public helpers at `ggml-cpu.c:863-938`) already wrap `tensor->data` with type-aware accessors. The precedent for device-staging is `ggml_backend_sched_compute_splits` itself, which stages via `tensor_copy`. The refactor packages that pattern in a compute-loop-friendly form. |

**Phases (scoping, not a binding plan):**

| Phase | Work | Lines | Test surface |
|-------|------|-------|--------------|
| P1 | Add `cpu_tensor_data()` helper + `cpu_tensor_stage_scratch` RAII; unit tests against synthetic non-host buffers | ~80 | new test file |
| P2 | Migrate §2a HIGH-risk sites (11 sites / 7 ops) | ~40 line diffs | existing test-backend-ops |
| P3 | Migrate §2b MEDIUM-risk indexing/mask/bias sub-category (FLASH_ATTN mask, MUL_MAT_ID ids, ROPE freq, SET_ROWS idx, CONCAT axis-op-params-backed, etc. — ~25 sites) | ~60 | existing tests |
| P4 | Migrate the bulk MEDIUM float-data sites (~120 sites) — optional, only if a crash is seen in this tier | ~200 | full bench sweep |
| P5 | CI check: grep-level ban on new raw `src*->data` reads in ggml-cpu (mirror of `scripts/check-sycl-tensor-data-usage.sh` from `llama.cpp-wjvse` T9) | ~30 | new CI job |

**Total estimate:** P1+P2 = ~120 lines, ~1 week for one engineer. Adding P3 = ~180 lines, ~2 weeks. Full P1-P5 = ~350 lines, ~1 month.

## 6. Cross-reference with existing beads

| Bead | Overlap | Verdict |
|------|---------|---------|
| `llama.cpp-wjvse` ([EPIC] Safe tensor access — eliminate raw ->data in SYCL dispatch) | Same *pattern*, different *backend*. wjvse wraps `ggml_tensor *` in a `sycl_tensor` class that forbids `->data` in SYCL op code, routing reads through the unified cache. Scope: SYCL backend only. | **Separate epic.** wjvse covers SYCL; this audit's follow-up would cover ggml-cpu. The wjvse T9 `scripts/check-sycl-tensor-data-usage.sh` CI pattern can be forked for ggml-cpu in P5 above. |
| `llama.cpp-4oi3i` ([BUG] GPT-OSS 20B MoE ADD_ID bias staging crash/hang on B580) | Direct parent. The ADD_ID crash cited in §2a and the T5 wedge-resolution doc is this bug. | **Unblocks:** this audit's P2 closes the ADD_ID case at ggml-cpu level, letting a future cleanup remove option Z's SYCL-side self-staging once both landed. |
| `llama.cpp-tlcjr` ([EPIC] MXFP4 compute parity — dense + MoE, GPU + CPU paths) | Absorbed 4oi3i (per plan §T5 scope amendment). Option Z (SYCL self-staging for ADD_ID) is in flight as a tlcjr deliverable. | **Complementary.** tlcjr provides the pragmatic workaround; this audit's P1+P2 is the architectural complement. No overlap in touched files (tlcjr = SYCL, this = ggml-cpu). |
| `llama.cpp-4ewji` ([FIX] MoE expert dispatch: eliminate raw tensor->data fallback) | SYCL-side expert routing fix (closed dependency of 4oi3i). | No overlap. Different layer. |
| MXFP4 audit (`docs/plans/2026-04-17-mxfp4-parity-audit.md`) | Same project, different scope: MXFP4 audit enumerated *SYCL dispatch gates* admitting host-resident MXFP4 to GPU kernels. This audit enumerates *CPU compute ops* that cannot tolerate non-CPU src pointers. | Complementary; together they describe both sides of the boundary. |

**Absorb/extend/separate verdict:** **Separate epic.** The refactor proposed here is a new epic targeting `ggml/src/ggml-cpu/**`. It should reference `llama.cpp-wjvse` as architectural prior art (same wrapper pattern, other backend) and `llama.cpp-4oi3i` as the motivating bug. Attempting to absorb it into wjvse would mis-scope wjvse's CI invariants (wjvse's grep guard excludes ggml-cpu's infrastructure files by design).

## 7. Recommended next steps

### Quick-win (≤50 lines, candidate for tlcjr inlining)

The ADD_ID case is closable at the ggml-cpu level with the single-op variant of Option (a). Concretely:

- Add a static helper at the top of `ggml-cpu/ops.cpp` (or in a new internal header) that given a `const ggml_tensor *` returns a CPU-readable pointer:
  - If `t->buffer->iface.is_host(t->buffer->buft)` → return `t->data` (zero-copy, 99% of calls).
  - Else → stage bytes via a thread-local scratch (sized `ggml_nbytes(t)`) using `ggml_backend_tensor_get(t, scratch, 0, ggml_nbytes(t))`, return scratch.
- Replace `src2->data` reads in `ggml_compute_forward_add_id_f32` (:734, :777) with the helper.

**Estimated size:** helper = ~25 lines, ADD_ID changes = 2 lines. Ship with a `static_assert` sized scratch (ids tensors are ≤ 32 KiB in GPT-OSS-class MoE). Test by running the same reproducer from T5 Step 1 without option Z enabled.

**This is flagged to the tlcjr team** as a candidate quick-win to consider inlining — the work is trivially small (≤30 LoC), it is architecturally correct, and it lets option Z eventually be reverted. If the team prefers to defer and ship option Z first, that is also fine — option Z already shipped in flight and this audit's P2 absorbs it cleanly.

### Follow-up epic scope

**Proposed epic:** "Safe tensor access in ggml-cpu — eliminate raw ->data in CPU compute ops" (name parallels `llama.cpp-wjvse` SYCL version).

- **Team size:** 1 implementer, 1 reviewer
- **Phases:** P1 (helper + tests) → P2 (HIGH-risk 11 sites) → P3 (MEDIUM-risk indexing subset) → [optional] P4 (all MEDIUM) → P5 (CI guard)
- **Effort:** P1+P2 = 1 engineer-week, ~120 lines; P1+P2+P3 = 2 engineer-weeks, ~180 lines; full P1-P5 = 1 engineer-month, ~350 lines
- **Blocking?** No — the SYCL option Z workaround already mitigates the known crash. This epic is for structural robustness across all non-SYCL backends (Vulkan, CUDA multi-device, Metal with host-visible MTLBuffer) that could exhibit the same pattern in the future.
- **Risk flags:** Thread-local scratch lifetime needs care (must be per-OMP-worker, freed on thread exit). Mirror the `g_leaf_staging_cache` pattern from SYCL cpu-dispatch.cpp (per user memory entry `llama.cpp-fimd.2`).
- **Sequencing:** Start after tlcjr T6 closes (no file conflict; tlcjr touches SYCL, this touches ggml-cpu, but sequencing keeps the ADD_ID bug-fix ownership clean).

### Related hygiene

- Audit MEDIUM-risk site #4 (FLASH_ATTN_EXT mask/src3) at `ops.cpp:~6900` — in GPT-OSS 20B CPU offload, the attention mask can be a SYCL-produced tensor. If a crash surfaces there, promote to P2 coverage.
- Evaluate whether `ggml_backend_cpu_device_supports_buft` (`ggml-cpu/ggml-cpu.cpp:462`) should become more conservative: it currently trusts any `buft->iface.is_host`. The SYCL host buffer type legitimately sets `is_host=true` (because it's USM-host), but a future backend that mis-declares is_host would leak in undetected. Consider adding an assert in DEBUG builds that dereferences the first byte of each src to catch malformed buffers.

---

**Summary:** 11 HIGH-risk index-dereference sites (ADD_ID, 4×GET_ROWS, 2×GET_ROWS_BACK, ROPE pos, COUNT_EQUAL, SET(i32)) + ~150 MEDIUM-risk float-data reads. ADD_ID is the only one actually crashing because its src2 has been observed landing in non-staged SYCL memory under planner-inactive GPT-OSS 20B MoE. A ~30-line ggml-cpu-side helper closes ADD_ID architecturally and complements the SYCL option Z pragmatic fix. Broader refactor is tractable as a new epic (~1 engineer-month, ~350 LoC across 5 phases), separate from but patterned after `llama.cpp-wjvse`.
