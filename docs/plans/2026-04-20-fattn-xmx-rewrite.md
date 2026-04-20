# Blueprint: Ground-Up SYCL Flash Attention Rewrite

**Status**: Design blueprint — no code in this document. Ready for user review before implementation dispatch.
**Predecessor**: `/Apps/llama.cpp/docs/plans/2026-04-19-l144i-rootcause.md` §15 (supersedes the kernel-printf "determinism" conclusion — today's input-probe diagnostic is ground truth).
**Sibling**: `/Apps/llama.cpp/docs/plans/2026-04-20-fattn-alt-path-epic.md` (Alt-A/B/C/D). This blueprint is the structurally-aligned rewrite that Alt-B/D hints at — fleshed out with oneDNN evaluation and CUDA-fattn-verbatim structure.

---

## §1 — oneDNN SDPA feasibility study (the crux question)

### 1.1 What oneDNN 3.11 ships for SDPA on Xe2

Shipping version on this machine: **oneDNN 3.11.3** (via oneAPI 2025.3). Cumulative improvements relevant here:

- **Graph-API SDPA pattern** (since 3.4, stable shape since 3.8): DAG of `MatMul(Q, K, transpose_b=true)` → `Divide` (or `Multiply`) → `Add` (mask, optional) → `SoftMax(axis=-1, mode="inf_as_zero")` → `MatMul(probs, V)` → optional `Reorder`. Source: oneDNN graph SDPA guide, `examples/graph/sdpa.cpp`.
- **Optimized Intel-GPU kernel** is selected when: f16 or bf16 Q/K/V with f32 intermediate, XMX-capable device. D limit is driver-dependent — query `compiled_partition` capability at runtime; don't hardcode.
- **Xe2/Battlemage optimizations already in our 3.11**:
  - 3.8: broad production-quality SDPA optimizations
  - 3.10: **GQA 2nd-token (decode) perf improvements on Xe** — directly relevant to Mistral TG
  - 3.11: **small-head f32 SDPA improvements** — applicable to D=64/128
- **Not yet in 3.11** (would need upgrade):
  - 3.12/3.13: implicit bottom-right causal fusion
  - 3.13: compressed KV cache partitions (Q8_0 KV)
- **GQA pattern** (`examples/graph/gqa.cpp`): 5-D tensors `Q=(N, H_kv, N_rep, S, D)`, `K/V=(N, H_kv, 1, S, D)`, with `N_rep = H_q / H_kv`. **MQA is a special case** (`H_kv = 1`, `N_rep = H_q`). Requires f16.
- **Causal mask**: implicit top-left / bottom-right or explicit user-provided tensor (Add).

### 1.2 What oneDNN SDPA **cannot** do (show-stoppers for gpt-oss-20b)

This is the crux. gpt-oss-20b needs all three of these simultaneously; oneDNN supports none:

1. **Attention sinks**. gpt-oss attention has a learned per-head scalar that participates in the softmax denominator only: `softmax(x_i) = exp(x_i - m) / (sum_j exp(x_j - m) + exp(sink_h - m))`. This is not any op in the oneDNN graph SDPA DAG. Inserting a `Concat` of a virtual sink-score column before `SoftMax` would change the pattern and break fusion (oneDNN's matcher is rigid — `MatMul → Divide → Add → SoftMax → MatMul` only; extra ops partition into separate subgraphs per release-note phrasing "floating-point SDPA patterns are usually implemented with matmul (with post-ops) and softmax primitives" as fallback). Confirmed: no mention of sinks in v3.8, v3.9, v3.10, v3.11, v3.12, v3.13 release notes; no example in `examples/graph/`.
2. **Logit softcap**. gpt-oss uses `logit = softcap * tanh(x / softcap)` between Q@K^T and softmax. This requires inserting a `Tanh` op between `Divide` and `Add`. Same rigidity issue — `Tanh` is a valid `op::kind` in the general graph API but is **not in the SDPA matcher template**. It would break fusion and split into unfused matmul + tanh + softmax + matmul primitives (functional but slow; no decode-2nd-token optimization).
3. **Custom axis / flexible head layout**. llama.cpp stores KV as `[D, n_kv, n_heads]` paged or `[D, n_kv, n_heads, batch]` contiguous. oneDNN expects Q `(N, H_q, S, D)` / KV `(N, H_kv, S, D)` contiguous. Transposing KV to that layout on every layer is a separate memcpy cost; alternatively one ggml `ggml_permute` per layer per tensor. Feasible but adds overhead and a strided-load code path that the Xe2 optimized kernel may reject (strided V has been a historical limitation in several oneDNN releases).

### 1.3 Verdict — oneDNN SDPA is NOT the answer for gpt-oss-20b

**oneDNN graph SDPA cannot express gpt-oss-20b's attention**. The rigid `MatMul→Divide→Add→SoftMax→MatMul` matcher has no slot for either sinks or softcap. Extra ops force the matcher to partition the graph, in which case what the "SDPA" path runs is just unfused matmul + softmax + matmul — no online-softmax, O(S²) intermediate memory, no Xe2 decode optimization. That is **worse** than the current XMX kernel on every axis except correctness.

**Secondary verdict — oneDNN IS viable for models without sinks/softcap** (Mistral, Llama-family without Gemma-style softcap). For those models we can route through oneDNN graph SDPA and it will likely match or beat the current XMX fattn on PP (since PP on Arc B580 is already oneDNN-GEMM-dominated at ~1480 tok/s). For TG (S=1), oneDNN's 3.10 "2nd-token GQA" optimization should help — but no published numbers for Xe2/Battlemage exist; this is Phase 3 work.

### 1.4 Xe2 micro-architecture — query, don't assume

**All hardware parameters must be discovered at runtime, never hardcoded.** The rewrite queries the device and adapts. Required queries at backend init (once per device):

```cpp
const auto dev = ctx.device();
const size_t slm_bytes     = dev.get_info<info::device::local_mem_size>();
const auto   sg_sizes      = dev.get_info<info::device::sub_group_sizes>();
const size_t max_wg_size   = dev.get_info<info::device::max_work_group_size>();
const size_t max_grf_per_t = dev.get_info<ext::intel::info::device::gpu_subslices_per_slice>(); // + grf_size if available
const auto   matrix_combos = dev.get_info<ext::oneapi::experimental::info::device::matrix_combinations>();
```

`matrix_combinations` is the authoritative source for supported `joint_matrix` `(M, K, N, A_type, B_type, C_type)` tuples. Kernel tile selection iterates this list and picks the best fit for `(ncols, D)` rather than baking in `M=8 N=16 K=16`. If a future Xe3 doubles M to 16, we pick it up for free.

**Empirical Arc B580 values (for reference — NOT hardcoded in the kernel)**: typical Xe2 desktop SKUs report ~128 KB SLM, sub-group sizes {8, 16, 32}, and joint_matrix combos including `m∈{1..8,16,32} × n=16 × k=16` for `fp16×fp16→fp32`. Confirm on this machine via the canary test in §9.3.

**Key architectural insight regardless of exact numbers**: for ncols=1 (TG), matrix-tile primitives waste capacity because they're designed for GEMM, not attention decode. The VEC path (sub-group dot product) sidesteps this entirely — use DPAS only where it fits (ncols ≥ 4 PP).

## §2 — CUDA fattn architectural summary

Source files read: `ggml/src/ggml-cuda/fattn.cu`, `fattn-mma-f16.cuh`, `fattn-tile.cu`, `fattn-tile.cuh`, `fattn-vec.cuh`, `fattn.cuh`.

### 2.1 Three-tier dispatch

`ggml_cuda_get_best_fattn_kernel()` at `fattn.cu:213-355` selects one of:

| Tier | Enum | When selected | Structure |
|------|------|---------------|-----------|
| VEC  | `BEST_FATTN_KERNEL_VEC` | `ncols * gqa_eff ≤ 2` AND D ∈ {64, 128, 256} (varies per arch) | One warp per `(query, head)`; per-lane K-elem; warp-reduce for max/sum |
| TILE | `BEST_FATTN_KERNEL_TILE` | Volta/no-tensor-core, moderate ncols | `nwarps` warps, SLM tile for K/V, scalar accum with warp reductions |
| MMA  | `BEST_FATTN_KERNEL_MMA_F16` | Turing+ with tensor cores, large ncols | Full `mma::sync` tile pipeline, Q-in-reg option, cp_async staging, multi-stage |
| WMMA | `BEST_FATTN_KERNEL_WMMA_F16` | Volta, legacy WMMA | Older WMMA path |

Intel-SYCL analog: **VEC → scalar sub-group dot product**, **TILE → MMA-style SLM+scalar**, **MMA → SYCL `joint_matrix` with proper tile control flow**.

### 2.2 Shared invariants (all three paths implement)

Every CUDA fattn path follows this recipe, which the SYCL rewrite must mirror exactly:

1. **Per-query register state**: each thread/warp owns `KQ_max[cols_per_thread]`, `KQ_sum[cols_per_thread]`, `VKQ[cols_per_thread][D_per_thread]` in **registers**, never SLM. Init: `KQ_max = -FLT_MAX/2.0f`, `KQ_sum = 0`, `VKQ = 0`.
2. **Online softmax** (per KV batch): compute `batch_max = fmax` over the KV tile's QK row → `new_max = fmax(KQ_max, batch_max)` → `scale_old = expf(KQ_max - new_max)` → rescale `VKQ *= scale_old`, `KQ_sum *= scale_old` → `KQ_max = new_max` → accumulate `KQ_sum += sum(exp(QK - new_max))` and `VKQ += sum(exp(QK - new_max) * V)`.
3. **Warp reduction** for `batch_max` across lanes: `__shfl_xor_sync(0xFFFFFFFF, KQ_max_new, offset, WARP_SIZE)` — see `fattn-mma-f16.cuh:545, 606`. This is shuffle-based, no SLM.
4. **No SLM aliasing**: every SLM buffer has exactly one logical use. Q tile, K tile, V tile, KQ scratch — each a separate allocation. CUDA `__shared__` declarations sit at file scope and don't overlap. **The SYCL XMX kernel violates this** (see §3.3).
5. **FTZ trick for rescaling**: `*((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;` (`fattn-mma-f16.cuh:634`). Flushes small rescale factors to zero bit-exactly without calling `expf` below a threshold. Preserves determinism.
6. **Attention sinks applied OUTSIDE the KV loop** (`fattn-mma-f16.cuh:1027-1080`): after all KV batches processed, compute `KQ_max_new = fmax(sink, KQ_max)`, rescale VKQ and KQ_sum by `expf(KQ_max - KQ_max_new)`, add sink contribution `KQ_sum += expf(sink - KQ_max_new)`. This is **strictly after** the main loop — simpler and race-free compared to initialising KQ_max to sink at loop start (which CUDA comment suggests as alternative).
7. **Logit softcap INSIDE Q@K^T**, before mask/softmax: `dot = softcap * tanhf(dot)` immediately after the matmul accumulator is reduced, before the max computation. Must also adjust `scale /= softcap` at dispatch time (`fattn.cpp:937-940` in SYCL already does this).

### 2.3 MMA path specifics (the template for the rewrite)

`ggml/src/ggml-cuda/fattn-mma-f16.cuh` — 1500 lines, battle-tested.

- **Compile-time config struct** per `(DKQ, DV, ncols, arch)`: `nthreads`, `nwarps`, `occupancy`, `nbatch_fa` (KV tile size), `nbatch_K2` / `nbatch_V2` (inner-D tile), `nstages` (cp_async pipelining), `Q_in_reg` flag. Lines 10-205. SYCL analog: template struct `fattn_xmx_config<D, ncols>` selecting these at compile time.
- **Load-tile helpers** at `fattn-mma-f16.cuh:206-365`: `flash_attn_ext_f16_load_tile` with/without `cp_async`; `oob_check` template param for bounds on partial tiles. SYCL analog: lambda or inline functions using `sycl::group::async_work_group_copy` or plain vectorized SLM stores. No cp_async on Xe2; use two-stage manual double-buffer with `group_barrier`.
- **Process-tile function** `flash_attn_ext_f16_process_tile` at line 809: the hot loop body. Loads Q into regs (if `Q_in_reg`), iterates KV tiles, calls `flash_attn_ext_f16_iter`. Online softmax + MMA fused.
- **K in `[D, N]` transposed layout**: K is stored with stride `nb11` at element size 2 bytes (half). For `mma::sync`, the kernel treats `K` as if transposed by loading with swapped dims into `mma::B` fragments. No runtime transpose.
- **Shuffle-based warp reduction** for `KQ_max_new` and `KQ_rowsum_add`: butterfly reduction over 32-lane warp via `__shfl_xor_sync`. Lines 535, 545, 588, 606. Completely SLM-free.

### 2.4 TILE path specifics (fallback for Volta / low-ncols)

`ggml/src/ggml-cuda/fattn-tile.cuh` — scalar-tile implementation.

- `nwarps` warps share SLM `tile_K`, `tile_V`, `tile_Q`, `tile_KQ`. Each warp computes one Q row against the K tile via scalar dot products in `flash_attn_tile_iter_KQ`.
- `warp_reduce_max<np>(KQ_max_new[0])` across `np = nwarps/ncols` parallel warps per Q col — lines 585-598.
- **This is the closest CUDA analog to our current broken `fattn-mma-f16.hpp` SYCL scalar fallback**. The SYCL fallback is functionally almost identical but simpler (no multi-warp split per col, just one sub-group per Q row). The fact that the SYCL scalar fallback is correct confirms the TILE pattern works on Xe2; the bug is in the XMX (MMA-equivalent) implementation.

### 2.5 VEC path specifics (TG fast path)

`ggml/src/ggml-cuda/fattn-vec.cuh` — one warp per `(query, head)`, lane-parallel on K elements. Best for `ncols * gqa_ratio_eff ≤ 2`.

- **No SLM for K/V** (!): each lane loads K[lane] and V[lane] directly from global; warp-wide butterfly for Q@K reduction; per-lane VKQ register accumulator.
- Fits in register file because D ≤ 256 and only one Q row.
- **This should be the SYCL TG fastpath for gpt-oss-20b with S=1.** The current XMX kernel uses M=8 matrix tiles for ncols=1 which wastes 7/8 of the matrix every launch — a SYCL vec-style implementation will be faster AND correct.

## §3 — Per-path SYCL port design

### 3.1 File plan (5 new files, 1 kept, 1 deleted)

| Path | File | Status | Purpose |
|------|------|--------|---------|
| VEC  | `ggml/src/ggml-sycl/fattn-vec-f16.hpp` | **NEW** | Sub-group-dot-product decode path for `ncols * gqa_eff ≤ 2` |
| TILE | `ggml/src/ggml-sycl/fattn-tile-f16.hpp` | **NEW** | Scalar-SLM-tile path for moderate ncols, D ≤ 128, used as fallback and when oneDNN SDPA is unsupported |
| MMA  | `ggml/src/ggml-sycl/fattn-xmx-f16-v2.hpp` | **NEW** | XMX joint_matrix path, structurally mirroring `fattn-mma-f16.cuh` exactly |
| ONEDNN | `ggml/src/ggml-sycl/fattn-onednn.cpp/hpp` | **NEW** | oneDNN graph SDPA dispatch for models WITHOUT sinks AND WITHOUT softcap |
| Dispatch | `ggml/src/ggml-sycl/fattn.cpp` | **MODIFY** | Rewrite `ggml_sycl_flash_attn_ext_dispatch_ncols` to pick one of {vec, tile, xmx-v2, onednn} instead of current {esimd, xmx, mma} |
| Fallback | `ggml/src/ggml-sycl/fattn-mma-f16.hpp` | **KEEP AS-IS** | Known-correct scalar last-resort. Rename symbol to `launch_fattn_scalar_f16` to match its actual behavior. Used only when fattn-xmx-f16-v2 fails validation for a given shape, or when `GGML_SYCL_FA_DETERMINISTIC=1`. |
| Broken | `ggml/src/ggml-sycl/fattn-xmx-f16.hpp` | **DELETE** after v2 validates | The racy kernel. Keep during Phase 1-3 validation period behind `GGML_SYCL_FA_XMX_V1=1` for A/B comparison; delete in Phase 4. |
| ESIMD | `ggml/src/ggml-sycl/fattn-esimd-f16.hpp` | **KEEP** | Orthogonal ESIMD path, currently +7% for TG single-query on Mistral. Unchanged. |
| V2-partition | `ggml/src/ggml-sycl/fattn-v2-partition.hpp` | **KEEP** | Paged partitioned path. Will dispatch into v2 leaf kernels below it, those leaves get swapped to the v2 kernels from this blueprint. |

### 3.2 The VEC path — `fattn-vec-f16.hpp`

**Mirrors**: `ggml-cuda/fattn-vec.cuh`.

**When selected** (dispatch policy §4): `ncols == 1` (TG) for any D, OR `ncols == 2 && gqa_ratio % 2 == 0` with GQA opt.

**Work distribution**:
- Grid: `(n_kv_head * gqa_ratio) * (ne03) * n_q_blocks` work-groups.
- Work-group: ONE sub-group (16 lanes), sub-group size 16.
- Per sub-group: one `(query, head)` pair; lanes partition across D dimension for Q load and across K-position for the inner loop.

**Register state (per lane)**:
- `float VKQ_partial[D / SG_SIZE]` — each lane owns `D/16` elements of the output.
- `float KQ_max` — scalar per lane, reduced at end of each KV tile with `sycl::reduce_over_group(sg, KQ_max, sycl::maximum<float>{})`.
- `float KQ_sum` — scalar per lane, sub-group-reduced same way.
- Lane-local `Q_row[D / SG_SIZE]` loaded once at kernel start.

**SLM**: ZERO. No `local_accessor` needed. This is the big determinism win.

**Inner loop** (per KV position `kv`):
1. Lanes collectively load `K[kv]`: each lane `l` reads `K[kv][l * D/SG_SIZE .. (l+1) * D/SG_SIZE]`.
2. Lane-local dot: `float dot_partial = sum_d (Q_row[d] * K_strip[d])`.
3. Sub-group reduce: `float dot = sycl::reduce_over_group(sg, dot_partial, sycl::plus<float>{})`. **SYCL spec guarantees this reduction is deterministic for a given launch configuration** (implementation-defined but fixed per-kernel-compile tree).
4. Softcap (if enabled): `dot = softcap * sycl::tanh(dot)`.
5. Mask: `dot += slope * mask_val` (lane 0 loads mask, broadcasts via shuffle).
6. Online softmax update (per lane, symmetric — all lanes see the same `dot`):
   - `new_max = sycl::fmax(KQ_max, dot)`
   - `scale_old = sycl::exp(KQ_max - new_max)` (FTZ threshold applied via `as_uint32(scale_old) *= (KQ_max - new_max >= SOFTMAX_FTZ_THRESHOLD)`)
   - `VKQ_partial[d] *= scale_old` for d in lane's strip
   - `KQ_sum = KQ_sum * scale_old + sycl::exp(dot - new_max)`
   - `KQ_max = new_max`
7. Load `V[kv][lane's strip]` and accumulate: `VKQ_partial[d] += sycl::exp(dot - KQ_max) * V_strip[d]`.

**Finalization**:
- Apply sinks (if present): see §2.2 item 6. Symmetric across all lanes — no reduction needed because `sink = sinks[head]` is scalar.
- Normalize: `VKQ_partial[d] /= KQ_sum` per lane.
- Lanes cooperatively store to `dst`: lane `l` writes `dst[head, query, l * D/SG_SIZE .. (l+1) * D/SG_SIZE]`.

**Why this is race-free**: (a) Every lane's register state is private. (b) The ONLY cross-lane operations are `reduce_over_group(sg, _, _)` and `group_broadcast`, both of which SYCL defines as deterministic per-launch. (c) Zero SLM = zero opportunity for the aliasing bug that currently breaks the XMX kernel.

**Divergence from current XMX kernel that matters**: the current kernel forces M=8 joint_matrix tile even for ncols=1, wasting 7 of 8 rows. This new kernel has zero wasted work.

### 3.3 The MMA (XMX) path — `fattn-xmx-f16-v2.hpp`

**Mirrors**: `ggml-cuda/fattn-mma-f16.cuh`.

**When selected**: `ncols ≥ 4 && D ≤ 256`, NOT oneDNN-eligible (has sinks or softcap), NOT safe-decode.

**The ten rules that the current fattn-xmx-f16.hpp violates and the v2 must obey**:

| # | Rule | Current kernel violation | Line ref |
|---|------|--------------------------|----------|
| 1 | **No SLM buffer aliasing** — each SLM region has one name and one purpose | `SV_acc` reused as `batch_max_shared` (reuses float bytes for different data) | `fattn-xmx-f16.hpp:802, 957` |
| 2 | **All cross-lane softmax state in registers**, via sub-group collectives | Current kernel writes `batch_max` to SLM `batch_max_shared`, reads it back after barrier | `fattn-xmx-f16.hpp:843-853` |
| 3 | **One barrier per SLM phase transition**, no barrier-on-alias | Current kernel has 6+ `group_barrier` calls per KV iteration, some redundant, some protecting aliased regions | `fattn-xmx-f16.hpp:336, 437, 542, 795, 845, 929, 959, 983, 1134, 1151` |
| 4 | **Zero overlap between producer/consumer regions** | `SV_acc` (S@V output) is reused for `batch_max_shared` and sum storage on the NEXT iteration — producer barrier is before consumer write, but compiler-reordered loads within a sub-group could read stale data even with the barrier | Same |
| 5 | **No lane-order-dependent computation** beyond what XMX/sub-group primitives guarantee | Current QK_acc storage via `joint_matrix_store` to SLM, then scalar reduction across lanes reading SLM — lane ordering of reads is unspecified, and the DPAS store layout on Xe2 is implementation-defined | `fattn-xmx-f16.hpp:535-540` |
| 6 | **Fixed KV tile stride** that doesn't depend on `ncols` | Current kernel uses `ncols_padded = max(ncols, XMX_TM)` and allocates SLM proportional to it — an 8x over-allocation at ncols=1 creates more chances for aliasing bugs | `fattn-xmx-f16.hpp:267` |
| 7 | **No `[[sycl::reqd_sub_group_size(N)]]` mismatch** | Current kernel is OK here (`[[sycl::reqd_sub_group_size(XMX_SG)]]` at line 1335) | — |
| 8 | **FTZ on rescale factor via bit-manipulation**, not comparison+branch | Current kernel has inconsistent FTZ handling between batch_max computation and softmax weight computation (comment at line 806 admits this) | `fattn-xmx-f16.hpp:806-807` |
| 9 | **Sinks applied after KV loop, not interleaved** | Current kernel's sink handling is a TODO / not fully present in every code path | Audit pending |
| 10 | **Mask loaded per-KV-tile with explicit stride**, not precomputed per-query-range | Current kernel's multi-seq logic at lines 344-400 mixes mask and seq-bounds, hard to reason about | `fattn-xmx-f16.hpp:344-400` |

**SLM layout (v2 — strict, non-aliasing)**:
```
tile_Q[ncols][D]                       half  — Q tile, used for the entire kernel lifetime
tile_K[BATCH_KV][D]                    half  — K tile, loaded per KV iteration, reused via barriers
tile_V[BATCH_KV][D]                    half  — V tile, loaded per KV iteration, separate from tile_K
```
Total: `(ncols + 2*BATCH_KV) * D * 2` bytes. For D=128, ncols=8, BATCH_KV=32: `(8 + 64) * 128 * 2 = 18432 bytes`. BATCH_KV is sized at runtime to fit within the queried `local_mem_size`; see `fattn_xmx_v2_config::compute_batch_kv(D, ncols, slm_bytes_available)`.

**Register state (per sub-group, per lane as appropriate)**:
- `float KQ_max[cols_per_thread]` — per-lane running max
- `float KQ_sum[cols_per_thread]` — per-lane running sum
- `float VKQ[cols_per_thread][D_per_thread]` — per-lane output accumulator
- Intermediate `joint_matrix` fragments for Q@K^T and S@V

**Per-iteration control flow** (single KV batch):
1. Sub-groups cooperatively load `tile_K` and `tile_V` from global to SLM. One `group_barrier`.
2. Each sub-group computes its slice of `QK = Q @ K^T` using `joint_matrix_mad`. Accumulator `mat_QK` lives in registers (fragment storage). **NO SLM write of QK here** — unlike the current kernel which writes `QK_acc[ncols * BATCH_KV]` to SLM.
3. Each lane reads its owned QK element directly from the fragment (via `joint_matrix_apply` or explicit load to C-fragment register per Intel SYCL matrix spec). For Xe2 `m8n16k16` fp16→fp32 accumulator, each lane in a sub-group of 16 owns exactly `(8*16)/16 = 8` output elements (documented lane layout).
4. Apply softcap/mask per-lane in registers.
5. Sub-group reduce for `batch_max` via `sycl::reduce_over_group(sg, lane_max, sycl::maximum<float>{})` — deterministic.
6. Online softmax update: all register state, all lanes see the same `batch_max`.
7. Convert softmax weights back to half and store to `tile_S` SLM (new allocation for this purpose — NO ALIASING with any other buffer). One `group_barrier`.
8. Compute `SV += tile_S @ tile_V` via `joint_matrix_mad`. Accumulator in registers.
9. Extract lane's owned SV slice from fragment, add to `VKQ` registers (with the `scale_old` rescale from step 6).

**Tile selection — compile-time instantiation, runtime dispatch**:

SYCL `joint_matrix<SG, T, use::a, M, K, layout>` requires M/K/N as template parameters (the fragment type itself is compile-time-parameterized; a sub-group lane-layout for `m8n16k16` is a different type than `m16n16k16`). So we **can't** build one universal kernel that consumes a runtime-selected tile shape — but we also **don't hardcode** a single shape.

Pattern, mirroring the existing `DISPATCH_NCOLS` macro in `fattn.cpp`:

1. **Compile time**: instantiate the kernel for every `(D, ncols, M, K, N)` plausible on any Xe GPU we support (e.g. `m8n16k16`, `m16n16k16`, `m32n16k16` for fp16→fp32). This is ~6-10 template instantiations total. Each gets its own code-object slot.
2. **Backend init (per device)**: query `matrix_combinations` on that device. Compute `best_tile_for(D, ncols, matrix_combinations)` — a pure function that scores each supported combo and returns the winner (covers ncols in M without wasting >50%, K divides D evenly, N matches sub-group size).
3. **Dispatch (per op)**: look up the cached best tile for this device, `switch` into the matching compile-time-instantiated launcher.

Result: **the tile shape IS hardware-discovered** — the hardware's own `matrix_combinations` list is the authoritative input to the selection. We pre-compile the universe of possible kernels (low cost — matrix kernels are small) and at runtime only launch the one the queried hardware actually supports. A future Xe3 reporting `m32n32k16` picks up that path automatically, no source changes.

This is the same pattern we use for `ncols` today (template-instantiated 1/2/4/8 variants, runtime-picked launcher). No hardcoded numbers leak out.

**Divergences from CUDA MMA path** (acceptable — these are platform differences):
- No `cp_async`: use explicit two-phase SLM load + `group_barrier` instead. Xe2 LSU is fast enough that single-buffer loading is fine at our tile sizes; skip double-buffering for simplicity.
- Sub-group size 16, not 32. `cols_per_thread` and `D_per_thread` scale accordingly.
- `__shfl_xor_sync` → `sycl::permute_group_by_xor(sg, val, offset)` or `sycl::reduce_over_group`.
- `__syncthreads()` → `sycl::group_barrier(item.get_group())`.
- `__syncwarp()` → implicit (sub-group execution is lock-step within a sub-group on Xe) or `sycl::group_barrier(sg)` for explicit fence.

### 3.4 The TILE path — `fattn-tile-f16.hpp`

**Mirrors**: `ggml-cuda/fattn-tile.cuh` AND the known-correct current `fattn-mma-f16.hpp` (SYCL).

**When selected**: non-Xe2 GPUs (older Arc, iGPU without full XMX), OR `D > 256 && D < 512` (beyond MMA's efficient range), OR `GGML_SYCL_FA_NO_XMX=1`, OR the XMX-v2 path fails runtime capability check.

**Structure**: essentially today's `fattn-mma-f16.hpp` (which is scalar-dot-product despite the name), **extended** to match CUDA-tile's warp-reduce-max pattern more closely and unified with the new dispatcher. Confirmed correct in the l144i investigation (§15: "GGML_SYCL_FA_NO_XMX=1 → bit-identical correct output"). Rename the file to reflect reality.

**No design changes needed** for the kernel body beyond renames. Changes only at the dispatch and symbol level.

### 3.5 The oneDNN path — `fattn-onednn.cpp`

**When selected**: `sinks == nullptr && logit_softcap == 0.0f && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && D ≤ 512 && !multi_seq && !paged_v2`. Applies to Mistral, Llama-2/3 without softcap, Qwen, etc. NOT gpt-oss (sinks + softcap), NOT Gemma-2 (softcap).

**Interface**:
- One `dnnl::graph::graph` built once per `(D, ncols, has_mask)` tuple, cached in a `std::unordered_map` on the backend context keyed by shape signature.
- Reuse the existing `ctx.engine_dnnl(queue)` / `ctx.stream_dnnl(queue)` interop helpers at `common.hpp:2912-2944`.
- Tensors wrapped in-place via `dnnl::sycl_interop::make_memory(md, eng, usm_ptr)` — no copy, no staging. Q/K/V are already USM-pointer-addressable in the SYCL backend.
- Mask passed as explicit `Add` node input (user-generated mask tensor, not implicit causal — llama.cpp already builds the mask explicitly).

**Layout handling**:
- Q is `[D, n_q, n_heads, batch]` in ggml; oneDNN wants `(batch, n_heads, n_q, D)`. Same memory, just different dim-order view — `memory::desc` with strided layout and transposed dim order. No data movement.
- K is `[D, n_kv, n_kv_heads, batch]`; oneDNN wants `(batch, n_kv_heads, n_kv, D)`. Same trick.
- V same as K.
- For GQA: use the 5-D GQA pattern from `examples/graph/gqa.cpp` with `H_kv = n_kv_heads`, `N_rep = n_heads / n_kv_heads`.

**Cache policy**: graph partition + compiled_partition per shape signature. Hit rate is 100% after layer 0 since all layers share the same `(D, n_kv, n_heads)` config within a forward pass. Cache lives for the lifetime of `ggml_backend_sycl_context`.

**Eligibility check**: at fattn dispatch time, check: mask layout contiguous in expected dims, f16 KV, no sinks, no softcap, not paged-v2 mode, `ne11 > 0` (non-empty KV). If any fails, fall through to the SYCL kernel path.

**Risk**: empirical benchmark needed vs. v2 XMX kernel at TG (oneDNN graph overhead per-layer submission vs. a fused kernel). Decode-path optimization in oneDNN 3.10+ claims "GQA 2nd token" improvements but no Xe2 numbers published. **Phase 3 gate**: measure on Mistral 7B TG; if oneDNN is slower than v2 XMX, route Mistral TG through v2 XMX instead. PP should always pick oneDNN (large-batch SDPA is where oneDNN's fused kernel shines).

### 3.6 Sub-group reductions — the SYCL primitives that replace SLM for softmax state

This is the ONE insight that makes the rewrite simpler than the current kernel:

| CUDA primitive | SYCL equivalent | Xe2 codegen | Determinism |
|----------------|------------------|-------------|-------------|
| `__shfl_xor_sync(0xFFFFFFFF, v, off, 32)` | `sycl::permute_group_by_xor(sg, v, off)` | SIMD lane swap instruction | deterministic per launch |
| `__shfl_sync(0xFFFFFFFF, v, src, 32)` | `sycl::select_from_group(sg, v, src)` | SIMD broadcast | deterministic |
| warp butterfly reduce | `sycl::reduce_over_group(sg, v, op)` | tree-reduction SIMD instr on Xe2 | **spec: deterministic per-kernel-compile** |
| `__syncwarp()` | implicit for sub-group OR `sycl::group_barrier(sg)` | no-op OR fence | deterministic |
| `__shared__ float buf[N]` | `sycl::local_accessor<float, 1>(N, cgh)` | SLM allocation | deterministic IF no aliasing |

The current broken kernel uses SLM+barrier for what should be a 2-line sub-group reduction. Moving to register+reduce_over_group eliminates the aliasing hazard class AND simplifies the code.

## §4 — Dispatch policy

Replace `ggml_sycl_flash_attn_ext_dispatch_ncols` in `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn.cpp:740-828`.

### 4.1 Decision tree

```
on entry to ggml_sycl_flash_attn_ext with (Q, K, V, mask, sinks, dst):
    D = Q->ne[0]
    ncols = Q->ne[1]
    ne11 = K->ne[1]  // KV length
    has_sinks = (sinks != nullptr)
    has_softcap = (logit_softcap != 0.0f)
    is_paged_v2 = (g_sycl_paged_v2_enabled && ne11 > 512)
    multi_seq = (n_seqs > 1)
    kv_is_fp8 = (K->type == GGML_TYPE_F8_E4M3)
    safe_decode_env = getenv("GGML_SYCL_FA_DETERMINISTIC")

    if safe_decode_env:                        -> TILE (scalar, last-resort correctness)
    elif is_paged_v2:                          -> v2-partition (unchanged)
    elif (!has_sinks && !has_softcap
          && !kv_is_fp8 && !multi_seq
          && D <= 512 && f16_KV
          && g_sycl_fa_onednn_enabled):        -> ONEDNN
    elif ncols == 1 && D <= 256 && !kv_is_fp8: -> VEC       (TG fast path, deterministic by construction)
    elif ncols == 2 && D <= 256 && gqa_opt
         && !kv_is_fp8:                        -> VEC       (GQA decode)
    elif D <= 256 && gpu_has_xmx:              -> XMX-v2    (MMA-style, mirrors CUDA fattn-mma-f16)
    else:                                      -> TILE      (scalar SLM, known-correct today)
```

### 4.2 Env vars

- `GGML_SYCL_FA_DETERMINISTIC=1` → force TILE everywhere. Guaranteed bit-identical across runs by construction (scalar, per-thread private VKQ, no DPAS, no SLM aliasing). Slower but this is the "gun to the head, just work" mode.
- `GGML_SYCL_FA_ONEDNN=0` → disable oneDNN SDPA path, fall through. Default ON when model lacks sinks+softcap.
- `GGML_SYCL_FA_XMX_V1=1` → force the old (broken) kernel for A/B comparison during Phase 2 validation. Removed in Phase 4.
- `GGML_SYCL_FA_NO_XMX=1` → legacy alias, now means "use TILE instead of XMX-v2". Same semantics as `GGML_SYCL_FA_DETERMINISTIC=1`.
- `GGML_SYCL_FA_FORCE_VEC=1` → force VEC path even for ncols > 2 (diagnostic).

### 4.3 Coverage matrix

| Model family | D | KV type | Sinks | Softcap | Paths at ncols=1 (TG) | Paths at ncols=8 (PP) |
|--------------|---|---------|-------|---------|------------------------|-------------------------|
| Mistral 7B | 128 | f16 | no | no | ONEDNN (or VEC fallback) | ONEDNN (or XMX-v2 fallback) |
| Llama-3 70B | 128 | f16 | no | no | ONEDNN | ONEDNN |
| gpt-oss-20b | 64, 128 | f16 | **yes** | **yes** | VEC | XMX-v2 |
| Gemma-2 | 256 | f16 | no | **yes** | VEC | XMX-v2 |
| Llama-2 70B Q8_0 KV | 128 | Q8_0→f16 dequant | no | no | ONEDNN after dequant | ONEDNN after dequant |
| gpt-oss FP8 KV | 64, 128 | FP8_E4M3 | yes | yes | VEC (with on-the-fly dequant) | XMX-v2 (dequant in tile load) |

FP8 dequant stays inside each kernel's tile load, same as today.

## §5 — Implementation map

### 5.1 Files to create

| File | Est. LOC | Depends on |
|------|----------|------------|
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-vec-f16.hpp` | 350 | `fattn-common.hpp`, `common.hpp` |
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-tile-f16.hpp` | 500 (renamed from fattn-mma-f16.hpp with minor unification edits) | `fattn-common.hpp` |
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-xmx-f16-v2.hpp` | 900 (down from 1390 in v1 — the simplification is the point) | `fattn-common.hpp`, Intel SYCL matrix extension |
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-onednn.hpp` | 80 (interface) | `dnnl-ops.hpp`, `common.hpp` |
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-onednn.cpp` | 450 (graph build + cache + dispatch) | `dnnl.hpp`, `dnnl_graph.hpp`, `dnnl_sycl.hpp` |

### 5.2 Files to modify

| File | Change |
|------|--------|
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn.cpp` | Rewrite `ggml_sycl_flash_attn_ext_dispatch_ncols` per §4; remove use_xmx branch; wire oneDNN/vec/tile/xmx-v2 branches; remove the l144i input-probe block (or gate on env var) |
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn.hpp` | Add `ggml_sycl_flash_attn_ext_onednn_eligible()` helper; export new kernel launchers |
| `/Apps/llama.cpp/ggml/src/ggml-sycl/CMakeLists.txt` | Add new .cpp files |
| `/Apps/llama.cpp/ggml/src/ggml-sycl/common.hpp` | If not already present, add `sg_deterministic_reduce<T, Op>` wrapper around `sycl::reduce_over_group` for code clarity |
| `/Apps/llama.cpp/CLAUDE.md` | Document new env vars under "SYCL Environment Variables" |
| `/Apps/llama.cpp/docs/backend/SYCL.md` | Document which models get which path |

### 5.3 Files to delete (at Phase 4, after XMX-v2 proves out)

| File | Reason |
|------|--------|
| `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-xmx-f16.hpp` | Racy. Superseded by fattn-xmx-f16-v2. |

Keep for the duration of Phases 1-3 as the A/B reference (gated by `GGML_SYCL_FA_XMX_V1=1`).

### 5.4 Symbol renames

| Current symbol | New symbol | File |
|----------------|------------|------|
| `launch_fattn_xmx_f16` | `launch_fattn_xmx_v1_f16` (during transition, then delete) | fattn-xmx-f16.hpp |
| `launch_fattn_mma_f16` | `launch_fattn_tile_f16` | fattn-tile-f16.hpp |
| `fattn_mma_f16_available` | `fattn_tile_f16_available` | fattn-tile-f16.hpp |
| `fattn_xmx_f16_available` | `fattn_xmx_v2_f16_available` | fattn-xmx-f16-v2.hpp |

### 5.5 Integration points untouched

- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-common.hpp` — `fattn_params` struct stays, all kernels share it.
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-v2-partition.hpp` — partitioned path wrapper stays; its leaf kernels swap from v1-xmx to v2-xmx.
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-esimd-f16.hpp` — ESIMD path orthogonal; still dispatched for `GGML_SYCL_FA_ESIMD=1` at ncols=1.
- `/Apps/llama.cpp/ggml/src/ggml-sycl/kv-cache-quant.hpp` — FP8 dequant helpers stay.

## §6 — Data flow

### 6.1 Common prefix (unchanged)

1. `ggml_backend_sycl_graph_compute` dispatches `GGML_OP_FLASH_ATTN_EXT` to `ggml_sycl_flash_attn_ext` (fattn.cpp:831).
2. Tensor pointer resolution via `ggml_sycl_resolve_tensor_ptr` — USM pointers on the active device.
3. `fattn_params` built (scale/max_bias/logit_softcap/strides/etc).
4. `init_paged_v2_config`, `init_kv_fp8_config`, `init_fa_esimd_config` called (unchanged).
5. Dispatch-policy decision tree runs (§4).

### 6.2 VEC path flow

```
ggml_sycl_flash_attn_ext
  └─ dispatch_ncols(D, Q_type) [VEC branch]
       └─ launch_fattn_vec_f16<D, ncols, softcap, Q_t>(params, stream)
            └─ stream->submit:
                 ├─ cgh.parallel_for<1-sub-group-per-query-head>(
                 │     nd_range = (n_q_blocks * ne02 * ne03, 1, 16),
                 │     local = (1, 1, 16),
                 │     reqd_sub_group_size=16)
                 └─ [kernel body] no local_accessor, register-only state
                    ├─ load Q strip
                    ├─ for kv in [0, ne11):
                    │    ├─ load K[kv] strip
                    │    ├─ dot = reduce_over_group(sg, Q_strip · K_strip, plus<float>)
                    │    ├─ apply softcap / mask
                    │    ├─ online softmax register update
                    │    ├─ load V[kv] strip
                    │    └─ VKQ_partial += exp(dot - KQ_max) * V_strip
                    ├─ apply sinks (if present)
                    └─ store VKQ_partial / KQ_sum to dst
```

### 6.3 XMX-v2 path flow

```
dispatch_ncols(D, Q_type) [XMX-v2 branch]
  └─ launch_fattn_xmx_v2_f16<D, ncols, softcap, Q_t>(params, stream)
       └─ stream->submit:
            ├─ local_accessor<half, 1> slm(ncols*D + 2*BATCH_KV*D, cgh)   [no aliasing]
            ├─ cgh.parallel_for<N-sub-groups-per-work-group>(
            │     nd_range = (n_q_blocks * ne02 * ne03, 1, XMX_NTHREADS),
            │     reqd_sub_group_size=16)
            └─ [kernel body]
               ├─ cooperatively load tile_Q from global to SLM
               ├─ group_barrier
               ├─ for kv_start in [0, ne11, BATCH_KV):
               │    ├─ cooperatively load tile_K, tile_V from global to SLM
               │    ├─ group_barrier
               │    ├─ joint_matrix_mad: mat_QK = tile_Q @ tile_K^T (registers)
               │    ├─ per-lane: extract QK slice from mat_QK fragment
               │    ├─ apply softcap, mask per-lane (register)
               │    ├─ reduce_over_group(sg, lane_max, maximum) for batch_max
               │    ├─ online softmax register update (all lanes in sg see same batch_max)
               │    ├─ convert softmax weights to half, write to tile_S SLM
               │    ├─ group_barrier
               │    ├─ joint_matrix_mad: mat_SV = tile_S @ tile_V (registers)
               │    └─ VKQ += lane's slice of mat_SV
               ├─ apply sinks (if present) — register-only
               └─ store VKQ / KQ_sum to dst
```

### 6.4 oneDNN path flow

```
dispatch_ncols(D, Q_type) [ONEDNN branch]
  └─ ggml_sycl_flash_attn_ext_onednn(params, stream)
       ├─ build or lookup cached dnnl::graph::compiled_partition
       │    keyed by (D, ncols, ne11, has_mask)
       │    ┌─ if miss:
       │    │    1. logical_tensor Q, K, scale, mask, V, output (strided)
       │    │    2. op MatMul(bmm1, transpose_b=true): Q, K -> score
       │    │    3. op Divide(scale_div): score, scale -> scaled_score
       │    │    4. op Add(mask_add): scaled_score, mask -> masked_score
       │    │    5. op SoftMax(axis=-1, mode="inf_as_zero"): masked_score -> probs
       │    │    6. op MatMul(bmm2): probs, V -> output
       │    │    7. graph.finalize(); partitions = graph.get_partitions();
       │    │    8. cp = partitions[0].compile({Q_lt, K_lt, ...}, {out_lt}, eng);
       │    └─ store cp in map
       ├─ wrap USM ptrs as dnnl::memory via sycl_interop::make_memory
       ├─ cp.execute(stream_dnnl(q), {Q_mem, K_mem, scale_mem, mask_mem, V_mem}, {out_mem})
       └─ [no explicit wait — SYCL event chain via stream_dnnl]
```

## §7 — Build sequence (phased)

### Phase 1 — VEC path (deterministic TG fast path) [est. 2 days]

- [ ] Create `fattn-vec-f16.hpp` per §3.2. Implement for D=64, 128, 256, ncols=1, both f16 and f32 Q, with/without softcap, with/without sinks, with/without mask.
- [ ] Create backend-ops tests covering every shape+flag combination.
- [ ] Dispatch gate: `if ncols == 1 && D <= 256` → VEC.
- [ ] Run gpt-oss-20b completion test: `GGML_SYCL_FA_VEC=1 llama-completion -p '1, 2, 3, 4, 5,' -n 30 --seed 42 --temp 0` — expect coherent English, deterministic across 3 runs.
- [ ] Validate bit-identical output probes (`l144i-probe.hpp` `cf/dst_exit`) across 3 runs.
- [ ] Benchmark TG on Mistral 7B Q4_0 and gpt-oss-20b. Record baseline.

**Gate**: gpt-oss-20b TG produces coherent output with bit-identical hashes across 3 runs. TG ≥ current MMA fallback.

### Phase 2 — TILE path unification [est. 1 day]

- [ ] Rename `fattn-mma-f16.hpp` → `fattn-tile-f16.hpp`. Rename symbols per §5.4. Audit for any remaining references.
- [ ] Remove dead `use_xmx && !safe_decode` branch — replace with the new policy.
- [ ] Add explicit handling for ncols=8, D=256 (the worst-case coverage).
- [ ] Re-run `test-backend-ops -o FLASH_ATTN_EXT` and confirm pass rate stays ≥ baseline.

**Gate**: test-backend-ops pass count ≥ baseline; no perf regression on the shapes TILE handles.

### Phase 3 — oneDNN SDPA path [est. 3 days]

- [ ] Create `fattn-onednn.{hpp,cpp}` per §3.5.
- [ ] Build graph cache keyed by shape signature.
- [ ] Wire eligibility check into dispatcher.
- [ ] Test against Mistral 7B Q4_0 on every bench config.
- [ ] Benchmark Mistral TG and PP. Target: TG ≥ current (81 tok/s), PP ≥ current (1480 tok/s).
- [ ] If oneDNN TG slower than VEC on Mistral, restrict ONEDNN branch to `ncols ≥ 8` (PP only), leave TG on VEC.

**Gate**: Mistral TG ≥ 81 tok/s, PP512 ≥ 1480 tok/s (matches CLAUDE.md targets), correctness unchanged.

### Phase 4 — XMX-v2 path (CUDA-MMA-structural port) [est. 5 days]

- [ ] Create `fattn-xmx-f16-v2.hpp` per §3.3. Port `fattn-mma-f16.cuh` structurally.
- [ ] Implement for D=64, 128, 256, ncols=2,4,8, f16 Q, with/without softcap, with/without sinks, with/without mask, with/without FP8 KV.
- [ ] A/B comparison vs current `fattn-xmx-f16.hpp`: `GGML_SYCL_FA_XMX_V1=1` vs v2 on same shapes. Confirm v2 output bit-identical across 3 runs (same seed) on gpt-oss-20b. Confirm v1 non-deterministic (reproduce the bug as a regression test).
- [ ] Benchmark PP on gpt-oss-20b and Mistral 7B. Confirm no regression vs v1's PP numbers.
- [ ] Delete `fattn-xmx-f16.hpp`; remove v1 env gate.

**Gate**: gpt-oss-20b with `ncols ≥ 4` (PP/prefill) bit-identical across 3 runs, perf ≥ v1 PP.

### Phase 5 — Cleanup and docs [est. 1 day]

- [ ] Remove v1 XMX file and `GGML_SYCL_FA_XMX_V1` env gate.
- [ ] Update `/Apps/llama.cpp/docs/backend/SYCL.md` with the path matrix from §4.3.
- [ ] Update `/Apps/llama.cpp/CLAUDE.md` env var table.
- [ ] Close `llama.cpp-l144i` bead.

**Total**: ~12 days if all phases go smoothly. Phase 4 has the most risk because it reintroduces `joint_matrix_mad`; see §9.

## §8 — Critical details

### 8.1 Error handling

- **SLM size query-then-fit**: at backend init, cache `slm_bytes_available = device.get_info<info::device::local_mem_size>()`. `fattn_xmx_v2_config::compute_batch_kv(D, ncols, slm_bytes_available)` picks the largest `BATCH_KV` that fits (ncols + 2*BATCH_KV) * D * 2 bytes. Assert at kernel entry that computed size ≤ queried size. Never hardcode.
- **joint_matrix unavailable**: compile-time gate `#if SYCL_JOINT_MATRIX_AVAILABLE` (already present). Runtime gate `gpu_has_xmx(device)` (already present). If both pass but v2 kernel throws at runtime: catch in the launcher, log, fall through to TILE path.
- **oneDNN graph compile failure**: wrap partition.compile() in try/catch; on failure disable ONEDNN for that shape signature permanently (add to negative cache), fall through to kernel path.
- **FP8 KV on VEC path**: requires on-the-fly dequant in lane strip load; implement as inline `fp8_e4m3_to_float()` per lane, no cross-lane state.

### 8.2 State management

- No global mutable state in kernel bodies (current kernel is already clean here).
- oneDNN graph cache: `std::unordered_map<shape_key, dnnl::graph::compiled_partition>` owned by `ggml_backend_sycl_context`, destroyed at context teardown.
- V2 partition / paged attention buffer cache: untouched — reuses `g_v2_auto` thread-local (see fattn.cpp:112).
- Thread-local seq-id buffers: untouched (fattn.cpp:183).

### 8.3 Testing plan

1. **Unit**: `test-backend-ops -o FLASH_ATTN_EXT` — must pass for every shape the current baseline passes.
2. **Integration (correctness)**: `llama-completion -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0` — output `6, 7, 8, 9, 10, 11, 12, 13, 14, 15` deterministic across 3 runs on:
   - Mistral 7B Q4_0 (no sinks/softcap; ONEDNN path)
   - Gemma-2 (softcap; XMX-v2 path)
   - gpt-oss-20b-mxfp4 (sinks+softcap; VEC for TG, XMX-v2 for PP)
3. **Integration (determinism)**: `l144i-probe.hpp` `cf/dst_exit` hashes for `__fattn__-0..23` bit-identical across 3 runs on all three models.
4. **Perplexity**: `llama-perplexity` on wikitext-2 subset — match CPU reference within 0.01 PPL on Mistral 7B, within 0.05 on gpt-oss-20b.
5. **Perf**:
   - Mistral 7B Q4_0: TG128 ≥ 70 tok/s (ideally ≥ 81), PP512 ≥ 1480 tok/s.
   - gpt-oss-20b: TG ≥ 15 tok/s, PP ≥ 54 tok/s (per l144i targets).
6. **A/B regression**: keep `GGML_SYCL_FA_XMX_V1=1` available through Phase 4. Automated probe-hash script runs v1 twice, confirms non-deterministic output; runs v2 twice, confirms deterministic. A regression in v2 would flip the v2 test; a Heisenbug fix attempt would flip the v1 test.
7. **Thermal throttle guard** (per MEMORY.md): ≥ 30 s sleep between benchmarks.

### 8.4 Performance considerations

- **SLM bank conflicts on Xe2**: 32 banks of 4 bytes, sub-group-size 16. Half-element stride of 2 bytes means 2 lanes share a bank on consecutive access — bank conflict. Mitigation: pad leading dimension of tile_K and tile_V to odd half-stride (e.g. `D + 1` instead of `D`). Current v1 kernel has PAD in places but not consistently; v2 must audit every tile stride.
- **L1 cache line 64 B**: vectorized half8 loads from global (16 B per lane per issue × 16 lanes = 256 B = 4 cache lines). Align tile_K/V loads to 64 B boundaries at the per-thread level. ggml KV cache alignment is already 32 B minimum; verify 64 B for Xe2.
- **Occupancy**: `XMX_NTHREADS` derives from `sub_group_size * n_subgroups_per_wg`, where `sub_group_size` is queried and `n_subgroups_per_wg` is chosen to balance SLM use against parallelism. Occupancy per Xe core = `floor(slm_per_xe_core / slm_per_wg)` work-groups — both numerator (queried) and denominator (computed from config) available at launch. Verify ≥ 4 wgs/Xe-core during perf tuning; below that, reduce `BATCH_KV` or `n_subgroups_per_wg`.
- **Register pressure**: `VKQ[ncols][D_per_thread]` at ncols=8, D=256, D_per_thread=D/nthreads=2 → 16 floats per lane per Q col × 8 cols = 128 floats = 512 bytes of register per lane. Plus `KQ_max[8]`, `KQ_sum[8]` = 64 B more. Plus joint_matrix fragments (lane portion of 8×16 C matrix = 8 floats). Total ~640 B per lane ≈ 160 GRF x 4B entries. Xe2 has 128 GRF per lane standard, 256 GRF extended. May need `[[intel::grf_size(256)]]` kernel attribute for large-ncols cases. Benchmark with and without.

### 8.5 Security

- No user-controlled memory access. Kernel inputs are device-local or already-validated USM pointers.
- Strided loads bounded by `ne11`, `ne01` checks at kernel entry (present in v1).

### 8.7 Multi-GPU (tensor-parallel and pipeline) first-class support

The existing llama.cpp SYCL backend supports tensor-split across multiple devices (e.g. `GGML_SYCL_SPLIT_RATIO="60,32,8"` for B580+B50+CPU). The fattn rewrite must preserve this. Rules:

**Per-device state, never global:**
- **Device capability cache**: `struct fattn_device_caps { size_t slm_bytes; size_t sg_size; std::vector<matrix_combo> combos; tile_selection best[D_max][ncols_max]; };` — one instance per device, stored on `ggml_backend_sycl_context` keyed by `device_index`. Queried once at `ggml_backend_sycl_device_init` time per device, not globally. Heterogeneous configs (B580 + B50 both Xe2 but different EU counts) may resolve to different `BATCH_KV` sizes per device.
- **oneDNN graph cache**: `std::unordered_map<(device_id, shape_key), dnnl::graph::compiled_partition>`. Compiled partitions are **not portable across devices** in oneDNN — they're compiled for a specific engine which wraps a specific device. Cross-device reuse is a silent bug (partition runs on wrong device or fails).
- **Kernel launch**: always uses `ctx.stream(device_idx)` — the per-device queue. Never use a global stream. (Already the current pattern; preserve it.)

**Cross-device tensor ops** (the dispatch layer handles these, unchanged from today):
- Tensor-split attention: each device runs fattn on its head slice. The dispatcher calls our per-device `ggml_sycl_flash_attn_ext` for each device with its owned (Q, K, V) pointers. Each device independently picks VEC/TILE/XMX-v2/oneDNN based on its own capabilities.
- Pipeline MoE (`GGML_SYCL_PIPELINE_MOE=1`): device 1 (B50) runs experts, device 0 (B580) runs attention. Fattn path selection on device 0 is independent of device 1's path.
- KV cache residency: KV may be on device 0 while Q is on device 1 (host-weights streaming mode). Current fattn entry resolves tensors via `ggml_sycl_resolve_tensor_ptr(t, ctx.device)` — this stays. The new kernels must not assume Q/K/V are all on the same device; they inherit current behavior (staging via `cpy_to_device` in the caller before fattn entry).

**Determinism under multi-GPU**: each device's kernel is independently deterministic (per §8.6). Cross-device reductions (e.g. combining head slices) happen in the dispatch layer via existing AllReduce paths — not inside the fattn kernel. So multi-GPU determinism = single-GPU determinism × N, same contract.

**Test matrix**: Phase 1 VEC tested on single-device, then tensor-split 2-device, then tensor-split 3-device (B580+B50+CPU with CPU fallback). Same for each subsequent phase. Existing `GGML_SYCL_SPLIT_RATIO` bench scripts cover this.

**Risk**: heterogeneous hardware (B580 Xe2 + iGPU Xe-LPG with different SLM sizes / tile combos). The per-device capability cache handles this by resolving different kernels on different devices within the same inference pass. Tested empirically in Phase 1 gate.

### 8.6 Determinism contract (the whole point)

The rewrite is **contractually deterministic** under:
1. Same input tensor bytes (verified by l144i probes).
2. Same dispatch path selection (deterministic function of shape params).
3. Same kernel compile (SYCL JIT cache on, `SYCL_CACHE_PERSISTENT=1` — already recommended).
4. Same device, same driver, same library versions.

Sources of non-determinism that each path eliminates:
- **VEC**: no SLM, no joint_matrix — only sub-group reduces. SYCL spec guarantees these are deterministic per kernel-compile.
- **TILE**: per-thread register VKQ, SLM for tile_K/V only, scalar accum with fixed loop order. No DPAS.
- **XMX-v2**: no SLM aliasing, no QK SLM round-trip, joint_matrix_mad inputs and consumption strictly register-local, sub-group reductions for cross-lane state. **If joint_matrix_mad itself is hardware-nondeterministic** (the l144i §14.6 hypothesis that the parallel diagnostic was refining), even v2 will fail determinism — in which case the fallback is TILE for all affected shapes. The blueprint pre-plans this exit: v2 validation in Phase 4 explicitly tests the hw-nondet hypothesis via A/B against TILE on identical inputs.
- **ONEDNN**: the library's SDPA implementation is assumed deterministic (not stated in docs; verified empirically in Phase 3). If it isn't, disable for determinism-critical workloads via `GGML_SYCL_FA_ONEDNN=0`.

## §9 — Risks and unknowns

| Risk | Likelihood | Impact | Mitigation | Confidence |
|------|-----------|--------|------------|------------|
| joint_matrix_mad is actually hw-nondet on Xe2 DPAS | **Medium** (l144i §14 localized race to kernel body, hasn't fully cleared DPAS) | XMX-v2 fails Phase 4 determinism | Fall back to VEC for TG, TILE for PP on affected models. VEC + TILE cover every shape. | **Medium** — need the parallel l144i diagnostic to land |
| oneDNN graph SDPA produces different output than ggml CPU reference (rounding, order-of-ops differ) | Low | PPL mismatch on Mistral | Tolerate ≤ 0.01 PPL delta; if exceeded, tighten eligibility | **High** confidence it'll be within tolerance; `f32` intermediate is spec'd |
| oneDNN SDPA slower than v2 XMX for Mistral TG | Medium | No TG perf win from oneDNN | Restrict ONEDNN to PP; route Mistral TG through VEC | **Medium** — no Xe2 TG numbers published, 3.10 "2nd token" optimization claimed but unquantified |
| sycl-tla (CUTLASS-SYCL) flash-attention is superior to everything we write | Low | Should have used it | Research item, not a commitment. Blueprint acknowledges but doesn't depend. If Phase 5 slack, spike-evaluate. | **Low** — sycl-tla is bf16-focused, sinks/softcap support undocumented |
| Intel SYCL matrix spec changes `joint_matrix` lane layout between oneAPI versions | Low | v2 port breaks | Pin oneAPI to 2025.3+; use `joint_matrix_apply` (layout-opaque) instead of fragment-x[] indexing where possible | **High** confidence spec is stable; architecture IDs for BMG are locked |
| Xe2 SLM bank conflicts degrade v2 below v1 PP perf | Medium | Phase 4 perf gate fails | Add explicit padding; measure with Intel GPU Occupancy Calculator | **Medium** — v1 already has ad-hoc XMX_PAD handling |
| oneDNN graph API is not ABI-stable, breaks on minor upgrades | Low | Runtime crash on new oneDNN | CI builds with exact oneDNN version pinned; graceful fallback on partition.compile exception | **High** — graph API has been stable since 3.0 |
| Multi-device (TP) breaks with per-device graph cache | Low | Wrong output on tensor-split | Cache keyed by device ID too | **High** — easy fix, explicit |
| The real bug is NOT in fattn-xmx but in data corruption before it (l144i input probes are racy themselves) | **Low** (inputs proven identical via probe+wait) | Rewrite fixes nothing | The probe is `memcpy + .wait()` — synchronous, reliable. Inputs were confirmed identical 20/20 runs. | **High** confidence probe is sound |

### 9.1 Open questions for user

1. **Should we gate the rewrite on first confirming `joint_matrix_mad` is deterministic?** If DPAS itself is hw-nondet, XMX-v2 (Phase 4) won't fix it — we'd route everything through VEC+TILE+ONEDNN. That's **still a complete solution** but sacrifices ~20-30% PP on gpt-oss-20b relative to a working XMX. **Recommendation**: start with VEC (Phase 1) since it unblocks gpt-oss correctness regardless; decide XMX-v2 vs. TILE-forever after Phase 1 lands.
2. **oneDNN version policy**: the machine has **3.11.3** via oneAPI 2025.3 — we already have GQA 2nd-token decode opts (3.10) and small-head f32 SDPA perf improvements (3.11). Not yet in 3.11: implicit bottom-right causal fusion (3.12/3.13), compressed KV partitions (3.13). **Recommendation**: use 3.11 for Phase 3; upgrade only if Phase 3 benchmarks fall short of targets.
3. **Is sycl-tla worth a spike?** Their FA has FP8 + KV cache + BMG support but no docs on sinks/softcap. A 2-day spike would tell us if it replaces Phase 4 entirely. **Recommendation**: deprioritize unless Phase 4 XMX-v2 gets blocked.

### 9.3 Pre-flight canary tests (run BEFORE Phase 1)

Standalone tests that validate the plan's load-bearing assumptions. Run these first — their outcomes gate or tune downstream phases.

**Test A: DPAS determinism canary** (`tests/sycl-canary/dpas-determinism.cpp`, ~150 lines)
- **Gates**: Phase 4 (XMX-v2). If DPAS is hw-nondeterministic, Phase 4 is unreachable and we commit permanently to VEC+TILE+oneDNN.
- **Approach**: standalone SYCL program. Allocate fixed fp16 A (8×16) and B (16×16) with deterministic non-trivial values. In a kernel, call `joint_matrix_mad` N times (target ~10,000) across work-groups and sub-groups, writing each result to a distinct dst offset. No SLM, no aliasing, no shared mutable state — pure DPAS → register → DRAM. Compute FNV-64 of entire dst buffer. Run the binary 3 times; compare hashes.
- **Pass**: all 3 runs produce identical hash. DPAS is deterministic; the current kernel bug is 100% control-flow (as the plan assumes). Phase 4 is safe.
- **Fail**: hashes differ. DPAS itself is racy on Xe2. Skip Phase 4; enlarge VEC coverage to handle ncols ≥ 2, enlarge TILE coverage for ncols ≥ 4 with reduced perf.
- **Cost**: ~1 hour to write + run.

**Test B: oneDNN graph SDPA fusion smoke** (`tests/sycl-canary/onednn-sdpa-fusion.cpp`, ~100 lines)
- **Gates**: Phase 3 TG perf target. Confirms the graph API actually fuses the 5-op SDPA DAG into one partition on this specific oneDNN 3.11.3 + Arc B580 toolchain.
- **Approach**: build the SDPA graph (MatMul→Div→Add→SoftMax→MatMul) with Mistral-7B dimensions (D=128, S=512, H_q=32, H_kv=8, f16). Call `graph.finalize()` then `graph.get_partitions()`. Inspect partition count and op list per partition.
- **Pass**: one partition containing all 5 ops. Phase 3 TG target achievable.
- **Soft-fail**: multiple partitions (e.g. MatMul+Div+Add+SoftMax fused, MatMul separate). Phase 3 still works but expect ~2x latency overhead from cross-partition execution. Tightens restrict to PP-only.
- **Hard-fail**: partition compile throws on Xe2. oneDNN doesn't support graph SDPA on this driver; restrict oneDNN usage to the primitive API (MatMul + Softmax + MatMul unfused, much slower).
- **Cost**: ~30 min.

**Test C: Xe device capability dump** (`tests/sycl-canary/device-info.cpp`, ~50 lines)
- **Gates**: nothing directly — informs config defaults. Confirms the runtime-query assumptions hold and surfaces the actual numbers we're building against.
- **Approach**: on every SYCL device visible, dump: `local_mem_size`, `sub_group_sizes`, `max_work_group_size`, `max_compute_units`, `matrix_combinations` (every supported `(M, K, N, A, B, C)` tuple), `ext::intel::info::device::gpu_eu_count`, GRF size if queryable.
- **Output**: `docs/plans/data/xe2-b580-caps.txt` for the plan's record.
- **Cost**: ~15 min to write, runs in seconds.

**Recommended execution order**: C → A → B. C first (no cost, informs A's test constants). A next (gates Phase 4). B last (can run in parallel with Phase 1 implementation).

### 9.2 Confidence summary

| Element | Confidence |
|---------|-----------|
| oneDNN SDPA cannot express gpt-oss-20b (sinks+softcap) | **High** — documented op list is fixed, no extension points |
| VEC path will be correct and deterministic for gpt-oss-20b TG | **High** — no SLM, no DPAS, spec-guaranteed sub-group reductions |
| VEC path will meet TG perf target (≥ current MMA fallback = ~current TILE) | **High** — TILE is slow scalar; VEC is per-lane dot with zero SLM round-trips |
| TILE path is correct (already verified via `GGML_SYCL_FA_NO_XMX=1`) | **High** — empirically shown deterministic, correct tokens |
| XMX-v2 will fix the SLM aliasing bug | **High** — the aliasing hazard is removed by construction |
| XMX-v2 will be deterministic if joint_matrix_mad is HW-deterministic | **High** — control flow matches known-correct CUDA fattn-mma |
| XMX-v2 determinism if joint_matrix_mad is HW-nondeterministic | **Zero** — same primitive, same problem; hence Phase 4 gate tests this |
| oneDNN graph SDPA will match current Mistral PP (~1480 tok/s) | **High** — PP is already oneDNN-dominated via oneDNN GEMM today |
| oneDNN graph SDPA will match current Mistral TG (~81 tok/s) | **Low** — per-layer graph execute overhead unknown, may regress |
| Total implementation effort ~12 days for single implementer | **Medium** — depends on how clean the Intel SYCL matrix API is in practice |

---

## Sources

- [Scaled Dot-Product Attention (SDPA) — oneDNN v3.8.2](https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_graph_sdpa.html)
- [Scaled Dot-Product Attention (SDPA) — oneDNN v3.13.0](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa.html)
- [Grouped Query Attention (GQA) — oneDNN v3.8.2](https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_graph_gqa.html)
- [SDPA with Compressed Key and Value — oneDNN v3.8.1](https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_graph_sdpa_compressed_kv.html)
- [Intel oneDNN Release Notes (all versions)](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-deep-neural-network-library-release-notes.html)
- [Intel oneDNN 3.8 Release Notes — Phoronix](https://www.phoronix.com/news/Intel-oneDNN-3.8)
- [oneDNN releases on GitHub](https://github.com/uxlfoundation/oneDNN/releases)
- [Programming Intel XMX Using SYCL Joint Matrix Extension](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-2/programming-intel-xmx-using-sycl-joint-matrix.html)
- [sycl_ext_oneapi_matrix spec — intel/llvm](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)
- [SYCL-TLA (CUTLASS-SYCL) — intel/sycl-tla](https://github.com/intel/sycl-tla)
- [gpt-oss attention sinks in llama.cpp — PR #15091](https://github.com/ggml-org/llama.cpp/pull/15091)
- [SYCL Flash Attention iGPU issue — #19276](https://github.com/ggml-org/llama.cpp/issues/19276)

---

**Relevant file paths** (all absolute, for implementation reference):

- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn.cpp` — dispatcher rewrite target (lines 740-828 for ncols dispatch, 831+ for ggml_sycl_flash_attn_ext entry)
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn.hpp` — public header
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-xmx-f16.hpp` — BROKEN kernel, 1390 lines, to replace (SV_acc/batch_max_shared aliasing at lines 802, 957)
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-mma-f16.hpp` — scalar-correct "TILE" kernel, 493 lines, to rename and keep
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-common.hpp` — shared `fattn_params`
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-esimd-f16.hpp` — orthogonal ESIMD TG path, unchanged
- `/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-v2-partition.hpp` — paged partition wrapper
- `/Apps/llama.cpp/ggml/src/ggml-sycl/dnnl-ops.hpp` — existing oneDNN wrappers (Softmax, Eltwise) for pattern reference
- `/Apps/llama.cpp/ggml/src/ggml-sycl/common.hpp` — `engine_dnnl()` / `stream_dnnl()` interop at lines 2912-2944
- `/Apps/llama.cpp/ggml/src/ggml-cuda/fattn.cu` — CUDA dispatcher (reference for §2.1)
- `/Apps/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh` — CUDA MMA reference for XMX-v2 port (§2.3, §3.3)
- `/Apps/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh` — CUDA TILE reference (§2.4)
- `/Apps/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` — CUDA VEC reference for Phase 1 (§2.5)
- `/Apps/llama.cpp/docs/plans/2026-04-19-l144i-rootcause.md` — predecessor diagnostic
- `/Apps/llama.cpp/docs/plans/2026-04-20-fattn-alt-path-epic.md` — sibling epic (Alt-A/B/C/D alternatives)
