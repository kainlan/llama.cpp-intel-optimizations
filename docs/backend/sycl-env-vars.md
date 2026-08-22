# SYCL Backend Environment Variables (fork tuning)

This is a curated catalog of the `GGML_SYCL_*` tuning and debugging variables
most useful for this fork; it is not an exhaustive inventory. `CLAUDE.md` keeps
only the handful of load-bearing performance opt-outs.

To inventory names in the current source without depending on which environment
accessor reads them, search string literals and allow digits:

```bash
find ggml/src/ggml-sycl -type f \
  \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) -print0 \
  | xargs -0 cat | grep -oE '"GGML_SYCL[A-Z0-9_]*"' | sort -u
```

The output includes every matching literal, including names retained only for
compatibility, diagnostics, comments, or removal notices; confirm a live read
before treating any result as an active setting.

## Performance-critical (all default ON, opt-out)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_UNIFIED_SOA=0` | ON | Disable SOA memory layout (AOS fallback, ~4x slower TG) |
| `GGML_SYCL_TG_FAST=0` | ON | Disable the batch=1 MMVQ fast-path (slower TG). Also the **only** thing that lets `GGML_SYCL_FORCE_DMMV` or a kernel choice from `GGML_SYCL_LAYOUT_OVERRIDE` bind at batch=1 — see below. |
| `GGML_SYCL_DISABLE_GRAPH=1` | OFF | Disable SYCL graph replay (minimal TG impact ~3%, mainly helps PP) |
| `GGML_SYCL_ONEDNN_PP=0` | ON | Disable oneDNN for prompt processing |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (skip unified kernel) |
| `GGML_SYCL_XMX_TILED_PP=0` | ON | Disable the XMX_TILED grouped-DPAS PP route for MXFP4 gate/up (falls back to SOA at PP). Productized 2026-08-17 (llama.cpp-rzy7) from the verified diagnostic state (llama.cpp-e3xj): pp512 461±16 B70 / 149.9±0.9 B50, gates pass, TG neutral-positive, zero UR errors. The older `GGML_SYCL_XMX_MOE_ALLOW_UNSAFE_PP=0` / `GGML_SYCL_XMX_MOE_PP=0` names are still honored as a compatibility opt-out when this variable is unset — despite the "unsafe" name, neither is a diagnostic-only knob anymore. Does not affect the separate `GGML_SYCL_XMX_MOE` prompt-XMX-forced diagnostic path. |
| `GGML_SYCL_MOE_DOWN_XMX_TILED=0` | ON | Disable the XMX_TILED grouped-DPAS route for the MoE **down** projection (falls back to SOA/MXFP4_I8 planning, unchanged). **Semantics changed 2026-08-20 (llama.cpp-sk67): default flipped OFF→ON.** Before sk67 the flag existed but the down dispatcher (`mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4`, `mmvq.cpp`) rejected `GGML_LAYOUT_XMX_TILED` outright at its entry guard regardless of this flag, so `=1` was a no-op — OFF+non-functional. sk67 added a grouped-DPAS arm reusing the existing single-matrix kernel `mxfp4_xmx_tiled_grouped_direct_q8_sycl` (the same kernel already dispatched for gate/up XMX_TILED weights) against the down role's cached Q8 activation artifact, and widened the planner/admission gates (`ggml_sycl_moe_prompt_xmx_tiled_supported`, `ggml_sycl_select_moe_planned_graph_layout`, `ggml_sycl_moe_prompt_down_specialized_layout_proven`, the PP transactional executor's `executor_layouts_ok`/`abi_ok` admission, and the decode fusion path's `down_layout_table_eligible` query) to actually select and reach it — now ON+functional. Expected effect: on B50, all 24 MoE layers reach the grouped-DPAS route for down instead of the slow SOA path (was 92% of B50 MoE device time per the `llama.cpp-sk67` mission). I8 stays preferred wherever the planner already materializes/plans it (e.g. B70 where it fits); this flag only extends coverage to layers that would otherwise fall back to SOA. Also gates part of the decode-phase fused dispatch (`ggml-sycl.cpp`'s `decode_pair_glu_dispatched` block) via `ggml_sycl_moe_decode_xmx_tiled_supported`'s DOWN branch — see `llama.cpp-y0it` for the companion fix (a refused down dispatch used to silently mark itself "handled" and drop the output; now the flag/return there is conditional on the dispatch actually succeeding). A separate, narrower flag `GGML_SYCL_MOE_PHASE_DOWN_XMX` (default OFF, unchanged by sk67) gates a different phase-materialization decision point (`ggml_sycl_moe_phase_target_layout`) not touched by this ticket. |
| `GGML_SYCL_Q8_DENSE_AOS=1` | OFF | Route Q8_0 dense projections (ATTENTION_WEIGHT/FFN_WEIGHT/OUTPUT_WEIGHT — `layout_policy::get_optimal` in `common.hpp`) AOS instead of COALESCED so PP batches reach `ONEDNN_AOS` (the oneDNN jit:gemm arm; COALESCED batches land on `MMQ_COALESCED` ~3 TFLOPS since that case has no oneDNN arm). MEASURED 2026-08-20 (llama.cpp-e3xj C25/C26): +38% B70 GPT-OSS pp512 (847→1160) but **−34% B50 tg128 (32.1→21.3)** because TG moves from the coalesced kernel to MMVQ_AOS — it FAILS the single-layout all-consumers rule, so the default is OFF. Enable only for PP-dominated workloads that accept the TG loss. Superseded for the PP win by `GGML_SYCL_Q8_ONEDNN_COALESCED` below, which gets the oneDNN arm without moving TG off the coalesced kernel. Only affects Q8_0; MoE expert weights are untouched by this flag. |
| `GGML_SYCL_Q8_ONEDNN_COALESCED=0` | ON | The layout-neutral recapture for the `GGML_SYCL_Q8_DENSE_AOS` tradeoff above (llama.cpp-e3xj): adds an `ONEDNN_COALESCED` kernel choice, selected for Q8_0 dense weights in COALESCED layout at batch > `MMVQ_MAX_BATCH_SIZE` (8). Dequants the COALESCED weight straight to FP16 (`dequantize_row_q8_0_coalesced_to_fp16_rowmajor`, reusing the same tile/offset math as the live `DMMV_COALESCED`/`MMVQ_COALESCED` Q8_0 kernels) and routes through the same oneDNN `row_gemm` the `ONEDNN_AOS` arm uses — no AOS materialization, so TG stays on the coalesced kernel (unlike `GGML_SYCL_Q8_DENSE_AOS`). If the COALESCED weight isn't device/shared-resident when the op runs, falls straight through to `MMQ_COALESCED` (the pre-existing behavior); never errors. Set `=0` to force the pre-existing `MMQ_COALESCED` PP path for an A/B. |
| `GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=1` | OFF | **Route policy**, not just an executor switch (llama.cpp-dboi, first-class integration 2026-08-21; layout scope corrected llama.cpp-1tjn, Option T, same day). One predicate — `ggml_sycl_moe_pp_onednn_batched_route_selected()` in `common.hpp`, shared by dispatch and the unified-cache planner — flips these together: (i) the MXFP4 MoE layout chokepoint (`ggml_sycl_select_mxfp4_moe_layout`) is a single per-weight, planning-time decision with no PP/decode caller identity ("one layout per weight" — no duplicates, no dispatch-time layout shifts). **Gate/up are NOT touched by this policy** — they keep the default XMX_TILED materialization unconditionally, so decode's XMX_TILED DPAS kernel (`used_xmx_tiled_dpas`, `mmvq.cpp`) is unaffected. **Down alone still plans SOA under the policy** (the down-i8 planning upgrade stays disabled there — a deliberate, separate tradeoff: down's decode delta is ~1 ms/token, its PP-batched win is banked, tracked apart from gate/up); (ii) at PP the fused pair-GLU executor declines, the per-row MMVQ probes yield MXFP4 gate/up/down, and prompt routing skips the hybrid per-entry world so dispatch reaches the batched executor; (iii) the executor itself (`try_pp_mxfp4_soa_onednn_f16_batched`, `ggml-sycl.cpp`) runs. **Updated 2026-08-22 (llama.cpp-sr83, track C3):** `ggml_sycl_moe_pp_onednn_batched_claims_tensor` now claims gate/up too (XMX_TILED, planned layout) alongside down (SOA) — the interim "gate/up fall back to a slower PP path pending track C" window this row described is closed. By default (`GGML_SYCL_MOE_PP_WOQ` unset/1, see its own row below) the executor repacks 4-bit-to-4-bit straight into a `{nibbles,e8m0-scales}` WOQ scratch shape (`repack_mxfp4_soa_to_woq` for down, `repack_mxfp4_xmx_tiled_to_woq` for gate/up, both `convert.cpp`) and hands them to a batched oneDNN WOQ matmul (`DnnlGemmWrapper::woq_gemm_batch_mxfp4`, `gemm.hpp`) — the f16 dequant is deleted on this arm entirely, ~4x smaller scratch. Setting `GGML_SYCL_MOE_PP_WOQ=0` restores the pre-C3 per-expert f16 dequant + `gemm_batch_strided` arm for A/B or as a fallback if the WOQ primitive ever regresses; experts are still sorted by row count and split into grouped strided oneDNN batches (group `n` padded to 64-multiples for primitive reuse) to avoid ~5x rectangular padding on GPT-OSS's skewed routing, unchanged by the WOQ/f16 choice. **Fail-closed contract:** once an XMX_TILED-claimed op reaches this executor, every internal decline point throws (`ggml_sycl_fallback_error`) instead of returning false, because the outer dispatch has already skipped every other route for it and the remaining per-expert staging fallback cannot decode XMX_TILED weights (llama.cpp-71hx); SOA-claimed (down) declines keep the original fall-through. Decode (`ne12==1`) is untouched by dispatch, and gate/up decode runs its normal tiled/DPAS routes unchanged regardless of this policy; only down's decode path sees the policy's SOA choice. A SOA-native direct-XMX/DPAS kernel for gate/up decode (`used_direct_xmx`, wired via `moe_layer_direct_xmx_check_role`, env-gated by `GGML_SYCL_MOE_DIRECT_XMX_GLU`/`GGML_SYCL_DIRECT_XMX_MOE_PROBE`, off by default) was measured llama.cpp-1tjn 2026-08-21 as a **2x regression** (25.8 vs 12.26 ms/token) — do not re-enable it as a decode speedup; it is documentation of a measured dead end, not a viable route. Scratch ring (`moe-scratch-admission.hpp` + `llama_model_sycl_populate_inventory`) is sized for all `n_expert` experts, admission fail-closed since `f2bdfbffe`; `reserve_pp_moe_onednn_scratch` satisfies from an existing adequate ring without reallocating -- sizing itself now branches on `GGML_SYCL_MOE_PP_WOQ` read at plan time (~130 MB total for GPT-OSS 20B under the default WOQ arm, vs ~776 MB under the f16 opt-out arm; see the `GGML_SYCL_MOE_PP_WOQ` row). Measured clean-host 2026-08-21 (r=2, PRE-Option-T, gate/up still SOA-planned at measurement time): B50 pp512 **569** / tg128 25.4 vs default 122/32.2; B70 pp512 **922** / tg128 37.1 (B70 default is broken on a clean device — llama.cpp-o3h1). These PP numbers predate both Option T's gate/up admission and C3's WOQ arm; re-measure before citing as current. Default stays OFF pending the per-card pp/tg-tradeoff ruling; the per-card default seam lives in the policy helper. `_TRACE=1` and `_COMPARE=1` (+`_COMPARE_LIMIT`) harnesses built in. |
| `GGML_SYCL_MOE_PP_WOQ=0` | ON | Opt out of the WOQ (weight-only-quantized) arm of the batched PP MoE oneDNN executor above (llama.cpp-sr83, track C3) — only meaningful when `GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED` (or its capability-derived default, once wired) has selected the batched route at all. `=0` restores the pre-C3 per-expert f16 dequant (`dequantize_row_mxfp4_soa_to_fp16_rowmajor`) + `gemm_batch_strided` GEMM. Read at TWO points that must stay in sync: `ggml-sycl.cpp`'s executor lambda (per-dispatch) and `llama-model.cpp`'s `llama_model_sycl_populate_inventory` (plan-time scratch-slot sizing) — the plan-time read decides which arm's byte formula sizes the ring, and an under-sized ring makes admission fail-closed refuse every dispatch (`f2bdfbffe`), so flipping this must happen before the model loads, not mid-run. **The f16 arm is SOA-only** (llama.cpp-sr83 fix cycle, defect 1, 2026-08-22): `dequantize_row_mxfp4_soa_to_fp16_rowmajor` has no tiled-aware counterpart, so `GGML_SYCL_MOE_PP_WOQ=0` combined with an XMX_TILED-claimed gate/up op now refuses (`f16-arm-no-tiled-dequant`, throws via the fail-closed contract) rather than misreading tiled bytes as SOA — which previously poisoned every output entry to NaN (the OCP-MX reserved exponent `0xFF` reads as a NaN-equivalent). `=0` therefore only exercises down at PP when gate/up are XMX_TILED-planned. |
| `GGML_SYCL_MOE_PP_WOQ_3D=0` | ON | Forces the WOQ arm's 2-D per-batch-element GEMM loop (`DnnlGemmWrapper::woq_gemm_batch_mxfp4`, `gemm.hpp` — C1's exact proven non-batched recipe, mask 3/group_dims `{group_size,1}`, looped once per expert) instead of attempting the batch-dim-grouped 3-D scale mask (mask 7/group_dims `{1,group_size,1}`, unvalidated on hardware) first. Added llama.cpp-sr83 fix cycle (2026-08-22) as a discriminating experiment: the 3-D primitive is *accepted* by oneDNN (no refusal in the log) but produced wrong numerics on the down role (COMPARE evidence, team lead) — accepted is not the same as correct (the same trap as `ext_oneapi_enable_peer_access`, CLAUDE.md P2P section). Does not touch the per-device capability cache (`tri_state` in `gemm.hpp`), so it is safe to flip between runs without recompiling. `GGML_SYCL_DEBUG=1` prints `[ONEDNN][WOQ-MXFP4-BATCH] device=%d arm=3D\|2D-loop` once per device on whichever arm actually dispatches — printing only existed on 3-D *refusal* before this fix cycle, which made "3-D worked" and "message never reached the log" indistinguishable. |

`GGML_SYCL_DISABLE_GRAPH` controls graph replay, not graph-compute concurrency.
The graph-compute mutex is process-global, but does not universally serialize
submission. Direct/fallback paths release it before `compute_impl` submission.
In contrast, persistent-TG/deferred-copy paths and command-graph record/replay
paths submit while the process-global mutex remains held. Completion may still
outlive the lock where a path permits deferred exit. Thus host submission can
overlap across calls on direct/fallback paths, and device execution may overlap
across calls and devices; pure-GPU decode may also return with kernels still in
flight. Do not infer supported concurrent inference or cache safety from either
overlap. The process-global `unified_cache_set_graph_compute_active(bool)` flag
is an eviction guard, not a per-device concurrency control. Same-device
concurrent inference also remains unsupported for the distinct context/arena
ownership reasons documented in the canonical contract §5.

⚠️ **`GGML_SYCL_DISABLE_GRAPH=1` is currently load-bearing for multi-context
workloads, and that is an open question rather than a setting to recommend.**
Measured at `98deb46ed` with replay ON (the default): the first context to begin
a tracked graph on a device holds its invocation indefinitely — a replay-active
graph never reaches a terminal state, so `release_invocation` silently no-ops —
and every other context's claim on that device is refused `DEVICE_BUSY` exactly
once, with no path by which it later succeeds. The discriminator is clean:
`test-thread-safety` (3 models × 4 contexts, decodes serialized per device)
fails at 3.4 s with replay on and passes at 6.95 s with replay off, both pinned
`level_zero:0,1`.

Consequences worth knowing before you reach for this flag:

- The `test-thread-safety` **ctest registration sets it** (the
  `llama.cpp-b16a`/`llama.cpp-cnre` disposition). That makes the test pass; it
  does not resolve the production question. Do not read that green result as
  evidence that multi-context works on the default configuration.
- Whether winner-holds-forever is intended exclusivity or a gap is being
  adjudicated on `llama.cpp-u7vj` (open, P2), with the analysis in the canonical
  contract §5.3.
- The blast radius is larger than the device count suggests: on current Level
  Zero multi-GPU runtimes the scheduler-visible device set collapses to **one**
  device, so all contexts in a process share device 0's registry slot regardless
  of how many physical GPUs are installed.

## Experimental (opt-in, off by default)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PP_PIPELINE=1` | OFF | Enable double-buffered FP16 weight dequant prefetch. B50 GPT-OSS `llama-bench` PP improves to ~1030-1043 tok/s, but GPT-OSS chat correctness currently fails with repeated `isNaN`, so keep this opt-in until fixed. |
| `GGML_SYCL_XMX_MOE_SORTED=1` | OFF | Arms the diagnostic-only `try_xmx_sorted_moe` sorted MoE wrapper. Decoupled from `GGML_SYCL_XMX_MOE` (llama.cpp-twl6): at higher dispatch priority it used to pre-empt the grouped-DPAS PP route whenever `GGML_SYCL_XMX_MOE=1` was set, measuring 5x slower on the B70 and an rc=134 abort on the B50. Diagnostic use only. |

## Kernel dispatch tuning

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_FORCE_MMVQ=1` | Force MMVQ kernels for all batch sizes |
| `GGML_SYCL_FORCE_ESIMD=1` | Force ESIMD kernels |
| `GGML_SYCL_FORCE_MMQ=1` | Force MMQ kernels |
| `GGML_SYCL_FORCE_DMMV=1` | Force DMMV kernels. ⚠️ **No effect at batch=1 on its own** — the TG fast-path returns upstream of both sites that read it. Pair with `GGML_SYCL_TG_FAST=0`; see "The TG fast-path claims batch=1" below. |
| `GGML_SYCL_ESIMD_MIN_BATCH=N` | Min batch size for ESIMD dispatch |
| `GGML_SYCL_ONEDNN_PP_MIN_BATCH=N` | Min batch for oneDNN PP path |
| `GGML_SYCL_ONEDNN_MUL=1` | Enable oneDNN for element-wise MUL (default OFF, SYCL kernel is 2.3x faster) |
| `GGML_SYCL_FA_ONEDNN_MATERIALIZE=1` | Opt back into `MATERIALIZE_REQUIRED` for GQA/MQA flash-attention shapes whose K/V token stride differs from D — the `KV_NC_STRIDE_MISMATCH` gate in `ggml_sycl_flash_attn_ext_onednn_plan()` (`fattn-onednn.cpp`), which is the single materialize-vs-`DIRECT` decision site. Default **OFF**; unset and `=0` are identical and plan `DIRECT`, skipping the dense f16 K/V repack before the oneDNN SDPA execute. ⚠️ **The polarity reversed on 2026-08-10** (`llama.cpp-olpg`, owner ruling): this variable was introduced by `llama.cpp-l7rt` defaulting **ON** as a pure measurement axis, and any text describing it that way predates the ruling. `DIRECT` is numerically correct at D=16 and D=128 within the descriptors fixture's tolerance and end-to-end on the Mistral digit gate; on B50 pp512 it measured at worst parity across five interleaved pairs (two at parity, three ahead — the magnitude is deliberately not quoted, as the pairs were anti-correlated and only the direction is supported). The heap-corruption abort once recorded against `DIRECT` was a harness artifact of a stashed-library A/B, not a product defect (`llama.cpp-l7rt`). `=1` is retained as the **A/B axis**: it makes the decision revisitable from any build with two env-var runs, no patching and no second `libggml-sycl.so`. |
| `GGML_SYCL_BATCH_EXPERTS=0` | Disable batched expert kernel launches (default ON) |
| `GGML_SYCL_ESIMD_DEQUANT=1` | Opt-in retest hatch for ESIMD small-block dequant; standard SYCL is the default. ⚠️ The 1.9x-slower figure behind that default was measured on an **Arc B580 + oneAPI 2025.3** — that card is no longer in this machine (replaced by the B70) and the toolchain has moved on, so treat it as *historical justification*, not a current measurement. The conclusion is still believed to hold (block granularity too small to amortize LSC loads), but it has not been re-measured on Battlemage G31. Same caveat applies to the copy of this claim in `CLAUDE.md`. |
| `GGML_SYCL_LAYOUT_OVERRIDE=<mode>` | Force a weight layout: `aos`, `soa`, `coalesced`, or `xmx_tiled`. Overrides the layout policy's own choice — use for A/B isolation, not as a default. (Migrated from AGENTS.md 2026-07-25, which was its only documentation.) |
| `GGML_SYCL_USE_XMX_GEMM=1` | Route quantized MUL_MAT through the experimental XMX GEMM kernels (measured 5–11x **slower** for quantized models). Needs a build carrying **both** `GGML_SYCL_XMX_GEMM` and `GGML_SYCL_MMQ_XMX`; in a default build it does nothing. `=0` disables it on both dispatch paths (it did not until `llama.cpp-wvbw`; see below). |
| `GGML_SYCL_XMX_THRESHOLD=N` | Upper batch bound for the XMX GEMM path; the gate is `batch >= 1 && batch < N`. Default **64**, stated only by the settings table in `ggml_check_sycl()` — not by the global's initializer. Same build requirement as above. See below. |
| `GGML_SYCL_MXFP4_GROUPED_DPAS_ROW_LIST_TILES=N` | Caps the grouped-DPAS MXFP4 MoE row-list chunk size at `caps.N * N` rows per submission. Default **256** (raised from 16 on 2026-08-17, llama.cpp-e3xj): 16 forced ~23 chunks/layer on GPT-OSS pp512, each re-reading expert weight tiles; 256 measured +7.5% pp512 on B50 (136.4->146.7, interleaved r=2 pairs) and ~+30% on B70 (356->471-493). |

### The TG fast-path claims batch=1, so the force/override variables never see it

`ggml_sycl_mul_mat()` short-circuits batch=1 quantized `MUL_MAT` on a
GPU-accessible weight that resolved to a non-AoS layout: it dispatches MMVQ with
q8_1 activations against that layout and **returns**, upstream of both
`GGML_SYCL_FORCE_DMMV` reads and of the kernel choice made by
`GGML_SYCL_LAYOUT_OVERRIDE`. Its own comment says it "bypasses orchestrator, name
parsing, prefetch, TP checks for maximum speed".

So for Q4_0 (and any type `ggml_sycl_supports_reorder_mmvq()` accepts):

- **MMVQ is the production TG kernel.** The coalesced DMMV kernel is a
  fallback/debug path in a default configuration.
- **`GGML_SYCL_FORCE_DMMV=1` by itself changes nothing at batch=1.** It is not
  overridden — it is never read. `GGML_SYCL_TG_FAST=0` is what makes it bind.
- **`GGML_SYCL_LAYOUT_OVERRIDE` still binds on materialization.** The overridden
  buffer is really built, in the requested layout, byte-correct — and then MMVQ
  reads it. A passing layout precondition is not evidence about which kernel ran.

`llama.cpp-szv8` spent three RCA rounds on a test that hit this: every structural
precondition passed and the kernel under test never executed. Confirm the kernel
from output, not from setup — either `GGML_SYCL_MUL_MAT_ROUTE_TRACE=1` (below) or
an oracle-fit ratio, never a tolerance. Full write-up, including the by-design
accuracy contract of q8_1 activation quantization, is in
`docs/backend/sycl-perf-baselines.md` ("The production TG path is MMVQ with q8_1
activations — not DMMV").

### `GGML_SYCL_USE_XMX_GEMM` / `GGML_SYCL_XMX_THRESHOLD` — the XMX GEMM path

**Both are compile-gated.** The globals `g_ggml_sycl_use_xmx_gemm` and
`g_ggml_sycl_xmx_threshold`, their `sycl_env_settings` rows in
`ggml_check_sycl()`, and both dispatch sites all sit inside
`#ifdef GGML_SYCL_XMX_GEMM`, which is `option(... OFF)` in
`ggml/src/ggml-sycl/CMakeLists.txt`. In a stock build the variables are not read
at all — setting them measures nothing.

⚠️ **`-DGGML_SYCL_XMX_GEMM` alone does not compile.** The GEMM blocks call
`ggml_sycl_xmx_available()` and `ggml_sycl_xmx_supports_type()`, declared in
`mmq_xmx.hpp`, which `ggml-sycl.cpp` includes only under `GGML_SYCL_MMQ_XMX` — a
second, independently-defaulted CMake option. **Configure both**, e.g.
`-DGGML_SYCL_XMX_GEMM=ON -DGGML_SYCL_MMQ_XMX=ON`.

The requirement is now enforced in two places (`llama.cpp-d6d6`, fixed
2026-07-31), because one of them cannot cover the other:

- `ggml/src/ggml-sycl/CMakeLists.txt` fails the configure with a
  `message(FATAL_ERROR)` naming `GGML_SYCL_MMQ_XMX`. It is a hard error rather
  than an implicit force-ON so that nothing rewrites your cache behind your
  back — `-DGGML_SYCL_XMX_GEMM=ON` will *not* silently turn the other option on
  for you, and `GGML_SYCL_MMQ_XMX` in `CMakeCache.txt` always means what it says.
- `ggml-sycl.cpp` carries the same condition as an `#error` next to the
  conditional include, because the original reproducer was a direct
  `icpx -fsyntax-only -DGGML_SYCL_XMX_GEMM` that never runs CMake at all.

⚠️ **The four undeclared-identifier errors still appear** — the `#error` is the
*first* diagnostic and names the missing flag, but clang does not stop at
`#error`, so it goes on to hit the two dispatch sites and re-emit the old
cascade below it. Read the top of the output, not the bottom: a compile that
ends in `use of undeclared identifier 'ggml_sycl_xmx_available'` is still the
missing-option failure, not a broken XMX path.

**Where the threshold's default comes from — read this before quoting a number.**
The authoritative default is the `GGML_SYCL_XMX_THRESHOLD` row of the
`sycl_env_settings` table in `ggml_check_sycl()`, currently **64**. That parse
writes the global *unconditionally* at backend init, before any `mul_mat`
dispatch can read it, so it wins in every run. Do **not** take the default from
the declaration of `g_ggml_sycl_xmx_threshold`: that initializer is a deliberate
fail-closed `0` covering only the pre-parse window, and
`scripts/check-sycl-xmx-threshold-default.sh` (ctest
`test-sycl-xmx-threshold-policy`) enforces that it stays `0` so a second,
competing literal cannot reappear.

That guard exists because the two literals disagreed for months. `a0ede18b3`
introduced both as `64`; `05519d18f` (*"Increase XMX threshold to 1024 (was 64)
for broader XMX usage"*) changed **only** the initializer, so the increase never
took effect. `43d04b327` made the table row the single source
(`llama.cpp-d5h0`).

⚠️ **So 64 is the value that has always been *effective*, not the value that won
a measurement.** The 1024 hypothesis was never evaluated — the code silently
ignored it, including in that commit's own B50 numbers, which were taken at 64.
`llama.cpp-eju9` is open to actually measure 64 vs 1024 (interleaved paired A/B,
both cards, plus the Mistral completion gate, since a threshold change reroutes
kernel dispatch). If you are tuning this you are in unmeasured territory, not
second-guessing a benchmarked choice.

**The gate**, identical in `ggml_sycl_select_preferred_kernel` and
`ggml_sycl_mul_mat`:

```c
use_xmx = batch >= 1 && batch < g_ggml_sycl_xmx_threshold;
```

so `GGML_SYCL_XMX_THRESHOLD=0` or `=1` disables the XMX path outright, and
"XMX for every batch" is only reachable by naming a large `N`.

✅ **`GGML_SYCL_USE_XMX_GEMM=0` disables XMX everywhere** — fixed 2026-08-01,
`llama.cpp-wvbw`. Both dispatch sites now read the parsed global
`g_ggml_sycl_use_xmx_gemm`, so the value is honoured and the startup report
cannot disagree with the behaviour.

⚠️ **The advice this section carried until then was a workaround for a bug, and
if you find it repeated anywhere else it is now wrong.** It read: *"one of them
ignores the value … `ggml_sycl_select_preferred_kernel` tests
`std::getenv(…) != nullptr` **or** the global … To disable, leave the variable
unset; do not set it to `0`."* That was accurate at the time —
`select_preferred_kernel` tested mere *presence*, so `=0` enabled XMX there while
`ggml_sycl_mul_mat` read the same `0` as disabled, giving one process two
opposite dispatch configurations and a report that printed `0` regardless. The
presence form had been copied from the `GGML_SYCL_FORCE_MMQ` /
`GGML_SYCL_FORCE_DMMV` idiom directly below it, where set-to-anything *is* the
intent because those have no settings row, no global and no report line.

Regression gate: `scripts/check-sycl-xmx-enable-single-source.sh` (textual — the
block is `#ifdef GGML_SYCL_XMX_GEMM`, so an ordinary green build compiles zero
lines of it and certifies nothing). It carries its own fixture suite:
`scripts/check-sycl-xmx-enable-single-source.sh --self-test`.

`ggml_check_sycl()` reports the threshold only when the enable flag is on, and
that report is `GGML_LOG_INFO` — see CLAUDE.md on why those lines need raised
verbosity to appear at all.

## Binbcast completion event

`ggml_sycl_op_bin_bcast` (ADD / SUB / MUL / DIV / REPEAT) publishes one completion
event to two consumers: `ggml_sycl_set_tensor_ready_event(dst, ...)` for
`GGML_OP_MUL`, and `unified_cache::unpin_on_event`, which parks the weight-cache
lease release on it. `GGML_SYCL_BINBCAST_EVENT_MODE` selects where that event
comes from.

| Value | Default | Effect |
|-------|---------|--------|
| `barrier` | **yes** (also the fallback for any unrecognised value) | Manufacture the event with `ext_oneapi_submit_barrier()`. On an in-order queue this is silently promoted to `safe`. |
| `safe` | | Manufacture the event with an empty `single_task` marker kernel. |
| `reuse` | | Return the binbcast kernel's own submission event — no extra submission. Falls back to a real `safe` submission if no kernel event was captured. |

`reuse` exists to remove the empty marker submissions from the decode path (72 per
token on GPT-OSS 20B). The kernel event carries the same guarantee both consumers
need: the kernel is what writes `dst` and what reads the pinned weights, so its
completion means `dst` is ready and the pins are safe to release.

⚠️ **There is no "no event" option, and adding one is not an optimisation.** A
default-constructed `sycl::event` reads as **already complete**, so handing one to
`unpin_on_event` releases the weight-cache lease while the GPU is still reading the
weight — `DEVICE_LOST` or silent corruption. That failure mode presents as a
**speedup**, so a faster run is not evidence that it is correct. Every mode, on
every path, must yield a real completion event.

Debug: with `GGML_SYCL_DEBUG=1` the unpin path logs
`[SYCL-BINBCAST] unpin event mode=<mode> source=<kernel|submission> pins=<n>`.
`source` is what actually happened — a configured `reuse` that fell back still
reports `submission`. `GGML_SYCL_BINBCAST_TRACE=1` adds `[BINBCAST]` staging and
launch traces from the same file.

Gate (all three modes, plus the host-only event-source policy check):

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-unpin-event --mode=compare
```

## Persistent TG kernel (experimental, opt-in)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PERSISTENT_TG=1` | OFF | **Required** to enable persistent TG kernel. Without this, all other PERSISTENT_TG_* vars are ignored |
| `GGML_SYCL_PERSISTENT_TG_PHASE=0` | ON | Disable phase-based scheduling (falls back to DAG or legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_DAG=0` | ON | Disable DAG scheduling (falls back to legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_N_WGS=N` | auto | Override work-group count (auto: max_compute_units/4, clamped 4-64) |
| `GGML_SYCL_PERSISTENT_TG_LOG_POLICY=1` | OFF | Print kernel dispatch mode (phase/dag/split/n_wgs) on each launch |
| `GGML_SYCL_MOE_BLOCK_GRAPHLETS=1` | OFF | Enable experimental MoE block command graphlets |
| `GGML_SYCL_PERSISTENT_SPLIT=1` | OFF | Enable persistent kernel for multi-device row-split |

Testing persistent TG modes:

```bash
# Phase mode (default when persistent TG enabled)
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -n 128

# DAG mode (disable phase, enable DAG)
GGML_SYCL_PERSISTENT_TG=1 GGML_SYCL_PERSISTENT_TG_PHASE=0 GGML_SYCL_PERSISTENT_TG_DAG=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -n 128

# Correctness check — must pass the Mistral completion gate.
# Output is "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" and STOPS at 10 (EOS).
# This line used to say "ends 6..15" -- unreachable at -n 15, so a passing
# gate reads as a failure and sends you hunting a nonexistent TG bug.
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

## Memory budget and pressure hierarchy

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_VRAM_BUDGET_PCT=N` | **100** | VRAM budget as % of total (triggers CPU offload when model exceeds). ⚠️ This row said **90** until 2026-07-30; that was correct when written but went stale at `9b0fe06aa` (2026-04-02, *"budget = actual free VRAM, default 100% — remove artificial headroom"*), which moved the two load-bearing defaults to 100 and removed the 256 MB/10% headroom deduction. 100% is **not** an unconditional whole-card guarantee: the budget is still capped to actual free VRAM at cache init, plus a structural runtime-slack reservation. Two further copies of this literal (`unified-cache.cpp:9770`, `:9926`) were missed by that commit and still read 90 — see `llama.cpp-ytr7`; one is log-only, the other a pre-init fallback. |
| `GGML_SYCL_KV_HOST=1` | OFF | Force KV cache to host pinned memory (Level 1 offload) |
| `GGML_SYCL_KV_HOT_LAYERS=N` | auto | Hot layer count for per-layer KV hot/cold tiering |
| `GGML_SYCL_KV_HOT_PCT=N` | auto | Hot window as % of total KV buffer |
| `GGML_SYCL_FORCE_STREAMING=1` | OFF | Enable GPU weight streaming (Level 5, last resort) |
| `GGML_SYCL_HOST_COMPUTE=1` | OFF | Use host-pinned compute buffers (eliminates staging for CPU-dispatched layers) |
| ~~`GGML_SYCL_PIPELINE_MOE=1`~~ | — | ⚠️ **DEAD — does nothing.** No code reads this name (comments only), and its gate `ggml_sycl_pipeline_moe_enabled()` (`ggml-sycl.cpp:14178`) is a hardcoded `return false;`. Added `ae5eae507`, getenv removed `3cf9b5fdf` (2026-05-09, listed under "Removed (hardcoded to defaults)"), then **this row was created `f3c36987f` two months after the removal** — it was never accurate. Setting it measures nothing; do not conclude multi-GPU MoE overlap "doesn't help" from it. `GGML_SYCL_PIPELINE_CPU` below **is** live. |
| `GGML_SYCL_PIPELINE_CPU=1` | OFF | Pipeline CPU expert compute with GPU attention across MoE layers: CPU experts from layer N run during layer N+1 attention |

## Cache and memory

| Variable | Effect |
|----------|--------|
| ~~`GGML_SYCL_UNIFIED_CACHE=0`~~ | **Removed by `9a0670712` with optional cache enablement and its enable/disable branches.** Setting this name no longer disables anything; there is no replacement opt-out. |
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Select cache topology (`auto`, `global`, or `per_device`) only; it cannot disable the cache. |
| `GGML_SYCL_NO_PINNED=1` | Disable pinned host memory |
| `GGML_SYCL_WEIGHTS_EVICTABLE=1` | Allow weight eviction under memory pressure |
| `GGML_SYCL_MEM_BUDGET=<MB>` | Set VRAM budget in MB |

For the architectural contract and migration history behind these two rows, see
§1.2 and §9.3 of `docs/design/sycl-canonical-memory-architecture.md`.

## Debugging

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_DEBUG=1` | Enable detailed kernel dispatch logging (MASSIVE output) |
| `GGML_SYCL_UNIFIED_DEBUG=1` | Debug unified kernel dispatch |
| `GGML_SYCL_NAN_CHECK=1` | Enable NaN detection in outputs |
| `GGML_SYCL_VALIDATE=1` | Enable A/B validation between kernel paths |
| `GGML_SYCL_SET_ROWS_VALIDATE=1` | Default OFF (`set_rows.cpp`). Reads the SET_ROWS index tensor back to the host and reports the entries the kernels will drop, as one `[SET_ROWS_VALIDATE]` line per op **only when at least one index is out of range** — silence means every index was in `[0, ne1)`. Reports, never aborts: an out-of-range index is a caller error the CPU backend catches with `GGML_ASSERT(i1 >= 0 && i1 < ne1)` (`ggml-cpu/ops.cpp:5073`), while SET_ROWS has no status channel on any backend, so the SYCL kernels merely *contain* it by dropping the element. This flag exists because that containment is otherwise silent. Costs a device→host copy plus a queue sync per SET_ROWS, and the op runs once per layer per ubatch on the KV write path — diagnostic only, never in production. **Skipped while a command graph is recording**, where a blocking copy is not legal, so pair it with `GGML_SYCL_DISABLE_GRAPH=1` to see the KV-cache writes that graph replay would otherwise hide. Read once and cached. |
| `GGML_SYCL_ADD_ID_VALIDATE=1` / `GGML_SYCL_ADD_ID_VALIDATE_LIMIT=<N>` | Default OFF (`add-id.cpp:13-21`). The ADD_ID sibling of the above and the pattern it was modelled on: reads src1/src2 back and prints an `[ADD_ID_VALIDATE]` line carrying `ids_min`/`ids_max`/`ids_oob` plus NaN/inf counts over the selected rows. Reports, never aborts. Three things to know before reading a capture, because each can make it look like nothing is wrong: it fires **twice per op** (`site=before` and `site=after`); it is **filtered to tensors whose name contains `ffn_moe_`**, so a non-MoE model prints nothing however bad the ids are; and it emits at most `GGML_SYCL_ADD_ID_VALIDATE_LIMIT` lines (default 32) before going quiet, which is a cap and not a clean bill of health. Unlike `GGML_SYCL_SET_ROWS_VALIDATE` it re-reads `getenv` per call and is **not** suppressed during graph recording. |
| `GGML_SYCL_MUL_MAT_ROUTE_TRACE=1` / `GGML_SYCL_MUL_MAT_ROUTE_TRACE_LIMIT=<N>` | Default OFF. One `[MUL-MAT-ROUTE] <stage> idx=… kernel=… layout=…` line per `MUL_MAT` stage, on raw `fprintf(stderr)` so it survives the `GGML_LOG_INFO` verbosity gate. **This is how you prove which kernel produced a number.** Stages: `entry` and `graph` (bookkeeping, no kernel named); `selected` / `dispatch-legacy` / `dispatched-legacy` and their `-failed` and `-retry-aos` siblings (orchestrator route); `dispatch-tg-fast` / `dispatched-tg-fast` / `-failed` and the `-split` variants (the batch=1 fast-path — added by `llama.cpp-erf1`, before which a traced fast-path run named no kernel at all). The `kernel=` token separates `DMMV_COALESCED` from `DMMV_SOA`, which `GGML_SYCL_DMMV_Q8_DEBUG=1`'s `[DMMV] dispatch` line cannot. `…_LIMIT` caps traced ops at N (default **256**) — a cap, not a clean bill of health. Both are read once and latched. |
| `GGML_SYCL_DMMV_Q8_DEBUG=1` | Default OFF (`dmmv.cpp`). Emits `[DMMV] dispatch: … src0_type=… layout=… src1_q8=…` when a DMMV kernel runs, so its **absence** proves DMMV did not run. It does not say *which* DMMV kernel — pair with the route trace above for that. |
| `GGML_SYCL_GRAPH_RERECORD=1` | Use graph re-record instead of replay (very slow, diagnostic only) |
| `GGML_SYCL_OP_TIMEOUT_MS=<N>` | Abort with diagnostic if no inference progress for N ms (default 30000, set to 0 to disable). Fires before the xe driver's 10s GT reset cascade. Effective detection latency is `timeout + ~500 ms`. |
| `GGML_SYCL_SAFE_MODE=1` | Drain the SYCL queue after every op submit so a fault surfaces at the op that caused it (2-3x slowdown, implies `GGML_SYCL_DISABLE_GRAPH=1`). Useful for CI canaries and correlating intermittent hangs 1:1 with their triggering op. |
| `GGML_SYCL_FUSION_BIT1_REACH_DEBUG=1` | Default OFF (`ggml-sycl.cpp:83907`). Logs one `[FUSION-BIT1-GUARD]` line per bit1 fusion candidate: `is_weight`, the accumulated `view_offs`, `nb[0]`, element size, the per-operand `safe` verdict, and the full gate chain. Built for the gemma3n cross-model investigation (`llama.cpp-8t4s`), and still the way to answer "did the 81gx view-offset guard classify this operand correctly?" — in the healthy 2-arch run at HEAD, all 44 of gemma3n's unsafe views log `is_weight=0` and are refused. Diagnostic only; the block is guarded by the flag. |
| `GGML_SYCL_HANDLE_STRICT=1` | Default OFF; reports `ggml_tensor_extra_gpu` `data_handle`/`data_device` divergence (first 16 only) without needing `GGML_SYCL_DEBUG=1`. Diagnostic only, no perf effect. Plan: `docs/plans/2026-07-30-extra-device-indexed-handle-storage.md`. |
| `GGML_SYCL_MOE_LAYOUT_DEBUG=1` | Emit the `[MOE-LAYOUT]` per-pass summary unconditionally. The down-i8 / gateup-i8 lines already fire on ANY decline without this; the variable adds the lines a fully-successful pass would otherwise not print. |
| `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS=<N>` | Hard cap on how many down tensors the MoE I8 layout pass upgrades. Unset (or negative) = no cap, the shipping behaviour; `0` disables the upgrade. **Diagnostic only — do not set in production.** See the measured cost below. |
| `GGML_SYCL_ARENA_PP_PROFILE=1` | Emit `[ARENA-PP-*]` counters, including `[ARENA-PP-ONEDNN] … reserve_req_mb=W/A` — the summed oneDNN weights/activations reservation requests. This is the **only** log that reports what was actually asked for, as opposed to what was planned. |
| `GGML_SYCL_ZONE_RESET_AUDIT=1\|2` | Default OFF. Phase 0 of the retire-zone-reset epic (llama.cpp-iiff): every reset/drain site reports what is still live in its zone as `[ZONE-RESET-AUDIT]` lines, with the allocation's own `alloc_id`/`cohort`/`role`/`category` attribution. `=1` changes no behaviour and is safe across the whole gate set. `=2` additionally suppresses the reset even when the zone is clean — host SCRATCH/STAGING are reset-only by design, so that leaks without bound. See below. ⚠️ **The variable name and its site strings (`host-zone-reset/*`, `scratch-pool-reset/bump`, `device-zone-reset/*`, `weight-reclaim/*`) were deliberately NOT renamed when `llama.cpp-37ba` renamed the underlying API to `*_boundary_check` / `*_reclaim` / `scratch_pool_epoch_boundary`.** They are stable historical identifiers that keep captures 01–17 baseline-comparable. They look stale against the current code; they are not. Do not "fix" them. |
| `GGML_SYCL_EXT_ALLOC_TRACE=1` | Default OFF (`unified-cache.cpp`). Logs one `[EXT-ALLOC] device=… bytes=… cohort=… role=… category=… prefer_vram_zone=… total_external=…` line whenever a device allocation escapes the arena's per-zone accounting — a `prefer_vram_zone==COUNT` request, or a full/inactive zone falling through to the raw `unified_cache_malloc_device_tracked` device malloc. `total_external` is a running byte total (process-lifetime, function-local `static`), so a capture doubles as a drain profile of what is bypassing the zone system. `role`/`category` are the raw `alloc_role`/`runtime_category` enum ordinals (see `unified-cache.hpp`); `cohort` is `?` when the caller left `alloc_intent::cohort_id` unset. Read once and cached; zero cost when unset beyond the one atomic load. Diagnostic only. |

### `GGML_SYCL_ZONE_RESET_AUDIT` — reading a capture

Four sites report: `device-zone-reset/<zone>:devN`, `host-zone-reset/<zone>`,
`scratch-pool-reset/bump:devN`, and `weight-reclaim/<mode>:devN` (the mode being
`load-boundary`, `mid-load-replan` or `model-teardown`).

Read `visits_with_live` against `visits` first. They exist because an **empty
capture is not evidence of zero escapes** — it is equally consistent with the
site never having been reached. `visits=0` for a site means it does not appear at
all; a run that reached no site at all says so outright
(`NO RESET SITE WAS VISITED … this run proves NOTHING`).

A full inventory is re-emitted every 256 site visits, at process exit, from the
SYCL watchdog before its `_Exit(1)`, and from a `SIGSEGV`/`SIGABRT` handler — the
runs with the confirmed escapes are the runs that crash. The signal handler
chains to whatever handler it displaced (the planner canaries install their own),
falling back to `SIG_DFL` only when there was none.

#### Every line is printed twice — do not `grep -c`

Output goes to WARN **plus** a raw `stderr` copy. The raw copy exists because
`GGML_LOG_INFO` is dropped at default verbosity in every tool, so an INFO-level
audit line would produce an empty capture indistinguishable from a clean one —
but WARN is *not* dropped, so in the normal case **both copies survive and every
count is doubled**.

| regime | factor | how to dedup |
|---|---|---|
| any tool calling `common_init()` — `llama-cli`, `llama-completion`, `test-thread-safety`, `test-llama-archs` (**all four captures in the protocol**) | x2 | `common_init()` enables the log prefix, so the copies differ: the WARN copy is preceded by a timestamp and a `W ` marker, the raw copy starts at column 0. **`grep '^\[ZONE-RESET-AUDIT\]'` selects exactly the raw copy.** |
| `llama-bench` without `-v` (null log callback) | x1 | nothing to do — only the raw copy exists |
| a binary left on ggml's default log sink (itself a bare `fputs`) | x2 | byte-identical copies, **no discriminator — expect x2 and halve** |

Two practical consequences:

- Confirm the factor empirically before counting anything:
  `grep -c 'ZONE-RESET-AUDIT] ==== end inventory'` against the number of reports
  you expect.
- For a true escape count use `grep 'NEW-ESCAPE' <capture> | sort -u | wc -l` —
  those lines are unique per `(site, cohort, size)`, so `sort -u` collapses the
  duplication without discarding real repeats.

The two writes are separate calls and `common_log` is asynchronous, so their
order is not guaranteed and they can interleave with other output. This is the
same hazard that corrupts `test-llama-archs`' results table; prefer the prose
lines over anything column-aligned.

#### `role` / `category` / `tier` mean different things at the weight-reclaim site

At the three zone sites those columns are read straight off the allocation's
`alloc_handle` and mean what they say. A weight **cache entry** has no such
handle, so `weight-reclaim/*` rows substitute unrelated fields into the same
column names:

| column | zone sites | `weight-reclaim/*` rows |
|---|---|---|
| `alloc_id` | allocation id | weight `name_hash` |
| `role` | `alloc_role` | `cache_layout` |
| `category` | `runtime_category` | **lease count** |
| `tier` | `alloc_tier` | `cache_location` |

This is kept for parity with c-jec1's published inventory, which was read this
way — its "`role=4` = the MoE expert weights" decoding is a `cache_layout`, not
an `alloc_role` — and matching that format is what lets the new capture diff
against the old one. Decoding the columns to names is tracked as
`llama.cpp-u7vi`.

**Read the site name before reading these columns, and never compare them across
sites.**

#### `weight-reclaim/*` cohort names

`unified-cache.cpp`'s `reclaim_weight_entries()` labels every preserved entry
with one of four cohorts, computed from `live` (lease count, i.e. `category`
above) and `owned_by_live` (owner-mask overlap with the current live-model
mask):

| cohort | condition | meaning |
|---|---|---|
| `weight:leaked_lease` | `live != 0`, no live owner, entry was tagged | a real leak — `entries_leaked` counts it, `GGML_SYCL_STRICT_LEASES=1` aborts on it |
| `weight:leased` | `live != 0`, owned by a live model OR never tagged | benign — either correct concurrent ownership, or a lease on an entry the code never claimed to attribute |
| `weight:owned_by_live_model` | `live == 0`, owned by a live model | correct: another live model's idle weight |
| `weight:unattributed` | `live == 0`, not owned by a live model | preserved because it was never tagged (`!entry.owner_tagged`) — see below, never a leak |

`GGML_SYCL_STRICT_LEASES=1` can abort **only** on `weight:leaked_lease` —
`entries_leaked` increments exclusively inside the `live != 0` branch of
`reclaim_weight_entries()`, so `weight:unattributed` (`live == 0` by
definition) can never reach it. This was adjudicated from source in
`llama.cpp-zjz6` after the Phase-0 recapture reclassified GPT-OSS's 1536
MoE-expert `weight-reclaim/model-teardown` entries from `weight:leased`
(pre-`2wv5`) to `weight:unattributed` (post-`2wv5`): the change is that their
lease dropped to zero earlier, not that their ownership tag changed — they
were untagged in both captures. They stay untagged because they are
materialized by `ggml_sycl_materialize_moe_tensor_phase_layout()`'s bulk-XMX
branch (role/`cache_layout` 4 = `GGML_LAYOUT_XMX_TILED`), reached from the
graph-compute path's `GGML_OP_MUL_MAT_ID` scan — a runtime path with no load
transaction bound, so the load-transaction-scoped `stamp_pending_owner()` /
`note_model_load_end()` tagging never reaches them. (`moe_prestage_popular_experts()`
is a different function; it stages only `GGML_LAYOUT_SOA`/`GGML_LAYOUT_AOS`
and merely *pins* an already-existing `XMX_TILED` entry, never creates one.)
`MODEL_TEARDOWN` preserves untagged entries unconditionally, so they are not
reclaimed until the next `MID_LOAD_REPLAN` (immediately, regardless of
`live_mask` — untagged means `owner_mask == 0`, which never overlaps
`live_mask`) or `LOAD_BOUNDARY` (only once `live_mask == 0`). See
`docs/design/sycl-canonical-memory-architecture.md` §1.2 for the full chain
and the practical consequence (delayed, not denied, reclaim).

The attribution is read from the allocation, never re-derived. It became
trustworthy only with llama.cpp-f9tg (`85eb63dcb` / `810ae7fef`); captures taken
before that fix mislabel COMPUTE/CONTROL allocations as GRAPH.

The audit frees nothing and resets nothing at `=1`. Everything it reports stays
owned by whoever already holds its handle.

### `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS` — what it costs to cap

Measured on GPT-OSS 20B MXFP4, B50, `-p 512 -n 128 -fa 1`, 6 interleaved rounds
per level (`docs/plans/2026-07-25-moe-down-i8-dose-response-findings.md`):

| granted | pp512 | tg128 |
|--------:|------:|------:|
| 0 | 605.51 | 34.14 |
| 2 | 913.50 | 35.13 |
| 5 | 907.13 | 35.56 |

**Setting this to 0 costs 33.7% of pp512.** The layout upgrade is load-bearing
for the ~894 pp512 baseline, not a marginal tuning knob. Beyond 2 layers the
trade is small and roughly symmetric: −0.70% pp512 for +1.24% tg128 going 2→5.
The variable exists to isolate the layout from the VRAM reclaim that pays for
it; `GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB` cannot do that job because it
moves the arena, the zones and the resident set at the same time.
