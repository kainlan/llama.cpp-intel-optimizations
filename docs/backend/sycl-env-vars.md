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
| `GGML_SYCL_ONEDNN_CACHE_ALLOCATOR=0` | ON | Disable the `dnnl::graph::allocator` that routes the compiled SDPA partition's per-execute scratch through the unified cache's ONEDNN VRAM zone instead of oneDNN's default SYCL allocator (llama.cpp-gwno, S4 root cause `jmc5` `c-uxch`: ~12 MiB `zeMemAllocDevice`/`xe_vm_bind` round trip, 35x/ubatch on the B70 pp512, ~2.7 ms host block each). `=0` restores the pre-fix `dnnl::sycl_interop::make_engine()` (no allocator) for an A/B; also falls back automatically, with a `GGML_LOG_WARN`, if `make_engine_with_allocator()` itself throws — unlatched, so it fires once per queue per backend context (engines are cached per queue in `engine_map`), not once per process. See `docs/backend/sycl-memory-design.md`'s "oneDNN Graph allocations are a cache consumer too" section. |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (skip unified kernel) |
| `GGML_SYCL_XMX_TILED_PP=0` | ON | Disable the XMX_TILED grouped-DPAS PP route for MXFP4 gate/up (falls back to SOA at PP). Productized 2026-08-17 (llama.cpp-rzy7) from the verified diagnostic state (llama.cpp-e3xj): pp512 461±16 B70 / 149.9±0.9 B50, gates pass, TG neutral-positive, zero UR errors. The older `GGML_SYCL_XMX_MOE_ALLOW_UNSAFE_PP=0` / `GGML_SYCL_XMX_MOE_PP=0` names are still honored as a compatibility opt-out when this variable is unset — despite the "unsafe" name, neither is a diagnostic-only knob anymore. Does not affect the separate `GGML_SYCL_XMX_MOE` prompt-XMX-forced diagnostic path. |
| `GGML_SYCL_MOE_DOWN_XMX_TILED=0` | ON | Disable the XMX_TILED grouped-DPAS route for the MoE **down** projection (falls back to SOA/MXFP4_I8 planning, unchanged). **Semantics changed 2026-08-20 (llama.cpp-sk67): default flipped OFF→ON.** Before sk67 the flag existed but the down dispatcher (`mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4`, `mmvq.cpp`) rejected `GGML_LAYOUT_XMX_TILED` outright at its entry guard regardless of this flag, so `=1` was a no-op — OFF+non-functional. sk67 added a grouped-DPAS arm reusing the existing single-matrix kernel `mxfp4_xmx_tiled_grouped_direct_q8_sycl` (the same kernel already dispatched for gate/up XMX_TILED weights) against the down role's cached Q8 activation artifact, and widened the planner/admission gates (`ggml_sycl_moe_prompt_xmx_tiled_supported`, `ggml_sycl_select_moe_planned_graph_layout`, `ggml_sycl_moe_prompt_down_specialized_layout_proven`, the PP transactional executor's `executor_layouts_ok`/`abi_ok` admission, and the decode fusion path's `down_layout_table_eligible` query) to actually select and reach it — now ON+functional. Expected effect: on B50, all 24 MoE layers reach the grouped-DPAS route for down instead of the slow SOA path (was 92% of B50 MoE device time per the `llama.cpp-sk67` mission). I8 stays preferred wherever the planner already materializes/plans it (e.g. B70 where it fits); this flag only extends coverage to layers that would otherwise fall back to SOA. Also gates part of the decode-phase fused dispatch (`ggml-sycl.cpp`'s `decode_pair_glu_dispatched` block) via `ggml_sycl_moe_decode_xmx_tiled_supported`'s DOWN branch — see `llama.cpp-y0it` for the companion fix (a refused down dispatch used to silently mark itself "handled" and drop the output; now the flag/return there is conditional on the dispatch actually succeeding). A separate, narrower flag `GGML_SYCL_MOE_PHASE_DOWN_XMX` (default OFF, unchanged by sk67) gates a different phase-materialization decision point (`ggml_sycl_moe_phase_target_layout`) not touched by this ticket. |
| `GGML_SYCL_Q8_DENSE_AOS=1` | OFF | Route Q8_0 dense projections (ATTENTION_WEIGHT/FFN_WEIGHT/OUTPUT_WEIGHT — `layout_policy::get_optimal` in `common.hpp`) AOS instead of COALESCED so PP batches reach `ONEDNN_AOS` (the oneDNN jit:gemm arm; COALESCED batches land on `MMQ_COALESCED` ~3 TFLOPS since that case has no oneDNN arm). MEASURED 2026-08-20 (llama.cpp-e3xj C25/C26): +38% B70 GPT-OSS pp512 (847→1160) but **−34% B50 tg128 (32.1→21.3)** because TG moves from the coalesced kernel to MMVQ_AOS — it FAILS the single-layout all-consumers rule, so the default is OFF. Enable only for PP-dominated workloads that accept the TG loss. Superseded for the PP win by `GGML_SYCL_Q8_ONEDNN_COALESCED` below, which gets the oneDNN arm without moving TG off the coalesced kernel. Only affects Q8_0; MoE expert weights are untouched by this flag. |
| `GGML_SYCL_Q8_ONEDNN_COALESCED=0` | ON | The layout-neutral recapture for the `GGML_SYCL_Q8_DENSE_AOS` tradeoff above (llama.cpp-e3xj): adds an `ONEDNN_COALESCED` kernel choice, selected for Q8_0 dense weights in COALESCED layout at batch > `MMVQ_MAX_BATCH_SIZE` (8). Dequants the COALESCED weight straight to FP16 (`dequantize_row_q8_0_coalesced_to_fp16_rowmajor`, reusing the same tile/offset math as the live `DMMV_COALESCED`/`MMVQ_COALESCED` Q8_0 kernels) and routes through the same oneDNN `row_gemm` the `ONEDNN_AOS` arm uses — no AOS materialization, so TG stays on the coalesced kernel (unlike `GGML_SYCL_Q8_DENSE_AOS`). If the COALESCED weight isn't device/shared-resident when the op runs, falls straight through to `MMQ_COALESCED` (the pre-existing behavior); never errors. Set `=0` to force the pre-existing `MMQ_COALESCED` PP path for an A/B. |
| `GGML_SYCL_Q8_DENSE_LAYOUT=soa` | coalesced\* | **Opt-in force (llama.cpp-nz1k, prefill L2b phase 1).** Materialize EVERY Q8_0 dense projection (ATTENTION_WEIGHT/FFN_WEIGHT/OUTPUT_WEIGHT — `layout_policy::get_optimal` in `common.hpp`) SOA instead of COALESCED, regardless of shape. Only the exact value `soa` selects SOA; anything else is the default coalesced *resolution from `get_optimal`*. \*As of llama.cpp-pktr this is no longer the whole story: `ggml_sycl_adjust_layout_for_tensor` (`ggml-sycl.cpp`) and `planner_default_device_layout` (`unified-cache.cpp`) now apply an automatic PER-WEIGHT rule on top of `get_optimal`'s coalesced default — a Q8_0 dense row whose `blocks_per_row` (`ne00`/32) is NOT a multiple of `MMVQ_COALESCED_TILE_BLOCKS` (32) resolves SOA even with this env unset, because the coalesced kernel pays for a padded tail tile on that shape every token (measured: gemma4-E4B's K=2560 gate/up/q/k/v/o, 80 blocks/row, up to 14 points of B50/B70 bandwidth; tile-aligned shapes — Mistral K=4096/14336, gemma4 K=10240/2048 — are unaffected and stay coalesced). This env still means what it always meant: force ALL dense projections SOA, including tile-aligned ones the automatic rule would otherwise leave coalesced. One layout serves every consumer (ruling 5): TG runs the existing `DMMV_SOA`/`MMVQ_SOA` `mulmat.mmvq.q8_0_soa` kernels, PP lands on `MMQ_SOA` unless `GGML_SYCL_Q8_ONEDNN_SOA` (two rows below, default ON since pktr) routes it to the oneDNN SOA arm. EMBEDDING is untouched by this env (the automatic per-weight rule does apply to EMBEDDING/OUTPUT_WEIGHT — that is the pre-existing llama.cpp-os8k net this task extended, not new in pktr). Precedence: `GGML_SYCL_Q8_DENSE_AOS=1` is checked first and wins over this variable, and this variable's `soa` (when set) is itself resolved by `get_optimal` before `ggml_sycl_adjust_layout_for_tensor` ever runs, so the automatic per-weight rule cannot re-promote it back to coalesced. |
| `GGML_SYCL_ONEDNN_WOQ_Q8=1` | OFF | **Opt-in (llama.cpp-nz1k, prefill L2b phase 1); scope narrowed by llama.cpp-pktr — read the next row first.** Gates ONLY the int8-plane WoQ-int8 EXECUTE path inside `ggml_sycl_op_mul_mat_sycl` (`ggml-sycl.cpp`): the prepare/stage/GEMM sequence through `DnnlGemmWrapper::woq_gemm_q8_0` (`gemm.hpp`), running probe llama.cpp-ovkn's V4 form — **f16 activations × the stored int8 qs plane read as-is** (logical (K,N) strides {1,K}), K/32 grouped f16 scales (mask (1<<0)\|(1<<1), groups {32,1}), `fpmath f16 apply_to_int` — measured equal to f16×f16 speed on the B70 and +25% on the B50, with **no per-call weight dequant and half the weight bytes**. The only per-call staging is the scale-plane transpose `[N][K/32] → [K/32][N]` (`q8_0_soa_scale_plane_to_kbn_sycl`, `convert.cpp`; 2 bytes per 32 weights, 1/32 of the 64 bytes per 32 weights the f16 dequant wrote; it uses at most 1/32 of the dequant-sized PP scratch reservation the caller already holds) into the existing oneDNN PP weights scratch (`acquire_onednn_pp_scratch`, ruling 1 — no new allocator). The SOA d-plane order bound directly is accepted-but-garbage (probe V5) and is never passed. Declines — never errors — into the SOA f16 dequant + `row_gemm` when rows are partial, K is not a multiple of 32, the SOA plane is not device-resident, **this env is off** (new decline reason `woq_disabled`, llama.cpp-pktr), the PP scratch is absent, or oneDNN refuses the primitive/throws (the primitive is checked before the scale staging is submitted, so a declining shape pays no transpose); `GGML_SYCL_MUL_MAT_ROUTE_TRACE=1` prints `[MUL-MAT-ROUTE] onednn_woq_q8 ... arm=woq_q8\|dequant_f16 decline=<reason>` per dispatch. Kernel-profiler rows: `mulmat.onednn_woq_q8.execute` (the GEMM) and `mulmat.onednn_woq_q8.stage_scales`. Phase 1 is an opt-in interim only (per-call staging is not design-compliant as a default); phase 2 (llama.cpp-2zsc) stores the d plane in `[K/32][N]` order, retires the transpose, and owns the default decision. **llama.cpp-pktr changed what "off" means:** before pktr, this env ALSO controlled whether the `ONEDNN_SOA` kernel choice was reachable at all — with it off, an SOA-planned Q8_0 weight's PP batches fell all the way to `MMQ_SOA` (~3 TFLOPS). Now `ONEDNN_SOA` kernel-choice eligibility is the separate `GGML_SYCL_Q8_ONEDNN_SOA` (default ON, next row); with THIS env off but that one on (the new default combination), an SOA-planned weight still reaches oneDNN PP through the SOA-aware f16 dequant + `row_gemm` fallback in this same function — the SOA-plane lookup (`q8_0_soa_ptr`) now runs unconditionally of this env so that fallback can address the plane correctly. Source-gated by `tests/test-sycl-onednn-woq-q8-source.py`. **MEASURED 2026-09-03 (510de3676, one binary, 3 interleaved pairs per card, Mistral 7B Q8_0 `-p 512 -n 128 -r 2`):** B50 pp512 1145.8 → **2039.5 (+78%)**, tg128 27.23 → 26.78 (−1.7%, real: run spread 0.1%); B70 pp512 2403.8 → **5241.4 (+118%)**, tg128 67.95 → 66.30 (−2.4%). Profiler: `mulmat.onednn_woq_q8.execute` at 119–135 TOPS on the three large Mistral shapes (probe predicted 124–127), scale staging = 10% of GEMM time. All correctness gates green on both cards with the arm engaged on 221/221 PP dispatches. Those figures predate the nz1k review rounds (bf2988282..e411bc505 changed the arm's entry condition and split the GEMM wrapper); **re-verified at e411bc505 (fresh binaries, all nine gates green, arm 221/221, one interleaved anchor pair per card):** B50 pp512 1124.7 → 2005.7 (+78%), tg128 26.68 → 26.26 (−1.6%); B70 pp512 2304.3 → 4865.9 (+111%), tg128 65.31 → 63.26 (−3.1%) — same ratios, absolutes a few percent lower in BOTH arms because the codescout embedder was re-indexing on the CPU during the run. The B50 decode cost is why the default int8-plane execute stays opt-in: the flip is decided on llama.cpp-2zsc. This row's PP measurements predate llama.cpp-pktr's eligibility-gate split and were taken with the old combined gate (this env alone controlled both eligibility and execute); re-measure with `GGML_SYCL_Q8_ONEDNN_SOA` at its pktr default (ON) before citing as current for the SOA-aware f16 dequant fallback's own numbers. |
| `GGML_SYCL_Q8_ONEDNN_SOA=0` | ON | **New in llama.cpp-pktr.** The layout-neutral eligibility gate for the `ONEDNN_SOA` kernel choice (`pick_kernel_for_layout` / `k_mul_mat_priority`, the override-check switch, and `ggml_sycl_select_preferred_kernel`, all `ggml-sycl.cpp`), mirroring `GGML_SYCL_Q8_ONEDNN_COALESCED` above but for the SOA side: selected for any Q8_0 weight resident in SOA layout (pktr demotions, the `GGML_SYCL_Q8_DENSE_LAYOUT=soa` opt-in set, the os8k tied head) at batch > `MMVQ_MAX_BATCH_SIZE` (8) — not scoped to dense weights only. Distinct from `GGML_SYCL_ONEDNN_WOQ_Q8` (previous row) — this only decides whether `ONEDNN_SOA` may be chosen at all, not whether the int8-plane WoQ primitive executes once it is. Exists because the pktr per-weight layout rule (see `GGML_SYCL_Q8_DENSE_LAYOUT`'s row) now materializes tile-misaligned dense Q8_0 rows SOA **at default env** (e.g. gemma4-E4B's K=2560 head), and those PP batches need a route to oneDNN GEMM parity with `ONEDNN_COALESCED` — without this gate they would land on `MMQ_SOA` (~3 TFLOPS) and trade prefill down relative to the pre-pktr all-coalesced baseline. With `GGML_SYCL_ONEDNN_WOQ_Q8` at its own default (OFF), a selected `ONEDNN_SOA` batch executes through `ggml_sycl_op_mul_mat_sycl`'s SOA-aware f16 dequant + `row_gemm` fallback — the same oneDNN f16 GEMM `ONEDNN_COALESCED` uses, just reading the SOA plane instead of the coalesced one — so default-env PP throughput for a pktr-demoted-to-SOA weight is intended to match a tile-aligned COALESCED weight's PP, not `MMQ_SOA`'s. Set `=0` to opt out and keep `MMQ_SOA` for every SOA Q8_0 batch, as before pktr. **MEASURED 2026-09-04 (2c2570fe3, B70, GPT-OSS `-p 512 -n 128`, 3 interleaved pairs):** pp512 1772/1769/1774 (default ON) vs 1344/1340/1343 (`=0`) — **+32%**; tg128 39.0/39.7/39.7 vs 40.0/39.9/39.2 — unchanged within run spread. `GGML_SYCL_Q8_DENSE_LAYOUT=soa` users therefore get oneDNN PP under this gate by default. Source-gated by `tests/test-sycl-onednn-woq-q8-source.py` and `tests/test-sycl-q8-dense-layout-rule-source.py`. |
| `GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=0` | ON | **Route policy**, not just an executor switch (llama.cpp-dboi, first-class integration 2026-08-21; layout scope corrected llama.cpp-1tjn, Option T, same day; **flipped to default-ON on both cards llama.cpp-iikr, 2026-08-25, owner ruling c-mnd7** — `=0` opts out; see the closing sentence of this row for the certification that drove the flip). One predicate — `ggml_sycl_moe_pp_onednn_batched_route_selected()` in `common.hpp`, shared by dispatch and the unified-cache planner — flips these together: (i) the MXFP4 MoE layout chokepoint (`ggml_sycl_select_mxfp4_moe_layout`) is a single per-weight, planning-time decision with no PP/decode caller identity ("one layout per weight" — no duplicates, no dispatch-time layout shifts). **Gate/up are NOT touched by this policy** — they keep the default XMX_TILED materialization unconditionally, so decode's XMX_TILED DPAS kernel (`used_xmx_tiled_dpas`, `mmvq.cpp`) is unaffected. **Down alone still plans SOA under the policy** (the down-i8 planning upgrade stays disabled there — a deliberate, separate tradeoff: down's decode delta is ~1 ms/token, its PP-batched win is banked, tracked apart from gate/up); (ii) at PP the fused pair-GLU executor declines, the per-row MMVQ probes yield MXFP4 gate/up/down, and prompt routing skips the hybrid per-entry world so dispatch reaches the batched executor; (iii) the executor itself (`try_pp_mxfp4_soa_onednn_f16_batched`, `ggml-sycl.cpp`) runs. **Updated 2026-08-22 (llama.cpp-sr83, track C3):** `ggml_sycl_moe_pp_onednn_batched_claims_tensor` now claims gate/up too (XMX_TILED, planned layout) alongside down (SOA) — the interim "gate/up fall back to a slower PP path pending track C" window this row described is closed. By default (`GGML_SYCL_MOE_PP_WOQ` unset/1, see its own row below) the executor repacks 4-bit-to-4-bit straight into a `{nibbles,e8m0-scales}` WOQ scratch shape (`repack_mxfp4_soa_to_woq` for down, `repack_mxfp4_xmx_tiled_to_woq` for gate/up, both `convert.cpp`) and hands them to a batched oneDNN WOQ matmul (`DnnlGemmWrapper::woq_gemm_batch_mxfp4`, `gemm.hpp`) — the f16 dequant is deleted on this arm entirely, ~4x smaller scratch. Setting `GGML_SYCL_MOE_PP_WOQ=0` restores the pre-C3 per-expert f16 dequant + `gemm_batch_strided` arm for A/B or as a fallback if the WOQ primitive ever regresses; experts are still sorted by row count and split into grouped strided oneDNN batches (group `n` padded to 64-multiples for primitive reuse) to avoid ~5x rectangular padding on GPT-OSS's skewed routing, unchanged by the WOQ/f16 choice. **Fail-closed contract:** once an XMX_TILED-claimed op reaches this executor, every internal decline point throws (`ggml_sycl_fallback_error`) instead of returning false, because the outer dispatch has already skipped every other route for it and the remaining per-expert staging fallback cannot decode XMX_TILED weights (llama.cpp-71hx); SOA-claimed (down) declines keep the original fall-through. Decode (`ne12==1`) is untouched by dispatch, and gate/up decode runs its normal tiled/DPAS routes unchanged regardless of this policy; only down's decode path sees the policy's SOA choice. A SOA-native direct-XMX/DPAS kernel for gate/up decode (`used_direct_xmx`, wired via `moe_layer_direct_xmx_check_role`, env-gated by `GGML_SYCL_MOE_DIRECT_XMX_GLU`/`GGML_SYCL_DIRECT_XMX_MOE_PROBE`, off by default) was measured llama.cpp-1tjn 2026-08-21 as a **2x regression** (25.8 vs 12.26 ms/token) — do not re-enable it as a decode speedup; it is documentation of a measured dead end, not a viable route. Scratch ring (`moe-scratch-admission.hpp` + `llama_model_sycl_populate_inventory`) is sized for all `n_expert` experts, admission fail-closed since `f2bdfbffe`; `reserve_pp_moe_onednn_scratch` satisfies from an existing adequate ring without reallocating -- sizing itself now branches on `GGML_SYCL_MOE_PP_WOQ` read at plan time (~130 MB weight-slot total for GPT-OSS 20B under the default WOQ arm, vs ~506 MB weight-slot total under the f16 opt-out arm — like-for-like, both weight-slot-only figures, matching the `llama_model_sycl_populate_inventory` comment in `llama-model.cpp`; see the `GGML_SYCL_MOE_PP_WOQ` row). Measured clean-host 2026-08-21 (r=2, PRE-Option-T, gate/up still SOA-planned at measurement time): B50 pp512 **569** / tg128 25.4 vs default 122/32.2; B70 pp512 **922** / tg128 37.1 (B70 default is broken on a clean device — llama.cpp-o3h1). These PP numbers predate both Option T's gate/up admission and C3's WOQ arm; re-measure before citing as current. **Default flipped to ON on both cards (llama.cpp-iikr, 2026-08-25, owner ruling c-mnd7)** — the per-card pp/tg-tradeoff ruling this row used to say was pending resolved in favor of the batched arm on BOTH cards, not a per-card split: certified clean-host (o3h1 chain complete, `4f3b70d75`, zero co-resident tenants) at B50 pp512=871.43±15.59 tg128=33.68±0.16 (best-ever tg) and B70 pp512=1730.68±31.40 tg128=40.85±1.15. The comparison baselines this ruling superseded were themselves invalid: on a genuinely clean host the *default* (non-batched) arm collapses to ~130 pp512 (B50) / ~515 (B70) — the old ~894/~1415 default-arm baselines were measured with codescout's embedder secretly resident on the B50, the same masking-bug class llama.cpp-o3h1 fixed one layer down, recurring here one layer up. That default-route collapse is filed separately as llama.cpp-9klr (P2) and is NOT fixed by this flip — see `docs/backend/sycl-perf-baselines.md`'s note on why its old default-route rows are known-stale and must not be gated against. `_TRACE=1` and `_COMPARE=1` (+`_COMPARE_LIMIT`) harnesses built in. |
| `GGML_SYCL_MOE_PP_WOQ=0` | ON | Opt out of the WOQ (weight-only-quantized) arm of the batched PP MoE oneDNN executor above (llama.cpp-sr83, track C3) — only meaningful when `GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED` has selected the batched route at all, which it now does by default on both cards (llama.cpp-iikr flip, 2026-08-25) rather than the per-card "capability-derived default" this row used to anticipate -- the ruling landed as one flat default, not a device-identity-threaded one. `=0` restores the pre-C3 per-expert f16 dequant (`dequantize_row_mxfp4_soa_to_fp16_rowmajor`) + `gemm_batch_strided` GEMM. Read at TWO points that must stay in sync: `ggml-sycl.cpp`'s executor lambda (per-dispatch) and `llama-model.cpp`'s `llama_model_sycl_populate_inventory` (plan-time scratch-slot sizing) — the plan-time read decides which arm's byte formula sizes the ring, and an under-sized ring makes admission fail-closed refuse every dispatch (`f2bdfbffe`), so flipping this must happen before the model loads, not mid-run. **The f16 arm is SOA-only** (llama.cpp-sr83, defect 1 found 2026-08-22, fixed at the claim rather than downstream): `dequantize_row_mxfp4_soa_to_fp16_rowmajor` has no tiled-aware counterpart, so feeding it tiled bytes previously poisoned every output entry to NaN (the OCP-MX reserved exponent `0xFF` reads as a NaN-equivalent). The fix is upstream of the executor: `ggml_sycl_moe_pp_onednn_batched_claims_tensor`'s XMX_TILED branch is itself gated on this same env, so with `=0` an XMX_TILED gate/up op is simply **never claimed** — no throw, no fail-closed hit, it falls straight through to whatever route handled gate/up before C3 existed. `=0` therefore only routes down through this executor at PP; gate/up run their pre-C3 path unchanged. |
| `GGML_SYCL_MOE_PP_WOQ_3D=1` | OFF | Opts IN to the WOQ arm's batch-dim-grouped 3-D scale mask (mask 7/group_dims `{1,group_size,1}`) instead of the default 2-D per-batch-element GEMM loop (`DnnlGemmWrapper::woq_gemm_batch_mxfp4`, `gemm.hpp` — C1's exact proven non-batched recipe, mask 3/group_dims `{group_size,1}`, looped once per expert). **Default meaning inverted 2026-08-22 (llama.cpp-sr83 fix cycle 2) on hardware-proven grounds, not just as an A/B knob:** the variable's own spelling and OFF/ON labeling are unchanged, but what "unset" selects flipped -- unset previously meant "attempt the 3-D primitive first, fall back to 2-D only on refusal"; unset now means "use the 2-D loop directly, no 3-D attempt", and only `=1` opts into 3-D. The 3-D primitive is *accepted* by oneDNN (no refusal in the log) but reproducibly (B70, two runs, byte-identical) produces wrong numerics on the down role — blk.0 gpu=-5.21 vs cpu=-484.47 (rel_at_max_abs 0.989), NaN cascade by blk.1 — while the 2-D loop on the *same* hardware is clean across every role and block (rel_at_max_abs <=0.0258, zero NaNs). Accepted is not the same as correct (the same trap as `ext_oneapi_enable_peer_access`, CLAUDE.md P2P section). **The 3-D encoding is known-broken; do not re-enable it in production.** It remains opt-in only so a follow-up task (tracked separately from C3; candidate fix direction: mask 3/group_dims `{group_size,1}` with the batch dimension carried implicitly on the scale memory's own leading dim, rather than an explicit mask bit + group_dims entry for it) can iterate without needing the 2-D path recompiled out. Setting this env does not touch the per-device capability cache (`tri_state` in `gemm.hpp`), so it is safe to flip between runs without recompiling. `GGML_SYCL_DEBUG=1` prints `[ONEDNN][WOQ-MXFP4-BATCH] device=%d arm=3D\|2D-loop` once per device on whichever arm actually dispatches. Perf with the (default, correct) 2-D loop, B70 policy p512/n128/r3: pp512 752.59±7.89 / tg128 39.48±0.13 vs pre-C3 B70 default 394.84/36.74 — +91% PP, +7.5% TG; both axes up even without the (broken) 3-D path. |
| `GGML_SYCL_AUTO_UBATCH=0` | ON | Disable the `-ub auto` / no-`-ub` micro-batch selection trial (`llama_context::sycl_select_auto_ubatch()`, llama.cpp-nphx Task 4a wires the switch, Task 4b llama.cpp-xojq lands the trial itself) -- falls back to the library default 512 (clamped to `n_batch`, as it always was). An explicit `-ub N` always wins over the trial regardless of this variable; if that explicit value does not fit, the refusal (largest `-ub` that fits, named in the error) comes from Task 1's runtime-context transaction (llama.cpp-ibj0), not from this variable or this task. Any other NON-EMPTY value is treated as enabled and logs one `GGML_LOG_WARN`. An empty value (`GGML_SYCL_AUTO_UBATCH=`) is treated the same as unset, silently -- deliberate, matching `env_mb_override()`'s own convention (`unified-cache.cpp`), not a gap in the WARN coverage. Read once per process (memoized in `unified_cache_auto_ubatch_enabled()`), so changing it mid-run (e.g. via `setenv`) has no effect on an already-running process. SYCL-only, and only for a CAUSAL model (a non-causal model keeps its `n_ubatch == n_batch` semantics, trial or not). When the trial runs, it logs exactly one `[SYCL-PLAN] auto n_ubatch=%u for n_ctx=%u n_batch=%u (tried %s; %s); pass -ub N to override` WARN (a context whose n_ctx cannot admit even the 512 rung, and a model with no published SYCL token, take the pre-trial path and log nothing; a context with no ladder rung between its own `n_ubatch` and the cap -- reachable only from the raw API, with `n_ubatch` above 512 and `n_ubatch_auto` left true, e.g. `n_batch=1000 n_ubatch=600`, or a MoE model whose 512 routing ceiling sits below an explicit `n_ubatch=1024` -- also takes the pre-trial path and skips this WARN and the cache store, but only after the `tuning cache` WARN below has printed its `miss` or `disabled` line), whose trailing `%s` names why the ladder stopped where it did -- `ladder exhausted`, `MoE GPU routing ceiling`, `transaction refused`, `transaction busy`, `not the published model`, `KV would be demoted`, `compute buffer fell back to host`, or (since llama.cpp-pyu4) `compute buffers did not fit`, or (since llama.cpp-7n6n, Task 5) `cached` -- so that single line is the place to look, not the (now INFO-level, quiet) per-candidate refusals underneath it. `MoE GPU routing ceiling` is reported whenever that ceiling (`ggml_backend_sycl_moe_gpu_ubatch_max()`, llama.cpp-ohkx) is the BINDING cap, not only when it narrows a larger batch/ctx cap (llama.cpp-pyu4) -- concretely, every MoE model is pinned at 512 today by that ceiling, which is why GPT-OSS reports this reason while a dense model like Mistral 7B can climb the ladder to 2048. Task 5 also adds a SECOND, separate `[SYCL-PLAN] tuning cache %s: n_ubatch=%u (%s)` WARN logged once per start, before the ladder decision above it: `%s` is `hit`, `miss`, or `disabled`, reporting whether the persisted cache (`GGML_SYCL_TUNING_CACHE` row below) supplied a value that passed the exact same per-candidate validation a ladder rung would (a `hit` is what produces the `cached` stop reason above; a `miss` or `disabled` state still runs the ladder exactly as before this task). Since llama.cpp-y8xv, llama-bench's `n_ubatch` output column and CSV/JSON field print the RESOLVED `llama_n_ubatch(ctx)` (read after context creation) instead of the raw `-ub` value -- this changed what an unrelated existing row can print: `llama_context` clamps `n_ubatch` to `n_batch`, which is itself clamped to `n_ctx` only for causal models (`src/llama-context.cpp`), so a small-context row (e.g. a `tg`-only instance, or one with a short prompt) now prints that clamp, not the requested `-ub`, even with an explicit `-ub` and even outside SYCL. The `n_batch` column is NOT resolved the same way -- it still prints the REQUESTED value (`llama-bench.cpp`'s `test` constructor reads `inst.n_batch` there, not a value read back from the context) -- so e.g. a `tg`-only row started with `-b 2048 -ub 512` prints `n_batch=2048 n_ubatch=128`, not `n_batch=128`. Because `scripts/compare-llama-bench.py` keys row pairing on `n_ubatch` (`LLAMA_BENCH_KEY_PROPERTIES`), runs recorded before and after llama.cpp-y8xv's merge (`6419b06bf`) will silently fail to pair whenever the clamp changed the recorded value (e.g. a default `-n 128` run recorded `512` before, `128` now) -- the affected rows are dropped from the comparison with no warning, and if nothing pairs at all the tool exits 1 with `No comparable data was found`, which does not name the cause. |
| `GGML_SYCL_TUNING_CACHE=0` | ON | **llama.cpp-7n6n (nphx Task 5).** Master switch for the persisted auto n_ubatch cache that wires `tuning-cache-io.hpp` for the trial above: `=0` disables BOTH `ggml_backend_sycl_ubatch_cache_lookup()` and `ggml_backend_sycl_ubatch_cache_store()` (`ubatch-tuning-cache.cpp`) -- every call then returns `false` without touching the filesystem, so the trial behaves exactly as it did before this task (every start runs the ladder; nothing is ever written). Any other NON-EMPTY value is treated as enabled, with one `GGML_LOG_WARN`, mirroring `GGML_SYCL_AUTO_UBATCH`'s own convention (`unified_cache_auto_ubatch_enabled()`) exactly. Memoized (read once per process); changing it mid-run has no effect on an already-running process. The store is advisory: a stale, corrupt, or wrong-version entry can only cost one extra ladder run on the next start, never a wrong choice, because a cache hit is revalidated through the identical per-candidate steps (probe, publish, reserve, host-fallback check) a ladder rung uses. **A hit's STORED REASON decides what "advisory" means, and this is not optional framing -- an earlier version of this row promised "never a wrong choice" while a hit that merely revalidated pinned its value forever, even after a purely transient loss.** A hit whose stored reason is `ladder exhausted` or `MoE GPU routing ceiling` is TERMINAL (nothing above it was ever going to fit) and still skips the ladder outright; any other stored reason (a lost race, a refused publish, a host-pinned fallback) resumes the ladder from the next rung above the cached value instead of trusting it forever, so a value that pinned low on a bad day can climb back up on a later start. The persisted `reason` is now the trial's own actual stop reason, not a fixed literal -- `cached` (a hit whose outcome did not change) and the two pure-race reasons (`transaction busy`, `not the published model`) are never stored, since persisting a race as if it were a shape limit would reintroduce the same sticky-hit bug one level up. A failed store logs one `[SYCL-PLAN] tuning cache store failed: ...` `GGML_LOG_WARN` and is never fatal. |
| `GGML_SYCL_TUNING_CACHE_DIR=<path>` | unset | Override the directory the persisted auto n_ubatch cache reads and writes (`ggml/src/ggml-sycl/ubatch-tuning-cache.cpp`'s memoized accessor). Unset falls back to `tuning-cache-io.hpp`'s `get_cache_dir()` (XDG: `$XDG_CACHE_HOME/llama.cpp/sycl-tuning`, else `$HOME/.cache/llama.cpp/sycl-tuning`, else `/tmp/llama.cpp/sycl-tuning`). The file itself is `<dir>/<sanitized device name>-ubatch.json` (the `-ubatch` suffix keeps it from colliding with the pre-existing, still-unwired matmul dispatch-tuning cache's own `<dir>/<sanitized device name>.json` for the same device). An entry's key is `(sanitize_device_name(name) + "@" + driver_version, device_set_hash, model general.name, model total tensor bytes, a 64-bit FNV-1a hash over the tensor inventory's names and byte sizes, n_ctx, n_batch, n_seq_max, flash_attn, type_k, type_v, kv_unified, swa_full)` (device_set_hash is an FNV-1a hash over the ordered in-process device indices of every SYCL backend, so `level_zero:0` and `level_zero:0,1` are different entries; kv_unified is cparams.kv_unified, added in CACHE_VERSION 3 once KV sizing started depending on it; swa_full is llama_context_params::swa_full, added in CACHE_VERSION 4 (llama.cpp-uajm) because every SWA layer is sized as a FULL layer under it, so a CLI run (common's default false) and a raw-API context (llama_context_default_params()'s true) have different KV demand) -- no PCI id (it moves across boots on this host); a driver upgrade or a re-quantised model file changes the key and so misses cleanly rather than applying a stale value. The file keeps at most the 64 newest entries per device (by `created`); older ones are evicted on the next store. |
| `GGML_SYCL_BLOCK_EXEC_DENSE=0` | ON | Disable the dense-split block executor (`block-exec-dense.hpp`, `ggml_sycl_block_exec_dense_enabled` in `ggml-sycl.cpp`). The value is read with `atoi`, so **any value that parses to 0 disables it** -- `0`, an empty value, and non-numeric values such as `off`, `false` or even `true`; only an unset variable or a non-zero number leaves it on. When a dense model is split over two cards, the executor cuts each graph into contiguous node ranges, one per device, and runs each range through the ordinary node loop on its own device, copying only the tensors that cross a range edge; without it every op of the second card goes through the per-op route. Per-graph gates decline MoE/`MUL_MAT_ID` graphs, fewer than two active layer blocks, KV off the execution device, CPU offload and single-device graphs, so single-card and MoE runs are unaffected. Measured at `8f5ad20bc` (split `level_zero:0,1`, `GGML_SYCL_VRAM_BUDGET_PCT=22`, Mistral 7B Q4_0): pp512 2396 / tg128 90.8 on vs 2246 / 67.4 off. With `GGML_SYCL_BLOCK_EXEC_TRACE=1` it prints one `[SYCL-BLOCK-EXEC-DENSE-RESULT] ... executor=executed\|inactive gate=<reason>` line per graph compute, **except** when it is disabled (no line at all, so an absent line under TRACE means disabled) and when the older candidate executor (`GGML_SYCL_BLOCK_EXEC_EXECUTE`) already ran the graph. |
| `GGML_SYCL_BLOCK_EXEC_DENSE_GRAPH=0` | ON | Keep every dense-executor range on direct dispatch instead of recording and replaying a SYCL command graph per range (decode graphs only). |

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
| `GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW` | unset = AUTO | Three-way override for whether decode `FLASH_ATTN_EXT` nodes are allowed into SYCL command-graph recording/replay (llama.cpp-dyi3). **Owner ruling (round 9, decode graph replay now proven correct on hardware — see the task's comment log, dyi3 c-z3di): unset is now AUTO**, the observation-gated mode: the graph engages only when every decode-shape FA dispatch this context has observed reached a kernel in the verified-safe allowlist (`fa_decode_kernel_observation::kernel_family`, `common.hpp`) — currently `ESIMD_PARTITIONED` (D<=256, dyi3's original route) and `D512_TILE` (gemma4's D=512 global-attention layers, `launch_fattn_tile_d512`, added llama.cpp-86a7: verified individually, not inherited from ESIMD's clearance — same graph-safe launch idiom, no wait()/malloc at submission). This is a positive, named allowlist, not a threshold: any other observed kernel family (oneDNN SDPA — structurally excluded from decode entirely, since `ne01=1` is below its `MIN_NCOLS` floor — VEC safe-decode, XMX, or a future kernel this task never looked at) keeps the exclusion until it is individually verified and added by name. `=auto` is the same mode, spelled explicitly. `=1` (or any other nonzero integer) force-engages the graph regardless of the observation — the original diagnostic-only behavior, for FA shapes/paths never verified replay-safe. `=0` disables it (force-off, today's per-op dispatch with no decode-FA graph at all). **Why the default moved:** rounds 5-8 held it at FORCE_OFF pending proof; that proof is now in. `llama-server` `/completion` with `n_probs=5` (45 decode replays/run), chosen-token logprob deltas measured *with a nondeterminism control* (two no-graph runs against each other, same host, same everything) on both cards — B70 control max 4.91e-02 / mean 9.19e-03 / median 3.14e-03 vs. forced max 1.44e-01 / mean 1.56e-02 / median 2.33e-03, all 48 tokens identical either way; B50 control agrees on only 11/48 tokens while forced agrees on 14/48, i.e. forced graph replay agrees with a no-graph run *longer* than two no-graph runs agree with each other. Forced sits inside the machine's own nondeterminism envelope on both cards. MoE/`MUL_MAT_ID` models never engage whole-graph decode replay at all (separate FORCE_ON-only graphlet/segmented mechanisms, unaffected by this default), so they see no behavior change either way. Do not set `=1` outside a controlled diagnostic run on a model/shape never verified replay-safe by the observation gate — that combination is still unverified, not corrected by this ruling. A value that is neither `auto` nor a valid integer (e.g. a typo like `atuo`) is not silently accepted as unset — it prints one `[SYCL] unknown GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW=…` warning (read once, matching the `GGML_SYCL_FA_FORCE_PATH` parser's precedent) before falling back to force-off. ⚠️ **SET `=0` IF YOU RUN TWO `llama_context`s ON ONE DEVICE.** That configuration returns deterministic wrong logits (llama.cpp-79o5) — no abort, no warning, fluent plausible output — and the defect is replay-dependent: it disappears entirely when no decode graph is engaged. It is PRE-EXISTING (reproduces byte-identically on a master that predates every graph default change, 79o5 c-4s35) and the configuration is documented-unsupported by the canonical memory contract §5, which states that context-keyed KV/RUNTIME arena ownership is absent. But the *reachable surface* grew twice on 2026-09-02: round 9 made unset mean AUTO (so all-ESIMD dense models like Mistral engage), and llama.cpp-86a7 added `D512_TILE` (so gemma4 engages). A two-context deployment that previously could not reach the defect now can. `=0` restores the eager path, which is measured correct in that configuration, at the cost of the decode win (~-40% tg128 on gemma4 B70). Single-context use — the supported case, and what `llama-server` does with `--parallel N` since that is one context with N sequences — is unaffected. |

## Kernel dispatch tuning

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_DISPATCH_TUNING=0` | Master switch for `dispatch-tuning.{hpp,cpp}`'s per-shape kernel-choice override (`ggml_sycl::dispatch_tuning::lookup_kernel()`, consulted from `ggml-sycl.cpp`'s kernel selection). Default **ON**; `=0` (or any value `atoi`s to 0) disables lookups outright, so every mul_mat dispatches through the normal layout/eligibility rules with nothing overridden or cached. |
| `GGML_SYCL_DISPATCH_TUNING_JSON=<path>` | **Opt-in, new in llama.cpp-o65k.** Path to a `sycl-kernel-bench --emit-json` summary (`tools/sycl-kernel-bench/README.md`) to load per-shape kernel-choice overrides from. Unset or empty: `tuning_path()` returns nothing, `ensure_model_loaded()` returns before touching the filesystem, and nothing is logged — the pre-o65k behaviour (a hardcoded default of `/tmp/onednn_unified_bench.json`, tried unconditionally and WARNing on every model load when absent, which it was on essentially every run) is removed, not just made conditional. Set: load behaviour is unchanged from before, and a successful load now logs one `GGML_LOG_WARN` per model (was `GGML_LOG_INFO`, dropped at default verbosity in every tool — see CLAUDE.md's "llama-bench traps" section — so a successful opt-in load used to be invisible). ⚠️ **The lookup key (`DispatchTuningKey`: quant type plus M/N/K bucket, `dispatch-tuning.hpp`) carries no device or driver identity** — a JSON measured on one card is applied to every card, regardless of driver version; only point this at a file measured on the hardware you are actually running. An empty value is unset; a whitespace-only value is treated as a path and fails to open with one WARN per model. The path is only consulted while `GGML_SYCL_DISPATCH_TUNING` is on; with the master switch off no load is attempted and neither WARN appears. |
| `GGML_SYCL_FORCE_MMVQ=1` | Force MMVQ kernels for all batch sizes |
| `GGML_SYCL_FORCE_ESIMD=1` | Force ESIMD kernels |
| `GGML_SYCL_FORCE_MMQ=1` | Force MMQ kernels |
| `GGML_SYCL_FORCE_DMMV=1` | Force DMMV kernels. ⚠️ **No effect at batch=1 on its own** — the TG fast-path returns upstream of both sites that read it. Pair with `GGML_SYCL_TG_FAST=0`; see "The TG fast-path claims batch=1" below. |
| `GGML_SYCL_ESIMD_MIN_BATCH=N` | Min batch size for ESIMD dispatch |
| `GGML_SYCL_ONEDNN_PP_MIN_BATCH=N` | Min batch for oneDNN PP path |
| `GGML_SYCL_ONEDNN_MUL=1` | Enable oneDNN for element-wise MUL (default OFF, SYCL kernel is 2.3x faster) |
| `GGML_SYCL_FA_ONEDNN_MIN_NCOLS=N` | Minimum `ne01` (query rows per flash-attention call) for oneDNN SDPA eligibility — `ggml_sycl_flash_attn_ext_onednn_plan()`'s `BELOW_MIN_NCOLS` gate (`fattn-onednn.cpp`), the very first check in the planner. Default **8**; `=0` disables the threshold entirely (all batch sizes eligible, subject to every other gate). Pre-existing infrastructure (introduced `79eb6710d`, 2026-04-30), not new to any attention-migration ticket — `llama.cpp-bn5k` item 1 uses it as a measurement hatch rather than adding one. `ne01` is the number of query positions in ONE call: single-token autoregressive decode (TG) is always `ne01=1`, so it is **unconditionally below the default threshold and never reaches oneDNN today**, regardless of D, scale, or model — this is true even for gemma4's D≤256 SWA layers, which `llama.cpp-p0f5` already made scale-eligible for *prefill*. Prompt-processing (PP) calls carry `ne01` up to the ubatch size (commonly 512), comfortably above the floor, except a prompt's final partial chunk if it is smaller than the threshold. The default of 8 exists to bound oneDNN Graph JIT-recompilation cost: the compiled-partition cache (`sdpa_partition_cache`) keys on the full shape including `ne11` (KV length), so admitting `ne01=1` unconditionally would mean autoregressive decode compiles (or looks up) a new-or-growing-KV-length partition on every single generated token — a real JIT-cost risk to measure before lowering the default, not just a historical assumption. ⚠️ **DEAD ZONE at `ne01=1` (llama.cpp-bn5k item 1, closed 2026-08-31): lowering this variable alone has ZERO effect on single-token autoregressive decode benchmarks, and a real A/B first ran straight into it.** `ggml_sycl_flash_attn_ext_onednn_plan()` — the function this threshold lives in — is never even called for `ne01=1` under normal dispatch: `ggml_sycl_flash_attn_ext_dispatch_ncols()` (`fattn.cpp`) hits two fast-decode interceptors that `return` BEFORE the oneDNN branch is reached — the ESIMD decode fast path (`fattn.cpp:~3057-3065`, ON by default whenever the device supports it) and the VEC fast path (`fattn.cpp:~3072-3077`) right behind it. Only a call that falls through BOTH reaches the oneDNN check, where this variable finally matters — so it genuinely gates the `ne01` **2-7** range (speculative-decode drafts, small batched decode), just not `ne01=1`, the shape that dominates default TG measurement. **Measured anyway** via the pre-existing `GGML_SYCL_FA_FORCE_PATH=onednn` override (sits ahead of both interceptors, calls the same plan function): forced-oneDNN decode on gemma4/B70 correctly engaged (300/300 `force_onednn`, digit gate exact) and still lost to `esimd_f16` — 28.83/28.89 forced vs 29.82/30.15 default tg128 (-3.8%), consistent with the JIT/partition-churn cost this default already guards against. **Do not re-run "lower `MIN_NCOLS`, bench TG" as a decode probe — it measures nothing at `ne01=1`**; use `GGML_SYCL_FA_FORCE_PATH=onednn` instead (with a correctness gate under the same env first — its `DIRECT` plan kind is production-verified for prefill's KV layout only, not decode's). |
| `GGML_SYCL_FA_ONEDNN_D512_SCALE=1` | Measurement hatch (`llama.cpp-bn5k` item 2, default **OFF**) that relaxes the "CONSERVATIVE SCOPE, D > 256" scale check in `ggml_sycl_flash_attn_ext_onednn_plan()` (`fattn-onednn.cpp`): with it set, D>256 (i.e. D=512, since D>512 is rejected earlier by `UNSUPPORTED_D`) is screened by the same finite-and-nonzero check D≤256 already uses (`llama.cpp-p0f5`), instead of the strict `1/sqrt(D)` requirement — letting gemma-3n's D=512 global-attention layers (`kq_scale=1.0`) reach oneDNN so they can be A/B'd against the tile route (`GGML_SYCL_FA_TILE_D512`, below) without a code change. Deliberately a **different** switch from `GGML_SYCL_FA_ONEDNN_D512` immediately below — that one gates whether D=512 traffic reaches this planner *at all*; this one only changes what the planner does with the scale once asked, and does nothing unless the other switch (plus `GGML_SYCL_FA_ONEDNN`/`GGML_SYCL_FA_NO_XMX`/`GGML_SYCL_PAGED_V2`, all of which `ggml_sycl_fattn_d512_onednn_admissible()` also checks) is already open. The name was deliberately chosen to NOT be a token-order transposition of `GGML_SYCL_FA_ONEDNN_D512` (i.e. not `GGML_SYCL_ONEDNN_FA_D512`) — two independently-acting switches whose names differ only in whether "FA"/"ONEDNN" are swapped is a footgun for anyone typing an A/B script by hand. Exercised by `test_planner_d512_gemma3n_scale_follows_relax_hatch`/`test_supports_op_d512_gemma3n_scale_follows_relax_hatch` in `tests/test-sycl-fattn-onednn-gates.cpp`. |
| `GGML_SYCL_FA_ONEDNN_MATERIALIZE=1` | Opt back into `MATERIALIZE_REQUIRED` for GQA/MQA flash-attention shapes whose K/V token stride differs from D — the `KV_NC_STRIDE_MISMATCH` gate in `ggml_sycl_flash_attn_ext_onednn_plan()` (`fattn-onednn.cpp`), which is the single materialize-vs-`DIRECT` decision site. Default **OFF**; unset and `=0` are identical and plan `DIRECT`, skipping the dense f16 K/V repack before the oneDNN SDPA execute. ⚠️ **The polarity reversed on 2026-08-10** (`llama.cpp-olpg`, owner ruling): this variable was introduced by `llama.cpp-l7rt` defaulting **ON** as a pure measurement axis, and any text describing it that way predates the ruling. `DIRECT` is numerically correct at D=16 and D=128 within the descriptors fixture's tolerance and end-to-end on the Mistral digit gate; on B50 pp512 it measured at worst parity across five interleaved pairs (two at parity, three ahead — the magnitude is deliberately not quoted, as the pairs were anti-correlated and only the direction is supported). The heap-corruption abort once recorded against `DIRECT` was a harness artifact of a stashed-library A/B, not a product defect (`llama.cpp-l7rt`). `=1` is retained as the **A/B axis**: it makes the decision revisitable from any build with two env-var runs, no patching and no second `libggml-sycl.so`. |
| `GGML_SYCL_FA_ONEDNN_D512=0` | Kill switch for the oneDNN half of D=512 flash-attention (llama.cpp-jahv, e.g. gemma-3n's 7 global-attention layers). `=0` removes oneDNN from the D=512 route, leaving only the tile route below (or CPU fallback if that is also disabled/inadmissible). Default **ON**, but note this is frequently a no-op even at default: oneDNN's compiled-partition softmax divisor is hardcoded to `sqrt(D)` (`build_and_compile_sdpa`, `fattn-onednn.cpp`), so any model whose `kq_scale != 1/sqrt(D)` — gemma3n hardcodes `f_attention_scale = 1.0f` (`src/models/gemma3n.cpp`), same as phi2/gemma2/gemma3/gemma4/gemma4-assistant/gemma-embedding — still rejects via `SCALE_UNSUPPORTED` regardless of this switch, and TG/decode (`ne01=1`) still falls below `GGML_SYCL_FA_ONEDNN_MIN_NCOLS` (default 8) and still declines to CPU (oneDNN's, not the whole route's). The wiring is real and exercised (`test_planner_accepts_d512_with_matching_scale`/`test_planner_d512_gemma3n_scale_follows_relax_hatch` plus the supports_op-level `test_supports_op_*` cases in `tests/test-sycl-fattn-onednn-gates.cpp`), but on its own unblocks only a future D=512 model with standard `1/sqrt(D)` scaling and a prefill-shaped (`ne01>=8`) call — gemma3n's actual decode-phase global layers are served by the tile route instead (llama.cpp-dtpk, below), not by this switch. `GGML_SYCL_FA_ONEDNN_D512_SCALE` (above) is the separate, opt-in measurement hatch that relaxes the scale requirement itself for this route. **The oneDNN half of D=512 also declines, independent of this switch, whenever `GGML_SYCL_FA_ONEDNN=0`/`GGML_SYCL_FA_NO_XMX=1` (oneDNN disabled globally) or `GGML_SYCL_PAGED_V2=1` (paged-attention's block-table K/V layout can't represent oneDNN's contiguous `[D, n_kv]` expectation) is set** — `ggml_sycl_fattn_d512_onednn_admissible()` checks all of these, not just this one variable, so do not read a D=512 op landing on tile or CPU under one of those other settings as evidence this switch itself is broken. Spec review llama.cpp-jahv/c-c3y6 found and fixed a **residual correctness gap**: D=512's oneDNN execute can still fail at dispatch time for reasons invisible to the admissibility check (SYCL command graph recording in progress, Q/K/V host-pinned under `GGML_SYCL_KV_HOST`/host-resident weight streaming/warmup, or a materialization allocation/repack failure) — llama.cpp-dtpk changed the outcome of that gap: rather than `GGML_ABORT`ing (the original behavior, still what happens if the tile route below is also inadmissible/disabled), the D==512 dispatch branch now falls through to the tile route first, since tile has none of oneDNN's execute-time-only decline causes. |
| `GGML_SYCL_FA_TILE_D512=0` | Kill switch for the tile half of D=512 flash-attention (llama.cpp-dtpk) — the gemma-viable route: `fattn-tile.hpp`'s `flash_attn_tile<>` takes attention scale as a runtime value (not oneDNN's compiled-in `1/sqrt(D)`) and has no `ne01` floor, so it is the only route that serves gemma-3n's actual global-layer calls (`kq_scale=1.0`, decode `ne01=1`). `=0` removes tile from the D=512 route, leaving only oneDNN (or CPU fallback if that is also inadmissible). Default **ON**. ⚠️ **`=0` does not restore the pre-`llama.cpp-dtpk` F16 Q cast** -- `build_attn_mha()`'s skip of that cast is keyed on `D==512` alone, unconditionally, not on this switch; it restores which route a D=512 op is admitted into, not what dtype Q arrives as. With tile disabled, a D=512 op still reaches oneDNN (or the CPU fallback) with F32 Q, taking their respective F32-Q code paths (oneDNN's materialize-to-F16 step; the CPU kernel's `q_to_vec_dot()` branch) rather than the F16 fast path either had before this ticket. `ggml_sycl_fattn_d512_tile_admissible()` (`fattn.cpp`) additionally declines outright (not this switch's doing) whenever the op carries a paged block table or continuous-batching seq-id sources (`dst->src[5..8]`, `submit_fattn_tile_d512` doesn't implement either), the Q/K head counts aren't an integer GQA ratio, or Q is not F32 — the last one matters because `flash_attn_tile<>` has no `Q_type` template (unlike every sibling SYCL FA kernel family) and hardcodes an F32 load; `llama-graph.cpp`'s `build_attn_mha()` skips the usual `GGML_SYCL_FATTN_Q_TYPE`(=F16 under `GGML_SYCL_F16`) cast specifically for `D==512` so real ops arrive as F32, and the dtype check here is a defensive backstop against anything that doesn't. Route composition (`ggml_sycl_flash_attn_ext()`'s D==512 branch): oneDNN is tried first when admissible (a vendor-tuned GEMM-based SDPA, plausibly faster for the standard-scale/prefill shapes it can serve); tile is tried next, including catching oneDNN's accepted-then-declined-at-execute case described under `GGML_SYCL_FA_ONEDNN_D512` above. Exercised by the `test_supports_op_admits_d512_tile_*`/`test_supports_op_declines_d512_tile_*` cases in `tests/test-sycl-fattn-onednn-gates.cpp`. |
| `GGML_SYCL_FA_D512_DECODE_ESIMD=0` | Kill switch for the decode-shaped (`ne01==1` ONLY, see the ⚠️ below) D=512 flash-attention ESIMD tier (llama.cpp-zwsj, plan Task P3, part 2 of llama.cpp-ebxw). The P3 spike (`GGML_SYCL_KERNEL_PROFILE` captures of gemma4's `fattn.decode.tile_d512`, both cards, `n_kv` in {32,128,512,2048}) found the tile route's decode cost **flat in `n_kv`** (B70 115.9 µs at every point; B50 124.7-134.0 µs) — a fixed per-launch cost of one work-group serially doing the whole D=512 reduction, not per-key work, so a split-KV tier inside the tile kernel (the plan's other branch) cannot help a short decode call: there is nothing to split. Default **ON**: with the toggle on, an admissible (`ggml_sycl_fattn_d512_tile_admissible()`) D=512 op at `ne01==1` is routed to `fattn_esimd_f16<512, float>()` (`fattn-esimd-f16.hpp`) instead of `launch_fattn_tile_d512()` — the SAME partitioned decode kernel already production-verified for D≤256 (checked: its "optimized" launcher has no D≤256-specific branching; the `D==128` special-casing elsewhere in that header belongs to the separate BATCHED prefill kernel, never reached at `ne01<=1`), so D=512 is a new template instantiation of an existing, verified kernel, not a new algorithm. `=0` falls back to the tile route unconditionally at any `ne01`, matching pre-llama.cpp-zwsj behaviour exactly. Prefill (`ne01>8`) is untouched either way — it always uses the tile route regardless of this switch. Also gated (same as the D≤256 decode path, `fattn.cpp:~2525/2630/3038`) by `g_sycl_fa_esimd_enabled`/`fattn_esimd_f16_available()`: on a build or device without ESIMD, the toggle has no effect and D=512 decode stays on tile — `fattn_esimd_f16<>`'s non-ESIMD stub `GGML_ASSERT`s rather than declining, so this guard is load-bearing, not defensive theatre. Observed decode-shape dispatches (`ne01<=1`) are recorded into the SAME `fa_decode_kernel_observation` allowlist bucket ("esimd_f16") as the D≤256 path, since `classify()` keys on kernel name, not `D`, and this is genuinely the identical verified graph-safe launch idiom. Exercised by `tests/test-sycl-fattn-tile-d512-decode.cpp`, which runs the full `n_kv` sweep under both toggle states in separate re-exec'd processes (the accessor is cached into a function-local static at the dispatch call site, so it cannot be toggled mid-process). ⚠️ **Engagement was narrowed from `ne01<=8` to `ne01==1` after hardware testing (lead hardware finding, llama.cpp-zwsj/c-1ha7, finding A) found the multi-query (`ne01>1`) path returns garbage on a real masked `ne01=4` op (94% of elements wrong) while `ne01==1` measured correct on both cards.** Root cause is not established (D=512-specific vs. a latent bug shared with the D≤256 ESIMD family's own multi-query path, `launch_fattn_esimd_f16_batched`, which nothing exercised with a mask before this ticket either) — tracked as follow-up work on **llama.cpp-wais**, which owns a D=256 `ne01=4 mask=1` control to decide shared-kernel vs D=512-specific and whether this gate can re-widen. `ne01` in 2..8 now falls through to the tile route unconditionally regardless of this toggle — this is gemma4's actual production shape anyway (decode is always `ne01==1` outside speculative decoding). |
| `GGML_SYCL_FA_XMX_V1_PP=1` | Opt simple D=128 prompt-processing flash attention into the old XMX-v1 kernel, for A/B comparison only. Default **OFF**: unset, `=0`, and any value other than exactly `1` select XMX-v2 (`xmx_v2_f16_ncols16_large` for `ne01>16`). XMX-v1 produces non-deterministic, intermittently wrong output on this shape: bit-identical Q/K/V/mask inputs give a different result on every call, and Mistral 7B end-to-end requests fail intermittently on both cards (llama.cpp-b1ov). The shape is D=128 with `ne01>=8` and `ne01%8==0`, no sinks, softcap, FP8 KV, paged or multi-sequence layout, or multi-token decode (`can_use_xmx_v1_runtime()`, `fattn.cpp`), and it reaches native FA only when oneDNN SDPA does not take the call. Cases that route it to native FA include `GGML_SYCL_FA_ONEDNN=0`, a non-DNNL build, host-tier KV layers that oneDNN refuses at execute time, SYCL command-graph recording (oneDNN declines while a graph is being recorded), ALiBi (`max_bias != 0`, `MAX_BIAS_UNSUPPORTED`) and a zero or non-finite scale (`SCALE_UNSUPPORTED`; at D=128 any other scale is accepted) in `ggml_sycl_flash_attn_ext_onednn_plan()` (`fattn-onednn.cpp`). Every such call used to run v1 by default and now runs v2. Decode (`ne01=1`) never reaches v1. Parsed by `ggml_sycl_fattn_xmx_v1_simple_pp_requested()` into a `static const bool` inside the templated `ggml_sycl_flash_attn_ext_dispatch_ncols<D, Q_type>`, so it is read once per instantiation, at its first use, and decided by `ggml_sycl_fattn_xmx_v1_select_simple_pp()` (`fattn-xmx-f16.hpp`), pinned by `test-sycl-fattn-xmx-policy`. The broader v1 hatches `GGML_SYCL_FA_XMX_V1=1` and `GGML_SYCL_FA_FORCE_PATH=xmx-v1` are unchanged and equally A/B-only. |
| `GGML_SYCL_BATCH_EXPERTS=0` | Disable batched expert kernel launches (default ON) |
| `GGML_SYCL_GETROWS_Q8_OPT=0` | Restores the pre-`llama.cpp-go37` scalar (1-element/thread, hand-rolled dequant) Q8_0 AoS GET_ROWS kernel. Default **ON**: the 2-elements/thread kernel that reuses the shared `dequantize_q8_0()` primitive instead (same launch geometry as the file's Q4_0/Q4_1/Q5_0/Q5_1 AoS GET_ROWS, same `i01` row-bounds guard the scalar kernel added for correctness). This is a **hygiene switch, not a performance lever** — an interleaved A/B on both cards (tracker llama.cpp-go37, comment c-m6hd) measured no PP/TG change on either arm: GET_ROWS hides behind MUL_MAT in the pipelined graph, so the OP_TIMING drain-mode share this ticket started from was never recoverable time. Kept as a cheap, already-tested rollback hatch given `per_layer_token_embd.weight`'s history of correctness incidents (dkw0), not because a regression is expected. |
| `GGML_SYCL_ESIMD_DEQUANT=1` | Opt-in retest hatch for ESIMD small-block dequant; standard SYCL is the default. ⚠️ The 1.9x-slower figure behind that default was measured on an **Arc B580 + oneAPI 2025.3** — that card is no longer in this machine (replaced by the B70) and the toolchain has moved on, so treat it as *historical justification*, not a current measurement. The conclusion is still believed to hold (block granularity too small to amortize LSC loads), but it has not been re-measured on Battlemage G31. Same caveat applies to the copy of this claim in `CLAUDE.md`. |
| `GGML_SYCL_LAYOUT_OVERRIDE=<mode>` | Force a weight layout: `aos`, `soa`, `coalesced`, or `xmx_tiled`. Overrides the layout policy's own choice — use for A/B isolation, not as a default. (Migrated from AGENTS.md 2026-07-25, which was its only documentation.) |
| `GGML_SYCL_USE_XMX_GEMM=1` | Route quantized MUL_MAT through the experimental XMX GEMM kernels (measured 5–11x **slower** for quantized models). Needs a build carrying **both** `GGML_SYCL_XMX_GEMM` and `GGML_SYCL_MMQ_XMX`; in a default build it does nothing. `=0` disables it on both dispatch paths (it did not until `llama.cpp-wvbw`; see below). |
| `GGML_SYCL_XMX_THRESHOLD=N` | Upper batch bound for the XMX GEMM path; the gate is `batch >= 1 && batch < N`. Default **64**, stated only by the settings table in `ggml_check_sycl()` — not by the global's initializer. Same build requirement as above. See below. |
| `GGML_SYCL_MXFP4_GROUPED_DPAS_ROW_LIST_TILES=N` | Caps the grouped-DPAS MXFP4 MoE row-list chunk size at `caps.N * N` rows per submission. Default **256** (raised from 16 on 2026-08-17, llama.cpp-e3xj): 16 forced ~23 chunks/layer on GPT-OSS pp512, each re-reading expert weight tiles; 256 measured +7.5% pp512 on B50 (136.4->146.7, interleaved r=2 pairs) and ~+30% on B70 (356->471-493). |
| `GGML_SYCL_MXFP4_GATEUP_KSPLIT=<1\|2\|3\|4\|auto>` | **Opt-in (llama.cpp-lis9, plan Task G2, parent llama.cpp-30ak7 lever 2).** Splits the m2 gate/up decode DPAS kernel's K reduction (`mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl`, `mmvq.cpp`) across `S` work-items per M-tile-pair, combined by a second small reduction kernel that applies the bias/GLU epilogue after the full sum. Default **1**: the ksplit kernels are not even reached (see the submit wrapper's early-out), but both the S=1 kernel's K-reduction/DPAS body (`mxfp4_pair_glu_xmx_tiled_dpas_m2_k_reduce`) and its epilogue (`mxfp4_bundle4_store_glu_tile`) are shared with the ksplit kernels, not isolated from them (quality round 2 corrected an earlier claim of isolation here) — a change to the shared K-reduce or epilogue helper made for ksplit's benefit lands on the default S=1 path too. The safety argument for the default is therefore not code-path isolation, it is: the K-reduction extraction is verified byte-identical at the source level to the pre-refactor S=1/ksplit loop bodies, and the unmodified S=1 numerics gate, the GPT-OSS chat gates, and the B70/B50 A-B (B70 tg128 +6% under `auto`, B50 unchanged, profiled m2 kernel mean unchanged at 112.2 us) all pass on the resulting binary. `auto` = `clamp(compute_units / 128, 1, 4)`: llama.cpp-ulp9 (Task G1) found the B70 gate/up launch (k=4 experts, 720 threads) flat from k=1 to k=8 — 8x the K-tile bytes costs only +14% time, elbow between k=8 and k=16 — so splitting it toward that elbow should recover most of the gap; the B50 (128 CUs) resolves `auto` to exactly 1 (it was already near its own occupancy elbow at k=4, so a split there would only add combine overhead — the formula's own arithmetic gives this without a separate `compute_units <= 128` special case). Explicit integer values are clamped into `[1, 4]`, the only range the kernel's K-tile partition is exercised at (`ggml_sycl_mxfp4_gateup_ksplit`, `common.cpp`). Scope-limited to the `n_tokens==1` **default** TG1Index=false launch (llama.cpp-ulp9's profiled shape) — with `GGML_SYCL_MOE_GATEUP_M2_TG1_INDEX` set, ksplit is not consulted and the unmodified S=1 kernel always runs. Profile row `mxfp4.gateup.xmx_tiled_dpas_m2`'s metadata gains `;ksplit=<S>` when `S>1` (the combine kernel gets its own separate row, `mxfp4.gateup.xmx_tiled_dpas_m2_ksplit_combine`). Numerics gate: `tests/test-sycl-mxfp4-gateup-ksplit-numerics.cpp` (S=1/2/3/4 agree within `1e-4 + 1e-3*|ref|` per element — DPAS int8 accumulation is exact, only the float combine across k-parts reorders the final sum). At this shape (GPT-OSS gate/up, `k_tiles=90`), S=4 is the only point covering a REMAINDER partition (`90/4=22 r2`, i.e. two K-parts of 23 tiles and two of 22, exercising `k_tiles_rem`); S=2 (`90/2=45`) and S=3 (`90/3=30`) both divide evenly. S=3 is in the set for a clean-division split ratio between the already-covered S=2 and S=4 cases, not to exercise the remainder branch itself (see the S-set comment above `kNumKsplitPoints` in that test file). Default stays `1` until the B70/B50 A-B and the GPT-OSS chat gate pass on the same build, per the fork's opt-in rule. |

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
| `GGML_SYCL_DEV_LAYER_SYNC=0` | ON | **llama.cpp-30h4.** `dev_layer(il)`/`dev_output()` used to keep reporting the `-ngl`-derived prefix split even after the SYCL planner tiered a layer to host by any OTHER route (budget-percent tiering, `-ot`, `-nkvo`, ...) — `get_layer_buft_list()` only ever marks the `[0, i_gpu_start)` prefix as CPU and has no way to see a later planner decision. Two real consumers acted on that stale value: FA executor placement (`llama-context.cpp:1236`) and KV buft selection (`llama-kv-cache.cpp:57-75/287`), both of which then treated a host-tiered layer's weight as still device-resident. Default **ON**: right after the LATE replan (`llama_model_sycl_set_late_inventory()`, `src/llama-model.cpp:2395`), each layer's actual placement is read from the buffer type its tensors were actually created in (`ml.ctx_map`) and `dev_layer[il].dev`/`dev_output.dev` (never `.buft_list`) is corrected to the CPU device when that buft is ggml-cpu-owned (e.g. `CPU_REPACK`). An earlier revision asked the planner (`ggml_backend_sycl_planned_target_device()`) instead; that query returns `NO_PLAN`, never host, for exactly these tensors, because CPU-owned bufts are excluded from the LATE inventory — so it corrected 0 layers in every arm (`llama.cpp-30h4` c-pj9b). `SYCL_Host` is SYCL-owned and does not trigger a correction. Only the EXPLICIT literal string `"0"` or `"false"` disables it and restores the pre-fix stale behaviour — mirrors the established default-ON/opt-out idiom this codebase already uses for the same shape of switch (e.g. `ggml_sycl_fa_onednn_d512_enabled()`, `fattn.cpp`); the check is a plain string compare, not `atoi()`-based, specifically so an unparseable or unexpected value (a typo, empty string) cannot silently fold to the disabled branch the way it did for `GGML_SYCL_KV_HOST` on `llama.cpp-80w9`. Known side effect once a layer's `.dev` flips to `cpu_dev`: `llama_kv_cache_sycl_hooks_for()` requires the device's backend reg to be named `"SYCL"`, so that layer's KV falls through to a plain `ggml_backend_cpu_buffer_type()` instead of `SYCL_KV_Host` — expected, not a regression, since the layer's weights are host-resident too. **Positive control for the `=0` arm:** `dev_layer`'s own per-layer assignment log (`get_layer_buft_list()`) is `LLAMA_LOG_DEBUG` and INFO is dropped at default verbosity in every tool on this tree, so this fix also emits one `LLAMA_LOG_WARN` — `"[SYCL] dev_layer sync: corrected N layer(s)[+ output] ..."` — only when it actually changed something; it cannot print under `=0`/`=false` (the whole correction block is skipped), so its presence/absence at default verbosity is a clean two-arm signal rather than an empty grep that proves nothing either way. |
| `GGML_SYCL_KV_HOT_LAYERS=N` | auto | Hot layer count for per-layer KV hot/cold tiering |
| `GGML_SYCL_KV_HOT_PCT=N` | auto | Hot window as % of total KV buffer |
| `GGML_SYCL_FORCE_STREAMING=1` | OFF | Enable GPU weight streaming (Level 5, last resort) |
| `GGML_SYCL_HOST_COMPUTE=1` | OFF | Use host-pinned compute buffers (eliminates staging for CPU-dispatched layers) |
| ~~`GGML_SYCL_PIPELINE_MOE=1`~~ | — | ⚠️ **DEAD — does nothing.** No code reads this name (comments only), and its gate `ggml_sycl_pipeline_moe_enabled()` (`ggml-sycl.cpp:14178`) is a hardcoded `return false;`. Added `ae5eae507`, getenv removed `3cf9b5fdf` (2026-05-09, listed under "Removed (hardcoded to defaults)"), then **this row was created `f3c36987f` two months after the removal** — it was never accurate. Setting it measures nothing; do not conclude multi-GPU MoE overlap "doesn't help" from it. `GGML_SYCL_PIPELINE_CPU` below **is** live. |
| `GGML_SYCL_PIPELINE_CPU=1` | OFF | Pipeline CPU expert compute with GPU attention across MoE layers: CPU experts from layer N run during layer N+1 attention |
| `GGML_SYCL_BLOCK_EXEC_DENSE_MAX_ARENA_MB=<n>` | **1024** | Largest arena, per device, that the dense-split block executor (`GGML_SYCL_BLOCK_EXEC_DENSE`) may allocate for the tensors crossing between its ranges; a plan needing more is declined with `gate=arena-too-large` and the graph runs on the per-op path. A value that is not a positive number (`0`, negative, empty, non-numeric) falls back to 1024. |

## Cache and memory

| Variable | Effect |
|----------|--------|
| ~~`GGML_SYCL_UNIFIED_CACHE=0`~~ | **Removed by `9a0670712` with optional cache enablement and its enable/disable branches.** Setting this name no longer disables anything; there is no replacement opt-out. |
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Select cache topology (`auto`, `global`, or `per_device`) only; it cannot disable the cache. |
| `GGML_SYCL_NO_PINNED=1` | Disable pinned host memory |
| `GGML_SYCL_WEIGHTS_EVICTABLE=1` | Allow weight eviction under memory pressure |
| `GGML_SYCL_MEM_BUDGET=<MB>` | Set VRAM budget in MB |
| `GGML_SYCL_HOST_RESERVE_MB=<MB>` | Overrides the auto-computed pinned-**pool BUDGET** (the host arena's overall cap; `min(total_ram*pct, available - max(8 GiB, total/32))` by default) with a fixed byte count, in MB. Wired in llama.cpp-glkg -- until then this name was parsed only by `resolve_host_reserve_bytes()`, which had zero callers (a documented-but-dead variable): setting it changed nothing, and the NAS report that surfaced this used it expecting a budget increase and instead saw the request fail EARLIER (`alloc_tensor_range: failed to allocate SYCL_Host buffer of size 1963470336`), because raising it also raised the un-budgeted pre-zone footprint the report's own repro was hitting. Do not confuse this with `GGML_SYCL_HOST_STAGING_MB`/`GGML_SYCL_MMAP_STAGING_MB` below, which size a small fixed buffer, not the pool's overall cap; the only other way to change the pool budget is `ggml_backend_sycl_set_unified_cache_host_budget_pct()` (a **percentage**, C-API only, not reachable from any CLI flag) -- this variable is the CLI-reachable byte-count form of the same underlying budget. Skipped if that C-API override was already used (nonzero), since that is an explicit programmatic choice. Accepted syntax (llama.cpp-16el): an unsigned decimal integer, optionally surrounded by whitespace, where `0` keeps the auto calculation; anything else -- a sign (`-1`, `+4096`), trailing or embedded characters (`6144junk`, `0x10`), a value that does not fit `unsigned long long`, or a MiB count above 17592186044415 whose byte conversion would overflow `size_t` -- is rejected with one `GGML_LOG_WARN` naming the variable, the raw string and the reason, and the auto calculation then runs exactly as if the variable were unset. Also useful as a deliberate reproduction lever for host-zone-capacity issues: a small value here (comfortably below a model's host-resident weight footprint) reproduces `"pinned pool capacity 0.0 MB is below requested zone footprint ..."` on hardware whose ordinary budget would never hit it. |
| `GGML_SYCL_HOST_STAGING_MB=<MB>` / `GGML_SYCL_MMAP_STAGING_MB=<MB>` | Sizes the unified cache's OWN small internal staging buffer (`resolve_host_staging_bytes()`, default 64 MB), allocated once at cache bootstrap before the host arena/pinned pool exists -- **not** a pinned-pool zone, and not the pool's overall budget (that is `GGML_SYCL_HOST_RESERVE_MB` above). Raising this does not enlarge the WEIGHT/KV/STAGING/SCRATCH host zones or the pool's budget cap; it only changes this one bootstrap allocation's size. |
| `GGML_SYCL_ONEDNN_GRAPH_ZONE_MB=<MB>` | ⚠️ **Semantics changed under llama.cpp-0oxf.** Was a flat 64 MiB default (previously 512, until llama.cpp-gwno's `event_complete()` fix — history below); the DEFAULT (no env var set) is now **shape-derived**: `floor = max(64 MiB, 1.5 x max(n_head_ctx_max x n_ctx, n_head_swa_max x min(n_ctx, n_swa + n_ubatch)) x n_ubatch x sizeof(f32))`, where `n_head_ctx_max`/`n_head_swa_max`/`n_ubatch`/`n_ctx`/`n_swa` are the planner's max query-head count per attention window class (non-SWA vs SWA, over layers eligible for the oneDNN SDPA route), planned ubatch size, planned context length, and the sliding-window size (see `unified-cache.cpp`'s `onednn_graph_scratch_zone_floor_bytes_swa()`; llama.cpp-o3a0 split the earlier single `n_head` into this per-class pair so a SWA layer's effective KV length is `min(n_ctx, n_swa + n_ubatch)`, not `n_ctx` — a ubatch of `n_ubatch` queries against an `n_swa`-key sliding window spans `n_swa + n_ubatch` keys in total, GPU-verified, not `n_swa` alone, which an earlier revision of this same fix used and which under-provisioned SWA layers). ⚠️ **An earlier version of this fix anchored the floor to `n_ctx` alone — that was WRONG** (task llama.cpp-0oxf, ticket correction c-xcop): the oneDNN Graph-scratch request is the PEAK OUTSTANDING size across however many compiled SDPA partitions are concurrently alive, one per distinct (n_head, ubatch-rows, KV-length) shape a call produces, and it scales with `n_ubatch` just as much as with `n_ctx` — measured identically 144 MB at n_kv 2048/4096/8192 on Mistral 7B Q4_0's default ubatch, but smaller at a smaller ubatch. The `1.5x` factor (not `1.0x`) covers ~5 SDPA scratch buffers measured concurrently in flight even on an idle host. Using `n_ctx` as an upper bound on the real KV length (`ne11`) for every layer under-explained one measurement until traced further: gemma4's oneDNN-served attention layers are sliding-window, so that flat formula over-provisioned rather than under-covered (bounded by the 25% budget clamp below, but wasting VRAM) — llama.cpp-o3a0 closed this, first with `min(n_ctx, n_swa)` (`n_swa` alone) for the SWA class's `ne11`. ⚠️ **That `n_swa`-alone form was ALSO WRONG** (GPU-verified on the B50, `GGML_SYCL_DEBUG=1` probe of gemma4 E4B's own SWA layers): measured Graph-scratch requests at `n_ubatch=512` were 24.00/12.00 MB (high-water 24.0 MB), and at `n_ubatch=256` were 9.00/6.00/3.00 MB (high-water 9.0 MB) — solving `1.5 x 8 x n_ubatch x K x 4 B` for `K` gives exactly `K = n_swa + n_ubatch` both times against gemma4's real `n_swa=512` (1024 = 512+512; 768 = 512+256), because a ubatch of `n_ubatch` queries against an `n_swa`-key sliding window spans `n_swa + n_ubatch` keys, not `n_swa` alone. This also resolves the original 24 MB measurement this row's own history cites: `1.5 x 8 x 512 x (512+512) x 4 B == 24 MB` exactly. The formula above now uses `min(n_ctx, n_swa + n_ubatch)` as `ne11` for the SWA class and takes the max across classes, so gemma4 (at its real `n_swa=512`, `n_ubatch=512`) now computes 64 MiB (24 MiB raw, clamped) instead of the 192 MiB the flat formula predicted, matching the GPU probe exactly. ⚠️ **This zone is sized once, at MODEL LOAD, against a conservative `n_ctx=512` default — it is never resized once the real runtime `n_ctx` is known** (verification 2 on the ticket: `-p 8192 -ub 512` still took the DIRECT path on every request despite this formula, because `arena_reserve()`'s existing-arena branch ignores the zone-size arguments it is passed once weights are resident — see `docs/backend/sycl-memory-design.md`'s "cannot actually reach the DIRECT path's steady state" note for the full mechanics). ⚠️ **This directly bears on the gemma4 example above too:** the `n_swa + n_ubatch < n_ctx` condition that makes the SWA class's window term actually narrower than the ctx class's (rather than equal to it) only binds once the REAL `n_ctx` is the one being planned against — but the placement envelope's own `n_ctx` field is set to 0 at load (`llama_model_sycl_make_placement_envelope()`) and its only reader anywhere is a diagnostic log line, not `planner_n_ctx` (unlike `n_ubatch`, which the envelope DOES feed via its own separate `envelope->n_ubatch` field), so nothing today threads a real `n_ctx` into the planner via the envelope at all; the later runtime-context update (`ggml_backend_sycl_set_runtime_context()`) also does not re-call `unified_cache_set_planned_onednn_graph_scratch_shape()`, so the Graph-scratch shape (and this floor) stays frozen at whatever it was planned against at load regardless. Re-planning this shape on a runtime context change, so an ordinary `-c 8192` run gets the narrower SWA-aware floor by default, is tracked separately (llama.cpp-fkpg) — this row documents the formula, not (yet) that plumbing. The `[SYCL-PLAN] oneDNN Graph-scratch zone floor:` log line (`unified-cache.cpp`, added alongside this formula) prints the `n_ctx` a given run actually planned against, which is how to tell which case you are in -- but that line is `GGML_LOG_INFO`, which is dropped at default verbosity in every tool (see CLAUDE.md's "llama-bench traps" section), so it needs `-v` (`llama-bench`) or a raised verbosity threshold to actually reach a normal run's log, the same caveat the DIRECT pool summary row above carries for its own `GGML_LOG_INFO` lines. So for a model whose real shape needs more than this floor provisioned at load, the DIRECT path below is the STEADY STATE for that shape, not an occasional fallback — its own reuse pool and eviction bound are what actually carries that workload, not this floor. Setting this env var still **always wins over the FORMULA**, unchanged behaviour — it remains the override for "the formula is wrong for my workload right now". ⚠️ **It does NOT win over the 25% budget clamp described below:** the clamp applies to the whole planned ONEDNN zone (pair + whichever floor produced it, formula or override), so an override large enough to exceed 25% of the device's available budget is still clamped down to that 25% (floored at the primitive-API pair's own bare requirement), exactly as an unclamped formula result would be. This is not a correctness problem either way — the DIRECT path absorbs whatever either the formula or an override loses to the clamp — only the override's REACH is bounded, not its precedence over the formula. This value is extra headroom `unified_cache_get_planned_onednn_scratchpad_bytes()` adds, additive, on top of the primitive-API weights+activations estimate, to hold the Graph API SDPA allocator's (llama.cpp-gwno, `GGML_SYCL_ONEDNN_CACHE_ALLOCATOR` above) outstanding scratch demand. Why a flat floor was not safe to leave alone: a request that does not fit this zone takes the DIRECT (non-arena) allocation fallback in `onednn_graph_scratch_alloc()`, and under host CPU contention that fallback's bounded cap and wait (`GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB` below) can still be exhausted and abort — so under-sizing this zone trades a graceful, planned allocation for a bounded-but-still-fallible one. The planned zone (pair + this floor) is itself clamped to 25% of the device's available budget (floored at the primitive-API pair's own bare requirement, which has no DIRECT-path fallback of its own) if the shape-derived floor would otherwise demand more — a `[VRAM-ARENA]` `GGML_LOG_WARN` names the shortfall once when this triggers, and the DIRECT path absorbs the difference. ⚠️ **512 was sized for the wrong model** (historical, pre-0oxf): it assumed many compiled-partition buffers could be concurrently outstanding across a ubatch (their frees only bulk-drained once per ubatch) — true only while the free path deferred reclaim until each buffer's completion event fired. Once that blocking `event_complete()` check was removed from the zone-backed free path (`unified_cache::onednn_graph_scratch_free`'s in-order-queue comment — reclaim is immediate now, safe because every Graph-scratch consumer submits on the same in-order compute queue), at most ~1 buffer is outstanding at a time in practice. Measured high-water (`onednn_graph_scratch_high_water_bytes()`, logged once at cache teardown): 12.0 MB on a B70 gemma4 pp512 run, 3.0 MB on a B50 two-model (gemma4-Q8_0 + mistral-Q4_0) `GGML_SYCL_STRICT_LEASES=1` run. 512 was not just wasteful, it was an active landing blocker: a second model's load-time zone-growth request (its own weights+activations pair plus the 512 MB floor) could exceed what `ensure_planned_arena_zones()` can grow into while the first model's leases are still live, aborting with `[VRAM-ARENA] planned zones exceed active arena but live allocations prevent rebuild` on a run that passes on master (which never needs this zone to grow). Raise this (or let the shape-derived default do it) if `unified_cache::onednn_graph_scratch_alloc`'s "did not fit the ONEDNN zone" `GGML_LOG_WARN` fires, `GGML_SYCL_DEBUG=1` shows repeated distinct-size DIRECT-path request prints, or a multi-model/strict-leases run aborts sizing the VRAM arena. Only takes effect at planning time, before weights load — has no effect mid-inference. An unparsable or out-of-range value (including overflow, or anything above a 1 TiB sanity cap) is ignored and falls back to the shape-derived formula, logging one `GGML_LOG_WARN`; an explicit `0` is accepted as a genuine override, not treated as "unset", bypassing this formula's own floor entirely, and WARNs once too, so it is never mistaken for a value that was silently ignored. |
| `GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB=<MB>` | New in llama.cpp-0oxf. Caps outstanding bytes in the DIRECT (non-arena) Graph-scratch fallback (`onednn_graph_scratch_alloc()`'s path taken when a request does not fit the ONEDNN zone above) — for many models this is the STEADY-STATE path, not an occasional fallback, because the zone is sized at model load against a conservative `n_ctx` and the real runtime `n_ctx` arrives too late to resize it (see the ZONE_MB row's ⚠️ and `docs/backend/sycl-memory-design.md`'s "cannot actually reach the DIRECT path's steady state" note). ⚠️ **Semantics changed after this variable's introduction**: a freed DIRECT buffer is no longer handed to the shared background drain worker in `mem-handle.cpp` — it is parked in a size-bucketed reuse pool (`onednn_graph_scratch_reuse_pool_`) and handed back to the next request of the identical size instead of a fresh `unified_alloc()`/`zeMemAllocDevice` round trip, since SDPA calls of a fixed shape repeat across `-r N` benchmark reps and decode steps. This cap still bounds the pool: a pooled-but-idle buffer's bytes stay charged against it (it is still real resident VRAM, just idle), so pool growth competes with fresh allocations for the same headroom. Default: the smaller of 1 GiB and 25% of `available_budget()` **snapshotted the last time this device's arena was successfully (re)planned** (`ensure_planned_arena_zones()`) — not live budget, so the cap does not shrink out from under the allocator as the arena's own zones consume it; falls back to a flat 1 GiB if the arena was never planned for this device (e.g. arena disabled). This is a SEPARATE 25% from the ZONE_MB row's own budget clamp on the planned zone SIZE — one bounds how big the static zone is allowed to be, this one bounds how much can pile up (checked out plus pooled) in the DIRECT fallback at runtime. Before allocating past the cap for a size that misses the pool, the allocator evicts (a real release) completed pool entries -- of any size, including the requested size's own bucket (a same-size entry that is complete but fails this request's alignment is exactly the kind of entry this can evict) -- until it fits — preserving as much of the pool as possible — waiting (bounded, with a heartbeat so the watchdog does not fire) for an in-flight pool entry to complete first if none are immediately evictable. A single request larger than the whole cap by itself is a distinct early-out (llama.cpp-pqgl): the eviction sweep above still runs and releases whatever it genuinely can, but the bounded wait is skipped entirely — logged once — since no amount of additional waiting could ever make it fit, and the allocation is then attempted directly, the same as after a timed-out wait below. A wait that does not resolve within the total timeout is logged and the allocation is attempted anyway rather than refused pre-emptively -- that timeout is measured from this call's own entry, before the pre-loop eviction sweep and the oversized-request early-out above both run, not from the first poll iteration, so time spent in the sweep already counts against it. If the underlying `unified_alloc()` then genuinely fails (even after one drain-and-retry), the allocator logs the sizes and `GGML_ABORT`s — it no longer returns a null scratch pointer to oneDNN, which is what let a prior version of this code execute the SDPA kernel against invalid memory (a GPU page fault / engine reset / segfault on the B50 under host load; the root cause llama.cpp-0oxf fixes). Raise this only after checking why the DIRECT path is engaging at all (see the ZONE_MB row above) — a bigger cap widens the window before the abort fires, it does not remove the underlying zone under-sizing. `onednn_graph_scratch_direct_wait_count()` reports how often a run actually waited for eviction, and `onednn_graph_scratch_pool_hit_count()` reports how often a request was served from the pool instead of a fresh allocation. The pool is reclaimed (real release of every entry) at cache teardown, at `arena_reserve()`'s context-reclaim branch, at `ggml_backend_sycl_set_runtime_context()` on every successful runtime update (this one does not go through `arena_reserve()` at all, so it needs its own reclaim call), and, per llama.cpp-me60, once per live cache from `shutdown_unified_cache()` itself, unless SYCL is already shutting down or that cache's queue context is already invalid (llama.cpp-3lgu), immediately BEFORE its own pre-teardown census (a parked DIRECT buffer's own `EXTERNAL_EXACT` allocation control would otherwise refuse that census). The four production sites do not all log in the same order relative to the clear: the context-reclaim, runtime-update, and pre-census sites share `reclaim_pool()`, which clears the pool and only then logs the summary; teardown instead logs the summary early — right after the high-water WARN, well before either of its early-return branches — and does not actually clear the pool until much later, after `drain_all_queues_noexcept()` has synced every queue (not the shared background worker's retained-handle drain -- `shutdown_resources()` has no such drain; it is `drain_all_queues_noexcept()` that guarantees every pooled entry's release event is complete by the time the clear runs). Either way the line logged is `[UNIFIED-CACHE] oneDNN Graph scratch DIRECT pool summary (%s): hits=%zu misses=%zu evictions=%zu waits=%zu peak_pooled=%.1f MB retired_flag_slots=%zu (cumulative for this process, not just this reclaim)` (silent if the pool was never used). Only the teardown call logs at `GGML_LOG_LEVEL_WARN`; the other three calls (context-reclaim, runtime-context-update, and module-shutdown pre-census) log at `GGML_LOG_LEVEL_INFO`, which is dropped at default verbosity in every tool (see CLAUDE.md's "llama-bench traps" section), so those three summaries do not reach a normal run's log. |
| `GGML_SYCL_NONFA_ATTN_SCRATCH_MB=<MB>` | New in llama.cpp-oyfl. Overrides `unified_cache_nonfa_attn_scratch_demand_bytes()`'s shape-derived floor for the non-flash-attention batched mul_mat path's scratch demand `d`: `d = max(16 MiB, 3 x n_head x n_ubatch x n_ctx x sizeof(f16))`. The `c=3` concurrency factor is MEASURED from the llama.cpp-oyfl repro log's own SCRATCH_ZONE occupancy at the moment of failure (460 MB of 512 MB, plus a 256 MiB request — ~2.8x, not the oneDNN sibling's c=1.5 an earlier version of this formula copied by analogy and which under-covered the repro). This consumer is unconditional (not gated behind `GGML_SYCL_DNNL`), and it feeds `d`, the demand term of `ggml_backend_sycl_set_runtime_context()`'s runtime-context-update refusal (llama.cpp-pvjr): refuse iff `d + R > free`, where `R` is the EMPIRICAL `unified_cache_nonfa_attn_outside_arena_reserve_bytes()` = **928 MiB** and `free` is the device's LIVE free memory read at guard time. It does not feed the plan-time zone-raise the ONEDNN sibling above uses (which cannot take effect automatically for the standard model-load flow either — `llama-model.cpp` hardcodes the pre-load envelope's `n_ctx` to 0; piping the real context size back that early is llama.cpp-fkpg's scope). ⚠️ **This is NOT a working remediation for a refused context, and the runtime refusal does not suggest it.** It only ever replaces `d`, never `R` or the headroom comparison itself, so it cannot make a genuinely over-budget shape fit — use it only to experiment while investigating llama.cpp-k1ev (the B50's own still-unattributed ~0.1–0.9 GB of extra outside-arena consumption, apparently absent on the B70 over the measured range), not as advice to give a user hitting the refusal; `-fa 1`/`auto` or a smaller `-c` (the refusal's own largest-fitting estimate, labeled **"headroom-limited"**) are the only remediations with hardware support. See `docs/backend/sycl-memory-design.md`'s "A non-tensor consumer" subsection for the full history, including the 2026-09-09 sweep that replaced the earlier zone-capacity predicate with this headroom-based one. An explicit non-negative value always wins over the formula. An unparsable or out-of-range value (including overflow, or anything above a 1 TiB sanity cap) is ignored and falls back to the shape-derived formula, logging one `GGML_LOG_WARN`; an explicit `0` is accepted as a genuine override, not treated as "unset" — it disables the runtime-context-update refusal outright (`unified_cache_nonfa_attn_scratch_guard_disabled()` returns true and the guard returns before any comparison at all), WARNing once, rather than merely zeroing `d` and still being checked against the reserve and live free memory — and, because the override also feeds the plan-time raise in `ensure_planned_arena_zones()`, an explicit `0` additionally suppresses the shape-derived SCRATCH-zone raise that an unset variable would have produced. |
| `GGML_SYCL_ONEDNN_GRAPH_POOL_DEPTH_PER_SIZE=<N>` | New in llama.cpp-0oxf. Caps how many idle entries the reuse pool may hold PER DISTINCT SIZE (default **8**) before `onednn_graph_scratch_free()` starts releasing overflow for real (via the same event-gated drain path the pool itself superseded) instead of parking it. This is a SECOND, orthogonal bound from `GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB` above: that one bounds total pooled+checked-out bytes across ALL sizes; this one bounds how many buffers of ONE size can accumulate, which matters for a workload that walks many distinct sizes (a pp8192 run touches ~16 distinct ne11-derived shapes) where the byte cap alone would not prevent one popular size from crowding out every other size's reuse opportunity. 8 is a generous multiple of the "~5 SDPA scratch buffers concurrently in flight" measurement the zone floor's own 1.5x factor is based on (see the ZONE_MB row) — headroom for noise without keeping every historical shape's buffers alive forever. |
| `GGML_SYCL_ONEDNN_GRAPH_TEST_HOOKS=1` | **Test-only, default off.** New in llama.cpp-0oxf. Gates `ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail()` and `ggml_sycl_test_onednn_graph_scratch_suppress_abort()` (declared in `unified-cache.hpp`) -- both setters are no-ops unless this is set, so nothing running in a production process can flip either behavior by calling these always-compiled, exported symbols. Also gates `ggml_sycl_test_onednn_graph_scratch_force_blocking_pool_check()` (llama.cpp-c6ah): while forced, `onednn_graph_scratch_free()`'s park site skips arming the completion flag (a device marker kernel writing a host-USM slot) for the entry it is parking, so that entry's completion check falls back to a direct `event_complete()` query on its `release_event` instead -- reproducing, on demand, the pre-fix behavior of waiting for an in-flight SDPA kernel instead of skipping it, so a test can measure the difference between the two. Two tests set it: `tests/test-sycl-onednn-graph-scratch-direct.cpp`'s ctest registration sets it via `ENVIRONMENT` (that same test also sets it itself for a bare, non-ctest invocation, since a direct run would otherwise silently no-op the hooks and false-fail), and `ggml/src/ggml-sycl/tests/test-unified-runtime-alloc.cpp` sets it directly in `main()` (llama.cpp-me60, so its flag-slab shutdown cases can poll the completion flag via `onednn_graph_scratch_pool_entry_flag_true_for_test()`). Never set this outside those two tests. |

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
| `GGML_SYCL_GRAPH_REPLAY_PROBE=1` / `GGML_SYCL_GRAPH_REPLAY_PROBE_LIMIT=<N>` (default 4) / `GGML_SYCL_GRAPH_REPLAY_PROBE_NAMES=<comma list>` | Default OFF (llama.cpp-dyi3, `ggml-sycl.cpp`). Decisive-experiment probe for the decode FA graph-replay divergence: for the first N decode-phase `graph_compute` calls, substring-matches cgraph tensor names against the (overridable) name list and prints one `[GRAPH-REPLAY-PROBE]` line per match — op, `ne`/`nb`, `data=`/`dev_ptr=`/`view_root=`/`view_offs=` pointers, `nelements`/`finite_count`, `sum`/`abs_sum`/`sq_sum` (finite-values-only, same convention as `common/debug.cpp`'s `common_debug_cb_eval`), and the first 4 decoded values. Run once with `GGML_SYCL_DISABLE_GRAPH=1` and once with `GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW=1`, diff by matching `call=N` — the first tensor whose `sq_sum`/pointer fields differ at the same call names the un-refreshed input. Kept as a permanent diagnostic, not removed once the bug is found. |
| `GGML_SYCL_GRAPH_CENSUS=1` | Default OFF (llama.cpp-dyi3, `mem-ops.cpp`). Read once and cached. When set, `mem_copy_direct_submit` warns once per call site, `[SYCL-GRAPH-CENSUS] mem_copy_direct_submit (non-capturable) called during recording at <file>:<line> (<function>)…`, whenever a copy reaches this non-graph-capturable path (profile label `sycl.memcpy.mem_ops`) while `ggml_sycl_graph_recording_active()` is true — such a copy executes once at record time and is never re-issued on replay. Was unconditional through round 6, which meant any model whose SYCL command graph engages at all printed it in ordinary runs; now opt-in. |
| `GGML_SYCL_OP_TIMEOUT_MS=<N>` | Abort with diagnostic if no inference progress for N ms (default 30000, set to 0 to disable). Fires before the xe driver's 10s GT reset cascade. Effective detection latency is `timeout + ~500 ms`. |
| `GGML_SYCL_SAFE_MODE=1` | Drain the SYCL queue after every op submit so a fault surfaces at the op that caused it (2-3x slowdown, implies `GGML_SYCL_DISABLE_GRAPH=1`). Useful for CI canaries and correlating intermittent hangs 1:1 with their triggering op. |
| `GGML_SYCL_FUSION_BIT1_REACH_DEBUG=1` | Default OFF (`ggml-sycl.cpp:83907`). Logs one `[FUSION-BIT1-GUARD]` line per bit1 fusion candidate: `is_weight`, the accumulated `view_offs`, `nb[0]`, element size, the per-operand `safe` verdict, and the full gate chain. Built for the gemma3n cross-model investigation (`llama.cpp-8t4s`), and still the way to answer "did the 81gx view-offset guard classify this operand correctly?" — in the healthy 2-arch run at HEAD, all 44 of gemma3n's unsafe views log `is_weight=0` and are refused. Diagnostic only; the block is guarded by the flag. |
| `GGML_SYCL_HANDLE_STRICT=1` | Default OFF; reports `ggml_tensor_extra_gpu` `data_handle`/`data_device` divergence (first 16 only) without needing `GGML_SYCL_DEBUG=1`. Diagnostic only, no perf effect. Plan: `docs/plans/2026-07-30-extra-device-indexed-handle-storage.md`. |
| `GGML_SYCL_EXTRA_LEAK_PROBE=1` | Default OFF (`ggml-sycl.cpp`, `ggml_backend_sycl_buffer_reset`'s COMPUTE-buffer branch; llama.cpp-dfo0 plan tasks L1/L2). At `GGML_LOG_WARN`, one `[EXTRA-LEAK-PROBE] buf=%p kept=%zu extras this call (~%.1f MB), sum_of_sizes=%zu @ sizeof=%zu released=%zu gen=%llu` line per compute-buffer reset. Field semantics: `buf=` this call's single buffer (`ggml_vbuffer_reset` resets multiple chunks and `ggml_gallocr_alloc_graph` loops over buffer types, so several `buf=` values can appear per graph allocation); `kept=` the post-release size of THIS buffer's `tensor_extras` vector — the figure to compare against RSS growth, and what should stay BOUNDED across rebuilds rather than grow every one (the L1/L2 fix this probe verifies); the `~%.1f MB` figure is `kept * sizeof(ggml_tensor_extra_gpu)`, this call's own bytes only, divided by `1024.0 * 1024.0` in the `[EXTRA-LEAK-PROBE]` `GGML_LOG_WARN` call in `ggml-sycl.cpp` (cited by the format string rather than a line number, which drifts) — the printed unit is MiB wearing an `MB` label, not decimal MB; `sum_of_sizes=` is a PROCESS-WIDE running total of `kept` across every call (one function-local `static std::atomic`, all compute buffers) — trailing diagnostic context only, never quote its MB equivalent as "leaked" and never compare it against RSS growth (it only ever grows); `sizeof=` is `sizeof(ggml_tensor_extra_gpu)` read live, so it self-reports the post-h9uv-split size (25,048 B) without needing a separate update when the struct changes — see S1/S2, llama.cpp-aenv, and `ggml/src/ggml-sycl/tests/test-sycl-extra-gpu-size.cpp`; `released=`/`gen=` are the same per-call counters `GGML_SYCL_DEBUG`'s adjacent `[SOA-DEBUG]` line already prints, repeated here so a `-v` capture (needed to see `GGML_LOG_WARN` under `llama-bench`; see the CPU-fallback-blindness section above) does not require enabling both. `kept` and the `n` used for the MB figure are the SAME NUMBER by construction (read right after the release loop) and are printed once, as `kept=`, not twice under two names. Used by: the L1 probe (this variable's original purpose), the L2/L2b compute-buffer- and KV-view-extra-reuse GPU tests (`tests/test-sycl-compute-buffer-extra-reuse.cpp`, `tests/test-sycl-kv-view-extra-reuse.cpp` — both read RSS/counters independently and do not require this flag, but a hardware capture with it enabled is how the L1/L2 fixes were originally verified), and llama.cpp-h9uv's own acceptance criterion (the per-reset MB probe this variable provides, referenced directly in `test-sycl-extra-gpu-size.cpp`'s header comment). The release loop that fixes the underlying leak always runs regardless of this flag — it is the correctness fix, not a diagnostic; this flag only gates the logging and the atomic bookkeeping above it, so the unset-env hot path costs one magic-static guard load plus a cached bool test. |
| `GGML_SYCL_MOE_LAYOUT_DEBUG=1` | Emit the `[MOE-LAYOUT]` per-pass summary unconditionally. The down-i8 / gateup-i8 lines already fire on ANY decline without this; the variable adds the lines a fully-successful pass would otherwise not print. |
| `GGML_SYCL_STORED_GEMM_DEBUG=1` | Default OFF (`mxfp4-stored-gemm.cpp`, llama.cpp-kcya round 6, the non-dispatched stored-SOA small-M MXFP4 GEMM kernel). Prints one `[mxfp4-stored-gemm]` launch-geometry line per `ggml_sycl_mxfp4_soa_gemm_dpas` call, all ten fields in the order printed: `M_TILE` (1/2/4/8), `n_out`, `n_k`, `compute_units` (as queried for the K-split heuristic below), `n_tiles` (N-tiles, one work-group per 16 output rows), `k_tiles`, the resolved `ksplit`, `partial_work_items` (`n_tiles * ksplit`, always launched), `combine_work_items` (`n_tiles` when `ksplit > 1`, else **0** — the combine kernel is never submitted on the `ksplit <= 1` direct-write path, and this print says so rather than advertising a launch that does not happen), and `k_tiles_per_partition(max)` — the size of the FIRST `k_tiles % ksplit` partitions; the remaining partitions process one tile fewer (e.g. at the B50's default geometry, k_tiles=90/ksplit=12, six partitions process 8 tiles and the other six process 7 — the printed max is the size of that first group, not "every partition but the last"). Built for comparing this kernel's launch shape against the production reference kernel's (`mxfp4_pair_glu_xmx_tiled_dpas_m2`, `mmvq.cpp`) without a GPU-side profiler. Diagnostic only; no perf effect when unset. Sibling `GGML_SYCL_STORED_GEMM_KSPLIT=<N>` overrides the K-split factor this print reports: accepted only when it parses as an integer AND is `> 0`, then clamped at the UPPER end to `k_tiles`; a non-positive or non-numeric value is IGNORED outright (the computed heuristic runs instead), it is not clamped up to 1. **Profiler `bytes` reconciliation** (llama.cpp-kcya c-amir should-fix 3): with `GGML_SYCL_KERNEL_PROFILE=1`, this kernel emits two rows, `mxfp4.stored_gemm.soa.partial` and `.combine`. On the `ksplit <= 1` path the single `.partial` row's `bytes` equals the useful weight + activation + dst traffic — the SAME figure a caller-side `bytes_moved` calculation (e.g. the test's own) would compute. On the `ksplit > 1` path the two rows' `bytes` ALSO include the K-split's internal scratch round-trip (written by `.partial`, read back by `.combine`) — real DRAM traffic this dispatch causes, so it belongs in the profiler's own accounting, but not "useful" work a caller cares about, so a caller-side `bytes_moved` figure deliberately excludes it. Do not sum the two rows' `bytes` and expect a caller's `bytes_moved` to fall out on that path; do still sum their `mean_ns` for one logical launch's total device time (see `mxfp4-stored-gemm.cpp`'s dispatcher comment). This variable belongs to llama.cpp-kcya, which carried this kernel's `>= 50%-of-peak` bandwidth criterion forward to **llama.cpp-wjqn** (the lane-contiguous/group-repacked layout variant this needs) after two hardware rounds (checkpoint: B50 M=8 N=K=2880 237 us -> ~109 us, round 7's instruction-count levers refuted on `spike/kcya-round7`) rather than closing it on kcya itself. |
| `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS=<N>` | Hard cap on how many down tensors the MoE I8 layout pass upgrades. Unset (or negative) = no cap, the shipping behaviour; `0` disables the upgrade. **Diagnostic only — do not set in production.** See the measured cost below. |
| `GGML_SYCL_ARENA_PP_PROFILE=1` | Emit `[ARENA-PP-*]` counters, including `[ARENA-PP-ONEDNN] … reserve_req_mb=W/A` — the summed oneDNN weights/activations reservation requests. This is the **only** log that reports what was actually asked for, as opposed to what was planned. |
| `GGML_SYCL_ZONE_RESET_AUDIT=1\|2` | Default OFF. Phase 0 of the retire-zone-reset epic (llama.cpp-iiff): every reset/drain site reports what is still live in its zone as `[ZONE-RESET-AUDIT]` lines, with the allocation's own `alloc_id`/`cohort`/`role`/`category` attribution. `=1` changes no behaviour and is safe across the whole gate set. `=2` additionally suppresses the reset even when the zone is clean — host SCRATCH/STAGING are reset-only by design, so that leaks without bound. See below. ⚠️ **The variable name and its site strings (`host-zone-reset/*`, `scratch-pool-reset/bump`, `device-zone-reset/*`, `weight-reclaim/*`) were deliberately NOT renamed when `llama.cpp-37ba` renamed the underlying API to `*_boundary_check` / `*_reclaim` / `scratch_pool_epoch_boundary`.** They are stable historical identifiers that keep captures 01–17 baseline-comparable. They look stale against the current code; they are not. Do not "fix" them. |
| `GGML_SYCL_EXT_ALLOC_TRACE=1` | Default OFF (`unified-cache.cpp`). Logs one `[EXT-ALLOC] device=… bytes=… cohort=… role=… category=… prefer_vram_zone=… total_external=…` line whenever a device allocation escapes the arena's per-zone accounting — a `prefer_vram_zone==COUNT` request, or a full/inactive zone falling through to the raw `unified_cache_malloc_device_tracked` device malloc. `total_external` is a running byte total (process-lifetime, function-local `static`), so a capture doubles as a drain profile of what is bypassing the zone system. `role`/`category` are the raw `alloc_role`/`runtime_category` enum ordinals (see `unified-cache.hpp`); `cohort` is `?` when the caller left `alloc_intent::cohort_id` unset. Read once and cached; zero cost when unset beyond the one atomic load. Diagnostic only. |
| `GGML_SYCL_KERNEL_PROFILE=1` | Default OFF. Aggregate per-kernel-label device-time census (`sycl-kernel-profiler.cpp/.hpp`). Sub-variables: `GGML_SYCL_KERNEL_PROFILE_OUTPUT=<path>` (CSV/JSON out), `GGML_SYCL_KERNEL_PROFILE_FORMAT=csv\|json\|both`, `GGML_SYCL_KERNEL_PROFILE_RAW=1` (per-event rows with `timestamp_status`). Coverage, status semantics and the partial-event caveat: see below. |
| `GGML_SYCL_E2E_TG_PROFILE=1` | Default OFF (`e2e-profile.cpp/.hpp`). Per-stage (DISPATCH/MOE/ATTENTION/KV/ELEMENTWISE/GRAPH/CACHE/TRANSFER/…) **host-time** breakdown, one `[SYCL-E2E-TG-PROFILE]` summary + `[SYCL-E2E-TG-STAGE]` rows per graph_compute (one decode token or one prefill ubatch). Its `device_us` column is host-only — see below. |
| `GGML_SYCL_MXFP4_PP_PROFILE=1` | Default OFF (llama.cpp-6405, perf-recovery epic PP gap analysis sub-task). Device-event breakdown of the batched PP MoE oneDNN executor (`try_pp_mxfp4_soa_onednn_f16_batched`, `ggml-sycl.cpp`; only compiled under `GGML_SYCL_DNNL`). Prints `[MXFP4-PP-BATCHED-PROFILE]` (one per graph eval) and `[MXFP4-PP-BATCHED-PROFILE-TENSOR]` (per-tensor detail) lines at `GGML_LOG_WARN`, partitioning each eval's device time into repack/GEMM/staging/`other` buckets. Reuses the same env var name `mmvq.cpp`'s `mmvq_moe_pp_profile_enabled()` already reads for a complementary, older instrument timing the small-batch MMVQ GEMV MoE path. See below for the exact line format, what `other` does and does not mean, the graph-replay caveat, and the host-wait/zero-cost contract. |

### `GGML_SYCL_KERNEL_PROFILE` — reading a capture

Reads real `sycl::event` device timestamps (`command_start`/`command_end`) via a
deferred/pending-event flush, not at the submit site, so it adds no host wait
to the dispatch path. Aggregate rows are keyed on `{name, category, metadata}`;
`GGML_SYCL_KERNEL_PROFILE_RAW=1` adds per-event rows carrying `timestamp_status`:
`ok` for a real device-event read, `host_span_only` for a call site that has no
`sycl::event` to attach to and records a host wall-clock span instead
(`ggml_sycl_kernel_profile_record_host_span`). A site whose returned event covers
only part of the real work — the oneDNN Graph SDPA execute in `fattn-onednn.cpp`,
whose event's profiling window is the fused pattern's last kernel — still records
through `record_event` with status `ok` and carries the host submit span on the
same sample: read its device time as a lower bound and its host span as the
per-call cost. One record call per submit; recording a device event and a host
span separately under one label double-counts in the aggregate rows.

Coverage as of llama.cpp-qmen: every decode MUL_MAT kernel family; the dense and
MXFP4-batched oneDNN GEMM executors (`mulmat.onednn_gemm.execute`,
`mxfp4.pp.gemm.execute`); the un-batched oneDNN GEMM primitive
(`DnnlGemmWrapper::gemm`, label `mulmat.onednn_gemm.unbatched` — the
Q8_0/Q4_0 dequant-then-GEMM route most prefill Q8_0 M≥512 GEMMs take); the full
RMS_NORM family (`norm.rms_norm`, `norm.rms_norm_mul`, `norm.rms_norm_mul_add`,
`norm.add_rms_norm`) plus `norm.norm` / `norm.group_norm`. `l2_norm` is
deliberately unwrapped.

Zero overhead when unset for the qmen wraps: the per-call metadata string is
built only under `ggml_sycl_kernel_profile_enabled()`. Older per-batch-element
sites (e.g. `mxfp4.pp.gemm.execute`'s 2-D fallback loop) predate that convention
and still build their metadata unconditionally — a small per-call heap
allocation with the profiler off; not fixed here.

#### `failed_timestamps` / `graph_recorded` — one number, two causes today (S5, llama.cpp-aenv; root cause llama.cpp-mmbg c-x4it)

`failed_timestamps` (CSV/JSON per-row column, `sycl-kernel-profiler.cpp`) counts
a launch for which no usable device timing was obtained. It increments from
exactly four production sites in `flush_pending_events`, ALL of which hardcode
`graph_recorded=false` regardless of what actually happened:

  - `status_error` — the pre-wait `command_execution_status` query threw.
  - `wait_error` — `event.wait_and_throw()` threw.
  - `invalid_range` — `command_end < command_start` (a real device-timestamp
    anomaly).
  - `query_failed` — `command_start`/`command_end`/`command_submit` retrieval
    itself threw, after the wait already succeeded.

These four causes are genuinely different (a queue/device fault vs. a garbled
but present timestamp vs. an event that simply carries no profiling info at
all), but the CSV's `failed_timestamps` integer and its neighboring
`graph_recorded` boolean (`format_csv_rows`/`format_json_rows`) do not
distinguish them in any real capture, because every production call site that
increments the counter passes the same hardcoded `false` for `graph_recorded`.
The ONLY caller anywhere in the tree that ever passes `true` is the test hook
`ggml_sycl_kernel_profile_add_failed_timestamp_for_test`, used solely by
`tests/test-sycl-kernel-profiler.cpp`. So `graph_recorded` is always `0` in
every production/hardware capture today, and cannot currently be used to tell
apart a genuine instrument failure from the shape below.

**The dominant real-world cause is not an instrument failure at all.**
llama.cpp-mmbg (root cause comment c-x4it) found that with SYCL command-graph
replay enabled (the default), every `mulmat.mmvq.*` decode row reports
`failed_timestamps == count/2` — e.g. `count=128 failed_timestamps=64` — while
`GGML_SYCL_DISABLE_GRAPH=1` on the identical workload gives `failed_timestamps=0`
at 4.5x the `count` (576 vs 128), with IDENTICAL `mean_ns` across both arms. The
reading: replayed graph launches are not counted by the profiler at all (a
separate, already-documented gap — see `kernel-profiler-count-excludes-
recorded-kernels`), so what the profiler DOES see, with graph replay on, is only
the warm-up pass that RECORDS the command graph — half direct submissions
(timestamped normally, `ok`) and half record-mode submissions whose
`sycl::event` carries no usable profiling info (one of the four causes above,
almost always `query_failed` or `invalid_range` in practice). The half that
fails is not broken instrumentation and not a device/queue fault; it is the
graph-recording pass being counted as if it were an ordinary timestamped
launch. `failed_timestamps == count/2` on a graph-replay-eligible kernel family
is this shape, not noise, and not evidence of a profiler regression.

**Consequence for reading any capture:** you cannot currently tell, from the
CSV/JSON alone, whether a given `failed_timestamps` count is (a) the expected
record-pass shape above, (b) a genuine device/queue-level timing failure, or
(c) some mix, because `graph_recorded` never reaches `1` in a production run to
disambiguate. As a practical proxy: `failed_timestamps` at or near `count/2`
with `mean_ns` otherwise stable and plausible is almost certainly (a); a
`failed_timestamps` count with no such clean fraction, or accompanied by
implausible/zero `mean_ns`, warrants treating it as (b) and reproducing with
`GGML_SYCL_DISABLE_GRAPH=1` to rule the graph-recording pass out (as the mmbg
capture above did).

**This documents the CURRENT semantics only; it does not fix them.** The code
fix — plumbing the actual graph-capture-active state into the four production
call sites so record-pass launches are tagged `graph_recorded=1` and excluded
from `failed_timestamps` rather than counted as failures, after which
per-token attribution needs the separate replay count — is tracked on
llama.cpp-mmbg and intentionally NOT made here; this task (llama.cpp-aenv) is
documentation only.

### `GGML_SYCL_E2E_TG_PROFILE` — the `device_us` column is host-only

No call site supplies a SYCL device timestamp. The accumulator and its printed
field work (`test-sycl-e2e-profile.cpp` feeds a synthetic non-zero value end to
end), and the one non-zero producer in the tree is a host-clock span —
`e2e_tg_profile_record_transfer("peer_host_bounce_measure", …)` passes the chrono
time of a host-mediated peer-link bounce copy as `device_us` — so the column is
not universally `0.0`, but it is never device time. Reading event profiling info
at the per-op `e2e_tg_scope` bracket would be an implicit host wait on an
incomplete event, which the no-host-waits rule forbids on the dispatch path (the
`assert_no_waits` source guards in `tests/test-sycl-e2e-profile-*-source.py`
enforce it). Use `GGML_SYCL_KERNEL_PROFILE` for device time; use this instrument
for the host-side stage split (`total_host`, per-stage `host`, and `wall`).

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

### `GGML_SYCL_MXFP4_PP_PROFILE` — reading a capture

Device-event breakdown of the batched PP MoE oneDNN executor
(`try_pp_mxfp4_soa_onednn_f16_batched`, `ggml-sycl.cpp`; only compiled under
`GGML_SYCL_DNNL`). Mechanism is copied from the proven decode instrument
`GGML_SYCL_MXFP4_TG_PROFILE` (`mmvq.cpp`): real SYCL device-event profiling
(`get_profiling_info<command_start/command_end>`), not host chrono deltas. No
queue-creation step is needed to activate profiling — every ggml-sycl GPU
queue already carries `sycl::property::queue::enable_profiling` via
`common.hpp`'s `default_queue_properties()` (confirmed by `unified-cache.hpp`'s
`dma_queue_` comment: "every backend stream is created with...
`enable_profiling`").

#### Line format

One `[MXFP4-PP-BATCHED-PROFILE]` line per graph eval (one
`ggml_backend_sycl_graph_compute()` call, i.e. roughly one ubatch — printed
only when that eval dispatched the batched executor at least once, via
`mxfp4_pp_batched_profile_record_graph_total`) partitioning the eval's whole
device time:

```
[MXFP4-PP-BATCHED-PROFILE] device=<N> dispatches=<N> graph_total=<ms> accounted=<ms> other=<ms>
  tiled_repack=<ms>/<calls> soa_repack=<ms>/<calls>
  woq_gemm=<ms>/<calls>(2d=<N>,3d_requested=<N>) f16_gemm=<ms>/<calls>
  stage=<ms>/<calls> graph_total_measured=<0|1> read_failures=<N>
```

(wrapped here for readability; the real line is one line). `device` is the
SYCL device ordinal, captured once at graph-compute entry alongside the reset
below — attributes a multi-device capture to the right card on both line
families. The buckets:

- `tiled_repack`/`soa_repack` — the batched repack kernels
  `repack_mxfp4_xmx_tiled_to_woq_coalesced_batched` /
  `repack_mxfp4_soa_to_woq_coalesced_batched` (`convert.cpp`; the coalesced
  forms since llama.cpp-0vqt `a89cfba79` — each falls back internally to its
  pre-coalesced `_batched` sibling when its shape gate fails), one begin/end
  marker bracket per **tensor-dispatch** since llama.cpp-1lon (`c6ecc70aa`)
  — the whole active-expert set is one kernel launch, so the `/calls` count
  reads ~46 tiled + ~23 soa per GPT-OSS eval.
  Before 1lon the per-slot kernels ran one launch (and one bracket) per active
  expert per dispatch, so pre-1lon captures (e.g. the `892eb5d7d` baselines)
  read ~1064/~532 calls per eval instead — the counts are not comparable
  across that commit, only the ms figures are. The per-slot form still exists
  as a fallback, taken only when a dispatch has more than 128 active expert
  slots (unreached by any shipped MXFP4 model); on that path the bracket is
  again per-slot.
- `woq_gemm` — `DnnlGemmWrapper::woq_gemm_batch_mxfp4` (the `GGML_SYCL_MOE_PP_WOQ`
  arm). ⚠️ **Timing is a begin/end marker bracket around the whole GEMM `try{}`
  block, NOT `gemm_events`' own profiling info** (fix cycle F1, confirmed on
  hardware before the fix: `woq_gemm=0.000 ms/459-462` on both B70 and B50
  profiling runs). The shipped 2-D WOQ arm (`gemm.hpp`'s `woq_gemm_batch_mxfp4`
  fallback) returns `q->ext_oneapi_submit_barrier(per_batch_events)` — a
  barrier event whose `command_start`/`command_end` cover the near-instant
  barrier command itself, not the batched GEMM executes it merges, so reading
  `gemm_events`' own timestamps silently pushed the entire WOQ GEMM device time
  into `other`, inverting the parent task's question. `gemm_events` is still
  consulted, but only for its per-group count (the `2d=`/`3d_requested=` split,
  by the env-*requested* 2-D/3-D primitive — see `GGML_SYCL_MOE_PP_WOQ_3D`
  above; the device can still silently fall back 3-D→2-D per `gemm.hpp`'s
  per-device `tri_state` cache, which this instrument cannot observe, so the
  split is a request label, not a guaranteed-executed one), never its
  timestamps. A subtler risk in the SAME marker-bracket mechanism: the GEMM
  calls depend explicitly on their own `deps_in` barrier, not on the begin
  marker — the marker is ordered after `deps_in` only by this in-order
  queue's submission order, not by an explicit dependency edge. This file
  already documents (llama.cpp-dboi) that the batched oneDNN primitive's
  execution is not always reliably ordered with this queue by submission
  order alone; if that ever applies to the marker/GEMM pair too, the recorded
  span would UNDERCOUNT the true GEMM time, with the missing slice landing in
  `other` rather than being lost outright.
- `f16_gemm` — `gemm_batch_strided`, the `GGML_SYCL_MOE_PP_WOQ=0` arm. Timed the
  same begin/end-bracket way as `woq_gemm`. **The WOQ=0 arm's per-expert f16
  dequant (`dequantize_row_mxfp4_soa_to_fp16_rowmajor`) is NOT separately
  timed** — only its GEMM is — so that dequant's device time is unmeasured and
  lands in `other` too on any capture taken with `GGML_SYCL_MOE_PP_WOQ=0`.
- `stage` — the per-expert activation copy-in
  (`k_copy_src1_to_contiguous_f16_mapped`) + output scatter-out
  (`k_copy_dst_from_contiguous`) kernels, one event per active expert per loop
  per dispatch.
- `other` — `graph_total` minus the four named buckets above, printed
  unclamped/signed on purpose: a negative value is itself a signal (some
  device work is being double-counted, overlapping, or — see below — running
  on a queue this instrument never brackets), not something to hide by
  flooring at 0.
- `read_failures` — every `*_us` sum above excludes a failed profiling-info
  read (`mxfp4_pp_event_duration_us`/`_span_us` returning `-1.0` — a bad event
  or a query that threw) rather than being poisoned by it; the failed
  operation's own CALL count still increments (the op did happen), and the
  failure is tallied here instead. The same applies to the graph-total span
  itself: on a failed read, `graph_total_us` stays `0.0`,
  `graph_total_measured=0`, and `read_failures` is incremented, rather than
  printing `graph_total_measured=1` on the exact failure that field exists to
  flag. Nonzero `read_failures` means some of the ms figures on that line are
  undercounts, not that an operation failed.

A second line family answers the epic's suspected-waste question directly —
whether a tensor's WOQ repack re-runs more than once per graph eval
(per-ubatch/per-dispatch) instead of once per layer:

```
[MXFP4-PP-BATCHED-PROFILE-TENSOR] device=<N> tensor=<name> dispatches=<N> repacks=<N> repacks_per_dispatch=<float>
```

`repacks_per_dispatch` is `repacks / dispatches` for that tensor this eval,
printed for tensor names in sorted (deterministic) order so two captures of
the identical eval never diff spuriously on ordering alone, capped at the
first 96 distinct tensor names with a truncation line if more were seen.

⚠️ **Semantics changed at llama.cpp-1lon (`c6ecc70aa`).** `repacks` counts
marker brackets, and the batched repack kernels bracket once per
tensor-dispatch — so on the batched path `repacks_per_dispatch` reads **1.00
by construction** and no longer measures the active-expert count. Pre-1lon
captures counted one repack per active expert slot, so the `892eb5d7d`
baselines read mean 23.33 (min 15, max 32 = the active-slot distribution);
do not compare that figure across the commit, and do not read a post-1lon
1.00 as "one active expert". A value above 1.00 today is only reachable via
the >128-active-slot per-slot fallback. The question this line family was
built for — does a tensor's repack re-run every eval instead of once per
weight — is still answered by `dispatches` and the repack ms buckets, which
kept their meaning.

#### Host-side phase breakdown (`HOST`/`HOST2`–`HOST5`)

⚠️ **Known doc gap, flagged rather than backfilled from memory: this
subsection covers the `[MXFP4-PP-BATCHED-PROFILE-HOST]` through `-HOST6`
lines added across ~11 commits of the same investigation
(`llama.cpp-iikr`, the "unphased-host-time" → "promptadmit remainder" →
"B50 residual-pool" cycles) that produced the device-event lines above.
Everything below is accurate to the source as of this writing, but this
family has not yet accumulated the multi-round hardware-caveat depth the
`other`/graph-replay/host-waits sections above have — those were found by
repeated hardware captures over time; this is a from-the-source writeup.
Treat a claim here as unverified-on-hardware unless it cites a measured
figure, and correct this section the first time a capture contradicts it.**

Unlike the lines above (SYCL device-event profiling, `command_start`/
`command_end`), these are **host CPU wall-clock** phases — a running
`std::chrono::high_resolution_clock`, reset at each mark (`host_phase_mark`
in `ggml_sycl_mul_mat_id`), so consecutive marks partition elapsed HOST
dispatch time rather than measuring device execution. All five lines share
the `GGML_SYCL_MXFP4_PP_PROFILE` gate and print once per eval, same as
`[MXFP4-PP-BATCHED-PROFILE]` above.

```
[MXFP4-PP-BATCHED-PROFILE-HOST] device=<N> readback=<ms>/<calls> ptrtable=<ms>/<calls> routeresolve=<ms>/<calls> grouping=<ms>/<calls> repackstage=<ms>/<calls> submission=<ms>/<calls>
[MXFP4-PP-BATCHED-PROFILE-HOST2] device=<N> whole=<ms>/<calls> prologue=<ms>/<calls> unphased=<ms> gemm_2d_args_map=<ms>/<calls>
[MXFP4-PP-BATCHED-PROFILE-HOST3] device=<N> entry=<ms>/<calls> promptadmit=<ms>/<calls> routeeval=<ms>/<calls> fastpath=<ms>/<calls>
[MXFP4-PP-BATCHED-PROFILE-HOST4] device=<N> admit_idswait=<ms>/<calls> admit_sweep_a=<ms>/<calls> admit_sweep_b=<ms>/<calls> admit_publish=<ms>/<calls> admit_lookup=<ms>/<calls> admit_resolve=<ms>/<calls> admit_snapshot=<ms>/<calls> admit_cache_hits=<N> admit_cache_misses=<N> admit_publish_aos=<N> admit_publish_nonaos=<N> admit_probe_entries=<N> admit_resolve_fastpath=<N> admit_resolve_fallback=<N> sweep_layout_reused=<N> sweep_layout_swept=<N>
[MXFP4-PP-BATCHED-PROFILE-HOST5] device=<N> resolve_inner=<ms>/<calls> resolve_outer=<ms>/<calls> resolve_memo_hit=<ms>/<calls> fastpath_rebuild=<ms>/<calls> fastpath_rebuild_reused=<N> fastpath_rebuild_built=<N>
[MXFP4-PP-BATCHED-PROFILE-HOST6] device=<N> admit_resolve_layout=<ms>/<calls> admit_resolve_batch=<ms>/<calls> admit_resolve_align=<ms>/<calls> admit_resolve_populate=<ms>/<calls>
```

(each wrapped here for readability; every real line is one line.)

**HOST** — the original six phases inside the batched-executor loop itself:
`readback` (device→host ids readback), `ptrtable` (expert pointer table
build/upload), `routeresolve` (per-expert route resolution inside the
dispatch loop), `grouping` (expert grouping for the batched kernel calls),
`repackstage` (repack-kernel staging), `submission` (kernel submission
itself).

**HOST2** — `whole` brackets `ggml_sycl_mul_mat_id`'s entire body (RAII,
every exit path, success or throw). `prologue` here is narrower than its
name once suggested: the marks run sequentially `entry` →
`promptadmit` → `routeeval` → `fastpath` → `prologue` → HOST's own
`readback` (first of the original six), so by the time `prologue`'s own
mark fires it only covers the small tail segment between the end of
`fastpath` and the start of `readback` — HOST3's four sub-marks are
separate, EARLIER segments in that same sequential chain, not contained
inside `prologue`. (The name is a holdover: before that five-way
bisection, "prologue" meant the whole undifferentiated gap from function
entry to the first of the six original phases; HOST3 split most of that
gap out from under it, leaving `prologue` as just the final slice.)
`unphased` is derived at print time (`whole` minus every phase-mark
total: HOST's six + HOST3's four + HOST2's own `prologue`) — a print-
time-only quantity, not separately accumulated; large `unphased` is the
signal to bisect further, not a target in itself. `gemm_2d_args_map` is a
*different* instrument folded in here:
`gemm.hpp`'s own `ggml_sycl_gemm_profile` accumulator (declared in that
header, not this struct, since `gemm.hpp` is included earlier in this TU),
timing the WOQ 2-D fallback's per-iteration `std::unordered_map<int,
dnnl::memory>` construction — measured negligible on hardware (0.252 ms
total across 1596 iterations, one full cycle's finding, not re-verified
since).

**HOST3** — the four sequential marks that precede HOST2's own `prologue`
mark in the same chain (see HOST2 above for the ordering), covering most
of what "prologue" originally meant before this split, in the function's
own top-level code order: `entry` (function entry through weight/id/ne
setup), `promptadmit`
(the `ne12 != 1` prompt-admission block — itself further split by HOST4,
see below; `promptadmit` here is DERIVED as the sum of HOST4's six
sub-phases, not separately accumulated, so it stays comparable across the
several cycles that narrowed it: 133.6 → 20.5 ms/eval as of `bbc8c3352`,
the most recent captured commit at the time of this writing — a further
drop is expected once the routeeval caching fix (`d14b5bdd6`, see
`routeeval` just below) is captured, but that number is not yet measured),
`routeeval` (retained-route lambda definitions and, as of `d14b5bdd6`, a
single cached read of the fused-fast-path admissibility check that used
to be recomputed 3x per layer — see that commit's message for the full
attribution), `fastpath` (the planner-fastpath booleans through the
batched-oneDNN route-probe chain).

**HOST4** — bisects `promptadmit` into six sequential sub-phases in that
block's own code order: `admit_idswait` (ids D2H cache fetch/refresh,
including the first-consumer device wait), `admit_sweep_a`/`admit_sweep_b`
(the two independent per-dispatch 32-expert layout-validation sweeps the
analyst read `c-nq1u` named — `select_moe_planned_graph_layout` and
`moe_layout_for_selected_rows` respectively), `admit_publish`
(`publish_mmid_canonical_aos_experts`, AOS-route-only — see
`admit_publish_aos`/`admit_publish_nonaos` below for whether it ever
actually runs), `admit_lookup` (the promptadmit admission-cache key build
and hit/hash check), `admit_resolve` (the cache-miss path: the three
`build_moe_resolved_batch` calls — see HOST5 for this one's own further
split). `admit_snapshot` is **not part of the six-phase partition** —
it separately, additionally times the ids-snapshot copy plus the two
`ids_hash` call sites via their own before/after timestamps (they are not
adjacent in source, so they cannot share one running-clock mark the way
the six phases do); it overlaps whichever of the six phases contains each
site and must not be summed into `promptadmit`.
`admit_cache_hits`/`admit_cache_misses` count the promptadmit admission
cache's own hit rate. `admit_publish_aos`/`admit_publish_nonaos` classify
every dispatch by `retained_prompt_layout`'s final value right before the
`admit_publish` call — `aos=0` across an eval means the AOS-gated body
never ran and `admit_publish` should read near-zero too, corroborating
each other. `admit_probe_entries`/`admit_resolve_fastpath`/
`admit_resolve_fallback` are **any-caller totals**, not scoped to just the
two promptadmit sweeps — they come from thread_local counters declared
next to `ggml_sycl_probe_moe_planned_layout` and
`ggml_sycl_resolve_moe_expert_route` themselves (declared far earlier in
this TU than this struct, same "declared at the call site" split as
`gemm_2d_args_map`), so they total every call across the whole profiled
pipeline for that eval while profiling is on, matching the existing
`g_moe_decode_route_census` precedent for the same function ("...entries
(any caller)").
`sweep_layout_reused`/`sweep_layout_swept` (`fcf1860a8`, mechanism 1) classify
every dispatch by whether `sweep_a`/`sweep_b` above (the two-sweep computation
of `retained_prompt_layout` for src0 specifically, at the TOP of the `ne12 !=
1` block, before the triad-lookup below) actually ran or were skipped: a
read-only, additive probe checks whether src0 is one of the layer's 3 sibling
weights and the layer's admission-cache entry already carries a non-rejected
bundle (populated by whichever sibling dispatched first this eval) — if so,
`retained_prompt_layout` is read straight off that role's `actual_layout()`
and both sweeps are skipped (`reused`); otherwise the full sweep runs as
before (`swept`). This is a SEPARATE probe from the existing promptadmit
cache lookup further down (which still runs unconditionally on every
dispatch) — the two independently look up the same map by the same key, so a
`reused` classification here does not change `admit_cache_hits`/`_misses`
below it. `swept` dominating on a triad-heavy graph (expected only 1 of every
3 sibling dispatches per layer, the layer's first) would mean the reuse probe
is declining more than expected and is worth a follow-up capture.

**HOST5** — bisects `admit_resolve`'s `build_moe_resolved_batch` calls
into `resolve_inner` (the per-unique-expert `resolver()` call itself) and
`resolve_outer` (loop overhead plus full field-stamping for a genuinely
new expert), further splitting `resolve_outer` into `resolve_memo_hit`
(the repeat-token-occurrence branch specifically) once a capture found
`resolve_outer` dominating `resolve_inner` 12:1 and `resolve_memo_hit`
dominating in turn — see `moe-resolved-batch.hpp`'s own
`ggml_sycl_resolve_batch_profile` namespace comment for the full
attribution chain (69.0 ms/eval, 279,432 copies, before `ba987da15`'s
canonical-payload fix).
`fastpath_rebuild`/`fastpath_rebuild_reused`/`fastpath_rebuild_built`
(`9573da32f`, design finding 2) are a SEPARATE, additional bracket around
a different call site: `fastpath`'s own `ggml_sycl_build_moe_resolved_batch(
src0, ...)` inside the `ne12 != 1` block, not `admit_resolve`'s three
gate/up/down calls above. Like `admit_snapshot` in HOST4, this OVERLAPS
`fastpath`'s window in HOST3 rather than partitioning it, and must not be
summed into that total. `fastpath_rebuild_reused`/`_built` classify every
call: `reused` took the cheap `moe_resolved_batch` copy of an
already-admitted role (src0 matched one of the triad's 3 roles and its
layout agreed), `built` fell through to the independent resolve (no triad
match, an unpopulated/rejected bundle, or a layout disagreement). A
`reused`-heavy split with `fastpath_rebuild_us` well below its pre-fix
figure confirms the reuse path is actually engaging on hardware, not just
compiling; `built`-heavy despite triad tensors in the graph would mean the
layout-equality guard is declining more than expected and is worth a
follow-up capture.

**HOST6** — bisects `admit_resolve`'s CACHE-MISS body (`1378fe65b`, mechanism
2, instrument-only — added to characterize the pool, not yet acted on) into
the 4 sequential sub-phases the cache-miss branch actually contains, in code
order: `admit_resolve_layout` (the 3x `role_layout()` calls for gate/up/down
— each its own `select_moe_planned_graph_layout`+`moe_layout_for_selected_rows`
sweep pair, the SAME recipe HOST4's `sweep_a`/`sweep_b` run for src0 alone,
here paid for all 3 roles since this is the layer's first dispatch),
`admit_resolve_batch` (the 3x `ggml_sycl_build_moe_resolved_batch()` calls —
OVERLAPS HOST5's `resolve_inner`/`resolve_outer`/`resolve_memo_hit`, which
measure only what happens *inside* that call; this bracket is the superset
including per-call/loop overhead outside those), `admit_resolve_align`
(`align_moe_retained_role_batches()`, expected cheap — an O(operand count)
comparison loop, no allocation), `admit_resolve_populate` (the cache-entry
construction, including the `ggml_sycl_moe_validate_retained_role_bundle()`
call `d14b5bdd6` moved here from routeeval). Same running `host_phase_clock`
as every other mark in this function, so on the common (resolve-succeeds)
path the four sum exactly to `admit_resolve` above — same partition guarantee
`HOST4`'s six top-level phases already give the old single `promptadmit`
figure. A failed resolve (gate/up/down not all admitted) skips the
`admit_resolve_align`/`admit_resolve_populate` marks; that near-zero
condition-check overhead lands in `admit_resolve` itself rather than a named
sub-phase.

#### What `other` does and does not mean

`other` is not a pure "everything this instrument doesn't track" bucket —
several independent things land in it, and a raw negative-or-positive
magnitude alone does not tell you which:

1. oneDNN work elsewhere in the same graph that this instrument does not
   separately measure (e.g. attention SDPA, `fattn-onednn.cpp`, out of this
   instrument's file scope) lands inside `other`, alongside genuine
   host/dispatch-gap time.
2. The WOQ=0 arm's unmeasured per-expert f16 dequant (see `f16_gemm` above).
3. **The instrumentation itself inflates `other`, on both the per-dispatch and
   the graph-level side.** Per-dispatch: the `stream->wait()` drain (see Host
   waits below) de-pipelines submission, so the device sits idle between
   dispatches waiting for the host to finish reading back and accumulating the
   previous dispatch's events before it submits the next one's kernels, and
   that idle time falls inside the graph-total bracket without landing in any
   of the four named buckets. Graph-level: the graph-compute wrapper's own
   `end_ev.wait()` forces every bit of the graph's tail work to actually
   complete on the device before `ggml_backend_sycl_graph_compute()` returns —
   an UNPROFILED call does not do this (pure-GPU decode can legitimately
   return with kernels still in flight; see the SYCL Memory Ownership section
   of `CLAUDE.md`), so a profiled `graph_total` can be longer than what an
   unprofiled run would have waited for at that exact point, and that extra
   wait time has nowhere to land but `other`. The sibling `mmvq.cpp` instrument
   enabled by the same shared env var (`mmvq_moe_pp_profile_enabled()`, its own
   `stream->wait()` calls at `mmvq.cpp:17817-17822` and `:19254`) does the same
   to the small-batch MMVQ GEMV path within the same graph.
4. A negative `other` specifically can also mean device work is happening on a
   queue this instrument never brackets — the graph-level span times only
   `profile_ctx->stream()` (this device's primary compute queue); MoE row
   aggregation can dispatch experts to OTHER GPUs
   (`pp_per_gpu_entries_for_log` and friends), and any component time
   accumulated from THAT device's events would sum into `accounted` without a
   matching contribution to `graph_total`, since `graph_total` never measured
   that other queue in the first place.

**Do not read `other` as an unprofiled run's host-gap time** — it is specific
to a profiled run and larger than the true unprofiled gap because the
instrument's own synchronization (points 3 above) is part of what it measures.
Read `other` as an upper bound on the non-MoE-PP, non-instrumented remainder,
not as a pure host-overhead figure.

#### Graph replay can silently discard an entire captured shape

⚠️ **Under SYCL graph replay (the default), the instrument can go silent for
an entire captured graph shape — potentially printing NOTHING, not just "less
often".** Two separate mechanisms combine: (1) the RECORDING eval (the one
call that captures the graph) DOES re-enter `ggml_sycl_mul_mat_id`/the batched
lambda — `compute_impl()` runs for real while capturing — but
`mxfp4_pp_batched_profile_enabled()` (this instrument's dispatch-level guard,
`pp_profile` inside the lambda) reads `ggml_sycl_graph_recording_active()` and
is itself gated off for exactly that window, so nothing is collected even
though the dispatch happened; (2) a subsequently REPLAYED eval (same graph
shape, `ext_oneapi_graph()` fast path) does not call
`compute_impl()`/`ggml_sycl_mul_mat_id` at all, so there is nothing to collect
there either. `dispatch_evals` is therefore 0 on BOTH the eval that recorded
and every eval that replays it. The graph-compute wrapper's OUTER bracket is a
different guard (evaluated before descending into
`ggml_backend_sycl_graph_compute_unchecked`, where graph recording has not
started yet) and does NOT see recording-active on either kind of eval, so it
still pays for its begin/end markers and the `.wait()` every time (see Host
waits below) — but `mxfp4_pp_batched_profile_record_graph_total` sees
`dispatch_evals == 0` and silently resets without printing (by design — see
the comment there — since there is nothing to attribute). Net effect: once a
PP cgraph shape gets captured into a replay graph, this instrument can print
zero lines for it ever again while still paying its full per-eval cost, with
no signal that anything was skipped. **Pair `GGML_SYCL_MXFP4_PP_PROFILE=1` with
`GGML_SYCL_DISABLE_GRAPH=1` to get a printed line for every eval.** Guarded the
same way as the TG/PP profile variants in `mmvq.cpp`: skipped while a SYCL
command graph is recording (`ggml_sycl_graph_recording_active()`), since this
diagnostic both submits extra no-op marker kernels (`ggml_sycl_submit_marker`)
and reads event profiling info, which can implicitly wait on an incomplete
event — neither is legal during graph capture.

#### The two PP MoE instruments can fire together in one eval

Reuses the SAME env var name `mmvq_moe_pp_profile_enabled()` (`mmvq.cpp`)
already reads for a complementary, older instrument that times the
small-batch MMVQ GEMV MoE kernels (the non-batched PP path, `[MXFP4-MOE-PP-PROFILE]`
lines). **Per-dispatch, the two never double-count** — a single tensor
dispatch goes through exactly one of the two executors, never both. That does
NOT mean the two instruments never fire together within the same eval: with
`GGML_SYCL_MOE_DOWN_XMX_TILED` default OFF, one eval routinely has `down`
routed through the MMVQ path (timed by `mmvq.cpp`'s instrument) while
`gate`/`up` go through this batched executor — different line prefixes keep
the two unambiguous, but both instruments' `stream->wait()` calls perturb the
SAME graph in the SAME eval (see point 3 under "What `other` does and does not
mean" above).

#### Host waits and zero cost

**Host waits**: on the batched-executor side, one `stream->wait()` per
successful dispatch (not per event) after every component-1/2/3/4 event for
that dispatch has been submitted; on the graph-compute wrapper side, one
marker + `.wait()` bracketing the whole
`ggml_backend_sycl_graph_compute_unchecked()` call, success path only — an
eval that throws is simply not recorded, so this diagnostic never layers extra
device work onto the existing GPU-fallback recovery path.

**Zero cost when unset**: every insertion point is gated on the same cached
bool (`mxfp4_pp_batched_profile_enabled()`), so with the flag off,
`ggml_backend_sycl_graph_compute()` takes the unmodified direct-return path and
the batched executor's per-dispatch loops skip every marker/event push (the
collection vectors stay empty, no extra kernels are submitted). The two GEMM
markers are `std::optional<sycl::event>`, not default-constructed
`sycl::event` objects, so with the flag off neither marker is even
constructed — "zero cost when unset" is literal, not just "cheap".

**Accumulator reset**: the accumulator is reset to a fresh state at
`ggml_backend_sycl_graph_compute()` entry, before the begin marker, on EVERY
call (not only after a successful print) — this is the primary mechanism
keeping two evals from bleeding into each other, covering the case where the
PREVIOUS call threw (the `fallback_error` recovery path, which used to leave
accumulated dispatches merging into the NEXT eval's line) as well as the case
where it silently returned with `dispatch_evals` still 0 (the graph-replay
case above). Purely host-side; no device work is added to any recovery path.
