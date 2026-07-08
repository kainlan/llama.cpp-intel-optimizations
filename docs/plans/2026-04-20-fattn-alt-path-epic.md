# 2026-04-20 — Deterministic SYCL Flash-Attention alternative path (epic)

**Status**: PLAN — not implemented
**Branch**: `feature/sycl-coalescing`
**HEAD at plan time**: `6709d5d06`
**Predecessor bead**: `llama.cpp-l144i` (IN_PROGRESS)
**Parallel workstream**: kernel-printf diagnostic on `fattn-xmx-f16.hpp` — branch point in §8.

## §1 — Problem statement

`llama-completion` on `gpt-oss-20b-mxfp4.gguf` produces garbled, non-deterministic tokens across runs with `--seed 42 --temp 0`. Session 4 of `llama.cpp-l144i` (see `docs/plans/2026-04-19-l144i-rootcause.md` §13 and §14) localized the race to inside the SYCL XMX flash-attention kernel `flash_attn_xmx_f16_kernel` in `ggml/src/ggml-sycl/fattn-xmx-f16.hpp`. Evidence:

- §14.2: all 5 input tensors (Q/K/V/mask/sinks) D2H-hash bit-identical across runs (probes `fa/src*` in `fattn.cpp:875-905`).
- §14.1: `__fattn__-0` (layer 0 SWA) output hash differs across runs for the same bit-identical inputs.
- §14.4: barrier audit passed — every SLM RAW sequence is preceded by `group_barrier`.
- §14.5, §14.7.5: four empirical fixes rejected — end-of-loop barrier (hash changed but still non-det); `SYCL_CACHE_PERSISTENT=1` (still non-det); `XMX_PAD=8` (still non-det); MMA F16 force (still non-det per rootcause doc).

Leading hypothesis (unconfirmed): `sycl::ext::oneapi::experimental::matrix::joint_matrix_mad` on Intel Arc B580 DPAS is non-deterministic across launches for some shape×input combinations. Parallel agent is currently running kernel-printf diagnostic to capture DPAS input/output byte-exactly and distinguish hw-nondet from code-nondet. This plan is pre-emptive: if hw-nondet is confirmed or strongly indicated, XMX is ruled out as a correctness primitive on this hardware and we need a byte-exact deterministic replacement.

The user directive ("no fallbacks, no disabling features, no workarounds, correct approach only") rules out "turn fattn off" or "gate fattn via env var". A fattn path that is byte-exact deterministic across runs is the correct approach — not a workaround.

## §2 — Target state

A SYCL fattn path that:

- **S1 (determinism)**: produces bit-identical F32 output across N runs for identical input (N≥5). The existing `l144i-probe.hpp` `cf/dst_exit` hash for `__fattn__-0..23` matches byte-for-byte across runs.
- **S2 (Mistral perf)**: TG128 ≥ 81 tok/s, PP512 ≥ 1480 tok/s on Arc B580 level_zero:0 per CLAUDE.md targets.
- **S3 (20B perf)**: TG ≥ 15 tok/s, PP ≥ 54 tok/s on `gpt-oss-20b-mxfp4.gguf` per §13.9 of rootcause.
- **S4 (coverage)**: covers every shape the current XMX path serves — head dim D ∈ {64, 128, 256}, ncols ∈ {1, 2, 4, 8}, F16 and FP8 E4M3 KV cache, F32 and F16 query, logit_softcap on/off, mask present/absent, sinks present/absent, multi-sequence, paged V2, multi-token decode, GQA (ne02/ne12 ratio).
- **S5 (20B coherence)**: `llama-completion -p "1, 2, 3, 4, 5," -n 30 --seed 42 --temp 0` on `gpt-oss-20b-mxfp4.gguf` produces coherent English text deterministically across 3 runs. Closes Gate 4 of l144i.
- **S6 (no regression)**: test-backend-ops `-o FLASH_ATTN_EXT` passes on SYCL backend (same pass count as baseline at plan HEAD or better; no new NMSE failures).

## §3 — Design alternatives

### Alternative A — promote the existing `launch_fattn_mma_f16` scalar path to default on Arc B580

**What**: `ggml/src/ggml-sycl/fattn-mma-f16.hpp` already exists and is currently selected only when `use_xmx == false` (non-XMX GPUs) or when `safe_decode && ne01 <= 1`. It is **despite the "MMA" in the name not a joint_matrix kernel** — the dot product at lines 240-244 is scalar `for (d=0; d<D; d++) dot += Q[d]*K[d]` in F32 accumulator, with F16 operand reads from SLM. Confirmed by `grep joint_matrix_mad fattn-mma-f16.hpp` → 0 matches. Output writeback (lines 367-402) is per-(j, d_idx) with no cross-thread reduction — each thread owns its `D_per_thread` slice of VKQ.

The code already handles D=64/128/256 and ncols=1/2/4/8 (dispatcher `ggml_sycl_flash_attn_ext_dispatch_ncols` at `fattn.cpp:740-819` has the MMA branch explicit).

**How to deploy**: change the `use_xmx` decision so it becomes false on Arc B580 (either unconditionally for now — simplest — or gated by a shape-specific heuristic once perf is measured). Dispatcher change is localized to `fattn.cpp:740-819`.

**Pros**:
- Zero new kernel code. Alternative is already compiled into every build.
- Already serves every D, ncols, KV-type, mask, sinks, paged, multi-seq case the XMX path serves (need to confirm feature parity for FP8, paged, multi-seq — Phase 1 work).
- Scalar accumulation order is compile-time fixed (loop `for d in [0, D)`), matching a textbook reference. No DPAS; no SLM bank conflicts across sub-groups for the reduction.
- A scalar kernel with F32 accumulator and `for (d=0; d<D; d++)` order is the CUDA/CPU reference that l144i's §14 already treats as ground truth for correctness.

**Cons / unknowns**:
- Perf unknown on Arc B580. The scalar path was historically used for non-XMX GPUs. On an XMX-capable GPU, moving Q@K^T off XMX is a large throughput hit **in theory** (XMX is 8x the FP16 FMA rate vs scalar SIMD16). But Mistral TG on Arc B580 currently shows XMX doesn't dominate — ESIMD single-query path is already +7% faster than XMX for decode (§fattn.cpp:411-419 comment). So the XMX advantage for ncols=1 TG is not clean.
- FP8 E4M3 path in `fattn-xmx-f16.hpp` may not have a parallel implementation in `fattn-mma-f16.hpp` — needs audit in Phase 1.
- Paged V2 path (`should_use_paged_attention_v2`) bypasses the dispatcher — unclear interaction.
- Scalar PP perf for ncols=8 at D=128 will almost certainly regress from XMX; need measurement.

### Alternative B — CUDA-MMA-style port (`fattn-mma-f16.cuh` → SYCL joint_matrix)

**What**: port the CUDA `fattn-mma-f16.cuh` implementation structure to SYCL using `joint_matrix` primitives, with its deterministic Ampere-style tile assignment (configs at lines 39-72 of `fattn-mma-f16.cuh`: explicit nthreads/occupancy/nbatch_fa/nbatch_K2/nbatch_V2 per (D, ncols)). F32 accumulator, no atomic reduction, explicit per-sub-group tile assignment.

**Pros**: design is proven correct at scale; broad shape coverage; CUDA path is the reference llama.cpp treats as ground truth.

**Cons**: still uses `joint_matrix_mad` — which is exactly the primitive l144i §14.6 hypothesizes as hw-nondet. **If the parallel diagnostic confirms hw-nondet on DPAS, Alternative B is ruled out as a correctness solution.** Port is also a large mechanical lift (several hundred lines) for possibly no correctness gain. Punt unless diagnostic clears DPAS.

### Alternative C — new scalar-reduction fattn tuned for Arc B580 EUs

**What**: write a new scalar fattn optimized for Arc B580's SIMD16 EU, no `joint_matrix`, no DPAS. Two-pass online softmax; per-sub-group K-tile; sub-group reduction via `sycl::reduce_over_group` for dot product (hardware-deterministic per SYCL spec — reduce_over_group on associative/commutative float op is implementation-defined order, but per-launch deterministic because `sycl::plus<float>` reduce tree is fixed at kernel compile).

**Pros**: guaranteed deterministic even if DPAS is hw-nondet. Tailored to Arc B580 (known cache/SLM sizes, SIMD16 subgroup).

**Cons**: significant perf hit for large D and large batches. Net-new kernel (~500 lines) to write, test, and maintain in parallel with the existing `fattn-mma-f16.hpp` scalar. Much of what this would become is just a Arc-tuned fork of `fattn-mma-f16.hpp` — so in practice it degrades to "Alternative A plus tuning".

### Alternative D — promote existing MMA path as baseline, add Arc-specific tuning on top (A + targeted C)

**What**: adopt Alternative A as the immediate correctness primitive, then if-and-only-if Phase 4 perf gates fail, fork `fattn-mma-f16.hpp` into an Arc-tuned variant (`fattn-arc-f16.hpp`) that tightens thread count, SLM tile sizes, loop unrolling, and subgroup reduction for Arc B580's specific hardware. This is A with a perf escape valve; C becomes Phase-4-conditional work, not up-front work.

**Pros**:
- Fastest path to correctness (A alone = flip a dispatcher flag).
- Optional perf tuning only if A's numbers miss the gates.
- The Arc-tuned fork, if ever needed, starts from an already-working correct baseline — much safer than writing a new kernel.

**Cons**: small risk of two kernels to maintain long-term if Phase 4 requires the fork.

## §4 — Current-state map

**Consumers of XMX fattn on Arc B580**:
- Single entry: `ggml_sycl_flash_attn_ext` at `fattn.cpp:822`, called from `ggml-sycl.cpp` for every `GGML_OP_FLASH_ATTN_EXT` op.
- Dispatcher: `ggml_sycl_flash_attn_ext_dispatch_ncols<D, Q_type>` at `fattn.cpp:740-819`. `use_xmx = gpu_has_xmx(dev)` is true on Arc B580. For `ne01 <= 1 && safe_decode` (default enabled via `GGML_SYCL_FA_SAFE_DECODE=1`), it falls back to the scalar MMA F16 path already. For `ne01 >= 2` or `safe_decode` off, it dispatches to `launch_fattn_xmx_f16` (lines 788-800).
- ESIMD detour (lines 759-772): `g_sycl_fa_esimd_enabled && fattn_esimd_f16_available() && ne01 <= 1` → `fattn_esimd_f16` (ESIMD path, single-query only, D=64/128). This is a SEPARATE deterministic-ish path but only covers ncols=1.
- Paged V2 detour (line 1230+): `g_sycl_paged_v2_enabled && should_use_paged_attention_v2(max_context_len)` bypasses the dispatcher entirely for context > 512. Not active in our 20B repro (ctx=30).

**Shape space XMX serves at HEAD (Arc B580, safe_decode on)**:
- D ∈ {64, 128, 256}, ncols ∈ {2, 4, 8} always XMX.
- D ∈ {64, 128, 256}, ncols=1 goes to MMA F16 scalar (already via `safe_decode`). So **TG ncols=1 on Arc B580 at HEAD is already on the scalar path.** This means the observed non-determinism for `__fattn__-0` in 20B TG (which is ncols=1 prompt processing with 20 query tokens → ncols=8) is specifically the ncols=8 XMX path.
- 20B repro has ne01=20 (prompt "1, 2, 3, 4, 5,") → ncols=8 at first fattn (prefill), then ne01=1 from token 2 onward → ncols=1 (already scalar). The layer-0 divergence seen in §14.1 is therefore **the prefill pass at ncols=8, which is XMX on Arc B580 despite safe_decode**.
- 20B TG perf target (15 tok/s) is decode-dominated, ncols=1, already on scalar. So promoting ncols=2/4/8 to scalar mainly affects **PP, not TG**.

**What XMX kernel carries that the scalar MMA F16 does not** (audit needed — Phase 1):
- FP8 E4M3 KV cache dequantization (`kv_is_fp8` template parameter in XMX kernel).
- Paged attention block-table indirection (XMX has `use_paged_attn` branch).
- Multi-sequence boundary handling (`n_seqs`, `seq_q_offsets`, `seq_kv_offsets`).
- Multi-token decode (`multi_token_decode`, `q_positions`, `kv_base_pos`).
- Attention sinks (present in both — confirmed in `fattn-mma-f16.hpp:338-361`).

Parity audit of `fattn-mma-f16.hpp` vs `fattn-xmx-f16.hpp` for the above five items is Phase 1 blocker work.

**Test coverage**:
- `test-backend-ops -o FLASH_ATTN_EXT` on SYCL: exists, uses `test_flash_attn_ext` struct (`tests/test-backend-ops.cpp:5908`). Shapes in `make_test_cases_eval()` exercise hsk/hsv ∈ {64, 80, 96, 112, 128, 256, 576}, nh ∈ {8, 32}, kv ∈ {1..5776}, nb ∈ {1, 8, 130+} (line 8019-8023, 8262).
- **Determinism harness missing**: no existing test runs the same fattn op twice and compares bit-for-bit. A new micro-bench `test-fattn-determinism` under `tests/` that runs the SYCL fattn 5x and byte-compares dst is required. Phase 1 work. Very small (~100 LOC).

**Interaction with existing epics**:
- `mxfp4-compute-parity` (CLOSED per `docs/plans/2026-04-17-mxfp4-compute-parity.md` existence in status): no fattn surface — MXFP4 is MUL_MAT_ID territory. No conflict.
- `project_tensor_access_redesign` (IN_PROGRESS per MEMORY): makes raw `->data` inaccessible from GPU dispatch. fattn resolves pointers via `ggml_sycl_resolve_tensor_ptr` and `safe_dst.resolve_as<float>()` already (see `fattn.cpp:945-950`) — already aligned with the redesign direction. No conflict.
- `unified-memory-epic` (CLOSED): no fattn surface.

## §5 — Picked alternative + justification

**Recommendation: Alternative D (A first, C-style fork conditional on Phase 4 perf gate)**.

Rationale:

1. **Correctness-first**: A is byte-exact scalar F32 accumulation in a fixed loop order. It is the same shape of kernel the CUDA backend and the CPU backend already pass our regression gates with. If the parallel diagnostic confirms DPAS hw-nondet, A needs nothing more to satisfy S1. If the diagnostic instead finds a code race in the XMX kernel (Branch B/C in §8), A is still a correct fallback we can adopt while the XMX fix is prepared.

2. **Minimal code surface**: the switch in §6 Phase 2 is a few lines in `ggml_sycl_flash_attn_ext_dispatch_ncols` to set `use_xmx = false` unconditionally (or gate on an env var during development). No new kernel to write, test, review. The Phase 1 parity audit is the only blocker, and the existing `fattn-mma-f16.hpp` already covers D=64/128/256, ncols=1/2/4/8, F16 KV, sinks, mask, logit_softcap — ~90% of the shape space we need.

3. **Addresses user directive cleanly**: a deterministic scalar kernel is not a workaround or a disabled feature. Flash attention remains enabled, all ops are served; the dispatch just picks a deterministic primitive over a non-deterministic one. "No fallbacks" is satisfied in letter and spirit.

4. **Per-shape perf envelope estimate**:
   - TG128 Mistral: ncols=1 already scalar (safe_decode). A changes nothing. **Expect ≥81 tok/s preserved.**
   - TG 20B: ncols=1 for expert decoding. Already scalar. **Expect ≥15 tok/s preserved.**
   - PP512 Mistral: ncols=8 currently XMX. Moving to scalar is the perf risk. Conservative estimate: −15 to −35% PP512 (scalar F32 accum vs DPAS FP16 MAD) → 1480 → 960..1260. **Phase 4 gate will measure**; if it misses 1480, Phase 4.5 triggers a C-style Arc-tuned fork with subgroup reduction and tighter tiling.
   - PP 20B: 20B PP is CPU-expert-dominated today (see MOE-STATS in rootcause §13.9). fattn is not the PP bottleneck; expect ≥54 tok/s preserved trivially.
   - PP 120B: untested at HEAD. Treat as information-only gate; if it regresses substantially, Phase 4.5 escalation is justified.

5. **Rollback is trivial**: single env var flag (`GGML_SYCL_FA_USE_XMX=1`) during development phases keeps the XMX path available for A/B measurement. Removed after Phase 6.

6. **Interaction with parallel diagnostic** (see §8): A is correct under every outcome of the diagnostic.

**Alternative B (CUDA-MMA port) is explicitly rejected** at this planning stage because it stakes correctness on `joint_matrix_mad`, which is the primitive under suspicion. Revisit only if the parallel diagnostic clears DPAS and the real root cause turns out to be a SYCL extension / JIT issue fixable by restructuring around `joint_matrix` differently.

## §6 — Phased transition plan

Phase gates are cumulative — each phase must pass before the next starts. Bead IDs in §9.

### Phase 1 — Parity audit + determinism harness + env gate

**Tasks**:
1. Audit `fattn-mma-f16.hpp` for feature gaps vs `fattn-xmx-f16.hpp`: FP8 E4M3 dequantization, paged attention indirection, multi-sequence handling, multi-token decode. Produce a feature-parity table. Any missing feature becomes a pre-Phase-3 blocker.
2. Add `tests/test-fattn-determinism.cpp` (new, ≤150 LOC): construct a fixed FLASH_ATTN_EXT case (D=128, ncols=8, ne11=20, F16 KV), execute on SYCL 5 times, byte-compare dst. Fails today (reproduces l144i), passes when A is live.
3. Add env gate `GGML_SYCL_FA_USE_XMX` (default ON at Phase 1 landing so master behaviour is preserved) to `ggml_sycl_flash_attn_ext_dispatch_ncols`. Reads once, caches, gates the `use_xmx` decision at line 750. Off = force scalar MMA F16 path for all ncols.

**Acceptance**:
- Feature parity table committed to this plan doc (§4 extended).
- `test-fattn-determinism` compiles, runs, FAILS at HEAD (demonstrates the race is testable).
- `GGML_SYCL_FA_USE_XMX=0` compiles, produces valid output on Mistral `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` canonical.

### Phase 2 — Feature-parity backfill in `fattn-mma-f16.hpp`

**Tasks** (any parity gap found in Phase 1):
1. Add FP8 E4M3 dequant template branch to `flash_attn_mma_f16_kernel` mirroring `fattn-xmx-f16.hpp:kv_is_fp8`.
2. Add paged attention block-table indirection if used by any target model.
3. Add multi-sequence handling (`n_seqs`, offsets).
4. Add multi-token decode handling (`multi_token_decode`, `q_positions`, `kv_base_pos`).

**Acceptance**:
- `GGML_SYCL_FA_USE_XMX=0 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT` on SYCL backend passes same count as `=1` (XMX default).
- No new NMSE failures.

### Phase 3 — Determinism gate (Mistral)

**Tasks**:
1. Run `test-fattn-determinism` with `GGML_SYCL_FA_USE_XMX=0` → must pass (5/5 byte-identical).
2. Run Mistral Gate 1 canonical (`1, 2, 3, 4, 5, 6, 7, 8, 9, 10`) with `GGML_SYCL_FA_USE_XMX=0` → 5 consecutive runs, all identical tokens.
3. Run `GGML_SYCL_L144I_PROBE=1` with scalar path → hash `cf/dst_exit __fattn__-*` byte-identical across 3 runs on Mistral.

**Acceptance**: determinism confirmed on Mistral with scalar path. `test-fattn-determinism` becomes permanent regression gate.

### Phase 4 — Perf gate (Mistral)

**Tasks**:
1. `llama-bench -p 512 -n 128` Mistral Q4_0 with `GGML_SYCL_FA_USE_XMX=0` → record PP512, TG128.
2. Compare against `=1` baseline at same HEAD.
3. Compare against CLAUDE.md targets (PP512 ≥ 1480, TG128 ≥ 81).

**Acceptance**:
- TG128 ≥ 81 tok/s (expected; ncols=1 already scalar).
- PP512 ≥ 1480 tok/s — if this fails, **Phase 4.5 triggers** (C-style Arc-tuned fork of `fattn-mma-f16.hpp`).

**Phase 4.5 (conditional)** — Arc-tuned scalar fork:
1. Fork `fattn-mma-f16.hpp` into `fattn-arc-f16.hpp`. New kernel.
2. Tune `nthreads`, `BATCH_KV`, sub-group reduction (`sycl::reduce_over_group<sycl::plus<float>>`), Q register tiling for Arc B580.
3. Re-gate dispatcher to use `fattn-arc-f16` when `gpu_has_xmx(dev)` (Arc) but `use_xmx` off.
4. Repeat Phase 3 and Phase 4 gates with the Arc-tuned kernel.

### Phase 5 — 20B determinism gate (primary close)

**Tasks**:
1. `llama-completion -p "1, 2, 3, 4, 5," -n 30 --seed 42 --temp 0` on `gpt-oss-20b-mxfp4.gguf` with `GGML_SYCL_FA_USE_XMX=0`. Run 3 times.
2. All 3 runs produce identical token sequences (deterministic).
3. Output text is coherent English (human inspection).
4. `llama-bench -p 20 -n 30` on 20B → TG ≥ 15, PP ≥ 54.

**Acceptance**:
- l144i Gate 4 passes. Bead `llama.cpp-l144i` closes.
- 20B PP/TG gates pass.

### Phase 6 — Flip default, retire env gate

**Tasks**:
1. Change dispatcher so scalar path is the default on Arc B580 (or universally) with no env var needed.
2. Remove `GGML_SYCL_FA_USE_XMX` env var reads.
3. Keep XMX launcher code compiled for reference / for non-Arc GPUs that pass its determinism test (if any). Alternatively remove XMX files entirely — decision deferred to Phase 6 planning once data is in.
4. Update `CLAUDE.md` SYCL Backend Structure section to reflect the new default.
5. Merge to master via PR with `ggml-ci` trigger in commit message.

**Acceptance**: all Phase 1–5 gates green without any env var set. CI green. PR merged.

## §7 — Risk analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| PP512 Mistral regresses > 30% | Medium | High (blocks S2) | Phase 4.5 Arc-tuned fork; subgroup reduction for dot product; larger BATCH_KV tiles |
| PP 20B regresses materially | Low | Medium | 20B PP is CPU-expert bound today; fattn not the bottleneck |
| `fattn-mma-f16.hpp` missing FP8/paged/multi-seq features larger than estimated | Medium | High | Phase 1 audit is explicit gate; Phase 2 backfill is a full phase not a side task |
| Scalar kernel itself has a determinism bug we haven't noticed | Low | High | Phase 3 determinism gate on Mistral catches it before 20B; `test-fattn-determinism` catches regressions forever |
| DPAS hw-nondet is disproven by parallel diagnostic (race turns out to be code) | Low | Medium (plan still useful, but XMX becomes repairable) | Phase 1 already gives us a deterministic path by env var; XMX repair can land in parallel. Alternative B becomes viable. |
| Moving PP off XMX reduces throughput enough that users revolt | Low | Medium | Env var `GGML_SYCL_FA_USE_XMX=1` can be retained as an opt-in "performance mode" after Phase 6 for users who accept non-determinism — not a fallback, an explicit user choice for a known-broken-on-this-hw primitive |
| Non-Arc SYCL devices (non-XMX) are unaffected | Certain | None | They already use the scalar path; Phase 6 default flip just matches their behaviour |
| Interaction with persistent-TG kernel (`GGML_SYCL_PERSISTENT_TG=1`) | Low | Low | Persistent TG is opt-in and separate from fattn dispatch; no cross-talk |
| Interaction with SYCL graph replay | Low | Medium | Scalar kernel has no special requirements; fattn is already graph-replayable. Re-verify `GGML_SYCL_DISABLE_GRAPH=0` in Phase 4 |

**Rollback plan**: all phases are guarded by `GGML_SYCL_FA_USE_XMX`. A failing gate reverts the dispatcher commit and leaves the env var & test infrastructure in place (valuable for future diagnostics). Phase 6 default flip is a one-line revert. There is no disruptive state change.

## §8 — Interaction with parallel diagnostic

The parallel agent's kernel-printf diagnostic inside `fattn-xmx-f16.hpp` Phase 4 will produce one of three outcomes:

**Branch A — DPAS hw-nondet confirmed**:
Two runs produce different `mat_SV` output from `joint_matrix_mad` for byte-identical `tile_S` / `tile_V` input. This plan proceeds unchanged. Alternative B is dead for this hardware. Alternative A (scalar) becomes the permanent architecture on Arc B580. Phase 6 flips default globally.

**Branch B — Code race found inside the XMX kernel (non-DPAS)**:
E.g., SLM aliasing across sub-groups, missed barrier condition that only fires under certain batch sizes, uninit read. The parallel agent lands a fix; their fix is orthogonal to this plan. In this case: **Phases 1–3 of this plan still land** (the determinism harness, the scalar env gate, the parity backfill) because they are regression-prevention infrastructure. Phase 4–6 may be cancelled or the plan may be reprioritised: keep XMX default, keep scalar as a permanent deterministic env-gated option for CI determinism runs. Escape valve preserved.

**Branch C — Inconclusive (printf perturbs timing, can't isolate)**:
Proceed with this plan as if Branch A. Determinism gained, DPAS is not load-bearing for correctness on Arc B580 going forward. The diagnostic bead stays open; this plan does not depend on it.

This plan's Phases 1–3 are **safe to start immediately** regardless of branch — they don't touch XMX code, they only add a dispatch gate and a regression harness. Phase 4+ is deferrable if the diagnostic lands a code fix in the XMX kernel before Phase 4 starts.

## §9 — Beads to create (draft `bd create` commands — do NOT run)

Epic bead:
```
bd create \
  --title "Deterministic SYCL fattn path — replace XMX for correctness" \
  --type epic \
  --priority 1 \
  --description "Epic covering the transition from XMX-based SYCL flash attention to a deterministic scalar path on Arc B580. Plan: docs/plans/2026-04-20-fattn-alt-path-epic.md. Closes llama.cpp-l144i Gate 4 (20B coherent output) once Phase 5 lands."
```

Phase task beads (each `--parent` pointing to the epic ID once created):
```
bd create \
  --title "Phase 1 — fattn parity audit + determinism harness + env gate" \
  --type task --priority 1 \
  --description "Audit fattn-mma-f16.hpp vs fattn-xmx-f16.hpp feature set (FP8 E4M3, paged, multi-seq, multi-token). Add tests/test-fattn-determinism.cpp. Add GGML_SYCL_FA_USE_XMX env gate in ggml_sycl_flash_attn_ext_dispatch_ncols. Acceptance: gate compiles, Mistral canonical passes with gate=0, test-fattn-determinism reproduces race at HEAD."

bd create \
  --title "Phase 2 — Feature-parity backfill in fattn-mma-f16.hpp" \
  --type task --priority 1 \
  --description "Close any Phase-1-identified gaps (FP8 E4M3 dequant, paged block-table indirection, multi-sequence, multi-token decode) in flash_attn_mma_f16_kernel. Acceptance: test-backend-ops -o FLASH_ATTN_EXT same pass count with USE_XMX=0 vs =1."

bd create \
  --title "Phase 3 — Determinism gate (Mistral scalar path)" \
  --type task --priority 1 \
  --description "Run test-fattn-determinism with USE_XMX=0 (must pass 5/5). Run Mistral Gate 1 canonical 5x deterministically. L144I probe confirms __fattn__ hash byte-identical across 3 runs."

bd create \
  --title "Phase 4 — Perf gate (Mistral, scalar path)" \
  --type task --priority 1 \
  --description "llama-bench Mistral Q4_0 -p 512 -n 128 with USE_XMX=0. Acceptance: TG128 >= 81, PP512 >= 1480. If PP missed, escalate to Phase 4.5."

bd create \
  --title "Phase 4.5 (conditional) — Arc-tuned scalar fork fattn-arc-f16.hpp" \
  --type task --priority 2 \
  --description "Only if Phase 4 misses PP512 >= 1480. Fork fattn-mma-f16.hpp into Arc-tuned variant with subgroup reduction, tuned BATCH_KV/nthreads for Arc B580. Re-run Phase 3 + 4 gates on the new kernel."

bd create \
  --title "Phase 5 — 20B determinism + perf gate (closes l144i Gate 4)" \
  --type task --priority 0 \
  --description "llama-completion gpt-oss-20b-mxfp4 with USE_XMX=0: 3 identical runs, coherent English. llama-bench 20B: TG>=15, PP>=54. Closes llama.cpp-l144i."

bd create \
  --title "Phase 6 — Flip default, retire env gate" \
  --type task --priority 1 \
  --description "Default dispatcher to scalar on Arc B580 (or globally). Remove GGML_SYCL_FA_USE_XMX env var reads. Update CLAUDE.md. PR with ggml-ci tag."
```

All Phase 1–5 beads depend on the epic bead as parent. Phase N+1 depends on Phase N. Phase 4.5 is conditional on Phase 4 failure.

## §10 — Open questions

1. **Parallel diagnostic outcome**: which of Branch A/B/C landed? Determines whether Phase 4+ goes forward or is re-scoped. If B, should this plan still land Phases 1–3 as regression infrastructure?
2. **Scope of Phase 6 default flip**: flip only on Arc B580 (`gpu_has_xmx && is_arc_b580`), flip on all XMX-capable devices, or flip globally? Default recommendation: flip on all Arc (B580 + Pro B50 + future) since DPAS is common hardware. But no data yet on whether other Arc SKUs exhibit the same nondeterminism.
3. **Retain XMX as opt-in after Phase 6?** User directive says "no workarounds / disabled features" — does keeping XMX behind an opt-in env var for power users violate that, or does it satisfy it (feature remains available, default is correct)?
4. **FP8 E4M3 KV cache**: is this actually exercised in any current model configuration in our environment, or is it speculative future work? If speculative, Phase 2's FP8 backfill can be deferred past Phase 6 (the feature is simply not used and not a regression).
5. **Paged V2 attention** (`GGML_SYCL_PAGED_V2=1`): it bypasses the dispatcher entirely. Determinism state of the paged V2 kernel is unknown — is it also XMX-based? Does it also exhibit nondeterminism? If yes, Phase 2/3/4 need a V2 parallel track. If V2 isn't on by default in any CI/user path, defer.
6. **Do we need a `test-fattn-determinism` that also runs on CPU backend and CUDA backend for cross-backend determinism comparison?** Valuable as a reference but maybe out of scope for this bead.
7. **Timeline**: is the user willing to accept a temporary PP512 Mistral regression during Phase 1–5 (only measurable with `GGML_SYCL_FA_USE_XMX=0`) as long as master default behaviour is preserved? Assumed yes because the gate is opt-in until Phase 6.
