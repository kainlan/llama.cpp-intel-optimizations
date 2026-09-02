# Upstream absorbed — every op SYCL claims is implemented and dispatched

- **Slug:** op-coverage-and-upstream-ports
- **Title:** Upstream absorbed — every op SYCL claims is implemented and dispatched
- **Epic ticket:** llama.cpp-1yr6
- **Priority:** 1
- **Description:** ## Goal
Close the honesty gap between what `docs/ops/SYCL.csv` claims the SYCL backend supports, what `ggml_backend_sycl_device_supports_op` actually admits, and what `ggml_sycl_compute_forward`'s dispatch switch actually reaches. Land the Phase-C queue of small, single-commit upstream SYCL ports identified by the post-b10630-merge audit (`docs/merge/upstream-b10630-sycl-audit.md`), implement `GGML_OP_TRI` (missing outright, not just undispatched), wire the seven implemented-but-undispatched ops from earlier lost merge hunks, and close the `bin_bcast`/`SET_ROWS`/`SSM_SCAN` capability holes that leave the scheduler silently falling back to CPU (correct but slow) instead of the SYCL kernel a model actually needs.

## Why now
The b10630 merge (2026-08-26, `1ebfa4e4a`) landed a large upstream sync and, per the audit, dropped or never carried forward 27 small SYCL commits plus assorted hunks from earlier merges. Every dropped hunk is currently invisible: `file(GLOB GGML_SOURCES_SYCL "*.cpp")` (`ggml/src/ggml-sycl/CMakeLists.txt:76`) auto-compiles any standalone `.cpp` a merge does carry, so a missing *dispatch* case produces no link error, no test failure (`supports_op`'s default arm returns false, so the scheduler correctly falls back to CPU), and no log line — "every signal says fine" while `docs/ops/SYCL.csv` keeps claiming coverage that isn't wired. Two of the seven op families found this way (COL2IM_1D, CONV_2D family, CONV_3D, CROSS_ENTROPY_LOSS/BACK) and `GGML_OP_TRI` are already known-broken from the earlier `llama.cpp-roz3` audit; the 27 [ports] tickets are the newer, ledger-tracked backlog from the same failure mode. Left undone, the gap compounds at every future merge.

## Current state (file:line refs)
- `docs/ops/SYCL.csv:14240-14242` lists `TRI` as `"support":"1"` — confirmed **false**: zero `GGML_OP_TRI` occurrences anywhere in `ggml/src/ggml-sycl/ggml-sycl.cpp` and no `tri.{cpp,hpp}` file exists (verified live, 2026-09-01).
- `GGML_OP_SSM_SCAN` **is** dispatched (`ggml-sycl.cpp:79308`, `:102696`; `ggml/src/ggml-sycl/ssm_scan.cpp` implements `ssm_scan_f32_group`/`ssm_scan_f32_sycl`/`ggml_sycl_op_ssm_scan`) — this contradicts `llama.cpp-xwnkf`'s premise (filed 2026-04-25, before the dispatch existed or was found). The live gap is narrower: `ssm_scan.cpp` does not yet read the `int64_t K` rollback-slot parameter the core `ggml_ssm_scan()` signature gained with the b10630 merge (`llama.cpp-8vwp`).
- `ggml/src/ggml-sycl/ggml-sycl.cpp` is 94k+ lines; codescout's index silently skips it as oversize — every finding above and in the member tickets was verified with `cat ... | grep -n`, not `search_text`, per CLAUDE.md.
- `ggml/src/ggml-sycl/ggml-sycl.cpp` carries 13 live `ggml_sycl_pool_alloc`/`ctx.pool()` sites (legacy allocator, pre-dates the unified-cache mandate) — one port in this epic (`llama.cpp-ag2d`) removes one of the 13; none of the other member ports touch this backlog.
- `fattn-onednn.cpp`'s oneDNN SDPA gate (`ggml_sycl_flash_attn_ext_onednn_supported()`) still hard-refuses any KV type other than F16 (`llama.cpp-g7hq`); `fattn-vec.hpp`'s flash-attention vec kernel still uses a flat 128-thread compile-time macro with no Battlemage-specific 256-thread path (`llama.cpp-ada5`/`llama.cpp-d6is`).

## Design constraints
- **Ruling 1/2** (unified cache is the sole allocator; `mem_handle` is the only ownership token): any port whose upstream diff stages scratch via `ggml_sycl_pool_alloc`/`ctx.pool()` (confirmed live in `llama.cpp-g7hq`'s upstream diff, 11 occurrences) must be re-expressed through `unified_allocate`/`unified_allocate_owner()` → `mem_handle` **before** landing, not pasted as-is. Most member ports (dequant/cpy/set_rows/concat type-table gaps, launch-geometry fixes) touch **no** allocation at all — confirmed per-ticket by grepping the upstream diff for `malloc`/`pool_alloc`/TLSF and getting zero hits; those are ownership-screen-clean by inspection, not by exemption.
- **Ruling 9** (correctness before throughput; CPU fallback is fine, wrong numbers are not): every currently-undispatched op runs correctly on CPU today via `supports_op`'s honest `default: return false`. Landing any of these ports can only introduce a *new* failure, never fix an existing wrong answer — every port needs a pre-change baseline (`test-backend-ops -o <OP>`, lead-run) to compare against, not just a post-change green.
- **Ruling 11** (verify a knob still exists before restoring it): `llama.cpp-hkuh`/`llama.cpp-8vwp` both cite `depends on: llama.cpp-90ns`, a task id absent from the open tracker — the core-ggml-wave prerequisite it names (the `ggml_ssm_scan` `K` param, `ggml_rope_set_offset()`) has already landed with the b10630 merge itself (confirmed present in `ggml/include/ggml.h`); the dependency edge is stale and should be cleared, not treated as a live blocker.
- **Ruling 10** (device selection belongs to oneAPI): `llama.cpp-jah9`'s iGPU-classification port must only change what `ggml_backend_sycl_device_get_type()` *reports* (GPU vs IGPU) — it must not add any parsing of or special-casing on `ONEAPI_DEVICE_SELECTOR`.

## External / upstream status
See `web_findings` below for full citations. Summary: the b10630 merge already absorbed the bulk of upstream SYCL work through late August 2026; the remaining gap is specifically the 27-commit Phase-C audit ledger plus the pre-existing lost-hunk ops. Two of the audited ports carry real, hardware-relevant upstream-measured payoffs on this fork's own card family (B70/B50): `llama.cpp-g7hq` (oneDNN SDPA for quantized KV) shows 2.37–3.21x prefill speedup upstream on Qwen/Gemma at Q4_K_XL/Q5_K_XL; `llama.cpp-9ppc`/`llama.cpp-922d` (Q2_0 mul_mat/cpy) closed a SYCL unit-test crash upstream, not just a coverage gap. A live, still-open upstream issue (#21893, filed 2026-04-14, unconfirmed) reports weight corruption on B70 with `GGML_SYCL_DEVICE_ARCH=bmg_g21` + F16 unless `GGML_SYCL_DISABLE_OPT=1` is set — no fix has landed upstream; this epic's ports do not touch the code path implicated (Xe2 DPAS reorder/sync), but any B70 correctness gate run for this epic should note the workaround exists if a similar symptom appears.

## Work breakdown (prerequisites first)
1. `llama.cpp-8cou` — wire the 7 implemented-but-undispatched lost-hunk ops (COL2IM_1D, CONV_2D family, CONV_3D, CROSS_ENTROPY_LOSS/BACK); defines the CSV-honesty bar and the parity-check method the rest of the epic is measured against.
2. `llama.cpp-tu4t` — implement `GGML_OP_TRI` from scratch (standalone `tri.cpp`/`tri.hpp`, `sycl_tensor`/`resolve_as<T>()` convention, not upstream's raw pointer paste).
3. `llama.cpp-9uic`, `llama.cpp-9vm3` — close the `bin_bcast` stride/type-chain gaps and the `SET_ROWS` F16-src gap, both already-certified-honest declines that should become real coverage.
4. `llama.cpp-jah9` — iGPU classification fix (small, standalone, unblocks nothing else but should land early since other tickets reference the same `common.hpp`/`ggml-sycl.cpp` device-info neighborhood).
5. `llama.cpp-5pgc` — zero-devices-not-abort robustness fix (standalone, low risk).
6. Independent single-file type-table ports (parallelizable, no ordering constraint between them): `llama.cpp-0w52` (dmmv row off-by-one), `llama.cpp-1kcx` (conv2d_dw F16), `llama.cpp-3zzs` (ssm_conv coalescing), `llama.cpp-tjt9` (get_rows Q2_K/Q4_K/Q5_K), `llama.cpp-h2uz` (UE4M3 dead-branch cleanup), `llama.cpp-qz0f` (concat launch geometry), `llama.cpp-xoi2` (cpy launch geometry), `llama.cpp-vvci` (*glu flat path), `llama.cpp-tp03` (xielu), `llama.cpp-al2o` (OPT_STEP_ADAMW/SGD), `llama.cpp-kdcm` (FWHT Hadamard hint), `llama.cpp-v7lu` (TQ2_0 refusal correctness), `llama.cpp-ag2d` (gemm scratch removal), `llama.cpp-tfap` (concat quantized types), `llama.cpp-596y` (set_rows missing types).
7. `llama.cpp-9ppc` + `llama.cpp-922d` — land together (Q2_0 mul_mat + Q2_0 cpy; the type is unusable end-to-end without both).
8. `llama.cpp-ada5` + `llama.cpp-d6is` — land together, single PR (256-thread fattn-vec for Battlemage + the `if constexpr` companion fix that prevents a 64KB work-group overflow entry 3 alone would introduce). Sequence with/after any attention-path work in the `prefill-xmx-int8-peak` epic to avoid two people in `fattn*` at once.
9. `llama.cpp-g7hq` — oneDNN SDPA for non-FP16 KV; depends on the ownership screen (re-express new dequant-to-F16 scratch through `mem_handle`, not `ctx.pool()`); sequence after `llama.cpp-ada5`/`llama.cpp-d6is` since both touch `fattn*`.
10. `llama.cpp-8vwp` — ssm_scan `K` rollback param (core signature already landed; only the SYCL kernel write is missing).
11. `llama.cpp-hkuh` — rope `n_offs` threading (core `ggml_rope_set_offset()` already landed).
12. `llama.cpp-pmzl` — DSv4 ops (`LIGHTNING_INDEXER`, `DSV4_HC_*`); blocker (`llama.cpp-t9l9`, the merge) is closed, so this is now unblocked — lowest priority of the substantial ports since no DSv4 model exists locally to correctness-gate against.
13. `llama.cpp-j6qq` — administrative only: file the CUDA argmax sentinel defect upstream, referencing this fork's own `llama.cpp-f7jd`/`llama.cpp-irdj` fixes as precedent; close once filed.
14. `csv-parity-gate` (new task) — regenerate `docs/ops/SYCL.csv` from the live dispatch table and add the standing CI check; run last, after the bulk of the op-coverage ports above have landed, so the regenerated CSV reflects the post-epic reality rather than needing a second pass.

## Bar / closing gate
Every [ports] ticket lands with its upstream sha cited in the commit message, or closes with a stated not-applicable reason. Concrete commands (lead-run only, single invocation, never in a subagent — memory-exhaustion hazard):
```bash
source /opt/intel/oneapi/setvars.sh --force
# Per-op correctness (repeat per op the port touches; one at a time)
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o <OP>
# Full-suite regression gate once the queue is landed (excludes the OOM-hazard family per CLAUDE.md)
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build --output-on-failure -j 1 \
  -LE 'residency|mem-handle|cache' -E '^test-backend-ops$'
# Correctness gates (must still pass byte-identical after every landed port)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0   # expect: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli -m /models/gpt-oss-20b-mxfp4.gguf -ngl 99 -c 4096 \
  -cnv -st --simple-io --no-display-prompt --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0   # expect: 1, 2, 3, 4, 5
# Architecture regression (single-arch runs affected by ssm_scan/rope/DSv4 ports)
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a mamba2; echo "rc=$?"   # rc=0 required, not the corruptible results table
```
For `llama.cpp-g7hq` and `llama.cpp-ada5`/`llama.cpp-d6is` specifically: an interleaved paired A/B `llama-bench -p 512 -n 128` on both B70 and B50 must show no PP512/TG128 regression against `docs/backend/sycl-perf-baselines.md`. `docs/ops/SYCL.csv` regenerated (not hand-edited) with zero rows claiming an op the dispatch table cannot reach.

## Out of scope
- The legacy `ggml_sycl_pool_alloc`/`ctx.pool()` migration backlog SYCL-backend-wide (referenced by `llama.cpp-ag2d` and `llama.cpp-g7hq` as "a separate ticket") — no such ticket exists among this epic's members; if the lead wants it tracked, it needs its own ticket, not folded in here.
- `llama.cpp-achg`, `llama.cpp-oqv9`, `llama.cpp-7phf`, `llama.cpp-17zg` — open tasks that `llama.cpp-1yr6` (the pre-existing epic ticket) lists as dependencies but which are NOT members of this taxonomy epic; they belong to a different epic/theme and are left untouched here.
- Any attention-kernel work beyond the two named fattn ports (`ada5`/`d6is`, `g7hq`) — broader flash-attention throughput work belongs to `prefill-xmx-int8-peak`.
- The upstream #21893 B70 weight-corruption issue — unconfirmed, no upstream fix, not caused by or fixed by any port in this queue; noted only as a risk to watch during B70 verification.

## Dependencies on other epics
- Sequence `llama.cpp-ada5`/`llama.cpp-d6is`/`llama.cpp-g7hq` (all touch `fattn*`) against whatever the `prefill-xmx-int8-peak` epic is doing in the same files, per the taxonomy ordering hint, to avoid two people editing `fattn-vec.hpp`/`fattn-onednn.cpp` concurrently.
- None of this epic's members depend on the memory-ownership epic's zone-reset elimination work landing first — every port here is either allocation-free or additive against the unified cache as already required.
## Tasks

### llama.cpp-1yr6

  - **Id:** llama.cpp-1yr6
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** This is the umbrella epic ticket itself (existing_epic_id for this taxonomy epic) — it tracks the 27-commit Phase-C upstream-SYCL port queue plus the TRI/7-op/bin_bcast/set_rows/ssm_scan capability-gap tickets as one closing unit.

**Design:** No independent code change. Its dependency list (`llama.cpp-ada5, llama.cpp-tp03, ...`) should be reconciled against this epic's actual member list: several ids it names (`llama.cpp-achg`, `llama.cpp-oqv9`, `llama.cpp-7phf`, `llama.cpp-17zg`) are NOT members of this taxonomy epic and belong elsewhere; two ids it names (`llama.cpp-90ns`, `llama.cpp-t9l9`) are no longer open (the merge landed) and their dependency edges on `llama.cpp-8vwp`/`llama.cpp-hkuh`/`llama.cpp-pmzl` are stale and should be cleared.

**Bar:** Closes only when every member ticket in the Work breakdown above is landed-with-sha or closed-not-applicable, and the epic-level closing gate (docs/ops/SYCL.csv regeneration + parity CI check + test-backend-ops sweep + Mistral/GPT-OSS gates) passes.

**Constraints:** All rulings cited in the epic description apply transitively to every child.

**Acceptance test:** See epic ## Bar / closing gate.

**Depends on:** every task in this epic's Work breakdown.

### llama.cpp-8cou

  - **Id:** llama.cpp-8cou
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Wire the 7 remaining implemented-but-undispatched ops (COL2IM_1D, CONV_2D/CONV_2D_DW/CONV_TRANSPOSE_2D, CONV_3D, CROSS_ENTROPY_LOSS/CROSS_ENTROPY_LOSS_BACK) into both `ggml_sycl_compute_forward` and `supports_op`, and regenerate `docs/ops/SYCL.csv` so it stops lying.

**Design:**
- Add each op's `case GGML_OP_*` to `ggml_sycl_compute_forward`'s switch, calling the already-compiled-and-linked `ggml_sycl_*` entry point (confirmed present in `libggml-sycl.so.0` via `nm -C --defined-only`).
- Add matching `case` arms to `ggml_backend_sycl_device_supports_op`, carrying each commit's original type/shape guards verbatim (not `return true`).
- Add any missing forward declarations to `backend.hpp`.
- Both switches must end with identical `case` sets, no duplicates (the `32b210a25` invariant, currently 72/72).
- Write the standing parity check this ticket's acceptance criteria already call for: a script comparing `ggml_sycl_*` defined symbols against dispatched `GGML_OP_*` cases, wired into CI so this class of regression cannot recur silently at the next merge.
- Regenerate `docs/ops/SYCL.csv` from the corrected dispatch table.

**Bar:** `test-backend-ops -o <OP>` green per op on the B70 (7 separate lead-run invocations, never batched/looped); CSV round-trips clean against the parity script; the parity script itself must be demonstrated to fire on a synthetic "missing dispatch" case (positive control) before being trusted.

**Constraints:** Ruling 9 — these 7 ops run correctly on CPU today; landing this can only introduce a new failure. Compare every post-change run against a pre-change baseline, not just green-in-isolation.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o COL2IM_1D` (repeat for the other 6 ops), lead-run, one at a time.

**Depends on:** none — this is a prerequisite for the epic's CSV-honesty bar; the `csv-parity-gate` new task builds on its parity-script output.

### llama.cpp-tu4t

  - **Id:** llama.cpp-tu4t
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Implement `GGML_OP_TRI` for SYCL from scratch — confirmed absent entirely (zero occurrences in `ggml-sycl.cpp`, no `tri.{cpp,hpp}` file), unlike the 7-op sibling gap this is not a wiring problem.

**Design:**
- `git show c7358ddf6 -- ggml/src/ggml-sycl/ggml-sycl.cpp` for the 69-line upstream reference implementation (inline, no standalone file upstream).
- Do NOT paste it inline. Create standalone `ggml/src/ggml-sycl/tri.cpp` + `tri.hpp`, matching this fork's convention (`ssm_conv.cpp`/`roll.cpp`/`gla.cpp`/`wkv.cpp`/`count_equal.cpp`) — the CMake glob picks it up automatically.
- Use `sycl_tensor` + `resolve_as<T>()` and `safe_dst`, NOT upstream's raw `tensor->data` access — raw pointers are transient ABI views per this fork's ownership contract, and `ggml_sycl_get_data_ptr` (`common.hpp:3842`) consults `extra->data_device_ptr(device)` first; a raw-pointer paste is benign single-device but wrong under any cross-device storage.
- Dispatch in both `ggml_sycl_compute_forward` and `supports_op`, preserving upstream's guards.
- Fix `docs/ops/SYCL.csv`'s existing false `TRI: support=1` row (currently the only op where the CSV claims total support with zero implementation, vs. 8cou's ops which are compiled-but-unwired).

**Bar:** `test-backend-ops -o TRI` green on the B70.

**Constraints:** Ruling 2 (mem_handle/no raw pointers as source of truth). Ruling 9 — TRI runs correctly on CPU today; this can only introduce a new failure, sequence accordingly.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o TRI`, lead-run only, never in a subagent (memory-exhaustion hazard).

**Depends on:** none.

### llama.cpp-xwnkf

  - **Id:** llama.cpp-xwnkf
  - **Action:** close
  - **Close evidence:** Premise is stale/false. Ticket claims 'ggml/src/ggml-sycl has no GGML_OP_SSM_SCAN support/dispatch entry' (filed 2026-04-25). Verified live 2026-09-01: `ggml/src/ggml-sycl/ggml-sycl.cpp:79308` and `:102696` both carry `case GGML_OP_SSM_SCAN:`, and `ggml/src/ggml-sycl/ssm_scan.cpp` fully implements `ssm_scan_f32_group`/`ssm_scan_f32_sycl`/`ggml_sycl_op_ssm_scan`. SSM_SCAN dispatch and routing exist and are tested. The one real remaining gap — the SYCL kernel not yet reading the `int64_t K` rollback-slot parameter the core `ggml_ssm_scan()` signature gained with the b10630 merge — is already tracked precisely by `llama.cpp-8vwp`, which supersedes this ticket's intent. Close as superseded-by-llama.cpp-8vwp; the lead should re-verify the two grep results above before closing.
  - **Priority:** 3
### Depends on

  - _(empty)_

### llama.cpp-8vwp

  - **Id:** llama.cpp-8vwp
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `1692f9e50bb2` — the core `ggml_ssm_scan()` signature gained a trailing `int64_t K` parameter (recurrent-state rollback, K rollback slots instead of always 1) with the b10630 merge; the SYCL side never picked up the matching hunk. **This ticket absorbs `llama.cpp-xwnkf`'s residual scope** — xwnkf's original premise ("SYCL has no SSM_SCAN dispatch at all") is false and closed as superseded-by-this-ticket, but the one real gap xwnkf's audit surfaced (the kernel not reading `K`) lives here.

**Design:**
- `ggml/src/ggml-sycl/ssm_scan.cpp` (`ssm_scan_f32_group`, `ssm_scan_f32_sycl`, `ggml_sycl_op_ssm_scan`) — confirmed live 2026-09-01: no `K`/`ggml_get_op_params_i32` usage anywhere in the file; matches the pre-patch (K==1-implicit) signature verbatim.
- Dispatch and routing already exist (`ggml-sycl.cpp:79308`, `:102696`) and need no change — this is a snapshot-write hunk inside the existing kernel, not a new dispatch case.
- **Ownership surface:** none — writes the rollback snapshot into the pre-existing `dst` tensor buffer at a computed offset; no allocation, so no ownership-screen work needed.
- **Ruling 11:** clear the stale `depends on: llama.cpp-90ns` edge — that task id is absent from the open tracker; the core `ggml.h` signature change it names already landed with the b10630 merge (confirmed present in `ggml/include/ggml.h`).

**Bar:** SSM_SCAN with `K > 1` produces the correct rolled-back recurrent state; existing `K == 1` behavior (today's only exercised case) is unchanged.

**Constraints:** Ruling 9 — no allocation surface, in-buffer index/offset math only; verify against a pre-change baseline.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o SSM_SCAN`, lead-run, covering a `K > 1` case; then `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a mamba2; echo "rc=$?"` (rc=0 required — no local Nemotron-style model exists to exercise `K>1` end-to-end, so this remains correctness-gated at the op level until one is available).

**Depends on:** none (the `llama.cpp-90ns` dependency is stale; the merge already landed the prerequisite).

### llama.cpp-0w52

  - **Id:** llama.cpp-0w52
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `0bd0ec60998d` — fix an off-by-one row-bounds check in the non-reorder Q2_K/Q3_K/Q4_K DMMV kernels.

**Design:**
- `ggml/src/ggml-sycl/dmmv.cpp:1981` (`dequantize_mul_mat_vec_q2_k`): `if (row > nrows)` → `if (row >= nrows)`.
- Same fix at `:2089` (`_q3_k`) and `:2203` (`_q4_k`).
- Q6_K (`:2475`) is already correct (`>=`) — no change needed there.
- Q5_K needs no fix: the fork's non-reorder Q5_K kernel already distributes both `im` halves across separate work-group threads (`:2366-2374`) and its launcher already grids exactly `nrows` groups with `local_range(1)==1` (`:2941-2952`), making an `nrows` guard structurally inert; the reorder-side Q5_K restructure upstream also ships is unreachable in this fork (`ggml_sycl_supports_reorder_dmmv()` returns true only for Q4_0).

**Bar:** Three one-character diffs; DMMV output for Q2_K/Q3_K/Q4_K unchanged in the common case, corrected only at the boundary row.

**Constraints:** Ruling 9 — no allocation surface, pure bounds-comparison fix; verify against a pre-change baseline since it changes which rows are computed at the edge.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o MUL_MAT` with Q2_K/Q3_K/Q4_K shapes at odd row counts (the DMMV kernels are reached via the MUL_MAT op family — there is no separate `MUL_MAT_VEC` op in `ggml_op_desc`), lead-run.

**Depends on:** none.

### llama.cpp-1kcx

  - **Id:** llama.cpp-1kcx
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `ae9291e16b97` — add an F16 kernel-tensor code path to depthwise conv2d.

**Design:**
- `ggml/src/ggml-sycl/conv2d-dw.cpp:122` currently asserts `kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32` unconditionally.
- Add the F16-kernel branch upstream's commit provides, following its type-dispatch shape.
- No interaction with any rewritten subsystem (mul_mat/fattn/memory/graph/MoE) — this op sits outside all of them.

**Bar:** CONV_2D_DW with an F16 kernel tensor produces correct output; F32 path unchanged.

**Constraints:** Ruling 9 — no ownership surface (in-place op on existing buffers); compare against pre-change baseline.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o CONV_2D_DW`, lead-run only.

**Depends on:** none.

### llama.cpp-3zzs

  - **Id:** llama.cpp-3zzs
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `f8e30266d23f` — reorder `kernel_ssm_conv`'s index decomposition from channel-fastest to token-fastest for coalesced loads.

**Design:**
- `ggml/src/ggml-sycl/ssm_conv.cpp` function `kernel_ssm_conv`: change the flattened-index decomposition from `channel = idx % d_inner; token = (idx / d_inner) % n_t` to upstream's token-fastest layout (confirmed the fork is still byte-for-byte the pre-patch layout).
- Pure index-math reordering; same src/weights/dst pointers, no allocation, no interaction with any rewritten subsystem.
- Capture a `test-backend-ops perf -o SSM_CONV` before/after on this fork's B70/B50 — the claimed 1.85–1.87x is an upstream measurement, not yet validated on this hardware/driver combination.

**Bar:** SSM_CONV throughput measurably improves (target: upstream's 1.85–1.87x, allow this fork's noise floor); +1.8–2.2% end-to-end pp2048 on a Mamba-family model if one is available locally, else SSM_CONV microbenchmark alone suffices.

**Constraints:** Ruling 9 — no allocation surface; run the Mistral/GPT-OSS correctness gates since this touches a live kernel path, even though those models don't exercise SSM_CONV directly (the gate proves the build/dispatch didn't break elsewhere).

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops perf -o SSM_CONV`, lead-run, before/after comparison.

**Depends on:** none.

### llama.cpp-596y

  - **Id:** llama.cpp-596y
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `c074cb3f763d` — add `set_rows` support for the K-quant and IQ types the fork's dispatch currently omits.

**Design:**
- `ggml/src/ggml-sycl/set_rows.cpp` dispatch (~lines 935-976) currently covers only F32/F16/BF16/Q8_0/Q5_1/Q5_0/Q4_1/Q4_0/IQ4_NL.
- Add `Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_XS` arms.
- Reuse `dequantize.hpp`/`vecdotq.hpp` quantize helpers where already ported by sibling tickets (`llama.cpp-tjt9` adds Q2_K/Q4_K/Q5_K dequant helpers for get_rows — check whether they're reusable here before writing new ones).

**Bar:** Every newly-added type moves from `not supported [SYCL0]` to `OK` in `test-backend-ops -o SET_ROWS` output.

**Constraints:** Ruling 9 — no allocation surface confirmed (`grep`'d for malloc/pool_alloc in set_rows.cpp — no hits); per-row quantize math writing into the existing destination buffer only.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o SET_ROWS`, lead-run, count each newly-covered type explicitly.

**Depends on:** consider sequencing after `llama.cpp-tjt9` if dequant helper reuse is worthwhile; not a hard blocker either way.

### llama.cpp-5pgc

  - **Id:** llama.cpp-5pgc
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `6cc504a2e90d` — report zero SYCL devices instead of letting a `dpct::dev_mgr::instance().device_count()` exception crash SYCL-agnostic tools (e.g. `llama-quantize`) on a host with the SYCL runtime installed but no usable device.

**Design:**
- `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_sycl_init()` (currently `raw_device_count` acquisition, unguarded) — wrap in try/catch, fall back to `info.device_count = 0` plus a log line on exception.
- This sits ahead of, and is unaffected by, the fork's own `GGML_SYCL_VISIBLE_DEVICES`/`GGML_SYCL_DEVICE` device-map parsing that runs after the raw count.
- No interaction with any rewritten subsystem — small, low-risk robustness fix.

**Bar:** A SYCL-agnostic binary run on a host/environment with the SYCL runtime present but no device degrades to `device_count=0` and continues, instead of throwing.

**Constraints:** Ruling 10 — do not let this become a place to add device-selector parsing/validation; it only catches an enumeration exception.

**Acceptance test:** No reliable way to induce a real `dpct::dev_mgr::instance().device_count()` throw exists on this host (an invalid `ONEAPI_DEVICE_SELECTOR` value yields zero enumerated devices, not a throw, so it cannot produce the RED state this fix targets) — treat the bar as **code-inspection plus a no-regression run**, not a reproduced-throw test: (1) inspect the diff to confirm the try/catch wraps exactly the `device_count()` call and falls back to `info.device_count = 0` plus a log line; (2) run `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-quantize --help` (metadata-only, no device touch) to confirm no regression in the unexceptional path. If the lead later finds a concrete way to force the throw (e.g. a broken/stubbed Level Zero loader `.so`), prefer that as a stronger positive control, but do not block the port on finding one.

**Depends on:** none.

### llama.cpp-922d

  - **Id:** llama.cpp-922d
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-9ppc

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `d5d3e05bf8d2` — add Q2_0↔F32 cpy (quantize/dequantize) support, pairing with `llama.cpp-9ppc`'s Q2_0 mul_mat so the type is usable end-to-end.

**Design:**
- `ggml/src/ggml-sycl/cpy.hpp`: new inline `cpy_blck_f32_q2_0` (F32→Q2_0 quantize).
- `ggml/src/ggml-sycl/cpy.cpp`: new static `cpy_blck_q2_0_f32` (Q2_0→F32 dequantize) plus the dispatch-table pairing (corrected assignment per `git show --stat d5d3e05bf8d2`: the ledger's earlier draft had both in `cpy.hpp`, which is wrong).
- Land in the same change/PR as `llama.cpp-9ppc` — Q2_0 has no usable end-to-end path without both.

**Bar:** CPY correctness for the F32↔Q2_0 pair; Q2_0 tensors can round-trip through cpy the same way every other quant type does.

**Constraints:** Ruling 9 — no allocation surface confirmed (grep for malloc/pool_alloc — no hits); per-block quantize/copy math over existing src/dst buffers only.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o CPY` with the new Q2_0 type pair, lead-run.

**Depends on:** `llama.cpp-9ppc` (land together).

### llama.cpp-9ppc

  - **Id:** llama.cpp-9ppc
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `a2be61dc8794` — add MMVQ dot-product and dequantize support for `GGML_TYPE_Q2_0`, which currently has zero mul_mat path despite the type existing in `ggml.h`.

**Design:**
- `ggml/src/ggml-sycl/{dequantize.hpp,convert.cpp,vecdotq.hpp,mmvq.cpp}` — add Q2_0 dequant/dot-product/convert arms, purely additive new `case` entries against the fork's own MMVQ dispatch (not a rewrite of it).
- Land paired with `llama.cpp-922d` (cpy-side Q2_0 support).
- Upstream confirms real payoff: this exact commit resolved a SYCL unit-test **crash** (not just a coverage gap) — `test-backend-ops` UT cases added for Q2_0 mul_mat previously crashed the SYCL backend since the type was entirely unhandled (verified via upstream PR #26231, merged 2026-07-31).

**Bar:** MUL_MAT with Q2_0 src produces correct output instead of a crash/abort; no real Q2_0-quantized model exists locally, so use a synthetic test tensor via `test-backend-ops`.

**Constraints:** Ruling 9 — no allocation surface confirmed (grep for malloc/pool_alloc in mmvq.cpp/convert.cpp — no hits); dot-product/dequant/convert code reading existing weight buffers only.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o MUL_MAT` with Q2_0 shapes, lead-run.

**Depends on:** none, but land with `llama.cpp-922d`.

### llama.cpp-ada5

  - **Id:** llama.cpp-ada5
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `c1063ac9d75f` — use a 256-thread work group for the flash-attention vec kernel on Battlemage (`bmg_g21`/`bmg_g31` — this fork's exact B50/B70 cards), replacing the flat 128-thread compile-time macro.

**Design:**
- `ggml/src/ggml-sycl/fattn-common.hpp:22`: replace the flat `#define FATTN_VEC_NTHREADS 128` with an arch-conditional helper (detect `gpu_arch::intel_gpu_bmg_g21/g31/lnl_m`).
- `ggml/src/ggml-sycl/fattn-vec.hpp`: the launch site currently hardcodes `nthreads` from the macro — switch to the new helper.
- **MUST land in the same commit/PR as `llama.cpp-d6is`** — alone, this port instantiates the D>256, nthreads==256 template branch even though it's never executed, overflowing 64KB work-group local memory. `llama.cpp-d6is` converts the runtime `if` to `if constexpr` specifically to prevent that instantiation; write it that way from the start rather than porting this entry and then immediately following with the fix.

**Bar:** Flash-attention correctness gate passes on both cards; B70/B50 fattn PP throughput does not regress against `docs/backend/sycl-perf-baselines.md`, and should improve on the affected shapes per upstream's measurement.

**Constraints:** Ruling 9 — pure launch-geometry constant, no allocation. Sequence with/after the `prefill-xmx-int8-peak` epic's attention-path work to avoid concurrent edits to `fattn*`.

**Acceptance test:** Mistral completion gate (`llama-completion` — must still emit `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`) plus `llama-bench -p 512 -n 128` on B70 and B50, interleaved paired A/B against baseline.

**Land together:** `llama.cpp-d6is` (no dependency edge from this ticket — the edge lives on `llama.cpp-d6is` pointing back to this one, to avoid a mutual-dependency cycle; both must land in the same commit/PR regardless of which surfaces as "ready" first).

### llama.cpp-ag2d

  - **Id:** llama.cpp-ag2d
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `154d57af3eec` — in the non-oneDNN fallback GEMM branch, let `dpct::gemm` promote directly to F32 output instead of writing F16 to a scratch buffer and running a separate conversion kernel.

**Design:**
- `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_sycl_op_mul_mat_sycl`, non-oneDNN F16 GEMM branch — the real landing zone is `dst_f16(ctx.pool())` at `:42119` and the `dpct::gemm` call at `:42123` (the `:41320-41335`/`:41327` coordinates in this ticket's original filing are stale and land inside an unrelated POOL_AVG kernel; re-locate live via `grep -n 'dst_f16(ctx.pool())'` before editing) — replace the `ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols)` + `to_fp32_sycl` conversion pair with a single `dpct::gemm` call that promotes directly to F32.
- This is a **rare ownership-screen-favorable** port: it REMOVES one of the fork's 13 `ggml_sycl_pool_alloc`/`ctx.pool()` legacy sites rather than adding one, moving this call site closer to canonical-contract compliance without needing new `mem_handle` plumbing.
- Verify `dpct::gemm`'s direct-F32-output signature against the installed oneAPI 2025.3 dpct headers before assuming API parity with upstream's version.
- Leave the two nearby, textually-similar-looking sites alone once re-located (the input-promotion `src0_as_f16` site immediately before, and the separate `src0_ddq_as_f32`/`src1_ddq_as_f32` f32-precision GEMM arm immediately after).

**Bar:** Bit-identical output before/after on the Mistral completion gate; `dpct::gemm` call count and pool-alloc census (13 → 12) confirm the intended removal.

**Constraints:** Ruling 9 — net effect removes an allocation, introduces none.

**Acceptance test:** Build `ggml-sycl` target, run the Mistral completion gate plus `test-backend-ops -o MUL_MAT` on the non-oneDNN fallback path specifically, lead-run.

**Depends on:** none.

### llama.cpp-al2o

  - **Id:** llama.cpp-al2o
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `37a215c9e909` — add SYCL-backend training/fine-tuning support via `OPT_STEP_ADAMW`/`OPT_STEP_SGD` optimizer-step kernels.

**Design:**
- New `ggml/src/ggml-sycl/opt-step.{cpp,hpp}` implementing `ggml_sycl_opt_step_adamw`/`ggml_sycl_opt_step_sgd` (in-place parameter update from gradient + optimizer state).
- Wire `GGML_OP_OPT_STEP_ADAMW`/`GGML_OP_OPT_STEP_SGD` into `ggml_sycl_compute_forward`'s op switch and `ggml_backend_sycl_device_supports_op`.
- Confirmed absent entirely today — no existing SYCL-backend training test to gate against, so this ticket must define its own acceptance bar rather than reuse an existing one.

**Bar:** A CPU-vs-SYCL numeric parity check (small synthetic optimizer-step tensor, compare AdamW/SGD update output between backends) since no existing test-backend-ops case covers this without the port landing first.

**Constraints:** Ruling 9 — in-place update of existing parameter/gradient/optimizer-state buffers, no allocation.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o OPT_STEP_ADAMW` and `-o OPT_STEP_SGD` once the port adds test coverage for these ops, lead-run.

**Depends on:** none.

### llama.cpp-d6is

  - **Id:** llama.cpp-d6is
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-ada5

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `eef5f3e3430a` — the required companion fix for `llama.cpp-ada5`: convert a runtime `if (D <= 256 && nthreads == 256)` to `if constexpr (D <= 256) { if (nthreads == 256) {...; return;} }` to prevent template instantiation of the `nthreads==256` branch at `D > 256`, which overflows 64KB work-group local memory even when never executed.

**Design:**
- Same landing zone as `llama.cpp-ada5` — the arch-conditional launch dispatch in `ggml/src/ggml-sycl/fattn-vec.hpp`.
- Write this `if constexpr` form from the start when implementing `ada5`, rather than porting `ada5`'s runtime `if` first and patching it afterward — there is no intermediate state where `ada5` alone is safe to ship.
- Has **zero standalone value** without `ada5` — the fork's current flat-macro `fattn-vec.hpp` has no per-arch dispatch for this fix to apply to.

**Bar:** No 64KB work-group local-memory overflow at D>256 shapes; flash-attention correctness gate passes for every D size the fork's models exercise.

**Constraints:** Ruling 9 — compile-time template-instantiation fix only, no allocation, no runtime behavior change beyond fixing the overflow.

**Acceptance test:** Same as `llama.cpp-ada5` — both land in one commit/PR and are verified together.

**Depends on:** `llama.cpp-ada5` (single PR, not sequential; task_dep should be added so this cannot be picked up alone).

### llama.cpp-g7hq

  - **Id:** llama.cpp-g7hq
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-ada5
  - llama.cpp-d6is

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `66fa168a5617` — extend the oneDNN SDPA gate to accept Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and F32 KV caches (currently hard-refused to F16-only), by dequantizing/converting K/V to dense F16 before the SDPA graph.

**Design:**
- `ggml/src/ggml-sycl/fattn-onednn.cpp`, `ggml_sycl_flash_attn_ext_onednn_supported()` (line ~277) — relax the `params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16` hard gate.
- Reuse the fork's existing `MATERIALIZE_REQUIRED` K/V materialization machinery (already unified-cache-backed, per the file's own comments) for the new dequant-to-F16 conversion step, rather than writing a parallel staging path.
- **Ownership screen is load-bearing here**: upstream's own diff stages the converted K/V in `ggml_sycl_pool_alloc<sycl::half>` scratch (11 occurrences, confirmed via `git show b10630:.../fattn-onednn.cpp`) — that is upstream's code and must NOT be copied as-is. The fork's file currently has ZERO `ggml_sycl_pool_alloc`/`ctx.pool()` occurrences (already fully on the unified-cache surface). The NEW dequant-to-F16 scratch this port adds must go through `unified_allocate`/`unified_allocate_owner()` → `mem_handle` (or `alloc_owner` via `mem_handle::from_owned_alloc()`), matching the standard the rest of the file already meets.
- Gated to prefill-only, K≥1024, Q≥32 (matches upstream's own gating conditions).

**Bar:** Upstream measured 2.37–3.21x prefill speedup on Qwen 27B Q5_K_XL / Gemma 12B+E4B Q4_K_XL on Intel Arc GPUs (PR #25874, merged 2026-08-04) — this fork's own perf baselines already benchmark "q8_0 KV" runs that cannot reach this path today; expect a comparable prefill win once landed, verified on B70/B50.

**Constraints:** Rulings 1/2 — the new scratch allocation is the one part of this port that must be re-expressed before landing, not after. Sequence after `llama.cpp-ada5`/`llama.cpp-d6is` (same `fattn*` neighborhood).

**Acceptance test:** A correctness gate for quantized-KV prefill is currently missing from this fork's canonical gates — add a perplexity or completion-gate run with `--cache-type-k q8_0 --cache-type-v q8_0` on Mistral, confirm output matches the F16-KV baseline within tolerance; then `llama-bench` PP512 A/B on B70/B50 against `docs/backend/sycl-perf-baselines.md`.

**Depends on:** `llama.cpp-ada5`, `llama.cpp-d6is` (sequencing, not a hard code dependency).

### llama.cpp-h2uz

  - **Id:** llama.cpp-h2uz
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `fc3f10b3895e` — drop a dead `x == 0xff → 0.0f` special case from `ggml_sycl_ue4m3_to_fp32()`, left over from upstream's old buggy signed-decode formula which the fork's current unsigned-decode formula already fixed.

**Design:**
- `ggml/src/ggml-sycl/common.hpp`, function `ggml_sycl_ue4m3_to_fp32` (now at line ~7086, drifted from the ticket's originally-cited 6797) — drop the `x == 0xff` clause from the leading guard (`x==0 || x==0x7f || x==0xff`).
- Confirmed dead, not live-data: `ggml_fp32_to_ue4m3` saturates at 0x7E and never sets bit 7, so no real encoder can ever emit byte `0xFF`. The fork's own test documents this exact divergence (`ggml/src/ggml-sycl/tests/test-q1-nvfp4-adapter-device.cpp:271-276`).
- Update that test file's comment once the branch is removed — it currently documents a divergence that will no longer exist.

**Bar:** `ggml_sycl_ue4m3_to_fp32` matches `ggml_ue4m3_to_fp32`'s behavior exactly (no remaining special case); the nvfp4 adapter test's comment reflects reality.

**Constraints:** Ruling 9 — pure scalar decode function, no allocation, no live-data path reaches the removed branch (low risk by construction).

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o <the op exercising NVFP4/UE4M3 decode>` once, lead-run; not strictly required (dead branch) but cheap insurance.

**Depends on:** none.

### llama.cpp-hkuh

  - **Id:** llama.cpp-hkuh
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `749f688fcaa4` — thread a new `n_offs` parameter through the SYCL `rope_norm`/`rope_neox`/`rope_multi` kernels so only the `[n_offs, n_offs+n_dims)` channel range is rotated, and remove the placeholder guard that currently blocks it.

**Design:**
- `ggml/src/ggml-sycl/rope.cpp` (`rope_norm`, `rope_neox`, `rope_multi`) — add `n_offs` to the per-thread bounds check and index math; confirmed unmodified pre-patch today (`if (i0 >= n_dims)`, no `n_offs` anywhere).
- `ggml_backend_sycl_device_supports_op` (function begins `:102023`, not `:98736` — that coordinate is ~3.4k lines stale and lands inside the persistent-TG MoE graph probe; re-locate the ROPE placeholder guard live via `grep -n 'GGML_OP_ROPE' ggml/src/ggml-sycl/ggml-sycl.cpp` before editing) — remove the placeholder guard so `case GGML_OP_ROPE/ROPE_BACK:` falls through to the general `return true;` group.
- The core `ggml_rope_set_offset()` prerequisite (`ggml.h:2075`) already landed with the b10630 merge — confirmed present. **Clear the stale `depends on: llama.cpp-90ns` edge** — that task id does not exist in the open tracker; the merge already satisfied it.

**Bar:** ROPE/ROPE_BACK with a non-zero `n_offs` produces correct output (only the specified channel range rotates); existing zero-offset behavior unchanged.

**Constraints:** Ruling 9 — no allocation surface, index-math change over existing x/dst buffers.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o ROPE`, lead-run, must cover an `n_offs != 0` case once the port adds test coverage.

**Depends on:** none (the `llama.cpp-90ns` dependency is stale; the merge already landed the prerequisite).

### llama.cpp-jah9

  - **Id:** llama.cpp-jah9
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `272700b36094` — add `l0_device_type_valid` so a failed `zeDeviceGetProperties` query no longer silently misclassifies a device as integrated, and make `ggml_backend_sycl_device_get_type()` return `GGML_BACKEND_DEVICE_TYPE_IGPU` for actually-integrated GPUs instead of always `GGML_BACKEND_DEVICE_TYPE_GPU`.

**Design:**
- `ggml/src/ggml-sycl/common.hpp` (`sycl_device_info` struct) — add `l0_device_type_valid` field.
- `ggml/src/ggml-sycl/ggml-sycl.cpp` (`ggml_sycl_init()`'s L0 properties query, ~line 167; `ggml_backend_sycl_device_get_type()`, currently `ggml-sycl.cpp:101761`, unconditionally `return GGML_BACKEND_DEVICE_TYPE_GPU;`) — wire the new field through.
- Complements, does NOT duplicate, the fork's existing `host_unified_memory`/`ggml_sycl_device_is_host_unified()` VRAM-budget fix (llama.cpp-403s) — that fix scales the budget calculation; this fix lets ggml-core's backend scheduler itself see the iGPU as `IGPU` rather than a generic `GPU`.

**Bar:** The iGPU (Arrow Lake-S, `level_zero` device 2 in this fork's current numbering) reports `IGPU`; both discrete cards (B70, B50) report `GPU`.

**Constraints:** Ruling 10 — this must only change what the classification reports, never add parsing/refusal logic on `ONEAPI_DEVICE_SELECTOR` itself.

**Acceptance test:** No existing test asserts `device_get_type()`'s per-device return value — add a CPU-only/host-side unit check that calls `ggml_backend_sycl_device_get_type()` directly for each enumerated device and asserts IGPU-vs-GPU by `host_unified_memory` (the same signal `llama.cpp-403s`'s VRAM-budget fix already reads at `common.cpp:716` — no new device query needed). Do NOT verify this by running any binary with the iGPU unpinned (CLAUDE.md: 127.8 GB Shmem unpinned vs 2.4 GB pinned) and do not use `sycl-ls` (hang hazard). If a live run is ever needed to cross-check, it must be `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2 <binary>` under the same pre/post `Shmem`/`MemAvailable` sampling discipline as every other GPU command in this epic, lead-run only.

**Depends on:** none.

### llama.cpp-kdcm

  - **Id:** llama.cpp-kdcm
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `0882c7bc8907` — add a Fast Walsh-Hadamard Transform kernel and an early-return in `ggml_sycl_mul_mat()` that runs FWHT instead of GEMM when `GGML_HINT_SRC0_IS_HADAMARD` is set, measured 3.3–6x faster than GEMM on the affected shapes upstream.

**Design:**
- New `ggml/src/ggml-sycl/fwht.{cpp,hpp}` ("port of `ggml-cuda/fwht.cu`") implementing `ggml_sycl_op_fwht`.
- `ggml-sycl.cpp`: add the Hadamard-hint check at the very top of `ggml_sycl_mul_mat()` — re-verify the current line number before landing (ticket cites ~line 58456 as of 2026-08-26; the 94k-line file has grown since, re-grep `ggml_sycl_mul_mat(` first) — must sit BEFORE the fork's own customized layout/route dispatch (the fork-added `forced_layout` parameter), not compete with it.
- Core prerequisite (`GGML_HINT_SRC0_IS_HADAMARD`, `ggml_mul_mat_set_hint()`) already exists in `ggml/include/ggml.h` — no missing compile-time dependency, unlike the DSv4 port.

**Bar:** FWHT-path output matches GEMM-path output on a synthetic Hadamard-flagged tensor (bit-for-bit or within float tolerance); measured speedup on the flagged shapes.

**Constraints:** Ruling 9 — no allocation surface (grep for malloc/pool_alloc in the new file — no hits expected); reads src1/writes dst directly.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o MUL_MAT` with the Hadamard hint set on a synthetic tensor, lead-run.

**Depends on:** none.

### llama.cpp-pmzl

  - **Id:** llama.cpp-pmzl
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `31558dbb7657` — implement four DeepSeek-v4-specific ops (`LIGHTNING_INDEXER`, `DSV4_HC_COMB/POST/PRE`) as fork-native SYCL kernels; the previously-blocking merge (`llama.cpp-t9l9`) is closed, so this ticket's BLOCKED-ON-MERGE status is stale and should be cleared.

**Design:**
- New `ggml/src/ggml-sycl/{dsv4-hc,lightning-indexer}.{cpp,hpp}` — fork-native implementations (the fork's dispatch has diverged from upstream's single switch shape, so this is a reimplementation against the fork's structure, not a copy of upstream's diff).
- Wire into `ggml_sycl_compute_forward`'s op switch and `ggml_backend_sycl_device_supports_op`'s capability switch.
- Do **NOT** port the `docs/ops/SYCL.csv` churn this commit carries upstream (23,541 lines) — that file must be auto-regenerated by this epic's `csv-parity-gate` task, not hand-copied from upstream's version.
- The enum/CPU prerequisites (`GGML_OP_LIGHTNING_INDEXER`, `GGML_OP_DSV4_HC_*`) already landed with the b10630 merge — confirmed present in `ggml/include/ggml.h`.

**Bar:** No DSv4 model exists locally to correctness-gate against directly — the bar is `test-backend-ops` coverage for the four new ops plus a synthetic-shape numeric check against the CPU reference implementation the merge also landed.

**Constraints:** Ruling 9. Ownership screen already checked clean by the ticket writer (positive-control grep against `llama.cpp-g7hq`'s known-allocating diff returns 6 hits; this commit's diff returns 0) — confirm no scratch beyond work-group local memory is actually needed once the kernels are written.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o LIGHTNING_INDEXER` and `-o DSV4_HC_COMB/POST/PRE`, lead-run, once the port adds test coverage; cross-check against the CPU reference the merge landed.

**Depends on:** none (the `llama.cpp-t9l9` blocker is closed).

### llama.cpp-qz0f

  - **Id:** llama.cpp-qz0f
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `6c8dcaa7ae41` — widen the non-contiguous concat kernel's launch from a single-lane `(1,1,1)` work-group to `(1,1,SYCL_CONCAT_BLOCK_SIZE)`. Upstream measured +9.4% PP2048 on Arc Pro B70 — this fork's own hardware.

**Design:**
- `ggml/src/ggml-sycl/concat.cpp`, function `concat_T_sycl_non_cont` (`:130`) — confirmed still `sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1))`; widen per upstream.
- Pure launch-geometry change, no interaction with any rewritten subsystem.

**Bar:** Measured +9.4% PP2048 transfers to this fork's B70 build (or a comparable win — driver/hardware may shift the exact number).

**Constraints:** Ruling 9 — launch-geometry only, no allocation.

**Acceptance test:** Before/after `llama-bench -p 2048 -n 0` on B70, interleaved paired A/B (not required to land the fix, since it's narrow and low-risk, but required before claiming the perf win).

**Depends on:** none.

### llama.cpp-tfap

  - **Id:** llama.cpp-tfap
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `d415e65a5784` — add `concat_impl_{q4_0,q4_1,q5_0,q5_1,q8_0}_sycl` block-quantized concat paths, plus drop two spurious `.wait()`s after already-in-order device-to-device memcpys in the pre-existing float path.

**Design:**
- `ggml/src/ggml-sycl/concat.cpp`, `ggml_sycl_op_concat` dispatch switch — currently covers only F32/F16/BF16/I32/I16/I64/I8; add the 5 quant-type arms mirroring the existing `concat_impl_sycl<T>` template shape.
- Drop the two unnecessary `.wait()` calls in the float path while touching this file — this is directly aligned with Ruling 6 (no host waits; ordering via SYCL event dependencies), a rare case where the upstream diff itself removes a wait rather than adding one.

**Bar:** CONCAT correctness for all 5 new quant types; the removed `.wait()`s produce no ordering regression (same-queue in-order completion already guarantees correctness).

**Constraints:** Ruling 6 (removing waits is aligned with, not against, design) and Ruling 9 — no allocation surface, quantized paths use `stream->memcpy`/kernel writes into the pre-existing dst buffer.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o CONCAT` with each of the 5 new quant types, lead-run.

**Depends on:** none.

### llama.cpp-tjt9

  - **Id:** llama.cpp-tjt9
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `d3fba0c79db8` — add `get_rows` support for Q2_K, Q4_K, Q5_K, currently absent from the fork's `get_rows` dispatch entirely.

**Design:**
- `ggml/src/ggml-sycl/dequantize.hpp` — add `dequantize_{q2_K,q4_K,q5_K}_f32` helpers.
- `ggml/src/ggml-sycl/getrows.cpp` — add the three `case GGML_TYPE_*` arms to both dispatch switches (~lines 2054, 2829).
- Shared-quant-table category, same as the other type-coverage ports in this epic.

**Bar:** GET_ROWS correctness for Q2_K/Q4_K/Q5_K sources.

**Constraints:** Ruling 9 — pure per-element dequantize math reading existing weight-buffer bytes, no allocation.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o GET_ROWS` with the 3 new types, lead-run.

**Depends on:** none. (`llama.cpp-596y`'s SET_ROWS port may reuse these dequant helpers — not a hard dependency in either direction.)

### llama.cpp-tp03

  - **Id:** llama.cpp-tp03
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `22b208b1cacb` — implement the xIELU activation unary op, currently declared in `ggml.h` but with zero SYCL kernel anywhere in the backend.

**Design:**
- `ggml/src/ggml-sycl/element_wise.{cpp,hpp}` — new `ggml_sycl_op_xielu`/`ggml_sycl_xielu`, mirroring the existing `ggml_sycl_op_clamp`/`ggml_sycl_elu` pattern via `dispatch_ggml_sycl_op_unary`.
- `ggml-sycl.cpp` — add the `GGML_UNARY_OP_XIELU` case to `ggml_sycl_compute_forward`'s unary-op switch.
- Small, self-contained; no rewritten subsystem touched.

**Bar:** UNARY(XIELU) produces correct output for representative alpha/beta parameter values; no local model exercises xIELU today, so correctness is `test-backend-ops`-only.

**Constraints:** Ruling 9 — dispatch helper consumes already-resolved src/dst pointers from existing tensor buffers; no new allocation.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o XIELU` (`ggml_op_desc` returns the unary op's own name, e.g. `XIELU`, never `UNARY`, for `GGML_OP_UNARY` — `-o UNARY` matches zero cases), lead-run.

**Depends on:** none.

### llama.cpp-v7lu

  - **Id:** llama.cpp-v7lu
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Fix two genuine TQ2_0-refusal gaps in `supports_op` — NOT a literal port of `814d84bc9da5`, whose three named sites mostly don't apply to this fork's already-different dispatch shape (MUL_MAT and SET_ROWS already correctly refuse TQ2_0 via positive allowlists).

**Design:**
- `ggml_backend_sycl_device_supports_op` (begins `:102023`), `GGML_OP_CPY`'s same-type contiguous fast path — add a `TQ2_0` exclusion alongside the existing `BF16` one; currently a `TQ2_0`→`TQ2_0` contiguous copy is accepted by `supports_op` and then hits `cpy.cpp`'s dispatch chain's `else { GGML_ABORT("fatal error"); }` (no `TQ2_0`↔`TQ2_0` arm exists there). NOTE: the `:98985`/`:98745` coordinates in this ticket's original filing are ~3.4k lines stale and land inside the persistent-TG MoE graph probe, not `supports_op` — re-locate both sites live via `grep -n` before editing (CLAUDE.md's mandatory-grep rule for this file) rather than trusting either line number.
- The `GGML_OP_ADD_ID`/`GGML_OP_MUL_MAT_ID` unconditional-`true` block (sits BEFORE `ggml_sycl_mul_mat_type_supported()`'s allowlist) — add a `TQ2_0` check before its `return true`.
- Diagnostic-message improvement (naming the offending type in `GGML_ABORT`), matched per-file rather than assumed: `set_rows.cpp:983` matches upstream's pre-patch text verbatim, direct port. `mmvq.cpp` has been restructured (dispatches via `ggml_sycl_mmvq_dispatch`, `:21787`) — not a mechanical port; optionally improve its own default case (`:22150`) with the same "name the type" intent.

**Bar:** A TQ2_0 CPY or MUL_MAT_ID/ADD_ID test case that currently `GGML_ABORT`s cleanly refuses post-fix (moves to `not supported [SYCL0]`, matching the honest-decline pattern the rest of the backend already uses).

**Constraints:** Ruling 9 — refusal logic and diagnostic strings only, no allocation. This ticket is a case study in verifying each claim against the current fork rather than name-grepping upstream's checks — a fail-closed allowlist's absence of a type name means it's already refused, so a naive port would have "fixed" two sites that need nothing.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o CPY` and `-o MUL_MAT_ID`/`ADD_ID` with TQ2_0 shapes, lead-run — confirm clean refusal, not abort.

**Depends on:** none.

### llama.cpp-vvci

  - **Id:** llama.cpp-vvci
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `6b5c2efb4e2f` — consolidate the fused-GLU kernels (geglu/reglu/swiglu/geglu_erf/geglu_quick) behind one shared launcher and add a contiguous fast path when `o0 == n && o1 == n`. Upstream measured +14% f16 / +4% f32 on SWIGLU on Arc Pro B70 — this fork's own hardware.

**Design:**
- `ggml/src/ggml-sycl/element_wise.cpp` — the separate, non-consolidated `gated_op_fused_{geglu,reglu,swiglu,...}` kernels (currently `(o0, o1)` params, no contiguous-fast-path branch) get the shared launcher plus the identity-collapse fast path.
- Different, additive change from the sibling generic-unary-op fast path in the same file (a separate ticket, not in this epic's members).

**Bar:** SWIGLU throughput improves on B70; upstream's +14%/+4% figures were measured on plain upstream, not this fork — must be re-validated here via interleaved paired A/B before being cited as a landed win.

**Constraints:** Ruling 9 — kernel-consolidation and index-math fast path over existing src/dst buffers, no allocation (grep'd, no hits).

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops perf -o SWIGLU`, interleaved paired A/B before/after on B70, lead-run.

**Depends on:** none.

### llama.cpp-xoi2

  - **Id:** llama.cpp-xoi2
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Port `f275595dd16f` — fix two launch-geometry defects in `cpy.cpp`'s block-quantized launchers: 11 cross-type conversion launchers under-launch (single-lane-per-block), and 5 same-type passthrough launchers over-launch (treat element count as block count, up to 32x wasted threads). Upstream measured 20.21→158.19 GB/s (7.8x) on q4_0→f32 on an Arc Pro B70 — this fork's own hardware.

**Design:**
- Under-launch fix: 11 launchers (`ggml_cpy_{f32_q8_0,q8_0_f32,f32_q4_0,q4_0_f32,f32_q4_1,q4_1_f32,f32_q5_0,q5_0_f32,f32_q5_1,q5_1_f32,f32_iq4_nl}_sycl`, `f32_iq4_nl` at `:842`) — widen from `sycl::range<3>(1,1,1)` single-thread work-groups to `ceil_div(ne/QK, SYCL_CPY_BLOCK_SIZE)` work-groups of `SYCL_CPY_BLOCK_SIZE` threads, matching the file's own f16/f32 launchers.
- Over-launch fix: 5 same-type passthroughs (`ggml_cpy_q8_0_q8_0` `:1176`, `q5_0_q5_0` `:1203`, `q5_1_q5_1` `:1230`, `q4_0_q4_0` `:1258`, `q4_1_q4_1` `:1285`) — correct `num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE)` to `ceil_div(ne / QK*, SYCL_CPY_BLOCK_SIZE)`.
- Skip upstream's hunks for q2_0/mxfp4/K-quant/IQ-type variants that don't exist in this fork's `cpy.cpp`.
- The over-launch bug is not a memory-safety issue (the shared `cpy_q_q` kernel bounds-checks internally) — pure waste, not a correctness fix, but still needs a pre-change baseline per Ruling 9 since it changes grid dimensions.

**Bar:** q4_0↔f32 / q8_0↔f32 CPY throughput measurably improves (target: near upstream's 7.8x, allow this fork's noise floor); no OOB now that launch width increases (explicit check, not assumed from upstream's clean result).

**Constraints:** Ruling 9 — launch-geometry/grid-sizing change only, same src/dst buffers.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops -o CPY` correctness for every touched type pair, plus a throughput A/B on q4_0/q8_0↔f32, lead-run.

**Depends on:** none.

### llama.cpp-9uic

  - **Id:** llama.cpp-9uic
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Close the two `bin_bcast` capability gaps CUDA/CPU already handle: non-unit innermost strides on src0/src1/dst, and the type-combination chain's `GGML_ABORT` for anything outside 5 hardcoded combos.

**Design:**
- Stride gap: `ggml/src/ggml-sycl/binbcast.cpp` `k_bin_bcast`/`k_bin_bcast_unravel` currently assert away non-unit innermost strides (14 asserts at `:633-649`) plus an unasserted src0 contiguity assumption. Thread both strides through the two kernels (mirroring `binbcast.cu:85-90`'s `s00`/`s10` pattern) and add the missing multiply to the innermost loop of ADD/MUL.
- Then relax the decline in `ggml_sycl_binbcast_layout_supported` (function begins `:101923`, not `:~98583` — that coordinate is ~3.4k lines stale; re-locate live via `grep -n` before editing) — a single deletable seam per the prior `llama.cpp-83e4` quality review.
- Type-chain gap: `ggml_sycl_op_bin_bcast`'s dispatch chain ends in `GGML_ABORT` (`binbcast.cpp:960`) for anything outside its 5 covered type combinations, while ADD/SUB/MUL/DIV `supports_op` still admits every type — not currently reached by `test-backend-ops`' enumerated cases, but unguarded against future graphs. Left open deliberately by the prior ticket since a type decline could interact with the persistent-TG and fused-ADD fast paths (`ggml-sycl.cpp:92083`/`93511` at time of writing).
- Method template for either fix: reuse `llama.cpp-83e4`'s host-only pre-registration sim pattern (`binbcast-predicate-sim.cpp`).

**Bar:** Non-unit-stride bin_bcast shapes produce correct output instead of assert-failing; the type-chain gap is either closed (real coverage) or the `supports_op` allowlist is narrowed to match what the dispatch chain actually accepts (no silent `GGML_ABORT` reachable from an admitted type combo).

**Constraints:** Ruling 9 — the stride fix adds a multiply to the ADD/MUL hot loop, needs GPU throughput verification (not just correctness) since it's on a hot path.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-backend-ops -o ADD` / `-o MUL` with non-unit-stride shapes, lead-run; discriminating probe string for the type-chain gap: `ggml_sycl_op_bin_bcast: unsupported types:` should never fire from an admitted `supports_op` case.

**Depends on:** none.

### llama.cpp-9vm3

  - **Id:** llama.cpp-9vm3
  - **Action:** keep
  - **Priority:** 3
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Add the F16-src `SET_ROWS` instantiation CUDA and CPU already have (dst also F16); the fork's general dispatch path currently asserts F32-only, which `llama.cpp-9wjb` correctly made `supports_op` decline rather than let abort, leaving this as the honest remaining gap.

**Design:**
- `ggml/src/ggml-sycl/set_rows.cpp:1064` — relax the `GGML_ASSERT` from F32-only to the CUDA form: `src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16)`.
- Instantiate `set_rows_sycl<sycl::half, ...>` at `:1095/1097` when the F16/F16 condition holds — the template `set_rows_sycl<TIn, TIdx, TOut>` is already generic in `TIn`, so this is additive, not a rewrite.
- Re-widen the `supports_op` predicate `llama.cpp-9wjb` (`b949c9816`) narrowed, matching the new coverage.
- Consider a memcpy-equivalent fast path for F16→F16 rather than round-tripping through the `convert`-style half→float→half casts the general path uses — the persistent-TG SET_ROWS path already handles an F16 src this way (`ggml-sycl.cpp:93847`, `src_type = (src0->type == GGML_TYPE_F16) ? 1 : ...`), so precedent exists in-tree.

**Bar:** `test-backend-ops -o SET_ROWS -b SYCL0` moves the four `type_src=f16` cases from `not supported [SYCL0]` to `OK`: 196 not-supported → 192, 135 OK → 139, total 331, rc=0. This exact before/after count is the acceptance bar as stated by the ticket.

**Constraints:** Ruling 9 — no shipped model on this fork currently produces an F16-src SET_ROWS (would have aborted pre-`b949c9816`); this is coverage/parity, not a correctness regression fix. Worth doing before any model whose KV write feeds SET_ROWS an F16 source lands, since the CPU-fallback split would otherwise fragment the graph mid-KV-write.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops -o SET_ROWS -b SYCL0`, lead-run (memory hazard), confirm the exact count transition above.

**Depends on:** none.

### llama.cpp-j6qq

  - **Id:** llama.cpp-j6qq
  - **Action:** keep
  - **Priority:** 4
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Purely administrative — file the CUDA argmax all-`-inf` `-1`-sentinel defect (same class already fixed in this fork's SYCL argmax/TOP_K kernels) upstream to `ggml-org/llama.cpp`. No code change in this fork; CUDA is unused on Intel hardware here.

**Design:**
- File an upstream issue/PR referencing `ggml/src/ggml-cuda/argmax.cu:8-21` (seeds `maxval=-FLT_MAX, argmax=-1`, admits on `val > maxval`, so an all-`-inf` row never takes any column and emits `-1`).
- Cite this fork's own precedent fixes as the pattern upstream should apply: `llama.cpp-f7jd` (SYCL argmax) and `llama.cpp-irdj`/`a5008abf9` (SYCL TOP_K, occupancy-ranked admission).
- Note for the report: this fork's `n_finite` ARGMAX/TOP_K test-backend-ops cases would RED on CUDA as true positives if run there; upstream would want the equivalent kernel fix alongside adopting those cases. CUDA TOP_K should be checked for the same pattern while filing — not yet inspected.

**Bar:** Issue/PR filed to `ggml-org/llama.cpp`, referencing this fork's fix commits as precedent.

**Constraints:** None — no fork code touched.

**Acceptance test:** N/A — close this ticket once filed, or once the owner adjudicates it not-our-problem.

**Depends on:** none.

## New tasks

### SYCL op-coverage parity gate: regenerate docs/ops/SYCL.csv from the live dispatch table + standing CI check

  - **Title:** SYCL op-coverage parity gate: regenerate docs/ops/SYCL.csv from the live dispatch table + standing CI check
  - **Description:** ## Design addendum (triage 2026-09-01)

**Intent:** `docs/ops/SYCL.csv` is hand-maintained and currently wrong (e.g. it claims `TRI` and all 7 of `llama.cpp-8cou`'s lost-hunk ops are supported when they are not dispatched at all). This epic's bar requires the CSV be regenerated from the actual dispatch table, with a CI gate that fails if an op is ever claimed-but-unreachable again — the mechanism `llama.cpp-8cou`'s acceptance criteria #5/#6 already call for, promoted to its own deliverable since it should run AFTER the bulk of this epic's ports land, not as part of the first ticket alone.

**Design:**
- Write a script (Python, matching the fork's convention in `examples/sycl/update-ops-doc.sh` if reusable) that enumerates every `ggml_sycl_*` defined symbol in the built `libggml-sycl.so.0` (`nm -C --defined-only`) and cross-references it against the dispatched `case GGML_OP_*`/`case GGML_UNARY_OP_*` entries in both `ggml_sycl_compute_forward` and `ggml_backend_sycl_device_supports_op`.
- Regenerate `docs/ops/SYCL.csv` from that cross-reference, not by hand-editing rows.
- Add the script as a CI check (or a `test-sycl-*` gate matching this fork's existing pure-Python-gate convention, e.g. `test-sycl-env-report.cpp`'s pattern) that fails the build if a compiled-and-linked `ggml_sycl_*` entry point has no matching dispatch case, or if the CSV disagrees with the live dispatch table.
- Demonstrate the check fires on a synthetic negative (temporarily stub out one dispatch case) before trusting it — a positive control, per this epic's own emphasis on not shipping a check that would have been zero regardless.

**Bar:** `docs/ops/SYCL.csv` has zero rows claiming support for an op the dispatch table cannot reach; the parity script/CI check is demonstrated to fire on a synthetic missing-dispatch case.

**Constraints:** Ruling 9 — this is a documentation/tooling task, no runtime behavior change; must not itself become a place that silently passes (mirrors the CLAUDE.md warning about `GGML_ABORT` string greps and empty-probe traps — verify the check's positive control before trusting a zero result).

**Acceptance test:** Run the regeneration script, diff the new CSV against the current one, confirm every previously-false-positive row (TRI, the 8cou ops, and any others surfaced) is now accurate; run the CI check against a deliberately broken dispatch table and confirm it fails.

**Depends on:** `llama.cpp-8cou` (defines the parity-check method and acceptance criteria this task promotes to a standalone deliverable); should run after the bulk of this epic's ports (priority-2 items) have landed so the regenerated CSV reflects post-epic reality.
  - **Priority:** 2
### Depends on

  - llama.cpp-8cou

## Web findings

### Item 1

  - **Claim:** Upstream PR #25874 ('Extended SYCL oneDNN SDPA to non-FP16 KV caches', merged 2026-08-04) measured 2.37x-3.21x prefill speedup on Qwen 27B Q5_K_XL and Gemma 12B/E4B Q4_K_XL on Intel Arc GPUs when quantized KV (Q4_0-Q8_0) or F32 KV is dequantized to dense F16 before the oneDNN SDPA graph, gated to prefill K>=1024, batch>=32.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/25874
  - **Date:** 2026-08-04
  - **Impact:** Makes llama.cpp-g7hq's port materially higher-value than a generic coverage gap: this is a real, hardware-relevant throughput win on this fork's exact card family. Measured range is wider than the headline number, and asymmetric by card: B70 1.42x-2.51x, B50 2.21x-3.21x (the '2.37x-3.21x' figure some summaries quote is the optimistic slice, and the PR's own worked example is 'Gemma 4 E4B Q4_K_XL q8_0, 3.21x on B50', not the Qwen 27B/Gemma 12B pairing referenced in some earlier notes). g7hq's Bar should carry the B70 low end (1.42x) as the pass threshold so a ~1.5x result on the bigger card reads as a successful port, not a shortfall.

### Item 2

  - **Claim:** Upstream PR #26231 ('[SYCL] Support q2 mul_mat', merged 2026-07-31) was filed to fix a SYCL unit-test crash (SIGSEGV-class), not just a missing feature: test-backend-ops UT cases added for Q2_0 mul_mat crashed the SYCL backend before this landed, since GGML_TYPE_Q2_0 had zero dot-product/dequant path.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/26231
  - **Date:** 2026-07-31
  - **Impact:** Reframes llama.cpp-9ppc/llama.cpp-922d from pure coverage-gap to crash-avoidance: any future test-backend-ops sync that adds Q2_0 UT cases without this port landed first will crash test-backend-ops, which this fork treats as a memory-hazard binary already.

### Item 3

  - **Claim:** Upstream PR #26515 ('sycl: enhance OP set_rows to support all missed data types', merged 2026-08-07) was filed against issue #26462, a reported SIGSEGV on Intel processors for set_rows with an unsupported type; some added unit test cases could not be verified on CPU (no native support) but were shipped after review as 'merge ready'.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/26515
  - **Date:** 2026-08-07
  - **Impact:** Confirms llama.cpp-596y's type list (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ*) is the same set upstream shipped and tested; note some cases are only GPU-verifiable, matching this fork's own lead-only test-backend-ops constraint.

### Item 4

  - **Claim:** Upstream issue #21893 ('[SYCL] Intel Xe2 (Battlemage) B70: Weight corruption/nonsense output without GGML_SYCL_DISABLE_OPT=1', filed 2026-04-14, status bug-unconfirmed as of the fetch) reports garbled output on B70 with GGML_SYCL_F16=ON + GGML_SYCL_DEVICE_ARCH=bmg_g21 unless GGML_SYCL_DISABLE_OPT=1 is set; root cause suspected in Xe2 kernel reordering/DPAS synchronization; PR #21527 partially addressed a related silent-allocation-failure symptom but did not resolve the core issue; no fix has landed as of this research.
  - **Url:** https://github.com/ggml-org/llama.cpp/issues/21893
  - **Date:** 2026-04-14
  - **Impact:** Not caused by, or fixed by, any port in this epic (none touch Xe2 DPAS reorder/sync code), but is a live open risk on this fork's exact B70 hardware: if any B70 correctness gate in this epic shows unexplained corruption, check this issue and the GGML_SYCL_DISABLE_OPT=1 workaround before assuming the new port is at fault.

### Item 5

  - **Claim:** Intel Compute Runtime 26.09.37435.1 (Phoronix coverage) added Battlemage-specific fixes (PMT counter offsets, fence allocation/synchronization simplification, new Battlemage device IDs) and enabled counter-based allocation peer sharing for in-order command lists with multi-GPU event scenarios, as part of ongoing 2026 Level Zero feature work.
  - **Url:** https://www.phoronix.com/news/Intel-Compute-26.09.37435.1
  - **Date:** 2026-08 (approx, release-numbered 26.09)
  - **Impact:** Peer-sharing work is for counter-based/in-order multi-GPU event scenarios, not general P2P DMA between the B70 and B50 -- does not contradict this fork's durable 'no direct P2P between the two discrete cards' finding (different PCIe root ports, kernel-level refusal). No action needed for this epic; noted for awareness only, since the fork's compute-runtime is currently pinned to 26.31 per CLAUDE.md, not this 26.09 line.

### Item 6

  - **Claim:** The b10630 merge (2026-08-26, sha 1ebfa4e4a) already lands the core-ggml prerequisites two epic members were waiting on: ggml_ssm_scan()'s trailing int64_t K parameter (confirmed present in ggml.h, consumed by llama.cpp-8vwp) and ggml_rope_set_offset() (confirmed present at ggml.h:2075, consumed by llama.cpp-hkuh) -- both tickets' 'depends on: llama.cpp-90ns' edges point to a task id no longer in the open tracker, i.e. already satisfied and stale rather than a live blocker.
  - **Url:** https://github.com/ggml-org/llama.cpp (repo state, verified via local grep against ggml/include/ggml.h at HEAD fed0b58e2)
  - **Date:** 2026-08-26
  - **Impact:** Clears the stale dependency edges on llama.cpp-8vwp and llama.cpp-hkuh; both can be scheduled as standalone SYCL-only work with no remaining core-ggml blocker.