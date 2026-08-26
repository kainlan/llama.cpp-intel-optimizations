# Upstream b10630 Merge Campaign Record (Phase C)

**Landed:** 2026-08-26, fork master `35a2c86eb..410433dd5` (pushed).
**Plan:** `docs/plans/2026-08-25-phase-c-upstream-merge.md` (25 tasks).
**Merge shape:** single `git merge` of upstream `b10630` (tip `d222767c7`,
702 commits past fork point `81ff7abe5`) with staged 5-wave resolution;
`ggml/src/ggml-sycl/` resolved keep-ours with whole-tree identity check;
upstream's 49 sycl commits audited into a port ledger
(`docs/merge/upstream-b10630-sycl-audit.md`, epic `llama.cpp-1yr6`).

## Certifications (all at the landed tip or its certified ancestors)

**Correctness (T22, `llama.cpp-phbr`).** Full `test-backend-ops` on the B70
scored against pre-merge master by ERR-line census (the case name is inline in
each ERR line, immune to the interleave attribution trap): failure population
identical except master's 3 bf16 MUL_MAT_ID wrong-answer cases, which the
merge refuses cleanly. Zero failures outside the c-wps7 enumerated classes.
Filtered ctest sweep triaged to a master-equal state; cache|mem-handle family
run serially (one shared pre-existing failure, identical on master);
skip-policy behavioral arm re-proven (no-model exits 77).

**Memory-design contract.** All enforcement gates pass on the merged tree:
`test-sycl-alloc-policy`, `test-sycl-handle-policy`,
`test-sycl-ensure-cached-alloc-policy`, `canonical-checksum-owner-scope`,
`canonical-checksum-source-contract`, `scripts/check-sycl-alloc-usage.sh`.
The merge introduced zero out-of-contract allocation, copy, or ownership
sites; the five raw memcpy sites it brought in were converted to the
sanctioned `mem_copy_ptr_async` primitive (`14e864c86`).

**Performance (T23, `llama.cpp-6r31`).** Four arms x five interleaved pairs,
bench-guard wrapped (`artifacts/merge-b10630/perf-final/` in the evidence
worktree): match-or-exceed on every arm. Mistral parity on both cards;
GPT-OSS parity after the nextn-crop fix below. Parser B50 floors OK; the
three B70 "band" failures are stale-band false alarms (two rows EXCEED their
bands by ~20%; the tg row is missed identically by the pre-merge binary
in-pair) — band refresh is owner-review task `llama.cpp-kzug`.

**Landing (T24, `llama.cpp-t9l9`).** ff-only into master; push; 51/51 fork
workflows `disabled_manually` (2 new upstream arrivals disabled post-push),
guard rc=0; fresh-clone `sycl-build.sh` rc=0 with `GGML_SYCL=ON`.

## Defects found and fixed during certification

- **Upstream perf bug — nextn last-layer crop** (`2e6dc4782`,
  `llama.cpp-qfae`): b10630's nextn/MTP support made the last-layer
  output-row crop conditional on `embeddings_nextn_masked` alone in
  `openai-moe`/`qwen35`/`qwen3next`/`gemma4`, so with defaults every prompt
  row ran the final layer's FFN/MoE for an unread hidden state (+1/n_layer
  MoE PP work; measured −3.5% pp512 on GPT-OSS, both cards, kernel-profiled
  to +4.3% staged rows at equal per-dispatch cost). Fixed with deepseek2's
  `!embeddings_nextn` guard pattern. **Candidate for an upstream PR.**
- **SSM_SCAN suite abort** (`916024d39`): fork kernel predates upstream's K
  rollback snapshots; supports_op now mirrors the kernel's shape identity
  (refusal, not abort).
- **ROPE `n_offs` / ROLL permuted-src wrong answers** (`da9769a05`): fork
  kernels ignore `op_params[15]` and assume packed src respectively; both
  fail closed until ported.
- **MMID type gate** (`186348705`): `MUL_MAT_ID`/`ADD_ID` admission was
  type-blind; upstream's q2_0 computed ERR≈90 wrong answers through the MoE
  executor. Admission now consults the MUL_MAT allowlist, with Q1_0/NVFP4
  deliberately exempt (the sanctioned runtime-oracle fail-closed class,
  population verified equal to master). Both source contracts re-locked with
  new mutation arms (unconditional re-admission and exemption widening
  rejected).
- **MMID aux-tensor workspace** (`fe94878a7`): only `.weight` names drive
  workspace geometry (nvfp4/fp8 `.scale`/`.input_scale` companions skipped).
- **Source-assert adaptations** (`3069cfb54`): ROPE inventory parser extended
  for the `rope_set_offset` block; backend count/get null-census updated for
  upstream's dedup/removals, with the fork null-guard added to the new
  `tools/tuning` site.
- **woq-repack TU compile fix** (`410433dd5`, `llama.cpp-1lrx`): defaulted
  `device` arg does not survive function-pointer conversion; both test TUs
  now build (pre-existing breakage, byte-identical on master).

## Tickets filed (open)

- `llama.cpp-zsyj` (P1): tiered-KV does not handle recurrent
  `cache_r_l`/`cache_s_l` tensors — SSM/hybrid state writes refused by the
  mem-ops range guard. Pre-existing (strict dominance: master aborts
  byte-identically on the same fixture); newly covered by upstream's
  rollback tests.
- `llama.cpp-a6sy` (P1): MoE-fixture route refusal class (arch sweep
  exclusions; re-triage vs zsyj once fixed).
- `llama.cpp-kzug` (P2): owner review — refresh stale B70 perf bands.
- `llama.cpp-1yr6` (epic): 27 port-candidate upstream sycl commits.

## Evidence

Worktree `/Apps/llama.cpp-merge-b10630` (KEPT until owner review):
`artifacts/merge-b10630/sweeps/` (backend-ops master/r3/r4, filtered),
`artifacts/merge-b10630/perf{,-eqwork,-final}/` (paired logs + manifests),
`artifacts/merge-b10630/gates/`. Fresh-clone proof at
`/Apps/llama.cpp-freshclone` (deletable after review).

## Perf history (2026-08-26, ambient load ~60, driver 26.27, landed tip)

| arm | pre-merge (fbdb31f43) | landed (410433dd5) |
|-----|----------------------:|-------------------:|
| B70 Mistral pp512 / tg128 | ~3018 / ~101.9 | 3018.33 / 101.34 |
| B70 GPT-OSS pp512 / tg128 | ~1717 / ~40.5 | 1712.42 / 40.22 |
| B50 Mistral pp512 / tg128 | ~1293 / ~46.7 | 1290.82 / 46.67 |
| B50 GPT-OSS pp512 / tg128 | ~857 / ~32.3 | 852.40 / 31.92 |

Interleaved in-pair deltas all within noise; absolute values under ambient
load are not baselines (standing rule).
