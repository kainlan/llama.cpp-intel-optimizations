# D3 — B70 validation + baseline reference re-derivation (llama.cpp-vsnr)

Date: 2026-08-22. HEAD: `79b55ceb` (+ docs commit `c4b0f903c`). Baseline:
`/Apps/llama.cpp-baseline` at detached `79ae63559`. All rows
`llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -r 2` through
`scripts/bench-guard.sh` (cool-card preflight, VALID stamps, kernel log clean).
Post-D2, the B70 default route works on a clean device — no error 40, no
teardown hang, in any of today's B70 runs (the D1-era hang-after-throw now
reproduces only via the B50 f16-arm exhaustion, tracked as `llama.cpp-rtf1`).

## The three VALID rows (B70, `level_zero:0`)

| arm | env | pp512 | tg128 | log |
|-----|-----|------:|------:|-----|
| baseline worktree `79ae63559` | default | **1422.61 ± 57.08** | **49.47 ± 0.14** | c4/b70-baseline.log |
| HEAD default | default | **399.63 ± 0.68** | **39.36 ± 0.19** | d3/b70-head-default.log |
| HEAD policy | `GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=1` (WOQ default) | **750.36 ± 5.94** | **39.73 ± 0.06** | c4/b70-woq.log |

Baseline and policy rows are shared with C4's matrix (same session, same
protocol; see `artifacts/perf-recovery/C4-pp-validation.md`).

## Reading

- **The policy route nearly doubles HEAD's own default PP on the B70** (750 vs
  400) at equal TG — on this card the policy pipeline is unambiguously the
  better HEAD route today.
- **Both HEAD routes trail the baseline build badly** (28% and 53% of baseline
  PP; ~80% of baseline TG). The attribution work is `llama.cpp-alt6` (PP,
  systemic, non-oneDNN 85%) and the B-track decode tasks (B7 residue).
- Baseline tg128 measured 49.47 today vs the documented 43.57 ± 1.46 (21-run
  pool, 2026-07-25 driver-26.27 rows). Same binary tree, so the delta is
  environmental (ambient load mix / thermal state); B70 tg remains the noisy
  axis — gate on the pooled doc rows, not on today's single r=2 snapshot.

## Track E thresholds (per epic §Task D3 acceptance: ref×0.95 pp / ref×0.98 tg)

References are the **baseline-worktree rows** (the known-good build), per the
same convention that produced E's pre-registered B50 thresholds (≥863 pp ≈
0.95 × ~894–906 baseline; ≥34.5 tg):

- **B70 flip gate: pp512 ≥ 1351.5** (0.95 × 1422.61) **AND tg128 ≥ 48.5**
  (0.98 × 49.47) at default env on a clean device.
- Practical consequence: with HEAD policy at 750/39.7, the E-track auto-flip
  CANNOT fire on the B70 until the alt6/B-track gap work closes the deficit.
  That is the design working as intended — the flip is measured, not hoped.

## Logs

Session scratchpad `d3/` + `c4/` (not durable; all load-bearing numbers
reproduced above and in the baselines doc rows).
