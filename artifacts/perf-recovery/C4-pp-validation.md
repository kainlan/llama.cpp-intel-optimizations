# C4 — PP validation A/B: WOQ vs f16 vs baseline (llama.cpp-lvzb)

Date: 2026-08-22. HEAD: `79b55ceb` (C3 complete: WOQ batched executor, 2-D arm
default). Baseline: `/Apps/llama.cpp-baseline` worktree at detached `79ae63559`
(binary carries B1's env-gated inert probe; acceptable for throughput per
B1-decode-diagnosis.md). All runs `llama-bench -m /models/gpt-oss-20b-mxfp4.gguf
-p 512 -n 128 -r 2` through `scripts/bench-guard.sh` (cool-card preflight,
timeout -k, VALID/SUSPECT stamping). Host under permanent ambient load (owner
ruling 2026-08-07): absolute values are not quiet-host baselines; the A/B
structure is the evidence.

## Matrix (pp512 / tg128, bench-guard VALID unless noted)

| card | arm | env | pp512 | tg128 |
|------|-----|-----|------:|------:|
| B70 `level_zero:0` | baseline `79ae63559` | default | **1422.61 ± 57.08** | 49.47 ± 0.14 |
| B70 | HEAD policy WOQ | `F16_BATCHED=1` | **750.36 ± 5.94** | 39.73 ± 0.06 |
| B70 | HEAD policy f16 | `F16_BATCHED=1 WOQ=0` | 424.41 ± 0.38 | 40.48 ± 0.17 |
| B50 `level_zero:1` | baseline `79ae63559` | default | **906.42 ± 6.61** | 36.35 ± 0.08 |
| B50 | HEAD policy WOQ | `F16_BATCHED=1` | **426.80 ± 3.69** | 32.02 ± 0.07 |
| B50 | HEAD policy f16 | `F16_BATCHED=1 WOQ=0` | **error 40** (see below) | — |

Reference points from the same epic (B70 HEAD default, pre-C3, clean device):
394.84 ± 1.13 / 36.74 ± 0.26. B50 HEAD default (mixed-mode, post-D2):
274.65 ± 0.43 / 31.43 ± 0.03.

## Verdict against the pre-registered C4 thresholds

**B50 policy pp512 = 426.80 < 700 → track C does NOT close outright; the gap
analysis is mandatory** (epic §Task C4: ≥863 closes; 700–863 files gap
analysis; below 700 a fortiori). Correctness gates were green the same session
at HEAD: Mistral digit gate, GPT-OSS chat gate at default env AND with
`F16_BATCHED=1` (exact `1, 2, 3, 4, 5`), COMPARE clean on the shipped 2-D arm
(rel ≤ 0.026, zero NaNs, B70).

## What the split says

1. **WOQ is a large, real win over the f16 arm on both cards**: B70 750 vs 424
   (+77%), and on the B50 the f16 arm cannot even complete (below). C3's
   executor did its job.
2. **Both cards' WOQ arms sit at ~half their baselines** — B70 750/1422 = 53%,
   B50 427/906 = 47%. The residual PP gap is *systemic to the policy
   pipeline*, not card-specific.
3. **The GEMMs are not the gap.** `ONEDNN_VERBOSE=1` snapshot (B50, WOQ arm,
   `-p 512 -n 0 -r 1`, pp512 410.51): 3457 oneDNN GPU exec calls totalling
   385.9 ms across ~2 evals, of which 3219 are `f4_e2m1` WOQ matmuls at
   348.8 ms → ≈174 ms WOQ GEMM per eval against ≈1247 ms pp512 wall.
   **oneDNN ≈ 15% of PP wall; ~85% is outside oneDNN** — repack kernels,
   attention, the 2-D loop's per-expert submission cadence (~1610 prims/eval ≈
   0.1 ms each), and the rest of the graph. Gap analysis must event-time the
   non-oneDNN 85%, not tune the GEMMs.

## B50 f16 arm: broken escape hatch (filed llama.cpp-rtf1)

`WOQ=0` on the B50 dies with `error 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)` in
`FLASH_ATTN_EXT` (node 880/1374, dst handle_size 407633920) at
`ggml-sycl.cpp:74382`, then hangs until `timeout -k` kills it — two runs killed
(900 s and 1800 s budgets), kernel log clean both times. One throttled run
completed at 235.49 ± 50.31 (spread 21% on a card whose normal pp cv is 0.3% —
thermal-corrupt, not a measurement). Suspected tipping factor: D2's derived
headroom floor (652 MB → 1 GiB pipeline-on) + the f16 arm's ~506 MB down-slot
ring on a 16 GB card; pre-C3 the same env combo measured 274.65 clean under the
652 MB floor. Two defects tracked in `llama.cpp-rtf1`: the fail-closed planning
gap (admission accepted a plan whose runtime attention allocation cannot fit)
and the hang-after-throw (D1's second-order bug, now with a repro).

## Logs

Session scratchpad `c4/`: b70-{woq,f16,baseline}.log, b50-{woq,baseline}.log,
b50-f16.log (throttle-SUSPECT), b50-f16-rerun{,2}.log (timeout-killed, error-40
evidence), b50-woq-onednn-verbose.log. Scratchpad is not durable — all
load-bearing numbers are reproduced above and in the tracker comment.
