# PP attribution baselines, fixed instrument (llama.cpp-6405 final) — pre-1lon reference

Date: 2026-08-22. HEAD: `b50442bcb` (instrument complete: 091a4acc2 +
5ed0c5a7d fix cycle + b50442bcb polish; verify review PASS, task closed).
Both runs GPT-OSS 20B MXFP4, policy-WOQ arm
(`GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=1`, `GGML_SYCL_MOE_PP_WOQ` default ON),
`GGML_SYCL_MXFP4_PP_PROFILE=1 GGML_SYCL_DISABLE_GRAPH=1`, `llama-bench
-p 512 -n 0 -r 1 -v` via `scripts/bench-guard.sh`, **both stamped VALID**,
`read_failures=0` on every line. Figures below are the steady-state (second)
eval; the first eval carries first-touch overhead (B70 1877 ms vs 783; B50
1591 vs 1274). Host under permanent ambient load (owner ruling 2026-08-07).

Purpose: these are the **pre-fix baselines for llama.cpp-1lon** (batch the
per-expert-slot WOQ repacks into one launch per tensor-dispatch). Unlike the
alt6 capture (HEAD 091a4acc2), the WOQ GEMM bucket is now measured by the
instrument itself (F1 marker-bracket fix) instead of recovered out-of-band
from `ONEDNN_VERBOSE` — the post-1lon A/B can come from this instrument alone.

## Steady-state eval, verbatim summary lines

```
B70 (level_zero:0, profiled pp512 652.48):
[MXFP4-PP-BATCHED-PROFILE] device=0 dispatches=69 graph_total=783.376 ms accounted=384.477 ms other=398.899 ms tiled_repack=215.027 ms/1064 soa_repack=68.738 ms/532 woq_gemm=78.243 ms/462(2d=462,3d_requested=0) f16_gemm=0.000 ms/0 stage=22.468 ms/3192 graph_total_measured=1 read_failures=0

B50 (level_zero:1, profiled pp512 401.55):

[MXFP4-PP-BATCHED-PROFILE] device=0 dispatches=69 graph_total=1273.715 ms accounted=800.641 ms other=473.074 ms tiled_repack=453.487 ms/1064 soa_repack=152.087 ms/532 woq_gemm=153.860 ms/459(2d=459,3d_requested=0) f16_gemm=0.000 ms/0 stage=41.207 ms/3192 graph_total_measured=1 read_failures=0
```

(`device=0` on both is the post-selector in-process index, not the card —
key off the selector. B70 profiled pp512 = 652.48; B50 = 401.55.)

## Comparison against alt6 (broken-GEMM instrument @ 091a4acc2)

| component | B70 alt6 | B70 now | B50 alt6 | B50 now |
|---|---:|---:|---:|---:|
| tiled→WOQ repack | 214 | 215.0 | 457 | 453.5 |
| SOA→WOQ repack | 69 | 68.7 | 153 | 152.1 |
| **repack total** | **283** | **283.8** | **610** | **605.6** |
| WOQ GEMM | ~93 (ONEDNN_VERBOSE) | 78.2 (bracket) | ~174 (ONEDNN_VERBOSE) | 153.9 (bracket) |
| staging | 23 | 22.5 | 41 | 41.2 |
| graph_total (profiled) | — | 783.4 | — | 1273.7 |

Repack attribution reproduces within 1%; the alt6 finding stands unchanged.
GEMM bracket reads ~15% under the `ONEDNN_VERBOSE` sums — consistent with the
documented undercount direction (marker ordered by submission only; missing
slice lands in `other` — sycl-env-vars.md § reading a capture, `woq_gemm`
bullet) and/or `ONEDNN_VERBOSE` including non-batched-executor oneDNN calls.
Treat the bracket figure as a floor, the VERBOSE figure as a ceiling; both
say the GEMMs are ~10-15% of wall, not the gap.

## Per-tensor repack answer (unchanged from alt6)

138 `[MXFP4-PP-BATCHED-PROFILE-TENSOR]` lines per capture (72 distinct
tensors × 2 evals, minus non-dispatching entries): every tensor dispatches
**once** per eval; repacks run per active expert slot — mean **23.33
repacks_per_dispatch** (min 15, max 32 = full expert count) on both cards,
≈1600 repack launches/eval. This is llama.cpp-1lon's target: one launch per
tensor-dispatch ⇒ ~138 launches/eval (acceptance: repack ≤100 ms/eval
combined on B50, calls/eval ≈138).

## Ledger context (unprofiled, from C4/D3)

B70 baseline 1422.61 / policy-WOQ 750.36 pp512; B50 baseline 906.42 /
policy-WOQ 426.80. Raw profiled logs: session scratchpad
`profile-b{70,50}-b50442bcb.log` (not committed; summary lines above are the
load-bearing content).
