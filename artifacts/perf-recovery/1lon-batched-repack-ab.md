# llama.cpp-1lon batched-repack A/B — hypothesis FALSIFIED (perf regression, correctness clean)

Date: 2026-08-22. Post-fix HEAD: `c6ecc70aa` (batched WOQ repack kernels).
Pre-fix baseline: `artifacts/perf-recovery/6405-fixed-instrument-baselines.md`
@ `892eb5d7d`. All runs GPT-OSS 20B MXFP4, policy-WOQ arm
(`GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=1`), via `scripts/bench-guard.sh`,
**every run stamped VALID**, `read_failures=0` on every profile line. Host
under permanent ambient load (owner ruling 2026-08-07); profiled runs add
`GGML_SYCL_MXFP4_PP_PROFILE=1 GGML_SYCL_DISABLE_GRAPH=1`, `-p 512 -n 0 -r 1
-v`; unprofiled runs `-p 512 -n 128 -r 2`, no profile env.

## Verdict

The batching **engaged exactly as designed and did not recover any time**:
all 138 `[MXFP4-PP-BATCHED-PROFILE-TENSOR]` lines read
`repacks_per_dispatch=1.00` (was mean 23.33), repack launches collapsed
~1600 → 69/eval (46 tiled + 23 soa = one per tensor-dispatch) — and repack
device time went **up**, graph time went up, pp512 went down, on both cards:

| metric (steady-state eval) | B70 pre | B70 post | B50 pre | B50 post |
|---|---:|---:|---:|---:|
| tiled_repack ms | 215.0 | 243.0 | 453.5 | 552.1 |
| soa_repack ms | 68.7 | 85.8 | 152.1 | 198.1 |
| **repack total ms** | **283.8** | **328.7** | **605.6** | **750.2** |
| woq_gemm ms | 78.2 | 78.0 | 153.9 | 155.2 |
| stage ms | 22.5 | 19.2 | 41.2 | 34.2 |
| other ms | 398.9 | 377.1 | 473.1 | 437.1 |
| graph_total ms | 783.4 | 803.0 | 1273.7 | 1376.6 |
| profiled pp512 | 652.48 | 636.54 | 401.55 | 371.54 |
| **unprofiled pp512** | **750.36** | **708.97 ± 9.68** | **426.80** | **384.46 ± 1.74** |
| unprofiled tg128 | 39.73 | 39.73 ± 0.02 | 32.02 | 31.39 ± 0.22 |

Unprofiled pp512: **B70 −5.5%, B50 −9.9%** (B50 is the steady card, cv
~0.3% — decisively real). tg128 unchanged (TG does not traverse this path).
Acceptance criteria (3) repack ≤100 ms/eval and (4) pp512 improvement are
**FAILED**. The pre-registered "repack calls/eval ~138" expectation was
mis-derived from the per-capture (2-eval) tensor-line count; the per-eval
truth is 69 = `dispatches`, exactly one launch per tensor-dispatch, which is
what the acceptance meant.

## What this falsifies, and what survives

Falsified: the alt6/6405 attribution's *mechanism* hypothesis — that the
repack cost was per-launch overhead (~1600 launches × ~0.2–0.4 ms). The
launch count dropped 23x with zero time recovered, so the cost is **kernel
execution itself**. The batched grid (all active slots co-resident in one
launch, slot as the slower-varying nd_range dim) is ~15–24% *slower* than
the same kernels run slot-sequentially — consistent with interleaved
multi-expert memory streams degrading DRAM/cache behavior, though the
microarchitectural cause is unproven.

Survives: the attribution's *magnitude* — repack is still the dominant PP
sink (~40–55% of profiled graph time), it still moves ~12 GB/eval of static
weight bytes every eval, and the effective bandwidth is still pathological:
B50 12 GB/0.750 s ≈ **16 GB/s**, B70 ≈ **37 GB/s**, on ~450 GB/s-class
cards. A repack kernel running at even 200 GB/s would cost ~60 ms/eval and
meet the original ≤100 ms acceptance without any caching amendment — the
recovery lever is per-byte kernel efficiency (access pattern/coalescing/
vector width), not launch count.

Small real wins hiding in the numbers: `other` shrank ~22/36 ms (B70/B50)
and `stage` ~3/7 ms — the actual inter-launch overhead that batching
removed, an order of magnitude smaller than the attribution assumed.

## Correctness record (all VALID, all green)

- COMPARE (`GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_COMPARE=1`, LIMIT=16, B70):
  16/16 `status=PASS`, roles up/gate/down all covered, no `failed=` lines.
- Unit tests (GREEN at c6ecc70aa, B50, prior session): both
  `test-sycl-mxfp4-woq-repack` and `-tiled-` all-OK rc=0, including
  batched-vs-per-slot byte identity; RED sensitivity proven via injected
  slot^1 bug (task comment c-gcay).
- Mistral digit gate (B50, default env): `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`,
  rc=0.
- GPT-OSS chat gate (B50, `-c 4096`), BOTH default env AND
  `GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=1`: `1, 2, 3, 4, 5`, rc=0.

So `c6ecc70aa` is numerically correct and contained to the opt-in
policy-WOQ arm (default env unaffected), but is a pp512 regression within
that arm. Disposition (keep/iterate/revert) is recorded on llama.cpp-1lon.

Raw logs: session scratchpad `compare-b70-*.log`, `profile-b{70,50}-*.log`,
`pp512-b{70,50}-*.log`, `gptoss-gate*-*.log`, `mistral-gate-*.log` (not
committed; the summary lines above are the load-bearing content).

## Verbatim steady-state profile lines

```
B70 (level_zero:0):
[MXFP4-PP-BATCHED-PROFILE] device=0 dispatches=69 graph_total=802.998 ms accounted=425.944 ms other=377.054 ms tiled_repack=242.959 ms/46 soa_repack=85.784 ms/23 woq_gemm=77.996 ms/462(2d=462,3d_requested=0) f16_gemm=0.000 ms/0 stage=19.204 ms/3192 graph_total_measured=1 read_failures=0

B50 (level_zero:1):
[MXFP4-PP-BATCHED-PROFILE] device=0 dispatches=69 graph_total=1376.644 ms accounted=939.556 ms other=437.088 ms tiled_repack=552.107 ms/46 soa_repack=198.091 ms/23 woq_gemm=155.192 ms/459(2d=459,3d_requested=0) f16_gemm=0.000 ms/0 stage=34.166 ms/3192 graph_total_measured=1 read_failures=0
```

(`device=0` is the post-selector in-process index — key off the selector.)
