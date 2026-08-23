# llama.cpp-0vqt coalesced+vectorized repack — full gate battery at `a89cfba79`

Date: 2026-08-22/23. Chain: `d6f41cd86` (SLM-tiled coalesced kernels, r1) →
`c8a8d174c` (r2, perf-neutral) → `4b31ba66f`/`205a5d691` (diagnostic forms) →
`56946e399` (write-phase vectorization, r3) → `a89cfba79` (executor wiring:
batched call sites swapped to the coalesced_batched forms; original kernels
retained as internal fallbacks). Baselines: `892eb5d7d` artifact (pre-1lon)
and `artifacts/perf-recovery/1lon-batched-repack-ab.md`. All runs GPT-OSS 20B
MXFP4 via `scripts/bench-guard.sh`, **every run VALID**, `read_failures=0`.

## Kernel-level (standalone microbench, device-event, mean GB/s)

| form | B50 orig → final | B70 orig → final |
|---|---:|---:|
| soa per-slot → coalesced per-slot | 26.96 → 76.98 | 67.10 → 194.92 |
| xmx per-slot → coalesced per-slot | 19.40 → 75.03 | 44.71 → 180.93 |
| soa batched → coalesced batched | 19.50 → 71.77 | 53.00 → 185.48 |
| xmx batched → coalesced batched | 16.63 → 63.71 | 39.09 → 165.47 |

~4x cumulative. Attribution chain (diagnostic forms, committed in the bench):
the limiter was **byte-granular destination stores** (write-only scalar 36/71
GB/s vs vectorized 205/345), NOT the strided reads (read-only 134/257). The
1lon batched-grid penalty vanished once stores were vectorized. B50's final
kernel runs at 94% of its SLM-roundtrip reference — the SLM-staged shape's
structural ceiling.

## In-graph gate battery at `a89cfba79` (policy-WOQ arm)

| metric (steady eval) | B70 pre-1lon | B70 now | B50 pre-1lon | B50 now |
|---|---:|---:|---:|---:|
| repack total ms | 283.8 | **83.2** ✅≤100 | 605.6 | **188.1** ❌≤100 |
| woq_gemm ms | 78.2 | 78.1 | 153.9 | 156.5 |
| stage ms | 22.5 | 19.4 | 41.2 | 33.8 |
| graph_total ms | 783.4 | 530.8 | 1273.7 | 804.7 |
| profiled pp512 | 652.48 | 962.19 | 401.55 | 635.25 |
| **unprofiled pp512** | **750.36** | **1118.01 ± 2.03 (+49%)** | **426.80** | **696.51 ± 5.37 (+63%)** |
| unprofiled tg128 | 39.73 | 39.84 | 32.02 | 32.25 |

Correctness: COMPARE 16/16 PASS (roles up/gate/down, B70, LIMIT=16); unit
tests 7/7 both binaries (RED-proven three times — 1lon slot^1, r1 slot-index,
r3 pack-loop byte-pair); Mistral digit gate PASS (interleave-scored); GPT-OSS
chat gate (`-c 4096`) PASS at default env AND batched arm. Default env is
untouched by the wiring (opt-in route policy only).

## Strategic position vs the E-flip thresholds (Amendment 1)

Flip thresholds imply eval budgets ≈379 ms (B70, pp≥1351.5) / ≈593 ms (B50,
≥863). Measured eval now: B70 458 ms (512/1118.01), B50 735 ms (512/696.51);
non-repack portion: B70 ≈375 ms, B50 ≈547 ms. Therefore the arm reaches its
flip threshold only if repack drops to **≤~4 ms on B70 and ≤~46 ms on B50**
— i.e. effectively *eliminated*, which the SLM-staged kernel shape cannot do
(ceiling measured) and oneDNN cannot absorb (option A dead, llama.cpp-4nlr
tripwire at `70c57f790`). Remaining levers, for the owner: (a) option C
custom PP GEMM consuming the stored layout (eliminates repack, vyjl c-ui9m,
effort L); (b) attack the arm's other 150+ ms (woq_gemm) — unexamined so
far; (c) revisit the flip thresholds/role of the WOQ arm. Note the arm now
stands at 79% (B70) / 77% (B50) of the default path's pp512 with tg parity.

Raw logs: session scratchpad `compare-b70-a89cfba79.log`,
`profile-b{70,50}-a89cfba79.log`, `pp512-b{70,50}-a89cfba79.log`,
`mistral-gate-a89cfba79.log`, `gptoss-{default,batched}-a89cfba79.log`.

## Verbatim steady-state profile lines

```
B70 (level_zero:0):
[MXFP4-PP-BATCHED-PROFILE] device=0 dispatches=69 graph_total=530.829 ms accounted=180.750 ms other=350.079 ms tiled_repack=59.310 ms/46 soa_repack=23.937 ms/23 woq_gemm=78.112 ms/462(2d=462,3d_requested=0) f16_gemm=0.000 ms/0 stage=19.392 ms/3192 graph_total_measured=1 read_failures=0

B50 (level_zero:1):
[MXFP4-PP-BATCHED-PROFILE] device=0 dispatches=69 graph_total=804.698 ms accounted=378.448 ms other=426.250 ms tiled_repack=131.207 ms/46 soa_repack=56.896 ms/23 woq_gemm=156.507 ms/459(2d=459,3d_requested=0) f16_gemm=0.000 ms/0 stage=33.838 ms/3192 graph_total_measured=1 read_failures=0
```

(`device=0` is the post-selector in-process index — key off the selector.)
