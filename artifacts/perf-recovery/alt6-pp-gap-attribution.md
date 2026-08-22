# PP gap attribution — the repack is the gap (llama.cpp-alt6)

Date: 2026-08-22. HEAD: `091a4acc2` (adds the `GGML_SYCL_MXFP4_PP_PROFILE`
device-event instrument, llama.cpp-6405). All runs GPT-OSS 20B MXFP4,
`-p 512 -n 0 -r 1` via bench-guard, VALID stamps. Steady-state (second) eval
figures; the profiled runs carry ~13% instrument overhead (per-dispatch
drain), so component ms are device-event-true but `graph_total` slightly
exceeds unprofiled wall — shares below are quoted against unprofiled wall
(512 / pp512 tok/s).

## Per-eval attribution

| component | B70 ms | B70 share | B50 ms | B50 share |
|---|---:|---|---:|---|
| tiled→WOQ repack (gate/up) | 214 | ~31% | 457 | ~38% |
| SOA→WOQ repack (down) | 69 | ~10% | 153 | ~13% |
| WOQ GEMMs (ONEDNN_VERBOSE) | ~93 | ~14% | ~174 | ~15% |
| staging copies | 23 | ~3% | 41 | ~3% |
| other oneDNN (f16 jit:gemm) | ~8 | ~1% | n/m | — |
| unattributed (FA, norms, routing, gaps) | ~276 | ~40% | ~374 | ~31% |
| **unprofiled wall/eval** | **683** (750.4 tok/s) | | **1199** (426.8 tok/s) | |
| baseline wall/eval | 360 (1422.6) | | 565 (906.4) | |

Instrument notes: `woq_gemm` reads 0.000 ms — the merged
`ext_oneapi_submit_barrier` events carry no profiling info (instrument
limitation, fed back to llama.cpp-6405's review); GEMM time taken from
`ONEDNN_VERBOSE` instead (B70 185.0 ms/3219 calls per 2 evals; B50 348.8/3219,
C4 artifact). The 2d/3d split is a request label; all 462 GEMMs ran the 2-D
arm as shipped.

## The finding

**The policy route re-converts static weights every eval, and does it at ~5%
of memory bandwidth.**

- Repack totals: B70 283 ms/eval (~41% of wall), B50 610 ms/eval (~51%) — on
  the B50 the repack alone exceeds the baseline build's entire eval.
- The per-tensor lines show why it is slow: repacks are submitted **per
  expert slot** — mean 23.3 repack launches per tensor-dispatch (B50), 138
  tensor-dispatches/eval, ≈1600 kernel launches of ~0.2–0.4 ms each. Moving
  ≈12 GB/eval at an effective ~20 GB/s (B50) / ~44 GB/s (B70) on cards with
  ~450 GB/s: **launch/occupancy-bound, not bandwidth-bound**.
- Repacks re-run every eval because the WOQ form lives in per-dispatch
  scratch (Option T: stored layout stays XMX_TILED/SOA; WOQ is transient
  executor format). Within one eval each tensor dispatches once at p512, so
  there is no intra-eval reuse to exploit — the cost is the conversion
  itself.

## Fix path (filed llama.cpp-1lon)

Ruling-compliant (no layout duplicate, no dispatch-time layout shift — the
conversion already exists, it just runs badly): **batch the per-expert-slot
repack launches into one kernel per tensor-dispatch** (grid over active
expert slots), for both `repack_mxfp4_xmx_tiled_to_woq` and
`repack_mxfp4_soa_to_woq`. Target ≥300 GB/s effective → repack budget
~30–60 ms/eval. Projected recovery if achieved: B50 ~427 → ~800+ tok/s,
B70 ~750 → ~1200+ — within sight of the Track E flip thresholds (B50 863 /
B70 1351.5), with the unattributed bucket (~40%/~31% — FA kernels, norms,
routing) as the remaining ceiling, to be re-profiled after the repack fix
lands. A persistent cross-eval WOQ cache would also work but is a second
stored layout — forbidden by the one-layout-per-weight ruling absent an
explicit owner amendment; not needed if batching reaches bandwidth.

## Logs

Session scratchpad `alt6/`: b70-woq-ppprofile.log, b70-woq-onednn-verbose.log,
b50-woq-ppprofile.log (+ C4's b50-woq-onednn-verbose.log). Load-bearing
numbers reproduced above.
