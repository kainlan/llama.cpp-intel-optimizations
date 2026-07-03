# SYCL MoE Decode: Graph-Replay & Gate/Up→Down Fusion Roadmap

> **Status: ROADMAP, not an executable TDD plan.** The two tracks below are the
> two highest-upside structural levers for B50 GPT-OSS decode, but which one to
> build first — and whether track B is even worth it — is **gated on the
> diagnostic numbers from `llama.cpp-t6u6`** (decode kernel-coverage %) and the
> lead's achieved-GB/s measurement. Do not open implementation tasks from this
> document until the decision gate below is resolved with real numbers. This is
> the "spike the unknowns first" step: the spike is the measurement, not code.

**Companion campaign:** `docs/plans/2026-07-03-sycl-mxfp4-tg-down-gateup-cooptimization.md`
(the current proof campaign, epic `llama.cpp-j69a`) optimizes the two named
hot kernels in place. This roadmap covers the two structural moves that campaign
explicitly defers.

---

## Why this exists (the reframe)

Latest valid FA-on B50 GPT-OSS profile (commit `1c004a9a1`): `PP512 1211.90`,
`TG128 36.97 tok/s` = **27.05 ms/token**.

| Bucket | ms/token | Source |
|--------|----------|--------|
| `gateup.xmx_tiled_dpas_m2` | 5.55 | named profiler |
| `down.q8_soa` | 4.89 | named profiler |
| `fattn.compute.xmx_v2` | ~0.76 | named profiler |
| **Unattributed (norm, routing, elementwise, launch overhead, inter-kernel gaps, host sync)** | **~15.8** | 27.05 − above |

Target `45 tok/s` = `22.22 ms/token` → need **−4.83 ms/token**. The two named
hot kernels total 10.44 ms; a heroic 20% off both is only ~−2.1 ms → ~40 tok/s.
**45 is not reachable by kernel-local micro-optimization.** The reachable time is
disproportionately in the ~15.8 ms unattributed bucket. This roadmap attacks
that bucket (Track A: launch/gap via graph replay) and the byte traffic that
bounds the named kernels (Track B: fusion).

---

## Decision Gate (resolve BEFORE opening implementation tasks)

Run `llama.cpp-t6u6`'s extended parser on a lead FA-on decode profile with
`--wall-ms <decode wall from llama-bench>` and `--kernel-bytes` for the two hot
kernels. Then:

| Measurement | Reading | First track to build |
|-------------|---------|----------------------|
| `kernel_coverage_pct` well below ~100 (e.g. <80) | Decode wall is dominated by time OUTSIDE kernels — launch overhead, gaps, host sync | **Track A (graph replay)** — highest upside, model-agnostic |
| `kernel_coverage_pct` near 100 AND hot kernels at <60% of card peak GB/s | Kernels own the wall but are occupancy-bound | Occupancy/coalescing work on the kernels first; revisit A/B after |
| `kernel_coverage_pct` near 100 AND hot kernels at >85% of peak GB/s | Genuinely bandwidth-bound; only fewer bytes helps | **Track B (fusion)** — kill the intermediate VRAM round-trip |

The gate exists because Track A and Track B have very different cost/risk and
only the measurement says which is the real wall. Do not build both blind.

---

## Track A — Make MoE decode dispatch graph-replayable

**Hypothesis:** GPT-OSS decode re-records (or partially bypasses) the SYCL command
graph each token because the per-token expert pointer table is data-dependent on
routing, so the ~15.8 ms bucket carries per-submit launch overhead across ~24
MoE layers × several submits that graph replay would otherwise amortize (graph
replay is worth ~13% on Mistral per `CLAUDE.md`).

**Grounding already in-tree (verify during the spike, do not assume):**
- `ggml_sycl_trace_memcpy_during_recording` / `ggml_sycl_trace_op_during_recording`
  in `ggml/src/ggml-sycl/ggml-sycl.cpp` — instrumentation that exists precisely
  to catch memcpys/ops during graph recording (which invalidate replay).
- `docs/plans/2026-04-19-cache-expert-invariant-investigation.md` — prior
  investigation noting the gap is graph-related, not cache-miss-related.
- `docs/plans/2026-05-18-moe-layer-execution-architecture.md` — "Pointer tables
  are launch ABI data. Their source of truth is the selected [expert set]."
- Per `CLAUDE.md` SYCL Memory Ownership rules: pointer tables must derive from
  the stable identity/hash carried by `mem_handle`, and handles for queued/graph
  work must be retained for the lifetime of the executable graph. A
  graph-capturable table keyed by routing index (not raw device addresses) is
  consistent with that contract.

**Design sketch (to be firmed up by the spike):**
1. Confirm with `GGML_SYCL_KERNEL_PROFILE` + `t6u6` coverage whether decode is
   actually replaying. If a per-token memcpy or a re-record fires, `*_during_recording`
   tracers will show it.
2. If the expert pointer table is rebuilt per token: make it a **stable,
   graph-captured buffer** sized to `[n_layers][n_experts]` device pointers,
   populated once at weight residency time, and index it inside the kernel by the
   routing result (already a device buffer) instead of rebuilding a per-token
   `down_ptrs_device` / gate/up pointer array on the host and copying it in.
3. The routing-dependent part becomes an index load inside an otherwise-static
   captured graph, so a single `graph.replay()` covers the whole MoE layer.

**Risk / why it's non-trivial:** the current dispatch builds pointer tables on the
host per token; moving that to a captured, index-driven form touches the hot MoE
dispatch and the residency/handle lifetime rules. Correctness gate (GPT-OSS count
gate + Mistral gate) is mandatory after every step. Fail-closed behind an env
(e.g. `GGML_SYCL_MOE_DECODE_GRAPH=1`) until proven.

**Upside if hypothesis holds:** this is the only lever that can plausibly recover
multiple ms/token at once, and it helps every MoE model, not just GPT-OSS.

---

## Track B — Fuse gate/up → activation → down (kill the intermediate round-trip)

**Hypothesis:** the two hot kernels are separated by a full VRAM round-trip of the
expert intermediate. `gateup` writes the intermediate (per active expert,
`ff_exp` wide) to VRAM; `down` reads it back. At batch=1 that round-trip is pure
overhead — the intermediate never needs to leave the EU/SLM. Fusing gate/up,
the GLU activation, and down into one kernel eliminates that write+read entirely.

**This is the deferred "direct-final down" route** that campaign `j69a` Task 4
documents and guards (`activation/sycl-mxfp4-down-direct-final-proof.md`). Prior
direct-final and grouped variants produced correctness/perf regressions, so this
track is high-upside AND high-risk and must not be attempted blind.

**Design sketch (to be firmed up by a spike):**
1. One kernel per MoE layer: for each active expert, compute `gate·x` and `up·x`
   into registers/SLM, apply SwiGLU/SwiGLU-OAI, then immediately accumulate
   `down·(glu)` into the token's output — never materializing the intermediate to
   global memory.
2. Reuse the existing MXFP4 SoA weight layouts and q8 activation packing; the
   change is dataflow (fuse), not numerics.
3. New distinct profiler labels so the fused kernel is comparable against the
   split `gateup`+`down` baseline event times.

**Preconditions (hard):**
- A model-free numerical reference/parity harness for the fused path (prior
  regressions were correctness, not just perf). Build the parity test BEFORE the
  kernel.
- SLM/register budget check: `ff_exp × n_active_experts` intermediate must fit the
  chosen tiling without spilling, or the round-trip saving is eaten by spills.
- Fail-closed env; correctness gates mandatory.

**Upside if bandwidth-bound:** removes one full read+write of the expert
intermediate per token per layer — the biggest single byte-traffic reduction
available on this path.

---

## Adjacent quick checks (cheap, not tracks)

These are worth a lead measurement pass; each is small and may buy something or
rule itself out:

- **Is XMX/DPAS even right for batch=1 decode?** DPAS wants RHS width ≥8; at TG the
  RHS is one activation vector, so `xmx_tiled_dpas_m2` runs a matvec on a matmul
  engine. The Mistral TG path hits 81 tok/s on a pure MMVQ vector-dot kernel.
  A/B a MMVQ-style down/gateup decode kernel vs DPAS-M2. (The row-group variants
  in `j69a` Task 1 grope toward this; a dedicated vector kernel may win outright.)
- **Batch the top-k active experts as one k-wide GEMM.** The k active experts share
  the same activation vector; stacking their weights into one `[k·ff_exp × hidden]`
  call gives an RHS-width-k matmul that actually feeds DPAS. `GGML_SYCL_BATCH_EXPERTS`
  (default ON) batches launches, not GEMM shape — this is a different, additive idea.
- **B50 power/clock headroom.** B50 runs a 70 W cap. Measure actual GPU frequency
  during a TG run; if it sags below max clock, decode is power-throttled and a cap
  bump (board permitting) is a zero-code win. Won't reach 45 alone but is free signal.

Do NOT touch attention (0.76 ms/token) or pack/quant (~0.1 ms/token) — already
negligible.

---

## Constraints (same as `j69a`)

- Active checkout: `/Apps/llama.cpp-mxfp4-tg-runtime`, branch `feature/sycl-mxfp4-tg-runtime`.
- Workers must not run B50/B580/model gates, `/Storage/GenAI/models`, `llama-bench`,
  `sycl-kernel-bench`, VTune, `sycl-ls`, `/dev/dri`/DRM probes, `lsof`, P2P probes,
  or real harness execution. Lead owns all executable/hardware validation.
- Every runtime variant fails closed behind an explicit `GGML_SYCL_*` env; defaults
  unchanged until a promotion plan passes correctness + `PP512 >= 1150` + `TG128 >= 45`.
- Regression guardrails from `CLAUDE.md` hold: B50 GPT-OSS FA-on ≥1100 PP512 / ~50+
  TG128; B580 Mistral >2000 PP512 / >85 TG128; count gates correct.

---

## Next action

1. Land `llama.cpp-t6u6` (worker-safe parser diagnostic).
2. Lead runs a FA-on decode profile through the extended parser → record
   `kernel_coverage_pct` and hot-kernel achieved GB/s in the `j69a` validation
   record (`activation/sycl-mxfp4-tg-cooptimization-validation.md`).
3. Resolve the Decision Gate above → open a focused, TDD-structured
   implementation plan for the winning track (A or B) with a spike task first.
