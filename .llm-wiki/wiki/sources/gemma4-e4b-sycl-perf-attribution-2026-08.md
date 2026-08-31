---
type: source
title: "gemma-4-E4B SYCL perf: measured attribution and D=512 route map"
status: insight
category: architecture
created: 2026-08-31
updated: 2026-08-31
slug: gemma4-e4b-sycl-perf-attribution-2026-08
---

# gemma-4-E4B SYCL perf: measured attribution and D=512 route map

Settled knowledge from the 2026-08-31 six-ticket gemma perf campaign (tracker: llama.cpp-0oad and children; all claims hardware-verified on the B50/B70, build lineage 4432f195d..f2b9ca638).

## The model
`stock-gemma-4-E4B-it.Q8_0.gguf` is **LLM_ARCH_GEMMA4** (fork arch, `src/models/gemma4.cpp`) — NOT gemma-3n: no AltUp, no Laurel, zero such ops in the traced graph. It borrows gemma-3n's per-layer-embedding tensor naming. 42 layers; interleaved attention with D=512 globals (every 6th layer, 7 total) and D=256 SWA; the last 18 layers share KV (`has_kv(il) = il < 24`); `f_attention_scale = 1.0` (pre-scaled Q — long-standing upstream convention).

## Measured baseline (Q8_0, warm, load ~50)
B70: pp512 666.5, tg128 28.2. B50: pp512 393.3, tg128 13.8. Both ~3.5-3.8x below Mistral 7B Q4_0 on the same cards — the deficit is architectural mapping, not card-specific.

## Attribution (what does NOT explain it)
- KV boundary copies: NEGLIGIBLE (~0.5MB/token). The measured 8MB×140 crossings were one-time context-init reserve probes; upstream K/V views are already n_kv-bounded (n_pad=1 → floor 256; decode fill n≈20). Premise disproven, llama.cpp-0375.
- MUL_MAT call count: NORMAL (344/token ≈ 8.1/layer). The "1003/token" was an OP_TIMING 5-graph fold artifact (instrument since fixed to disclose multiplicity).
- Host submission overhead: secondary (resolve 5.8µs/op, 4.8% of host time).
- Zero-alloc warnings: were a stale-baseline instrument artifact affecting all models (fixed; per-device + per-phase baselines now).

## What DOES explain it
- The 7 D=512 global layers decline SYCL FA (dispatch D set {64,128,256}) → CPU landing → 221 graph splits/graph (104 CPU) → split-boundary sync latency + CPU attention compute, plus graph-replay futility (the fragmented regime disables replay — correctly, per the dkw0 defect-#4 fix).
- Ordinary GPU compute on the Q8_0 shape mix (MUL_MAT ~80% of device time in decode).

## D=512 route map (the critical path)
- oneDNN SDPA is WIRED for D=512 (llama.cpp-jahv, kill switch GGML_SYCL_FA_ONEDNN_D512) but **scale-blocked for gemma**: oneDNN's compiled partition bakes 1/sqrt(D); gemma's kq_scale=1.0 is SCALE_UNSUPPORTED-rejected (this is also why gemma's D=256 layers never used oneDNN). Decode additionally blocked by MIN_NCOLS=8 vs ne01=1. Upstream CUDA has the same D=512 kernel gap (FA≤256; cuDNN SDPA is their escape).
- The gemma-viable route is the TILE path (llama.cpp-dtpk, P1): fattn-tile.hpp already carries D=512 config rows, unwired; the fork's own kernels take runtime scale and handle gemma's D=256 correctly today.
- Scale generalization for oneDNN: llama.cpp-p0f5 (riskier — touches Mistral/GPT-OSS production path).
- PLE: Google designed per-layer embeddings for CPU/host placement; the fork VRAM-places the 2.99GB tensor (llama.cpp-kmeq holds the host-placement experiment).
- Per-layer-embd gate+proj (84 calls/token, 24% of MUL_MATs) is NOT batchable — true sequential dependency (each layer's correction consumes that layer's own output).

## Ops facts worth keeping
vLLM's ~5000 pp on a B580 comes from all-GPU attention (Sycl-TLA/Triton kernels), bf16 GEMM-saturating prefill, chunked prefill, PagedAttention — engine-level differences beyond the gemma-mapping gap. The `<unused32>` knife-edge token at ngl99 matches upstream issue #12433 (model quirk, not fork corruption; llama.cpp-wy31).

*Category: architecture*

---
*Captured: 2026-08-31*

## Related

_Add links to related pages._
