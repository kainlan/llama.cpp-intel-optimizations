---
type: concept
title: WOQ GEMM (weight-only-quantized)
description: Batched oneDNN matmul consuming 4-bit MXFP4 weights plus e8m0 scales directly (repacked into ~4x-smaller scratch), avoiding f16 dequant; runs woq_gemm_batch_mxfp4.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# WOQ GEMM (weight-only-quantized)

Batched oneDNN matmul that repacks MXFP4 weights straight into
`{nibbles, e8m0-scales}` scratch — **~4x smaller than f16 dequant**
(≈130 MB vs ≈506 MB for GPT-OSS 20B) — and runs
`woq_gemm_batch_mxfp4`, avoiding the f16 dequant step entirely. The f16
dequant arm is retained as a **SOA-only A/B fallback**
(`GGML_SYCL_MOE_PP_WOQ=0`).

## Key points

- The **3-D scale-mask variant is known-broken**: accepted by oneDNN (no
  refusal in the log) yet reproducibly produces wrong numerics on the down
  role (blk.0 gpu=−5.21 vs cpu=−484.47, NaN cascade by blk.1) while the 2-D
  loop on the same hardware is clean. Documented contradiction — do not
  re-enable in production.
- Hence the general rule here: **accepted is not the same as correct** (the
  same trap as `ext_oneapi_enable_peer_access`).

## Merged from

Consolidates former duplicate page `woq-weight-only-quantized-arm`.

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
- [GGML_SYCL_MOE_PP_WOQ](/entities/ggmlsyclmoeppwoq.md)
- [MXFP4](/entities/mxfp4.md)
- [Accepted is not the same as correct](/concepts/accepted-is-not-the-same-as-correct.md)
