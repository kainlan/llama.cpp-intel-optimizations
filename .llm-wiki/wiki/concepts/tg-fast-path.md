---
type: concept
title: TG fast-path
description: The ggml_sycl_mul_mat() shortcut for batch=1 quantized non-AoS MUL_MAT — it dispatches MMVQ and returns before any FORCE_DMMV or LAYOUT_OVERRIDE read, so MMVQ is the production TG kernel.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-001
    resource: /sources/SRC-2026-08-29-001.md
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# TG fast-path

The `ggml_sycl_mul_mat()` shortcut whose own comment says it "bypasses
orchestrator, name parsing, prefetch, TP checks for maximum speed." For
`src1->ne[1] == 1` (batch=1) with a quantized, GPU-accessible `src0` whose
resolved layout is not `GGML_LAYOUT_AOS`, it dispatches the MMVQ coalesced
kernel (e.g. `quantize_and_reorder_q8_1_soa` →
`ggml_sycl_op_mul_mat_vec_q`) and **returns** — before the
`GGML_SYCL_FORCE_DMMV` consultation sites and the `GGML_SYCL_LAYOUT_OVERRIDE`
kernel choice.

## Consequences

- **MMVQ is the production TG kernel** for reorder-eligible types (Q4_0 always
  qualifies); the Q4_0 coalesced DMMV kernel is a fallback/debug path.
- `GGML_SYCL_FORCE_DMMV=1` alone is a no-op at batch=1 — not ignored, never
  read. Pair it with `GGML_SYCL_TG_FAST=0` (the fast-path's documented
  disable) for the forcing to bind.
- A layout override binds on materialization, not on kernel selection — the
  `llama.cpp-szv8` COALESCED buffer was really built and byte-exact while the
  named DMMV kernel never ran.
- Prove which kernel actually ran from output, not setup:
  `GGML_SYCL_MUL_MAT_ROUTE_TRACE=1`.

## Merged from

Consolidates former duplicate pages `tg-fast-path-claims-batch1`,
`the-tg-fast-path-claims-batch1` (same mechanism, three names).

## Links

- [SRC-2026-08-29-001](/sources/SRC-2026-08-29-001.md) · [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
- [GGML_SYCL_TG_FAST](/entities/ggmlsycltgfast.md)
- [MMVQ/MMQ/DMMV/ESIMD kernel families](/concepts/mmvqmmqdmmvesimd-kernel-families.md)
- [Kernel confirmation from output, not setup](/concepts/kernel-confirmation-from-output-not-setup.md)
