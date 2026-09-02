---
type: entity
title: oneDNN
description: DNN library whose scratch allocations and calls flow through the unified cache; its on-device dequant (e.g. MXFP4 → f16) is a permitted format-conversion staging.
created: 2026-08-29
updated: 2026-09-02
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
  - id: SRC-2026-09-02-007
    resource: /sources/SRC-2026-09-02-007.md
---

# oneDNN

DNN library whose scratch allocations and calls flow through the unified cache; its on-device dequant (e.g. MXFP4 → f16) is a permitted format-conversion staging.

## Overview

[Key facts]

## Grouped-GEMM for MoE (2026-09-01 research, unverified against this fork's code)

oneDNN 2026.x documents `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY` grouped-GEMM
support intended for MoE token routing, plus claimed improved Xe2/Xe3 matmul
performance including float16-with-low-precision-weight paths
([uxlfoundation.github.io/oneDNN](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html)).
Flagged as directly actionable for `llama.cpp-0yi9` (evaluate oneDNN grouped
GEMM before hand-writing a new dense per-expert GEMM kernel for F16/F32/BF16
MMID) — this fork already delegates comparable work (SDPA, batched-PP-WOQ) to
oneDNN rather than a bespoke device kernel. Not yet evaluated; see the
[epic taxonomy's Open Questions](/syntheses/sycl-fork-epic-taxonomy-2026-09.md).

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
- [SRC-2026-09-02-007](/sources/SRC-2026-09-02-007.md) — `one-layout-honest-routes` epic web findings
- [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
