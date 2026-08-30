---
type: concept
title: XMX_TILED grouped-DPAS route
description: Intel XMX matrix-extend grouped-DPAS kernel route for MXFP4 MoE gate/up/down projections, selected by planner/admission gates at layout time; productized for PP and enabled for the down projection.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# XMX_TILED grouped-DPAS route

Intel XMX matrix-extend **grouped-DPAS** kernel route for MXFP4 MoE
gate/up/down projections, selected by planner/admission gates at layout time.
Productized for PP, and (since `llama.cpp-sk67`) enabled for the **down**
projection where I8 does not already plan.

## Key points

- Admission is **fail-closed**: once an XMX_TILED-claimed op reaches the
  batched path, the admission contract must hold (see the fail-closed
  admission contract).
- Relies on the [XMX_TILED weight
  layout](/concepts/weight-memory-layouts-soa-aos-coalesced-xmxtiled.md);
  thresholds and gating via `GGML_SYCL_XMX_THRESHOLD` /
  `GGML_SYCL_USE_XMX_GEMM`.

## Merged from

Consolidates former duplicate page `xmx-grouped-dpas`.

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
- [Fail-closed admission contract](/concepts/fail-closed-admission-contract.md)
- [Weight memory layouts (SOA / AOS / COALESCED / XMX_TILED)](/concepts/weight-memory-layouts-soa-aos-coalesced-xmxtiled.md)
- [GGML_SYCL_XMX_THRESHOLD](/entities/ggmlsyclxmxthreshold.md)
