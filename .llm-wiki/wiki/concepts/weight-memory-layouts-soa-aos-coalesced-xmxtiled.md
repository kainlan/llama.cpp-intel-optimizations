---
type: concept
title: Weight memory layouts (SOA / AOS / COALESCED / XMX_TILED)
description: Per-weight data layouts for quantized weights, chosen by the layout policy at planning time under the one-layout-per-weight rule; overridable per-weight via GGML_SYCL_LAYOUT_OVERRIDE.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# Weight memory layouts (SOA / AOS / COALESCED / XMX_TILED)

Per-weight data layouts (AOS, SOA, COALESCED, XMX_TILED) for quantized
weights, chosen at planning time by the layout policy under the
[one-layout-per-weight](/concepts/one-layout-per-weight.md) rule. The layout
choice determines which kernel arms (oneDNN, MMVQ, DMMV, DPAS) are reachable,
so it is load-bearing for both performance and correctness.

## Key points

- Overridable per-weight via `GGML_SYCL_LAYOUT_OVERRIDE` — but the override
  binds on **materialization, not on kernel selection** (the `szv8` trap: a
  COALESCED buffer really built and byte-exact, while the named kernel never
  ran).
- Layout follows residency: wherever a weight lives it is materialized in the
  optimal layout for the processor that executes on it (per device — the B50
  and B70 have different tile shapes / XMX generation; host-resident weights
  use the CPU's optimal format, AOS today).
- The route layout follows the materialized layout, never the reverse: a
  dispatch must consume what the cache actually materialized. The 2026-08-16
  MMID NaN family was routes advertising SOA over AOS bytes.

## Merged from

Consolidates former duplicate page `weight-layouts-aossoacoalescedxmxtiled`.

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
- [GGML_SYCL_LAYOUT_OVERRIDE](/entities/ggmlsycllayoutoverride.md)
- [One layout per weight](/concepts/one-layout-per-weight.md)
- [XMX_TILED grouped-DPAS route](/concepts/xmxtiled-grouped-dpas-route.md)
