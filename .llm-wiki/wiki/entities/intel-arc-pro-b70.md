---
type: entity
title: Intel Arc Pro B70
description: Primary benchmark card in this machine (Battlemage G31, 256 CU, PCI 0000:03:00.0, level_zero:0, ~32.6 GB VRAM); replaced the B580 on 2026-07-24.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-001
    resource: /sources/SRC-2026-08-29-001.md
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
  - id: SRC-2026-08-29-003
    resource: /sources/SRC-2026-08-29-003.md
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# Intel Arc Pro B70

Primary benchmark card in this machine. Battlemage G31 discrete card (256 CU)
at PCI `0000:03:00.0`, selected as `level_zero:0`, ~32.6 GB VRAM (32602 MiB
free on clean runs). Replaced the former card (B580 per the perf-baseline and
env-var docs; "A580" per compute-runtime.md — see
[Intel Arc B580](/entities/intel-arc-b580.md) for the naming discrepancy) on
2026-07-24.

## Key facts

- Weights resident in B70 VRAM are materialized in the **B70's optimal kernel
  layout** — per-device, not global (different tile shapes / XMX generation
  from the B50).
- The noisier benchmark card (tg cv ~3.3% across 21 runs): treat single-run
  tg differences below ~10% as nothing. Best for orientation; Mistral is the
  steadier guardrail workload on this card.
- B70 runs require `GGML_SYCL_OP_TIMEOUT_MS=180000` — its cold prestage
  exceeds the 30 s default watchdog; that is not a hang.
- A B70 run reporting ~13.8 GB free is confounded by a co-resident tenant
  (e.g. ComfyUI once held 18.3 GiB) — discard, do not rationalise.
- No direct P2P to the B50 (PCI topology restriction; `can_access_peer` false
  both directions, D2D USM copy fails both directions with a misleading
  OUT_OF_DEVICE_MEMORY).

## Merged from

Consolidates former duplicate pages `b70`, `intel-arc-b70`, `arc-pro-b70`,
`intel-arc-pro-b70` (minted independently by per-source synthesis).

## Links

- [SRC-2026-08-29-001](/sources/SRC-2026-08-29-001.md) · [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md) · [SRC-2026-08-29-003](/sources/SRC-2026-08-29-003.md) · [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
- [Intel Arc Pro B50](/entities/intel-arc-pro-b50.md) — peer card, no direct P2P
- [PCI/PCIe topology restriction](/concepts/pcipcie-topology-restriction.md)
