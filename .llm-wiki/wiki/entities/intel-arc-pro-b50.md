---
type: entity
title: Intel Arc Pro B50
description: Secondary benchmark card in this machine (Battlemage G21, PCI 0000:07:00.0, level_zero:1, ~16.2 GB VRAM); unchanged through the B580→B70 swap.
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

# Intel Arc Pro B50

Secondary benchmark card in this machine. Battlemage G21 discrete card at PCI
`0000:07:00.0`, selected as `level_zero:1`, ~16.2 GB VRAM (bench runs log
16250 MiB free; the only two committed real `-v` captures report 14677/14679
MiB free on a healthy card, so the documented figure is unresolved). Remained
installed and unchanged through the 2026-07-24 B580→B70 swap.

## Key facts

- Weights resident in B50 VRAM are materialized in the **B50's optimal kernel
  layout** — layout is a per-device answer (tile shapes / XMX generation
  differ from the B70); host-resident weights use the CPU's optimal format (AOS).
- Inherently steady benchmark card (tg cv ~0.7%, pp cv ~0.3%): Mistral 7B
  Q4_0 measures ~1188 PP512 / ~47 TG128, reproducing its historical values —
  the card/driver are delivering, which localises the GPT-OSS PP gap to the
  GPT-OSS/MoE path.
- GPT-OSS 20B MXFP4: 893.77 PP512 / 32.06 TG128 measured — ~19% below the
  retired ≥1100 guardrail (retired as a gate 2026-07-25; the residual gap is
  an open analysis, tracked under the driver/TG partial attribution).
- No direct P2P to the B70 (PCI topology restriction); host-bounce
  (`level_zero:0,1`) is the only multi-device transfer path.

## Merged from

Consolidates former duplicate pages `b50`, `intel-arc-b50`, `arc-pro-b50`,
`intel-arc-pro-b50` (minted independently by per-source synthesis).

## Links

- [SRC-2026-08-29-001](/sources/SRC-2026-08-29-001.md) · [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md) · [SRC-2026-08-29-003](/sources/SRC-2026-08-29-003.md) · [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
- [Intel Arc Pro B70](/entities/intel-arc-pro-b70.md) — peer card, no direct P2P
- [PCI/PCIe topology restriction](/concepts/pcipcie-topology-restriction.md)
