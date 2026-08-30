---
type: concept
title: Scratch ring admission
description: The MoE oneDNN scratch ring is sized for all n_expert experts and admission is fail-closed (since f2bdfbffe), so plan-time sizing mistakes (e.g. stale WOQ arm choice) make admission refuse every dispatch.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# Scratch ring admission

The MoE oneDNN scratch ring is sized for all n_expert experts and admission is fail-closed (since f2bdfbffe), so plan-time sizing mistakes (e.g. stale WOQ arm choice) make admission refuse every dispatch.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
