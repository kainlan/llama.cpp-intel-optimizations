---
type: concept
title: Layout follows residency
description: Wherever a weight lives, it is materialized in the optimal layout for the processor that executes on it (per device), and dispatch routes must consume the materialized layout, never the reverse.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Layout follows residency

Wherever a weight lives, it is materialized in the optimal layout for the processor that executes on it (per device), and dispatch routes must consume the materialized layout, never the reverse.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)

**Tracked by:** `llama.cpp-30ak7`, `llama.cpp-xihy`, `one-layout-honest-routes` (new epic, no tracker id yet) — see [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
