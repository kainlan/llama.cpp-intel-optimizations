---
type: concept
title: WEIGHT handle leases
description: Weights are handed out as WEIGHT-kind handles keyed by tensor identity that hold a lease (in_use_count) on the cache entry; lazy generation-counter resolution permits transparent VRAM↔host migration, and eviction may only remove entries with in_use_count == 0.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# WEIGHT handle leases

Weights are handed out as WEIGHT-kind handles keyed by tensor identity that hold a lease (in_use_count) on the cache entry; lazy generation-counter resolution permits transparent VRAM↔host migration, and eviction may only remove entries with in_use_count == 0.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)

**Tracked by:** `llama.cpp-rg2ft`, `refcount-reclaim-no-zone-resets` (new epic, no tracker id yet) — see [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
