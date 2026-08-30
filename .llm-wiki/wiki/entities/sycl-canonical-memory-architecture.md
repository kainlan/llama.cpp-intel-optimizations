---
type: entity
title: SYCL canonical memory architecture (design doc)
description: docs/design/sycl-canonical-memory-architecture.md — the authoritative, enforceable version of the unified-cache + mem_handle design.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# SYCL canonical memory architecture (design doc)

`docs/design/sycl-canonical-memory-architecture.md` — the **authoritative,
enforceable** version of the SYCL memory-ownership design (the narrative
onboarding is [sycl-memory-design.md](/sources/SRC-2026-08-29-002.md)).

## Key facts

- Carries the exact **allowlist of permitted allocation and pointer-resolution
  entry points**, and the migration inventory of not-yet-compliant sites.
- §1 states the formal contract: the
  [unified cache](/entities/unified-cache.md) owns placement, the
  [mem_handle](/entities/mem-handle.md) owns lifetime, a raw pointer is only a
  transient view.
- §3.1 specifies the `allocation_control_class` taxonomy
  (`CACHE_BACKING` / `CACHE_SUBALLOCATION` / `EXTERNAL_EXACT`) — the
  enforceable part of the owner-first contract, including why
  `CACHE_BACKING` cannot be requested through any public field and the two
  distinct mechanisms that can mint it.
- A region's bump pointer is never exposed as a stored raw pointer outside the
  immediate `get_scratch()` / `return_scratch()` pair.

## Merged from

Consolidates former duplicate pages `sycl-canonical-memory-architecturemd`,
`docsdesignsycl-canonical-memory-architecturemd` (same doc, two names).

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
- [unified_cache](/entities/unified-cache.md)
- [mem_handle](/entities/mem-handle.md)
