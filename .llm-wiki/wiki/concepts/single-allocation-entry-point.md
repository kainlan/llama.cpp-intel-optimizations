---
type: concept
title: Single allocation entry point
description: All runtime/scratch/staging/KV/compute allocation goes through unified_allocate, which routes by alloc_intent to a tier and VRAM zone; a raw sycl::malloc may occur only inside the cache implementation.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Single allocation entry point

All runtime/scratch/staging/KV/compute allocation goes through unified_allocate, which routes by alloc_intent to a tier and VRAM zone; a raw sycl::malloc may occur only inside the cache implementation.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
