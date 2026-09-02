---
type: concept
title: Single-allocator invariant
description: All backend allocation flows through one allocator (the unified cache) and is owned by one lifetime token (mem_handle); raw allocation calls, raw pointer keying, and stored raw pointers are forbidden.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Single-allocator invariant

All backend allocation flows through one allocator (the unified cache) and is owned by one lifetime token (mem_handle); raw allocation calls, raw pointer keying, and stored raw pointers are forbidden.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)

**Tracked by:** `llama.cpp-mubmt`, `llama.cpp-rg2ft` — see [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
