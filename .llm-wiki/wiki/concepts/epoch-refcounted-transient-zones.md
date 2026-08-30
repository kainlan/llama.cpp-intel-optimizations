---
type: concept
title: Epoch-refcounted transient zones
description: A ring of equal-sized regions each carrying a live-handle counter (the epoch) that rewinds on zero, giving reset-only scratch/staging zones a real per-allocation release path without replacing the bump allocator or the reset.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Epoch-refcounted transient zones

A ring of equal-sized regions each carrying a live-handle counter (the epoch) that rewinds on zero, giving reset-only scratch/staging zones a real per-allocation release path without replacing the bump allocator or the reset.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
