---
type: concept
title: Lazy re-resolution
description: A global generation counter bumped whenever a pointer could have moved makes resolve() a ~3 ns compare-and-return fast path, with resolve_slow() re-querying the cache on a miss so weights can migrate VRAM↔host transparently.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Lazy re-resolution

A global generation counter bumped whenever a pointer could have moved makes resolve() a ~3 ns compare-and-return fast path, with resolve_slow() re-querying the cache on a miss so weights can migrate VRAM↔host transparently.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
