---
type: concept
title: Leases (in_use_count)
description: A live WEIGHT handle increments the cache entry's in_use_count; eviction may only remove entries with in_use_count == 0, so a held lease means a missing release to fix, never a forced eviction.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Leases (in_use_count)

A live WEIGHT handle increments the cache entry's in_use_count; eviction may only remove entries with in_use_count == 0, so a held lease means a missing release to fix, never a forced eviction.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
