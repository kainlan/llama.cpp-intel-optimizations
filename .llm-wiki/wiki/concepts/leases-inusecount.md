---
type: concept
title: Leases (in_use_count)
description: A live WEIGHT handle increments the cache entry's in_use_count; in_use_count == 0 is necessary but not sufficient for eviction — ownership-aware reclaim (weight_entry_reclaimable()) also preserves live-model-owned and MID_LOAD_REPLAN entries.
created: 2026-08-29
updated: 2026-09-02
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
  - id: SRC-2026-09-02-001
    resource: /sources/SRC-2026-09-02-001.md
---

# Leases (in_use_count)

A live WEIGHT handle increments the cache entry's in_use_count; eviction may
only remove entries with `in_use_count == 0`, so a held lease means a
missing release to fix, never a forced eviction.

## ⚠️ `in_use_count == 0` is necessary, not sufficient

Do not infer reclaimability from `in_use_count` alone. Current weight
reclaim is ownership- and mode-aware and must go through
`weight_entry_reclaimable()`, which additionally preserves: entries owned by
a still-live model even when `in_use_count == 0` (several `llama_model`
objects can be loaded at once, and another live model's weights are correct
to keep, not leaked); and unattributed entries when the reclaim mode and
live-model mask require preservation. Only entries the predicate permits may
be reclaimed. Outside `MID_LOAD_REPLAN`, an attributed entry that is still
leased with no live model owner is reported as an ownerless leaked lease
(and may abort under `GGML_SYCL_STRICT_LEASES=1`); that ownerless
classification is suppressed during `MID_LOAD_REPLAN`. This nuance was
learned expensively: an earlier commit replaced preserve-and-continue with
`GGML_ABORT` on the assumption that any live handle at "cleanup" was a leak,
which broke `test-llama-archs` and `test-thread-safety` until reverted.

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md) · [SRC-2026-09-02-001](/sources/SRC-2026-09-02-001.md)
- [WEIGHT handle leases](/concepts/weight-handle-leases.md)
