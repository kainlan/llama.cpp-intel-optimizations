---
type: entity
title: mem_handle
description: "ggml_sycl::mem_handle - the backend's ownership/lifetime token: lightweight, copyable, ref-counted; resolves to the current pointer on dereference."
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# mem_handle

`ggml_sycl::mem_handle` — the **ownership and lifetime token** of the SYCL
backend memory model. Lightweight, copyable, ref-counted; resolves to the
current pointer on dereference. Holding a handle guarantees the backing
allocation cannot be freed or evicted underneath you.

## Key facts

- Implements the core invariant: the [unified cache](/entities/unified-cache.md)
  owns **placement**, the handle owns **lifetime**; a raw pointer is only a
  transient view resolved for one immediate use, never stored.
- Resolves lazily and re-resolves on staleness: a global generation counter is
  bumped whenever a pointer could have moved (evict/promote/flush);
  `resolve()` is a ~3 ns compare-and-return on a hit, `resolve_slow()`
  re-queries the cache on a miss. This is what lets the cache migrate a
  weight VRAM↔host transparently.
- Holds a lease: while alive it increments the cache entry's `in_use_count`;
  eviction may only remove entries with `in_use_count == 0`. A non-evictable
  cache means a missing release to fix — never force eviction.
- Kinds: `WEIGHT` (keyed by `ggml_sycl_cache_id` tensor identity), `DIRECT`
  (raw-pointer wrapper for buffers the cache never moves),
  `ARENA_RUNTIME/SCRATCH/ONEDNN` (views into fixed VRAM zones), `CHUNK_LEASE`
  (pointer + lease on its arena chunk).
- Owner-first variant: `unified_allocate_owner()` allocates the intrusive
  `alloc_owner_control` *before* the physical allocation so an allocation can
  never exist without an owner; release is coordinator-mediated and can be
  refused (`RELEASED` / `RETRY_SCHEDULED`).
- Lives in `ggml/src/ggml-sycl/mem-handle.{hpp,cpp}`.

## Merged from

Consolidates former duplicate pages `memhandle`, `ggmlsyclmemhandle`.

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
- [unified_cache](/entities/unified-cache.md)
- [SYCL canonical memory architecture](/entities/sycl-canonical-memory-architecture.md)
