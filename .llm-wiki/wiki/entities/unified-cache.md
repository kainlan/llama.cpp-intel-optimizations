---
type: entity
title: unified_cache
description: ggml_sycl::unified_cache — the sole allocator and owner of SYCL backend memory (tiered weight cache, VRAM arena and zones, host pinned pool).
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# unified_cache

`ggml_sycl::unified_cache` — the **sole allocator and owner** of backend
memory. Every GPU, host-pinned, staging, scratch, graph-temporary, KV, oneDNN
and weight-layout allocation in the SYCL backend flows through it; no code in
the backend calls `sycl::malloc_device` / `sycl::malloc_host` / `sycl::free`
directly.

## Key facts

- Holds the tiered weight cache (device VRAM / pinned host / mmap, LRU
  eviction), the VRAM arena and its zones (KV, WEIGHT, ONEDNN, RUNTIME,
  SCRATCH), the host pinned pool, and all runtime/scratch/KV allocations.
- Enforces the VRAM budget `min(total*pct, free_at_init)`; ref-counted
  eviction — only entries with `in_use_count == 0` may be evicted.
- One allocation entry point: `unified_allocate(const alloc_request&)`
  returns a [mem_handle](/entities/mem-handle.md); the owner-first
  `unified_allocate_owner()` is the dominant shape in the backend (48 call
  sites at `0b7b49e07`). The old `unified_alloc`/`unified_free` pair backs the
  same machinery; `alloc_handle::as_mem_handle()` bridges them.
- The single sanctioned out-of-arena path, `ensure_cached_alloc()`, is
  **test-only** (all 34 call sites under `tests/`); a third production grep
  hit is the violation signature.
- Tiering is what makes placement moves possible: the cache can move an
  allocation between tiers (evict a weight to host, promote it back) at any
  time — which is exactly why raw VRAM pointers may not be stored anywhere.

## Implementation files

`ggml/src/ggml-sycl/unified-cache.{hpp,cpp}` (allocator) and
`mem-handle.{hpp,cpp}` (handle) — formerly tracked as a separate page.

## Merged from

Consolidates former duplicate pages `unifiedcache`, `ggmlsyclunifiedcache`,
`unified-cachehpp-unified-cachecpp-mem-handlehpp-mem-handlecpp`.

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
- [mem_handle](/entities/mem-handle.md)
- [SYCL canonical memory architecture](/entities/sycl-canonical-memory-architecture.md)
