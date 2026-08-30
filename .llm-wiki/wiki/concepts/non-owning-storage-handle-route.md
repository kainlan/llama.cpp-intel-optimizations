---
type: concept
title: Non-owning storage-handle route
description: Per-expert slices of the buffer's managed_handle that are owning ref-counted leases but not cache entries — published transactionally and consulted by the expert resolver before any cache key is formed.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Non-owning storage-handle route

The older answer for backend-buffer tensors that need no cache identity — and
what the MoE expert resolver consults **first**.
`ggml_sycl_publish_backend_aos_expert_handles()` slices the buffer's own
`managed_handle` per expert
(`ctx->managed_handle.slice(offset, expert_size)`).

## Mechanism

- Every slice is **validated before any is published** (pointer matches the
  expected address, on-device, correct layout, correct device,
  `has_stable_owner_identity()` true), and stored on the tensor's `extra`
  keyed by `moe_storage_handle_key(expert_id, layout)`.
- **Publication is transactional**: previous records are taken aside first and
  rolled back if any step throws — a partially installed prefix is never
  observable. Runs where an AoS upload becomes complete (`set_tensor`,
  `buffer_cpy_tensor`, MMID decode branch).
- `ggml_sycl_resolve_moe_expert_route()` tries
  `ggml_sycl_try_moe_storage_handle_route()` for each candidate layout
  **before forming any cache key**; a hit returns `FOUND` and the unified
  cache is never consulted.

## Properties

- The slices are **owning but not cache entries**: real ref-counted leases on
  the buffer's allocation (bytes can't go away under a kernel), and because
  the cache never learns of them, no eviction/free predicate can reach them.
  "Non-owning" means *the cache* does not own them, not that nobody does.
- **Layout discipline is structural**: the key includes the layout, so a
  request for SOA/XMX/packed **misses** rather than silently aliasing the AoS
  view.

## Merged from

Consolidates former duplicate page `non-owning-storage-handle`.

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
- [mem_handle](/entities/mem-handle.md)
- [Buffer-scoped weight provenance](/concepts/buffer-scoped-weight-provenance.md)
