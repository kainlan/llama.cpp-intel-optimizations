---
type: concept
title: Buffer-scoped weight provenance
description: A second legitimate owner-minting occasion at buffer allocation, seeded from managed_meta.id and namespaced by the top bit (1ull << 63), so buffer and model Registry id spaces are disjoint by enforcement.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Buffer-scoped weight provenance

A second legitimate minting occasion for weight ownership — at **buffer
allocation** (the first is model load via `Registry::begin_outer()` →
`ModelToken`). Needed because tensors that never went through a model load
have no owner and would be refused entirely.

## Mechanism

- **Minted at buffer allocation**: the buffer context's unique per-allocation
  id (`managed_meta.id`, from the cache's retention-identity counter) *seeds*
  a buffer owner. It cannot *be* the token — `alloc_metadata` carries no
  model-identity field at all.
- **Namespaced by the top bit**: buffer owner ids carry `1ull << 63`; the
  model `Registry` counts up from 1 and asserts the tag is clear, so the two
  id spaces are disjoint **by enforcement**, not assumption. Buffer owners
  take a sentinel slot outside the 32-slot model mask (unattributed class).
- **Consumed at `init_tensor`**: each tensor gets `extra->model_id` stamped
  and a row filed under `ggml_sycl_owner_name_key(buffer_owner, name)`, with
  a truthful synthetic parent identity (real byte offset, real `ggml_nbytes`).
- **Dropped at buffer free via stored cache ids**: the buffer destructor must
  not dereference tensors (`ggml_context` may already be gone — gallocr frees
  buffers first), so each tensor's `ggml_sycl_cache_id` is computed at
  registration and stored; teardown drops by stored key.

No new reclamation mechanism: these are lookup-table erasures.

## Merged from

Consolidates former duplicate page `buffer-scoped-provenance`.

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
- [tensor identity (ggml_sycl_cache_id)](/concepts/tensor-identity-ggmlsyclcacheid.md)
- [mem_handle](/entities/mem-handle.md)
- [Non-owning storage-handle route](/concepts/non-owning-storage-handle-route.md)
