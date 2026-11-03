//
// The kinds a host allocation request is routed by: its role, its category, the host zone it lands in, and the
// lifetime and owner it is checked against (docs/design/sycl-mmap-host-tier.md §4.4, "Host-side non-weight requests:
// routed by role" and "Lifetimes").
//
// SYCL-free on purpose. The zone choice and the lifetime checks are pure functions over these kinds
// (host-zone-plan.hpp), and their host-only tests build without -fsycl, so the kinds cannot live in a header that
// pulls in SYCL. unified-cache.hpp and pinned-pool.hpp include this one.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

namespace ggml_sycl {

// Classification of runtime (non-weight) VRAM reservations for diagnostics.
enum class runtime_category : uint8_t {
    KV_CACHE     = 0,  // KV buffer allocations
    COMPUTE      = 1,  // Compute buffer pool + scratch
    STAGING      = 2,  // Expert staging, DMA staging, BLAS fallback
    GRAPH        = 3,  // Persistent TG temp allocs, get_rows_pool
    HOST_COMPUTE = 4,  // Host-pinned scratch for CPU offload layers
    EXPERT_CACHE = 5,  // MoE expert VRAM cache pool
    CONTROL      = 6,  // Small CPU-produced control tensors read by any GPU
    OTHER        = 7,  // Everything else (default for backward compat)
    COUNT        = 8
};

enum class alloc_role : uint8_t {
    WEIGHT         = 0,
    COMPUTE        = 1,
    KV             = 2,
    STAGING        = 3,
    GRAPH_TMP      = 4,
    TP_TMP         = 5,
    EXPERT_STAGING = 6,
    CONTROL        = 7,
    OTHER          = 8,
    // The lazy-table roles (llama.cpp-th32 T1), appended so no existing value moves.
    // unified_cache_role_bytes_for_testing reads the live bytes of each.
    LAZY_ROW_CACHE = 9,   // a lazy table's row-cache slab and index
    LAZY_STAGING   = 10,  // landing slabs, epoch-ring captures and the block ring
    LAZY_OUTPUT    = 11,  // the gathered rows a lazy GET_ROWS writes
};

// Host memory zone identifiers. Mirrors the VRAM arena zoning model.
enum class host_zone_id : uint8_t {
    WEIGHT  = 0,
    KV      = 1,
    STAGING = 2,
    SCRATCH = 3,
    COUNT   = 4,
};

// How long a host allocation's owner keeps it, and so where its leak check runs (§4.4 "Lifetimes"). There is no
// PROCESS value: a process-lifetime allocation has no releasing owner to check it.
//   GRAPH    the default. The graph-boundary check of the SCRATCH and STAGING zones counts only these records.
//   CONTEXT  owned by a ggml_backend_buffer (checked at that buffer's free_buffer), by a ggml_backend_sycl_context
//            member (checked after `delete sycl_ctx`), or by an execution context's tenant reservation.
//   MODEL    owned by a model's ModelId, checked at that model's teardown.
//   DEVICE   a device-scoped holder shared by every context on the device, checked when the last live backend on
//            that device is freed (§M62 (D)).
//   UNOWNED  the interim marker of the file-scope holders §4.4's site table leaves to their refactor. select_zone
//            sends it to WEIGHT, where those holders land today, and no boundary check counts it.
enum class alloc_lifetime : uint8_t {
    GRAPH   = 0,
    CONTEXT = 1,
    MODEL   = 2,
    DEVICE  = 3,
    UNOWNED = 4,
};

// Which owner a non-GRAPH record names. The kind keeps the id spaces apart: a buffer serial, a backend-context serial,
// a ModelId, a device index and an execution-context id can hold the same number.
enum class alloc_owner_kind : uint8_t {
    NONE              = 0,
    BUFFER            = 1,
    BACKEND           = 2,
    MODEL             = 3,
    DEVICE            = 4,
    EXECUTION_CONTEXT = 5,
};

struct alloc_owner_ref {
    alloc_owner_kind kind = alloc_owner_kind::NONE;
    uint64_t         id   = 0;
};

}  // namespace ggml_sycl
