//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

// Private cache-backing provenance.
//
// CACHE_BACKING is an ownership authority, not a hint: shutdown exempts that
// class from destructive-teardown refusal. It was previously minted from
// alloc_constraints.cache_backing, a public caller-writable bool, so any caller
// could forge it.
//
// There are exactly TWO ways to mint CACHE_BACKING, and they are enforced
// differently. Neither is reachable from a public alloc_request:
//
//   1. The token below, for the pinned pool's backing chunks. Enforced by the
//      compiler: the constructor is private to pinned_chunk_pool, so an
//      unauthorised caller cannot even name a valid argument.
//   2. A plain bool parameter on unified_cache_adopt_raw_host_allocation(), for
//      the cache's own staging buffer. That adopt runs during unified_cache
//      construction and cannot route through the coordinator without a circular
//      dependency, so it stays a bootstrap mint. Enforced instead by the helper
//      being TU-static in unified-cache.cpp — unreachable from any other
//      translation unit — plus a source gate pinning it to a single call site.
//
// Mechanism 2 is not weaker in reach, only in the kind of proof: staticness and
// the gate are checked by the build and the test, not by the type system. Do not
// describe this file as the only mint path.
//
// Do NOT include this header outside unified-cache.cpp and pinned-pool.cpp.
// tests/test-sycl-owner-allocation-migration.py enforces that restriction, the
// absence of cache_backing from the public request structs, and mechanism 2's
// staticness and single call site. Mechanism 1 is enforced by the compiler.

#include "unified-cache.hpp"

namespace ggml_sycl {

// Unforgeable proof that a standalone host USM base is physical backing owned
// by a cache/pool. Only the friend below can construct one, so possession is the
// authority; no request field can substitute for it. (The cache's own staging
// buffer does not use this token — see mechanism 2 above.)
class cache_backing_token {
    friend class pinned_chunk_pool;  // the sole minter of this token

    // User-provided, not defaulted: at C++17 aggregate initialization would
    // bypass a defaulted private ctor. tests/test-cache-backing-forgery-compile.py
    // fails if this is ever "cleaned up" back to `= default`.
    cache_backing_token() {}
};

// Backing counterpart of unified_allocate_owner(). Classifies the resulting
// owner CACHE_BACKING. The request must ask for a standalone host USM base
// (must_host_pinned + require_host_usm_base) — backing is never an interior
// suballocation — and is rejected as INVALID_REQUEST otherwise.
allocation_result unified_allocate_owner_backing(const alloc_request & req, cache_backing_token) noexcept;

}  // namespace ggml_sycl
