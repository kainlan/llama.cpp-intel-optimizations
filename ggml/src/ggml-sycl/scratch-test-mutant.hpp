//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_SCRATCH_TEST_MUTANT_HPP
#define GGML_SYCL_SCRATCH_TEST_MUTANT_HPP

// The planted faults of the SCRATCH return's three device REDs (docs/design/sycl-mmap-host-tier.md §7 C4:
// test-sycl-scratch-trial-return, test-sycl-configure-refusal, test-sycl-scratch-owner-end). Each fault is a block at
// its site in the pinned pool, the unified cache or the backend, under `#if defined(GGML_SYCL_SCRATCH_MUTANTS)`, taken
// only when a device test has switched it on, with the production statements after it unchanged. They are runtime
// seams, never compile-time variants: the private fixtures are the one backend build those tests link (§M82 case (2)),
// and a variant would need a second one. The macro is defined only in a GGML_SYCL_PRIVATE_TESTING build, so the
// shipped libggml-sycl compiles none of the blocks, and a source gate reads the text with every block dropped
// (tests/sycl_gate.py resolve_mutants), as production compiles it.

#include <cstdint>

#if defined(GGML_SYCL_PRIVATE_TESTING)
#    define GGML_SYCL_SCRATCH_MUTANTS 1
#    include <atomic>
#endif

namespace ggml_sycl {

// One bit each in the selector; a mutant that carries two faults by construction sets two.
enum class scratch_test_mutant : int {
    // The pinned pool.
    DOUBLE_DECREMENT          = 0,   // release_zone_chunk also lowers the zone's size
    FOOTPRINT_BEFORE_FREE     = 1,   // release_zone_chunk subtracts the backing before the free
    CARRIER_DTOR_NOOP         = 2,   // a carrier's destructor does nothing
    MOVED_FROM_RELEASES       = 3,   // a moved-from carrier's destructor subtracts the backing again
    NO_PRUNE                  = 4,   // the reconcile never prunes a stranded entry
    PRUNE_WITHOUT_QUERY       = 5,   // the reconcile prunes without asking the registry
    SUBTRACT_AT_REFUSAL       = 6,   // a free the registry refused subtracts the backing at once
    DTOR_NO_REPORT            = 7,   // the pool's destructor does not report a stranded entry
    STRANDED_RESERVE_8        = 8,   // stranded_ is reserved for eight entries
    LEASE_CHECKED_FIRST       = 9,   // release_zone_chunk names a lease before a live allocation
    WITHDRAW_TRANSITION_BLIND = 10,  // a second withdrawal lowers the zone's size again
    CONFIGURE_REBUILDS        = 11,  // configure_zones rebuilds over a withdrawn chunk
    CARRIER_ASSERT_REMOVED    = 12,  // the pool's destructor does not assert on an outstanding carrier
    // The unified cache and the registry.
    REGISTERED_EXACT_PTR_ONLY = 13,  // unified_allocation_registered_exact compares the ptr alone
    RETURN_RELEASES_LIVE      = 14,  // a return tries to release every chunk it files, empty or not
    RETURN_NO_WITHDRAW        = 15,  // a return files its chunks without withdrawing them
    RETURN_NO_FILE            = 16,  // a return withdraws its chunks without filing them in the marked set
    RETURN_NO_IMMEDIATE       = 17,  // a return leaves an empty chunk for a begin to release
    RELEASE_NO_ERASE          = 18,  // a release leaves the chunk's marked-set entry behind
    CONFIGURE_DIRECT_RELEASE  = 19,  // configure_host_zones calls release_zone_chunk directly
    STRICT_ABORT_DROPPED      = 20,  // configure_host_zones returns false under GGML_SYCL_STRICT_LEASES=1, no abort
    ABORT_NON_STRICT          = 21,  // configure_host_zones aborts on a refusal without GGML_SYCL_STRICT_LEASES=1
    ORPHAN_FREE_IGNORED       = 22,  // the free that empties an orphaned chunk does not release it
    OWNER_END_NO_EMPTY        = 23,  // the owner's end leaves its empty chunks withdrawn
    OWNER_END_FIRST_ONLY      = 24,  // the owner's end frees only the first backing it collected
    LEASE_DROP_NO_INDEX       = 25,  // the last lease's drop on an orphaned chunk does not release it
    BEGIN_NAMES_ANY_OWNER     = 26,  // a begin names another owner's live chunk as refused
    // The SCRATCH transaction and its backend entry points.
    RETURN_NOOP               = 27,  // RETURN does nothing: the ensure keeps the high-water
    RTM_LEFT_IN_TXN           = 28,  // RETURN_TO_MARK files the tail but leaves it in the transaction
    KEEP_WALKS_SET            = 29,  // KEEP clears the withdrawn flag of every chunk in the device's marked set
    OWNER_END_REMOVED         = 30,  // ggml_backend_sycl_free makes no owner-end call
    COUNT                     = 31,
};

#if defined(GGML_SYCL_SCRATCH_MUTANTS)
// The process's selector. A test sets it before the arm it scores, in the child process that runs that arm.
inline std::atomic<uint64_t> g_scratch_test_mutants{ 0 };

inline void scratch_test_mutants_set(uint64_t bits) noexcept {
    g_scratch_test_mutants.store(bits, std::memory_order_release);
}

inline bool scratch_test_mutant_on(scratch_test_mutant m) noexcept {
    return ((g_scratch_test_mutants.load(std::memory_order_acquire) >> static_cast<int>(m)) & 1u) != 0;
}
#endif

}  // namespace ggml_sycl

#endif  // GGML_SYCL_SCRATCH_TEST_MUTANT_HPP
