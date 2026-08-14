#pragma once

// Private BUILD_TESTING-only failpoints for the transactional staging-replacement
// region of ggml-sycl.cpp introduced by 4f01993c9. Kept beside its sole test, in
// the same shape as q1-nvfp4-production-route-test-seam.hpp, and never installed
// with ggml-sycl.h.
//
// Every name here deliberately carries "failpoint". The ordinary-DSO check is
// `nm -D libggml-sycl.so | grep -ci failpoint` -> 0, and a probe whose pattern
// cannot match any symbol the build ever produces is worth nothing. The positive
// control is the same pattern against the private-fixture objects, where it must
// be non-zero.
//
// Arming is deterministic and count-limited rather than sticky: a seam fires at
// most `fires` times and then stops, so a test can arm exactly the number of
// traversals it reasons about and assert the remainder went unarmed.

// Relative on purpose: the two includers of this header (ggml-sycl.cpp and its
// fixture) do not share an include path that reaches the backend directory.
#include "../mem-handle.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#    define GGML_SYCL_STAGING_FAILPOINT_LOCAL __attribute__((visibility("hidden")))
#else
#    define GGML_SYCL_STAGING_FAILPOINT_LOCAL
#endif

enum ggml_sycl_staging_failpoint {
    // The device allocation itself succeeded; force the caller to treat the
    // replacement as unusable so the fresh owner must unwind before publication.
    GGML_SYCL_STAGING_FAILPOINT_ALLOCATION          = 0,
    // ext_oneapi_submit_barrier() throws inside the retirement transaction,
    // before any old owner has been handed to retention.
    GGML_SYCL_STAGING_FAILPOINT_BARRIER_SUBMIT      = 1,
    // retain_handles_until_event_transactional() throws after the barrier has
    // already been submitted -- the case the drain exists for.
    GGML_SYCL_STAGING_FAILPOINT_RETENTION_PUBLISH   = 2,
    // Force the grow decision even where capacity already suffices, i.e. resize
    // while prior secondary work may still reference the outgoing allocation.
    GGML_SYCL_STAGING_FAILPOINT_RESIZE_IN_FLIGHT    = 3,
    // Force the checked size arithmetic to take its rejection branch at a call
    // site whose operands cannot be driven to a boundary from outside.
    GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY = 4,
    // The host-side table views built in the same transaction as the device
    // allocation throw std::bad_alloc.
    GGML_SYCL_STAGING_FAILPOINT_HOST_VECTOR         = 5,
    GGML_SYCL_STAGING_FAILPOINT_COUNT               = 6,
};

// Arm `fp` for its next `fires` traversals. Arming is absolute, not additive.
GGML_SYCL_STAGING_FAILPOINT_LOCAL void ggml_sycl_staging_failpoint_arm(ggml_sycl_staging_failpoint fp, uint32_t fires);

// Disarm every seam and zero every counter. Call between scenarios.
GGML_SYCL_STAGING_FAILPOINT_LOCAL void ggml_sycl_staging_failpoint_reset();

// Times `fp` actually fired. A scenario that goes green with 0 here executed no
// armed path and is a no-op, not a pass.
GGML_SYCL_STAGING_FAILPOINT_LOCAL uint32_t ggml_sycl_staging_failpoint_reached(ggml_sycl_staging_failpoint fp);

// Times the seam site was traversed at all, armed or not. Lets an unarmed
// control prove the site is on the path under test before anything is armed.
GGML_SYCL_STAGING_FAILPOINT_LOCAL uint32_t ggml_sycl_staging_failpoint_traversals(ggml_sycl_staging_failpoint fp);

// Fires still owed on `fp`. Non-zero after a scenario means the scenario reached
// the seam fewer times than it claimed to.
GGML_SYCL_STAGING_FAILPOINT_LOCAL uint32_t ggml_sycl_staging_failpoint_armed_remaining(ggml_sycl_staging_failpoint fp);

// Rollback drains taken by the retirement transaction (secure() catch plus
// publication_failed()). This counts the staging region's own drains only, not
// every ggml_sycl_drain_direct_stage_queue() caller.
GGML_SYCL_STAGING_FAILPOINT_LOCAL uint32_t ggml_sycl_staging_failpoint_retirement_drains();

// ---------------------------------------------------------------------------
// Host-reachable drivers. Each one calls the production definition directly --
// they are accessors, not reimplementations, so a fixture cannot pass against a
// copy of the logic that has drifted from the shipped one.
// ---------------------------------------------------------------------------

GGML_SYCL_STAGING_FAILPOINT_LOCAL bool ggml_sycl_staging_failpoint_checked_add(size_t a, size_t b, size_t * out);
GGML_SYCL_STAGING_FAILPOINT_LOCAL bool ggml_sycl_staging_failpoint_checked_mul(size_t a, size_t b, size_t * out);
GGML_SYCL_STAGING_FAILPOINT_LOCAL bool ggml_sycl_staging_failpoint_checked_round_up(size_t   value,
                                                                                    size_t   alignment,
                                                                                    size_t * out);

// Runs the production same-allocation host-vector construction used by the MoE
// pointer-table publication. Returns false when it threw std::bad_alloc; on
// false neither out-vector is modified.
GGML_SYCL_STAGING_FAILPOINT_LOCAL bool ggml_sycl_staging_failpoint_build_table_views(
    size_t                               count,
    std::vector<ggml_sycl::mem_handle> * handles,
    std::vector<void *> *                payload);

// Drives one production ggml_sycl_old_owner_retirement transaction over
// caller-supplied escrow owners and returns whether secure() succeeded. A null
// queue is permitted only here: it means there is no submission to fence or
// drain, which lets the barrier/publication rollback be proved without a device.
// The transaction object is destroyed before returning, so on failure every
// escrow copy has already been dropped when the caller inspects its counters.
GGML_SYCL_STAGING_FAILPOINT_LOCAL bool ggml_sycl_staging_failpoint_run_retirement(
    sycl::queue *                              queue,
    const std::vector<ggml_sycl::mem_handle> & owners);
