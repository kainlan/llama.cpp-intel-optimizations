// Unified-cache residency request/plan and diagnostics types.
//
// These declarations intentionally stay small and ownership-focused so they can
// be shared by unified-cache code, MoE planning code, and synthetic tests
// without dragging MoE executor internals into the cache header.

#pragma once

#include "mem-handle.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

enum class residency_phase : uint8_t {
    UNKNOWN    = 0,
    MOE_DECODE = 1,
};

enum class residency_role : uint8_t {
    UNKNOWN       = 0,
    MOE_GATE      = 1,
    MOE_UP        = 2,
    MOE_DOWN      = 3,
    POINTER_TABLE = 4,
};

enum class residency_reject_reason : uint8_t {
    NONE                = 0,
    BUDGET              = 1,
    FRAGMENTATION       = 2,
    MISSING_LAYOUT      = 3,
    LIVE_LEASE_PRESSURE = 4,
    UNSUPPORTED         = 5,
};

const char * residency_reject_reason_name(residency_reject_reason reason);

struct residency_entry_request {
    const char *     tensor_name = "";
    ggml_layout_mode layout      = GGML_LAYOUT_AOS;
    size_t           bytes       = 0;
    residency_role   role        = residency_role::UNKNOWN;

    // Optional lease to retain when the reservation is accepted. Requests that
    // only preflight budget/fragmentation can leave this invalid; production
    // optimized paths should provide handles for pointer-table/descriptor users.
    mem_handle handle;

    // True when this entry represents an already-materialized resident layout
    // and the caller requires a valid handle/layout/device proof rather than
    // a budget-only reservation estimate.  This lets production preflight
    // reject stale/missing handles as MISSING_LAYOUT before any optimized
    // kernel sees a raw pointer table.
    bool require_handle = false;
};

struct residency_request {
    const char *                         debug_name = "";
    int                                  device     = -1;
    residency_phase                      phase      = residency_phase::UNKNOWN;
    std::vector<residency_entry_request> entries;
};

struct residency_budget {
    size_t bytes_available    = 0;
    size_t largest_free_block = 0;
};

struct residency_plan_entry {
    residency_entry_request request;
    mem_handle              handle;
};

struct residency_plan {
    bool                              accepted           = false;
    residency_reject_reason           reason             = residency_reject_reason::NONE;
    size_t                            bytes_requested    = 0;
    size_t                            bytes_reserved     = 0;
    size_t                            bytes_available    = 0;
    size_t                            largest_free_block = 0;
    std::vector<residency_plan_entry> entries;
};

struct residency_diagnostics_snapshot {
    uint64_t accept_count                         = 0;
    uint64_t reject_budget                        = 0;
    uint64_t reject_fragmentation                 = 0;
    uint64_t reject_missing_layout                = 0;
    uint64_t reject_live_lease_pressure           = 0;
    uint64_t reject_unsupported                   = 0;
    uint64_t stale_descriptor_rejects             = 0;
    uint64_t stale_descriptor_invalid_handle      = 0;
    uint64_t stale_descriptor_identity_mismatch   = 0;
    uint64_t stale_descriptor_generation_mismatch = 0;
    uint64_t stale_descriptor_layout_mismatch     = 0;
    uint64_t stale_descriptor_device_mismatch     = 0;
    uint64_t live_handle_count                    = 0;
    size_t   last_bytes_requested                 = 0;
    size_t   last_bytes_available                 = 0;
    size_t   last_largest_free_block              = 0;
    char     last_live_owner_tag[96]              = {};
    char     last_live_allocation_class[64]       = {};
};

residency_plan evaluate_residency_request_for_test(const residency_request & req, const residency_budget & budget);

void                           residency_diagnostics_reset_for_test();
void                           residency_diagnostics_record_accept_for_test(size_t bytes_requested,
                                                                            size_t bytes_available,
                                                                            size_t largest_free_block);
void                           residency_diagnostics_record_reject_for_test(residency_reject_reason reason,
                                                                            size_t                  bytes_requested,
                                                                            size_t                  bytes_available,
                                                                            size_t                  largest_free_block);
void                           residency_diagnostics_record_live_handle_for_test(const char * owner_tag,
                                                                                 const char * allocation_class,
                                                                                 size_t       bytes);
void                           residency_diagnostics_record_stale_descriptor_for_test();
void                           residency_diagnostics_record_stale_descriptor_invalid_handle_for_test();
void                           residency_diagnostics_record_stale_descriptor_identity_mismatch_for_test();
void                           residency_diagnostics_record_stale_descriptor_generation_mismatch_for_test();
void                           residency_diagnostics_record_stale_descriptor_layout_mismatch_for_test();
void                           residency_diagnostics_record_stale_descriptor_device_mismatch_for_test();
residency_diagnostics_snapshot residency_diagnostics_snapshot_for_test();

}  // namespace ggml_sycl
