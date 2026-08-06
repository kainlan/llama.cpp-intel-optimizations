//
// Exact MoE pointer-table descriptor planning and CONTROL zone sizing.
//
// The RUNTIME zone used to budget MoE expert pointer tables as
// MAX_EXPERTS (256) * n_moe_layers * sizeof(void *) -- a count that is
// independent of the model's actual expert count and of how many expert
// tensors each layer really carries. On a model with 32 experts that is an 8x
// over-count per table, and the layer term is derived from the largest layer id
// seen rather than from the descriptors themselves, so a model with gate/up
// fused into one tensor is charged for two.
//
// The sizing here is EXACT: it enumerates the expert tensor descriptors, gives
// each one table slot per expert, and lays the routing ids, compaction table
// and missing-expert counter out after them with checked arithmetic. Every
// step that can overflow reports `arithmetic_overflow` rather than wrapping,
// and every step that can disagree with the descriptors reports
// `metadata_mismatch`; a layout that is not `valid` yields a
// `moe_control_requirement` that is not `valid`, which the zone arithmetic
// refuses to add. Sizing failures therefore fail CLOSED -- they never silently
// become a smaller number.
//
// This header is deliberately dependency-free, for the same reason
// zone-sizing.hpp is: the logic is pure, so it can be unit-tested on a host
// with no GPU, no SYCL runtime and no backend library (see
// tests/test-moe-control-plan.cpp). Callers adapt their own inventory type
// into moe_control_tensor_desc at the call site, and unified-cache.cpp
// static_asserts that its expert_tensor_role agrees with moe_expert_role here.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Mirrors unified-cache.hpp's expert_tensor_role value-for-value. It is
// duplicated rather than included because including that header would drag in
// SYCL and the whole backend, which is exactly what this unit must not depend
// on. unified-cache.cpp static_asserts the two agree, so a divergence is a
// compile error at the adapter rather than a silent misclassification here.
enum class moe_expert_role : uint8_t {
    UNKNOWN    = 0,
    GATE       = 1,
    UP         = 2,
    DOWN       = 3,
    GATE_UP    = 4,  // fused gate+up (ffn_gate_up_exps); has no partner entry
    CHUNK_GATE = 5,  // grovemoe ffn_gate_chexps
    CHUNK_UP   = 6,  // grovemoe ffn_up_chexps
    CHUNK_DOWN = 7,  // grovemoe ffn_down_chexps
    COUNT      = 8,
};

// What an "_exps"-bearing name turned out to be. MALFORMED is a distinct
// outcome from IGNORED on purpose: a name that advertises itself as an expert
// tensor but does not parse is a descriptor this planner cannot size, and
// sizing it as zero would under-budget the zone. It is counted and it makes
// the plan structurally incomplete.
enum class moe_expert_candidate : uint8_t {
    IGNORED,    // not an expert matrix; nothing to size (norms, biases, scales)
    MATRIX,     // a per-expert weight matrix; gets one pointer table
    MALFORMED,  // claims to be an expert tensor but does not parse
};

struct moe_expert_identity {
    moe_expert_candidate candidate = moe_expert_candidate::IGNORED;
    moe_expert_role      role      = moe_expert_role::UNKNOWN;
    int                  layer_id  = -1;
};

// Strict, anchored parse. Unlike the substring classifier in unified-cache.hpp
// (which answers "what role is this, roughly" for placement), this one also
// answers "is this name one I am willing to size", and it says no to anything
// that is not exactly blk.<digits>.<known suffix>. The distinction matters
// because an unrecognised expert tensor must widen the budget or block the
// plan, never be dropped.
moe_expert_identity moe_parse_expert_identity(const char * name);

// One bit per LOGICAL role a layer needs before its experts can be routed.
// GATE_UP sets both the gate and up bits because the fused tensor supplies
// both; that is why a completeness check cannot just count entries.
uint8_t moe_expert_logical_role_mask(moe_expert_role role);

// True when a layer's accumulated mask covers gate, up and down -- in either
// the split (GATE|UP|DOWN) or fused (GATE_UP|DOWN) spelling. A layer missing
// one of them cannot be dispatched, so a plan containing it is not a plan.
bool moe_logical_role_mask_complete(uint8_t mask);

// Minimal descriptor: deliberately NOT placement_tensor_info, so this header
// stays free of unified-cache.hpp. `type` mirrors ggml_type as a plain int for
// the same reason zone_tensor_desc does -- it is only ever compared for
// equality, when deciding whether two entries with the same name are the same
// tensor or a genuine conflict.
struct moe_control_tensor_desc {
    std::string name;
    size_t      size  = 0;
    int         type  = -1;
    int64_t     ne[4] = {};
};

struct moe_ptr_table_plan_entry {
    std::string     tensor_name;
    int             layer_id    = -1;
    moe_expert_role role        = moe_expert_role::UNKNOWN;
    size_t          table_index = SIZE_MAX;
    size_t          byte_offset = 0;
    size_t          byte_size   = 0;
};

struct moe_ptr_table_plan {
    std::vector<moe_ptr_table_plan_entry>   entries;
    std::unordered_map<std::string, size_t> index_by_name;
    int                                     n_expert                         = 0;
    size_t                                  table_bytes                      = 0;
    size_t                                  table_stride_bytes               = 0;
    size_t                                  total_bytes                      = 0;
    size_t                                  duplicate_count                  = 0;
    size_t                                  conflicting_duplicate_count      = 0;
    size_t                                  malformed_expert_candidate_count = 0;
    size_t                                  non_moe_ignored_count            = 0;
    size_t                                  missing_role_count               = 0;
    bool                                    arithmetic_overflow              = false;
    bool                                    structurally_complete            = false;

    const moe_ptr_table_plan_entry * lookup(const std::string & name) const;
};

// count * stride with the multiplication checked. Returns false and leaves
// *out untouched on overflow, so a caller that ignores the result cannot pick
// up a wrapped value.
bool moe_checked_pool_bytes(size_t count, size_t stride, size_t * out);

moe_ptr_table_plan moe_build_ptr_table_plan(const std::vector<moe_control_tensor_desc> & inventory, int n_expert);

// Above this micro-batch the routing ids and compaction table stop being a
// control-sized allocation, and the GPU routing path is not admitted. It is a
// property of the layout, not of the device, which is why it is decided here.
static constexpr uint32_t MOE_GPU_UBATCH_MAX = 512;

struct moe_context_control_layout {
    int      n_expert            = 0;
    int      n_expert_used       = 0;
    uint32_t n_ubatch            = 0;
    size_t   table_count         = 0;
    size_t   table_bytes         = 0;
    size_t   table_stride        = 0;
    size_t   tables_offset       = 0;
    size_t   tables_bytes        = 0;
    size_t   ids_offset          = 0;
    size_t   ids_bytes           = 0;
    size_t   compact_offset      = 0;
    size_t   compact_bytes       = 0;
    size_t   missing_offset      = 0;
    size_t   missing_bytes       = 0;
    size_t   total_bytes         = 0;
    bool     valid               = false;
    bool     gpu_eligible        = false;
    bool     exceeds_gpu_ubatch  = false;
    bool     metadata_mismatch   = false;
    bool     arithmetic_overflow = false;
};

// Re-derives every geometry field from (n_expert, entry count) and REJECTS the
// plan when the plan's own numbers disagree. The plan is an input from another
// pass, so trusting its arithmetic would make this a formatting exercise; the
// point is that a plan built with different constants cannot be sized here.
moe_context_control_layout moe_build_context_control_layout(const moe_ptr_table_plan & plan,
                                                            int                        n_expert_used,
                                                            uint32_t                   n_ubatch);

// A byte count paired with whether it is meaningful. `valid == false` is NOT
// "zero bytes": it propagates through the zone arithmetic and makes the whole
// requirement unanswerable, so a caller cannot accidentally read a failed
// sizing as a satisfied one.
struct moe_control_requirement {
    size_t bytes = 0;
    bool   valid = true;
};

// `required` says whether this model actually needs GPU MoE routing. A layout
// that is merely not gpu_eligible on a model that does not need it is a
// zero-byte VALID requirement; the same layout on a model that needs it is
// invalid.
moe_control_requirement moe_control_requirement_from_layout(const moe_context_control_layout & layout, bool required);

// pp_pipeline + pp_moe_onednn + control, with both additions checked. Returns
// false without writing *out when any term is invalid or the sum overflows.
bool moe_checked_runtime_zone_requirement(size_t                          pp_pipeline_bytes,
                                          size_t                          pp_moe_onednn_bytes,
                                          const moe_control_requirement & base_context_control,
                                          size_t *                        out);

enum class moe_control_reason : uint8_t {
    ADMITTED,
    EMPTY_INVENTORY,  // no expert tensors; zero bytes is the right answer
    NO_ROUTE_CAPABILITY,
    RUNTIME_CAPACITY,
    INVALID_LAYOUT,
    INVALID_PLAN_ID,
    PENDING_CONVERSION,  // another thread holds this plan's conversion
};

struct moe_runtime_capacity_snapshot {
    bool   arena_active      = false;
    bool   runtime_available = false;
    size_t runtime_capacity  = 0;
    size_t runtime_used      = 0;
    size_t weight_capacity   = 0;
};

struct moe_context_control_device_plan {
    uint64_t           plan_id         = 0;
    int                device_id       = -1;
    size_t             required_bytes  = 0;
    size_t             available_bytes = 0;
    bool               admitted        = false;
    moe_control_reason reason          = moe_control_reason::RUNTIME_CAPACITY;
};

// Pure admission arithmetic: no ledger, no lock, same answer for the same
// arguments. moe_reserve_context_control is the one that records.
moe_context_control_device_plan moe_plan_context_control(int                             device_id,
                                                         size_t                          runtime_available,
                                                         const moe_control_requirement & requirement,
                                                         bool                            arena_active,
                                                         bool                            route_capable = true);

enum class moe_control_reservation_state : uint8_t {
    NONE,
    UNCONSUMED,          // reserved, no allocation has converted it yet
    CONVERSION_PENDING,  // a conversion is in flight and owns the entry
    ADMITTED_CONVERTED,  // converted; the bytes are real and no longer reserved
};

struct moe_control_reservation_snapshot {
    size_t                        total_unconsumed_bytes = 0;
    size_t                        plan_unconsumed_bytes  = 0;
    size_t                        plan_reference_count   = 0;
    size_t                        plan_count             = 0;
    moe_control_reservation_state plan_state             = moe_control_reservation_state::NONE;
};

// Records a per-(device, plan) reservation against `capacity`, discounting
// every OTHER plan's unconverted reservation on that device so two plans
// cannot both be admitted for the same bytes.
//
// The capacity snapshot is a PARAMETER rather than something this function
// queries, which is the one deliberate departure from the branch this was
// salvaged from. There the ledger lock was taken while resolving the live
// cache, which forced a documented lock-ordering dance; passing the snapshot
// in makes the ledger mutex strictly leaf (canonical §12.5) and costs the
// caller one line.
moe_context_control_device_plan moe_reserve_context_control(uint64_t                              plan_id,
                                                            int                                   device_id,
                                                            const moe_control_requirement &       requirement,
                                                            const moe_runtime_capacity_snapshot & capacity,
                                                            bool                                  route_capable);

bool moe_retain_context_control_reservation(uint64_t plan_id, int device_id);
bool moe_retain_context_control_reservations(uint64_t plan_id);
void moe_release_context_control_reservations(uint64_t plan_id);

// A claim on one plan/device reservation for the duration of one allocation
// attempt. `first` distinguishes the conversion that must be committed or
// rolled back from a later allocation against an already-converted
// reservation, which owns nothing and must not roll anything back.
struct moe_control_conversion_ticket {
    uint64_t plan_id       = 0;
    int      device_id     = -1;
    uint64_t conversion_id = 0;
    bool     valid         = false;
    bool     first         = false;
};

// Moves an UNCONSUMED reservation to CONVERSION_PENDING and stamps it with a
// fresh generation. An already-converted reservation yields a valid ticket
// with `first == false`. Anything else yields an invalid ticket.
moe_control_conversion_ticket moe_begin_control_conversion(uint64_t plan_id, int device_id, size_t bytes);

struct moe_control_allocation_grant {
    bool   granted            = false;
    bool   first_conversion   = false;
    size_t runtime_used_after = 0;
};

// Strict authorization: an allocation is granted only against a live
// reservation whose generation matches, and only when the bytes still fit
// after every other plan's unconverted reservation is subtracted. Granting a
// first conversion CLAIMS it, so a second attempt on the same ticket fails.
moe_control_allocation_grant moe_authorize_control_allocation(uint64_t                              plan_id,
                                                              int                                   device_id,
                                                              uint64_t                              conversion_id,
                                                              size_t                                bytes,
                                                              const moe_runtime_capacity_snapshot & capacity);

// Commit is only legal once the ticket's conversion has been claimed by a
// granted authorization; committing an unclaimed ticket is a caller bug and
// returns false rather than converting bytes nothing allocated.
bool moe_commit_control_conversion(const moe_control_conversion_ticket & ticket);

// Restores CONVERSION_PENDING to UNCONSUMED, or erases the entry when its last
// reference went away mid-conversion. Never throws: it runs from a destructor
// on the failure path, and a reservation ledger that can be left wedged by an
// exception is worse than one that aborts.
void moe_rollback_control_conversion(const moe_control_conversion_ticket & ticket) noexcept;

// Rolls the ticket back unless commit() was reached. This is the whole
// transactional guarantee: every early return, every exception and every
// failure branch between begin and commit leaves the reservation exactly as it
// was, generation included.
struct moe_control_conversion_guard {
    moe_control_conversion_ticket ticket;
    bool                          committed = false;

    moe_control_conversion_guard() = default;

    explicit moe_control_conversion_guard(const moe_control_conversion_ticket & t) : ticket(t) {}

    moe_control_conversion_guard(const moe_control_conversion_guard &)             = delete;
    moe_control_conversion_guard & operator=(const moe_control_conversion_guard &) = delete;

    bool commit() {
        committed = moe_commit_control_conversion(ticket);
        return committed;
    }

    ~moe_control_conversion_guard() {
        if (!committed) {
            moe_rollback_control_conversion(ticket);
        }
    }
};

// Per-plan planned CONTROL bytes, summed per device to size the RUNTIME zone
// before any arena exists. A zero-byte VALID requirement ERASES the plan's
// entry rather than storing zero, so a model that turned out to have no MoE
// layers stops contributing; an INVALID one is stored, so it can poison the
// device total and refuse the sizing.
void moe_publish_planned_control_bytes(uint64_t plan_id, int device_id, const moe_control_requirement & requirement);
void moe_reset_planned_control_bytes(uint64_t plan_id);
moe_control_requirement moe_planned_control_bytes_for_device(int device_id);
moe_control_requirement moe_planned_control_bytes_for_plan(uint64_t plan_id, int device_id);

moe_control_reservation_snapshot moe_control_reservation_state_for(int device_id, uint64_t plan_id);

// Test-only: drops every reservation and every planned publication. Production
// code releases by plan id; a global clear would discard a live plan's ledger.
void moe_control_reset_all_state();

}  // namespace ggml_sycl
