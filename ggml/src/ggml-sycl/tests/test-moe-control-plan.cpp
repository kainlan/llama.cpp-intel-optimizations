// Exact MoE pointer-table descriptor planning, CONTROL sizing and the strict
// reservation ledger (llama.cpp-dimc).
//
// Host-only: no SYCL queue, no device, no model, no arena. Every capacity the
// reservation ledger sees is a snapshot this file constructs, which is what
// makes the admission and conversion arithmetic testable at all.
//
// The case this whole port exists for is `exact-beats-coarse-both-ways`: the
// MAX_EXPERTS * n_moe_layers budget it replaces is wrong in BOTH directions --
// 4x too large on a fused-gate_up model, and 3x too small on a split
// gate/up/down one. A sizing that is merely "conservative" would only need the
// first half.

#include "moe-control-plan.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using ggml_sycl::moe_authorize_control_allocation;
using ggml_sycl::moe_begin_control_conversion;
using ggml_sycl::moe_build_context_control_layout;
using ggml_sycl::moe_build_ptr_table_plan;
using ggml_sycl::moe_checked_pool_bytes;
using ggml_sycl::moe_checked_runtime_zone_requirement;
using ggml_sycl::moe_commit_control_conversion;
using ggml_sycl::moe_context_control_layout;
using ggml_sycl::moe_control_conversion_guard;
using ggml_sycl::moe_control_reason;
using ggml_sycl::moe_control_requirement;
using ggml_sycl::moe_control_requirement_from_layout;
using ggml_sycl::moe_control_reservation_state;
using ggml_sycl::moe_control_reservation_state_for;
using ggml_sycl::moe_control_reset_all_state;
using ggml_sycl::moe_control_tensor_desc;
using ggml_sycl::moe_expert_candidate;
using ggml_sycl::moe_expert_role;
using ggml_sycl::MOE_GPU_UBATCH_MAX;
using ggml_sycl::moe_parse_expert_identity;
using ggml_sycl::moe_plan_context_control;
using ggml_sycl::moe_planned_control_bytes_for_device;
using ggml_sycl::moe_ptr_table_plan;
using ggml_sycl::moe_publish_planned_control_bytes;
using ggml_sycl::moe_release_context_control_reservations;
using ggml_sycl::moe_reserve_context_control;
using ggml_sycl::moe_retain_context_control_reservations;
using ggml_sycl::moe_runtime_capacity_snapshot;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static moe_control_tensor_desc desc(const std::string & name, size_t size = 1024, int type = 8) {
    moe_control_tensor_desc d;
    d.name  = name;
    d.size  = size;
    d.type  = type;
    d.ne[0] = 64;
    d.ne[1] = 64;
    d.ne[2] = 32;
    d.ne[3] = 1;
    return d;
}

// blk.<layer>.ffn_<role>_exps.weight for every layer, one entry per role.
static std::vector<moe_control_tensor_desc> build_inventory(int n_layers, const std::vector<std::string> & roles) {
    std::vector<moe_control_tensor_desc> inventory;
    for (int layer = 0; layer < n_layers; ++layer) {
        for (const std::string & role : roles) {
            inventory.push_back(desc("blk." + std::to_string(layer) + ".ffn_" + role + ".weight"));
        }
    }
    return inventory;
}

// What populate_host_zone_sizing budgeted before this port: MAX_EXPERTS slots
// per MoE layer, regardless of the model's expert count or how many expert
// tensors each layer carries.
static size_t coarse_budget(int n_moe_layers) {
    return static_cast<size_t>(256) * sizeof(void *) * static_cast<size_t>(n_moe_layers);
}

// --- descriptor identity -----------------------------------------------------

static void case_identity_parse_is_anchored() {
    check(moe_parse_expert_identity("blk.7.ffn_gate_exps.weight").candidate == moe_expert_candidate::MATRIX,
          "canonical gate name must be a matrix");
    check(moe_parse_expert_identity("blk.7.ffn_gate_exps.weight").layer_id == 7, "layer id must parse");
    check(moe_parse_expert_identity("blk.7.ffn_gate_exps.weight").role == moe_expert_role::GATE, "role must parse");
    check(moe_parse_expert_identity("blk.12.ffn_gate_up_exps.weight").role == moe_expert_role::GATE_UP,
          "fused gate_up must not classify as gate");

    check(moe_parse_expert_identity("blk.3.ffn_norm_exps.weight").candidate == moe_expert_candidate::IGNORED,
          "a norm is not a sizable expert matrix");
    check(moe_parse_expert_identity("blk.3.ffn_gate_exps.bias").candidate == moe_expert_candidate::IGNORED,
          "a bias is not a sizable expert matrix");
    check(moe_parse_expert_identity("output.weight").candidate == moe_expert_candidate::IGNORED,
          "a non-expert tensor is ignored, not malformed");

    // The anchoring is the point: anything that advertises itself as an expert
    // tensor but does not parse must be MALFORMED, never quietly ignored.
    check(moe_parse_expert_identity("prefix/blk.3.ffn_gate_exps.weight").candidate == moe_expert_candidate::MALFORMED,
          "an embedded path must be malformed");
    check(moe_parse_expert_identity("blk.x.ffn_gate_exps.weight").candidate == moe_expert_candidate::MALFORMED,
          "a non-numeric layer must be malformed");
    check(moe_parse_expert_identity("blk.3.ffn_wat_exps.weight").candidate == moe_expert_candidate::MALFORMED,
          "an unknown expert suffix must be malformed");
    check(moe_parse_expert_identity(nullptr).candidate == moe_expert_candidate::IGNORED, "null must not crash");
}

static void case_grovemoe_chexps_family_parses() {
    // "_chexps" does not contain "_exps"; a guard keyed only on the latter
    // drops the whole grovemoe family silently, which is a zero-byte budget
    // for tensors that exist.
    check(moe_parse_expert_identity("blk.2.ffn_gate_chexps.weight").candidate == moe_expert_candidate::MATRIX,
          "chunked gate must be a matrix");
    check(moe_parse_expert_identity("blk.2.ffn_gate_chexps.weight").role == moe_expert_role::CHUNK_GATE,
          "chunked gate must carry its own role");
    check(moe_parse_expert_identity("blk.2.ffn_down_chexps.weight").role == moe_expert_role::CHUNK_DOWN,
          "chunked down must carry its own role");
    check(moe_parse_expert_identity("blk.2.ffn_up_chexps.bias").candidate == moe_expert_candidate::IGNORED,
          "a chunked bias is ignored, not malformed");
}

// --- exact sizing ------------------------------------------------------------

static void case_dense_model_is_a_valid_empty_layout() {
    const moe_ptr_table_plan plan = moe_build_ptr_table_plan({ desc("output.weight") }, 0);
    check(plan.structurally_complete, "a dense model is a complete plan");
    check(plan.entries.empty(), "a dense model has no expert tables");

    const moe_context_control_layout layout = moe_build_context_control_layout(plan, 0, 512);
    check(layout.valid, "an empty layout is valid");
    check(!layout.gpu_eligible, "there is nothing to route on a dense model");
    check(layout.total_bytes == 0, "an empty layout costs nothing");

    // Valid-and-zero, even when routing is demanded: the model genuinely needs
    // no CONTROL memory.
    check(moe_control_requirement_from_layout(layout, true).valid, "empty is a real zero, not a failure");
    check(moe_control_requirement_from_layout(layout, true).bytes == 0, "empty costs nothing");
}

static void case_exact_beats_coarse_both_ways() {
    // GPT-OSS shape: fused gate_up plus down, 24 layers, 32 experts.
    // exact = 48 tables x align256(32 * 8) = 48 * 256 = 12288
    // coarse = 256 * 8 * 24 = 49152  -> the coarse figure is 4x too LARGE.
    {
        const auto inventory = build_inventory(24, { "gate_up_exps", "down_exps" });
        const auto plan      = moe_build_ptr_table_plan(inventory, 32);
        check(plan.structurally_complete, "fused gate_up model must plan cleanly");
        check(plan.entries.size() == 48, "24 layers x 2 tensors");

        const auto layout = moe_build_context_control_layout(plan, 4, 512);
        check(layout.valid && layout.gpu_eligible, "fused gate_up model must be routable");
        check(layout.table_bytes == 32 * sizeof(void *), "one pointer per expert");
        check(layout.table_stride == 256, "stride is the 256B control alignment");
        check(layout.tables_bytes == 48 * 256, "exact table bytes");
        check(layout.tables_bytes < coarse_budget(24), "coarse over-counts the fused model");
        check(coarse_budget(24) / layout.tables_bytes == 4, "coarse is 4x too large here");
    }

    // Split gate/up/down, 24 layers, 256 experts.
    // exact = 72 tables x align256(256 * 8) = 72 * 2048 = 147456
    // coarse = 256 * 8 * 24 = 49152  -> the coarse figure is 3x too SMALL.
    {
        const auto inventory = build_inventory(24, { "gate_exps", "up_exps", "down_exps" });
        const auto plan      = moe_build_ptr_table_plan(inventory, 256);
        check(plan.entries.size() == 72, "24 layers x 3 tensors");

        const auto layout = moe_build_context_control_layout(plan, 8, 512);
        check(layout.valid && layout.gpu_eligible, "split model must be routable");
        check(layout.tables_bytes == 72 * 2048, "exact table bytes");
        check(layout.tables_bytes > coarse_budget(24), "coarse UNDER-counts the split model");
        check(layout.tables_bytes / coarse_budget(24) == 3, "coarse is 3x too small here");
    }
}

static void case_layout_regions_are_aligned_and_disjoint() {
    const auto plan   = moe_build_ptr_table_plan(build_inventory(4, { "gate_exps", "up_exps", "down_exps" }), 32);
    const auto layout = moe_build_context_control_layout(plan, 4, 256);
    check(layout.valid && layout.gpu_eligible, "layout must be routable");

    check(layout.tables_offset == 0, "tables come first");
    check(layout.ids_offset % 256 == 0, "ids must be aligned");
    check(layout.compact_offset % 256 == 0, "compact must be aligned");
    check(layout.missing_offset % 256 == 0, "missing must be aligned");
    check(layout.total_bytes % 256 == 0, "the pool total must be aligned");

    check(layout.ids_offset >= layout.tables_offset + layout.tables_bytes, "ids must clear the tables");
    check(layout.compact_offset >= layout.ids_offset + layout.ids_bytes, "compact must clear the ids");
    check(layout.missing_offset >= layout.compact_offset + layout.compact_bytes, "missing must clear compact");
    check(layout.total_bytes >= layout.missing_offset + layout.missing_bytes, "the total must cover every region");

    check(layout.ids_bytes == 4u * 256u * sizeof(int32_t), "ids are n_expert_used x n_ubatch int32");
    check(layout.compact_bytes == 4u * 256u * sizeof(void *), "compact is n_expert_used x n_ubatch pointers");
}

static void case_table_indices_are_order_independent() {
    auto forward  = build_inventory(6, { "gate_exps", "up_exps", "down_exps" });
    auto shuffled = forward;
    std::reverse(shuffled.begin(), shuffled.end());

    const auto a = moe_build_ptr_table_plan(forward, 32);
    const auto b = moe_build_ptr_table_plan(shuffled, 32);
    check(a.entries.size() == b.entries.size(), "same inventory, same entry count");
    for (size_t i = 0; i < a.entries.size(); ++i) {
        check(a.entries[i].tensor_name == b.entries[i].tensor_name, "table index must not depend on inventory order");
        check(a.entries[i].byte_offset == b.entries[i].byte_offset, "byte offset must not depend on inventory order");
    }
}

static void case_malformed_descriptor_fails_closed() {
    auto inventory = build_inventory(2, { "gate_exps", "up_exps", "down_exps" });
    inventory.push_back(desc("blk.rogue.ffn_gate_exps.weight"));

    const auto plan = moe_build_ptr_table_plan(inventory, 32);
    check(plan.malformed_expert_candidate_count == 1, "the malformed name must be counted");
    check(!plan.structurally_complete, "a malformed descriptor makes the plan incomplete");

    const auto layout = moe_build_context_control_layout(plan, 4, 256);
    check(!layout.valid, "an incomplete plan has no valid layout");
    check(layout.metadata_mismatch, "and it reports why");

    // The whole point of `required`: the same unusable layout is a hard
    // refusal for a model that needs routing, and a real zero for one that
    // does not.
    check(!moe_control_requirement_from_layout(layout, true).valid, "a needed-but-unsizable layout must refuse");
    check(moe_control_requirement_from_layout(layout, false).valid, "an unneeded one degrades to zero");
    check(moe_control_requirement_from_layout(layout, false).bytes == 0, "and that zero is zero");
}

static void case_duplicate_descriptors_are_distinguished() {
    auto exact = build_inventory(2, { "gate_exps", "up_exps", "down_exps" });
    exact.push_back(exact.front());
    const auto exact_plan = moe_build_ptr_table_plan(exact, 32);
    check(exact_plan.duplicate_count == 1, "an identical repeat is a benign duplicate");
    check(exact_plan.conflicting_duplicate_count == 0, "and not a conflict");
    check(exact_plan.structurally_complete, "a benign duplicate does not break the plan");

    auto conflicting = build_inventory(2, { "gate_exps", "up_exps", "down_exps" });
    auto clash       = conflicting.front();
    clash.size += 1;  // same name, different tensor
    conflicting.push_back(clash);
    const auto conflicting_plan = moe_build_ptr_table_plan(conflicting, 32);
    check(conflicting_plan.conflicting_duplicate_count == 1, "a disagreeing repeat is a conflict");
    check(!conflicting_plan.structurally_complete, "a conflict makes the plan incomplete");
}

static void case_incomplete_layer_is_refused() {
    // A layer with gate and up but no down cannot be dispatched.
    std::vector<moe_control_tensor_desc> inventory = build_inventory(1, { "gate_exps", "up_exps", "down_exps" });
    inventory.push_back(desc("blk.1.ffn_gate_exps.weight"));
    inventory.push_back(desc("blk.1.ffn_up_exps.weight"));

    const auto plan   = moe_build_ptr_table_plan(inventory, 32);
    const auto layout = moe_build_context_control_layout(plan, 4, 256);
    check(!layout.valid, "a layer missing a role has no valid layout");
    check(layout.metadata_mismatch, "and it reports a metadata mismatch");
}

static void case_role_collision_within_a_layer_is_refused() {
    // Fused gate_up alongside a split gate: the fused tensor already supplies
    // the gate, so this layer claims the gate role twice.
    std::vector<moe_control_tensor_desc> inventory = {
        desc("blk.0.ffn_gate_up_exps.weight"),
        desc("blk.0.ffn_gate_exps.weight"),
        desc("blk.0.ffn_down_exps.weight"),
    };
    const auto plan   = moe_build_ptr_table_plan(inventory, 32);
    const auto layout = moe_build_context_control_layout(plan, 4, 256);
    check(!layout.valid, "a doubled logical role has no valid layout");
    check(layout.metadata_mismatch, "and it reports a metadata mismatch");
}

static void case_partial_chunk_trio_is_refused() {
    std::vector<moe_control_tensor_desc> complete = build_inventory(1, { "gate_exps", "up_exps", "down_exps" });
    for (const char * const role : { "gate_chexps", "up_chexps", "down_chexps" }) {
        complete.push_back(desc(std::string("blk.0.ffn_") + role + ".weight"));
    }
    const auto whole = moe_build_context_control_layout(moe_build_ptr_table_plan(complete, 32), 4, 256);
    check(whole.valid && whole.gpu_eligible, "a full primary + chunk layer is routable");
    check(whole.table_count == 6, "both trios get tables");

    std::vector<moe_control_tensor_desc> partial = build_inventory(1, { "gate_exps", "up_exps", "down_exps" });
    partial.push_back(desc("blk.0.ffn_gate_chexps.weight"));
    const auto broken = moe_build_context_control_layout(moe_build_ptr_table_plan(partial, 32), 4, 256);
    check(!broken.valid, "half a chunk trio is not a plan");
    check(broken.metadata_mismatch, "and it reports a metadata mismatch");
}

static void case_oversized_ubatch_is_not_gpu_eligible() {
    const auto plan   = moe_build_ptr_table_plan(build_inventory(2, { "gate_exps", "up_exps", "down_exps" }), 32);
    const auto layout = moe_build_context_control_layout(plan, 4, MOE_GPU_UBATCH_MAX + 1);
    check(!layout.valid, "an oversized ubatch has no layout");
    check(layout.exceeds_gpu_ubatch, "and it says which limit it broke");
    check(!layout.arithmetic_overflow, "it is a policy limit, not an overflow");
}

static void case_checked_pool_bytes_refuses_overflow() {
    // The layout's own arithmetic cannot be driven to overflow from this side:
    // n_expert is an int and n_ubatch is capped at MOE_GPU_UBATCH_MAX, so every
    // product stays well inside a 64-bit size_t. The guard is still the thing
    // standing between a bad count and a wrapped allocation size, so it is
    // tested where it can actually be exercised.
    size_t out = 999;
    check(!moe_checked_pool_bytes(std::numeric_limits<size_t>::max(), 2, &out), "an overflowing product is refused");
    check(out == 999, "and it must not write a wrapped value");
    check(!moe_checked_pool_bytes(2, std::numeric_limits<size_t>::max(), &out), "either way round");
    check(!moe_checked_pool_bytes(1, 1, nullptr), "a null out is refused");

    check(moe_checked_pool_bytes(std::numeric_limits<size_t>::max() / 8, 8, &out), "the largest exact product fits");
    check(out == (std::numeric_limits<size_t>::max() / 8) * 8, "and is exact");
    check(moe_checked_pool_bytes(std::numeric_limits<size_t>::max(), 0, &out), "a zero stride never overflows");
    check(out == 0, "and yields zero");

    // A large-but-representable expert count must produce a layout whose
    // regions still stack up, which is what proves nothing wrapped.
    const auto plan   = moe_build_ptr_table_plan(build_inventory(1, { "gate_exps", "up_exps", "down_exps" }), 1 << 20);
    const auto layout = moe_build_context_control_layout(plan, 8, 256);
    check(layout.valid && layout.gpu_eligible, "a large expert count is still sizable");
    check(layout.tables_bytes == 3ull * (1ull << 23), "3 tables x 1M experts x 8 bytes");
    check(layout.total_bytes > layout.missing_offset, "the total must still exceed the last region's offset");
}

static void case_zone_requirement_arithmetic_is_checked() {
    size_t                  out         = 0;
    moe_control_requirement requirement = {};
    requirement.bytes                   = 4096;

    check(moe_checked_runtime_zone_requirement(1024, 2048, requirement, &out), "a plain sum must succeed");
    check(out == 1024 + 2048 + 4096, "and it must be the sum");

    moe_control_requirement invalid = {};
    invalid.valid                   = false;
    out                             = 12345;
    check(!moe_checked_runtime_zone_requirement(1024, 2048, invalid, &out), "an invalid control term poisons the sum");
    check(out == 12345, "and it must not write a partial answer");

    moe_control_requirement huge = {};
    huge.bytes                   = std::numeric_limits<size_t>::max();
    check(!moe_checked_runtime_zone_requirement(1024, 0, huge, &out), "an overflowing sum must be refused");
    check(!moe_checked_runtime_zone_requirement(0, 0, requirement, nullptr), "a null out must be refused");
}

// --- admission and the strict reservation ledger -----------------------------

static moe_runtime_capacity_snapshot capacity(size_t total, size_t used) {
    moe_runtime_capacity_snapshot snapshot;
    snapshot.arena_active      = true;
    snapshot.runtime_available = true;
    snapshot.runtime_capacity  = total;
    snapshot.runtime_used      = used;
    return snapshot;
}

static moe_control_requirement bytes(size_t n) {
    moe_control_requirement requirement;
    requirement.bytes = n;
    return requirement;
}

static void case_admission_arithmetic_is_pure() {
    check(moe_plan_context_control(0, 8192, bytes(4096), true, true).admitted, "it fits");
    check(!moe_plan_context_control(0, 1024, bytes(4096), true, true).admitted, "it does not fit");
    check(moe_plan_context_control(0, 1024, bytes(4096), true, true).reason == moe_control_reason::RUNTIME_CAPACITY,
          "and the reason is capacity");
    check(!moe_plan_context_control(0, 8192, bytes(4096), false, true).admitted, "no arena, no admission");
    check(!moe_plan_context_control(0, 8192, bytes(4096), true, false).admitted, "no route capability, no admission");
    check(moe_plan_context_control(0, 8192, bytes(4096), true, false).reason == moe_control_reason::NO_ROUTE_CAPABILITY,
          "and the reason says so");

    moe_control_requirement invalid = {};
    invalid.valid                   = false;
    check(moe_plan_context_control(0, 8192, invalid, true, true).reason == moe_control_reason::INVALID_LAYOUT,
          "an invalid requirement is never admitted");
}

static void case_reservations_do_not_double_admit() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);

    check(moe_reserve_context_control(1, 0, bytes(6000), space, true).admitted, "the first plan fits");
    // The second plan sees 10000 - 0 - 6000 = 4000 available, not 10000.
    const auto second = moe_reserve_context_control(2, 0, bytes(6000), space, true);
    check(!second.admitted, "the second plan must be charged the first plan's reservation");
    check(second.available_bytes == 4000, "and it must see the reduced availability");
    check(moe_reserve_context_control(3, 0, bytes(4000), space, true).admitted, "what remains is still admissible");

    moe_control_reset_all_state();
}

static void case_zero_and_invalid_requirements_are_distinguished() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);

    check(moe_reserve_context_control(1, 0, bytes(6000), space, true).admitted, "seed a reservation");

    // Invalid must refuse WITHOUT disturbing the live reservation.
    moe_control_requirement invalid = {};
    invalid.valid                   = false;
    const auto refused              = moe_reserve_context_control(1, 0, invalid, space, true);
    check(!refused.admitted, "an invalid requirement is refused");
    check(refused.reason == moe_control_reason::INVALID_LAYOUT, "and says why");
    check(moe_control_reservation_state_for(0, 1).plan_unconsumed_bytes == 6000,
          "and it must leave the live reservation alone");

    // A zero requirement is an admitted transition to dense, and it releases.
    const auto emptied = moe_reserve_context_control(1, 0, bytes(0), space, true);
    check(emptied.admitted, "a zero requirement is admitted");
    check(emptied.reason == moe_control_reason::EMPTY_INVENTORY, "as an empty inventory");
    check(moe_control_reservation_state_for(0, 1).plan_count == 0, "and it drops the reservation");

    const auto no_plan_id = moe_reserve_context_control(0, 0, bytes(4096), space, true);
    check(!no_plan_id.admitted, "bytes without a plan id cannot be reserved");
    check(no_plan_id.reason == moe_control_reason::INVALID_PLAN_ID, "and it says so");

    moe_control_reset_all_state();
}

static void case_reservation_references_are_counted() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);

    check(moe_reserve_context_control(1, 0, bytes(4096), space, true).admitted, "reserve");
    check(moe_retain_context_control_reservations(1), "retain the plan across devices");
    check(moe_control_reservation_state_for(0, 1).plan_reference_count == 2, "two references now");

    moe_release_context_control_reservations(1);
    check(moe_control_reservation_state_for(0, 1).plan_reference_count == 1, "one reference left");
    check(moe_control_reservation_state_for(0, 1).plan_count == 1, "the entry survives");

    moe_release_context_control_reservations(1);
    check(moe_control_reservation_state_for(0, 1).plan_count == 0, "the last release drops it");
    check(!moe_retain_context_control_reservations(1), "and it cannot be retained afterwards");

    moe_control_reset_all_state();
}

// --- the transactional conversion protocol -----------------------------------

static void case_conversion_commits_and_stops_being_reserved() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);
    check(moe_reserve_context_control(1, 0, bytes(4096), space, true).admitted, "reserve");
    check(moe_control_reservation_state_for(0, 1).total_unconsumed_bytes == 4096, "reserved but unconverted");

    const auto ticket = moe_begin_control_conversion(1, 0, 4096);
    check(ticket.valid && ticket.first, "a fresh conversion owns the transaction");
    check(ticket.conversion_id != 0, "and carries a generation");
    check(moe_control_reservation_state_for(0, 1).plan_state == moe_control_reservation_state::CONVERSION_PENDING,
          "the entry is pending");

    const auto grant = moe_authorize_control_allocation(1, 0, ticket.conversion_id, 4096, space);
    check(grant.granted && grant.first_conversion, "the allocation is authorized against its own reservation");

    check(moe_commit_control_conversion(ticket), "commit succeeds once claimed");
    check(moe_control_reservation_state_for(0, 1).plan_state == moe_control_reservation_state::ADMITTED_CONVERTED,
          "the entry is converted");
    check(moe_control_reservation_state_for(0, 1).total_unconsumed_bytes == 0,
          "converted bytes stop being counted as reserved -- they are in runtime_used now");

    moe_control_reset_all_state();
}

static void case_failed_conversion_rolls_back_and_regenerates() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);
    check(moe_reserve_context_control(1, 0, bytes(4096), space, true).admitted, "reserve");

    uint64_t first_generation = 0;
    {
        // A guard that is never committed: this is every early return, every
        // failure branch and every exception between begin and commit.
        moe_control_conversion_guard guard(moe_begin_control_conversion(1, 0, 4096));
        first_generation = guard.ticket.conversion_id;
        check(guard.ticket.valid && guard.ticket.first, "the conversion started");
        check(moe_authorize_control_allocation(1, 0, guard.ticket.conversion_id, 4096, space).granted,
              "and was authorized");
    }

    const auto after = moe_control_reservation_state_for(0, 1);
    check(after.plan_state == moe_control_reservation_state::UNCONSUMED, "rollback restores the reservation");
    check(after.plan_unconsumed_bytes == 4096, "with its bytes intact");
    check(after.plan_reference_count == 1, "and its reference intact");

    // Generation rollback must not be regressed: the retry gets a NEW
    // generation, and the rolled-back one is dead.
    const auto retry = moe_begin_control_conversion(1, 0, 4096);
    check(retry.valid && retry.first, "the retry starts a conversion");
    check(retry.conversion_id != first_generation, "the retry must carry a fresh generation");
    check(!moe_authorize_control_allocation(1, 0, first_generation, 4096, space).granted,
          "the rolled-back generation must be dead");
    check(!moe_commit_control_conversion({ 1, 0, first_generation, true, true }), "and a stale ticket must not commit");

    moe_control_reset_all_state();
}

static void case_a_claim_cannot_be_spent_twice() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);
    check(moe_reserve_context_control(1, 0, bytes(4096), space, true).admitted, "reserve");

    const auto ticket = moe_begin_control_conversion(1, 0, 4096);
    check(moe_authorize_control_allocation(1, 0, ticket.conversion_id, 4096, space).granted, "first claim");
    check(!moe_authorize_control_allocation(1, 0, ticket.conversion_id, 4096, space).granted,
          "the same conversion cannot be claimed twice");

    // A second begin while one is pending must not hand out a second ticket.
    check(!moe_begin_control_conversion(1, 0, 4096).valid, "a pending conversion blocks a second begin");

    moe_control_reset_all_state();
}

static void case_commit_requires_an_authorized_claim() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);
    check(moe_reserve_context_control(1, 0, bytes(4096), space, true).admitted, "reserve");

    const auto ticket = moe_begin_control_conversion(1, 0, 4096);
    check(ticket.valid, "the conversion started");
    check(!moe_commit_control_conversion(ticket), "commit without an authorized allocation must fail");
    check(moe_control_reservation_state_for(0, 1).plan_state == moe_control_reservation_state::CONVERSION_PENDING,
          "and it must not convert anything");

    moe_control_reset_all_state();
}

static void case_authorization_respects_competing_reservations() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);
    check(moe_reserve_context_control(1, 0, bytes(4000), space, true).admitted, "plan 1 reserves");
    check(moe_reserve_context_control(2, 0, bytes(4000), space, true).admitted, "plan 2 reserves");

    const auto ticket = moe_begin_control_conversion(1, 0, 4000);
    check(ticket.valid && ticket.first, "plan 1 converts");
    // Plan 1 spends its own 4000; plan 2's 4000 still stands, so only
    // 10000 - 4000(other) = 6000 is spendable and 4000 fits.
    check(moe_authorize_control_allocation(1, 0, ticket.conversion_id, 4000, space).granted,
          "a plan may spend its own reservation");
    // But it may not spend more than the competing reservation leaves.
    check(!moe_authorize_control_allocation(1, 0, ticket.conversion_id, 7000, space).granted,
          "it may not spend into another plan's reservation");

    moe_control_reset_all_state();
}

static void case_release_during_conversion_defers_to_rollback() {
    moe_control_reset_all_state();
    const auto space = capacity(10000, 0);
    check(moe_reserve_context_control(1, 0, bytes(4096), space, true).admitted, "reserve");

    const auto ticket = moe_begin_control_conversion(1, 0, 4096);
    check(ticket.valid && ticket.first, "conversion in flight");

    // The last holder goes away mid-conversion. The entry must survive so the
    // in-flight transaction has something to roll back, but publication must
    // now fail.
    moe_release_context_control_reservations(1);
    check(moe_control_reservation_state_for(0, 1).plan_count == 1, "the entry survives its last release");
    check(moe_control_reservation_state_for(0, 1).plan_reference_count == 0, "with no references");
    check(!moe_authorize_control_allocation(1, 0, ticket.conversion_id, 4096, space).granted,
          "an unreferenced reservation cannot authorize");
    check(!moe_commit_control_conversion(ticket), "and it cannot commit");

    ggml_sycl::moe_rollback_control_conversion(ticket);
    check(moe_control_reservation_state_for(0, 1).plan_count == 0, "rollback reaps the abandoned entry");

    moe_control_reset_all_state();
}

// --- planned-bytes publication ----------------------------------------------

static void case_planned_bytes_sum_and_poison() {
    moe_control_reset_all_state();

    moe_publish_planned_control_bytes(1, 0, bytes(4096));
    moe_publish_planned_control_bytes(2, 0, bytes(2048));
    check(moe_planned_control_bytes_for_device(0).valid, "two valid plans sum");
    check(moe_planned_control_bytes_for_device(0).bytes == 6144, "to their total");

    // A valid zero ERASES rather than storing zero, so a model that turned out
    // dense stops contributing.
    moe_publish_planned_control_bytes(2, 0, bytes(0));
    check(moe_planned_control_bytes_for_device(0).bytes == 4096, "a valid zero drops the plan's contribution");

    // An invalid one poisons the device total instead of vanishing.
    moe_control_requirement invalid = {};
    invalid.valid                   = false;
    moe_publish_planned_control_bytes(3, 0, invalid);
    check(!moe_planned_control_bytes_for_device(0).valid, "an invalid plan poisons the device total");

    size_t out = 0;
    check(!moe_checked_runtime_zone_requirement(1024, 1024, moe_planned_control_bytes_for_device(0), &out),
          "and that poison reaches the zone sizing");

    ggml_sycl::moe_reset_planned_control_bytes(3);
    check(moe_planned_control_bytes_for_device(0).valid, "resetting the bad plan clears the poison");
    check(moe_planned_control_bytes_for_device(0).bytes == 4096, "leaving the good one");

    moe_control_reset_all_state();
    check(moe_planned_control_bytes_for_device(0).bytes == 0, "reset clears everything");
}

int main() {
    struct test_case {
        const char * name;
        void (*run)();
    };

    const test_case cases[] = {
        { "identity-parse-is-anchored",          case_identity_parse_is_anchored                      },
        { "grovemoe-chexps-family-parses",       case_grovemoe_chexps_family_parses                   },
        { "dense-model-is-valid-empty",          case_dense_model_is_a_valid_empty_layout             },
        { "exact-beats-coarse-both-ways",        case_exact_beats_coarse_both_ways                    },
        { "layout-regions-aligned-disjoint",     case_layout_regions_are_aligned_and_disjoint         },
        { "table-indices-order-independent",     case_table_indices_are_order_independent             },
        { "malformed-descriptor-fails-closed",   case_malformed_descriptor_fails_closed               },
        { "duplicates-are-distinguished",        case_duplicate_descriptors_are_distinguished         },
        { "incomplete-layer-refused",            case_incomplete_layer_is_refused                     },
        { "role-collision-refused",              case_role_collision_within_a_layer_is_refused        },
        { "partial-chunk-trio-refused",          case_partial_chunk_trio_is_refused                   },
        { "oversized-ubatch-not-eligible",       case_oversized_ubatch_is_not_gpu_eligible            },
        { "checked-pool-bytes-refuses-overflow", case_checked_pool_bytes_refuses_overflow             },
        { "zone-requirement-arithmetic-checked", case_zone_requirement_arithmetic_is_checked          },
        { "admission-arithmetic-is-pure",        case_admission_arithmetic_is_pure                    },
        { "reservations-do-not-double-admit",    case_reservations_do_not_double_admit                },
        { "zero-vs-invalid-distinguished",       case_zero_and_invalid_requirements_are_distinguished },
        { "reservation-references-counted",      case_reservation_references_are_counted              },
        { "conversion-commits",                  case_conversion_commits_and_stops_being_reserved     },
        { "failed-conversion-regenerates",       case_failed_conversion_rolls_back_and_regenerates    },
        { "claim-cannot-be-spent-twice",         case_a_claim_cannot_be_spent_twice                   },
        { "commit-requires-a-claim",             case_commit_requires_an_authorized_claim             },
        { "authorization-respects-competitors",  case_authorization_respects_competing_reservations   },
        { "release-during-conversion",           case_release_during_conversion_defers_to_rollback    },
        { "planned-bytes-sum-and-poison",        case_planned_bytes_sum_and_poison                    },
    };

    for (const test_case & entry : cases) {
        try {
            entry.run();
        } catch (const std::exception & error) {
            std::cerr << "FAIL: " << entry.name << ": " << error.what() << '\n';
            return 1;
        }
    }

    std::cout << "moe control plan: PASS (" << (sizeof(cases) / sizeof(cases[0])) << " cases)\n";
    return 0;
}
