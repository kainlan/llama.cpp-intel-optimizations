//
// Exact MoE pointer-table descriptor planning and CONTROL zone sizing.
// See moe-control-plan.hpp for why this unit depends on nothing.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "moe-control-plan.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>

namespace ggml_sycl {

namespace {

// Every offset in the CONTROL pool is aligned to this. It is the pointer-table
// stride as well, so a table's base address is usable by a kernel ABI that
// wants aligned loads without the layout having to special-case the first one.
constexpr size_t k_control_alignment = 256;

constexpr uint8_t k_role_bit_gate       = 1u << 0;
constexpr uint8_t k_role_bit_up         = 1u << 1;
constexpr uint8_t k_role_bit_down       = 1u << 2;
constexpr uint8_t k_role_bit_chunk_gate = 1u << 3;
constexpr uint8_t k_role_bit_chunk_up   = 1u << 4;
constexpr uint8_t k_role_bit_chunk_down = 1u << 5;

constexpr uint8_t k_role_mask_primary = k_role_bit_gate | k_role_bit_up | k_role_bit_down;
constexpr uint8_t k_role_mask_chunk   = k_role_bit_chunk_gate | k_role_bit_chunk_up | k_role_bit_chunk_down;

bool checked_add(size_t a, size_t b, size_t * out) {
    if (out == nullptr || b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    *out = a + b;
    return true;
}

bool checked_align(size_t value, size_t * out) {
    if (out == nullptr || value > std::numeric_limits<size_t>::max() - (k_control_alignment - 1)) {
        return false;
    }
    *out = (value + k_control_alignment - 1) & ~(k_control_alignment - 1);
    return true;
}

// Materialization order within a layer: gate (or the fused gate_up standing in
// for it), then up, then down, then the chunk trio. Entries are sorted by it so
// two runs over the same inventory produce byte-identical table indices --
// index stability is what lets a recorded graph keep using the same slot.
int materialization_role_order(moe_expert_role role) {
    switch (role) {
        case moe_expert_role::GATE:
        case moe_expert_role::GATE_UP:  // fused gate+up materializes where gate would
            return 0;
        case moe_expert_role::UP:
            return 1;
        case moe_expert_role::DOWN:
            return 2;
        case moe_expert_role::CHUNK_GATE:
            return 3;
        case moe_expert_role::CHUNK_UP:
            return 4;
        case moe_expert_role::CHUNK_DOWN:
            return 5;
        case moe_expert_role::UNKNOWN:
        default:
            return 6;
    }
}

bool tensor_identity_equal(const moe_control_tensor_desc & a, const moe_control_tensor_desc & b) {
    if (a.name != b.name || a.size != b.size || a.type != b.type) {
        return false;
    }
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            return false;
        }
    }
    return true;
}

struct reservation_entry {
    size_t                        bytes              = 0;
    size_t                        references         = 1;
    moe_control_reservation_state state              = moe_control_reservation_state::UNCONSUMED;
    uint64_t                      conversion_id      = 0;
    bool                          allocation_claimed = false;
};

// Strictly leaf (canonical §12.5, L5): nothing is acquired while it is held,
// nothing is allocated under it beyond the ledger's own nodes, and no device
// work is submitted. That is only true because the capacity snapshot is passed
// in rather than queried here -- see the note on moe_reserve_context_control.
std::mutex                                                                     g_mutex;
std::unordered_map<int, std::unordered_map<uint64_t, reservation_entry>>       g_reservations;
std::unordered_map<int, std::unordered_map<uint64_t, moe_control_requirement>> g_planned_bytes;
std::atomic<uint64_t>                                                          g_conversion_id{ 1 };

// Sum of every reservation on `device_id` that has NOT converted, excluding
// one plan. Converted reservations are excluded because their bytes are
// already counted in the arena's runtime_used; counting them here would charge
// them twice. Saturates rather than wrapping.
size_t reserved_bytes_locked(int device_id, uint64_t excluded_plan_id) {
    auto device_it = g_reservations.find(device_id);
    if (device_it == g_reservations.end()) {
        return 0;
    }
    size_t total = 0;
    for (const auto & item : device_it->second) {
        if (item.first == excluded_plan_id || item.second.state == moe_control_reservation_state::ADMITTED_CONVERTED) {
            continue;
        }
        if (item.second.bytes > std::numeric_limits<size_t>::max() - total) {
            return std::numeric_limits<size_t>::max();
        }
        total += item.second.bytes;
    }
    return total;
}

// Drops a plan's reservation unless a conversion is in flight, in which case
// the in-flight transaction owns the entry until it commits or rolls back.
void erase_reservation_locked(int device_id, uint64_t plan_id) {
    if (plan_id == 0) {
        return;
    }
    auto device_it = g_reservations.find(device_id);
    if (device_it == g_reservations.end()) {
        return;
    }
    auto plan_it = device_it->second.find(plan_id);
    if (plan_it == device_it->second.end() ||
        plan_it->second.state != moe_control_reservation_state::CONVERSION_PENDING) {
        device_it->second.erase(plan_id);
    }
    if (device_it->second.empty()) {
        g_reservations.erase(device_it);
    }
}

}  // namespace

uint8_t moe_expert_logical_role_mask(moe_expert_role role) {
    switch (role) {
        case moe_expert_role::GATE:
            return k_role_bit_gate;
        case moe_expert_role::UP:
            return k_role_bit_up;
        case moe_expert_role::DOWN:
            return k_role_bit_down;
        case moe_expert_role::GATE_UP:
            return k_role_bit_gate | k_role_bit_up;
        case moe_expert_role::CHUNK_GATE:
            return k_role_bit_chunk_gate;
        case moe_expert_role::CHUNK_UP:
            return k_role_bit_chunk_up;
        case moe_expert_role::CHUNK_DOWN:
            return k_role_bit_chunk_down;
        case moe_expert_role::UNKNOWN:
        default:
            return 0;
    }
}

bool moe_logical_role_mask_complete(uint8_t mask) {
    // The primary trio must be whole. The grovemoe chunk trio is optional --
    // most models have none -- but a layer carrying part of it is missing a
    // tensor the dispatch needs, so partial is rejected rather than ignored.
    const uint8_t chunk = mask & k_role_mask_chunk;
    return (mask & k_role_mask_primary) == k_role_mask_primary && (chunk == 0 || chunk == k_role_mask_chunk);
}

moe_expert_identity moe_parse_expert_identity(const char * name) {
    moe_expert_identity identity;
    // grovemoe spells its chunked family "_chexps", which does NOT contain
    // "_exps" -- the underscore is before the "ch". Both spellings have to be
    // admitted here or the whole grovemoe family parses as "not an expert
    // tensor" and is silently dropped from the sizing.
    if (!name || (std::strstr(name, "_exps") == nullptr && std::strstr(name, "_chexps") == nullptr)) {
        return identity;
    }

    // Any expert-looking identity not accepted below is fatal rather than
    // ignored. Canonical identities start exactly at blk.<digits> and contain
    // no leading or embedded path.
    identity.candidate = moe_expert_candidate::MALFORMED;
    if (std::strncmp(name, "blk.", 4) != 0) {
        return identity;
    }
    const char * cursor = name + 4;
    if (*cursor < '0' || *cursor > '9') {
        return identity;
    }
    int layer = 0;
    do {
        const int digit = *cursor - '0';
        if (layer > (std::numeric_limits<int>::max() - digit) / 10) {
            return identity;
        }
        layer = layer * 10 + digit;
        ++cursor;
    } while (*cursor >= '0' && *cursor <= '9');
    if (*cursor != '.') {
        return identity;
    }

    struct accepted_suffix {
        const char *         suffix;
        moe_expert_candidate candidate;
        moe_expert_role      role;
    };

    static constexpr accepted_suffix accepted[] = {
        { ".ffn_gate_up_exps.weight",   moe_expert_candidate::MATRIX,  moe_expert_role::GATE_UP    },
        { ".ffn_gate_exps.weight",      moe_expert_candidate::MATRIX,  moe_expert_role::GATE       },
        { ".ffn_up_exps.weight",        moe_expert_candidate::MATRIX,  moe_expert_role::UP         },
        { ".ffn_down_exps.weight",      moe_expert_candidate::MATRIX,  moe_expert_role::DOWN       },
        { ".ffn_gate_chexps.weight",    moe_expert_candidate::MATRIX,  moe_expert_role::CHUNK_GATE },
        { ".ffn_up_chexps.weight",      moe_expert_candidate::MATRIX,  moe_expert_role::CHUNK_UP   },
        { ".ffn_down_chexps.weight",    moe_expert_candidate::MATRIX,  moe_expert_role::CHUNK_DOWN },
        { ".ffn_norm_exps.weight",      moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_up_exps.bias",     moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_up_exps.scale",    moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_up_exps.scales",   moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_exps.bias",        moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_exps.scale",       moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_exps.scales",      moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_exps.input_scale", moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_up_exps.bias",          moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_up_exps.scale",         moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_up_exps.scales",        moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_up_exps.input_scale",   moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_down_exps.bias",        moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_down_exps.scale",       moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_down_exps.scales",      moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_down_exps.input_scale", moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_chexps.bias",      moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_chexps.scale",     moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_gate_chexps.scales",    moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_up_chexps.bias",        moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_up_chexps.scale",       moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_up_chexps.scales",      moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_down_chexps.bias",      moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_down_chexps.scale",     moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
        { ".ffn_down_chexps.scales",    moe_expert_candidate::IGNORED, moe_expert_role::UNKNOWN    },
    };
    for (const accepted_suffix & item : accepted) {
        if (std::strcmp(cursor, item.suffix) == 0) {
            identity.candidate = item.candidate;
            identity.role      = item.role;
            identity.layer_id  = layer;
            return identity;
        }
    }
    return identity;
}

const moe_ptr_table_plan_entry * moe_ptr_table_plan::lookup(const std::string & name) const {
    const auto it = index_by_name.find(name);
    return it == index_by_name.end() ? nullptr : &entries[it->second];
}

bool moe_checked_pool_bytes(size_t count, size_t stride, size_t * out) {
    if (out == nullptr || (stride != 0 && count > std::numeric_limits<size_t>::max() / stride)) {
        return false;
    }
    *out = count * stride;
    return true;
}

moe_ptr_table_plan moe_build_ptr_table_plan(const std::vector<moe_control_tensor_desc> & inventory, int n_expert) {
    moe_ptr_table_plan plan;
    plan.n_expert = std::max(n_expert, 0);
    if (n_expert <= 0) {
        plan.structurally_complete = true;
        return plan;
    }

    std::unordered_map<std::string, moe_control_tensor_desc> unique;
    unique.reserve(inventory.size());
    for (const moe_control_tensor_desc & tensor : inventory) {
        if (tensor.name.find("_exps") == std::string::npos && tensor.name.find("_chexps") == std::string::npos) {
            plan.non_moe_ignored_count++;
            continue;
        }

        auto inserted = unique.emplace(tensor.name, tensor);
        if (!inserted.second) {
            // Two entries under one name are only benign when they describe
            // the same tensor. A disagreement means the inventory cannot say
            // what this tensor is, which is not something to average over.
            if (tensor_identity_equal(inserted.first->second, tensor)) {
                plan.duplicate_count++;
            } else {
                plan.conflicting_duplicate_count++;
            }
        }
    }

    // Parse only after every candidate has been deduplicated by full name and
    // metadata. This gives malformed identities the same exact/conflict
    // accounting as classified descriptors, and counts each identity once.
    for (const auto & item : unique) {
        const moe_expert_identity identity = moe_parse_expert_identity(item.first.c_str());
        if (identity.candidate == moe_expert_candidate::IGNORED) {
            plan.non_moe_ignored_count++;
            continue;
        }
        if (identity.candidate == moe_expert_candidate::MALFORMED) {
            plan.malformed_expert_candidate_count++;
            continue;
        }

        moe_ptr_table_plan_entry entry;
        entry.tensor_name = item.first;
        entry.layer_id    = identity.layer_id;
        entry.role        = identity.role;
        plan.entries.push_back(std::move(entry));
    }

    // Deterministic order, so table indices are reproducible across runs. The
    // name is the final tie-break precisely because the map above has none.
    std::sort(plan.entries.begin(), plan.entries.end(),
              [](const moe_ptr_table_plan_entry & a, const moe_ptr_table_plan_entry & b) {
                  if (a.layer_id != b.layer_id) {
                      return a.layer_id < b.layer_id;
                  }
                  const int a_role = materialization_role_order(a.role);
                  const int b_role = materialization_role_order(b.role);
                  if (a_role != b_role) {
                      return a_role < b_role;
                  }
                  return a.tensor_name < b.tensor_name;
              });

    // The alignment goes through checked_align rather than being open-coded:
    // moe_build_context_control_layout re-derives these same three values with
    // these same helpers and REFUSES the plan when they disagree, so a second
    // spelling of the rounding here would show up as a metadata mismatch on a
    // plan that is actually fine. Short-circuit leaves the later fields at 0,
    // which is what the overflow flag already means.
    if (!moe_checked_pool_bytes(static_cast<size_t>(plan.n_expert), sizeof(void *), &plan.table_bytes) ||
        !checked_align(plan.table_bytes, &plan.table_stride_bytes) ||
        !moe_checked_pool_bytes(plan.entries.size(), plan.table_stride_bytes, &plan.total_bytes)) {
        plan.arithmetic_overflow = true;
    }

    plan.index_by_name.reserve(plan.entries.size());
    for (size_t i = 0; i < plan.entries.size(); ++i) {
        moe_ptr_table_plan_entry & entry = plan.entries[i];
        entry.table_index                = i;
        entry.byte_size                  = plan.table_bytes;
        if (!plan.arithmetic_overflow) {
            // Bare multiply on purpose: entries.size() * table_stride_bytes was
            // just checked into total_bytes, and i < entries.size(), so this
            // product is strictly below a value already proven not to overflow.
            // Routing it through moe_checked_pool_bytes would add a branch that
            // provably cannot be taken. If the guard above is ever loosened,
            // this line stops being safe -- it depends on that check, not on
            // its own bounds.
            entry.byte_offset = i * plan.table_stride_bytes;
        }
        plan.index_by_name.emplace(entry.tensor_name, i);
    }

    plan.structurally_complete = plan.malformed_expert_candidate_count == 0 && plan.conflicting_duplicate_count == 0 &&
                                 !plan.arithmetic_overflow;
    return plan;
}

moe_context_control_layout moe_build_context_control_layout(const moe_ptr_table_plan & plan,
                                                            int                        n_expert_used,
                                                            uint32_t                   n_ubatch) {
    moe_context_control_layout layout;
    layout.n_expert      = plan.n_expert;
    layout.n_expert_used = n_expert_used;
    layout.n_ubatch      = n_ubatch;
    layout.table_count   = plan.entries.size();
    layout.tables_offset = 0;

    if (n_ubatch > MOE_GPU_UBATCH_MAX) {
        layout.exceeds_gpu_ubatch = true;
        return layout;
    }
    if (plan.arithmetic_overflow) {
        layout.arithmetic_overflow = true;
        return layout;
    }
    if (plan.n_expert < 0 || n_expert_used < 0) {
        layout.metadata_mismatch = true;
        return layout;
    }
    // Role COMPLETENESS is not checked here -- it is enforced per layer by the
    // layer_role_masks pass below, which is the only place that can see which
    // roles a layer actually accumulated.
    if (!plan.entries.empty() &&
        (plan.n_expert <= 0 || n_expert_used == 0 || n_expert_used > plan.n_expert || n_ubatch == 0 ||
         !plan.structurally_complete || plan.malformed_expert_candidate_count != 0 ||
         plan.conflicting_duplicate_count != 0 || plan.index_by_name.size() != plan.entries.size())) {
        layout.metadata_mismatch = true;
        return layout;
    }

    // Re-derive the geometry instead of reading the plan's. The plan is an
    // input from another pass; if it was built against different constants the
    // two disagree here and the layout is refused.
    size_t expected_table_bytes  = 0;
    size_t expected_table_stride = 0;
    if (!moe_checked_pool_bytes(static_cast<size_t>(plan.n_expert), sizeof(void *), &expected_table_bytes) ||
        !checked_align(expected_table_bytes, &expected_table_stride) ||
        !moe_checked_pool_bytes(layout.table_count, expected_table_stride, &layout.tables_bytes)) {
        layout.arithmetic_overflow = true;
        return layout;
    }
    layout.table_bytes  = expected_table_bytes;
    layout.table_stride = expected_table_stride;

    const bool invalid_geometry = plan.table_bytes != expected_table_bytes ||
                                  plan.table_stride_bytes != expected_table_stride ||
                                  plan.total_bytes != layout.tables_bytes;
    if (invalid_geometry) {
        if (plan.total_bytes > std::numeric_limits<size_t>::max() - (k_control_alignment - 1)) {
            layout.arithmetic_overflow = true;
        } else {
            layout.metadata_mismatch = true;
        }
        return layout;
    }

    // Per-entry audit: the entry's own index/offset/size must be what this
    // geometry says, its name must still parse to the role it claims, and no
    // layer may claim a logical role twice. The last of those is what catches
    // a fused gate_up sitting alongside a split gate in the same layer.
    std::map<int, uint8_t> layer_role_masks;
    for (size_t i = 0; i < plan.entries.size(); ++i) {
        size_t expected_offset = 0;
        if (!moe_checked_pool_bytes(i, expected_table_stride, &expected_offset)) {
            layout.arithmetic_overflow = true;
            return layout;
        }
        const moe_ptr_table_plan_entry & entry           = plan.entries[i];
        const auto                       index_it        = plan.index_by_name.find(entry.tensor_name);
        const uint8_t                    role_mask       = moe_expert_logical_role_mask(entry.role);
        const moe_expert_identity        parsed_identity = moe_parse_expert_identity(entry.tensor_name.c_str());
        if (entry.table_index != i || entry.byte_offset != expected_offset || entry.byte_size != expected_table_bytes ||
            entry.layer_id < 0 || entry.role == moe_expert_role::UNKNOWN || role_mask == 0 ||
            parsed_identity.candidate != moe_expert_candidate::MATRIX || parsed_identity.layer_id != entry.layer_id ||
            parsed_identity.role != entry.role || index_it == plan.index_by_name.end() || index_it->second != i ||
            (layer_role_masks[entry.layer_id] & role_mask) != 0) {
            layout.metadata_mismatch = true;
            return layout;
        }
        layer_role_masks[entry.layer_id] |= role_mask;
    }
    for (const auto & layer : layer_role_masks) {
        if (!moe_logical_role_mask_complete(layer.second)) {
            layout.metadata_mismatch = true;
            return layout;
        }
    }

    // A dense model is a valid layout of zero bytes, and deliberately NOT
    // gpu_eligible: there is nothing to route.
    if (plan.entries.empty()) {
        layout.valid = true;
        return layout;
    }

    size_t routed_slots = 0;
    if (!moe_checked_pool_bytes(static_cast<size_t>(n_expert_used), static_cast<size_t>(n_ubatch), &routed_slots) ||
        !moe_checked_pool_bytes(routed_slots, sizeof(int32_t), &layout.ids_bytes) ||
        !moe_checked_pool_bytes(routed_slots, sizeof(void *), &layout.compact_bytes)) {
        layout.arithmetic_overflow = true;
        return layout;
    }
    layout.missing_bytes = sizeof(uint32_t);

    size_t cursor = 0;
    if (!checked_add(layout.tables_offset, layout.tables_bytes, &cursor) ||
        !checked_align(cursor, &layout.ids_offset) || !checked_add(layout.ids_offset, layout.ids_bytes, &cursor) ||
        !checked_align(cursor, &layout.compact_offset) ||
        !checked_add(layout.compact_offset, layout.compact_bytes, &cursor) ||
        !checked_align(cursor, &layout.missing_offset) ||
        !checked_add(layout.missing_offset, layout.missing_bytes, &cursor) ||
        !checked_align(cursor, &layout.total_bytes)) {
        layout.arithmetic_overflow = true;
        return layout;
    }
    layout.valid        = true;
    layout.gpu_eligible = true;
    return layout;
}

moe_control_requirement moe_control_requirement_from_layout(const moe_context_control_layout & layout, bool required) {
    if (!required) {
        // Nothing on this device will route on the GPU, so it needs no CONTROL
        // pool whatever the layout says.
        return {};
    }
    // `valid` is the axis that matters, NOT `gpu_eligible`: a valid layout with
    // nothing to route is already total_bytes == 0, so it reports a real zero,
    // while a layout that could not be computed reports invalid and refuses to
    // be summed. Keying on gpu_eligible instead would make every dense model an
    // unsizable one.
    moe_control_requirement requirement;
    requirement.bytes = layout.total_bytes;
    requirement.valid = layout.valid;
    return requirement;
}

bool moe_checked_runtime_zone_requirement(size_t                          pp_pipeline_bytes,
                                          size_t                          pp_moe_onednn_bytes,
                                          const moe_control_requirement & base_context_control,
                                          size_t *                        out) {
    if (out == nullptr || !base_context_control.valid) {
        return false;
    }
    size_t total = 0;
    if (!checked_add(pp_pipeline_bytes, pp_moe_onednn_bytes, &total) ||
        !checked_add(total, base_context_control.bytes, &total)) {
        return false;
    }
    *out = total;
    return true;
}

moe_context_control_device_plan moe_plan_context_control(int                             device_id,
                                                         size_t                          runtime_available,
                                                         const moe_control_requirement & requirement,
                                                         bool                            arena_active,
                                                         bool                            route_capable) {
    moe_context_control_device_plan result;
    result.device_id       = device_id;
    result.required_bytes  = requirement.bytes;
    result.available_bytes = runtime_available;
    if (!requirement.valid) {
        result.reason = moe_control_reason::INVALID_LAYOUT;
    } else if (requirement.bytes == 0) {
        result.admitted = true;
        result.reason   = moe_control_reason::EMPTY_INVENTORY;
    } else if (!route_capable) {
        result.reason = moe_control_reason::NO_ROUTE_CAPABILITY;
    } else if (!arena_active || requirement.bytes > runtime_available) {
        result.reason = moe_control_reason::RUNTIME_CAPACITY;
    } else {
        result.admitted = true;
        result.reason   = moe_control_reason::ADMITTED;
    }
    return result;
}

moe_context_control_device_plan moe_reserve_context_control(uint64_t                              plan_id,
                                                            int                                   device_id,
                                                            const moe_control_requirement &       requirement,
                                                            const moe_runtime_capacity_snapshot & capacity,
                                                            bool                                  route_capable) {
    if (!requirement.valid) {
        // Invalid sizing is not an empty/dense transition. Fail without
        // touching this or any overlapping plan's live reservation.
        auto result    = moe_plan_context_control(device_id, 0, requirement, false, route_capable);
        result.plan_id = plan_id;
        return result;
    }
    if (requirement.bytes > 0 && plan_id == 0) {
        auto result     = moe_plan_context_control(device_id, 0, requirement, false, route_capable);
        result.plan_id  = plan_id;
        result.admitted = false;
        result.reason   = moe_control_reason::INVALID_PLAN_ID;
        return result;
    }
    if (requirement.bytes == 0 || !route_capable) {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto                        result = moe_plan_context_control(device_id, 0, requirement, false, route_capable);
        result.plan_id                     = plan_id;
        erase_reservation_locked(device_id, plan_id);
        return result;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto device_it = g_reservations.find(device_id);
    if (device_it != g_reservations.end()) {
        auto existing = device_it->second.find(plan_id);
        if (existing != device_it->second.end() &&
            existing->second.state == moe_control_reservation_state::CONVERSION_PENDING) {
            // Someone is converting this exact reservation. Join it by taking
            // a reference rather than re-admitting bytes it already holds.
            moe_context_control_device_plan result;
            result.plan_id         = plan_id;
            result.device_id       = device_id;
            result.required_bytes  = requirement.bytes;
            result.available_bytes = existing->second.bytes;
            if (existing->second.references == 0 || existing->second.bytes != requirement.bytes ||
                existing->second.references == std::numeric_limits<size_t>::max()) {
                result.reason = moe_control_reason::PENDING_CONVERSION;
                return result;
            }
            existing->second.references++;
            result.admitted = true;
            result.reason   = moe_control_reason::ADMITTED;
            return result;
        }
    }

    const size_t other_reserved = reserved_bytes_locked(device_id, plan_id);

    size_t available = 0;
    if (capacity.arena_active && capacity.runtime_available && capacity.runtime_used <= capacity.runtime_capacity) {
        available = capacity.runtime_capacity - capacity.runtime_used;
        available = available > other_reserved ? available - other_reserved : 0;
    }
    auto result    = moe_plan_context_control(device_id, available, requirement,
                                              capacity.arena_active && capacity.runtime_available, route_capable);
    result.plan_id = plan_id;
    if (!result.admitted || requirement.bytes == 0) {
        erase_reservation_locked(device_id, plan_id);
        return result;
    }
    if (plan_id == 0) {
        return result;
    }

    auto & reservations = g_reservations[device_id];
    auto   existing     = reservations.find(plan_id);
    if (existing == reservations.end()) {
        reservation_entry entry;
        entry.bytes = requirement.bytes;
        reservations.emplace(plan_id, entry);
    } else if (existing->second.state == moe_control_reservation_state::UNCONSUMED) {
        existing->second.bytes = requirement.bytes;
    } else if (existing->second.bytes != requirement.bytes) {
        // Already converted at a different size. Re-admitting would let a
        // second context believe it owns bytes the first one allocated.
        result.admitted = false;
        result.reason   = moe_control_reason::RUNTIME_CAPACITY;
    }
    return result;
}

bool moe_retain_context_control_reservation(uint64_t plan_id, int device_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto                        device_it = g_reservations.find(device_id);
    if (device_it == g_reservations.end()) {
        return false;
    }
    auto plan_it = device_it->second.find(plan_id);
    if (plan_it == device_it->second.end() || plan_it->second.references == 0 ||
        plan_it->second.references == std::numeric_limits<size_t>::max()) {
        return false;
    }
    plan_it->second.references++;
    return true;
}

bool moe_retain_context_control_reservations(uint64_t plan_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    // Two passes: a partial retain would leave some devices holding a
    // reference the caller believes it does not have, and there is no
    // unwinding step that could tell them apart afterwards.
    bool                        found = false;
    for (const auto & device : g_reservations) {
        auto plan_it = device.second.find(plan_id);
        if (plan_it != device.second.end()) {
            if (plan_it->second.references == 0 || plan_it->second.references == std::numeric_limits<size_t>::max()) {
                return false;
            }
            found = true;
        }
    }
    if (!found) {
        return false;
    }
    for (auto & device : g_reservations) {
        auto plan_it = device.second.find(plan_id);
        if (plan_it != device.second.end()) {
            plan_it->second.references++;
        }
    }
    return true;
}

void moe_release_context_control_reservations(uint64_t plan_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto device_it = g_reservations.begin(); device_it != g_reservations.end();) {
        auto plan_it = device_it->second.find(plan_id);
        if (plan_it != device_it->second.end()) {
            if (plan_it->second.references > 1) {
                plan_it->second.references--;
            } else if (plan_it->second.state == moe_control_reservation_state::CONVERSION_PENDING) {
                // The in-flight transaction owns the entry until its nofail
                // commit or rollback. Zero references makes publication fail,
                // and rollback erases the entry it finds unreferenced.
                plan_it->second.references = 0;
            } else {
                device_it->second.erase(plan_it);
            }
        }
        if (device_it->second.empty()) {
            device_it = g_reservations.erase(device_it);
        } else {
            ++device_it;
        }
    }
}

moe_control_conversion_ticket moe_begin_control_conversion(uint64_t plan_id, int device_id, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto                        device_it = g_reservations.find(device_id);
    if (plan_id == 0 || device_it == g_reservations.end()) {
        return {};
    }
    auto plan_it = device_it->second.find(plan_id);
    if (plan_it == device_it->second.end() || plan_it->second.references == 0) {
        return {};
    }

    auto & entry = plan_it->second;
    if (entry.state == moe_control_reservation_state::ADMITTED_CONVERTED) {
        // Already converted: a valid ticket that owns no transaction, so its
        // guard must not roll anything back.
        return { plan_id, device_id, 0, true, false };
    }
    if (entry.state != moe_control_reservation_state::UNCONSUMED || bytes > entry.bytes) {
        return {};
    }

    uint64_t conversion_id = g_conversion_id.fetch_add(1, std::memory_order_relaxed);
    if (conversion_id == 0) {
        conversion_id = g_conversion_id.fetch_add(1, std::memory_order_relaxed);
    }
    entry.state              = moe_control_reservation_state::CONVERSION_PENDING;
    entry.conversion_id      = conversion_id;
    entry.allocation_claimed = false;
    return { plan_id, device_id, conversion_id, true, true };
}

moe_control_allocation_grant moe_authorize_control_allocation(uint64_t                              plan_id,
                                                              int                                   device_id,
                                                              uint64_t                              conversion_id,
                                                              size_t                                bytes,
                                                              const moe_runtime_capacity_snapshot & capacity) {
    moe_control_allocation_grant grant;
    std::lock_guard<std::mutex>  lock(g_mutex);

    auto device_it = g_reservations.find(device_id);
    if (plan_id == 0 || device_it == g_reservations.end()) {
        return grant;
    }
    auto plan_it = device_it->second.find(plan_id);
    if (plan_it == device_it->second.end() || plan_it->second.references == 0) {
        return grant;
    }

    auto & entry = plan_it->second;
    if (conversion_id != 0) {
        if (entry.state != moe_control_reservation_state::CONVERSION_PENDING || entry.conversion_id != conversion_id ||
            entry.allocation_claimed || bytes > entry.bytes) {
            return grant;
        }
        grant.first_conversion = true;
    } else if (entry.state != moe_control_reservation_state::ADMITTED_CONVERTED) {
        return grant;
    }

    // A first conversion is spending its OWN reservation, so that reservation
    // is excluded from the competing total; any later allocation is spending
    // genuinely unreserved capacity and competes with everything.
    const size_t reserved = reserved_bytes_locked(device_id, grant.first_conversion ? plan_id : uint64_t{ 0 });
    if (!capacity.arena_active || !capacity.runtime_available || capacity.runtime_used > capacity.runtime_capacity ||
        reserved > capacity.runtime_capacity - capacity.runtime_used ||
        bytes > capacity.runtime_capacity - capacity.runtime_used - reserved) {
        grant.first_conversion = false;
        return grant;
    }

    if (grant.first_conversion) {
        entry.allocation_claimed = true;
    }
    grant.granted            = true;
    grant.runtime_used_after = capacity.runtime_used + bytes;
    return grant;
}

bool moe_commit_control_conversion(const moe_control_conversion_ticket & ticket) {
    if (!ticket.valid) {
        return false;
    }
    if (!ticket.first) {
        return true;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto                        device_it = g_reservations.find(ticket.device_id);
    if (device_it == g_reservations.end()) {
        return false;
    }
    auto plan_it = device_it->second.find(ticket.plan_id);
    if (plan_it == device_it->second.end() ||
        plan_it->second.state != moe_control_reservation_state::CONVERSION_PENDING ||
        plan_it->second.conversion_id != ticket.conversion_id || !plan_it->second.allocation_claimed ||
        plan_it->second.references == 0) {
        return false;
    }
    auto & entry             = plan_it->second;
    entry.state              = moe_control_reservation_state::ADMITTED_CONVERTED;
    entry.conversion_id      = 0;
    entry.allocation_claimed = false;
    return true;
}

void moe_rollback_control_conversion(const moe_control_conversion_ticket & ticket) noexcept {
    if (!ticket.valid || !ticket.first) {
        return;
    }
    // noexcept is load-bearing rather than decorative: this runs from a
    // destructor on the failure path, and the only way it can fail is the lock
    // itself. Terminating beats returning with the entry stuck in
    // CONVERSION_PENDING, which would refuse every later reservation for this
    // plan with no way back.
    std::lock_guard<std::mutex> lock(g_mutex);
    auto                        device_it = g_reservations.find(ticket.device_id);
    if (device_it == g_reservations.end()) {
        return;
    }
    auto plan_it = device_it->second.find(ticket.plan_id);
    if (plan_it == device_it->second.end() ||
        plan_it->second.state != moe_control_reservation_state::CONVERSION_PENDING ||
        plan_it->second.conversion_id != ticket.conversion_id) {
        return;
    }
    if (plan_it->second.references == 0) {
        // The last holder released while this conversion was in flight and
        // deliberately left the entry for us to clean up.
        device_it->second.erase(plan_it);
        if (device_it->second.empty()) {
            g_reservations.erase(device_it);
        }
        return;
    }
    plan_it->second.state              = moe_control_reservation_state::UNCONSUMED;
    plan_it->second.conversion_id      = 0;
    plan_it->second.allocation_claimed = false;
}

void moe_publish_planned_control_bytes(uint64_t plan_id, int device_id, const moe_control_requirement & requirement) {
    if (plan_id == 0 || device_id < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (requirement.valid && requirement.bytes == 0) {
        auto device_it = g_planned_bytes.find(device_id);
        if (device_it != g_planned_bytes.end()) {
            device_it->second.erase(plan_id);
            if (device_it->second.empty()) {
                g_planned_bytes.erase(device_it);
            }
        }
        return;
    }
    g_planned_bytes[device_id][plan_id] = requirement;
}

void moe_reset_planned_control_bytes(uint64_t plan_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto device_it = g_planned_bytes.begin(); device_it != g_planned_bytes.end();) {
        device_it->second.erase(plan_id);
        if (device_it->second.empty()) {
            device_it = g_planned_bytes.erase(device_it);
        } else {
            ++device_it;
        }
    }
}

moe_control_requirement moe_planned_control_bytes_for_device(int device_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto                        device_it = g_planned_bytes.find(device_id);
    if (device_it == g_planned_bytes.end()) {
        return {};
    }
    moe_control_requirement total;
    for (const auto & item : device_it->second) {
        if (!item.second.valid || !checked_add(total.bytes, item.second.bytes, &total.bytes)) {
            total.valid = false;
            total.bytes = 0;
            return total;
        }
    }
    return total;
}

moe_control_requirement moe_planned_control_bytes_for_plan(uint64_t plan_id, int device_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto                        device_it = g_planned_bytes.find(device_id);
    if (device_it == g_planned_bytes.end()) {
        return {};
    }
    auto plan_it = device_it->second.find(plan_id);
    return plan_it == device_it->second.end() ? moe_control_requirement{} : plan_it->second;
}

moe_control_reservation_snapshot moe_control_reservation_state_for(int device_id, uint64_t plan_id) {
    std::lock_guard<std::mutex>      lock(g_mutex);
    moe_control_reservation_snapshot snapshot;
    auto                             device_it = g_reservations.find(device_id);
    if (device_it == g_reservations.end()) {
        return snapshot;
    }
    snapshot.plan_count = device_it->second.size();
    for (const auto & item : device_it->second) {
        const bool unconsumed = item.second.state != moe_control_reservation_state::ADMITTED_CONVERTED;
        if (unconsumed && item.second.bytes > std::numeric_limits<size_t>::max() - snapshot.total_unconsumed_bytes) {
            snapshot.total_unconsumed_bytes = std::numeric_limits<size_t>::max();
        } else if (unconsumed) {
            snapshot.total_unconsumed_bytes += item.second.bytes;
        }
        if (item.first == plan_id) {
            snapshot.plan_unconsumed_bytes = unconsumed ? item.second.bytes : 0;
            snapshot.plan_reference_count  = item.second.references;
            snapshot.plan_state            = item.second.state;
        }
    }
    return snapshot;
}

void moe_control_reset_all_state() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_reservations.clear();
    g_planned_bytes.clear();
}

}  // namespace ggml_sycl
