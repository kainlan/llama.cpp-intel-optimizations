//
// Structural path-scoped arena zone sizing implementation.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "zone-sizing.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>

// This TU deliberately depends on nothing: that is what lets the host-only
// test target link it without libggml-sycl and build in seconds. The
// diagnostics below still have to reach the backend's log in a production
// build, so the log macro is switched here rather than the header being
// included unconditionally — the same split gpu-arch.cpp uses for
// GGML_SYCL_GPU_ARCH_STANDALONE, and for the same reason.
#if defined(ZONE_SIZING_STANDALONE)
// The stub must still COMPILE its arguments, not discard them: a macro that
// swallowed __VA_ARGS__ would let a bad format specifier survive the test build
// and only break the library build. Guarding a real fprintf with `if (false)`
// keeps -Wformat checking while the optimizer drops the call, and keeps the
// unit test's output clean.
#    define ZONE_SIZING_LOG_WARN(...)              \
        do {                                       \
            if (false) {                           \
                std::fprintf(stderr, __VA_ARGS__); \
            }                                      \
        } while (0)
#else
#    include "ggml-impl.h"
#    define ZONE_SIZING_LOG_WARN(...) GGML_LOG_WARN(__VA_ARGS__)
#endif

namespace ggml_sycl {

namespace {

// Grouping key. Two tensors belong to the same family when they agree on type
// and on all four dimensions — the structural signature of a weight that is
// instantiated once per block. No naming convention is consulted.
struct zone_group_key {
    int     type  = -1;
    int64_t ne[4] = {};

    bool operator<(const zone_group_key & other) const {
        if (type != other.type) {
            return type < other.type;
        }
        for (int i = 0; i < 4; i++) {
            if (ne[i] != other.ne[i]) {
                return ne[i] < other.ne[i];
            }
        }
        return false;
    }
};

zone_group_key group_key(const zone_tensor_desc & tensor) {
    zone_group_key key;
    key.type = tensor.type;
    for (int i = 0; i < 4; i++) {
        key.ne[i] = tensor.ne[i];
    }
    return key;
}

// Cardinality of every (type, ne) group in the inventory.
//
// Shapeless entries are skipped outright rather than grouped. Their ne is
// {0,0,0,0}, so every shapeless entry of a given type would key identically
// and form one large spurious family that clears the threshold and drags each
// path-scoped maximum back up. Both reference models reported a valid shape
// for all 459 / 291 entries, but that is an observation about two models.
std::map<zone_group_key, size_t> zone_group_frequencies(const std::vector<zone_tensor_desc> & inventory) {
    std::map<zone_group_key, size_t> freq;
    for (size_t i = 0; i < inventory.size(); i++) {
        if (!inventory[i].has_shape) {
            continue;
        }
        freq[group_key(inventory[i])]++;
    }
    return freq;
}

size_t group_cardinality_of(const std::map<zone_group_key, size_t> & freq, const zone_tensor_desc & tensor) {
    if (!tensor.has_shape) {
        return 0;
    }
    const std::map<zone_group_key, size_t>::const_iterator it = freq.find(group_key(tensor));
    return it == freq.end() ? 0 : it->second;
}

}  // namespace

bool zone_is_per_layer_weight(const zone_tensor_desc & tensor, size_t group_cardinality) {
    // A shapeless entry is explicitly NOT a per-layer weight: without a shape
    // there is no structural evidence of repetition, and the zeros must not be
    // allowed to vote.
    if (!tensor.has_shape) {
        return false;
    }
    return group_cardinality >= k_zone_per_layer_min_group;
}

// KNOWN, INSTRUMENTED RISK — do not "fix" here.
//
// This rule classifies the LM head as NOT a per-layer weight and so excludes
// it from onednn_eligible. But the LM head IS consumed by MUL_MAT and may be a
// genuine oneDNN reorder subject — unlike the token embedding, which is a
// GET_ROWS lookup and legitimately never reaches that path. If the LM head
// does reach oneDNN, this under-estimates.
//
// That is survivable by construction: the zone grows on demand and the
// underestimate is counted, so it surfaces as a loud warning rather than a
// crash or a silent slowdown, and end-to-end validation is the experiment that
// settles it. Do not pre-emptively widen the predicate, and above all do not
// add a name check to special-case the LM head — a name predicate is exactly
// what this unit exists to replace.
bool zone_is_moe_expert_tensor(const zone_tensor_desc & tensor) {
    // Without a shape there is no evidence either way, and the zeros must not
    // be allowed to vote — same rule as zone_is_per_layer_weight. Returning
    // false here means a shapeless entry is not excluded on THIS ground; it is
    // already excluded by the per-layer predicate below.
    if (!tensor.has_shape) {
        return false;
    }
    return tensor.ne[2] > 1 || tensor.ne[3] > 1;
}

bool zone_is_onednn_reorder_eligible(const zone_tensor_desc & tensor, size_t group_cardinality) {
    // See the header: expert weights go through the PP-MoE oneDNN ring, and
    // reserve_onednn_scratch is measurably unreachable on a MoE model.
    if (zone_is_moe_expert_tensor(tensor)) {
        return false;
    }
    return zone_is_per_layer_weight(tensor, group_cardinality);
}

bool zone_is_cpu_quant_eligible(const zone_tensor_desc & tensor, size_t group_cardinality) {
    return zone_is_per_layer_weight(tensor, group_cardinality);
}

bool zone_is_dma_streamed(const zone_tensor_desc & tensor, size_t group_cardinality) {
    return zone_is_per_layer_weight(tensor, group_cardinality);
}

path_scoped_maxima zone_scoped_maxima(const std::vector<zone_tensor_desc> & inventory) {
    const std::map<zone_group_key, size_t> freq = zone_group_frequencies(inventory);

    path_scoped_maxima maxima;
    for (size_t i = 0; i < inventory.size(); i++) {
        const zone_tensor_desc & tensor = inventory[i];

        // `size` is the authoritative magnitude. Never derive one from ne:
        // expert tensors are 3-D (ne[2] = 32 experts) and ne[0] * ne[1]
        // understates them by 32x.
        maxima.any_tensor = std::max(maxima.any_tensor, tensor.size);

        const size_t cardinality = group_cardinality_of(freq, tensor);

        const bool onednn_eligible = zone_is_onednn_reorder_eligible(tensor, cardinality);
        if (onednn_eligible) {
            maxima.onednn_eligible = std::max(maxima.onednn_eligible, tensor.size);
            // Maxed independently of onednn_eligible, over the SAME eligible
            // set. The two winners need not be the same tensor: expansion is
            // per type, so the largest stored weight and the largest
            // dequantized reorder buffer diverge on a mixed-quantization model.
            maxima.onednn_reorder  = std::max(maxima.onednn_reorder, tensor.reorder_size);
        }
        if (zone_is_cpu_quant_eligible(tensor, cardinality)) {
            maxima.cpu_quant_eligible = std::max(maxima.cpu_quant_eligible, tensor.size);
        }
        if (zone_is_dma_streamed(tensor, cardinality)) {
            maxima.dma_streamed = std::max(maxima.dma_streamed, tensor.size);
        }
        // A tensor the loader says is only ever gathered (GET_ROWS) is no MUL_MAT operand whatever marks the adapter
        // set on it; the three marks below are the MUL_MAT-side plans (llama.cpp-8ony).
        if (tensor.get_rows_only) {
            continue;
        }
        // Not gated on a per-layer family or on the expert predicate: the adapter
        // already said whether this is a dense MUL_MAT operand (non-zero).
        maxima.mmq_src1_bytes_per_token = std::max(maxima.mmq_src1_bytes_per_token, tensor.mmq_src1_bytes_per_token);
        maxima.dequant_f16_weight_bytes = std::max(maxima.dequant_f16_weight_bytes, tensor.dequant_f16_weight_bytes);
        maxima.dequant_f16_src1_bytes_per_token =
            std::max(maxima.dequant_f16_src1_bytes_per_token, tensor.dequant_f16_src1_bytes_per_token);
        if (tensor.dequant_f16_if_unsupplied_weight_bytes != 0 &&
            zone_dequant_f16_planned_when_unsupplied(tensor.pp_scratch_type_enabled, onednn_eligible)) {
            maxima.dequant_f16_weight_bytes =
                std::max(maxima.dequant_f16_weight_bytes, tensor.dequant_f16_if_unsupplied_weight_bytes);
            maxima.dequant_f16_src1_bytes_per_token = std::max(maxima.dequant_f16_src1_bytes_per_token,
                                                               tensor.dequant_f16_if_unsupplied_src1_bytes_per_token);
        }
    }
    return maxima;
}

zone_onednn_plan zone_onednn_plan_keep(const zone_onednn_plan & held, const zone_onednn_plan & live) {
    // The pair plan is one thing: its halves sum into its bare plan. Take the whole pair of the plan with the larger
    // bare plan (the held one on a tie), never the maximum of each half, which would build a pair no plan had. The
    // Graph floor is a separate figure the zone also has to hold, so it keeps its own maximum.
    zone_onednn_plan kept  = live.bare_bytes > held.bare_bytes ? live : held;
    kept.graph_floor_bytes = std::max(held.graph_floor_bytes, live.graph_floor_bytes);
    return kept;
}

bool zone_dequant_f16_planned_when_unsupplied(bool pp_scratch_type_enabled, bool pair_eligible) {
    return !(pp_scratch_type_enabled && pair_eligible);
}

bool zone_mmq_src1_row_bytes(int64_t ne10, size_t * out) {
    if (!out || ne10 <= 0) {
        return false;
    }
    const int64_t pad    = k_zone_mmq_src1_row_padding;
    const int64_t padded = ne10 > INT64_MAX - (pad - 1) ? 0 : (ne10 + pad - 1) / pad * pad;
    if (padded <= 0) {
        return false;
    }
    const size_t blocks = static_cast<size_t>(padded / k_zone_mmq_src1_block_elems);
    if (blocks > SIZE_MAX / k_zone_mmq_src1_block_bytes) {
        return false;
    }
    *out = blocks * k_zone_mmq_src1_block_bytes;
    return true;
}

bool zone_mmq_src1_required_bytes(int64_t nrows, int64_t ne10, bool with_overflow_pad, size_t * out) {
    size_t row = 0;
    if (!out || nrows <= 0 || !zone_mmq_src1_row_bytes(ne10, &row)) {
        return false;
    }
    if (static_cast<size_t>(nrows) > (SIZE_MAX - k_zone_mmq_src1_overflow_pad) / row) {
        return false;
    }
    *out = static_cast<size_t>(nrows) * row + (with_overflow_pad ? k_zone_mmq_src1_overflow_pad : 0);
    return true;
}

bool zone_mmq_src1_bytes_per_token(int64_t ne0, int64_t ne2, int64_t ne3, size_t * out) {
    size_t row = 0;
    if (!out || ne2 <= 0 || ne3 <= 0 || !zone_mmq_src1_row_bytes(ne0, &row)) {
        return false;
    }
    const size_t batch = static_cast<size_t>(ne2);
    if (batch > SIZE_MAX / row || static_cast<size_t>(ne3) > SIZE_MAX / (row * batch)) {
        return false;
    }
    *out = row * batch * static_cast<size_t>(ne3);
    return true;
}

bool zone_mmq_src1_scratch_bytes(size_t bytes_per_token, uint32_t n_ubatch, size_t * out) {
    if (!out) {
        return false;
    }
    if (bytes_per_token == 0 || n_ubatch == 0) {
        *out = 0;
        return true;
    }
    const size_t slack = k_zone_mmq_src1_overflow_pad + (k_zone_mmq_src1_align - 1);
    if (bytes_per_token > (SIZE_MAX - slack) / n_ubatch) {
        return false;
    }
    const size_t raw = bytes_per_token * n_ubatch + k_zone_mmq_src1_overflow_pad;
    *out             = (raw + k_zone_mmq_src1_align - 1) / k_zone_mmq_src1_align * k_zone_mmq_src1_align;
    return true;
}

bool zone_dequant_f16_weight_bytes(int64_t ne0, int64_t ne1, size_t * out) {
    if (!out || ne0 <= 0 || ne1 <= 0) {
        return false;
    }
    const size_t cols = static_cast<size_t>(ne0);
    const size_t rows = static_cast<size_t>(ne1);
    if (cols > SIZE_MAX / rows || cols * rows > SIZE_MAX / k_zone_dequant_f16_elem_bytes) {
        return false;
    }
    *out = cols * rows * k_zone_dequant_f16_elem_bytes;
    return true;
}

bool zone_dequant_f16_src1_bytes_per_token(int64_t ne0, int64_t ne2, int64_t ne3, size_t * out) {
    if (!out || ne0 <= 0 || ne2 <= 0 || ne3 <= 0) {
        return false;
    }
    const size_t row   = static_cast<size_t>(ne0);
    const size_t batch = static_cast<size_t>(ne2);
    if (row > SIZE_MAX / batch || row * batch > SIZE_MAX / static_cast<size_t>(ne3) ||
        row * batch * static_cast<size_t>(ne3) > SIZE_MAX / k_zone_dequant_f16_elem_bytes) {
        return false;
    }
    *out = row * batch * static_cast<size_t>(ne3) * k_zone_dequant_f16_elem_bytes;
    return true;
}

bool zone_dequant_f16_region_bytes(int64_t elems, size_t * out) {
    if (!out || elems < 0) {
        return false;
    }
    const size_t n = static_cast<size_t>(elems);
    if (n > (SIZE_MAX - (k_zone_dequant_f16_align - 1)) / k_zone_dequant_f16_elem_bytes) {
        return false;
    }
    const size_t raw = n * k_zone_dequant_f16_elem_bytes;
    *out             = (raw + k_zone_dequant_f16_align - 1) / k_zone_dequant_f16_align * k_zone_dequant_f16_align;
    return true;
}

bool zone_dequant_f16_plan_bytes(size_t   max_weight_bytes,
                                 size_t   src1_bytes_per_token,
                                 uint32_t n_ubatch,
                                 size_t * src0_bytes,
                                 size_t * src1_bytes) {
    if (!src0_bytes || !src1_bytes) {
        return false;
    }
    const size_t align = k_zone_dequant_f16_align;
    if (max_weight_bytes > SIZE_MAX - (align - 1)) {
        return false;
    }
    if (src1_bytes_per_token != 0 && n_ubatch != 0 && src1_bytes_per_token > (SIZE_MAX - (align - 1)) / n_ubatch) {
        return false;
    }
    const size_t acts_raw = src1_bytes_per_token * n_ubatch;
    *src0_bytes           = (max_weight_bytes + align - 1) / align * align;
    *src1_bytes           = (acts_raw + align - 1) / align * align;
    return true;
}

bool zone_onednn_pp_scratch_planned(bool   arena_active,
                                    size_t pair_bound_bytes,
                                    size_t weights_bytes,
                                    size_t activations_bytes) {
    if (!arena_active) {
        return true;
    }
    if (weights_bytes > SIZE_MAX - activations_bytes) {
        return false;
    }
    return weights_bytes + activations_bytes <= pair_bound_bytes;
}

void zone_onednn_scratch_reserve_target(bool     arena_active,
                                        size_t   pair_bound_bytes,
                                        size_t   held_weights_bytes,
                                        size_t   held_activations_bytes,
                                        size_t   planned_weights_bytes,
                                        size_t   planned_activations_bytes,
                                        size_t   requested_weights_bytes,
                                        size_t   requested_activations_bytes,
                                        size_t * weights_bytes,
                                        size_t * activations_bytes) {
    // A pair that cannot exist inside the bound (or whose sum is unrepresentable) is not a target.
    const auto fits = [&](size_t w, size_t a) {
        return !arena_active || (w <= SIZE_MAX - a && w + a <= pair_bound_bytes);
    };
    const size_t merged_weights     = std::max(held_weights_bytes, requested_weights_bytes);
    const size_t merged_activations = std::max(held_activations_bytes, requested_activations_bytes);
    // With an arena the planned pair is a floor: reserving it up front is what keeps the pair from ever regrowing.
    const size_t planned_w          = arena_active ? std::max(merged_weights, planned_weights_bytes) : merged_weights;
    const size_t planned_a =
        arena_active ? std::max(merged_activations, planned_activations_bytes) : merged_activations;
    size_t target_w = requested_weights_bytes;  // used as asked when nothing merged fits
    size_t target_a = requested_activations_bytes;
    if (fits(planned_w, planned_a)) {
        target_w = planned_w;
        target_a = planned_a;
    } else if (fits(merged_weights, merged_activations)) {
        target_w = merged_weights;
        target_a = merged_activations;
    }
    if (weights_bytes) {
        *weights_bytes = target_w;
    }
    if (activations_bytes) {
        *activations_bytes = target_a;
    }
}

size_t zone_onednn_pp_pair_bound(size_t capacity_bytes, size_t bare_plan_bytes, size_t graph_floor_bytes) {
    const size_t slack_bound = capacity_bytes > graph_floor_bytes ? capacity_bytes - graph_floor_bytes : 0;
    return std::min(capacity_bytes, std::max(bare_plan_bytes, slack_bound));
}

bool zone_onednn_pp_scratch_type_enabled(int env_mode, bool default_type) {
    return env_mode > 0 || (env_mode < 0 && default_type);
}

bool zone_onednn_pp_scratch_supplies(bool   pp_candidate,
                                     bool   type_enabled,
                                     bool   arena_active,
                                     size_t pair_bound_bytes,
                                     size_t weights_bytes,
                                     size_t activations_bytes) {
    return pp_candidate && type_enabled &&
           zone_onednn_pp_scratch_planned(arena_active, pair_bound_bytes, weights_bytes, activations_bytes);
}

bool zone_unified_pp_draws_dequant(bool primary_unified,
                                   bool unified_type,
                                   bool src1_plain,
                                   bool pp_candidate,
                                   bool scratch_supplies) {
    return primary_unified && unified_type && src1_plain && pp_candidate && !scratch_supplies;
}

bool zone_walk_f16_node_draws(bool prec_default, bool legacy_route_draws, bool unified_route_draws) {
    return (prec_default && legacy_route_draws) || unified_route_draws;
}

bool zone_dense_scratch_total_bytes(size_t   mmq_bytes_per_token,
                                    size_t   f16_weight_bytes,
                                    size_t   f16_src1_bytes_per_token,
                                    uint32_t n_ubatch,
                                    size_t * out) {
    if (!out) {
        return false;
    }
    size_t q8   = 0;
    size_t src0 = 0;
    size_t src1 = 0;
    if (!zone_mmq_src1_scratch_bytes(mmq_bytes_per_token, n_ubatch, &q8) ||
        !zone_dequant_f16_plan_bytes(f16_weight_bytes, f16_src1_bytes_per_token, n_ubatch, &src0, &src1)) {
        return false;
    }
    if (q8 > SIZE_MAX - src0 || q8 + src0 > SIZE_MAX - src1) {
        return false;
    }
    *out = q8 + src0 + src1;
    return true;
}

uint32_t zone_dense_scratch_largest_ubatch(size_t   mmq_bytes_per_token,
                                           size_t   f16_weight_bytes,
                                           size_t   f16_src1_bytes_per_token,
                                           size_t   other_runtime_bytes,
                                           size_t   capacity_bytes,
                                           uint32_t search_max) {
    constexpr uint32_t k_row_group = 32;
    if (other_runtime_bytes >= capacity_bytes) {
        return 0;
    }
    const size_t avail = capacity_bytes - other_runtime_bytes;
    uint32_t     ub    = search_max / k_row_group * k_row_group;
    // The total is monotonic in n_ubatch, so step down from the bound until it fits. A hopeless zone (the weight
    // copy alone is too big) walks the whole range, at most search_max / 32 cheap steps, once per transaction.
    for (; ub >= k_row_group; ub -= k_row_group) {
        size_t total = 0;
        if (zone_dense_scratch_total_bytes(mmq_bytes_per_token, f16_weight_bytes, f16_src1_bytes_per_token, ub,
                                           &total) &&
            total <= avail) {
            return ub;
        }
    }
    return 0;
}

bool zone_planned_scratch_hold_bytes(const zone_planned_buffer * buffers, size_t count, size_t * out) {
    if (!out || (count != 0 && !buffers)) {
        return false;
    }
    size_t hold = 0;
    for (size_t i = 0; i < count; ++i) {
        if (buffers[i].capacity >= buffers[i].plan) {
            continue;
        }
        if (buffers[i].plan > SIZE_MAX - hold) {
            return false;
        }
        hold += buffers[i].plan;
    }
    *out = hold;
    return true;
}

bool zone_runtime_alloc_respects_hold(size_t available, size_t hold, size_t size) {
    // Written as two comparisons so that neither a huge hold nor a huge size can wrap into "fits".
    return hold <= available && size <= available - hold;
}

static size_t zone_hold_scale_request(size_t request, uint32_t from_n_ubatch, uint32_t to_n_ubatch) {
    if (from_n_ubatch == 0 || to_n_ubatch == 0 || from_n_ubatch == to_n_ubatch || request == 0) {
        return request;
    }
    if (request > SIZE_MAX / to_n_ubatch) {
        return SIZE_MAX;
    }
    const size_t product = request * to_n_ubatch;
    return product > SIZE_MAX - (from_n_ubatch - 1) ? SIZE_MAX : (product + from_n_ubatch - 1) / from_n_ubatch;
}

size_t zone_hold_spill_bound(size_t plan, size_t request_hwm, uint32_t hwm_n_ubatch, uint32_t n_ubatch) {
    if (plan == 0) {
        return 0;
    }
    const size_t request = zone_hold_scale_request(request_hwm, hwm_n_ubatch, n_ubatch);
    return request > SIZE_MAX - plan ? SIZE_MAX : plan + request;
}

size_t zone_hold_spill_raw_demand(size_t spill_bound, size_t kv_zone_free) {
    return spill_bound > kv_zone_free ? spill_bound - kv_zone_free : 0;
}

size_t zone_kv_room_for_compute(size_t kv_largest_free, size_t kv_pending_bytes) {
    return kv_pending_bytes >= kv_largest_free ? 0 : kv_largest_free - kv_pending_bytes;
}

bool zone_hold_spill_bound_fits(size_t free_before, size_t headroom_target, size_t bound) {
    // The hold is blamed only when ITS demand is what pushes the card under the headroom: the card was above it
    // without the demand and is below it with. Two comparisons, so that neither a huge bound nor a huge headroom wraps.
    if (bound == 0 || free_before < headroom_target) {
        return true;
    }
    return bound <= free_before && free_before - bound >= headroom_target;
}

size_t zone_hold_fit_request(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    if (!in.rungs || in.n_rungs == 0) {
        return 0;
    }
    bool   own         = false;
    size_t own_request = 0;
    for (size_t i = 0; i < in.n_rungs; ++i) {
        if (in.rungs[i].n_ubatch == n_ubatch) {
            own         = true;
            own_request = std::max(own_request, in.rungs[i].bytes);
        }
    }
    if (own) {
        return own_request;
    }
    size_t scaled = 0;
    for (size_t i = 0; i < in.n_rungs; ++i) {
        scaled = std::max(scaled, zone_hold_scale_request(in.rungs[i].bytes, in.rungs[i].n_ubatch, n_ubatch));
    }
    return scaled;
}

size_t zone_hold_fit_demand(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    if (n_ubatch == 0 || !in.plan_of) {
        return 0;
    }
    const size_t plan = in.plan_of(in.plan_ctx, n_ubatch);
    // The request is already the rung's own (or scaled to it), so the bound is asked with it at this n_ubatch.
    return zone_hold_spill_raw_demand(
        zone_hold_spill_bound(plan, zone_hold_fit_request(in, n_ubatch), n_ubatch, n_ubatch), in.kv_room);
}

bool zone_hold_fit(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    return zone_hold_spill_bound_fits(in.free_before, in.headroom_target, zone_hold_fit_demand(in, n_ubatch));
}

uint32_t zone_hold_fit_largest_ub(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    if (n_ubatch == 0) {
        return 0;
    }
    if (zone_hold_fit(in, n_ubatch)) {
        return n_ubatch;
    }
    if (n_ubatch < 32) {
        return 0;
    }
    uint32_t rung = 32;
    while (rung <= n_ubatch / 2) {
        rung <<= 1;
    }
    for (; rung >= 32; rung >>= 1) {
        if (zone_hold_fit(in, rung)) {
            return rung;
        }
    }
    return 0;
}

size_t zone_hold_free_cold(size_t driver_free, size_t raw_live) {
    return raw_live > SIZE_MAX - driver_free ? SIZE_MAX : driver_free + raw_live;
}

size_t zone_hold_free_before(size_t cold, size_t persistent_raw) {
    return persistent_raw >= cold ? 0 : cold - persistent_raw;
}

size_t zone_hold_persistent_raw(size_t raw_held, size_t compute_live, bool rung_live) {
    if (!rung_live) {
        return raw_held;
    }
    return compute_live >= raw_held ? 0 : raw_held - compute_live;
}

size_t zone_hold_cold_update(bool have_baseline, size_t baseline, size_t candidate) {
    return have_baseline && baseline > candidate ? baseline : candidate;
}

size_t zone_hold_pick_kv_room(bool rung_live, bool have_epoch, size_t epoch_kv_room, size_t live_kv_room) {
    return rung_live && have_epoch ? epoch_kv_room : live_kv_room;
}

size_t zone_hold_nonfa_demand(size_t nonfa_scratch, size_t hold_spill) {
    return hold_spill > SIZE_MAX - nonfa_scratch ? SIZE_MAX : nonfa_scratch + hold_spill;
}

bool zone_runtime_spill_prefers_kv_zone(bool   compute_spill_flag,
                                        bool   runtime_zone,
                                        bool   forbid_spill,
                                        bool   zone_misses,
                                        size_t kv_zone_free,
                                        size_t alloc_size) {
    return compute_spill_flag && runtime_zone && !forbid_spill && zone_misses && alloc_size <= kv_zone_free;
}

bool zone_runtime_alloc_held_back(bool   runtime_zone,
                                  bool   forbid_spill,
                                  size_t zone_available,
                                  size_t hold,
                                  size_t alloc_size) {
    // Held back means the zone ALONE would have served the request and the hold is what keeps it out. A request
    // larger than the free bytes spills with or without a hold (an ordinary zone-full spill, which the allocator
    // handles exactly as it did before the hold existed), and with no hold there is nothing to keep it out.
    return runtime_zone && !forbid_spill && hold > 0 && alloc_size <= zone_available &&
           !zone_runtime_alloc_respects_hold(zone_available, hold, alloc_size);
}

bool zone_route_draws_scratch(bool decision_valid,
                              bool primary_is_unified,
                              bool primary_draws,
                              bool fallback_valid,
                              bool fallback_draws) {
    if (!decision_valid) {
        return false;
    }
    if (!primary_is_unified) {
        return primary_draws;
    }
    return fallback_valid && fallback_draws;
}

size_t zone_dense_scratch_merge_input(size_t prev, size_t next, bool other_model_live) {
    return other_model_live && prev > next ? prev : next;
}

namespace {

struct underestimate_record {
    size_t count = 0;  // genuine sizing misses only; never fragmentation

    // The worst overshoot is what sizes the fix, so the largest request is
    // retained rather than the most recent one. planned_at_max is the zone
    // size that request was measured against — kept as a pair so the two
    // numbers in the report describe the same event; a "last planned" field
    // could pair the largest request with an unrelated zone size.
    size_t max_requested  = 0;
    size_t planned_at_max = 0;

    // Reservations that consulted this path's planned zone at all. Zero here
    // means the path was never entered, which is NOT the same as the predicate
    // being right — see the header.
    size_t observations = 0;
};

struct underestimate_state {
    std::mutex                                  mutex;
    std::map<std::string, underestimate_record> table;
};

// Process-lifetime diagnostic state. A unified_cache can be destroyed from the
// global cache registry during static destruction, after function-local static
// destructors registered later in main have already run. Giving this tiny
// accounting table a destructor therefore makes a late cache shutdown read a
// destroyed std::map (and used to produce exit 139 after every test passed).
//
// Leak the state deliberately: it owns no SYCL resources, remains bounded by
// the handful of sizing path names, and must also survive ordinary DSO teardown
// until every cache destructor has reported. Keeping the mutex and table in one
// process-lifetime object removes both destructor-order dependencies.
underestimate_state & underestimates() {
    static underestimate_state * state = new underestimate_state();
    return *state;
}

const char * path_key(const char * path) {
    return path ? path : "unknown";
}

size_t underestimate_field(const char * path, size_t underestimate_record::* field) {
    underestimate_state &                                      state = underestimates();
    std::lock_guard<std::mutex>                                 lock(state.mutex);
    std::map<std::string, underestimate_record>::const_iterator it = state.table.find(path_key(path));
    return it == state.table.end() ? 0 : it->second.*field;
}

}  // namespace

void zone_sizing_record_underestimate(const char * path, size_t requested_bytes, size_t planned_bytes) {
    underestimate_state &       state = underestimates();
    std::lock_guard<std::mutex> lock(state.mutex);
    underestimate_record &      record = state.table[path_key(path)];
    record.count += 1;
    if (requested_bytes > record.max_requested) {
        record.max_requested  = requested_bytes;
        record.planned_at_max = planned_bytes;
    }
}

size_t zone_sizing_underestimate_count(const char * path) {
    return underestimate_field(path, &underestimate_record::count);
}

size_t zone_sizing_max_underestimate_bytes(const char * path) {
    return underestimate_field(path, &underestimate_record::max_requested);
}

void zone_sizing_record_observation(const char * path) {
    underestimate_state &       state = underestimates();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.table[path_key(path)].observations += 1;
}

void zone_sizing_record_observations(const char * path, size_t count) {
    if (count == 0) {
        return;
    }
    underestimate_state &       state = underestimates();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.table[path_key(path)].observations += count;
}

size_t zone_sizing_observation_count(const char * path) {
    return underestimate_field(path, &underestimate_record::observations);
}

void zone_sizing_reset_underestimates() {
    underestimate_state &       state = underestimates();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.table.clear();
}

bool zone_sizing_log_underestimate_summary() {
    // Snapshot under the lock, report outside it: the log callback is backend
    // code that must not run while this unit's mutex is held.
    std::map<std::string, underestimate_record> snapshot;
    {
        underestimate_state &       state = underestimates();
        std::lock_guard<std::mutex> lock(state.mutex);
        snapshot = state.table;
    }

    bool reported = false;
    for (std::map<std::string, underestimate_record>::const_iterator it = snapshot.begin(); it != snapshot.end();
         ++it) {
        // Observations alone are the healthy case and say nothing worth a
        // warning. Only a recorded miss breaks the silence.
        if (it->second.count == 0) {
            continue;
        }
        reported = true;
        ZONE_SIZING_LOG_WARN(
            "[SYCL-PLAN] zone sizing under-estimated the '%s' path %zu time(s) out of %zu reservation(s): largest "
            "request %.1f MB against a planned %.1f MB. The zone grew at runtime each time, so this is a slowdown "
            "rather than a failure -- re-check that path's predicate in zone-sizing.hpp against this model.\n",
            it->first.c_str(), it->second.count, it->second.observations, it->second.max_requested / (1024.0 * 1024.0),
            it->second.planned_at_max / (1024.0 * 1024.0));
    }
    return reported;
}

zone_collapse_signal zone_detect_collapse(const std::vector<zone_tensor_desc> & inventory,
                                          const path_scoped_maxima &            maxima) {
    // Too few tensors to expect a per-layer family at all: absence of one is
    // not evidence of anything, and a diagnostic that fires where it cannot
    // judge is worse than none.
    if (inventory.size() < k_zone_collapse_min_inventory) {
        return zone_collapse_signal::NONE;
    }

    // Every path is checked, not just one. The predicates are identical today
    // but are documented to diverge, and a collapse in the shared grouping
    // input takes all of them down together — one path narrowing while the
    // others do not is a predicate difference, not a classifier failure.
    const bool none_classified =
        maxima.onednn_eligible == 0 && maxima.cpu_quant_eligible == 0 && maxima.dma_streamed == 0;
    if (none_classified) {
        return zone_collapse_signal::NO_FAMILY;
    }

    const bool none_narrowed = maxima.onednn_eligible == maxima.any_tensor &&
                               maxima.cpu_quant_eligible == maxima.any_tensor &&
                               maxima.dma_streamed == maxima.any_tensor;
    if (none_narrowed) {
        return zone_collapse_signal::NO_NARROWING;
    }

    return zone_collapse_signal::NONE;
}

void zone_sizing_warn_if_collapsed(const std::vector<zone_tensor_desc> & inventory, const path_scoped_maxima & maxima) {
    const zone_collapse_signal signal = zone_detect_collapse(inventory, maxima);
    if (signal == zone_collapse_signal::NONE) {
        return;
    }

    // Recomputed rather than plumbed through: this runs once per plan, only on
    // the already-degraded path, and the counts are what make the warning
    // actionable instead of merely alarming.
    const std::map<zone_group_key, size_t> freq = zone_group_frequencies(inventory);

    size_t shapeless = 0;
    for (size_t i = 0; i < inventory.size(); i++) {
        if (!inventory[i].has_shape) {
            shapeless++;
        }
    }

    if (signal == zone_collapse_signal::NO_FAMILY) {
        ZONE_SIZING_LOG_WARN(
            "[SYCL-PLAN] zone sizing found no repeated (type, ne) group in %zu inventory entries (%zu distinct "
            "groups, %zu without a shape): every path-scoped maximum collapsed to zero and each zone falls back to "
            "the global %.1f MB maximum, reclaiming nothing. If this model really is all singletons that is "
            "legitimate; otherwise the inventory adapter is not carrying type/ne/has_shape into zone_tensor_desc.\n",
            inventory.size(), freq.size(), shapeless, maxima.any_tensor / (1024.0 * 1024.0));
        return;
    }

    ZONE_SIZING_LOG_WARN(
        "[SYCL-PLAN] zone sizing narrowed nothing: every path-scoped maximum equals the global %.1f MB maximum over "
        "%zu inventory entries in %zu distinct (type, ne) groups. Legitimate when the largest tensor is itself a "
        "per-layer weight; otherwise the inventory adapter is losing shape or type detail, so unrelated tensors are "
        "grouping together.\n",
        maxima.any_tensor / (1024.0 * 1024.0), inventory.size(), freq.size());
}

}  // namespace ggml_sycl
