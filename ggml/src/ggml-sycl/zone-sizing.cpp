//
// Structural path-scoped arena zone sizing implementation.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "zone-sizing.hpp"

#include "tlsf-allocator.hpp"

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
//
// This TU's classifier stays name-free. One decision about the head IS made, but
// by the adapter, from classifiers that exist already (infer_tensor_usage's
// OUTPUT_WEIGHT and the tied-embedding classifier), and it concerns a different
// reservation: the f16 dequant buffers for a type whose PP route is the dequant
// arm (the IQ family) leave the head out, because a completion runs the head on
// one row per ubatch and its f16 copy (1.2 GB at a 248k vocabulary) would
// otherwise size the whole plan. The Q4_0 / MXFP4 head stays planned (owner
// decision, llama.cpp-8ony) until llama.cpp-fkpg delivers n_outputs; that gate
// reinstates both. See aos_dequant_f16_plan_claims in unified-types.hpp.
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

bool zone_onednn_plan_same(const zone_onednn_plan & a, const zone_onednn_plan & b) {
    return a.bare_bytes == b.bare_bytes && a.graph_floor_bytes == b.graph_floor_bytes &&
           a.weights_bytes == b.weights_bytes && a.activations_bytes == b.activations_bytes;
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

bool zone_moe_mmvq_op_bytes(const zone_moe_mmvq_operand & op, int64_t n_tokens, int64_t n_expert_used, size_t * out) {
    if (!out) {
        return false;
    }
    *out = 0;
    if (op.k <= 0 || op.grouped_experts < 0 || n_tokens <= 0 || n_expert_used <= 0) {
        return false;
    }
    *out = SIZE_MAX;  // every return below is an overflow

    // GGML_PAD(ne10, QK8_1) / QK8_1 blocks of sizeof(block_q8_1) bytes: the dispatch's q8_1_row_size.
    const int64_t block  = k_zone_mmq_src1_block_elems;
    const size_t  blocks = static_cast<size_t>(op.k / block + (op.k % block != 0 ? 1 : 0));
    const size_t  tokens = static_cast<size_t>(n_tokens);
    const size_t  slots  = static_cast<size_t>(n_expert_used);
    if (blocks > SIZE_MAX / k_zone_mmq_src1_block_bytes || tokens > SIZE_MAX / slots) {
        return false;
    }
    const size_t row         = blocks * k_zone_mmq_src1_block_bytes;
    const size_t token_slots = tokens * slots;  // the dispatch's total_batches
    const size_t rows        = op.row_per_slot ? token_slots : tokens;
    if (rows > SIZE_MAX / row || token_slots > SIZE_MAX / sizeof(int32_t)) {
        return false;
    }
    const size_t q8  = rows * row;
    const size_t ids = token_slots * sizeof(int32_t);
    if (ids > SIZE_MAX - q8) {
        return false;
    }

    // The grouped tables, in entries: experts, offsets (experts + 1), rows (total_batches), chunk groups and chunk
    // starts (max_chunks each: every expert's rows cut into chunks of k_zone_mxfp4_moe_group_rows), and the counter.
    size_t tables = 0;
    if (op.grouped_experts > 0) {
        const size_t experts         = static_cast<size_t>(op.grouped_experts);
        const size_t group_rows      = static_cast<size_t>(k_zone_mxfp4_moe_group_rows);
        const size_t chunks_a_expert = token_slots / group_rows + (token_slots % group_rows != 0 ? 1 : 0);
        if (experts > (SIZE_MAX - 2) / 2 || chunks_a_expert > SIZE_MAX / experts) {
            return false;
        }
        const size_t max_chunks = experts * chunks_a_expert;
        const size_t fixed      = 2 * experts + 2;  // experts, offsets, and the counter
        if (max_chunks > (SIZE_MAX - fixed) / 2 || token_slots > SIZE_MAX - fixed - 2 * max_chunks) {
            return false;
        }
        const size_t entries = fixed + token_slots + 2 * max_chunks;
        if (entries > SIZE_MAX / sizeof(int32_t)) {
            return false;
        }
        tables = entries * sizeof(int32_t);
    }
    if (tables > SIZE_MAX - q8 - ids) {
        return false;
    }
    *out = q8 + ids + tables;
    return true;
}

bool zone_moe_mmvq_scratch_bytes(const std::vector<zone_moe_mmvq_operand> & ops,
                                 int64_t                                    n_tokens,
                                 int64_t                                    n_expert_used,
                                 zone_moe_mmvq_scratch *                    out) {
    if (!out) {
        return false;
    }
    *out = {};
    if (n_tokens <= 0 || n_expert_used <= 0 ||
        std::any_of(ops.begin(), ops.end(),
                    [](const zone_moe_mmvq_operand & op) { return op.k <= 0 || op.grouped_experts < 0; })) {
        return false;
    }
    bool exact = true;
    for (const zone_moe_mmvq_operand & op : ops) {
        size_t bytes = 0;
        exact        = zone_moe_mmvq_op_bytes(op, n_tokens, n_expert_used, &bytes) && exact;
        out->per_op  = std::max(out->per_op, bytes);
        if (bytes > SIZE_MAX - out->graph) {
            out->graph = SIZE_MAX;
            exact      = false;
        } else {
            out->graph += bytes;
        }
    }
    return exact;
}

size_t zone_scratch_required_bytes(size_t default_bytes, size_t nonfa_bytes) {
    return std::max(default_bytes, nonfa_bytes);
}

zone_scratch_plan zone_scratch_zone_plan(size_t default_bytes,
                                         size_t nonfa_bytes,
                                         size_t moe_mmvq_bytes,
                                         size_t budget_cap_bytes,
                                         size_t room_bytes) {
    zone_scratch_plan plan;
    plan.required         = zone_scratch_required_bytes(default_bytes, nonfa_bytes);
    // The room is on top of what the zone must hold; SIZE_MAX, no limit, saturates.
    const size_t room_cap = room_bytes > SIZE_MAX - plan.required ? SIZE_MAX : plan.required + room_bytes;
    const size_t limit    = std::min(budget_cap_bytes, room_cap);
    plan.wanted           = std::max(plan.required, std::min(moe_mmvq_bytes, limit));
    if (moe_mmvq_bytes > plan.required && moe_mmvq_bytes > limit) {
        plan.clamp = room_cap <= budget_cap_bytes ? zone_scratch_clamp::ROOM : zone_scratch_clamp::BUDGET;
    }
    return plan;
}

size_t zone_scratch_kept_bytes(const zone_scratch_plan & plan, size_t capacity_bytes) {
    if (capacity_bytes >= plan.wanted || capacity_bytes < plan.required) {
        return plan.wanted;
    }
    return capacity_bytes;
}

bool zone_scratch_warn_due(size_t latched_bytes, size_t bytes) {
    return bytes > latched_bytes;
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

bool zone_runtime_kept_above_plan(size_t live_zone_bytes, size_t floor_bytes, size_t requirement) {
    // The reserved zone already carries the arena's rounding, so every byte above it was grown.
    size_t reserved = 0;
    return zone_runtime_reserved_bytes(floor_bytes, requirement, &reserved) && live_zone_bytes > reserved;
}

size_t zone_arena_zone_bytes(size_t bytes) {
    return (bytes + k_zone_arena_alignment - 1) & ~(k_zone_arena_alignment - 1);
}

size_t zone_arena_round_alloc_bytes(size_t bytes) {
    return (bytes / k_zone_arena_size_alignment) * k_zone_arena_size_alignment;
}

size_t zone_arena_per_chunk_cap(size_t max_alloc_size, size_t safe_max_alloc_size) {
    if (safe_max_alloc_size == 0) {
        return max_alloc_size;
    }
    return max_alloc_size > 0 ? std::min(max_alloc_size, safe_max_alloc_size) : safe_max_alloc_size;
}

bool zone_arena_single_chunk_tried(size_t alloc_bytes, size_t max_alloc_size, size_t per_chunk_cap) {
    return max_alloc_size == 0 || alloc_bytes <= per_chunk_cap;
}

size_t zone_arena_tail_kv_reserve(size_t alloc_bytes) {
    constexpr size_t k_min_tail_kv_reserve = 128ull * 1024ull * 1024ull;
    constexpr size_t k_max_tail_kv_reserve = 512ull * 1024ull * 1024ull;
    const size_t     proportional_kv       = std::max<size_t>(alloc_bytes / 128, k_zone_arena_min_shared_bytes);
    return std::min(k_max_tail_kv_reserve, std::max(k_min_tail_kv_reserve, proportional_kv));
}

size_t zone_arena_tail_chunk_bytes(size_t alloc_bytes,
                                   size_t per_chunk_cap,
                                   size_t tail_bytes,
                                   size_t tail_kv_reserve) {
    const size_t wanted = tail_kv_reserve > SIZE_MAX - tail_bytes ? SIZE_MAX : tail_bytes + tail_kv_reserve;
    return std::min({ wanted, per_chunk_cap, alloc_bytes });
}

bool zone_arena_chunk_holds_tail(size_t chunk_bytes, size_t tail_bytes) {
    return tail_bytes <= SIZE_MAX - k_zone_arena_min_shared_bytes &&
           chunk_bytes >= tail_bytes + k_zone_arena_min_shared_bytes;
}

bool zone_arena_weight_chunks_fit(size_t alloc_bytes, size_t tail_chunk_bytes) {
    return tail_chunk_bytes <= alloc_bytes && alloc_bytes - tail_chunk_bytes >= k_zone_arena_min_shared_bytes;
}

zone_arena_reserve_shape zone_arena_reserve_shape_of(size_t budget_bytes,
                                                     size_t device_total_vram,
                                                     size_t max_alloc_size,
                                                     size_t safe_max_alloc_size) {
    zone_arena_reserve_shape shape;
    const size_t arena  = device_total_vram > 0 && budget_bytes > device_total_vram ? device_total_vram : budget_bytes;
    shape.alloc_bytes   = zone_arena_round_alloc_bytes(arena);
    shape.per_chunk_cap = zone_arena_per_chunk_cap(max_alloc_size, safe_max_alloc_size);
    shape.single_chunk  = zone_arena_single_chunk_tried(shape.alloc_bytes, max_alloc_size, shape.per_chunk_cap);
    shape.tail_kv_reserve = zone_arena_tail_kv_reserve(shape.alloc_bytes);
    return shape;
}

bool zone_arena_single_chunk_accepts(const zone_arena_reserve_shape & shape, size_t tail_bytes) {
    return shape.single_chunk && zone_arena_chunk_holds_tail(shape.alloc_bytes, tail_bytes);
}

bool zone_arena_n_chunk_accepts(const zone_arena_reserve_shape & shape, size_t tail_bytes) {
    const size_t tail_chunk =
        zone_arena_tail_chunk_bytes(shape.alloc_bytes, shape.per_chunk_cap, tail_bytes, shape.tail_kv_reserve);
    return zone_arena_chunk_holds_tail(tail_chunk, tail_bytes) &&
           zone_arena_weight_chunks_fit(shape.alloc_bytes, tail_chunk);
}

size_t zone_arena_scratch_room(const zone_arena_reserve_shape & shape, size_t other_tail_bytes, size_t scratch_bytes) {
    const size_t scratch = zone_arena_zone_bytes(scratch_bytes);
    if (scratch < scratch_bytes || other_tail_bytes > SIZE_MAX - scratch) {
        return 0;  // the rounding or the tail wrapped: no zone the reserve could carve
    }
    const size_t base    = other_tail_bytes + scratch;
    const bool   single  = zone_arena_single_chunk_accepts(shape, base);
    const bool   n_chunk = zone_arena_n_chunk_accepts(shape, base);
    if (!single && !n_chunk) {
        return 0;
    }
    // Each check fails only as the tail grows, so the tails that keep every check passing are those up to one largest.
    const auto passes = [&](size_t tail) {
        return (!single || zone_arena_single_chunk_accepts(shape, tail)) &&
               (!n_chunk || zone_arena_n_chunk_accepts(shape, tail));
    };
    // Every passing tail lies inside the arena (base passed one check, so base < alloc), which bounds the search.
    size_t lo = 0;
    size_t hi = (shape.alloc_bytes - base) / k_zone_arena_alignment;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo + 1) / 2;
        if (passes(base + mid * k_zone_arena_alignment)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo * k_zone_arena_alignment;
}

bool zone_runtime_requirement(const zone_runtime_terms & terms, size_t * out) {
    if (out == nullptr) {
        return false;
    }
    const size_t added[] = { terms.moe_host_scatter, terms.mmq_src1, terms.dequant_f16, terms.compute, terms.state };
    size_t       total   = terms.base;
    for (const size_t term : added) {
        if (term > SIZE_MAX - total) {
            return false;
        }
        total += term;
    }
    *out = total;
    return true;
}

bool zone_arena_rebuild_allowed(const zone_arena_holders & holders) {
    // scratch_outside is not read: a rebuild leaves an allocation outside the arena where it is.
    return holders.live_zone_bytes == 0 && holders.chunk_leases == 0 && holders.scratch_in_arena == 0;
}

bool zone_arena_only_ring_holds(bool rebuild_needed, const zone_arena_holders & holders) {
    // The ring's slot buffers are among scratch_in_arena, and their bytes among live_zone_bytes, so those two are read
    // through the ring's own counts; a lease is never the ring's.
    return rebuild_needed && holders.ring_in_arena != 0 && holders.scratch_in_arena == holders.ring_in_arena &&
           holders.live_beside_ring == 0 && holders.chunk_leases == 0;
}

bool zone_arena_shortfall_is_reservation_only(const zone_arena_sizes & have,
                                              const zone_arena_sizes & need,
                                              size_t                   onednn_pair_bytes) {
    // Only the ONEDNN zone is short, and it still holds the pair. Not widened to SCRATCH: the non-FA scratch it holds
    // has no DIRECT path of its own.
    return have.scratch >= need.scratch && have.runtime >= need.runtime && have.onednn < need.onednn &&
           have.onednn >= onednn_pair_bytes;
}

size_t zone_onednn_kept_pair_bytes(const zone_onednn_plan & kept, size_t held_pair_bytes) {
    return std::max(kept.bare_bytes, held_pair_bytes);
}

zone_onednn_plan zone_onednn_plan_keep_short(const zone_onednn_plan & held,
                                             const zone_onednn_plan & live,
                                             size_t                   capacity_bytes,
                                             size_t                   held_pair_bytes) {
    zone_onednn_plan kept  = zone_onednn_plan_keep(held, live);
    const size_t     pair  = zone_onednn_kept_pair_bytes(kept, held_pair_bytes);
    const size_t     room  = capacity_bytes > pair ? capacity_bytes - pair : 0;
    kept.graph_floor_bytes = std::min(room, kept.graph_floor_bytes);
    return kept;
}

size_t zone_graph_direct_cap_snapshot_kept(size_t held_bytes, size_t current_bytes) {
    return held_bytes == 0 ? current_bytes : std::min(held_bytes, current_bytes);
}

bool zone_kept_onednn_warn_due(uint64_t latched_txn, uint64_t load_txn) {
    return load_txn == 0 || latched_txn != load_txn;
}

// `bytes` in MB at one decimal, in KB under 0.1 MB, and in bytes under 1 KB, so a small figure never prints as 0.0 MB.
static std::string zone_bytes_text(size_t bytes) {
    char out[48];
    if (bytes < 1024) {
        std::snprintf(out, sizeof(out), "%zu bytes", bytes);
    } else if (static_cast<double>(bytes) < 0.1 * 1024.0 * 1024.0) {
        std::snprintf(out, sizeof(out), "%.1f KB", bytes / 1024.0);
    } else {
        std::snprintf(out, sizeof(out), "%.1f MB", bytes / (1024.0 * 1024.0));
    }
    return out;
}

// The holders refusal's clause: each kind of holder that points into the arena with its count, and the remedy the
// kinds allow. The cache's own scratch buffers other than the PP MoE ring stay in the arena until the cache is torn
// down at process exit, whatever else holds it. The ring is the cache's too, but a load releases it when it is the
// arena's only holder and no prefill's claim on one of its slots is held; a claim outlives its prefill until a context
// on the device is freed, so freeing that context is the ring's remedy. Leases with nothing but the ring live in the
// zones outlived what they point at, so no free releases them either. Live allocations belong to a model or context,
// which freeing releases, unless the cache still has releases of its own to run: their bytes count as live too, so then
// the remedy says they may be those and asks for the load again first.
static std::string zone_arena_unfinished_text(const zone_arena_holders & h) {
    struct kind {
        size_t       count;
        const char * one;
        const char * many;
    };

    const kind kinds[] = {
        { h.queued_frees,    "queued free",                 "queued frees"                 },
        { h.release_retries, "release waiting for a retry", "releases waiting for a retry" },
        { h.retired_pending, "retired entry",               "retired entries"              },
        { h.evictions,       "eviction in flight",          "evictions in flight"          },
    };
    std::string parts[4];
    int         n = 0;
    for (const kind & k : kinds) {
        if (k.count != 0) {
            parts[n++] = std::to_string(k.count) + " " + (k.count == 1 ? k.one : k.many);
        }
    }
    std::string text;
    for (int i = 0; i < n; ++i) {
        text += (i == 0 ? "" : i == n - 1 ? " and " : ", ") + parts[i];
    }
    return text;
}

static std::string zone_arena_held_text(const zone_arena_holders & h, int device) {
    char        part[96];
    std::string held[3];
    int         n = 0;
    if (h.live_zone_bytes != 0) {
        held[n++] = zone_bytes_text(h.live_zone_bytes) + " of live allocations in its zones";
    }
    if (h.chunk_leases != 0) {
        std::snprintf(part, sizeof(part), "%u lease%s on its chunks", h.chunk_leases, h.chunk_leases == 1 ? "" : "s");
        held[n++] = part;
    }
    if (h.scratch_in_arena != 0 && h.ring_in_arena != 0) {
        std::snprintf(part, sizeof(part), "%u of the cache's own scratch buffers (%u of them its PP MoE ring's slots)",
                      h.scratch_in_arena, h.ring_in_arena);
        held[n++] = part;
    } else if (h.scratch_in_arena != 0) {
        std::snprintf(part, sizeof(part), "%u of the cache's own scratch buffers", h.scratch_in_arena);
        held[n++] = part;
    }
    std::string text = "; ";
    if (n == 0) {
        text += "allocations of a model or context already on the device";
    }
    for (int i = 0; i < n; ++i) {
        text += (i == 0 ? "" : i == n - 1 ? " and " : ", ") + held[i];
    }
    text += " hold the arena (the [VRAM-ARENA] lines above name them), so its zones cannot grow";
    const uint32_t other_scratch = h.scratch_in_arena > h.ring_in_arena ? h.scratch_in_arena - h.ring_in_arena : 0;
    // With no slot of the ring inside the arena, every live byte belongs to a model or context.
    const bool     live_beside   = h.ring_in_arena == 0 ? h.live_zone_bytes != 0 : h.live_beside_ring != 0;
    if (other_scratch == 0 && (live_beside || (h.chunk_leases == 0 && h.ring_in_arena == 0))) {
        const std::string unfinished = zone_arena_unfinished_text(h);
        if (unfinished.empty()) {
            return text + ". Free the model or context they belong to, then load this one again";
        }
        return text + ". Some of those bytes may be releases the cache has not finished (" + unfinished +
               "), which it finishes on its own. Load this one again; if it is refused with the same bytes, free the "
               "model or context they belong to";
    }
    char remedy[256];
    if (other_scratch != 0) {
        std::snprintf(remedy, sizeof(remedy),
                      ". The cache keeps its own scratch buffers in the arena until the process exits, so freeing a "
                      "model or context does not let its zones grow. Load this model in a new process before any "
                      "other on device %d",
                      device);
    } else if (h.chunk_leases != 0) {
        std::snprintf(remedy, sizeof(remedy),
                      ". %s is live in its zones, so the leases outlived what they point at and are held until the "
                      "process exits. Load this model in a new process before any other on device %d",
                      h.ring_in_arena == 0 ? "No allocation" : "Nothing but the cache's PP MoE ring", device);
    } else {
        std::snprintf(remedy, sizeof(remedy),
                      ". Only the cache's PP MoE ring holds it, and a prefill's claim on one of its slots keeps it: "
                      "the claim outlives its prefill until a context on device %d is freed. Free the context that "
                      "ran the prefill, then load this one again",
                      device);
    }
    return text + remedy;
}

// The refused teardown's clause: why it refused, and what that left. Only a refusal named as one before the teardown
// closed the arena leaves it usable; any other, a missing cause included, is read as closed, which no reserve may
// follow and no later load in the process can use.
static std::string zone_arena_teardown_text(zone_arena_teardown_refusal cause, int device) {
    std::string text = "; the arena could not be torn down";
    switch (cause) {
        case zone_arena_teardown_refusal::HOST_STATE:
            return text +
                   ": the host could not reserve the teardown's own bookkeeping, and the arena is as it was. Free "
                   "host memory, then load this one again";
        case zone_arena_teardown_refusal::LIVE_ALLOCATIONS:
            text += ": allocations were still live in it when the teardown closed it";
            break;
        case zone_arena_teardown_refusal::DRAIN_SNAPSHOT:
            text += ": the host could not record the allocations the teardown waits on";
            break;
        case zone_arena_teardown_refusal::DRAIN_TIMEOUT:
            text += ": a lease on one of its allocations did not end while the teardown waited";
            break;
        case zone_arena_teardown_refusal::FREE_FAILED:
            text += ": the device refused to free one of its chunks, so part of it may already be freed";
            break;
        case zone_arena_teardown_refusal::NONE:
            break;
    }
    char closed[160];
    std::snprintf(closed, sizeof(closed),
                  ". The arena stays closed to new allocations, so no model can load on device %d in this process. "
                  "Load this model in a new process",
                  device);
    return text + closed;
}

void zone_scratch_use_note_alloc(zone_scratch_use & use, size_t request_bytes, size_t used_bytes, size_t live) {
    use.requests += use.requests < SIZE_MAX ? 1 : 0;
    use.largest_request_bytes = std::max(use.largest_request_bytes, request_bytes);
    if (used_bytes > use.high_water_bytes) {
        use.high_water_bytes   = used_bytes;
        use.live_at_high_water = live;
    }
}

void zone_scratch_use_note_miss(zone_scratch_use & use, size_t request_bytes) {
    use.misses += use.misses < SIZE_MAX ? 1 : 0;
    use.miss_bytes         = request_bytes > SIZE_MAX - use.miss_bytes ? SIZE_MAX : use.miss_bytes + request_bytes;
    use.largest_miss_bytes = std::max(use.largest_miss_bytes, request_bytes);
}

std::string zone_scratch_use_text(const zone_scratch_use & use, int device, size_t capacity_bytes) {
    if (use.requests == 0 && use.misses == 0) {
        return {};
    }
    constexpr double mib = 1024.0 * 1024.0;
    char             line[512];
    std::snprintf(line, sizeof(line),
                  "[UNIFIED-CACHE] SCRATCH zone use: device=%d high-water=%.2f MB of %.2f MB with %zu allocations "
                  "live, largest request=%.2f MB, requests placed=%zu; zone misses (placements the zone refused, "
                  "each spilled to raw device memory or refused by its caller): requests=%zu bytes=%.2f MB "
                  "largest=%.2f MB",
                  device, use.high_water_bytes / mib, capacity_bytes / mib, use.live_at_high_water,
                  use.largest_request_bytes / mib, use.requests, use.misses, use.miss_bytes / mib,
                  use.largest_miss_bytes / mib);
    return line;
}

std::string zone_arena_refusal_text(const zone_arena_refusal & refusal) {
    if (refusal.reason == zone_arena_refusal_reason::NONE) {
        return {};
    }
    constexpr double KiB = 1024.0;
    constexpr double MiB = 1024.0 * KiB;
    char             part[192];
    std::snprintf(part, sizeof(part), "[SYCL-PLAN] model load refused on device %d: VRAM arena zones", refusal.device);
    std::string text = part;
    // Every zone with its need and have, and a shortfall only on a zone that is short, so the line says which did not
    // fit. A shortfall under 0.1 MB would print as 0.0 MB beside a need and a have that print the same, so that zone's
    // three figures are printed in KB, and in bytes under 1 KB.
    const auto  zone = [&](const char * sep, const char * name, size_t need, size_t have) {
        const size_t short_bytes = need > have ? need - have : 0;
        if (short_bytes != 0 && short_bytes < 1024) {
            std::snprintf(part, sizeof(part), "%s%s need %zu bytes have %zu bytes (%zu bytes short)", sep, name, need,
                           have, short_bytes);
        } else if (short_bytes != 0 && static_cast<double>(short_bytes) < 0.1 * MiB) {
            std::snprintf(part, sizeof(part), "%s%s need %.1f KB have %.1f KB (%.1f KB short)", sep, name, need / KiB,
                           have / KiB, short_bytes / KiB);
        } else {
            std::snprintf(part, sizeof(part), "%s%s need %.1f MB have %.1f MB", sep, name, need / MiB, have / MiB);
            text += part;
            if (short_bytes == 0) {
                return;
            }
            std::snprintf(part, sizeof(part), " (%.1f MB short)", short_bytes / MiB);
        }
        text += part;
    };
    zone(" ", "SCRATCH", refusal.need.scratch, refusal.have.scratch);
    zone(", ", "ONEDNN", refusal.need.onednn, refusal.have.onednn);
    zone(", ", "RUNTIME", refusal.need.runtime, refusal.have.runtime);
    switch (refusal.reason) {
        case zone_arena_refusal_reason::REBUILD_HELD:
            text += zone_arena_held_text(refusal.holders, refusal.device);
            break;
        case zone_arena_refusal_reason::RESERVE_FAILED:
            text += "; the device could not reserve an arena with these zones";
            if (refusal.released.scratch != 0 || refusal.released.onednn != 0 || refusal.released.runtime != 0) {
                // The rebuild destroyed the arena before the reserve failed, so the device now has none: the have
                // above says so, and this names what it had.
                std::snprintf(part, sizeof(part),
                              ", and the arena it had (SCRATCH %.1f MB, ONEDNN %.1f MB, RUNTIME %.1f MB) was released "
                              "for the rebuild",
                              refusal.released.scratch / MiB, refusal.released.onednn / MiB,
                              refusal.released.runtime / MiB);
                text += part;
            }
            std::snprintf(part, sizeof(part), ". Free VRAM on device %d, then load this one again", refusal.device);
            text += part;
            break;
        case zone_arena_refusal_reason::NO_CACHE:
            text += "; the device has no unified cache to size them in";
            break;
        case zone_arena_refusal_reason::TEARDOWN_REFUSED:
            text += zone_arena_teardown_text(refusal.teardown, refusal.device);
            break;
        case zone_arena_refusal_reason::NONE:
            break;
    }
    return text;
}

bool zone_arena_teardown_left_closed(zone_arena_teardown_refusal cause) {
    return cause != zone_arena_teardown_refusal::NONE && cause != zone_arena_teardown_refusal::HOST_STATE;
}

// A compute-buffer request observed at `from_n_ubatch`, scaled linearly to `to_n_ubatch` (compute buffers grow about
// linearly with the micro-batch), rounded up. Unchanged when either micro-batch is unknown, they are equal, or the
// request is 0; SIZE_MAX when the product overflows. The hold-spill fit's one estimator for a compute buffer at a
// micro-batch nobody measured: the spill bound and the fit's request scale a rung's record with it.
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

bool zone_hold_fit_rung_has_record(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    if (!in.rungs) {
        return false;
    }
    for (size_t i = 0; i < in.n_rungs; ++i) {
        if (in.rungs[i].n_ubatch == n_ubatch) {
            return true;
        }
    }
    return false;
}

bool zone_hold_fit_rung_has_compute_room(const zone_hold_fit_inputs & in,
                                         uint32_t                     n_ubatch,
                                         size_t                       request,
                                         bool                         own_record) {
    if (n_ubatch == 0 || in.compute_room_n_ubatch == 0 || n_ubatch > in.compute_room_n_ubatch ||
        n_ubatch == in.held_out_n_ubatch) {
        return false;
    }
    if (own_record) {
        return request <= in.compute_room_bytes;
    }
    return in.compute_room_measured_n_ubatch != 0 && n_ubatch <= in.compute_room_measured_n_ubatch;
}

zone_hold_fit_terms zone_hold_fit_terms_at(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    zone_hold_fit_terms t;
    if (n_ubatch == 0) {
        return t;
    }
    // No plan function, no hold: the ring is still a demand.
    t.plan             = in.plan_of ? in.plan_of(in.plan_ctx, n_ubatch) : 0;
    // The request is already the rung's own (or scaled to it), so the bound is asked with it at this n_ubatch. One
    // the zone was planned to serve is not held back, so it spills nothing; the plan stays in the bound as it is.
    t.request          = zone_hold_fit_request(in, n_ubatch);
    t.request_credited =
        zone_hold_fit_rung_has_compute_room(in, n_ubatch, t.request, zone_hold_fit_rung_has_record(in, n_ubatch));
    t.kv_room          = in.kv_room;
    t.ring_uncovered   = in.ring_bytes;
    const size_t spill = zone_hold_spill_raw_demand(
        zone_hold_spill_bound(t.plan, t.request_credited ? 0 : t.request, n_ubatch, n_ubatch), in.kv_room);
    // The ring's slots are the planned room's bytes and lent bytes, so only the part neither the planned room nor a
    // loan covers is a demand on free memory (llama.cpp-kjef). The one place it is added, so every consumer of the fit
    // reads it. Saturating, so a wrapped sum never reads as a fit.
    t.demand = in.ring_bytes > SIZE_MAX - spill ? SIZE_MAX : spill + in.ring_bytes;
    return t;
}

size_t zone_hold_fit_demand(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    return zone_hold_fit_terms_at(in, n_ubatch).demand;
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

zone_context_shape zone_plan_context_shape(bool     n_ctx_is_runtime,
                                           uint32_t plan_n_ctx,
                                           uint32_t n_ctx_context,
                                           uint32_t plan_n_ubatch,
                                           uint32_t n_ubatch_context) {
    zone_context_shape s;
    s.n_ctx    = !n_ctx_is_runtime && n_ctx_context != 0 ? n_ctx_context : plan_n_ctx;
    s.n_ubatch = !n_ctx_is_runtime && n_ubatch_context != 0 ? n_ubatch_context : plan_n_ubatch;
    if (s.n_ctx != 0 && s.n_ubatch > s.n_ctx) {
        s.n_ubatch = s.n_ctx;
    }
    return s;
}

uint32_t zone_planning_n_ubatch(uint32_t envelope_n_ubatch, uint32_t n_ubatch) {
    return envelope_n_ubatch != 0 ? envelope_n_ubatch : n_ubatch;
}

zone_context_shape zone_plan_context_shape_for_envelope(bool     n_ctx_is_runtime,
                                                        uint32_t n_ctx,
                                                        uint32_t n_ctx_context,
                                                        uint32_t envelope_n_ubatch,
                                                        uint32_t n_ubatch,
                                                        uint32_t n_ubatch_context) {
    return zone_plan_context_shape(n_ctx_is_runtime, n_ctx, n_ctx_context,
                                   zone_planning_n_ubatch(envelope_n_ubatch, n_ubatch), n_ubatch_context);
}

size_t zone_onednn_graph_scratch_floor_bytes(uint32_t n_head_ctx_max,
                                             uint32_t n_head_swa_max,
                                             uint32_t n_swa,
                                             uint32_t n_ubatch,
                                             uint32_t n_ctx,
                                             size_t   direct_cap) {
    constexpr uint64_t k_floor_min_bytes = 64ull * 1024ull * 1024ull;
    constexpr uint64_t k_sizeof_f32      = 4;
    const uint64_t     ctx_term          = uint64_t(n_head_ctx_max) * uint64_t(n_ctx);
    // Widened before the add: n_swa + n_ubatch in 32 bits can wrap for a huge window.
    const uint64_t     swa_term = uint64_t(n_head_swa_max) * std::min<uint64_t>(n_ctx, uint64_t(n_swa) + n_ubatch);
    const uint64_t     elems    = std::max(ctx_term, swa_term) * uint64_t(n_ubatch);
    // Fixed-point 3/2 rather than a floating-point 1.5: exact at the MiB boundaries the formula was measured at.
    uint64_t           modeled  = (elems * k_sizeof_f32 * 3) / 2;
    if (direct_cap != 0) {
        modeled = std::min<uint64_t>(modeled, direct_cap);
    }
    return static_cast<size_t>(std::max<uint64_t>(k_floor_min_bytes, modeled));
}

size_t zone_onednn_graph_uncovered_bytes(size_t demand, size_t graph_room) {
    return demand > graph_room ? demand - graph_room : 0;
}

uint32_t zone_onednn_graph_fit_largest_ub(uint32_t n_head_ctx_max,
                                          uint32_t n_head_swa_max,
                                          uint32_t n_swa,
                                          uint32_t n_ubatch,
                                          uint32_t n_ctx,
                                          size_t   direct_cap,
                                          size_t   graph_room,
                                          size_t   free_outside,
                                          size_t   headroom,
                                          size_t   hold_spill) {
    constexpr uint32_t k_smallest_rung = 32;
    auto               fits            = [&](uint32_t ub) {
        const size_t demand =
            zone_onednn_graph_scratch_floor_bytes(n_head_ctx_max, n_head_swa_max, n_swa, ub, n_ctx, direct_cap);
        const size_t uncovered = zone_onednn_graph_uncovered_bytes(demand, graph_room);
        // A scratch the room holds draws nothing outside the arena: this check has nothing to answer for, and the hold
        // check judges its own spill.
        if (uncovered == 0) {
            return true;
        }
        // The same saturating sum the non-FA check charges the hold's spill with.
        const size_t outside = zone_hold_nonfa_demand(uncovered, hold_spill);
        return outside <= free_outside && free_outside - outside >= headroom;
    };
    if (n_ubatch == 0) {
        return 0;
    }
    if (fits(n_ubatch)) {
        return n_ubatch;
    }
    if (n_ubatch <= k_smallest_rung) {
        return 0;
    }
    // The largest power of two below n_ubatch, at least the smallest rung. The demand does not fall as the
    // micro-batch grows, so the first rung that fits going down is the largest.
    uint32_t rung = k_smallest_rung;
    while (rung <= (n_ubatch - 1) / 2) {
        rung <<= 1;
    }
    for (; rung >= k_smallest_rung; rung >>= 1) {
        if (fits(rung)) {
            return rung;
        }
    }
    return 0;
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

bool zone_compute_term_bytes(const uint64_t * chunk_bytes, size_t n_chunks, size_t alignment, size_t * out) {
    if (out == nullptr || (chunk_bytes == nullptr && n_chunks != 0)) {
        return false;
    }
    size_t sum = 0;
    for (size_t i = 0; i < n_chunks; ++i) {
        if (chunk_bytes[i] == 0) {
            continue;
        }
        if (static_cast<uint64_t>(static_cast<size_t>(chunk_bytes[i])) != chunk_bytes[i]) {
            return false;  // a chunk a size_t cannot carry
        }
        const size_t occupied = tlsf_allocator::round_request(static_cast<size_t>(chunk_bytes[i]), alignment);
        if (occupied == 0 || occupied > SIZE_MAX - sum) {
            return false;
        }
        sum += occupied;
    }
    *out = sum;
    return true;
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
