//
// Structural path-scoped arena zone sizing implementation.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "zone-sizing.hpp"

#include <algorithm>
#include <map>

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
bool zone_is_onednn_reorder_eligible(const zone_tensor_desc & tensor, size_t group_cardinality) {
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

        if (zone_is_onednn_reorder_eligible(tensor, cardinality)) {
            maxima.onednn_eligible = std::max(maxima.onednn_eligible, tensor.size);
        }
        if (zone_is_cpu_quant_eligible(tensor, cardinality)) {
            maxima.cpu_quant_eligible = std::max(maxima.cpu_quant_eligible, tensor.size);
        }
        if (zone_is_dma_streamed(tensor, cardinality)) {
            maxima.dma_streamed = std::max(maxima.dma_streamed, tensor.size);
        }
    }
    return maxima;
}

}  // namespace ggml_sycl
