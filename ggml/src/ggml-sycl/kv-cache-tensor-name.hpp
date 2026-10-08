#pragma once
//
// Layer id of a llama_kv_cache tensor, parsed from its name.
//
// llama_kv_cache names its tensors "cache_%sk_l%d" / "cache_%sv_l%d" (llama-kv-cache.cpp) with a
// per-cache name tag: empty for the main attention cache, "idx_" for qwen4exp's indexer cache. The
// tiered KV buffer remaps each tensor onto its layer's own allocation by that layer id, so a parser
// that knows only the untagged prefixes leaves a tagged cache's tensors on the buffer's synthetic
// host span (llama.cpp-4ot7). Pure C++ with no ggml or SYCL include, so a host test can pin it.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <climits>
#include <cstring>

namespace ggml_sycl {

// "cache_<tag>(k|v)_l<N>" -> N, for any tag (including none); -1 for any other name: no "cache_" prefix,
// no digit run to the end of the name, a sign or other junk around the number, a state tensor whose letter
// is not k or v ("cache_r_l3", "cache_s_l3", "cache_ple_r_l3"), or an N that does not fit an int.
//
// Parsed from the end of the name, where the layer number sits, so the tag is free to hold anything.
inline int kv_cache_tensor_layer_id(const char * name) {
    static const char prefix[]   = "cache_";
    const size_t      prefix_len = sizeof(prefix) - 1;
    if (name == nullptr || strncmp(name, prefix, prefix_len) != 0) {
        return -1;
    }
    const size_t len = strlen(name);

    size_t digits_begin = len;
    while (digits_begin > prefix_len && name[digits_begin - 1] >= '0' && name[digits_begin - 1] <= '9') {
        digits_begin--;
    }
    // At least one digit, and room before it for the k/v letter, '_' and 'l' (three chars) that sit at or
    // after the end of "cache_": the bound keeps the look-behind reads below out of the prefix.
    if (digits_begin == len || digits_begin < prefix_len + 3) {
        return -1;
    }
    if (name[digits_begin - 1] != 'l' || name[digits_begin - 2] != '_') {
        return -1;
    }
    const char kind = name[digits_begin - 3];
    if (kind != 'k' && kind != 'v') {
        return -1;
    }

    long long layer = 0;
    for (size_t i = digits_begin; i < len; i++) {
        layer = layer * 10 + (name[i] - '0');
        if (layer > INT_MAX) {
            return -1;
        }
    }
    return static_cast<int>(layer);
}

}  // namespace ggml_sycl
