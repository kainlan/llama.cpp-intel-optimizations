//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-sycl.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ggml_sycl {

namespace detail {

static inline size_t cache_hash_combine(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

static inline bool cache_id_equal(const ggml_sycl_cache_id & a, const ggml_sycl_cache_id & b) {
    // GGUF-backed weights already carry stable file identity.  Do not include
    // model_id in that case: graph-local wrappers for the same loaded weight
    // can churn model_id while still needing to resolve to the same smart
    // mem_handle/cache entry.
    const bool compare_model_id = !(a.has_gguf && b.has_gguf);
    if (a.valid != b.valid || (compare_model_id && a.model_id != b.model_id) || a.has_gguf != b.has_gguf ||
        a.file_idx != b.file_idx || a.file_offs != b.file_offs || a.nbytes != b.nbytes || a.name_hash != b.name_hash ||
        a.type != b.type || a.tp_sharded != b.tp_sharded || a.tp_rank != b.tp_rank ||
        a.tp_world_size != b.tp_world_size || a.aux_id != b.aux_id) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a.ne[i] != b.ne[i] || a.tp_local_ne[i] != b.tp_local_ne[i] || a.tp_offset_ne[i] != b.tp_offset_ne[i]) {
            return false;
        }
    }
    return true;
}

struct cache_id_equal_fn {
    bool operator()(const ggml_sycl_cache_id & a, const ggml_sycl_cache_id & b) const { return cache_id_equal(a, b); }
};

struct cache_id_hash {
    size_t operator()(const ggml_sycl_cache_id & id) const {
        size_t h = 0;
        h        = cache_hash_combine(h, std::hash<bool>()(id.valid));
        h        = cache_hash_combine(h, std::hash<bool>()(id.has_gguf));
        if (!id.has_gguf) {
            h = cache_hash_combine(h, std::hash<uint64_t>()(id.model_id));
        }
        h = cache_hash_combine(h, std::hash<uint16_t>()(id.file_idx));
        h = cache_hash_combine(h, std::hash<size_t>()(id.file_offs));
        h = cache_hash_combine(h, std::hash<size_t>()(id.nbytes));
        h = cache_hash_combine(h, std::hash<uint64_t>()(id.name_hash));
        h = cache_hash_combine(h, std::hash<int>()(id.type));
        h = cache_hash_combine(h, std::hash<bool>()(id.tp_sharded));
        h = cache_hash_combine(h, std::hash<int>()(id.tp_rank));
        h = cache_hash_combine(h, std::hash<int>()(id.tp_world_size));
        h = cache_hash_combine(h, std::hash<uint64_t>()(id.aux_id));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            h = cache_hash_combine(h, std::hash<int64_t>()(id.ne[i]));
            h = cache_hash_combine(h, std::hash<int64_t>()(id.tp_local_ne[i]));
            h = cache_hash_combine(h, std::hash<int64_t>()(id.tp_offset_ne[i]));
        }
        return h;
    }
};

}  // namespace detail

// Type of cached entry.
enum class cache_entry_type {
    DENSE_WEIGHT,  // Regular weight tensor (attention, FFN, embeddings)
    MOE_EXPERT     // MoE expert weight
};

// Key for identifying a cached entry.
struct unified_cache_key {
    cache_entry_type   type;
    ggml_sycl_cache_id id;         // Identity for weights/MoE (no layout)
    int                layer_id;   // Layer ID (for expert identification)
    int                expert_id;  // Expert ID (-1 for dense weights)

    bool operator==(const unified_cache_key & other) const {
        return type == other.type && detail::cache_id_equal(id, other.id) && layer_id == other.layer_id &&
               expert_id == other.expert_id;
    }
};

struct unified_cache_key_hash {
    size_t operator()(const unified_cache_key & k) const {
        size_t h = 0;
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(k.type)));
        h        = detail::cache_hash_combine(h, detail::cache_id_hash{}(k.id));
        h        = detail::cache_hash_combine(h, std::hash<int>()(k.layer_id));
        h        = detail::cache_hash_combine(h, std::hash<int>()(k.expert_id));
        return h;
    }
};

struct layer_weight_set;
struct layer_weight_pointers;
struct unified_cache_entry;
struct alloc_handle;

}  // namespace ggml_sycl
