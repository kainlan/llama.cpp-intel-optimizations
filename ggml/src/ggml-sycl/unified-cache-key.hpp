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

// ggml_sycl_cache_id's field list is written out by hand in FOUR places, and a
// field added to only three of them is a silent identity bug: two weights that
// differ solely in the new field compare equal and share one cache entry. There
// is no compiler diagnostic for that, so this tripwire stands in for one --
// adding, removing or resizing a field breaks the build until someone visits
// all four. Update the size once every site is correct.
//
// The four: cache_id_equal and cache_id_hash below; same_logical_moe_expert in
// unified-cache.cpp; retained_cache_id_less in cpu-dispatch.cpp.
//
// 216 is the size on the LP64 targets this backend builds for. On a platform
// with a different layout this fires with nothing wrong -- read the four sites,
// confirm they are complete, and record the new size.
static_assert(sizeof(ggml_sycl_cache_id) == 216,
              "ggml_sycl_cache_id changed: update cache_id_equal, cache_id_hash, "
              "same_logical_moe_expert (unified-cache.cpp) and retained_cache_id_less (cpu-dispatch.cpp)");

namespace detail {

static inline size_t cache_hash_combine(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

// A GGUF-backed weight IS the bytes it names on disk, so its identity is
// physical: (file_id, file_idx, file_offs, nbytes, type, ne) and the TP slice.
// Neither model_id nor name_hash participates -- both split an identity that is
// physically one.  Dropping model_id keeps graph-local wrappers, whose model_id
// churns across PP/TG, resolving to the same mem_handle; dropping name_hash is
// what lets a tied embedding/output head pair, or two models mapping the same
// file, share a single cache entry instead of staging the bytes twice.
//
// The safety of that sharing rests entirely on file_id being a WHOLE-FILE
// identity: file_idx alone is a split index within one model, so every model has
// a file_idx 0 and two unrelated weights at equal offsets would alias.  See
// ggml_backend_sycl_register_gguf_file_identity(); a split whose identity was
// never published falls back to a per-model file_id, which forfeits the sharing
// and keeps the isolation.
//
// Without GGUF identity there is no physical fact to key on, so model_id,
// name_hash and aux_id all stay in and such weights never share across models.
static inline bool cache_id_equal(const ggml_sycl_cache_id & a, const ggml_sycl_cache_id & b) {
    if (a.valid != b.valid || a.load_scoped != b.load_scoped || a.has_gguf != b.has_gguf) {
        return false;
    }
    if (a.load_scoped &&
        (a.model_id != b.model_id || a.load_txn_id != b.load_txn_id || a.model_slot != b.model_slot ||
         a.slot_generation != b.slot_generation)) {
        return false;
    }
    const bool compare_logical = !a.has_gguf;
    if (compare_logical && (a.model_id != b.model_id || a.name_hash != b.name_hash)) {
        return false;
    }
    if (a.file_id != b.file_id || a.file_idx != b.file_idx || a.file_offs != b.file_offs || a.nbytes != b.nbytes ||
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
        // Must hash exactly the fields cache_id_equal compares, and no others:
        // model_id and name_hash are excluded for GGUF-backed weights so that
        // two physically identical weights land in the same bucket.
        size_t h = 0;
        h        = cache_hash_combine(h, std::hash<bool>()(id.valid));
        h        = cache_hash_combine(h, std::hash<bool>()(id.load_scoped));
        h        = cache_hash_combine(h, std::hash<bool>()(id.has_gguf));
        if (id.load_scoped) {
            h = cache_hash_combine(h, std::hash<uint64_t>()(id.model_id));
            h = cache_hash_combine(h, std::hash<uint64_t>()(id.load_txn_id));
            h = cache_hash_combine(h, std::hash<uint32_t>()(id.model_slot));
            h = cache_hash_combine(h, std::hash<uint64_t>()(id.slot_generation));
        }
        if (!id.has_gguf) {
            h = cache_hash_combine(h, std::hash<uint64_t>()(id.model_id));
            h = cache_hash_combine(h, std::hash<uint64_t>()(id.name_hash));
        }
        h = cache_hash_combine(h, std::hash<uint64_t>()(id.file_id));
        h = cache_hash_combine(h, std::hash<uint16_t>()(id.file_idx));
        h = cache_hash_combine(h, std::hash<size_t>()(id.file_offs));
        h = cache_hash_combine(h, std::hash<size_t>()(id.nbytes));
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
