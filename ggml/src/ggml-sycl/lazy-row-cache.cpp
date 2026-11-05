//
// The hot-row cache of a STORAGE lazy table (th32 §6.3, §6.4). See lazy-row-cache.hpp.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "lazy-row-cache.hpp"

namespace ggml_sycl {

// A stub: no cache is built yet, so every build refuses and every operation finds nothing.
struct lazy_row_cache::shard {};

struct lazy_row_cache::chunk {};

const char * lazy_row_cache_status_name(lazy_row_cache_status status) {
    switch (status) {
        case lazy_row_cache_status::OK:
            return "ok";
        case lazy_row_cache_status::INVALID_CONFIG:
            return "invalid-config";
        case lazy_row_cache_status::SHORTFALL:
            return "shortfall";
    }
    return "unknown";
}

lazy_row_cache::lazy_row_cache() = default;

lazy_row_cache::~lazy_row_cache() = default;

lazy_row_cache_build lazy_row_cache::build(const lazy_row_cache_config &  config,
                                           lazy_row_cache_memory_source & source) {
    (void) config;
    (void) source;
    lazy_row_cache_build result;
    result.status = lazy_row_cache_status::SHORTFALL;
    result.reason = "no row cache is built";
    return result;
}

bool lazy_row_cache::lookup_copy(int64_t row, void * dst) {
    (void) row;
    (void) dst;
    return false;
}

bool lazy_row_cache::insert(int64_t row, const void * src) {
    (void) row;
    (void) src;
    return false;
}

size_t lazy_row_cache::yield(size_t deficit, uint64_t asking_load_txn_id) {
    (void) deficit;
    (void) asking_load_txn_id;
    return 0;
}

size_t lazy_row_cache::yield_entry(void * ctx, size_t deficit, uint64_t load_txn_id) noexcept {
    return static_cast<lazy_row_cache *>(ctx)->yield(deficit, load_txn_id);
}

void lazy_row_cache::publish_yieldable() {}

bool lazy_row_cache::resident(int64_t row) const {
    return chunk_holding(row) >= 0;
}

int64_t lazy_row_cache::chunk_holding(int64_t row) const {
    (void) row;
    return -1;
}

uint32_t lazy_row_cache::shard_of(int64_t row) const {
    (void) row;
    return 0;
}

lazy_row_cache_counters lazy_row_cache::counters() const {
    return {};
}

size_t lazy_row_cache::chunk_bytes(size_t chunk) const {
    (void) chunk;
    return 0;
}

size_t lazy_row_cache::shard_buckets(uint32_t shard) const {
    (void) shard;
    return 0;
}

size_t lazy_row_cache::shard_limit(uint32_t shard) const {
    (void) shard;
    return 0;
}

size_t lazy_row_cache::charged_bytes() const {
    return 0;
}

size_t lazy_row_cache::yieldable_bytes() const {
    return 0;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
void lazy_row_cache::test_set_hook(lazy_row_cache_test_hook hook, void * ctx) {
    (void) hook;
    (void) ctx;
}

bool lazy_row_cache::test_any_shard_held() const {
    return false;
}

void lazy_row_cache::test_fire(lazy_row_cache_test_point point, uint32_t shard) const {
    (void) point;
    (void) shard;
}
#endif

}  // namespace ggml_sycl
