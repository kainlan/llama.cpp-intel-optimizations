//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "pinned-pool.hpp"

#include "common.hpp"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <future>
#include <memory>
#include <thread>

namespace ggml_sycl {
namespace {

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool host_zone_debug_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_HOST_ZONE_DEBUG");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    return enabled != 0;
}

bool pinned_trace_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_PINNED_TRACE");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    return enabled != 0;
}

size_t resolve_chunk_size() {
    const char * env = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
    if (!env || env[0] == '\0') {
        size_t chunk = pinned_chunk_pool::CHUNK_SIZE;
        // Use the default chunk size (2 GB). Chunks must be large enough to
        // hold individual weight tensors (MoE experts can be 2+ GB each).
        return chunk;
    }

    char * end = nullptr;
    long   mb  = std::strtol(env, &end, 10);
    if (end == env || mb <= 0) {
        GGML_LOG_WARN("[SYCL] Invalid GGML_SYCL_PINNED_CHUNK_MB='%s', using default chunk size\n", env);
        return pinned_chunk_pool::CHUNK_SIZE;
    }

    return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
}

size_t resolve_alloc_timeout_ms() {
    const char * env = std::getenv("GGML_SYCL_PINNED_ALLOC_TIMEOUT_MS");
    if (!env || env[0] == '\0') {
        return 0;
    }

    char * end = nullptr;
    long   ms  = std::strtol(env, &end, 10);
    if (end == env || ms <= 0) {
        GGML_LOG_WARN("[SYCL] Invalid GGML_SYCL_PINNED_ALLOC_TIMEOUT_MS='%s', disabling timeout\n", env);
        return 0;
    }

    return static_cast<size_t>(ms);
}

mem_handle allocate_pinned_chunk_owner(sycl::queue & queue, size_t chunk_size, bool runtime_pool) {
    alloc_request req{};
    req.queue                                    = &queue;
    req.device                                   = ggml_sycl_get_device_id_from_queue(queue);
    req.size                                     = chunk_size;
    req.intent.role                              = alloc_role::STAGING;
    req.intent.category                          = runtime_category::HOST_COMPUTE;
    req.intent.cohort_id                         = runtime_pool ? "pinned_chunk:runtime" : "pinned_chunk:base";
    req.intent.constraints.must_host_pinned      = true;
    // This pool is the backing arena for host-pinned suballocations.  Re-entering
    // host_pool_alloc() here would recurse, so request a standalone USM base
    // while still registering ownership through unified_alloc().
    req.intent.constraints.require_host_usm_base = true;

    mem_handle owner    = unified_allocate(req);
    auto       resolved = owner.resolve();
    if (!resolved.ptr || resolved.on_device) {
        return {};
    }
    return owner;
}

}  // namespace

const char * host_zone_name(host_zone_id zone) {
    switch (zone) {
        case host_zone_id::WEIGHT:
            return "WEIGHT";
        case host_zone_id::KV:
            return "KV";
        case host_zone_id::STAGING:
            return "STAGING";
        case host_zone_id::SCRATCH:
            return "SCRATCH";
        default:
            return "UNKNOWN";
    }
}

pinned_chunk_pool::pinned_chunk_pool(sycl::queue & queue, size_t budget) :
    queue_(queue),
    budget_(budget),
    chunk_size_(resolve_chunk_size()),
    alloc_timeout_ms_(resolve_alloc_timeout_ms()) {
    GGML_LOG_INFO("[SYCL] Pinned chunk pool created with %.1f GB budget, committed=0.0 GB (chunk=%.1f MB)\n",
                  budget / (1024.0 * 1024.0 * 1024.0), chunk_size_ / (1024.0 * 1024.0));
}

pinned_chunk_pool::~pinned_chunk_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      chunk_count = chunks_.size() + runtime_chunks_.size();

    // llama.cpp-dyhdl: wait for all chunk leases to drain before sycl::free.
    // If a lease is still held at destruction, a downstream caller holds a
    // raw pointer into this chunk and will SEGV on any subsequent use.
    // Assert-with-detail after 5 s rather than spin forever.
    for (auto & c : chunks_) {
        if (c.base) {
            wait_for_chunk_drain_or_assert(c, "pinned-base");
            free_chunk_owner(c, "pinned-base");
        }
    }
    for (auto & c : runtime_chunks_) {
        if (c.base) {
            wait_for_chunk_drain_or_assert(c, "pinned-runtime");
            free_chunk_owner(c, "pinned-runtime");
        }
    }
    chunks_.clear();
    runtime_chunks_.clear();
    total_allocated_ = 0;

    GGML_LOG_INFO("[SYCL] Pinned chunk pool destroyed, released %zu chunks\n", chunk_count);
}

segmented_buffer pinned_chunk_pool::allocate_segmented(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    segmented_buffer            result;
    result.total_size = size;

    size                    = align_up(size, alignment);
    const size_t tlsf_align = std::max(alignment, (size_t) 256);

    for (auto & c : chunks_) {
        size_t offset = c.allocator->allocate(size, tlsf_align);
        if (offset != SIZE_MAX) {
            void * ptr = static_cast<char *>(c.base) + offset;
            result.segments.push_back({ ptr, size });
            return result;
        }
    }

    // Need to grow - allocate a new chunk
    if (total_allocated_ + chunk_size_ > budget_) {
        return {};
    }

    if (!grow_into(chunks_, std::max(size, chunk_size_), false)) {
        return {};
    }

    auto & c      = chunks_.back();
    size_t offset = c.allocator->allocate(size, tlsf_align);
    if (offset != SIZE_MAX) {
        void * ptr = static_cast<char *>(c.base) + offset;
        result.segments.push_back({ ptr, size });
        return result;
    }
    return {};
}

void * pinned_chunk_pool::allocate(size_t size, size_t alignment) {
    segmented_buffer buf = allocate_segmented(size, alignment);
    return buf.segments.empty() ? nullptr : buf.segments[0].ptr;
}

void * pinned_chunk_pool::allocate_runtime(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocate_from_chunks(runtime_chunks_, size, alignment, true);
}

void pinned_chunk_pool::deallocate(void * ptr, size_t /* size */) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (deallocate_from_chunks(runtime_chunks_, ptr)) {
        return;
    }
    (void) deallocate_from_chunks(chunks_, ptr);
}

size_t pinned_chunk_pool::pre_allocate(size_t total_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate how many chunks are needed to cover total_bytes,
    // minus capacity already available in existing chunks.
    size_t existing_capacity = 0;
    for (const auto & c : chunks_) {
        existing_capacity += c.allocator ? c.allocator->available() : c.size;
    }
    if (existing_capacity >= total_bytes) {
        GGML_LOG_INFO("[SYCL] Pinned pool pre_allocate: already have %.1f MB free (need %.1f MB)\n",
                      existing_capacity / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0));
        return 0;
    }

    const size_t deficit       = total_bytes - existing_capacity;
    const size_t chunks_needed = (deficit + chunk_size_ - 1) / chunk_size_;
    size_t       chunks_grown  = 0;

    for (size_t i = 0; i < chunks_needed; i++) {
        if (total_allocated_ + chunk_size_ > budget_) {
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate: budget exhausted after %zu/%zu chunks\n", chunks_grown,
                          chunks_needed);
            break;
        }
        if (!grow(chunk_size_)) {
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate: grow failed at chunk %zu/%zu\n", chunks_grown,
                          chunks_needed);
            break;
        }
        chunks_grown++;
    }

    GGML_LOG_INFO("[SYCL] Pinned pool pre_allocate: grew %zu chunks for %.1f MB working set (total=%.1f GB)\n",
                  chunks_grown, total_bytes / (1024.0 * 1024.0), total_allocated_ / (1024.0 * 1024.0 * 1024.0));
    return chunks_grown;
}

size_t pinned_chunk_pool::pre_allocate_all(size_t model_weight_bytes) {
    constexpr double k_headroom_factor = 1.2;
    const size_t     total_need = static_cast<size_t>(static_cast<double>(model_weight_bytes) * k_headroom_factor);

    GGML_LOG_INFO(
        "[SYCL] Pinned pool pre_allocate_all: "
        "model=%.1f MB, total=%.1f MB (%.0f%% headroom)\n",
        model_weight_bytes / (1024.0 * 1024.0), total_need / (1024.0 * 1024.0), (k_headroom_factor - 1.0) * 100.0);

    return pre_allocate(total_need);
}

size_t pinned_chunk_pool::pre_allocate_runtime_chunks(size_t total_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_bytes == 0) {
        return 0;
    }

    size_t current_free = 0;
    for (const auto & c : runtime_chunks_) {
        current_free += c.allocator ? c.allocator->available() : c.size;
    }
    if (current_free >= total_bytes) {
        GGML_LOG_INFO("[SYCL] Pinned pool pre_allocate_runtime: already have %.1f MB free (need %.1f MB)\n",
                      current_free / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0));
        return 0;
    }

    size_t       need          = total_bytes - current_free;
    const size_t chunks_needed = (need + chunk_size_ - 1) / chunk_size_;
    size_t       chunks_grown  = 0;

    for (size_t i = 0; i < chunks_needed; ++i) {
        if (!grow_into(runtime_chunks_, chunk_size_, true)) {
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate_runtime: grow failed at chunk %zu/%zu\n", chunks_grown,
                          chunks_needed);
            break;
        }
        ++chunks_grown;
    }
    GGML_LOG_INFO("[SYCL] Pinned pool pre_allocate_runtime: grew %zu chunks for %.1f MB (total need %.1f MB)\n",
                  chunks_grown, chunks_grown * chunk_size_ / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0));
    return chunks_grown;
}

void pinned_chunk_pool::configure_zones(size_t weight_bytes,
                                        size_t kv_bytes,
                                        size_t staging_bytes,
                                        size_t scratch_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    weight_bytes  = align_up(weight_bytes, DEFAULT_ALIGNMENT);
    kv_bytes      = align_up(kv_bytes, DEFAULT_ALIGNMENT);
    staging_bytes = align_up(staging_bytes, DEFAULT_ALIGNMENT);
    scratch_bytes = align_up(scratch_bytes, DEFAULT_ALIGNMENT);

    const size_t total_zone_bytes = weight_bytes + kv_bytes + staging_bytes + scratch_bytes;

    // Grow chunks until we have enough total capacity.
    size_t total_capacity = 0;
    for (const auto & c : chunks_) {
        total_capacity += c.size;
    }
    while (total_capacity < total_zone_bytes && total_allocated_ + chunk_size_ <= budget_) {
        if (!grow(chunk_size_)) {
            break;
        }
        total_capacity += chunks_.back().size;
    }

    if (total_zone_bytes > total_capacity) {
        GGML_LOG_WARN(
            "[HOST-ARENA] pinned pool capacity %.1f MB is below requested zone footprint %.1f MB; "
            "disabling host zones and falling back to runtime pinned allocations\n",
            total_capacity / (1024.0 * 1024.0), total_zone_bytes / (1024.0 * 1024.0));
        for (auto & zone : zones_) {
            zone.size  = 0;
            zone.start = 0;
        }
        zones_configured_ = false;
        return;
    }

    // Build flat_spans_: each chunk maps to one span in the logical address space.
    flat_spans_.clear();
    size_t logical_cursor = 0;
    for (size_t ci = 0; ci < chunks_.size(); ci++) {
        flat_spans_.push_back({ logical_cursor, ci, 0ULL, chunks_[ci].size });
        logical_cursor += chunks_[ci].size;
    }

    // Assign zone sizes.
    auto set_zone = [&](host_zone_id zone_id, size_t sz) {
        auto & z = zones_[static_cast<size_t>(zone_id)];
        z.size   = sz;
    };
    set_zone(host_zone_id::WEIGHT, weight_bytes);
    set_zone(host_zone_id::KV, kv_bytes);
    set_zone(host_zone_id::STAGING, staging_bytes);
    set_zone(host_zone_id::SCRATCH, scratch_bytes);

    // Compute zone logical starts (contiguous layout).
    size_t cursor = 0;
    for (size_t z = 0; z < static_cast<size_t>(host_zone_id::COUNT); ++z) {
        zones_[z].start = cursor;
        cursor += zones_[z].size;
    }

    zones_configured_ = true;

    // Create per-zone-per-chunk TLSF allocators.
    for (size_t z = 0; z < static_cast<size_t>(host_zone_id::COUNT); ++z) {
        zone_allocators_[z].clear();
        auto & zone = zones_[z];
        if (zone.size == 0) {
            continue;
        }

        const size_t zone_logical_start = zone.start;
        const size_t zone_logical_end   = zone.start + zone.size;

        for (const auto & span : flat_spans_) {
            const size_t span_end = span.logical_start + span.span_size;
            if (span.logical_start >= zone_logical_end || span_end <= zone_logical_start) {
                continue;
            }

            const size_t overlap_start = std::max(span.logical_start, zone_logical_start);
            const size_t overlap_end   = std::min(span_end, zone_logical_end);
            const size_t overlap_size  = overlap_end - overlap_start;

            const size_t chunk_offset = span.chunk_start + (overlap_start - span.logical_start);

            zone_chunk_state zcs;
            zcs.chunk_idx  = span.chunk_idx;
            zcs.zone_start = chunk_offset;
            zcs.zone_size  = overlap_size;
            zcs.allocator  = std::make_unique<tlsf_allocator>(overlap_size);
            zone_allocators_[z].push_back(std::move(zcs));
        }
    }

    GGML_LOG_INFO(
        "[HOST-ARENA] configured zones: WEIGHT=%.1f MB  KV=%.1f MB  "
        "STAGING=%.1f MB  SCRATCH=%.1f MB  (total-cap=%.1f MB)\n",
        weight_bytes / (1024.0 * 1024.0), kv_bytes / (1024.0 * 1024.0), staging_bytes / (1024.0 * 1024.0),
        scratch_bytes / (1024.0 * 1024.0), total_capacity / (1024.0 * 1024.0));
}

segmented_buffer pinned_chunk_pool::zone_alloc_segmented(host_zone_id zone, size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    segmented_buffer            result;

    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return {};
    }

    // Early capacity check: reject if zone doesn't have enough total space
    const auto & z = zones_[zi];
    if (z.size > 0) {
        size_t cur_used = 0;
        for (const auto & zcs : zone_allocators_[zi]) {
            cur_used += zcs.allocator->used();
        }
        if (cur_used + size > z.size) {
            return {};
        }
    } else {
        return {};
    }

    const size_t tlsf_align = std::max(alignment, (size_t) 256);
    size_t       remaining  = size;

    for (auto & zcs : zone_allocators_[zi]) {
        if (remaining == 0) {
            break;
        }

        const size_t avail = zcs.allocator->largest_free_block();
        if (avail < tlsf_align) {
            continue;
        }

        size_t try_size = std::min(remaining, avail);
        // Round UP to TLSF alignment to avoid leaving a sub-alignment remainder
        // that would round to 0 on the next iteration and cause allocation failure.
        try_size        = (try_size + tlsf_align - 1) & ~(tlsf_align - 1);
        if (try_size == 0 || try_size > avail) {
            continue;
        }

        const size_t offset = zcs.allocator->allocate(try_size, tlsf_align);
        if (offset == SIZE_MAX) {
            continue;
        }

        // The rounded-up try_size may exceed remaining (e.g. remaining=100,
        // tlsf_align=256 → try_size=256).  Record only the bytes we need from
        // this segment so remaining -= consume never underflows.
        const size_t consume = std::min(try_size, remaining);
        void *       ptr     = static_cast<uint8_t *>(chunks_[zcs.chunk_idx].base) + zcs.zone_start + offset;
        result.segments.push_back({ ptr, consume });
        result.total_size += consume;
        remaining -= consume;
    }

    if (remaining > 0) {
        // Rollback: free what we already allocated in this call.
        for (auto & seg : result.segments) {
            zone_free_unlocked(zone, seg.ptr, seg.size);
        }
        result.segments.clear();
        result.total_size = 0;
        GGML_LOG_ERROR("[HOST-ZONE] segmented alloc failed: zone=%s size=%zu remaining=%zu\n", host_zone_name(zone),
                       size, remaining);
        return {};
    }
    return result;
}

void * pinned_chunk_pool::zone_alloc(host_zone_id zone, size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t                zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return nullptr;
    }

    size                    = align_up(size, alignment);
    const size_t tlsf_align = std::max(alignment, (size_t) 256);

    for (auto & zcs : zone_allocators_[zi]) {
        if (!zcs.allocator || zcs.allocator->largest_free_block() < size) {
            continue;
        }

        const size_t offset = zcs.allocator->allocate(size, tlsf_align);
        if (offset == SIZE_MAX) {
            continue;
        }

        return static_cast<uint8_t *>(chunks_[zcs.chunk_idx].base) + zcs.zone_start + offset;
    }

    return nullptr;
}

void pinned_chunk_pool::zone_free_unlocked(host_zone_id zone, void * ptr, size_t /* size */) {
    if (!ptr) {
        return;
    }
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return;
    }

    const uint8_t * p = static_cast<const uint8_t *>(ptr);
    for (auto & zcs : zone_allocators_[zi]) {
        const uint8_t * chunk_base = static_cast<const uint8_t *>(chunks_[zcs.chunk_idx].base);
        const uint8_t * zone_base  = chunk_base + zcs.zone_start;
        if (p >= zone_base && p < zone_base + zcs.zone_size) {
            const size_t offset = static_cast<size_t>(p - zone_base);
            zcs.allocator->free(offset);
            return;
        }
    }
    GGML_LOG_WARN("[HOST-ZONE] zone_free_unlocked: ptr %p not found in zone=%s\n", ptr, host_zone_name(zone));
}

void pinned_chunk_pool::zone_free(host_zone_id zone, void * ptr, size_t size) {
    if (!ptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    zone_free_unlocked(zone, ptr, size);
}

void pinned_chunk_pool::zone_reset(host_zone_id zone) {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & zcs : zone_allocators_[zi]) {
        zcs.allocator->reset();
    }
    if (host_zone_debug_enabled()) {
        GGML_LOG_INFO("[HOST-ZONE] reset zone=%s\n", host_zone_name(zone));
    }
}

size_t pinned_chunk_pool::zone_used(host_zone_id zone) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t                zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return 0;
    }
    size_t total = 0;
    for (const auto & zcs : zone_allocators_[zi]) {
        total += zcs.allocator->used();
    }
    return total;
}

size_t pinned_chunk_pool::zone_capacity(host_zone_id zone) const {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return 0;
    }
    return zones_[zi].size;
}

// Return the largest single-chunk contiguous free block currently available in
// the given zone.  Used by callers that must hand out a contiguous pointer and
// therefore cannot consume a cross-chunk (fragmented) allocation.
size_t pinned_chunk_pool::zone_largest_free_block(host_zone_id zone) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t                zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return 0;
    }
    size_t largest = 0;
    for (const auto & zcs : zone_allocators_[zi]) {
        if (!zcs.allocator) {
            continue;
        }
        const size_t block = zcs.allocator->largest_free_block();
        if (block > largest) {
            largest = block;
        }
    }
    return largest;
}

bool pinned_chunk_pool::grow_zone(host_zone_id zone, size_t additional_bytes) {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return false;
    }

    // Phase gate: early rejection without holding mutex.
    // Note: this is a best-effort check — the phase could transition between
    // this check and the actual grow() call below. Match grow_into(): default
    // mode warns only, while mode >= 2 is the strict audit path.
    static std::atomic<int> s_phase_gate{ -1 };
    int                     mode = s_phase_gate.load(std::memory_order_relaxed);
    if (mode < 0) {
        const char * env = std::getenv("GGML_SYCL_HOST_ALLOC_PHASE_GATE");
        mode             = (env != nullptr) ? std::atoi(env) : 1;
        s_phase_gate.store(mode, std::memory_order_relaxed);
    }
    if (mode > 0) {
        const auto phase = ggml_sycl::offload_stats_phase();
        if (phase == ggml_sycl::offload_phase::PP || phase == ggml_sycl::offload_phase::TG) {
            GGML_LOG_WARN("[HOST-POOL] grow_zone(%s, %.1f MB) during %s phase; planner should pre-size\n",
                          host_zone_name(zone), additional_bytes / (1024.0 * 1024.0),
                          ggml_sycl::offload_phase_name(phase));
            if (mode >= 2) {
                GGML_ASSERT(false && "host pool zone growth during inference");
                return false;
            }
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate how many chunks we need to add
    size_t chunks_needed = (additional_bytes + chunk_size_ - 1) / chunk_size_;
    if (chunks_needed == 0) {
        return true;
    }

    // Check budget before allocating
    if (total_allocated_ + chunks_needed * chunk_size_ > budget_) {
        GGML_LOG_WARN(
            "[SYCL] Pinned pool grow_zone: budget exhausted for zone %zu "
            "(need %zu chunks, budget=%.1f GB, used=%.1f GB)\n",
            zi, chunks_needed, budget_ / (1024.0 * 1024.0 * 1024.0), total_allocated_ / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    // Record the old end of the pool (where new chunks will start)
    size_t old_chunk_count = chunks_.size();

    // Allocate new chunks
    for (size_t i = 0; i < chunks_needed; i++) {
        if (!grow(chunk_size_)) {
            GGML_LOG_WARN("[SYCL] Pinned pool grow_zone: grow failed at chunk %zu/%zu\n", i, chunks_needed);
            break;
        }
    }

    size_t new_chunks_added = chunks_.size() - old_chunk_count;
    if (new_chunks_added == 0) {
        return false;
    }
    if (new_chunks_added < chunks_needed) {
        GGML_LOG_WARN("[HOST-POOL] grow_zone: partial growth %zu/%zu chunks for zone %zu\n", new_chunks_added,
                      chunks_needed, zi);
    }

    // Compute the logical cursor for the new chunks (just past the existing logical space).
    size_t logical_cursor = 0;
    for (const auto & span : flat_spans_) {
        logical_cursor = std::max(logical_cursor, span.logical_start + span.span_size);
    }

    // Extend flat_spans_ and add per-zone TLSF allocators for the new chunks.
    for (size_t i = old_chunk_count; i < chunks_.size(); i++) {
        flat_spans_.push_back({ logical_cursor, i, 0ULL, chunks_[i].size });
        logical_cursor += chunks_[i].size;

        zone_chunk_state zcs;
        zcs.chunk_idx  = i;
        zcs.zone_start = 0;
        zcs.zone_size  = chunks_[i].size;
        zcs.allocator  = std::make_unique<tlsf_allocator>(chunks_[i].size);
        zone_allocators_[zi].push_back(std::move(zcs));
    }

    // Extend the zone's capacity to include the new chunks.
    size_t additional_capacity = new_chunks_added * chunk_size_;
    zones_[zi].size += additional_capacity;

    GGML_LOG_INFO(
        "[SYCL] Pinned pool grow_zone: zone %zu grown by %.1f MB "
        "(%zu new chunks, zone now %.1f MB)\n",
        zi, additional_capacity / (1024.0 * 1024.0), new_chunks_added, zones_[zi].size / (1024.0 * 1024.0));
    return true;
}

bool pinned_chunk_pool::contains(const void * ptr) const {
    if (!ptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const char *                p = static_cast<const char *>(ptr);
    for (const auto & c : chunks_) {
        const char * base = static_cast<const char *>(c.base);
        if (p >= base && p < base + c.size) {
            return true;
        }
    }
    for (const auto & c : runtime_chunks_) {
        const char * base = static_cast<const char *>(c.base);
        if (p >= base && p < base + c.size) {
            return true;
        }
    }
    return false;
}

size_t pinned_chunk_pool::allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

size_t pinned_chunk_pool::chunk_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return chunks_.size() + runtime_chunks_.size();
}

void pinned_chunk_pool::free_chunk_owner(chunk & c, const char * ctx) {
    if (!c.base) {
        return;
    }
    if (c.owner.valid()) {
        c.owner = {};
        c.base  = nullptr;
        return;
    }

    GGML_LOG_ERROR("[PINNED-POOL] %s chunk %p missing unified owner size=%.1f MB\n", ctx, c.base,
                   c.size / (1024.0 * 1024.0));
    GGML_ASSERT(false && "pinned chunk missing unified allocation owner");
    c.base = nullptr;
}

bool pinned_chunk_pool::grow(size_t min_size) {
    return grow_into(chunks_, min_size, false);
}

void * pinned_chunk_pool::allocate_from_chunks(std::vector<chunk> & chunks,
                                               size_t               size,
                                               size_t               alignment,
                                               bool                 runtime_pool) {
    size                    = align_up(size, alignment);
    const size_t tlsf_align = std::max(alignment, (size_t) 256);

    if (pinned_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] pinned alloc request: size=%zu align=%zu chunks=%zu used=%.1f GB mode=%s\n", size,
                      alignment, chunks.size(), total_allocated_ / (1024.0 * 1024.0 * 1024.0),
                      runtime_pool ? "runtime" : "zone-base");
    }

    for (auto & c : chunks) {
        size_t offset = c.allocator->allocate(size, tlsf_align);
        if (offset != SIZE_MAX) {
            return static_cast<char *>(c.base) + offset;
        }
    }

    size_t new_chunk_size = std::max(chunk_size_, align_up(size, DEFAULT_ALIGNMENT));
    if (total_allocated_ + new_chunk_size > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    if (pinned_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] pinned pool grow: min=%zu chunk=%zu total=%.1f GB budget=%.1f GB mode=%s\n", size,
                      new_chunk_size, total_allocated_ / (1024.0 * 1024.0 * 1024.0),
                      budget_ / (1024.0 * 1024.0 * 1024.0), runtime_pool ? "runtime" : "zone-base");
    }
    if (!grow_into(chunks, size, runtime_pool)) {
        return nullptr;
    }

    auto & c      = chunks.back();
    size_t offset = c.allocator->allocate(size, tlsf_align);
    if (offset != SIZE_MAX) {
        return static_cast<char *>(c.base) + offset;
    }
    GGML_LOG_ERROR("[SYCL] Pinned pool TLSF allocation %zu MB failed in new chunk %zu MB.\n", size / (1024 * 1024),
                   c.size / (1024 * 1024));
    return nullptr;
}

bool pinned_chunk_pool::deallocate_from_chunks(std::vector<chunk> & chunks, void * ptr) {
    char * p = static_cast<char *>(ptr);
    for (auto & c : chunks) {
        char * base = static_cast<char *>(c.base);
        if (p >= base && p < base + c.size) {
            size_t offset = p - base;
            c.allocator->free(offset);
            return true;
        }
    }
    return false;
}

bool pinned_chunk_pool::grow_into(std::vector<chunk> & chunks, size_t min_size, bool runtime_pool) {
    // Phase gate: warn/assert when allocating chunks during inference
    static std::atomic<int> s_phase_gate{ -1 };
    int                     mode = s_phase_gate.load(std::memory_order_relaxed);
    if (mode < 0) {
        const char * env = std::getenv("GGML_SYCL_HOST_ALLOC_PHASE_GATE");
        mode             = (env != nullptr) ? std::atoi(env) : 1;
        s_phase_gate.store(mode, std::memory_order_relaxed);
    }
    if (mode > 0) {
        const auto phase = ggml_sycl::offload_stats_phase();
        if (phase == ggml_sycl::offload_phase::PP || phase == ggml_sycl::offload_phase::TG) {
            GGML_LOG_WARN("[HOST-POOL] chunk allocation during %s phase — planner should pre-size (%zu bytes)\n",
                          ggml_sycl::offload_phase_name(phase), min_size);
            if (mode >= 2) {
                GGML_ASSERT(false && "host pool chunk allocation during inference");
            }
        }
    }

    // Use chunk_size_ as the default, but allow larger chunks when the
    // allocation exceeds chunk_size_ (e.g., 615 MB reorder buffers for
    // MoE models).  Level Zero's ~11 GB per-allocation limit is the
    // real cap, not chunk_size_.
    size_t usable_size = align_up(min_size, DEFAULT_ALIGNMENT);
    if (usable_size < chunk_size_) {
        usable_size = chunk_size_;
    }
    const size_t backing_size = usable_size + DEFAULT_ALIGNMENT;
    if (total_allocated_ + backing_size > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    if (alloc_timeout_ms_ > 0) {
        if (pinned_trace_enabled()) {
            GGML_LOG_INFO("[SYCL] pinned chunk malloc_host begin: size=%zu timeout=%zu ms\n", backing_size,
                          alloc_timeout_ms_);
        }
        auto future = std::async(std::launch::async, [&, backing_size]() {
            return allocate_pinned_chunk_owner(queue_, backing_size, runtime_pool);
        });

        const auto status = future.wait_for(std::chrono::milliseconds(alloc_timeout_ms_));
        if (status != std::future_status::ready) {
            GGML_LOG_ERROR("[SYCL] Pinned chunk allocation timed out after %zu ms (size=%zu). Aborting.\n",
                           alloc_timeout_ms_, backing_size);
            std::fflush(stderr);
            std::abort();
        }

        try {
            auto owner    = future.get();
            auto resolved = owner.resolve();
            if (!resolved.ptr) {
                GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes, nullptr)\n", backing_size);
                return false;
            }
            chunks.emplace_back();
            chunks.back().base      = static_cast<uint8_t *>(resolved.ptr) + DEFAULT_ALIGNMENT;
            chunks.back().size      = usable_size;
            chunks.back().allocator = std::make_unique<tlsf_allocator>(usable_size);
            chunks.back().owner     = std::move(owner);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes): %s\n", backing_size, e.what());
            return false;
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes): %s\n", backing_size, e.what());
            return false;
        }
    } else {
        if (pinned_trace_enabled()) {
            GGML_LOG_INFO("[SYCL] pinned chunk malloc_host begin: size=%zu\n", backing_size);
        }
        try {
            auto owner    = allocate_pinned_chunk_owner(queue_, backing_size, runtime_pool);
            auto resolved = owner.resolve();
            if (!resolved.ptr) {
                GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes, nullptr)\n", backing_size);
                return false;
            }
            chunks.emplace_back();
            chunks.back().base      = static_cast<uint8_t *>(resolved.ptr) + DEFAULT_ALIGNMENT;
            chunks.back().size      = usable_size;
            chunks.back().allocator = std::make_unique<tlsf_allocator>(usable_size);
            chunks.back().owner     = std::move(owner);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[SYCL] Failed to allocate pinned chunk (%zu bytes): %s\n", backing_size, e.what());
            return false;
        }
    }

    void * ptr = chunks.back().base;
    if (pinned_trace_enabled()) {
        const sycl::usm::alloc alloc_type = ggml_sycl_get_alloc_type(ptr);
        const char *           alloc_name = alloc_type == sycl::usm::alloc::host   ? "host" :
                                            alloc_type == sycl::usm::alloc::shared ? "shared" :
                                            alloc_type == sycl::usm::alloc::device ? "device" :
                                                                                     "unknown";
        GGML_LOG_INFO("[SYCL] pinned chunk malloc_host ok: ptr=%p type=%s size=%.1f MB\n", ptr, alloc_name,
                      backing_size / (1024.0 * 1024.0));
    }

    total_allocated_ += backing_size;

    GGML_LOG_INFO("[SYCL] Allocated pinned %s chunk %zu (size=%.1f MB, total=%.1f GB)\n",
                  runtime_pool ? "runtime" : "base", chunks.size(), usable_size / (1024.0 * 1024.0),
                  total_allocated_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

// === Chunk lease API (llama.cpp-dyhdl) ===
//
// Raw-pointer handouts from this pool (e.g. tensor->data = chunk.base + off)
// outlive any per-allocation handle lifetime.  Without a refcount at the chunk
// layer, a chunk can in principle be sycl::free'd while a tensor still holds
// a derived raw pointer — this is the vtf7f §9 crash class.  Lease acquire
// bumps a per-chunk atomic; the destruction gate waits for the count to drop
// to 0 before freeing.  Non-pool pointers return INVALID_CHUNK_HANDLE which
// is a safe no-op on release — callers treat it as "no protection needed."

uint64_t pinned_chunk_pool::find_chunk_handle(const void * ptr) const {
    if (!ptr) {
        return INVALID_CHUNK_HANDLE;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const char *                p = static_cast<const char *>(ptr);
    for (size_t i = 0; i < chunks_.size(); ++i) {
        const char * base = static_cast<const char *>(chunks_[i].base);
        if (base && p >= base && p < base + chunks_[i].size) {
            return encode_chunk_handle(false, i);
        }
    }
    for (size_t i = 0; i < runtime_chunks_.size(); ++i) {
        const char * base = static_cast<const char *>(runtime_chunks_[i].base);
        if (base && p >= base && p < base + runtime_chunks_[i].size) {
            return encode_chunk_handle(true, i);
        }
    }
    return INVALID_CHUNK_HANDLE;
}

uint64_t pinned_chunk_pool::acquire_chunk_lease(const void * ptr) {
    if (!ptr) {
        return INVALID_CHUNK_HANDLE;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const char *                p = static_cast<const char *>(ptr);
    for (size_t i = 0; i < chunks_.size(); ++i) {
        const char * base = static_cast<const char *>(chunks_[i].base);
        if (base && p >= base && p < base + chunks_[i].size) {
            chunks_[i].lease_count.fetch_add(1);
            return encode_chunk_handle(false, i);
        }
    }
    for (size_t i = 0; i < runtime_chunks_.size(); ++i) {
        const char * base = static_cast<const char *>(runtime_chunks_[i].base);
        if (base && p >= base && p < base + runtime_chunks_[i].size) {
            runtime_chunks_[i].lease_count.fetch_add(1);
            return encode_chunk_handle(true, i);
        }
    }
    return INVALID_CHUNK_HANDLE;
}

void pinned_chunk_pool::release_chunk_lease(uint64_t handle) {
    if (handle == INVALID_CHUNK_HANDLE) {
        return;
    }
    const bool   is_runtime = (handle & CHUNK_KIND_RUNTIME_BIT) != 0;
    const size_t idx        = static_cast<size_t>(handle & ~CHUNK_KIND_RUNTIME_BIT);

    std::lock_guard<std::mutex> lock(mutex_);
    auto &                      vec = is_runtime ? runtime_chunks_ : chunks_;
    if (idx >= vec.size()) {
        GGML_LOG_ERROR("[PINNED-POOL] release_chunk_lease: invalid handle=0x%" PRIx64 " (idx=%zu, size=%zu)\n", handle,
                       idx, vec.size());
        return;
    }
    const uint32_t prev = vec[idx].lease_count.fetch_sub(1);
    if (prev == 0) {
        GGML_LOG_ERROR("[PINNED-POOL] release_chunk_lease: underflow on chunk idx=%zu runtime=%d\n", idx, is_runtime);
        // Restore to avoid wrap-around.
        vec[idx].lease_count.fetch_add(1);
    }
}

bool pinned_chunk_pool::chunk_has_leases(uint64_t handle) const {
    if (handle == INVALID_CHUNK_HANDLE) {
        return false;
    }
    const bool   is_runtime = (handle & CHUNK_KIND_RUNTIME_BIT) != 0;
    const size_t idx        = static_cast<size_t>(handle & ~CHUNK_KIND_RUNTIME_BIT);

    std::lock_guard<std::mutex> lock(mutex_);
    const auto &                vec = is_runtime ? runtime_chunks_ : chunks_;
    if (idx >= vec.size()) {
        return false;
    }
    return vec[idx].lease_count.load() > 0;
}

void pinned_chunk_pool::wait_for_chunk_drain_or_assert(const chunk & c, const char * ctx) const {
    // Caller (destructor) already holds mutex_.  We read lease_count
    // lock-free — writers (acquire/release) acquire the mutex before
    // mutating the vector, but the atomic count itself is observable
    // without the lock.
    const auto deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    uint32_t   outstanding = c.lease_count.load();
    if (outstanding == 0) {
        return;
    }
    GGML_LOG_WARN("[PINNED-POOL] %s chunk %p size=%.1f MB has %u outstanding lease(s) at destruction, waiting ≤5s\n",
                  ctx, c.base, c.size / (1024.0 * 1024.0), outstanding);
    while ((outstanding = c.lease_count.load()) > 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            GGML_LOG_ERROR(
                "[PINNED-POOL] %s chunk %p size=%.1f MB still has %u outstanding lease(s) after 5s — aborting to "
                "surface the bug (llama.cpp-dyhdl)\n",
                ctx, c.base, c.size / (1024.0 * 1024.0), outstanding);
            std::fflush(stderr);
            GGML_ASSERT(false && "pinned chunk freed while leases outstanding (dyhdl timeout)");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace ggml_sycl
