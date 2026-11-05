//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "pinned-pool.hpp"

#include "allocation-provenance.hpp"
#include "common.hpp"
#include "ggml-impl.h"
#include "scratch-test-mutant.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <future>
#include <memory>
#include <new>
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

mem_handle allocate_pinned_chunk_owner(sycl::queue &       queue,
                                       size_t              chunk_size,
                                       bool                runtime_pool,
                                       cache_backing_token backing) {
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

    // CACHE_BACKING comes from the private token, not from anything in req: a
    // request field would let any caller claim to be a cache's backing arena.
    allocation_result allocation = unified_allocate_owner_backing(req, backing);
    if (!allocation) {
        return {};
    }
    mem_handle owner    = mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);
    auto       resolved = owner.resolve();
    if (!resolved.ptr || resolved.on_device) {
        return {};
    }
    return owner;
}

}  // namespace

struct pinned_chunk_pool::stranded_entry {
    alloc_metadata metadata;
    size_t         backing = 0;
};

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
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::CARRIER_ASSERT_REMOVED)) {
        owning_carriers_.store(0, std::memory_order_release);
    }
#endif
    GGML_ASSERT(owning_carriers_.load(std::memory_order_acquire) == 0 &&
                "released_chunk outstanding at pool destruction");
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      chunk_count = 0;

    // llama.cpp-dyhdl: wait for all chunk leases to drain before sycl::free.
    // If a lease is still held at destruction, a downstream caller holds a
    // raw pointer into this chunk and will SEGV on any subsequent use.
    // Assert-with-detail after 5 s rather than spin forever.
    for (auto & c : chunks_) {
        if (c.base) {
            wait_for_chunk_drain_or_assert(c, "pinned-base");
            free_chunk_owner(c, "pinned-base");
            ++chunk_count;
        }
    }
    for (auto & c : runtime_chunks_) {
        if (c.base) {
            wait_for_chunk_drain_or_assert(c, "pinned-runtime");
            free_chunk_owner(c, "pinned-runtime");
            ++chunk_count;
        }
    }
    // A released backing the registry still holds now (a control the registry abandoned at shutdown, or a retry not
    // yet run) is reported here: after this the pool counts nothing.
    reconcile_stranded_locked();
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::DTOR_NO_REPORT)) {
        stranded_.clear();
    }
#endif
    for (const auto & entry : stranded_) {
        GGML_LOG_WARN("[PINNED-POOL] stranded chunk backing %.1f MB still registered at pool destruction\n",
                      entry.backing / (1024.0 * 1024.0));
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
        if (!c.base) {
            continue;
        }
        size_t offset = c.allocator->allocate(size, tlsf_align);
        if (offset != SIZE_MAX) {
            void * ptr = static_cast<char *>(c.base) + offset;
            result.segments.push_back({ ptr, size });
            return result;
        }
    }

    // Need to grow - allocate a new chunk
    if (footprint_locked() + chunk_size_ > budget_) {
        return {};
    }

    if (!grow_into(chunks_, std::max(size, chunk_size_), false, "allocate_segmented")) {
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
        if (!c.base) {
            continue;
        }
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
        if (footprint_locked() + chunk_size_ > budget_) {
            GGML_LOG_WARN("[SYCL] Pinned pool pre_allocate: budget exhausted after %zu/%zu chunks\n", chunks_grown,
                          chunks_needed);
            break;
        }
        if (!grow(chunk_size_, "pre_allocate")) {
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
        if (!grow_into(runtime_chunks_, chunk_size_, true, "pre_allocate_runtime")) {
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

bool pinned_chunk_pool::configure_zones(size_t weight_bytes,
                                        size_t kv_bytes,
                                        size_t staging_bytes,
                                        size_t scratch_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    // A surviving withdrawn chunk refuses the rebuild before any state changes (§4.4 "configure_zones refuses while a
    // withdrawn chunk survives"): the rebuild below would drop its allocator, and the live allocation in it, while the
    // device cache's marked set still names the chunk.
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::CONFIGURE_REBUILDS)) {
        for (auto & states : zone_allocators_) {
            for (auto & zcs : states) {
                zcs.withdrawn = false;
            }
        }
    }
#endif
    for (const auto & states : zone_allocators_) {
        for (const auto & zcs : states) {
            if (zcs.withdrawn) {
                GGML_LOG_ERROR("[HOST-ARENA] configure_zones refused: chunk %zu still withdrawn, %zu B allocated\n",
                               zcs.chunk_idx, zcs.allocator->used());
                return false;
            }
        }
    }

    weight_bytes  = align_up(weight_bytes, DEFAULT_ALIGNMENT);
    kv_bytes      = align_up(kv_bytes, DEFAULT_ALIGNMENT);
    staging_bytes = align_up(staging_bytes, DEFAULT_ALIGNMENT);
    scratch_bytes = align_up(scratch_bytes, DEFAULT_ALIGNMENT);

    const size_t total_zone_bytes = weight_bytes + kv_bytes + staging_bytes + scratch_bytes;

    // Grow chunks until we have enough total capacity.
    size_t total_capacity = 0;
    for (const auto & c : chunks_) {
        if (!c.base) {
            continue;
        }
        total_capacity += c.size;
    }
    while (total_capacity < total_zone_bytes && footprint_locked() + chunk_size_ <= budget_) {
        if (!grow(chunk_size_, "configure_zones")) {
            break;
        }
        total_capacity += chunks_.back().size;
    }

    if (total_zone_bytes > total_capacity) {
        // llama.cpp-glkg: name every term of the arithmetic that produced this
        // refusal, not just the two the old message compared. `total_capacity`
        // above sums `chunks_` ONLY, but the growth guard just above
        // (`total_allocated_ + chunk_size_ <= budget_`) reads
        // `total_allocated_`, which also counts `runtime_chunks_` -- bytes
        // already committed by the PRE-ZONE fallback (see the
        // "[HOST-ARENA] runtime pinned chunk" WARN in grow_into() above). A
        // reader who only sees "capacity 0.0 MB" next to a multi-GB budget
        // cannot tell those two accumulators apart; this line makes the
        // budget, the zone-eligible chunk bytes, the runtime-chunk bytes
        // already spent, and the requested footprint all explicit.
        size_t runtime_capacity = 0;
        for (const auto & c : runtime_chunks_) {
            if (!c.base) {
                continue;
            }
            runtime_capacity += c.size;
        }
        GGML_LOG_WARN(
            "[HOST-ARENA] pinned pool capacity %.1f MB is below requested zone footprint %.1f MB "
            "(budget=%.1f MB, zone-chunk bytes=%.1f MB, runtime-chunk bytes=%.1f MB, total committed=%.1f MB); "
            "disabling host zones and falling back to runtime pinned allocations\n",
            total_capacity / (1024.0 * 1024.0), total_zone_bytes / (1024.0 * 1024.0), budget_ / (1024.0 * 1024.0),
            total_capacity / (1024.0 * 1024.0), runtime_capacity / (1024.0 * 1024.0),
            total_allocated_ / (1024.0 * 1024.0));
        for (auto & zone : zones_) {
            zone.size  = 0;
            zone.start = 0;
        }
        zones_configured_ = false;
        return true;
    }

    // Build flat_spans_: each chunk maps to one span in the logical address space.
    flat_spans_.clear();
    size_t logical_cursor = 0;
    for (size_t ci = 0; ci < chunks_.size(); ci++) {
        if (!chunks_[ci].base) {
            continue;
        }
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
    return true;
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
        // Like with like: z.size excludes a withdrawn chunk, so its bytes are not summed either.
        size_t cur_used = 0;
        for (const auto & zcs : zone_allocators_[zi]) {
            if (zcs.withdrawn) {
                continue;
            }
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
        if (zcs.withdrawn) {
            continue;
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
            // The segments came from chunks that are not withdrawn, so none is an orphaned chunk's.
            (void) zone_free_unlocked(zone, seg.ptr, seg.size);
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
        if (zcs.withdrawn || !zcs.allocator || zcs.allocator->largest_free_block() < size) {
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

size_t pinned_chunk_pool::zone_free_unlocked(host_zone_id zone, void * ptr, size_t /* size */) {
    if (!ptr) {
        return NO_CHUNK;
    }
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return NO_CHUNK;
    }

    const uint8_t * p = static_cast<const uint8_t *>(ptr);
    for (auto & zcs : zone_allocators_[zi]) {
        const uint8_t * chunk_base = static_cast<const uint8_t *>(chunks_[zcs.chunk_idx].base);
        const uint8_t * zone_base  = chunk_base + zcs.zone_start;
        if (p >= zone_base && p < zone_base + zcs.zone_size) {
            const size_t offset = static_cast<size_t>(p - zone_base);
            zcs.allocator->free(offset);
            // The orphaned chunk's first release event (§4.4 "Where the two flags live"): a flag test on the state
            // this free already found, so a free into any other chunk gains no lookup.
            return (zcs.orphaned && zcs.allocator->used() == 0) ? zcs.chunk_idx : NO_CHUNK;
        }
    }
    GGML_LOG_WARN("[HOST-ZONE] zone_free_unlocked: ptr %p not found in zone=%s\n", ptr, host_zone_name(zone));
    return NO_CHUNK;
}

size_t pinned_chunk_pool::zone_free(host_zone_id zone, void * ptr, size_t size) {
    if (!ptr) {
        return NO_CHUNK;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return zone_free_unlocked(zone, ptr, size);
}

void pinned_chunk_pool::zone_reset(host_zone_id zone) {
    const size_t zi = static_cast<size_t>(zone);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & zcs : zone_allocators_[zi]) {
        // A withdrawn chunk may still hold a returned buffer's live allocation; resetting its allocator would let the
        // release free memory a live handle owns.
        if (zcs.withdrawn) {
            continue;
        }
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
        if (zcs.withdrawn) {
            continue;
        }
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
        if (zcs.withdrawn || !zcs.allocator) {
            continue;
        }
        const size_t block = zcs.allocator->largest_free_block();
        if (block > largest) {
            largest = block;
        }
    }
    return largest;
}

bool pinned_chunk_pool::grow_zone(host_zone_id zone, size_t additional_bytes, size_t * grown_chunk) {
    if (grown_chunk) {
        *grown_chunk = NO_CHUNK;
    }
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
    // A thread inside a TRANSACTION-kind re-plan token is the planner pre-sizing
    // (a context's first publish carves its host slots here), so the gate does not
    // apply to it. LOAD and LIFECYCLE tokens are NOT exempt: a load's pinned fills
    // stay gated, and the kind is read before the phase.
    if (mode > 0 && !ggml_sycl::ggml_sycl_replan_token_held(ggml_sycl::GGML_SYCL_REPLAN_KIND_TRANSACTION)) {
        const auto phase = ggml_sycl::offload_stats_phase();
        if (phase == ggml_sycl::offload_phase::PP || phase == ggml_sycl::offload_phase::TG) {
            GGML_LOG_WARN(
                "[HOST-POOL] grow_zone(%s, %.1f MB) during %s phase site=grow_zone; planner should pre-size\n",
                host_zone_name(zone), additional_bytes / (1024.0 * 1024.0), ggml_sycl::offload_phase_name(phase));
            if (mode >= 2) {
                GGML_ASSERT(false && "host pool zone growth during inference");
                return false;
            }
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (additional_bytes == 0) {
        return true;
    }

    // llama.cpp-nsl3: grow by ONE chunk sized to the request. zone_alloc()
    // hands out a contiguous pointer from a single chunk's TLSF, so a request
    // larger than chunk_size_ can only be served by a chunk at least that
    // large; N fixed chunk_size_ chunks add aggregate capacity that no
    // single-chunk allocation can consume (Gemma 4 E4B: a 2856 MB host-placed
    // embedding against 2048 MB chunks grew the WEIGHT zone by two more
    // 2048 MB chunks and still failed model load). grow() already sizes a
    // chunk as max(chunk_size_, align_up(min_size)) -- the rule the runtime
    // path (allocate_from_chunks) has always used -- so the zone path follows
    // it, and the budget is checked against the bytes that chunk actually
    // commits (grow_into() adds one DEFAULT_ALIGNMENT header per chunk).
    const chunk_footprint footprint = chunk_footprint_for(additional_bytes);
    if (footprint_locked() + footprint.backing > budget_) {
        GGML_LOG_WARN(
            "[SYCL] Pinned pool grow_zone: budget exhausted for zone %zu "
            "(need %.1f MB, budget=%.1f GB, used=%.1f GB)\n",
            zi, footprint.backing / (1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0),
            total_allocated_ / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    // Record the old end of the pool (where the new chunk will start). grow()
    // appends exactly one chunk on success and none on failure, so the loop
    // below visits that one chunk.
    const size_t old_chunk_count = chunks_.size();

    if (!grow(footprint.usable, "grow_zone")) {
        GGML_LOG_WARN("[SYCL] Pinned pool grow_zone: grow failed for zone %zu (%.1f MB chunk)\n", zi,
                      footprint.usable / (1024.0 * 1024.0));
        return false;
    }

    // Compute the logical cursor for the new chunks (just past the existing logical space).
    size_t logical_cursor = 0;
    for (const auto & span : flat_spans_) {
        logical_cursor = std::max(logical_cursor, span.logical_start + span.span_size);
    }

    // Extend flat_spans_ and add per-zone TLSF allocators for the new chunks,
    // accumulating the zone's added capacity from the chunks actually added
    // rather than from chunk_size_, which under-reports an oversize chunk.
    size_t additional_capacity = 0;
    for (size_t i = old_chunk_count; i < chunks_.size(); i++) {
        flat_spans_.push_back({ logical_cursor, i, 0ULL, chunks_[i].size });
        logical_cursor += chunks_[i].size;

        zone_chunk_state zcs;
        zcs.chunk_idx  = i;
        zcs.zone_start = 0;
        zcs.zone_size  = chunks_[i].size;
        zcs.allocator  = std::make_unique<tlsf_allocator>(chunks_[i].size);
        zone_allocators_[zi].push_back(std::move(zcs));
        additional_capacity += chunks_[i].size;
    }

    // Extend the zone's capacity to include the new chunks.
    zones_[zi].size += additional_capacity;
    if (grown_chunk && chunks_.size() > old_chunk_count) {
        *grown_chunk = old_chunk_count;
    }

    GGML_LOG_INFO(
        "[SYCL] Pinned pool grow_zone: zone %zu grown by %.1f MB "
        "(%zu new chunks, zone now %.1f MB)\n",
        zi, additional_capacity / (1024.0 * 1024.0), chunks_.size() - old_chunk_count,
        zones_[zi].size / (1024.0 * 1024.0));
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
        if (base && p >= base && p < base + c.size) {
            return true;
        }
    }
    for (const auto & c : runtime_chunks_) {
        const char * base = static_cast<const char *>(c.base);
        if (base && p >= base && p < base + c.size) {
            return true;
        }
    }
    return false;
}

bool pinned_chunk_pool::contains_backing_allocation(const void * ptr, size_t size) const {
    if (!ptr || size == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const uintptr_t             lo = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t             hi = lo + size;
    auto in_owner = [&](const chunk & c) {
        if (!c.base || !c.owner.valid()) {
            return false;
        }
        auto resolved = c.owner.resolve();
        if (!resolved.ptr) {
            return false;
        }
        const uintptr_t owner_lo = reinterpret_cast<uintptr_t>(resolved.ptr);
        const uintptr_t owner_hi = owner_lo + c.owner.size();
        return lo >= owner_lo && hi <= owner_hi;
    };
    for (const auto & c : chunks_) {
        if (in_owner(c)) {
            return true;
        }
    }
    for (const auto & c : runtime_chunks_) {
        if (in_owner(c)) {
            return true;
        }
    }
    return false;
}

size_t pinned_chunk_pool::allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return footprint_locked();
}

size_t pinned_chunk_pool::chunk_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      live = 0;
    for (const auto & c : chunks_) {
        live += c.base ? 1 : 0;
    }
    for (const auto & c : runtime_chunks_) {
        live += c.base ? 1 : 0;
    }
    return live;
}

size_t pinned_chunk_pool::footprint_locked() const {
    reconcile_stranded_locked();
    return total_allocated_;
}

void pinned_chunk_pool::reconcile_stranded_locked() const {
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::NO_PRUNE)) {
        return;
    }
    if (scratch_test_mutant_on(scratch_test_mutant::PRUNE_WITHOUT_QUERY)) {
        for (const stranded_entry & entry : stranded_) {
            total_allocated_ -= entry.backing;
        }
        stranded_.clear();
        return;
    }
#endif
    if (stranded_.empty()) {
        return;
    }
    // In place (remove_if then erase), so the capacity grow_into reserved is never given back.
    const auto gone = std::remove_if(stranded_.begin(), stranded_.end(), [this](const stranded_entry & entry) {
        if (unified_allocation_registered_exact(entry.metadata) != ALLOCATION_REGISTERED_GONE) {
            return false;
        }
        total_allocated_ -= entry.backing;
        return true;
    });
    stranded_.erase(gone, stranded_.end());
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
size_t pinned_chunk_pool::stranded_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    reconcile_stranded_locked();
    return stranded_.size();
}

bool pinned_chunk_pool::test_chunk_metadata(size_t chunk_idx, alloc_metadata * out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (out == nullptr || chunk_idx >= chunks_.size() || !chunks_[chunk_idx].base) {
        return false;
    }
    return chunks_[chunk_idx].owner.owner_metadata(out);
}

#    if defined(GGML_SYCL_SCRATCH_MUTANTS)
void pinned_chunk_pool::test_unwithdraw_zone_chunk(host_zone_id zone, size_t chunk_idx) {
    const size_t                zi = static_cast<size_t>(zone);
    std::lock_guard<std::mutex> lock(mutex_);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT)) {
        return;
    }
    for (auto & zcs : zone_allocators_[zi]) {
        if (zcs.chunk_idx == chunk_idx && zcs.withdrawn) {
            zcs.withdrawn = false;
            zones_[zi].size += zcs.zone_size;
        }
    }
}
#    endif
#endif

bool pinned_chunk_pool::withdraw_zone_chunk(host_zone_id zone, size_t chunk_idx) {
    const size_t                zi = static_cast<size_t>(zone);
    std::lock_guard<std::mutex> lock(mutex_);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || !zones_configured_ || chunk_idx >= chunks_.size() ||
        !chunks_[chunk_idx].base) {
        return false;
    }
    for (auto & zcs : zone_allocators_[zi]) {
        if (zcs.chunk_idx != chunk_idx) {
            continue;
        }
        // Only a chunk wholly the zone's can leave it: a shared chunk's other zones would keep using it.
        if (zcs.zone_start != 0 || zcs.zone_size != chunks_[chunk_idx].size) {
            return false;
        }
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
        if (zcs.withdrawn && scratch_test_mutant_on(scratch_test_mutant::WITHDRAW_TRANSITION_BLIND)) {
            zones_[zi].size -= zcs.zone_size;
        }
#endif
        if (!zcs.withdrawn) {
            zcs.withdrawn = true;
            zones_[zi].size -= zcs.zone_size;
        }
        return true;
    }
    return false;
}

pinned_chunk_release pinned_chunk_pool::release_zone_chunk(host_zone_id     zone,
                                                           size_t           chunk_idx,
                                                           released_chunk & out,
                                                           bool             orphan_if_live) {
    GGML_ASSERT(!out.owns_backing() && "release_zone_chunk: the carrier slot already owns a chunk");
    const size_t                zi = static_cast<size_t>(zone);
    std::lock_guard<std::mutex> lock(mutex_);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || chunk_idx >= chunks_.size() || !chunks_[chunk_idx].base) {
        return PINNED_CHUNK_RELEASE_ALREADY;
    }
    auto & states = zone_allocators_[zi];
    auto   state  = std::find_if(states.begin(), states.end(),
                                 [chunk_idx](const zone_chunk_state & s) { return s.chunk_idx == chunk_idx; });
    GGML_ASSERT(state != states.end() && state->withdrawn &&
                "release_zone_chunk: only a withdrawn chunk of the zone is released");

    chunk &    c    = chunks_[chunk_idx];
    const bool live = state->allocator->used() > 0 || (c.allocator && c.allocator->used() > 0);
    if (live || c.lease_count.load() > 0) {
        if (orphan_if_live) {
            state->orphaned = true;
            return PINNED_CHUNK_RELEASE_ORPHANED;
        }
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
        if (c.lease_count.load() > 0 && scratch_test_mutant_on(scratch_test_mutant::LEASE_CHECKED_FIRST)) {
            return PINNED_CHUNK_RELEASE_REFUSED_LEASE;
        }
#endif
        return live ? PINNED_CHUNK_RELEASE_REFUSED_LIVE : PINNED_CHUNK_RELEASE_REFUSED_LEASE;
    }

    // Unlink the span and the zone allocator. The zone's capacity already left at the withdrawal, and the footprint
    // leaves when the carrier finds the memory gone.
    flat_spans_.erase(std::remove_if(flat_spans_.begin(), flat_spans_.end(),
                                     [chunk_idx](const flat_span_info & s) { return s.chunk_idx == chunk_idx; }),
                      flat_spans_.end());
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::DOUBLE_DECREMENT)) {
        zones_[zi].size -= state->zone_size;
    }
#endif
    states.erase(state);

    out.pool_    = this;
    out.backing_ = c.size + DEFAULT_ALIGNMENT;  // what grow_into committed: usable plus the base header
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::FOOTPRINT_BEFORE_FREE)) {
        total_allocated_ -= out.backing_;
    }
#endif
    out.removed_ = std::move(c);
    // The tombstone: find_chunk_handle and acquire_chunk_lease skip a null base under this lock, so no lease can be
    // taken on the chunk from here on, and no other chunk's index moves.
    c.base       = nullptr;
    c.size       = 0;
    c.allocator.reset();
    owning_carriers_.fetch_add(1, std::memory_order_acq_rel);
    return PINNED_CHUNK_RELEASE_DONE;
}

void pinned_chunk_pool::release_chunk_backing(released_chunk && carrier) noexcept {
    released_chunk consumed(std::move(carrier));
}

pinned_chunk_usage pinned_chunk_pool::zone_chunk_usage(host_zone_id zone, size_t chunk_idx) const {
    pinned_chunk_usage          usage;
    const size_t                zi = static_cast<size_t>(zone);
    std::lock_guard<std::mutex> lock(mutex_);
    if (zi >= static_cast<size_t>(host_zone_id::COUNT) || chunk_idx >= chunks_.size() || !chunks_[chunk_idx].base) {
        return usage;
    }
    const chunk & c = chunks_[chunk_idx];
    for (const auto & zcs : zone_allocators_[zi]) {
        if (zcs.chunk_idx != chunk_idx) {
            continue;
        }
        usage.found     = true;
        usage.withdrawn = zcs.withdrawn;
        usage.orphaned  = zcs.orphaned;
        usage.used      = zcs.allocator->used() + (c.allocator ? c.allocator->used() : 0);
        usage.lo        = reinterpret_cast<uintptr_t>(c.base) + zcs.zone_start;
        usage.hi        = usage.lo + zcs.zone_size;
        break;
    }
    return usage;
}

pinned_chunk_pool::released_chunk::released_chunk(released_chunk && other) noexcept :
    pool_(other.pool_),
    removed_(std::move(other.removed_)),
    backing_(other.backing_) {
    other.pool_    = nullptr;
    other.backing_ = 0;
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::MOVED_FROM_RELEASES)) {
        // The moved-from carrier keeps its backing figure and the pool, so its destructor subtracts it again.
        other.test_moved_from_ = pool_;
        other.backing_         = backing_;
    }
#endif
}

pinned_chunk_pool::released_chunk & pinned_chunk_pool::released_chunk::operator=(released_chunk && other) noexcept {
    if (this != &other) {
        // A chunk this carrier still owned is released here, as a dropped carrier's is.
        released_chunk previous(std::move(*this));
        pool_          = other.pool_;
        removed_       = std::move(other.removed_);
        backing_       = other.backing_;
        other.pool_    = nullptr;
        other.backing_ = 0;
    }
    return *this;
}

pinned_chunk_pool::released_chunk::~released_chunk() {
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (pool_ == nullptr && test_moved_from_ != nullptr && backing_ != 0) {
        std::lock_guard<std::mutex> lock(test_moved_from_->mutex_);
        test_moved_from_->total_allocated_ -= backing_;
    }
    if (pool_ != nullptr && scratch_test_mutant_on(scratch_test_mutant::CARRIER_DTOR_NOOP)) {
        return;
    }
#endif
    if (pool_ == nullptr) {
        return;
    }
    // (1) The backing's registry identity, read before the owner's reset releases it.
    alloc_metadata metadata{};
    GGML_ASSERT(removed_.owner.owner_metadata(&metadata) && "released chunk has no allocation owner");
    // (2) The owner's reset is the release (P2); free_chunk_owner keeps the missing-owner assertion.
    pool_->free_chunk_owner(removed_, "released_chunk");
#if defined(GGML_SYCL_PRIVATE_TESTING)
    if (pool_->test_free_point_hook_) {
        pool_->test_free_point_hook_(pool_->test_free_point_ctx_, *pool_);
    }
#endif
    // (3) Whether the registry still holds that exact row: the one fact that says the memory is gone (§M92).
    const allocation_registered_state state = unified_allocation_registered_exact(metadata);
    // (4) The footprint leaves only with the memory. A row still present strands the bytes until a reconcile finds it
    // gone; the row cannot say why its release is pending, so the line names no cause (§M97).
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (state != ALLOCATION_REGISTERED_GONE && scratch_test_mutant_on(scratch_test_mutant::SUBTRACT_AT_REFUSAL)) {
        {
            std::lock_guard<std::mutex> lock(pool_->mutex_);
            pool_->total_allocated_ -= backing_;
        }
        pool_->owning_carriers_.fetch_sub(1, std::memory_order_acq_rel);
        return;
    }
#endif
    {
        std::lock_guard<std::mutex> lock(pool_->mutex_);
        if (state == ALLOCATION_REGISTERED_GONE) {
            pool_->total_allocated_ -= backing_;
        } else {
            if (pool_->stranded_.size() >= pool_->stranded_.capacity()) {
                GGML_ABORT("stranded_ capacity %zu exhausted at %zu entries", pool_->stranded_.capacity(),
                           pool_->stranded_.size());
            }
            pool_->stranded_.emplace_back(stranded_entry{ metadata, backing_ });
            GGML_LOG_WARN(
                "[PINNED-POOL] released chunk backing %.1f MB: release not yet complete after the owner reset "
                "(registry row %s); footprint stranded until the registry drops it\n",
                backing_ / (1024.0 * 1024.0), state == ALLOCATION_REGISTERED_LIVE ? "LIVE" : "RELEASING");
        }
    }
    pool_->owning_carriers_.fetch_sub(1, std::memory_order_acq_rel);
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

bool pinned_chunk_pool::grow(size_t min_size, const char * site) {
    return grow_into(chunks_, min_size, false, site);
}

pinned_chunk_pool::chunk_footprint pinned_chunk_pool::chunk_footprint_for(size_t min_size) const {
    chunk_footprint footprint;
    footprint.usable  = std::max(chunk_size_, align_up(min_size, DEFAULT_ALIGNMENT));
    footprint.backing = footprint.usable + DEFAULT_ALIGNMENT;
    return footprint;
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
        if (!c.base) {
            continue;
        }
        size_t offset = c.allocator->allocate(size, tlsf_align);
        if (offset != SIZE_MAX) {
            return static_cast<char *>(c.base) + offset;
        }
    }

    const chunk_footprint footprint      = chunk_footprint_for(size);
    const size_t          new_chunk_size = footprint.usable;
    if (footprint_locked() + footprint.backing > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    if (pinned_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] pinned pool grow: min=%zu chunk=%zu total=%.1f GB budget=%.1f GB mode=%s\n", size,
                      new_chunk_size, total_allocated_ / (1024.0 * 1024.0 * 1024.0),
                      budget_ / (1024.0 * 1024.0 * 1024.0), runtime_pool ? "runtime" : "zone-base");
    }
    if (!grow_into(chunks, size, runtime_pool, "allocate_from_chunks")) {
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
        if (base && p >= base && p < base + c.size) {
            size_t offset = p - base;
            c.allocator->free(offset);
            return true;
        }
    }
    return false;
}

bool pinned_chunk_pool::grow_into(std::vector<chunk> & chunks, size_t min_size, bool runtime_pool, const char * site) {
    // Phase gate: warn/assert when allocating chunks during inference
    static std::atomic<int> s_phase_gate{ -1 };
    int                     mode = s_phase_gate.load(std::memory_order_relaxed);
    if (mode < 0) {
        const char * env = std::getenv("GGML_SYCL_HOST_ALLOC_PHASE_GATE");
        mode             = (env != nullptr) ? std::atoi(env) : 1;
        s_phase_gate.store(mode, std::memory_order_relaxed);
    }
    // The same exemption as grow_zone(): only a TRANSACTION-kind token skips it.
    if (mode > 0 && !ggml_sycl::ggml_sycl_replan_token_held(ggml_sycl::GGML_SYCL_REPLAN_KIND_TRANSACTION)) {
        const auto phase = ggml_sycl::offload_stats_phase();
        if (phase == ggml_sycl::offload_phase::PP || phase == ggml_sycl::offload_phase::TG) {
            GGML_LOG_WARN(
                "[HOST-POOL] chunk allocation during %s phase site=%s — planner should pre-size (%zu bytes)\n",
                ggml_sycl::offload_phase_name(phase), site != nullptr ? site : "unknown", min_size);
            if (mode >= 2) {
                GGML_ASSERT(false && "host pool chunk allocation during inference");
            }
        }
    }

    // Use chunk_size_ as the default, but allow larger chunks when the
    // allocation exceeds chunk_size_ (e.g., 615 MB reorder buffers for
    // MoE models).  Level Zero's ~11 GB per-allocation limit is the
    // real cap, not chunk_size_.
    const chunk_footprint footprint    = chunk_footprint_for(min_size);
    const size_t          usable_size  = footprint.usable;
    const size_t          backing_size = footprint.backing;
    if (footprint_locked() + backing_size > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0), budget_ / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    // Room for this chunk's stranded entry before the chunk exists (§4.4 "The append never allocates, and the reserve
    // comes first"): a chunk is released once and strands at most once, so one entry per chunk suffices, and a failed
    // reserve never creates and releases a registered owner. Caught here because the synchronous path below catches
    // only sycl::exception.
    try {
        stranded_.reserve(chunks_.size() + runtime_chunks_.size() + 1);
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR("[SYCL] Pinned chunk refused: no room for its stranded entry (%zu chunks)\n",
                       chunks_.size() + runtime_chunks_.size());
        return false;
    }
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    if (scratch_test_mutant_on(scratch_test_mutant::STRANDED_RESERVE_8) && stranded_.size() <= 8) {
        std::vector<stranded_entry> eight;
        eight.reserve(8);
        eight.assign(stranded_.begin(), stranded_.end());
        stranded_.swap(eight);
    }
#endif

    if (alloc_timeout_ms_ > 0) {
        if (pinned_trace_enabled()) {
            GGML_LOG_INFO("[SYCL] pinned chunk malloc_host begin: size=%zu timeout=%zu ms\n", backing_size,
                          alloc_timeout_ms_);
        }
        auto future = std::async(std::launch::async, [&, backing_size]() {
            return allocate_pinned_chunk_owner(queue_, backing_size, runtime_pool, cache_backing_token{});
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
            auto owner    = allocate_pinned_chunk_owner(queue_, backing_size, runtime_pool, cache_backing_token{});
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

    // llama.cpp-glkg: this is the log the original NAS repro was missing.
    // A runtime chunk grown BEFORE host zones exist (zones_configured_ ==
    // false) is exactly the pre-zone fallback path
    // (pinned_chunk_pool::allocate_runtime() -> allocate_from_chunks(
    // runtime_chunks_, ..., runtime_pool=true) -> here) that can silently
    // consume the whole pinned-pool budget while model weights load, so that
    // by the time configure_zones() finally runs (today: eagerly, per the
    // ordering fix above; historically: lazily at S1-PRELOAD) its growth
    // guard is already false and it reports 0.0 MB of capacity with no
    // warning naming why. GGML_LOG_INFO above is dropped at default
    // verbosity in every tool (see CLAUDE.md's "llama-bench traps"), so this
    // is a WARN, not a second INFO line, specifically for the pre-zone case.
    if (runtime_pool && !zones_configured_) {
        GGML_LOG_WARN(
            "[HOST-ARENA] runtime pinned chunk %zu: +%.1f MB, %.1f of %.1f GB committed before host zones exist\n",
            chunks.size(), usable_size / (1024.0 * 1024.0), total_allocated_ / (1024.0 * 1024.0 * 1024.0),
            budget_ / (1024.0 * 1024.0 * 1024.0));
    }

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

size_t pinned_chunk_pool::release_chunk_lease(uint64_t handle) {
    if (handle == INVALID_CHUNK_HANDLE) {
        return NO_CHUNK;
    }
    const bool   is_runtime = (handle & CHUNK_KIND_RUNTIME_BIT) != 0;
    const size_t idx        = static_cast<size_t>(handle & ~CHUNK_KIND_RUNTIME_BIT);

    std::lock_guard<std::mutex> lock(mutex_);
    auto &                      vec = is_runtime ? runtime_chunks_ : chunks_;
    if (idx >= vec.size()) {
        GGML_LOG_ERROR("[PINNED-POOL] release_chunk_lease: invalid handle=0x%" PRIx64 " (idx=%zu, size=%zu)\n", handle,
                       idx, vec.size());
        return NO_CHUNK;
    }
    const uint32_t prev = vec[idx].lease_count.fetch_sub(1);
    if (prev == 0) {
        GGML_LOG_ERROR("[PINNED-POOL] release_chunk_lease: underflow on chunk idx=%zu runtime=%d\n", idx, is_runtime);
        // Restore to avoid wrap-around.
        vec[idx].lease_count.fetch_add(1);
        return NO_CHUNK;
    }
    // The orphaned chunk's second release event (§4.4 "the lease-drop trigger"): only a drop to zero on a base chunk
    // looks, and only at the SCRATCH states, so every other drop pays one comparison.
    if (prev != 1 || is_runtime) {
        return NO_CHUNK;
    }
    for (const auto & zcs : zone_allocators_[static_cast<size_t>(host_zone_id::SCRATCH)]) {
        if (zcs.chunk_idx == idx) {
            return (zcs.orphaned && zcs.allocator->used() == 0) ? idx : NO_CHUNK;
        }
    }
    return NO_CHUNK;
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
