//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_PINNED_POOL_HPP
#define GGML_SYCL_PINNED_POOL_HPP

#include "mem-handle.hpp"
#include "tlsf-allocator.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl {

// Copy-preserving atomic u32 adapter.  std::atomic<uint32_t> is non-copyable,
// but std::vector<chunk> grows via push_back which may internally move.
// For chunk.lease_count (llama.cpp-dyhdl), the invariant is that vector growth
// only happens under mutex_ at chunk-creation time, when lease_count is 0, so
// the "copy the value across a move" semantic is correct by construction.
//
// This mirrors the copyable_atomic_u32 pattern in unified-cache.hpp (vtf7f);
// we duplicate it here to avoid a cyclic include (unified-cache.hpp already
// depends on pinned-pool.hpp).
struct pool_atomic_u32 {
    std::atomic<uint32_t> v{ 0 };

    pool_atomic_u32() = default;

    pool_atomic_u32(const pool_atomic_u32 & o) : v(o.v.load(std::memory_order_relaxed)) {}

    pool_atomic_u32(pool_atomic_u32 && o) noexcept : v(o.v.load(std::memory_order_relaxed)) {}

    pool_atomic_u32 & operator=(const pool_atomic_u32 & o) {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    pool_atomic_u32 & operator=(pool_atomic_u32 && o) noexcept {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    uint32_t fetch_add(uint32_t n) { return v.fetch_add(n, std::memory_order_acq_rel); }

    uint32_t fetch_sub(uint32_t n) { return v.fetch_sub(n, std::memory_order_acq_rel); }

    uint32_t load() const { return v.load(std::memory_order_acquire); }

    void store(uint32_t x) { v.store(x, std::memory_order_release); }
};

// Host memory zone identifiers. Mirrors the VRAM arena zoning model.
enum class host_zone_id : uint8_t {
    WEIGHT  = 0,
    KV      = 1,
    STAGING = 2,
    SCRATCH = 3,
    COUNT   = 4,
};

const char * host_zone_name(host_zone_id zone);

struct host_zone {
    size_t start = 0;  // Logical byte offset where this zone begins
    size_t size  = 0;  // Zone capacity (for accounting/reporting)
};

// Per-chunk TLSF allocator state for a zone. Each zone-chunk intersection
// gets its own allocator managing [zone_start, zone_start + zone_size) within
// the chunk.
struct zone_chunk_state {
    size_t                          chunk_idx;   // Index into chunks_ vector
    size_t                          zone_start;  // Byte offset within chunk where this zone starts
    size_t                          zone_size;   // Size of zone region within this chunk
    std::unique_ptr<tlsf_allocator> allocator;
};

// Segment of a pinned buffer.
struct buffer_segment {
    void * ptr  = nullptr;
    size_t size = 0;
};

// A buffer that may consist of one or more non-contiguous pinned segments.
struct segmented_buffer {
    std::vector<buffer_segment> segments;
    size_t                      total_size = 0;

    void * contiguous_ptr() const { return segments.size() == 1 ? segments[0].ptr : nullptr; }

    bool is_contiguous() const { return segments.size() <= 1; }
};

// Pool allocator for pinned host memory using multiple chunks.
// Bypasses Intel Level Zero's ~11GB per-allocation limit by using
// multiple 2GB malloc_host allocations.
// Optional: set GGML_SYCL_PINNED_CHUNK_MB to override the default chunk size.
// Optional: set GGML_SYCL_PINNED_ALLOC_TIMEOUT_MS to abort if a chunk allocation hangs.
//
// Uses per-zone-per-chunk TLSF sub-allocators for O(1) alloc/free with coalescing.
// Each zone has isolated TLSF allocators, enabling proper zone_reset() without
// disturbing other zones' allocations.
// Chunks are only released when the pool is destroyed.
class pinned_chunk_pool {
  public:
    static constexpr size_t CHUNK_SIZE        = 2ULL * 1024 * 1024 * 1024;  // 2GB default chunk
    static constexpr size_t DEFAULT_ALIGNMENT = 64;                         // Cache line alignment

    // Create a pool with the given budget (maximum total memory to allocate)
    pinned_chunk_pool(sycl::queue & queue, size_t budget);
    ~pinned_chunk_pool();

    // Non-copyable, non-movable (owns SYCL allocations)
    pinned_chunk_pool(const pinned_chunk_pool &)             = delete;
    pinned_chunk_pool & operator=(const pinned_chunk_pool &) = delete;
    pinned_chunk_pool(pinned_chunk_pool &&)                  = delete;
    pinned_chunk_pool & operator=(pinned_chunk_pool &&)      = delete;

    // Allocate from pool. Returns a segmented_buffer containing one or more pinned segments.
    // All segments are aligned to DEFAULT_ALIGNMENT (64 bytes).
    segmented_buffer allocate_segmented(size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Legacy wrapper: returns first segment's pointer. For large allocations, use allocate_segmented().
    [[deprecated("use allocate_segmented() for large allocations")]] void * allocate(
        size_t size,
        size_t alignment = DEFAULT_ALIGNMENT);

    // Runtime allocation: uses a separate pool of chunks that are not part of
    // the zone layout.  Suitable for large contiguous allocations (e.g., 615 MB
    // reorder buffers) that don't fit within a single 256 MB zone chunk.
    // Falls back to growing the runtime pool if no existing chunk has space.
    void * allocate_runtime(size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Free an allocation. TLSF reclaims memory immediately via coalescing.
    void deallocate(void * ptr, size_t size);

    // Pre-allocate enough chunks to hold total_bytes without any runtime allocation.
    // Call at init time (e.g. after MoE prestage) so that inference-time allocate()
    // never triggers sycl::malloc_host which blocks the Level Zero driver.
    // Returns the number of chunks actually allocated.
    size_t pre_allocate(size_t total_bytes);

    // Pre-allocate ALL chunks needed for the full model weight set plus headroom.
    // Called once after model header is parsed and total weight size is known.
    // Returns the number of new chunks allocated.
    size_t pre_allocate_all(size_t model_weight_bytes);

    // Pre-allocate runtime pool chunks to hold total_bytes without lazy growth.
    // Call after placement plan is computed and before first inference pass.
    // Returns the number of chunks actually allocated.
    size_t pre_allocate_runtime_chunks(size_t total_bytes);

    // Configure logical zones across the pool's free capacity.
    // Must be called after pre-allocation and before any zone_alloc().
    void configure_zones(size_t weight_bytes, size_t kv_bytes, size_t staging_bytes, size_t scratch_bytes);

    // Allocate from a specific host zone. Returns a segmented_buffer.
    // With per-zone TLSF, each allocator manages its own sub-region per chunk.
    segmented_buffer zone_alloc_segmented(host_zone_id zone, size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Allocate contiguous memory from a zone. Returns nullptr on failure.
    void * zone_alloc(host_zone_id zone, size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Free a zone allocation. ptr must have been returned by zone_alloc / zone_alloc_segmented.
    void zone_free(host_zone_id zone, void * ptr, size_t size = 0);

    // Reset a zone — bulk-resets all per-zone TLSF allocators.
    // Callers must ensure no outstanding users remain.
    void zone_reset(host_zone_id zone);

    size_t zone_used(host_zone_id zone) const;
    size_t zone_capacity(host_zone_id zone) const;

    // Largest single-chunk contiguous free block currently available in `zone`.
    // Callers that need a contiguous pointer (cannot consume a fragmented
    // multi-segment allocation) use this to decide how large a single
    // allocation they can safely request.
    size_t zone_largest_free_block(host_zone_id zone) const;

    // Grow a zone by allocating additional chunks and extending the zone's span.
    // Used when compute buffer needs exceed the initially planned SCRATCH zone.
    // Returns true if the zone was successfully grown.
    bool grow_zone(host_zone_id zone, size_t additional_bytes);

    bool zones_configured() const { return zones_configured_; }

    // Check if a pointer belongs to this pool (falls within any chunk).
    bool contains(const void * ptr) const;

    // Statistics
    size_t budget() const { return budget_; }

    size_t allocated() const;  // Total bytes allocated (in chunks)
    size_t chunk_count() const;

    // === Chunk lease API (llama.cpp-dyhdl) ===
    // Reference-count a chunk while any mem_handle holds a raw pointer
    // derived from it.  Defense-in-depth beneath the vtf7f cache_entry
    // refcount: cache_entry refcount prevents cache-layer invalidation;
    // chunk lease prevents arena-layer munmap / sycl::free.
    //
    // Chunk indices are opaque u64s encoding (pool_kind, slot) so callers
    // don't need to know whether the pointer came from chunks_ or
    // runtime_chunks_. INVALID_CHUNK_HANDLE (returned on lookup miss) is
    // safe to pass to release_chunk_lease (no-op).

    static constexpr uint64_t INVALID_CHUNK_HANDLE = UINT64_MAX;

    // Find the chunk containing `ptr`. Returns INVALID_CHUNK_HANDLE if not
    // owned by this pool.  Does NOT bump the refcount — use
    // acquire_chunk_lease when the caller wants lifetime protection.
    uint64_t find_chunk_handle(const void * ptr) const;

    // Acquire a lease on the chunk containing `ptr`. Returns
    // INVALID_CHUNK_HANDLE if the pointer is not in any chunk (caller
    // treats as "no protection needed"). On success, bumps the chunk's
    // lease_count; caller MUST call release_chunk_lease(handle) exactly once.
    uint64_t acquire_chunk_lease(const void * ptr);

    // Release a lease previously acquired via acquire_chunk_lease.
    // No-op when handle == INVALID_CHUNK_HANDLE.
    void release_chunk_lease(uint64_t handle);

    // True while any lease is held on the given chunk.  Destruction-time
    // gate in ~pinned_chunk_pool waits for this to return false before
    // sycl::free'ing the chunk.
    bool chunk_has_leases(uint64_t handle) const;

  private:
    struct chunk {
        void *                          base;         // aligned pointer into backing host allocation
        size_t                          size;         // Chunk capacity in bytes
        std::unique_ptr<tlsf_allocator> allocator;    // Per-chunk TLSF for runtime/pre-zone allocs
        mem_handle                      owner;        // owning handle for the backing chunk
        pool_atomic_u32                 lease_count;  // llama.cpp-dyhdl: live raw-ptr refs
    };

    // Encoding of uint64_t chunk handles:
    //   bit 63     — 0 for base chunks (chunks_), 1 for runtime (runtime_chunks_)
    //   bits 62..0 — slot index
    static constexpr uint64_t CHUNK_KIND_RUNTIME_BIT = 1ULL << 63;

    static uint64_t encode_chunk_handle(bool runtime, size_t idx) {
        return (runtime ? CHUNK_KIND_RUNTIME_BIT : 0ULL) | static_cast<uint64_t>(idx);
    }

    // Safe 5-second wait for lease_count -> 0 before chunk free.  ASSERTs
    // with chunk details on timeout rather than spinning forever.
    void wait_for_chunk_drain_or_assert(const chunk & c, const char * ctx) const;

    // Mapping from logical address space position to a specific chunk region.
    struct flat_span_info {
        size_t logical_start;  // Start of span in the pool's logical address space
        size_t chunk_idx;      // Which chunk this span lives in
        size_t chunk_start;    // Byte offset within the chunk where the span starts
        size_t span_size;      // Size of this span
    };

    void * allocate_from_chunks(std::vector<chunk> & chunks, size_t size, size_t alignment, bool runtime_pool);
    bool   deallocate_from_chunks(std::vector<chunk> & chunks, void * ptr);
    bool   grow_into(std::vector<chunk> & chunks, size_t min_size, bool runtime_pool);

    // Allocate a new chunk (>= min_size). Returns false if over budget or allocation fails.
    bool grow(size_t min_size);

    void free_chunk_owner(chunk & c, const char * ctx);

    // Internal free without acquiring mutex_ (caller must hold it).
    void zone_free_unlocked(host_zone_id zone, void * ptr, size_t size);

    sycl::queue &                                                   queue_;
    size_t                                                          budget_;
    size_t                                                          total_allocated_  = 0;
    size_t                                                          chunk_size_       = CHUNK_SIZE;
    size_t                                                          alloc_timeout_ms_ = 0;
    std::vector<chunk>                                              chunks_;
    std::vector<chunk>                                              runtime_chunks_;
    std::array<host_zone, static_cast<size_t>(host_zone_id::COUNT)> zones_{};
    bool                                                            zones_configured_ = false;
    std::vector<flat_span_info>                                     flat_spans_;
    // Per-zone, per-chunk TLSF allocators
    std::vector<zone_chunk_state> zone_allocators_[static_cast<size_t>(host_zone_id::COUNT)];
    mutable std::mutex            mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_PINNED_POOL_HPP
