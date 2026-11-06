//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_PINNED_POOL_HPP
#define GGML_SYCL_PINNED_POOL_HPP

#include "host-zone-kinds.hpp"
#include "mem-handle.hpp"
#include "scratch-test-mutant.hpp"
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

// host_zone_id lives in host-zone-kinds.hpp.
const char * host_zone_name(host_zone_id zone);

struct host_zone {
    size_t start = 0;  // Logical byte offset where this zone begins
    size_t size  = 0;  // Zone capacity (for accounting/reporting)
};

// Per-chunk TLSF allocator state for a zone. Each zone-chunk intersection
// gets its own allocator managing [zone_start, zone_start + zone_size) within
// the chunk.
//
// `withdrawn` and `orphaned` are the only per-chunk states of a returned
// SCRATCH chunk, and this is their one home (docs/design/sycl-mmap-host-tier.md
// §4.4 "Where the two flags live"). A withdrawn chunk takes no new zone
// allocation and its bytes leave the zone's capacity; frees into it still reach
// its allocator. An orphaned chunk's owner has ended, so the free that empties
// it, or the drop of its last lease, releases it.
struct zone_chunk_state {
    size_t                          chunk_idx;   // Index into chunks_ vector
    size_t                          zone_start;  // Byte offset within chunk where this zone starts
    size_t                          zone_size;   // Size of zone region within this chunk
    std::unique_ptr<tlsf_allocator> allocator;
    bool                            withdrawn = false;
    bool                            orphaned  = false;
};

// What pinned_chunk_pool::release_zone_chunk() decided, under the pool's lock, with the chunk's state (§4.4 "The state
// re-read"). Only DONE fills the carrier. ORPHANED is returned only to a call made with `orphan_if_live`, in place of
// REFUSED_LIVE or REFUSED_LEASE (a live allocation is checked first, then a lease), after setting `orphaned`.
enum pinned_chunk_release {
    PINNED_CHUNK_RELEASE_DONE,
    PINNED_CHUNK_RELEASE_ALREADY,        // the chunk is a tombstone
    PINNED_CHUNK_RELEASE_REFUSED_LIVE,   // the zone's or the chunk's allocator is not empty
    PINNED_CHUNK_RELEASE_REFUSED_LEASE,  // a lease is held on the chunk
    PINNED_CHUNK_RELEASE_ORPHANED,
};

// One zone chunk as the pool sees it: its flags, the bytes still allocated in it, and its usable span, for the
// SCRATCH return-refused line and the test accessors. `found` is false when the zone holds no state for the chunk.
struct pinned_chunk_usage {
    bool      found     = false;
    bool      withdrawn = false;
    bool      orphaned  = false;
    size_t    used      = 0;
    uintptr_t lo        = 0;
    uintptr_t hi        = 0;
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
// A chunk is released at the pool's destruction, or earlier by
// release_zone_chunk() when it is a withdrawn SCRATCH chunk whose allocators are
// empty and which holds no lease (§4.4 "The support gap it closes"). Its entry in
// chunks_ stays as a tombstone (base null, size 0), so no other chunk's index
// moves, and every loop over chunks_ skips a null base. Its backing leaves the
// footprint only when the memory is gone (the released_chunk carrier below).
class pinned_chunk_pool {
  public:
    static constexpr size_t CHUNK_SIZE        = 2ULL * 1024 * 1024 * 1024;  // 2GB default chunk
    static constexpr size_t DEFAULT_ALIGNMENT = 64;                         // Cache line alignment
    // No chunk index: what zone_free() and release_chunk_lease() return when the call released no orphaned chunk.
    static constexpr size_t NO_CHUNK          = SIZE_MAX;

    // The release of one chunk's backing, as a move-only RAII carrier defined after this class: its destructor frees
    // the chunk owner and moves the footprint, so a carrier dropped on any path releases and accounts exactly as a
    // consumed one does.
    class released_chunk;

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
    // reorder buffers) that don't fit within a single zone chunk (CHUNK_SIZE is 2 GiB).
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
    // Returns false only when a withdrawn chunk survives, before touching any
    // state: the rebuild would discard a live allocation's allocator (§4.4
    // "configure_zones refuses while a withdrawn chunk survives"). A capacity
    // shortfall leaves the zones unconfigured and returns true, as before.
    bool configure_zones(size_t weight_bytes, size_t kv_bytes, size_t staging_bytes, size_t scratch_bytes);

    // Allocate from a specific host zone. Returns a segmented_buffer.
    // With per-zone TLSF, each allocator manages its own sub-region per chunk.
    segmented_buffer zone_alloc_segmented(host_zone_id zone, size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Allocate contiguous memory from a zone. Returns nullptr on failure.
    void * zone_alloc(host_zone_id zone, size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Free a zone allocation. ptr must have been returned by zone_alloc / zone_alloc_segmented.
    // Returns the chunk's index when the free emptied an orphaned chunk, which the caller then releases through the
    // device cache after this lock has dropped; NO_CHUNK otherwise.
    size_t zone_free(host_zone_id zone, void * ptr, size_t size = 0);

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
    // Returns true if the zone was successfully grown. `grown_chunk`, when given,
    // receives the index of the chunk the growth appended (NO_CHUNK when none),
    // which a SCRATCH transaction records.
    bool grow_zone(host_zone_id zone, size_t additional_bytes, size_t * grown_chunk = nullptr);

    // Withdraw a chunk from zone allocation (§4.4 "a marked chunk is withdrawn at
    // once"). The chunk must belong to `zone` alone. The zone's capacity drops by
    // the chunk's bytes only on the flag's false-to-true transition, so a second
    // withdrawal changes nothing; this is the one site that lowers a zone's size.
    // False when the chunk is not wholly the zone's.
    bool withdraw_zone_chunk(host_zone_id zone, size_t chunk_idx);

    // Release a withdrawn chunk whose allocators are empty and which holds no
    // lease: unlink its span and zone allocator, leave a tombstone in chunks_, and
    // move the chunk into `out`, whose destruction frees the owner. The zone's
    // capacity and the footprint are not touched here. With `orphan_if_live` a
    // chunk that cannot be released is marked orphaned under the same hold.
    pinned_chunk_release release_zone_chunk(host_zone_id     zone,
                                            size_t           chunk_idx,
                                            released_chunk & out,
                                            bool             orphan_if_live = false);

    // Consume a carrier early: its release runs before this returns.
    static void release_chunk_backing(released_chunk && carrier) noexcept;

    pinned_chunk_usage zone_chunk_usage(host_zone_id zone, size_t chunk_idx) const;

#if defined(GGML_SYCL_PRIVATE_TESTING)
    // Test accessor: released backings whose registry row was still present when their carrier ran. Their footprint
    // stays counted until the registry drops the row.
    size_t stranded_count() const;
#endif

    bool zones_configured() const { return zones_configured_; }

    // Check if a pointer belongs to this pool (falls within any usable chunk).
    bool contains(const void * ptr) const;

    // Check if a raw backing allocation range belongs to this pool, including
    // the pre-base alignment/header bytes that usable chunk pointers skip.
    bool contains_backing_allocation(const void * ptr, size_t size = 1) const;

    // Statistics
    size_t budget() const { return budget_; }

    size_t chunk_size() const { return chunk_size_; }

    size_t allocated() const;    // Total bytes allocated (in chunks), after the stranded reconcile
    size_t chunk_count() const;  // Live chunks; a released chunk's tombstone is not counted

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
    // No-op when handle == INVALID_CHUNK_HANDLE. Returns the chunk's index when
    // this drop took an orphaned, empty SCRATCH chunk's count to zero, which the
    // caller then releases after this lock has dropped; NO_CHUNK otherwise.
    size_t release_chunk_lease(uint64_t handle);

    // True while any lease is held on the given chunk.  Destruction-time
    // gate in ~pinned_chunk_pool waits for this to return false before
    // sycl::free'ing the chunk.
    bool chunk_has_leases(uint64_t handle) const;

#if defined(GGML_SYCL_PRIVATE_TESTING)
    // The free point of a released chunk (§7 C4 "The footprint leaves with the memory"): called by the carrier after
    // the owner's release and before the registry query, with no lock held, so a test reads "freed but still counted".
    void test_set_free_point_hook(void (*hook)(void * ctx, const pinned_chunk_pool & pool), void * ctx) {
        test_free_point_hook_ = hook;
        test_free_point_ctx_  = ctx;
    }

    // The SCRATCH return's device REDs compute every expected figure from the pool's own chunk rule (§7 C4 "computed
    // as a sum of chunk_footprint_for terms"): the usable and backing bytes of one new chunk able to hold `min_size`.
    size_t test_chunk_usable_for(size_t min_size) const { return chunk_footprint_for(min_size).usable; }

    size_t test_chunk_backing_for(size_t min_size) const { return chunk_footprint_for(min_size).backing; }

    // The registry metadata of a live base chunk's backing, for the registry hold (§M88 (1)). False for a tombstone or
    // an index past the end.
    bool test_chunk_metadata(size_t chunk_idx, alloc_metadata * out) const;

#    if defined(GGML_SYCL_SCRATCH_MUTANTS)
    // Clears a withdrawn chunk's flag and gives its bytes back to the zone: the KEEP_WALKS_SET and RETURN_NO_WITHDRAW
    // faults' own step, so they can be planted in the cache, which reaches the flag only through the pool.
    void test_unwithdraw_zone_chunk(host_zone_id zone, size_t chunk_idx);
#    endif
#endif

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
    // `site` names the caller; the inference-phase gate's WARN prints it.
    bool   grow_into(std::vector<chunk> & chunks, size_t min_size, bool runtime_pool, const char * site);

    // Allocate a new chunk (>= min_size). Returns false if over budget or allocation fails.
    bool grow(size_t min_size, const char * site);

    // Byte footprint of ONE new chunk able to hold `min_size` usable bytes.
    // `usable` is what the chunk's TLSF arena manages (max(chunk_size_,
    // align_up(min_size))); `backing` is what grow_into() commits against
    // budget_ (usable plus the DEFAULT_ALIGNMENT header it skips at the
    // base). Every pre-growth budget check (grow_zone, allocate_from_chunks)
    // and grow_into()'s own check must use this same arithmetic: llama.cpp-nsl3
    // was a divergence between them (grow_zone budgeted N * chunk_size_ for
    // chunks grow_into() sized differently), and a pre-check that admits a
    // growth grow_into() then refuses surfaces as an allocation failure with
    // a misleading budget line.
    struct chunk_footprint {
        size_t usable;
        size_t backing;
    };

    chunk_footprint chunk_footprint_for(size_t min_size) const;

    void free_chunk_owner(chunk & c, const char * ctx);

    // Internal free without acquiring mutex_ (caller must hold it). Returns the
    // index of an orphaned chunk the free emptied, NO_CHUNK otherwise.
    size_t zone_free_unlocked(host_zone_id zone, void * ptr, size_t size);

    // Every decision read of the footprint (§4.4 "footprint_locked()"): reconcile the stranded list, then answer
    // total_allocated_. Caller holds mutex_.
    size_t footprint_locked() const;
    // Drop each stranded entry the registry no longer holds, subtracting its bytes at that moment; prunes in place, so
    // the list's capacity never shrinks. Caller holds mutex_; takes g_runtime_alloc_mutex per entry (canon §12.5 L5).
    void   reconcile_stranded_locked() const;

    // {registry metadata, backing bytes} of a released backing still registered; defined in pinned-pool.cpp, where
    // alloc_metadata is complete.
    struct stranded_entry;

    sycl::queue &                                                   queue_;
    size_t                                                          budget_;
    // Footprint: the backing bytes of every chunk whose memory is not yet gone. Written by grow_into's commit, the
    // carrier's GONE branch, the reconcile and the destructor's reset; read for decisions only via footprint_locked().
    mutable size_t                                                  total_allocated_  = 0;
    size_t                                                          chunk_size_       = CHUNK_SIZE;
    size_t                                                          alloc_timeout_ms_ = 0;
    std::vector<chunk>                                              chunks_;
    std::vector<chunk>                                              runtime_chunks_;
    std::array<host_zone, static_cast<size_t>(host_zone_id::COUNT)> zones_{};
    bool                                                            zones_configured_ = false;
    std::vector<flat_span_info>                                     flat_spans_;
    // Per-zone, per-chunk TLSF allocators
    std::vector<zone_chunk_state> zone_allocators_[static_cast<size_t>(host_zone_id::COUNT)];

    // Reserved by grow_into to one entry per chunk before the chunk exists, so the carrier's append never allocates.
    mutable std::vector<stranded_entry> stranded_;
    // Owning released_chunk carriers alive; ~pinned_chunk_pool asserts 0 first.
    std::atomic<size_t>                 owning_carriers_{ 0 };
#if defined(GGML_SYCL_PRIVATE_TESTING)
    void (*test_free_point_hook_)(void * ctx, const pinned_chunk_pool & pool) = nullptr;
    void * test_free_point_ctx_                                               = nullptr;
#endif
    mutable std::mutex mutex_;
};

// A released chunk's backing (§4.4 "released_chunk is the release, and it is RAII"). Built only by
// release_zone_chunk(); a move transfers the release and leaves the count unchanged; a default-constructed or
// moved-from carrier owns nothing. Its destructor, for a carrier that still owns a chunk: (1) takes the owner's
// registry metadata, (2) frees the chunk owner, (3) asks the registry whether that exact row is still registered, and
// (4) under the pool's lock subtracts the backing on GONE, or else strands it for the reconcile. It never outlives its
// pool.
class pinned_chunk_pool::released_chunk {
  public:
    released_chunk() = default;
    released_chunk(released_chunk && other) noexcept;
    released_chunk & operator=(released_chunk && other) noexcept;
    released_chunk(const released_chunk &)             = delete;
    released_chunk & operator=(const released_chunk &) = delete;
    ~released_chunk();

    bool owns_backing() const { return pool_ != nullptr; }

    size_t usable_size() const { return pool_ ? removed_.size : 0; }

    size_t backing_size() const { return pool_ ? backing_ : 0; }

  private:
    friend class pinned_chunk_pool;

    pinned_chunk_pool * pool_ = nullptr;
    chunk               removed_{};
    size_t              backing_ = 0;
#if defined(GGML_SYCL_PRIVATE_TESTING)
    // The MOVED_FROM_RELEASES fault's pool (scratch-test-mutant.hpp): set only on a carrier moved from under it.
    pinned_chunk_pool * test_moved_from_ = nullptr;
#endif
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_PINNED_POOL_HPP
