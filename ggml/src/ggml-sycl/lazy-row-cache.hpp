#pragma once

// The hot-row cache of a STORAGE lazy table (docs/design/sycl-storage-tier-lazy-tables.md §6.3, §6.4).
//
// One cache per STORAGE table, owned by the table, keyed by the table's int64 row index and holding raw row bytes in
// the table's own type: the cache is table-agnostic (§9), so a row is row_size opaque bytes. Its memory is the index,
// one allocation of 13.3 B per slot (4 B slot ids at load factor <= 0.75 plus one 8 B key word per slot, the CLOCK
// reference bit folded into the key word's top bit), and the slab, in chunks of LAZY_ROW_CACHE_CHUNK_BYTES, each its
// own allocation, so that the chunks above the floor can be yielded back one at a time (§6.4).
//
// Sharding: hash(row) mod n_shards picks a shard, and each shard has one mutex, a leaf lock (L5, canonical contract
// §12.5). Slot s belongs to shard s mod n_shards, so every chunk spans every shard. A shard lock is never held across
// an IO wait, a callback outside the cache, an allocation, a release, a log call or the ledger; it covers the index
// probe, the copy of one row and the CLOCK hand. The yield takes the shard locks one at a time, never two at once.
//
// No leases (§6.3): no pointer into the cache leaves it. A hit is copied to the caller's landing position under the
// shard lock, and an insertion copies the caller's row in under the lock. Nothing outside the cache can hold a row, so
// a chunk can be yielded at any moment without waiting for anybody: under each shard lock in turn its slots leave the
// index, then, with no lock held, its memory is released.
//
// Where the memory comes from is the one seam (lazy_row_cache_memory_source): the lazy table's production source
// allocates through the owner surfaces (HOST_CPU, alloc_role::LAZY_ROW_CACHE, charged in the host ledger), and the
// host test hands out fixture memory. Device-free on purpose, like lazy-io-pool.hpp: this header and
// lazy-row-cache.cpp include no SYCL header, so the host test links the translation unit alone.

#include "lazy-io-pool.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace ggml_sycl {

// The shards of one cache (§6.3). Each shard keeps at least 4 index buckets, so a small cache has fewer.
// An enumerator, so the header declares no namespace-scope object (§11).
enum : uint32_t { LAZY_ROW_CACHE_SHARDS = 16 };

// One allocation the cache holds: the index or one slab chunk. Destroying it releases the memory, the sole release
// point (P2): the production form drops its mem_handle, whose free credits the host ledger. The cache destroys a
// chunk with no lock held.
class lazy_row_cache_memory {
  public:
    lazy_row_cache_memory(const lazy_row_cache_memory &)             = delete;
    lazy_row_cache_memory & operator=(const lazy_row_cache_memory &) = delete;
    virtual ~lazy_row_cache_memory()                                 = default;

    uint8_t * data() const { return data_; }

    size_t bytes() const { return bytes_; }

    // The bytes the allocation charged in the host ledger (host_cpu_mapped_size of the request in production): what a
    // yield of this chunk gives back.
    size_t charged_bytes() const { return charged_; }

    // Called once, by lazy_row_cache::publish_yieldable, for each chunk above the floor (§6.4): from then on the chunk
    // counts as yieldable until it is destroyed. The production form adds its charge to the ledger's yieldable bytes
    // here and subtracts it in its destructor.
    virtual void set_yieldable() {}

  protected:
    lazy_row_cache_memory(uint8_t * data, size_t bytes, size_t charged) :
        data_(data),
        bytes_(bytes),
        charged_(charged) {}

  private:
    uint8_t * data_    = nullptr;
    size_t    bytes_   = 0;
    size_t    charged_ = 0;
};

// The seam between the cache and its memory (§6.4). Called only while the cache is built, with no cache lock held.
class lazy_row_cache_memory_source {
  public:
    virtual ~lazy_row_cache_memory_source() = default;

    // One allocation of at least `bytes` usable bytes, aligned for 8-byte words, or null when it is refused.
    virtual std::unique_ptr<lazy_row_cache_memory> allocate(size_t bytes) = 0;
};

// What a cache is built from: the table's row geometry and the planner's figures, which the cache takes as given
// (§4.3 steps 4-5).
struct lazy_row_cache_config {
    std::string name;                                      // the table's name, for every line
    size_t      row_size    = 0;                           // the bytes of one row in the table's own type
    int64_t     n_rows      = 0;                           // the table's rows: a key is in [0, n_rows)
    size_t      cache_slots = 0;                           // the plan's slots, floor plus yieldable share
    size_t      floor_slots = 0;                           // the plan's floor: 2 x n_ubatch x rows_per_token slots
    uint64_t    load_txn_id = 0;                           // the load that creates the chunks: a yield skips it
    size_t      chunk_bytes = LAZY_ROW_CACHE_CHUNK_BYTES;  // the slab's chunk; production passes the default
    uint32_t    n_shards    = LAZY_ROW_CACHE_SHARDS;
};

enum class lazy_row_cache_status : uint8_t {
    OK = 0,
    INVALID_CONFIG,  // the geometry or the figures are not a cache's
    SHORTFALL,       // an allocation of the plan was refused: the planned chunks are not all charged
};

const char * lazy_row_cache_status_name(lazy_row_cache_status status);

// The counters (§10; Task 25's hit-rate harness reads them). Plain relaxed atomics, compiled in every build.
struct lazy_row_cache_counters {
    uint64_t hits            = 0;  // lookups that found the row and copied it out
    uint64_t misses          = 0;  // lookups that did not find it
    uint64_t inserts         = 0;  // rows copied in
    uint64_t evictions       = 0;  // rows the CLOCK hand replaced
    uint64_t yields          = 0;  // yield calls that freed at least one chunk
    uint64_t chunks_yielded  = 0;  // chunks the yields freed
    uint64_t chunks_resident = 0;  // slab chunks still held
};

class lazy_row_cache;

struct lazy_row_cache_build {
    std::unique_ptr<lazy_row_cache> cache;
    lazy_row_cache_status           status = lazy_row_cache_status::INVALID_CONFIG;
    std::string                     reason;                   // the refusal's figures; empty on success
    size_t                          allocations_planned = 0;  // the index and every slab chunk
    size_t                          allocations_made    = 0;  // before the refusal; all of them on success
    size_t                          charged_bytes       = 0;  // what the allocations made had charged

    explicit operator bool() const { return status == lazy_row_cache_status::OK && cache != nullptr; }
};

#if defined(GGML_SYCL_PRIVATE_TESTING)
// The points where a fixture can hold a thread inside the cache (§11 T4's shard and yield arms).
enum class lazy_row_cache_test_point : uint8_t {
    HIT_BEFORE_LOCK,         // a lookup, before it takes its shard lock: no lock held
    HIT_UNDER_LOCK,          // a lookup, holding its shard lock, before its probe
    YIELD_UNDER_SHARD_LOCK,  // a yield, holding one shard lock while it removes a chunk's slots from that shard
};

using lazy_row_cache_test_hook = void (*)(void * ctx, lazy_row_cache_test_point point, uint32_t shard);
#endif

class lazy_row_cache {
  public:
    lazy_row_cache(const lazy_row_cache &)             = delete;
    lazy_row_cache & operator=(const lazy_row_cache &) = delete;
    ~lazy_row_cache();

    // Builds the cache: the index first, then each slab chunk in order, every allocation through `source`. A refused
    // allocation releases everything made and returns SHORTFALL with the counts and the charge made; the caller names
    // it (a step-4 shortfall is a [HOST-PLAN-BUG], §6.4). Logs the cache's slots, its capacity and its yieldable bytes.
    static lazy_row_cache_build build(const lazy_row_cache_config & config, lazy_row_cache_memory_source & source);

    // A hit copies the row's bytes to `dst` under the shard lock and sets the row's reference bit; a miss writes
    // nothing. Returns whether it hit.
    bool lookup_copy(int64_t row, void * dst);

    // Copies `src`, one row, into the cache (§6.3: a block completion inserts the row it carried). A free slot is taken
    // first; otherwise the CLOCK hand evicts the first row whose reference bit is clear, clearing the bits it passes.
    // Returns false when the row was already resident (nothing changes) or when no slot of a live chunk is left.
    bool insert(int64_t row, const void * src);

    // Frees slab chunks, the last first, until `deficit` bytes are freed or no chunk above the floor is left, skipping
    // the chunks whose load is `asking_load_txn_id` when it is nonzero (§4.2: a yield excludes the caller's own load).
    // Under each shard lock in turn a chunk's slots leave the index; then, with no lock held, its memory is released.
    // Nothing waits. Returns the bytes freed, the sum of the freed chunks' charges, and logs the chunk count and the
    // asking load.
    size_t yield(size_t deficit, uint64_t asking_load_txn_id);

    // host_ledger_yield_fn's shape (unified-cache.hpp), with `ctx` the cache: the owner registers it.
    static size_t yield_entry(void * ctx, size_t deficit, uint64_t load_txn_id) noexcept;

    // Marks every resident chunk above the floor yieldable, once, before the cache is shared (§6.4).
    void publish_yieldable();

    // Whether `row` is resident, without touching its reference bit.
    bool resident(int64_t row) const;

    // The slab chunk that holds `row`, or -1 when it is not resident. Does not touch the reference bit.
    int64_t chunk_holding(int64_t row) const;

    uint32_t shard_of(int64_t row) const;

    lazy_row_cache_counters counters() const;

    // The accounting, all fixed at build but the last two.
    const std::string & name() const { return name_; }

    size_t row_size() const { return row_size_; }

    size_t cache_slots() const { return cache_slots_; }

    size_t floor_slots() const { return floor_slots_; }

    // The rows the cache can hold at once: whole rows per chunk (a row never straddles two), within each shard's
    // load-factor cap.
    size_t capacity_slots() const { return capacity_slots_; }

    size_t index_bytes() const { return index_bytes_; }

    size_t slab_bytes() const { return slab_bytes_; }

    uint32_t n_shards() const { return n_shards_; }

    size_t n_chunks() const { return n_chunks_; }

    size_t n_floor_chunks() const { return n_floor_chunks_; }

    size_t chunk_bytes(size_t chunk) const;

    // A shard's index buckets and the live rows it may hold: at most 0.75 of its buckets.
    size_t shard_buckets(uint32_t shard) const;
    size_t shard_limit(uint32_t shard) const;

    // The ledger charge of every allocation still held: the index's and the resident chunks'.
    size_t charged_bytes() const;

    // The charge of the resident chunks above the floor.
    size_t yieldable_bytes() const;

#if defined(GGML_SYCL_PRIVATE_TESTING)
    // Arms the hook at every test point; null disarms. Set before the threads it holds start.
    void test_set_hook(lazy_row_cache_test_hook hook, void * ctx);

    // Whether any thread holds one of this cache's shard locks now.
    bool test_any_shard_held() const;
#endif

  private:
    struct shard;
    struct chunk;

    lazy_row_cache();

    shard &   shard_at(uint32_t j) const;
    size_t    chunk_of_slot(size_t slot) const;
    uint8_t * slot_bytes(size_t slot) const;
    size_t    find_locked(const shard & sh, int64_t row) const;
    void      erase_bucket_locked(shard & sh, size_t bucket);
    void      place_locked(shard & sh, size_t slot, int64_t row, const void * src);

    bool row_valid(int64_t row) const { return row >= 0 && row < n_rows_; }

    static size_t home_bucket(const shard & sh, uint64_t hash, uint32_t n_shards);

    std::string                            name_;
    size_t                                 row_size_       = 0;
    int64_t                                n_rows_         = 0;
    size_t                                 cache_slots_    = 0;
    size_t                                 floor_slots_    = 0;
    size_t                                 capacity_slots_ = 0;
    size_t                                 index_bytes_    = 0;
    size_t                                 slab_bytes_     = 0;
    size_t                                 whole_slots_    = 0;  // the rows one whole chunk holds
    uint32_t                               n_shards_       = 0;
    size_t                                 n_chunks_       = 0;
    size_t                                 n_floor_chunks_ = 0;
    uint64_t                               load_txn_id_    = 0;
    std::unique_ptr<lazy_row_cache_memory> index_;
    uint64_t *                             keys_ = nullptr;  // one key word per slot, in the index allocation
    std::unique_ptr<shard[]>               shards_;
    std::unique_ptr<chunk[]>               chunks_;          // the chunk table, a cache member (§11 static census, T4)

    std::atomic<uint64_t> hits_{ 0 };
    std::atomic<uint64_t> misses_{ 0 };
    std::atomic<uint64_t> inserts_{ 0 };
    std::atomic<uint64_t> evictions_{ 0 };
    std::atomic<uint64_t> yields_{ 0 };
    std::atomic<uint64_t> chunks_yielded_{ 0 };
    std::atomic<uint64_t> chunks_resident_{ 0 };
    std::atomic<size_t>   charged_live_{ 0 };

#if defined(GGML_SYCL_PRIVATE_TESTING)
    std::atomic<lazy_row_cache_test_hook> test_hook_{ nullptr };
    std::atomic<void *>                   test_hook_ctx_{ nullptr };
    void                                  test_fire(lazy_row_cache_test_point point, uint32_t shard) const;
#endif
#if defined(LAZY_ROW_CACHE_MUTANT_YIELD_WAITS_FOR_READERS)
    std::atomic<uint32_t> mutant_readers_{ 0 };
#endif
};

}  // namespace ggml_sycl
