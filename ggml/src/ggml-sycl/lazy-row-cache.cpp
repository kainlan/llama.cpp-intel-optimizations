//
// The hot-row cache of a STORAGE lazy table (th32 §6.3, §6.4): the build from the plan's figures, the CLOCK lookup and
// insertion under one shard lock, and the yield that withdraws a chunk shard by shard and releases it with no lock
// held. See lazy-row-cache.hpp.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "lazy-row-cache.hpp"

#include "ggml-impl.h"
#include "host-cpu-mapping.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(LAZY_ROW_CACHE_MUTANT_YIELD_WAITS_FOR_READERS)
#    include <thread>
#endif

namespace ggml_sycl {

namespace {

// The constants are enumerators, so the cache keeps no namespace-scope object (§11).

// A key word holds the row index in its low 63 bits and the CLOCK reference bit in its top bit (§6.3). An empty slot's
// word is a row index no table has.
enum : uint64_t {
    KEY_REF   = uint64_t{ 1 } << 63,
    KEY_EMPTY = KEY_REF - 1,
};

// An index bucket holds a 4 B slot id; an empty bucket holds none.
enum : uint32_t { BUCKET_EMPTY = UINT32_MAX };

enum : size_t { NPOS = SIZE_MAX };

// The fewest buckets a shard keeps, so that its 0.75 cap still holds rows.
enum : size_t { MIN_SHARD_BUCKETS = 4 };

// splitmix64's finalizer: the row's hash, whose residue picks the shard and whose quotient the home bucket.
uint64_t hash_row(int64_t row) {
    uint64_t x = static_cast<uint64_t>(row);
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

}  // namespace

// One shard: its mutex, its share of the index's buckets, and its CLOCK state. A shard owns the slots s with
// s mod n_shards == its index; `local` below is s / n_shards.
struct lazy_row_cache::shard {
    std::mutex mutex;
    uint32_t * buckets   = nullptr;  // n_buckets slot ids, in the index allocation
    size_t     n_buckets = 0;
    size_t     owned     = 0;        // the slots this shard owns
    size_t     limit     = 0;        // the live rows it may hold: min(owned, 0.75 of its buckets)
    size_t     live      = 0;
    size_t     fill      = 0;        // the local slots ever handed out: [0, fill) is the hand's range
    size_t     hand      = 0;        // the CLOCK hand, a local slot in [0, fill)
#if defined(GGML_SYCL_PRIVATE_TESTING)
    std::atomic<bool> held{ false };
#endif
};

// One slab chunk. Its slots are [first_slot, first_slot + n_slots); `base` is its memory's bytes, resolved once at
// build and valid until `memory` is reset, which happens only once no shard can reach a slot of it.
struct lazy_row_cache::chunk {
    std::unique_ptr<lazy_row_cache_memory> memory;
    uint8_t *                              base        = nullptr;
    size_t                                 first_slot  = 0;
    size_t                                 n_slots     = 0;
    size_t                                 charged     = 0;
    uint64_t                               load_txn_id = 0;  // the load that created it, stamped once (§4.2)
    bool                                   floor       = false;
    std::atomic<bool>                      withdrawn{ false };
};

namespace {

// The lock of shard `sh`, recorded as held for the fixtures that check no release runs under one.
class shard_lock {
  public:
    explicit shard_lock(std::mutex & m
#if defined(GGML_SYCL_PRIVATE_TESTING)
                        ,
                        std::atomic<bool> & held
#endif
                        ) :
        lock_(m)
#if defined(GGML_SYCL_PRIVATE_TESTING)
        ,
        held_(held)
#endif
    {
#if defined(GGML_SYCL_PRIVATE_TESTING)
        held_.store(true, std::memory_order_release);
#endif
    }

    ~shard_lock() {
#if defined(GGML_SYCL_PRIVATE_TESTING)
        held_.store(false, std::memory_order_release);
#endif
    }

  private:
    std::lock_guard<std::mutex> lock_;
#if defined(GGML_SYCL_PRIVATE_TESTING)
    std::atomic<bool> & held_;
#endif
};

}  // namespace

#if defined(GGML_SYCL_PRIVATE_TESTING)
#    define LAZY_ROW_CACHE_LOCK(sh) shard_lock guard_((sh).mutex, (sh).held)
#else
#    define LAZY_ROW_CACHE_LOCK(sh) shard_lock guard_((sh).mutex)
#endif

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

// Why the cache does not build a geometry, or empty when it does: a row of some bytes, a floor that fits the slots, a
// slot count a 4 B slot id names (§6.3; the empty bucket's id is none of them) whose slab fits a size_t, and a chunk
// that holds a row. The one rule for "a geometry the cache refuses": the build refuses these and the chunk helpers
// return 0 for them.
static std::string lazy_row_cache_geometry_refusal(size_t cache_slots,
                                                   size_t floor_slots,
                                                   size_t row_size,
                                                   size_t chunk_bytes) {
    if (row_size == 0) {
        return "row_size=0 is not a table's row geometry";
    }
    if (floor_slots == 0 || cache_slots < floor_slots) {
        return "cache_slots=" + std::to_string(cache_slots) + " floor_slots=" + std::to_string(floor_slots) +
               ": the plan's slots must hold its floor";
    }
    if (cache_slots >= BUCKET_EMPTY || cache_slots > SIZE_MAX / row_size) {
        return "cache_slots=" + std::to_string(cache_slots) + " is more slots than a 4 B slot id names";
    }
    if (chunk_bytes < row_size) {
        return "chunk_bytes=" + std::to_string(chunk_bytes) + " row_size=" + std::to_string(row_size) +
               ": a chunk must hold a row";
    }
    return {};
}

size_t lazy_row_cache_floor_chunks(size_t cache_slots, size_t floor_slots, size_t row_size, size_t chunk_bytes) {
    if (!lazy_row_cache_geometry_refusal(cache_slots, floor_slots, row_size, chunk_bytes).empty()) {
        return 0;
    }
    const size_t slab      = cache_slots * row_size;
    const size_t n_whole   = slab / chunk_bytes;
    const size_t per_whole = chunk_bytes / row_size;
    const size_t need      = floor_slots / per_whole + (floor_slots % per_whole != 0 ? 1 : 0);
    if (need <= n_whole) {
        return need;
    }
    // The whole chunks hold fewer rows than the floor, so the remainder joins it: the floor is every chunk.
    return n_whole + (slab % chunk_bytes != 0 ? 1 : 0);
}

#if defined(LAZY_ROW_CACHE_MUTANT_YIELDABLE_BY_SUBTRACTION)
// Mutant: the planner's former figure, the resident bytes at cache_slots less those at floor_slots.
static size_t lazy_row_cache_mutant_resident(size_t slots, size_t row_size, size_t chunk_bytes) {
    const size_t slab  = slots * row_size;
    size_t       bytes = (slab / chunk_bytes) * host_cpu_mapped_size(chunk_bytes);
    if (slab % chunk_bytes != 0) {
        bytes += host_cpu_mapped_size(slab % chunk_bytes);
    }
    return bytes + host_cpu_mapped_size(lazy_row_cache_index_raw_bytes(slots));
}
#endif

size_t lazy_row_cache_yieldable_bytes(size_t cache_slots, size_t floor_slots, size_t row_size, size_t chunk_bytes) {
    const size_t floor_chunks = lazy_row_cache_floor_chunks(cache_slots, floor_slots, row_size, chunk_bytes);
    if (floor_chunks == 0) {
        return 0;
    }
#if defined(LAZY_ROW_CACHE_MUTANT_YIELDABLE_BY_SUBTRACTION)
    return lazy_row_cache_mutant_resident(cache_slots, row_size, chunk_bytes) -
           lazy_row_cache_mutant_resident(floor_slots, row_size, chunk_bytes);
#else
    const size_t slab      = cache_slots * row_size;
    const size_t n_whole   = slab / chunk_bytes;
    const size_t remainder = slab % chunk_bytes;
    size_t       bytes     = 0;
    if (floor_chunks < n_whole) {
        bytes += (n_whole - floor_chunks) * host_cpu_mapped_size(chunk_bytes);
    }
    if (remainder != 0 && floor_chunks <= n_whole) {
        bytes += host_cpu_mapped_size(remainder);
    }
    return bytes;
#endif
}

lazy_row_cache::lazy_row_cache() = default;

lazy_row_cache::~lazy_row_cache() {
    // The last reference's teardown: nothing else holds the cache, so its chunks and its index go with no lock held.
    chunks_.reset();
    index_.reset();
}

lazy_row_cache::shard & lazy_row_cache::shard_at(uint32_t j) const {
    return shards_[j];
}

// The shard whose mutex guards shard j's state: shard j itself, each shard its own leaf lock (§6.3).
lazy_row_cache::shard & lazy_row_cache::shard_to_lock(uint32_t j) const {
#if defined(LAZY_ROW_CACHE_MUTANT_ONE_LOCK)
    (void) j;
    return shard_at(0);  // Mutant: one lock for every shard.
#else
    return shard_at(j);
#endif
}

size_t lazy_row_cache::chunk_of_slot(size_t slot) const {
    // Every chunk but the last is whole, so the whole chunks' slot count divides the slot ids among them.
    if (whole_slots_ == 0) {
        return 0;
    }
    return std::min(slot / whole_slots_, n_chunks_ - 1);
}

uint8_t * lazy_row_cache::slot_bytes(size_t slot) const {
    const chunk & ch = chunks_[chunk_of_slot(slot)];
    return ch.base + (slot - ch.first_slot) * row_size_;
}

size_t lazy_row_cache::home_bucket(const shard & sh, uint64_t hash, uint32_t n_shards) {
    return static_cast<size_t>((hash / n_shards) % sh.n_buckets);
}

// The bucket of `row` in its shard, or NPOS. The shard holds at most 0.75 of its buckets, so a probe meets an empty
// bucket.
size_t lazy_row_cache::find_locked(const shard & sh, int64_t row) const {
    size_t i = home_bucket(sh, hash_row(row), n_shards_);
    for (;;) {
        const uint32_t s = sh.buckets[i];
        if (s == BUCKET_EMPTY) {
            return NPOS;
        }
        if ((keys_[s] & ~KEY_REF) == static_cast<uint64_t>(row)) {
            return i;
        }
        i = i + 1 == sh.n_buckets ? 0 : i + 1;
    }
}

// Linear probing's backward-shift deletion: the run after the freed bucket moves up wherever its home allows, so no
// tombstone is left and a probe still ends at the first empty bucket.
void lazy_row_cache::erase_bucket_locked(shard & sh, size_t bucket) {
    size_t i = bucket;
    size_t j = bucket;
    for (;;) {
        j                = j + 1 == sh.n_buckets ? 0 : j + 1;
        const uint32_t s = sh.buckets[j];
        if (s == BUCKET_EMPTY) {
            break;
        }
        const size_t k            = home_bucket(sh, hash_row(static_cast<int64_t>(keys_[s] & ~KEY_REF)), n_shards_);
        // the entry at j may fill the hole at i unless its home lies cyclically in (i, j]
        const bool   home_between = i <= j ? (i < k && k <= j) : (i < k || k <= j);
        if (!home_between) {
            sh.buckets[i] = s;
            i             = j;
        }
    }
    sh.buckets[i] = BUCKET_EMPTY;
}

// Copies `src` into `slot`, keys it as `row` with its reference bit clear (a row earns its second chance by a hit),
// and links it into the shard's index.
void lazy_row_cache::place_locked(shard & sh, size_t slot, int64_t row, const void * src) {
    std::memcpy(slot_bytes(slot), src, row_size_);
    keys_[slot] = static_cast<uint64_t>(row);
    size_t i    = home_bucket(sh, hash_row(row), n_shards_);
    while (sh.buckets[i] != BUCKET_EMPTY) {
        i = i + 1 == sh.n_buckets ? 0 : i + 1;
    }
    sh.buckets[i] = static_cast<uint32_t>(slot);
    ++sh.live;
    inserts_.fetch_add(1, std::memory_order_relaxed);
}

lazy_row_cache_build lazy_row_cache::build(const lazy_row_cache_config &  config,
                                           lazy_row_cache_memory_source & source) {
    lazy_row_cache_build result;
    // The refusal is the caller's to name, in its one WARN line (a [HOST-PLAN-BUG] for a shortfall); result.reason
    // carries every figure there, so the cache's own line is INFO.
    const auto           refuse = [&](lazy_row_cache_status status, const std::string & reason) {
        result.status = status;
        result.reason = reason;
        GGML_LOG_INFO("[LAZY-ROW-CACHE] %s refused: %s: %s\n", config.name.c_str(), lazy_row_cache_status_name(status),
                      reason.c_str());
        return std::move(result);
    };

    const size_t slots = config.cache_slots;
    const size_t row   = config.row_size;
    // The geometry by the chunk helpers' rule, then what only a built cache needs: keys for its rows and a shard.
    const std::string geometry = lazy_row_cache_geometry_refusal(slots, config.floor_slots, row, config.chunk_bytes);
    if (!geometry.empty()) {
        return refuse(lazy_row_cache_status::INVALID_CONFIG, geometry);
    }
    if (config.n_rows <= 0 || static_cast<uint64_t>(config.n_rows) >= KEY_EMPTY) {
        return refuse(lazy_row_cache_status::INVALID_CONFIG,
                      "n_rows=" + std::to_string(config.n_rows) + " is not a table's row count");
    }
    if (config.n_shards == 0) {
        return refuse(lazy_row_cache_status::INVALID_CONFIG, "n_shards=0: a cache needs a shard");
    }

    std::unique_ptr<lazy_row_cache> c(new lazy_row_cache());
    c->name_        = config.name;
    c->row_size_    = row;
    c->n_rows_      = config.n_rows;
    c->cache_slots_ = slots;
    c->floor_slots_ = config.floor_slots;
    c->load_txn_id_ = config.load_txn_id;
    c->slab_bytes_  = slots * row;
    c->index_bytes_ = lazy_row_cache_index_raw_bytes(slots);

    // The slab is the plan's chunks (§3.3): whole chunks of chunk_bytes, then one remainder. A row never straddles two
    // chunks, so a chunk of b bytes holds b / row_size rows.
    const size_t n_whole       = c->slab_bytes_ / config.chunk_bytes;
    const size_t remainder     = c->slab_bytes_ % config.chunk_bytes;
    c->n_chunks_               = n_whole + (remainder != 0 ? 1 : 0);
    c->whole_slots_            = n_whole != 0 ? config.chunk_bytes / row : 0;
    result.allocations_planned = 1 + c->n_chunks_;

    std::unique_ptr<lazy_row_cache_memory> index = source.allocate(c->index_bytes_);
    if (index == nullptr) {
        return refuse(lazy_row_cache_status::SHORTFALL,
                      "the index of " + std::to_string(c->index_bytes_) + " B was refused: 0 of " +
                          std::to_string(result.allocations_planned) + " allocations made, 0 B charged");
    }
    result.allocations_made = 1;
    result.charged_bytes    = index->charged_bytes();

    c->chunks_.reset(new chunk[c->n_chunks_]);
    size_t usable = 0;
    for (size_t i = 0; i < c->n_chunks_; ++i) {
        const size_t                           bytes = i < n_whole ? config.chunk_bytes : remainder;
        std::unique_ptr<lazy_row_cache_memory> mem   = source.allocate(bytes);
        if (mem == nullptr) {
            const std::string reason = "chunk " + std::to_string(i) + " of " + std::to_string(bytes) +
                                       " B was refused: " + std::to_string(result.allocations_made) + " of " +
                                       std::to_string(result.allocations_planned) + " allocations made, " +
                                       std::to_string(result.charged_bytes) + " B charged";
            // Everything made is released here, by the cache's destruction, before the refusal is logged.
            index.reset();
            c.reset();
            return refuse(lazy_row_cache_status::SHORTFALL, reason);
        }
        chunk & ch     = c->chunks_[i];
        ch.base        = mem->data();
        ch.first_slot  = usable;
        ch.n_slots     = bytes / row;
        ch.charged     = mem->charged_bytes();
        ch.load_txn_id = config.load_txn_id;
        ch.memory      = std::move(mem);
        usable += ch.n_slots;
        result.allocations_made++;
        result.charged_bytes += ch.charged;
    }
    // The floor is the chunk rule's, the one the planner counts the yieldable bytes by, and is never yielded (§6.4).
    c->n_floor_chunks_ = lazy_row_cache_floor_chunks(slots, config.floor_slots, row, config.chunk_bytes);
    for (size_t i = 0; i < c->n_floor_chunks_; ++i) {
        c->chunks_[i].floor = true;
    }

    // The index, laid out as lazy_row_cache_index_raw_bytes sized it: one key word per slot, then the buckets, split
    // over the shards. Each shard keeps at least MIN_SHARD_BUCKETS, and holds at most its buckets' load-factor cap.
    c->index_              = std::move(index);
    c->keys_               = reinterpret_cast<uint64_t *>(c->index_->data());
    uint32_t *   buckets   = reinterpret_cast<uint32_t *>(c->index_->data() + sizeof(uint64_t) * slots);
    const size_t n_buckets = lazy_row_cache_index_buckets(slots);
    for (size_t s = 0; s < slots; ++s) {
        c->keys_[s] = KEY_EMPTY;
    }
    std::memset(buckets, 0xFF, sizeof(uint32_t) * n_buckets);
    c->n_shards_ = static_cast<uint32_t>(
        std::max<size_t>(1, std::min<size_t>(config.n_shards, std::min(n_buckets / MIN_SHARD_BUCKETS, usable))));
    c->shards_.reset(new shard[c->n_shards_]);
    size_t bucket_at = 0;
    for (uint32_t j = 0; j < c->n_shards_; ++j) {
        shard & sh   = c->shards_[j];
        sh.n_buckets = n_buckets / c->n_shards_ + (j < n_buckets % c->n_shards_ ? 1 : 0);
        sh.buckets   = buckets + bucket_at;
        bucket_at += sh.n_buckets;
        sh.owned = usable / c->n_shards_ + (j < usable % c->n_shards_ ? 1 : 0);
        sh.limit = std::min(sh.owned, lazy_row_cache_bucket_cap(sh.n_buckets));
        c->capacity_slots_ += sh.limit;
    }
    // The capacity can fall short of cache_slots from two sources, and the build line prints the two side by side: a
    // row never straddles two chunks, so each chunk boundary can lose one row (§6.3, §3.3), and each shard holds at
    // most floor(0.75 x its buckets), so the per-shard caps can lose up to n_shards - 1 rows to rounding.
    c->chunks_resident_.store(c->n_chunks_, std::memory_order_relaxed);
    c->charged_live_.store(result.charged_bytes, std::memory_order_relaxed);

    const size_t yieldable = c->yieldable_bytes();
    GGML_LOG_INFO(
        "[LAZY-ROW-CACHE] %s built cache_slots=%zu capacity=%zu floor_slots=%zu row_size=%zu chunks=%zu "
        "floor_chunks=%zu chunk_bytes=%zu slab=%zu index=%zu shards=%u charged=%zu yieldable=%zu load_txn=%llu\n",
        c->name_.c_str(), slots, c->capacity_slots_, config.floor_slots, row, c->n_chunks_, c->n_floor_chunks_,
        config.chunk_bytes, c->slab_bytes_, c->index_bytes_, c->n_shards_, result.charged_bytes, yieldable,
        (unsigned long long) config.load_txn_id);

    result.cache  = std::move(c);
    result.status = lazy_row_cache_status::OK;
    return result;
}

uint32_t lazy_row_cache::shard_of(int64_t row) const {
    return static_cast<uint32_t>(hash_row(row) % n_shards_);
}

bool lazy_row_cache::lookup_copy(int64_t row, void * dst) {
    if (!row_valid(row) || dst == nullptr) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const uint32_t j = shard_of(row);
#if defined(LAZY_ROW_CACHE_MUTANT_YIELD_WAITS_FOR_READERS)
    // Mutant: a lookup is a lease the yield waits for.
    mutant_readers_.fetch_add(1, std::memory_order_seq_cst);

    struct reader_done {
        std::atomic<uint32_t> & n;

        ~reader_done() { n.fetch_sub(1, std::memory_order_seq_cst); }
    } done{ mutant_readers_ };
#endif
#if defined(GGML_SYCL_PRIVATE_TESTING)
    test_fire(lazy_row_cache_test_point::HIT_BEFORE_LOCK, j);
#endif
    shard & sh = shard_at(j);
    LAZY_ROW_CACHE_LOCK(shard_to_lock(j));
#if defined(GGML_SYCL_PRIVATE_TESTING)
    test_fire(lazy_row_cache_test_point::HIT_UNDER_LOCK, j);
#endif
    const size_t bucket = find_locked(sh, row);
    if (bucket == NPOS) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const uint32_t slot = sh.buckets[bucket];
    // The copy is made under the shard lock (§6.3): no pointer into the cache outlives it.
    std::memcpy(dst, slot_bytes(slot), row_size_);
#if !defined(LAZY_ROW_CACHE_MUTANT_NO_REFERENCE_BIT)
    keys_[slot] |= KEY_REF;
#endif
    hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool lazy_row_cache::insert(int64_t row, const void * src) {
    if (!row_valid(row) || src == nullptr) {
        return false;
    }
    const uint32_t j  = shard_of(row);
    shard &        sh = shard_at(j);
    LAZY_ROW_CACHE_LOCK(shard_to_lock(j));
    if (find_locked(sh, row) != NPOS) {
        return false;
    }
    const auto global = [&](size_t local) {
        return local * n_shards_ + j;
    };
    const auto live_slot = [&](size_t slot) {
        return !chunks_[chunk_of_slot(slot)].withdrawn.load(std::memory_order_acquire);
    };
    // A free slot first: the next one never handed out, skipping slots of withdrawn chunks (§6.4: no insertion picks
    // a withdrawn chunk).
    if (sh.live < sh.limit) {
        while (sh.fill < sh.owned && !live_slot(global(sh.fill))) {
            ++sh.fill;
        }
        if (sh.fill < sh.owned) {
            const size_t slot = global(sh.fill++);
            place_locked(sh, slot, row, src);
            return true;
        }
    }
    // Otherwise the CLOCK hand over [0, fill): a set reference bit is cleared and passed, the first clear one is the
    // victim. Two passes suffice, since the first clears every bit it passes.
    for (size_t step = 0; sh.fill != 0 && step < 2 * sh.fill + 1; ++step) {
        const size_t slot = global(sh.hand);
        sh.hand           = sh.hand + 1 == sh.fill ? 0 : sh.hand + 1;
        if (!live_slot(slot)) {
            continue;
        }
        const uint64_t key = keys_[slot];
        if (key == KEY_EMPTY) {
            if (sh.live < sh.limit) {
                place_locked(sh, slot, row, src);
                return true;
            }
            continue;
        }
        if ((key & KEY_REF) != 0) {
            keys_[slot] = key & ~KEY_REF;
            continue;
        }
        erase_bucket_locked(sh, find_locked(sh, static_cast<int64_t>(key)));
        keys_[slot] = KEY_EMPTY;
        --sh.live;
        evictions_.fetch_add(1, std::memory_order_relaxed);
        place_locked(sh, slot, row, src);
        return true;
    }
    // Every slot of this shard is in a withdrawn chunk: the row is dropped, which is correct, since the landing slab
    // already holds it (§6.4).
    return false;
}

size_t lazy_row_cache::yield(size_t deficit, uint64_t asking_load_txn_id) {
    size_t freed        = 0;
    size_t n_freed      = 0;
    size_t skipped_self = 0;
#if defined(LAZY_ROW_CACHE_MUTANT_YIELD_WAITS_FOR_READERS)
    // Mutant: the yield waits until no lookup is running.
    while (mutant_readers_.load(std::memory_order_seq_cst) != 0) {
        std::this_thread::yield();
    }
#endif
#if defined(LAZY_ROW_CACHE_MUTANT_YIELD_FLOOR)
    const size_t lowest = 0;  // Mutant: the floor chunks are yieldable too.
#else
    const size_t lowest = n_floor_chunks_;
#endif
    for (size_t i = n_chunks_; i > lowest && freed < deficit; --i) {
        chunk & ch = chunks_[i - 1];
        if (asking_load_txn_id != 0 && ch.load_txn_id == asking_load_txn_id) {
            ++skipped_self;
            continue;
        }
        // The withdrawal is claimed once: a chunk another yield withdrew is that yield's to release.
        if (ch.withdrawn.exchange(true, std::memory_order_acq_rel)) {
            continue;
        }
        // Under each shard lock in turn, the chunk's slots leave that shard's index. The flag above is already set, so
        // an insertion that takes the lock after this pass sees it and picks another chunk.
        const size_t end = ch.first_slot + ch.n_slots;
        for (uint32_t j = 0; j < n_shards_; ++j) {
            shard & sh = shard_at(j);
            LAZY_ROW_CACHE_LOCK(shard_to_lock(j));
#if defined(GGML_SYCL_PRIVATE_TESTING)
            test_fire(lazy_row_cache_test_point::YIELD_UNDER_SHARD_LOCK, j);
#endif
            const size_t first = ch.first_slot + (j + n_shards_ - ch.first_slot % n_shards_) % n_shards_;
            for (size_t s = first; s < end; s += n_shards_) {
                const uint64_t key = keys_[s];
                if (key == KEY_EMPTY) {
                    continue;
                }
                erase_bucket_locked(sh, find_locked(sh, static_cast<int64_t>(key & ~KEY_REF)));
                keys_[s] = KEY_EMPTY;
                --sh.live;
            }
#if defined(LAZY_ROW_CACHE_MUTANT_RELEASE_UNDER_SHARD_LOCK)
            // Mutant: the chunk is released holding the last shard's lock.
            if (j + 1 == n_shards_) {
                ch.memory.reset();
            }
#endif
        }
        // With no lock held, the release point (canon P2): the production memory drops its mem_handle, whose free
        // credits the ledger by an atomic subtract. Nothing waits: no row of the chunk is held outside the cache.
        ch.memory.reset();
        freed += ch.charged;
        ++n_freed;
        chunks_resident_.fetch_sub(1, std::memory_order_relaxed);
        chunks_yielded_.fetch_add(1, std::memory_order_relaxed);
        charged_live_.fetch_sub(ch.charged, std::memory_order_relaxed);
    }
    if (n_freed != 0) {
        yields_.fetch_add(1, std::memory_order_relaxed);
    }
    GGML_LOG_INFO(
        "[LAZY-ROW-CACHE] %s yielded chunks=%zu bytes=%zu deficit=%zu tenant_load_txn=%llu skipped_own_load=%zu "
        "chunks_resident=%llu\n",
        name_.c_str(), n_freed, freed, deficit, (unsigned long long) asking_load_txn_id, skipped_self,
        (unsigned long long) chunks_resident_.load(std::memory_order_relaxed));
    return freed;
}

size_t lazy_row_cache::yield_entry(void * ctx, size_t deficit, uint64_t load_txn_id) noexcept {
    // A yield runs inside a ledger charge, which has no failure path for a throw: the log's allocation is the only
    // thing that can throw here, and a cache that cannot log still answers what it freed.
    lazy_row_cache * c = static_cast<lazy_row_cache *>(ctx);
    try {
        return c->yield(deficit, load_txn_id);
    } catch (...) {
        return 0;
    }
}

void lazy_row_cache::publish_yieldable() {
    for (size_t i = n_floor_chunks_; i < n_chunks_; ++i) {
        chunk & ch = chunks_[i];
        if (!ch.withdrawn.load(std::memory_order_acquire) && ch.memory != nullptr) {
            ch.memory->set_yieldable();
        }
    }
}

void lazy_row_cache::unpublish_yieldable() {
    for (size_t i = n_floor_chunks_; i < n_chunks_; ++i) {
        chunk & ch = chunks_[i];
        if (ch.memory != nullptr) {
            ch.memory->clear_yieldable();
        }
    }
}

bool lazy_row_cache::resident(int64_t row) const {
    return chunk_holding(row) >= 0;
}

int64_t lazy_row_cache::chunk_holding(int64_t row) const {
    if (!row_valid(row)) {
        return -1;
    }
    const uint32_t j  = shard_of(row);
    shard &        sh = shard_at(j);
    LAZY_ROW_CACHE_LOCK(shard_to_lock(j));
    const size_t bucket = find_locked(sh, row);
    return bucket == NPOS ? -1 : static_cast<int64_t>(chunk_of_slot(sh.buckets[bucket]));
}

lazy_row_cache_counters lazy_row_cache::counters() const {
    lazy_row_cache_counters n;
    n.hits            = hits_.load(std::memory_order_relaxed);
    n.misses          = misses_.load(std::memory_order_relaxed);
    n.inserts         = inserts_.load(std::memory_order_relaxed);
    n.evictions       = evictions_.load(std::memory_order_relaxed);
    n.yields          = yields_.load(std::memory_order_relaxed);
    n.chunks_yielded  = chunks_yielded_.load(std::memory_order_relaxed);
    n.chunks_resident = chunks_resident_.load(std::memory_order_relaxed);
    return n;
}

size_t lazy_row_cache::chunk_bytes(size_t i) const {
    return i < n_chunks_ ? chunks_[i].n_slots * row_size_ : 0;
}

size_t lazy_row_cache::shard_buckets(uint32_t j) const {
    return j < n_shards_ ? shards_[j].n_buckets : 0;
}

size_t lazy_row_cache::shard_limit(uint32_t j) const {
    return j < n_shards_ ? shards_[j].limit : 0;
}

size_t lazy_row_cache::charged_bytes() const {
    return charged_live_.load(std::memory_order_relaxed);
}

size_t lazy_row_cache::yieldable_bytes() const {
    size_t bytes = 0;
    for (size_t i = n_floor_chunks_; i < n_chunks_; ++i) {
        if (!chunks_[i].withdrawn.load(std::memory_order_acquire)) {
            bytes += chunks_[i].charged;
        }
    }
    return bytes;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
void lazy_row_cache::test_set_hook(lazy_row_cache_test_hook hook, void * ctx) {
    test_hook_ctx_.store(ctx, std::memory_order_release);
    test_hook_.store(hook, std::memory_order_release);
}

bool lazy_row_cache::test_any_shard_held() const {
    for (uint32_t j = 0; j < n_shards_; ++j) {
        if (shards_[j].held.load(std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

void lazy_row_cache::test_fire(lazy_row_cache_test_point point, uint32_t shard) const {
    const lazy_row_cache_test_hook hook = test_hook_.load(std::memory_order_acquire);
    if (hook != nullptr) {
        hook(test_hook_ctx_.load(std::memory_order_acquire), point, shard);
    }
}
#endif

}  // namespace ggml_sycl
