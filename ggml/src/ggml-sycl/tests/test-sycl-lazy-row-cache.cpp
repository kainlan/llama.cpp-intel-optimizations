//
// The hot-row cache of a STORAGE lazy table (th32 §6.3, §6.4; row T4b), host-only on fixture memory: the CLOCK
// replacement, shard independence, the index width, the no-leases rule, the yield that frees a chunk with no wait
// while every other row stays gatherable, the counters, and the yieldable rule the planner shares. The cache's memory
// comes through its one seam, lazy_row_cache_memory_source, which here hands out plain memory: the binary links
// lazy-row-cache.cpp, host-cpu-mapping.cpp and ggml-base alone, with no device, no unified cache and no SYCL module.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Every arm names its engagement check. An arm whose engagement check fails is VOID, and a VOID arm fails the binary
// like a FAIL. Every arm runs, so one failure does not hide the rest.
//
// The arms that prove something did NOT happen (no wait, no shared lock, no release under a lock, no floor chunk
// freed, the reference bit honoured) each have a registered mutant: a -D macro that plants that defect in
// lazy-row-cache.cpp, built as test-sycl-lazy-row-cache-mutant-<suffix>, whose ctest passes only on the arm's FAIL.

#include "../lazy-row-cache.hpp"
#include "host-ledger-test-harness.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace host_ledger_test_harness;
using namespace host_ledger_test_harness::arm_loop;
using ggml_sycl::lazy_row_cache;
using ggml_sycl::lazy_row_cache_build;
using ggml_sycl::lazy_row_cache_config;
using ggml_sycl::lazy_row_cache_counters;
using ggml_sycl::lazy_row_cache_memory;
using ggml_sycl::lazy_row_cache_memory_source;
using ggml_sycl::lazy_row_cache_status;
using ggml_sycl::lazy_row_cache_test_point;

namespace {

// The no-leases rule, at compile time: a lookup and an insertion answer a bool and take the caller's buffer, so no
// member hands a pointer into the cache out.
static_assert(std::is_same<decltype(&lazy_row_cache::lookup_copy), bool (lazy_row_cache::*)(int64_t, void *)>::value,
              "a hit is a copy into the caller's buffer");
static_assert(std::is_same<decltype(&lazy_row_cache::insert), bool (lazy_row_cache::*)(int64_t, const void *)>::value,
              "an insertion copies the caller's row in");

// How long a thread may take for work that holds no lock it must wait for. A cache operation takes microseconds; the
// bound fails a mutant that waits instead of hanging the arm.
constexpr std::chrono::milliseconds k_bound{ 3000 };

constexpr size_t k_row = 170;  // Qwen's PLE row, Q8_0 of 160 elements

// --- the fixture memory ---

class fixture_source;

class fixture_memory final : public lazy_row_cache_memory {
  public:
    fixture_memory(fixture_source & source, uint8_t * data, size_t bytes, size_t ordinal);
    ~fixture_memory() override;

    void set_yieldable() override;

  private:
    fixture_source & source_;
    size_t           ordinal_;
};

// Hands out plain memory, charged at its own size, and records every request and release. The index is allocation 0
// and slab chunk c is allocation c + 1, since the cache asks in that order.
class fixture_source final : public lazy_row_cache_memory_source {
  public:
    std::vector<size_t>    requests;
    std::vector<int>       released;             // per allocation: times released
    std::vector<int>       released_under_lock;  // per allocation: releases that saw a shard lock held
    std::vector<int>       yieldable;            // per allocation: set_yieldable calls
    long                   refuse_at = -1;       // the allocation ordinal to refuse
    bool                   poison    = false;    // fill a released allocation with 0xEE before freeing it
    size_t                 live      = 0;
    const lazy_row_cache * watch     = nullptr;  // the cache whose shard locks a release checks

    std::unique_ptr<lazy_row_cache_memory> allocate(size_t bytes) override {
        const size_t ordinal = requests.size();
        requests.push_back(bytes);
        released.push_back(0);
        released_under_lock.push_back(0);
        yieldable.push_back(0);
        if (refuse_at >= 0 && ordinal == static_cast<size_t>(refuse_at)) {
            return nullptr;
        }
        uint8_t * p = static_cast<uint8_t *>(::operator new[](bytes, std::align_val_t(64)));
        ++live;
        return std::unique_ptr<lazy_row_cache_memory>(new fixture_memory(*this, p, bytes, ordinal));
    }
};

fixture_memory::fixture_memory(fixture_source & source, uint8_t * data, size_t bytes, size_t ordinal) :
    lazy_row_cache_memory(data, bytes, bytes),
    source_(source),
    ordinal_(ordinal) {}

fixture_memory::~fixture_memory() {
    if (source_.watch != nullptr && source_.watch->test_any_shard_held()) {
        ++source_.released_under_lock[ordinal_];
    }
    ++source_.released[ordinal_];
    --source_.live;
    // Poisoned before the free when asked, so a row read from a released chunk cannot pass for the row it was.
    if (source_.poison) {
        std::memset(data(), 0xEE, bytes());
    }
    ::operator delete[](data(), std::align_val_t(64));
}

void fixture_memory::set_yieldable() {
    ++source_.yieldable[ordinal_];
}

// --- rows ---

void fill_row(int64_t row, size_t size, uint8_t * out) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<uint8_t>((static_cast<uint64_t>(row) * 131u + i * 7u) ^ 0x5au);
    }
}

std::vector<uint8_t> row_bytes(int64_t row, size_t size) {
    std::vector<uint8_t> b(size);
    fill_row(row, size, b.data());
    return b;
}

bool insert_row(lazy_row_cache & c, int64_t row) {
    const std::vector<uint8_t> b = row_bytes(row, c.row_size());
    return c.insert(row, b.data());
}

// A hit whose bytes are the row's.
bool hit_matches(lazy_row_cache & c, int64_t row) {
    std::vector<uint8_t> got(c.row_size(), 0);
    return c.lookup_copy(row, got.data()) && got == row_bytes(row, c.row_size());
}

lazy_row_cache_config config(size_t slots, size_t floor, size_t chunk_bytes, uint32_t shards) {
    lazy_row_cache_config cfg;
    cfg.name        = "fixture.lazy";
    cfg.row_size    = k_row;
    cfg.n_rows      = 10000000;
    cfg.cache_slots = slots;
    cfg.floor_slots = floor;
    cfg.load_txn_id = 0x51;
    cfg.chunk_bytes = chunk_bytes;
    cfg.n_shards    = shards;
    return cfg;
}

// Whether `b` built. When it did not, records a VOID naming `what`: the arm needs a built cache to reach its subject.
bool built(arm_result & r, const lazy_row_cache_build & b, const std::string & what) {
    if (b) {
        return true;
    }
    void_arm(r, what + " did not build: " + ggml_sycl::lazy_row_cache_status_name(b.status) + ": " + b.reason);
    return false;
}

// --- holding a thread inside the cache ---

// Parks the first thread that reaches `point` (on `shard`, or any shard when it is UINT32_MAX) until released.
struct park {
    lazy_row_cache_test_point point{};
    uint32_t                  shard = UINT32_MAX;
    std::mutex                mutex;
    std::condition_variable   cv;
    bool                      armed         = true;
    bool                      entered       = false;
    bool                      released      = false;
    uint32_t                  entered_shard = UINT32_MAX;
};

void park_hook(void * ctx, lazy_row_cache_test_point point, uint32_t shard) {
    park *                       p = static_cast<park *>(ctx);
    std::unique_lock<std::mutex> lock(p->mutex);
    if (!p->armed || point != p->point || (p->shard != UINT32_MAX && shard != p->shard)) {
        return;
    }
    p->armed         = false;
    p->entered       = true;
    p->entered_shard = shard;
    p->cv.notify_all();
    p->cv.wait(lock, [p] { return p->released; });
}

bool wait_entered(park & p) {
    std::unique_lock<std::mutex> lock(p.mutex);
    return p.cv.wait_for(lock, k_bound, [&p] { return p.entered; });
}

void release(park & p) {
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        p.released = true;
    }
    p.cv.notify_all();
}

// Runs `work` on its own thread and reports whether it finished within the bound. The thread is joined by finish().
class timed {
  public:
    explicit timed(std::function<void()> work) :
        thread_([this, work] {
            work();
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
            cv_.notify_all();
        }) {}

    ~timed() { finish(); }

    bool done_within_bound() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, k_bound, [this] { return done_; });
    }

    void finish() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

  private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    done_ = false;
    std::thread             thread_;
};

// The releases of slab chunk `chunk`, allocation chunk + 1.
int src_released(const fixture_source & src, int64_t chunk) {
    const size_t ordinal = static_cast<size_t>(chunk) + 1;
    return ordinal < src.released.size() ? src.released[ordinal] : 0;
}

// --- the arms ---

// (1) The index is 13.3 B per slot and the slab is the plan's chunks (§6.3, §3.3), read from the cache's own
// accounting and from the requests its memory source saw, at Qwen's row and the production chunk. 3,200,000 slots of
// 170 B are a 544,000,000 B slab: two whole 256 MiB chunks and a remainder of 7,129,088 B. A row never straddles two
// chunks, so a whole chunk holds floor(268435456 / 170) = 1,579,032 rows and the remainder 41,935, 3,199,999 in all.
// The index has M = (4 x 3,200,000 + 2) / 3 = 4,266,667 buckets over 16 shards, 266,667 for the first 11 and 266,666
// for the rest, each capped at 0.75 of its buckets: 200,000 and 199,999 live rows, so the capacity is
// 11 x 200,000 + 5 x 199,999 = 3,199,995. Engagement: the cache built.
arm_result arm_index_width() {
    arm_result           r;
    fixture_source       src;
    const size_t         slots = 3200000;
    lazy_row_cache_build b     = lazy_row_cache::build(
        config(slots, 16384, ggml_sycl::LAZY_ROW_CACHE_CHUNK_BYTES, ggml_sycl::LAZY_ROW_CACHE_SHARDS), src);
    if (!built(r, b, "the Qwen-sized cache")) {
        return r;
    }
    const lazy_row_cache & c       = *b.cache;
    const size_t           raw     = 8 * slots + 4 * ((4 * slots + 2) / 3);
    const double           per_row = 4.0 / 0.75 + 8.0;
    expect(r, raw == ggml_sycl::lazy_row_cache_index_raw_bytes(slots),
           "the hand figure " + num(raw) + " B is not lazy_row_cache_index_raw_bytes");
    expect(r, c.index_bytes() == raw, "the cache counts " + num(c.index_bytes()) + " B of index, want " + num(raw));
    expect(r, !src.requests.empty() && src.requests[0] == raw,
           "the index request is " + (src.requests.empty() ? std::string("absent") : num(src.requests[0])) +
               " B, want " + num(raw));
    const double width = static_cast<double>(c.index_bytes()) / static_cast<double>(slots);
    expect(r, width >= per_row - 1e-5 && width <= per_row + 8.0 / slots,
           "the index is " + std::to_string(width) + " B per slot, want 13.33 within rounding");
    const std::vector<size_t> want_chunks = { 268435456, 268435456, 7129088 };
    expect(r, c.n_chunks() == 3, "the slab is " + num(c.n_chunks()) + " chunks, want 3");
    expect(r, src.requests.size() == 4 && std::equal(want_chunks.begin(), want_chunks.end(), src.requests.begin() + 1),
           "the chunk requests are not 268435456, 268435456, 7129088 B");
    expect(r, c.slab_bytes() == slots * k_row, "the slab is " + num(c.slab_bytes()) + " B, want " + num(slots * k_row));
    expect(r, c.capacity_slots() == 3199995, "the capacity is " + num(c.capacity_slots()) + " rows, want 3199995");
    expect(r, c.n_shards() == 16, "the cache has " + num(c.n_shards()) + " shards, want 16");
    for (uint32_t j = 0; j < c.n_shards(); ++j) {
        expect(r, 4 * c.shard_limit(j) <= 3 * c.shard_buckets(j),
               "shard " + num(j) + " may hold " + num(c.shard_limit(j)) + " rows in " + num(c.shard_buckets(j)) +
                   " buckets, past the 0.75 load factor");
    }
    expect(r, c.n_floor_chunks() == 1, "the floor is " + num(c.n_floor_chunks()) + " chunks, want 1");
    expect(r, c.charged_bytes() == raw + slots * k_row,
           "the cache charges " + num(c.charged_bytes()) + " B, want " + num(raw + slots * k_row));
    return r;
}

// (2) CLOCK replacement (§6.3), on one shard of four slots: rows 10-13 fill it, hits on 10 and 12 set their reference
// bits, then 14, 15 and 16 each evict one row. The hand starts at slot 0: inserting 14 clears 10's bit and evicts 11;
// 15 clears 12's bit and evicts 13; 16 finds 10's bit cleared and evicts 10. Then the counters read back: 2 hits,
// 1 miss, 7 inserts (a resident row's second insertion is not one), 3 evictions. Engagement: the cache built with one
// shard holding four rows. Mutant: LAZY_ROW_CACHE_MUTANT_NO_REFERENCE_BIT (CLOCK becomes FIFO: 14 evicts 10).
arm_result arm_clock() {
    arm_result            r;
    fixture_source        src;
    lazy_row_cache_config cfg = config(4, 4, 4 * k_row, 1);
    lazy_row_cache_build  b   = lazy_row_cache::build(cfg, src);
    if (!built(r, b, "the four-slot cache")) {
        return r;
    }
    lazy_row_cache & c = *b.cache;
    if (c.n_shards() != 1 || c.capacity_slots() != 4) {
        void_arm(r, "the cache is " + num(c.n_shards()) + " shards of capacity " + num(c.capacity_slots()) +
                        ", want 1 shard holding 4 rows");
        return r;
    }
    for (int64_t row = 10; row <= 13; ++row) {
        expect(r, insert_row(c, row), "row " + num(row) + " was not inserted into a free slot");
    }
    expect(r, hit_matches(c, 10) && hit_matches(c, 12), "a resident row did not hit with its bytes");
    expect(r, !insert_row(c, 12), "a resident row's second insertion reported an insertion");
    std::vector<uint8_t> scratch(k_row);
    expect(r, !c.lookup_copy(99, scratch.data()), "a row never inserted hit");

    const int64_t newcomers[3] = { 14, 15, 16 };
    const int64_t victims[3]   = { 11, 13, 10 };
    const char *  why[3]       = { "10 and 12 carry the reference bit", "10's bit was cleared, 12 carries it",
                                   "the hand cleared 10's bit on its first pass" };
    for (int i = 0; i < 3; ++i) {
        std::vector<int64_t> before;
        for (int64_t row = 10; row <= 16; ++row) {
            if (c.resident(row)) {
                before.push_back(row);
            }
        }
        expect(r, insert_row(c, newcomers[i]), "row " + num(newcomers[i]) + " was not inserted");
        int64_t evicted = -1;
        for (int64_t row : before) {
            if (!c.resident(row)) {
                evicted = row;
            }
        }
        expect(r, evicted == victims[i],
               "inserting " + num(newcomers[i]) + " evicted " + (evicted < 0 ? std::string("nothing") : num(evicted)) +
                   ", want " + num(victims[i]) + " (" + why[i] + ")");
    }
    expect(r, hit_matches(c, 14) && hit_matches(c, 16), "an inserted row did not hit with its bytes");
    const lazy_row_cache_counters n = c.counters();
    expect(r, n.hits == 4 && n.misses == 1 && n.inserts == 7 && n.evictions == 3,
           "the counters read hits=" + num(n.hits) + " misses=" + num(n.misses) + " inserts=" + num(n.inserts) +
               " evictions=" + num(n.evictions) + ", want 4, 1, 7, 3");
    expect(r, n.chunks_resident == 1 && n.yields == 0 && n.chunks_yielded == 0,
           "the chunk counters read resident=" + num(n.chunks_resident) + " yields=" + num(n.yields) +
               " chunks_yielded=" + num(n.chunks_yielded) + ", want 1, 0, 0");
    return r;
}

// The shard-independence cache: 4096 slots of 170 B in 64 chunks of 64 rows, 16 shards, a floor of 1024 slots (16
// chunks), rows 0-1999 resident. 1999 rows over 16 shards stay below every shard's limit, so none is evicted.
struct striped_cache {
    fixture_source       src;
    lazy_row_cache_build b;

    striped_cache() : b(lazy_row_cache::build(config(4096, 1024, 64 * k_row, 16), src)) {
        src.poison = true;
        if (b) {
            src.watch = b.cache.get();
        }
    }
};

bool fill_striped(arm_result & r, lazy_row_cache & c, int64_t rows) {
    for (int64_t row = 0; row < rows; ++row) {
        if (!insert_row(c, row)) {
            void_arm(r, "row " + num(row) + " was not inserted into the striped cache");
            return false;
        }
    }
    return true;
}

// (3) Shard independence (§6.3): a lookup holding one shard's lock does not stop a lookup in another shard, and a yield
// holding one shard's lock while it removes a chunk's slots does not stop a lookup in another shard; both lookups hit
// with their rows' bytes within the bound. Engagement: the held thread reached its hook. Mutant:
// LAZY_ROW_CACHE_MUTANT_ONE_LOCK (every shard shares one mutex).
arm_result arm_shard_independence() {
    arm_result    r;
    striped_cache sc;
    if (!built(r, sc.b, "the striped cache") || !fill_striped(r, *sc.b.cache, 2000)) {
        return r;
    }
    lazy_row_cache & c  = *sc.b.cache;
    const int64_t    ra = 7;
    int64_t          rb = -1;
    for (int64_t row = 8; row < 2000 && rb < 0; ++row) {
        if (c.shard_of(row) != c.shard_of(ra)) {
            rb = row;
        }
    }
    if (rb < 0) {
        void_arm(r, "every row hashes to shard " + num(c.shard_of(ra)));
        return r;
    }

    park held;
    held.point = lazy_row_cache_test_point::HIT_UNDER_LOCK;
    held.shard = c.shard_of(ra);
    c.test_set_hook(park_hook, &held);
    bool  a_hit = false;
    timed holder([&] { a_hit = hit_matches(c, ra); });
    if (!wait_entered(held)) {
        release(held);
        void_arm(r, "the lookup of row " + num(ra) + " never reached its hook under shard " + num(held.shard));
        c.test_set_hook(nullptr, nullptr);
        return r;
    }
    bool       b_hit = false;
    timed      other([&] { b_hit = hit_matches(c, rb); });
    const bool other_done = other.done_within_bound();
    release(held);
    holder.finish();
    other.finish();
    c.test_set_hook(nullptr, nullptr);
    expect(r, other_done,
           "a lookup in shard " + num(c.shard_of(rb)) + " did not return within " + num(k_bound.count()) +
               " ms while shard " + num(c.shard_of(ra)) + " was held");
    expect(r, a_hit && b_hit, "a held or a concurrent lookup did not hit with its row's bytes");

    // A yield holding one shard lock: it frees the last chunk and is held at the shard that is not row a's.
    const int64_t holding = c.chunk_holding(ra);
    park          yielding;
    yielding.point = lazy_row_cache_test_point::YIELD_UNDER_SHARD_LOCK;
    yielding.shard = (c.shard_of(ra) + 1) % c.n_shards();
    c.test_set_hook(park_hook, &yielding);
    size_t freed = 0;
    timed  yielder([&] { freed = c.yield(1, 0x99); });
    if (!wait_entered(yielding)) {
        release(yielding);
        yielder.finish();
        c.test_set_hook(nullptr, nullptr);
        void_arm(r, "the yield never reached its hook under shard " + num(yielding.shard));
        return r;
    }
    bool       a_again = false;
    timed      during([&] { a_again = hit_matches(c, ra); });
    const bool during_done = during.done_within_bound();
    release(yielding);
    yielder.finish();
    during.finish();
    c.test_set_hook(nullptr, nullptr);
    expect(r, holding >= 0 && holding < 63, "row " + num(ra) + " sits in chunk " + num(holding) + ", want one of 0-62");
    expect(r, during_done,
           "a lookup in shard " + num(c.shard_of(ra)) + " did not return within " + num(k_bound.count()) +
               " ms while a yield held shard " + num(yielding.shard));
    expect(r, a_again, "row " + num(ra) + " did not hit with its bytes during the yield");
    expect(r, freed == 64 * k_row, "the yield freed " + num(freed) + " B, want one chunk of " + num(64 * k_row));
    return r;
}

// (4) No leases (§6.3): a hit copies the row out, so the caller's copy survives the row's eviction and its chunk's
// release (the fixture poisons a released chunk), and the cache keeps no reference to it. The static_asserts at the top
// pin that neither a lookup nor an insertion hands a pointer out. Engagement: the row hit and its chunk was freed.
arm_result arm_no_leases() {
    arm_result    r;
    striped_cache sc;
    if (!built(r, sc.b, "the striped cache") || !fill_striped(r, *sc.b.cache, 2000)) {
        return r;
    }
    lazy_row_cache & c   = *sc.b.cache;
    int64_t          row = -1;
    for (int64_t k = 0; k < 2000 && row < 0; ++k) {
        if (c.chunk_holding(k) >= static_cast<int64_t>(c.n_floor_chunks())) {
            row = k;
        }
    }
    if (row < 0) {
        void_arm(r, "no resident row sits above the floor");
        return r;
    }
    std::vector<uint8_t> copy(k_row, 0);
    if (!c.lookup_copy(row, copy.data())) {
        void_arm(r, "row " + num(row) + " did not hit");
        return r;
    }
    const int64_t chunk = c.chunk_holding(row);
    c.yield(SIZE_MAX, 0x99);
    if (src_released(sc.src, chunk) == 0) {
        void_arm(r, "chunk " + num(chunk) + " was not released by a yield of everything above the floor");
        return r;
    }
    expect(r, copy == row_bytes(row, k_row), "the caller's copy of row " + num(row) + " changed with its chunk");
    expect(r, !c.resident(row), "row " + num(row) + " is still resident after its chunk was released");
    return r;
}

// (5) Row T4's yield fixture (§6.4): free a chunk with no wait while rows remain gatherable. The striped cache is
// filled past its capacity (rows 0-9999), so the last chunk holds rows. A reader is parked inside a lookup of one of
// those rows, before its shard lock (it holds nothing); a yield of one byte, asked by another load, must return within
// the bound having freed exactly the last chunk: one chunk's charge, released once, with no shard lock held. Then every
// resident row outside that chunk hits with its bytes, every row of it misses, the yield logged its chunk count and the
// asking load, and the counters read 1 yield, 1 chunk yielded, 63 resident. Rows inserted afterwards never land in the
// withdrawn chunk. A yield asked by the cache's own load frees nothing (§4.2). A yield of everything frees the 47
// chunks above the floor and keeps the 16 floor chunks, whose rows still hit. The cache's destruction releases the
// rest. Engagement: the reader reached its hook and the last chunk held rows. Mutants:
// LAZY_ROW_CACHE_MUTANT_YIELD_WAITS_FOR_READERS (the yield waits for the parked reader),
// LAZY_ROW_CACHE_MUTANT_RELEASE_UNDER_SHARD_LOCK (the chunk is released holding a shard lock),
// LAZY_ROW_CACHE_MUTANT_YIELD_FLOOR (a yield of everything frees the floor too).
arm_result arm_yield() {
    arm_result r;
    {
        striped_cache sc;
        if (!built(r, sc.b, "the striped cache")) {
            return r;
        }
        lazy_row_cache & c = *sc.b.cache;
        for (int64_t row = 0; row < 10000; ++row) {
            insert_row(c, row);
        }
        c.publish_yieldable();
        const size_t chunk_charge = 64 * k_row;
        const size_t last         = c.n_chunks() - 1;
        expect(r, c.yieldable_bytes() == (c.n_chunks() - c.n_floor_chunks()) * chunk_charge,
               "the yieldable bytes are " + num(c.yieldable_bytes()) + ", want " +
                   num((c.n_chunks() - c.n_floor_chunks()) * chunk_charge));
        size_t marked = 0;
        for (size_t i = 0; i < sc.src.yieldable.size(); ++i) {
            marked += sc.src.yieldable[i];
        }
        expect(r, marked == c.n_chunks() - c.n_floor_chunks(),
               num(marked) + " chunks were marked yieldable, want " + num(c.n_chunks() - c.n_floor_chunks()));

        std::vector<int64_t> in_last;
        std::vector<int64_t> elsewhere;
        for (int64_t row = 0; row < 10000; ++row) {
            const int64_t ch = c.chunk_holding(row);
            if (ch == static_cast<int64_t>(last)) {
                in_last.push_back(row);
            } else if (ch >= 0) {
                elsewhere.push_back(row);
            }
        }
        if (in_last.empty()) {
            void_arm(r, "the last chunk holds no row, so its yield could not show a row become a miss");
            return r;
        }

        log_clear();
        const lazy_row_cache_counters before = c.counters();
        park                          reader_park;
        reader_park.point = lazy_row_cache_test_point::HIT_BEFORE_LOCK;
        c.test_set_hook(park_hook, &reader_park);
        bool  reader_hit = true;
        timed reader([&] {
            std::vector<uint8_t> got(k_row);
            reader_hit = c.lookup_copy(in_last[0], got.data());
        });
        if (!wait_entered(reader_park)) {
            release(reader_park);
            reader.finish();
            c.test_set_hook(nullptr, nullptr);
            void_arm(r, "the reader never reached its hook");
            return r;
        }
        size_t     freed = 0;
        timed      yielder([&] { freed = c.yield(1, 0x99); });
        const bool yield_done = yielder.done_within_bound();
        release(reader_park);
        reader.finish();
        yielder.finish();
        c.test_set_hook(nullptr, nullptr);
        expect(r, yield_done,
               "the yield did not return within " + num(k_bound.count()) +
                   " ms while a reader was inside a lookup of a row of the chunk (holding no lock)");
        expect(r, freed == chunk_charge, "the yield freed " + num(freed) + " B, want " + num(chunk_charge));
        expect(r, src_released(sc.src, last) == 1,
               "chunk " + num(last) + " was released " + num(src_released(sc.src, last)) + " times, want once");
        expect(r, sc.src.released_under_lock[last + 1] == 0,
               "chunk " + num(last) + " was released while a shard lock was held");
        expect(r, !reader_hit, "the parked reader hit row " + num(in_last[0]) + " after its chunk was yielded");

        size_t lost = 0;
        for (int64_t row : elsewhere) {
            lost += hit_matches(c, row) ? 0 : 1;
        }
        size_t kept = 0;
        for (int64_t row : in_last) {
            kept += c.resident(row) ? 1 : 0;
        }
        expect(r, lost == 0, num(lost) + " of " + num(elsewhere.size()) + " rows outside the chunk stopped hitting");
        expect(r, kept == 0,
               num(kept) + " of " + num(in_last.size()) + " rows of the yielded chunk are still resident");
        const lazy_row_cache_counters after = c.counters();
        expect(r, after.yields == before.yields + 1 && after.chunks_yielded == 1 && after.chunks_resident == 63,
               "the counters read yields=" + num(after.yields) + " chunks_yielded=" + num(after.chunks_yielded) +
                   " chunks_resident=" + num(after.chunks_resident) + ", want 1, 1, 63");
        expect(r, after.hits - before.hits == elsewhere.size() && after.misses - before.misses == 1,
               "the yield's lookups counted " + num(after.hits - before.hits) + " hits and " +
                   num(after.misses - before.misses) + " misses, want " + num(elsewhere.size()) + " and 1");
        const auto lines = log_lines_containing("[LAZY-ROW-CACHE] fixture.lazy yielded");
        expect(r, lines.size() == 1, "the yield printed " + num(lines.size()) + " lines, want 1");
        if (lines.size() == 1) {
            expect(r, printed(lines[0], "chunks") == 1 && printed(lines[0], "tenant_load_txn") == 0x99,
                   "the yield line does not name its chunk count and the asking load: " + lines[0]);
        }

        bool landed_in_withdrawn = false;
        for (int64_t row = 20000; row < 21000; ++row) {
            insert_row(c, row);
            landed_in_withdrawn = landed_in_withdrawn || c.chunk_holding(row) == static_cast<int64_t>(last);
        }
        expect(r, !landed_in_withdrawn, "a row inserted after the yield landed in the withdrawn chunk");

        expect(r, c.yield(SIZE_MAX, 0x51) == 0, "a yield asked by the cache's own load freed bytes");
        expect(r, c.counters().chunks_resident == 63, "a yield asked by the cache's own load freed a chunk");

        const size_t all = c.yield(SIZE_MAX, 0x99);
        expect(r, c.counters().chunks_resident == c.n_floor_chunks(),
               "a yield of everything left " + num(c.counters().chunks_resident) + " chunks, want the " +
                   num(c.n_floor_chunks()) + " floor chunks");
        expect(r, all == 47 * chunk_charge,
               "a yield of everything freed " + num(all) + " B, want " + num(47 * chunk_charge));
        expect(r, c.yieldable_bytes() == 0,
               "the yieldable bytes after a yield of everything are " + num(c.yieldable_bytes()));
        size_t floor_rows = 0, floor_lost = 0;
        for (int64_t row = 0; row < 21000; ++row) {
            const int64_t ch = c.chunk_holding(row);
            if (ch >= 0) {
                ++floor_rows;
                floor_lost += (ch < static_cast<int64_t>(c.n_floor_chunks()) && hit_matches(c, row)) ? 0 : 1;
            }
        }
        expect(r, floor_rows > 0 && floor_lost == 0,
               num(floor_lost) + " of " + num(floor_rows) + " resident rows are outside the floor or stopped hitting");
        size_t under_lock = 0;
        for (int n : sc.src.released_under_lock) {
            under_lock += n;
        }
        expect(r, under_lock == 0, num(under_lock) + " chunks were released while a shard lock was held");
        sc.src.watch = nullptr;
        sc.b.cache.reset();
        expect(r, sc.src.live == 0, "the cache's destruction left " + num(sc.src.live) + " allocations live");
    }
    return r;
}

// (6) A refused allocation is a shortfall (§6.4: a step-4 shortfall is plan != reality, which the table names): the
// build releases everything it made and reports the counts and the charge made. Allocation 2 (the second chunk) is
// refused: the index and one chunk were made. Engagement: the source saw the refused request.
arm_result arm_build_shortfall() {
    arm_result     r;
    fixture_source src;
    src.refuse_at          = 2;
    lazy_row_cache_build b = lazy_row_cache::build(config(4096, 1024, 64 * k_row, 16), src);
    if (src.requests.size() < 3) {
        void_arm(r, "the build made " + num(src.requests.size()) + " requests, so the refusal never ran");
        return r;
    }
    const size_t index = ggml_sycl::lazy_row_cache_index_raw_bytes(4096);
    expect(r, b.status == lazy_row_cache_status::SHORTFALL,
           std::string("the build returned ") + ggml_sycl::lazy_row_cache_status_name(b.status) + ", want shortfall");
    expect(r, !b.cache, "a shortfall returned a cache");
    expect(r, b.allocations_planned == 65 && b.allocations_made == 2,
           "the shortfall counts " + num(b.allocations_made) + " of " + num(b.allocations_planned) +
               " allocations, want 2 of 65");
    expect(r, b.charged_bytes == index + 64 * k_row,
           "the shortfall's charge is " + num(b.charged_bytes) + " B, want " + num(index + 64 * k_row));
    expect(r, src.requests.size() == 3, "the build asked " + num(src.requests.size()) + " times, want 3");
    expect(r, src.live == 0, "the shortfall left " + num(src.live) + " allocations live");
    expect(r, b.reason.find("2 of 65") != std::string::npos,
           "the shortfall's reason does not name its counts: " + b.reason);
    return r;
}

// (7) A config that is not a cache's refuses before any allocation: a floor above the slots, a chunk smaller than a
// row, no row size, and more slots than a 4 B slot id names. Engagement: none needed; the source counts requests.
arm_result arm_invalid_config() {
    arm_result r;

    struct bad {
        const char *          what;
        lazy_row_cache_config cfg;
    };

    std::vector<bad> cases;
    cases.push_back({ "a floor above the slots", config(1024, 2048, 64 * k_row, 16) });
    cases.push_back({ "a chunk smaller than a row", config(1024, 1024, k_row - 1, 16) });
    lazy_row_cache_config no_row = config(1024, 1024, 64 * k_row, 16);
    no_row.row_size              = 0;
    cases.push_back({ "no row size", no_row });
    lazy_row_cache_config wide = config(size_t{ UINT32_MAX }, 1024, ggml_sycl::LAZY_ROW_CACHE_CHUNK_BYTES, 16);
    cases.push_back({ "more slots than a 4 B slot id names", wide });
    for (const bad & k : cases) {
        fixture_source             src;
        const lazy_row_cache_build b = lazy_row_cache::build(k.cfg, src);
        expect(r, b.status == lazy_row_cache_status::INVALID_CONFIG && !b.cache,
               std::string(k.what) + " built as " + ggml_sycl::lazy_row_cache_status_name(b.status) +
                   ", want invalid-config");
        expect(r, src.requests.empty(), std::string(k.what) + " allocated before refusing");
    }
    return r;
}

// (8) The cache and the planner count one yieldable figure (§6.4, §10): lazy_row_cache_floor_chunks and
// lazy_row_cache_yieldable_bytes are the chunk rule, and a built cache's floor and yieldable bytes are theirs.
// First at Qwen's geometry, arm (1)'s: the floor of 16,384 rows sits in the first 256 MiB chunk, so the yieldable bytes
// are the second chunk and the remainder's mapping, 268,435,456 + 8,388,608 = 276,824,064 B. The planner's subtraction,
// the resident bytes at cache_slots less those at floor_slots, counts the first chunk's unused part and the index
// difference as well: 583,008,256 B.
// Then three caches of 20 rows of 1 MiB in 8 MiB chunks (8, 8 and 4 rows), each chunk a 2 MiB multiple, so a chunk's
// fixture charge is its mapping: a floor inside the first chunk (3 rows) and one exactly at its boundary (8 rows) are
// one floor chunk and 12 MiB yieldable; a floor larger than one chunk (9 rows) is two floor chunks and 4 MiB.
// Last, the mapping rounding with 100 B rows in 1000 B chunks, and a geometry the cache refuses, which is 0.
// Engagement: each cache built. Mutant: LAZY_ROW_CACHE_MUTANT_YIELDABLE_BY_SUBTRACTION, the planner's subtraction.
arm_result arm_yieldable_rule() {
    arm_result   r;
    const size_t mib = size_t{ 1 } << 20;
    const size_t qwen =
        ggml_sycl::lazy_row_cache_yieldable_bytes(3200000, 16384, k_row, ggml_sycl::LAZY_ROW_CACHE_CHUNK_BYTES);
    expect(r, qwen == 276824064, "the yieldable bytes at Qwen's geometry are " + num(qwen) + ", want 276824064");

    struct edge {
        const char * what;
        size_t       floor;
        size_t       floor_chunks;
        size_t       yieldable;
    };

    const edge edges[] = {
        { "a floor inside the first chunk",      3, 1, 12 * mib },
        { "a floor exactly at a chunk boundary", 8, 1, 12 * mib },
        { "a floor larger than one chunk",       9, 2, 4 * mib  },
    };
    for (const edge & e : edges) {
        const size_t      floor_chunks = ggml_sycl::lazy_row_cache_floor_chunks(20, e.floor, mib, 8 * mib);
        const size_t      yieldable    = ggml_sycl::lazy_row_cache_yieldable_bytes(20, e.floor, mib, 8 * mib);
        const std::string at           = std::string(e.what) + " (" + num(e.floor) + " rows): ";
        expect(r, floor_chunks == e.floor_chunks,
               at + "the rule's floor is " + num(floor_chunks) + " chunks, want " + num(e.floor_chunks));
        expect(r, yieldable == e.yieldable,
               at + "the rule's yieldable bytes are " + num(yieldable) + ", want " + num(e.yieldable));
        fixture_source        src;
        lazy_row_cache_config cfg = config(20, e.floor, 8 * mib, 16);
        cfg.row_size              = mib;
        lazy_row_cache_build b    = lazy_row_cache::build(cfg, src);
        if (!built(r, b, at + "the cache")) {
            continue;
        }
        expect(
            r, b.cache->n_floor_chunks() == floor_chunks,
            at + "the cache's floor is " + num(b.cache->n_floor_chunks()) + " chunks, the rule's " + num(floor_chunks));
        expect(r, b.cache->yieldable_bytes() == yieldable,
               at + "the cache publishes " + num(b.cache->yieldable_bytes()) + " yieldable bytes, the rule " +
                   num(yieldable));
    }
    const size_t round_one = ggml_sycl::lazy_row_cache_yieldable_bytes(25, 10, 100, 1000);
    const size_t round_two = ggml_sycl::lazy_row_cache_yieldable_bytes(25, 11, 100, 1000);
    const size_t round_all = ggml_sycl::lazy_row_cache_yieldable_bytes(25, 25, 100, 1000);
    expect(r, round_one == 2 * 2 * mib && round_two == 2 * mib && round_all == 0,
           "100 B rows in 1000 B chunks yield " + num(round_one) + ", " + num(round_two) + " and " + num(round_all) +
               " B at floors of 10, 11 and 25 rows, want two, one and no 2 MiB mappings");
    expect(r,
           ggml_sycl::lazy_row_cache_yieldable_bytes(10, 11, 100, 1000) == 0 &&
               ggml_sycl::lazy_row_cache_floor_chunks(10, 11, 100, 1000) == 0,
           "a floor above the slots is not 0 floor chunks and 0 yieldable bytes");
    return r;
}

}  // namespace

int main() {
    ggml_log_set(capture_log, nullptr);
    std::vector<named_arm> arms;
    arms.push_back({ "(1) the index is 13.3 B per slot and the slab is the plan's chunks", arm_index_width });
    arms.push_back({ "(2) CLOCK evicts by the reference bit and the hand, and the counters read back", arm_clock });
    arms.push_back({ "(3) a lookup in one shard never waits on another shard's lock", arm_shard_independence });
    arms.push_back({ "(4) a hit is a copy, so no row outlives the shard lock", arm_no_leases });
    arms.push_back({ "(5) a yield frees a chunk with no wait while rows remain gatherable", arm_yield });
    arms.push_back({ "(6) a refused allocation is a shortfall that leaves nothing live", arm_build_shortfall });
    arms.push_back({ "(7) a config that is not a cache's refuses before any allocation", arm_invalid_config });
    arms.push_back({ "(8) the cache and the planner count one yieldable figure", arm_yieldable_rule });
    return run_arms(arms, "lazy row cache arms");
}
