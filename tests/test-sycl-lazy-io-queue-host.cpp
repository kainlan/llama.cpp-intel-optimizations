// The IO engine's queue (docs/design/sycl-storage-tier-lazy-tables.md §5.2.1) and the reaper's delivery (§5.3):
// the (epoch, deadline) key, lazy rows before expert slices at an equal layer, equal keys in enqueue order, the
// epoch's wrap, the contexts' turns, class 1 behind every class 0, the promotion of a joined lookahead read, the
// 512 KiB chunks, the in-flight dedup, the same-context re-key that keeps its number, the second entry of a join
// from another context, the test-only FIFO override, the dispatch log, and the landing copy's epoch and cancel check.
//
// Host-only, because §5.2.1's order is a property of the queue alone: the queue is a data structure the pool calls
// under its queue lock, so these arms drive it on one thread with fixture requests, no worker, no ring and no device.
// Links ggml-sycl/lazy-io-pool.cpp and the row cache it inserts into (lazy-row-cache.cpp, host-cpu-mapping.cpp), with
// ggml-base for the log.
//
// Every arm names the mutant it kills; each mutant is a registered CMake variant whose ctest passes only when the arm's
// FAIL line appears. Every arm runs, so one failure does not hide the rest.

#include "../ggml/src/ggml-sycl/lazy-io-pool.hpp"
#include "../ggml/src/ggml-sycl/lazy-row-cache.hpp"
#include "../ggml/src/ggml-sycl/tests/host-ledger-test-harness.hpp"
#include "../ggml/src/ggml-sycl/tests/lazy-io-host-fixture.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

using namespace ggml_sycl;
using host_ledger_test_harness::arm_result;
using host_ledger_test_harness::expect;
using host_ledger_test_harness::num;
using host_ledger_test_harness::arm_loop::named_arm;
using host_ledger_test_harness::arm_loop::run_arms;

namespace {

constexpr uint64_t k_file = 7;

lazy_io_span block_span(uint64_t block) {
    return { k_file, block * LAZY_IO_BLOCK_BYTES, LAZY_IO_BLOCK_BYTES };
}

lazy_io_key key0(uint32_t epoch, int32_t layer, lazy_io_kind kind = lazy_io_kind::TABLE_ROWS) {
    return { lazy_io_class::PREFETCH, epoch, layer, kind };
}

lazy_io_key key1(uint32_t epoch, int32_t layer) {
    return { lazy_io_class::LOOKAHEAD, epoch, layer, lazy_io_kind::TABLE_ROWS };
}

size_t put(lazy_io_queue &        q,
           uint64_t               context,
           uint64_t               block,
           const lazy_io_key &    key,
           const lazy_io_waiter & w = {}) {
    std::vector<ring_lease> spare;
    return q.submit(context, -1, block_span(block), key, w, spare);
}

uint64_t block_of(const lazy_io_read * r) {
    return r->span.offset / LAZY_IO_BLOCK_BYTES;
}

struct pick_rec {
    uint64_t block   = 0;
    size_t   waiters = 0;
};

// Picks until nothing is queued, completing each read at once, and returns the picks in order. Bounded, so a queue that
// never empties ends the arm rather than the binary.
std::vector<pick_rec> drain_records(lazy_io_queue & q) {
    std::vector<pick_rec> out;
    while (out.size() < 100000) {
        lazy_io_read * r = q.pick();
        if (r == nullptr) {
            break;
        }
        out.push_back({ block_of(r), r->waiters.size() });
        q.complete(r);
    }
    return out;
}

std::vector<uint64_t> drain(lazy_io_queue & q) {
    std::vector<uint64_t> blocks;
    for (const pick_rec & p : drain_records(q)) {
        blocks.push_back(p.block);
    }
    return blocks;
}

std::string join(const std::vector<uint64_t> & v) {
    if (v.empty()) {
        return "(nothing)";
    }
    std::string s;
    for (uint64_t x : v) {
        s += (s.empty() ? "" : " ") + num(x);
    }
    return s;
}

std::vector<uint64_t> range(uint64_t first, uint64_t last) {
    std::vector<uint64_t> v;
    for (uint64_t b = first; b <= last; ++b) {
        v.push_back(b);
    }
    return v;
}

std::vector<uint64_t> cat(std::vector<uint64_t> a, const std::vector<uint64_t> & b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// The pick index of `block` in `order`, or SIZE_MAX.
size_t position(const std::vector<uint64_t> & order, uint64_t block) {
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] == block) {
            return i;
        }
    }
    return SIZE_MAX;
}

std::string pos_text(size_t p) {
    return p == SIZE_MAX ? std::string("-") : num(p);
}

// Within one context class 0 dispatches by (epoch, deadline): every read of epoch 5 before any of epoch 6, the earliest
// layer first within an epoch, a table's rows before an expert slice at an equal layer, and equal keys in the order
// they were queued.
// Mutants: LAZY_IO_QUEUE_MUTANT_DEADLINE_ONLY (no epoch term), LAZY_IO_QUEUE_MUTANT_SLICES_FIRST (the kind term
// reversed), LAZY_IO_QUEUE_MUTANT_EQUAL_KEY_LIFO (equal keys in reverse enqueue order).
arm_result arm_key_order() {
    arm_result    r;
    lazy_io_queue q;
    put(q, 1, 100, key0(6, 1));
    put(q, 1, 101, key0(6, 1));
    put(q, 1, 102, key0(5, 14));
    put(q, 1, 103, key0(5, 2, lazy_io_kind::EXPERT_SLICE));
    put(q, 1, 104, key0(5, 14, lazy_io_kind::EXPERT_SLICE));
    put(q, 1, 105, key0(5, 14));
    put(q, 1, 106, key0(5, 2));
    const std::vector<uint64_t> want = { 106, 103, 102, 105, 104, 100, 101 };
    const std::vector<uint64_t> got  = drain(q);
    expect(r, got == want, "dispatched " + join(got) + ", want " + join(want));
    return r;
}

// The epoch compares by serial-number arithmetic, so epoch 0xffffffff goes before epoch 0, which goes before epoch 1.
// Mutant: LAZY_IO_QUEUE_MUTANT_PLAIN_EPOCH_COMPARE (an unsigned compare, which puts the wrapped epoch last).
arm_result arm_epoch_wrap() {
    arm_result    r;
    lazy_io_queue q;
    put(q, 1, 200, key0(1, 1));
    put(q, 1, 201, key0(0xffffffffu, 30));
    put(q, 1, 202, key0(0, 1));
    const std::vector<uint64_t> want = { 201, 202, 200 };
    const std::vector<uint64_t> got  = drain(q);
    expect(r, got == want, "dispatched " + join(got) + ", want " + join(want));
    return r;
}

// The contexts take turns: context 2's four reads, queued behind context 1's hundred, dispatch at picks 1, 3, 5 and 7.
// Mutant: LAZY_IO_QUEUE_MUTANT_NO_ROUND_ROBIN (the lowest context is served until it is empty).
arm_result arm_round_robin() {
    arm_result    r;
    lazy_io_queue q;
    for (uint64_t b = 0; b < 100; ++b) {
        put(q, 1, b, key0(1, 1));
    }
    for (uint64_t b = 200; b < 204; ++b) {
        put(q, 2, b, key0(1, 1));
    }
    const std::vector<uint64_t> got = drain(q);
    std::string                 at;
    bool                        ok = true;
    for (uint64_t b = 200; b < 204; ++b) {
        const size_t p = position(got, b);
        at += (at.empty() ? "" : " ") + pos_text(p);
        ok = ok && p == 1 + 2 * (b - 200);
    }
    expect(r, ok, "context 2's reads dispatched at picks " + at + ", want 1 3 5 7");
    expect(r, got.size() == 104, "dispatched " + num(got.size()) + " reads, want 104");
    return r;
}

// Class 1 is served only when no context has a class-0 read queued, even when its key is more urgent: context 1's
// lookahead reads wait for context 2's class-0 reads, and the contexts take turns within class 1.
// Mutant: LAZY_IO_QUEUE_MUTANT_CLASS_BLIND (one queue per context, keyed without the class).
arm_result arm_class_order() {
    arm_result    r;
    lazy_io_queue q;
    put(q, 1, 0, key1(1, 1));
    put(q, 1, 1, key1(1, 1));
    put(q, 1, 2, key1(1, 1));
    put(q, 2, 10, key0(9, 30));
    put(q, 2, 11, key0(9, 30));
    put(q, 2, 12, key0(9, 30));
    put(q, 2, 13, key1(0, 0));
    const std::vector<uint64_t> want = { 10, 11, 12, 0, 13, 1, 2 };
    const std::vector<uint64_t> got  = drain(q);
    expect(r, got == want, "dispatched " + join(got) + ", want " + join(want));
    return r;
}

// A class-0 request for a block a class-1 read has in flight promotes it: from the same context the read moves to
// class 0 at the class-0 key, ahead of the context's other lookahead reads although its layer is later; from another
// context it gains a class-0 entry there. The joined read has both waiters.
// Mutant: LAZY_IO_QUEUE_MUTANT_NO_PROMOTION (a join never changes the read's class).
arm_result arm_promotion() {
    arm_result    r;
    lazy_io_queue q;
    put(q, 1, 0, key1(1, 5));
    put(q, 1, 1, key1(1, 5));
    put(q, 1, 2, key1(1, 5));
    put(q, 1, 10, key0(1, 3));
    put(q, 1, 1, key0(1, 9));
    put(q, 2, 2, key0(7, 9));
    const std::vector<pick_rec> recs = drain_records(q);
    std::vector<uint64_t>       got;
    size_t                      waiters_1 = 0;
    for (const pick_rec & p : recs) {
        got.push_back(p.block);
        if (p.block == 1) {
            waiters_1 = p.waiters;
        }
    }
    const std::vector<uint64_t> want = { 10, 2, 1, 0 };
    expect(r, got == want, "dispatched " + join(got) + ", want " + join(want));
    expect(r, waiters_1 == 2, "block 1's read had " + num(waiters_1) + " waiters, want 2");
    return r;
}

// A span is queued as reads of at most 512 KiB, each with its share of the waiter: 1.5 MiB + 4 KiB is four reads,
// landing at offsets 0, 512 KiB, 1 MiB and 1.5 MiB, and the ticket counts four waiters. A span that is not
// block-aligned is refused and queues nothing.
// Mutant: LAZY_IO_QUEUE_MUTANT_NO_CHUNKING (one read of the whole span).
arm_result arm_chunks() {
    arm_result          r;
    lazy_io_queue       q;
    lazy_gather_request req;
    lazy_io_ticket      t;
    t.request = &req;
    std::vector<ring_lease> spare;
    const uint64_t          len = 3 * LAZY_IO_CHUNK_BYTES + LAZY_IO_BLOCK_BYTES;
    lazy_io_waiter          w;
    w.ticket           = &t;
    w.epoch            = 3;
    w.bytes            = len;
    const size_t added = q.submit(1, -1, { k_file, 1ull << 20, len }, key0(3, 2, lazy_io_kind::EXPERT_SLICE), w, spare);
    expect(r, q.in_flight() == 4,
           "the span was queued as " + num(q.in_flight()) + " reads, want 4 of at most " + num(LAZY_IO_CHUNK_BYTES) +
               " B");
    expect(r, added == 4 && t.outstanding.load() == 4,
           "the submit added " + num(added) + " waiters and the ticket counts " + num(t.outstanding.load()) +
               ", want 4 and 4");
    for (uint64_t i = 0; i < 4; ++i) {
        lazy_io_read * rd = q.pick();
        if (rd == nullptr) {
            expect(r, false, "pick " + num(i) + " found nothing queued, want chunk " + num(i));
            break;
        }
        const uint64_t want_len = i < 3 ? LAZY_IO_CHUNK_BYTES : LAZY_IO_BLOCK_BYTES;
        const bool span_ok = rd->span.offset == (1ull << 20) + i * LAZY_IO_CHUNK_BYTES && rd->span.length == want_len;
        const bool w_ok    = rd->waiters.size() == 1 && rd->waiters[0].src_offset == 0 &&
                          rd->waiters[0].bytes == want_len && rd->waiters[0].landing_offset == i * LAZY_IO_CHUNK_BYTES;
        expect(r, span_ok && w_ok,
               "chunk " + num(i) + " is [" + num(rd->span.offset) + ", +" + num(rd->span.length) + ") with " +
                   num(rd->waiters.size()) + " waiter(s), want [" + num((1ull << 20) + i * LAZY_IO_CHUNK_BYTES) +
                   ", +" + num(want_len) + ") landing at " + num(i * LAZY_IO_CHUNK_BYTES));
        q.complete(rd);
    }
    lazy_io_queue q2;
    const size_t  refused = q2.submit(1, -1, { k_file, 100, LAZY_IO_BLOCK_BYTES }, key0(3, 2), {}, spare);
    expect(r, refused == 0 && q2.in_flight() == 0,
           "a span at offset 100 added " + num(refused) + " waiters and " + num(q2.in_flight()) +
               " reads, want it refused");
    return r;
}

// A request for a block that is already pending joins its read rather than reading it again: two requests for block 5
// and one for block 6 make two reads and three waiters; a request for block 5 after its pick only adds a waiter, and
// block 5's completion hands back all three of its waiters.
// Mutant: LAZY_IO_QUEUE_MUTANT_NO_DEDUP (every request is its own read).
arm_result arm_dedup() {
    arm_result          r;
    lazy_io_queue       q;
    lazy_gather_request req;
    lazy_io_ticket      t;
    t.request = &req;
    lazy_io_waiter w;
    w.ticket = &t;
    put(q, 1, 5, key0(1, 1), w);
    put(q, 1, 5, key0(1, 1), w);
    put(q, 1, 6, key0(1, 1), w);
    expect(r, q.in_flight() == 2,
           "after two requests for block 5 and one for block 6, " + num(q.in_flight()) + " reads in flight, want 2");
    expect(r, t.outstanding.load() == 3, "the ticket counts " + num(t.outstanding.load()) + " waiters, want 3");
    lazy_io_read * first = q.pick();
    expect(r, first != nullptr && block_of(first) == 5, "the first pick is not block 5");
    put(q, 2, 5, key0(1, 1), w);
    expect(r, q.in_flight() == 2 && q.queued_entries() == 1,
           "a request for dispatched block 5 left " + num(q.in_flight()) + " reads and " + num(q.queued_entries()) +
               " queue entries, want 2 and 1");
    if (first != nullptr) {
        std::unique_ptr<lazy_io_read> done = q.complete(first);
        expect(r, done != nullptr && done->waiters.size() == 3,
               "block 5's completion handed back " + num(done ? done->waiters.size() : 0) + " waiters, want 3");
    }
    const std::vector<uint64_t> rest = drain(q);
    expect(r, rest == std::vector<uint64_t>{ 6 }, "after block 5 the queue dispatched " + join(rest) + ", want 6");
    return r;
}

// A join from the same context re-keys a queued read to the more urgent key and keeps its sequence number (§5.2.1,
// witness 4's across-epochs shape): Y, queued last in epoch 6's blk.14 ticket and then joined by epoch 5's blk.14
// ticket, dispatches after the epoch-4 backlog and ahead of every other read of both tickets.
// Mutant: LAZY_IO_QUEUE_MUTANT_JOIN_KEEPS_POSITION (a join never re-keys, so Y dispatches last).
arm_result arm_same_context_rekey() {
    arm_result    r;
    lazy_io_queue q;
    for (uint64_t b = 0; b < 10; ++b) {
        put(q, 1, b, key0(4, 20));
    }
    for (uint64_t b = 20; b < 25; ++b) {
        put(q, 1, b, key0(6, 14));
    }
    put(q, 1, 99, key0(6, 14));
    for (uint64_t b = 30; b < 35; ++b) {
        put(q, 1, b, key0(5, 14));
    }
    put(q, 1, 99, key0(5, 14));
    const std::vector<uint64_t> want = cat(cat(cat(range(0, 9), { 99 }), range(30, 34)), range(20, 24));
    const std::vector<uint64_t> got  = drain(q);
    expect(r, got == want, "dispatched " + join(got) + ", want " + join(want));
    return r;
}

// A join from another context leaves the owner's entry alone and adds a second entry at the joiner's key: block 97,
// near the end of context 1's hundred-read backlog, dispatches at context 2's turn, pick 5, and is dispatched once, so
// the 104 requests dispatch the 103 blocks they name, each once.
// Mutants: LAZY_IO_QUEUE_MUTANT_NO_SECOND_ENTRY (the join only adds a waiter, so 97 waits for context 1's backlog);
// LAZY_IO_QUEUE_MUTANT_CROSS_CONTEXT_REKEY (the join re-keys context 1's entry by context 2's epoch, comparing two
// contexts' counters).
arm_result arm_cross_context_join() {
    arm_result    r;
    lazy_io_queue q;
    for (uint64_t b = 0; b < 100; ++b) {
        put(q, 1, b, key0(7, 10));
    }
    put(q, 2, 200, key0(0, 0));
    put(q, 2, 201, key0(0, 0));
    put(q, 2, 97, key0(0, 0));
    put(q, 2, 202, key0(0, 0));
    const std::vector<uint64_t>                  got     = drain(q);
    const std::vector<lazy_io_dispatch_record> & log     = q.dispatch_log_for_testing();
    const size_t                                 p       = position(got, 97);
    const uint64_t                               context = p != SIZE_MAX && p < log.size() ? log[p].context : 0;
    expect(r, p == 5 && context == 2,
           "block 97 dispatched at pick " + pos_text(p) + " from context " + num(context) +
               ", want pick 5 from context 2");
    std::vector<uint64_t> sorted = got;
    std::sort(sorted.begin(), sorted.end());
    const std::vector<uint64_t> blocks = cat(range(0, 99), range(200, 202));
    expect(r, sorted == blocks,
           "dispatched " + num(got.size()) + " reads, block 97 " + num(std::count(got.begin(), got.end(), 97)) +
               " times, want each of the 103 blocks once");
    return r;
}

// The test-only FIFO override serves reads in enqueue order, whatever their class, context or key: witness 1's
// positive control, which must show the backlog delay that the order of §5.2.1 removes.
// Mutant: LAZY_IO_QUEUE_MUTANT_FIFO_IGNORED (the override is ignored, so context 2's reads take turns).
arm_result arm_fifo_override() {
    arm_result    r;
    lazy_io_queue q;
    q.set_fifo_for_testing(true);
    for (uint64_t b = 0; b < 10; ++b) {
        put(q, 1, b, key0(9, 30));
    }
    for (uint64_t b = 200; b < 204; ++b) {
        put(q, 2, b, key0(0, 0));
    }
    put(q, 1, 50, key1(0, 0));
    const std::vector<uint64_t> want = cat(cat(range(0, 9), range(200, 203)), { 50 });
    const std::vector<uint64_t> got  = drain(q);
    expect(r, got == want, "dispatched " + join(got) + ", want " + join(want));
    return r;
}

// The dispatch log numbers a read when a worker picks it up, not when it completes: three reads picked in order and
// completed in reverse are logged in pick order with numbers 1, 2, 3, and each read carries its number.
// Mutant: LAZY_IO_QUEUE_MUTANT_LOG_AT_COMPLETION (the number is taken at the completion).
arm_result arm_dispatch_log() {
    arm_result    r;
    lazy_io_queue q;
    for (uint64_t b = 0; b < 3; ++b) {
        put(q, 1, b, key0(1, 1));
    }
    std::vector<lazy_io_read *> picked;
    for (int i = 0; i < 3; ++i) {
        lazy_io_read * rd = q.pick();
        if (rd != nullptr) {
            picked.push_back(rd);
        }
    }
    expect(r, picked.size() == 3, "picked " + num(picked.size()) + " reads, want 3");
    std::vector<uint64_t> seqs;
    for (size_t i = picked.size(); i-- > 0;) {
        std::unique_ptr<lazy_io_read> done = q.complete(picked[i]);
        seqs.insert(seqs.begin(), done ? done->dispatch_seq : 0);
    }
    std::vector<uint64_t> logged_blocks;
    std::vector<uint64_t> logged_seqs;
    for (const lazy_io_dispatch_record & rec : q.dispatch_log_for_testing()) {
        logged_blocks.push_back(rec.span.offset / LAZY_IO_BLOCK_BYTES);
        logged_seqs.push_back(rec.dispatch_seq);
    }
    const std::vector<uint64_t> want_blocks = { 0, 1, 2 };
    const std::vector<uint64_t> want_seqs   = { 1, 2, 3 };
    expect(r, logged_blocks == want_blocks, "the dispatch log lists blocks " + join(logged_blocks) + ", want 0 1 2");
    expect(r, logged_seqs == want_seqs && seqs == want_seqs,
           "the log numbers " + join(logged_seqs) + " and the reads carry " + join(seqs) + ", want 1 2 3 and 1 2 3");
    return r;
}

// --- the delivery arm's row cache, on plain memory ---

class plain_memory final : public lazy_row_cache_memory {
  public:
    plain_memory(uint8_t * data, size_t bytes) : lazy_row_cache_memory(data, bytes, bytes) {}

    ~plain_memory() override { ::operator delete[](data(), std::align_val_t(64)); }
};

class plain_source final : public lazy_row_cache_memory_source {
  public:
    std::unique_ptr<lazy_row_cache_memory> allocate(size_t bytes) override {
        uint8_t * p = static_cast<uint8_t *>(::operator new[](bytes, std::align_val_t(64)));
        std::memset(p, 0, bytes);
        return std::unique_ptr<lazy_row_cache_memory>(new plain_memory(p, bytes));
    }
};

constexpr size_t k_row = 170;

bool cache_holds(lazy_row_cache & c, int64_t row, const uint8_t * want) {
    std::vector<uint8_t> got(k_row);
    return c.lookup_copy(row, got.data()) && std::memcmp(got.data(), want, k_row) == 0;
}

// The reaper copies a waiter's bytes into the landing slot only while the slot's epoch is the waiter's and the request
// is neither cancelled nor draining; it inserts the row into the cache either way and counts the ticket down, the last
// delivery returning true. A failed read copies and inserts nothing, keeps the errno in the ticket, and counts down.
// Mutant: LAZY_IO_QUEUE_MUTANT_DELIVER_IGNORES_EPOCH (the copy skips the epoch and cancel check).
arm_result arm_delivery() {
    arm_result            r;
    plain_source          source;
    lazy_row_cache_config cfg;
    cfg.name                   = "fixture.lazy";
    cfg.row_size               = k_row;
    cfg.n_rows                 = 1000;
    cfg.cache_slots            = 8;
    cfg.floor_slots            = 8;
    cfg.load_txn_id            = 0x51;
    cfg.chunk_bytes            = 8 * k_row;
    cfg.n_shards               = 1;
    lazy_row_cache_build build = lazy_row_cache::build(cfg, source);
    if (!build) {
        host_ledger_test_harness::void_arm(r, "the fixture row cache did not build");
        return r;
    }
    lazy_row_cache & cache = *build.cache;

    std::vector<uint8_t> block(LAZY_IO_BLOCK_BYTES);
    for (size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<uint8_t>((i * 7) ^ 0x5a);
    }
    std::vector<uint8_t> slab(4 * k_row, 0);
    lazy_landing_slot    slot;
    slot.epoch = 5;
    slot.data  = slab.data();
    slot.bytes = slab.size();
    lazy_gather_request req;
    lazy_io_ticket      t;
    t.request  = &req;
    req.ticket = &t;
    t.outstanding.store(4);

    auto waiter = [&](uint32_t epoch, size_t landing_offset, int64_t row) {
        lazy_io_waiter w;
        w.ticket         = &t;
        w.epoch          = epoch;
        w.landing        = &slot;
        w.landing_offset = landing_offset;
        w.src_offset     = 100;
        w.bytes          = k_row;
        w.cache          = &cache;
        w.row            = row;
        return w;
    };
    const uint8_t *            src = block.data() + 100;
    const std::vector<uint8_t> zeros(k_row, 0);

    const bool last0 = lazy_io_deliver(waiter(5, 0, 3), block.data(), 0);
    expect(r, std::memcmp(slab.data(), src, k_row) == 0, "the epoch-5 waiter's bytes did not land in the epoch-5 slot");
    const bool last1 = lazy_io_deliver(waiter(4, k_row, 4), block.data(), 0);
    expect(r, std::memcmp(slab.data() + k_row, zeros.data(), k_row) == 0,
           "a waiter for epoch 4 wrote into the slot at epoch 5");
    req.cancelled.store(true);
    const bool last2 = lazy_io_deliver(waiter(5, 2 * k_row, 5), block.data(), 0);
    expect(r, std::memcmp(slab.data() + 2 * k_row, zeros.data(), k_row) == 0,
           "a waiter of a cancelled request wrote into the slot");
    req.cancelled.store(false);
    req.drain.store(true);
    const bool last3 = lazy_io_deliver(waiter(5, 3 * k_row, 6), block.data(), 0);
    expect(r, std::memcmp(slab.data() + 3 * k_row, zeros.data(), k_row) == 0,
           "a waiter of a draining request wrote into the slot");
    expect(r,
           cache_holds(cache, 3, src) && cache_holds(cache, 4, src) && cache_holds(cache, 5, src) &&
               cache_holds(cache, 6, src),
           "rows 3 to 6 are not all in the cache: the insert must happen whether or not the copy did");
    expect(r, !last0 && !last1 && !last2 && last3 && t.outstanding.load() == 0,
           "the four deliveries returned " + num(last0) + num(last1) + num(last2) + num(last3) + " leaving " +
               num(t.outstanding.load()) + " outstanding, want 0001 leaving 0");

    lazy_io_ticket t2;
    t2.request = &req;
    t2.outstanding.store(1);
    req.drain.store(false);
    lazy_io_waiter w5 = waiter(5, 0, 7);
    w5.ticket         = &t2;
    std::memset(slab.data(), 0, slab.size());
    const bool last5 = lazy_io_deliver(w5, block.data(), EIO);
    expect(r, std::memcmp(slab.data(), zeros.data(), k_row) == 0 && !cache.resident(7),
           "a failed read copied into the slot or inserted its row");
    expect(r, last5 && t2.read_error.load() == EIO && t2.outstanding.load() == 0,
           "a failed read left read_error " + num(static_cast<uint64_t>(t2.read_error.load())) + " and " +
               num(t2.outstanding.load()) + " outstanding, want " + num(EIO) + " and 0, returning true");
    return r;
}

}  // namespace

int main() {
    const std::vector<named_arm> arms = {
        { "class 0 dispatches by (epoch, deadline), rows first, equal keys in order", arm_key_order          },
        { "the epoch order survives the counter's wrap",                              arm_epoch_wrap         },
        { "the contexts take turns",                                                  arm_round_robin        },
        { "class 1 waits for every context's class 0",                                arm_class_order        },
        { "a class-0 join promotes a lookahead read",                                 arm_promotion          },
        { "a span is read in chunks of at most 512 KiB",                              arm_chunks             },
        { "a request for a pending block joins its read",                             arm_dedup              },
        { "a same-context join takes the more urgent key and keeps its number",       arm_same_context_rekey },
        { "a join from another context adds an entry at the joiner's key",            arm_cross_context_join },
        { "the FIFO override serves reads in enqueue order",                          arm_fifo_override      },
        { "the dispatch log numbers a read when a worker picks it up",                arm_dispatch_log       },
        { "the reaper copies only into its own epoch's live slot",                    arm_delivery           },
    };
    return run_arms(arms, "lazy io queue host arms");
}
