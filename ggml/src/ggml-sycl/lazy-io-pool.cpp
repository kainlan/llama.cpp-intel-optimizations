#include "lazy-io-pool.hpp"

#include "ggml-impl.h"
#include "lazy-row-cache.hpp"

#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

// The IO pool registry (docs/design/sycl-storage-tier-lazy-tables.md §3.3). Ownership: while a pool lives the
// registry's slot owns it, and the pool owns its reap node, allocated with the pool outside the lock. The node holds a
// back-pointer to the pool. The drop that reaches zero parks the pool: under the leaf mutex it marks the pool DYING,
// clears the slot and releases the node into the reap list, and from then on the node owns the pool. So the park
// links two pointers and allocates nothing, and a take never sees a parked pool. A FULL reap frees each parked pool and
// then its node. A pool that dies without parking, the install-race loser, frees its unparked node in its destructor.
//
// The registry's leaf mutex (canonical contract §12.5, L5) guards the slot and its generation, the reap list, the lease
// counts, the DYING flags, the counters and the hold hook. Nothing is taken, allocated, freed, waited on or logged
// under it. The slot is a raw pointer on purpose: a static owner would free the pool during static destruction under a
// lease that still points at it, while the shut-down flag lets such a lease decrement without the mutex instead.

namespace ggml_sycl {

struct lazy_io_pool;

struct lazy_reap_node {
    lazy_io_pool *   pool = nullptr;  // a back-pointer while the pool lives; the owning pointer once it is parked
    lazy_reap_node * next = nullptr;
};

struct lazy_io_pool {
    std::atomic<uint64_t>           leases{ 0 };     // changed under the leaf mutex, and without it once shut down
    uint64_t                        generation = 0;  // the generation its take reserved, set before the install
    bool                            dying      = false;
    std::unique_ptr<lazy_reap_node> node;            // owned until the park releases it into the reap list
};

namespace {

std::mutex        g_lazy_registry_mutex;
lazy_io_pool *    g_lazy_pool       = nullptr;  // the live pool, owned by the registry
uint64_t          g_lazy_generation = 0;        // the last installed pool's generation; a take reserves the next one
lazy_reap_node *  g_lazy_reap_list  = nullptr;  // the parked pools, each node owning its pool
std::atomic<bool> g_lazy_shut_down{ false };
uint64_t          g_lazy_parked_nodes           = 0;
uint64_t          g_lazy_full_reaps_run         = 0;
uint64_t          g_lazy_reaps_run_at_last_take = 0;
uint64_t          g_lazy_parked_at_last_take    = 0;
#if defined(GGML_SYCL_PRIVATE_TESTING)
// The one-shot hold hook and the point it is armed at: the first FULL reap, or the first take that builds a pool.
enum class lazy_hold_point : uint8_t { REAP, TAKE };

lazy_hold_point g_lazy_hold_point = lazy_hold_point::REAP;
lazy_hold_fn    g_lazy_hold       = nullptr;
void *          g_lazy_hold_ctx   = nullptr;

// Hands the hook to its first caller at the point it is armed at, and disarms it. Caller holds g_lazy_registry_mutex.
lazy_hold_fn lazy_hold_claim_locked(lazy_hold_point point, void ** ctx) {
    if (g_lazy_hold == nullptr || g_lazy_hold_point != point) {
        return nullptr;
    }
    lazy_hold_fn hold = g_lazy_hold;
    *ctx              = g_lazy_hold_ctx;
    g_lazy_hold       = nullptr;
    g_lazy_hold_ctx   = nullptr;
    return hold;
}

void lazy_hold_arm(lazy_hold_point point, lazy_hold_fn hold, void * ctx) {
    std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
    g_lazy_hold_point = point;
    g_lazy_hold       = hold;
    g_lazy_hold_ctx   = hold != nullptr ? ctx : nullptr;
}
#endif

// The take-time snapshots, read with the counters. Caller holds g_lazy_registry_mutex.
void lazy_snapshot_take_locked() {
    g_lazy_reaps_run_at_last_take = g_lazy_full_reaps_run;
    g_lazy_parked_at_last_take    = g_lazy_parked_nodes;
}

}  // namespace

ring_lease::~ring_lease() {
    reset();
}

ring_lease::ring_lease(ring_lease && other) noexcept : pool_(other.pool_) {
    other.pool_ = nullptr;
}

ring_lease & ring_lease::operator=(ring_lease && other) noexcept {
    if (this != &other) {
        reset();
        pool_       = other.pool_;
        other.pool_ = nullptr;
    }
    return *this;
}

ring_lease ring_lease::clone() const noexcept {
    if (pool_ == nullptr) {
        return ring_lease();
    }
    bool dying = false;
    {
        std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
        dying = pool_->dying;
        if (!dying) {
            pool_->leases.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (dying) {
        GGML_LOG_WARN("[HOST-PLAN-BUG] ring_lease::clone found its pool DYING under a held lease\n");
        GGML_ABORT("ring_lease::clone: the pool is DYING while a lease on it is held");
    }
    return ring_lease(pool_);
}

uint64_t ring_lease::ring_bytes() const noexcept {
    return 0;
}

void ring_lease::reset() noexcept {
    lazy_io_pool * pool = pool_;
    if (pool == nullptr) {
        return;
    }
    pool_ = nullptr;
    if (g_lazy_shut_down.load(std::memory_order_acquire)) {
        pool->leases.fetch_sub(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
    if (pool->leases.fetch_sub(1, std::memory_order_relaxed) != 1 || g_lazy_shut_down.load(std::memory_order_relaxed)) {
        return;
    }
    pool->dying = true;
    if (g_lazy_pool == pool) {
        g_lazy_pool = nullptr;
    }
    lazy_reap_node * node = pool->node.release();
    node->next            = g_lazy_reap_list;
    g_lazy_reap_list      = node;
    ++g_lazy_parked_nodes;
}

// Canon's reserve, build-outside, install pattern (§3.3). Under the lock a take joins the live pool, or reserves the
// generation after the last one installed. It builds the pool and its reap node after the unlock, and under the lock
// again installs the pool only if the generation still matches, that is, if no take installed a pool meanwhile. A take
// that lost joins the pool the winner installed; when that pool has already parked, the loser destroys its own after
// the unlock and takes again. So a loser never installs, and no two pools carry one generation.
//
// LAZY_IO_POOL_MUTANT_INSTALL_ON_EMPTY_SLOT installs on an empty slot whatever the generation: a take whose generation
// another take installed and parked during its build then installs a second pool of that generation. Only
// test-sycl-lazy-io-pool-host's mutant target defines it.
ring_lease lazy_io_pool_take() {
    for (;;) {
        uint64_t reserved = 0;
#if defined(GGML_SYCL_PRIVATE_TESTING)
        lazy_hold_fn hold     = nullptr;
        void *       hold_ctx = nullptr;
#endif
        {
            std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
            if (g_lazy_shut_down.load(std::memory_order_relaxed)) {
                return ring_lease();
            }
            if (g_lazy_pool != nullptr) {
                g_lazy_pool->leases.fetch_add(1, std::memory_order_relaxed);
                lazy_snapshot_take_locked();
                return ring_lease(g_lazy_pool);
            }
            reserved = g_lazy_generation + 1;
#if defined(GGML_SYCL_PRIVATE_TESTING)
            hold = lazy_hold_claim_locked(lazy_hold_point::TAKE, &hold_ctx);
#endif
        }

        std::unique_ptr<lazy_io_pool> built(new lazy_io_pool());
        built->generation = reserved;
        built->node.reset(new lazy_reap_node());
        built->node->pool = built.get();
#if defined(GGML_SYCL_PRIVATE_TESTING)
        if (hold != nullptr) {
            hold(hold_ctx);
        }
#endif

        lazy_io_pool * held  = nullptr;
        bool           again = false;
        {
            std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
#if defined(LAZY_IO_POOL_MUTANT_INSTALL_ON_EMPTY_SLOT)
            const bool install = g_lazy_pool == nullptr;
#else
            const bool install = g_lazy_generation + 1 == reserved;
#endif
            if (g_lazy_shut_down.load(std::memory_order_relaxed)) {
                // built is destroyed outside the lock, with its unparked node
            } else if (install) {
                built->leases.store(1, std::memory_order_relaxed);
                g_lazy_generation = reserved;
                g_lazy_pool       = built.release();
                held              = g_lazy_pool;
                lazy_snapshot_take_locked();
            } else if (g_lazy_pool != nullptr) {
                // A take installed a pool meanwhile: join it, and destroy this one outside the lock.
                g_lazy_pool->leases.fetch_add(1, std::memory_order_relaxed);
                held = g_lazy_pool;
                lazy_snapshot_take_locked();
            } else {
                // A take installed a pool meanwhile and it has already parked: destroy this one and take again.
                again = true;
            }
        }
        if (held != nullptr) {
            return ring_lease(held);
        }
        if (!again) {
            return ring_lease();
        }
    }
}

void ggml_sycl_lazy_ring_reap(lazy_reap_scope scope) noexcept {
    GGML_UNUSED(scope);  // FULL is the only scope until the IO engine adds WORKER
#if defined(GGML_SYCL_PRIVATE_TESTING)
    lazy_hold_fn hold     = nullptr;
    void *       hold_ctx = nullptr;
#endif
    {
        std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
        ++g_lazy_full_reaps_run;
#if defined(GGML_SYCL_PRIVATE_TESTING)
        hold = lazy_hold_claim_locked(lazy_hold_point::REAP, &hold_ctx);
#endif
    }
#if defined(GGML_SYCL_PRIVATE_TESTING)
    if (hold != nullptr) {
        hold(hold_ctx);
    }
#endif
    lazy_reap_node * list = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
        list                = g_lazy_reap_list;
        g_lazy_reap_list    = nullptr;
        g_lazy_parked_nodes = 0;
    }
    while (list != nullptr) {
        lazy_reap_node * next = list->next;
        delete list->pool;
        delete list;
        list = next;
    }
}

void lazy_io_pool_mark_shut_down() noexcept {
    std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
    g_lazy_shut_down.store(true, std::memory_order_release);
}

// LAZY_IO_POOL_MUTANT_NO_ENTRY_REAP and LAZY_IO_POOL_MUTANT_NO_CLOSING_REAP each drop one of the wrapper's two reaps.
// Only test-sycl-lazy-io-pool-host's mutant targets define them; each passes only when the host test prints that
// reap's FAIL text.
lazy_coverage_scope::reap_guard::reap_guard() noexcept {
#if !defined(LAZY_IO_POOL_MUTANT_NO_ENTRY_REAP)
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
#endif
}

lazy_coverage_scope::reap_guard::~reap_guard() {
#if !defined(LAZY_IO_POOL_MUTANT_NO_CLOSING_REAP)
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
#endif
}

bool lazy_coverage_scope::take_if(bool has_lazy_candidate) {
    if (!has_lazy_candidate) {
        return false;
    }
    lease_ = lazy_io_pool_take();
    return lease_.held();
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
bool ggml_sycl_lazy_registry_for_testing(lazy_registry_state * out) {
    if (out == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
    out->live_pools             = g_lazy_pool != nullptr ? 1 : 0;
    out->live_leases            = g_lazy_pool != nullptr ? g_lazy_pool->leases.load(std::memory_order_relaxed) : 0;
    out->generation             = g_lazy_generation;
    out->live_generation        = g_lazy_pool != nullptr ? g_lazy_pool->generation : 0;
    out->parked_nodes           = g_lazy_parked_nodes;
    out->full_reaps_run         = g_lazy_full_reaps_run;
    out->reaps_run_at_last_take = g_lazy_reaps_run_at_last_take;
    out->parked_at_last_take    = g_lazy_parked_at_last_take;
    return true;
}

void ggml_sycl_test_lazy_reap_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::REAP, hold, ctx);
}

void ggml_sycl_test_lazy_take_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::TAKE, hold, ctx);
}

void ggml_sycl_test_lazy_mark_dying(const ring_lease & lease) {
    if (lease.pool_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_lazy_registry_mutex);
    lease.pool_->dying = true;
}

lazy_test_coverage_pin::lazy_test_coverage_pin(ring_lease lease) noexcept : lease_(std::move(lease)) {}

lazy_test_coverage_pin::~lazy_test_coverage_pin() {
    lease_.reset();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
}

lazy_test_coverage_pin ggml_sycl_test_make_coverage_pin(uint64_t ring_bytes) {
    if (ring_bytes != 0) {
        GGML_ABORT("ggml_sycl_test_make_coverage_pin: ring_bytes=%llu; only a ring-less pool (0) is accepted",
                   (unsigned long long) ring_bytes);
    }
    return lazy_test_coverage_pin(lazy_io_pool_take());
}
#endif

// ---- the IO engine's queue (§5.2.1) ----
//
// Mutant macros, each defined only by one of test-sycl-lazy-io-queue-host's mutant targets, which passes only when the
// arm that kills it prints its FAIL line: LAZY_IO_QUEUE_MUTANT_DEADLINE_ONLY, _SLICES_FIRST, _EQUAL_KEY_LIFO,
// _PLAIN_EPOCH_COMPARE, _NO_ROUND_ROBIN, _CLASS_BLIND, _NO_PROMOTION, _NO_CHUNKING, _NO_DEDUP, _JOIN_KEEPS_POSITION,
// _NO_SECOND_ENTRY, _CROSS_CONTEXT_REKEY, _FIFO_IGNORED, _LOG_AT_COMPLETION and _DELIVER_IGNORES_EPOCH.

namespace {

// Serial-number order: a before b when b is less than half the counter's range ahead of a.
bool lazy_io_epoch_before(uint32_t a, uint32_t b) {
#if defined(LAZY_IO_QUEUE_MUTANT_PLAIN_EPOCH_COMPARE)
    return a < b;
#else
    return static_cast<int32_t>(a - b) < 0;
#endif
}

// Whether `a` is more urgent than `b` (§5.2.1): (class, epoch, deadline, kind) lexicographically. Only keys of one
// context are ever compared.
bool lazy_io_key_before(const lazy_io_key & a, const lazy_io_key & b) {
#if !defined(LAZY_IO_QUEUE_MUTANT_CLASS_BLIND)
    if (a.cls != b.cls) {
        return a.cls < b.cls;
    }
#endif
#if !defined(LAZY_IO_QUEUE_MUTANT_DEADLINE_ONLY)
    if (a.epoch != b.epoch) {
        return lazy_io_epoch_before(a.epoch, b.epoch);
    }
#endif
    if (a.deadline != b.deadline) {
        return a.deadline < b.deadline;
    }
#if defined(LAZY_IO_QUEUE_MUTANT_SLICES_FIRST)
    return a.kind > b.kind;
#else
    return a.kind < b.kind;
#endif
}

// The class queue a key sits in: 0 for class 0, 1 for class 1.
size_t lazy_io_class_index(const lazy_io_key & key) {
#if defined(LAZY_IO_QUEUE_MUTANT_CLASS_BLIND)
    GGML_UNUSED(key);
    return 0;
#else
    return key.cls == lazy_io_class::PREFETCH ? 0 : 1;
#endif
}

bool lazy_io_span_less(const lazy_io_span & a, const lazy_io_span & b) {
    if (a.file != b.file) {
        return a.file < b.file;
    }
    if (a.offset != b.offset) {
        return a.offset < b.offset;
    }
    return a.length < b.length;
}

}  // namespace

struct lazy_io_queue::state {
    // One queue entry: a read's place in one context's class queue. A read has at most one entry per context.
    struct entry {
        lazy_io_read * read    = nullptr;
        uint64_t       context = 0;
        lazy_io_key    key;
        uint64_t       seq = 0;  // the enqueue sequence number, kept across a re-key
    };

    struct entry_order {
        bool operator()(const entry * a, const entry * b) const {
            if (lazy_io_key_before(a->key, b->key)) {
                return true;
            }
            if (lazy_io_key_before(b->key, a->key)) {
                return false;
            }
#if defined(LAZY_IO_QUEUE_MUTANT_EQUAL_KEY_LIFO)
            return a->seq > b->seq;
#else
            return a->seq < b->seq;
#endif
        }
    };

    using entry_set = std::set<entry *, entry_order>;

    struct context_queues {
        entry_set by_class[2];
    };

    // A read in flight, queued or dispatched, with the entries that queue it.
    struct node {
        std::unique_ptr<lazy_io_read>       read;
        std::vector<std::unique_ptr<entry>> entries;
    };

    struct span_order {
        bool operator()(const lazy_io_span & a, const lazy_io_span & b) const { return lazy_io_span_less(a, b); }
    };

    // A multimap only so the no-dedup mutant can hold two reads of one span; production finds at most one.
    std::multimap<lazy_io_span, node, span_order> in_flight;
    std::map<uint64_t, context_queues>            contexts;
    std::map<uint64_t, entry *>                   by_seq;  // every entry by its sequence number: the FIFO order
    uint64_t                                      next_seq      = 0;
    uint64_t                                      next_dispatch = 0;
    uint64_t                                      cursor        = 0;  // the context the last pick served
    bool                                          cursor_set    = false;
    size_t                                        class0        = 0;  // entries queued in class 0, every context
#if defined(GGML_SYCL_PRIVATE_TESTING)
    bool                                 fifo = false;
    std::vector<lazy_io_dispatch_record> log;
#endif

    node * find(const lazy_io_span & span) {
        auto it = in_flight.find(span);
        return it == in_flight.end() ? nullptr : &it->second;
    }

    node * node_of(const lazy_io_read * read) {
        auto range = in_flight.equal_range(read->span);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second.read.get() == read) {
                return &it->second;
            }
        }
        return nullptr;
    }

    void insert_entry(entry * e) {
        contexts[e->context].by_class[lazy_io_class_index(e->key)].insert(e);
        by_seq[e->seq] = e;
        class0 += lazy_io_class_index(e->key) == 0 ? 1 : 0;
    }

    // Takes `e` out of its context's queue and the FIFO index, leaving it owned by its node.
    void unlink_entry(entry * e) {
        auto ctx = contexts.find(e->context);
        if (ctx != contexts.end()) {
            ctx->second.by_class[lazy_io_class_index(e->key)].erase(e);
            if (ctx->second.by_class[0].empty() && ctx->second.by_class[1].empty()) {
                contexts.erase(ctx);
            }
        }
        by_seq.erase(e->seq);
        class0 -= lazy_io_class_index(e->key) == 0 ? 1 : 0;
    }

    void add_entry(node & n, uint64_t context, const lazy_io_key & key) {
        std::unique_ptr<entry> e(new entry());
        e->read    = n.read.get();
        e->context = context;
        e->key     = key;
        e->seq     = next_seq++;
        insert_entry(e.get());
        n.entries.push_back(std::move(e));
    }

    // A re-key: out of its queue under the old key, back in under the new one, keeping its sequence number.
    void rekey(entry * e, const lazy_io_key & key) {
        unlink_entry(e);
        e->key = key;
        insert_entry(e);
    }

    void drop_entries(node & n) {
        for (std::unique_ptr<entry> & e : n.entries) {
            unlink_entry(e.get());
        }
        n.entries.clear();
    }

    entry * entry_in(node & n, uint64_t context) {
        for (std::unique_ptr<entry> & e : n.entries) {
            if (e->context == context) {
                return e.get();
            }
        }
        return nullptr;
    }

    // A join of a queued read (§5.2.1). From the same context: the more urgent of the two keys, the number kept. From
    // another context: a second entry in the joiner's queue at the joiner's key, the owner's entry untouched.
    void join_queued(node & n, uint64_t context, const lazy_io_key & key) {
        entry * own = entry_in(n, context);
        if (own != nullptr) {
#if defined(LAZY_IO_QUEUE_MUTANT_JOIN_KEEPS_POSITION)
            return;
#elif defined(LAZY_IO_QUEUE_MUTANT_NO_PROMOTION)
            if (key.cls == own->key.cls && lazy_io_key_before(key, own->key)) {
                rekey(own, key);
            }
            return;
#else
            if (lazy_io_key_before(key, own->key)) {
                rekey(own, key);
            }
            return;
#endif
        }
#if defined(LAZY_IO_QUEUE_MUTANT_NO_SECOND_ENTRY)
        return;
#elif defined(LAZY_IO_QUEUE_MUTANT_CROSS_CONTEXT_REKEY)
        if (!n.entries.empty() && lazy_io_key_before(key, n.entries.front()->key)) {
            rekey(n.entries.front().get(), key);
        }
        return;
#else
        add_entry(n, context, key);
#endif
    }

    // The round-robin pick within one class: the first context after the last one served whose class queue is not
    // empty, wrapping.
    entry * pick_round_robin(size_t cls) {
#if defined(LAZY_IO_QUEUE_MUTANT_NO_ROUND_ROBIN)
        for (auto & ctx : contexts) {
            if (!ctx.second.by_class[cls].empty()) {
                return *ctx.second.by_class[cls].begin();
            }
        }
        return nullptr;
#else
        auto start = cursor_set ? contexts.upper_bound(cursor) : contexts.begin();
        for (auto it = start; it != contexts.end(); ++it) {
            if (!it->second.by_class[cls].empty()) {
                return *it->second.by_class[cls].begin();
            }
        }
        for (auto it = contexts.begin(); it != start; ++it) {
            if (!it->second.by_class[cls].empty()) {
                return *it->second.by_class[cls].begin();
            }
        }
        return nullptr;
#endif
    }

    void record_dispatch(lazy_io_read * read, const entry & e) {
        read->dispatch_seq = ++next_dispatch;
#if defined(GGML_SYCL_PRIVATE_TESTING)
        lazy_io_dispatch_record rec;
        rec.dispatch_seq = read->dispatch_seq;
        rec.context      = e.context;
        rec.span         = read->span;
        rec.key          = e.key;
        rec.enqueue_seq  = e.seq;
        log.push_back(rec);
#else
        GGML_UNUSED(e);
#endif
    }

#if defined(LAZY_IO_QUEUE_MUTANT_LOG_AT_COMPLETION)
    std::map<const lazy_io_read *, entry> picked_by;
#endif
};

lazy_io_queue::lazy_io_queue() : s_(new state()) {}

lazy_io_queue::~lazy_io_queue() = default;

size_t lazy_io_queue::submit(uint64_t                  context,
                             int                       fd,
                             const lazy_io_span &      span,
                             const lazy_io_key &       key,
                             const lazy_io_waiter &    waiter,
                             std::vector<ring_lease> & spare_leases) {
    if (span.length == 0 || span.offset % LAZY_IO_BLOCK_BYTES != 0 || span.length % LAZY_IO_BLOCK_BYTES != 0 ||
        span.length > UINT64_MAX - span.offset) {
        return 0;
    }
    const uint64_t end   = span.offset + span.length;
    const uint64_t w0    = span.offset + waiter.src_offset;
    const uint64_t w1    = w0 + waiter.bytes;
    size_t         added = 0;
    for (uint64_t c0 = span.offset; c0 < end;) {
#if defined(LAZY_IO_QUEUE_MUTANT_NO_CHUNKING)
        const uint64_t c1 = end;
#else
        const uint64_t c1 = end - c0 > LAZY_IO_CHUNK_BYTES ? c0 + LAZY_IO_CHUNK_BYTES : end;
#endif
        const lazy_io_span chunk     = { span.file, c0, c1 - c0 };
        lazy_io_waiter     share     = waiter;
        bool               has_share = waiter.bytes == 0;
        if (waiter.bytes != 0) {
            const uint64_t lo = w0 > c0 ? w0 : c0;
            const uint64_t hi = w1 < c1 ? w1 : c1;
            if (hi > lo) {
                share.src_offset     = lo - c0;
                share.bytes          = static_cast<size_t>(hi - lo);
                share.landing_offset = waiter.landing_offset + static_cast<size_t>(lo - w0);
                has_share            = true;
            }
        }
#if defined(LAZY_IO_QUEUE_MUTANT_NO_DEDUP)
        state::node * n = nullptr;
#else
        state::node * n = s_->find(chunk);
#endif
        if (n == nullptr) {
            state::node fresh;
            fresh.read.reset(new lazy_io_read());
            fresh.read->span = chunk;
            fresh.read->fd   = fd;
            n                = &s_->in_flight.emplace(chunk, std::move(fresh))->second;
            s_->add_entry(*n, context, key);
            // The lease moves in last, so a throw above leaves it with the caller, who drops it after unlocking.
            if (!spare_leases.empty()) {
                n->read->lease = std::move(spare_leases.back());
                spare_leases.pop_back();
            }
        } else if (!n->read->dispatched) {
            s_->join_queued(*n, context, key);
        }
        if (has_share) {
            n->read->waiters.push_back(share);
            if (share.ticket != nullptr) {
                share.ticket->outstanding.fetch_add(1, std::memory_order_relaxed);
            }
            ++added;
        }
        c0 = c1;
    }
    return added;
}

lazy_io_read * lazy_io_queue::pick() {
    state & s = *s_;
    if (s.by_seq.empty()) {
        return nullptr;
    }
    state::entry * e = nullptr;
#if defined(GGML_SYCL_PRIVATE_TESTING)
    if (s.fifo) {
        e = s.by_seq.begin()->second;
    }
#endif
    if (e == nullptr) {
        e = s.pick_round_robin(s.class0 > 0 ? 0 : 1);
    }
    if (e == nullptr) {
        return nullptr;
    }
    lazy_io_read * read = e->read;
    state::node *  n    = s.node_of(read);
    s.cursor            = e->context;
    s.cursor_set        = true;
    read->dispatched    = true;
#if defined(LAZY_IO_QUEUE_MUTANT_LOG_AT_COMPLETION)
    s.picked_by[read] = *e;
#else
    s.record_dispatch(read, *e);
#endif
    if (n != nullptr) {
        s.drop_entries(*n);
    }
    return read;
}

std::unique_ptr<lazy_io_read> lazy_io_queue::complete(lazy_io_read * read) {
    state & s     = *s_;
    auto    range = s.in_flight.equal_range(read->span);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.read.get() == read) {
            s.drop_entries(it->second);
            std::unique_ptr<lazy_io_read> out = std::move(it->second.read);
            s.in_flight.erase(it);
#if defined(LAZY_IO_QUEUE_MUTANT_LOG_AT_COMPLETION)
            auto picked = s.picked_by.find(read);
            if (picked != s.picked_by.end()) {
                s.record_dispatch(read, picked->second);
                s.picked_by.erase(picked);
            }
#endif
            return out;
        }
    }
    return nullptr;
}

size_t lazy_io_queue::queued_entries() const {
    return s_->by_seq.size();
}

size_t lazy_io_queue::in_flight() const {
    return s_->in_flight.size();
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
void lazy_io_queue::set_fifo_for_testing(bool fifo) {
#    if defined(LAZY_IO_QUEUE_MUTANT_FIFO_IGNORED)
    GGML_UNUSED(fifo);
#    else
    s_->fifo = fifo;
#    endif
}

const std::vector<lazy_io_dispatch_record> & lazy_io_queue::dispatch_log_for_testing() const {
    return s_->log;
}
#endif

bool lazy_io_deliver(const lazy_io_waiter & waiter, const uint8_t * read_bytes, int read_errno) noexcept {
    lazy_io_ticket * const ticket = waiter.ticket;
    if (read_errno == 0) {
        if (waiter.landing != nullptr && waiter.bytes != 0) {
            lazy_landing_slot &         slot = *waiter.landing;
            std::lock_guard<std::mutex> lock(slot.lock);
            const lazy_gather_request * req  = ticket != nullptr ? ticket->request : nullptr;
            const bool                  stop = req != nullptr && (req->cancelled.load(std::memory_order_acquire) ||
                                                 req->drain.load(std::memory_order_acquire));
#if defined(LAZY_IO_QUEUE_MUTANT_DELIVER_IGNORES_EPOCH)
            GGML_UNUSED(stop);
            const bool live = true;
#else
            const bool live = slot.epoch == waiter.epoch && !stop;
#endif
            if (live && waiter.landing_offset <= slot.bytes && waiter.bytes <= slot.bytes - waiter.landing_offset) {
                std::memcpy(slot.data + waiter.landing_offset, read_bytes + waiter.src_offset, waiter.bytes);
            }
        }
        if (waiter.cache != nullptr) {
            waiter.cache->insert(waiter.row, read_bytes + waiter.src_offset);
        }
    } else if (ticket != nullptr) {
        int none = 0;
        ticket->read_error.compare_exchange_strong(none, read_errno, std::memory_order_release,
                                                   std::memory_order_relaxed);
    }
    if (ticket == nullptr) {
        return false;
    }
    return ticket->outstanding.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

}  // namespace ggml_sycl
