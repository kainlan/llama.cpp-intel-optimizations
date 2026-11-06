#include "lazy-io-pool.hpp"

#include "ggml-impl.h"

#include <atomic>
#include <memory>
#include <mutex>
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

// ---- the IO engine's queue (§5.2.1): RED stubs, replaced by the queue ----

struct lazy_io_queue::state {
#if defined(GGML_SYCL_PRIVATE_TESTING)
    std::vector<lazy_io_dispatch_record> log;
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
    GGML_UNUSED(context);
    GGML_UNUSED(fd);
    GGML_UNUSED(span);
    GGML_UNUSED(key);
    GGML_UNUSED(waiter);
    GGML_UNUSED(spare_leases);
    return 0;
}

lazy_io_read * lazy_io_queue::pick() {
    return nullptr;
}

std::unique_ptr<lazy_io_read> lazy_io_queue::complete(lazy_io_read * read) {
    GGML_UNUSED(read);
    return nullptr;
}

size_t lazy_io_queue::queued_entries() const {
    return 0;
}

size_t lazy_io_queue::in_flight() const {
    return 0;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
void lazy_io_queue::set_fifo_for_testing(bool fifo) {
    GGML_UNUSED(fifo);
}

const std::vector<lazy_io_dispatch_record> & lazy_io_queue::dispatch_log_for_testing() const {
    return s_->log;
}
#endif

bool lazy_io_deliver(const lazy_io_waiter & waiter, const uint8_t * read_bytes, int read_errno) noexcept {
    GGML_UNUSED(waiter);
    GGML_UNUSED(read_bytes);
    GGML_UNUSED(read_errno);
    return false;
}

}  // namespace ggml_sycl
