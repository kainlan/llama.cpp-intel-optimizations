#include "lazy-io-pool.hpp"

#include "ggml-impl.h"
#include "host-cpu-mapping.hpp"
#include "lazy-row-cache.hpp"
#include "witnessed-mutex.hpp"

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
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
//
// The pool also owns its block ring and the IO engine whose workers read into it (§3.3, §5.2), both installed once by
// block_ring_ensure under the leaf mutex. A WORKER reap takes a parked pool's ring out under the mutex and frees it
// after the unlock; a FULL reap destroys the pool, whose destructor stops and joins the workers before the ring goes.

namespace ggml_sycl {

struct lazy_io_pool;

// The IO engine (§5.2): the queue of §5.2.1 under its lock, and LAZY_BLOCK_RING_QD pread workers, worker i reading into
// slot i of the ring, span bytes at ring + i * span. Its lock (canonical contract §12.5, L5) is a leaf: nothing is
// taken under it, and it is never held across a read, a delivery, a lease drop or a log. A worker's wait releases it.
// It is a witnessed lock (§3.3, §11 T5), so its condition variables are condition_variable_any: a test build's
// witnessed_mutex records its holder and is not a std::mutex.
struct lazy_io_engine {
    witnessed_mutex             mutex{ GGML_SYCL_WITNESS_NAME("lazy_io_engine::mutex") };
    std::condition_variable_any work;  // a worker waits here for a queued read or the stop
    std::condition_variable_any idle;  // shutdown waits here for the picked reads to finish
    lazy_io_queue               queue;
    std::vector<std::thread>    workers;
    uint8_t *                   ring = nullptr;
    size_t                      span = 0;
    size_t                      busy = 0;  // reads a worker has picked and not finished, its lease dropped included
    bool                        stop = false;

    lazy_io_engine() = default;
    ~lazy_io_engine();
    lazy_io_engine(const lazy_io_engine &)             = delete;
    lazy_io_engine & operator=(const lazy_io_engine &) = delete;

    // Starts `n` workers over `ring_data` in slots of `slot_span` bytes. Throws what std::thread throws, after stopping
    // and joining the workers already started.
    void start(uint8_t * ring_data, size_t slot_span, size_t n);

    // Sets the stop flag, wakes every worker and joins the ones started; the destructor and a failed start share it.
    void stop_and_join() noexcept;

    // Worker `slot`'s loop: pick under the lock, read and deliver with none held, drop the read's lease, then reap.
    void run(size_t slot);
};

struct lazy_reap_node {
    lazy_io_pool *   pool = nullptr;  // a back-pointer while the pool lives; the owning pointer once it is parked
    lazy_reap_node * next = nullptr;
};

struct lazy_io_pool {
    std::atomic<uint64_t>           leases{ 0 };     // changed under the leaf mutex, and without it once shut down
    uint64_t                        generation = 0;  // the generation its take reserved, set before the install
    bool                            dying      = false;
    std::unique_ptr<lazy_reap_node> node;            // owned until the park releases it into the reap list

    // The block ring and its engine, installed together under the leaf mutex by ensure's step (3). The engine is
    // declared after the ring, so the pool's destruction joins the workers before it frees the ring.
    std::unique_ptr<lazy_io_host_memory> ring;
    std::atomic<uint64_t>                ring_bytes{ 0 };  // the ring's usable bytes, read through a lease lock-free
    size_t                               ring_span        = 0;  // one slot, need / LAZY_BLOCK_RING_QD
    uint64_t                             ring_load_txn    = 0;  // the transaction whose ensure created it; 0 for none
    bool                                 ensuring         = false;
    uint64_t                             ring_allocations = 0;  // rings ensure allocated for this pool, under the mutex
    std::unique_ptr<lazy_io_engine>      engine;
};

// The engine's paths into a lease's pool, the one friend of ring_lease they share.
struct lazy_io_pool_access {
    static lazy_io_pool * pool(const ring_lease & lease) noexcept { return lease.pool_; }

    // A lease on `pool`, whose count the caller incremented under the leaf mutex.
    static ring_lease adopt(lazy_io_pool * pool) noexcept { return ring_lease(pool); }
};

namespace {

witnessed_mutex   g_lazy_registry_mutex{ GGML_SYCL_WITNESS_NAME("g_lazy_registry_mutex") };
lazy_io_pool *    g_lazy_pool       = nullptr;  // the live pool, owned by the registry
uint64_t          g_lazy_generation = 0;        // the last installed pool's generation; a take reserves the next one
lazy_reap_node *  g_lazy_reap_list  = nullptr;  // the parked pools, each node owning its pool
std::atomic<bool> g_lazy_shut_down{ false };
uint64_t          g_lazy_parked_nodes           = 0;
uint64_t          g_lazy_full_reaps_run         = 0;
uint64_t          g_lazy_reaps_run_at_last_take = 0;
uint64_t          g_lazy_parked_at_last_take    = 0;

// §10's lazy_ring_coverage_reads: one relaxed increment per coverage read, in every build, no lock.
std::atomic<uint64_t> g_lazy_ring_coverage_reads{ 0 };
#if defined(GGML_SYCL_PRIVATE_TESTING)
// The one-shot hold hook and the point it is armed at: the first FULL reap, the first take that builds a pool, the
// first ensure that reserves `ensuring`, the first planner caller to reach the plan point, the first shutdown to reach
// its wait for the picked reads, the first coverage scope's take, or the first drop of a held lease.
enum class lazy_hold_point : uint8_t { REAP, TAKE, ENSURE, PLAN, SHUTDOWN, STAGE_TAKE, DROP };

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
    std::lock_guard lock(g_lazy_registry_mutex);
    g_lazy_hold_point = point;
    g_lazy_hold       = hold;
    g_lazy_hold_ctx   = hold != nullptr ? ctx : nullptr;
}

// Claims the hook if it is armed at `point` and calls it after the unlock, with no registry or IO-engine lock held.
void lazy_hold_run(lazy_hold_point point) {
    lazy_hold_fn hold     = nullptr;
    void *       hold_ctx = nullptr;
    {
        std::lock_guard lock(g_lazy_registry_mutex);
        hold = lazy_hold_claim_locked(point, &hold_ctx);
    }
    if (hold != nullptr) {
        hold(hold_ctx);
    }
}

// The engine's read source and the ring-write recorder (§11): atomics, read by the workers with no lock held.
std::atomic<lazy_io_read_fn> g_lazy_io_read_fn{ nullptr };
std::atomic<void *>          g_lazy_io_read_ctx{ nullptr };
std::atomic<uint64_t>        g_lazy_ring_write_end{ 0 };

// The one runtime mutant selector (§11): NONE unless an arm's registration selects one.
std::atomic<uint8_t> g_lazy_ring_mutant{ 0 };

// The witness's counts and the join census (§3.3, §11 T5): relaxed atomics read lock-free, since a reap that refuses
// under the registry's own mutex counts there. The one-shot throw at the FULL reap's first step is the test seam's.
std::atomic<uint64_t> g_lazy_reaps_refused{ 0 };
std::atomic<uint64_t> g_lazy_reaps_failed{ 0 };
std::atomic<uint64_t> g_lazy_lease_drops_locked{ 0 };
std::atomic<uint64_t> g_lazy_worker_reaps{ 0 };
std::atomic<uint64_t> g_lazy_joins{ 0 };
std::atomic<uint64_t> g_lazy_joins_from_worker{ 0 };
std::atomic<uint64_t> g_lazy_self_joins{ 0 };
thread_local bool     t_lazy_io_worker = false;  // set by a pool worker on entry, so a join knows who runs it
std::atomic<uint64_t> g_lazy_drops_after_shut_down{ 0 };
std::atomic<bool>     g_lazy_reap_throw_next{ false };

// The test seam's throw at the FULL reap's first step: what std::mutex::lock throws.
void lazy_reap_first_step_seam() {
    if (g_lazy_reap_throw_next.exchange(false, std::memory_order_relaxed)) {
        throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur),
                                "the test seam's throw at the reap's first step");
    }
}

// The reap's witness (§3.3, §11 T5): the reap takes the registry's leaf mutex and joins IO workers, so it runs with no
// witnessed lock held. One called with a lock held refuses before taking any: it names the lock in one WARN, counts the
// refusal and frees nothing, leaving its parked pools to the next reap.
bool lazy_reap_refused_under_held_lock() noexcept {
    const char * held = witnessed_mutex_held_any();
    if (held == nullptr) {
        return false;
    }
    g_lazy_reaps_refused.fetch_add(1, std::memory_order_relaxed);
    GGML_LOG_WARN("[HOST-PLAN-BUG] ring reap refused: %s held\n", held);
    return true;
}

// The drop's witness (§3.3, §11 T5): a final drop parks its pool under the registry's leaf mutex, so a drop runs with
// no witnessed lock held. One that runs with a lock held names it in one WARN, counts it and completes; under the
// registry's own mutex the drop would take that mutex again, so it aborts naming it.
void lazy_drop_witness() noexcept {
    const char * held = witnessed_mutex_held_any();
    if (held == nullptr) {
        return;
    }
    g_lazy_lease_drops_locked.fetch_add(1, std::memory_order_relaxed);
    GGML_LOG_WARN("[HOST-PLAN-BUG] ring lease dropped with %s held\n", held);
    if (witnessed_mutex_held(g_lazy_registry_mutex)) {
        GGML_ABORT("ring_lease::reset: the dropping thread holds the IO pool registry's leaf mutex");
    }
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
        std::lock_guard lock(g_lazy_registry_mutex);
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
    return pool_ != nullptr ? pool_->ring_bytes.load(std::memory_order_acquire) : 0;
}

void ring_lease::reset() noexcept {
    lazy_io_pool * pool = pool_;
    if (pool == nullptr) {
        return;
    }
    pool_ = nullptr;
    if (g_lazy_shut_down.load(std::memory_order_acquire)) {
#if defined(GGML_SYCL_PRIVATE_TESTING)
        g_lazy_drops_after_shut_down.fetch_add(1, std::memory_order_relaxed);
#endif
        pool->leases.fetch_sub(1, std::memory_order_relaxed);
        return;
    }
#if defined(GGML_SYCL_PRIVATE_TESTING)
    lazy_drop_witness();
    // The drop's hold point (§3.3's race): a fixture holds this drop before its decrement while a take joins the pool.
    lazy_hold_run(lazy_hold_point::DROP);
#endif
    std::lock_guard lock(g_lazy_registry_mutex);
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
            std::lock_guard lock(g_lazy_registry_mutex);
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
            std::lock_guard lock(g_lazy_registry_mutex);
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

namespace {

#if defined(GGML_SYCL_PRIVATE_TESTING)
// WORKER_REAP_DESTROYS_POOL: the WORKER pass runs the parked pools' destruction, whose join of their workers the census
// records, on the thread the pass runs on. It stops and joins each pool's engine and then leaks the pool, since freeing
// it would free the engine the calling worker returns into.
void lazy_reap_worker_destroys_pools() {
    lazy_reap_node * list = nullptr;
    {
        std::lock_guard lock(g_lazy_registry_mutex);
        list                = g_lazy_reap_list;
        g_lazy_reap_list    = nullptr;
        g_lazy_parked_nodes = 0;
    }
    while (list != nullptr) {
        lazy_reap_node * next = list->next;
        if (list->pool->engine != nullptr) {
            list->pool->engine->stop_and_join();
        }
        delete list;
        list = next;
    }
}
#endif

// A WORKER pass: one ring per pass, so nothing is allocated under the leaf mutex. The first parked pool still holding a
// ring gives it up under the lock, and the ring is freed after the unlock, its handle's drop crediting the budget
// ledger. The nodes stay on the list with their pools, whose workers only a FULL reap joins. Its locks can throw
// std::system_error, which the reap catches.
void lazy_reap_worker_pass() {
    for (;;) {
        std::unique_ptr<lazy_io_host_memory> ring;
        {
            std::lock_guard lock(g_lazy_registry_mutex);
            for (lazy_reap_node * node = g_lazy_reap_list; node != nullptr; node = node->next) {
                if (node->pool->ring != nullptr) {
                    ring = std::move(node->pool->ring);
                    node->pool->ring_bytes.store(0, std::memory_order_release);
                    break;
                }
            }
        }
        if (ring == nullptr) {
#if defined(GGML_SYCL_PRIVATE_TESTING)
            if (lazy_ring_mutant_is(lazy_ring_mutant::WORKER_REAP_DESTROYS_POOL)) {
                lazy_reap_worker_destroys_pools();
            }
            g_lazy_worker_reaps.fetch_add(1, std::memory_order_relaxed);
#endif
            return;
        }
        ring.reset();
    }
}

// A FULL pass: takes the parked list under the lock and frees every pool on it after the unlock, joining its workers.
// Its locks can throw std::system_error, which the reap catches; a throw before the list is taken leaves every pool
// parked for the next reap.
void lazy_reap_full_pass() {
#if defined(GGML_SYCL_PRIVATE_TESTING)
    lazy_reap_first_step_seam();
    lazy_hold_fn hold     = nullptr;
    void *       hold_ctx = nullptr;
#endif
    {
        std::lock_guard lock(g_lazy_registry_mutex);
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
        std::lock_guard lock(g_lazy_registry_mutex);
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

}  // namespace

void ggml_sycl_lazy_ring_reap(lazy_reap_scope scope) noexcept {
#if defined(GGML_SYCL_PRIVATE_TESTING)
    if (lazy_reap_refused_under_held_lock()) {
        return;
    }
#endif
    // A failure to reset is logged and never thrown (§3.3): std::mutex::lock can throw std::system_error, and the reap
    // runs from destructors and from noexcept teardown. Only that type is caught: anything else is a defect this
    // noexcept function ends the process on.
    try {
        if (scope == lazy_reap_scope::WORKER) {
            lazy_reap_worker_pass();
        } else {
            lazy_reap_full_pass();
        }
    } catch (const std::system_error & e) {
#if defined(GGML_SYCL_PRIVATE_TESTING)
        g_lazy_reaps_failed.fetch_add(1, std::memory_order_relaxed);
#endif
        GGML_LOG_WARN("[HOST-PLAN-BUG] ring reap failed: %s\n", e.what());
    }
}

void lazy_io_pool_mark_shut_down() noexcept {
    std::lock_guard lock(g_lazy_registry_mutex);
    g_lazy_shut_down.store(true, std::memory_order_release);
}

// LAZY_IO_POOL_MUTANT_GENERATION_LIVE_IGNORES_PARKED, and the runtime selector's GENERATION_LIVE_IGNORES_PARKED, read
// the slot alone, so a parked pool's generation reads gone before the FULL reap has freed its ring. Only
// test-sycl-lazy-io-pool-host's mutant target defines the macro; it passes only when the host test prints the parked
// pool's FAIL text.
bool lazy_io_pool_generation_live(uint64_t generation) noexcept {
    if (generation == 0) {
        return false;
    }
    std::lock_guard lock(g_lazy_registry_mutex);
    if (g_lazy_pool != nullptr && g_lazy_pool->generation == generation) {
        return true;
    }
#if defined(LAZY_IO_POOL_MUTANT_GENERATION_LIVE_IGNORES_PARKED)
    return false;
#else
#    if defined(GGML_SYCL_PRIVATE_TESTING)
    if (lazy_ring_mutant_is(lazy_ring_mutant::GENERATION_LIVE_IGNORES_PARKED)) {
        return false;
    }
#    endif
    for (const lazy_reap_node * node = g_lazy_reap_list; node != nullptr; node = node->next) {
        if (node->pool != nullptr && node->pool->generation == generation) {
            return true;
        }
    }
    return false;
#endif
}

// LAZY_IO_POOL_MUTANT_NO_ENTRY_REAP and LAZY_IO_POOL_MUTANT_NO_CLOSING_REAP each drop one of the wrapper's two reaps.
// Only test-sycl-lazy-io-pool-host's mutant targets define them; each passes only when the host test prints that
// reap's FAIL text.
// The runtime selector's NO_ENTRY_REAP and NO_CLOSING_REAP plant the same two defects for the device arms.
lazy_coverage_scope::reap_guard::reap_guard() noexcept {
#if !defined(LAZY_IO_POOL_MUTANT_NO_ENTRY_REAP)
#    if defined(GGML_SYCL_PRIVATE_TESTING)
    if (lazy_ring_mutant_is(lazy_ring_mutant::NO_ENTRY_REAP)) {
        return;
    }
#    endif
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
#endif
}

lazy_coverage_scope::reap_guard::~reap_guard() {
#if !defined(LAZY_IO_POOL_MUTANT_NO_CLOSING_REAP)
#    if defined(GGML_SYCL_PRIVATE_TESTING)
    if (lazy_ring_mutant_is(lazy_ring_mutant::NO_CLOSING_REAP)) {
        return;
    }
#    endif
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
#endif
}

bool lazy_coverage_scope::take_if(bool has_lazy_candidate) {
#if defined(GGML_SYCL_PRIVATE_TESTING)
    has_lazy_candidate = has_lazy_candidate || lazy_ring_mutant_is(lazy_ring_mutant::TAKE_WITHOUT_CANDIDATE);
#endif
    if (!has_lazy_candidate) {
        return false;
    }
#if defined(GGML_SYCL_PRIVATE_TESTING)
    // The stage-take hold point (§3.3's race): after the scope's entry reap and before its take, so a fixture can park
    // the live pool between the two or read what the entry reap left.
    lazy_hold_run(lazy_hold_point::STAGE_TAKE);
#endif
    lease_ = lazy_io_pool_take();
    return lease_.held();
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
bool ggml_sycl_lazy_registry_for_testing(lazy_registry_state * out) {
    if (out == nullptr) {
        return false;
    }
    std::lock_guard lock(g_lazy_registry_mutex);
    out->live_pools             = g_lazy_pool != nullptr ? 1 : 0;
    out->live_leases            = g_lazy_pool != nullptr ? g_lazy_pool->leases.load(std::memory_order_relaxed) : 0;
    out->generation             = g_lazy_generation;
    out->live_generation        = g_lazy_pool != nullptr ? g_lazy_pool->generation : 0;
    out->parked_nodes           = g_lazy_parked_nodes;
    out->full_reaps_run         = g_lazy_full_reaps_run;
    out->reaps_run_at_last_take = g_lazy_reaps_run_at_last_take;
    out->parked_at_last_take    = g_lazy_parked_at_last_take;
    out->ring_bytes             = g_lazy_pool != nullptr ? g_lazy_pool->ring_bytes.load(std::memory_order_relaxed) : 0;
    out->ring_allocations       = g_lazy_pool != nullptr ? g_lazy_pool->ring_allocations : 0;
    out->ensuring               = g_lazy_pool != nullptr && g_lazy_pool->ensuring;
    out->reaps_refused          = g_lazy_reaps_refused.load(std::memory_order_relaxed);
    out->reaps_failed           = g_lazy_reaps_failed.load(std::memory_order_relaxed);
    out->lease_drops_locked     = g_lazy_lease_drops_locked.load(std::memory_order_relaxed);
    out->worker_reaps           = g_lazy_worker_reaps.load(std::memory_order_relaxed);
    out->joins                  = g_lazy_joins.load(std::memory_order_relaxed);
    out->joins_from_worker      = g_lazy_joins_from_worker.load(std::memory_order_relaxed);
    out->self_joins             = g_lazy_self_joins.load(std::memory_order_relaxed);
    out->drops_after_shut_down  = g_lazy_drops_after_shut_down.load(std::memory_order_relaxed);
    return true;
}

bool registry_mutex_held() {
    return witnessed_mutex_held(g_lazy_registry_mutex);
}

const char * lazy_holding_status_name(lazy_holding_status status) {
    switch (status) {
        case lazy_holding_status::HELD:
            return "held";
        case lazy_holding_status::NOT_WIRED:
            return "not-wired";
        case lazy_holding_status::NO_ENGINE:
            return "no-engine";
        case lazy_holding_status::UNKNOWN:
            return "unknown";
    }
    return "unknown";
}

lazy_reap_holding_result ggml_sycl_test_lazy_reap_under_held_lock() {
    lazy_reap_holding_result result;
    result.held           = witnessed_mutex_held_any();
    result.witnessed      = result.held != nullptr;
    result.registry_held  = registry_mutex_held();
    const uint64_t before = g_lazy_reaps_refused.load(std::memory_order_relaxed);
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    result.refused = g_lazy_reaps_refused.load(std::memory_order_relaxed) != before;
    return result;
}

lazy_holding_status ggml_sycl_test_lazy_run_holding(int lock, lazy_hold_fn fn, void * ctx) {
    if (lock == LAZY_LOCK_STRAND) {
        return lazy_holding_status::NOT_WIRED;
    }
    if (lock == LAZY_LOCK_REGISTRY) {
        std::lock_guard registry_lock(g_lazy_registry_mutex);
        fn(ctx);
        return lazy_holding_status::HELD;
    }
    if (lock != LAZY_LOCK_QUEUE) {
        return lazy_holding_status::UNKNOWN;
    }
    // The join lease keeps the pool, and so its engine, alive across the call, and drops after the queue lock's scope.
    const ring_lease       lease  = lazy_io_pool_join();
    lazy_io_engine * const engine = lease.held() ? lazy_io_pool_access::pool(lease)->engine.get() : nullptr;
    if (engine == nullptr) {
        return lazy_holding_status::NO_ENGINE;
    }
    {
        std::lock_guard queue_lock(engine->mutex);
        fn(ctx);
    }
    return lazy_holding_status::HELD;
}

namespace {

void lazy_reap_under_held_lock_into(void * out) {
    *static_cast<lazy_reap_holding_result *>(out) = ggml_sycl_test_lazy_reap_under_held_lock();
}

}  // namespace

lazy_reap_holding_result ggml_sycl_test_lazy_reap_holding_pool(int lock) {
    lazy_reap_holding_result  result;
    const lazy_holding_status status = ggml_sycl_test_lazy_run_holding(lock, lazy_reap_under_held_lock_into, &result);
    result.status                    = status;
    return result;
}

void ggml_sycl_test_lazy_reap_throw_next() {
    g_lazy_reap_throw_next.store(true, std::memory_order_relaxed);
}

void ggml_sycl_test_lazy_reap_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::REAP, hold, ctx);
}

void ggml_sycl_test_lazy_take_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::TAKE, hold, ctx);
}

void ggml_sycl_test_lazy_ensure_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::ENSURE, hold, ctx);
}

void ggml_sycl_test_lazy_plan_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::PLAN, hold, ctx);
}

void ggml_sycl_test_lazy_shutdown_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::SHUTDOWN, hold, ctx);
}

void ggml_sycl_test_lazy_stage_take_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::STAGE_TAKE, hold, ctx);
}

void ggml_sycl_test_lazy_drop_hold(lazy_hold_fn hold, void * ctx) {
    lazy_hold_arm(lazy_hold_point::DROP, hold, ctx);
}

void ggml_sycl_test_lazy_plan_point() {
    lazy_hold_run(lazy_hold_point::PLAN);
}

void ggml_sycl_test_lazy_mark_dying(const ring_lease & lease) {
    if (lease.pool_ == nullptr) {
        return;
    }
    std::lock_guard lock(g_lazy_registry_mutex);
    lease.pool_->dying = true;
}

lazy_test_coverage_pin::lazy_test_coverage_pin(ring_lease lease) noexcept : lease_(std::move(lease)) {}

lazy_test_coverage_pin::~lazy_test_coverage_pin() {
    lease_.reset();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
}

lazy_test_coverage_pin ggml_sycl_test_make_coverage_pin(uint64_t ring_bytes) {
    ring_lease     lease = lazy_io_pool_take();
    const uint64_t live  = lease.ring_bytes();
    if (live != 0) {
        // The take joined a live ring (§3.3): this pin holds the ring the live guard's pool already has, so it asks for
        // that size and ensures nothing. An ensure would refuse it, since transaction 0 is never the ring's creator and
        // the creating plan's charge is then a mismatch.
        if (ring_bytes != live) {
            GGML_ABORT("ggml_sycl_test_make_coverage_pin: ring_bytes=%llu joins a live ring of %llu B",
                       (unsigned long long) ring_bytes, (unsigned long long) live);
        }
    } else if (ring_bytes != 0) {
        // The charge of a creating plan, the mapped need (§10): what a ring that ensure creates is charged.
        const size_t                  need = static_cast<size_t>(ring_bytes);
        const lazy_ring_ensure_status status =
            block_ring_ensure(need, need / LAZY_BLOCK_RING_QD, 0, host_cpu_mapped_size(need));
        if (status != lazy_ring_ensure_status::OK) {
            GGML_ABORT("ggml_sycl_test_make_coverage_pin: ring_bytes=%llu: block_ring_ensure refused: %s",
                       (unsigned long long) ring_bytes, lazy_ring_ensure_status_name(status));
        }
    }
    return lazy_test_coverage_pin(std::move(lease));
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

    // Indexes `e` in its context's class queue and the FIFO index, all or nothing: a throw leaves it in neither and
    // removes a context entry made for it, so the caller still owns an entry no index holds. The rollback is a guard's
    // destructor, not a handler, so it runs for every exception and the exception propagates as thrown.
    void insert_entry(entry * e) {
        const size_t cls = lazy_io_class_index(e->key);
        auto         ctx = contexts.try_emplace(e->context);

        struct undo_insert {
            state &                                      s;
            std::map<uint64_t, context_queues>::iterator pos;
            bool                                         made;
            size_t                                       cls;
            entry *                                      e;
            bool                                         armed = true;

            ~undo_insert() {
                if (armed) {
                    pos->second.by_class[cls].erase(e);
                    if (made && pos->second.by_class[cls].empty()) {
                        s.contexts.erase(pos);
                    }
                }
            }
        } undo{ *this, ctx.first, ctx.second, cls, e };
#if defined(LAZY_IO_QUEUE_MUTANT_PARTIAL_INSERT)
        undo.armed = false;
#endif
        ctx.first->second.by_class[cls].insert(e);
        by_seq[e->seq] = e;
        undo.armed     = false;
        class0 += cls == 0 ? 1 : 0;
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

    // The node owns the entry before any index holds it, and a failed index takes it back out through a guard, so a
    // throw never frees an entry an index still points at.
    void add_entry(node & n, uint64_t context, const lazy_io_key & key) {
        std::unique_ptr<entry> owned(new entry());
        entry *                e = owned.get();
        e->read                  = n.read.get();
        e->context               = context;
        e->key                   = key;
        e->seq                   = next_seq++;
#if defined(LAZY_IO_QUEUE_MUTANT_INDEX_BEFORE_OWN)
        insert_entry(e);
        n.entries.push_back(std::move(owned));
#else
        n.entries.push_back(std::move(owned));

        struct undo_own {
            std::vector<std::unique_ptr<entry>> & entries;
            bool                                  armed = true;

            ~undo_own() {
                if (armed) {
                    entries.pop_back();
                }
            }
        } undo{ n.entries };

        insert_entry(e);
        undo.armed = false;
#endif
    }

    // A re-key: the entry moves to its class queue under the new key, keeping its sequence number and its FIFO
    // index entry. The set node is extracted and reinserted, which allocates nothing, so a re-key cannot throw and
    // cannot leave an entry its node owns out of every index.
    void rekey(entry * e, const lazy_io_key & key) {
        auto ctx = contexts.find(e->context);
        GGML_ASSERT(ctx != contexts.end());
        const size_t from = lazy_io_class_index(e->key);
        const size_t to   = lazy_io_class_index(key);
        auto         nh   = ctx->second.by_class[from].extract(e);
        GGML_ASSERT(!nh.empty());
        e->key = key;
        ctx->second.by_class[to].insert(std::move(nh));
        class0 = class0 - (from == 0 ? 1 : 0) + (to == 0 ? 1 : 0);
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

#if defined(GGML_SYCL_PRIVATE_TESTING)
    // The indexes against their owners. It compares pointers and reads only entries a node owns, so a dangling pointer
    // in an index is reported, never followed.
    std::string check_indexes() const {
        std::set<const entry *> owned;
        size_t                  owned_class0 = 0;
        for (const auto & kv : in_flight) {
            const node & n = kv.second;
            if (!n.read->dispatched && n.entries.empty()) {
                return "the queued read at offset " + std::to_string(n.read->span.offset) + " has no entry";
            }
            for (const std::unique_ptr<entry> & e : n.entries) {
                owned.insert(e.get());
                owned_class0 += lazy_io_class_index(e->key) == 0 ? 1 : 0;
            }
        }
        size_t queued = 0;
        for (const auto & ctx : contexts) {
            if (ctx.second.by_class[0].empty() && ctx.second.by_class[1].empty()) {
                return "context " + std::to_string(ctx.first) + " is indexed with no entry";
            }
            for (size_t cls = 0; cls < 2; ++cls) {
                for (const entry * e : ctx.second.by_class[cls]) {
                    if (owned.count(e) == 0) {
                        return "a class queue holds an entry no read owns";
                    }
                    if (e->context != ctx.first || lazy_io_class_index(e->key) != cls) {
                        return "an entry is queued under another context or class";
                    }
                    ++queued;
                }
            }
        }
        for (const auto & kv : by_seq) {
            if (owned.count(kv.second) == 0) {
                return "the FIFO index holds an entry no read owns";
            }
            if (kv.second->seq != kv.first) {
                return "the FIFO index files entry " + std::to_string(kv.second->seq) + " under " +
                       std::to_string(kv.first);
            }
        }
        if (queued != owned.size() || by_seq.size() != owned.size()) {
            return "the reads own " + std::to_string(owned.size()) + " entries, the class queues hold " +
                   std::to_string(queued) + " and the FIFO index " + std::to_string(by_seq.size());
        }
        if (class0 != owned_class0) {
            return "the class-0 count is " + std::to_string(class0) + ", the reads own " +
                   std::to_string(owned_class0) + " class-0 entries";
        }
        return std::string();
    }
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
#if !defined(LAZY_IO_QUEUE_MUTANT_CACHED_WAITER_UNCHECKED)
    // A waiter with a row cache inserts one whole row from one read, so it must lie in one chunk; a share holding part
    // of a row would insert that part as the row. A row's read covers only the blocks that hold it (§5.2), so a row
    // smaller than a chunk always does.
    GGML_ASSERT(waiter.cache == nullptr ||
                (waiter.bytes != 0 && waiter.src_offset / LAZY_IO_CHUNK_BYTES ==
                                          (waiter.src_offset + waiter.bytes - 1) / LAZY_IO_CHUNK_BYTES));
#endif
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
#if !defined(LAZY_IO_QUEUE_MUTANT_CACHED_WAITER_UNCHECKED)
                // Only a share that is the whole waiter inserts the row; the assert above makes every cached share one.
                if (share.bytes != waiter.bytes) {
                    share.cache = nullptr;
                    share.row   = -1;
                }
#endif
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
            auto it          = s_->in_flight.emplace(chunk, std::move(fresh));
            n                = &it->second;

            // A read with no entry would never be picked, so one whose entry could not be added is taken back out by
            // the guard's destructor as the throw leaves.
            struct undo_read {
                state &      s;
                decltype(it) pos;
                bool         armed = true;

                ~undo_read() {
                    if (armed) {
                        s.in_flight.erase(pos);
                    }
                }
            } undo{ *s_, it };

            s_->add_entry(*n, context, key);
            undo.armed = false;
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

std::vector<std::unique_ptr<lazy_io_read>> lazy_io_queue::cancel_queued() {
    state &                                    s = *s_;
    std::vector<std::unique_ptr<lazy_io_read>> out;
    for (auto it = s.in_flight.begin(); it != s.in_flight.end();) {
        if (it->second.read->dispatched) {
            ++it;
            continue;
        }
        s.drop_entries(it->second);
        out.push_back(std::move(it->second.read));
        it = s.in_flight.erase(it);
    }
    return out;
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

std::string lazy_io_queue::check_indexes_for_testing() const {
    return s_->check_indexes();
}
#endif

bool lazy_io_deliver(const lazy_io_waiter & waiter, const uint8_t * read_bytes, int read_errno) noexcept {
    lazy_io_ticket * const ticket = waiter.ticket;
    if (read_errno == 0) {
        if (waiter.landing != nullptr && waiter.bytes != 0) {
            lazy_landing_slot & slot       = *waiter.landing;
            bool                outside    = false;  // live, and the range does not fit the slot
            size_t              slot_bytes = 0;
            {
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
                if (live) {
                    if (waiter.landing_offset <= slot.bytes && waiter.bytes <= slot.bytes - waiter.landing_offset) {
                        std::memcpy(slot.data + waiter.landing_offset, read_bytes + waiter.src_offset, waiter.bytes);
                    } else {
                        outside    = true;
                        slot_bytes = slot.bytes;
                    }
                }
            }
#if defined(LAZY_IO_QUEUE_MUTANT_LANDING_OOB_SILENT)
            GGML_UNUSED(outside);
            GGML_UNUSED(slot_bytes);
#else
            // A landing past its live slot is a defect in the request, never a short copy: the ticket fails with
            // EINVAL, the first errno winning as for a failed read, and the WARN names the range after the unlock.
            if (outside) {
                if (ticket != nullptr) {
                    int none = 0;
                    ticket->read_error.compare_exchange_strong(none, EINVAL, std::memory_order_release,
                                                               std::memory_order_relaxed);
                }
                GGML_LOG_WARN(
                    "[LAZY-IO] landing outside its slot: %zu B at offset %zu of a %zu B slot (epoch %u), "
                    "the request fails with EINVAL\n",
                    waiter.bytes, waiter.landing_offset, slot_bytes, (unsigned) waiter.epoch);
            }
#endif
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

// ---- the block ring and the IO engine (§3.3, §5.2) ----

namespace {

// A [HOST-PLAN-BUG] of the ring (§10): one WARN naming both numbers. Ensure returns its typed refusal after it, and the
// strict abort is its caller's (§3.3). Called with no lock held.
void lazy_ring_plan_bug(const char * what, size_t got, size_t want) {
    GGML_LOG_WARN("[HOST-PLAN-BUG] %s %zu B, want %zu B\n", what, got, want);
}

// One read of `length` bytes of `fd` at `offset` into `dst` (§5.2): pread until done, retried on EINTR. A short read
// at the end of the file zero-fills the rest of the span, since no row reaches past the file's end. Returns 0 or the
// errno.
int lazy_io_read_span(int fd, uint8_t * dst, size_t length, uint64_t offset) {
#if defined(GGML_SYCL_PRIVATE_TESTING)
    const lazy_io_read_fn source = g_lazy_io_read_fn.load(std::memory_order_acquire);
    if (source != nullptr) {
        const long got = source(g_lazy_io_read_ctx.load(std::memory_order_acquire), fd, dst, length, offset);
        if (got < 0) {
            return static_cast<int>(-got);
        }
        if (static_cast<size_t>(got) < length) {
            std::memset(dst + got, 0, length - static_cast<size_t>(got));
        }
        return 0;
    }
#endif
    size_t done = 0;
    while (done < length) {
        const ssize_t got = ::pread(fd, dst + done, length - done, static_cast<off_t>(offset + done));
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return errno;
        }
        if (got == 0) {
            std::memset(dst + done, 0, length - done);
            break;
        }
        done += static_cast<size_t>(got);
    }
    return 0;
}

}  // namespace

lazy_io_engine::~lazy_io_engine() {
    stop_and_join();
}

void lazy_io_engine::stop_and_join() noexcept {
    {
        std::lock_guard lock(mutex);
        stop = true;
    }
    work.notify_all();
    for (std::thread & worker : workers) {
        if (!worker.joinable()) {
            continue;
        }
#if defined(GGML_SYCL_PRIVATE_TESTING)
        // The join census (§3.3: a pool worker never joins itself). A worker's join of itself throws inside this
        // noexcept function, so a test build records it and detaches instead, and the arm reads the count. A release
        // build records nothing: only a FULL reap destroys a pool, and no worker runs one.
        if (worker.get_id() == std::this_thread::get_id()) {
            g_lazy_self_joins.fetch_add(1, std::memory_order_relaxed);
            worker.detach();
            continue;
        }
        g_lazy_joins.fetch_add(1, std::memory_order_relaxed);
        if (t_lazy_io_worker) {
            g_lazy_joins_from_worker.fetch_add(1, std::memory_order_relaxed);
        }
#endif
        worker.join();
    }
    workers.clear();
}

void lazy_io_engine::start(uint8_t * ring_data, size_t slot_span, size_t n) {
    ring = ring_data;
    span = slot_span;
    workers.reserve(n);

    // A thread whose creation throws leaves the ones already started running over a ring the caller is about to free,
    // so the guard stops and joins them on that exit, and the exception reaches the caller unchanged.
    struct start_guard {
        lazy_io_engine * engine;

        ~start_guard() {
            if (engine != nullptr) {
                engine->stop_and_join();
            }
        }
    } guard{ this };

    for (size_t slot = 0; slot < n; ++slot) {
        workers.emplace_back(&lazy_io_engine::run, this, slot);
    }
    guard.engine = nullptr;
}

void lazy_io_engine::run(size_t slot) {
#if defined(GGML_SYCL_PRIVATE_TESTING)
    t_lazy_io_worker = true;
#endif
    uint8_t * const dst = ring + slot * span;
    for (;;) {
        lazy_io_read * read = nullptr;
        {
            std::unique_lock lock(mutex);
            for (;;) {
                if (stop) {
                    return;
                }
                read = queue.pick();
                if (read != nullptr) {
                    break;
                }
                work.wait(lock);
            }
            ++busy;
        }
        const size_t length = static_cast<size_t>(read->span.length);
        const int    err    = lazy_io_read_span(read->fd, dst, length, read->span.offset);
#if defined(GGML_SYCL_PRIVATE_TESTING)
        const uint64_t end  = static_cast<uint64_t>(slot * span + length);
        uint64_t       seen = g_lazy_ring_write_end.load(std::memory_order_relaxed);
        while (seen < end && !g_lazy_ring_write_end.compare_exchange_weak(seen, end, std::memory_order_relaxed)) {
        }
#endif
        std::unique_ptr<lazy_io_read> done;
        {
            std::lock_guard lock(mutex);
            done = queue.complete(read);
        }
        if (err != 0) {
            const std::string why = std::generic_category().message(err);
            GGML_LOG_WARN("[LAZY-IO] read failed: errno %d (%s), file %016llx, block offset %llu, %zu B\n", err,
                          why.c_str(), (unsigned long long) read->span.file, (unsigned long long) read->span.offset,
                          length);
        }
        if (done != nullptr) {
            for (const lazy_io_waiter & waiter : done->waiters) {
                lazy_io_deliver(waiter, dst, err);
            }
        }
        // The read's lease goes last; when it was the pool's last, the pool parks and this worker frees only its ring.
        done.reset();
        ggml_sycl_lazy_ring_reap(lazy_reap_scope::WORKER);
        {
            std::lock_guard lock(mutex);
            --busy;
        }
        idle.notify_all();
    }
}

ring_lease lazy_io_pool_join() {
    std::lock_guard lock(g_lazy_registry_mutex);
    if (g_lazy_shut_down.load(std::memory_order_relaxed) || g_lazy_pool == nullptr ||
        g_lazy_pool->ring_bytes.load(std::memory_order_relaxed) == 0) {
        return ring_lease();
    }
    g_lazy_pool->leases.fetch_add(1, std::memory_order_relaxed);
    return lazy_io_pool_access::adopt(g_lazy_pool);
}

const char * lazy_ring_ensure_status_name(lazy_ring_ensure_status status) {
    switch (status) {
        case lazy_ring_ensure_status::OK:
            return "ok";
        case lazy_ring_ensure_status::UNPINNED:
            return "lazy-table-ring-unpinned";
        case lazy_ring_ensure_status::TOO_NARROW:
            return "lazy-table-ring-too-narrow";
        case lazy_ring_ensure_status::CONCURRENT:
            return "lazy-table-ring-concurrent-ensure";
        case lazy_ring_ensure_status::SHUT_DOWN:
            return "lazy-table-ring-shut-down";
        case lazy_ring_ensure_status::MISMATCH:
            return "lazy-table-ring-mismatch";
        case lazy_ring_ensure_status::REFUSED:
            return "lazy-table-ring-refused";
    }
    return "lazy-table-ring-unknown";
}

lazy_ring_ensure_status block_ring_ensure(size_t   need,
                                          size_t   span,
                                          int      device,
                                          size_t   plan_charge,
                                          uint64_t load_txn_id) {
    // O_DIRECT alignment is a plan fact (§5.2): each of the LAZY_BLOCK_RING_QD slots of need / LAZY_BLOCK_RING_QD bytes
    // must be whole 4 KiB blocks, since a worker reads a block-aligned span into its slot. Any other need is a plan
    // bug, refused before any lock or state and never rounded up; the line names the next need that splits.
    constexpr size_t ring_unit = LAZY_BLOCK_RING_QD * LAZY_IO_BLOCK_BYTES;
#if defined(GGML_SYCL_PRIVATE_TESTING)
    const bool alignment_check = !lazy_ring_mutant_is(lazy_ring_mutant::NO_ALIGNMENT_CHECK);
#else
    const bool alignment_check = true;
#endif
    if (alignment_check && need % ring_unit != 0) {
        const size_t pad = ring_unit - need % ring_unit;
        lazy_ring_plan_bug("block ring need not whole 4 KiB blocks per slot: need", need,
                           need <= SIZE_MAX - pad ? need + pad : need - need % ring_unit);
        return lazy_ring_ensure_status::MISMATCH;
    }

    // Step (1), under the leaf mutex: refuse, accept the live ring, or reserve `ensuring` and take a counted reference
    // that keeps the reserved pool alive through the unlocked allocation. Every log waits for the unlock.
    enum class step1 : uint8_t { RESERVED, SHUT_DOWN, UNPINNED, CONCURRENT, LIVE };

    step1          outcome      = step1::RESERVED;
    uint64_t       live         = 0;
    uint64_t       creator      = 0;
    size_t         ring_charged = 0;
    lazy_io_pool * pool         = nullptr;
    ring_lease     hold;
#if defined(GGML_SYCL_PRIVATE_TESTING)
    lazy_hold_fn hook        = nullptr;
    void *       hook_ctx    = nullptr;
    const bool   no_ensuring = lazy_ring_mutant_is(lazy_ring_mutant::NO_ENSURING);
#else
    const bool no_ensuring = false;
#endif
    {
        std::lock_guard lock(g_lazy_registry_mutex);
        if (g_lazy_shut_down.load(std::memory_order_relaxed)) {
            outcome = step1::SHUT_DOWN;
        } else if (g_lazy_pool == nullptr) {
            outcome = step1::UNPINNED;
        } else if (g_lazy_pool->ring_bytes.load(std::memory_order_relaxed) != 0) {
            outcome      = step1::LIVE;
            live         = g_lazy_pool->ring_bytes.load(std::memory_order_relaxed);
            creator      = g_lazy_pool->ring_load_txn;
            ring_charged = g_lazy_pool->ring->charged_bytes();
        } else if (g_lazy_pool->ensuring && !no_ensuring) {
            outcome = step1::CONCURRENT;
        } else {
            pool           = g_lazy_pool;
            pool->ensuring = true;
            pool->leases.fetch_add(1, std::memory_order_relaxed);
            hold = lazy_io_pool_access::adopt(pool);
#if defined(GGML_SYCL_PRIVATE_TESTING)
            hook = lazy_hold_claim_locked(lazy_hold_point::ENSURE, &hook_ctx);
#endif
        }
    }
    switch (outcome) {
        case step1::SHUT_DOWN:
            GGML_LOG_WARN("[LAZY-RING] block ring ensure refused: the IO pool is shut down (need %zu B)\n", need);
            return lazy_ring_ensure_status::SHUT_DOWN;
        case step1::UNPINNED:
            GGML_LOG_WARN("[LAZY-RING] block ring ensure refused: no live IO pool pins a ring (need %zu B)\n", need);
            return lazy_ring_ensure_status::UNPINNED;
        case step1::CONCURRENT:
            GGML_LOG_WARN(
                "[HOST-PLAN-BUG] concurrent ring ensure: need %zu B, span %zu B, another caller is ensuring "
                "the ring\n",
                need, span);
            return lazy_ring_ensure_status::CONCURRENT;
        case step1::LIVE:
            // A later call (§3.3, §10): the live ring must hold the need and the span. The model whose ensure created
            // the ring carries its charge, the ring row's charged bytes, in every table's plan; any other model's plan
            // charged no ring. A transaction of 0 names no model, so it is never the creator.
            if (need > live) {
                lazy_ring_plan_bug("block ring too narrow for the plan's need: need", need, static_cast<size_t>(live));
                return lazy_ring_ensure_status::TOO_NARROW;
            }
            if (span > live / LAZY_BLOCK_RING_QD) {
                lazy_ring_plan_bug("block ring slot too narrow for the table's span: span", span,
                                   static_cast<size_t>(live / LAZY_BLOCK_RING_QD));
                return lazy_ring_ensure_status::TOO_NARROW;
            }
            if (load_txn_id != 0 && load_txn_id == creator) {
                if (plan_charge != ring_charged) {
                    lazy_ring_plan_bug("block ring charge differs from the ledger's: ring charge", plan_charge,
                                       ring_charged);
                    return lazy_ring_ensure_status::MISMATCH;
                }
            } else if (plan_charge != 0) {
                lazy_ring_plan_bug("block ring charged by a plan the live ring covers: ring charge", plan_charge, 0);
                return lazy_ring_ensure_status::MISMATCH;
            }
            return lazy_ring_ensure_status::OK;
        case step1::RESERVED:
            break;
    }

    // The guard clears `ensuring` on every exit from here on, an exception included.
    struct ensuring_guard {
        lazy_io_pool * pool;
        bool           armed = true;

        ~ensuring_guard() {
#if defined(GGML_SYCL_PRIVATE_TESTING)
            if (std::uncaught_exceptions() > 0 && lazy_ring_mutant_is(lazy_ring_mutant::ENSURING_KEPT_ON_THROW)) {
                return;
            }
#endif
            if (armed) {
                std::lock_guard lock(g_lazy_registry_mutex);
                pool->ensuring = false;
            }
        }
    } guard{ pool };

#if defined(GGML_SYCL_PRIVATE_TESTING)
    const bool span_check = !lazy_ring_mutant_is(lazy_ring_mutant::NO_SPAN_CHECK);
#else
    const bool span_check = true;
#endif
    if (span_check && span > need / LAZY_BLOCK_RING_QD) {
        lazy_ring_plan_bug("block ring need too short for the table's span: need", need, LAZY_BLOCK_RING_QD * span);
        return lazy_ring_ensure_status::TOO_NARROW;
    }
#if defined(GGML_SYCL_PRIVATE_TESTING)
    if (hook != nullptr) {
        hook(hook_ctx);
    }
#endif

    // Step (2), no lock held: allocate exactly `need`, owned by the reserved pool's generation, check it against the
    // plan, and start the workers over it. The generation was set before the pool's install and is never written
    // again, and the counted reference keeps the pool alive, so it is read here without the lock.
    std::unique_ptr<lazy_io_host_memory> ring = lazy_io_allocate_host(need, device, pool->generation, load_txn_id);
    if (ring == nullptr) {
        GGML_LOG_WARN("[LAZY-RING] block ring allocation refused: need %zu B on device %d\n", need, device);
        return lazy_ring_ensure_status::REFUSED;
    }
    if (ring->size() != need) {
        lazy_ring_plan_bug("block ring allocated size differs from the plan's need: allocated", ring->size(), need);
        return lazy_ring_ensure_status::MISMATCH;
    }
    if (plan_charge != ring->charged_bytes()) {
        lazy_ring_plan_bug("block ring charge differs from the ledger's: ring charge", plan_charge,
                           ring->charged_bytes());
        return lazy_ring_ensure_status::MISMATCH;
    }
    std::unique_ptr<lazy_io_engine> engine(new lazy_io_engine());
    engine->start(ring->data(), need / LAZY_BLOCK_RING_QD, LAZY_BLOCK_RING_QD);

    // Step (3), under the leaf mutex: install the ring and its engine unless the registry shut down meanwhile, in
    // which case both are freed after the unlock.
#if defined(GGML_SYCL_PRIVATE_TESTING)
    const bool install_after_shut_down = lazy_ring_mutant_is(lazy_ring_mutant::INSTALL_AFTER_SHUT_DOWN);
#else
    const bool install_after_shut_down = false;
#endif
    bool shut_down = false;
    {
        std::lock_guard lock(g_lazy_registry_mutex);
        ++pool->ring_allocations;
        shut_down = g_lazy_shut_down.load(std::memory_order_relaxed) && !install_after_shut_down;
        if (!shut_down && pool->ring == nullptr) {
            pool->ring          = std::move(ring);
            pool->engine        = std::move(engine);
            pool->ring_span     = need / LAZY_BLOCK_RING_QD;
            pool->ring_load_txn = load_txn_id;
            pool->ring_bytes.store(need, std::memory_order_release);
        }
        pool->ensuring = false;
        guard.armed    = false;
    }
    if (shut_down) {
        GGML_LOG_WARN(
            "[LAZY-RING] block ring ensure refused: the IO pool shut down during the allocation (need %zu "
            "B)\n",
            need);
        return lazy_ring_ensure_status::SHUT_DOWN;
    }
    GGML_LOG_INFO("[LAZY-RING] block ring: %zu B on device %d, %zu slots of %zu B, charged %zu B\n", need, device,
                  LAZY_BLOCK_RING_QD, need / LAZY_BLOCK_RING_QD, plan_charge);
    return lazy_ring_ensure_status::OK;
}

size_t lazy_io_submit(uint64_t               context,
                      int                    fd,
                      const lazy_io_span &   span,
                      const lazy_io_key &    key,
                      const lazy_io_waiter & waiter) {
    ring_lease lease = lazy_io_pool_join();
    if (!lease.held()) {
        GGML_LOG_WARN("[LAZY-IO] read refused: no live block ring (file %016llx, block offset %llu)\n",
                      (unsigned long long) span.file, (unsigned long long) span.offset);
        return 0;
    }
    lazy_io_pool * const pool = lazy_io_pool_access::pool(lease);
    if (span.length == 0 || span.length > pool->ring_span) {
        GGML_LOG_WARN("[LAZY-IO] read refused: span %llu B, block ring slot %zu B\n", (unsigned long long) span.length,
                      pool->ring_span);
        return 0;
    }
    // One lease per read the span may queue; the reads take theirs and the rest drop after the unlock. The join
    // lease is held until after the notify, so the engine outlives this call whatever the workers do meanwhile.
    const size_t            reads = static_cast<size_t>((span.length + LAZY_IO_CHUNK_BYTES - 1) / LAZY_IO_CHUNK_BYTES);
    std::vector<ring_lease> spare;
    spare.reserve(reads);
    for (size_t i = 0; i < reads; ++i) {
        spare.push_back(lease.clone());
    }
    lazy_io_engine * const engine = pool->engine.get();
    size_t                 added  = 0;
    size_t                 fresh  = 0;  // the reads this call queued, each holding one of the spare leases
    {
        std::lock_guard lock(engine->mutex);
        added = engine->queue.submit(context, fd, span, key, waiter, spare);
        fresh = reads - spare.size();
    }
    // One wake per new read, and none for a join, which queues nothing a worker could pick. A wake lost to a busy
    // worker loses no read: a worker picks before it waits, so it takes what was queued while it worked.
    for (size_t i = 0; i < fresh; ++i) {
        engine->work.notify_one();
    }
    return added;
}

void lazy_io_pool_shutdown() noexcept {
    ring_lease lease = lazy_io_pool_join();
    if (!lease.held()) {
        return;
    }
    lazy_io_engine * const                     engine = lazy_io_pool_access::pool(lease)->engine.get();
    std::vector<std::unique_ptr<lazy_io_read>> cancelled;
    {
        std::lock_guard lock(engine->mutex);
        cancelled = engine->queue.cancel_queued();
    }
    for (std::unique_ptr<lazy_io_read> & read : cancelled) {
        for (const lazy_io_waiter & waiter : read->waiters) {
            lazy_io_deliver(waiter, nullptr, ECANCELED);
        }
        read.reset();
    }
    if (!cancelled.empty()) {
        GGML_LOG_WARN("[LAZY-IO] shutdown cancelled %zu queued reads\n", cancelled.size());
    }
#if defined(GGML_SYCL_PRIVATE_TESTING)
    // The wait's two test points: the mutant returns without waiting, and the hold hook runs just before the wait with
    // no lock held. The hook is claimed under the registry's lock alone: the engine's is never held while it is taken.
    if (lazy_ring_mutant_is(lazy_ring_mutant::SHUTDOWN_NO_WAIT)) {
        return;
    }
    lazy_hold_fn hold     = nullptr;
    void *       hold_ctx = nullptr;
    {
        std::lock_guard registry_lock(g_lazy_registry_mutex);
        hold = lazy_hold_claim_locked(lazy_hold_point::SHUTDOWN, &hold_ctx);
    }
    if (hold != nullptr) {
        hold(hold_ctx);
    }
#endif
    std::unique_lock lock(engine->mutex);
    engine->idle.wait(lock, [engine] { return engine->busy == 0; });
}

uint64_t lazy_ring_coverage_read(const ring_lease * pin) noexcept {
    if (pin == nullptr) {
#if defined(GGML_SYCL_PRIVATE_TESTING)
        if (lazy_ring_mutant_is(lazy_ring_mutant::COUNT_WITHOUT_PIN)) {
            g_lazy_ring_coverage_reads.fetch_add(1, std::memory_order_relaxed);
        }
#endif
        return 0;
    }
    g_lazy_ring_coverage_reads.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_SYCL_PRIVATE_TESTING)
    if (lazy_ring_mutant_is(lazy_ring_mutant::IGNORE_LIVE_RING)) {
        return 0;
    }
    if (lazy_ring_mutant_is(lazy_ring_mutant::LEASE_PRESENCE_AS_COVERAGE) && pin->held()) {
        return UINT64_MAX;
    }
#endif
    return pin->ring_bytes();
}

uint64_t lazy_ring_coverage_reads() noexcept {
    return g_lazy_ring_coverage_reads.load(std::memory_order_relaxed);
}

bool lazy_block_ring_bytes(uint64_t * out) {
    if (out == nullptr) {
        return false;
    }
    std::lock_guard lock(g_lazy_registry_mutex);
    if (g_lazy_pool == nullptr || g_lazy_pool->ring_bytes.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    *out = g_lazy_pool->ring_bytes.load(std::memory_order_relaxed);
    return true;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
void ggml_sycl_test_lazy_io_read_source(lazy_io_read_fn read, void * ctx) {
    if (read == nullptr) {
        g_lazy_io_read_fn.store(nullptr, std::memory_order_release);
        g_lazy_io_read_ctx.store(nullptr, std::memory_order_release);
        return;
    }
    g_lazy_io_read_ctx.store(ctx, std::memory_order_release);
    g_lazy_io_read_fn.store(read, std::memory_order_release);
}

uint64_t ggml_sycl_test_lazy_ring_write_end() {
    return g_lazy_ring_write_end.load(std::memory_order_relaxed);
}

void ggml_sycl_test_lazy_ring_write_end_reset() {
    g_lazy_ring_write_end.store(0, std::memory_order_relaxed);
}

void ggml_sycl_test_lazy_ring_mutant_select(lazy_ring_mutant mutant) {
    g_lazy_ring_mutant.store(static_cast<uint8_t>(mutant), std::memory_order_relaxed);
}

bool lazy_ring_mutant_is(lazy_ring_mutant mutant) noexcept {
    return g_lazy_ring_mutant.load(std::memory_order_relaxed) == static_cast<uint8_t>(mutant);
}
#endif

}  // namespace ggml_sycl
