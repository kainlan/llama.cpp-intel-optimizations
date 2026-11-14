// The block ring's ensure and the IO engine (docs/design/sycl-storage-tier-lazy-tables.md §3.3, §5.2, §5.3, §10):
// the first ensure allocates exactly the plan's need and later calls allocate nothing, every mismatch with the plan
// refuses naming both numbers, one flight at a time (a concurrent ensure refuses at once with its WARN line), a
// throwing allocation leaves no ensuring state behind, shut-down refuses and frees the reserver's ring, a second
// coverage pin while the first is live joins its ring and ensures nothing (another size aborts), a worker's read
// lands in its waiter's slot, a failed read reaches the ticket with one WARN, a read's last drop on a worker frees only
// the ring and leaves the join of the pool's workers to the next FULL reap, shutdown cancels the queued reads and
// waits for the picked ones, a ticket's wait returns only once every read submitted on it is delivered, with the
// first errno a delivery stored, and a ubatch gather lands every row byte for byte from the row cache or the file and
// returns only after the reads it submitted are delivered, on its error paths too. Ensure never aborts on a refusal:
// under strict leases its one production caller does, and test-sycl-lazy-ring scores that abort. Entered with a
// witnessed lock held, it aborts at its entry naming the lock.
//
// The lock witness (§3.3, §11 T5): a witnessed lock's depth table follows lock, try_lock, unlock, a unique_lock's
// relock and a condition_variable_any wait; the reap refuses, naming it, when it is called with the registry's leaf
// mutex or the queue lock held, and with none held it runs; a held lease dropped under a witnessed lock is logged, and
// under the registry's own mutex aborts; and a reap whose step throws logs it and returns. Under strict leases (§10)
// each of those [HOST-PLAN-BUG] lines, and the one a reap prints for a parked pool a take joined, aborts after its
// WARN, scored in children whose fixture strict switch is on.
//
// Host-only, because none of this needs a device: ensure allocates through lazy_io_allocate_host, a link seam the
// fixture header defines with a counting ledger, and the workers read through the PRIVATE_TESTING read source. Links
// ggml-sycl/lazy-io-pool.cpp and the row cache its reaper inserts into (lazy-row-cache.cpp, host-cpu-mapping.cpp), with
// ggml-base for the log and the abort.
//
// The mutants are the runtime selector's (§11), one --mutant registration each, which passes only when the arm that
// kills it prints its FAIL line. Every hold is released on an observed event, never on a timer, within a 5 s bound past
// which the arm is VOID, which fails the binary like a FAIL. The exceptions release their reads after a 200 ms window
// in which the call must not return, since a wait asleep on its condition variable cannot be observed without a hook
// in the wait itself: the ticket-wait arm once its wait's thread has started with its three reads held; the gated
// gather arm once every read each gather submits is held, which puts that gather past its last submit; and the
// throwing-submit arm once the gather has returned or one of its reads is held, the earliest point a throw that leaves
// a read queued is seen.
//
// The throwing-submit arm replaces the global operator new for the whole binary: disarmed, it is malloc and free; on a
// thread that armed its countdown, the countdown's last allocation throws std::bad_alloc.

#include "../ggml/src/ggml-sycl/lazy-gather.hpp"
#include "../ggml/src/ggml-sycl/lazy-io-pool.hpp"
#include "../ggml/src/ggml-sycl/lazy-row-cache.hpp"
#include "../ggml/src/ggml-sycl/tests/lazy-io-host-fixture.hpp"
#include "../ggml/src/ggml-sycl/tests/lazy-io-test-util.hpp"
#include "../ggml/src/ggml-sycl/tests/lazy-table-test-util.hpp"
#include "../ggml/src/ggml-sycl/witnessed-mutex.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

// The throwing-submit arm's fault injection: while this thread's countdown is armed, its n-th allocation throws
// std::bad_alloc. The replacement covers the whole binary, lazy-io-pool.cpp included; disarmed, it is malloc and free.
namespace {
thread_local long g_alloc_countdown = 0;  // 0: disarmed
}  // namespace

void * operator new(std::size_t bytes) {
    if (g_alloc_countdown > 0 && --g_alloc_countdown == 0) {
        throw std::bad_alloc();
    }
    void * p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

void operator delete(void * p) noexcept {
    std::free(p);
}

void operator delete(void * p, std::size_t) noexcept {
    std::free(p);
}

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using host_ledger_test_harness::arm_loop::named_arm;
using host_ledger_test_harness::arm_loop::run_arms;
using lazy_io_test::file_byte;
using lazy_io_test::fill_source;
using lazy_io_test::gate;
using lazy_io_test::gated_source;
using lazy_io_test::wait_for;
using lazy_table_test::registry;

namespace {

constexpr size_t   k_span  = 2 * LAZY_IO_BLOCK_BYTES;      // one row of at most 4 KiB: two blocks
constexpr size_t   k_need  = LAZY_BLOCK_RING_QD * k_span;  // the plan's need for that span
constexpr uint64_t k_txn   = 42;
constexpr uint64_t k_file  = 9;
constexpr auto     k_bound = std::chrono::seconds(5);

// The mismatched-pin child's switch: a pin of k_need, then one of twice that while the first is live.
const char * k_pin_child = "--mismatched-pin-child";

// The witness arms' children (§11 T5): nine witnessed locks held at once; a FULL reap under the lock named by the next
// argument; a held lease dropped under the registry's own mutex; and a FULL reap whose first step throws.
const char * k_table_child   = "--witness-table-child";
const char * k_holding_child = "--reap-holding-child";
const char * k_drop_child    = "--drop-holding-child";
const char * k_throw_child   = "--reap-throw-child";

// The strict arms' child (§10), its site the next argument and its mutant, if any, the one after.
const char * k_strict_child = "--strict-child";

// The ensure entry witness's child (§3.3, §11 T5): an ensure with a witnessed lock held when the next argument is
// "held", with none when it is "none"; its mutant, if any, the one after.
const char * k_ensure_child = "--ensure-holding-child";

// The --mutant this process runs under, passed on to the strict children; null in a plain run.
const char * g_mutant = nullptr;

std::string name(lazy_ring_ensure_status s) {
    return lazy_ring_ensure_status_name(s);
}

uint64_t allocations() {
    return lazy_io_host_fixture::g_state.allocations.load();
}

uint64_t live_charged() {
    return lazy_io_host_fixture::g_state.live_charged.load();
}

size_t mapped(size_t bytes) {
    return host_cpu_mapped_size(bytes);
}

// Waits for the reads' leases to drop, since a worker drops its read's lease after the delivery the arm observed, then
// drops `lease` and runs a FULL reap, so the next arm finds no pool. True when no pool is left, live or parked.
bool settle(ring_lease & lease) {
    const uint64_t own   = lease.held() ? 1 : 0;
    const bool     quiet = wait_for([own] { return registry().live_leases <= own; });
    lease.reset();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    const lazy_registry_state s = registry();
    return quiet && s.live_pools == 0 && s.parked_nodes == 0;
}

void gate_hold(void * ctx) {
    static_cast<gate *>(ctx)->hold();
}

void gate_release(void * ctx) {
    static_cast<gate *>(ctx)->release();
}

// Releases its gate on every path out of the scope that declares it, so no arm leaves a read held.
struct gate_release_guard {
    gate & g;

    ~gate_release_guard() { g.release(); }
};

// A gate for one gated call, never destroyed: a held read still takes the gate's mutex on its way out of hold() after
// release() returns, so the gate must outlive every holder, and an arm that makes one gate per attempt cannot keep
// them all static.
gate & lasting_gate() {
    return *new gate;
}

// Whether `g` holds at least `n` reads now, read under its mutex.
bool holding(gate & g, size_t n) {
    std::lock_guard<std::mutex> lock(g.mutex);
    return g.holders >= n;
}

// Fails the read at block `*ctx` with EIO and serves every other block.
long eio_source(void * ctx, int fd, uint8_t * dst, size_t length, uint64_t offset) {
    if (offset == *static_cast<const uint64_t *>(ctx)) {
        return -EIO;
    }
    return fill_source(nullptr, fd, dst, length, offset);
}

lazy_io_span block_span(uint64_t block) {
    return { k_file, block * LAZY_IO_BLOCK_BYTES, k_span };
}

lazy_io_key key0(uint32_t epoch, int32_t layer) {
    return { lazy_io_class::PREFETCH, epoch, layer, lazy_io_kind::TABLE_ROWS };
}

// Runs `fn` on its own detached thread and returns its result's future. `fn` must own everything it touches (capture a
// shared_ptr by value): a call that never returns outlives the arm.
std::future<int> run_detached(std::function<int()> fn) {
    auto             task = std::make_shared<std::packaged_task<int()>>(std::move(fn));
    std::future<int> done = task->get_future();
    std::thread([task] { (*task)(); }).detach();
    return done;
}

// Runs `fn` detached and waits at most `seconds`: a call that does not return fails its arm instead of hanging the
// binary, so a mutant registration still sees its own message.
bool returns_within(std::function<int()> fn, int seconds, int & rc) {
    std::future<int> done = run_detached(std::move(fn));
    if (done.wait_for(std::chrono::seconds(seconds)) != std::future_status::ready) {
        return false;
    }
    rc = done.get();
    return true;
}

// Polls `ticket` to 0 within the hook bound, then takes its wait, which returns at once: the wait's lock orders the
// ticket's later free after the last delivery's notify and unlock, which a 0 read alone does not. False at the bound,
// when a worker may still deliver to the ticket.
bool delivered(lazy_io_ticket & ticket) {
    if (!wait_for([&ticket] { return ticket.outstanding.load() == 0; })) {
        return false;
    }
    (void) lazy_io_ticket_wait(ticket);
    return true;
}

// One STORAGE request's ticket and landing slot at `epoch`, for one waiter of `bytes` from `src_offset` in its read.
struct request_fixture {
    lazy_gather_request  req;
    lazy_io_ticket       ticket;
    lazy_landing_slot    slot;
    std::vector<uint8_t> slab;

    explicit request_fixture(uint32_t epoch) : slab(LAZY_IO_BLOCK_BYTES, 0) {
        req.epoch      = epoch;
        req.ticket     = &ticket;
        ticket.request = &req;
        slot.epoch     = epoch;
        slot.data      = slab.data();
        slot.bytes     = slab.size();
    }

    lazy_io_waiter waiter(uint64_t src_offset, size_t bytes) {
        lazy_io_waiter w;
        w.ticket     = &ticket;
        w.epoch      = req.epoch;
        w.landing    = &slot;
        w.src_offset = src_offset;
        w.bytes      = bytes;
        return w;
    }
};

// The first ensure allocates exactly the plan's need, under the load's transaction, and only when the table's span fits
// need / QD. Later calls allocate nothing: the creating model's next table carries the ring row's charge and another
// model's covered plan carries 0, and a need or a span the live ring cannot hold, or a charge other than that rule's,
// refuses with one [HOST-PLAN-BUG] line naming both numbers (§3.3, §10's ring check). A transaction of 0 names no
// model, so it is never the creator.
// Mutant: no-span-check, under which the need one block per slot short allocates a ring.
arm_result arm_ensure_rules() {
    arm_result r;
    log_clear();
    ring_lease     lease      = lazy_io_pool_take();
    const uint64_t a0         = allocations();
    const size_t   short_need = k_need - LAZY_BLOCK_RING_QD * LAZY_IO_BLOCK_BYTES;

    lazy_ring_ensure_status st = block_ring_ensure(short_need, k_span, 0, mapped(short_need), k_txn);
    expect(r, st == lazy_ring_ensure_status::TOO_NARROW && allocations() == a0,
           "the ensure one block per slot short returned " + name(st) + " with " + num(allocations() - a0) +
               " allocation(s), want lazy-table-ring-too-narrow and none");
    const std::string short_numbers = "need " + num(short_need) + " B, want " + num(k_need) + " B";
    expect(r, log_count_with("[HOST-PLAN-BUG]", short_numbers.c_str()) == 1,
           "the ensure one block per slot short printed " +
               num(log_count_with("[HOST-PLAN-BUG]", short_numbers.c_str())) + " [HOST-PLAN-BUG] lines naming '" +
               short_numbers + "', want 1");

    st               = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    uint64_t   bytes = 0;
    const bool live  = lazy_block_ring_bytes(&bytes);
    expect(r, st == lazy_ring_ensure_status::OK && allocations() == a0 + 1 && live && bytes == k_need,
           "the full ensure returned " + name(st) + " with " + num(allocations() - a0) + " allocation(s) and ring " +
               (live ? num(bytes) : std::string("none")) + ", want ok, 1 and " + num(k_need));
    expect(r, live_charged() == mapped(k_need) && lazy_io_host_fixture::g_state.last_load_txn.load() == k_txn,
           "the ring is charged " + num(live_charged()) + " B under txn " +
               num(lazy_io_host_fixture::g_state.last_load_txn.load()) + ", want " + num(mapped(k_need)) + " under " +
               num(k_txn));

    st = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    expect(r, st == lazy_ring_ensure_status::OK && allocations() == a0 + 1,
           "the creating model's next table, charged the ring row's bytes, returned " + name(st) + " after " +
               num(allocations() - a0) + " allocations, want ok and still 1");
    st = block_ring_ensure(k_need / 2, k_span, 0, 0, k_txn + 1);
    expect(r, st == lazy_ring_ensure_status::OK && allocations() == a0 + 1,
           "another model's covered ensure with fewer bytes and ring= 0 returned " + name(st) +
               ", want ok with no allocation");

    struct mismatch_case {
        const char * what;
        size_t       charge;
        uint64_t     txn;
        size_t       want;
    };

    const mismatch_case mismatches[] = {
        { "the creating model's table charged no ring",              0,                  k_txn,     mapped(k_need) },
        { "another model's covered plan charged a ring",             mapped(k_need / 2), k_txn + 1, 0              },
        { "a plan with no transaction charged the ring row's bytes", mapped(k_need),     0,         0              },
    };
    for (const mismatch_case & m : mismatches) {
        log_clear();
        st                          = block_ring_ensure(k_need / 2, k_span, 0, m.charge, m.txn);
        const std::string numbers   = "ring charge " + num(m.charge) + " B, want " + num(m.want) + " B";
        const size_t      bug_lines = log_count_with("[HOST-PLAN-BUG]", numbers.c_str());
        expect(r, st == lazy_ring_ensure_status::MISMATCH && bug_lines == 1,
               std::string(m.what) + ": returned " + name(st) + " with " + num(bug_lines) + " lines naming '" +
                   numbers + "', want lazy-table-ring-mismatch and 1");
    }
    log_clear();
    st                             = block_ring_ensure(2 * k_need, k_span, 0, 0, k_txn);
    const std::string wide_numbers = "need " + num(2 * k_need) + " B, want " + num(k_need) + " B";
    expect(r, st == lazy_ring_ensure_status::TOO_NARROW && log_count_with("[HOST-PLAN-BUG]", wide_numbers.c_str()) == 1,
           "an ensure with a larger need returned " + name(st) + ", want lazy-table-ring-too-narrow naming '" +
               wide_numbers + "'");
    st                             = block_ring_ensure(k_need, 2 * k_span, 0, 0, k_txn);
    const std::string span_numbers = "span " + num(2 * k_span) + " B, want " + num(k_span) + " B";
    expect(r, st == lazy_ring_ensure_status::TOO_NARROW && log_count_with("[HOST-PLAN-BUG]", span_numbers.c_str()) == 1,
           "an ensure with a wider span returned " + name(st) + ", want lazy-table-ring-too-narrow naming '" +
               span_numbers + "'");
    expect(r, allocations() == a0 + 1, "the refusals allocated " + num(allocations() - a0 - 1) + " ring(s), want 0");

    expect(r, settle(lease) && live_charged() == 0,
           "after the drop and a FULL reap " + num(live_charged()) + " ring bytes are charged, want 0");
    return r;
}

// A creating ensure whose allocation is refused, or falls short of the need, or whose plan charged other than the
// ledger does, refuses with its typed reason, frees what it allocated and leaves no ensuring state, so the next ensure
// creates the ring (§3.3, §10).
arm_result arm_ensure_creation_refusals() {
    arm_result r;
    log_clear();
    ring_lease     lease = lazy_io_pool_take();
    const uint64_t a0    = allocations();

    lazy_io_host_fixture::g_state.refuse_next = true;
    lazy_ring_ensure_status st                = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    expect(r, st == lazy_ring_ensure_status::REFUSED && !registry().ensuring,
           "a refused allocation returned " + name(st) + ", want lazy-table-ring-refused with ensuring clear");

    lazy_io_host_fixture::g_state.short_next = LAZY_IO_BLOCK_BYTES;
    st                                       = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    const std::string short_numbers =
        "allocated " + num(k_need - LAZY_IO_BLOCK_BYTES) + " B, want " + num(k_need) + " B";
    expect(
        r, st == lazy_ring_ensure_status::MISMATCH && log_count_with("[HOST-PLAN-BUG]", short_numbers.c_str()) == 1,
        "a short allocation returned " + name(st) + ", want lazy-table-ring-mismatch naming '" + short_numbers + "'");

    st                               = block_ring_ensure(k_need, k_span, 0, 0, k_txn);
    const std::string charge_numbers = "ring charge 0 B, want " + num(mapped(k_need)) + " B";
    expect(r, st == lazy_ring_ensure_status::MISMATCH && log_count_with("[HOST-PLAN-BUG]", charge_numbers.c_str()) == 1,
           "a creating plan that charged no ring returned " + name(st) + ", want lazy-table-ring-mismatch naming '" +
               charge_numbers + "'");
    expect(r, live_charged() == 0 && registry().ring_bytes == 0 && !registry().ensuring,
           "after the refusals " + num(live_charged()) + " B are charged and the ring holds " +
               num(registry().ring_bytes) + " B, want both 0 with ensuring clear");

    st = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    expect(r, st == lazy_ring_ensure_status::OK && registry().ring_bytes == k_need && allocations() == a0 + 3,
           "the ensure after the refusals returned " + name(st) + " with ring " + num(registry().ring_bytes) +
               ", want ok and " + num(k_need));
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// O_DIRECT alignment is a plan fact (§5.2): a need that is not whole 4 KiB blocks per slot is a plan bug, refused as
// lazy-table-ring-mismatch with one [HOST-PLAN-BUG] line naming it and the next need that splits, before ensure
// reserves or allocates anything, and never rounded up into a ring.
// Mutant: no-alignment-check, under which the unaligned need allocates a ring.
arm_result arm_ensure_unaligned_need() {
    arm_result r;
    log_clear();
    ring_lease     lease = lazy_io_pool_take();
    const uint64_t a0    = allocations();
    const size_t   need  = k_need + LAZY_IO_BLOCK_BYTES;
    const size_t   want  = k_need + LAZY_BLOCK_RING_QD * LAZY_IO_BLOCK_BYTES;

    const lazy_ring_ensure_status st = block_ring_ensure(need, k_span, 0, mapped(need), k_txn);
    expect(r, st == lazy_ring_ensure_status::MISMATCH && allocations() == a0,
           "the unaligned need returned " + name(st) + " with " + num(allocations() - a0) +
               " allocation(s), want lazy-table-ring-mismatch and none");
    const std::string numbers = "need " + num(need) + " B, want " + num(want) + " B";
    expect(r, log_count_with("[HOST-PLAN-BUG]", numbers.c_str()) == 1,
           "the unaligned need printed " + num(log_count_with("[HOST-PLAN-BUG]", numbers.c_str())) +
               " [HOST-PLAN-BUG] lines naming '" + numbers + "', want 1");
    expect(r, registry().ring_bytes == 0 && !registry().ensuring,
           "after the refusal the ring holds " + num(registry().ring_bytes) + " B with ensuring " +
               (registry().ensuring ? "set" : "clear") + ", want 0 and clear");
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// With no live pool, no lease pins a ring: ensure refuses as lazy-table-ring-unpinned and allocates nothing (§3.3).
arm_result arm_ensure_unpinned() {
    arm_result     r;
    const uint64_t a0 = allocations();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    const lazy_ring_ensure_status st = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    expect(r, st == lazy_ring_ensure_status::UNPINNED && allocations() == a0,
           "an ensure with no live pool returned " + name(st) + " after " + num(allocations() - a0) +
               " allocation(s), want lazy-table-ring-unpinned and none");
    return r;
}

// One flight at a time (§3.3): while the hook holds the reserver between its reserve and its allocation, a second
// ensure returns lazy-table-ring-concurrent-ensure at once with one "[HOST-PLAN-BUG] concurrent ring ensure" line, and
// the reserver then installs its ring with one allocation. Engagement: the reserver is held when the second ensure
// runs. Mutant: no-ensuring, under which the second caller allocates and installs a ring of its own.
arm_result arm_ensure_concurrent() {
    arm_result r;
    log_clear();
    ring_lease     lease = lazy_io_pool_take();
    const uint64_t a0    = allocations();
    gate           g;
    ggml_sycl_test_lazy_ensure_hold(gate_hold, &g);
    std::atomic<int> first{ -1 };
    std::thread      reserver([&] { first = (int) block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn); });
    if (!g.wait_holders(1)) {
        void_arm(r, "the hook did not hold the reserver within 5 s");
        ggml_sycl_test_lazy_ensure_hold(nullptr, nullptr);
        g.release();
        reserver.join();
        settle(lease);
        return r;
    }
    const lazy_ring_ensure_status second = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    expect(r, second == lazy_ring_ensure_status::CONCURRENT,
           "the second ensure returned " + name(second) + ", want lazy-table-ring-concurrent-ensure");
    expect(r, log_count_with("[HOST-PLAN-BUG] concurrent ring ensure") == 1,
           "the second ensure printed " + num(log_count_with("[HOST-PLAN-BUG] concurrent ring ensure")) +
               " '[HOST-PLAN-BUG] concurrent ring ensure' lines, want 1");
    g.release();
    reserver.join();
    expect(r, first == (int) lazy_ring_ensure_status::OK && allocations() == a0 + 1 && registry().ring_allocations == 1,
           "the reserver returned " + name((lazy_ring_ensure_status) first.load()) + " after " +
               num(allocations() - a0) + " allocation(s), the pool counting " + num(registry().ring_allocations) +
               ", want ok, 1 and 1");
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// A throwing allocation leaves ensuring clear (§3.3's scope guard), so the next ensure allocates.
// Mutant: ensuring-kept-on-throw, under which the next ensure finds ensuring set and refuses.
arm_result arm_ensure_throw() {
    arm_result     r;
    ring_lease     lease                     = lazy_io_pool_take();
    const uint64_t a0                        = allocations();
    lazy_io_host_fixture::g_state.throw_next = true;
    bool threw                               = false;
    try {
        (void) block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    expect(r, threw, "the throwing allocation did not reach the caller as std::bad_alloc");
    expect(r, !registry().ensuring, "after the throw the pool's ensuring is still set, want it cleared by the guard");
    const lazy_ring_ensure_status st = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    expect(r, st == lazy_ring_ensure_status::OK && allocations() == a0 + 1,
           "the ensure after the throw returned " + name(st) + " after " + num(allocations() - a0) +
               " allocation(s), want ok and 1");
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// A worker reads its span into its slot of the ring and the reaper copies the waiter's bytes into the landing slot
// (§5.2, §5.3); the ring-write recorder's largest end stays within the ring.
arm_result arm_engine_read_lands() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    ggml_sycl_test_lazy_ring_write_end_reset();
    ggml_sycl_test_lazy_io_read_source(fill_source, nullptr);
    request_fixture      f(3);
    const uint64_t       block = 3;
    const size_t         added = lazy_io_submit(1, -1, block_span(block), key0(3, 5), f.waiter(100, 1000));
    const bool           done  = added == 1 && delivered(f.ticket);
    std::vector<uint8_t> want(1000);
    for (size_t i = 0; i < want.size(); ++i) {
        want[i] = file_byte(block * LAZY_IO_BLOCK_BYTES + 100 + i);
    }
    expect(r, done, "the read of block 3 queued " + num(added) + " waiter(s) and did not complete within 5 s");
    expect(r, done && std::memcmp(f.slab.data(), want.data(), want.size()) == 0 && f.ticket.read_error.load() == 0,
           "the landing slot does not hold bytes 100-1099 of block 3's read");
    const uint64_t end = ggml_sycl_test_lazy_ring_write_end();
    expect(r, end >= k_span && end <= k_need,
           "the largest ring write ends at " + num(end) + ", want between " + num(k_span) + " and " + num(k_need));
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// A failed read copies nothing, stores its errno in the ticket and still delivers, with one WARN naming the errno, the
// file identity and the block offset (§5.3).
arm_result arm_engine_read_error() {
    arm_result r;
    log_clear();
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    uint64_t failing = 6 * LAZY_IO_BLOCK_BYTES;
    ggml_sycl_test_lazy_io_read_source(eio_source, &failing);
    request_fixture f(4);
    const size_t    added = lazy_io_submit(1, -1, block_span(6), key0(4, 2), f.waiter(0, 512));
    const bool      done  = added == 1 && delivered(f.ticket);
    expect(r, done, "the failing read queued " + num(added) + " waiter(s) and did not complete within 5 s");
    expect(r, done && f.ticket.read_error.load() == EIO && f.slab[0] == 0,
           "the failing read left read_error " + num(static_cast<uint64_t>(f.ticket.read_error.load())) +
               " and a landing byte " + num(f.slab[0]) + ", want " + num(EIO) + " and 0");
    const std::string where = "block offset " + num(failing);
    expect(r,
           log_count_with("[LAZY-IO] read failed: errno 5", where.c_str()) == 1 &&
               log_count_with("[LAZY-IO] read failed", "file 0000000000000009") == 1,
           "the failing read printed " + num(log_count_with("[LAZY-IO] read failed")) +
               " WARN lines, want 1 naming errno 5, file 0000000000000009 and " + where);
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// A read holds a lease from its submission until its delivery (§3.3), so when the arm drops its own lease the read's is
// the last: its drop on the worker parks the pool and the worker's WORKER reap frees the ring and leaves the node,
// with the pool and its workers, for the next FULL reap. Engagement: the read is held on the worker when the arm drops.
arm_result arm_engine_worker_reap() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    gate g;
    ggml_sycl_test_lazy_io_read_source(gated_source, &g);
    request_fixture f(5);
    const size_t    added = lazy_io_submit(1, -1, block_span(8), key0(5, 1), f.waiter(0, 64));
    if (added != 1 || !g.wait_holders(1)) {
        void_arm(r, "the read was not held on a worker within 5 s (" + num(added) + " waiter(s) queued)");
        g.release();
        (void) delivered(f.ticket);
        ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
        settle(lease);
        return r;
    }
    lease.reset();
    const lazy_registry_state held = registry();
    expect(r, held.live_pools == 1 && held.live_leases == 1 && held.parked_nodes == 0,
           "with the read held after the arm's drop: " + num(held.live_pools) + " live pools, " +
               num(held.live_leases) + " leases, " + num(held.parked_nodes) + " parked, want 1, 1 and 0");
    g.release();
    const bool freed = wait_for([&] { return live_charged() == 0 && registry().parked_nodes == 1; });
    expect(r, freed,
           "after the read's delivery " + num(live_charged()) + " ring bytes are charged and " +
               num(registry().parked_nodes) +
               " pool(s) parked, want 0 and 1: the worker's drop frees only "
               "the ring");
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    expect(r, registry().parked_nodes == 0, "the FULL reap left " + num(registry().parked_nodes) + " parked, want 0");
    return r;
}

// Shutdown's step (§3.3): with every worker holding a read, lazy_io_pool_shutdown delivers the queued read ECANCELED
// and returns only after the picked reads have finished. The picked reads are held on a latch that only shutdown's hold
// hook releases, and the hook runs once the queued read is cancelled, just before the wait, so a shutdown that returns
// without waiting returns with the latch still held. Shutdown's return reads the latch and the deliveries; the arm has
// no timer. Engagement: every worker holds a read when shutdown starts.
arm_result arm_engine_shutdown() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    gate g;
    ggml_sycl_test_lazy_io_read_source(gated_source, &g);
    std::vector<std::unique_ptr<request_fixture>> picked;
    for (size_t i = 0; i < LAZY_BLOCK_RING_QD; ++i) {
        picked.emplace_back(new request_fixture(6));
        (void) lazy_io_submit(1, -1, block_span(2 * i), key0(6, 1), picked.back()->waiter(0, 64));
    }
    if (!g.wait_holders(LAZY_BLOCK_RING_QD)) {
        void_arm(r, "not every worker held a read within the hook bound");
        g.release();
    }
    request_fixture queued(6);
    (void) lazy_io_submit(1, -1, block_span(2 * LAZY_BLOCK_RING_QD), key0(6, 2), queued.waiter(0, 64));
    ggml_sycl_test_lazy_shutdown_hold(gate_release, &g);
    std::atomic<bool> returned{ false };
    bool              released_at_return  = false;
    size_t            delivered_at_return = 0;
    std::thread       stopper([&] {
        lazy_io_pool_shutdown();
        released_at_return = g.is_released();
        for (const auto & f : picked) {
            delivered_at_return += f->ticket.outstanding.load() == 0 && f->ticket.read_error.load() == 0;
        }
        returned = true;
    });
    const bool        bounded = wait_for([&] { return returned.load(); });
    // Frees the picked reads that a shutdown returning without waiting, or one that never reached its hook, left held.
    g.release();
    stopper.join();
    for (const auto & f : picked) {
        (void) delivered(f->ticket);
    }
    ggml_sycl_test_lazy_shutdown_hold(nullptr, nullptr);
    expect(r, bounded, "shutdown had not returned within the hook bound, want its hook to free the picked reads");
    const bool cancelled = queued.ticket.outstanding.load() == 0;
    expect(r, cancelled && queued.ticket.read_error.load() == ECANCELED,
           "the queued read was delivered " + std::string(cancelled ? "with read_error " : "not at all, read_error ") +
               num(static_cast<uint64_t>(queued.ticket.read_error.load())) + ", want " + num(ECANCELED));
    expect(r, released_at_return,
           "shutdown returned while the picked reads were still held on their latch, want it to return only after "
           "they finish");
    expect(r, delivered_at_return == LAZY_BLOCK_RING_QD,
           num(delivered_at_return) + " of the " + num(LAZY_BLOCK_RING_QD) +
               " picked reads were delivered when shutdown returned, want all");
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// A second coverage pin while the first is live joins its pool (§3.3): it asks for the live ring's size, so it ensures
// nothing, allocates nothing and prints no [HOST-PLAN-BUG] line, and the ring outlives the second pin's drop. A second
// pin that asks for another size aborts naming both sizes, in a child, since the abort ends the process.
arm_result arm_coverage_pin_joins() {
    arm_result r;
    log_clear();
    const uint64_t a0 = allocations();
    {
        const lazy_test_coverage_pin first = ggml_sycl_test_make_coverage_pin(k_need);
        {
            const lazy_test_coverage_pin second = ggml_sycl_test_make_coverage_pin(k_need);
            const lazy_registry_state    s      = registry();
            expect(r, allocations() == a0 + 1 && s.live_pools == 1 && s.live_leases == 2 && s.ring_bytes == k_need,
                   "two pins of " + num(k_need) + " B made " + num(allocations() - a0) + " allocation(s), " +
                       num(s.live_pools) + " live pool(s) and " + num(s.live_leases) + " lease(s) on a ring of " +
                       num(s.ring_bytes) + " B, want 1, 1, 2 and " + num(k_need) + " B");
        }
        expect(r, registry().ring_bytes == k_need,
               "the second pin's drop left a ring of " + num(registry().ring_bytes) + " B, want " + num(k_need));
    }
    expect(r, log_count_with("[HOST-PLAN-BUG]") == 0,
           "the pins printed " + num(log_count_with("[HOST-PLAN-BUG]")) + " [HOST-PLAN-BUG] lines, want 0");
    const lazy_registry_state s = registry();
    expect(r, s.live_pools == 0 && s.parked_nodes == 0 && live_charged() == 0,
           "after both pins: " + num(s.live_pools) + " live pool(s), " + num(s.parked_nodes) + " parked and " +
               num(live_charged()) + " B charged, want 0, 0 and 0");

    const child_run child = run_child_strings({ "test-sycl-lazy-io-engine-host", k_pin_child },
                                              child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
    if (!child.ran) {
        fail(r, "the mismatched-pin child did not finish within " + num(k_bound.count()) + " s");
        return r;
    }
    const std::string want = "ggml_sycl_test_make_coverage_pin: ring_bytes=" + num(2 * k_need) +
                             " joins a live ring of " + num(k_need) + " B";
    const bool aborted = WIFSIGNALED(child.status) && WTERMSIG(child.status) == SIGABRT;
    expect(r, aborted && text_count_with(child.err, want.c_str()) == 1,
           "the mismatched second pin ended with status " + std::to_string(child.status) + " and " +
               num(text_count_with(child.err, want.c_str())) + " lines '" + want + "', want SIGABRT and 1");
    expect(r, text_count_with(child.err, "PIN-CHILD returned") == 0, "the mismatched second pin returned");
    return r;
}

// Shut-down (§3.3): a reserver that reaches its install after the registry shut down refuses as
// lazy-table-ring-shut-down and frees its ring after the unlock, and an ensure that begins after it refuses the same
// way. Engagement: the reserver is held when the registry shuts down.
// Mutant: install-after-shut-down, under which the reserver installs its ring and returns ok.
arm_result arm_ensure_shut_down() {
    arm_result     r;
    ring_lease     lease = lazy_io_pool_take();
    const uint64_t c0    = live_charged();
    gate           g;
    ggml_sycl_test_lazy_ensure_hold(gate_hold, &g);
    std::atomic<int> first{ -1 };
    std::thread      reserver([&] { first = (int) block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn); });
    if (!g.wait_holders(1)) {
        void_arm(r, "the hook did not hold the reserver within 5 s");
    }
    lazy_io_pool_mark_shut_down();
    g.release();
    reserver.join();
    expect(r, first == (int) lazy_ring_ensure_status::SHUT_DOWN,
           "the reserver that reached its install after shut-down returned " +
               name((lazy_ring_ensure_status) first.load()) + ", want lazy-table-ring-shut-down");
    expect(r, live_charged() == c0 && registry().ring_bytes == 0,
           "after shut-down " + num(live_charged() - c0) + " ring bytes stay charged and the ring holds " +
               num(registry().ring_bytes) + " B, want both 0: the reserver frees its ring");
    const lazy_ring_ensure_status st = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    expect(r, st == lazy_ring_ensure_status::SHUT_DOWN,
           "an ensure after shut-down returned " + name(st) + ", want lazy-table-ring-shut-down");
    lease.reset();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    return r;
}

// ---- the lock witness (§3.3, §11 T5) ----

// The name the witness gives the first witnessed lock the calling thread holds, or "(none)".
std::string held_name() {
    const char * held = witnessed_mutex_held_any();
    return held != nullptr ? held : "(none)";
}

std::string depth_of(const witnessed_mutex & m) {
    return std::to_string(witnessed_mutex_depth(m));
}

// Polls `done` until it holds or the 5 s bound of §11 T5 passes: an observation, never a sleep standing in for an
// event.
template <typename F> bool within_bound(F done) {
    const auto deadline = std::chrono::steady_clock::now() + k_bound;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// The first line of `text` that contains `needle`, or "(none)".
std::string first_line_with(const std::string & text, const std::string & needle) {
    const size_t at = text.find(needle);
    if (at == std::string::npos) {
        return "(none)";
    }
    const size_t start = text.rfind('\n', at);
    const size_t end   = text.find('\n', at);
    const size_t from  = start == std::string::npos ? 0 : start + 1;
    return text.substr(from, end == std::string::npos ? std::string::npos : end - from);
}

std::string status_text(int status) {
    if (WIFSIGNALED(status)) {
        return "signal " + std::to_string(WTERMSIG(status));
    }
    return "exit " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
}

// The depth table (§11 T5): a witnessed lock's depth is 1 on the thread that holds it, through lock, try_lock and
// unlock, and 0 on every other thread, and each instance is its own entry, so the witness names the one still held.
arm_result arm_witness_depth() {
    arm_result      r;
    witnessed_mutex a{ GGML_SYCL_WITNESS_NAME("witness-arm-a") };
    witnessed_mutex b{ GGML_SYCL_WITNESS_NAME("witness-arm-b") };
    expect(r, held_name() == "(none)", "before any lock the witness names " + held_name() + ", want (none)");
    a.lock();
    int a_elsewhere = -1;
    std::thread([&] { a_elsewhere = witnessed_mutex_depth(a); }).join();
    expect(r, witnessed_mutex_depth(a) == 1 && held_name() == "witness-arm-a" && a_elsewhere == 0,
           "after a.lock() the witness reads depth " + depth_of(a) + " naming " + held_name() + ", and depth " +
               std::to_string(a_elsewhere) + " on another thread, want 1 naming witness-arm-a, and 0");
    b.lock();
    a.unlock();
    expect(r, witnessed_mutex_depth(a) == 0 && witnessed_mutex_depth(b) == 1 && held_name() == "witness-arm-b",
           "after b.lock() and a.unlock() the witness reads depths " + depth_of(a) + " and " + depth_of(b) +
               " naming " + held_name() + ", want 0 and 1 naming witness-arm-b");
    b.unlock();
    expect(
        r, witnessed_mutex_depth(b) == 0 && held_name() == "(none)",
        "after b.unlock() the witness reads depth " + depth_of(b) + " naming " + held_name() + ", want 0 and (none)");
    const bool took           = a.try_lock();
    bool       took_elsewhere = true;
    std::thread([&] {
        took_elsewhere = a.try_lock();
        if (took_elsewhere) {
            a.unlock();
        }
    }).join();
    const int depth_taken = witnessed_mutex_depth(a);
    if (took) {
        a.unlock();
    }
    expect(r, took && !took_elsewhere && depth_taken == 1 && witnessed_mutex_depth(a) == 0,
           "a try_lock that took the lock read depth " + std::to_string(depth_taken) + " and " + depth_of(a) +
               " after its unlock, want 1 and 0");
    return r;
}

// A unique_lock that unlocks and relocks, and a condition_variable_any wait, which unlocks and relocks through the same
// calls, keep the depth exact (§11 T5: the guard sites whose unique_lock unlocks and relocks, and the engine's waits).
// Engagement, VOID counting as failed: this thread takes the lock while the waiter waits, so the wait released it.
arm_result arm_witness_relock() {
    arm_result      r;
    witnessed_mutex m{ GGML_SYCL_WITNESS_NAME("witness-arm-cv") };
    int             depths[4] = { -1, -1, -1, -1 };
    {
        std::unique_lock lock(m);
        depths[0] = witnessed_mutex_depth(m);
        lock.unlock();
        depths[1] = witnessed_mutex_depth(m);
        lock.lock();
        depths[2] = witnessed_mutex_depth(m);
    }
    depths[3] = witnessed_mutex_depth(m);
    expect(r, depths[0] == 1 && depths[1] == 0 && depths[2] == 1 && depths[3] == 0,
           "a unique_lock's lock, unlock, relock and scope end read depths " + std::to_string(depths[0]) + " " +
               std::to_string(depths[1]) + " " + std::to_string(depths[2]) + " " + std::to_string(depths[3]) +
               ", want 1 0 1 0");

    std::condition_variable_any cv;
    bool                        waiting      = false;
    bool                        go           = false;
    int                         in_wait      = 2;  // the least depth the predicate read; it runs with the lock held
    int                         after_wait   = -1;
    int                         after_unlock = -1;
    std::thread                 waiter([&] {
        std::unique_lock lock(m);
        waiting = true;
        cv.wait(lock, [&] {
            in_wait = std::min(in_wait, witnessed_mutex_depth(m));
            return go;
        });
        after_wait = witnessed_mutex_depth(m);
        lock.unlock();
        after_unlock = witnessed_mutex_depth(m);
    });
    const bool                  released = within_bound([&] {
        std::lock_guard lock(m);
        if (waiting) {
            go = true;
        }
        return waiting;
    });
    if (!released) {
        std::lock_guard lock(m);
        go = true;
    }
    cv.notify_all();
    waiter.join();
    if (!released) {
        void_arm(r, "this thread did not take the lock while the waiter waited within 5 s");
    }
    expect(r, in_wait == 1 && after_wait == 1 && after_unlock == 0,
           "the waiter's predicate read depth " + std::to_string(in_wait) + ", the wait's return " +
               std::to_string(after_wait) + " and the unlock " + std::to_string(after_unlock) + ", want 1, 1 and 0");
    return r;
}

// The table is fixed and allocates nothing (§11 T5), so a lock that finds it full aborts naming that lock rather than
// go unwitnessed: nine witnessed locks held at once, in a child, since the abort ends the process.
arm_result arm_witness_table_full() {
    arm_result      r;
    const child_run child = run_child_strings({ "test-sycl-lazy-io-engine-host", k_table_child },
                                              child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
    if (!child.ran) {
        void_arm(r, "the nine-lock child did not end within " + num(k_bound.count()) + " s");
        return r;
    }
    const char * want    = "witnessed_mutex: the depth table is full taking witness-full-8";
    const bool   aborted = WIFSIGNALED(child.status) && WTERMSIG(child.status) == SIGABRT;
    expect(r, aborted && text_count_with(child.err, want) == 1 && text_count_with(child.err, "TABLE-CHILD") == 0,
           "the ninth witnessed lock ended the child with " + status_text(child.status) + ", " +
               num(text_count_with(child.err, want)) + " line(s) '" + want + "' and " +
               num(text_count_with(child.err, "TABLE-CHILD")) + " return line(s), want SIGABRT, 1 and 0");
    return r;
}

int witness_table_child() {
    static const char * const        names[9] = { "witness-full-0", "witness-full-1", "witness-full-2",
                                                  "witness-full-3", "witness-full-4", "witness-full-5",
                                                  "witness-full-6", "witness-full-7", "witness-full-8" };
    std::unique_ptr<witnessed_mutex> locks[9];
    for (size_t i = 0; i < 9; ++i) {
        (void) names[i];
        locks[i].reset(new witnessed_mutex{ GGML_SYCL_WITNESS_NAME(names[i]) });
    }
    for (size_t i = 0; i < 9; ++i) {
        locks[i]->lock();
    }
    std::fprintf(stderr, "TABLE-CHILD returned with 9 locks held\n");
    for (size_t i = 9; i-- > 0;) {
        locks[i]->unlock();
    }
    return 0;
}

// The reap's negative control (§11 T5): with no witnessed lock held a FULL reap runs, counts, and refuses nothing.
arm_result arm_reap_unheld() {
    arm_result r;
    log_clear();
    const lazy_registry_state s0 = registry();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    const lazy_registry_state s1 = registry();
    expect(r,
           s1.full_reaps_run == s0.full_reaps_run + 1 && s1.reaps_refused == s0.reaps_refused &&
               log_count_with("[HOST-PLAN-BUG] ring reap refused") == 0,
           "with no lock held the reap ran " + num(s1.full_reaps_run - s0.full_reaps_run) + " time(s), refused " +
               num(s1.reaps_refused - s0.reaps_refused) + " and printed " +
               num(log_count_with("[HOST-PLAN-BUG] ring reap refused")) + " refusal line(s), want 1, 0 and 0");
    return r;
}

// The reap's positive control, the pool's half (§11 T5): a FULL reap called with the registry's leaf mutex, or the live
// engine's queue lock, held reads that lock through the witness, prints one "[HOST-PLAN-BUG] ring reap refused: <name>
// held" line and reaps nothing, and registry_mutex_held() reads the registry's mutex held and nothing else. Each run is
// a child bounded at 5 s, since a reap that takes the registry's mutex under itself never returns. A table's strand
// lock is not in this tree, and the control says so: not-wired.
arm_result arm_reap_refuses_held_lock() {
    struct lock_case {
        int          lock;
        const char * name;
        bool         registry;
    };

    static const lock_case cases[] = {
        { LAZY_LOCK_REGISTRY, "g_lazy_registry_mutex", true  },
        { LAZY_LOCK_QUEUE,    "lazy_io_engine::mutex", false },
    };
    arm_result r;
    for (const lock_case & c : cases) {
        const child_run child =
            run_child_strings({ "test-sycl-lazy-io-engine-host", k_holding_child, std::to_string(c.lock) },
                              child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
        if (!child.ran) {
            void_arm(r, std::string("the reap under ") + c.name + " did not return within " + num(k_bound.count()) +
                            " s, want it to refuse and return");
            continue;
        }
        const std::string want = std::string("REAP-HOLDING status=held witnessed=1 registry_held=") +
                                 (c.registry ? "1" : "0") + " refused=1 held=" + c.name;
        const std::string warn = std::string("[HOST-PLAN-BUG] ring reap refused: ") + c.name + " held";
        expect(r,
               WIFEXITED(child.status) && WEXITSTATUS(child.status) == 0 &&
                   text_count_with(child.err, want.c_str()) == 1 && text_count_with(child.err, warn.c_str()) == 1,
               std::string("the reap under ") + c.name + " ended with " + status_text(child.status) + ", printed '" +
                   first_line_with(child.err, "REAP-HOLDING") + "' and " +
                   num(text_count_with(child.err, "[HOST-PLAN-BUG] ring reap refused")) +
                   " refusal line(s), want exit 0, '" + want + "' and 1 '" + warn + "'");
    }
    const lazy_reap_holding_result strand = ggml_sycl_test_lazy_reap_holding_pool(LAZY_LOCK_STRAND);
    expect(r, strand.status == lazy_holding_status::NOT_WIRED,
           std::string("the strand lock's control reported ") + lazy_holding_status_name(strand.status) +
               ", want not-wired");
    return r;
}

int reap_holding_child(int lock) {
    ring_lease lease;
    if (lock == LAZY_LOCK_QUEUE) {
        lease = lazy_io_pool_take();
        if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
            std::fprintf(stderr, "REAP-HOLDING-CHILD: the ring was not ensured\n");
            return 3;
        }
    }
    const lazy_reap_holding_result res = ggml_sycl_test_lazy_reap_holding_pool(lock);
    std::fprintf(stderr, "REAP-HOLDING status=%s witnessed=%d registry_held=%d refused=%d held=%s\n",
                 lazy_holding_status_name(res.status), res.witnessed ? 1 : 0, res.registry_held ? 1 : 0,
                 res.refused ? 1 : 0, res.held != nullptr ? res.held : "(none)");
    lease.reset();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    return 0;
}

// The drop's witness (§3.3, §11 T5): a held lease dropped by a thread that holds a witnessed lock prints one
// "[HOST-PLAN-BUG] ring lease dropped with <name> held" line, counts it, and completes; a drop with nothing held prints
// nothing. Under the registry's own mutex the drop would take that mutex again, so it prints the line and aborts, in a
// child bounded at 5 s.
arm_result arm_drop_witness() {
    arm_result r;
    log_clear();
    const lazy_registry_state s0     = registry();
    ring_lease                lease  = lazy_io_pool_take();
    ring_lease                second = lease.clone();
    witnessed_mutex           m{ GGML_SYCL_WITNESS_NAME("witness-arm-drop") };
    {
        std::lock_guard lock(m);
        second.reset();
    }
    const lazy_registry_state s1   = registry();
    const char *              warn = "[HOST-PLAN-BUG] ring lease dropped with witness-arm-drop held";
    expect(r, log_count_with(warn) == 1 && s1.lease_drops_locked == s0.lease_drops_locked + 1 && s1.live_leases == 1,
           "the drop under witness-arm-drop printed " + num(log_count_with(warn)) + " line(s) '" + warn +
               "', counted " + num(s1.lease_drops_locked - s0.lease_drops_locked) + " and left " + num(s1.live_leases) +
               " lease(s), want 1, 1 and 1");
    lease.reset();
    expect(r, log_count_with("[HOST-PLAN-BUG] ring lease dropped") == 1,
           "the drop with nothing held printed a drop line: " +
               num(log_count_with("[HOST-PLAN-BUG] ring lease dropped")) + " in all, want 1");
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);

    const child_run child = run_child_strings({ "test-sycl-lazy-io-engine-host", k_drop_child },
                                              child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
    if (!child.ran) {
        void_arm(r, "the drop under g_lazy_registry_mutex did not end within " + num(k_bound.count()) +
                        " s, want it to abort naming that mutex");
        return r;
    }
    const char * line  = "[HOST-PLAN-BUG] ring lease dropped with g_lazy_registry_mutex held";
    const char * abort = "ring_lease::reset: the dropping thread holds the IO pool registry's leaf mutex";
    const bool   died  = WIFSIGNALED(child.status) && WTERMSIG(child.status) == SIGABRT;
    expect(r,
           died && text_count_with(child.err, line) == 1 && text_count_with(child.err, abort) == 1 &&
               text_count_with(child.err, "DROP-CHILD") == 0,
           "the drop under g_lazy_registry_mutex ended with " + status_text(child.status) + ", " +
               num(text_count_with(child.err, line)) + " drop line(s) and " + num(text_count_with(child.err, abort)) +
               " abort line(s), want SIGABRT, 1 and 1");
    return r;
}

void reset_lease(void * lease) {
    static_cast<ring_lease *>(lease)->reset();
}

int drop_holding_child() {
    ring_lease lease  = lazy_io_pool_take();
    ring_lease second = lease.clone();
    (void) ggml_sycl_test_lazy_run_holding(LAZY_LOCK_REGISTRY, reset_lease, &second);
    std::fprintf(stderr, "DROP-CHILD returned\n");
    return 0;
}

// A failure to reset is logged and never thrown (§3.3): a FULL reap whose first step throws std::system_error, as
// std::mutex::lock can, prints one "[HOST-PLAN-BUG] ring reap failed: <what>" line, counts it and returns with its
// parked pool left for the next reap, which frees it. In a child, since a throw that escapes the noexcept reap ends
// the process.
arm_result arm_reap_throw() {
    arm_result      r;
    const child_run child = run_child_strings({ "test-sycl-lazy-io-engine-host", k_throw_child },
                                              child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
    if (!child.ran) {
        void_arm(r, "the throwing reap's child did not end within " + num(k_bound.count()) + " s");
        return r;
    }
    expect(r, WIFEXITED(child.status) && WEXITSTATUS(child.status) == 0,
           "the reap's thrown step ended the child with " + status_text(child.status) +
               ", want the reap to log it and return");
    const char * warn = "[HOST-PLAN-BUG] ring reap failed: the test seam's throw at the reap's first step";
    expect(r, text_count_with(child.err, warn) == 1,
           "the throwing reap printed " + num(text_count_with(child.err, "[HOST-PLAN-BUG] ring reap failed")) +
               " failure line(s), want 1 '" + warn + "'");
    expect(r,
           text_count_with(child.err, "THROW-CHILD after the throw: parked=1 failed=1") == 1 &&
               text_count_with(child.err, "THROW-CHILD after the next reap: parked=0") == 1,
           "the child printed '" + first_line_with(child.err, "THROW-CHILD after the throw") + "' and '" +
               first_line_with(child.err, "THROW-CHILD after the next reap") +
               "', want parked=1 failed=1, then parked=0");
    return r;
}

int reap_throw_child() {
    lazy_io_pool_take().reset();  // the last drop parks the pool
    ggml_sycl_test_lazy_reap_throw_next();
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    lazy_registry_state s = registry();
    std::fprintf(stderr, "THROW-CHILD after the throw: parked=%llu failed=%llu\n", (unsigned long long) s.parked_nodes,
                 (unsigned long long) s.reaps_failed);
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    s = registry();
    std::fprintf(stderr, "THROW-CHILD after the next reap: parked=%llu\n", (unsigned long long) s.parked_nodes);
    return 0;
}

// The strict arms (§10: each [HOST-PLAN-BUG] line of the ring reap and the lease drop is a WARN that aborts under
// ggml_sycl_strict_enabled()). Each of the pool's four such sites is reached in a child whose fixture strict switch
// stands for GGML_SYCL_STRICT_LEASES=1: a FULL reap refused under a held witnessed lock, a lease dropped under one, a
// FULL reap that finds a parked pool a take joined, and a FULL reap whose first step throws. The child prints the
// site's one [HOST-PLAN-BUG] line, then aborts (SIGABRT) with one line naming the same figures under
// GGML_SYCL_STRICT_LEASES=1, and never returns. This process keeps the switch off, so the arms above score the same
// sites' WARN-and-continue path.
// Engagement: the child ran within 5 s; at the parked site, its ensure made a ring and its take joined that pool.
// Mutant: pool-strict-abort-dropped, under which each site logs and continues under strict leases too, so the child
// returns from its site. One registration per site scores that site's return line after its one WARN:
// pool-strict-abort-dropped the throwing reap's, and its aliases pool-strict-abort-dropped-refused, -drop and -parked
// (the same selector) the refused reap's, the dropped lease's and the parked pool's.
struct strict_site {
    const char * name;        // the child's site argument
    const char * warn;        // the site's one [HOST-PLAN-BUG] line
    const char * abort;       // the strict abort's line, or the part of it this arm fixes
    const char * abort_tail;  // the rest the abort's line also carries, or nullptr when `abort` is the whole line
};

const strict_site k_strict_refused = {
    "refused",
    "[HOST-PLAN-BUG] ring reap refused: strict-child-lock held",
    "ring reap refused: strict-child-lock held under GGML_SYCL_STRICT_LEASES=1",
    nullptr,
};
const strict_site k_strict_dropped = {
    "dropped",
    "[HOST-PLAN-BUG] ring lease dropped with strict-child-lock held",
    "ring lease dropped with strict-child-lock held under GGML_SYCL_STRICT_LEASES=1",
    nullptr,
};
const strict_site k_strict_parked = {
    "parked",
    "[HOST-PLAN-BUG] reap found a parked pool holding 1 lease(s), want 0",
    "reap found a parked pool holding 1 lease(s) under GGML_SYCL_STRICT_LEASES=1",
    nullptr,
};
// The abort names the exception's what(), and a std::system_error's ends with the platform's message for its error
// code, which this arm does not fix. So the line is scored by the two parts the pool writes: the seam's text, which
// the WARN line shares, and the strict suffix, which only the abort carries.
const strict_site k_strict_failed = {
    "failed",
    "[HOST-PLAN-BUG] ring reap failed: the test seam's throw at the reap's first step",
    "ring reap failed: the test seam's throw at the reap's first step",
    "under GGML_SYCL_STRICT_LEASES=1",
};

// The child's return is scored first, and its FAIL line carries the count of the site's WARN lines. The return line is
// the last the child writes before _Exit, so every WARN counted was written before it, and a registration that names a
// count of 1 matches only a child whose site logged its WARN and then returned. A child that died before its site,
// whatever killed it, fails on the abort's line instead, and one that returned without its WARN fails with a count of
// 0, neither of which a registration names.
arm_result score_strict_site(const strict_site & site) {
    arm_result               r;
    std::vector<std::string> args = { "test-sycl-lazy-io-engine-host", k_strict_child, site.name };
    if (g_mutant != nullptr) {
        args.emplace_back(g_mutant);
    }
    const child_run child = run_child_strings(args, child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
    if (!child.ran) {
        void_arm(
            r, std::string("the strict child at ") + site.name + " did not end within " + num(k_bound.count()) + " s");
        return r;
    }
    const std::string engaged = first_line_with(child.err, "STRICT-CHILD void:");
    if (engaged != "(none)") {
        void_arm(r, engaged);
        return r;
    }
    const size_t warns = text_count_with(child.err, site.warn);
    expect(r, text_count_with(child.err, "STRICT-CHILD returned") == 0,
           "the strict child returned from its site after " + num(warns) + " line(s) '" + site.warn + "': '" +
               first_line_with(child.err, "STRICT-CHILD returned") + "'");
    const size_t      aborts     = text_count_with(child.err, site.abort, site.abort_tail);
    const std::string abort_text = site.abort_tail == nullptr ?
                                       "'" + std::string(site.abort) + "'" :
                                       "with '" + std::string(site.abort) + "' and '" + site.abort_tail + "'";
    expect(r, aborts == 1, "the strict child printed " + num(aborts) + " line(s) " + abort_text + ", want 1");
    expect(r, WIFSIGNALED(child.status) && WTERMSIG(child.status) == SIGABRT,
           "the strict child ended with " + status_text(child.status) + ", want SIGABRT");
    expect(r, warns == 1 && text_count_with(child.err, "[HOST-PLAN-BUG]") == 1,
           "the strict child printed " + num(warns) + " line(s) '" + site.warn + "' among " +
               num(text_count_with(child.err, "[HOST-PLAN-BUG]")) + " [HOST-PLAN-BUG] line(s), want 1 and 1");
    return r;
}

arm_result arm_strict_refused() {
    return score_strict_site(k_strict_refused);
}

arm_result arm_strict_dropped() {
    return score_strict_site(k_strict_dropped);
}

arm_result arm_strict_parked() {
    return score_strict_site(k_strict_parked);
}

arm_result arm_strict_failed() {
    return score_strict_site(k_strict_failed);
}

// The strict child's way out when its site returns or it cannot engage: its STRICT-CHILD line, then _Exit, with
// nothing torn down, so no teardown can stand in for the abort the arm scores.
[[noreturn]] void strict_child_exit(const std::string & line, int code) {
    std::fprintf(stderr, "STRICT-CHILD %s\n", line.c_str());
    std::fflush(stderr);
    std::_Exit(code);
}

// The strict arms' child: the fixture's strict switch on, `mutant` planted by the caller, then the site named `site`.
[[noreturn]] void strict_child(const char * site, lazy_ring_mutant mutant) {
    lazy_io_host_fixture::g_state.strict.store(true);
    if (std::strcmp(site, k_strict_refused.name) == 0) {
        witnessed_mutex m{ GGML_SYCL_WITNESS_NAME("strict-child-lock") };
        std::lock_guard lock(m);
        ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    } else if (std::strcmp(site, k_strict_dropped.name) == 0) {
        ring_lease      lease  = lazy_io_pool_take();
        ring_lease      second = lease.clone();
        witnessed_mutex m{ GGML_SYCL_WITNESS_NAME("strict-child-lock") };
        std::lock_guard lock(m);
        second.reset();
    } else if (std::strcmp(site, k_strict_parked.name) == 0) {
        ring_lease lease = lazy_io_pool_take();
        if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
            strict_child_exit("void: the ring was not ensured", 3);
        }
        lease.reset();  // the last drop parks the pool with its ring
        // The defect the site reports: a take blind to DYING joins the parked pool, as TAKE_IGNORES_DYING plants it.
        ggml_sycl_test_lazy_ring_mutant_select(lazy_ring_mutant::TAKE_IGNORES_DYING);
        ring_lease joined = lazy_io_pool_take();
        ggml_sycl_test_lazy_ring_mutant_select(mutant);
        if (joined.ring_bytes() != k_need) {
            strict_child_exit("void: the take did not join the parked pool: ring " + num(joined.ring_bytes()) +
                                  " B, want " + num(k_need),
                              3);
        }
        ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    } else if (std::strcmp(site, k_strict_failed.name) == 0) {
        lazy_io_pool_take().reset();  // the last drop parks the pool
        ggml_sycl_test_lazy_reap_throw_next();
        ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    } else {
        strict_child_exit(std::string("unknown site ") + site, 2);
    }
    strict_child_exit("returned", 0);
}

// Ensure's entry witness (§3.3, §11 T5): ensure allocates through unified_allocate_owner, which charges the budget
// ledger and may yield, so it is called with no witnessed lock held. One entered with a lock held aborts there, before
// it refuses, reserves or allocates anything, with one line naming the lock; one entered with none held ensures as
// before. Each runs in a child that takes the pool first, so the ensure finds a live pool: under a local witnessed
// lock, ensure-child-lock, and, the negative control, with no lock held.
// Engagement, VOID counting as failed: both children ended within 5 s.
// Mutant: ensure-witness-dropped, under which the held child's ensure runs and returns ok, and its registration
// scores that return line.
const char * k_ensure_abort = "block_ring_ensure: the calling thread holds ensure-child-lock";

arm_result arm_ensure_held_lock() {
    arm_result               r;
    std::vector<std::string> held_args = { "test-sycl-lazy-io-engine-host", k_ensure_child, "held" };
    std::vector<std::string> none_args = { "test-sycl-lazy-io-engine-host", k_ensure_child, "none" };
    if (g_mutant != nullptr) {
        held_args.emplace_back(g_mutant);
        none_args.emplace_back(g_mutant);
    }
    const child_run held = run_child_strings(held_args, child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
    const child_run none = run_child_strings(none_args, child_environment({ "GGML_NO_BACKTRACE=1" }), k_bound);
    if (!held.ran || !none.ran) {
        void_arm(r, std::string("the ensure child ") + (held.ran ? "with no lock held" : "holding the lock") +
                        " did not end within " + num(k_bound.count()) + " s");
        return r;
    }
    // The ensure's return first, so the mutant's FAIL line is one only a held child that ran its ensure prints. A held
    // child that died before its ensure, whatever killed it, fails on the abort's line instead, which no registration
    // names.
    expect(r, text_count_with(held.err, "ENSURE-CHILD returned") == 0,
           "the held child's ensure returned: '" + first_line_with(held.err, "ENSURE-CHILD returned") + "'");
    expect(r, text_count_with(held.err, k_ensure_abort) == 1,
           "the held child printed " + num(text_count_with(held.err, k_ensure_abort)) + " line(s) '" + k_ensure_abort +
               "', want 1");
    expect(r, WIFSIGNALED(held.status) && WTERMSIG(held.status) == SIGABRT,
           "the held child ended with " + status_text(held.status) + ", want SIGABRT");
    const bool none_ok = WIFEXITED(none.status) && WEXITSTATUS(none.status) == 0;
    expect(r,
           none_ok && text_count_with(none.err, "ENSURE-CHILD returned ok") == 1 &&
               text_count_with(none.err, "block_ring_ensure: the calling thread holds") == 0,
           "the child with no lock held ended with " + status_text(none.status) + " and printed '" +
               first_line_with(none.err, "ENSURE-CHILD") + "', want exit 0 and 'ENSURE-CHILD returned ok'");
    return r;
}

// The entry witness's child: a take, so the ensure finds a live pool, then the ensure, with ensure-child-lock held when
// `held`. It prints the ensure's status and ends through _Exit with nothing torn down.
[[noreturn]] void ensure_holding_child(bool held) {
    ring_lease              lease = lazy_io_pool_take();
    lazy_ring_ensure_status st    = lazy_ring_ensure_status::OK;
    if (held) {
        witnessed_mutex m{ GGML_SYCL_WITNESS_NAME("ensure-child-lock") };
        std::lock_guard lock(m);
        st = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    } else {
        st = block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn);
    }
    std::fprintf(stderr, "ENSURE-CHILD returned %s\n", name(st).c_str());
    std::fflush(stderr);
    std::_Exit(0);
}

// The worker-drop RED (§3.3, §11 T5): model A is unloaded with a read in flight, so the read's lease is the last, and
// its drop and WORKER reap run on a pool worker. Within 5 s the ring's bytes return to the pre-ensure figure through
// the fixture ledger, with no FULL reap run and no join on the completion path: the pool, with its workers, waits on
// the reap list for the next FULL reap, which joins every one of its LAZY_BLOCK_RING_QD workers from this thread. The
// join census records each join by its caller, so a join from a pool worker, or a worker asked to join itself, is a
// finding.
// Engagement, VOID counting as failed: the read is held on a worker when A's lease drops, and the worker's reap ran to
// its end and gave the ring's bytes back within the bound.
// Mutant: worker-reap-destroys-pool, under which the WORKER reap deletes the parked pool, joining its workers from one
// of them, which asks itself to join.
arm_result arm_engine_worker_drop() {
    arm_result     r;
    const uint64_t charged0 = live_charged();
    ring_lease     lease    = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    gate g;
    ggml_sycl_test_lazy_io_read_source(gated_source, &g);
    request_fixture f(7);
    const size_t    added = lazy_io_submit(1, -1, block_span(10), key0(7, 1), f.waiter(0, 64));
    if (added != 1 || !g.wait_holders(1)) {
        void_arm(r, "the read was not held on a worker within the hook bound (" + num(added) + " waiter(s) queued)");
        g.release();
        (void) delivered(f.ticket);
        ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
        settle(lease);
        return r;
    }
    const lazy_registry_state s0 = registry();
    lease.reset();  // A's unload: the read's lease is now the pool's last
    g.release();
    const bool done = within_bound([&] {
        const lazy_registry_state s = registry();
        return s.worker_reaps > s0.worker_reaps && live_charged() == charged0;
    });
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    if (!done) {
        void_arm(r, "the worker's drop and reap did not finish within " + num(k_bound.count()) +
                        " s: " + num(live_charged() - charged0) + " ring bytes still charged, " +
                        num(registry().parked_nodes) + " parked");
        ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
        return r;
    }
    const lazy_registry_state s1 = registry();
    expect(r, s1.full_reaps_run == s0.full_reaps_run,
           num(s1.full_reaps_run - s0.full_reaps_run) + " FULL reap(s) ran during the worker's drop, want 0");
    expect(r, s1.joins_from_worker == s0.joins_from_worker && s1.self_joins == s0.self_joins && s1.joins == s0.joins,
           "the worker's drop ran " + num(s1.joins_from_worker - s0.joins_from_worker) +
               " join(s) on a pool worker and " + num(s1.self_joins - s0.self_joins) +
               " self-join(s), want 0 and 0: no join runs on the completion path");
    expect(r, s1.parked_nodes == 1,
           "after the worker's drop " + num(s1.parked_nodes) + " pool(s) wait on the reap list, want 1");
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    const lazy_registry_state s2 = registry();
    expect(r,
           s2.joins - s1.joins == LAZY_BLOCK_RING_QD && s2.joins_from_worker == s1.joins_from_worker &&
               s2.self_joins == s1.self_joins && s2.parked_nodes == 0,
           "the next FULL reap recorded " + num(s2.joins - s1.joins) + " join(s), " +
               num(s2.joins_from_worker - s1.joins_from_worker) + " of them from a pool worker, and left " +
               num(s2.parked_nodes) + " parked, want " + num(LAZY_BLOCK_RING_QD) +
               ", 0 and 0: the pool's workers are joined there, from this thread");
    return r;
}

// A ticket's wait returns once its three reads are delivered, every row landed and no error; a ticket with nothing
// submitted returns at once; a failed read returns its errno. Every wait is bounded. The three reads are held on a
// gate until the wait has been seen started and not returned for 200 ms, so the wait cannot pass by starting after they
// finished. Engagement, VOID counting as failed: the three reads are held on workers before the wait starts, and the
// wait has started before the 200 ms window opens.
// Mutants: ticket-wait-returns-at-once, under which the wait returns with the reads held, and ticket-zero-no-notify,
// under which the wait is never woken once they are released.
arm_result arm_ticket_wait() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    ggml_sycl_test_lazy_io_read_source(fill_source, nullptr);
    {
        auto       idle    = std::make_shared<request_fixture>(1);
        int        idle_rc = -1;
        const bool back    = returns_within([idle] { return lazy_io_ticket_wait(idle->ticket); }, 5, idle_rc);
        expect(r, back && idle_rc == 0,
               back ? "an empty ticket's wait returned " + std::to_string(idle_rc) + ", want 0" :
                      std::string("an empty ticket's wait did not return within 5 s"));
    }

    // Static, so it outlives its holders: a held read still takes the gate's mutex on its way out of hold() after
    // release() returns. The guard releases it on every path out of this arm.
    static gate        g;
    gate_release_guard release_on_exit{ g };

    ggml_sycl_test_lazy_io_read_source(gated_source, &g);
    auto   f     = std::make_shared<request_fixture>(2);
    size_t added = 0;
    for (uint64_t b = 0; b < 3; ++b) {
        lazy_io_waiter w = f->waiter(0, 100);
        w.landing_offset = b * 100;
        added += lazy_io_submit(1, -1, block_span(10 + b), key0(2, 0), w);
    }
    if (added != 3 || !g.wait_holders(3)) {
        void_arm(
            r, "the three reads were not held on workers within the hook bound (" + num(added) + " waiter(s) queued)");
        g.release();
        (void) delivered(f->ticket);
        ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
        settle(lease);
        return r;
    }
    auto             entered = std::make_shared<std::atomic<bool>>(false);
    std::future<int> waited  = run_detached([f, entered] {
        entered->store(true);
        return lazy_io_ticket_wait(f->ticket);
    });
    if (!wait_for([&entered] { return entered->load(); })) {
        void_arm(r, "the wait's thread had not started within the hook bound");
        g.release();
        (void) delivered(f->ticket);
        ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
        settle(lease);
        return r;
    }
    expect(r, waited.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout,
           "the wait returned with 3 reads held");
    g.release();
    const bool back = waited.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    const int  rc   = back ? waited.get() : -1;
    expect(r, back && rc == 0,
           back ? "after the release the wait returned " + std::to_string(rc) + ", want 0" :
                  std::string("after the release the wait did not return within 5 s"));
    for (uint64_t b = 0; b < 3; ++b) {
        expect(r, f->slab[b * 100] == file_byte((10 + b) * LAZY_IO_BLOCK_BYTES), "row " + num(b) + " did not land");
    }

    static uint64_t bad = 20 * LAZY_IO_BLOCK_BYTES;
    auto            e   = std::make_shared<request_fixture>(3);
    ggml_sycl_test_lazy_io_read_source(eio_source, &bad);
    (void) lazy_io_submit(1, -1, block_span(20), key0(3, 0), e->waiter(0, 100));
    int        eio_rc   = -1;
    const bool eio_back = returns_within([e] { return lazy_io_ticket_wait(e->ticket); }, 5, eio_rc);
    expect(r, eio_back && eio_rc == EIO,
           eio_back ? "a failed read's wait returned " + std::to_string(eio_rc) + ", want " + std::to_string(EIO) :
                      std::string("a failed read's wait did not return within 5 s"));
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// ---- the ubatch gather (STORAGE tables) ----

// The gather arms' row cache, on plain memory.
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

constexpr size_t   k_gather_row    = 170;     // a Q8_0 row of 160 values, the PLE table's
constexpr uint64_t k_gather_offset = 4032;    // the table's first byte in the file, not on a block
constexpr int64_t  k_gather_rows   = 100000;  // the table's rows
constexpr size_t   k_gather_max    = 4;       // the most rows one gather of these arms asks for

// One gather's world in one shared state: the cache, the table's place in the file, the destination and the stats. A
// gather that never returns keeps it alive through its capture, so it cannot outlive the memory it writes.
struct gather_state {
    plain_source         mem;  // before the cache, which it outlives
    lazy_row_cache_build cache;
    lazy_gather_source   src;
    std::vector<uint8_t> dst = std::vector<uint8_t>(k_gather_max * k_gather_row, 0);
    lazy_gather_stats    s;
};

// A state whose cache holds 64 rows of the 100000-row table at file offset 4032; its cache is null when the build was
// refused.
std::shared_ptr<gather_state> make_gather_state() {
    auto                  st = std::make_shared<gather_state>();
    lazy_row_cache_config cfg;
    cfg.name            = "fixture.gather";
    cfg.row_size        = k_gather_row;
    cfg.n_rows          = k_gather_rows;
    cfg.cache_slots     = 64;
    cfg.floor_slots     = 64;
    cfg.chunk_bytes     = 64 * k_gather_row;
    cfg.n_shards        = 1;
    st->cache           = lazy_row_cache::build(cfg, st->mem);
    st->src.fd          = -1;
    st->src.file_id     = k_file;
    st->src.file_offset = k_gather_offset;
    st->src.row_size    = k_gather_row;
    st->src.n_rows      = k_gather_rows;
    return st;
}

int gather_into(gather_state & s, const std::vector<int32_t> & rows, uint32_t epoch) {
    return lazy_gather_rows(s.cache.cache.get(), s.src, 1, epoch, 0, rows.data(), rows.size(), s.dst.data(),
                            s.dst.size(), &s.s);
}

// The file's byte `j` of table row `row`.
uint8_t row_byte(int64_t row, size_t j) {
    return file_byte(k_gather_offset + static_cast<uint64_t>(row) * k_gather_row + j);
}

// Fills the destination of each row of `rows` with bytes that differ from the file's at every position, so a row the
// gather did not copy reads as a mismatch. False when some sentinel byte equals its file byte, whose check would then
// be blind.
bool fill_sentinel(std::vector<uint8_t> & dst, const std::vector<int32_t> & rows) {
    bool distinct = true;
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < k_gather_row; ++j) {
            const uint8_t want        = row_byte(rows[i], j);
            dst[i * k_gather_row + j] = static_cast<uint8_t>(want ^ 0xA5u);
            distinct                  = distinct && dst[i * k_gather_row + j] != want;
        }
    }
    return distinct;
}

// The first index into `rows` whose destination does not hold the file's bytes, or rows.size() when every row does.
size_t first_inexact(const std::vector<uint8_t> & dst, const std::vector<int32_t> & rows) {
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < k_gather_row; ++j) {
            if (dst[i * k_gather_row + j] != row_byte(rows[i], j)) {
                return i;
            }
        }
    }
    return rows.size();
}

// One refused gather's own state: a gather that is not refused may still be running when its arm moves on.
struct refusal_state {
    std::shared_ptr<gather_state> owner;  // keeps the cache alive
    lazy_gather_source            src;
    bool                          with_cache = false;
    std::vector<uint8_t>          dst;
    std::vector<uint8_t>          before;
    lazy_gather_stats             s;
};

// Gathers `row` from `src` into a patterned dst of `dst_bytes`, bounded: true when the gather returned EINVAL with
// nothing looked up or submitted and dst unchanged. `got` is what it returned, or -1 when it did not return.
bool refused_at_entry(const std::shared_ptr<gather_state> & owner,
                      const lazy_gather_source &            src,
                      bool                                  with_cache,
                      int32_t                               row,
                      size_t                                dst_bytes,
                      std::string &                         got) {
    auto rs        = std::make_shared<refusal_state>();
    rs->owner      = owner;
    rs->src        = src;
    rs->with_cache = with_cache;
    rs->dst.resize(dst_bytes);
    for (size_t i = 0; i < dst_bytes; ++i) {
        rs->dst[i] = static_cast<uint8_t>(i * 7u + 1u);
    }
    rs->before      = rs->dst;
    int        rc   = -1;
    const bool back = returns_within(
        [rs, row] {
            return lazy_gather_rows(rs->with_cache ? rs->owner->cache.cache.get() : nullptr, rs->src, 1, 9, 0, &row, 1,
                                    rs->dst.data(), rs->dst.size(), &rs->s);
        },
        5, rc);
    got = back ? std::to_string(rc) + " after " + num(rs->s.rows) + " lookup(s) and " + num(rs->s.reads) +
                     " read(s), dst " + (rs->dst == rs->before ? "unchanged" : "written") :
                 std::string("no return within 5 s");
    return back && rc == EINVAL && rs->s.rows == 0 && rs->s.reads == 0 && rs->dst == rs->before;
}

// A gather of rows 3, 40000, 3 lands each row byte-exact from the file, over a sentinel; a second gather of row 3 hits
// the cache, submits nothing and lands the cache's copy; a row past the table is EINVAL and submits nothing; a gather
// whose read fails returns EIO after its other reads are delivered. A cache of another row size, a row wider than one
// IO chunk can hold, and a table whose end does not fit a uint64_t are each refused EINVAL before anything is looked up
// or submitted, with dst unchanged. Every gather is bounded.
arm_result arm_gather_rows() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    const std::shared_ptr<gather_state> st = make_gather_state();
    if (!st->cache) {
        void_arm(r, "the fixture's row cache was not built: " + st->cache.reason);
        settle(lease);
        return r;
    }
    ggml_sycl_test_lazy_io_read_source(fill_source, nullptr);
    const auto gather = [&r](const std::shared_ptr<gather_state> & s, const std::vector<int32_t> & rows, uint32_t epoch,
                             int & rc) {
        if (!fill_sentinel(s->dst, rows)) {
            void_arm(r, "a sentinel byte equals its file byte");
            return false;
        }
        const bool done = returns_within([s, rows, epoch] { return gather_into(*s, rows, epoch); }, 5, rc);
        expect(r, done, "a gather did not return within 5 s");
        return done;
    };

    int                        rc    = -1;
    const std::vector<int32_t> first = { 3, 40000, 3 };
    if (gather(st, first, 1, rc)) {
        const size_t bad = first_inexact(st->dst, first);
        expect(r, rc == 0 && bad == first.size(),
               "the gather returned " + std::to_string(rc) +
                   (bad < first.size() ? ", and row " + num(first[bad]) + " is not the file's bytes" : ""));
        expect(r, st->s.rows == 3 && st->s.reads >= 2,
               "rows " + num(st->s.rows) + ", reads " + num(st->s.reads) + ", want 3 and at least 2");
    }
    const uint64_t hits0 = st->cache.cache->counters().hits;
    if (gather(st, { 3 }, 2, rc)) {
        expect(r, rc == 0 && st->s.hits == 1 && st->s.reads == 0 && first_inexact(st->dst, { 3 }) == 1,
               "the repeated row did not hit and land: returned " + std::to_string(rc) + ", hits " + num(st->s.hits) +
                   ", reads " + num(st->s.reads));
        expect(r, st->cache.cache->counters().hits == hits0 + 1,
               "the cache counted " + num(st->cache.cache->counters().hits - hits0) + " hit(s) for the repeat, want 1");
    }
    if (gather(st, { static_cast<int32_t>(k_gather_rows) }, 3, rc)) {
        expect(r, rc == EINVAL && st->s.reads == 0,
               "a row past the table returned " + std::to_string(rc) + " after " + num(st->s.reads) +
                   " read(s), want EINVAL and none");
    }
    {
        std::string        got;
        lazy_gather_source other_size = st->src;
        other_size.row_size           = k_gather_row - 10;
        expect(r, refused_at_entry(st, other_size, true, 77777, k_gather_row, got),
               "a cache of " + num(k_gather_row) + " B rows for a table of " + num(other_size.row_size) +
                   " B rows returned " + got + ", want EINVAL, nothing looked up or read, dst unchanged");
        lazy_gather_source wide = st->src;
        wide.row_size           = LAZY_IO_CHUNK_BYTES - 2 * (LAZY_IO_BLOCK_BYTES - 1) + 1;
        expect(r, refused_at_entry(st, wide, false, 0, wide.row_size, got),
               "a row of " + num(wide.row_size) + " B, whose block span exceeds one IO chunk, returned " + got +
                   ", want EINVAL, nothing looked up or read, dst unchanged");
        lazy_gather_source wrapping = st->src;
        wrapping.file_offset        = UINT64_MAX - 100 * LAZY_IO_BLOCK_BYTES;
        expect(r, refused_at_entry(st, wrapping, false, 5000, k_gather_row, got),
               "a table whose end wraps a uint64_t returned " + got +
                   ", want EINVAL, nothing looked up or read, dst unchanged");
    }
    static uint64_t failing = (k_gather_offset + 50000ull * k_gather_row) / LAZY_IO_BLOCK_BYTES * LAZY_IO_BLOCK_BYTES;
    ggml_sycl_test_lazy_io_read_source(eio_source, &failing);
    if (gather(st, { 50000, 7 }, 4, rc)) {
        expect(r, rc == EIO, "a failed read returned " + std::to_string(rc) + ", want " + std::to_string(EIO));
    }
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    // The reads' deliveries insert into the state's cache, so the arm keeps the state until the reads have settled.
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// A gather waits for the reads it submitted: with every read held on a gate, a gather of rows 20000, 30000, 20000,
// 40000 has not returned 200 ms after its three blocks' reads are held, and once the gate opens it returns 0 with all
// four rows byte-exact over a sentinel. The last row's read is submitted after the third row's lookup, so three reads
// held proves the gather is past every submit, and the gate is still shut when the third row looks up row 20000: it
// misses and joins the first row's read, queued or held, never hitting the cache. Nothing in the wait itself can be
// observed without a hook there. Then a gather of rows 25000 and one past the table, which stops at the second row
// without a throw, has not returned 200 ms after the first row's read is held, and once released returns EINVAL with
// row 25000 byte-exact.
// Engagement, VOID counting as failed: each gather's reads are held within the hook bound.
// Mutants: gather-returns-unwaited, under which the first gather returns with the three reads held, and
// gather-error-returns-unwaited, under which the stopped gather returns with its read held.
arm_result arm_gather_waits() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    const std::shared_ptr<gather_state> st   = make_gather_state();
    const std::vector<int32_t>          rows = { 20000, 30000, 20000, 40000 };
    if (!st->cache || !fill_sentinel(st->dst, rows)) {
        void_arm(r, "the fixture's row cache was not built, or a sentinel byte equals its file byte");
        settle(lease);
        return r;
    }
    gate &             g = lasting_gate();
    gate_release_guard release_on_exit{ g };
    ggml_sycl_test_lazy_io_read_source(gated_source, &g);
    std::future<int> gathered = run_detached([st, rows] { return gather_into(*st, rows, 1); });
    if (!g.wait_holders(3)) {
        void_arm(r, "the gather's three reads were not held on workers within the hook bound");
        g.release();
        (void) gathered.wait_for(std::chrono::seconds(5));
        ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
        settle(lease);
        return r;
    }
    expect(r, gathered.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout,
           "the gather returned with 3 reads held");
    g.release();
    const bool   back = gathered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    const int    rc   = back ? gathered.get() : -1;
    const size_t bad  = back ? first_inexact(st->dst, rows) : 0;
    expect(r, back && rc == 0 && bad == rows.size(),
           !back   ? std::string("after the release the gather did not return within 5 s") :
           rc != 0 ? "after the release the gather returned " + std::to_string(rc) + ", want 0" :
                     "row " + num(rows[bad]) + " at index " + num(bad) + " is not the file's bytes");
    expect(
        r, !back || (st->s.rows == 4 && st->s.hits == 0 && st->s.reads == 4),
        "rows " + num(st->s.rows) + ", hits " + num(st->s.hits) + ", reads " + num(st->s.reads) + ", want 4, 0 and 4");

    const std::vector<int32_t> stopping = { 25000, static_cast<int32_t>(k_gather_rows) };
    gate &                     g2       = lasting_gate();
    gate_release_guard         release_g2_on_exit{ g2 };
    if (back && fill_sentinel(st->dst, stopping)) {
        ggml_sycl_test_lazy_io_read_source(gated_source, &g2);
        std::future<int> stopped = run_detached([st, stopping] { return gather_into(*st, stopping, 2); });
        if (g2.wait_holders(1)) {
            expect(r, stopped.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout,
                   "the gather that stopped at a row past the table returned with 1 read held");
        } else {
            void_arm(r, "the stopping gather's read was not held on a worker within the hook bound");
        }
        g2.release();
        const bool back2 = stopped.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        const int  rc2   = back2 ? stopped.get() : -1;
        expect(r, back2 && rc2 == EINVAL,
               back2 ? "the stopping gather returned " + std::to_string(rc2) + ", want EINVAL" :
                       std::string("after the release the stopping gather did not return within 5 s"));
        expect(r, !back2 || (first_inexact(st->dst, { 25000 }) == 1 && st->s.rows == 1 && st->s.reads == 1),
               "after stopping, row 25000 is " +
                   std::string(first_inexact(st->dst, { 25000 }) == 1 ? "exact" : "not the file's bytes") + ", rows " +
                   num(st->s.rows) + ", reads " + num(st->s.reads) + ", want exact, 1 and 1");
    } else if (back) {
        void_arm(r, "a sentinel byte of the stopping gather equals its file byte");
    }
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

// A submit that throws, at every allocation the gather makes: the gather still waits for every read it counted before
// its frame dies, returns ENOMEM with the counted rows landed, and no read the throw left queued is stranded, since a
// worker picks each one with no further submit. The throw is the fault injection's: the gather's thread arms its
// countdown at k, and k grows from 1 until a countdown outlasts every allocation and the gather returns 0 with its rows
// landed. The gated source holds every read it serves, so no read, held or queued, is delivered before the gate opens:
// a gather that returned before then returned with a counted read undelivered, and the reads still holding a lease on
// the pool before then are exactly the reads queued or held, every one of which must reach the gate.
// Engagement, VOID counting as failed: some countdown throws after a counted submit.
// Mutants: gather-error-returns-unwaited, under which the gather returns as soon as its submit throws, and
// submit-throw-unwoken, under which a throw leaves a fresh read queued with no worker woken.
arm_result arm_gather_throw_waits() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    const std::shared_ptr<gather_state> st = make_gather_state();
    if (!st->cache) {
        void_arm(r, "the fixture's row cache was not built: " + st->cache.reason);
        settle(lease);
        return r;
    }
    // The attempts' rows: k's pair is 60000 + 300 k and 150 rows on, and the last pair must stay inside the table.
    constexpr long k_first = 60000;
    constexpr long k_step  = 300;
    constexpr long k_apart = 150;
    constexpr long k_most  = (k_gather_rows - 1 - k_apart - k_first) / k_step;
    static_assert(k_first + k_step * k_most + k_apart < k_gather_rows, "the last attempt's rows leave the table");
    long found = 0;  // attempts that threw after a counted submit
    long last  = 0;  // the countdown the search ended at
    for (long k = 1; k <= k_most && r.result == outcome::PASS; ++k) {
        last                 = k;
        gate &             g = lasting_gate();
        gate_release_guard release_on_exit{ g };
        ggml_sycl_test_lazy_io_read_source(gated_source, &g);
        // Rows of their own each attempt, in two blocks, so neither is cached or in flight from an earlier attempt.
        const int32_t              a    = static_cast<int32_t>(k_first + k_step * k);
        const std::vector<int32_t> rows = { a, static_cast<int32_t>(a + k_apart) };
        if (!fill_sentinel(st->dst, rows)) {
            void_arm(r, "a sentinel byte equals its file byte");
            break;
        }
        std::future<int> gathered = run_detached([st, rows, k] {
            g_alloc_countdown = k;
            const int rc      = gather_into(*st, rows, static_cast<uint32_t>(k));
            g_alloc_countdown = 0;
            return rc;
        });
        const auto       returned = [&gathered](std::chrono::milliseconds within) {
            return gathered.wait_for(within) == std::future_status::ready;
        };
        if (!wait_for([&] { return returned(std::chrono::milliseconds(0)) || holding(g, 1); })) {
            void_arm(r, "a gather under countdown " + num(k) + " neither returned nor held a read within the bound");
            break;
        }
        const bool shut_return = returned(std::chrono::milliseconds(0)) || returned(std::chrono::milliseconds(200));
        // The arm's own lease is the one lease that is not a read's; a read delivered by an earlier attempt drops its
        // lease just after its delivery, so the count is polled.
        size_t     queued      = 0;
        const bool picked      = wait_for([&] {
            const uint64_t leases = registry().live_leases;
            queued                = leases > 0 ? static_cast<size_t>(leases - 1) : 0;
            return holding(g, queued);
        });
        g.release();
        if (!returned(std::chrono::seconds(5))) {
            fail(r, "after the release a gather under countdown " + num(k) + " did not return within 5 s");
            break;
        }
        const int rc = gathered.get();
        if (!picked) {
            // The stranded read waits for the next fresh read's wake, since no held worker may exist to loop back to
            // it, so its lease can outlive this arm and leave a later arm VOID. Only a defect reaches this branch.
            fail(r, "after a submit threw, a queued read was never picked by a worker: " + num(queued) +
                        " read(s) queued or held under countdown " + num(k));
            break;
        }
        if (rc == 0) {
            // The countdown outlasted every allocation the gather made: no throw, so every row must have landed.
            const size_t bad = first_inexact(st->dst, rows);
            expect(r, bad == rows.size(),
                   "a gather under countdown " + num(k) + " returned 0, and row " +
                       num(rows[bad < rows.size() ? bad : 0]) + " is not the file's bytes");
            break;
        }
        if (rc != ENOMEM) {
            fail(r, "a gather under countdown " + num(k) + " returned " + std::to_string(rc) + ", want ENOMEM or 0");
            break;
        }
        if (st->s.reads == 0) {
            continue;  // the throw came before any row's submit was counted
        }
        ++found;
        expect(r, !shut_return, "after a submit threw, the gather returned with a submitted read still held");
        expect(r, first_inexact(st->dst, { a }) == 1,
               "after a submit threw under countdown " + num(k) + ", the counted row " + num(a) +
                   " is not the file's bytes");
    }
    if (found == 0) {
        void_arm(r, "no countdown from 1 to " + num(last) + ", where the search stopped, threw after a counted submit");
    }
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    expect(r, settle(lease) && live_charged() == 0,
           "after " + num(found) + " throwing attempt(s) the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}

lazy_ring_mutant mutant_named(const char * text) {
    struct entry {
        const char *     text;
        lazy_ring_mutant mutant;
    };

    static const entry table[] = {
        { "no-ensuring",                       lazy_ring_mutant::NO_ENSURING                   },
        { "ensuring-kept-on-throw",            lazy_ring_mutant::ENSURING_KEPT_ON_THROW        },
        { "install-after-shut-down",           lazy_ring_mutant::INSTALL_AFTER_SHUT_DOWN       },
        { "no-span-check",                     lazy_ring_mutant::NO_SPAN_CHECK                 },
        { "no-alignment-check",                lazy_ring_mutant::NO_ALIGNMENT_CHECK            },
        { "shutdown-no-wait",                  lazy_ring_mutant::SHUTDOWN_NO_WAIT              },
        { "worker-reap-destroys-pool",         lazy_ring_mutant::WORKER_REAP_DESTROYS_POOL     },
        { "pool-strict-abort-dropped",         lazy_ring_mutant::POOL_STRICT_ABORT_DROPPED     },
        { "pool-strict-abort-dropped-refused", lazy_ring_mutant::POOL_STRICT_ABORT_DROPPED     },
        { "pool-strict-abort-dropped-drop",    lazy_ring_mutant::POOL_STRICT_ABORT_DROPPED     },
        { "pool-strict-abort-dropped-parked",  lazy_ring_mutant::POOL_STRICT_ABORT_DROPPED     },
        { "ensure-witness-dropped",            lazy_ring_mutant::ENSURE_WITNESS_DROPPED        },
        { "ticket-wait-returns-at-once",       lazy_ring_mutant::TICKET_WAIT_RETURNS_AT_ONCE   },
        { "ticket-zero-no-notify",             lazy_ring_mutant::TICKET_ZERO_NO_NOTIFY         },
        { "gather-returns-unwaited",           lazy_ring_mutant::GATHER_RETURNS_UNWAITED       },
        { "gather-error-returns-unwaited",     lazy_ring_mutant::GATHER_ERROR_RETURNS_UNWAITED },
        { "submit-throw-unwoken",              lazy_ring_mutant::SUBMIT_THROW_UNWOKEN          },
    };
    for (const entry & e : table) {
        if (std::strcmp(e.text, text) == 0) {
            return e.mutant;
        }
    }
    std::fprintf(stderr, "unknown mutant %s\n", text);
    std::exit(2);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], k_pin_child) == 0) {
        const lazy_test_coverage_pin first  = ggml_sycl_test_make_coverage_pin(k_need);
        const lazy_test_coverage_pin second = ggml_sycl_test_make_coverage_pin(2 * k_need);
        std::fprintf(stderr, "PIN-CHILD returned\n");
        return 0;
    }
    ggml_log_set(capture_log, nullptr);
    if (argc == 2 && std::strcmp(argv[1], k_table_child) == 0) {
        return witness_table_child();
    }
    if (argc == 3 && std::strcmp(argv[1], k_holding_child) == 0) {
        return reap_holding_child(std::atoi(argv[2]));
    }
    if (argc == 2 && std::strcmp(argv[1], k_drop_child) == 0) {
        return drop_holding_child();
    }
    if (argc == 2 && std::strcmp(argv[1], k_throw_child) == 0) {
        return reap_throw_child();
    }
    if ((argc == 3 || argc == 4) && std::strcmp(argv[1], k_strict_child) == 0) {
        const lazy_ring_mutant mutant = argc == 4 ? mutant_named(argv[3]) : lazy_ring_mutant::NONE;
        ggml_sycl_test_lazy_ring_mutant_select(mutant);
        strict_child(argv[2], mutant);
    }
    if ((argc == 3 || argc == 4) && std::strcmp(argv[1], k_ensure_child) == 0) {
        ggml_sycl_test_lazy_ring_mutant_select(argc == 4 ? mutant_named(argv[3]) : lazy_ring_mutant::NONE);
        ensure_holding_child(std::strcmp(argv[2], "held") == 0);
    }
    if (argc == 3 && std::strcmp(argv[1], "--mutant") == 0) {
        g_mutant = argv[2];
        ggml_sycl_test_lazy_ring_mutant_select(mutant_named(argv[2]));
    }
    const std::vector<named_arm> arms = {
        { "ensure allocates the need once and refuses what the plan did not charge",    arm_ensure_rules             },
        { "a creating ensure refuses a refused, short or mischarged ring and frees it", arm_ensure_creation_refusals },
        { "a need that is not whole blocks per slot refuses before ensure reserves",    arm_ensure_unaligned_need    },
        { "an ensure with no live pool refuses as unpinned",                            arm_ensure_unpinned          },
        { "a concurrent ensure refuses at once while the reserver is held",             arm_ensure_concurrent        },
        { "a throwing allocation leaves ensuring clear for the next ensure",            arm_ensure_throw             },
        { "an ensure entered holding a witnessed lock aborts naming it",                arm_ensure_held_lock         },
        { "a worker's read lands in its waiter's slot",                                 arm_engine_read_lands        },
        { "a failed read reaches the ticket with one WARN",                             arm_engine_read_error        },
        { "a read's last drop on a worker frees only the ring",                         arm_engine_worker_reap       },
        { "a worker's last drop leaves its pool's join to the next FULL reap",          arm_engine_worker_drop       },
        { "shutdown cancels queued reads and waits for picked ones",                    arm_engine_shutdown          },
        { "a second coverage pin joins the live ring and a mismatched one aborts",      arm_coverage_pin_joins       },
        { "a witnessed lock's depth follows lock, try_lock and unlock",                 arm_witness_depth            },
        { "a unique_lock relock and a condition_variable_any wait keep the depth",      arm_witness_relock           },
        { "a ninth witnessed lock held at once aborts naming it",                       arm_witness_table_full       },
        { "a FULL reap with no witnessed lock held runs and refuses nothing",           arm_reap_unheld              },
        { "a reap under the registry's mutex or the queue lock refuses naming it",      arm_reap_refuses_held_lock   },
        { "a lease dropped under a witnessed lock is logged, or aborts",                arm_drop_witness             },
        { "a reap whose step throws logs it and returns",                               arm_reap_throw               },
        { "a reap refused under a held lock aborts under strict leases",                arm_strict_refused           },
        { "a lease dropped under a held lock aborts under strict leases",               arm_strict_dropped           },
        { "a reap finding a parked pool with a lease aborts under strict leases",       arm_strict_parked            },
        { "a reap whose first step throws aborts under strict leases",                  arm_strict_failed            },
        { "a ticket's wait returns when its reads are delivered",                       arm_ticket_wait              },
        { "a gather lands its rows from the row cache and the file",                    arm_gather_rows              },
        { "a gather waits for the reads it submitted",                                  arm_gather_waits             },
        { "a gather whose submit throws still waits for the reads it queued",           arm_gather_throw_waits       },
        // Last: the shut-down flag is never cleared.
        { "ensure refuses after shut-down and frees the reserver's ring",               arm_ensure_shut_down         },
    };
    return run_arms(arms, "lazy io engine host arms");
}
