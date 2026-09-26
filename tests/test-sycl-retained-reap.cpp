// Host test for release_retained_referencing(): the reap that drops the
// retained handles (retain_handles_until_event()) still holding a reference to
// an allocation owner, so a caller that retires an allocation can have its
// bytes back without waiting for the drain worker to get to them.
//
// Owners are real allocation owner controls minted by the private fixture
// factory, with an injected release backend in place of a physical free, so
// "released" below is the owner's last reference actually going. Incomplete
// events come from a host_task on the CPU SYCL device, held on a gate the test
// opens; the drain worker is paused at its test points to put a record in its
// hands, and a reap on another thread is known to be waiting on it when
// retained_reap_test_yielding() says so -- no case orders threads by sleeping.
// Every owner release checks that its thread does not hold the retained-store
// mutex. Nothing here touches a GPU (the registration pins the selector to
// the OpenCL CPU device).
//
// Usage:
//   ./build/bin/test-sycl-retained-reap                          # every case
//   ./build/bin/test-sycl-retained-reap strict-child             # STRICT backstop child
//   ./build/bin/test-sycl-retained-reap strict-ownerless-child   # STRICT ownerless child

#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

using namespace ggml_sycl;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

#define CHECK_EQ(got, want, msg)                                                                                       \
    do {                                                                                                               \
        const long long got_v_  = (long long) (got);                                                                   \
        const long long want_v_ = (long long) (want);                                                                  \
        if (got_v_ != want_v_) {                                                                                       \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, got_v_, want_v_); \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

std::atomic<uint64_t> g_next_id{ 1000 };
std::atomic<int>      g_released{ 0 };
std::atomic<int>      g_released_under_store_mutex{ 0 };

release_attempt count_release(const alloc_metadata &, void *) noexcept {
    g_released.fetch_add(1);
    if (retained_store_mutex_held()) {
        g_released_under_store_mutex.fetch_add(1);
    }
    return { release_attempt_status::RELEASED };
}

// An owner handle over a fresh owner control of `size` bytes.
mem_handle make_owner(size_t size) {
    static std::vector<std::vector<unsigned char>> backing;
    backing.emplace_back(size);
    const uint64_t id = g_next_id.fetch_add(1);
    alloc_metadata metadata{};
    metadata.ptr      = backing.back().data();
    metadata.size     = size;
    metadata.device   = 0;
    metadata.id       = id;
    metadata.alloc_id = id;
    metadata.epoch_id = 1;
    metadata.tier     = alloc_tier::HOST_PINNED;
    auto fixture      = allocation_owner_test_create(metadata, count_release, nullptr);
    if (!fixture.result) {
        std::fprintf(stderr, "FAIL: allocation_owner_test_create failed\n");
        std::exit(1);
    }
    return mem_handle::from_owned_alloc(std::move(fixture.result.owner));
}

// A handle with no owner control: an ownerless DIRECT view of 64 host bytes.
constexpr size_t OWNERLESS_BYTES = 64;

mem_handle make_ownerless() {
    static unsigned char buf[OWNERLESS_BYTES];
    return mem_handle::from_direct(buf, GGML_LAYOUT_AOS, false, mem_handle::HOST_DEVICE, sizeof(buf));
}

// Whether the test's handle is now the owner's last reference: resetting it
// releases the allocation only then.
bool only_reference(mem_handle & owner) {
    return owner.reset_owned_allocation().released();
}

// A host_task on the CPU device that runs until open() is called.
struct gate {
    sycl::queue &     queue;
    std::atomic<bool> is_open{ false };
    sycl::event       event;

    explicit gate(sycl::queue & q) : queue(q) {
        event = queue.submit([this](sycl::handler & h) {
            h.host_task([this] {
                while (!is_open.load()) {
                    std::this_thread::yield();
                }
            });
        });
    }

    void open() {
        is_open.store(true);
        event.wait();
    }

    ~gate() {
        if (!is_open.load()) {
            open();
        }
    }
};

void wait_parked(bool parked) {
    while (retained_drain_test_parked() != parked) {
        std::this_thread::yield();
    }
}

// Hold the worker at its pop on a blocker record of its own, so records
// published after this stay queued until release_worker().
mem_handle hold_worker() {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle blocker = make_owner(64);
    retain_handles_until_event({ blocker }, sycl::event{});
    wait_parked(true);
    return blocker;
}

void release_worker() {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    wait_parked(false);
    (void) drain_retained_handles(true);
}

// A reap on its own thread. yields() waits until it is waiting on the drain
// worker's in-hand record (true) or has returned without (false).
struct reaper {
    retained_reap_result r;
    std::atomic<bool>    done{ false };
    std::thread          thread;

    template <typename F>
    explicit reaper(F f) :
        thread([this, f] {
            r = f();
            done.store(true);
        }) {}

    bool yields() const {
        for (;;) {
            if (retained_reap_test_yielding() > 0) {
                return true;
            }
            if (done.load()) {
                return false;
            }
            std::this_thread::yield();
        }
    }

    // Whether the reap returns within `limit`: a bound on a wait for an
    // outcome, not an ordering.
    bool returns_within(std::chrono::milliseconds limit) const {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        while (!done.load()) {
            if (std::chrono::steady_clock::now() > deadline) {
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    }

    retained_reap_result join() {
        thread.join();
        return r;
    }
};

// Opens `g` once the backstop counter has moved past `before`, so the reap
// cannot have seen its event complete. If it never moves (a reap that skips
// the backstop), the gate opens after 10 s so the case fails instead of
// hanging.
std::thread open_after_backstop(gate & g, size_t before) {
    return std::thread([&g, before] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (retained_reap_backstop_incomplete() == before && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        g.open();
    });
}

retained_reap_result reap(const std::vector<mem_handle> & owners,
                          retained_reap_precondition      pre,
                          bool *                          pending = nullptr) {
    retained_reap_request request;
    request.owners        = owners.data();
    request.n_owners      = owners.size();
    request.pre           = pre;
    request.reason        = "test";
    request.owner_pending = pending;
    return release_retained_referencing(request);
}

// A queued record whose event is complete is dropped in QUERY mode; one whose
// event is not is kept and reported pending, its owner marked, its bytes once.
void test_queued_query(sycl::queue & q) {
    mem_handle blocker = hold_worker();
    mem_handle a       = make_owner(1000);
    mem_handle b       = make_owner(3000);
    gate       g(q);
    retain_handles_until_event({ a }, sycl::event{});
    retain_handles_until_event({ b, b.slice(0, 1000) }, g.event);

    bool                       pending[2] = { true, true };
    const retained_reap_result r          = reap({ a, b }, RETAINED_REAP_QUERY_EVENT_STATUS, pending);
    CHECK_EQ(r.entries_dropped, 1, "queued-query: the complete record is dropped");
    CHECK_EQ(r.entries_pending, 1, "queued-query: the incomplete record is one entry pending");
    CHECK_EQ(r.pending_bytes, 3000, "queued-query: its owner's bytes, once for two handles of it");
    CHECK_EQ(r.in_hand_yields, 0, "queued-query: nothing in hand matched");
    CHECK(!pending[0] && pending[1], "queued-query: only b's owner is pending");
    CHECK(only_reference(a), "queued-query: a's last reference was the test's");

    g.open();
    release_worker();
    CHECK(only_reference(b), "queued-query: the kept record went with the worker once complete");
    CHECK(only_reference(blocker), "queued-query: blocker drained");
}

// A record naming two owners is one entry for either; a record naming neither
// is left alone.
void test_shared_and_non_owner() {
    mem_handle blocker = hold_worker();
    mem_handle a       = make_owner(100);
    mem_handle b       = make_owner(200);
    mem_handle c       = make_owner(300);
    retain_handles_until_event({ a, b }, sycl::event{});
    retain_handles_until_event({ c }, sycl::event{});

    const retained_reap_result r = reap({ b }, RETAINED_REAP_QUERY_EVENT_STATUS);
    CHECK_EQ(r.entries_dropped, 1, "shared: the record naming b is dropped, a with it");
    CHECK(only_reference(a), "shared: a's reference went with the record");
    CHECK(only_reference(b), "shared: b's too");
    CHECK(!only_reference(c), "non-owner: c's record is untouched");

    release_worker();
    CHECK(only_reference(blocker), "shared: blocker drained");
}

// The worker has popped a record whose event is complete and not dropped it
// yet (the race a caller that waited on the event itself still loses): the
// reap waits for the worker to finish with it rather than report it pending.
void test_in_hand_yield() {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle a = make_owner(500);
    retain_handles_until_event({ a }, sycl::event{});
    wait_parked(true);

    reaper     t([&] { return reap({ a }, RETAINED_REAP_QUERY_EVENT_STATUS); });
    const bool yielded = t.yields();
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    const retained_reap_result r = t.join();
    CHECK(yielded, "in-hand: the reap waited on the worker's record");
    CHECK_EQ(r.in_hand_yields, 1, "in-hand: the reap yielded to the worker");
    CHECK_EQ(r.entries_pending, 0, "in-hand: nothing pending");
    CHECK(only_reference(a), "in-hand: the worker's reference is gone when the reap returns");
    (void) drain_retained_handles(true);
}

// In hand with its event still running: QUERY does not wait on device work.
void test_in_hand_incomplete(sycl::queue & q) {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle a = make_owner(700);
    gate       g(q);
    retain_handles_until_event({ a }, g.event);
    wait_parked(true);
    bool                       pending = false;
    const retained_reap_result r       = reap({ a }, RETAINED_REAP_QUERY_EVENT_STATUS, &pending);
    CHECK_EQ(r.in_hand_yields, 0, "in-hand-incomplete: no yield");
    CHECK_EQ(r.entries_pending, 1, "in-hand-incomplete: pending");
    CHECK_EQ(r.pending_bytes, 700, "in-hand-incomplete: its bytes");
    CHECK(pending, "in-hand-incomplete: owner marked");
    g.open();
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    (void) drain_retained_handles(true);
    CHECK(only_reference(a), "in-hand-incomplete: the worker dropped it once complete");
}

// A handle the worker parked for command-graph lifetime is pending in QUERY
// mode and dropped, as unwaitable, in COMPLETE mode. COMPLETE takes only the
// parked handles its owners name: it never releases graph_unwaitable as a
// whole (release_graph_retained_handles()).
void test_parked() {
    mem_handle a = make_owner(900);
    mem_handle b = make_owner(50);
    retained_drain_test_fail_next_wait_as_command_graph();
    retain_handles_until_event({ a, b }, sycl::event{});
    (void) drain_retained_handles(true);
    CHECK(graph_retained_handle_count() >= 2, "parked: both handles in graph_unwaitable");

    bool                 pending = false;
    retained_reap_result r       = reap({ a }, RETAINED_REAP_QUERY_EVENT_STATUS, &pending);
    CHECK_EQ(r.entries_pending, 1, "parked-query: one handle pending");
    CHECK_EQ(r.entries_dropped, 0, "parked-query: never dropped");
    CHECK(pending, "parked-query: owner marked");
    CHECK_EQ(r.pending_bytes, 900, "parked-query: its bytes");

    const size_t parked = graph_retained_handle_count();
    r                   = reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    CHECK_EQ(r.entries_dropped, 1, "parked-complete: dropped");
    CHECK_EQ(r.unwaitable_dropped, 1, "parked-complete: as unwaitable");
    CHECK_EQ(parked - graph_retained_handle_count(), 1, "parked-complete: only a's handle left graph_unwaitable");
    CHECK(only_reference(a), "parked-complete: a is free");
    CHECK(!only_reference(b), "parked-complete: b, not named, stays parked");
    release_graph_retained_handles();
}

// The worker throws "command graph" on a record the reap is yielding to: the
// reap finds it in graph_unwaitable after the yield.
void test_yield_then_parked() {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle a = make_owner(400);
    retained_drain_test_fail_next_wait_as_command_graph();
    retain_handles_until_event({ a }, sycl::event{});
    wait_parked(true);

    reaper     t([&] { return reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER); });
    const bool yielded = t.yields();
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    const retained_reap_result r = t.join();
    CHECK(yielded, "yield-parked: the reap waited on the worker's record");
    CHECK_EQ(r.in_hand_yields, 1, "yield-parked: yielded");
    CHECK_EQ(r.unwaitable_dropped, 1, "yield-parked: found parked after the yield");
    CHECK(only_reference(a), "yield-parked: a is free");
    (void) drain_retained_handles(true);
}

// The worker has dropped its in-hand record's handles and not yet cleared the
// record (its AFTER_DROP point): the record is still in hand, so a reap still
// waits on it. It is cleared only after the drop, never before, so a reap
// that finds nothing in hand finds the handles gone.
void test_mid_drop() {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_DROP);
    mem_handle a = make_owner(600);
    retain_handles_until_event({ a }, sycl::event{});
    wait_parked(true);

    reaper     t([&] { return reap({ a }, RETAINED_REAP_QUERY_EVENT_STATUS); });
    const bool yielded = t.yields();
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    const retained_reap_result r = t.join();
    CHECK(yielded, "mid-drop: a dropped record is still in hand until the worker clears it");
    CHECK_EQ(r.in_hand_yields, 1, "mid-drop: the reap yielded to it");
    CHECK(only_reference(a), "mid-drop: a is free");
    (void) drain_retained_handles(true);
}

// Two owners: b's record is in the worker's hands on a running event, a's is
// queued and complete. QUERY drops a's and keeps b's, marking only b.
void test_two_owners_one_in_hand(sycl::queue & q) {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle a = make_owner(1100);
    mem_handle b = make_owner(2200);
    gate       g(q);
    retain_handles_until_event({ b }, g.event);
    wait_parked(true);
    retain_handles_until_event({ a }, sycl::event{});

    bool                       pending[2] = { true, false };
    const retained_reap_result r          = reap({ a, b }, RETAINED_REAP_QUERY_EVENT_STATUS, pending);
    CHECK(!pending[0] && pending[1], "two-owner in-hand: owner_pending is {false, true}");
    CHECK_EQ(r.entries_dropped, 1, "two-owner in-hand: a's queued record is dropped");
    CHECK_EQ(r.entries_pending, 1, "two-owner in-hand: b's in-hand record is pending");
    CHECK_EQ(r.pending_bytes, 2200, "two-owner in-hand: b's bytes");
    CHECK_EQ(r.in_hand_yields, 0, "two-owner in-hand: no yield on a running event");
    CHECK(only_reference(a), "two-owner in-hand: a is free");

    g.open();
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    (void) drain_retained_handles(true);
    CHECK(only_reference(b), "two-owner in-hand: b went with the worker once complete");
}

// A kept record naming two owners marks both, and counts both owners' bytes.
void test_kept_shared(sycl::queue & q) {
    mem_handle blocker = hold_worker();
    mem_handle a       = make_owner(1300);
    mem_handle b       = make_owner(2600);
    gate       g(q);
    retain_handles_until_event({ a, b }, g.event);

    bool                       pending[2] = { false, false };
    const retained_reap_result r          = reap({ a, b }, RETAINED_REAP_QUERY_EVENT_STATUS, pending);
    CHECK(pending[0] && pending[1], "kept-shared: both owners of the kept record are marked");
    CHECK_EQ(r.entries_pending, 1, "kept-shared: one entry");
    CHECK_EQ(r.pending_bytes, 3900, "kept-shared: both owners' bytes");
    CHECK_EQ(r.entries_dropped, 0, "kept-shared: nothing dropped");

    g.open();
    release_worker();
    CHECK(only_reference(a), "kept-shared: a went with the worker once complete");
    CHECK(only_reference(b), "kept-shared: b too");
    CHECK(only_reference(blocker), "kept-shared: blocker drained");
}

// In hand with an event whose status query throws: an event-bound record not
// known complete, so QUERY keeps it pending and does not wait on the worker.
void test_in_hand_unqueryable() {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle a = make_owner(750);
    retain_handles_until_event({ a }, sycl::event{});
    wait_parked(true);

    retained_reap_test_fail_next_query();
    bool       pending = false;
    reaper     t([&] { return reap({ a }, RETAINED_REAP_QUERY_EVENT_STATUS, &pending); });
    const bool yielded = t.yields();
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    const retained_reap_result r = t.join();
    CHECK(!yielded, "in-hand-unqueryable: QUERY does not wait on a record it cannot query complete");
    CHECK_EQ(r.in_hand_yields, 0, "in-hand-unqueryable: no yield");
    CHECK_EQ(r.entries_pending, 1, "in-hand-unqueryable: pending");
    CHECK_EQ(r.pending_bytes, 750, "in-hand-unqueryable: its bytes");
    CHECK(pending, "in-hand-unqueryable: owner marked");
    (void) drain_retained_handles(true);
    CHECK(only_reference(a), "in-hand-unqueryable: the worker dropped it");
}

// An owner with no owner control cannot be matched, so it is never reported
// clean: it is marked pending with its size, beside an owner that is clean.
void test_ownerless_owner() {
    mem_handle a         = make_owner(1500);
    mem_handle ownerless = make_ownerless();
    CHECK_EQ(ownerless.owner_control_id(), 0, "ownerless: the handle has no owner control");
    retain_handles_until_event({ a }, sycl::event{});
    (void) drain_retained_handles(true);

    bool                       pending[2] = { true, false };
    const retained_reap_result r          = reap({ a, ownerless }, RETAINED_REAP_QUERY_EVENT_STATUS, pending);
    CHECK(!pending[0] && pending[1], "ownerless: owner_pending is {false, true}");
    CHECK_EQ(r.pending_bytes, OWNERLESS_BYTES, "ownerless: its size is pending");
    CHECK_EQ(r.entries_pending, 0, "ownerless: no entry is pending");

    bool                       alone   = false;
    const retained_reap_result r_alone = reap({ ownerless }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER, &alone);
    CHECK(alone, "ownerless: alone, it is still not clean");
    CHECK_EQ(r_alone.pending_bytes, OWNERLESS_BYTES, "ownerless: alone, its size is pending");
    CHECK(only_reference(a), "ownerless: a is free");
}

// R11c: which entries go unwaited is fixed at retention. A queued record is
// event-bound, so a status query that throws on it means "not known
// complete": COMPLETE reports the backstop and waits for it, and does not
// drop it as unwaitable. The positive control is a graph-lifetime entry -- a
// handle the worker parked -- which COMPLETE drops unwaited, with no backstop.
void test_query_throw_is_incomplete(sycl::queue & q) {
    mem_handle   blocker = hold_worker();
    mem_handle   a       = make_owner(820);
    gate         g(q);
    const size_t before = retained_reap_backstop_incomplete();
    retain_handles_until_event({ a }, g.event);
    retained_reap_test_fail_next_query();
    std::thread                opener         = open_after_backstop(g, before);
    const retained_reap_result r              = reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    const bool                 open_at_return = g.is_open.load();
    opener.join();
    CHECK(open_at_return, "query-throws: the reap waited for the record's event");
    CHECK_EQ(retained_reap_backstop_incomplete() - before, 1, "query-throws: the backstop is reported");
    CHECK_EQ(r.unwaitable_dropped, 0, "query-throws: an event-bound record is not dropped as unwaitable");
    CHECK_EQ(r.entries_dropped, 1, "query-throws: dropped after the wait");
    CHECK(only_reference(a), "query-throws: a is free");
    release_worker();
    CHECK(only_reference(blocker), "query-throws: blocker drained");

    mem_handle c = make_owner(410);
    retained_drain_test_fail_next_wait_as_command_graph();
    retain_handles_until_event({ c }, sycl::event{});
    (void) drain_retained_handles(true);
    const size_t               control_before = retained_reap_backstop_incomplete();
    const retained_reap_result control        = reap({ c }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    CHECK_EQ(control.unwaitable_dropped, 1, "query-throws control: a parked handle is dropped unwaited");
    CHECK_EQ(retained_reap_backstop_incomplete() - control_before, 0, "query-throws control: no backstop");
    CHECK(only_reference(c), "query-throws control: c is free");
}

// A record whose wait fails is parked whatever the exception says: its
// handles are kept for graph lifetime, never freed early.
void test_wait_failure_parks() {
    mem_handle   a      = make_owner(450);
    const size_t before = graph_retained_handle_count();
    retained_drain_test_fail_next_wait(RETAINED_DRAIN_TEST_WAIT_DEVICE_ERROR);
    retain_handles_until_event({ a }, sycl::event{});
    (void) drain_retained_handles(true);
    CHECK_EQ(graph_retained_handle_count() - before, 1, "wait-failure: a failed wait parks the record's handle");
    const retained_reap_result r = reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    CHECK_EQ(r.unwaitable_dropped, 1, "wait-failure: COMPLETE drops it as graph-lifetime");
    CHECK(only_reference(a), "wait-failure: a is free after the reap");
}

// R11b: the in-hand backstop. The worker is inside its wait on a gated event,
// so the record is in hand and not complete. COMPLETE reports the backstop
// once and returns only after the gate opens and the worker drops it.
void test_in_hand_backstop(sycl::queue & q) {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle a = make_owner(850);
    gate       g(q);
    retain_handles_until_event({ a }, g.event);
    wait_parked(true);
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);  // on into its wait, still in hand

    const size_t               before         = retained_reap_backstop_incomplete();
    std::thread                opener         = open_after_backstop(g, before);
    const retained_reap_result r              = reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    const bool                 open_at_return = g.is_open.load();
    opener.join();
    CHECK(open_at_return, "in-hand-backstop: the reap returned only after the gate opened");
    CHECK_EQ(retained_reap_backstop_incomplete() - before, 1, "in-hand-backstop: reported once");
    CHECK_EQ(r.in_hand_yields, 1, "in-hand-backstop: it yielded to the worker");
    CHECK(only_reference(a), "in-hand-backstop: the worker's reference is gone at return");
    (void) drain_retained_handles(true);
}

// R13b: the yield is bounded by the in-hand record's sequence. The reap
// snapshots record A in hand and stops at REAPER_AFTER_SNAPSHOT; the worker
// drops A and pops an unrelated record B on a gated event. Let go, the reap
// returns while B is still in hand, never waiting on B's work.
void test_seq_bound(sycl::queue & q) {
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_AFTER_POP);
    mem_handle a = make_owner(310);
    mem_handle b = make_owner(620);
    gate       g(q);
    retain_handles_until_event({ a }, sycl::event{});
    wait_parked(true);
    retain_handles_until_event({ b }, g.event);

    retained_reap_test_hold_after_snapshot(true);
    reaper t([&] { return reap({ a }, RETAINED_REAP_QUERY_EVENT_STATUS); });
    while (!retained_reap_test_parked() && !t.done.load()) {
        std::this_thread::yield();
    }
    CHECK(retained_reap_test_parked(), "seq-bound: the reap stopped after its snapshot of A");
    const uint64_t parks = retained_drain_test_parks();
    retained_drain_test_release_once();  // the worker drops A and stops with B in hand
    while (retained_drain_test_parks() == parks) {
        std::this_thread::yield();
    }
    retained_reap_test_hold_after_snapshot(false);
    const bool returned = t.returns_within(std::chrono::seconds(10));
    CHECK(returned, "seq-bound: the reap returned while an unrelated record was in hand");

    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    g.open();
    const retained_reap_result r = t.join();
    CHECK_EQ(r.in_hand_yields, 1, "seq-bound: it yielded to A");
    CHECK(only_reference(a), "seq-bound: a is free");
    (void) drain_retained_handles(true);
    CHECK(only_reference(b), "seq-bound: b went with the worker once complete");
}

// COMPLETE with an event the caller did not finish: the backstop waits for it
// instead of freeing early, and reports the plan bug.
void test_backstop(sycl::queue & q) {
    mem_handle   blocker = hold_worker();
    mem_handle   a       = make_owner(800);
    gate         g(q);
    const size_t before = retained_reap_backstop_incomplete();
    retain_handles_until_event({ a }, g.event);
    std::thread                opener         = open_after_backstop(g, before);
    const retained_reap_result r              = reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    const bool                 open_at_return = g.is_open.load();
    opener.join();
    CHECK(open_at_return, "backstop: the reap returned only after the gate opened");
    CHECK_EQ(r.entries_dropped, 1, "backstop: dropped after the wait");
    CHECK_EQ(retained_reap_backstop_incomplete() - before, 1, "backstop: counted once");
    CHECK(only_reference(a), "backstop: a is free");
    release_worker();
    CHECK(only_reference(blocker), "backstop: blocker drained");
}

// Under GGML_SYCL_STRICT_PLAN=1 the same backstop aborts.
int strict_child(sycl::queue & q) {
    mem_handle blocker = hold_worker();
    mem_handle a       = make_owner(800);
    gate       g(q);
    retain_handles_until_event({ a }, g.event);
    // Not an ordering: the abort comes before any wait. Only a child that
    // fails to abort reaches the backstop wait, and this lets it finish.
    std::thread opener([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        g.open();
    });
    opener.detach();
    (void) reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    std::printf("strict-child: returned without aborting\n");
    return 0;
}

// Under GGML_SYCL_STRICT_PLAN=1 an owner with no owner control aborts.
int strict_ownerless_child() {
    (void) reap({ make_ownerless() }, RETAINED_REAP_QUERY_EVENT_STATUS);
    std::printf("strict-child: returned without aborting\n");
    return 0;
}

void test_strict_aborts(const char * self, const char * child, const char * line) {
    const std::string cmd = std::string("GGML_SYCL_STRICT_PLAN=1 ") + self + " " + child + " 2>&1";
    FILE *            p   = popen(cmd.c_str(), "r");
    CHECK(p != nullptr, "strict: child started");
    if (!p) {
        return;
    }
    std::string out;
    char        buf[512];
    while (std::fgets(buf, sizeof(buf), p)) {
        out += buf;
    }
    const int status = pclose(p);
    CHECK(status != 0, "strict: the child did not exit 0");
    CHECK(out.find(line) != std::string::npos, "strict: the child printed the plan-bug line");
    CHECK(out.find("returned without aborting") == std::string::npos, "strict: the child aborted in the reap");
    if (status == 0 || out.find(line) == std::string::npos) {
        std::fprintf(stderr, "strict: %s printed:\n%s", child, out.c_str());
    }
}

}  // namespace

int main(int argc, char ** argv) {
    sycl::queue q{ sycl::cpu_selector_v };
    if (argc > 1 && std::strcmp(argv[1], "strict-child") == 0) {
        return strict_child(q);
    }
    if (argc > 1 && std::strcmp(argv[1], "strict-ownerless-child") == 0) {
        return strict_ownerless_child();
    }
    test_queued_query(q);
    test_shared_and_non_owner();
    test_in_hand_yield();
    test_in_hand_incomplete(q);
    test_mid_drop();
    test_two_owners_one_in_hand(q);
    test_kept_shared(q);
    test_in_hand_unqueryable();
    test_ownerless_owner();
    test_parked();
    test_yield_then_parked();
    test_query_throw_is_incomplete(q);
    test_wait_failure_parks();
    test_in_hand_backstop(q);
    test_seq_bound(q);
    test_backstop(q);
    test_strict_aborts(argv[0], "strict-child", "[CONTEXT-PLAN-BUG] retained-reap backstop");
    test_strict_aborts(argv[0], "strict-ownerless-child", "has no owner control");
    // Every owner released above, by the worker, a reap or the test, was
    // released with the retained-store mutex not held by its thread.
    CHECK_EQ(g_released_under_store_mutex.load(), 0, "no owner is released under the retained-store mutex");
    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-retained-reap: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-retained-reap: all ok\n");
    return 0;
}
