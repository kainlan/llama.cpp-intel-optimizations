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
// hands. Nothing here touches a GPU (the registration pins the selector to
// the OpenCL CPU device).
//
// Usage:
//   ./build/bin/test-sycl-retained-reap                 # every case
//   ./build/bin/test-sycl-retained-reap strict-child    # the STRICT abort child

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

release_attempt count_release(const alloc_metadata &, void *) noexcept {
    g_released.fetch_add(1);
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

    retained_reap_result r;
    std::thread          reaper([&] { r = reap({ a }, RETAINED_REAP_QUERY_EVENT_STATUS); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let the reaper reach the yield
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    reaper.join();
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
// mode and dropped, as unwaitable, in COMPLETE mode.
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

    r = reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    CHECK_EQ(r.entries_dropped, 1, "parked-complete: dropped");
    CHECK_EQ(r.unwaitable_dropped, 1, "parked-complete: as unwaitable");
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

    retained_reap_result r;
    std::thread          reaper([&] { r = reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    retained_drain_test_hold(RETAINED_DRAIN_TEST_POINT_NONE);
    reaper.join();
    CHECK_EQ(r.in_hand_yields, 1, "yield-parked: yielded");
    CHECK_EQ(r.unwaitable_dropped, 1, "yield-parked: found parked after the yield");
    CHECK(only_reference(a), "yield-parked: a is free");
    (void) drain_retained_handles(true);
}

// COMPLETE with an event the caller did not finish: the backstop waits for it
// instead of freeing early, and reports the plan bug.
void test_backstop(sycl::queue & q) {
    mem_handle   blocker = hold_worker();
    mem_handle   a       = make_owner(800);
    gate         g(q);
    const size_t before = retained_reap_backstop_incomplete();
    retain_handles_until_event({ a }, g.event);
    std::thread                opener([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        g.open();
    });
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
    std::thread opener([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        g.open();
    });
    opener.detach();
    (void) reap({ a }, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);
    std::printf("strict-child: returned without aborting\n");
    return 0;
}

void test_strict_aborts(const char * self) {
    const std::string cmd = std::string("GGML_SYCL_STRICT_PLAN=1 ") + self + " strict-child 2>&1";
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
    CHECK(out.find("[CONTEXT-PLAN-BUG] retained-reap backstop") != std::string::npos,
          "strict: the child printed the plan-bug line");
    CHECK(out.find("returned without aborting") == std::string::npos, "strict: the child aborted in the reap");
}

}  // namespace

int main(int argc, char ** argv) {
    sycl::queue q{ sycl::cpu_selector_v };
    if (argc > 1 && std::strcmp(argv[1], "strict-child") == 0) {
        return strict_child(q);
    }
    test_queued_query(q);
    test_shared_and_non_owner();
    test_in_hand_yield();
    test_in_hand_incomplete(q);
    test_parked();
    test_yield_then_parked();
    test_backstop(q);
    test_strict_aborts(argv[0]);
    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-retained-reap: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-retained-reap: all ok\n");
    return 0;
}
