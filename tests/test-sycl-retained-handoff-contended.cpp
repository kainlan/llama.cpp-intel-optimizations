// Concurrent retained-handle ownership barrier (llama.cpp-fbj5, plan Task 12c).
//
// This is the device-free reproducer from artifacts/oze0/repro-drain-barrier.cpp
// with its POLARITY FLIPPED.  When it was written the finding was a defect: a
// thread that owned an allocation but had not YET handed it to
// retain_handles_until_event() was in neither state.queue nor state.active, so
// drain_retained_handles(true) reported "all clear" while that owner was live.
// The measurement was 12/12 runs, 79-99% of boundary crossings
// (artifacts/oze0/repro-drain-barrier-output.txt).
//
// 1f84f1a95 closed that window with a publisher ticket: begin_retained_handle_publish()
// increments state.publishers under the state mutex before the owner touches its
// allocation, the ticket is moved into retain_handles_until_event(), and only that
// call's parameter destructor decrements it -- AFTER the handles are already in
// state.queue.  drain_retained_handles(true) then waits on
//
//     state.queue.empty() && state.active == 0 && state.publishers == 0
//
// so there is no instant at which an owner is invisible to the barrier.
//
// This test asserts that contract under the old reproducer's concurrent shape.
// Its sibling test-sycl-retained-handoff-barrier covers the ticket's API contract
// single-threaded (a publisher blocks the drain, scope exit unblocks it); what it
// cannot show is that the handoff has no GAP, because a gap is only observable
// from another thread while the owner is mid-hold.
//
// GUARD UNDER TEST: the `state.publishers == 0` clause of the drain predicate
// (mem-handle.cpp, drain_retained_handles()).  Deleting that one clause restores
// the exact pre-1f84f1a95 predicate and must turn this test red.  See
// artifacts/fbj5/guard-mutation.diff and artifacts/fbj5/README.md.
//
// WHY THE RESULT IS NOT A COIN FLIP.  The owner takes its ticket before it
// publishes the hold, and drops it only inside retain_handles_until_event(),
// which it calls after it has left the hold.  So for a drain to report clear
// while an owner's hold state is BIT-IDENTICAL before and after that drain,
// state.publishers must have reached 0 during a window in which that owner's
// ticket was continuously live -- impossible while the clause is present.  The
// pass is therefore structural, not timing-dependent; only the RED side depends
// on timing, and only to the extent that it needs the contended window to occur
// at all.
//
// TIMING, AND THE LESSON FROM oze0.  kHoldWindow is 50 us because a real owner
// holds scratch across a stream_dma submission.  An earlier variant of the
// original reproducer used a 200-cycle hold -- far shorter than an honest
// submission -- and reproduced in only 1 of 5 runs, i.e. the too-short window
// flattered the barrier.  Do not shrink it to make this test faster.
//
// SELF-CHECK AGAINST A VACUOUS PASS.  A green run must not merely be a run in
// which nothing happened.  Each scenario counts CROSSINGS: drains that began
// while an owner was demonstrably mid-hold and went on to report clear.  Those
// are exactly the events that are violations when the guard is absent, so a
// scenario that cannot reach kMinCrossings of them has not tested anything and
// fails rather than passing.
//
// Device-free by construction: DIRECT handle views over stack buffers and
// default-constructed (already complete) events -- no device, queue, unified
// cache or model.  There is consequently no legitimate skip path; a run that
// cannot make progress fails loudly instead of exiting 77.

#include "mem-handle.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr int      kMaxOwners       = 3;
constexpr auto     kHoldWindow      = std::chrono::microseconds(50);
// Longer than the hold so that all owners are simultaneously idle often enough
// for the boundary to observe a legitimate clear.  With three owners at a 1:1
// duty cycle those moments get scarce, and a boundary that only ever times out
// would fail the crossing self-check below instead of measuring anything.
constexpr auto     kIdleWindow      = std::chrono::microseconds(100);
constexpr uint32_t kDrainTimeoutMs  = 50;
constexpr long     kMinCrossings    = 50;
constexpr auto     kScenarioTimeout = std::chrono::seconds(20);

// One cache line per owner: the boundary polls these in a tight loop and false
// sharing would perturb the very window being measured.
struct alignas(64) owner_slot {
    // (iteration << 1) | holding.  The low bit says whether this owner is inside
    // its hold; the rest makes consecutive holds distinguishable, so the boundary
    // can tell "still the same hold" from "a later hold that happens to look the
    // same".  Without that, an owner cycling every ~100 us aliases against a
    // blocking drain and manufactures false violations.
    std::atomic<uint64_t> hold_state{ 0 };
};

int fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

void cleanup() {
    ggml_sycl::drain_retained_handles(true, 5000);
    ggml_sycl::release_graph_retained_handles();
    ggml_sycl::drain_retained_handles(true, 5000);
}

// n_owners concurrent owners against one boundary (this thread).  Mirrors
// repro-drain-barrier-multi.cpp's owner sweep: the number of holders a boundary
// can find is a distribution bounded by the number of concurrent contexts, so
// the contract has to hold for each of 1, 2 and 3.
int run_scenario(int n_owners) {
    cleanup();

    alignas(64) std::uint8_t           buffers[kMaxOwners][64] = {};
    std::array<owner_slot, kMaxOwners> slots;
    std::atomic<bool>                  stop{ false };
    std::atomic<int>                   started{ 0 };

    std::vector<std::thread> owners;
    owners.reserve(n_owners);
    for (int k = 0; k < n_owners; ++k) {
        owners.emplace_back([&, k]() {
            started.fetch_add(1, std::memory_order_release);
            uint64_t iteration = 1;
            while (!stop.load(std::memory_order_acquire)) {
                // Production shape, getrows.cpp:233-248: allocate, resolve, THEN
                // take the ticket.  The ticket covers "this thread owns an
                // allocation" through "the handle is in the drain queue".
                auto ticket = ggml_sycl::begin_retained_handle_publish();

                ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_direct(buffers[k], GGML_LAYOUT_AOS, false);

                slots[k].hold_state.store((iteration << 1) | 1u, std::memory_order_release);

                // Stand-in for the window between acquiring scratch and submitting
                // the work that consumes it.
                std::this_thread::sleep_for(kHoldWindow);

                std::vector<ggml_sycl::mem_handle> retained;
                retained.push_back(std::move(h));

                // Leave the hold BEFORE publishing, never after.  The interval
                // [publish, ticket released] is one the boundary is allowed to see
                // as clear once the handle is queued, so counting it as a hold
                // would make green runs flaky for a reason that is not a defect.
                slots[k].hold_state.store(iteration << 1, std::memory_order_release);

                ggml_sycl::retain_handles_until_event(std::move(retained), sycl::event{}, std::move(ticket));

                ++iteration;
                std::this_thread::sleep_for(kIdleWindow);
            }
        });
    }

    while (started.load(std::memory_order_acquire) < n_owners) {
        std::this_thread::yield();
    }

    long     crossings       = 0;  // drains that began mid-hold and reported clear
    long     contended       = 0;  // drains that began mid-hold, cleared or not
    long     clears          = 0;  // drains that reported clear, contended or not
    long     violations      = 0;
    int      violating_owner = -1;
    uint64_t violating_state = 0;

    uint64_t   before[kMaxOwners] = {};
    const auto deadline           = std::chrono::steady_clock::now() + kScenarioTimeout;

    while (crossings < kMinCrossings && violations == 0 && std::chrono::steady_clock::now() < deadline) {
        bool any_holding = false;
        for (int k = 0; k < n_owners; ++k) {
            before[k] = slots[k].hold_state.load(std::memory_order_acquire);
            if (before[k] & 1u) {
                any_holding = true;
            }
        }
        if (any_holding) {
            ++contended;
        }

        // This is the graph boundary's question: is every retained-handle owner
        // accounted for, so that reclaiming the zone behind it is sound?
        if (!ggml_sycl::drain_retained_handles(true, kDrainTimeoutMs)) {
            continue;  // refused to report clear -- the correct answer mid-hold
        }
        ++clears;

        for (int k = 0; k < n_owners; ++k) {
            if ((before[k] & 1u) == 0) {
                continue;
            }
            // Same hold, bit for bit: the owner never left it, so the drain
            // reported clear while that owner held an unpublished allocation.
            if (slots[k].hold_state.load(std::memory_order_acquire) == before[k]) {
                ++violations;
                violating_owner = k;
                violating_state = before[k];
            } else {
                ++crossings;
            }
        }
    }

    stop.store(true, std::memory_order_release);
    for (auto & t : owners) {
        t.join();
    }

    const size_t graph_parked = ggml_sycl::graph_retained_handle_count();

    std::printf("owners=%d crossings=%ld contended=%ld clears=%ld violations=%ld graph_parked=%zu\n", n_owners,
                crossings, contended, clears, violations, graph_parked);

    if (violations > 0) {
        std::fprintf(stderr,
                     "FAIL: retained-handle barrier violated: drain reported clear while owner %d held an "
                     "unpublished handle (owners=%d hold_state=%llu crossings=%ld)\n",
                     violating_owner, n_owners, (unsigned long long) violating_state, crossings);
        cleanup();
        return 1;
    }

    // The three checks below are the positive controls.  Each one, if it fires,
    // means a green verdict would have been about nothing.
    if (crossings < kMinCrossings) {
        std::fprintf(stderr,
                     "FAIL: contended window never exercised: %ld/%ld crossings in %llds "
                     "(owners=%d contended=%ld clears=%ld) -- this run proves NOTHING about the barrier\n",
                     crossings, kMinCrossings, (long long) kScenarioTimeout.count(), n_owners, contended, clears);
        cleanup();
        return 1;
    }
    if (contended == 0 || clears == 0) {
        cleanup();
        return fail("boundary never both contended and cleared -- the violation check could not have fired");
    }
    if (graph_parked != 0) {
        // Handles routed to the command-graph list are deliberately invisible to
        // drain_retained_handles(), so a run that parked any of them measured the
        // wrong path.  Non-recording threads must never land there (llama.cpp-oze0).
        cleanup();
        return fail("handles were parked for command-graph lifetime; the event-bound path was not the one tested");
    }

    std::printf("PASS: owners=%d -- drain never reported clear across %ld contended crossings\n", n_owners, crossings);
    cleanup();
    return 0;
}

}  // namespace

int main() {
    for (int n_owners = 1; n_owners <= kMaxOwners; ++n_owners) {
        if (int rc = run_scenario(n_owners)) {
            return rc;
        }
    }
    std::puts("PASS: retained-handle publisher barrier holds against concurrent owners");
    return 0;
}
