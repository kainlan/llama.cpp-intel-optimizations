// The lazy tables' IO pool registry (docs/design/sycl-storage-tier-lazy-tables.md §3.2, §3.3): the counted lease, its
// clone, the park and the DYING transition at the last drop, the FULL reap and its list, the coverage wrapper and the
// helper guard, and the shut-down flag.
//
// Links ggml-sycl/lazy-io-pool.cpp alone, with ggml-base for the abort and the log: no SYCL header, no device, no
// unified cache, no SYCL module. The process holds one registry, so the arms run in order and each leaves it with no
// live pool and no parked node; the shut-down arm runs last because the flag is never cleared. The two abort paths run
// in a child process of this binary.
//
// Every arm names the mutant it kills. Every arm runs, so one failure does not hide the rest.

#include "../ggml/src/ggml-sycl/lazy-io-pool.hpp"
#include "../ggml/src/ggml-sycl/tests/host-ledger-test-harness.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern char ** environ;

using namespace ggml_sycl;
using namespace host_ledger_test_harness;

namespace {

constexpr std::chrono::seconds      k_hook_bound{ 10 };
constexpr std::chrono::milliseconds k_child_bound{ 30000 };

lazy_registry_state registry() {
    lazy_registry_state state;
    if (!ggml_sycl_lazy_registry_for_testing(&state)) {
        std::printf("FAIL: the registry accessor refused a valid pointer\n");
    }
    return state;
}

std::string describe(const lazy_registry_state & s) {
    return num(s.live_pools) + " live pools, " + num(s.live_leases) + " leases, " + num(s.parked_nodes) + " parked";
}

bool is(const lazy_registry_state & s, uint64_t pools, uint64_t leases, uint64_t parked) {
    return s.live_pools == pools && s.live_leases == leases && s.parked_nodes == parked;
}

// Frees whatever an arm left parked, so the next arm starts from an empty registry.
void settle() {
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
}

// Takes count and drops: a second take joins the live pool, a drop decrements, and the last drop parks the pool and
// empties the registry's slot. A held lease reads 0 ring bytes, since this pool has no ring.
// Mutants: a take that builds a pool while one is live (two pools); a drop that does not decrement; the last drop not
// parking (no parked node) or leaving the pool in the slot.
arm_result arm_take_and_drop() {
    arm_result r;
    ring_lease a = lazy_io_pool_take();
    expect(r, a.held(), "the first take returned an empty lease");
    expect(r, a.ring_bytes() == 0, "a held lease on a ring-less pool reads " + num(a.ring_bytes()) + " ring bytes");
    expect(r, is(registry(), 1, 1, 0), "after one take: " + describe(registry()) + ", want 1, 1, 0");
    ring_lease b = lazy_io_pool_take();
    expect(r, b.held(), "the second take returned an empty lease");
    expect(r, is(registry(), 1, 2, 0), "after a second take: " + describe(registry()) + ", want it joined: 1, 2, 0");
    a.reset();
    expect(r, !a.held(), "a reset lease still reads held");
    expect(r, is(registry(), 1, 1, 0), "after one drop of two: " + describe(registry()) + ", want 1, 1, 0");
    b.reset();
    expect(r, is(registry(), 0, 0, 1),
           "after the last drop: " + describe(registry()) + ", want the pool parked: 0, 0, 1");
    settle();
    expect(r, is(registry(), 0, 0, 0), "after a FULL reap: " + describe(registry()) + ", want 0, 0, 0");
    return r;
}

// A clone is another counted reference on the same pool, and a clone of an empty lease is empty; moves transfer a
// reference without counting, and a move-assignment drops the reference the target held.
// Mutants: a clone that does not count (the pool parks while the clone still holds it); a clone of an empty lease that
// takes a pool; a move that counts or a move-assignment that leaks the target's reference.
arm_result arm_clone_and_move() {
    arm_result r;
    ring_lease empty;
    ring_lease empty_clone = empty.clone();
    expect(r, !empty_clone.held(), "a clone of an empty lease is held");
    expect(r, is(registry(), 0, 0, 0), "a clone of an empty lease changed the registry: " + describe(registry()));

    ring_lease a     = lazy_io_pool_take();
    ring_lease clone = a.clone();
    expect(r, clone.held(), "a clone of a held lease is empty");
    expect(r, is(registry(), 1, 2, 0), "after a clone: " + describe(registry()) + ", want 1, 2, 0");
    a.reset();
    expect(r, is(registry(), 1, 1, 0),
           "the original dropped while its clone lives: " + describe(registry()) + ", want the pool kept: 1, 1, 0");

    ring_lease moved(std::move(clone));
    expect(r, moved.held() && !clone.held(), "a move did not transfer the reference");
    expect(r, is(registry(), 1, 1, 0), "a move changed the count: " + describe(registry()) + ", want 1, 1, 0");

    ring_lease target = lazy_io_pool_take();
    expect(r, is(registry(), 1, 2, 0), "a take beside a live lease: " + describe(registry()) + ", want 1, 2, 0");
    target = std::move(moved);
    expect(r, target.held() && !moved.held(), "a move-assignment did not transfer the reference");
    expect(r, is(registry(), 1, 1, 0),
           "a move-assignment over a held lease: " + describe(registry()) + ", want its reference dropped: 1, 1, 0");
    target.reset();
    expect(r, is(registry(), 0, 0, 1), "after the last drop: " + describe(registry()) + ", want 0, 0, 1");
    settle();
    return r;
}

// A take after the last drop builds a new pool and never revives the parked one; the reap list holds every parked
// pool, and one FULL reap frees them all. A FULL reap counts even when its list is empty.
// Mutants: a take reviving the DYING pool (the parked count falls without a reap); the reap freeing only the list's
// head; a reap of an empty list not counting.
arm_result arm_park_and_reap() {
    arm_result     r;
    const uint64_t c0 = registry().full_reaps_run;
    settle();
    expect(r, registry().full_reaps_run == c0 + 1,
           "a FULL reap of an empty list counted " + num(registry().full_reaps_run - c0) + " runs, want 1");

    ring_lease a = lazy_io_pool_take();
    a.reset();
    expect(r, is(registry(), 0, 0, 1), "after the first pool's last drop: " + describe(registry()) + ", want 0, 0, 1");
    ring_lease b = lazy_io_pool_take();
    expect(r, b.held(), "a take after the last drop returned an empty lease");
    expect(r, is(registry(), 1, 1, 1),
           "a take after the last drop: " + describe(registry()) + ", want a new pool beside the parked one: 1, 1, 1");
    b.reset();
    expect(r, is(registry(), 0, 0, 2), "after the second pool's last drop: " + describe(registry()) + ", want 0, 0, 2");
    const uint64_t c1 = registry().full_reaps_run;
    settle();
    expect(r, is(registry(), 0, 0, 0), "one FULL reap left " + describe(registry()) + ", want every node freed");
    expect(r, registry().full_reaps_run == c1 + 1, "the FULL reap did not count once");
    return r;
}

// Concurrent takes on an empty registry give one pool: a take that loses the install race joins the winner's pool and
// destroys its own. Several rounds, each released together from a spin barrier.
// Mutant: an install that does not check the generation (a second pool replaces the first, whose leases are lost from
// the slot's count, and two pools park at the drops).
arm_result arm_concurrent_takes() {
    arm_result    r;
    constexpr int threads = 16;
    constexpr int rounds  = 50;
    for (int round = 0; round < rounds && r.result == outcome::PASS; ++round) {
        std::vector<ring_lease>  leases(threads);
        std::vector<std::thread> workers;
        std::atomic<int>         ready{ 0 };
        std::atomic<bool>        go{ false };
        for (int i = 0; i < threads; ++i) {
            workers.emplace_back([&, i] {
                ready.fetch_add(1);
                while (!go.load()) {
                    std::this_thread::yield();
                }
                leases[i] = lazy_io_pool_take();
            });
        }
        while (ready.load() != threads) {
            std::this_thread::yield();
        }
        go.store(true);
        for (std::thread & t : workers) {
            t.join();
        }
        int held = 0;
        for (const ring_lease & l : leases) {
            held += l.held() ? 1 : 0;
        }
        expect(r, held == threads, "round " + num(round) + ": " + num(held) + " of " + num(threads) + " takes held");
        expect(r, is(registry(), 1, threads, 0),
               "round " + num(round) + ": " + describe(registry()) + ", want one pool with every lease");
        leases.clear();
        expect(r, is(registry(), 0, 0, 1),
               "round " + num(round) + " after every drop: " + describe(registry()) + ", want one pool parked");
        settle();
    }
    return r;
}

// The child legs. Each returns only if the abort it exists for did not happen.
int clone_dying_child() {
    ring_lease lease = lazy_io_pool_take();
    if (!lease.held()) {
        std::fprintf(stderr, "child: the take returned an empty lease\n");
        return 2;
    }
    ggml_sycl_test_lazy_mark_dying(lease);
    ring_lease clone = lease.clone();
    std::fprintf(stderr, "child: the clone of a DYING pool returned (%s)\n", clone.held() ? "held" : "empty");
    return 3;
}

int ring_bytes_child() {
    lazy_test_coverage_pin pin = ggml_sycl_test_make_coverage_pin(4096);
    std::fprintf(stderr, "child: a helper guard with 4096 ring bytes was made (%s)\n", pin.pin() ? "held" : "empty");
    return 3;
}

child_run run_leg(const char * flag) {
    char         arg0[] = "test-sycl-lazy-io-pool-host";
    std::string  arg1(flag);
    char * const argv[] = { arg0, &arg1[0], nullptr };
    return run_child(argv, environ, k_child_bound);
}

bool aborted(const child_run & c) {
    return c.ran && WIFSIGNALED(c.status) && WTERMSIG(c.status) == SIGABRT;
}

// A clone that finds its pool DYING logs [HOST-PLAN-BUG] and aborts: a noexcept clone cannot refuse, and a silent
// empty lease is forbidden (§10).
// Mutant: the clone returning an empty lease, or a counted one on the DYING pool (the child returns).
arm_result arm_clone_on_dying_aborts() {
    arm_result      r;
    const child_run c = run_leg("--clone-dying-child");
    expect(r, aborted(c), "the clone of a DYING pool did not abort; the child wrote: " + c.err);
    expect(r, c.err.find("[HOST-PLAN-BUG]") != std::string::npos,
           "the clone's abort logged no [HOST-PLAN-BUG] line; the child wrote: " + c.err);
    return r;
}

// The helper guard takes its pool through the registry, and its destructor drops the lease and runs a FULL reap, so the
// next fixture's guard finds no pool; a second guard while one lives joins its pool; any ring bytes but 0 abort.
// Mutants: the guard taking off-registry (no live pool); its destructor not reaping (a parked node survives it); a
// second guard building its own pool; a non-zero ring accepted.
arm_result arm_helper_guard() {
    arm_result     r;
    const uint64_t c0 = registry().full_reaps_run;
    {
        lazy_test_coverage_pin pin = ggml_sycl_test_make_coverage_pin(0);
        expect(r, pin.pin() != nullptr && pin.pin() == &pin.lease(), "the guard holds no lease");
        expect(r, is(registry(), 1, 1, 0), "with one guard: " + describe(registry()) + ", want 1, 1, 0");
        {
            lazy_test_coverage_pin second = ggml_sycl_test_make_coverage_pin(0);
            expect(r, is(registry(), 1, 2, 0), "with two guards: " + describe(registry()) + ", want them joined");
        }
        expect(r, is(registry(), 1, 1, 0), "after the inner guard: " + describe(registry()) + ", want 1, 1, 0");
    }
    expect(r, is(registry(), 0, 0, 0), "after the guards: " + describe(registry()) + ", want their pool reaped");
    expect(r, registry().full_reaps_run == c0 + 2,
           "the two guards ran " + num(registry().full_reaps_run - c0) + " FULL reaps, want one each");
    const child_run c = run_leg("--ring-bytes-child");
    expect(r, aborted(c) && c.err.find("4096") != std::string::npos,
           "a guard with 4096 ring bytes did not abort naming them; the child wrote: " + c.err);
    return r;
}

// The hold hook's rendezvous: the held reap reports itself and blocks until the fixture releases it.
struct reap_hold {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered  = false;
    bool                    released = false;
};

void release(reap_hold & hold) {
    {
        std::lock_guard<std::mutex> lock(hold.mutex);
        hold.released = true;
    }
    hold.cv.notify_all();
}

void hold_reap(void * ctx) {
    reap_hold *                  hold = static_cast<reap_hold *>(ctx);
    std::unique_lock<std::mutex> lock(hold->mutex);
    hold->entered = true;
    hold->cv.notify_all();
    hold->cv.wait(lock, [hold] { return hold->released; });
}

// The wrapper's reaps, scored at the take (§11 T3's wrapper arm, the same scoring as the lead-run
// test-sycl-lazy-coverage-wrapper): a guard's pool is parked and the guard's FULL reap is held in the hook before its
// list swap; with c0 read after the hook reports, the scope's take must snapshot c0 + 1 reaps run and no parked node,
// and the scope must leave no parked node and c0 + 2 reaps run. take_if(false) takes nothing and still reaps twice.
// Mutants: the entry reap dropped or run after the take (the take snapshots c0, or the parked node); the closing reap
// dropped (the scope's pool survives parked); take_if gated wrong (a lease without a candidate, none with one).
arm_result arm_wrapper_reaps() {
    arm_result r;
    reap_hold  hold;

    std::unique_ptr<lazy_test_coverage_pin> guard(new lazy_test_coverage_pin(ggml_sycl_test_make_coverage_pin(0)));
    ggml_sycl_test_lazy_reap_hold(hold_reap, &hold);
    std::thread dropper([&guard] { guard.reset(); });
    bool        entered = false;
    {
        std::unique_lock<std::mutex> lock(hold.mutex);
        entered = hold.cv.wait_for(lock, k_hook_bound, [&hold] { return hold.entered; });
    }
    const lazy_registry_state held = registry();
    const uint64_t            c0   = held.full_reaps_run;
    if (!entered) {
        // Released and disarmed before the scope exists, so a late reap cannot hold the scope's own entry reap.
        release(hold);
        ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
        fail(r, "the guard's drop never reached the hold hook within " + num(k_hook_bound.count()) + " s");
    } else {
        expect(r, is(held, 0, 0, 1), "while the guard's reap is held: " + describe(held) + ", want 0, 0, 1");
    }
    {
        lazy_coverage_scope       scope;
        const bool                took  = scope.take_if(true);
        const lazy_registry_state taken = registry();
        expect(r, took && scope.pin() != nullptr, "take_if(true) took no lease for a lazy candidate");
        expect(r, taken.reaps_run_at_last_take == c0 + 1,
               "the take snapshot " + num(taken.reaps_run_at_last_take) + " reaps run, want " + num(c0 + 1) +
                   ": the entry reap did not run before the take");
        expect(r, taken.parked_at_last_take == 0,
               "the take found " + num(taken.parked_at_last_take) + " parked nodes: the entry reap did not free them");
        release(hold);
        dropper.join();
        ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
    }
    const lazy_registry_state after = registry();
    expect(r, is(after, 0, 0, 0), "after the scope: " + describe(after) + ", want its pool reaped by its closing reap");
    expect(r, after.full_reaps_run == c0 + 2,
           "after the scope " + num(after.full_reaps_run) + " reaps have run, want " + num(c0 + 2));

    const uint64_t c1 = registry().full_reaps_run;
    {
        lazy_coverage_scope scope;
        expect(r, !scope.take_if(false) && scope.pin() == nullptr, "take_if(false) took a lease");
        expect(r, registry().live_pools == 0, "take_if(false) built a pool");
    }
    expect(r, registry().full_reaps_run == c1 + 2,
           "a scope with no candidate ran " + num(registry().full_reaps_run - c1) + " FULL reaps, want 2");
    return r;
}

// Once the registry is shut down a take returns an empty lease, take_if takes nothing, and a lease taken before only
// decrements at its drop: it never parks, so a lease destroyed during static destruction never touches the mutex's
// reap list. Runs last: the flag is never cleared.
// Mutants: a take after shut-down building or joining a pool; the drop after shut-down parking.
arm_result arm_shut_down() {
    arm_result r;
    ring_lease before = lazy_io_pool_take();
    expect(r, before.held(), "a take before shut-down returned an empty lease");
    lazy_io_pool_mark_shut_down();
    ring_lease after = lazy_io_pool_take();
    expect(r, !after.held(), "a take after shut-down returned a held lease");
    {
        lazy_coverage_scope scope;
        expect(r, !scope.take_if(true), "take_if(true) after shut-down took a lease");
    }
    expect(r, registry().live_leases == 1, "after shut-down's takes: " + describe(registry()) + ", want 1 lease");
    before.reset();
    const lazy_registry_state s = registry();
    expect(r, s.live_leases == 0 && s.parked_nodes == 0,
           "the drop after shut-down left " + describe(s) + ": want the count at 0 and nothing parked");
    return r;
}

const char * outcome_name(outcome o) {
    return o == outcome::PASS ? "PASS" : o == outcome::FAIL ? "FAIL" : "VOID";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--clone-dying-child") == 0) {
        return clone_dying_child();
    }
    if (argc == 2 && std::strcmp(argv[1], "--ring-bytes-child") == 0) {
        return ring_bytes_child();
    }

    struct named_arm {
        const char * name;
        arm_result (*run)();
    };

    const named_arm arms[] = {
        { "takes join one pool, drops count, the last drop parks", arm_take_and_drop         },
        { "a clone counts, moves transfer",                        arm_clone_and_move        },
        { "takes never revive a parked pool; one reap frees all",  arm_park_and_reap         },
        { "concurrent takes give one pool",                        arm_concurrent_takes      },
        { "a clone of a DYING pool aborts",                        arm_clone_on_dying_aborts },
        { "the helper guard takes through the registry and reaps", arm_helper_guard          },
        { "the wrapper reaps before its take and after its drop",  arm_wrapper_reaps         },
        { "shut-down: no take, and a drop only decrements",        arm_shut_down             },
    };
    int passed = 0, failed = 0, voided = 0;
    for (const named_arm & arm : arms) {
        const arm_result result = arm.run();
        std::printf("%s %s%s%s\n", outcome_name(result.result), arm.name, result.why.empty() ? "" : ": ",
                    result.why.c_str());
        std::fflush(stdout);
        passed += result.result == outcome::PASS;
        failed += result.result == outcome::FAIL;
        voided += result.result == outcome::VOID;
    }
    std::printf("lazy io pool host arms: %d passed, %d failed, %d void\n", passed, failed, voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}
