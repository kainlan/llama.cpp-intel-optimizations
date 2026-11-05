// The lazy tables' IO pool registry (docs/design/sycl-storage-tier-lazy-tables.md §3.2, §3.3): the counted lease, its
// clone, the install on a generation match, the park and the DYING transition at the last drop, the FULL reap and its
// list, the coverage wrapper and the helper guard, and the shut-down flag.
//
// Links ggml-sycl/lazy-io-pool.cpp alone, with ggml-base for the abort and the log: no SYCL header, no device, no
// unified cache, no SYCL module. The process holds one registry, so the arms run in order and each leaves it with no
// live pool and no parked node; the shut-down arm runs last because the flag is never cleared. The two abort paths run
// in a child process of this binary.
//
// Every arm names the mutant it kills. Every arm runs, so one failure does not hide the rest.

#include "../ggml/src/ggml-sycl/lazy-io-pool.hpp"
#include "../ggml/src/ggml-sycl/tests/lazy-table-test-util.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern char ** environ;

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using namespace lazy_table_test;

namespace {

constexpr std::chrono::milliseconds k_child_bound{ 30000 };

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
    expect(r, registry_is(registry(), 1, 1, 0), "after one take: " + describe(registry()) + ", want 1, 1, 0");
    ring_lease b = lazy_io_pool_take();
    expect(r, b.held(), "the second take returned an empty lease");
    expect(r, registry_is(registry(), 1, 2, 0),
           "after a second take: " + describe(registry()) + ", want it joined: 1, 2, 0");
    a.reset();
    expect(r, !a.held(), "a reset lease still reads held");
    expect(r, registry_is(registry(), 1, 1, 0), "after one drop of two: " + describe(registry()) + ", want 1, 1, 0");
    b.reset();
    expect(r, registry_is(registry(), 0, 0, 1),
           "after the last drop: " + describe(registry()) + ", want the pool parked: 0, 0, 1");
    settle();
    expect(r, registry_is(registry(), 0, 0, 0), "after a FULL reap: " + describe(registry()) + ", want 0, 0, 0");
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
    expect(r, registry_is(registry(), 0, 0, 0),
           "a clone of an empty lease changed the registry: " + describe(registry()));

    ring_lease a     = lazy_io_pool_take();
    ring_lease clone = a.clone();
    expect(r, clone.held(), "a clone of a held lease is empty");
    expect(r, registry_is(registry(), 1, 2, 0), "after a clone: " + describe(registry()) + ", want 1, 2, 0");
    a.reset();
    expect(r, registry_is(registry(), 1, 1, 0),
           "the original dropped while its clone lives: " + describe(registry()) + ", want the pool kept: 1, 1, 0");

    ring_lease moved(std::move(clone));
    expect(r, moved.held() && !clone.held(), "a move did not transfer the reference");
    expect(r, registry_is(registry(), 1, 1, 0), "a move changed the count: " + describe(registry()) + ", want 1, 1, 0");

    ring_lease target = lazy_io_pool_take();
    expect(r, registry_is(registry(), 1, 2, 0),
           "a take beside a live lease: " + describe(registry()) + ", want 1, 2, 0");
    target = std::move(moved);
    expect(r, target.held() && !moved.held(), "a move-assignment did not transfer the reference");
    expect(r, registry_is(registry(), 1, 1, 0),
           "a move-assignment over a held lease: " + describe(registry()) + ", want its reference dropped: 1, 1, 0");
    target.reset();
    expect(r, registry_is(registry(), 0, 0, 1), "after the last drop: " + describe(registry()) + ", want 0, 0, 1");
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
    expect(r, registry_is(registry(), 0, 0, 1),
           "after the first pool's last drop: " + describe(registry()) + ", want 0, 0, 1");
    ring_lease b = lazy_io_pool_take();
    expect(r, b.held(), "a take after the last drop returned an empty lease");
    expect(r, registry_is(registry(), 1, 1, 1),
           "a take after the last drop: " + describe(registry()) + ", want a new pool beside the parked one: 1, 1, 1");
    b.reset();
    expect(r, registry_is(registry(), 0, 0, 2),
           "after the second pool's last drop: " + describe(registry()) + ", want 0, 0, 2");
    const uint64_t c1 = registry().full_reaps_run;
    settle();
    expect(r, registry_is(registry(), 0, 0, 0),
           "one FULL reap left " + describe(registry()) + ", want every node freed");
    expect(r, registry().full_reaps_run == c1 + 1, "the FULL reap did not count once");
    return r;
}

// Concurrent takes on an empty registry give one pool: a take that loses the install race joins the winner's pool and
// destroys its own. Several rounds, each released together from a spin barrier.
// Mutant: an install that checks neither the generation nor the slot (a second pool replaces the first, whose leases
// are lost from the slot's count, and two pools park at the drops).
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
        expect(r, registry_is(registry(), 1, threads, 0),
               "round " + num(round) + ": " + describe(registry()) + ", want one pool with every lease");
        leases.clear();
        expect(r, registry_is(registry(), 0, 0, 1),
               "round " + num(round) + " after every drop: " + describe(registry()) + ", want one pool parked");
        settle();
    }
    return r;
}

// What take B saw inside take A's build.
struct take_race {
    bool                fired        = false;
    bool                b_held       = false;
    uint64_t            b_generation = 0;
    lazy_registry_state after_b;
};

// The take hook: take B runs inside take A's build window, installs a pool, and drops it, so it parks.
void take_and_drop_b(void * ctx) {
    take_race * race   = static_cast<take_race *>(ctx);
    race->fired        = true;
    ring_lease b       = lazy_io_pool_take();
    race->b_held       = b.held();
    race->b_generation = registry().live_generation;
    b.reset();
    race->after_b = registry();
}

// A take installs only on a generation match (§3.3): take A reserves the generation after g0 and, held in the take hook
// after its build, take B reserves the same one, installs and drops, so B's pool parks. A's generation no longer
// matches, so A destroys its pool and takes again: B's pool had g0 + 1, A's has g0 + 2, and A's pool is live with one
// lease beside B's parked one.
// Mutant: LAZY_IO_POOL_MUTANT_INSTALL_ON_EMPTY_SLOT, the empty-slot install rule (A installs the pool it built for the
// generation B already installed, so A's pool carries g0 + 1 too).
arm_result arm_generation_install() {
    arm_result r;
    settle();
    const uint64_t g0 = registry().generation;
    take_race      race;
    ggml_sycl_test_lazy_take_hold(take_and_drop_b, &race);
    ring_lease a = lazy_io_pool_take();
    // Disarmed in case the hook never fired, so it cannot hold a later arm's take.
    ggml_sycl_test_lazy_take_hold(nullptr, nullptr);
    const lazy_registry_state s = registry();
    if (!race.fired) {
        fail(r, "take A's build never reached the take hook: the hook is not between the reserve and the install");
    }
    expect(r, race.b_held && race.b_generation == g0 + 1,
           "take B, inside A's build, holds a pool of generation " + num(race.b_generation) + ", want " + num(g0 + 1));
    expect(r, registry_is(race.after_b, 0, 0, 1),
           "after B's drop: " + describe(race.after_b) + ", want B's pool parked");
    expect(r, a.held(), "take A returned an empty lease");
    expect(r, s.live_generation != race.b_generation,
           "take A's pool shares generation " + num(s.live_generation) +
               " with the pool installed and parked during its build");
    expect(r, s.live_generation == g0 + 2 && s.generation == g0 + 2,
           "take A's pool has generation " + num(s.live_generation) + " and the registry " + num(s.generation) +
               ", want " + num(g0 + 2) + " for both");
    expect(r, registry_is(s, 1, 1, 1),
           "after A's take: " + describe(s) + ", want A's pool live with 1 lease and B's parked");
    a.reset();
    settle();
    expect(r, registry_is(registry(), 0, 0, 0),
           "after A's drop and a FULL reap: " + describe(registry()) + ", want 0, 0, 0");
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
        expect(r, registry_is(registry(), 1, 1, 0), "with one guard: " + describe(registry()) + ", want 1, 1, 0");
        {
            lazy_test_coverage_pin second = ggml_sycl_test_make_coverage_pin(0);
            expect(r, registry_is(registry(), 1, 2, 0),
                   "with two guards: " + describe(registry()) + ", want them joined");
        }
        expect(r, registry_is(registry(), 1, 1, 0),
               "after the inner guard: " + describe(registry()) + ", want 1, 1, 0");
    }
    expect(r, registry_is(registry(), 0, 0, 0),
           "after the guards: " + describe(registry()) + ", want their pool reaped");
    expect(r, registry().full_reaps_run == c0 + 2,
           "the two guards ran " + num(registry().full_reaps_run - c0) + " FULL reaps, want one each");
    const child_run c = run_leg("--ring-bytes-child");
    expect(r, aborted(c) && c.err.find("4096") != std::string::npos,
           "a guard with 4096 ring bytes did not abort naming them; the child wrote: " + c.err);
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

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--clone-dying-child") == 0) {
        return clone_dying_child();
    }
    if (argc == 2 && std::strcmp(argv[1], "--ring-bytes-child") == 0) {
        return ring_bytes_child();
    }

    std::vector<named_arm> arms = {
        { "takes join one pool, drops count, the last drop parks", arm_take_and_drop         },
        { "a clone counts, moves transfer",                        arm_clone_and_move        },
        { "takes never revive a parked pool; one reap frees all",  arm_park_and_reap         },
        { "concurrent takes give one pool",                        arm_concurrent_takes      },
        { "a take installs only on a generation match",            arm_generation_install    },
        { "a clone of a DYING pool aborts",                        arm_clone_on_dying_aborts },
        { "the helper guard takes through the registry and reaps", arm_helper_guard          },
    };
    for (const named_arm & arm : wrapper_arms()) {
        arms.push_back(arm);
    }
    // Last: the shut-down flag is never cleared.
    arms.push_back({ "shut-down: no take, and a drop only decrements", arm_shut_down });
    return run_arms(arms, "lazy io pool host arms");
}
